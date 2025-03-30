import os
import sys
import torch
import numpy as np
import scipy as sci
from tqdm import tqdm
from random import randint
import shutil

from input_output import mmap_file_init
from matrix_utils import mx_inv_sqrt, _is_real_sym, torch_force_symmetric
from matmul import GpuSparseMatmul
from slice import PreMatmulCacheGen, csr_col_slice, csr_col_slice_transpose
from loss_mx_calc import LossMatrixCalc
from ctypes_interface import c_impl_available


from diff_op import DifferentialOperator

import pdb

class ManifoldEmbedLayer:
    def __init__(self, args, dset, sc_layer, proj=None):
        self.args = args
        if proj is not None:
            # loading from checkpoint (convert input projection matrix to mmap)
            
            # TODO: remove mmap stuff...
            t_fname = args.mmap_path + "/smt_proj_dense_T.bin"
            mmap_file_init(t_fname, proj)
            self.projection = np.memmap(t_fname, shape=proj.shape, dtype=proj.dtype)
            
            self.projection = torch.from_numpy(self.projection)
            self.embed_dim = embed_dim
            return
    
        
        # calculate inverse square root covariance matrix
        inv_sqrt_cov = self.inv_sqrt_cov(dset, sc_layer)


        pdb.set_trace()

        # TODO: calculate core ADD^TA^T loss matrix
        diff_op = DifferentialOperator(args, dset)


        # TODO: compute complete loss matrix, solve, get projection matrix




        if torch.cuda.is_available(): torch.cuda.empty_cache()
    
        assert alphas.dtype == np.float32
        assert isinstance(alphas, (sci.sparse.csr_array))
        print("Calculating embedding matrix...", flush=True)
        self.embed_dim = embed_dim
        self.args = args

        # Calculate inverse square root of the covariance matrix of the input
        inv_sqrt_alpha_cov = self._get_inv_sqrt_cov(args, alphas)
        
        # Solve optimization problem as outlined in equation 8 of (1)
        # Efficiently calculate alphas @ diff_op @ diff_op.T @ alphas.T for image datasets
        inner = self._calc_inner(args, alphas, diff_op)
        assert _np_is_real_sym(inner), "Inner is not real-symmetric."

        print("Generating closed form formulation...", flush=True)
        closed_form = inv_sqrt_alpha_cov @ inner @ inv_sqrt_alpha_cov
        assert _np_is_real_sym(closed_form, verbose=False, tol=1e-1), "Closed form is not (almost) real-symmetric."

        print("Solving closed form...", flush=True)
        success = False
        if torch.cuda.is_available():
            try:
                evals, evecs = torch.linalg.eigh(torch.from_numpy(closed_form).to('cuda:0'))
                evals = evals.cpu().numpy()
                evecs = evecs.cpu().numpy()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                success = True    
            except RuntimeError as e:
                if 'out of memory' not in str(e):
                    raise e
                elif torch.cuda.is_available(): torch.cuda.empty_cache()
        if not success:
            # run on CPU if no GPU or out of VRAM
            if sys.version_info < (3, 11):
                compiled_fn = torch.compile(sci.linalg.eigh)
                evals, evecs = compiled_fn(closed_form)
            else:
                evals, evecs = sci.linalg.eigh(closed_form)

        # Post-process eigendecomposition solution
        U, U_full = self._post_process_solution(args, evals, evecs)

        # Discrepancy between (1) and (2) on ordering of U and inv_sqrt_alpha_cov here, 
        # this way from (2) makes the shapes work out + makes sense intuitively. First, apply whitening
        # transform to the alpha vector that P is right-multiplied by, then apply U to obtain a spectral
        # embedding of this whitened vector
        P = U @ inv_sqrt_alpha_cov 
        self.projection_full = (U_full @ inv_sqrt_alpha_cov).astype(np.float32)
        assert P.shape[0] == self.embed_dim 
        # and P.shape[1] == args.dict_sz and 
        assert not np.any(np.isnan(P))
        
        # efficient dense-sparse matmul in __call__ requires all arguments are file-backed (and float32 format)
        P = P.astype(np.float32)
        t_fname = self.args.mmap_path + f"/smt_proj_dense_T_{randint(-sys.maxsize, sys.maxsize)}.bin"
        mmap_file_init(t_fname, P)
        self.projection = np.memmap(t_fname, shape=P.shape, dtype=P.dtype)


    def _gen_slice_cache(self, args, alphas, batch_sz, transpose_2d_slice=False, alphas_2d_slice=False, col_slice=False, batch_sz_2d=None, means=None):
        # Given a CSR matrix, slice it according to the proper batch size, then store as a mmap
        print("Generating sliced cache for faster matmuls...", flush=True)
        c_enable = torch.cuda.is_available()
        if col_slice and not transpose_2d_slice and c_impl_available():
            print("Using C-code for column slicing...")
            cache_dir = args.mmap_path + f"/col_slice_cache_{randint(-sys.maxsize, sys.maxsize)}"
            os.mkdir(cache_dir)
            return csr_col_slice(cache_dir, alphas, batch_sz, True), cache_dir
        elif col_slice and transpose_2d_slice and c_impl_available():
            print("Using C-code for tranposed column slicing...")
            return csr_col_slice_transpose(args, alphas, batch_sz)
        else:
            mgccg = PreMatmulCacheGen(args, alphas, batch_sz, transpose_2d_slice, alphas_2d_slice=alphas_2d_slice, col_slice=col_slice, batch_sz_2d=batch_sz_2d, means=means)
            mgccg.run(daemon=True)
            if col_slice and not transpose_2d_slice:
                return mgccg.cache, ""
            return mgccg.cache

    def inv_sqrt_cov(self, dset, sc_layer, stride=1):
        # Calculate mean over all patches
        mean = torch.zeros((self.args.dict_sz), device="cuda")
        for idx in tqdm(range(self.args.samples)):
            mean += sc_layer.sparse_code_img(dset.img_to_centered_patches(dset.train_set_image(idx, cuda=True)[0], stride)).sum(dim=1)
        mean /= (self.args.samples * dset.n_patch_per_img)
        mean = mean.unsqueeze(-1)

        # Calculate centered covariance matrix
        def sub(patches):
            return (sc_layer.sparse_code_img(patches) - mean).T
        cov_mx = dset.apply_and_reduce(sub, stride, cuda=True) / (self.args.samples * dset.n_patch_per_img)
        
        cov_mx = torch_force_symmetric(cov_mx)
        assert _is_real_sym(cov_mx)
        inv_sqrt_alpha_cov = mx_inv_sqrt(cov_mx).to("cuda", non_blocking=True)
        assert _is_real_sym(inv_sqrt_alpha_cov, tol=1e-1), "Inv. sqrt. covariance is not (almost) real-symmetric."

        inv_sqrt_alpha_cov = torch_force_symmetric(inv_sqrt_alpha_cov)
        return inv_sqrt_alpha_cov

    def _calc_inner(self, args, alphas, diff_op):
        # Calculate A @ D @ D^T @ A^T from A and D@D^T
        # Apply differential operator to the alphas matrix (part of inner computation)

        print("Computing final cost matrix...", flush=True)
        diff_op_cache, cache_dir = diff_op

        if torch.cuda.is_available(): torch.cuda.empty_cache()

        calc = LossMatrixCalc(args, alphas, diff_op_cache)
        calc.run(daemon=True)
        inner = calc.out

        inner = force_symmetric(inner)
        assert _np_is_real_sym(inner, verbose=False)

        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # g = (alphas @ diff_op).todense()
        # gt = g @ alphas.todense().T
        # diff = np.abs(gt - inner).max()
        # print("DIFF: ", diff)
        # sys.exit()
            
        for k in diff_op_cache:
            diff_op_cache[k].cleanup()
        if cache_dir != "":
            shutil.rmtree(cache_dir)
        return inner

    def _post_process_solution(self, args, evals, evecs):
        """Check solution for numerical instability, invert negative e'val, convert e'vec into (part of) the SMT embedding matrix."""

        neg_mask = evals < 0
        evecs[:, neg_mask] = evecs[:, neg_mask] * -1
        evals = np.abs(evals)
        
        # Handle negative eigenvectors that can occur from numerical instability. Larger negative eigenvalues likely a bug
        print(f"\tMin. (abs. value) eigenvalue: {np.abs(evals).min()}. Min e'val: {evals.min()}.")

        # Select the f eigenvectors with smallest eigenvalues (eigenvectors are COLUMNs of evec matrix (see torch.linalg.eig reference))
        # Need to convert them to rows to give a mapping to f-dimensional space
        if not args.disable_color_embed_drop:
            skip_first_n = 16
            indices = np.argsort(evals, kind='stable')[skip_first_n:(self.embed_dim + skip_first_n)] 
        else:
            # Note: drops the least e'vec (kernel of the Laplacian, constant vector)
            indices = np.argsort(evals, kind='stable')[1:(self.embed_dim+1)] 

        assert indices.shape[0] == self.embed_dim
        evals = evals[indices]       
        return evecs[:, indices].transpose(), evecs.transpose()

    @torch.compiler.disable
    def __call__(self, x, dense=False):
        """Apply calculated SMT to given inputs, return their embeddings."""        
    
        if dense:
            beta_flat = self.projection @ x
            beta_flat /= (torch.linalg.vector_norm(beta_flat, dim=1).unsqueeze(1) + 1e-10)
            return beta_flat
        else:
            # Want to calculate self.projection @ x, but spmm requires "sparse @ dense" format so instead we calculate (x.T @ self.projection.T).T
            cache = self._gen_slice_cache(self.args, x, self.args.proj_col_chunk, col_slice=True, transpose_2d_slice=True)        # column-slicing, but also transpose slices
            ssm = GpuSparseMatmul(self.projection, cache, False, self.args.proj_row_chunk, dense_matmul=True, mmap_path=self.args.mmap_path, a_shape=x.shape)
            ssm.run(daemon=True)
            beta_flat = ssm.result
            
            # L2 normalizes embeddings as in (2)
            print("Normalizing SMT embeddings...", flush=True)
            chnk_sz = int(beta_flat.shape[1] / 10)
            for start in tqdm(range(0, beta_flat.shape[1], chnk_sz)):
                end = min(beta_flat.shape[1], start + chnk_sz)
                beta_flat[:, start:end] /= (np.linalg.norm(beta_flat[:, start:end], ord=2, axis=0) + 1e-10)
            return beta_flat




