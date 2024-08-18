import os
import sys
import torch
import numpy as np
import scipy as sci
from tqdm import tqdm
from random import randint
import shutil

from input_output import mmap_file_init
from matrix_utils import mx_inv_sqrt, _np_is_real_sym, force_symmetric
from matmul import GpuSparseMatmul
from slice import PreMatmulCacheGen, csr_col_slice, csr_col_slice_transpose
from loss_mx_calc import LossMatrixCalc
from ctypes_interface import c_impl_available

class ManifoldEmbedLayer:
    def __init__(self, args, alphas, diff_op, embed_dim, proj=None):
        """Given a set of vectorized inputs and an associated differential operator, calculate their manifold embedding."""
        if proj is not None:
            # loading from checkpoint (convert input projection matrix to mmap)
            t_fname = args.mmap_path + "/smt_proj_dense_T.bin"
            mmap_file_init(t_fname, proj)
            self.projection = np.memmap(t_fname, shape=proj.shape, dtype=proj.dtype)
            self.embed_dim = embed_dim
            self.args = args
            return
    
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
        U = self._post_process_solution(args, evals, evecs)

        # Discrepancy between (1) and (2) on ordering of U and inv_sqrt_alpha_cov here, 
        # this way from (2) makes the shapes work out + makes sense intuitively. First, apply whitening
        # transform to the alpha vector that P is right-multiplied by, then apply U to obtain a spectral
        # embedding of this whitened vector
        P = U @ inv_sqrt_alpha_cov 
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

    def _get_inv_sqrt_cov(self, args, alphas):
        # Apply slicing + CSC caching to alphas CSR matrix
        alphas_cache = self._gen_slice_cache(args, alphas, args.cov_chunk, alphas_2d_slice=True, batch_sz_2d=args.cov_col_chunk)
        alphas_T_cache = self._gen_slice_cache(args, alphas, args.cov_chunk, transpose_2d_slice=True, batch_sz_2d=args.cov_col_chunk)

        # "cov" is not really the covariance, it is AA^T / N per reference (2)
        print("Calculating covariance matrix...", flush=True)
        if torch.cuda.is_available(): torch.cuda.empty_cache()        
        ssm = GpuSparseMatmul(alphas_cache, alphas_T_cache, True, args.cov_chunk, False, mmap_path=args.mmap_path, a_shape=alphas.shape)
        ssm.run(daemon=True)
        if torch.cuda.is_available(): torch.cuda.empty_cache()        
        cov = ssm.result / alphas.shape[1]
        
        cov = force_symmetric(cov)
        assert _np_is_real_sym(cov, verbose=False)
        cov = torch.from_numpy(cov)       # No (useful) sparse eigen-solvers give all eigenvalues and this is a (dict_sz x dict_sz) covariance matrix (which tend to be dense)
        
        inv_sqrt_alpha_cov = mx_inv_sqrt(cov).numpy()
        assert _np_is_real_sym(inv_sqrt_alpha_cov, False, tol=1), "Inv. sqrt. covariance is not (almost) real-symmetric."
        inv_sqrt_alpha_cov = force_symmetric(inv_sqrt_alpha_cov)
        for k in alphas_T_cache:
            for i in alphas_T_cache[k]:
                alphas_T_cache[k][i].cleanup()
        for k in alphas_cache:
            for i in alphas_cache[k]:
                if os.path.exists(i.fname):
                    i.cleanup()
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
        return evecs[:, indices].transpose()

    def __call__(self, x):
        """Apply calculated SMT to given inputs, return their embeddings."""        
        # Want to calculate self.projection @ x, but spmm requires "sparse @ dense" format so instead we calculate (x.T @ self.projection.T).T
        cache = self._gen_slice_cache(self.args, x, self.args.proj_col_chunk, col_slice=True, transpose_2d_slice=True)        # column-slicing, but also transpose slices
        ssm = GpuSparseMatmul(self.projection, cache, False, self.args.proj_row_chunk, dense_matmul=True, mmap_path=self.args.mmap_path, a_shape=x.shape)
        ssm.run(daemon=True)
        beta_flat = ssm.result
        
        # L2 normalizes embeddings as in (2)
        print("Normalizing SMT embeddings...", flush=True)

        # TODO(as) trivial to do a better version of this in C. For now, just chunk across embedding array so doesnt use so much RAM at once
        # (accumulator array, iterate through rows of CSR summing squared entries, sqrt, iterate through CSR entries again dividing appropriately)
        chnk_sz = int(beta_flat.shape[1] / 10)
        for start in tqdm(range(0, beta_flat.shape[1], chnk_sz)):
            end = min(beta_flat.shape[1], start + chnk_sz)
            beta_flat[:, start:end] /= (np.linalg.norm(beta_flat[:, start:end], ord=2, axis=0) + 1e-20)
        return beta_flat




