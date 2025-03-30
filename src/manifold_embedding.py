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

        # TODO: combine inv_sqrt_cov and inner passes through the dataset (can share the sparse code computation)    
        # calculate inverse square root covariance matrix
        inv_sqrt_cov = self.inv_sqrt_cov(dset, sc_layer)

        # calculate core ADD^TA^T loss matrix
        inner = self.core_loss_matrix(dset, sc_layer)

        # compute complete loss matrix
        inner = inner.to(torch.float64)
        inv_sqrt_cov = inv_sqrt_cov.to(torch.float64)
        closed_form = inv_sqrt_cov @ inner @ inv_sqrt_cov
        assert _is_real_sym(closed_form, tol=1e-3), "Closed form is not (almost) real-symmetric."

        # solve, populate projection matrix
        self.solve(closed_form, inv_sqrt_cov)

    def inv_sqrt_cov(self, dset, sc_layer, stride=1):
        # Calculate mean over all patches
        mean = dset.apply_and_sum(sc_layer.sparse_code_img, dim=1, stride=stride, cuda=True, whiten=True)    
        mean /= (self.args.samples * dset.n_patch_per_img)
        mean = mean.unsqueeze(-1)

        # Calculate centered covariance matrix
        def sub(patches):
            return (sc_layer.sparse_code_img(patches) - mean).T
        cov_mx = dset.apply_and_reduce(sub, stride, cuda=True) / (self.args.samples * dset.n_patch_per_img)
        
        # matrix inverse sqrt
        cov_mx = torch_force_symmetric(cov_mx)
        assert _is_real_sym(cov_mx)
        inv_sqrt_alpha_cov = mx_inv_sqrt(cov_mx).to("cuda", non_blocking=True)
        assert _is_real_sym(inv_sqrt_alpha_cov, tol=1e-1), "Inv. sqrt. covariance is not (almost) real-symmetric."
        inv_sqrt_alpha_cov = torch_force_symmetric(inv_sqrt_alpha_cov)
        return inv_sqrt_alpha_cov

    def core_loss_matrix(self, dset, sc_layer, stride=1):
        diff_op = DifferentialOperator(self.args, dset)
        def apply_sc(patches):
            return sc_layer.sparse_code_img(patches).T
        inner = dset.apply_and_reduce(apply_sc, stride,  bilinear_func=diff_op.get_bilinear_form, cuda=True)
        inner = torch_force_symmetric(inner)
        assert _is_real_sym(inner)
        print(f"Generated {diff_op.custom_dop} custom differential operators for images with zero code patches")
        return inner

    def solve(self, closed_form, inv_sqrt_cov):
        print("Solving closed form...", flush=True)
        success = False
        if torch.cuda.is_available():
            try:
                evals, evecs = torch.linalg.eigh(closed_form)
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
            print("Torch eigh GPU solver failed, falling back to scipy eigh")
            if sys.version_info < (3, 11):
                compiled_fn = torch.compile(sci.linalg.eigh)
                evals, evecs = compiled_fn(closed_form)
            else:
                evals, evecs = sci.linalg.eigh(closed_form)

        # Handle negative eigenvectors that can occur from numerical instability
        neg_mask = evals < 0
        evecs[:, neg_mask] = evecs[:, neg_mask] * -1
        evals = np.abs(evals)        
        print(f"\tMin. (abs. value) eigenvalue: {np.abs(evals).min()}. Min e'val: {evals.min()}.")

        # Select the f eigenvectors with smallest eigenvalues (eigenvectors are COLUMNs of evec matrix (see torch.linalg.eig reference))
        # Need to convert them to rows to give a mapping to f-dimensional space
        if not self.args.disable_color_embed_drop:
            skip_first_n = 16
            indices = np.argsort(evals, kind='stable')[skip_first_n:(self.args.embed_dim + skip_first_n)] 
        else:
            # Note: drops the least e'vec (kernel of the Laplacian, constant vector)
            indices = np.argsort(evals, kind='stable')[1:(self.args.embed_dim+1)] 

        assert indices.shape[0] == self.args.embed_dim
        evals = evals[indices]       
        U = torch.from_numpy(evecs[:, indices].transpose())
        U_full = torch.from_numpy(evecs.transpose())

        # Discrepancy between (1) and (2) on ordering of U and inv_sqrt_cov here, 
        # this way from (2) makes the shapes work out + makes sense intuitively. First, apply whitening
        # transform to the alpha vector that P is right-multiplied by, then apply U to obtain a spectral
        # embedding of this whitened vector
        self.projection = U @ inv_sqrt_cov.cpu()
        assert self.projection.shape[0] == self.args.embed_dim 
        assert not torch.any(torch.isnan(self.projection))
        self.projection_full = U_full @ inv_sqrt_cov.cpu()

        self.projection = self.projection.to(torch.float32)
        self.projection_full = self.projection_full.to(torch.float32)

    @torch.compiler.disable
    def __call__(self, x):
        """Apply calculated SMT to given inputs, return their embeddings."""        
        beta_flat = self.projection @ x
        beta_flat /= (torch.linalg.vector_norm(beta_flat, dim=1).unsqueeze(1) + 1e-10)
        return beta_flat


