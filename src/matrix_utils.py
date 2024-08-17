import torch
import numpy as np
from time import time
import scipy.sparse as sp

def _all_close(a, b, tol=1e-5):
    return (torch.abs(a - b) < tol).all()

def _is_real_sym(m, verbose=False):
    if verbose:
        print(f"is real sym: {torch.max(torch.abs(m - m.T))}", flush=True)
    return _all_close(m, m.T) and not m.is_complex()

def _np_is_real_sym(m, verbose=False, tol=1e-5):
    if verbose:
        print(f"is real sym: {np.max(np.abs(m - m.T))}", flush=True)
    return np.allclose(m, m.T, rtol=tol, atol=tol) and not np.iscomplex(m).any()

def mx_frac_pow(m, p, tol):
    # Calculate eigendecomposition and then exponentiate eigenvectors to get fractional/negative power of a matrix
    # NOTE: this assumes that input matrix is real, symmetric (this is only applied to a covariance matrix 
    # so this must be true)
    # NOTE: actually tol here is a critical parameter (esp. for whitening matrices)
    assert len(m.shape) == 2 and m.shape[0] == m.shape[1]
    assert _is_real_sym(m)
    orig_dtype = m.dtype
    m = m.to(torch.float32)
    if torch.cuda.is_available():
        m = m.to('cuda:0')
        m += torch.eye(m.shape[0], device='cuda:0', dtype=m.dtype) * tol      # handle singular mx
    else:
        m += torch.eye(m.shape[0]) * tol      # handle singular mx
    evals, evecs = torch.linalg.eigh(m)
    m = m.cpu()

    # invert any negative e'val/e'vec
    # TODO(as) really shouldnt need this...
    neg_mask = evals < 0
    evecs[:, neg_mask] = evecs[:, neg_mask] * -1
    evals = torch.abs(evals)
    # assert not torch.any(evals < 0), f"Negative eigenvalues: {evals.min()}"
    
    # shuffle around to reduce GPU memory usage
    evpow = evals ** p
    evals = evals.cpu()

    evec_inv = torch.inverse(evecs).cpu()

    out = evecs @ torch.diag(evpow)
    evecs = evecs.cpu()
    evpow = evpow.cpu()

    if torch.cuda.is_available():
        evec_inv = evec_inv.to('cuda:0')
    out = out @ evec_inv

    assert not torch.any(torch.isnan(out)), "Matrix inverse sqrt. is NaN"
    out = out.cpu()
    out = out.to(orig_dtype)
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return out

def mx_inv_sqrt(m, tol=1e-15):
    return mx_frac_pow(m, -1/2, tol)

def force_symmetric(m):
    # Copy upper triangle to lower triangle of given matrix
    return np.triu(m) + np.triu(m, k=1).T

def torch_force_symmetric(m):
    # Copy upper triangle to lower triangle of given matrix
    return torch.triu(m) + torch.triu(m, diagonal=1).T

def csr_row_view(csr, start, end):
    assert start >= 0 and end <= csr.shape[0] and start < end
    indptr = csr.indptr[start:end+1].copy()
    indices = csr.indices[indptr[0] : indptr[-1]]
    data = csr.data[indptr[0] : indptr[-1]]
    indptr -= indptr[0]     # recenter all indices at 0
    return sp.csr_array((data, indices, indptr), shape=(end-start, csr.shape[1]), dtype=csr.dtype, copy=False)

def profile_log(enable, running, out):
    if enable:
        print(f'{out} {time() - running}', flush=True)
        running = time()
    return running

