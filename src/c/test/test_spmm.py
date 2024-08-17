
import numpy as np
import scipy.sparse as sp
from random import randint

import os
from tqdm import tqdm
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))

from input_output import mmap_unified_write, mmap_unified_read, mmap_file_init
from ctypes_interface import spmm_batched, DenseSerialize, CSRSerialize

MMAP_PATH="/ssd1/smt-mmap"

def csr_write(csr):
    a_fname = MMAP_PATH + f"/test_spmm_{randint(-sys.maxsize, sys.maxsize)}.bin"
    mmap_unified_write(a_fname, (csr.data, csr.indices, csr.indptr))
    return a_fname

def dense_file_read(fname, shape):
    dtype_shp = [(np.float32, shape[0] * shape[1])]
    return mmap_unified_read(fname, dtype_shp)[0].reshape(shape)

def rand_dense(shape, sparsity=0.5):
    dense = np.zeros(shape=shape, dtype=np.float32)
    nnz = int(shape[0] * shape[1] * sparsity)
    nnz_idx = np.random.choice(shape[0] * shape[1], nnz, replace=False)
    dense.flat[nnz_idx] = np.random.rand(nnz)
    fname = MMAP_PATH + f"/test_spmm_{randint(-sys.maxsize, sys.maxsize)}.bin"
    assert dense.dtype == np.float32
    dense = dense.T.copy(order="C")
    mmap_file_init(fname, dense, False)
    return dense, fname


def rand_csr(shape, sparsity=0.25):
    # Generate random dense matrix of given shape + sparsity, then convert to dense
    dense = np.zeros(shape=shape, dtype=np.float32)
    nnz = int(shape[0] * shape[1] * sparsity)
    nnz_idx = np.random.choice(shape[0] * shape[1], nnz, replace=False)
    dense.flat[nnz_idx] = np.random.rand(nnz)
    s = sp.csr_array(dense, shape=shape, dtype=np.float32)
    s.indices = s.indices.astype(np.int32)
    s.indptr = s.indptr.astype(np.int32)
    s = s.T.tocsr()
    return s


def main():
    # we want dense @ sparse, but spmm only supports sparse @ dense, so we calculate dense @ sparse == (sparse.T @ dense.T).T
    # for now, C code will assume all transposes are performed by Python code
    a_shp = (1, 364500)
    b_shp = (364500, 2)
    n_iter = 1
    n_b = 5
    
    a, a_fname = rand_dense(a_shp)      # NOTE: row-major layout
    a_serial = DenseSerialize(str.encode(a_fname), a.shape[0], a.shape[1], -1, -1)
    
    bs = [rand_csr(b_shp) for _ in range(n_b)]
    b_fnames = [csr_write(b) for b in bs]
    b_serials = [CSRSerialize(str.encode(b_fname), b.nnz, b.shape[0], b.shape[1], -1, -1) for b_fname, b in zip(b_fnames, bs)]
    c_files = [MMAP_PATH + f"/test_spmm_res_{randint(-sys.maxsize, sys.maxsize)}.bin" for _ in range(n_b)]
    for _ in tqdm(range(n_iter)):       # run repeatedly as a sanity check
        spmm_batched(a_serial, b_serials, c_files)

    # Check correctness
    c = np.vstack([dense_file_read(f, (b_shp[1], a_shp[0])) for f in c_files])
    gt = np.vstack([b.todense() for b in bs]) @ a       # equivalent to the (a.T @ b.T).T approach used with spmm
    print(f"Result: ({gt.shape}) ({c.shape})")
    print("\t", np.max(np.abs(gt - c)))
    print(f"\n\n{gt}")
    print(f"\n\n{c}")


if __name__ == "__main__":
    main()




