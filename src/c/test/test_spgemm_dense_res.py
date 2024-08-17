
import numpy as np
import scipy.sparse as sp
from random import randint

import os
from tqdm import tqdm
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))

from input_output import mmap_unified_write, mmap_unified_read
from ctypes_interface import CSRSerialize, spgemm_batched_matmul

MMAP_PATH="/ssd1/smt-mmap"


def csr_write(csr):
    a_fname = MMAP_PATH + f"/test_spgemm_{randint(-sys.maxsize, sys.maxsize)}.bin"
    mmap_unified_write(a_fname, (csr.data, csr.indices, csr.indptr))
    return a_fname

def dense_file_read(fname, shape):
    dtype_shp = [(np.float32, shape[0] * shape[1])]
    return mmap_unified_read(fname, dtype_shp)[0].reshape(shape)

def rand_csr(shape, sparsity=0.1):
    # Generate random dense matrix of given shape + sparsity, then convert to dense
    dense = np.zeros(shape=shape, dtype=np.float32)
    nnz = int(shape[0] * shape[1] * sparsity)
    nnz_idx = np.random.choice(shape[0] * shape[1], nnz, replace=False)
    dense.flat[nnz_idx] = np.random.rand(nnz)
    return sp.csr_array(dense, shape=shape)

def result_processing(a_shape, b_shape, c_files):
    # NOTE: assumes all b matrices have the same shape
    # NOTE: assumes c_files are sorted according to the order of b columns
    result = np.zeros(shape=(a_shape[0], len(c_files)*b_shape[1]))
    for idx, fname in enumerate(c_files):
        b_col = b_shape[1]
        result[:, idx*b_col : (idx+1)*b_col] = dense_file_read(fname, (a_shape[0], b_col))
    return result


def main():
    # Create sparse test matrices
    a_shp = (20, 50000)
    a = rand_csr(a_shp)
    b_shp = (50000, 20)
    n_b = 25
    bs = [rand_csr(b_shp) for _ in range(n_b)]
    n_iter = 1

    # Prep to call C exe (format as would be expected in full program)
    a_fname = csr_write(a)
    b_fnames = [csr_write(b) for b in bs]

    # Call C function
    # assert a.dtype == np.float32 and b.dtype == np.float32, "C code assumes CSR data types"
    # assert a.indices.dtype == np.int32 and a.indptr.dtype == np.int32
    # assert b.indices.dtype == np.int32 and b.indptr.dtype == np.int32
    print(f"PYTHON: {a.data.shape} {a.indices.shape} {a.indices.dtype} {a.indptr.shape} {a.indptr.dtype}")
    
    a_serial = CSRSerialize(str.encode(a_fname), a.nnz, a.shape[0], a.shape[1], -1, -1)
    b_serials = [CSRSerialize(str.encode(b_fname), b.nnz, b.shape[0], b.shape[1], -1, -1) for b_fname, b in zip(b_fnames, bs)]
    c_files = [MMAP_PATH + f"/test_spgemm_res_{randint(-sys.maxsize, sys.maxsize)}.bin" for _ in range(n_b)]
    for _ in tqdm(range(n_iter)):       # run repeatedly as a sanity check
        spgemm_batched_matmul(a_serial, b_serials, c_files)

    # Process results 
    b_stack = sp.hstack(bs, dtype=np.float32)
    c = result_processing(a_shp, b_shp, c_files)

    # Check correctness
    gt = (a @ b_stack).todense()
    print("Result: ", np.max(np.abs(gt - c)))


if __name__ == "__main__":
    main()




