
import os
import sys
import numpy as np
import scipy.sparse as sp
from random import randint
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))

from input_output import mmap_unified_write, mmap_unified_read



EXE_PATH="src/c/bin/coo2csr.so"
MMAP_PATH="/ssd1/smt-mmap"

import ctypes
c_lib = ctypes.CDLL(EXE_PATH)
from ctypes_interface import CSRSerialize
c_lib.coo2csr_param.argtypes = (ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong, \
                                ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_float), \
                                ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_float))

def rand_coo(shape, sparsity=0.15):
    # Generate random dense matrix of given shape + sparsity, then convert to dense
    dense = np.zeros(shape=shape, dtype=np.float32)
    nnz = int(shape[0] * shape[1] * sparsity)
    nnz_idx = np.random.choice(shape[0] * shape[1], nnz, replace=False)
    dense.flat[nnz_idx] = np.random.rand(nnz)
    coo = sp.coo_array(dense, shape=shape, dtype=np.float32)
    coo.row = coo.row.astype(np.int64)
    coo.col = coo.col.astype(np.int64)
    return coo

def coo_write(coo):
    a_fname = MMAP_PATH + f"/test_coo2csr_{randint(-sys.maxsize, sys.maxsize)}.bin"
    mmap_unified_write(a_fname, (coo.data, coo.row, coo.col))
    return CSRSerialize(a_fname.encode(), coo.nnz, coo.shape[0], coo.shape[1], -1, -1)

def csr_read(coo, fname):
    # Assumes (data, indices, indptr) serialization + int64 index types
    dtype_sz = ((coo.data.dtype, coo.nnz), (np.int64, coo.nnz), (np.int64, coo.shape[0]+1))
    return sp.csr_array(mmap_unified_read(fname, dtype_sz), dtype=coo.dtype, shape=coo.shape)

def csr_alloc(coo):
    data = np.zeros_like(coo.data)
    indices = np.zeros_like(coo.row)
    indptr = np.zeros(shape=(coo.shape[0]+1,), dtype=np.int64)
    return data, indices, indptr


def main():
    # Create sparse test matrices
    a_shp = (20000, 1000)
    a = rand_coo(a_shp)


    csr_data, csr_indices, csr_indptr = csr_alloc(a)

    csr_data_p = csr_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    csr_indptr_p = csr_indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
    csr_indices_p = csr_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
    coo_row_p = a.row.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
    coo_col_p = a.col.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
    coo_data_p = a.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    c_lib.coo2csr_param(a.shape[0], a.shape[1], a.nnz, coo_row_p, coo_col_p, coo_data_p, csr_indices_p, csr_indptr_p, csr_data_p)
    res = sp.csr_array((csr_data, csr_indices, csr_indptr), shape=a.shape, dtype=a.dtype, copy=False)

    # Check correctness
    a_csr = a.tocsr()
    print("Max. diff: ", np.max(np.abs(a_csr - res)))


if __name__ == "__main__":
    main()



