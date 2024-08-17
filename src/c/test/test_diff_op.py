import numpy as np
import scipy.sparse as sp
from random import randint

import os
from tqdm import tqdm
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))

from input_output import mmap_unified_write, mmap_file_load_1d


EXE_PATH="src/c/bin/cuda_c_func.so"
MMAP_PATH="/ssd1/smt-mmap"

import ctypes
c_lib = ctypes.CDLL(EXE_PATH)
from ctypes_interface import CSRSerialize
c_lib.spgemm_batched_matmul_dop.argtypes = (ctypes.POINTER(ctypes.POINTER(CSRSerialize)), ctypes.POINTER(ctypes.POINTER(CSRSerialize)), ctypes.c_int, ctypes.POINTER(ctypes.c_char_p))
c_lib.csr_postproc.argtypes = (ctypes.POINTER(ctypes.c_char_p), ctypes.c_char_p, ctypes.c_longlong, ctypes.c_longlong, ctypes.POINTER(ctypes.c_longlong))

def csr_write(csr):
    a_fname = MMAP_PATH + f"/test_spgemm_{randint(-sys.maxsize, sys.maxsize)}.bin"
    mmap_unified_write(a_fname, (csr.data, csr.indices, csr.indptr))
    return a_fname

def rand_csr(shape, sparsity=0.1):
    # Generate random dense matrix of given shape + sparsity, then convert to dense
    dense = np.zeros(shape=shape, dtype=np.float32)
    nnz = int(shape[0] * shape[1] * sparsity)
    nnz_idx = np.random.choice(shape[0] * shape[1], nnz, replace=False)
    dense.flat[nnz_idx] = np.random.rand(nnz)
    return sp.csr_array(dense, shape=shape)

def main():
    # Create sparse test matrices
    a_shp = (2, 1000)
    a = rand_csr(a_shp)
    b_shp = (1000, 3)
    n_b = 1
    bs = [rand_csr(b_shp) for _ in range(n_b)]

    # Prep to call C exe (format as would be expected in full program)
    a_fname = csr_write(a)
    b_fnames = [csr_write(b) for b in bs]

    data_files = [str.encode(MMAP_PATH + f"/test_diffop_data_{randint(-sys.maxsize, sys.maxsize)}.bin") for _ in range(n_b)]
    data_files_arr = (ctypes.c_char_p * n_b)(*data_files)

    # NOTE(!!!): the order in which these are passed into the C function is the order in which their results will be stacked horizontally
    b_serials = [CSRSerialize(str.encode(b_fname), b.nnz, b.shape[0], b.shape[1], -1, -1) for b_fname, b in zip(b_fnames, bs)]
    arr = (ctypes.POINTER(CSRSerialize) * n_b)(*[ctypes.pointer(b_serial) for b_serial in b_serials])
    a_serials = [CSRSerialize(str.encode(a_fname), a.nnz, a.shape[0], a.shape[1], -1, -1) for _ in b_serials]
    a_arr = (ctypes.POINTER(CSRSerialize) * n_b)(*[ctypes.pointer(a_serial) for a_serial in a_serials])
    c_lib.spgemm_batched_matmul_dop(
        a_arr,
        arr,
        n_b,
        data_files_arr
    )

    b_cols = (ctypes.c_longlong * n_b)(*[b_shp[1] for _ in range(n_b)])
    final_file = str.encode(MMAP_PATH + f"/test_diffop_data_{randint(-sys.maxsize, sys.maxsize)}.bin")
    open(final_file, "wb").close()
    c_lib.csr_postproc(data_files_arr, final_file, n_b, a_shp[0], b_cols)
    final_shp = (n_b * b_shp[1], a_shp[0])      # now returns transposed results...
    c = mmap_file_load_1d(final_file, np.float32, final_shp[0] * final_shp[1]).reshape(final_shp).T

    # c = []
    # for f in data_files:
    #     c.append(mmap_file_load_1d(f, np.float32, a_shp[0] * b_shp[1], order='C').reshape((a_shp[0], b_shp[1])))
    # c = np.concatenate(c, axis=1)

    # Check correctness
    b_stack = sp.hstack(bs, dtype=np.float32)
    gt = (a @ b_stack).todense()
    print("Result: ", np.max(np.abs(gt - c)))

    # print(gt)
    # print("\n\n")
    # print(c)


if __name__ == "__main__":
    main()

