
import os
import sys
import numpy as np
import scipy.sparse as sp
from random import randint
from tqdm import tqdm
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))

from input_output import mmap_file_init, mmap_unified_read, mmap_unified_write

EXE_PATH="src/c/bin/csr_column_slice.so"
MMAP_PATH="/ssd1/smt-mmap"

import ctypes
c_lib = ctypes.CDLL(EXE_PATH)
from ctypes_interface import CSRSliceSerialize, CSRSliceSerialize64
c_lib.csr_col_slice.argtypes = (ctypes.POINTER(CSRSliceSerialize64),  ctypes.POINTER(ctypes.POINTER(CSRSliceSerialize)), ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.c_bool)
c_lib.csr_col_slice_param.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_longlong), ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.POINTER(ctypes.POINTER(CSRSliceSerialize)), ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.c_ssize_t)


def csr_write(csr):
    data_fname = MMAP_PATH + f"/test_slice_{randint(-sys.maxsize, sys.maxsize)}_data.bin"
    indices_fname = MMAP_PATH + f"/test_slice_{randint(-sys.maxsize, sys.maxsize)}_indices.bin"
    indptr_fname = MMAP_PATH + f"/test_slice_{randint(-sys.maxsize, sys.maxsize)}_indptr.bin"
    csr.indptr = csr.indptr.astype(np.int64)
    csr.indices = csr.indices.astype(np.int64)


    print(f"NOTE (python): {csr.nnz} {csr.shape} {csr.data.shape} {csr.indptr.shape} {csr.indices.shape}")
    print(f"\tNOTE (python): {data_fname} {indices_fname} {indptr_fname}")
    mmap_file_init(data_fname, csr.data)
    mmap_file_init(indices_fname, csr.indices)
    mmap_file_init(indptr_fname, csr.indptr)
    return CSRSliceSerialize64(data_fname.encode(), indices_fname.encode(), indptr_fname.encode(), csr.nnz, csr.shape[0], csr.shape[1], -1, -1)

    # fname = MMAP_PATH + f"/test_slice_{randint(-sys.maxsize, sys.maxsize)}.bin"
    # mmap_unified_write(fname, (csr.data, csr.indices, csr.indptr))
    # return CSRSerialize(fname.encode(), csr.nnz, csr.shape[0], csr.shape[1], -1, -1)

def rand_csr(shape, sparsity=0.1):
    # Generate random dense matrix of given shape + sparsity
    dense = np.zeros(shape=shape, dtype=np.float32)
    nnz = int(shape[0] * shape[1] * sparsity)
    nnz_idx = np.random.choice(shape[0] * shape[1], nnz, replace=False)
    dense.flat[nnz_idx] = np.random.rand(nnz)
    out = sp.csr_array(dense, shape=shape)
    out.indices = out.indices.astype(np.int32)
    out.indptr = out.indptr.astype(np.int32)
    return out

def csr_serial_read(serial):
    # # print("data...")
    # data = mmap_file_load_1d(serial.data_fname, np.float32, serial.nnz)
    # # print("indices...")
    # indices = mmap_file_load_1d(serial.indices_fname, np.int32, serial.nnz)
    # # print("indptr...")
    # indptr = mmap_file_load_1d(serial.indptr_fname, np.int32, serial.nrow+1)
    # return sp.csr_array((data, indices, indptr), dtype=np.float32, shape=(serial.nrow, serial.ncol))
    dtype_sz = ((np.float32, serial.nnz), (np.int32, serial.nnz), (np.int32, serial.nrow+1))
    return sp.csr_array(mmap_unified_read(serial.data_fname, dtype_sz), dtype=np.float32, shape=(serial.nrow, serial.ncol), copy=False)


def main():
    # Create sparse test matrix
    a_shp = (10, 5000000)
    a = rand_csr(a_shp, sparsity=0.1)
    col_chunk = 5000
    buf_len = 10000000
    debug_log = False


    # serialize info 
    a_serial = csr_write(a)
    res_serial = []
    for start in range(0, a.shape[1], col_chunk):
        end = min(start + col_chunk, a.shape[1])
        tstamp = randint(-sys.maxsize, sys.maxsize)
        d = str.encode(MMAP_PATH + f"/test_spgemm_res_{tstamp}_data.bin")
        ptr = str.encode(MMAP_PATH + f"/test_spgemm_res_{tstamp}_indptr.bin")
        ind = str.encode(MMAP_PATH + f"/test_spgemm_res_{tstamp}_indices.bin")
        res_serial.append(CSRSliceSerialize(d, ind, ptr, -1, a.shape[0], end-start, start, end))
    arr = (ctypes.POINTER(CSRSliceSerialize) * len(res_serial))(*[ctypes.pointer(serial) for serial in res_serial])

    if debug_log:
        print(f"indptr: {a.indptr}")


    # call C function
    c_lib.csr_col_slice(ctypes.byref(a_serial), arr, len(res_serial), col_chunk, buf_len, False)

    # ground truth generation
    gt = []
    for start in tqdm(range(0, a.shape[1], col_chunk)):
        end = min(start + col_chunk, a.shape[1])
        gt.append(a[:, start:end])

    total = 0
    if debug_log:
        print(f"A: {a.todense()}\n")
    for ga, serial in zip(gt, res_serial):
        ca = csr_serial_read(serial)
        g = ga.todense()
        c = ca.todense()
        diff = np.abs(g - c).max()
        total = max(diff, total)
        if debug_log:
            print(f"\t{diff}:")
            print(f"\t\t{ga.data}\t\t{ca.data}")
            print(f"\t\t{ga.indices}\t\t{ca.indices}")
            print(f"\t\t{ga.indptr}\t\t{ca.indptr}")


    print(f"\n\nMax error: {total}")





    res_serial = []
    for start in range(0, a.shape[1], col_chunk):
        end = min(start + col_chunk, a.shape[1])
        tstamp = randint(-sys.maxsize, sys.maxsize)
        d = str.encode(MMAP_PATH + f"/test_spgemm_res_{tstamp}_data.bin")
        ptr = str.encode(MMAP_PATH + f"/test_spgemm_res_{tstamp}_indptr.bin")
        ind = str.encode(MMAP_PATH + f"/test_spgemm_res_{tstamp}_indices.bin")
        res_serial.append(CSRSliceSerialize(d, ind, ptr, -1, a.shape[0], end-start, start, end))
    arr = (ctypes.POINTER(CSRSliceSerialize) * len(res_serial))(*[ctypes.pointer(serial) for serial in res_serial])

    # call C function

    csr_data_p = a.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    csr_indptr_p = a.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
    csr_indices_p = a.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
    c_lib.csr_col_slice_param(csr_data_p, csr_indices_p, csr_indptr_p, a.nnz, a.shape[0], a.shape[1], arr, len(res_serial), col_chunk, buf_len)
    
    total = 0
    for ga, serial in zip(gt, res_serial):
        ca = csr_serial_read(serial)
        g = ga.todense()
        c = ca.todense()
        diff = np.abs(g - c).max()
        total = max(diff, total)
    print(f"\n\nMax error (param): {total}")



main()



