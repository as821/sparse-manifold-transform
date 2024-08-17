import ctypes

class CSRSerialize(ctypes.Structure):
    _fields_ = [
                ("fname", ctypes.c_char_p),
                ("nnz", ctypes.c_ssize_t),
                ("nrow", ctypes.c_ssize_t),
                ("ncol", ctypes.c_ssize_t),
                ("start", ctypes.c_ssize_t),
                ("end", ctypes.c_ssize_t)]
    
class COOResultSerialize(ctypes.Structure):
    _fields_ = [
                ("data_fname", ctypes.c_char_p),
                ("row_fname", ctypes.c_char_p),
                ("col_fname", ctypes.c_char_p),
                ("nnz", ctypes.c_ssize_t),
                ("nrow", ctypes.c_ssize_t),
                ("ncol", ctypes.c_ssize_t)
                ]

class DenseSerialize(ctypes.Structure):
    _fields_ = [
                ("fname", ctypes.c_char_p),
                ("nrow", ctypes.c_ssize_t),
                ("ncol", ctypes.c_ssize_t),
                ("start", ctypes.c_ssize_t),
                ("end", ctypes.c_ssize_t)]

class CSRMultiFileSerialize(ctypes.Structure):
    _fields_ = [
                ("data_fname", ctypes.c_char_p),
                ("indices_fname", ctypes.c_char_p),
                ("indptr_fname", ctypes.c_char_p),
                ("nnz", ctypes.c_ssize_t),
                ("nrow", ctypes.c_ssize_t),
                ("ncol", ctypes.c_ssize_t)
                ]
                
class CSRSliceSerialize(ctypes.Structure):
    _fields_ = [
                ("data_fname", ctypes.c_char_p),
                ("indices_fname", ctypes.c_char_p),
                ("indptr_fname", ctypes.c_char_p),
                ("nnz", ctypes.c_ssize_t),
                ("nrow", ctypes.c_ssize_t),
                ("ncol", ctypes.c_ssize_t),
                ("start", ctypes.c_ssize_t),
                ("end", ctypes.c_ssize_t),
                ("col_start", ctypes.c_ssize_t),
                ("col_end", ctypes.c_ssize_t)
                ]
                
class CSRSliceSerialize64(ctypes.Structure):
    _fields_ = [
                ("data_fname", ctypes.c_char_p),
                ("indices_fname", ctypes.c_char_p),
                ("indptr_fname", ctypes.c_char_p),
                ("nnz", ctypes.c_ssize_t),
                ("nrow", ctypes.c_ssize_t),
                ("ncol", ctypes.c_ssize_t),
                ("start", ctypes.c_ssize_t),
                ("end", ctypes.c_ssize_t)
                ]

import os
if os.path.exists("src/c/bin/cuda_c_func.so"):
    c_lib = ctypes.CDLL("src/c/bin/cuda_c_func.so")
    c_lib.spgemm_batched_matmul.argtypes = (ctypes.POINTER(CSRSerialize), ctypes.POINTER(ctypes.POINTER(CSRSerialize)), ctypes.POINTER(ctypes.c_char_p), ctypes.c_int)
    c_lib.spmm_batched.argtypes = (ctypes.POINTER(DenseSerialize), ctypes.POINTER(ctypes.POINTER(CSRSerialize)), ctypes.POINTER(ctypes.c_char_p), ctypes.c_int)
    c_lib.spgemm_batched_matmul_dop.argtypes = (ctypes.POINTER(ctypes.POINTER(CSRSerialize)), ctypes.POINTER(ctypes.POINTER(CSRSerialize)), ctypes.c_int, ctypes.POINTER(ctypes.c_char_p))

if os.path.exists("src/c/bin/coo2csr.so"):
    coo2csr_lib = ctypes.CDLL("src/c/bin/coo2csr.so")
    coo2csr_lib.coo2csr_param.argtypes = (ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong, \
                                ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_float), \
                                ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_float))

if os.path.exists("src/c/bin/csr_column_slice.so"):
    col_slice = ctypes.CDLL("src/c/bin/csr_column_slice.so")
    col_slice.csr_col_slice.argtypes = (ctypes.POINTER(CSRSliceSerialize64),  ctypes.POINTER(ctypes.POINTER(CSRSliceSerialize)), ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.c_bool)
    col_slice.csr_col_slice_param.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_longlong), ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.POINTER(ctypes.POINTER(CSRSliceSerialize)), ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.c_ssize_t)


def spgemm_batched_matmul_c(a_serial, b_serials, c_files):
    arr = (ctypes.POINTER(CSRSerialize) * len(b_serials))(*[ctypes.pointer(b_serial) for b_serial in b_serials])
    c_files = [str.encode(c) for c in c_files]
    c_arr = (ctypes.c_char_p * len(c_files))(*c_files)
    c_lib.spgemm_batched_matmul(ctypes.byref(a_serial), arr, c_arr, len(b_serials))


def spmm_batched(a_serial, b_serials, c_files):
    arr = (ctypes.POINTER(CSRSerialize) * len(b_serials))(*[ctypes.pointer(b_serial) for b_serial in b_serials])
    c_files = [str.encode(c) for c in c_files]
    c_arr = (ctypes.c_char_p * len(c_files))(*c_files)
    c_lib.spmm_batched(ctypes.byref(a_serial), arr, c_arr, len(b_serials))

def spgemm_batched_matmul_dop(a_serials, b_serials, data_files):
    assert len(a_serials) == len(b_serials)
    a_arr = (ctypes.POINTER(CSRSerialize) * len(a_serials))(*[ctypes.pointer(a_serial) for a_serial in a_serials])
    b_arr = (ctypes.POINTER(CSRSerialize) * len(b_serials))(*[ctypes.pointer(b_serial) for b_serial in b_serials])
    data_files = [str.encode(c) for c in data_files]
    data_files_arr = (ctypes.c_char_p * len(b_serials))(*data_files)
    c_lib.spgemm_batched_matmul_dop(a_arr, b_arr, len(b_serials), data_files_arr)


def coo2csr_param(csr_data, csr_indptr, csr_indices, row, col, data, shp):
    csr_data_p = csr_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    csr_indptr_p = csr_indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
    csr_indices_p = csr_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
    coo_row_p = row.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
    coo_col_p = col.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
    coo_data_p = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    nnz = data.shape[0]
    coo2csr_lib.coo2csr_param(shp[0], shp[1], nnz, coo_row_p, coo_col_p, coo_data_p, csr_indices_p, csr_indptr_p, csr_data_p)

def csr_col_slice_param_c(csr, res_serial, chunk, buf_len):
    arr = (ctypes.POINTER(CSRSliceSerialize) * len(res_serial))(*[ctypes.pointer(serial) for serial in res_serial])
    csr_data_p = csr.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    csr_indptr_p = csr.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
    csr_indices_p = csr.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
    col_slice.csr_col_slice_param(csr_data_p, csr_indices_p, csr_indptr_p, csr.nnz, csr.shape[0], csr.shape[1], arr, len(res_serial), chunk, buf_len)    

def csr_col_slice_c(csr_serial, res_serial, chunk, buf_len, row_slice):
    arr = (ctypes.POINTER(CSRSliceSerialize) * len(res_serial))(*[ctypes.pointer(serial) for serial in res_serial])
    col_slice.csr_col_slice(ctypes.byref(csr_serial), arr, len(res_serial), chunk, buf_len, row_slice)


