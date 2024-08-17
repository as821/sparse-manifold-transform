import os
import numpy as np
import scipy.sparse as sp
from scipy.sparse._data import _data_matrix
from scipy.sparse._base import issparse

from random import randint
import sys
from scipy.sparse.sparsetools import coo_tocsr, csr_todense

from ctypes_interface import coo2csr_param

def mmap_file_init(filename, data, flush=True):
    """Create a file that contains the given data."""
    assert not os.path.exists(filename)
    with open(filename, 'wb') as fid:
        fid.write(data)
        if flush:
            fid.flush()

def mmap_file_load_1d(src_file, src_dtype, src_sz, order='C'):
    assert os.path.exists(src_file)
    t_bytes = np.dtype(src_dtype).itemsize
    if isinstance(src_sz, (tuple)):
        for k in src_sz:
            t_bytes *= k
    else:
        t_bytes *= src_sz
    with open(src_file, 'rb') as fid:
        data = fid.read(t_bytes)
    # print(f"{t_bytes} {len(data)}")
    return np.ndarray(shape=src_sz, dtype=src_dtype, buffer=data, order=order)

def mmap_unified_write(filename, data):
    # Serialize all data to bytes, then write to file. Caller's responsibility to handle bookkeeping for deserialization
    nbytes = sum([d.nbytes for d in data])
    byte_buf = bytearray(nbytes)
    np.concatenate([d.view(np.uint8) for d in data], out=np.ndarray(shape=nbytes, dtype=np.uint8, buffer=byte_buf))
    mmap_file_init(filename, byte_buf, False)

def mmap_unified_write_zero_copy(filename, data):
    assert not os.path.exists(filename)
    with open(filename, 'wb') as fid:
        for d in data:
            fid.write(d)

def mmap_unified_read(src_file, src_dtype_sz_list):
    # Read bytes from file, then deserialize according to src_dtype_sz_list
    # Determine bytes to read
    sz = 0
    nbts = []
    for s in src_dtype_sz_list:
        t_bytes = np.dtype(s[0]).itemsize
        if isinstance(s[1], (tuple)):
            for k in s[1]:
                t_bytes *= k
        else:
            t_bytes *= s[1]
        nbts.append(t_bytes)
        sz += t_bytes

    # Read from file
    with open(src_file, 'rb') as fid:
        data = fid.read(sz)
    data = np.ndarray(shape=sz, dtype=np.uint8, buffer=data, order='C')

    # Deserialize according to src_dtype_sz_list
    out = []
    accum = 0
    for dtype_sz, bts in zip(src_dtype_sz_list, nbts):
        slc = data[accum:accum+bts].view(dtype_sz[0])
        out.append(slc)
        accum += bts
    return tuple(out)

def mmap_coo_arrays_to_csr(args, data, row, col, dtype, shape, mmap=True, verbose=True, disable_c=False):
    # Convert COO mmap to CSR mmap without bringing entire matrix into memory
    if os.path.exists("src/c/bin/coo2csr.so") and row.dtype == np.int64 and col.dtype == np.int64 and not disable_c and mmap:
        # Give preference to C-version of COO to CSR if it exists
        indptr, indices, csr_data = _c_coo_to_csr(data, row, col, shape, True, args.mmap_path)
    else:
        indptr, indices, csr_data = _coo_to_csr(args, data, row, col, mmap=mmap, row_mx=shape[0], verbose=verbose)

    if mmap:
        assert isinstance(csr_data, (np.memmap))
        csr_data.flush()
        indices.flush()
        indptr.flush()
        csr = MemmapCSR((csr_data, indices, indptr), shape=shape, dtype=data.dtype, copy=False)
    else:
        csr = sp.csr_array((csr_data, indices, indptr), dtype=dtype, copy=False, shape=shape)
    return csr

def _coo_to_csr(args, data, row, col, mmap=True, row_mx=None, verbose=True):
    if verbose:
        print("\tConverting mmap COO to CSR...", flush=True)
    n = len(data)
    if n == 0:
        # mmap cannot be empty
        n = 1
    
    if row_mx is None:
        m = row.max() + 1
    else:
        m = row_mx

    # Create binary files for underlying CSR representation (data, indices, indptr)
    if verbose:
        print('\t\tcreating data structures...', flush=True)
    assert row.dtype == col.dtype and (row.dtype == np.int64 or row.dtype == np.int32)
    idx_dtype = row.dtype
    if mmap:
        timestamp = randint(-sys.maxsize, sys.maxsize)
        data_fname = args.mmap_path + f"/coo2csr_data_{timestamp}.bin"
        indices_fname = args.mmap_path + f"/coo2csr_indices_{timestamp}.bin"
        indptr_fname = args.mmap_path + f"/coo2csr_indptr_{timestamp}.bin"
        csr_data = np.memmap(data_fname, dtype=data.dtype, mode="w+", shape=n)
        indices = np.memmap(indices_fname, dtype=idx_dtype, mode="w+", shape=n)
        indptr = np.memmap(indptr_fname, dtype=idx_dtype, mode="w+", shape=m+1)
    else:
        csr_data = np.empty(dtype=data.dtype, shape=n)
        indices = np.empty(dtype=idx_dtype, shape=n)
        indptr = np.empty(dtype=idx_dtype, shape=m+1)
    
    if verbose: 
        print("\t\tconverting...", flush=True)

    # foo = np.bincount(row, minlength=m)
    # assert foo.shape[0] == m

    coo_tocsr(m, n, int(len(data)), row, col, data, indptr, indices, csr_data)

    if mmap:
        indptr.flush()
        indices.flush()
        csr_data.flush()

    return indptr, indices, csr_data

def mmap_csr_cleanup(csr):
    def mmap_cleanup(mmap):
        """Delete + clean up disk usage of a memmap array"""
        assert isinstance(mmap, (np.memmap))
        filename = mmap.filename
        mmap.flush()
        del mmap
        assert os.path.exists(filename)
        os.remove(filename)

    assert isinstance(csr, (sp.csr_array))
    mmap_cleanup(csr.data)
    mmap_cleanup(csr.indices)
    mmap_cleanup(csr.indptr)
    del csr

def _c_coo_to_csr(data, row, col, shp, mmap=False, mmap_path=None):
    # print("Running optimized C code for coo2csr...")

    # Allocate space
    if mmap:
        timestamp = randint(-sys.maxsize, sys.maxsize)
        data_fname = mmap_path + f"/c_coo2csr_data_{timestamp}.bin"
        indices_fname = mmap_path + f"/c_coo2csr_indices_{timestamp}.bin"
        indptr_fname = mmap_path + f"/c_coo2csr_indptr_{timestamp}.bin"
        csr_data = np.memmap(data_fname, dtype=np.float32, mode="w+", shape=data.shape[0])
        csr_indices = np.memmap(indices_fname, dtype=np.int64, mode="w+", shape=data.shape[0])
        csr_indptr = np.memmap(indptr_fname, dtype=np.int64, mode="w+", shape=shp[0]+1)
    else:
        csr_data = np.empty_like(data)
        csr_indices = np.empty_like(row)
        csr_indptr = np.empty(shape=(shp[0]+1,), dtype=np.int64)
    assert csr_data.dtype == np.float32 and csr_indices.dtype == np.int64 and csr_indptr.dtype == np.int64

    coo2csr_param(csr_data, csr_indptr, csr_indices, row, col, data, shp)
    return csr_indptr, csr_indices, csr_data


class File():
    def __init__(self, mmap_path, dtype, max_sz, buf_sz=None, fname=None):
        if fname is not None:
            self.fname = fname
        else:
            assert mmap_path is not None
            self.fname = mmap_path + f"/coo_mmap_{randint(-sys.maxsize, sys.maxsize)}.bin"
        self.ptr = 0
        self.max_sz = max_sz
        self.dtype = dtype
    
        # Create a file of the given size
        assert not os.path.exists(self.fname)
        t_bytes = np.dtype(dtype).itemsize * self.max_sz
        self.fid = open(self.fname, mode="w+b")
        self.fid.seek(t_bytes-1, 0)
        self.fid.write(b'\0')
        self.fid.flush()
        self.fid.seek(0, 0)

        # in-memory buffer to store updates until a large enough disk write makes sense
        if buf_sz is None:
            buf_sz = max(int(max_sz/100), 100000000)
        self.buffer = np.zeros(shape=int(buf_sz), dtype=dtype)
        assert self.buffer.shape[0] > 0
        self.buf_idx = 0

    def _buf_flush(self):
        self.fid.write(self.buffer[:self.buf_idx])
        self.fid.flush()
        self.ptr += self.buf_idx
        self.buf_idx = 0

    def _buf_insert(self, data):
        assert len(data.shape) == 1, "only accepts 1d input"
        if data.shape[0] + self.buf_idx >= self.buffer.shape[0]:
            self._buf_flush()
        if self.buf_idx + data.shape[0] >= self.buffer.shape[0]:
            # buffer too small, write to file directly
            assert self.buf_idx == 0
            self.fid.write(data)
            self.fid.flush()
            self.ptr += data.shape[0]
        else:
            assert self.buf_idx + data.shape[0] < self.buffer.shape[0]
            self.buffer[self.buf_idx:self.buf_idx+data.shape[0]] = data
            self.buf_idx += data.shape[0]

    def update(self, data):
        assert self.ptr + self.buf_idx + data.shape[0] <= self.max_sz, f"mmap too small: {self.max_sz} {data.shape}"
        self._buf_insert(data)

    def get_mmap(self):
        self._buf_flush()
        return np.memmap(self.fname, dtype=self.dtype, mode="r+", shape=(self.ptr,))
    
    def fid_close(self):
        assert self.buf_idx == 0
        self.fid.flush()
        self.fid.close()
        self.fid = None

    def cleanup(self):
        self.fid_close()
        os.remove(self.fname)




class MemmapCSR(sp.csr_array):
    # A version of scipy's CSR data type that does not page in the full array when a memmap is passed in and 
    # does not silently alter the given indptr/indices index data types.

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        _data_matrix.__init__(self)
        if issparse(arg1):
            self._set_self(arg1)
            return
        assert isinstance(arg1, tuple) and len(arg1) == 3
        (data, indices, indptr) = arg1
        self.indices = indices
        self.indptr = indptr
        self.data = data
        self.set_shape(shape)

    def set_shape(self, shape):
        self._shape = shape

    def todense(self):
        out = np.zeros(shape=self.shape)
        csr_todense(self.shape[0], self.shape[1], self.indptr, self.indices, self.data, out)
        return out


class Buffer():
    # Simple in-memory buffer (wrapper around an array and index counter)
    def __init__(self, dtype, max_sz):
        self.ptr = 0
        self.max_sz = max_sz
        self.buffer = np.zeros(shape=self.max_sz, dtype=dtype)
        assert self.buffer.shape[0] > 0
        self.buf_idx = 0

    def update(self, data):
        assert len(data.shape) == 1, "only accepts 1d input"
        assert self.buf_idx + data.shape[0] < self.buffer.shape[0]
        self.buffer[self.buf_idx:self.buf_idx+data.shape[0]] = data
        self.buf_idx += data.shape[0]

    def reset(self):
        self.buf_idx = 0

    def get(self):
        return self.buffer[:self.buf_idx]







