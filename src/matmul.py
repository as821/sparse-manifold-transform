import sys
import os
import torch
from queue import Empty
import numpy as np
import scipy.sparse as sp
from random import randint

from ctypes_interface import spgemm_batched_matmul_c, spmm_batched, DenseSerialize, c_impl_available
from multiproc import MultiProcessDispatch
from input_output import mmap_file_init, mmap_file_load_1d

class MatmulWorkSlice:
    def __init__(self, start, end, a, n_col, a_dense):
        self.start = start
        self.end = end
        self.a = a
        self.n_col = n_col
        self.a_dense = a_dense
    
    def process(self, args):
        b_cache, mmap_path, sym = args
        result = np.zeros(shape=(self.end-self.start, self.n_col))

        if c_impl_available():
            a_serial = self.a.serialize()
            b_serials = []
            for b_chunk in b_cache:
                (b_start, b_end) = b_chunk
                if sym:
                    # NOTE: symmetric matrix optimization disabled
                    bs = b_cache[b_chunk][self.a.col_start]
                else:
                    bs = b_cache[b_chunk]
                bs = bs.serialize()
                b_serials.append((b_start, bs))
            b_serials = [b[1] for b in sorted(b_serials, key=lambda x: x[0])]
            c_files = [mmap_path + f"/sym_work_slice_{randint(-sys.maxsize, sys.maxsize)}.bin" for _ in range(len(b_serials))]
            if self.a_dense:
                spmm_batched(a_serial, b_serials, c_files)
            else:
                spgemm_batched_matmul_c(a_serial, b_serials, c_files)

            # Coalesce all result files into a single dense array so the ResultSlice interface can still be used
            for b_serial, c_fname in zip(b_serials, c_files):
                if self.a_dense:
                    shp = (b_serial.nrow, self.a.shape[1])
                    res = mmap_file_load_1d(c_fname, np.float32, shp).T
                else:
                    shp = (self.a.shape[0], b_serial.ncol)
                    res = mmap_file_load_1d(c_fname, np.float32, shp)
                result[:, b_serial.start:b_serial.end] = res
                os.remove(c_fname)
        else:
            a_slice = self.a.get()
            for b_chunk in b_cache:
                # NOTE: symmetric matrix optimization disabled
                b_slice = b_cache[b_chunk]
                if sym:
                    b_slice = b_slice[self.a.col_start]
                    result[:, b_slice.start:b_slice.end] = (a_slice.todense() @ b_slice.get().todense())
                else:
                    assert self.a_dense
                    result[:, b_slice.start:b_slice.end] = (b_slice.get().todense() @ a_slice).T
        self.a.cleanup()
        return MatmulResultSlice(self.start, self.end, result)


class MatmulResultSlice:
    def __init__(self, start, end, result):
        self.start = start
        self.end = end
        self.result = result

class GpuSparseMatmul(MultiProcessDispatch):
    def __init__(self, a, b, symmetric, batch_sz, dense_matmul, mmap_path, a_shape=None):
        super().__init__(_matmul_worker)
        self.symmetric = symmetric
        self.a = a
        self.a_shape = a_shape
        self.b = b
        self.csc_caching = isinstance(b, (dict))
        self.batch_sz = batch_sz
        self.dense = dense_matmul
        self.mmap_path = mmap_path
        shp = a_shape[0] if a_shape is not None else a.shape[0]
        if self.dense and not symmetric:
            b_cols = a_shape[1]     # hacky
            self.result = np.zeros(shape=(a.shape[0], b_cols))
        elif self.csc_caching or self.dense:
            self.result = np.zeros(shape=(shp, shp), dtype=np.float32)
        else:
            assert not self.dense, "Dense sym. matmul assumes CSC caching."
            self.result = np.zeros(shape=(shp, b.shape[1]), dtype=a.dtype)

    def initial_assertions(self):
        if isinstance(self.a, (sp.csr_array)):
            assert isinstance(self.a.data, (np.memmap))
        elif isinstance(self.a, (dict)):
            return
        else:
            assert isinstance(self.a, (np.memmap))
        if not self.csc_caching:
            assert isinstance(self.b.data, (np.memmap))
            assert self.a.dtype == self.b.dtype
            # assert isinstance(self.b, (sp.csc_array))

    def worker_init_args(self):
        return (self.b, self.mmap_path, self.symmetric)
    
    def get_nbatch(self):
        n_batch = int(self.result.shape[0] / self.batch_sz)
        assert self.batch_sz > 1, "modulus logic below can fail if batch size == 1"
        if self.result.shape[0] % self.batch_sz != 0:
            n_batch += 1
        return n_batch
    
    def process_result(self, res):
        self.result[res.start : res.end, :] += res.result
    
    def generate_work(self, idx):
        start = self.batch_sz * idx
        end = min(start + self.batch_sz, self.result.shape[0])
        if isinstance(self.a, dict):
            return [MatmulWorkSlice(start, end, a, self.result.shape[1], a_dense=self.dense) for a in self.a[(start, end)]]
        else:
            # NOTE: self.a is a dense matrix, effectively does the same thing here as 2D alphas slice caching
            assert isinstance(self.a, np.memmap)
            a_slice = self.a[start:end, :].T        # NOTE: same as slicing the columns of self.a.T
            a_pkg = DenseSliceCacheEntry(self.mmap_path, start, end, a_slice.copy(order='C'), 0, self.a.shape[1])       # TODO(as) is this the correct col_end?
            return [MatmulWorkSlice(start, end, a_pkg, self.result.shape[1], a_dense=self.dense)]

    def fast_work_gen(self):
        return True

    def num_processes(self):
        return 1



class DenseSliceCacheEntry():
    def __init__(self, mmap_path, start, end, data, col_start=None, col_end=None):
        assert data.dtype == np.float32
        self.start = start
        self.end = end
        self.col_start = col_start
        self.col_end = col_end
        self.fname = mmap_path + f"/dense_slice_cache_unified_{randint(-sys.maxsize, sys.maxsize)}.bin"
        self.dtype = data.dtype
        self.shape = data.shape
        mmap_file_init(self.fname, data)

    def get(self):
        return mmap_file_load_1d(self.fname, self.dtype, self.shape, order='C').reshape(self.shape)

    def serialize(self):
        assert self.dtype == np.float32
        return DenseSerialize(str.encode(self.fname), self.shape[0], self.shape[1], self.start, self.end)

    def cleanup(self):
        os.remove(self.fname)


def _matmul_worker(w_q, r_q, w_done, i, w_init_args):
    # Load work from the work queue, put results onto the result queue
    def _proc(work_q, res_q, work_done, idx, worker_init_args):
        while not (work_done.is_set() and work_q.empty()):
            try:
                work_slice = work_q.get(block=True, timeout=0.5)
            except Empty:
                continue
            result = work_slice.process(worker_init_args)
            res_q.put(result)
    
    _proc(w_q, r_q, w_done, i, w_init_args)
    sys.exit(0)

