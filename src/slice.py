import os
import sys
import time
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
from random import randint
from queue import Empty


from input_output import mmap_unified_read, mmap_unified_write, mmap_unified_write_zero_copy, MemmapCSR
from matrix_utils import csr_row_view, profile_log
from multiproc import MultiProcessDispatch
from ctypes_interface import CSRSerialize, CSRSliceSerialize, CSRSliceSerialize64, csr_col_slice_param_c, csr_col_slice_c, c_impl_available



class SliceCacheEntry():
    def __init__(self, args, start, end, csr, col_start=None, col_end=None):
        self.start = start
        self.end = end

        self.col_start = col_start
        self.col_end = col_end

        assert csr.data.shape[0] <= csr.shape[0] * csr.shape[1], f"{csr.data.shape[0]} {csr.shape} {col_start} {col_end}"
        assert csr.indices.dtype == np.int32 and csr.indptr.dtype == np.int32 and csr.data.dtype == np.float32

        fname = args.mmap_path + f"/slice_unified_{randint(-sys.maxsize, sys.maxsize)}.bin"
        mmap_unified_write(fname, (csr.data, csr.indices, csr.indptr))

        self.fname = fname
        self.dtype_sz = ((csr.data.dtype, csr.data.shape), (csr.indices.dtype, csr.indices.shape), (csr.indptr.dtype, csr.indptr.shape))
        self.shape = csr.shape

    def get(self):
        return sp.csr_array(mmap_unified_read(self.fname, self.dtype_sz), dtype=self.dtype_sz[0][0], copy=False, shape=self.shape)

    def serialize(self):
        assert self.dtype_sz[0][0] == np.float32 and self.dtype_sz[1][0] == np.int32 and self.dtype_sz[2][0] == np.int32
        nnz = self.dtype_sz[0][1][0]        # nnz from data array shape
        assert nnz <= self.shape[0] * self.shape[1]
        return CSRSerialize(self.fname.encode(), nnz, self.shape[0], self.shape[1], self.start, self.end)

    def cleanup(self):
        os.remove(self.fname)

class SliceCacheWorkSlice():
    def __init__(self, start, end, transpose_2d_slice, alphas_2d_slice, col_slice, batch_sz_2d=None):
        self.start = start
        self.end = end
        self.transpose_2d_slice = transpose_2d_slice
        self.alphas_2d_slice = alphas_2d_slice
        self.col_slice = col_slice
        self.batch_sz_2d = batch_sz_2d

    def process(self, args, alphas, means):        
        assert means is None, "Not implemented"
        profile = False
        running = time.time()
        if self.col_slice:
            csr_slice = alphas[:, self.start:self.end]
            if self.transpose_2d_slice:
                csr_slice = csr_slice.T.tocsr()
            assert csr_slice.indptr[-1] < np.iinfo(np.int32).max, "cupy implicitly down-casts indptr+indices arrays to int32 + CUSPARSE/spgemm requires int32 indices"
            csr_slice.indptr = csr_slice.indptr.astype(np.int32)
            csr_slice.indices = csr_slice.indices.astype(np.int32)
            # print(f"col slice ({self.start}:{self.end}) sparsity: {csr_slice.nnz / (csr_slice.shape[0] * csr_slice.shape[1])}", flush=True)
            running = profile_log(profile, running, f"({self.start}): slice")

            # note where nonzero rows start/end in the slice...
            nnz_row_start = csr_slice.indptr.nonzero()[0][0] - 1
            nnz_row_end = (csr_slice.indptr - csr_slice.indptr[-1]).nonzero()[0][-1] + 1
            if not self.transpose_2d_slice:
                csr_slice = csr_slice[nnz_row_start:(nnz_row_end+1)]
            entry = SliceCacheEntry(args, self.start, self.end, csr_slice, nnz_row_start, nnz_row_end)
            running = profile_log(profile, running, f"({self.start}): entry")            
            return entry
        elif self.transpose_2d_slice:
            csr_slice = csr_row_view(alphas, self.start, self.end)
            running = profile_log(profile, running, f"({self.start}): row")

            csr_slice = csr_slice.T.tocsr()
            # print(f"\tTransposed input in: {time.time() - running}")

            out = []
            for start in range(0, csr_slice.shape[0], self.batch_sz_2d):
                end = min(start+self.batch_sz_2d, csr_slice.shape[0])
                slc = csr_row_view(csr_slice, start, end)
                
                assert slc.indptr[-1] < np.iinfo(np.int32).max, f"cupy implicitly down-casts indptr+indices arrays to int32 + CUSPARSE/spgemm requires int32 indices ({alphas.shape} {slc.shape} {slc.indptr[-1]} {np.iinfo(np.int32).max})"
                slc.indptr = slc.indptr.astype(np.int32)
                slc.indices = slc.indices.astype(np.int32)
                
                assert slc.data.shape[0] <= slc.shape[0] * slc.shape[1], f"{slc.data.shape[0]} {slc.shape} {self.start} {self.end} {start} {end} {slc.nnz}\n{csr_slice.data.shape[0]} {csr_slice.shape} {csr_slice.nnz}"
                entry = SliceCacheEntry(args, self.start, self.end, slc, col_start=start, col_end=end)  # actually "row" start, but who cares...
                out.append(entry)
            return out
        elif self.alphas_2d_slice:
            # Performs 2D slicing along single set of rows + then batches across all columns
            slc = csr_row_view(alphas, self.start, self.end)  
            if c_impl_available():
                tmp = csr_col_slice_param(args, slc, self.batch_sz_2d)
                out = []
                for o in tmp:
                    # print(f"{self.start} <-> {self.end}: {o.shape} {o.start} {o.end} {o.col_start} {o.col_end} {slc.shape}")
                    o.reset_shape(self.start, self.end)       # overwrites col_start, but should not have been set by csr_col_slice_param in the first place
                    out.append(o)
            else:            
                out = []
                for start in range(0, slc.shape[1], self.batch_sz_2d):
                    end = min(start+self.batch_sz_2d, slc.shape[1])
                    # print(f"\t{self.start}: col slicing {start} <-> {end}")
                    csr_slice = slc[:, start:end]

                    assert csr_slice.indptr[-1] < np.iinfo(np.int32).max, f"cupy implicitly down-casts indptr+indices arrays to int32 + CUSPARSE/spgemm requires int32 indices: {csr_slice.indptr[-1]} < {np.iinfo(np.int32).max} {csr_slice.shape}"
                    csr_slice.indptr = csr_slice.indptr.astype(np.int32)
                    csr_slice.indices = csr_slice.indices.astype(np.int32)
                    assert csr_slice.data.shape[0] <= csr_slice.shape[0] * csr_slice.shape[1], f"{csr_slice.data.shape[0]} {csr_slice.shape} {self.start} {self.end} {csr_slice.nnz}\n{alphas.data.shape[0]} {alphas.shape} {alphas.nnz}"
                    
                    entry = SliceCacheEntry(args, self.start, self.end, csr_slice, col_start=start, col_end=end)
                    out.append(entry)
            return out
        else:
            raise NotImplementedError

def _matmul_cache_gen_worker(w_q, r_q, w_done, idx, worker_init_args):
    # Load alphas mmap into the worker
    data_pkg, indptr_pkg, indices_pkg, dtype, shape, a, m = worker_init_args
    data = np.memmap(data_pkg[0], dtype=data_pkg[1], mode="r", shape=data_pkg[2])
    indptr = np.memmap(indptr_pkg[0], dtype=indptr_pkg[1], mode="r", shape=indptr_pkg[2])
    indices = np.memmap(indices_pkg[0], dtype=indices_pkg[1], mode="r", shape=indices_pkg[2])
    alphas = MemmapCSR((data, indices, indptr), shape=shape, dtype=data.dtype, copy=False)
    assert alphas.nnz <= alphas.shape[0] * alphas.shape[1], f"{alphas.nnz} {alphas.shape}"

    # Load work from the work queue, put results onto the result queue
    def proc(work_q, res_q, work_done, args, means):
        while not (work_done.is_set() and work_q.empty()):
            try:
                work_slice = work_q.get(block=True, timeout=0.5)
            except Empty:
                continue
            
            result = work_slice.process(args, alphas, means)
            res_q.put(result)

    proc(w_q, r_q, w_done, a, m)
    sys.exit(0)


class PreMatmulCacheGen(MultiProcessDispatch):
    def __init__(self, args, alphas, batch_sz, transpose_2d_slice=False, alphas_2d_slice=False, col_slice=False, batch_sz_2d=None, means=None):
        super().__init__(_matmul_cache_gen_worker)
        self.args = args
        assert alphas.nnz <= alphas.shape[0] * alphas.shape[1], f"{alphas.nnz} {alphas.shape}"
        self.alpha_pkg = ((alphas.data.filename, alphas.data.dtype, alphas.data.shape), 
                          (alphas.indptr.filename, alphas.indptr.dtype, alphas.indptr.shape), 
                          (alphas.indices.filename, alphas.indices.dtype, alphas.indices.shape), 
                          alphas.dtype, alphas.shape, args, means)
        self.alpha_shape = alphas.shape
        self.batch_sz = batch_sz
        self.cache = {}

        # work-type flags
        self.transpose_2d_slice = transpose_2d_slice
        self.batch_sz_2d = batch_sz_2d
        self.alphas_2d_slice = alphas_2d_slice
        self.col_slice = col_slice

    def worker_init_args(self):
        return self.alpha_pkg
    
    def get_nbatch(self):
        if self.col_slice and self.transpose_2d_slice:
            r_end = self.alpha_shape[1]
        elif self.transpose_2d_slice or self.alphas_2d_slice:
            r_end = self.alpha_shape[0]
        else:
            r_end = self.alpha_shape[1]
        n_batch = int(r_end / self.batch_sz)
        assert self.batch_sz > 1, "modulus logic below can fail if batch size == 1"
        if r_end % self.batch_sz != 0:
            n_batch += 1
        return n_batch
    
    def process_result(self, res):
        if self.col_slice:
            self.cache[(res.start, res.end)] = res
        elif self.transpose_2d_slice:
            k = (res[0].start, res[0].end)
            assert k not in self.cache
            self.cache[k] = {r.col_start : r for r in res}
        elif self.alphas_2d_slice:
            k = (res[0].start, res[0].end)
            assert k not in self.cache
            self.cache[k] = res
        else:
            raise NotImplementedError
    
    def generate_work(self, idx):
        start = idx * self.batch_sz
        if (self.transpose_2d_slice or self.alphas_2d_slice) and not self.col_slice:
            r_end = self.alpha_shape[0]
        else:
            r_end = self.alpha_shape[1]
        return SliceCacheWorkSlice(start, min(start + self.batch_sz, r_end), self.transpose_2d_slice, self.alphas_2d_slice, col_slice=self.col_slice, batch_sz_2d=self.batch_sz_2d)

    def num_processes(self):
        return self.args.proj_cache_proc
    
    def fast_work_gen(self):
        return True

class SliceCacheEntry_C():
    def __init__(self, serial):
        self.start = serial.start
        self.end = serial.end
        self.fname = serial.data_fname
        self.dtype_sz = ((np.float32, serial.nnz), (np.int32, serial.nnz), (np.int32, serial.nrow+1))
        # self.shape = (serial.nrow, serial.ncol)
        self.nnz = serial.nnz
        self.col_start = serial.col_start
        self.col_end = serial.col_end
        self.shape = (serial.col_end - serial.col_start + 1, serial.ncol)
        assert self.nnz > 0 and self.shape[0] > 0 and self.shape[1] > 0, f"{self.nnz} {self.shape}"

    def get(self):
        return sp.csr_array(mmap_unified_read(self.fname, self.dtype_sz), dtype=self.dtype_sz[0][0], copy=False, shape=self.shape)

    def reset_shape(self, start, end):
        self.shape = (end - start, self.shape[1])
        self.col_start = self.start
        self.col_end = self.end
        self.start = start
        self.end = end

    def serialize(self):
        assert self.dtype_sz[0][0] == np.float32 and self.dtype_sz[1][0] == np.int32 and self.dtype_sz[2][0] == np.int32
        assert self.nnz <= self.shape[0] * self.shape[1]
        assert self.nnz > 0 and self.shape[0] > 0 and self.shape[1] > 0
        return CSRSerialize(self.fname, self.nnz, self.shape[0], self.shape[1], self.start, self.end)

    def cleanup(self):
        os.remove(self.fname)


def csr_col_slice(mmap_path, csr, chunk, row_slice, buf_len=50000000):
    assert isinstance(csr.data, (np.memmap)) and isinstance(csr.indices, (np.memmap)) and isinstance(csr.indptr, (np.memmap))
    assert csr.data.filename != csr.indptr.filename and csr.indptr.filename != csr.indices.filename and csr.data.filename != csr.indices.filename, f"{csr.data.filename} {csr.indptr.filename} {csr.indices.filename}"
    assert csr.indptr.dtype == np.int64 and csr.indices.dtype == np.int64
    csr_serial = CSRSliceSerialize64(csr.data.filename.encode(), csr.indices.filename.encode(), csr.indptr.filename.encode(), csr.nnz, csr.shape[0], csr.shape[1], -1, -1)

    res_serial = []
    for start in range(0, csr.shape[1], chunk):
        end = min(start + chunk, csr.shape[1])
        tstamp = randint(-sys.maxsize, sys.maxsize)
        d = str.encode(mmap_path + f"/csr_col_slice_{tstamp}_data.bin")
        ptr = str.encode(mmap_path + f"/csr_col_slice_{tstamp}_indptr.bin")
        ind = str.encode(mmap_path + f"/csr_col_slice_{tstamp}_indices.bin")
        res_serial.append(CSRSliceSerialize(d, ind, ptr, -1, csr.shape[0], end-start, start, end))
    csr_col_slice_c(csr_serial, res_serial, chunk, buf_len, row_slice)

    # convert results into a variant of SliceCacheEntry (NOTE: C code converted results to unified format, stored in data file)
    cache = {}
    for res in res_serial:
        slc = SliceCacheEntry_C(res)
        assert(slc.shape[0] == (res.col_end - res.col_start + 1)), f"{slc.shape} != {res.col_end - res.col_start + 1}"
        cache[(res.start, res.end)] = slc
    return cache


def csr_col_slice_transpose(args, csr, chunk, buf_len=50000000):
    cache = csr_col_slice(args.mmap_path, csr, chunk, False, buf_len)

    # NOTE: could move this into C + perform transposes on GPU if this is too slow (these are guaranteed to be small enough to fit onto GPU due to slicing)
    # iterate through all entries in the cache, transpose them, then write out to disk again
    print("transposing cached column slices...", flush=True)
    out_cache = {}
    for k in tqdm(cache):
        c = cache[k]
        serial_nrow = c.dtype_sz[-1][1]-1           # does not use col_slice hack, fix the shape
        # c.reset_shape(0, serial_nrow)
        c.shape = (serial_nrow, c.shape[1])
        # print(f"{c.start} <-> {c.end}: {c.shape} {c.col_start} {c.col_end}")
        slc = c.get()
        slc = slc.T.tocsr()
        out_cache[(c.start, c.end)] = SliceCacheEntry(args, c.start, c.end, slc, c.start, c.end)
    return out_cache
    


def csr_col_slice_param(args, csr, chunk, buf_len=50000000):
    assert csr.indptr.dtype == np.int64 and csr.indices.dtype == np.int64

    res_serial = []
    for start in range(0, csr.shape[1], chunk):
        end = min(start + chunk, csr.shape[1])
        tstamp = randint(-sys.maxsize, sys.maxsize)
        d = str.encode(args.mmap_path + f"/csr_col_slice_{tstamp}_data.bin")
        ptr = str.encode(args.mmap_path + f"/csr_col_slice_{tstamp}_indptr.bin")
        ind = str.encode(args.mmap_path + f"/csr_col_slice_{tstamp}_indices.bin")
        res_serial.append(CSRSliceSerialize(d, ind, ptr, -1, csr.shape[0], end-start, start, end, -1, -1))
    csr_col_slice_param_c(csr, res_serial, chunk, buf_len)

    # convert results into a variant of SliceCacheEntry (NOTE: C code converted results to unified format, stored in data file)
    out = []
    for res in res_serial:
        out.append(SliceCacheEntry_C(res))
    return out





class SliceCacheEntry_Array():
    def __init__(self, cache_dir, start, end, data, indices, indptr, shape, col_start=None, col_end=None):
        self.start = start
        self.end = end

        self.col_start = col_start
        self.col_end = col_end

        assert data.shape[0] <= shape[0] * shape[1], f"{data.shape[0]} {shape} {col_start} {col_end}"
        assert indices.dtype == np.int32 and indptr.dtype == np.int32 and data.dtype == np.float32

        fname = cache_dir + f"/slice_unified_{randint(-sys.maxsize, sys.maxsize)}.bin"
        mmap_unified_write_zero_copy(fname, (data, indices, indptr))

        self.fname = fname
        self.dtype_sz = ((data.dtype, data.shape), (indices.dtype, indices.shape), (indptr.dtype, indptr.shape))
        self.shape = shape

    def get(self):
        return sp.csr_array(mmap_unified_read(self.fname, self.dtype_sz), dtype=self.dtype_sz[0][0], copy=False, shape=self.shape)

    def serialize(self):
        assert self.dtype_sz[0][0] == np.float32 and self.dtype_sz[1][0] == np.int32 and self.dtype_sz[2][0] == np.int32
        nnz = self.dtype_sz[0][1][0]        # nnz from data array shape
        assert nnz <= self.shape[0] * self.shape[1]
        return CSRSerialize(self.fname.encode(), nnz, self.shape[0], self.shape[1], self.start, self.end)

    def cleanup(self):
        os.remove(self.fname)
