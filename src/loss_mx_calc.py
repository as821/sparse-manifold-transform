import os
import sys
import shutil
import torch

from ctypes_interface import spmm_batched, spgemm_batched_matmul_dop, CSRSerialize, DenseSerialize, c_impl_available

import numpy as np
import scipy.sparse as sp
from random import randint
from queue import Empty
from time import time

from matrix_utils import csr_row_view
from input_output import mmap_unified_read, mmap_unified_write, mmap_file_load_1d
from slice import PreMatmulCacheGen
from multiproc import MultiProcessDispatch


class LossMatrixWorkSlice():
    def __init__(self, idx, pkg, row, out_shape):
        self.idx = idx
        self.pkg = pkg
        self.row_range = row
        self.out_shape = out_shape

    def process(self, args, diff_op_cache, alphas_T_cache, col_chunk):

        # NOTE: this assumes that all rows are processed in order!
        dir_path = args.mmap_path + f"/diff_op_mm_{self.idx}"
        os.mkdir(dir_path)
        result = np.zeros(shape=(self.row_range[1] - self.row_range[0], self.out_shape[1]), dtype=np.float32)
        
        out_fname = args.mmap_path + f"/diff_op_mm_{self.idx}.bin"
        assert not os.path.exists(out_fname)
        open(out_fname, "wb").close()

        if c_impl_available():
            a_serials = []
            b_serials = []
            shps = []
            nrow = -1
            for i in self.pkg:
                a_fname, a_dtype_sz, a_shape = i[0]
                a_nnz = a_dtype_sz[0][1][0]     # num. elements in the alpha data array
                a_serials.append(CSRSerialize(str.encode(a_fname), a_nnz, a_shape[0], a_shape[1], -1, -1))
                b_serials.append(diff_op_cache[i[1]].serialize())
                shps.append((a_shape[0], diff_op_cache[i[1]].shape[1]))

                # check all "a" slices have the same number of rows
                if nrow > 0:
                    assert a_shape[0] == nrow
                nrow = a_shape[0]

            assert len(a_serials) == len(b_serials)
            data_files = [dir_path + f"/diffop_mm_tmp_{randint(-sys.maxsize, sys.maxsize)}.bin" for _ in range(len(b_serials))]
            spgemm_batched_matmul_dop(a_serials, b_serials, data_files)
            
            # collect results (already transposed on GPU)
            dop_cache = []
            for fname, shp in zip(data_files, shps):
                dop_cache.append(DenseSerialize(str.encode(fname), shp[1], shp[0], -1, -1))

            # calculate every part of inner matrix that requires this dop slice
            t = time()
            for didx in range(len(dop_cache)):
                d = dop_cache[didx]
                
                # collect column chunks from each row slice of alphas_T_cache that correspond to this dop_cache entry
                b_serials = []
                b_serial_k = []
                for b_row_slice in alphas_T_cache:
                    ks = sorted(list(alphas_T_cache[b_row_slice].keys()))          # process columns in order
                    b_slice = alphas_T_cache[b_row_slice][ks[didx]]
                    if b_slice.end <= self.row_range[0]:            # ADD^TA^T is symmetric
                        continue                    
                    b_serials.append(b_slice.serialize())
                    b_serial_k.append(b_row_slice)
                
                # calc all matmuls that include this dop cache entry
                res_files = [dir_path + f"/inner_tmp_{randint(-sys.maxsize, sys.maxsize)}.bin" for _ in range(len(b_serials))]
                spmm_batched(d, b_serials, res_files)

                # collect results
                for b, res_file, k in zip(b_serials, res_files, b_serial_k):
                    res = mmap_file_load_1d(res_file, np.float32, (b.nrow, d.ncol))
                    os.remove(res_file)
                    result[:, k[0] : k[1]] += res.T
            # print(f"\tspmm matmuls: {time() - t}")

            # cleanup dop intermediate results
            for i in data_files:
                os.remove(i)

            # remove all alphas files in pkg
            for i in self.pkg:
                os.remove(i[0][0])
        else:
            dop = []
            for i in self.pkg:
                a_fname, a_dtype_sz, a_shape = i[0]
                img_alpha = sp.csr_array(mmap_unified_read(a_fname, a_dtype_sz), dtype=a_dtype_sz[0][0], copy=False, shape=a_shape)
                assert img_alpha.indptr.dtype == np.int32 and img_alpha.indices.dtype == np.int32
                
                d = diff_op_cache[i[1]].get()
                r = img_alpha @ d
                r = r.todense()
                dop.append(r)
                os.remove(i[0][0])

            # calculate every part of inner matrix that requires this dop slice
            for b_row_slice in alphas_T_cache:
                b_start, b_end = b_row_slice
                ks = sorted(list(alphas_T_cache[b_row_slice].keys()))          # process columns in order
                for d, b in zip(dop, ks):
                    a_slice = d.T
                    b_slice = alphas_T_cache[b_row_slice][b]
                    if b_slice.end <= self.row_range[0]:            # ADD^TA^T is symmetric
                        continue
                    result[:, b_start:b_end] += (b_slice.get().todense() @ a_slice).T

        with open(out_fname, 'ab') as fid:
            fid.write(result)
            fid.flush()
        shutil.rmtree(args.mmap_path + f"/diff_op_mm_{self.idx}")
        return (self.idx, self.row_range, out_fname, result.shape)

class LossMatrixCalc(MultiProcessDispatch):
    """Calculate A @ D @ A^T."""
    def __init__(self, args, alphas, diff_op_cache):
        super().__init__(_inner_worker)
        self.args = args
        self.alphas = alphas
        self.diff_op_cache = diff_op_cache

        self.alphas_row_slice = args.diff_op_a_chunk
        self.dop_mm_col_chunk = args.diff_op_d_chunk    # enforce this to avoid needing the slicing result of the diff op matmul
        self.alphas_T_slice = args.inner_chunk

        print("\tGenerating tranpose alphas cache...")

        mgccg = PreMatmulCacheGen(args, alphas, self.alphas_T_slice, False, alphas_2d_slice=True, col_slice=False, batch_sz_2d=self.dop_mm_col_chunk, means=None)
        mgccg.run(daemon=True)
        alphas_T_cache = mgccg.cache
        for c in alphas_T_cache:         # for each row, index the column slices
            alphas_T_cache[c] = {slc.col_start : slc for slc in alphas_T_cache[c]}
        self.alphas_T_cache = alphas_T_cache


        self.alphas_cache_dir = args.mmap_path + f"/diff_op_mm_a"
        os.mkdir(self.alphas_cache_dir)

        self.out = np.zeros(shape=(alphas.shape[0], alphas.shape[0]), dtype=np.float32)

    def worker_init_args(self):
        return (self.args, self.diff_op_cache, self.alphas_T_cache, self.dop_mm_col_chunk)
    
    def get_nbatch(self):
        n_batch = int(self.alphas.shape[0] / self.alphas_row_slice)
        assert self.alphas_row_slice > 1, "modulus logic below can fail if batch size == 1"
        if self.alphas.shape[0] % self.alphas_row_slice != 0:
            n_batch += 1
        return n_batch
    
    def generate_work(self, idx):
        # slice alphas rows + write to file, then pass to worker
        start = idx * self.alphas_row_slice
        end = min(start+self.alphas_row_slice, self.alphas.shape[0])
        alphas_slice = csr_row_view(self.alphas, start, end)
        assert alphas_slice.indptr[-1] < np.iinfo(np.int32).max, "CUSPARSE/spgemm requires int32 indices"
        alphas_slice.indptr = alphas_slice.indptr.astype(np.int32)
        alphas_slice.indices = alphas_slice.indices.astype(np.int32)

        # slice alphas to match the columns with nonzero elements for each diff op slice
        out = []
        ks = sorted(list(self.diff_op_cache.keys()), key=lambda a: a[0])     # will be hstack-ing so need to process columns in order
        for k in ks:
            # NOTE: same output dimensions, but prunes the internal dimension of the matmul to only cover the nonzero elements of the second operand
            entry = self.diff_op_cache[k]
            a_slc = alphas_slice[:, entry.col_start:entry.col_end]

            # tst = entry.get()
            # full = alphas_slice @ tst     # TODO(as) for this to work, need to disable column slicing in SliceCacheWorkSlice::process
            # less = alphas_slice[:, entry.col_start:entry.col_end] @ tst[entry.col_start:entry.col_end, :]
            # assert np.all(full.todense() == less.todense())

            assert a_slc.indices.dtype == np.int32 and a_slc.indptr.dtype == np.int32
            fname = self.alphas_cache_dir + f"/sparse_dop_aslice_{start}_{k[0]}_{k[1]}.bin"
            mmap_unified_write(fname, (a_slc.data, a_slc.indices, a_slc.indptr))
            pkg = (fname, ((np.float32, a_slc.data.shape), (np.int32, a_slc.indices.shape), (np.int32, a_slc.indptr.shape)), a_slc.shape)
            out.append((pkg, k))
        return (idx, out, (start, end), self.out.shape)

    def num_processes(self):
        return 1

    def fast_work_gen(self):
        return True
    
    def process_result(self, res):
        idx, row_range, fname, shp = res
        m = mmap_file_load_1d(fname, np.float32, shp, order='C')
        os.remove(fname)
        self.out[row_range[0]:row_range[1]] = m

    def postprocess_results(self):
        shutil.rmtree(self.alphas_cache_dir)
        for k in self.alphas_T_cache:
            for i in self.alphas_T_cache[k]:
                self.alphas_T_cache[k][i].cleanup()



def _inner_worker(w_q, r_q, w_done, idx, worker_init_args):    
    # Load work from the work queue, put results onto the result queue
    def _proc(work_q, res_q, w_done, w_init_args):
        while not (w_done.is_set() and work_q.empty()):
            try:
                work_slice = work_q.get(block=True, timeout=0.5)
            except Empty:
                continue

            res_q.put(LossMatrixWorkSlice(*work_slice).process(*w_init_args))

    _proc(w_q, r_q, w_done, worker_init_args)
    sys.exit(0)






