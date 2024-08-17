import numpy as np
from time import time
import torch
from tqdm import tqdm
import copy
import os
import sys
import psutil
import shutil
from queue import Empty
from scipy import sparse as sp

from input_output import mmap_coo_arrays_to_csr, Buffer
from matrix_utils import profile_log
from preprocessor import _context
from slice import SliceCacheEntry_Array, csr_col_slice, SliceCacheEntry_C
from multiproc import MultiProcessDispatch
from laplacian import sparse_cotan_laplacian

if torch.cuda.is_available():
    import cupy as cp

class DiffOpAssembly(MultiProcessDispatch):
    def __init__(self, args, context_sz, dset, zero_code, diff_op_chunk, alphas):
        super().__init__(_laplacian_worker)
        self.args = args
        self.n_patch_per_dim = dset.n_patch_per_dim
        self.n_patch_per_img = dset.n_patch_per_img
        
        self.zero_code = None
        self.diff_op_chunk = None
        self.last_written = -1
        self.ordered_write_queue = {}

        # NOTE: this assumes block sparsity of the full diff op mx. For each sample, we have a 
        # (num. patches) * (num. patches) block that can be non-zero. Worst case assume that this
        # entire block is non-zero for each image (full context window)
        if args.n_aug > 0 and args.inter_aug_ctx:
            # block diagonal for augmentation blocks
            assert args.samples % (args.n_aug+1) == 0
            n_unaug = int(args.samples / (args.n_aug+1))
            max_sz = n_unaug * (((args.n_aug + 1) * self.n_patch_per_img) ** 2)
        else:
            max_sz = args.samples * self.n_patch_per_img * self.n_patch_per_img
        self.out_shape = (self.n_patch_per_img * self.args.samples, self.n_patch_per_img * self.args.samples)
        
        # tmp buffering for each slice, as well as output cache
        self.tmp_data_file = Buffer(np.float32, max_sz)     # TODO(as) can reduce max_sz to just account for the size of each slice (block-diagonal so will have regular structure)
        self.tmp_indices_file = Buffer(np.int32, max_sz)
        self.tmp_indptr_file = Buffer(np.int32, max_sz)
        self.out_slice_cache = {}
        self.indptr_offset = 0
        self.cache_dir = args.mmap_path + "/diff_op_slices"
        os.mkdir(self.cache_dir)

        self.slice_cache_dir = args.mmap_path + "/alpha_slice_cache"
        os.mkdir(self.slice_cache_dir)

        # per-image diff. op. template
        self.alphas = None
        if args.optim == "one":
            if args.n_aug > 0 and args.inter_aug_ctx:
                template = opt1_partial_diff_op_aug(args.n_aug, self.n_patch_per_img)
                self.n_ctx_pairs_per_img = template[0].shape[0]
            else:
                ctx_pairs = dset.get_context_pairs(context_sz)
                self.n_ctx_pairs_per_img = len(ctx_pairs)
                template = opt1_partial_diff_op(self.n_patch_per_dim, ctx_pairs, set())
        elif args.optim == "two":
            if args.n_aug > 0 and args.inter_aug_ctx:
                template = opt2_partial_diff_op_preproc_aug(args.n_aug, context_sz, self.n_patch_per_img, self.n_patch_per_dim)
            else:
                template = opt2_partial_diff_op_preproc(context_sz, self.n_patch_per_img, self.n_patch_per_dim)
            self.n_ctx_pairs_per_img = self.n_patch_per_img         # hack to make optim 1 code work for optim 2
        elif args.optim == "laplacian":
            # cotan laplacian weights are context dependent, no easy way to make a template
            template = None
            self.alphas = None
            self.alphas_slices = None
            if torch.cuda.is_available():
                # use single pass C-code to perform column slicing on entire alphas matrix at once
                print("Column slicing alphas for cotan Laplacian generation...")
                self.alphas_slices = csr_col_slice(args.mmap_path, alphas, self.n_patch_per_img, False)
                for k in tqdm(self.alphas_slices):        # TODO(as): fact that this is needed means theres an off by 1 error somewhere in the C code?
                    a = self.alphas_slices[k]
                    # print(f"{k}: {a.start} {a.end} {a.col_start} {a.col_end}, {a.shape}")

                    # TODO(as): this shouldn't be needed, but row slice stuff on C side seems to have an off by 1 somewhere (even though it should be disabled here...)
                    assert a.shape[1] == self.n_patch_per_img
                    a.shape = (alphas.shape[0], self.n_patch_per_img)
                    
                    # TODO(as) debug --> confirmed this all passes, so underlying C code is correct
                    # slc = a.get()     
                    # slc = slc.todense()
                    # gt = alphas[0:alphas.shape[0], k[0]:k[1]].todense()
                    # assert np.all(slc == gt)
            else:
                self.alphas = alphas


        else:
            raise NotImplementedError

        self.processing(args, template, zero_code, diff_op_chunk)

    def processing(self, args, template, zero_code, diff_op_chunk):
        if args.optim == "laplacian":
            self.diff_op_chunk = diff_op_chunk
            self.zero_code = zero_code
            self.run(daemon=True)
            return
        
        profile = False
        running = time()
        gpu_avail = torch.cuda.is_available()
        
        if args.n_aug > 0 and args.inter_aug_ctx:
            chnk_sz = args.n_aug + 1
        else:
            chnk_sz = 1
        assert diff_op_chunk % chnk_sz == 0, "Splitting augmentation blocks across slices is not currently supported"

        # pre-compute nonzero element case
        if args.optim != "laplacian":
            dop_template = self._prune(args, template, set(), chnk_sz, gpu_avail, None)
            running = profile_log(profile, running, f"template")

        for start in tqdm(range(0, args.samples, chnk_sz)):
            end = min(start + chnk_sz, args.samples)

            base_idx = int(start / chnk_sz)
            zero_ind = np.where(zero_code[base_idx*self.n_patch_per_img : end*self.n_patch_per_img] == 0)[0]
            running = profile_log(profile, running, f"({start}): slice get")
            if zero_ind.shape[0] == 0 and args.optim != "laplacian":
                # fast-track for no zero-code patches
                dop = copy.deepcopy(dop_template)
                running = profile_log(profile, running, f"({start}): construct (fast)")
            else:
                dop = self._prune(args, template, zero_ind, chnk_sz, gpu_avail, start)
                running = profile_log(profile, running, f"({start}): construct (slow)")

            self.store_result(args, start, end, dop.data, dop.indices, dop.indptr, diff_op_chunk)
            running = profile_log(profile, running, f"({start}): result str")

        self.postprocess_results()
    
    def store_result(self, args, start, end, data, indices, indptr, slc_sz):
        # Performs column slicing of the differential operator as it is built. Since diff. op. is block diagonal (inter-image connections for augmented images are contiguous), 
        # grouping slices by images implicitly performs column slicing. Also prune/record non-zero rows for alphas column pruning

        
        # Offset rows and cols for insertion into the slice of the differential operator
        # (can build CSR directly rather than through COO construction/conversion since all entries in a given row are added at once + rows are processed in order)        

        # offset into the slice currently being built
        cur_slice = start % slc_sz
        indices += cur_slice * self.n_patch_per_img       # hstacking results (per slice)
        indptr += self.indptr_offset
        self.indptr_offset += indices.shape[0]

        profile = False
        running = time()
        end_of_slice = end % slc_sz == 0 or end == args.samples
        if end_of_slice:
            running = profile_log(profile, running, f"init")
            self.tmp_data_file.update(data)
            self.tmp_indices_file.update(indices)
            self.tmp_indptr_file.update(indptr)
            running = profile_log(profile, running, f"update")

            out_data = self.tmp_data_file.get()
            out_indices = self.tmp_indices_file.get()
            out_indptr = self.tmp_indptr_file.get()
            running = profile_log(profile, running, f"get")

            # indptr is appended to out_indptr, indptr[-1] == out_indptr[-1]
            assert indptr[-1] < np.iinfo(np.int32).max and indptr[-1] >= 0, "cupy implicitly down-casts indptr+indices arrays to int32 + CUSPARSE/spgemm requires int32 indices"

            slc_start = int(start / slc_sz) * slc_sz
            slc_end = min(slc_start + slc_sz, end)      # don't want to run past # of images in dataset
            slc_start *= self.n_patch_per_img
            slc_end *= self.n_patch_per_img

            # since slices are along image boundaries + due to block-diagonal structure we can determine where to slice rows
            # just from start/end column indices. a bit more conservative in some cases
            nnz_row_start = slc_start
            nnz_row_end = slc_end

            # cache object expected during matmul
            shp = (slc_end - slc_start, nnz_row_end - nnz_row_start)        # TODO(as) +1 really isnt right here...
            self.out_slice_cache[(slc_start, slc_end)] = SliceCacheEntry_Array(self.cache_dir, slc_start, slc_end, out_data, out_indices, out_indptr, shp, nnz_row_start, nnz_row_end)
            running = profile_log(profile, running, f"cache")

            # reset buffers for next slice
            self.tmp_data_file.reset()
            self.tmp_indices_file.reset()
            self.tmp_indptr_file.reset()
            self.indptr_offset = 0
            running = profile_log(profile, running, f"reset")
        else:
            indptr = indptr[:-1]        # only write out final indptr for last chunk in the slice
            self.tmp_data_file.update(data)
            self.tmp_indices_file.update(indices)
            self.tmp_indptr_file.update(indptr)

    def postprocess_results(self):
        if torch.cuda.is_available(): 
            cp.get_default_memory_pool().free_all_blocks()

        if self.args.optim == "laplacian":
            # flush any images that have queued up
            while self.last_written + 1 in self.ordered_write_queue:
                self.last_written += 1
                dop = self.ordered_write_queue[self.last_written]
                self.store_result(self.args, self.last_written, self.last_written+1, dop.data, dop.indices, dop.indptr, self.diff_op_chunk)
                del self.ordered_write_queue[self.last_written]

            # assert len(self.ordered_write_queue) == 0 and self.last_written+1 == self.get_nbatch()
        
        shutil.rmtree(self.slice_cache_dir)

    def _prune(self, args, template, zero_ind, chnk_sz, gpu_avail, start):
        temp = copy.deepcopy(template)
        if args.optim == "one":
            data, row, col = opt1_partial_diff_op_prune(*temp, set(zero_ind))

            if gpu_avail:                
                dop = sp.coo_array((data, (row, col)), shape=(chnk_sz*self.n_patch_per_img, self.n_ctx_pairs_per_img), dtype=np.float32, copy=False)
                dop = cp.array(dop.todense())
            else:
                dop = mmap_coo_arrays_to_csr(args, data, row, col, np.float32, (chnk_sz*self.n_patch_per_img, self.n_ctx_pairs_per_img), mmap=False, verbose=False, disable_c=True)    
                dop = dop.todense()
        elif args.optim == "two":
            dop = opt2_partial_diff_op_prune(temp, self.n_patch_per_img, set(zero_ind))
            if gpu_avail:
                dop = cp.array(dop)
        else:
            raise NotImplementedError
        dop = dop @ dop.T       # dense matmul
        
        # dense to CSR....
        if gpu_avail:
            dop = cp.sparse.csr_matrix(dop, dtype=np.float32)
            dop = dop.get()
        else:
            dop = sp.csr_array(dop, shape=(chnk_sz*self.n_patch_per_img, chnk_sz*self.n_patch_per_img), dtype=np.float32, copy=False)
        return dop
    
    def get_nbatch(self):
        assert self.args.optim == "laplacian"
        return self.args.samples
    
    def generate_work(self, idx):
        assert self.args.optim == "laplacian"
        end = min(idx + 1, self.args.samples)
        zero_ind = np.where(self.zero_code[idx*self.n_patch_per_img : end*self.n_patch_per_img] == 0)[0]

        # if fast C slicing available, use it
        if torch.cuda.is_available():
            # pass cache entry off to the worker, let it perform the read from disk
            img_alphas = self.alphas_slices[(idx*self.n_patch_per_img, (idx+1)*self.n_patch_per_img)]
        else:
            img_alphas = self.alphas[0:self.alphas.shape[0], idx*self.n_patch_per_img : (idx+1)*self.n_patch_per_img]
        return (img_alphas, idx, self.n_patch_per_img, set(zero_ind), self.args.lap_area_norm)

    def num_processes(self):
        assert self.args.optim == "laplacian"
        # cores = int(psutil.cpu_count() / 1.5)
        cores = psutil.cpu_count()
        print(f"Generating Laplacian with {cores} cores")
        return cores
    
    def n_work_gen_check(self):
        return 64
    
    def n_work_gen_advance(self):
        return 64

    def fast_work_gen(self):
        assert self.args.optim == "laplacian"
        return True
    
    def process_result(self, res):
        assert self.args.optim == "laplacian"
        dop, img_idx = res

        # dense to CSR on the GPU if possible
        if torch.cuda.is_available():
            dop = cp.array(dop)
        # dop = dop @ dop.T       # dense matmul

        if torch.cuda.is_available():
            dop = cp.sparse.csr_matrix(dop, dtype=np.float32)
            dop = dop.get()
        else:
            dop = sp.csr_array(dop, shape=(self.n_patch_per_img, self.n_patch_per_img), dtype=np.float32, copy=False)

        # write to output file in image index order even if intermediate results are out of order
        if self.last_written + 1 == img_idx:
            # this index is next one to be written
            self.store_result(self.args, img_idx, img_idx+1, dop.data, dop.indices, dop.indptr, self.diff_op_chunk)
            self.last_written += 1

            # flush any images that have queued up waiting for this one to arrive
            while self.last_written + 1 in self.ordered_write_queue:
                self.last_written += 1
                dop = self.ordered_write_queue[self.last_written]
                self.store_result(self.args, self.last_written, self.last_written+1, dop.data, dop.indices, dop.indptr, self.diff_op_chunk)
                del self.ordered_write_queue[self.last_written]
        else:
            assert self.last_written < img_idx
            self.ordered_write_queue[img_idx] = dop


def construct_diff_op(args, dset, alphas, context_sz, diff_op_chunk):
    print("Assembling differential operator...", flush=True)
    zero_code = alphas.sum(axis=0)
    num_zero = np.where(zero_code == 0)[0].shape[0]
    print("\t\tSC {:.5f} percent ({}) 'zero' patches...".format((num_zero / alphas.shape[1]) * 100, num_zero))    

    doa = DiffOpAssembly(args, context_sz, dset, zero_code, diff_op_chunk, alphas)
    return (doa.out_slice_cache, doa.cache_dir)

def opt1_partial_diff_op(y_range, ctx_pairs, zero_ind):
    """Implement first-derivative contextual operator for a single image. All images in the dataset have the same context pattern."""
    data = []
    rows = []
    cols = []
    index = {}
    for idx, px_pair in enumerate(ctx_pairs):
        pos_pixel = px_pair[0]
        neg_pixel = px_pair[1]
        
        # Pixels are returned as (x, y) tuples
        # pos_idx = pos_pixel[1] * y_range + pos_pixel[0]
        # neg_idx = neg_pixel[1] * y_range + neg_pixel[0]

        pos_idx = pos_pixel[0] * y_range + pos_pixel[1]     # NOTE: stored in row-major order
        neg_idx = neg_pixel[0] * y_range + neg_pixel[1]

        assert pos_idx != neg_idx

        # Check if either indices are for "zero" patches
        if pos_idx in zero_ind or neg_idx in zero_ind:
            continue
        
        data.append(1)
        rows.append(pos_idx)
        cols.append(idx)
        
        data.append(-1)
        rows.append(neg_idx)
        cols.append(idx)

        # store all indices in data, rows, cols parallel arrays that correspond to a given index
        if pos_idx not in index:
            index[pos_idx] = set()
        if neg_idx not in index:
            index[neg_idx] = set()
        idx0 = len(data)-1        # need to remove both pos and neg entries... (offset due to len. vs. zero indexing)
        idx1 = len(data)-2
        index[pos_idx].add(idx0)
        index[pos_idx].add(idx1)
        index[neg_idx].add(idx0)
        index[neg_idx].add(idx1)

    data = np.array(data, dtype=np.float32)
    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    return data, rows, cols, index

def opt1_partial_diff_op_aug(n_aug, n_patch_per_img):
    """Implement first-derivative contextual operator for a single image. All images in the dataset have the same context pattern."""
    data = []
    rows = []
    cols = []
    index = {}

    # connect every pixel in the original image to every other pixel in the original + augmented image (but no connections between augmented pixels)
    # assumes that the original image is placed first in the block
    cnt = 0
    for pos_idx in tqdm(range(n_patch_per_img)):
        for neg_idx in range(pos_idx, (n_aug+1) * n_patch_per_img, 1):
            data.append(1)
            rows.append(pos_idx)
            cols.append(cnt)
            
            data.append(-1)
            rows.append(neg_idx)
            cols.append(cnt)

            # store all indices in data, rows, cols parallel arrays that correspond to a given index
            if pos_idx not in index: index[pos_idx] = set()
            if neg_idx not in index: index[neg_idx] = set()
            idx0 = len(data)-1        # need to remove both pos and neg entries... (offset due to len. vs. zero indexing)
            idx1 = len(data)-2
            index[pos_idx].add(idx0)
            index[pos_idx].add(idx1)
            index[neg_idx].add(idx0)
            index[neg_idx].add(idx1)
            
            cnt += 1

    data = np.array(data, dtype=np.float32)
    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    return data, rows, cols, index

def opt1_partial_diff_op_prune(data, rows, cols, index, zero_ind):
    """Remove all zero index values from the differential operator"""
    # collect all indices in parallel array where element in zero_ind is used
    rm_idx = set()
    for idx in zero_ind:
        rm_idx.update(index[idx])

    # remove elements corresponding to zero indices
    data = np.delete(data, list(rm_idx), axis=0)
    rows = np.delete(rows, list(rm_idx), axis=0)
    cols = np.delete(cols, list(rm_idx), axis=0)
    assert data.shape[0] == rows.shape[0] and rows.shape[0] == cols.shape[0]
    return data, rows, cols
    
def opt2_partial_diff_op_preproc_aug(n_aug, ctx_sz, n_patch_per_img, n_patch_per_dim):
    # get single image context (for inter-original image patch context)
    dop = opt2_partial_diff_op_preproc(ctx_sz, n_patch_per_img, n_patch_per_dim)

    dim = (n_aug+1)*n_patch_per_img
    aug_dop = np.zeros(shape=(dim, dim), dtype=np.float32)
    aug_dop[:n_patch_per_img, :n_patch_per_img] = dop

    # make connections between original image patches + all augmented patches (note that no inter-augmented connections are generated)
    # (neighbors fill the rows, original images are columns. cols sum to 1)
    aug_dop[n_patch_per_img:, :n_patch_per_img] = -1
    return aug_dop

def opt2_partial_diff_op_preproc(ctx_sz, n_patch_per_img, n_patch_per_dim):
    dop = np.zeros(shape=(n_patch_per_img, n_patch_per_img), dtype=np.float32)
    for pidx in range(n_patch_per_img):
        x = int(pidx / n_patch_per_dim)
        y = int(pidx % n_patch_per_dim)
        ctx = _context(x, y, n_patch_per_dim, ctx_sz)
        for px_pair in ctx:
            n_x, n_y = px_pair
            neighbor_idx = n_x * n_patch_per_dim + n_y
            dop[neighbor_idx, pidx] = -1
    return dop

def opt2_partial_diff_op_prune(dop, n_patch_per_img, zero_code):
    # Outputs a (# patch per image x # patch per image) matrix. Diagonal entries are 1, all others are negative, columns must sum to zero

    # prune all references to zero codes
    dop[list(zero_code), :] = 0
    dop[:, list(zero_code)] = 0

    # scale remaining neighbor entries (normalize over columns for )
    s = np.sum(dop, axis=0, keepdims=True)
    s *= -1
    s_zero_idx = s == 0
    s[s_zero_idx] = 1       # avoid div by zero
    dop /= s

    # set diagonal for "original images" to 1 (if patch for column is not zero and if it has any nonzero neighbors)
    r = [i for i in range(n_patch_per_img) if i not in zero_code and not s_zero_idx[0][i]]
    dop[r, r] = 1

    return dop

def _laplacian_worker(w_q, r_q, w_done, idx, worker_init_args):    
    # Load work from the work queue, put results onto the result queue
    def _proc(work_q, res_q, w_done, w_init_args):
        while not (w_done.is_set() and work_q.empty()):
            try:
                work_slice = work_q.get(block=True, timeout=0.5)
            except Empty:
                continue

            img_alphas, img_idx, n_patch_per_img, zero_codes, area_norm = work_slice
            if isinstance(img_alphas, (SliceCacheEntry_C)):
                img_alphas = img_alphas.get()
            res_q.put((sparse_cotan_laplacian(img_alphas, img_idx, n_patch_per_img, zero_codes, area_norm), img_idx))

    _proc(w_q, r_q, w_done, worker_init_args)
    sys.exit(0)
