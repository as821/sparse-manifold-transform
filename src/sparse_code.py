import numpy as np
import torch
from tqdm import tqdm
import scipy.sparse as sp
import random

from input_output import mmap_coo_arrays_to_csr, File

from matrix_utils import profile_log
from time import time

if torch.cuda.is_available():
    import cupy as cp

import pdb

class SparseCodeLayer:
    """Encode inputs in the given dictionary basis. Optionally, generate a random basis from the first data passed to this object."""
    def __init__(self, dict_sz, phi, gq_thresh):
        self.basis = phi
        self.dict_sz = dict_sz
        self.gq_thresh = gq_thresh

    def __call__(self, args, data, test=False, dense=False):
        # Encode the given data in this object's dictionary basis
        assert self.basis.shape[1] == self.dict_sz
        if dense:
            return _general_sparse_coding_dense(args, data, self.basis, self.gq_thresh, test=test)
        else:
            return _general_sparse_coding(args, data, self.basis, self.gq_thresh, test=test)

def _coo_update_file(coo_data, coo_row, coo_col, result):
    coo_data.update(result.data)
    coo_row.update(result.row)
    coo_col.update(result.col)
    return (coo_data, coo_row, coo_col)

def _coo_init(mmap_path, shape):
    sz = int(shape[0] * shape[1] * 0.1)          # assume <10% sparsity
    data = File(mmap_path, np.float32, sz)
    row = File(mmap_path, np.int64, sz)
    col = File(mmap_path, np.int64, sz)
    return data, row, col


def _general_sparse_coding(args, data, phi, gq_thresh, test=False):
    """Implement k-sparse coding for the given data and dictionary."""
    print("Calculating sparse codes...", flush=True)
    coo_pkg = _coo_init(args.mmap_path, (phi.shape[1], data.shape[1]))

    assert phi.shape[1] < np.iinfo(np.int64).max
    assert data.shape[1] < np.iinfo(np.int64).max

    # Deduce the number of patched per image from flattened dataset size + number of samples
    if test:
        assert args.test_samples > 1, "modulus logic below can fail if samples == 1"
        assert data.shape[1] % args.test_samples == 0, "Assumes that all images have the same number of image patches"    
        n_patch_per_img = data.shape[1] // args.test_samples
        samples = args.test_samples
    else:
        assert args.samples > 1, "modulus logic below can fail if samples == 1"
        assert data.shape[1] % args.samples == 0, "Assumes that all images have the same number of image patches"
        n_patch_per_img = data.shape[1] // args.samples
        samples = args.samples
    # print(f"SC using # patch per image: {n_patch_per_img}, data shape: ({data.shape[0]}, {data.shape[1]}), # samples: {samples}")

    if torch.cuda.is_available():
        phi = phi.to(f'cuda:0', non_blocking=True)

    batch_sz = args.sc_chunk
    for start in tqdm(range(0, samples, batch_sz)):
        end = min(start + batch_sz, samples)
        batch = data[:, start*n_patch_per_img : end*n_patch_per_img]
        result = SparseWorkSlice(batch, test, gq_thresh, start * n_patch_per_img).process(args, 0, phi)
        coo_pkg = _coo_update_file(*coo_pkg, result)

    print("Generating mmap-backed CSR matrix through COO construction...", flush=True)
    d, r, c = coo_pkg[0].get_mmap(), coo_pkg[1].get_mmap(), coo_pkg[2].get_mmap()
    codes = mmap_coo_arrays_to_csr(args, d, r, c, np.float32, (phi.shape[1], data.shape[1]), mmap=True)
    for i in coo_pkg:
        i.cleanup()

    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return codes

@torch.compiler.disable
def _general_sparse_coding_dense(args, data, phi, gq_thresh, test=False):
    """Implement k-sparse coding for the given data and dictionary."""
    # Data and phi are both L2 normalized, so their cosine similarity is their dot product
    assert len(data.shape) == 3
    cosine_sim = phi.T @ data


    # TODO: do we still want to keep zero code handling?


    mask = cosine_sim >= gq_thresh
    cosine_sim[mask] = 1
    mask = ~mask
    cosine_sim[mask] = 0

    # cosine_sim.sub_(gq_thresh)
    # torch.nn.functional.relu(cosine_sim, inplace=True)
    # cosine_sim[cosine_sim > 0] = 1

    return cosine_sim

class SparseWorkSlice():
    def __init__(self, batch, test, thresh, offset):
        self.offset = offset
        self.batch = batch
        self.test = test
        self.thresh = thresh
            
    def process(self, args, gpu_idx, phi):
        profile = False
        running = time()
        running = profile_log(profile, running, "start")
        
        if torch.cuda.is_available() and gpu_idx >= 0:
            self.batch = self.batch.to(f'cuda:{gpu_idx}', non_blocking=True)

        # Data and phi are both L2 normalized, so their cosine similarity is their dot product
        cosine_sim = phi.T @ self.batch
        running = profile_log(profile, running, "matmul")

        # Ensure that each data point has at least 1 entry >= thresh (when applicable)
        if self.test or args.zero_code_disable:
            amax = cosine_sim.argmax(dim=0)
            ind = torch.arange(0, amax.shape[0])
            cosine_sim[amax, ind] = self.thresh   

        running = profile_log(profile, running, "misc")

        # only contains 0/1 entries
        codes = torch.zeros_like(cosine_sim)
        codes[cosine_sim >= self.thresh] = 1

        running = profile_log(profile, running, "set one")

        # col_sums = codes.sum(axis=0)
        # ind = col_sums > 0
        # running = profile_log(profile, running, "sum")

        if torch.cuda.is_available():
            # performs dense -> COO conversion on the GPU
            res = cp.sparse.coo_matrix(cp.asarray(codes)).get()
        else:
            res = sp.coo_array(codes.cpu().numpy())
        running = profile_log(profile, running, "coo conversion")
        res.row = res.row.astype(np.int64)
        res.col = res.col.astype(np.int64)
        res.col += self.offset
        assert res.row.dtype == res.col.dtype and res.row.dtype ==  np.int64, "Need large index dtypes to support very large datasets."
        running = profile_log(profile, running, "dtype + offset")

        return res





def generate_dict(args, dset, dict_sz, dict_thresh):
    """Generate dictionary elements from the already generated data points.
    NOTE: could run dictionary learning with multiple initialization and pick the best (like normal K-Means)
    """
    print("Generating dictionary...", flush=True)

    # Return a random selection of (unique) image patches as the dictionary to use
    ptr = 0 
    patch_dim = dset.n_inp_channels * args.patch_sz * args.patch_sz
    phi = torch.zeros(size=(patch_dim, dict_sz))
    if torch.cuda.is_available():
        phi = phi.to('cuda:0')
    shuf = [i for i in range(args.samples)]
    random.shuffle(shuf)
    pbar = tqdm(total=dict_sz)

    chnk_sz = 50
    for start in range(0, len(shuf), chnk_sz):
        end = min(len(shuf), start+chnk_sz)
        
        # generate all patches for the specified images
        cand = torch.zeros((patch_dim, chnk_sz * dset.n_patch_per_img))
        for idx in range(start, end):
            c_idx = idx - start
            cand[:, c_idx * dset.n_patch_per_img : (c_idx + 1) * dset.n_patch_per_img] = dset.get_single_train_image(shuf[idx])[0]

        if torch.cuda.is_available():
            cand = cand.to('cuda:0')

        # calc candidate similarity to existing dict elements and other candidates in the batch
        sim = cand.T @ phi
        c_sim = cand.T @ cand

        # check if candidates fit in the dictionary
        c_added_idx = []
        for i in range(start * dset.n_patch_per_img, end * dset.n_patch_per_img):
            c_idx = i - (start * dset.n_patch_per_img)
            slc = sim[c_idx, :]
            if not torch.any(slc > dict_thresh):
                if len(c_added_idx) > 0:
                    # Compare with any of the other candidates that have already been added during this iteration
                    t = torch.tensor(c_added_idx)

                    slc = c_sim[c_idx, t]
                    if torch.any(slc > dict_thresh).item():
                        continue

                phi[:, ptr] = cand[:, c_idx]
                ptr += 1
                pbar.update(1)
                c_added_idx.append(c_idx)

                if ptr == dict_sz:
                    break

        if ptr == dict_sz:
            print(f"Found {dict_sz} sufficiently (<={dict_thresh}) unique dictionary elements in {i} / {args.samples * dset.n_patch_per_img} ({i / (args.samples * dset.n_patch_per_img):0.4f}) tries.")
            break
    assert ptr == dict_sz, f"Unable to find {dict_sz} sufficiently (<={dict_thresh}) unique dictionary elements in the dataset (only found {ptr})."
    pbar.close()
    if torch.cuda.is_available():
        phi = phi.cpu()

    # shuffle dictionary elements to avoid having dict elements with densest codes being grouped together in the low indices of the dictionary (allows increasing size of GPU slices later)
    perm = torch.randperm(phi.shape[1])
    phi = phi[:, perm]

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return phi.float()






