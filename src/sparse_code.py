import torch
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
import random

from input_output import mmap_coo_arrays_to_csr, File

from matrix_utils import profile_log
from time import time

if torch.cuda.is_available():
    import cupy as cp

class SparseCodeLayer:
    """Encode inputs in the given dictionary basis. Optionally, generate a random basis from the first data passed to this object."""
    def __init__(self, dict_sz, phi, gq_thresh, idx_list=None):
        self.basis = phi
        self.dict_sz = dict_sz
        self.gq_thresh = gq_thresh
        self.idx_list = idx_list

    def __call__(self, args, data, test=False):
        # Encode the given data in this object's dictionary basis
        assert self.basis.shape[1] == self.dict_sz
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
    print("Calculating k-sparse interpolations...", flush=True)

    # TODO(as) if this is all too slow, then should come up with a way to build CSR matrix while generating sparse codes (likely involving C optim. too)

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
    print(f"SC using # patch per image: {n_patch_per_img}, data shape: ({data.shape[0]}, {data.shape[1]}), # samples: {samples}")

    if torch.cuda.is_available():
        phi = phi.to(f'cuda:0', non_blocking=True)
    running_stats = np.zeros(shape=8, dtype=np.float64)
    # dict_el_sum = np.zeros(shape=phi.shape[1], dtype=np.float64)
    niter = 0

    batch_sz = args.sc_chunk
    for start in tqdm(range(0, samples, batch_sz)):
        end = min(start + batch_sz, samples)
        batch = data[:, start*n_patch_per_img : end*n_patch_per_img]
        result, stats, tmp_dict_el_sum = SparseWorkSlice(batch, test, gq_thresh, start * n_patch_per_img).process(args, 0, phi)
        coo_pkg = _coo_update_file(*coo_pkg, result)
        stats[2] = (end-start)*n_patch_per_img
        running_stats[:stats.shape[0]] += stats
        # dict_el_sum += tmp_dict_el_sum
        running_stats[-1] += n_patch_per_img
        niter += 1

    # Process running statistics
    avg_n_dict_el = running_stats[0] / running_stats[1]     # excludes zero-codes
    avg_n_dict_el_zc = running_stats[0] / running_stats[2]  # include zero codes in normalization
    perc_zero = (running_stats[2] - running_stats[1]) / running_stats[2]
    avg_enc = running_stats[3] / running_stats[4]
    tot_avg_enc = running_stats[5] / running_stats[2]
    avg_sim = running_stats[6] / running_stats[4]
    print("Avg. encoding sz: {:.3f} ({:.3f}) ({:.5f}, {}, {}, {}), avg. similarity: {:.5f} ({:.5f}, {:.5f})".format(avg_n_dict_el, avg_n_dict_el_zc, perc_zero, running_stats[2] - running_stats[1], running_stats[2], running_stats[-1], avg_enc, tot_avg_enc, avg_sim))
    # # NOTE: Var(cX) = c^2 Var(X)
    # out_el_sum = (dict_el_sum / dict_el_sum.sum()) * 100
    # print("\t dict el: ({:.5f}, {:.5f}, {:.5f}, {:.5f})".format(out_el_sum.min(), out_el_sum.max(), out_el_sum.mean(), out_el_sum.var()))

    print("Generating mmap-backed CSR matrix through COO construction...", flush=True)
    d, r, c = coo_pkg[0].get_mmap(), coo_pkg[1].get_mmap(), coo_pkg[2].get_mmap()
    codes = mmap_coo_arrays_to_csr(args, d, r, c, np.float32, (phi.shape[1], data.shape[1]), mmap=True)
    for i in coo_pkg:
        i.cleanup()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return codes



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

        tot_avg_sim = cosine_sim.sum() / phi.shape[0]

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

        # note: this logging is slow compared to everything else here, but leave it enabled since its kinda useful
        # dict_el_sum = codes.sum(axis=1).to("cpu", non_blocking=True)
        avg_sim = cosine_sim[codes > 0].sum().to("cpu", non_blocking=True)
        running = profile_log(profile, running, "logging (0)")
        
        col_sums = codes.sum(axis=0)
        ind = col_sums > 0
        running = profile_log(profile, running, "sum")

        avg_norm = col_sums.sum().to("cpu", non_blocking=True)
        n_nonzero = col_sums[ind].shape[0]
        running = profile_log(profile, running, "logging (0.5)")

        if not args.disable_sc_norm:
            codes[:, ind] /= col_sums[ind]
        running = profile_log(profile, running, "norm")


        # track some basic stats
        avg_enc = codes[codes > 0]
        avg_enc_cnt = avg_enc.shape[0]
        avg_enc = avg_enc.sum()     # of the dictionary elements used in encoding, what is the average cosine similarity to that element

        if args.sc_only_normalize and args.sc_only:
            # L2 normalize codes for each patch
            n = torch.linalg.vector_norm(codes, ord=2, dim=0) + 1e-20      # avoid div by zero
            # print(f"{codes.shape} {n.shape}")
            codes /= n
        
        running = profile_log(profile, running, "logging (1)")
        if torch.cuda.is_available():
            res = cp.sparse.coo_matrix(cp.asarray(codes)).get()
        else:
            res = sp.coo_array(codes.cpu().numpy())
        running = profile_log(profile, running, "coo conversion")
        res.row = res.row.astype(np.int64)
        res.col = res.col.astype(np.int64)
        res.col += self.offset
        assert res.row.dtype == res.col.dtype and res.row.dtype ==  np.int64, "Need large index dtypes to support very large datasets."
        running = profile_log(profile, running, "dtype + offset")

        # return res, torch.tensor([avg_norm, n_nonzero, 0, avg_enc, avg_enc_cnt, tot_avg_sim, avg_sim]).numpy(), dict_el_sum.numpy()
        return res, torch.tensor([avg_norm, n_nonzero, 0, avg_enc, avg_enc_cnt, tot_avg_sim, avg_sim]).numpy(), None





def generate_dict(args, x, dict_sz, dict_thresh):
    """Generate dictionary elements from the already generated data points.
    NOTE: could run dictionary learning with multiple initialization and pick the best (like normal K-Means)
    """
    print("Generating dictionary...", flush=True)

    # print(f"Max norm: {np.linalg.norm(x, ord=2, axis=0).max()}")
    # assert args.dict_path != "", "Must specify a dictionary file to load from."
    # print(f"Loading dictionary from {args.dict_path}...", flush=True)
    # phi = np.load(args.dict_path, allow_pickle=True)
    # phi = torch.from_numpy(phi)

    # Return a random selection of (unique) image patches as the dictionary to use
    ptr = 0 
    phi = torch.zeros(size=(x.shape[0], dict_sz), dtype=x.dtype)
    if torch.cuda.is_available():
        phi = phi.to('cuda:0')
    shuf = [i for i in range(x.shape[1])]
    random.shuffle(shuf)
    pbar = tqdm(total=dict_sz)

    # for idx, s in enumerate(shuf):
    chnk_sz = 1000
    idx_list = []
    for start in range(0, len(shuf), chnk_sz):
        end = min(len(shuf), start+chnk_sz)
        cand = x[:, shuf[start:end]]
        if torch.cuda.is_available():
            cand = cand.to('cuda:0')


        # TODO(as) remove this assumption of unit length vectors...


        sim = cand.T @ phi
        c_sim = cand.T @ cand

        # For all candidates, check if room for them in the dictionary (also considering other new dict elements added on this iteration)
        c_added_idx = []
        for i in range(start, end):
            c_idx = i - start
            slc = sim[c_idx, :]
            if not torch.any(slc > dict_thresh):
                # mx = sim[c_idx, :].max()
                # mn = sim[c_idx, :].min()

                if len(c_added_idx) > 0:
                    # Compare with any of the other candidates that have already been added during this iteration
                    t = torch.tensor(c_added_idx)

                    slc = c_sim[c_idx, t]
                    if torch.any(slc > dict_thresh).item():
                        continue
                    # mx = max(mx, c_sim[c_idx, t].max())
                    # mn = min(mn, c_sim[c_idx, t].min())


                phi[:, ptr] = cand[:, c_idx]
                ptr += 1
                pbar.update(1)
                c_added_idx.append(c_idx)
                idx_list.append(shuf[i])

                # if args.vis and dset is not None:
                #     print(f"Max similarity: {mx}, min similarity: {mn}")
                #     dset.dict_vis(phi)

                if ptr == dict_sz:
                    break

        if ptr == dict_sz:
            print(f"Found {dict_sz} sufficiently (<={dict_thresh}) unique dictionary elements in {i} / {x.shape[1]} ({i / x.shape[1]}) tries.")
            break
    assert ptr == dict_sz, f"Unable to find {dict_sz} sufficiently (<={dict_thresh}) unique dictionary elements in the dataset (only found {ptr})."
    pbar.close()
    if torch.cuda.is_available():
        phi = phi.cpu()

    # shuffle dictionary elements to avoid having dict elements with densest codes being grouped together in the low indices of the dictionary (allows increasing size of GPU slices later)
    perm = torch.randperm(phi.shape[1])
    idx_list = [idx_list[i] for i in perm.tolist()]
    phi = phi[:, perm]
    if args.dict_path != "":
        out_phi = phi
        if isinstance(phi, (torch.Tensor,)):
            out_phi = phi.numpy()
        elif 'scipy.sparse' in str(type(phi)):      # janky check for sparse representations
            out_phi = phi.todense()
        np.save(args.dict_path, out_phi)

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return phi.float(), idx_list






