import numpy as np
import torch
from tqdm import tqdm
import random

from matrix_utils import profile_log
from time import time

import pdb

class SparseCodeLayer:
    """Encode inputs in the given dictionary basis. Optionally, generate a random basis from the first data passed to this object."""
    def __init__(self, dict_sz, phi, gq_thresh):
        self.basis = phi
        self.dict_sz = dict_sz
        self.gq_thresh = gq_thresh
        
        # usage of basis assumes it has unit norm
        assert (torch.linalg.norm(self.basis, dim=0) - 1).abs().max() < 1e-3

    def sparse_code_img(self, patches):
        # Given the centered and whitened patches for a single image, return their sparse codes
        return self.__call__(patches)

    @torch.compiler.disable
    def __call__(self, data):
        # Data and phi are both L2 normalized, so their cosine similarity is their dot product
        assert self.basis.shape[1] == self.dict_sz
        assert len(data.shape) == 2
        if self.basis.device != data.device:
            self.basis = self.basis.to(data.device, non_blocking=True)
        cosine_sim = self.basis.T @ data

        mask = cosine_sim >= self.gq_thresh
        cosine_sim[mask] = 1
        mask = ~mask
        cosine_sim[mask] = 0
        return cosine_sim



def generate_dict(args, dset, dict_sz, dict_thresh):
    """Generate dictionary elements from the already generated data points.
    NOTE: could run dictionary learning with multiple initialization and pick the best (like normal K-Means)
    """
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
        cand = torch.zeros((patch_dim, chnk_sz * dset.n_patch_per_img), device="cuda:0")
        for idx in range(start, end):
            c_idx = idx - start
            cand[:, c_idx * dset.n_patch_per_img : (c_idx + 1) * dset.n_patch_per_img] = dset.get_single_train_image(shuf[idx], cuda=True)[0]

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

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return phi.float()



