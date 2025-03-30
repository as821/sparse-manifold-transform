import torch
from tqdm import tqdm

import pdb

class DifferentialOperator():
    def __init__(self, args, dset):
        self.args = args
        self.dset = dset
        
        # initialize the differential operator to use if there are no patches with a code of all zeros
        self.ctx_pairs = dset.get_context_pairs(self.args.context_sz)
        if self.args.optim == "one":
            self.diff_op = self._opt1_diff_op()
        elif self.args.optim == "two":
            self.diff_op = opt2_diff_op(args, dset)
        else:
            assert False, "Invalid differential operator version"
        
        self.diff_op_sq = self.diff_op @ self.diff_op.T
        self.custom_dop = 0

    def get_bilinear_form(self, patches):
        # TODO: do we actually need to support this? how does it affect performance?
        code_sum = patches.sum(dim=1)
        if code_sum.min() > 0:
            if self.diff_op_sq.device != patches.device:
                self.diff_op_sq = self.diff_op_sq.to(patches.device, non_blocking=True)
            return self.diff_op_sq
        else:
            # construct a custom differential operator for this image to preserve sum to zero properties
            self.custom_dop += 1
            zero_ind = set(torch.where(code_sum == 0)[0].tolist())
            ret = self._opt1_diff_op(zero_ind)
            ret = ret @ ret.T
            if ret.device != patches.device:
                ret = ret.to(patches.device, non_blocking=True)
            return ret

    def _opt1_diff_op(self, zero_ind=set()):
        """Implement first-derivative contextual operator for a single image. (#patch / img) x (2 * num. neighbor pairs) matrix since context pairs needs to be symmetric"""
        mx = torch.zeros((self.dset.n_patch_per_img, len(self.ctx_pairs) * 2))
        for idx, px_pair in enumerate(self.ctx_pairs):
            pos_pixel = px_pair[0]
            neg_pixel = px_pair[1]
            
            # Pixels are returned as (x, y) tuples (stored in row-major order)
            pos_idx = pos_pixel[0] * self.dset.n_patch_per_dim + pos_pixel[1]
            neg_idx = neg_pixel[0] * self.dset.n_patch_per_dim + neg_pixel[1]
            assert pos_idx != neg_idx

            if pos_idx in zero_ind or neg_idx in zero_ind:
                continue

            # need to include both ways
            mx[pos_idx, 2 * idx] = 1
            mx[neg_idx, 2 * idx] = -1
            mx[pos_idx, 2 * idx + 1] = -1
            mx[neg_idx, 2 * idx + 1] = 1
        return mx

    
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

