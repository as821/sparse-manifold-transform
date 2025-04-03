import torch
from tqdm import tqdm

import pdb

from preprocessor import _context

class DifferentialOperator():
    def __init__(self, args, dset):
        self.args = args
        self.dset = dset
        self.custom_dop = 0
        
        # initialize the differential operator to use if there are no patches with a code of all zeros
        self.ctx_pairs = dset.get_context_pairs(self.args.context_sz)
        if self.args.optim == "one":
            self.diff_op, self.index = self._opt1_diff_op()
            self.diff_op_sq = self.diff_op @ self.diff_op.T
        elif self.args.optim == "two":
            self.diff_op, self.index = self._opt2_diff_op()
            full_dop = self._opt2_diff_op_prune(set())      # _opt2_diff_op only generates a partially populated matrix
            self.diff_op_sq = full_dop @ full_dop.T
        else:
            assert False, "Invalid differential operator version"        

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
            if self.args.optim == "one":
                ret = self._opt1_diff_op_prune(zero_ind).to("cuda", non_blocking=True)
            elif self.args.optim == "two":
                ret = self._opt2_diff_op_prune(zero_ind)
            else:
                assert False, "Invalid differential operator version"
            ret = ret @ ret.T
            if ret.device != patches.device:
                ret = ret.to(patches.device, non_blocking=True)
            return ret

    def _opt1_diff_op(self):
        """Implement first-derivative contextual operator for a single image. (#patch / img) x (2 * num. neighbor pairs) matrix since context pairs needs to be symmetric"""
        index = {}
        mx = torch.zeros((self.dset.n_patch_per_img, len(self.ctx_pairs) * 2))
        for idx, px_pair in enumerate(self.ctx_pairs):
            pos_pixel = px_pair[0]
            neg_pixel = px_pair[1]
            
            # Pixels are returned as (x, y) tuples (stored in row-major order)
            pos_idx = pos_pixel[0] * self.dset.n_patch_per_dim + pos_pixel[1]
            neg_idx = neg_pixel[0] * self.dset.n_patch_per_dim + neg_pixel[1]
            assert pos_idx != neg_idx

            if pos_idx not in index:
                index[pos_idx] = []
            if neg_idx not in index:
                index[neg_idx] = []

            # need to include both ways
            mx[pos_idx, 2 * idx] = 1
            mx[neg_idx, 2 * idx] = -1
            mx[pos_idx, 2 * idx + 1] = -1
            mx[neg_idx, 2 * idx + 1] = 1

            index[pos_idx].append((pos_idx, 2 * idx))
            index[pos_idx].append((neg_idx, 2 * idx))
            index[neg_idx].append((pos_idx, 2 * idx))
            index[neg_idx].append((neg_idx, 2 * idx))

        return mx, index

    def _opt1_diff_op_prune(self, zero_ind):
        op = self.diff_op.clone()
        for idx in zero_ind:
            for pr in self.index[idx]:
                op[pr[0], pr[1]] = 0
                op[pr[0], pr[1] + 1] = 0
        return op

    def _opt2_diff_op(self):
        # A partially constructed differential operator, without its central diagonal populated
        mx = torch.zeros((self.dset.n_patch_per_img, self.dset.n_patch_per_img))
        for pidx in range(self.dset.n_patch_per_img):
            x = int(pidx / self.dset.n_patch_per_dim)
            y = int(pidx % self.dset.n_patch_per_dim)
            ctx = _context(x, y, self.dset.n_patch_per_dim, self.args.context_sz)
            for px_pair in ctx:
                n_x, n_y = px_pair
                neighbor_idx = n_x * self.dset.n_patch_per_dim + n_y
                mx[neighbor_idx, pidx] = -1
        return mx, None

    def _opt2_diff_op_prune(self, zero_ind):
        # Outputs a (# patch per image x # patch per image) matrix. Diagonal entries are 1, all others are negative, columns must sum to zero
        # prune all references to zero codes
        op = self.diff_op.clone().to("cuda", non_blocking=True)
        op[list(zero_ind), :] = 0
        op[:, list(zero_ind)] = 0

        # scale remaining neighbor entries (normalize over columns for )
        s = torch.sum(op, dim=0, keepdims=True)
        s *= -1
        s_zero_idx = s == 0
        s[s_zero_idx] = 1       # avoid div by zero
        op /= s

        # set diagonal for "original images" to 1 (if patch for column is not zero and if it has any nonzero neighbors)
        r = [i for i in range(self.dset.n_patch_per_img) if i not in zero_ind and not s_zero_idx[0][i]]
        op[r, r] = 1
        return op

