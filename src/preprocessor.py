import torch
import math
from einops import rearrange
from tqdm import tqdm
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
import resource
import time
import numpy as np

from matrix_utils import mx_frac_pow, torch_force_symmetric

def generate_dset(args, split='train', train_set=None):
    # Select and generate dataset
    if args.dataset == "mnist":
        dset = ImagePreprocessor(args, torchvision.datasets.MNIST, split=split, n_channels=1, train_set=train_set)
    elif args.dataset == "cifar10":
        dset = ImagePreprocessor(args, torchvision.datasets.CIFAR10, split=split, train_set=train_set)
    else:
        raise NotImplementedError
    return dset


class ImagePreprocessor():
    def __init__(self, args, dset_obj, split='train', n_channels=None, train_set=None):
        self.args = args
        assert split in ['train', 'test']
        if train_set is not None:
            assert split == 'test'
        self.split = split
        trans = [transforms.ToTensor()]
        if args.grayscale_only:
            trans.append(transforms.Grayscale())
        
        self.dataset = dset_obj(root=self.args.dataset_path, train=split=='train', transform=transforms.Compose(trans), download=True)
        self.n_class = len(self.dataset.classes)
        
        data = self.dataset[0][0]
        self.img_sz = data.shape[1]
        if n_channels is not None:
            self.n_inp_channels = n_channels
        elif args.grayscale_only:
            self.n_inp_channels = 1
        else:
            self.n_inp_channels = data.shape[0]

        self.n_patch_per_dim = self.img_sz - self.args.patch_sz + 1
        self.n_patch_per_img = self.n_patch_per_dim ** 2
        self.input_patch_dim = args.patch_sz ** 2 * self.n_inp_channels

        self.whiten_op = None
        self.unwhiten_op = None
        self.context_sz = args.context_sz

        if train_set:
            self.whiten_op = train_set.whiten_op
            self.unwhiten_op = train_set.unwhiten_op

        self.patch_cpy = None
        self.c_means = None
        self.norm = None

        # Increase system limit on number of open files (limit "too many open files" errors during parallelization)
        # _, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (65536*16, 65536*16))

    def _context_mean(self, patches):
        """Calculate mean of context patches for each patch in each image. TODO: sliding window?"""

        # Setup padding so get a contextual mean for each patch location in the image
        conv = torch.nn.Conv2d(in_channels=self.input_patch_dim,
                                out_channels=1, 
                                kernel_size=(2 * self.context_sz + 1, 2 * self.context_sz + 1), 
                                stride=1,
                                padding=self.context_sz)

        # Set weights to 1s and bias to 0 to sum all entries in the conv filter window
        conv.weight.data.fill_(1)
        conv.bias.data.fill_(0)

        # Calculate context size for each patch location
        # NOTE: patches are stored in row-major order. x is used to index rows and y is used to index columns
        context_sz = torch.zeros(size=(self.n_patch_per_dim, self.n_patch_per_dim))
        for x in range(self.n_patch_per_dim):
            for y in range(self.n_patch_per_dim):
                context_sz[x, y] = len(_context(x, y, self.n_patch_per_dim, self.context_sz)) + 1

        # Apply conv. to all images + patches to get contextual sums
        tmp = rearrange(patches, "a b c d e -> a (d e) b c")
        if torch.cuda.is_available():
            conv = conv.to("cuda:0", non_blocking=True)

        # Perform means per-channel, treat different channels like different batches
        res = torch.zeros(size=(tmp.shape[0], 1, tmp.shape[2], tmp.shape[3]), dtype=patches.dtype)
        chnk = 300
        for start in tqdm(range(0, tmp.shape[0], chnk)):     # process each image separately
            end = min(tmp.shape[0], start+chnk)
            tmp_slice = tmp[start:end]
            if torch.cuda.is_available():
                tmp_slice = tmp_slice.to("cuda:0", non_blocking=True)
            res[start:end] = conv(tmp_slice.float()).to('cpu').type(patches.dtype)

        ctx_means = res / (context_sz.unsqueeze(0).unsqueeze(1) * self.input_patch_dim)
        return ctx_means.squeeze().unsqueeze(-1).unsqueeze(-1)

    def _calc_whitening(self, patches):
        """Given tensor of centered patches, calculate the whitening (and unwhitening) operators."""
        if self.whiten_op is not None:
            assert self.unwhiten_op is not None
            return
        assert self.unwhiten_op is None

        cov_mx = torch.cov(patches.T)
        cov_mx = torch_force_symmetric(cov_mx)
        tol = self.args.whiten_tol      # NOTE: this is actually a crucial parameter that can swing performance by multiple percentage points
        self.whiten_op = mx_frac_pow(cov_mx, -1/2, tol)
        self.unwhiten_op = mx_frac_pow(cov_mx, 1/2, tol)

    def train_set_image(self, idx):
        # default to using original + horizontal augmented images
        # alternate between "normal" and augmented datasets 
        if idx % 2 == 0:
            sample = self.dataset[int(idx / 2)]
        else:
            # manually apply a horizontal flip augmentation
            sample = self.dataset[int((idx-1) / 2)]
            img = transforms.functional.hflip(sample[0])
            if len(sample) == 1:
                sample = (img,)
            else:
                sample = (img, sample[1])
        return sample


    def generate_data(self, n_samples, test=False):
        """Generate (and patch-ify) given number of samples from the MNIST dataset"""
        print("Patch-ifying images...", flush=True)
        patches = torch.zeros(size=(n_samples, self.n_patch_per_dim, self.n_patch_per_dim, self.args.patch_sz**2, self.n_inp_channels), dtype=torch.float32)
        if self.n_class > 0:
            labels = torch.zeros(size=(n_samples, 1), dtype=torch.from_numpy(np.empty(shape=(1,), dtype=np.min_scalar_type(-1 * self.n_class))).dtype)  # get min. scalar type needed for labels
        else:
            labels = None

        patch_dim = self.args.patch_sz * self.args.patch_sz * self.n_inp_channels
        conv = torch.nn.Conv2d(in_channels=self.n_inp_channels, 
                                out_channels=patch_dim, 
                                kernel_size=(self.args.patch_sz, self.args.patch_sz), 
                                stride=1,
                                padding=0)

        # One filter for each location in the patch (flattens 3 color channels as well)
        conv.weight.data.fill_(0)
        conv.bias.data.fill_(0)
        for top_idx in range(self.args.patch_sz):
            for left_idx in range(self.args.patch_sz):
                for c_idx in range(self.n_inp_channels):
                    conv.weight.data[top_idx * self.args.patch_sz*self.n_inp_channels + left_idx*self.n_inp_channels + c_idx, c_idx, top_idx, left_idx] = 1

        if torch.cuda.is_available():
            conv = conv.to("cuda:0", non_blocking=True)

        # Convert images to image patches and store ground truth labels
        for idx in tqdm(range(n_samples)):
            if test:
                sample = self.dataset[idx]
            else:
                sample = self.train_set_image(idx)

            if len(sample) > 1:
                labels[idx] = sample[1]
            img = sample[0]
            if torch.cuda.is_available():
                img = img.to("cuda:0", non_blocking=True)
            res = conv(img).squeeze()
            res = rearrange(res, "(a b) c d -> a b c d", a=self.args.patch_sz**2, b=self.n_inp_channels, c=self.n_patch_per_dim, d=self.n_patch_per_dim)
            patches[idx] = res.permute((2, 3, 0, 1)).to("cpu")
        del conv, sample, res

        # Remove the contextual mean from each patch (centering before whitening)
        print("Calculating contextual patch means...", flush=True)
        c_means = self._context_mean(patches)
        patches = patches - c_means

        # Calculate whitening operator + apply it
        print("Calculating and applying whitening operator to all patches...", flush=True)
        patches = rearrange(patches, "a b c d e -> (a b c) (d e)")
        self._calc_whitening(patches)
        patches = patches.T
        patches = self.whiten_op @ patches 
        patches += 1e-20          # do not allow any patch to have a zero norm representation
        norm = torch.linalg.vector_norm(patches, ord=2, dim=0)
        patches /= norm

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return patches, labels

    def get_context_pairs(self, context_sz):
        """Return the set of all patch context pairs for a single image in this dataset."""
        pairs = {}      # dict of sets, indexed by (x, y) pixel location
        for x in range(self.n_patch_per_dim):
            for y in range(self.n_patch_per_dim):
                pairs[(x, y)] = set()
                for nbr in _context(x, y, self.n_patch_per_dim, context_sz):
                    if nbr not in pairs or (x, y) not in pairs[nbr]:
                        pairs[(x, y)].add(nbr)
        
        # Coallesce into list of pixel pairs
        out = []
        for pixel in pairs:
            out.extend([(pixel, nbr) for nbr in pairs[pixel]])
        return out

    def aggregate_image_embed(betas, ks=4, stride=2):
        """Aggregate patch-level embeddings into image-level embeddings for each image, following procedure outlined by (2)"""
        print("Aggregating image embeddings...")
        betas = torch.from_numpy(betas).permute((0, 3, 1, 2))

        pool = torch.nn.AvgPool2d(kernel_size=ks, stride=stride)
        dev = 'cpu'
        if torch.cuda.is_available():
            pool = pool.to("cuda", non_blocking=True)
            dev = 'cuda:0'
            torch.cuda.empty_cache()

        sz = pool(betas[0].to(dev)).shape[-1]       # pass single image through to get output shape

        chnk_sz = 50
        for start in tqdm(range(0, betas.shape[0], chnk_sz)):
            end = min(start + chnk_sz, betas.shape[0])
            chnk = pool(betas[start:end].to(dev))
            
            # Apply "point-wise L2 normalization" (L2 normalize each aggregated patch of each image)
            chnk /= (torch.linalg.vector_norm(chnk, ord=2, dim=1, keepdim=True) + 1e-20)
            betas[start:end, :, :sz, :sz] = chnk.cpu()

        betas = betas[:, :, :sz, :sz]
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return torch.flatten(betas, start_dim=1)

def _context(x, y, n_patches, context_sz):
    """Given the index of a patch in the image, return the indices of its neighbors (context). DOES NOT include the given index."""
    assert x < n_patches and y < n_patches
    neighbors = []
    for i in range(-1 * context_sz, context_sz+1):
        idx = i + x
        if idx < 0 or idx >= n_patches:
            continue
        for j in range(-1 * context_sz, context_sz+1):
            jdx = j + y
            if jdx < 0 or jdx >= n_patches or (i == 0 and j == 0):     # do not return passed in position 
                continue
            neighbors.append((idx, jdx))
    return neighbors

