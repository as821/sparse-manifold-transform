import torch
from einops import rearrange
from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import resource
import numpy as np
import time
import math

import pdb

from matrix_utils import mx_frac_pow, torch_force_symmetric

def generate_dset(args, split='train', train_set=None, whiten_op=None, unwhiten_op=None):
    # Select and generate dataset
    if args.dataset == "mnist":
        if train_set is None:
            dset = ImagePreprocessor(args, torchvision.datasets.MNIST, split=split, n_channels=1, whiten_op=whiten_op, unwhiten_op=unwhiten_op)
        else:
            assert whiten_op is None and unwhiten_op is None
            dset = ImagePreprocessor(args, torchvision.datasets.MNIST, split=split, n_channels=1, whiten_op=train_set.whiten_op, unwhiten_op=train_set.unwhiten_op)
    elif args.dataset == "cifar10":
        if train_set is None:
            dset = ImagePreprocessor(args, torchvision.datasets.CIFAR10, split=split, whiten_op=whiten_op, unwhiten_op=unwhiten_op)
        else:
            assert whiten_op is None and unwhiten_op is None
            dset = ImagePreprocessor(args, torchvision.datasets.CIFAR10, split=split, whiten_op=train_set.whiten_op, unwhiten_op=train_set.unwhiten_op)
    else:
        raise NotImplementedError
    return dset


class PatchDataset(Dataset):
    def __init__(self, parent, num_samples, stride):
        self.parent = parent
        self.num_samples = num_samples
        self.stride = stride

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = self.parent.train_set_image(idx, False)[0]
        return img

class ImagePreprocessor():
    def __init__(self, args, dset_obj, split='train', n_channels=None, whiten_op=None, unwhiten_op=None):
        self.args = args
        assert split in ['train', 'test']
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

        if whiten_op is not None:
            assert unwhiten_op is not None
            self.whiten_op = whiten_op
            self.unwhiten_op = unwhiten_op

        # Increase system limit on number of open files (limit "too many open files" errors during parallelization)
        # _, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (65536*16, 65536*16))

        self.ctx_kernel_size = 2 * self.args.context_sz + 1
        self.ctx_kernel = torch.ones((1, 1, self.ctx_kernel_size, self.ctx_kernel_size), device="cuda")

    def img_to_centered_patches(self, img, stride):
        patches = torch.nn.functional.unfold(img, self.args.patch_sz, stride=stride).clone()
        n_patch_per_dim = int(math.sqrt(patches.shape[1]))
        assert n_patch_per_dim ** 2 == patches.shape[1]
        patches = rearrange(patches, "(a b) (c d) -> c d b a", a=self.n_inp_channels, c=n_patch_per_dim, d=n_patch_per_dim)

        # context mean is per-channel (and per patch location as well)
        tmp_patches = rearrange(patches, "a b c d -> (c d) a b").unsqueeze(1)
        ctx_sums = torch.nn.functional.conv2d(tmp_patches, self.ctx_kernel, padding=self.args.context_sz)[:, 0, ...]
        
        # accurately determine the context size for each patch (should be able to just do this in the constructor)
        ones = torch.ones_like(tmp_patches)
        ctx_cnts = torch.nn.functional.conv2d(ones, self.ctx_kernel, padding=self.args.context_sz)[:, 0, ...]
        ctx_means = rearrange(ctx_sums / ctx_cnts, "(c d) a b -> a b c d", c=patches.shape[2])

        patches -= ctx_means
        patches = rearrange(patches, "b c d e -> (b c) (d e)")
        return patches

    def apply_and_reduce(self, func, stride=1, cuda=False, whiten=True, bilinear_func=None):
        # TODO: slowest part of this loop are the patches matmuls by far (followed by the addition at the bottom)

        # weirdly pinned memory is a bit slower, maybe due to very small allocation/transfer sizes?
        data_loader = DataLoader(PatchDataset(self, self.args.samples, stride), batch_size=1, shuffle=False, num_workers=16, pin_memory=False)
        
        # Generate centered (and possibly whitened) image patches for an image, apply the given function, then perform a matrix multiplication over the dataset
        out = None
        for img in tqdm(data_loader):
            assert img.shape[0] == 1        # batch size 1
            img = img[0]
            img = img.to("cuda", non_blocking=True)

            patches = self.img_to_centered_patches(img, stride)
            if whiten:
                patches = self._whiten_normalize_patch(patches)
            if func is not None:
                patches = func(patches)
            
            if bilinear_func is not None:
                form = bilinear_func(patches)
                prod = patches.T @ form @ patches
            else:
                prod = patches.T @ patches
            
            if out is None:
                out = prod
            else:
                out += prod
        return out
    
    def apply_and_sum(self, func, dim=0, stride=1, cuda=False, whiten=True):
        # Generate centered (and possibly whitened) image patches for an image, apply the given function, then perform a sum over the dataset
        out = None
        for idx in tqdm(range(self.args.samples)):
            patches = self.img_to_centered_patches(self.train_set_image(idx, cuda)[0], stride)
            if whiten:
                patches = self._whiten_normalize_patch(patches)
            if func is not None:
                patches = func(patches)
            if out is None:
                out = patches.sum(dim=dim)
            else:
                out += patches.sum(dim=dim)
        return out

    def calc_whitening(self, stride=1):
        # Calculate mean for each patch channel
        mean = self.apply_and_sum(None, dim=0, stride=stride, cuda=True, whiten=False)
        mean /= (self.args.samples * self.n_patch_per_img)

        # Calculate centered covariance matrix
        def sub(patches):
            return patches - mean
        cov_mx = self.apply_and_reduce(sub, stride, cuda=True, whiten=False) / (self.args.samples * self.n_patch_per_img)
        cov_mx = cov_mx.to(torch.float32)
        cov_mx = torch_force_symmetric(cov_mx)

        # Calculate whitening/unwhitening
        self.whiten_op = mx_frac_pow(cov_mx, -1/2, self.args.whiten_tol)
        self.unwhiten_op = mx_frac_pow(cov_mx, 1/2, self.args.whiten_tol)
        return self.whiten_op, self.unwhiten_op

    def train_set_image(self, idx, cuda=False):
        # default to using original + horizontal augmented images
        # alternate between "normal" and augmented datasets 
        if idx % 2 == 0:
            sample = self.dataset[int(idx / 2)]
            img = sample[0]
            if cuda:
                img = img.to("cuda:0", non_blocking=True)
            sample = (img, sample[1])
        else:
            # manually apply a horizontal flip augmentation
            sample = self.dataset[int((idx-1) / 2)]
            img = transforms.functional.hflip(sample[0])
            if cuda:
                img = img.to("cuda:0", non_blocking=True)
            if len(sample) == 1:
                sample = (img,)
            else:
                sample = (img, sample[1])
        return sample

    def _whiten_normalize_patch(self, patches):
        assert self.whiten_op is not None and self.unwhiten_op is not None
        patches = patches.T
        if self.whiten_op.device != patches.device:
            self.whiten_op = self.whiten_op.to(patches.device, non_blocking=True)
        patches = self.whiten_op @ patches 
        patches += 1e-20          # do not allow any patch to have a zero norm representation
        patches /= torch.linalg.vector_norm(patches, ord=2, dim=0)
        return patches

    def get_single_train_image(self, idx, stride=1, cuda=False):
        # Return a single preprocessed image
        assert idx < 2 * len(self.dataset)
        sample, label = self.train_set_image(idx, cuda=cuda)
        patches = self.img_to_centered_patches(sample, stride)
        patches = self._whiten_normalize_patch(patches)
        return patches, label

    def get_single_eval_image(self, idx, stride=1, cuda=False):
        # Return a single preprocessed image
        assert idx < len(self.dataset)
        sample = self.dataset[idx]
        patches = self.img_to_centered_patches(sample[0].to("cuda:0" if cuda else "cpu"), stride)
        patches = self._whiten_normalize_patch(patches)
        return patches, sample[1]

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
        
    def _context_mean(self, patches):
        """Calculate mean of context patches for each patch in each image."""

        # Setup padding so get a contextual mean for each patch location in the image
        conv = torch.nn.Conv2d(in_channels=self.input_patch_dim,
                                out_channels=1, 
                                kernel_size=(2 * self.context_sz + 1, 2 * self.context_sz + 1), 
                                stride=1,
                                padding=self.context_sz)

        # Set weights to 1s and bias to 0 to sum all entries in the conv filter window
        conv.weight.data.fill_(1)
        conv.bias.data.fill_(0)

        # Apply conv. to all images + patches to get contextual sums
        tmp = rearrange(patches, "a b c d e -> a (d e) b c")
        if torch.cuda.is_available():
            conv = conv.to("cuda:0", non_blocking=True)

        # Perform means per-channel, treat different channels like different batches
        # TODO: treats all channels as the same!! --> should be doing this per channel
        res = torch.zeros(size=(tmp.shape[0], 1, tmp.shape[2], tmp.shape[3]), dtype=patches.dtype)
        chnk = 300
        for start in tqdm(range(0, tmp.shape[0], chnk)):     # process each image separately
            end = min(tmp.shape[0], start+chnk)
            tmp_slice = tmp[start:end]
            if torch.cuda.is_available():
                tmp_slice = tmp_slice.to("cuda:0", non_blocking=True)
            res[start:end] = conv(tmp_slice.float()).to('cpu').type(patches.dtype)

        ctx_means = res / (self.context_sz_mx.unsqueeze(0).unsqueeze(1) * self.input_patch_dim)
        return ctx_means.squeeze().unsqueeze(-1).unsqueeze(-1)

    def aggregate_image_embed(betas, ks=4, stride=2):
        """Aggregate patch-level embeddings into image-level embeddings for each image, following procedure outlined by (2)"""
        print("Aggregating image embeddings...")
        betas = betas.permute((0, 3, 1, 2))

        pool = torch.nn.AvgPool2d(kernel_size=ks, stride=stride)
        dev = 'cpu'
        if torch.cuda.is_available():
            pool = pool.to("cuda", non_blocking=True)
            dev = 'cuda:0'

        sz = pool(betas[0].to(dev)).shape[-1]       # pass single image through to get output shape

        chnk_sz = 50
        for start in tqdm(range(0, betas.shape[0], chnk_sz)):
            end = min(start + chnk_sz, betas.shape[0])
            chnk = pool(betas[start:end].to(dev))
            
            # Apply "point-wise L2 normalization" (L2 normalize each aggregated patch of each image)
            chnk /= (torch.linalg.vector_norm(chnk, ord=2, dim=1, keepdim=True) + 1e-20)
            betas[start:end, :, :sz, :sz] = chnk.cpu()

        betas = betas[:, :, :sz, :sz]
        return torch.flatten(betas, start_dim=1)

    def generate_embeddings(self, sc_layer, smt_layer, stride=1, cuda=False):
        """Apply calculated SMT to this dataset. NOTE: uses full dataset regardless of args"""
        embed = torch.zeros((len(self.dataset), self.n_patch_per_img, self.args.embed_dim), device="cpu")
        labels = torch.zeros((len(self.dataset)), device="cpu")
        for idx in tqdm(range(len(self.dataset))):
            patches, label = self.get_single_eval_image(idx, stride, cuda)
            embed[idx, :] = smt_layer(sc_layer(patches)).T.cpu()
            labels[idx] = label
        return embed, labels

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

