import torch
from einops import rearrange
from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms
import resource
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

    def generate_data(self, n_samples, test=False):
        """Generate (and patch-ify) given number of samples from the MNIST dataset"""
        print("Patch-ifying images...", flush=True)
        patches = torch.zeros(size=(n_samples, self.n_patch_per_dim, self.n_patch_per_dim, self.args.patch_sz**2, self.n_inp_channels), dtype=torch.float32)
        if self.n_class > 0:
            labels = torch.zeros(size=(n_samples, 1), dtype=torch.from_numpy(np.empty(shape=(1,), dtype=np.min_scalar_type(-1 * self.n_class))).dtype)  # get min. scalar type needed for labels
        else:
            labels = None

        patches = rearrange(patches, "a b c d e -> a b c (d e)")
        for img_idx in tqdm(range(n_samples)):            
            labels[img_idx] = self.dataset[img_idx][1]
            sample = self.dataset.data[img_idx]
            for x in range(self.n_patch_per_dim):
                for y in range(self.n_patch_per_dim):
                    crop = sample[x:x+self.args.patch_sz, y:y+self.args.patch_sz]
                    crop = crop.flatten()
                    patches[img_idx][x][y] = torch.from_numpy(crop - crop.mean())

        # Calculate whitening operator + apply it
        print("Calculating and applying whitening operator to all patches...", flush=True)
        patches = rearrange(patches, "a b c d -> (a b c) d")
        patches = patches.T

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

