"""
Implementation of the sparse manifold transform (SMT).

References:
(1): The Sparse Manifold Transform https://arxiv.org/pdf/1806.08887.pdf 
(2): Minimalistic Unsupervised Representation Learning with the Sparse Manifold Transform https://arxiv.org/pdf/2209.15261.pdf
"""

# NOTE: (also, increasing whitening tolerance decreases sparsity, could use stricter dictionary + maybe make those work better?)

import torch
from einops import rearrange
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.append(os.path.join(os.getcwd(), 'src/c'))
from preprocessor import ImagePreprocessor
from manifold_embedding import ManifoldEmbedLayer
from diff_op import construct_diff_op
from util import generate_dset_dict_codes, generate_argparser, validate_args
from input_output import mmap_csr_cleanup
from classifier import test_set_classify

import numpy as np

def main(args):
    # Generate dataset and sparse codes for images
    dset, alphas, sc_layer, img_label = generate_dset_dict_codes(args)

    # convert argument specified in number of images into number of patches
    orig_diff_op_d_chunk = args.diff_op_d_chunk
    args.diff_op_d_chunk *= dset.n_patch_per_img
    print(f"Converted diff_op_d_chunk: {args.diff_op_d_chunk} images -> {args.diff_op_d_chunk * dset.n_patch_per_img} patches ({dset.n_patch_per_img} patch/img)")

    # Perform SMT calculations for the given dataset        
    diff_op = construct_diff_op(args, dset, alphas, args.context_sz, orig_diff_op_d_chunk)
    print(f"Total sparsity: {alphas.nnz / (alphas.shape[0] * alphas.shape[1])}", flush=True)

    # Generate embedding matrix (P matrix) and embeddings (betas)
    smt_layer = ManifoldEmbedLayer(args, alphas, diff_op, args.embed_dim)

    print(f"Calculating layer embeddings...", flush=True)
    betas = smt_layer(alphas)
    if isinstance(alphas.data, (np.memmap)):
        mmap_csr_cleanup(alphas)

    # Normalize image patches and aggregate into image-level representation
    betas = rearrange(betas, "d (a b c) -> a b c d", a=args.samples, b=dset.n_patch_per_dim, c=dset.n_patch_per_dim, d=args.embed_dim)
    assert betas.shape[0] == args.samples and betas.shape[1] == betas.shape[2] == dset.n_patch_per_dim and betas.shape[3] == args.embed_dim
    img_embed = ImagePreprocessor.aggregate_image_embed(betas)

    print("Test set accuracy: ", test_set_classify(args, dset, sc_layer, smt_layer, img_embed, img_label))


if __name__ == "__main__":

    # https://github.com/cupy/cupy/issues/3431#issuecomment-647931780
    # https://github.com/numpy/numpy/blob/da1621637b7c59c155ec29466fb5f810ebd902ac/numpy/__init__.py#L334-L353
    os.environ["NUMPY_MADVISE_HUGEPAGE"] = "1"

    args = validate_args(generate_argparser().parse_args())
    with torch.no_grad():
        main(args)
    print("Done")




