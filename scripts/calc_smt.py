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
import shutil
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.append(os.path.join(os.getcwd(), 'src/c'))

from preprocessor import ImagePreprocessor
from manifold_embedding import ManifoldEmbedLayer
from diff_op import construct_diff_op
from util import generate_dset_dict_codes, generate_argparser, validate_args
from input_output import mmap_csr_cleanup
from sparse_code import SparseCodeLayer, generate_dict
from classifier import test_set_classify

import numpy as np
from time import time
import json

def save_ckpt(args, sc_layer, smt_layer):
    # generate directory for checkpoint
    print("Saving checkpoint...", flush=True)
    if not os.path.exists(args.ckpt_path):
        os.mkdir(args.ckpt_path)
        path = args.ckpt_path
        if path[-1] != "/":
            path += "/"
    else:
        # if path exists, generate a subdirectory
        path = args.ckpt_path
        if path[-1] != "/":
            path += "/"
        path += f"ckpt_{int(time())}/"
        os.mkdir(path)

    # save dictionary, embedding matrix, + a copy of the arguments
    torch.save(sc_layer.basis, path + "sc_basis.pt")
    np.save(path + "smt_proj.npy", smt_layer.projection)
    with open(path + "args.json", "w") as file:
        json.dump(vars(args), file, indent=4)

def main(args):
    # Generate dataset and sparse codes for images
    dset, alphas, sc_layer, img_label = generate_dset_dict_codes(args)

    # convert argument specified in number of images into number of patches
    orig_diff_op_d_chunk = args.diff_op_d_chunk
    args.diff_op_d_chunk *= dset.n_patch_per_img
    print(f"Converted diff_op_d_chunk: {args.diff_op_d_chunk} images -> {args.diff_op_d_chunk * dset.n_patch_per_img} patches ({dset.n_patch_per_img} patch/img)")

    # Perform SMT calculations for the given dataset
    smt_layers = []
    sc_layers = []
    for idx, pr in enumerate(zip(args.dict_sz, args.dict_thresh, args.embed_dim, args.gq_thresh, args.context_sz)):
        dict_sz, dict_thresh, embed_dim, gq_thresh, context_sz = pr        
        
        diff_op = construct_diff_op(args, dset, alphas, context_sz, orig_diff_op_d_chunk)
        if idx != 0:
            # feed betas from last layer into the current layer
            assert dict_sz <= args.samples * dset.n_patch_per_img, "Dictionary cannot contain more landmarks than there are image patches."  
            betas = torch.from_numpy(betas).type(torch.float32)

            # TODO(as) should we also be mean removing + whitening betas here as well?
            phi = generate_dict(args, betas, dict_sz, dict_thresh)
            sc_layer = SparseCodeLayer(dict_sz, phi, gq_thresh)
            alphas = sc_layer(args, betas)

        print(f"Total sparsity ({idx}): {alphas.nnz / (alphas.shape[0] * alphas.shape[1])}", flush=True)

        # Generate embedding matrix (P matrix) and embeddings (betas)
        smt_layer = ManifoldEmbedLayer(args, alphas, diff_op, embed_dim)
        sc_layers.append(sc_layer)
        smt_layers.append(smt_layer)
    
        print(f"({idx}) Calculating layer embeddings...", flush=True)
        betas = smt_layer(alphas)
        if isinstance(alphas.data, (np.memmap)):
            mmap_csr_cleanup(alphas)

    # TODO(as) need to fix checkpointing to support multiple layers (support storage of multiple layers)
    if args.ckpt_path != "": 
        save_ckpt(args, sc_layer, smt_layer)

    # Normalize image patches and aggregate into image-level representation
    betas = rearrange(betas, "d (a b c) -> a b c d", a=args.samples, b=dset.n_patch_per_dim, c=dset.n_patch_per_dim, d=args.embed_dim[-1])
    assert betas.shape[0] == args.samples and betas.shape[1] == betas.shape[2] == dset.n_patch_per_dim and betas.shape[3] == args.embed_dim[-1]
    img_embed = ImagePreprocessor.aggregate_image_embed(betas)

    print("Test set accuracy: ", test_set_classify(args, dset, sc_layers, smt_layers, img_embed, img_label))


if __name__ == "__main__":

    # https://github.com/cupy/cupy/issues/3431#issuecomment-647931780
    # https://github.com/numpy/numpy/blob/da1621637b7c59c155ec29466fb5f810ebd902ac/numpy/__init__.py#L334-L353
    os.environ["NUMPY_MADVISE_HUGEPAGE"] = "1"

    args = validate_args(generate_argparser().parse_args())

    if args.vis_dir != "" and os.path.exists(args.vis_dir):
        shutil.rmtree(args.vis_dir)
        os.mkdir(args.vis_dir)

    with torch.no_grad():
        main(args)
    print("Done")




