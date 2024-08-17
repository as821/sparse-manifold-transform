"""
Implementation of the sparse manifold transform (SMT).

References:
(1): The Sparse Manifold Transform https://arxiv.org/pdf/1806.08887.pdf 
(2): Minimalistic Unsupervised Representation Learning with the Sparse Manifold Transform https://arxiv.org/pdf/2209.15261.pdf
"""

# NOTE: (also, increasing whitening tolerance decreases sparsity, could use stricter dictionary + maybe make those work better?)

import argparse
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
from classifier import WeightedKNNClassifier
from util import generate_dset_dict_codes, test_set_classify, generate_dset, generate_argparser
from input_output import mmap_csr_cleanup
from sparse_code import SparseCodeLayer, generate_dict
from patch_cluster import cluster_patches

import numpy as np
from time import time
import json



def load_ckpt(args):
    print("Loading checkpoint...", flush=True)
    path = args.ckpt_path
    if path[-1] != "/":
        path += "/"

    basis = torch.load(path + "sc_basis.pt")
    sc_layer = SparseCodeLayer(basis.shape[1], basis)
    smt_layer = ManifoldEmbedLayer(args, None, None, np.load(path + "smt_proj.npy"))
    with open(path + "args.json", "r") as file:
        ckpt_args = json.load(file)

    print("NOTE differences between checkpoint and current args:")
    cur_args = vars(args)
    for k in ckpt_args:
        if k in cur_args and cur_args[k] != ckpt_args[k]:
            print(f"\t{k}: ({ckpt_args[k]}) vs. current ({cur_args[k]})")

    # relies on global-scope arg parser
    ckpt_args_argparse = parser.parse_args(args=[], namespace=argparse.Namespace(**ckpt_args))
    return ckpt_args_argparse, sc_layer, smt_layer



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
    all_match = len(args.dict_sz) == len(args.dict_thresh) and len(args.dict_sz) == len(args.embed_dim) and len(args.dict_sz) == len(args.gq_thresh) and len(args.dict_sz) == len(args.context_sz)
    assert all_match, "number of SMT layers implicitly defined by number of dict size, threshold parameters, + embedding dimensions"

    if args.dataset != "clevr":
        assert args.patch_sz == 6
    if args.vis_dir != "" and args.depatchify == "center":
        assert args.patch_sz % 2 == 1

    if args.vis_dir != "" and os.path.exists(args.vis_dir):
        shutil.rmtree(args.vis_dir)
        os.mkdir(args.vis_dir)

    for i, j in zip(args.dict_sz, args.embed_dim):
        assert j <= i, f"Cannot have more embedding dimensions ({j}) than dictionary elements ({i}) (due to linear algebra)."
    if args.cov_chunk < 0: args.cov_chunk = args.dict_sz
    if args.inner_chunk < 0: args.inner_chunk = args.dict_sz
    if args.n_aug > 0: 
        args.samples *= (args.n_aug+1)        # +1 since keep the original as well
    elif args.dataset != "clevr":
        args.samples *= 2        # includes horizontal augmentations by default

    t_dset = None
    smt_layers = []
    sc_layers = []
    if args.load_ckpt:
        assert args.ckpt_path != "", "Invalid checkpoint path"
        assert not args.sc_only
        
        # load checkpoint
        _, sc_layer, smt_layer = load_ckpt(args)
        
        # load dataset, generate sparse codes
        print("Setting up dataset + generating sparse codes in given basis...", flush=True)
        dset = generate_dset(args)
        args.diff_op_d_chunk *= dset.n_patch_per_img
        x, img_label = dset.generate_data(args.samples)
        alphas = sc_layer(args, x)
        del x
        t_dset = dset       # use train set whitening matrix on the test set

        print("Calculating layer embeddings...", flush=True)
        betas = smt_layer(alphas)
        if isinstance(alphas.data, (np.memmap)):
            mmap_csr_cleanup(alphas)

        sc_layers.append(sc_layer)
        smt_layers.append(smt_layer)

        if args.vis_dir != "":
            cluster_patches(args, dset, betas)
            return
    elif args.sc_only:
        # Perform SMT calculations for the given dataset
        # Note: this function assumes no other memory maps have been used prior to this function running
        dset, alphas, sc_layer, img_label = generate_dset_dict_codes(args)
        args.diff_op_d_chunk *= dset.n_patch_per_img
        print(f"Total sparsity: {alphas.nnz / (alphas.shape[0] * alphas.shape[1])}", flush=True)
        t_dset = dset       # use train set whitening matrix on the test set
        num_zero = np.where(alphas.sum(axis=0) == 0)[0].shape[0]
        print("\tSC {:.5f} percent ({}) 'zero' patches...".format((num_zero / alphas.shape[1]) * 100, num_zero))    

        if args.vis:
            dset.dict_vis(sc_layer.basis, alphas.todense())

        classifier = WeightedKNNClassifier(k=args.nnclass_k, T=0.03)
        print("Alphas rearrange...")
        alphas = rearrange(alphas.todense(), "d (a b c) -> a b c d", a=args.samples, b=dset.n_patch_per_dim, c=dset.n_patch_per_dim)
        
        print("Aggregating train set image embeddings...")
        alphas = ImagePreprocessor.aggregate_image_embed(alphas)
        # classifier.update(train_features=alphas, train_targets=img_label)
        test_acc = test_set_classify(args, t_dset, sc_layer, None, classifier, alphas, img_label)
        print("Test set accuracy: ", test_acc, flush=True)      
        return      
    else:

        # Perform SMT calculations for the given dataset
        # Note: this function assumes no other memory maps have been used prior to this function running
        dset, alphas, sc_layer, img_label = generate_dset_dict_codes(args)
        print(f"Converted diff_op_d_chunk: {args.diff_op_d_chunk} -> {args.diff_op_d_chunk * dset.n_patch_per_img} ({dset.n_patch_per_img})")
        t_dset = dset       # use train set whitening matrix on the test set

        betas = None
        orig_diff_op_d_chunk = args.diff_op_d_chunk
        args.diff_op_d_chunk *= dset.n_patch_per_img
        for idx, pr in enumerate(zip(args.dict_sz, args.dict_thresh, args.embed_dim, args.gq_thresh, args.context_sz)):
            dict_sz, dict_thresh, embed_dim, gq_thresh, context_sz = pr        
            
            diff_op = construct_diff_op(args, dset, alphas, context_sz, orig_diff_op_d_chunk)
            if idx != 0:
                # feed betas from last layer into the current layer
                assert dict_sz <= args.samples * dset.n_patch_per_img, "Dictionary cannot contain more landmarks than there are image patches."  
                betas = torch.from_numpy(betas).type(torch.float32)
                
                # TODO(as) ACTUALLY! some betas have zero norm...



                # TODO(as): should we also be mean removing + whitening betas?                
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
        if args.ckpt_path != "": save_ckpt(args, sc_layer, smt_layer)

    if args.vis_dir != "":
        cluster_patches(args, dset, betas)

    # TODO(as) clevr metrics not set up yet
    if args.dataset == "clevr":
        return

    # Implement normalization / aggregation to image-level representation
    betas = rearrange(betas, "d (a b c) -> a b c d", a=args.samples, b=dset.n_patch_per_dim, c=dset.n_patch_per_dim, d=args.embed_dim[-1])
    assert betas.shape[0] == args.samples and betas.shape[1] == betas.shape[2] == dset.n_patch_per_dim and betas.shape[3] == args.embed_dim[-1]
    img_embed = ImagePreprocessor.aggregate_image_embed(betas)

    classifier = WeightedKNNClassifier(k=args.nnclass_k, T=args.knn_temp)
    test_acc = test_set_classify(args, t_dset, sc_layers, smt_layers, classifier, img_embed, img_label)
    print("Test set accuracy: ", test_acc, flush=True)


if __name__ == "__main__":

    # Tell numpy to use 
    # https://github.com/cupy/cupy/issues/3431#issuecomment-647931780
    # https://github.com/numpy/numpy/blob/da1621637b7c59c155ec29466fb5f810ebd902ac/numpy/__init__.py#L334-L353
    os.environ["NUMPY_MADVISE_HUGEPAGE"] = "1"

    args = generate_argparser().parse_args()
    
    # Clear + create mmap directory
    assert args.mmap_path[-1] != '/', "Filename format error. Assumes no trailing slash."
    if os.path.exists(args.mmap_path):
        shutil.rmtree(args.mmap_path)
    os.mkdir(args.mmap_path)

    with torch.no_grad():
        main(args)
    print("Done")






