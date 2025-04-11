
import torch
import sys
import os
import numpy as np
import pickle
import subprocess
import argparse
import shutil
import warnings
from time import time
import json

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))

from sparse_code import SparseCodeLayer, generate_dict
from manifold_embedding import ManifoldEmbedLayer


def validate_args(args):
    assert args.embed_dim <= args.dict_sz, f"Cannot have more embedding dimensions ({args.embed_dim}) than dictionary elements ({args.dict_sz})."
    assert args.samples > 0
    
    # account for horizontal augmentation
    args.samples *= 2
    return args


def generate_argparser():
    # Generic
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['mnist', 'cifar10'], help='dataset to use')
    parser.add_argument('--dataset-path', required=True, type=str, help='path to dataset (image datasets only)')
    parser.add_argument('--samples', default=50000, type=int, help='number of training samples to use')
    parser.add_argument('--test-samples', default=10000, type=int, help='number of training samples to use')
    parser.add_argument('--optim', default='two', choices=['one', 'two'], help='optimization equation to use from (2), naming follows the equation numbers from that paper. "one" is first deriv., "two" is second deriv.')

    parser.add_argument('--ckpt-path', default="/home/astange/smt_ckpt", type=str, help='path store checkpoint and visulizations')
    parser.add_argument('--debug_vis', action='store_true', help='store debug visualizations to the checkpoint directory')


    # Image pre-processor
    parser.add_argument('--patch-sz', default=6, type=int, help='image patch size')
    parser.add_argument('--context-sz', default=32, type=int, help='other patches within this number of pixels is considered a neighbor')
    parser.add_argument('--grayscale_only', action='store_true', help='convert all input images to grayscale')
    parser.add_argument('--whiten_tol', default=1e-3, type=float, help='scaling of identity added before whitening')

    # Dictionary
    parser.add_argument('--dict-sz', default=8192, type=int, help='sparse coding dictionary size. should be overcomplete, larger than input data dimension by around 10x')

    # Sparse-coding
    parser.add_argument('--gq_thresh', default=0.3, type=float, help='general sparse coding cosine similarity threshold')
    parser.add_argument('--dict_thresh', default=0.7, type=float, help='sparse coding dictionary element similarity threshold.')

    # SMT Embedding
    parser.add_argument('--embed-dim', default=384, type=int, help='feature manifold dimension (patch embedding dimension, image-level embedding will be much larger)')
    parser.add_argument('--disable_color_embed_drop', action='store_true', help='do not drop the first 16 embedding dim')
    parser.add_argument('--skip_first_n', default=16, type=int, help='feature manifold dimension (patch embedding dimension, image-level embedding will be much larger)')

    # Classifier
    parser.add_argument('--nnclass-k', default=30, type=int, help='value of k for k-NN classifier')
    parser.add_argument('--knn_temp', default=0.03, type=float, help='temperatur for soft k-NN classifier')
    parser.add_argument('--classify_chunk', default=500, type=int, help='chunk size used when classifying test-set images')
    parser.add_argument('--full_dset_eval', action='store_true', help='run classifier with full dataset, not just the train set')

    return parser

def load_ckpt(path):
    print("Loading checkpoint...", flush=True)
    if path[-1] != "/":
        path += "/"
    
    with open(path + "args.json", "r") as file:
        ckpt_args = json.load(file)
    parser = generate_argparser()
    for action in parser._actions:
        if action.required:
            action.required = False
    args = parser.parse_args(args=[], namespace=argparse.Namespace(**ckpt_args))

    basis = torch.load(path + "sc_basis.pt")
    sc_layer = SparseCodeLayer(basis.shape[1], basis, args.gq_thresh)
    smt_layer = ManifoldEmbedLayer(args, None, None, torch.from_numpy(np.load(path + "smt_proj.npy")))
    
    whiten_op = torch.load(path + "whiten_op.pt")
    unwhiten_op = torch.load(path + "unwhiten_op.pt")
    
    return args, sc_layer, smt_layer, whiten_op, unwhiten_op

def get_ckpt_path(ckpt_path):
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
        path = ckpt_path
        if path[-1] != "/":
            path += "/"
    else:
        # if path exists, generate a subdirectory
        path = ckpt_path
        if path[-1] != "/":
            path += "/"
        path += f"ckpt_{int(time())}/"
        os.mkdir(path)
    return path

def save_ckpt(path, args, sc_layer, smt_layer, dset):
    # generate directory for checkpoint
    assert os.path.exists(path)
    print("Saving checkpoint...", flush=True)

    # save dictionary, embedding matrix, + a copy of the arguments
    torch.save(sc_layer.basis, path + "sc_basis.pt")
    np.save(path + "smt_proj.npy", smt_layer.projection)
    np.save(path + "smt_proj_full.npy", smt_layer.projection_full)
    with open(path + "args.json", "w") as file:
        json.dump(vars(args), file, indent=4)

    # save whiten + unwhiten operators from the training set
    torch.save(dset.whiten_op, path + "whiten_op.pt")
    torch.save(dset.unwhiten_op, path + "unwhiten_op.pt")

