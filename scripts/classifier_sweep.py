import argparse 
import einops
import numpy as np
import torch
import math
from tqdm import tqdm

import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.append(os.path.join(os.getcwd(), 'src/c'))

from util import load_ckpt
from preprocessor import generate_dset, ImagePreprocessor
from manifold_embedding import ManifoldEmbedLayer
from classifier import eval_knn_classifier


import pdb

def main(a):
    # Sweep through classifier settings on a given checkpoint. Report best performing combinations
    args, sc_layer, _, whiten_op, unwhiten_op, U_full, inv_sqrt_cov = load_ckpt(a.path)
    dset = generate_dset(args, whiten_op=whiten_op, unwhiten_op=unwhiten_op)
    args.debug_vis = False    
    if a.n_train <= 0:
        args.full_dset_eval = True
    else:
        args.full_dset_eval = False
        args.samples = a.n_train

    k_val = a.classifier_k
    k_temp = a.classifier_temp
    if k_val is None:
        k_val = [args.nnclass_k]
    if k_temp is None:
        k_temp = [args.knn_temp]

    # sweep through sizes and offsets for projection matrices
    results = {}
    for offset in a.proj_offsets:
        for proj_dim in a.proj_dim:
            proj = U_full[offset : offset + proj_dim, :] @ inv_sqrt_cov
            smt_layer = ManifoldEmbedLayer(args, None, None, proj)
            for k in k_val:
                for temp in k_temp:
                    # TODO: actually don't need to regenerate embeddings for each classifier setting
                    args.nnclass_k = k
                    args.knn_temp = temp
                    results[(offset, proj_dim, k, temp)] = eval_knn_classifier(args, dset, sc_layer, smt_layer)
                    print(f"Test set accuracy ({offset}, {proj_dim}): {results[(offset, proj_dim, k, temp)]}\n\n")

    # print results sorted from best to worst top-1 performance
    print("\n\n\nSORTED RESULTS:")
    for idx, k in enumerate(sorted(results.keys(), key=lambda k: -1 * results[k][0])):
        print(f"{idx}: {k} -> {results[k]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/home/astange/smt_ckpt")
    parser.add_argument('--proj_offsets', nargs='+', type=int, help="offsets into the projection matrix")
    parser.add_argument('--proj_dim', nargs='+', type=int, help="projection matrix dimensions to use")
    parser.add_argument('--n_train', default=-1, type=int)
    parser.add_argument('--classifier_k', nargs='+', type=int, help="value of k for k-NN")
    parser.add_argument('--classifier_temp', nargs='+', type=float, help="exponential temp for k-NN (lower is sharper dist, higher is uniform)")
    
    with torch.no_grad():
        main(parser.parse_args())

