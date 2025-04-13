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
    args.full_dset_eval = True

    # sweep through sizes and offsets for projection matrices
    results = {}
    for offset in a.proj_offsets:
        for proj_dim in a.proj_dim:
            proj = U_full[offset : offset + proj_dim, :] @ inv_sqrt_cov
            smt_layer = ManifoldEmbedLayer(args, None, None, proj)
            results[(offset, proj_dim)] = eval_knn_classifier(args, dset, sc_layer, smt_layer)
            print(f"Test set accuracy ({offset}, {proj_dim}): {results[(offset, proj_dim)]}\n\n")

    # print results sorted from best to worst top-1 performance
    print("\n\n\nSORTED RESULTS:")
    for idx, k in enumerate(sorted(results.keys(), key=lambda k: -1 * results[k][0])):
        print(f"{idx}: {k} -> {results[k]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/home/astange/smt_ckpt")
    parser.add_argument('--proj_offsets', nargs='+', type=int, help="offsets into the projection matrix")
    parser.add_argument('--proj_dim', nargs='+', type=int, help="projection matrix dimensions to use")
    
    with torch.no_grad():
        main(parser.parse_args())

