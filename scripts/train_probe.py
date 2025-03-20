import argparse 
import einops
import numpy as np
import torch

import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.append(os.path.join(os.getcwd(), 'src/c'))

from attentive_probe import train_classifier_model
from util import load_ckpt, generate_dset

import pdb


def run_baseline(path):
    args, sc_layer, smt_layer, whiten_op, unwhiten_op = load_ckpt(path)

    # use entire train/test set for evaluation (even if SMT was trained with less than that)
    args.samples = 1000
    args.test_samples = 1000
    
    with torch.no_grad():
        train_set = generate_dset(args, whiten_op=whiten_op, unwhiten_op=unwhiten_op)
        test_set = generate_dset(args, 'test', train_set)

    # train probe with a stride=1 CNN layer, no SMT
    _, acc = train_classifier_model(train_set, test_set, baseline=True)
    print(f"\nTop accuracy: {acc}")

def main(path):
    args, sc_layer, smt_layer, whiten_op, unwhiten_op = load_ckpt(path)

    # use entire train/test set for evaluation (even if SMT was trained with less than that)
    args.samples = 1000
    args.test_samples = 1000
    
    with torch.no_grad():
        # generate train set embeddings
        train_set = generate_dset(args)
        x, train_labels = train_set.generate_data(args.samples, test=True)
        train_embed = smt_layer(sc_layer(args, x))
        train_embed = einops.rearrange(train_embed, "d (a b c) -> a (b c) d", a=args.samples, b=train_set.n_patch_per_dim, c=train_set.n_patch_per_dim, d=args.embed_dim).astype(np.float32)
        train_labels = train_labels[:, 0].to(torch.long)

        pdb.set_trace()

        # generate test set embeddings
        test_set = generate_dset(args, 'test', train_set)
        test_x, test_labels = test_set.generate_data(args.test_samples, test=True)
        test_embed = smt_layer(sc_layer(args, test_x, test=True))
        test_embed = einops.rearrange(test_embed, "d (a b c) -> a (b c) d", a=args.test_samples, b=test_set.n_patch_per_dim, c=test_set.n_patch_per_dim, d=args.embed_dim).astype(np.float32)
        test_labels = test_labels[:, 0].to(torch.long)

    # train an attentive probe that takes these SMT embeddings as input
    _, acc = train_classifier_model(train_embed, test_embed, train_labels, test_labels)
    print(f"\nTop accuracy: {acc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/home/astange/smt_ckpt")
    parser.add_argument('--baseline', action="store_true")
    
    args = parser.parse_args()
    if args.baseline:
        run_baseline(args.path)
    else:
        main(args.path)

