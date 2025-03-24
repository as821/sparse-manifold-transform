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


def main(a):
    with torch.no_grad():
        args, sc_layer, smt_layer, whiten_op, unwhiten_op = load_ckpt(a.path, dense=True)
        if a.baseline:
            sc_layer = None
            smt_layer = None

        # uses entire train/test set for evaluation (even if SMT was trained with less than that)
        train_set = generate_dset(args, whiten_op=whiten_op, unwhiten_op=unwhiten_op)
        test_set = generate_dset(args, 'test', train_set)

    _, acc = train_classifier_model(train_set, test_set, sc_layer, smt_layer, args, a)
    print(f"\nTop accuracy: {acc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/home/astange/smt_ckpt")
    parser.add_argument('--baseline', action="store_true")
    parser.add_argument('--wandb', action="store_true")

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--final_lr", type=float, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--n_probe_head", type=int, default=50)
    parser.add_argument('--bn', action="store_true")

    main(parser.parse_args())

