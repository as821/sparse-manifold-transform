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


def main(path, baseline):
    with torch.no_grad():
        args, sc_layer, smt_layer, whiten_op, unwhiten_op = load_ckpt(path, dense=True)

        # uses entire train/test set for evaluation (even if SMT was trained with less than that)
        train_set = generate_dset(args, whiten_op=whiten_op, unwhiten_op=unwhiten_op)
        test_set = generate_dset(args, 'test', train_set)

        if baseline:
            sc_layer = None
            smt_layer = None

    # train probe with a stride=1 CNN layer, no SMT
    _, acc = train_classifier_model(train_set, test_set, sc_layer, smt_layer, args)
    print(f"\nTop accuracy: {acc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/home/astange/smt_ckpt")
    parser.add_argument('--baseline', action="store_true")
    
    args = parser.parse_args()
    main(args.path, args.baseline)

