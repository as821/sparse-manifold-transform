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

from util import load_ckpt, get_ckpt_path
from preprocessor import generate_dset
from matrix_utils import visualize_matrix, visualize_histogram


import pdb


def main(a):
    args, sc_layer, smt_layer, whiten_op, unwhiten_op = load_ckpt(a.path)
    
    args.debug_vis = a.debug_vis
    
    dset = generate_dset(args, whiten_op=whiten_op, unwhiten_op=unwhiten_op)
    if a.test_set:
        dset = generate_dset(args, 'test', dset)

    if a.per_img:
        for idx in range(len(dset.dataset)):
            img, label = dset.get_single_eval_image(idx, a.stride)
            img = img.unsqueeze(0)
            sc = sc_layer(args, img, test=True, dense=True)
            smt = smt_layer(sc, dense=True)

            img = img[0]
            sc = sc[0]
            smt = smt[0]
            
            n_patch_per_dim = int(math.sqrt(sc.shape[1]))
            assert n_patch_per_dim ** 2 == sc.shape[1]
            print(f"Image: {idx} -> label {label}. patch per dim: {n_patch_per_dim}")
            print(f"\tImage patch vals: (mean: {img.mean():.4f}, min: {img.min():.4f}, max: {img.max():.4f})")

            # stats on number of dictionary members for the patch
            n_dict_per_patch = sc.sum(dim=0)
            print(f"\tDict el. per patch: (mean: {n_dict_per_patch.mean():.2f}, min: {n_dict_per_patch.min():.2f}, max: {n_dict_per_patch.max():.2f})")

            # stats on which dictionary elements are the most/least frequent
            n_patch_per_dict = sc.sum(dim=1)
            print(f"\tPatch per dict. el.: (mean: {n_patch_per_dict.mean():.2f}, min: {n_patch_per_dict.min():.2f}, max: {n_patch_per_dict.max():.2f})")

            # stats on reconstruction error from sparse coding
            recon = sc_layer.basis @ (sc / sc.sum(dim=0))       # normalized by number of dictionary elements that correspond to a patch to roughly preserve magnitude     
            recon_err = (recon - img).abs()
            print(f"\tRecon err per patch: (mean: {recon_err.mean():.4f}, min: {recon_err.min():.4f}, max: {recon_err.max():.4f})")
            
            # stats on embedding cosine similarity --> embeddings already normalized
            cos_sim = smt.T @ smt
            indices = torch.ones_like(cos_sim, dtype=torch.bool)
            indices.diagonal().fill_(False)     # remove diagonal elements
            cos_sim_no_diag = cos_sim[indices]
            print(f"\tEmbedding cos. sim: (mean: {cos_sim_no_diag.mean():.4f}, min: {cos_sim_no_diag.min():.4f}, max: {cos_sim_no_diag.max():.4f})")

            # stats on embeding cosine similarity with its neighborhood vs. the rest of the image
            cos_sim_patch = einops.rearrange(cos_sim, "a (b c) -> a b c", b=n_patch_per_dim)
            in_context_sim, out_context_sim = [], []
            for idx in range(cos_sim_patch.shape[0]):
                col_idx = idx % n_patch_per_dim
                col_start = max(0, col_idx - a.context_sz)
                col_end = min(n_patch_per_dim, col_idx + a.context_sz)
                
                row_idx = idx // n_patch_per_dim
                row_start = max(0, row_idx - a.context_sz)
                row_end = min(n_patch_per_dim, row_idx + a.context_sz)

                patch_sims = cos_sim_patch[idx]

                # always exclude diagonal elements
                mask = torch.zeros_like(patch_sims, dtype=torch.bool)
                mask[row_start:row_end, col_start:col_end] = True
                mask[row_idx, col_idx] = False
                in_context_sim.append(patch_sims[mask])
                
                mask[row_idx, col_idx] = True
                out_context_sim.append(patch_sims[~mask])
            in_context_sim = torch.cat(in_context_sim)
            out_context_sim = torch.cat(out_context_sim)
            
            print(f"\tEmbedding cos. sim (in-context): (mean: {in_context_sim.mean():.4f}, min: {in_context_sim.min():.4f}, max: {in_context_sim.max():.4f})")
            if out_context_sim.shape[0] > 0:
                print(f"\tEmbedding cos. sim (out-context): (mean: {out_context_sim.mean():.4f}, min: {out_context_sim.min():.4f}, max: {out_context_sim.max():.4f})")
                
            # number of patches with unique sparse codings
            uniq = torch.unique(sc, dim=1).shape[1]
            print(f"\tUnique sparse codings: {uniq} / {sc.shape[1]}")
            
            # print(f"\tEmbedding cos. sim: (mean: {cos_sim.min()}, min: {cos_sim.max()}, max: {cos_sim.max()})")



            print("\n\n\n")
            pdb.set_trace()
    else:
        sc_layer.basis = sc_layer.basis.to("cuda", non_blocking=True)
        smt_layer.projection = smt_layer.projection.to("cuda", non_blocking=True)

        n_patch_per_dict = torch.zeros(args.dict_sz)

        n_samples = 50
        embed = torch.zeros(n_samples, dset.n_patch_per_img, args.embed_dim)
        labels = torch.zeros(n_samples)
        for idx in tqdm(range(n_samples)):
            img, label = dset.get_single_eval_image(idx, a.stride)
            labels[idx] = label
            img = img.to("cuda", non_blocking=True)
            sc = sc_layer(img)
            embed[idx] = smt_layer(sc).T.unsqueeze(0).cpu()
            
            # stats on which dictionary elements are the most/least frequent
            n_patch_per_dict += sc.sum(dim=1).cpu()
        
        # stats on embedding cosine similarity --> embeddings already normalized
        embed = embed.flatten(0, 1)
        cos_sim = embed @ embed.T
        assert cos_sim.max() < (1 + 1e-3), f"Cosine similarity < 1. Bug somewhere: {cos_sim.max()}"
        
        visualize_matrix(args, cos_sim, "embed_cos_sim")
        visualize_matrix(args, torch.cov(embed.T), "embed_dim_cov")

        embed_hist = einops.rearrange(cos_sim, "(a b) (c d) -> a b c d", a=n_samples, c=n_samples)

        # cosine sim to other patches in the same image (double counts due to cosine similarity symmetry and includes self-similarity)
        intra_image_hist = embed_hist[torch.arange(n_samples), :, torch.arange(n_samples), :]
        visualize_histogram(args, intra_image_hist.flatten(), "intra_image_sim")

        # intra/inter class cosine similarity
        intra_class, inter_class = [], []
        for lab in torch.unique(labels):
            mask = labels == lab
            intra_class.append(embed_hist[mask][:, :, mask].flatten())
            inter_class.append(embed_hist[mask][:, :, ~mask].flatten())
        visualize_histogram(args, torch.concat(intra_class), "intra_class_sim")
        visualize_histogram(args, torch.concat(inter_class), "inter_class_sim")



        # TODO: visualize some nearest neighbors in patch embedding space (within image, within class, within dataset)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/home/astange/smt_ckpt")
    parser.add_argument('--test_set', action="store_true")
    parser.add_argument('--per_img', action="store_true")

    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--context_sz', type=int, default=32)
    
    # parser.add_argument('--ckpt_path', type=str, default="/home/astange/smt_vis")
    parser.add_argument('--debug_vis', action="store_true")

    with torch.no_grad():
        main(parser.parse_args())

