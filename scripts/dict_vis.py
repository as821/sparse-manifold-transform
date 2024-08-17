"""
Dictionary visualizations
"""

import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.append(os.path.join(os.getcwd(), 'src/c'))

from util import generate_dset_dict_codes, generate_argparser, validate_args

import numpy as np
import torch
import matplotlib.pyplot as plt
from einops import rearrange
import shutil
from tqdm import tqdm
import scipy.sparse as sp



def get_vis_path(args, name):
    dir = args.vis_dir + f"/d{args.dict_sz[0]}_dt{args.dict_thresh[0]}_g{args.gq_thresh[0]}_s{args.samples}_c{args.context_sz[0]}_wt{args.whiten_tol}_dw{int(args.disable_whiten)}"
    if not os.path.exists(dir):
        os.mkdir(dir)
    return dir + "/" + name

def dict_cluster_sz(args, alphas):
    """Visualize distribution of dictionary element activations"""
    print("Visualizing dictionary cluster sizes...")
    dict_cnt = alphas.sum(axis=1)
    plt.figure(figsize=(10, 6))
    plt.xlabel('Dictionary element index (Sorted)')
    plt.ylabel('# of Members in Element Cluster')
    plt.title('Sorted Distribution of Dictionary Element Cluster Size')
    plt.plot(np.arange(dict_cnt.shape[0]), np.sort(dict_cnt), linewidth=2)
    
    plt.annotate(f'Min: {int(np.min(dict_cnt))} ({100 * np.min(dict_cnt) / alphas.shape[1]:.3f}%)', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10, ha='left', va='top')
    plt.annotate(f'Max: {int(np.max(dict_cnt))} ({100 * np.max(dict_cnt) / alphas.shape[1]:.3f}%)', xy=(0.05, 0.90), xycoords='axes fraction', fontsize=10, ha='left', va='top')
    plt.annotate(f'Avg: {np.average(dict_cnt):.2f} ({100 * np.average(dict_cnt) / alphas.shape[1]:.3f}%)', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=10, ha='left', va='top')
    plt.annotate(f'Median: {np.median(dict_cnt):.2f} ({100 * np.median(dict_cnt) / alphas.shape[1]:.3f}%)', xy=(0.05, 0.8), xycoords='axes fraction', fontsize=10, ha='left', va='top')
    
    if args.vis_dir != '':
        plt.savefig(get_vis_path(args, "dict_cluster"))
        plt.close()
    else:
        plt.show()

def dict_extrema_vis(args, dset, alphas, sc_layer):
    """Visualize the dictionary elements with the most and least activations."""
    print("Visualizing dictionary elements with most/least members...")
    
    dict_cnt = alphas.sum(axis=1)

    k = 7
    ncol = 2
    fig, axes = plt.subplots(2*k + 1, ncol, figsize=(8, 10))  # 11 rows to account for the extra space
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.axis('off')

    fig.text(0.5, 0.975, 'Dict. Elements with Most Activations', ha='center', va='center', fontsize=12)

    # visualize 5 dictionary patches with the most activations
    arr = np.argsort(dict_cnt)[-k:][::-1].copy()
    for idx in range(k):
        p = sc_layer.basis[:, arr[idx]]

        # vis
        p = dset.preproc_to_orig(sc_layer.idx_list[arr[idx]], p)
        patch = rearrange(p.squeeze(), "(a b c) -> a b c", a=dset.n_inp_channels, b=args.patch_sz, c=args.patch_sz)
        patch = patch.permute((1, 2, 0))
        axes.flat[idx * ncol].imshow(patch)

        # original patch
        patch = rearrange(dset.patch_cpy, "a b c d e -> (a b c) (d e)")[sc_layer.idx_list[arr[idx]], :].squeeze()
        patch = rearrange(patch, "(a b c) -> a b c", a=dset.n_inp_channels, b=args.patch_sz, c=args.patch_sz)
        patch = patch.permute((1, 2, 0))
        axes.flat[idx * ncol + 1].imshow(patch)


    fig.text(0.5, 0.5, 'Dict. Elements with Least Activations', ha='center', va='center', fontsize=12)


    # visualize k dictionary patches with the least activations
    arr = np.argsort(dict_cnt)[:k].copy()
    k_smallest_ind = torch.from_numpy(arr)
    k_smallest = sc_layer.basis[:, k_smallest_ind]
    off = (k+1) * ncol      # largest + 1 empty row
    for idx in range(k):
        p = k_smallest[:, idx]
        
        # vis
        p = dset.preproc_to_orig(sc_layer.idx_list[arr[idx]], p)
        patch = rearrange(p.squeeze(), "(a b c) -> a b c", a=dset.n_inp_channels, b=args.patch_sz, c=args.patch_sz)
        patch = patch.permute((1, 2, 0))
        axes.flat[off + idx * ncol].imshow(patch)

        # original patch
        patch = rearrange(dset.patch_cpy, "a b c d e -> (a b c) (d e)")[sc_layer.idx_list[arr[idx]], :].squeeze()
        patch = rearrange(patch, "(a b c) -> a b c", a=dset.n_inp_channels, b=args.patch_sz, c=args.patch_sz)
        patch = patch.permute((1, 2, 0))
        axes.flat[off + idx * ncol + 1].imshow(patch)

    for ax in axes[k]:
        ax.remove()
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout, but leave space at the top

    if args.vis_dir != '':
        plt.savefig(get_vis_path(args, "dict_extrema"))
        plt.close()
    else:
        plt.show()

def patch_rep_sz(args, alphas):
    """Visualize distribution of the number of dictionary elements used to represent each patch."""
    print("Visualizing patch representation sizes...")
    patch_sz = alphas.sum(axis=0)
    plt.figure(figsize=(10, 6))
    plt.xlabel('Patch index (Sorted)')
    plt.ylabel('Representation Size (# of Dict. Element)')
    plt.title('Sorted Distribution of Patch Dictionary Element Activations')
    plt.plot(np.arange(patch_sz.shape[0]), np.sort(patch_sz), linewidth=2)
    
    plt.annotate(f'Min: {int(np.min(patch_sz))} ({100 * np.min(patch_sz) / alphas.shape[0]:.3f}%)', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10, ha='left', va='top')
    plt.annotate(f'Max: {int(np.max(patch_sz))} ({100 * np.max(patch_sz) / alphas.shape[0]:.3f}%)', xy=(0.05, 0.90), xycoords='axes fraction', fontsize=10, ha='left', va='top')
    plt.annotate(f'Avg: {np.average(patch_sz):.2f} ({100 * np.average(patch_sz) / alphas.shape[0]:.3f}%)', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=10, ha='left', va='top')
    plt.annotate(f'Median: {np.median(patch_sz):.2f} ({100 * np.median(patch_sz) / alphas.shape[0]:.3f}%)', xy=(0.05, 0.8), xycoords='axes fraction', fontsize=10, ha='left', va='top')
    
    if args.vis_dir != '':
        plt.savefig(get_vis_path(args, "patch_rep_sz"))
        plt.close()
    else:
        plt.show()

def vis_patch_recon(args, dset, alphas, sc_layer):
    """Visualize the reconstruction of patches from their dictionary elemements for the """
    print("Visualizing patch reconstructions...")


    k = 40
    ncol = 4
    fig, axes = plt.subplots(int(k / 2), ncol + 1, figsize=(10, 12))
    fig.subplots_adjust(hspace=0.5, wspace=0.05)
    for i, ax in enumerate(axes.flat):
        ax.axis('off')

    patches = rearrange(dset.patch_cpy, "a b c d e -> (a b c) (d e)")
    
    patch_sz = alphas.sum(axis=0)
    arr = np.argsort(patch_sz)

    # don't visualize zero codes
    idx = 0
    while patch_sz[arr[idx]] == 0:
        idx += 1
    arr = arr[idx:]

    # visualize K patches across spectrum of # of activated dictionary elements (including most + least activated)
    running = 0
    for i in range(k):
        # subplot + text formatting
        idx = int((i / k) * arr.shape[0])
        base = i * 2 + running
        if (base % (ncol + 1)) == 2:
            running += 1
            base += 1       # skip middle column
            
            # text on the right side
            fig.text(1-0.075, 0.925 - 0.045*int(i / 2), f'({idx}) {arr[idx]} : {int(patch_sz[arr[idx]])}', ha='center', va='center', fontsize=8)
        else:
            # text on left side
            fig.text(0.075, 0.925 - 0.045*int(i / 2), f'({idx}) {arr[idx]} : {int(patch_sz[arr[idx]])}', ha='center', va='center', fontsize=8)

        # recon
        code = alphas[0:alphas.shape[0], arr[idx]:arr[idx]+1].todense()
        recon = sc_layer.basis @ code.squeeze()
        recon /= code.sum()      # average of the dictionary elements

        recon = dset.preproc_to_orig(arr[idx], recon.type(torch.float))
        patch = rearrange(recon, "(a b c) -> a b c", a=dset.n_inp_channels, b=args.patch_sz, c=args.patch_sz)
        patch = patch.permute((1, 2, 0))
        axes.flat[base].imshow(patch)

        # original patch
        patch = patches[arr[idx], :].squeeze()
        patch = rearrange(patch, "(a b c) -> a b c", a=dset.n_inp_channels, b=args.patch_sz, c=args.patch_sz)
        patch = patch.permute((1, 2, 0))
        axes.flat[base + 1].imshow(patch)

    plt.tight_layout(rect=[0.1, 0, 0.9, 0.95])  # left, bottom, right, top
    
    if args.vis_dir != '':
        plt.savefig(get_vis_path(args, "patch_recon"))
        plt.close()
    else:
        plt.show()

def patch_recon_err(args, dset, alphas, sc_layer):
    """Visualize patch reconstruction error as a function of the number of dictionary elements"""
    print("Visualizing patch reconstruction error...")

    patches = rearrange(dset.patch_cpy, "a b c d e -> (a b c) (d e)")
    patch_sz = alphas.sum(axis=0)
    arr = np.argsort(patch_sz)

    # don't visualize zero codes
    idx = 0
    while patch_sz[arr[idx]] == 0:
        idx += 1
    arr = arr[idx:]

    # visualize K patches across spectrum of # of activated dictionary elements (including most + least activated)
    l2_err, n_dict = [], []
    
    # only visualize 1/100
    ind = [i for i in range(arr.shape[0]) if i % 2000 == 0]
    if ind[-1] != arr[-1]:
        ind.append(arr.shape[0]-1)

    csr = sp.csr_array(sc_layer.basis.numpy(), dtype=np.float32, copy=False)
    recon_mx = csr @ alphas[0:alphas.shape[0], ind]
    recon_mx = torch.from_numpy(recon_mx.todense())
    
    for idx in tqdm(range(len(ind))):
        i = ind[idx]
        
        # recon
        code = alphas[0:alphas.shape[0], arr[i]:arr[i]+1].todense()
        recon = recon_mx[:, idx].squeeze()
        recon /= code.sum()      # average of the dictionary elements
        recon = dset.preproc_to_orig_no_rescale(arr[i], recon.type(torch.float))

        # original patch
        patch = patches[arr[i], :].squeeze()

        # L1 and L2 reconstruction error
        l2 = ((patch - recon)**2).sum()
        l2_err.append(l2)
        n_dict.append(code.sum())

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Patch index (Sorted by # Dict Activations.)')
    ax1.set_ylabel('Reconstruction Error / # Activations')
    ax1.plot(np.arange(len(ind)), l2_err, linewidth=2, color='red', label='L2')
    ax1.tick_params(axis='y', labelcolor='red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('# Activations')
    ax2.plot(np.arange(len(ind)), n_dict, linewidth=2, color='blue', label='# dict. activations')
    ax2.tick_params(axis='y', labelcolor='blue')

    fig.suptitle('Error as Function of # Dict. Activations')
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)

    if args.vis_dir != '':
        plt.savefig(get_vis_path(args, "patch_recon_err"))
        plt.close()
    else:
        plt.show()




def main(args):
    if args.vis_dir != "" and os.path.exists(get_vis_path(args, "")):
        shutil.rmtree(get_vis_path(args, ""))

    # Generate dataset and sparse codes for images
    dset, alphas, sc_layer, _ = generate_dset_dict_codes(args)

    # plot distribution of how many patches each dictionary element "activates" for
    dict_cluster_sz(args, alphas)

    # visualize dict elements with most and least activations
    dict_extrema_vis(args, dset, alphas, sc_layer)

    # plot distribution of how many dictionary elements each patch is represented by
    patch_rep_sz(args, alphas)

    # visualize a patches and their reconstruction from dictionary elements
    vis_patch_recon(args, dset, alphas, sc_layer)

    # measure reconstruction error + relation to number of dictionary elements (diminishing returns on number of elements?)
    patch_recon_err(args, dset, alphas, sc_layer)


    # TODO(as) visualize all dictionary elements for some patches

    # TODO(as) visualize the K most/least similar patches in the largest dictionary patch cluster (what does 0.3, 0.5, etc. actually look like?)

    # TODO(as) plot distribution of image patches themselves (which patches are most common, how long is the long tail, etc.)

    # TODO(as) visualize which dictionary elements are associated with most/least common patches

    print('done')




if __name__ == "__main__":
    parser = generate_argparser()
    args = parser.parse_args()
    args.embed_dim = [-1]     # unused but required by arg validation
    args = validate_args(args)
    with torch.no_grad():
        main(args)




