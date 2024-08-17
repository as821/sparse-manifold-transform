import numpy as np
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sys
from random import randint


def mask_vis(vis_dir, vis_dict, masks, image, gt_masks, num_cluster):
    fig, axs = plt.subplots(1, num_cluster+2, figsize=(10, 10))
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    for j in range(num_cluster+2):
        axs[j].axis('off')
    img = image.permute(1, 2, 0).numpy()
    
    if gt_masks is not None:
        tot_mask = 0.0
        colors = cm.rainbow(np.linspace(0, 1, len(gt_masks)))
        for i, mask_i in enumerate(gt_masks):
            tot_mask += mask_i[:, :, None] * colors[i][:3]
        axs[0].imshow(tot_mask, alpha=0.5)
    else:
        colors = cm.rainbow(np.linspace(0, 1, len(masks)))      # TODO(as) does this work?
    
    axs[0].imshow(img)
    colors = cm.rainbow(np.linspace(0, 1, num_cluster))
    tot_mask = 0.0

    for i in range(num_cluster):
        mask_i = masks[i][:, :, None]
        axs[i+2].imshow(img)
        c = colors[i][:3]
        axs[i+2].imshow(mask_i * c, alpha=0.5)
        tot_mask += mask_i * c
    
    axs[1].imshow(img)
    axs[1].imshow(tot_mask, alpha=0.5)

    # plt.show()
    plt.savefig(vis_dir + "/" + str(randint(-sys.maxsize, sys.maxsize)))
    # vis_dict['slot_output'] = wandb.Image(fig)
    plt.close()
    return vis_dict

def cluster_patches(args, dset, betas):
    assert betas.shape[1] % dset.n_patch_per_img == 0

    # Visualize + store clustering of patches based on their embeddings: embeddings --> clusters --> masks
    n_clusters = args.n_obj
    n_img = 20
    assert args.samples >= n_img
    for idx in range(n_img):
        # convert patch-level embeddings into normalized pixel-level embeddings (averaging of embeddings of each patch that contains the pixel)
        emb = betas[:, idx*dset.n_patch_per_img : (idx+1)*dset.n_patch_per_img]
        per_pixel_emb = dset.de_patchify(emb)
        per_pixel_emb /= np.linalg.norm(per_pixel_emb, axis=0) + 1e-20
        
        # Run k-means on embeddings for each image 
        cluster_obj = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(per_pixel_emb.T)
        if args.depatchify != "center":
            labels_2d = cluster_obj.labels_.reshape(dset.img_sz, dset.img_sz)
        else:
            sz = int(np.sqrt(cluster_obj.labels_.shape[0]))
            assert sz ** 2 == cluster_obj.labels_.shape[0]
            labels_2d = cluster_obj.labels_.reshape(sz, sz)
        
        # Generate masks for each cluster
        img = dset.train_set_image(idx)[0]
        masks = []
        for i in range(n_clusters):
            mask_i = (labels_2d == i).astype(int)

            # if mask is smaller than image, center mask in image
            if args.depatchify == "center":
                zero_pad_mask = np.zeros(shape=(img.shape[1], img.shape[2]))
                x_diff = zero_pad_mask.shape[0] - mask_i.shape[0]
                y_diff = zero_pad_mask.shape[1] - mask_i.shape[1]
                assert x_diff % 2 == 0 and y_diff % 2 == 0
                x_diff /= 2
                y_diff /= 2
                x_diff = int(x_diff)
                y_diff = int(y_diff)
                zero_pad_mask[x_diff:-x_diff, y_diff:-y_diff] = mask_i
                mask_i = zero_pad_mask

            # plt.imshow(mask_i[:, :, None])
            masks.append(mask_i)

        mask_vis(args.vis_dir, {}, masks, img, None, n_clusters)


    return np.stack(masks)







