import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import wandb
import argparse
import einops
import matplotlib.pyplot as plt
from tqdm import tqdm

import pdb

import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))
from util import get_ckpt_path
from sparse_ae import SparseAutoEncoder

def feature_density_plot(args, loss_dict, vis_dict, prefix="test_"):
    assert "feature_density" in loss_dict
    log10 = loss_dict["feature_density"].cpu()
    vis_dict[prefix + "dead_neuron"] = log10[log10 == 0].shape[0]
    log10 = log10[log10 > 0].log10().numpy()        # only keep non-zero features
    
    plt.figure(figsize=(10, 6))
    plt.hist(log10, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title("Feature density")
    plt.xlabel("log_10 density")
    plt.ylabel("Counts")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    vis_dict[prefix + "feature_density"] = wandb.Image(plt)
    plt.close()

    return vis_dict


def visualize_dictionary(args, vis_dict, model, n_vis=100):
    foo = model.dec.weight.data.cpu().T  # shape: (a, b * c * d)
    foo = einops.rearrange(foo, "a (b c d) -> a b c d", b=3, c=args.patch_sz)

    # normalize using max/min dictionary values
    img_min = foo.amin(dim=(0, 2, 3), keepdim=True)
    img_max = foo.amax(dim=(0, 2, 3), keepdim=True)
    normalized = (foo - img_min) / (img_max - img_min + 1e-8)

    n_vis = min(n_vis, normalized.shape[0])

    upscale_factor = 20
    resized = torch.nn.functional.interpolate(normalized[:n_vis], scale_factor=upscale_factor, mode='nearest')

    fig, axes = plt.subplots(1, n_vis, figsize=(n_vis * 2, 3))
    if n_vis == 1:
        axes = [axes]
    for i in range(n_vis):
        img = resized[i].permute(1, 2, 0).numpy()  # CHW -> HWC
        axes[i].imshow(img)
        axes[i].axis('off')

    plt.tight_layout()
    vis_dict["dict"] = wandb.Image(plt)
    plt.close()

    return vis_dict


def visualize_recon(args, vis_dict, model, loader, n_vis=100, prefix=""):
    # generate from a single batch
    x = next(iter(loader)).cuda(non_blocking=True).flatten(0, 1)
    x = x[torch.randperm(x.shape[0])]       # randomize patches so not all are from a single image
    x_recon, _ = model(x)
    
    x = einops.rearrange(x, "a (b c d) -> a b c d", b=3, c=args.patch_sz).cpu().detach()
    x_recon = einops.rearrange(x_recon, "a (b c d) -> a b c d", b=3, c=args.patch_sz).cpu().detach()
    n_vis = min(n_vis, x.shape[0])

    # 0-1 normalize reconstrution (original should already be in range)
    img_min = x_recon.amin(dim=(0, 2, 3), keepdim=True)
    img_max = x_recon.amax(dim=(0, 2, 3), keepdim=True)
    x_recon = (x_recon - img_min) / (img_max - img_min + 1e-8)


    upscale_factor = 20
    resized_x = torch.nn.functional.interpolate(x[:n_vis], scale_factor=upscale_factor, mode='nearest')
    resized_x_recon = torch.nn.functional.interpolate(x_recon[:n_vis], scale_factor=upscale_factor, mode='nearest')

    fig, axes = plt.subplots(2, n_vis, figsize=(n_vis * 2, 3))
    for i in range(n_vis):
        axes[0, i].imshow(resized_x[i].permute(1, 2, 0).numpy())
        axes[0, i].axis('off')
        axes[1, i].imshow(resized_x_recon[i].permute(1, 2, 0).numpy())
        axes[1, i].axis('off')

    fig.text(0.01, 0.75, 'Original', va='center', ha='left', fontsize=12)
    fig.text(0.01, 0.25, 'Reconstructed', va='center', ha='left', fontsize=12)
    plt.tight_layout(rect=[0.03, 0, 1, 1])  # leave space for labels on the left
    
    vis_dict[prefix + "recon"] = wandb.Image(plt)
    plt.close()

    return vis_dict


class PatchDataset(Dataset):
    def __init__(self, dset, patch_sz, stride=1):
        self.dset = dset
        self.patch_sz = patch_sz
        self.stride = stride
        
        # CIFAR10 mean/std
        # self.mean = [0.4914, 0.4822, 0.4465]
        # self.std = [0.2023, 0.1994, 0.2010]
        # self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        # Get all patches for this image (and strip off labels)
        img, label = self.dset[idx]
        # img = self.normalize(img)       # normalize image with dataset-level stats prior to patchifying
        out = torch.nn.functional.unfold(img, self.patch_sz, stride=self.stride).clone().T
        # out -= torch.mean(out, dim=1, keepdim=True)
        return out

def run_epoch(args, model, loader, optimizer, epoch):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_recon, total_l1, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(loader)

    # feature density is the fraction of inputs that a dictionary element activates for
    dict_feature_density = torch.zeros((args.dict_sz,), device="cuda")

    with torch.enable_grad() if is_train else torch.no_grad():
        for data in data_bar:
            data = data.cuda(non_blocking=True)
            data = data.flatten(0, 1)
            recon, acts = model(data)
            loss, recon_loss, l1_loss = SparseAutoEncoder.loss(data, recon, acts, args.l1)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                model.unit_norm_decoder_weights_and_grad()
                optimizer.step()

            with torch.no_grad():
                total_num += data.size(0)
                total_loss += loss.item() * data.size(0)
                total_recon += recon_loss.item() * data.size(0)
                total_l1 += l1_loss.item() * data.size(0)

                # TODO: add a bunch of tracking for dead neurons + representation sparsity
                dict_feature_density += (acts > 0).sum(dim=0)

                # TODO: reinitialization of dead neurons

            data_bar.set_description(f"{'Train' if is_train else 'Test'} Epoch: [{epoch}] Loss: {total_loss / total_num :.4f}")

    loss_dict = {
        "loss" : total_loss / total_num,
        "recon_loss" : total_recon / total_num,
        "l1_loss" : total_l1 / total_num,
    }
    loss_dict["feature_density"] = dict_feature_density / total_num

    return loss_dict

def train(args):
    args.ckpt_path = get_ckpt_path(args.ckpt_path)

    train_set = PatchDataset(torchvision.datasets.CIFAR10(root=args.dset_path, train=True, transform=transforms.Compose([transforms.ToTensor()]), download=True), args.patch_sz, args.stride)
    test_set = PatchDataset(torchvision.datasets.CIFAR10(root=args.dset_path, train=False, transform=transforms.Compose([transforms.ToTensor()]), download=True), args.patch_sz, args.stride)
    train_loader = DataLoader(train_set, batch_size=args.batch_sz, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_sz, shuffle=False, num_workers=8, pin_memory=True)

    model = SparseAutoEncoder(3 * args.patch_sz * args.patch_sz, args.dict_sz)
    model = model.to("cuda", non_blocking=True)
    model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, fused=True)

    if args.wandb:
        wandb.init(config=args, project="smt_sae")
        wandb.watch(model, log_freq=5)

    best_loss = None
    for epoch in range(args.epochs):
        train_loss_dict = run_epoch(args, model, train_loader, optimizer, epoch)
        test_loss_dict = run_epoch(args, model, test_loader, None, epoch)

        if args.wandb:
            vis_dict = {}
            vis_dict["train_loss"] = train_loss_dict["loss"]
            vis_dict["train_recon"] = train_loss_dict["recon_loss"]
            vis_dict["train_l1"] = train_loss_dict["l1_loss"]

            vis_dict["test_loss"] = test_loss_dict["loss"]
            vis_dict["test_recon"] = test_loss_dict["recon_loss"]
            vis_dict["test_l1"] = test_loss_dict["l1_loss"]

            with torch.no_grad():
                model = model.eval()
                vis_dict = feature_density_plot(args, test_loss_dict, vis_dict)
                vis_dict = visualize_dictionary(args, vis_dict, model)

                vis_dict = visualize_recon(args, vis_dict, model, train_loader, prefix="train_")
                vis_dict = visualize_recon(args, vis_dict, model, test_loader, prefix="test_")
                model = model.train()


            wandb.log(vis_dict, step=epoch)

        if best_loss is None or test_loss_dict["loss"] < best_loss:
            best_loss = test_loss_dict["loss"]

        if best_loss == test_loss_dict["loss"]:
            torch.save(model.state_dict(), args.ckpt_path + "sae_best.pt")

    torch.save(model.state_dict(), args.ckpt_path + "sae_final.pt")
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dset_path', type=str, default="/home/astange/slots/data/cifar")
    parser.add_argument('--ckpt_path', type=str, default="/home/astange/sae_ckpt")
    
    parser.add_argument('--dict-sz', default=8192, type=int)
    parser.add_argument('--patch-sz', default=6, type=int, help='image patch size')
    parser.add_argument('--stride', type=int, default=1)

    parser.add_argument('--l1', type=float, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_sz', type=int, default=250)

    parser.add_argument('--wandb', action='store_true')


    train(parser.parse_args())

