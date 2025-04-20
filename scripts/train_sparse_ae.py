import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import wandb
import argparse
from tqdm import tqdm

import pdb

import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))
from util import get_ckpt_path
from sparse_ae import SparseAutoEncoder

class PatchDataset(Dataset):
    def __init__(self, dset, patch_sz, stride=1):
        self.dset = dset
        self.patch_sz = patch_sz
        self.stride = stride

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        # Get all patches for this image (and strip off labels)
        img, label = self.dset[idx]
        return torch.nn.functional.unfold(img, self.patch_sz, stride=self.stride).clone().T


def run_epoch(args, model, loader, optimizer, epoch):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_num, data_bar = 0.0, 0, tqdm(loader)

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

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)

            # TODO: add a bunch of tracking for dead neurons + representation sparsity

            # TODO: reinitialization of dead neurons

            data_bar.set_description(f"{'Train' if is_train else 'Test'} Epoch: [{epoch}] Loss: {total_loss / total_num :.4f}")

    return total_loss / total_num


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
        train_loss = run_epoch(args, model, train_loader, optimizer, epoch)
        test_loss = run_epoch(args, model, test_loader, None, epoch)

        if args.wandb:
            vis_dict = {}
            vis_dict["train_loss"] = train_loss
            vis_dict["test_loss"] = test_loss
            wandb.log(vis_dict, step=epoch)

        if best_loss is None or test_loss < best_loss:
            best_loss = test_loss

        if best_loss == test_loss:
            torch.save(model.state_dict(), args.ckpt_path + "sae_best.pt")

    torch.save(model.state_dict(), args.ckpt_path + "sae_final.pt")

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

