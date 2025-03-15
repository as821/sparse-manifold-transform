import torch
from torch import nn, optim
from torchvision.models.resnet import resnet50
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

import numpy as np
from tqdm import tqdm
from typing import OrderedDict
import einops
import pdb



# train or test linear classifier for one epoch
def train_val(net, data_loader, train_optimizer, epoch):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = (
        0.0,
        0.0,
        0.0,
        0,
        tqdm(data_loader),
    )
    with torch.enable_grad() if is_train else torch.no_grad():
        loss_criterion = nn.CrossEntropyLoss()
        for data, target in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = net(data)
            loss = loss_criterion(out, target)
            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum(
                (prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()
            ).item()
            total_correct_5 += torch.sum(
                (prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()
            ).item()

            data_bar.set_description(
                "{} Epoch: [{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%".format(
                    "Train" if is_train else "Test",
                    epoch,
                    total_loss / total_num,
                    total_correct_1 / total_num * 100,
                    total_correct_5 / total_num * 100,
                )
            )

    return (
        total_loss / total_num,
        total_correct_1 / total_num * 100,
        total_correct_5 / total_num * 100,
    )


def train_classifier_model(train_embed, test_embed, train_labels, test_labels, batch_size=512, epochs=50, lr=1e-2, save_path=None, save_name=None):
    top_acc = 0.0

    train_loader = DataLoader(CustomDataset(train_embed, train_labels), batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(CustomDataset(test_embed, test_labels), batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # only fully connected requires grad
    torch.set_float32_matmul_precision('high')
    model = AttentionPoolingClassifier(train_embed.shape[-1], 10)
    model = model.cuda()
    model = torch.compile(model)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6, fused=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if save_path is not None and save_name is None:
        save_name = str(np.random.rand() * 1e5)
    print(save_name)
    for epoch in range(1, epochs + 1):
        # train one epoch
        train_loss, train_acc_1, train_acc_5 = train_val(
            model, train_loader, optimizer, epoch
        )
        # test one epoch
        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None, epoch)
        scheduler.step()

        if test_acc_1 > top_acc:
            top_acc = test_acc_1

        if test_acc_1 == top_acc and save_path is not None and save_name is not None:
            save_str = save_path + save_name + ".pt"
            torch.save(model.state_dict(), save_str)

    return model, top_acc

# https://github.com/apple/ml-aim/blob/cb4171a25253dff87237f5fd5ee16fc633667d4f/aim-v1/aim/v1/torch/layers.py#L343
class AttentionPoolingClassifier(nn.Module):
    def __init__(
        self,
        dim: int,
        out_features: int,
        num_heads: int = 16,
        num_queries: int = 1,
        use_batch_norm: bool = True,
        qkv_bias: bool = False,
        linear_bias: bool = False,
        average_pool: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.average_pool = average_pool

        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.cls_token = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.linear = nn.Linear(dim, out_features, bias=linear_bias)
        # self.bn = (
        #     nn.BatchNorm1d(dim, affine=False, eps=1e-6)
        #     if use_batch_norm
        #     else nn.Identity()
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # x = self.bn(x.transpose(-2, -1)).transpose(-2, -1)
        cls_token = self.cls_token.expand(B, -1, -1)

        q = cls_token.reshape(B, self.num_queries, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        x_cls = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x_cls = x_cls.transpose(1, 2).reshape(B, self.num_queries, C)
        x_cls = x_cls.mean(dim=1) if self.average_pool else x_cls

        out = self.linear(x_cls)
        return out


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


