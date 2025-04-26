import torch
import torch.nn.functional as F

import pdb

class SparseAutoEncoder(torch.nn.Module):
    def __init__(self, input_sz, hidden_sz, dtype=torch.float32):
        super().__init__()

        self.b_dec = torch.nn.Parameter(torch.zeros(input_sz, dtype=dtype))

        self.enc = torch.nn.Sequential(
            torch.nn.Linear(input_sz, hidden_sz, bias=False),
            torch.nn.ReLU(),
            # torch.nn.Linear(hidden_sz, hidden_sz, bias=True),
            # torch.nn.ReLU(),
            # torch.nn.Linear(hidden_sz, hidden_sz, bias=True),
            # torch.nn.ReLU(),
        )

        self.dec = torch.nn.Linear(hidden_sz, input_sz, bias=False)
        with torch.no_grad():
            self.dec.weight.data = F.normalize(self.dec.weight.data, dim=0)



    def forward(self, x):
        x -= self.b_dec
        acts = self.enc(x)
        recon = self.dec(acts) + self.b_dec
        return recon, acts
    
    def loss(x, x_recon, acts, l1_coeff):
        l2_loss = (x_recon.float() - x.float()).pow(2).sum(-1).mean(0)
        l1_loss = l1_coeff * acts.float().abs().sum(1).mean()
        loss = l2_loss + l1_loss
        return loss, l2_loss.detach(), l1_loss.detach()

    @torch.no_grad()
    def unit_norm_decoder_weights_and_grad(self):
        # remove the portion of the gradient that is parallel to the dictionary elements
        dec_normed = self.dec.weight.data / self.dec.weight.data.norm(dim=-1, keepdim=True)
        dec_grad_proj = (self.dec.weight.grad * dec_normed).sum(-1, keepdim=True) * dec_normed     # component of the gradient in the direction of W_dec_normed (first mult + sum are a batched dot prod)
        self.dec.weight.grad -= dec_grad_proj
        
        # maintain decoder unit norm
        self.dec.weight.data = dec_normed

