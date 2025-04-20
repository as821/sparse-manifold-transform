import torch
import torch.nn.functional as F

import pdb

class SparseAutoEncoder(torch.nn.Module):
    def __init__(self, input_sz, hidden_sz, dtype=torch.float32):
        super().__init__()

        self.W_enc = torch.nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(input_sz, hidden_sz, dtype=dtype)))
        self.W_dec = torch.nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(hidden_sz, input_sz, dtype=dtype)))
        self.b_dec = torch.nn.Parameter(torch.zeros(input_sz, dtype=dtype))
        self.b_enc = torch.nn.Parameter(torch.zeros(hidden_sz, dtype=dtype))

        # decoder weights must be unit norm
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

    def forward(self, x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        return x_reconstruct, acts
    
    def loss(x, x_recon, acts, l1_coeff):
        l2_loss = (x_recon.float() - x.float()).pow(2).sum(-1).mean(0)
        l1_loss = l1_coeff * (acts.float().abs().sum())
        loss = l2_loss + l1_loss
        return loss, l2_loss.detach(), l1_loss.detach()

    @torch.no_grad()
    def unit_norm_decoder_weights_and_grad(self):
        # remove the portion of the gradient that is parallel to the dictionary elements
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed     # component of the gradient in the direction of W_dec_normed (first mult + sum are a batched dot prod)
        self.W_dec.grad -= W_dec_grad_proj
        
        # maintain decoder unit norm
        self.W_dec.data = W_dec_normed


