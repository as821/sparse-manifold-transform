import torch
import torch.nn.functional as F
import math

import pdb

class SparseAutoEncoder(torch.nn.Module):
    def __init__(self, input_sz, hidden_sz, activ_thresh, dtype=torch.float32):
        super().__init__()

        self.thresh = activ_thresh

        self.enc = torch.nn.Linear(input_sz, hidden_sz)

        self.dec = torch.nn.Parameter(torch.zeros(hidden_sz, input_sz))
        torch.nn.init.uniform_(self.dec, a=1)
        with torch.no_grad():
            self.dec.data[:] = F.normalize(self.dec.data, dim=0)



    def forward(self, x):
        # acts = torch.nn.functional.relu(x @ self.dec.T)


        acts = self.enc(x)
        acts = F.sigmoid(acts)

        hard_acts = (acts > 0.5).float()
        hard_acts = hard_acts.detach() - acts.detach() + acts  # STE trick



        # print(f"{acts.min()} {acts.mean()} {acts.max()}")
        # pdb.set_trace()

        # Set activations >= threshold == 1. This step is non-differentiable so use a straight through estimator
        # hard_acts = (acts >= self.thresh).float()
        # acts = acts * hard_acts.detach()        # mask out gradients as well
        # acts = acts + (hard_acts - acts).detach()

        # # (and take average of the activated dictionary members by dividing by N), keep this internal so as to not impact L1 loss
        # n_acts = torch.sum(acts, dim=1, keepdim=True)
        # n_acts = torch.clamp(n_acts, min=1)     # avoid div by zero
        # recon = (acts / n_acts) @ self.dec
        

        recon = hard_acts @ self.dec

        recon = torch.nn.functional.normalize(recon, dim=1)     # all patches are normalized
        return recon, acts
    
    def loss(x, x_recon, acts, l1_coeff):
        # l2_loss = torch.zeros((1,), device="cuda") #
        # l2_loss = (x_recon.float() - x.float()).pow(2).sum(-1).mean(0)
        # l1_loss = l1_coeff * acts.float().abs().sum(1).mean()

        l2_loss = F.mse_loss(x_recon, x)
        l1_loss = l1_coeff * torch.mean(acts)
        loss = l2_loss + l1_loss
        return loss, l2_loss.detach(), l1_loss.detach()

    @torch.no_grad()
    def unit_norm_decoder_weights_and_grad(self):
        # # remove the portion of the gradient that is parallel to the dictionary elements
        # dec_normed = self.dec.data / self.dec.data.norm(dim=-1, keepdim=True)
        # dec_grad_proj = (self.dec.grad * dec_normed).sum(-1, keepdim=True) * dec_normed     # component of the gradient in the direction of W_dec_normed (first mult + sum are a batched dot prod)
        # self.dec.grad -= dec_grad_proj
        
        # # maintain decoder unit norm
        # self.dec.data = dec_normed
        # print(self.dec.grad.max())
        pass

