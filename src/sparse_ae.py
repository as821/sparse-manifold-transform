import torch
import torch.nn.functional as F

import pdb

class SparseAutoEncoder(torch.nn.Module):
    def __init__(self, input_sz, hidden_sz, activ_thresh, dtype=torch.float32):
        super().__init__()

        self.thresh = activ_thresh

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
        
        print(f"{acts.detach().min()} {acts.detach().mean()} {acts.detach().max()}")


        # Set activations >= threshold == 1. This step is non-differentiable so use a straight through estimator
        hard_acts = (acts >= self.thresh).float()
        acts = acts + (hard_acts - acts).detach()

        # (and take average of the activated dictionary members by dividing by N), keep this internal so as to not impact L1 loss
        n_acts = torch.sum(acts, dim=1, keepdim=True)
                
        n_acts = torch.clamp(n_acts, min=1)     # avoid div by zero
        recon = self.dec(acts / n_acts)
        recon = recon + self.b_dec
        

        # all patches are normalized. force reconstruction to do the same
        recon = torch.nn.functional.normalize(recon, dim=1)
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

