import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter.scatter import scatter_add

class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        self.reset_parameters()
    
    def reset_parameters(self):
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = z1 @ z2.T
        dot_denominator = z1_norm @ z2_norm.T
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def cus_mul(self, pos, mat):
        row, col = pos
        mat_val = mat[row, col]
        return scatter_add(mat_val, row)

    def forward(self, z_mp, z_sc, pos):
        z_proj_mp = self.proj(z_mp)
        z_proj_sc = self.proj(z_sc)

        sim_mp2sc = self.sim(z_proj_mp, z_proj_sc)
        sim_sc2mp = sim_mp2sc.T
        
        sim_mp2sc = F.normalize(sim_mp2sc, p=1, dim=1)
        sim_sc2mp = F.normalize(sim_sc2mp, p=1, dim=1)

        lori_mp = -torch.log(self.cus_mul(pos, sim_mp2sc)).mean()
        lori_sc = -torch.log(self.cus_mul(pos, sim_sc2mp)).mean()

        return self.lam * lori_mp + (1 - self.lam) * lori_sc
