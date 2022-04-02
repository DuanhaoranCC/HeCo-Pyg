import torch.nn as nn
import torch.nn.functional as F
from module.mp_encoder import Mp_encoder
from module.sc_encoder import Sc_encoder
from module.contrast import Contrast

class HeCo(nn.Module):
    def __init__(
        self,
        use_data,
        hidden_dim,
        feat_drop,
        attn_drop,
        sample_rate,
        tau,
        lam
    ):
        super(HeCo, self).__init__()

        self.hidden_dim = hidden_dim

        self.fc = nn.ModuleDict({
            n_type: nn.Linear(
                use_data[n_type].x.shape[1], 
                hidden_dim, 
                bias=True
            )
            for n_type in use_data.use_nodes
        })

        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x

        self.mp = Mp_encoder(use_data, hidden_dim, attn_drop)
        self.sc = Sc_encoder(use_data, hidden_dim, sample_rate, attn_drop)      
        self.contrast = Contrast(hidden_dim, tau, lam)
        self.pos = use_data.main_node, 'pos', use_data.main_node

        self.reset_parameter()

    def reset_parameter(self):
        for fc in self.fc.values():
            nn.init.xavier_normal_(fc.weight, gain=1.414)

    def forward(self, data):  # p a s
        h = {}
        for n_type in data.use_nodes:
            h[n_type] = F.elu(
                self.feat_drop(
                    self.fc[n_type](data[n_type].x)
                )
            )
        z_mp = self.mp(h, data)
        z_sc = self.sc(h, data)
        return self.contrast(z_mp, z_sc, data[self.pos].edge_index)

    def get_embeds(self, data):
        m = data.main_node
        h = {m: F.elu(self.fc[m](data[m].x))}
        return self.mp(h, data).detach()
