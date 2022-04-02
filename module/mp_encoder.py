import torch
import torch.nn as nn
from module.aggregate import Aggregator
from torch_geometric.nn import GCNConv, Sequential


class Mp_encoder(nn.Module):
    def __init__(self, use_data, hidden_dim, attn_drop):
        super(Mp_encoder, self).__init__()

        self.mp_dict = {
            mp: i for i, mp in enumerate(use_data.metapath_dict)
        }
    
        self.intra = nn.ModuleList([
            Sequential('x, edge_index', [
                (GCNConv(hidden_dim, hidden_dim), 'x, edge_index -> x'),
                nn.PReLU()
            ])
            for _ in range(len(self.mp_dict))
        ])
        self.inter = Aggregator(hidden_dim, attn_drop)

    def forward(self, h, data):
        embeds = []

        for mp in data.metapath_dict:
            n = mp[0]
            i = self.mp_dict[mp]
            edge_index = data[mp].edge_index
            embed = self.intra[i](h[n], edge_index)
            embeds.append(embed)

        embeds = torch.stack(embeds)

        return self.inter(embeds)
