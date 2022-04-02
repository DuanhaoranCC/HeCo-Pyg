import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from module.aggregate import Aggregator
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, zeros

class GAT(MessagePassing):
    def __init__(self, in_channels, dropout, bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.dropout = dropout

        self.att_src = Parameter(torch.Tensor(1, in_channels))
        self.att_dst = Parameter(torch.Tensor(1, in_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(in_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)

    def forward(self, x, edge_index):
        x_src, x_dst = x
        
        alpha_src = (x_src * self.att_src).sum(-1)
        alpha_dst = (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        out = self.propagate(edge_index, x=x, alpha=alpha, size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

class Sc_encoder(nn.Module):
    def __init__(self, use_data, hidden_dim, sample_rate, attn_drop):
        super(Sc_encoder, self).__init__()
        self.schema_dict = {s: i for i, s in enumerate(use_data.schema_dict)}
        self.intra = nn.ModuleList([
            GAT(hidden_dim, attn_drop) 
            for _ in range(len(self.schema_dict))
        ])
        self.inter = Aggregator(hidden_dim, attn_drop)
        self.sample_rate = sample_rate

    def sample_edge_index(self, schema_type, edge_index):
        # [2, num_edges] -> [num_edges, 2]. index 1 is dst.
        sample_size = self.sample_rate[schema_type]
        num_nodes = int(edge_index[1].max() + 1)
        e = edge_index.clone().T
        buc = torch.zeros(num_nodes, dtype=torch.long, device=e.device)
        r = torch.randperm(e.shape[0], dtype=torch.long, device=e.device)
        e = e[r]

        edge_list = []
        for edge in e:
            _, dst = edge
            if buc[dst] < sample_size:
                edge_list.append(edge)
                buc[dst] += 1

        edge_index = torch.stack(edge_list)
        return edge_index.T


    def forward(self, h, data):
        embeds = []

        for sc in data.schema_dict:
            src, dst = sc
            edge_index = self.sample_edge_index(src, data[sc].edge_index)
            x = h[src], h[dst]
            embed = self.intra[self.schema_dict[sc]](x, edge_index)
            embeds.append(embed)

        embeds = torch.stack(embeds)

        return self.inter(embeds)
