import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, bias=None,
                 activation=None):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.bias = bias
        self.activation = activation

        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.in_feat,
                                                self.out_feat))

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))

        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))

        if self.bias:
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

    def forward(self, g):

        def message_func(edges):
            w = self.weight[edges.data['rel_type']]
            msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
            return {'msg': msg}

        def apply_func(nodes):
            h = nodes.data['h']
            if self.bias:
                h = h + self.bias
            if self.activation:
                h = self.activation(h)
            return {'h': h}

        g.update_all(message_func, fn.mean(msg='msg', out='h'), apply_func)


class RGCNModel(nn.Module):
    def __init__(self, h_dim, num_rels, num_layers, bias=None):
        super(RGCNModel, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(RGCNLayer(h_dim, h_dim, num_rels, activation=F.relu, bias=bias))
        self.layers = nn.ModuleList(layers)

    def forward(self, g):
        for layer in self.layers:
            layer(g)
        return g.ndata['h']
