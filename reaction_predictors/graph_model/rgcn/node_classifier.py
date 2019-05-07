import torch.nn as nn
import torch.nn.functional as F

from reaction_predictors.graph_model.layers import RGCNLayer
from functools import partial


class NodeClassifier(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels,
                 num_bases=-1, num_hidden_layers=1):
        super(NodeClassifier, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        for _ in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)
        h2o = self.build_output_layer()
        self.layers.append(nn.Linear(self.num_nodes, self.num_nodes))
        self.layers.append(nn.ReLU())
        self.layers.append(h2o)

    def build_input_layer(self):
        return RGCNLayer(self.num_nodes, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu, is_input_layer=True)

    def build_hidden_layer(self):
        return RGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu)

    def build_output_layer(self):
        return RGCNLayer(self.h_dim, self.out_dim, self.num_rels, self.num_bases,
                         activation=partial(F.softmax, dim=1))

    def forward(self, g):
        for layer in self.layers[:-3]:
            layer(g)
        in_lin = g.ndata['h'].permute(1, 0)
        out_lin = self.layers[-3](in_lin.unsqueeze(0)).permute(0, 2, 1)
        out_lin = self.layers[-2](out_lin)
        g.ndata['h'] = out_lin[0]        
        self.layers[-1](g)
        return g.ndata['h']
