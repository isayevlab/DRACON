from torch import nn

from reaction_predictors.graph_model.fcnn import FCNModel
from reaction_predictors.graph_model.rgcnn import RGCNModel
from reaction_predictors.graph_model.transformer import TransModel
from reaction_predictors.graph_model.emdedding import Embedding


class RGCNNClassifier(nn.Module):
    def __init__(self,
                 feat_sizes,
                 n_nodes,
                 batch_size,
                 h_dims,
                 num_rels,
                 num_conv_layers=4,
                 num_fcn_layers=2):
        super(RGCNNClassifier, self).__init__()
        self.n_nodes = n_nodes
        self.h_dim = sum(h_dims)
        self.batch_size = batch_size
        self.embed = Embedding(feat_sizes, h_dims)
        self.rgcn = RGCNModel(self.h_dim, num_rels, num_conv_layers, bias=None)
        self.fcn = FCNModel(self.h_dim, num_layers=num_fcn_layers)

    def forward(self, g):
        h = self.embed(g.ndata['feats'].T)
        g.ndata['h'] = h
        h = self.rgcn(g).view((self.batch_size, self.n_nodes, self.h_dim))
        out = self.fcn(h)
        return out


class RGCNNTrClassifier(nn.Module):
    def __init__(self,
                 feat_sizes,
                 n_nodes,
                 batch_size,
                 h_dims,
                 num_rels,
                 num_conv_layers=6,
                 num_trans_layers=1,
                 num_fcn_layers=1,
                 num_attention_heads=1):
        super(RGCNNTrClassifier, self).__init__()
        self.n_nodes = n_nodes
        self.h_dim = sum(h_dims)
        self.batch_size = batch_size
        self.embed = Embedding(feat_sizes, h_dims)
        self.rgcn = RGCNModel(self.h_dim, num_rels, num_conv_layers, bias=None)
        self.trans = TransModel(self.h_dim, num_attention_heads, num_trans_layers)
        self.fcn = FCNModel(self.h_dim, num_layers=num_fcn_layers)

    def forward(self, g):
        h = self.embed(g.ndata['feats'].T)
        g.ndata['h'] = h
        h = self.rgcn(g).view((self.batch_size, self.n_nodes, self.h_dim)).permute(1, 0, 2)
        h = self.trans(h).permute(1, 0, 2)
        out = self.fcn(h)
        return out
