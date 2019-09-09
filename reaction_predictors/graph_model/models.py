from torch import nn

from reaction_predictors.graph_model.fcnn import FCNModel
from reaction_predictors.graph_model.rgcnn import RGCNModel
from reaction_predictors.graph_model.transformer import TransModel


class RGCNNClassifier(nn.Module):
    def __init__(self,
                 in_feat,
                 n_nodes,
                 batch_size,
                 h_dim,
                 num_rels,
                 num_conv_layers=4,
                 num_fcn_layers=2):
        super(RGCNNClassifier, self).__init__()
        self.n_nodes = n_nodes
        self.h_dim = h_dim
        self.batch_size = batch_size
        self.rgcn = RGCNModel(in_feat, h_dim, num_rels, num_conv_layers, bias=None)
        self.fcn = FCNModel(h_dim, num_layers=num_fcn_layers)

    def forward(self, g):
        h = self.rgcn(g).view((self.batch_size, self.n_nodes, self.h_dim))
        out = self.fcn(h)
        return out


class RGCNNTrClassifier(nn.Module):
    def __init__(self,
                 in_feat,
                 n_nodes,
                 batch_size,
                 h_dim,
                 num_rels,
                 num_conv_layers=6,
                 num_trans_layers=1,
                 num_fcn_layers=1,
                 num_attention_heads=1):
        super(RGCNNTrClassifier, self).__init__()
        self.n_nodes = n_nodes
        self.h_dim = h_dim
        self.batch_size = batch_size
        self.rgcn = RGCNModel(in_feat, h_dim, num_rels, num_conv_layers, bias=None)
        self.trans = TransModel(h_dim, num_attention_heads, num_trans_layers)
        self.fcn = FCNModel(h_dim, num_layers=num_fcn_layers)

    def forward(self, g):
        h = self.rgcn(g).view((self.batch_size, self.n_nodes, self.h_dim)).permute(1, 0, 2)
        h = self.trans(h).permute(1, 0, 2)
        out = self.fcn(h)
        return out
