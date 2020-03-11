from torch import nn

from reaction_predictors.graph_model.fcnn import FCNModel
from reaction_predictors.graph_model.rgcnn import RGCNModel
from reaction_predictors.graph_model.transformer import TransModel
from reaction_predictors.graph_model.emdedding import Embedding


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
                 num_attention_heads=1,
                 n_model_heads=1):
        super(RGCNNTrClassifier, self).__init__()
        self.n_nodes = n_nodes
        self.h_dim = sum(h_dims)
        self.batch_size = batch_size
        self.embed = Embedding(feat_sizes, h_dims)
        if num_conv_layers > 0:
            self.rgcn = RGCNModel(self.h_dim, num_rels, num_conv_layers, bias=None)
        else:
            self.rgcn = None
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCEWithLogitsLoss()
        if num_trans_layers > 0:
            self.trans = TransModel(self.h_dim, num_attention_heads, num_trans_layers)
        else:
            self.trans = None
        self.fcns = nn.ModuleList([])

        for _ in range(n_model_heads):
            self.fcns.append(FCNModel(self.h_dim, num_layers=num_fcn_layers))
        self.n_model_heads = n_model_heads

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, g):
        h = self.embed(g.ndata['feats'].T)
        if self.rgcn is not None:
            g.ndata['h'] = h
            h = self.rgcn(g)
        h = h.view((self.batch_size, self.n_nodes, self.h_dim))
        if self.trans is not None:
            h = self.trans(h.permute(1, 0, 2)).permute(1, 0, 2)
        outs = []
        for fcn in self.fcns:
            outs.append(fcn(h)[:, :, 0])
        return tuple(outs)

    def get_loss(self, g, targets):
        outs = self.forward(g)
        loss = 0
        for out, target in zip(outs, targets):
            loss += self.criterion(out[target != -1], target[target != -1])
        return loss
