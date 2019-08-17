from torch import nn
from torch.nn import MultiheadAttention, Linear, Dropout
from torch.nn.modules.transformer import LayerNorm


class TransLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super(TransLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=1024, dropout=0.1):
        super(TransModel, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(TransLayer(d_model, nhead))
        self.layers = nn.ModuleList(layers)

    def forward(self, src):
        h = src
        for layer in self.layers:
            h = layer(h)
        return h
