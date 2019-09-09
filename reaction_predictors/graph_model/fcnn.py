from torch import nn


class FCNModel(nn.Module):
    def __init__(self, d_model, num_layers=3):
        super(FCNModel, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(d_model, d_model))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(d_model, 1))
        self.layers = nn.ModuleList(layers)

    def forward(self, src):
        h = src
        for layer in self.layers:
            h = layer(h)
        return h
