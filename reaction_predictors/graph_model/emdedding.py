import torch

from torch import nn


class Embedding(nn.Module):
    def __init__(self, vocab_sizes, embedding_sizes):
        super(Embedding, self).__init__()
        self.embeds = []
        for vocab_size, embedding_size in zip(vocab_sizes, embedding_sizes):
            self.embeds.append(nn.Embedding(vocab_size, embedding_size))
        self.embeds = nn.ModuleList(self.embeds)

    def forward(self, cat_features):
        res = []
        for embed, cat_feat in zip(self.embeds, cat_features):
            res.append(embed(cat_feat))
        return torch.cat(res, dim=1)
