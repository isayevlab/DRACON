import torch
import dgl
import numpy as np

from torch.utils import data

from lib.dataset.build_dgl_graph import get_graph


def graph_collate(batch):
    n_targets = len(batch[0]) - 1
    targets = []
    graphs = [item[0] for item in batch]
    for i in range(n_targets):
        targets.append([item[i + 1] for item in batch])
    graphs = dgl.batch(graphs)
    for j, tar in enumerate(targets):
        targets[j] = torch.stack(tar)
    return [graphs, *targets]


class Dataset(data.Dataset):
    def __init__(self, dataset, device, pad_length,
                 bond2label, node2label, n_molecule_level=1, n_reaction_level=1,
                 target_center=True, target_main_product=True,
                 feature_idxs=()):
        self.dataset = dataset
        self.list_idxs = list(dataset.keys())
        self.device = device
        self.pad_length = pad_length
        self.bond2label = bond2label
        self.node2label = node2label
        self.targets = [target_main_product, target_center,]
        self.feature_idxs = feature_idxs
        self.n_molecule_level = n_molecule_level
        self.n_reaction_level = n_reaction_level

    def __len__(self):
        return len(self.list_idxs)

    def __getitem__(self, index):
        g = get_graph(self.dataset[self.list_idxs[index]], self.bond2label,
                      self.node2label, pad_length=self.pad_length,
                      device=self.device, feature_idxs=self.feature_idxs,
                      n_molecule_level=self.n_molecule_level, n_reaction_level=self.n_reaction_level)
        if self.targets[0]:
            target_main_product = self.dataset[self.list_idxs[index]]['target_main_product']
            target_main_product = np.pad(target_main_product, (0, self.pad_length - len(target_main_product)),
                                         constant_values=-1)
            target_main_product = torch.from_numpy(target_main_product).to(self.device).float()
        if self.targets[1]:
            target_center = self.dataset[self.list_idxs[index]]['target_center']
            target_center = np.pad(target_center, (0, self.pad_length - len(target_center)), constant_values=-1)
            target_center = torch.from_numpy(target_center).to(self.device).float()
        if self.targets[0] and self.targets[1]:
            return g, target_main_product, target_center
        elif self.targets[0]:
            return g, target_main_product
        elif self.targets[1]:
            return g, target_center
        else:
            return g
