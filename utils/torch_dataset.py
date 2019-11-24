import torch
import dgl
import numpy as np

from torch.utils import data

from utils.graph_utils import get_graph


def graph_collate(batch):
    graphs = [item[0] for item in batch]
    mp_target = [item[1] for item in batch]
    c_target = [item[2] for item in batch]
    graphs = dgl.batch(graphs)
    mp_target = torch.stack(mp_target)
    c_target = torch.stack(c_target)
    return [graphs, mp_target, c_target]


class Dataset(data.Dataset):
    def __init__(self, dataset, device, pad_length,
                 bond2label, node2label,
                 target_center=True, target_main_product=True):
        self.dataset = dataset
        self.list_idxs = list(dataset.keys())
        self.device = device
        self.pad_length = pad_length
        self.bond2label = bond2label
        self.node2label = node2label
        self.targets = [target_center, target_main_product]

    def __len__(self):
        return len(self.list_idxs)

    def __getitem__(self, index):
        g = get_graph(self.dataset[self.list_idxs[index]], self.bond2label,
                      self.node2label, pad_length=self.pad_length,
                      device=self.device, features=True)
        target_main_product = self.dataset[self.list_idxs[index]]['target_main_product']
        target_main_product = np.pad(target_main_product, (0, self.pad_length - len(target_main_product)),
                                     constant_values=-1)
        target_center = self.dataset[self.list_idxs[index]]['target_center']
        target_center = np.pad(target_center, (0, self.pad_length - len(target_center)), constant_values=-1)

        target_main_product = torch.from_numpy(target_main_product).to(device).float()
        target_center = torch.from_numpy(target_center).to(self.device).float()
        if self.targets[0] and self.targets[1]:
            return g, target_main_product, target_center
        elif self.targets[0]:
            return g, target_main_product
        else:
            return g, target_center
