import sys
sys.path.append('../..')

import torch
import pickle

from torch.utils.data.dataloader import DataLoader

from reaction_predictors.graph_model.models import RGCNNTrClassifier
from utils.graph_utils import get_bonds, get_nodes
from utils.torch_dataset import Dataset, graph_collate
from utils.create_graph_dataset import prune_dataset_by_length
from reaction_predictors.graph_model.model_utils import train_epoch, evaluate, test


result_path = '../../data/results/mp/base_transformer.pkl'
dataset_path = '../../data/graphs/'
eval_names = ['Main product mapping']
n_molecule_level = 0
n_reaction_level = 0
device = 'cuda:0'
max_num_atoms = 50
self_bond = True
batch_size = 64
n_hidden = 64
lr = 1e-3
num_conv_layers = 4
num_fcn_layers = 2
num_trans_layers = 2
num_attention_heads = 4
num_model_heads = 1
n_epoches = 5
feature_embed_size = 0
exp_step_size = 10000
feature_idxs = ()
feature_sizes = []
target_center = False
target_main_product = True

meta = pickle.load(open(dataset_path + '/meta.pkl', 'rb'))

node2label = get_nodes(meta['node'], n_molecule_level=n_molecule_level, n_reaction_level=n_reaction_level)
bond2label = get_bonds(meta['type'], n_molecule_level=n_molecule_level, n_reaction_level=n_reaction_level, self_bond=True)

num_rels = len(bond2label)
pad_length = max_num_atoms + 15*n_molecule_level + n_molecule_level*n_reaction_level
num_nodes = pad_length

model = RGCNNTrClassifier([len(node2label)]+feature_sizes,
              num_nodes,
              batch_size,
              [n_hidden]+[feature_embed_size]*len(feature_sizes),
              num_rels,
              num_conv_layers,
              num_trans_layers,
              num_fcn_layers,
              num_attention_heads,
              num_model_heads,
             )
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=exp_step_size, gamma=0.1)


train_dataset = pickle.load(open(dataset_path + 'train.pkl', 'rb'))
test_dataset = pickle.load(open(dataset_path + 'test.pkl', 'rb'))
valid_dataset = pickle.load(open(dataset_path + 'valid.pkl', 'rb'))
train_dataset = prune_dataset_by_length(train_dataset, max_num_atoms)
test_dataset = prune_dataset_by_length(test_dataset, max_num_atoms)
valid_dataset = prune_dataset_by_length(valid_dataset, max_num_atoms)

tr_dataset = Dataset(train_dataset, device=device, pad_length=pad_length,
                     bond2label=bond2label, node2label=node2label, feature_idxs=feature_idxs,
                     target_main_product=target_main_product, target_center=target_center,
                     n_molecule_level=n_molecule_level, n_reaction_level=n_reaction_level)
ts_dataset = Dataset(test_dataset, device=device, pad_length=pad_length,
                     bond2label=bond2label, node2label=node2label, feature_idxs=feature_idxs,
                     target_main_product=target_main_product, target_center=target_center,
                     n_molecule_level=n_molecule_level, n_reaction_level=n_reaction_level)
vl_dataset = Dataset(valid_dataset, device=device, pad_length=pad_length,
                     bond2label=bond2label, node2label=node2label, feature_idxs=feature_idxs,
                     target_main_product=target_main_product, target_center=target_center,
                     n_molecule_level=n_molecule_level, n_reaction_level=n_reaction_level)

train_loader = DataLoader(tr_dataset, batch_size, drop_last=True, collate_fn=graph_collate)
test_loader = DataLoader(ts_dataset, batch_size, drop_last=True, collate_fn=graph_collate)
valid_loader = DataLoader(vl_dataset, batch_size, drop_last=True, collate_fn=graph_collate)

valid_scores = []
losses = []
print('Training is starting')
for epoch in range(n_epoches):
    losses.append(train_epoch(model, train_loader, optimizer, exp_lr_scheduler))
    print(f'Epoch number - {epoch}')
    valid_scores.append(evaluate(model, valid_loader, eval_names))
print(f'Final result')
test_score = evaluate(model, test_loader, eval_names)

with open(result_path, 'wb') as f:
    pickle.dump([test_score, losses, valid_scores], f)


