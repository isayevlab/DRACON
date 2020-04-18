import torch
import pickle
import argparse
import yaml
import init_path

from torch.utils.data.dataloader import DataLoader
from lib.node_classification_model.models import RGCNNTrClassifier
from lib.dataset.build_dgl_graph import get_bonds, get_nodes
from lib.dataset.torch_dataset import Dataset, graph_collate
from lib.node_classification_model.model_utils import train_epoch, evaluate
from lib.general_utils import convert
from lib.dataset.utils import filter_dataset


def main(config, device):
    model_cfg = convert(config["model"])
    data_cfg = convert(config["dataset"])
    train_cfg = convert(config["train"])
    paths = convert(config["paths"])

    meta = pickle.load(open(paths.dataset_path + 'meta.pkl', 'rb'))

    node2label = get_nodes(meta['node'], n_molecule_level=data_cfg.n_molecule_level,
                           n_reaction_level=data_cfg.n_reaction_level)
    bond2label = get_bonds(meta['type'], n_molecule_level=data_cfg.n_molecule_level,
                           n_reaction_level=data_cfg.n_reaction_level,
                           self_bond=data_cfg.self_bond)
    if data_cfg.same_bond:
        bond2label = {i: 0 if i in meta['type'] else bond2label[i] for i in bond2label}
    num_rels = len(bond2label)
    pad_length = data_cfg.max_num_atoms + data_cfg.max_num_reactants * data_cfg.n_molecule_level + \
                 data_cfg.n_molecule_level * data_cfg.n_reaction_level
    num_nodes = pad_length

    model = RGCNNTrClassifier([len(node2label)] + data_cfg.feature_sizes,
                              num_nodes,
                              train_cfg.batch_size,
                              [model_cfg.n_hidden] + [model_cfg.feature_embed_size] * len(data_cfg.feature_sizes),
                              num_rels,
                              model_cfg.num_conv_layers,
                              model_cfg.num_trans_layers,
                              model_cfg.num_fcn_layers,
                              model_cfg.num_attention_heads,
                              model_cfg.num_model_heads,
                              )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, betas=(0.9, 0.98), eps=1e-9)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=train_cfg.exp_step_size, gamma=0.1)

    train_dataset = pickle.load(open(paths.dataset_path + 'train.pkl', 'rb'))
    test_dataset = pickle.load(open(paths.dataset_path + 'test.pkl', 'rb'))
    valid_dataset = pickle.load(open(paths.dataset_path + 'valid.pkl', 'rb'))

    train_dataset = filter_dataset(train_dataset, data_cfg.max_num_atoms, data_cfg.max_num_reactants)
    test_dataset = filter_dataset(test_dataset, data_cfg.max_num_atoms, data_cfg.max_num_reactants)
    valid_dataset = filter_dataset(valid_dataset, data_cfg.max_num_atoms, data_cfg.max_num_reactants)

    tr_dataset = Dataset(train_dataset, device=device, pad_length=pad_length,
                         bond2label=bond2label, node2label=node2label, feature_idxs=data_cfg.feature_idxs,
                         target_main_product=data_cfg.target_main_product, target_center=data_cfg.target_center,
                         n_molecule_level=data_cfg.n_molecule_level, n_reaction_level=data_cfg.n_reaction_level)
    ts_dataset = Dataset(test_dataset, device=device, pad_length=pad_length,
                         bond2label=bond2label, node2label=node2label, feature_idxs=data_cfg.feature_idxs,
                         target_main_product=data_cfg.target_main_product, target_center=data_cfg.target_center,
                         n_molecule_level=data_cfg.n_molecule_level, n_reaction_level=data_cfg.n_reaction_level)
    vl_dataset = Dataset(valid_dataset, device=device, pad_length=pad_length,
                         bond2label=bond2label, node2label=node2label, feature_idxs=data_cfg.feature_idxs,
                         target_main_product=data_cfg.target_main_product, target_center=data_cfg.target_center,
                         n_molecule_level=data_cfg.n_molecule_level, n_reaction_level=data_cfg.n_reaction_level)

    train_loader = DataLoader(tr_dataset, train_cfg.batch_size, drop_last=True, collate_fn=graph_collate)
    test_loader = DataLoader(ts_dataset, train_cfg.batch_size, drop_last=True, collate_fn=graph_collate)
    valid_loader = DataLoader(vl_dataset, train_cfg.batch_size, drop_last=True, collate_fn=graph_collate)

    valid_scores = []
    losses = []
    print('Training is starting')
    for epoch in range(train_cfg.n_epoches):
        losses.append(train_epoch(model, train_loader, optimizer, exp_lr_scheduler))
        print(f'Epoch number - {epoch}')
        valid_scores.append(evaluate(model, valid_loader, train_cfg.eval_names))
        torch.save(model, paths.save_path)
    print(f'Final result')
    test_score = evaluate(model, test_loader, train_cfg.eval_names)

    with open(paths.result_path, 'wb') as f:
        pickle.dump([test_score, losses, valid_scores], f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="YAML configuration file")
    parser.add_argument("--device", default='cuda:0', help="Device to run and train model")
    args = parser.parse_args()
    with open(args.config, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    main(cfg, args.device)
