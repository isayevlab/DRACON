import torch
import pickle
import yaml
import pandas as pd

from torch import nn
from rdkit import Chem
from lib.node_classification_model.models import RGCNNTrClassifier
from lib.dataset.build_dgl_graph import get_bonds, get_nodes
from lib.dataset.torch_dataset import Dataset
from lib.general_utils import convert
from lib.draw_utils import get_molecule_svg
from lib.dataset.build_dataset import build_dataset
from rdkit.Chem import rdDepictor


def infer(smiles, device='cpu'):
    with open('../experiments/MT_EGTBF_demo.yml', 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    state_dict = torch.load('../data/models/model_50_demo.pth', map_location=device)
    model_cfg = convert(config["model"])
    data_cfg = convert(config["dataset"])
    paths = convert(config["paths"])
    meta = pickle.load(open(paths.dataset_path + '/meta.pkl', 'rb'))

    node2label = get_nodes(meta['node'], n_molecule_level=data_cfg.n_molecule_level,
                           n_reaction_level=data_cfg.n_reaction_level)
    bond2label = get_bonds(meta['type'], n_molecule_level=data_cfg.n_molecule_level,
                           n_reaction_level=data_cfg.n_reaction_level,
                           self_bond=data_cfg.self_bond)
    num_rels = len(bond2label)
    pad_length = data_cfg.max_num_atoms + 15 * data_cfg.n_molecule_level + \
                 data_cfg.n_molecule_level * data_cfg.n_reaction_level
    num_nodes = pad_length

    model = RGCNNTrClassifier([len(node2label)] + data_cfg.feature_sizes,
                              num_nodes,
                              1,
                              [model_cfg.n_hidden] + [model_cfg.feature_embed_size] * len(data_cfg.feature_sizes),
                              num_rels,
                              model_cfg.num_conv_layers,
                              model_cfg.num_trans_layers,
                              model_cfg.num_fcn_layers,
                              model_cfg.num_attention_heads,
                              model_cfg.num_model_heads,
                              )
    model = model.to(device)
    model.load_state_dict(state_dict)

    df = pd.DataFrame([smiles + '>>CC'], columns=['smarts'])
    dataset = build_dataset(df, atom_labels='False')
    length = len(dataset[0]['reactants']['nodes'])
    dataset[0]['reactants']['features'][-1] += 5
    print(dataset[0])
    dataset = Dataset(dataset, device=device, pad_length=pad_length,
                      bond2label=bond2label, node2label=node2label, feature_idxs=data_cfg.feature_idxs,
                      target_main_product=False, target_center=False,
                      n_molecule_level=data_cfg.n_molecule_level, n_reaction_level=data_cfg.n_reaction_level)
    g = dataset[0]
    sigmoid = nn.Sigmoid()
    model.eval()
    with torch.no_grad():
        outputs = model(g)
        predicted_mp = (sigmoid(outputs[0]) > .5).float().cpu().detach().numpy()[0]
        predicted_c = (sigmoid(outputs[1]) > .5).float().cpu().detach().numpy()[0]
    predicted_c[predicted_mp == 0] = 0
    predicted = predicted_mp + predicted_c

    fontsize = 0.98
    gt_colors = {1: (0.8, 1, 0.8), 2: (0.5, 0.8, 1)}
    r_mol = Chem.MolFromSmiles(smiles.split('>')[0])
    rdDepictor.Compute2DCoords(r_mol)
    r_svg = get_molecule_svg(r_mol, target=predicted[:length], target_type='GT',
                             gt_colors=gt_colors, dpa=100, fontsize=fontsize)
    return r_svg
