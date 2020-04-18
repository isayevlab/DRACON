import dgl
import torch
import numpy as np

from copy import deepcopy


def get_bonds(bond_types, n_molecule_level=1, n_reaction_level=1, self_bond=True):
    bond_types = list(bond_types)
    bond2label = dict(zip(bond_types, range(len(bond_types))))
    if self_bond:
        bond2label['SELF'] = len(bond2label)
    for i in range(n_molecule_level):
        bond2label[f'ML_{i}'] = len(bond2label)
    for i in range(n_molecule_level):
        for j in range(n_reaction_level):
            bond2label[f'RL_{i, j}'] = len(bond2label)
    return bond2label


def get_nodes(node_types, n_molecule_level=1, n_reaction_level=1):
    node_types = list(node_types)
    node2label = dict(zip(node_types, range(len(node_types))))
    node2label['EMPTY'] = len(node2label)

    for i in range(n_molecule_level):
        node2label[f'ML_{i}'] = len(node2label)

    for i in range(n_molecule_level):
        for j in range(n_reaction_level):
            node2label[f'RL_{i, j}'] = len(node2label)
    return node2label


def get_graph(rxn,
              bond2label,
              node2label,
              n_molecule_level=1,
              n_reaction_level=1,
              self_bond=True,
              pad_length=120,
              device='cuda',
              feature_idxs=()
              ):
    nodes = list(rxn['reactants']['nodes'])
    sender, reciever, bonds = list(rxn['reactants']['sender']), list(rxn['reactants']['reciever']), list(
        rxn['reactants']['types'])
    sender = sender + reciever
    reciever = reciever + sender[:len(reciever)]
    bonds = bonds + bonds

    nodes = [node2label[i] for i in nodes]
    bonds = [bond2label[i] for i in bonds]

    n_atoms = len(nodes)
    molecules_lengths = rxn['reactants']['lengths']
    n_molecules = len(molecules_lengths) - 1

    for i in range(n_molecule_level):
        nodes.extend([node2label[f'ML_{i}']] * n_molecules)

    for i in range(n_molecule_level):
        for j in range(n_reaction_level):
            nodes.append(node2label[f'RL_{i, j}'])

    mol_sender = []
    mol_reciever = []
    mol_bonds = []

    for i in range(n_molecule_level):
        for j, (st, fi) in enumerate(zip(molecules_lengths[:-1], molecules_lengths[1:])):
            mol_sender.extend(list(range(st, fi)) + [n_atoms + i * n_molecules + j] * (fi - st))
            mol_reciever.extend([n_atoms + i * n_molecules + j] * (fi - st) + list(range(st, fi)))
            mol_bonds.extend([bond2label[f'ML_{i}']] * (2 * (fi - st)))

    rxn_sender = []
    rxn_reciever = []
    rxn_bonds = []

    for i in range(n_molecule_level):
        for j in range(n_reaction_level):
            rxn_sender.extend(list(range(n_atoms + i * n_molecules, n_atoms + (i + 1) * n_molecules)) +
                              [n_atoms + n_molecule_level * n_molecules + i * n_molecule_level + j] * n_molecules)
            rxn_reciever.extend([n_atoms + n_molecule_level * n_molecules + i * n_molecule_level + j] * n_molecules +
                                list(range(n_atoms + i * n_molecules, n_atoms + (i + 1) * n_molecules)))
            rxn_bonds.extend([bond2label[f'RL_{i, j}']] * (2 * n_molecules))

    sender = sender + mol_sender + rxn_sender
    reciever = reciever + mol_reciever + rxn_reciever
    bonds = bonds + mol_bonds + rxn_bonds
    nodes = nodes + [node2label['EMPTY']] * (pad_length - len(nodes))
    if self_bond:
        sender = sender + list(range(len(nodes)))
        reciever = reciever + list(range(len(nodes)))
        bonds = bonds + [bond2label['SELF']] * len(nodes)

    g = dgl.DGLGraph()
    g.add_nodes(len(nodes))

    if len(feature_idxs) > 0:
        features = deepcopy(rxn['reactants']['features'])
        features += 1
        pad_features = np.pad(features[feature_idxs], ((0, 0), (0, len(nodes) - features.shape[1])))
        g.ndata['feats'] = torch.from_numpy(np.r_[np.array([nodes]), pad_features].T).to(device)
    else:
        g.ndata['feats'] = torch.from_numpy(np.array([nodes]).T).to(device)
    g.add_edges(np.array(sender), np.array(reciever))
    g.edata['rel_type'] = torch.from_numpy(np.array(bonds)).to(device)
    return g

