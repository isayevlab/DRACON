import dgl
import torch
import numpy as np

from rdkit import Chem


def get_molecule_lengths(smarts):
    lengths = []
    for mol_smarts in smarts.split('.'):
        mol = Chem.MolFromSmarts(mol_smarts)
        lengths.append(mol.GetNumAtoms())
    return np.cumsum([0] + lengths)


def get_norm(sender, reciever, encoded_types, num_atoms, num_bonds):
    adj_norm = np.zeros((num_atoms, num_bonds), dtype=np.float32)
    for rec, b_type in zip(reciever, encoded_types):
        adj_norm[rec, b_type] += 1
    for rec, b_type in zip(sender, encoded_types):
        adj_norm[rec, b_type] += 1
    adj_norm[adj_norm != 0] = 1./adj_norm[adj_norm != 0]
    return adj_norm


def label2onehot(labels, dim, encoder=None):
    if encoder is not None:
        labels = [encoder(label) for label in labels]
    return np.eye(dim, dtype=np.int32)[labels]


def load_graph_data(rxn, bond2label, node2label):
    sender, reciever, types = rxn.reactants.get_senders_recievers_types()
    encoded_types = [bond2label[t] for t in types]
    nodes = rxn.reactants.get_node_types()
    encoded_nodes = [node2label[node] for node in nodes]
    norm = get_norm(sender, reciever, encoded_types, rxn.reactants.get_num_atoms(), len(bond2label))
    sender = np.r_[sender, reciever]
    reciever = np.r_[reciever, sender[:len(reciever)]]
    encoded_types = encoded_types + encoded_types
    idxs = np.c_[[reciever, encoded_types]]
    norm = norm[idxs[0], idxs[1]]
    return encoded_nodes, list(sender), list(reciever), encoded_types, list(norm)


def get_bonds(datasets, n_molecule_level=1, n_reaction_level=1,  self_bond=True):
    bond_types = []
    for dataset in datasets:
        bond_types.extend(list(dataset.reactants_bond_types))
    bond_types = list(set(bond_types))
    bond2label = dict(zip(bond_types, range(len(bond_types))))
    if self_bond:
        bond2label['SELF'] = len(bond2label)
    for i in range(n_molecule_level):
        bond2label[f'ML_{i}'] = len(bond2label)
    for i in range(n_molecule_level):
        for j in range(n_reaction_level):
            bond2label[f'RL_{i, j}'] = len(bond2label)
    return bond2label


def get_nodes(datasets, n_molecule_level=1, n_reaction_level=1):
    node_types = []
    for dataset in datasets:
        node_types.extend(list(dataset.reactants_node_types))
    node_types = list(set(node_types))
    node2label = dict(zip(node_types, range(len(node_types))))
    node2label['EMPTY'] = len(node2label)

    for i in range(n_molecule_level):
        node2label[f'ML_{i}'] = len(node2label)

    for i in range(n_molecule_level):
        for j in range(n_reaction_level):
            node2label[f'RL_{i, j}'] = len(node2label)
    return node2label


def get_graph(dataset,
              idx,
              bond2label,
              node2label,
              n_molecule_level=1,
              n_reaction_level=1,
              self_bond=True,
              pad_length=120,
              device='cuda'
              ):
    rxn = dataset.dataset[idx]
    nodes, sender, reciever, bonds, norm = load_graph_data(rxn, bond2label, node2label)
    n_atoms = len(nodes)
    molecules_lengths = get_molecule_lengths(rxn.reactants.get_smarts())
    n_molecules = len(molecules_lengths) - 1

    for i in range(n_molecule_level):
        nodes.extend([node2label[f'ML_{i}']] * n_molecules)

    for i in range(n_molecule_level):
        for j in range(n_reaction_level):
            nodes.append(node2label[f'RL_{i, j}'])

    mol_sender = []
    mol_reciever = []
    mol_bonds = []
    mol_norms = []

    for i in range(n_molecule_level):
        for j, (st, fi) in enumerate(zip(molecules_lengths[:-1], molecules_lengths[1:])):
            rg = list(range(st, fi))
            mol_sender.extend(list(range(st, fi)) + [n_atoms + i * n_molecules + j] * (fi - st))
            mol_reciever.extend([n_atoms + i * n_molecules + j] * (fi - st) + list(range(st, fi)))
            mol_bonds.extend([bond2label[f'ML_{i}']] * (2 * (fi - st)))
            mol_norms.extend([1. / (fi - st)] * (fi - st) + [1.] * (fi - st))

    rxn_sender = []
    rxn_reciever = []
    rxn_bonds = []
    rxn_norms = []

    for i in range(n_molecule_level):
        for j in range(n_reaction_level):
            rxn_sender.extend(list(range(n_atoms + i * n_molecules, n_atoms + (i + 1) * n_molecules)) +
                              [n_atoms + n_molecule_level * n_molecules + i * n_molecule_level + j] * n_molecules)
            rxn_reciever.extend([n_atoms + n_molecule_level * n_molecules + i * n_molecule_level + j] * n_molecules +
                                list(range(n_atoms + i * n_molecules, n_atoms + (i + 1) * n_molecules)))
            rxn_bonds.extend([bond2label[f'RL_{i, j}']] * (2 * n_molecules))
            rxn_norms.extend([1. / n_molecules] * n_molecules + [1.] * n_molecules)

    norm = norm + mol_norms + rxn_norms
    sender = sender + mol_sender + rxn_sender
    reciever = reciever + mol_reciever + rxn_reciever
    bonds = bonds + mol_bonds + rxn_bonds
    nodes = nodes + [node2label['EMPTY']] * (pad_length - len(nodes))
    if self_bond:
        sender = sender + list(range(len(nodes)))
        reciever = reciever + list(range(len(nodes)))
        norm = norm + [1.] * len(nodes)
        bonds = bonds + [bond2label['SELF']] * len(nodes)

    g = dgl.DGLGraph()
    g.add_nodes(len(nodes))
    g.ndata['id'] = torch.from_numpy(np.array(nodes)).to(device)
    g.add_edges(np.array(sender), np.array(reciever))
    g.edata['norm'] = torch.from_numpy(np.array(norm).reshape(-1, 1)).to(device).float()
    g.edata['rel_type'] = torch.from_numpy(np.array(bonds)).to(device)
    target = list(dataset.dataset[idx].get_product_mask())
    target = target + [-1] * (len(nodes) - len(target))
    return g, target
