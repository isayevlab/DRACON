import sys
sys.path.append('..')

import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
from utils.reaction import Reaction
from utils.molecule import Molecule
from rdkit import Chem


def f(x):
    try:
        return x.split('|')[0]
    except Exception:
        return ''


def get_molecule_lengths(smarts):
    lengths = []
    for mol_smarts in smarts.split('.'):
        mol = Chem.MolFromSmarts(mol_smarts)
        lengths.append(mol.GetNumAtoms())
    return np.cumsum([0] + lengths)


def get_center_target(rxn):
    r_sender, r_reciever, r_bonds = rxn.reactants.get_senders_recievers_types()
    r_sender, r_reciever, r_bonds = list(r_sender), list(r_reciever), list(r_bonds)
    r_nodes = rxn.reactants.get_node_types()
    r_features = rxn.reactants.get_node_features()[-2:]

    p_sender, p_reciever, p_bonds = rxn.product.get_senders_recievers_types()
    p_nodes = rxn.product.get_node_types()
    p_features = rxn.product.get_node_features()[-2:]

    r_mask = rxn.reactants.get_atoms_mapping()
    p_mask = rxn.product.get_atoms_mapping()
    p_ids = np.where(r_mask > 0.)[0].flatten()
    num2id = dict(zip(r_mask, range(len(r_mask))))

    r_features = np.array([r_features[:, num2id[p_mask[i]]] for i in range(len(p_nodes))]).T

    p_sender = [num2id[p_mask[i]] for i in p_sender]
    p_reciever = [num2id[p_mask[i]] for i in p_reciever]

    reagents = set(zip(r_sender, r_reciever, r_bonds)) | \
               set(zip(r_reciever, r_sender, r_bonds))

    product = set(zip(p_sender, p_reciever, p_bonds)) | \
              set(zip(p_reciever, p_sender, p_bonds))
    new_reagents = set()
    for r in reagents:
        if (r[0] in p_ids) or (r[1] in p_ids):
            new_reagents.add(r)

    reagents = new_reagents
    interception = (reagents | product) - (reagents & product)
    target = np.ones_like(r_mask) * -1
    target[p_ids] = 0
    centers = []
    if len(interception) > 0:
        s, r, b = zip(*interception)
        centers += list(set(p_ids) & set(s))
    centers += [num2id[p_mask[i]] for i in np.where((p_features != r_features).sum(axis=0) > 0)[0]]
    target[centers] = 1
    return target


def get_smarts(mode='train', max_num_molecules=15):
    if mode == 'train':
        dataset = pd.read_csv('../data/US_patents_1976-Sep2016_1product_reactions_train.csv', header=2, sep='\t')
    elif mode == 'test':
        dataset = pd.read_csv('../data/US_patents_1976-Sep2016_1product_reactions_test.csv', header=2, sep='\t')
    else:
        dataset = pd.read_csv('../data/US_patents_1976-Sep2016_1product_reactions_valid.csv', header=2, sep='\t')
    smarts = dataset['OriginalReaction']
    smiles = dataset['CanonicalizedReaction']
    smarts = smarts.apply(f)
    idxs = []
    new_smarts = []
    new_smiles = []
    for j, (i, k) in enumerate(zip(smarts, smiles)):
        if ('>' in i):
            if (len(i.split('>')[0].split('.')) + len(i.split('>')[1].split('.'))) < 15:
                new_smarts.append(i)
                new_smiles.append(k)
                idxs.append(j)
    return list(zip(idxs, new_smarts, new_smiles))


def build_dataset(data):
    dataset = {}
    for idx, smarts, smiles in tqdm(data):
        product = {}
        reactants = {}
        rxn = Reaction(smarts)
        lengths = get_molecule_lengths(rxn.reactants.get_smarts())
        reactants['lengths'] = lengths
        sender, reciever, types = rxn.reactants.get_senders_recievers_types()
        reactants['sender'], reactants['reciever'], reactants['types'] = sender, reciever, types
        nodes = rxn.reactants.get_node_types()
        reactants['nodes'] = nodes
        react_features = rxn.reactants.get_node_features()
        reactants['features'] = react_features
        reactants['mask'] = rxn.reactants.get_atoms_mapping()

        sender, reciever, types = rxn.product.get_senders_recievers_types()
        product['sender'], product['reciever'], product['types'] = sender, reciever, types
        nodes = rxn.product.get_node_types()
        product['nodes'] = nodes
        prod_features = rxn.product.get_node_features()
        product['features'] = prod_features
        product['mask'] = rxn.product.get_atoms_mapping()


        target_main_product = rxn.get_product_mask()
        target_center = get_center_target(rxn)
        if len(product['mask']) == len(np.unique(product['mask'])):
            dataset[idx] = {'product': product,
                            'reactants': reactants,
                            'target_main_product': target_main_product,
                            'target_center': target_center,
                            'smarts': smarts,
                            'smiles': smiles}
    return dataset


def get_meta(datasets, faeture_names):
    meta = {'type': set(),
            'node': set(),
            'features': {}}
    for name in faeture_names:
        meta['features'][name] = set()
    for dataset in datasets:
        for idx in tqdm(dataset):
            for part in ['reactants', 'product']:
                meta['node'].update(list(dataset[idx][part]['nodes']))
                meta['type'].update(list(dataset[idx][part]['types']))
                for name, feature in zip(faeture_names, dataset[idx][part]['features']):
                    meta['features'][name].update(list(feature))
    meta['type'] = list(meta['type'])
    meta['node'] = list(meta['node'])
    for name in faeture_names:
        meta['features'][name] = list(meta['features'][name])
    return meta


def prune_dataset_by_length(dataset, max_len):
    new_dataset = {}
    for idx in dataset:
        if len(dataset[idx]['target_main_product']) <= max_len:
            new_dataset[idx] = dataset[idx]
    return new_dataset


if __name__ == '__main__':
    train = get_smarts()
    test = get_smarts(mode='test')
    valid = get_smarts(mode='valid')
    feature_names = Molecule.get_node_features_name()

    test_dataset = build_dataset(test)
    with open('../data/graphs/test.pkl', 'wb') as f:
        pickle.dump(test_dataset, f)

    train_dataset = build_dataset(train)
    with open('../data/graphs/train.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)

    valid_dataset = build_dataset(valid)
    with open('../data/graphs/valid.pkl', 'wb') as f:
        pickle.dump(valid_dataset, f)

    with open('../data/graphs/meta.pkl', 'wb') as f:
        pickle.dump(get_meta([train_dataset, test_dataset, valid_dataset], feature_names), f)
