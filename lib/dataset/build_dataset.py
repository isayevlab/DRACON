import numpy as np

from tqdm import tqdm
from lib.dataset.reaction import Reaction
from lib.dataset.molecule import Molecule
from lib.dataset.utils import get_molecule_lengths
from collections import OrderedDict


def get_center_target(rxn):
    r_sender, r_reciever, r_bonds = rxn.reactants.get_senders_recievers_types()
    r_sender, r_reciever, r_bonds = list(r_sender), list(r_reciever), list(r_bonds)
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


def build_dataset(initial_dataset, atom_labels=True):
    columns = initial_dataset.columns
    dataset = {}
    for idx in tqdm(initial_dataset.index):
        data = initial_dataset.loc[idx]
        smarts = data['smarts'].split('|')[0]
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
        dataset[idx] = {'product': product,
                        'reactants': reactants}
        for name in columns:
            dataset[idx][name] = data[name]
        if atom_labels:
            target_main_product = rxn.get_product_mask()
            target_center = get_center_target(rxn)
            dataset[idx]['target_main_product'] = target_main_product
            dataset[idx]['target_center'] = target_center
    return dataset


def get_meta(datasets):
    meta = {'type': set(),
            'node': set(),
            'features': OrderedDict()}
    feature_names = Molecule.get_node_features_names()
    for name in feature_names:
        meta['features'][name] = set()
    for dataset in datasets:
        for idx in tqdm(dataset):
            for part in ['reactants', 'product']:
                meta['node'].update(list(dataset[idx][part]['nodes']))
                meta['type'].update(list(dataset[idx][part]['types']))
                for name, feature in zip(feature_names, dataset[idx][part]['features']):
                    meta['features'][name].update(list(feature))
    meta['type'] = list(meta['type'])
    meta['node'] = list(meta['node'])
    for name in feature_names:
        meta['features'][name] = list(meta['features'][name])
    meta['features_min'] = {feature: min(meta['features'][feature]) for feature in meta['features']}
    for feature in meta['features']:
        norm_values = [i - meta['features_min'][feature] for i in meta['features'][feature]]
        meta['features'][feature] = norm_values
    return meta
