import numpy as np

from rdkit import Chem


def get_molecule_lengths(smarts):
    lengths = []
    for mol_smarts in smarts.split('.'):
        mol = Chem.MolFromSmarts(mol_smarts)
        lengths.append(mol.GetNumAtoms())
    return np.cumsum([0] + lengths)


def filter_dataset(dataset, max_len, max_reactants):
    new_dataset = {}
    for idx in dataset:
        r_mask = dataset[idx]['reactants']['mask']
        r_mask = r_mask[r_mask > 0]
        p_mask = dataset[idx]['reactants']['mask']
        p_mask = p_mask[p_mask > 0]
        length = len(dataset[idx]['reactants']['lengths']) - 1
        if (len(dataset[idx]['target_main_product']) <= max_len and
            len(r_mask) == len(np.unique(r_mask)) and
            len(p_mask) == len(np.unique(p_mask)) and
            length < max_reactants):
            new_dataset[idx] = dataset[idx]
    return new_dataset
