import numpy as np

from utils.reaction import Reaction


class Dataset:
    def __init__(self, smarts_list, max_num_atoms=100, min_num_atoms=10):
        self.dataset = []
        for smarts in smarts_list:
            rxn = Reaction(smarts)
            num_atoms = rxn.reactants.get_num_atoms()
            if (num_atoms < max_num_atoms) and (num_atoms > min_num_atoms):
                self.dataset.append(rxn)
        self.products_node_types = self.calculate_unique_values('product', 'get_node_types')
        self.products_bond_types = self.calculate_unique_values('product', 'get_adjacency_matrix')
        self.reactants_node_types = self.calculate_unique_values('reactants', 'get_node_types')
        self.reactants_bond_types = self.calculate_unique_values('reactants', 'get_adjacency_matrix')
        self.reactants_features = self.calculate_unique_values('reactants', 'get_node_features')

    def calculate_unique_values(self, reaction_part, field):
        values = None
        for rxn in self.dataset:
            value = np.unique(getattr(getattr(rxn, reaction_part), field)())
            if values:
                values = np.unique(np.c_[values, value])
            else:
                values = value
        return values


if __name__ == '__main__':
    smarts = '[Br:1][CH2:2][CH2:3][CH2:4][OH:5].[CH3:6][S:7](Cl)(=[O:9])=[O:8].CCOCC>C(N(CC)CC)C>[CH3:6][S:7]([O:5]' \
             '[CH2:4][CH2:3][CH2:2][Br:1])(=[O:9])=[O:8]'
    rxn = Reaction(smarts)
    dataset = Dataset([smarts])
    print(dataset.reactants_features)
