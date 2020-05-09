import numpy as np

from lib.dataset.molecule import Molecule


class Reaction:
    def __init__(self, smarts, drop_hs=True):
        reactants, agents, product = smarts.split('>')
        if agents != '':
            reactants = reactants + '.' + agents
        self.reactants = Molecule(reactants, drop_hs=drop_hs)
        self.product = Molecule(product, drop_hs=drop_hs)

    def get_product_mask(self):
        return np.int32(self.reactants.get_atoms_mapping() > 0)

    def get_reactants_adjacency_matrix(self):
        return self.reactants.get_adjacency_matrix()

    def get_reactants_node_types(self):
        return self.reactants.get_node_types()

    def get_reactants_node_features(self):
        return self.reactants.get_node_features()

    def get_product_adjacency_matrix(self):
        product_adjacency_matrix = self.product.get_adjacency_matrix()
        reactants_atom_mapping = self.reactants.get_atoms_mapping()
        product_atom_mapping = self.product.get_atoms_mapping()
        reactants_num_atoms = self.reactants.get_num_atoms()
        extended_product_matrix = np.zeros((reactants_num_atoms, reactants_num_atoms), dtype=str)
        indices = np.where(np.in1d(reactants_atom_mapping, product_atom_mapping))[0]
        for start, end in zip(*np.nonzero(product_adjacency_matrix)):
            extended_product_matrix[indices[start], indices[end]] = product_adjacency_matrix[start, end]
        return extended_product_matrix

    def get_product_node_types(self):
        reactants_atom_mapping = self.reactants.get_atoms_mapping()
        product_atom_mapping = self.product.get_atoms_mapping()
        reactants_num_atoms = self.reactants.get_num_atoms()
        product_node_types = self.product.get_node_types()
        indices = np.where(np.in1d(reactants_atom_mapping, product_atom_mapping))[0]
        extended_node_types = np.zeros(reactants_num_atoms, dtype=np.int32)
        for idx, node_type in enumerate(product_node_types):
            extended_node_types[indices[idx]] = node_type
        return extended_node_types


if __name__ == '__main__':
    smarts = '[Br:1][CH2:2][CH2:3][CH2:4][OH:5].[CH3:6][S:7](Cl)(=[O:9])=[O:8].CCOCC>C(N(CC)CC)C>[CH3:6][S:7]([O:5]' \
             '[CH2:4][CH2:3][CH2:2][Br:1])(=[O:9])=[O:8]'
    rxn = Reaction(smarts)
    rxn.get_product_mask()
    rxn.get_product_adjacency_matrix()
    rxn.get_product_node_types()
    rxn.get_reactants_adjacency_matrix()
    rxn.get_reactants_node_features()
    rxn.get_reactants_node_types()
