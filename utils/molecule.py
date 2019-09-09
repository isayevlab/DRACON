import numpy as np

from rdkit.Chem import AllChem
from rdkit import Chem

# bond_types = [ 'UNSPECIFIED', 'SINGLE', 'DOUBLE', 'TRIPLE', 'QUADRUPLE', 'QUINTUPLE', 'HEXTUPLE', 'ONEANDAHALF',
#                'TWOANDAHALF', 'THREEANDAHALF', 'FOURANDAHALF', 'FIVEANDAHALF', 'AROMATIC', 'IONIC', 'HYDROGEN',
#                'THREECENTER', 'DATIVEONE', 'DATIVE', 'DATIVEL', 'DATIVER', 'OTHER', 'ZERO']
# bond_decoder = dict(enumerate(bond_types))
# bond_encoder = dict(zip(bond_types, range(len(bond_types))))
#


def matrices2mol(node_labels, edge_labels):
    mol = Chem.RWMol()
    for node_label in node_labels:
        mol.AddAtom(Chem.Atom.SetAtomicNum(node_label))

    for start, end in zip(*np.nonzero(edge_labels)):
        if start > end:
            mol.AddBond(int(start), int(end), edge_labels[start, end])
    return mol


class Molecule:
    def __init__(self, smarts):
        self.rdkit_molecule = Chem.MolFromSmarts(smarts)
        self.rdkit_molecule.UpdatePropertyCache(strict=False)

    def get_senders_recievers_types(self):
        senders = [b.GetBeginAtomIdx() for b in self.rdkit_molecule.GetBonds()]
        receivers = [b.GetEndAtomIdx() for b in self.rdkit_molecule.GetBonds()]
        b_types = [str(b.GetBondType()) for b in self.rdkit_molecule.GetBonds()]
        return (np.array(senders, dtype=np.int32),
                np.array(receivers, dtype=np.int32),
                np.array(b_types, dtype='<U10'))

    def get_node_types(self):
        return np.array([atom.GetAtomicNum() for atom in self.rdkit_molecule.GetAtoms()], dtype=np.int32)

    def get_num_atoms(self):
        return self.rdkit_molecule.GetNumAtoms()

    def get_adjacency_matrix(self):
        senders, receivers, b_types = self.get_senders_recievers_types()
        num_atoms = self.get_num_atoms()
        adjacency_matrix = np.zeros((num_atoms, num_atoms), dtype='<U10')
        adjacency_matrix[senders, receivers] = b_types
        adjacency_matrix[receivers, senders] = b_types
        return adjacency_matrix

    def get_smiles(self):
        return Chem.MolToSmiles(self.rdkit_molecule)

    def get_smarts(self):
        return Chem.MolToSmarts(self.rdkit_molecule)

    def get_node_features(self):
        degrees = [a.GetDegree() for a in self.rdkit_molecule.GetAtoms()]
        exp_valences = [a.GetExplicitValence() for a in self.rdkit_molecule.GetAtoms()]
        hybridization = [int(a.GetHybridization()) for a in self.rdkit_molecule.GetAtoms()]
        imp_valences = [a.GetImplicitValence() for a in self.rdkit_molecule.GetAtoms()]
        is_aromatic = [a.GetIsAromatic() for a in self.rdkit_molecule.GetAtoms()]
        is_not_implicit = [a.GetNoImplicit() for a in self.rdkit_molecule.GetAtoms()]
        num_explicit_hs = [a.GetNumExplicitHs() for a in self.rdkit_molecule.GetAtoms()]
        num_implicit_hs = [a.GetNumImplicitHs() for a in self.rdkit_molecule.GetAtoms()]
        is_ring = [a.IsInRing() for a in self.rdkit_molecule.GetAtoms()]
        num_radical_electrons = [a.GetNumRadicalElectrons() for a in self.rdkit_molecule.GetAtoms()]
        features = [degrees, exp_valences, hybridization, imp_valences, is_aromatic, is_not_implicit,
                    num_explicit_hs, num_implicit_hs, is_ring, num_radical_electrons]
        features = np.stack(features)
        features = features.astype(np.int32)
        return features

    def get_atoms_mapping(self):
        map_num = [atom.GetAtomMapNum() for atom in self.rdkit_molecule.GetAtoms()]
        return np.array(map_num, dtype=np.int32)

    def get_degrees(self):
        return np.count_nonzero(self.get_adjacency_matrix(), -1)


if __name__ == "__main__":
    smarts = '[Br:1][CH2:2][CH2:3][CH2:4][OH:5].[CH3:6][S:7](Cl)(=[O:9])=[O:8].CCOCC.C(N(CC)CC)C'
    mol = Molecule(smarts)
    mol.get_senders_recievers_types()
    mol.get_adjacency_matrix()
    mol.get_smiles()
    mol.get_node_types()
    mol.get_node_features()
    mol.get_degrees()

