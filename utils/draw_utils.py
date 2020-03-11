import numpy as np
import re

from rdkit import Chem, Geometry
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
from rdkit.Chem.Draw import rdMolDraw2D


def draw_molecule(m, node2color, dpa):
    mol = Chem.Mol(m.ToBinary())
    rdDepictor.Compute2DCoords(mol, bondLength=-1.0)
    coords = mol.GetConformer(-1).GetPositions()
    min_p = Geometry.Point2D(*coords.min(0)[:2] - 1)
    max_p = Geometry.Point2D(*coords.max(0)[:2] + 1)
    w = int(dpa * (max_p.x - min_p.x)) + 1
    h = int(dpa * (max_p.y - min_p.y)) + 1
    Chem.Kekulize(mol)

    mcs_bonds = []
    b_col = []
    for i in range(0, len(mol.GetBonds())):
        bond = mol.GetBondWithIdx(i)
        beg_h, end_h = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

        if beg_h in node2color and end_h in node2color:
            mcs_bonds.append(i)
            b_col.append(max(node2color[beg_h], node2color[end_h]))

    b_col = dict(zip(mcs_bonds, b_col))

    drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
    drawer.SetScale(w, h, min_p, max_p)
    drawer.DrawMolecule(mol, highlightAtoms=list(node2color.keys()), highlightAtomColors=node2color, highlightBonds=mcs_bonds,
                        highlightBondColors=b_col, )
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    return png


def get_react_prouct(smarts, mapping=False):
    if not mapping:
        smarts = re.sub(r':[0-9]+', '', smarts)
    smarts = smarts.split('>')
    if len(smarts) == 3 and len(smarts[1]) > 0:
        reactant = smarts[0] + '.' + smarts[1]
    else:
        reactant = smarts[0]
    product = smarts[-1]
    r_mol = Chem.MolFromSmiles(reactant)
    p_mol = Chem.MolFromSmiles(product)
    return r_mol, p_mol


def get_target_product(data):
    centers = data['target_center']
    prod_mask = data['product']['mask']
    reac_mask = data['reactants']['mask']
    target = np.zeros_like(prod_mask)
    idxs = np.where(centers == 1)[0]
    maps = [reac_mask[i] for i in idxs]
    target[np.isin(prod_mask, maps)] = 1
    return target


def get_molecule_svg(mol, dpa=100, target=None, target_type=None, gt_colors=None):
    if target is None:
        png = draw_molecule(mol, {}, dpa)
    elif target_type == 'GT':
        atom2color = {}
        for j, i in enumerate(target):
            if i in gt_colors:
                atom2color[j] = gt_colors[i]
        png = draw_molecule(mol, atom2color, dpa)
    else:
        atom2color = {j: (i, 1-i, 1-i) for (j, i) in enumerate(target)}
        png = draw_molecule(mol, atom2color, dpa)
    return png


def draw_gt_reaction(data):
    r_mol, p_mol = get_react_prouct(data['smarts'])
    rdDepictor.Compute2DCoords(r_mol)
    rdDepictor.Compute2DCoords(p_mol)
    target_reactants = data['target_main_product'] + data['target_center']
    target_reactants[target_reactants == -1] = 0
    target_product = get_target_product(data) + 1
    r_png = get_molecule_svg(r_mol, target=target_reactants, target_type='GT',
                             gt_colors={1: (0.7, 1, 0.7), 2: (1, 0.7, 0.7)}, dpa=100)
    p_png = get_molecule_svg(p_mol, target=target_product, target_type='GT',
                             gt_colors={1: (0.7, 1, 0.7), 2: (1, 0.7, 0.7)}, dpa=100)
    return p_png, r_png

