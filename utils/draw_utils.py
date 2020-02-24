import rdkit
import numpy as np
import pickle
import re

from rdkit import Chem, Geometry
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG, Image
IPythonConsole.molSize = (400,400)


def draw_molecules(m, target, colors, dpa=50, output="png", outpath="", outfile=None):
    mol = Chem.Mol(m.ToBinary())
    rdDepictor.Compute2DCoords(mol, bondLength=-1.0)
    coords = mol.GetConformer(-1).GetPositions()
    min_p = Geometry.Point2D(*coords.min(0)[:2] - 1)
    max_p = Geometry.Point2D(*coords.max(0)[:2] + 1)
    w = int(dpa * (max_p.x - min_p.x)) + 1
    h = int(dpa * (max_p.y - min_p.y)) + 1
    Chem.Kekulize(mol)

    a_col = {i: colors[j] for (i, j) in enumerate(target)}
    mcs_bonds = []
    b_col = []
    for i in range(0, len(mol.GetBonds())):
        bond = mol.GetBondWithIdx(i)
        beg_h, end_h = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

        if target[beg_h] != 0 and target[end_h] != 0:
            mcs_bonds.append(i)
            b_col.append(colors[max(target[beg_h], target[end_h])])

    b_col = dict(zip(mcs_bonds, b_col))

    if output.lower() == "png":
        drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
        drawer.SetScale(w, h, min_p, max_p)
        drawer.DrawMolecule(mol, highlightAtoms=range(len(target)), highlightAtomColors=a_col, highlightBonds=mcs_bonds,
                            highlightBondColors=b_col)
        drawer.FinishDrawing()
        png = drawer.GetDrawingText()
        if outfile != None and outfile != "":
            drawer.WriteDrawingText(outpath + outfile + ".png")
        return png
    else:
        drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
        drawer.SetScale(w, h, min_p, max_p)
        drawer.DrawMolecule(mol, highlightAtoms=range(len(target)), highlightAtomColors=a_col, highlightBonds=mcs_bonds,
                            highlightBondColors=b_col)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        if outfile is not None and outfile != "":
            text_file = open(outpath + outfile + ".svg", "w")
            text_file.write(svg)
            text_file.close()
        return svg


def prune_dataset_by_length(dataset, max_len):
    new_dataset = {}
    for idx in dataset:
        if len(dataset[idx]['target_main_product']) <= max_len:
            new_dataset[idx] = dataset[idx]
    return new_dataset


def get_react_prouct(smarts):
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


def draw_reaction(data):
    r_mol, p_mol = get_react_prouct(data['smarts'])
    rdDepictor.Compute2DCoords(r_mol)
    rdDepictor.Compute2DCoords(p_mol)
    target_reactants = data['target_main_product'] + data['target_center']
    target_reactants[target_reactants == -1] = 0
    target_product = get_target_product(data) + 1
    r_svg = draw_molecules(r_mol, target_reactants, [(1, 1, 1), (0.8, 1, 0.8), (0.5, 0.5, 1)], output="svg", dpa=100)
    p_svg = draw_molecules(p_mol, target_product, [(1, 1, 1), (0.8, 1, 0.8), (0.5, 0.5, 1)], output="svg", dpa=100)
    return p_svg, r_svg
