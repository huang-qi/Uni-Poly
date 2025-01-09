import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data


def mol2_2Dcoords(mol): 
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    assert len(mol.GetAtoms()) == len(coordinates), f"2D coordinates shape is not aligned with { Chem.MolToSmiles(mol)}"
    return coordinates

def mol2_3Dcoords(mol, cnt):
    coordinate_list = []
    for seed in range(cnt):
        try:
            res = AllChem.EmbedMolecule(mol, randomSeed=seed)
            if res == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                    coordinates = mol.GetConformer().GetPositions()
                except:
                    coordinates = mol2_2Dcoords(mol)
            elif res == -1:
                mol_tmp = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
                AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=seed)
                mol_tmp = Chem.AddHs(mol_tmp, addCoords=True)
                try:
                    AllChem.MMFFOptimizeMolecule(mol_tmp)
                    coordinates = mol_tmp.GetConformer().GetPositions()
                except:
                    coordinates = mol2_2Dcoords(mol)
        except:
            coordinates = mol2_2Dcoords(mol)

        assert len(mol.GetAtoms()) == len(coordinates), f"3D coordinates shape is not aligned with { Chem.MolToSmiles(mol)}"
        coordinate_list.append(coordinates.astype(np.float32))
    return coordinate_list

def mol2coords(mol):
    mol = process_star_atoms(mol)
    cnt = 1  # conformer count: 10 3D + 1 2D
    if len(mol.GetAtoms()) > 400:
        coordinate_list = [mol2_2Dcoords(mol)] * (cnt + 1)
        print("Atom count > 400, using 2D coordinates")
    else:
        coordinate_list = mol2_3Dcoords(mol, cnt)
        # Add 2D conformer
        coordinate_list.append(mol2_2Dcoords(mol).astype(np.float32))

    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]  # After adding H
    positions = coordinate_list[0]  # Use the first set of coordinates

    # Create PyTorch Geometric Data object
    data = Data(
        z=torch.tensor(atomic_numbers, dtype=torch.long),
        pos=torch.tensor(positions, dtype=torch.float),
    )
    return data

def process_star_atoms(mol):
    mol = Chem.AddHs(mol)
    star_idx = []
    for idx, atom in enumerate(mol.GetAtoms()):
        if atom.GetSymbol() == '*':
            star_idx.append(idx)

    # 替换星号为氢原子
    for idx in star_idx[::-1]:
        mol.GetAtomWithIdx(idx).SetAtomicNum(1) 

    return mol

