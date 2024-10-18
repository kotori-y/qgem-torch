import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


class MoleculePositionToolKit:
    @staticmethod
    def get_atom_poses(mol, conf_idx=0):
        """tbd"""
        return mol.GetConformer(int(conf_idx)).GetPositions()

    @staticmethod
    def get_MMFF_atom_poses(mol, numConfs=10, numThreads=1):
        """the atoms of mol will be changed in some cases."""
        try:
            new_mol = Chem.AddHs(mol)
            res = AllChem.EmbedMultipleConfs(new_mol, numConfs=numConfs, numThreads=numThreads)
            res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
            res = [[x[0], x[1] if not x[0] else np.Inf] for x in res]
            index = np.argmin([x[1] for x in res])
        except:
            new_mol = mol
            AllChem.Compute2DCoords(new_mol)
            index = 0

        new_mol = Chem.RemoveHs(new_mol)
        return new_mol, MoleculePositionToolKit.get_atom_poses(new_mol, conf_idx=index)

    @staticmethod
    def get_2d_atom_poses(mol):
        """get 2d atom poses"""
        AllChem.Compute2DCoords(mol)
        return MoleculePositionToolKit.get_atom_poses(mol)


if __name__ == "__main__":
    smiles = "NS(=O)(=O)c1cc2c(cc1Cl)NC(CSCc1ccccc1)=NS2(=O)=O"
    mol = Chem.MolFromSmiles(smiles)

    mmff_poses = MoleculePositionToolKit.get_MMFF_atom_poses(mol, numConfs=5)
    poses = MoleculePositionToolKit.get_2d_atom_poses(mol)

    print(mmff_poses, poses)
