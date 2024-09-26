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
            new_mol = Chem.AddHs(mol, addCoords=True)
            AllChem.EmbedMultipleConfs(
                new_mol, numConfs=numConfs,
                numThreads=numThreads, maxAttempts=100,
                clearConfs=True, useExpTorsionAnglePrefs=True,
                useBasicKnowledge=True, useSmallRingTorsions=True,
            )
            prop = AllChem.MMFFGetMoleculeProperties(new_mol, mmffVariant="MMFF94s")
            ff = AllChem.MMFFGetMoleculeForceField(new_mol, prop, confId=0)
            ff.Minimize()
        except:
            new_mol = mol
            AllChem.Compute2DCoords(new_mol)
            index = 0

        new_mol = Chem.RemoveHs(new_mol)
        return new_mol, MoleculePositionToolKit.get_atom_poses(new_mol)

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
