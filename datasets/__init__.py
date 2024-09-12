import os
import random
import re

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdmolops import RemoveHs
from sklearn.metrics import pairwise_distances
from torch import Tensor
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from torch_sparse import SparseTensor

from datasets.featurizer import mol_to_egeognn_graph_data, mask_egeognn_graph
from datasets.utils import MoleculePositionToolKit
from tqdm import tqdm


class EgeognnPretrainedDataset(InMemoryDataset):
    def __init__(
            self,
            root,
            base_path,
            dataset_name,
            atom_names,
            bond_names,
            with_provided_3d,
            mask_ratio,
            transform=None,
            pre_transform=None,
            remove_hs=False,
    ):
        if remove_hs:
            self.folder = os.path.join(root, f"egeognn_{dataset_name}_rh")
        else:
            self.folder = os.path.join(root, f"egeognn_{dataset_name}")

        self.dataset_name = dataset_name
        self.remove_hs = remove_hs
        self.base_path = base_path
        self.with_provided_3d = with_provided_3d

        self.atom_names = atom_names
        self.bond_names = bond_names

        self.mask_ratio = mask_ratio

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # no error, since download function will not download anything
        return "graph.smi.gz"

    @property
    def processed_file_names(self):
        return f"egeognn_{self.dataset_name}_processed.pt"

    def download(self):
        if os.path.exists(self.processed_paths[0]):
            return
        else:
            assert os.path.exists(self.base_path)

    def process(self):
        print("Converting pickle files into graphs...")
        self.process_egeognn()

    def get_idx_split(self):
        path = os.path.join(self.root, "split")
        if os.path.isfile(os.path.join(path, "split_dict.pt")):
            return torch.load(os.path.join(path, "split_dict.pt"))

    def prepare_pretrain_task(self, graph, quantum_property=False):
        """
        prepare graph for pretrain task
        """
        n = len(graph['atom_pos'])
        dist_matrix = pairwise_distances(graph['atom_pos'])
        indice = np.repeat(np.arange(n).reshape([-1, 1]), n, axis=1)
        graph['Ad_node_i'] = indice.reshape([-1, 1])
        graph['Ad_node_j'] = indice.T.reshape([-1, 1])
        graph['atom_distance'] = dist_matrix.reshape([-1, 1])

        graph = mask_egeognn_graph(graph, mask_ratio=self.mask_ratio)

        if quantum_property:
            graph['atom_cm5'] = np.array(graph.get('cm5', []))
            graph['atom_espc'] = np.array(graph.get('espc', []))
            graph['atom_hirshfeld'] = np.array(graph.get('hirshfeld', []))
            graph['atom_npa'] = np.array(graph.get('npa', []))
            graph['bo_bond_order'] = np.array(graph.get('bond_order', []))

        return graph

    def mol_to_egeognn_graph_data_raw3d(self, mol):
        atom_poses = MoleculePositionToolKit.get_atom_poses(mol)
        return mol_to_egeognn_graph_data(
            mol=mol,
            atom_names=self.atom_names,
            bond_names=self.bond_names,
            atom_poses=atom_poses
        )

    def mol_to_egeognn_graph_data_MMFF3d(self, mol):
        if len(mol.GetAtoms()) <= 400:
            mol, atom_poses = MoleculePositionToolKit.get_MMFF_atom_poses(mol, numConfs=3, numThreads=0)
        else:
            atom_poses = MoleculePositionToolKit.get_2d_atom_poses(mol)

        return mol_to_egeognn_graph_data(
            mol=mol,
            atom_names=self.atom_names,
            bond_names=self.bond_names,
            atom_poses=atom_poses
        )

    def process_egeognn(self):
        input_file = os.path.join(self.base_path, f"{self.dataset_name}.smi")
        with open(input_file) as f:
            smiles_list = f.readlines()
        mol_list = [Chem.MolFromSmiles(x.strip()) for x in smiles_list]
        mol_list = [mol for mol in mol_list if mol is not None]

        valid_conformation = 0

        train_idx = []
        valid_idx = []
        test_idx = []
        data_list = []

        for mol in tqdm(mol_list):
            if self.remove_hs:
                try:
                    mol = RemoveHs(mol)
                except Exception:
                    continue

            if "." in Chem.MolToSmiles(mol):
                continue
            if mol.GetNumBonds() < 1:
                continue

            if self.with_provided_3d:
                graph = self.mol_to_egeognn_graph_data_raw3d(mol)
            else:
                graph = self.mol_to_egeognn_graph_data_MMFF3d(mol)

            # graph['smiles'] = Chem.MolToSmiles(mol)
            graph = self.prepare_pretrain_task(graph)
            # graph = self.rdk2graph(mol)
            # assert graph["edge_attr"].shape[0] == graph["edges"].shape[0]
            # assert graph["node_feat"].shape[0] == graph["num_nodes"]

            data = CustomData()
            data.AtomBondGraph_edges = torch.from_numpy(graph["edges"].T).to(torch.int64)
            data.BondAngleGraph_edges = torch.from_numpy(graph["BondAngleGraph_edges"].T).to(torch.int64)
            data.AngleDihedralGraph_edges = torch.from_numpy(graph["AngleDihedralGraph_edges"].T).to(torch.int64)

            data.node_feat = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_attr"]).to(torch.int64)
            # data.BondAngleGraph_edge_attr = torch.from_numpy(graph["BondAngleGraph_edge_attr"]).to(torch.float32)
            # data.AngleDihedralGraph_edge_attr = torch.from_numpy(graph["AngleDihedralGraph_edge_attr"]).to(torch.float32)

            # data.Bl_node_i = torch.from_numpy(graph["Bl_node_i"]).to(torch.int64)
            # data.Bl_node_j = torch.from_numpy(graph["Bl_node_j"]).to(torch.int64)
            data.bond_lengths = torch.from_numpy(graph["bond_length"]).to(torch.float32)

            # data.Ba_node_i = torch.from_numpy(graph["Ba_node_i"]).to(torch.int64)
            # data.Ba_node_j = torch.from_numpy(graph["Ba_node_j"]).to(torch.int64)
            # data.Ba_node_k = torch.from_numpy(graph["Ba_node_k"]).to(torch.int64)
            data.bond_angles = torch.from_numpy(graph["bond_angle"]).to(torch.float32)

            # data.Da_node_i = torch.from_numpy(graph["Da_node_i"]).to(torch.int64)
            # data.Da_node_j = torch.from_numpy(graph["Da_node_j"]).to(torch.int64)
            # data.Da_node_k = torch.from_numpy(graph["Da_node_k"]).to(torch.int64)
            # data.Da_node_l = torch.from_numpy(graph["Da_node_l"]).to(torch.int64)
            data.dihedral_angles = torch.from_numpy(graph["dihedral_angle"]).to(torch.float32)

            # data.Ad_node_i = torch.from_numpy(graph["Ad_node_i"]).to(torch.int64)
            # data.Ad_node_j = torch.from_numpy(graph["Ad_node_j"]).to(torch.int64)
            data.atom_distances = torch.from_numpy(graph["atom_distance"]).to(torch.float32)

            data.atom_poses = torch.from_numpy(graph["atom_pos"]).to(torch.float32)
            data.n_atoms = graph["num_nodes"]
            data.n_bonds = graph["num_edges"]
            data.n_angles = graph["num_angles"]
            # data.n_dihedral = graph["num_dihedral"]

            data.masked_atom_indices = torch.from_numpy(graph["masked_atom_indices"]).to(torch.int64)
            data.masked_bond_indices = torch.from_numpy(graph["masked_bond_indices"]).to(torch.int64)
            data.masked_angle_indices = torch.from_numpy(graph["masked_angle_indices"]).to(torch.int64)
            data.masked_dihedral_indices = torch.from_numpy(graph["masked_dihedral_indices"]).to(torch.int64)

            data_list.append(data)
            valid_conformation += 1

            if random.random() < 0.8:
                train_idx.append(valid_conformation)
                continue
            if random.random() < 0.5:
                valid_idx.append(valid_conformation)
                continue
            test_idx.append(valid_conformation)

        graphs, slices = self.collate(data_list)

        print("Saving...")
        torch.save((graphs, slices), self.processed_paths[0])
        os.makedirs(os.path.join(self.root, "split"), exist_ok=True)
        torch.save(
            {
                "train": torch.tensor(train_idx, dtype=torch.long),
                "valid": torch.tensor(valid_idx, dtype=torch.long),
                "test": torch.tensor(test_idx, dtype=torch.long),
            },
            os.path.join(self.root, "split", "split_dict.pt"),
        )


class CustomData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if isinstance(value, SparseTensor):
            return (0, 1)
        elif bool(re.search("(index|face|nei_tgt_mask|edges)", key)):
            return -1
        return 0

    def __inc__(self, key: str, value, *args, **kwargs):
        if 'batch' in key and isinstance(value, Tensor):
            return int(value.max()) + 1
        elif 'AtomBondGraph_edges' in key or key == 'face':
            return self.num_nodes
        elif 'BondAngleGraph_edges' in key:
            return self.AtomBondGraph_edges.size(1)
        elif 'AngleDihedralGraph_edges' in key:
            return self.BondAngleGraph_edges.size(1)
        elif 'masked_atom_indices' in key:
            return self.num_nodes
        elif 'masked_bond_indices' in key:
            return self.num_edges
        elif 'masked_angle_indices' in key:
            return len(self.bond_angles)
        elif 'masked_dihedral_indices' in key:
            return len(self.dihedral_angles)
        else:
            return 0


if __name__ == "__main__":
    import json
    from tqdm import tqdm

    with open("../input_feats.json") as f:
        configs = json.load(f)

    dataset = EgeognnPretrainedDataset(
        root='../data/demo', dataset_name='demo',
        remove_hs=True, base_path="../data/demo",
        atom_names=configs["atom_names"], bond_names=configs["bond_names"],
        with_provided_3d=False, mask_ratio=0.12
    )

    demo_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=10
    )

    pbar = tqdm(demo_loader, desc="Iteration")

    for step, batch in enumerate(pbar):
        ...
