import json
import os
import os.path as osp
import random
import re
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.rdmolops import RemoveHs
from sklearn.metrics import pairwise_distances
from torch import Tensor
from torch_geometric.data import InMemoryDataset, Data, Dataset
from torch_geometric.loader import DataLoader
from torch_sparse import SparseTensor
from tqdm import tqdm

from datasets.featurizer import mol_to_egeognn_graph_data, mask_egeognn_graph
from datasets.utils import MoleculePositionToolKit


class EgeognnPretrainedDataset(Dataset):
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
            mpi=False
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

        input_file = os.path.join(self.base_path, f"{self.dataset_name}.smi")
        with open(input_file) as f:
            smiles_list = f.readlines()

        self.mol_list = []

        for smiles in smiles_list:
            if "." in smiles:
                continue
            tmp_mol = Chem.MolFromSmiles(smiles.strip())
            try:
                mol = RemoveHs(tmp_mol) if self.remove_hs else tmp_mol
                if mol.GetNumBonds() < 1:
                    continue
                self.mol_list.append(mol)
            except Exception:
                continue

        random.shuffle(self.mol_list)

        super().__init__(self.folder, transform, pre_transform)

    @property
    def raw_file_names(self):
        # no error, since download function will not download anything
        return "graph.smi.gz"

    @property
    def processed_file_names(self):
        return [f"egeognn_{self.dataset_name}_processed_{i}.pt" for i in range(len(self.mol_list))]

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
            mol, atom_poses = MoleculePositionToolKit.get_MMFF_atom_poses(mol, numConfs=3, numThreads=1)
        else:
            atom_poses = MoleculePositionToolKit.get_2d_atom_poses(mol)

        return mol_to_egeognn_graph_data(
            mol=mol,
            atom_names=self.atom_names,
            bond_names=self.bond_names,
            atom_poses=atom_poses
        )

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[0]))
        return data


    def process_molecule(self, mol):
        if self.with_provided_3d:
            graph = self.mol_to_egeognn_graph_data_raw3d(mol)
        else:
            graph = self.mol_to_egeognn_graph_data_MMFF3d(mol)

        graph = self.prepare_pretrain_task(graph)

        data = CustomData()
        data.AtomBondGraph_edges = torch.from_numpy(graph["edges"].T).to(torch.int64)
        data.BondAngleGraph_edges = torch.from_numpy(graph["BondAngleGraph_edges"].T).to(torch.int64)
        data.AngleDihedralGraph_edges = torch.from_numpy(graph["AngleDihedralGraph_edges"].T).to(torch.int64)

        data.node_feat = torch.from_numpy(graph["node_feat"]).to(torch.int64)
        data.edge_attr = torch.from_numpy(graph["edge_attr"]).to(torch.int64)

        data.bond_lengths = torch.from_numpy(graph["bond_length"]).to(torch.float32)
        data.bond_angles = torch.from_numpy(graph["bond_angle"]).to(torch.float32)
        data.dihedral_angles = torch.from_numpy(graph["dihedral_angle"]).to(torch.float32)

        data.atom_poses = torch.from_numpy(graph["atom_pos"]).to(torch.float32)
        data.n_atoms = graph["num_nodes"]
        data.n_bonds = graph["num_edges"]
        data.n_angles = graph["num_angles"]

        data.masked_atom_indices = torch.from_numpy(graph["masked_atom_indices"]).to(torch.int64)
        data.masked_bond_indices = torch.from_numpy(graph["masked_bond_indices"]).to(torch.int64)
        data.masked_angle_indices = torch.from_numpy(graph["masked_angle_indices"]).to(torch.int64)
        data.masked_dihedral_indices = torch.from_numpy(graph["masked_dihedral_indices"]).to(torch.int64)

        return data

    def process_egeognn(self):
        # print(f"{os.cpu_count()} cpus to be used")
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(self.process_molecule, self.mol_list), total=len(self.mol_list)))

        for idx, data in enumerate(results):
            torch.save(data, os.path.join(self.processed_dir, self.processed_file_names[idx]))

        train_num = int(len(self.mol_list) * 0.8)
        valid_num = int((len(self.mol_list) - train_num) / 2)

        train_idx = range(train_num)
        valid_idx = range(train_num, train_num + valid_num)
        test_idx = range(train_num + valid_num, len(self.mol_list))

        os.makedirs(os.path.join(self.root, "split"), exist_ok=True)
        torch.save(
            {
                "train": torch.tensor(train_idx, dtype=torch.long),
                "valid": torch.tensor(valid_idx, dtype=torch.long),
                "test": torch.tensor(test_idx, dtype=torch.long),
            },
            os.path.join(self.root, "split", "split_dict.pt"),
        )


class EgeognnFinetuneDataset(InMemoryDataset):
    def __init__(
            self,
            root,
            base_path,
            dataset_name,
            atom_names,
            bond_names,
            endpoints,
            remove_hs=False,
            transform=None,
            pre_transform=None,
            dev=False
    ):
        assert dev in [True, False]

        if endpoints is None:
            endpoints = []

        if remove_hs:
            self.folder = os.path.join(root, f"egeognn_{dataset_name}_rh")
        else:
            self.folder = os.path.join(root, f"egeognn_{dataset_name}")

        self.remove_hs = remove_hs
        self.base_path = base_path

        self.atom_names = atom_names
        self.bond_names = bond_names

        self.dev = dev
        self.endpoints = endpoints

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # no error, since download function will not download anything
        return "graph.smi.gz"

    @property
    def processed_file_names(self):
        return f"egeognn_{self.dataset_name}_finetune_processed.pt"

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

    def mol_to_egeognn_graph_data_raw3d(self, mol):
        atom_poses = MoleculePositionToolKit.get_atom_poses(mol)
        return mol_to_egeognn_graph_data(
            mol=mol,
            atom_names=self.atom_names,
            bond_names=self.bond_names,
            atom_poses=atom_poses
        )

    def mol_to_egeognn_graph_data_MMFF3d(self, mol, numConfs=3):
        if len(mol.GetAtoms()) <= 400:
            mol, atom_poses = MoleculePositionToolKit.get_MMFF_atom_poses(mol, numConfs=numConfs, numThreads=0)
        else:
            atom_poses = MoleculePositionToolKit.get_2d_atom_poses(mol)

        return mol_to_egeognn_graph_data(
            mol=mol,
            atom_names=self.atom_names,
            bond_names=self.bond_names,
            atom_poses=atom_poses
        )

    def process_egeognn(self):
        train_idx = []
        valid_idx = []
        test_idx = []
        data_list = []

        valid_conformation = 0

        for endpoint in self.endpoints:
            input_file = os.path.join(self.base_path, f"{endpoint}.csv")
            df = pd.read_csv(input_file, nrows=[10, None][self.dev])

            mol_list = df['smiles'].map(lambda x: Chem.MolFromSmiles(x)).values
            labels = df[endpoint].values

            for i, mol in tqdm(enumerate(mol_list), total=len(mol_list)):
                if self.remove_hs:
                    try:
                        mol = RemoveHs(mol)
                    except Exception:
                        continue

                if "." in Chem.MolToSmiles(mol):
                    continue
                if mol.GetNumBonds() < 1:
                    continue

                graph = self.mol_to_egeognn_graph_data_MMFF3d(mol, numConfs=3)

                data = CustomData()
                data.AtomBondGraph_edges = torch.from_numpy(graph["edges"].T).to(torch.int64)
                data.BondAngleGraph_edges = torch.from_numpy(graph["BondAngleGraph_edges"].T).to(torch.int64)
                data.AngleDihedralGraph_edges = torch.from_numpy(graph["AngleDihedralGraph_edges"].T).to(torch.int64)

                data.node_feat = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph["edge_attr"]).to(torch.int64)
                data.bond_lengths = torch.from_numpy(graph["bond_length"]).to(torch.float32)
                data.bond_angles = torch.from_numpy(graph["bond_angle"]).to(torch.float32)
                data.dihedral_angles = torch.from_numpy(graph["dihedral_angle"]).to(torch.float32)

                data.atom_poses = torch.from_numpy(graph["atom_pos"]).to(torch.float32)
                data.n_atoms = graph["num_nodes"]
                data.n_bonds = graph["num_edges"]
                data.n_angles = graph["num_angles"]

                data.labels = torch.tensor(labels[i]).to(torch.float32)
                data.label_mean = torch.tensor(labels.mean()).to(torch.float32)
                data.label_std = torch.tensor(labels.std()).to(torch.float32)

                data_list.append(data)

                if random.random() < 0.8:
                    train_idx.append(valid_conformation)
                    valid_conformation += 1
                    continue
                if random.random() < 0.5:
                    valid_idx.append(valid_conformation)
                    valid_conformation += 1
                    continue
                test_idx.append(valid_conformation)
                valid_conformation += 1

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
    with open("../config.json") as f:
        configs = json.load(f)

    dataset = EgeognnPretrainedDataset(
        root='../data/demo',
        dataset_name='demo',
        remove_hs=True,
        with_provided_3d=False,
        base_path="../data/demo",
        atom_names=configs["atom_names"],
        bond_names=configs["bond_names"],
        mask_ratio=0.1
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
