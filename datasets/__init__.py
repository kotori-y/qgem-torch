import os
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
            mask_value,
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
        self.mask_value = mask_value

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

    def prepare_pretrain_task(self, graph):
        """
        prepare graph for pretrain task
        """
        n = len(graph['atom_pos'])
        dist_matrix = pairwise_distances(graph['atom_pos'])
        indice = np.repeat(np.arange(n).reshape([-1, 1]), n, axis=1)
        graph['Ad_node_i'] = indice.reshape([-1, 1])
        graph['Ad_node_j'] = indice.T.reshape([-1, 1])
        graph['atom_distance'] = dist_matrix.reshape([-1, 1])

        graph = mask_egeognn_graph(graph, mask_ratio=self.mask_ratio, mask_value=self.mask_value)
        #
        # graph['atom_cm5'] = np.array(graph.get('cm5', []))
        # graph['atom_espc'] = np.array(graph.get('espc', []))
        # graph['atom_hirshfeld'] = np.array(graph.get('hirshfeld', []))
        # graph['atom_npa'] = np.array(graph.get('npa', []))
        # graph['bo_bond_order'] = np.array(graph.get('bond_order', []))

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

            graph['smiles'] = Chem.MolToSmiles(mol)
            graph = self.prepare_pretrain_task(graph)
            # graph = self.rdk2graph(mol)
            # assert graph["edge_attr"].shape[0] == graph["edges"].shape[0]
            # assert graph["node_feat"].shape[0] == graph["num_nodes"]

            data = CustomData()
            data.edges = torch.from_numpy(graph["edges"]).to(torch.int64)
            data.BondAngleGraph_edges = torch.from_numpy(graph["BondAngleGraph_edges"]).to(torch.int64)
            data.AngleDihedralGraph_edges = torch.from_numpy(graph["AngleDihedralGraph_edges"]).to(torch.int64)

            data.node_feat = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_attr"]).to(torch.int64)
            data.BondAngleGraph_edge_attr = torch.from_numpy(graph["BondAngleGraph_edge_attr"]).to(torch.float32)
            data.AngleDihedralGraph_edge_attr = torch.from_numpy(graph["AngleDihedralGraph_edge_attr"]).to(torch.float32)

            data.Bl_node_i = torch.from_numpy(graph["Bl_node_i"]).to(torch.int64)
            data.Bl_node_j = torch.from_numpy(graph["Bl_node_j"]).to(torch.int64)
            data.bond_length = torch.from_numpy(graph["bond_length"]).to(torch.float32)

            data.Ba_node_i = torch.from_numpy(graph["Ba_node_i"]).to(torch.int64)
            data.Ba_node_j = torch.from_numpy(graph["Ba_node_j"]).to(torch.int64)
            data.Ba_node_k = torch.from_numpy(graph["Ba_node_k"]).to(torch.int64)
            data.bond_angle = torch.from_numpy(graph["bond_angle"]).to(torch.float32)

            data.Da_node_i = torch.from_numpy(graph["Da_node_i"]).to(torch.int64)
            data.Da_node_j = torch.from_numpy(graph["Da_node_j"]).to(torch.int64)
            data.Da_node_k = torch.from_numpy(graph["Da_node_k"]).to(torch.int64)
            data.Da_node_l = torch.from_numpy(graph["Da_node_l"]).to(torch.int64)
            data.dihedral_angle = torch.from_numpy(graph["dihedral_angle"]).to(torch.float32)

            data.Ad_node_i = torch.from_numpy(graph["Ad_node_i"]).to(torch.int64)
            data.Ad_node_j = torch.from_numpy(graph["Ad_node_j"]).to(torch.int64)
            data.atom_distance = torch.from_numpy(graph["atom_distance"]).to(torch.float32)

            data_list.append(data)

        graphs, slices = self.collate(data_list)

        print("Saving...")
        torch.save((graphs, slices), self.processed_paths[0])


class CustomData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if isinstance(value, SparseTensor):
            return (0, 1)
        elif bool(re.search("(index|face|nei_tgt_mask)", key)):
            return -1
        return 0

    def __inc__(self, key: str, value, *args, **kwargs):
        if 'batch' in key and isinstance(value, Tensor):
            return int(value.max()) + 1
        elif 'edge_index' in key or key == 'face':
            return self.num_nodes
        elif 'angle_index' in key:
            return self.edge_index.size(1)
        elif 'dihedral_index' in key:
            return self.angle_index.size(1)
        else:
            return 0


if __name__ == "__main__":
    import json
    from tqdm import tqdm

    with open("../configs/input_feats.json") as f:
        configs = json.load(f)

    dataset = EgeognnPretrainedDataset(
        root='../data/demo', dataset_name='demo',
        remove_hs=True, base_path="../data/demo",
        atom_names=configs["atom_names"], bond_names=configs["bond_names"],
        with_provided_3d=False, mask_value=-2, mask_ratio=0.12
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

    print()
