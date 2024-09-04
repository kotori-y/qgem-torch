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

        mask_egeognn_graph(graph, mask_ratio=0.1)
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

            # if "super_edge_index" in graph:
            #     graph.super_edge_index = torch.from_numpy(graph["super_edge_index"]).to(torch.int64)
            # if "supersuper_edge_index" in graph:
            #     graph.supersuper_edge_index = torch.from_numpy(graph["supersuper_edge_index"]).to(torch.int64)
            # if "angle_index" in graph:
            #     graph.angle_index = torch.from_numpy(graph["angle_index"]).to(torch.int64)
            # if "dihedral_index" in graph:
            #     graph.dihedral_index = torch.from_numpy(graph["dihedral_index"]).to(torch.int64)
            #
            # graph.n_nodes = graph["n_nodes"]
            # graph.n_edges = graph["n_edges"]
            # if 'n_angles' in graph:
            #     graph.n_angles = graph["n_angles"]
            # if 'n_dihedrals' in graph:
            #     graph.n_dihedrals = graph["n_dihedrals"]
            #
            # graph.pos = torch.from_numpy(mol.GetConformer(0).GetPositions()).to(torch.float)
            #
            # graph.rd_mol = copy.deepcopy(mol)
            # graph.isomorphisms = isomorphic_core(mol)
            # graph.nei_src_index = torch.from_numpy(graph["nei_src_index"]).to(torch.int64)
            # graph.nei_tgt_index = torch.from_numpy(graph["nei_tgt_index"]).to(torch.int64)
            # graph.nei_tgt_mask = torch.from_numpy(graph["nei_tgt_mask"]).to(torch.bool)
            #
            # if self.dataset_name in ['allsets']:
            #     if "test" in subset:
            #         test_idx.append(valid_conformation)
            #     else:
            #         train_idx.append(valid_conformation)
            #
            # else:
            #     if "train" in subset:
            #         train_idx.append(valid_conformation)
            #     elif "val" in subset:
            #         valid_idx.append(valid_conformation)
            #     else:
            #         test_idx.append(valid_conformation)
            #
            # valid_conformation += 1
            data_list.append(data)
            #
            # if self.pre_transform is not None:
            #     data_list = [self.pre_transform(graph) for graph in data_list]
            #
        graphs, slices = self.collate(data_list)

        print("Saving...")
        torch.save((graphs, slices), self.processed_paths[0])
        # os.makedirs(os.path.join(self.root, "split"), exist_ok=True)
        # if len(valid_idx) == 0:
        #     valid_idx = train_idx[:6400]
        # if len(test_idx) == 0:
        #     test_idx = train_idx[:6400]
        # torch.save(
        #     {
        #         "train": torch.tensor(train_idx, dtype=torch.long),
        #         "valid": torch.tensor(valid_idx, dtype=torch.long),
        #         "test": torch.tensor(test_idx, dtype=torch.long),
        #     },
        #     os.path.join(self.root, "split", "split_dict.pt"),
        # )
        print("DONE!!!")


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


# class Graph(Data):
#     def __init__(self, num_nodes, edges, node_feat, edge_feat):
#         super().__init__()
#         self.num_nodes = num_nodes
#         self.edges = edges
#         self.node_features = node_feat
#         self.edge_feat = edge_feat
#         self.n


class GeoPredCollateFn:
    def __init__(self):
        ...

    def __call__(self, batch_data_list):
        atom_bond_graph_list = []
        bond_angle_graph_list = []
        angle_dihedral_graph_list = []
        masked_atom_bond_graph_list = []
        masked_bond_angle_graph_list = []
        masked_angle_dihedral_graph_list = []
        Cm_node_i = []
        Cm_context_id = []
        Fg_morgan = []
        Fg_daylight = []
        Fg_maccs = []
        Ba_node_i = []
        Ba_node_j = []
        Ba_node_k = []
        Ba_bond_angle = []
        Adi_node_a = []
        Adi_node_b = []
        Adi_node_c = []
        Adi_node_d = []
        Adi_angle_dihedral = []
        Bl_node_i = []
        Bl_node_j = []
        Bl_bond_length = []
        Ad_node_i = []
        Ad_node_j = []
        Ad_atom_dist = []

        bo_bond_order = []
        atom_cm5 = []
        atom_espc = []
        atom_hirshfeld = []
        atom_npa = []

        graph_dict = {}

        for data in batch_data_list:
            ab_g = Data(edge_index=data['edges'], x=data['node_feat'], edge_attr=data['edge_attr'])
            ba_g = Data(edge_index=data['BondAngleGraph_edges'], edge_attr=data['bond_angle'])
            adi_g = Data(edge_index=data['AngleDihedralGraph_edges'], edge_attr=data['dihedral_angle'])

            atom_bond_graph_list.append(ab_g)
            bond_angle_graph_list.append(ba_g)
            angle_dihedral_graph_list.append(adi_g)

        atom_bond_graph, slices = self.collate(atom_bond_graph_list)
        graph_dict['atom_bond_graph'] = atom_bond_graph


if __name__ == "__main__":
    import json
    from tqdm import tqdm

    with open("../configs/input_feats.json") as f:
        configs = json.load(f)

    dataset = EgeognnPretrainedDataset(
        root='../data/demo', dataset_name='demo',
        remove_hs=True, base_path="../data/demo",
        atom_names=configs["atom_names"], bond_names=configs["bond_names"], with_provided_3d=False
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
