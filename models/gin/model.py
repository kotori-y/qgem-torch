from torch import nn
from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)

from datasets.featurizer import get_feature_dims
from models.gin.encoder import AtomBondEmbedding, BondAngleFloatRBF, DihedralAngleFloatRBF, EGeoGNNBlock, BondFloatRBF

_REDUCER_NAMES = {
    "sum": global_add_pool,
    "mean": global_mean_pool,
    "max": global_max_pool,
}


class EGeoGNNModel(nn.Module):
    def __init__(self, latent_size, encoder_dropout, n_layers, atom_names, bond_names, **kwargs):
        super(EGeoGNNModel, self).__init__()

        self.latent_size = latent_size
        self.dropout_rate = encoder_dropout
        self.n_layers = n_layers

        self.atom_names = atom_names
        self.bond_names = bond_names

        self.init_atom_embedding = AtomBondEmbedding(get_feature_dims(self.atom_names), self.latent_size)
        self.init_bond_embedding = AtomBondEmbedding(get_feature_dims(self.bond_names), self.latent_size)
        self.init_bond_float_rbf = BondFloatRBF(self.latent_size)
        self.init_bond_angle_float_rbf = BondAngleFloatRBF(self.latent_size)

        self.bond_embedding_list = nn.ModuleList()
        self.bond_float_rbf_list = nn.ModuleList()
        self.bond_angle_float_rbf_list = nn.ModuleList()
        self.dihedral_angle_float_rbf_list = nn.ModuleList()

        self.atom_bond_block_list = nn.ModuleList()
        self.bond_angle_block_list = nn.ModuleList()
        self.angle_dihedral_block_list = nn.ModuleList()

        for layer_id in range(self.n_layers):
            self.bond_embedding_list.append(
                AtomBondEmbedding(get_feature_dims(self.bond_names), self.latent_size)
            )
            self.bond_float_rbf_list.append(
                BondFloatRBF(self.latent_size)
            )
            self.bond_angle_float_rbf_list.append(
                BondAngleFloatRBF(self.latent_size)
            )
            self.dihedral_angle_float_rbf_list.append(
                DihedralAngleFloatRBF(self.latent_size)
            )
            self.atom_bond_block_list.append(
                EGeoGNNBlock(self.latent_size, self.dropout_rate, last_act=(layer_id != self.n_layers - 1))
            )
            self.bond_angle_block_list.append(
                EGeoGNNBlock(self.latent_size, self.dropout_rate, last_act=(layer_id != self.n_layers - 1))
            )
            self.angle_dihedral_block_list.append(
                EGeoGNNBlock(self.latent_size, self.dropout_rate, last_act=(layer_id != self.n_layers - 1))
            )

    @property
    def node_dim(self):
        """the out dim of graph_repr"""
        return self.embed_dim

    @property
    def graph_dim(self):
        """the out dim of graph_repr"""
        return self.embed_dim

    def forward(
            self, AtomBondGraph_edges, BondAngleGraph_edges, AngleDihedralGraph_edges,
            x, bond_attr, bond_lengths, bond_angles, dihedral_angles,
            atom_batch, num_graphs,
            masked_atom_indices, masked_bond_indices,
            masked_angle_indices, masked_dihedral_indices
    ):
        node_hidden = self.init_atom_embedding(x)
        bond_embed = self.init_bond_embedding(bond_attr)
        bond_hidden = bond_embed + self.init_bond_float_rbf(bond_lengths)
        angle_hidden = self.init_bond_angle_float_rbf(bond_angles)

        node_hidden_list = [node_hidden]
        edge_hidden_list = [bond_hidden]
        angle_hidden_list = [angle_hidden]

        for layer_id in range(self.layer_num):
            node_hidden = self.atom_bond_block_list[layer_id](
                node_hidden=node_hidden_list[layer_id],
                edge_hidden=edge_hidden_list[layer_id],
                edge_index=AtomBondGraph_edges
            )

            cur_edge_hidden = self.bond_embedding_list[layer_id](bond_attr)
            cur_edge_hidden = cur_edge_hidden + self.bond_float_rbf_list[layer_id](bond_lengths)
            edge_hidden = self.bond_angle_block_list[layer_id](
                node_hidden=cur_edge_hidden,
                edge_hidden=angle_hidden_list[layer_id],
                edge_index=BondAngleGraph_edges
            )

            cur_angle_hidden = self.bond_angle_float_rbf_list[layer_id](bond_angles)
            cur_dihedral_hidden = self.dihedral_angle_float_rbf_list[layer_id](dihedral_angles)
            angle_hidden = self.angle_dihedral_block_list[layer_id](
                node_hidden=cur_angle_hidden,
                edge_hidden=cur_dihedral_hidden,
                edge_index=AngleDihedralGraph_edges
            )

            node_hidden_list.append(node_hidden)
            edge_hidden_list.append(edge_hidden)
            angle_hidden_list.append(angle_hidden)

        node_repr = node_hidden_list[-1]
        edge_repr = edge_hidden_list[-1]
        graph_repr = global_mean_pool(node_repr)
        return node_repr, edge_repr, graph_repr
