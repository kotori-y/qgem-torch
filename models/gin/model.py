import copy
from typing import List

from torch import nn
from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)

from datasets.featurizer import get_feature_dims
from models.conv import MLP
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

    def mask_attr(
            self, x, bond_attr, bond_lengths, bond_angles, dihedral_angles,
            masked_atom_indices, masked_bond_indices,
            masked_angle_indices, masked_dihedral_indices
    ):
        _x = copy.deepcopy(x)
        _bond_attr = copy.deepcopy(bond_attr)
        _bond_lengths = copy.deepcopy(bond_lengths)
        _bond_angles = copy.deepcopy(bond_angles)
        _dihedral_angles = copy.deepcopy(dihedral_angles)

        atom_vocab_sizes = get_feature_dims(self.atom_names)
        for i in range(_x.shape[1]):
            if masked_atom_indices is not None:
                _x[:, i][masked_atom_indices] = atom_vocab_sizes[i] - 1

        bond_vocab_sizes = get_feature_dims(self.bond_names)
        for i in range(_bond_attr.shape[1]):
            if masked_bond_indices is not None:
                _bond_attr[:, i][masked_bond_indices] = bond_vocab_sizes[i] - 1

        _bond_lengths[masked_bond_indices] = 0
        _bond_angles[masked_angle_indices] = 0
        _dihedral_angles[masked_dihedral_indices] = 0

        return _x, _bond_attr, _bond_lengths, _bond_angles, _dihedral_angles

    def forward(
            self, AtomBondGraph_edges, BondAngleGraph_edges, AngleDihedralGraph_edges,
            x, bond_attr, bond_lengths, bond_angles, dihedral_angles,
            atom_batch, num_graphs,
            masked_atom_indices, masked_bond_indices,
            masked_angle_indices, masked_dihedral_indices
    ):
        x, bond_attr, bond_lengths, bond_angles, dihedral_angles = self.mask_attr(
            x, bond_attr, bond_lengths, bond_angles, dihedral_angles,
            masked_atom_indices, masked_bond_indices,
            masked_angle_indices, masked_dihedral_indices
        )

        node_hidden = self.init_atom_embedding(x)
        bond_embed = self.init_bond_embedding(bond_attr)
        bond_hidden = bond_embed + self.init_bond_float_rbf(bond_lengths)
        angle_hidden = self.init_bond_angle_float_rbf(bond_angles)

        node_hidden_list = [node_hidden]
        edge_hidden_list = [bond_hidden]
        angle_hidden_list = [angle_hidden]

        for layer_id in range(self.n_layers):
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
        angle_repr = angle_hidden_list[-1]
        dihedral_repr = cur_dihedral_hidden
        graph_repr = global_mean_pool(node_repr, atom_batch, size=num_graphs)
        return node_repr, edge_repr, angle_repr, dihedral_repr, graph_repr


class EGEM(nn.Module):
    def __init__(
            self, compound_encoder: EGeoGNNModel, pretrain_tasks: List[str],
            n_layers: int, hidden_size: int, dropout_rate: float,
            use_layer_norm: bool, use_bn: bool, adc_vocab: int
    ):
        super().__init__()

        self.compound_encoder = compound_encoder
        self.pretrain_tasks = pretrain_tasks

        # bond length with regression
        if 'Blr' in pretrain_tasks:
            self.Blr_mlp = MLP(
                input_size=compound_encoder.latent_size,
                output_sizes=[hidden_size] * n_layers + [1],
                use_layer_norm=use_layer_norm,
                use_bn=use_bn,
                activation=nn.ReLU,
                dropout=dropout_rate
            )
            self.Blr_loss = nn.SmoothL1Loss()

        # bond angle with regression
        if 'Bar' in pretrain_tasks:
            self.Bar_mlp = MLP(
                input_size=compound_encoder.latent_size,
                output_sizes=[hidden_size] * n_layers + [1],
                use_layer_norm=use_layer_norm,
                use_bn=use_bn,
                activation=nn.ReLU,
                dropout=dropout_rate
            )
            self.Bar_loss = nn.SmoothL1Loss()

        # dihedral angle with regression
        if 'Dar' in pretrain_tasks:
            self.Dar_mlp = MLP(
                input_size=compound_encoder.latent_size,
                output_sizes=[hidden_size] * n_layers + [1],
                use_layer_norm=use_layer_norm,
                use_bn=use_bn,
                activation=nn.ReLU,
                dropout=dropout_rate
            )
            self.Dar_loss = nn.SmoothL1Loss()

        # atom distance with classification
        if 'Adc' in pretrain_tasks:
            self.Adc_mlp = MLP(
                input_size=compound_encoder.latent_size * 2,
                output_sizes=[hidden_size] * n_layers + [adc_vocab + 3],
                use_layer_norm=use_layer_norm,
                use_bn=use_bn,
                activation=nn.ReLU,
                dropout=dropout_rate
            )
            self.Adc_loss = nn.CrossEntropyLoss()

    def _get_Blr_loss(self, bond_attr, bond_lengths, masked_bond_indices=None):
        if masked_bond_indices is not None:
            bond_lengths = bond_lengths[masked_bond_indices]
            bond_attr = bond_attr[masked_bond_indices]

        pred = self.Blr_mlp(bond_attr)
        loss = self.Blr_loss(pred, bond_lengths.unsqueeze(-1))
        return loss

    def _get_Bar_loss(self, angle_attr, bond_angles, masked_angle_indices=None):
        if masked_angle_indices is not None:
            bond_angles = bond_angles[masked_angle_indices]
            angle_attr = angle_attr[masked_angle_indices]

        pred = self.Bar_mlp(angle_attr)
        loss = self.Bar_loss(pred, bond_angles.unsqueeze(-1))
        return loss

    def _get_Dar_loss(self, dihedral_attr, dihedral_angles, masked_dihedral_indices=None):
        if masked_dihedral_indices is not None:
            dihedral_angles = dihedral_angles[masked_dihedral_indices]
            dihedral_attr = dihedral_attr[masked_dihedral_indices]

        pred = self.Dar_mlp(dihedral_attr)
        loss = self.Dar_loss(pred, dihedral_angles.unsqueeze(-1))
        return loss

    def compute_loss(
            self, bond_attr, angle_attr, dihedral_attr,
            bond_lengths, bond_angles, dihedral_angles,
            masked_bond_indices=None, masked_angle_indices=None,
            masked_dihedral_indices=None
    ):
        loss = 0
        loss_dict = {}

        if "Blr" in self.pretrain_tasks:
            bond_length_loss = self._get_Blr_loss(bond_attr, bond_lengths, masked_bond_indices)
            loss += bond_length_loss
            loss_dict["bond_length_loss"] = bond_length_loss.detach().item()

        if "Bar" in self.pretrain_tasks:
            bond_angle_loss = self._get_Bar_loss(angle_attr, bond_angles, masked_angle_indices)
            loss += bond_angle_loss
            loss_dict["bond_angle_loss"] = bond_angle_loss.detach().item()

        if "Dar" in self.pretrain_tasks:
            dihedral_angle_loss = self._get_Dar_loss(dihedral_attr, dihedral_angles, masked_dihedral_indices)
            loss += dihedral_angle_loss
            loss_dict["dihedral_angle_loss"] = dihedral_angle_loss.detach().item()

        loss_dict["loss"] = loss.detach().item()
        return loss, loss_dict

    def forward(
            self, AtomBondGraph_edges, BondAngleGraph_edges, AngleDihedralGraph_edges,
            x, bond_attr, bond_lengths, bond_angles, dihedral_angles,
            num_graphs, atom_batch,
            masked_atom_indices, masked_bond_indices,
            masked_angle_indices, masked_dihedral_indices,
            **kwargs
    ):

        atom_attr, bond_attr, angle_attr, dihedral_attr, u = self.compound_encoder(
            AtomBondGraph_edges=AtomBondGraph_edges,
            BondAngleGraph_edges=BondAngleGraph_edges,
            AngleDihedralGraph_edges=AngleDihedralGraph_edges,
            x=x, bond_attr=bond_attr, bond_lengths=bond_lengths,
            bond_angles=bond_angles, dihedral_angles=dihedral_angles,
            num_graphs=num_graphs, atom_batch=atom_batch,
            masked_atom_indices=masked_atom_indices,
            masked_bond_indices=masked_bond_indices,
            masked_angle_indices=masked_angle_indices,
            masked_dihedral_indices=masked_dihedral_indices
        )

        return self.compute_loss(
            bond_attr=bond_attr, angle_attr=angle_attr, dihedral_attr=dihedral_attr,
            bond_lengths=bond_lengths, bond_angles=bond_angles, dihedral_angles=dihedral_angles,
            masked_bond_indices=masked_bond_indices,
            masked_angle_indices=masked_angle_indices,
            masked_dihedral_indices=masked_dihedral_indices
        )
        # self.compute_loss(bond_attr, masked_bond_indices=masked_bond_indices)
