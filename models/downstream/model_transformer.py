import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models.conv import MLP, MultiHeadAttentionLayer
from models.gin import EGeoGNNModel


class DownstreamTransformerModel(nn.Module):
    def __init__(
            self, compound_encoder: EGeoGNNModel,
            hidden_size, dropout_rate, n_layers,
            endpoints, frozen_encoder, device,
            task_type='regression',
            inference=False
    ):
        super(DownstreamTransformerModel, self).__init__()
        self.task_type = task_type
        self.endpoints = endpoints
        self.frozen_encoder = frozen_encoder
        self.latent_size = compound_encoder.latent_size
        self.num_tasks = len(endpoints)

        self.endpoint_embedding = nn.Embedding(self.num_tasks, self.latent_size)

        self.attention = MultiHeadAttentionLayer(
            hid_dim=compound_encoder.latent_size,
            n_heads=8,
            dropout=dropout_rate,
            device=device
        )

        self.downstream_layer = MLP(
            input_size=compound_encoder.latent_size,
            output_sizes=[hidden_size] * n_layers + [1],
            use_layer_norm=False,
            activation=nn.ReLU,
            layernorm_before=False,
            use_bn=False,
            dropout=dropout_rate
        )

        self.compound_encoder = compound_encoder

        self.graph_feat_norm = nn.LayerNorm(self.latent_size)
        self.endpoint_attn_layer_norm = nn.LayerNorm(self.latent_size)
        self.dropout = nn.Dropout(dropout_rate)

        if self.task_type == 'class':
            self.out_act = nn.Sigmoid()

        self.inference = inference
        self.device = device

        self.loss_func = nn.MSELoss()

    def forward(
            self,
            AtomBondGraph_edges, BondAngleGraph_edges, AngleDihedralGraph_edges,
            pos, x, bond_attr, bond_lengths, bond_angles, dihedral_angles,
            num_atoms, num_bonds, num_angles, num_graphs, atom_batch,
            tgt_endpoints
    ):

        endpoint_index = torch.tensor([self.endpoints.index(x) for x in tgt_endpoints]).reshape(-1, 1).to(self.device)
        endpoint_embedding = self.endpoint_embedding(endpoint_index.int())

        if self.frozen_encoder:
            self.compound_encoder.eval()
            with torch.no_grad():
                _, _, _, _, graph_repr = self.compound_encoder(
                    AtomBondGraph_edges=AtomBondGraph_edges,
                    BondAngleGraph_edges=BondAngleGraph_edges,
                    AngleDihedralGraph_edges=AngleDihedralGraph_edges,
                    pos=pos, x=x, bond_attr=bond_attr, bond_lengths=bond_lengths,
                    bond_angles=bond_angles, dihedral_angles=dihedral_angles,
                    num_graphs=num_graphs, atom_batch=atom_batch,
                    num_atoms=num_atoms, num_bonds=num_bonds, num_angles=num_angles,
                    masked_atom_indices=None,
                    masked_bond_indices=None,
                    masked_angle_indices=None,
                    masked_dihedral_indices=None
                )
        else:
            _, _, _, _, graph_repr = self.compound_encoder(
                AtomBondGraph_edges=AtomBondGraph_edges,
                BondAngleGraph_edges=BondAngleGraph_edges,
                AngleDihedralGraph_edges=AngleDihedralGraph_edges,
                pos=pos, x=x, bond_attr=bond_attr, bond_lengths=bond_lengths,
                bond_angles=bond_angles, dihedral_angles=dihedral_angles,
                num_graphs=num_graphs, atom_batch=atom_batch,
                num_atoms=num_atoms, num_bonds=num_bonds, num_angles=num_angles,
                masked_atom_indices=None,
                masked_bond_indices=None,
                masked_angle_indices=None,
                masked_dihedral_indices=None
            )

        graph_repr = self.graph_feat_norm(graph_repr).unsqueeze(1)
        _graph_repr, _ = self.attention(graph_repr, endpoint_embedding, endpoint_embedding)
        graph_repr = self.endpoint_attn_layer_norm(graph_repr + self.dropout(_graph_repr)).squeeze(1)

        pred = self.downstream_layer(graph_repr)

        if self.task_type == 'class':
            pred = self.out_act(pred)
        return pred

    def compute_loss(self, pred, target):
        loss_dict = {}

        loss = self.loss_func(pred, target)
        loss_dict['loss'] = loss.detach().item()

        return loss, loss_dict
