from torch import nn

from models.conv import MLP
from models.gin import EGeoGNNModel


class DownstreamModel(nn.Module):
    def __init__(
            self, compound_encoder: EGeoGNNModel,
            hidden_size, dropout_rate, n_layers, task_type='class', num_tasks=1
    ):
        super(DownstreamModel, self).__init__()
        self.task_type = task_type
        self.num_tasks = num_tasks

        self.compound_encoder = compound_encoder
        self.norm = nn.LayerNorm(compound_encoder.latent_size)

        self.mlp = MLP(
            input_size=compound_encoder.latent_size,
            output_sizes=[hidden_size] * n_layers + [num_tasks],
            use_layer_norm=False,
            activation=nn.ReLU,
            layernorm_before=False,
            use_bn=False,
            dropout=dropout_rate
        )
        if self.task_type == 'class':
            self.out_act = nn.Sigmoid()

    def forward(
            self,
            AtomBondGraph_edges, BondAngleGraph_edges, AngleDihedralGraph_edges,
            x, bond_attr, bond_lengths, bond_angles, dihedral_angles,
            num_graphs, atom_batch
    ):
        _, _, _, _, graph_repr = self.compound_encoder(
            AtomBondGraph_edges=AtomBondGraph_edges,
            BondAngleGraph_edges=BondAngleGraph_edges,
            AngleDihedralGraph_edges=AngleDihedralGraph_edges,
            x=x, bond_attr=bond_attr, bond_lengths=bond_lengths,
            bond_angles=bond_angles, dihedral_angles=dihedral_angles,
            num_graphs=num_graphs, atom_batch=atom_batch,
            masked_atom_indices=None,
            masked_bond_indices=None,
            masked_angle_indices=None,
            masked_dihedral_indices=None
        )

        graph_repr = self.norm(graph_repr)
        pred = self.mlp(graph_repr)
        if self.task_type == 'class':
            pred = self.out_act(pred)
        return pred