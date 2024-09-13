import numpy as np
import torch
from torch import nn
from torch_geometric.nn.models import GIN
from torch_geometric.nn.norm import GraphNorm


class EGeoGNNBlock(nn.Module):
    def __init__(self, latent_size, dropout_rate, last_act):
        super(EGeoGNNBlock, self).__init__()

        self.latent_size = latent_size
        self.last_act = last_act

        self.gnn = GIN(latent_size, latent_size, num_layers=2)
        self.norm = nn.LayerNorm(latent_size)
        self.graph_norm = GraphNorm(latent_size)

        if last_act:
            self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, node_hidden, edge_hidden, edge_index):
        """tbd"""
        out = self.gnn(x=node_hidden, edge_attr=edge_hidden, edge_index=edge_index)
        out = self.norm(out)
        out = self.graph_norm(out)
        if self.last_act:
            out = self.act(out)
        if self.training:
            out = self.dropout(out)
        out = out + node_hidden
        return out


class AtomBondEmbedding(nn.Module):
    def __init__(self, vocab_sizes, latent_size):
        super(AtomBondEmbedding, self).__init__()

        self.embed_layers = nn.ModuleList()
        for vocab_size in vocab_sizes:
            embed = nn.Embedding(
                vocab_size,
                latent_size,
            )
            self.embed_layers.append(embed)

    def forward(self, node_features):
        out_embed = 0
        for i, layer in enumerate(self.embed_layers):
            out_embed += layer(node_features[:, i])
        return out_embed


class RBF(nn.Module):
    def __init__(self, centers, gamma):
        super(RBF, self).__init__()
        self.centers = torch.reshape(torch.tensor(centers, dtype=torch.float32), [1, -1])
        self.gamma = gamma

    def forward(self, x):
        x = torch.reshape(x, [-1, 1])
        return torch.exp(-self.gamma * torch.square(x - self.centers))


class BondFloatRBF(nn.Module):
    def __init__(self, latent_size):
        super(BondFloatRBF, self).__init__()

        centers = np.arange(0, 2, 0.1)
        gamma = 10.0

        self.rbf = RBF(centers, gamma)
        self.linear = nn.Linear(len(centers), latent_size)

    def forward(self, bond_lengths):
        rbf_x = self.rbf(bond_lengths)
        out_embed = self.linear(rbf_x)
        return out_embed


class BondAngleFloatRBF(nn.Module):
    def __init__(self, latent_size):
        super(BondAngleFloatRBF, self).__init__()

        centers = np.arange(0, np.pi, 0.01)
        gamma = 10.0

        self.rbf = RBF(centers, gamma)
        self.linear = nn.Linear(len(centers), latent_size)

    def forward(self, bond_angles):
        rbf_x = self.rbf(bond_angles)
        out_embed = self.linear(rbf_x)
        return out_embed


class DihedralAngleFloatRBF(nn.Module):
    def __init__(self, latent_size):
        super(DihedralAngleFloatRBF, self).__init__()

        centers = np.arange(-np.pi, np.pi, 0.01)
        gamma = 10

        self.rbf = RBF(centers, gamma)
        self.linear = nn.Linear(len(centers), latent_size)

    def forward(self, dihedral_angles):
        rbf_x = self.rbf(dihedral_angles)
        out_embed = self.linear(rbf_x)
        return out_embed
