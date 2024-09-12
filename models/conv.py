import torch
from torch import nn, Tensor
from typing import Optional
import math
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_geometric.nn.conv import TransformerConv


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        output_sizes,
        use_layer_norm=False,
        activation=nn.ReLU,
        dropout=0.0,
        layernorm_before=False,
        use_bn=False,
    ):
        super().__init__()
        module_list = []
        if not use_bn:
            if layernorm_before:
                module_list.append(nn.LayerNorm(input_size))
            for size in output_sizes:
                module_list.append(nn.Linear(input_size, size))
                if size != 1:
                    module_list.append(activation())
                    if dropout > 0:
                        module_list.append(nn.Dropout(dropout))
                input_size = size
            if not layernorm_before and use_layer_norm:
                module_list.append(nn.LayerNorm(input_size))
        else:
            for size in output_sizes:
                module_list.append(nn.Linear(input_size, size))
                if size != 1:
                    module_list.append(nn.BatchNorm1d(size))
                    module_list.append(activation())
                input_size = size

        self.module_list = nn.ModuleList(module_list)
        self.reset_parameters()

    def reset_parameters(self):
        for item in self.module_list:
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()

    def forward(self, x):
        for item in self.module_list:
            x = item(x)
        return x


class MLPwoLastAct(nn.Module):
    def __init__(
        self,
        input_size,
        output_sizes,
        use_layer_norm=False,
        activation=nn.ReLU,
        dropout=0.0,
        layernorm_before=False,
        use_bn=False,
    ):
        super().__init__()
        module_list = []
        if not use_bn:
            if layernorm_before:
                module_list.append(nn.LayerNorm(input_size))

            if dropout > 0:
                module_list.append(nn.Dropout(dropout))
            for i, size in enumerate(output_sizes):
                module_list.append(nn.Linear(input_size, size))
                if i < len(output_sizes) - 1:
                    module_list.append(activation())
                input_size = size
            if not layernorm_before and use_layer_norm:
                module_list.append(nn.LayerNorm(input_size))
        else:
            for i, size in enumerate(output_sizes):
                module_list.append(nn.Linear(input_size, size))
                if i < len(output_sizes) - 1:
                    module_list.append(nn.BatchNorm1d(size))
                    module_list.append(activation())
                input_size = size

        self.module_list = nn.ModuleList(module_list)
        self.reset_parameters()

    def reset_parameters(self):
        for item in self.module_list:
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()

    def forward(self, x):
        for item in self.module_list:
            x = item(x)
        return x


class DropoutIfTraining(nn.Module):
    """
    Borrow this implementation from deepmind
    """

    def __init__(self, p=0.0, submodule=None):
        super().__init__()
        assert p >= 0 and p <= 1
        self.p = p
        self.submodule = submodule if submodule else nn.Identity()

    def forward(self, x):
        x = self.submodule(x)
        newones = x.new_ones((x.size(0), 1))
        newones = F.dropout(newones, p=self.p, training=self.training)
        out = x * newones
        return out


class NodeAttn(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.emb_dim = emb_dim
        if num_heads is None:
            num_heads = emb_dim // 64
        self.num_heads = num_heads
        assert self.emb_dim % self.num_heads == 0

        self.w1 = nn.Linear(3 * emb_dim, emb_dim)
        self.w2 = nn.Parameter(torch.zeros((self.num_heads, self.emb_dim // self.num_heads)))
        self.w3 = nn.Linear(2 * emb_dim, emb_dim)
        self.head_dim = self.emb_dim // self.num_heads
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w2, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w3.weight, gain=1 / math.sqrt(2))

    def forward(self, q, k_v, k_e, index, nnode):
        """
        q: [N, C]
        k: [N, 2*c]
        v: [N, 2*c]
        """
        x = torch.cat([q, k_v, k_e], dim=1)
        x = self.w1(x).view(-1, self.num_heads, self.head_dim)
        x = F.leaky_relu(x)
        attn_weight = torch.einsum("nhc,hc->nh", x, self.w2).unsqueeze(-1)
        attn_weight = softmax(attn_weight, index)

        v = torch.cat([k_v, k_e], dim=1)
        v = self.w3(v).view(-1, self.num_heads, self.head_dim)
        x = (attn_weight * v).reshape(-1, self.emb_dim)
        x = scatter(x, index, dim=0, reduce="sum", dim_size=nnode)
        return x
