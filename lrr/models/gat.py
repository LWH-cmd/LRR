"""COMMON-GAT-v1 and its paper-main FullRank residual repair."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv

from .residual import FullRankResidual


class GATBackbone(nn.Module):
    architecture_version = "COMMON-GAT-v1"

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.conv1 = GATConv(input_dim, 128, heads=1, concat=True,
                             negative_slope=0.2, dropout=0.0, add_self_loops=True, bias=True)
        self.conv2 = GATConv(128, 64, heads=1, concat=True,
                             negative_slope=0.2, dropout=0.0, add_self_loops=True, bias=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv2(F.relu(self.conv1(x, edge_index)), edge_index)

    @staticmethod
    def decode(z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)


class FullRankGAT(nn.Module):
    TRAINABLE_NAMES = ("repair1.W", "repair2.W")

    def __init__(self, backbone: GATBackbone) -> None:
        super().__init__()
        self.backbone = backbone
        reference = next(backbone.parameters())
        self.repair1 = FullRankResidual(128, device=reference.device, dtype=reference.dtype)
        self.repair2 = FullRankResidual(64, device=reference.device, dtype=reference.dtype)
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def from_layer1(self, h1_base, edge_index, gate1, gate2):
        h1 = self.repair1(h1_base, gate1)
        return self.repair2(self.backbone.conv2(F.relu(h1), edge_index), gate2)

    def forward(self, x, edge_index, gate1, gate2):
        return self.from_layer1(self.backbone.conv1(x, edge_index), edge_index, gate1, gate2)

    def decode(self, z, edge_index):
        return self.backbone.decode(z, edge_index)
