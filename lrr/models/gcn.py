from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

from .residual import FullRankResidual


class GCNBackbone(nn.Module):
    """Frozen paper backbone with the explicit homogeneous input contract."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64,
                 *, input_mode: str = "feature_matrix", num_nodes: int | None = None,
                 cached: bool = False) -> None:
        super().__init__()
        if input_mode not in {"feature_matrix", "node_id_embedding"}:
            raise ValueError("input_mode must be feature_matrix or node_id_embedding")
        if input_mode == "node_id_embedding" and (num_nodes is None or num_nodes <= 0):
            raise ValueError("node_id_embedding requires num_nodes")
        self.input_mode = input_mode
        self.num_nodes = None if num_nodes is None else int(num_nodes)
        self.node_emb = (nn.Embedding(self.num_nodes, input_dim)
                         if input_mode == "node_id_embedding" else None)
        self.conv1 = GCNConv(input_dim, hidden_dim, cached=cached)
        self.conv2 = GCNConv(hidden_dim, output_dim, cached=cached)

    def common_input_preparation(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_mode == "feature_matrix":
            if x.ndim != 2 or not x.is_floating_point():
                raise TypeError("feature_matrix input must be floating [N,F]")
            if x.size(1) != self.conv1.in_channels:
                raise ValueError("feature dimension differs from pretrained backbone")
            return x
        if x.ndim != 1 or x.dtype != torch.long or x.numel() != self.num_nodes:
            raise TypeError("node_id_embedding input must contain all int64 node IDs")
        if self.node_emb is None:
            raise RuntimeError("node embedding is not initialized")
        return self.node_emb(x)

    @staticmethod
    def official_intermediate_processing(h: torch.Tensor) -> torch.Tensor:
        return F.relu(h)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.common_input_preparation(x)
        return self.conv2(self.official_intermediate_processing(self.conv1(h, edge_index)), edge_index)

    @staticmethod
    def decode(z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)


class FullRankGCN(nn.Module):
    """Frozen two-layer GCN scorer with two trainable FullRank repairs."""

    TRAINABLE_NAMES = ("repair1.W", "repair2.W")

    def __init__(self, backbone: GCNBackbone) -> None:
        super().__init__()
        self.backbone = backbone
        self.repair1 = FullRankResidual(backbone.conv1.out_channels)
        self.repair2 = FullRankResidual(backbone.conv2.out_channels)
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in self.repair1.parameters():
            parameter.requires_grad_(True)
        for parameter in self.repair2.parameters():
            parameter.requires_grad_(True)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def trainable_parameters(self):
        return (self.repair1.W, self.repair2.W)

    def common_input_preparation(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.common_input_preparation(x)

    @staticmethod
    def official_intermediate_processing(h: torch.Tensor) -> torch.Tensor:
        return GCNBackbone.official_intermediate_processing(h)

    @property
    def conv1(self) -> GCNConv:
        return self.backbone.conv1

    @property
    def conv2(self) -> GCNConv:
        return self.backbone.conv2

    def student_forward(self, h1_base: torch.Tensor, edge_index: torch.Tensor,
                        gate1: torch.Tensor, gate2: torch.Tensor) -> torch.Tensor:
        h1 = self.repair1(h1_base, gate1)
        h2_base = self.conv2(self.backbone.official_intermediate_processing(h1), edge_index)
        return self.repair2(h2_base, gate2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                gate1: torch.Tensor, gate2: torch.Tensor) -> torch.Tensor:
        h0 = self.backbone.common_input_preparation(x)
        h1_base = self.conv1(h0, edge_index)
        return self.student_forward(h1_base, edge_index, gate1, gate2)

    @staticmethod
    def decode(z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return GCNBackbone.decode(z, edge_index)
