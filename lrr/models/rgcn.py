from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import RGCNConv

from .residual import FullRankResidual


class RGCNBackbone(nn.Module):
    """Paper backbone layout: 128-d entity embeddings, RGCN, DistMult."""

    def __init__(self, num_entities: int, base_relations: int, *, embedding_dim: int = 128,
                 hidden_dim: int = 128, output_dim: int = 64) -> None:
        super().__init__()
        relation_count = 2 * int(base_relations)
        self.node_emb = nn.Embedding(int(num_entities), int(embedding_dim))
        self.conv1 = RGCNConv(embedding_dim, hidden_dim, relation_count)
        self.conv2 = RGCNConv(hidden_dim, output_dim, relation_count)
        self.W = nn.Parameter(torch.empty(int(base_relations), int(output_dim)))
        nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain("relu"))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        h0 = self.node_emb(x)
        h1 = self.conv1(h0, edge_index, edge_type)
        return self.conv2(F.relu(h1), edge_index, edge_type)

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor,
               edge_type: torch.Tensor) -> torch.Tensor:
        return (z[edge_index[0]] * self.W[edge_type] * z[edge_index[1]]).sum(dim=-1)


class FullRankRGCN(nn.Module):
    """Frozen RGCN + DistMult with student-only FullRank repair matrices."""

    TRAINABLE_NAMES = ("repair1.W", "repair2.W")

    def __init__(self, backbone: RGCNBackbone) -> None:
        super().__init__()
        self.backbone = backbone
        reference = next(backbone.parameters())
        self.repair1 = FullRankResidual(128, device=reference.device, dtype=reference.dtype)
        self.repair2 = FullRankResidual(64, device=reference.device, dtype=reference.dtype)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in self.repair1.parameters():
            parameter.requires_grad_(True)
        for parameter in self.repair2.parameters():
            parameter.requires_grad_(True)
        self.backbone.eval()

    def trainable_parameters(self):
        return (self.repair1.W, self.repair2.W)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def shared_first_layer(self, x: torch.Tensor, edge_index: torch.Tensor,
                           edge_type: torch.Tensor, gate1: torch.Tensor,
                           gate2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return student h2 and FROZEN_TEACHER h2, sharing only frozen h0/conv1."""
        h0 = self.backbone.node_emb(x)
        h1_base = self.backbone.conv1(h0, edge_index, edge_type)
        student_h1 = self.repair1(h1_base, gate1)
        student_h2_base = self.backbone.conv2(F.relu(student_h1), edge_index, edge_type)
        student_h2 = self.repair2(student_h2_base, gate2)
        with torch.no_grad():
            teacher_h2 = self.backbone.conv2(F.relu(h1_base), edge_index, edge_type)
        return student_h2, teacher_h2

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor, gate1: torch.Tensor,
                gate2: torch.Tensor) -> torch.Tensor:
        h0 = self.backbone.node_emb(x)
        h1_base = self.backbone.conv1(h0, edge_index, edge_type)
        h1 = self.repair1(h1_base, gate1)
        h2_base = self.backbone.conv2(F.relu(h1), edge_index, edge_type)
        return self.repair2(h2_base, gate2)

    def original_representations(self, x: torch.Tensor, edge_index: torch.Tensor,
                                 edge_type: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.backbone(x, edge_index, edge_type)

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor,
               edge_type: torch.Tensor) -> torch.Tensor:
        return self.backbone.decode(z, edge_index, edge_type)
