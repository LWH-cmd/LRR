from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """Two-layer GCN with an explicit homogeneous-graph input contract."""

    _INPUT_MODES = {"feature_matrix", "node_id_embedding"}

    def __init__(
        self,
        args,
        num_nodes: Optional[int] = None,
        input_mode: Optional[str] = None,
        expected_layer_dims: Optional[Tuple[int, int]] = None,
        require_explicit_input_mode: bool = False,
        **kwargs,
    ):
        super().__init__()

        resolved_input_mode = (
            input_mode if input_mode is not None else getattr(args, "input_mode", None)
        )
        if resolved_input_mode is None and not require_explicit_input_mode:
            resolved_input_mode = "feature_matrix"
            self.input_contract_status = "LEGACY_FEATURE_MATRIX_COMPATIBILITY"
        else:
            self.input_contract_status = "EXPLICIT_COMMON_HG_GCN_INPUT_CONTRACT"
        if resolved_input_mode not in self._INPUT_MODES:
            raise ValueError(
                "input_mode must be explicitly set to 'feature_matrix' or "
                f"'node_id_embedding', got {resolved_input_mode!r}"
            )

        self.input_mode = resolved_input_mode
        self.num_nodes = None if num_nodes is None else int(num_nodes)
        if self.input_mode == "node_id_embedding":
            if self.num_nodes is None or self.num_nodes <= 0:
                raise ValueError(
                    "node_id_embedding requires a positive explicit num_nodes"
                )
            self.node_emb = nn.Embedding(self.num_nodes, int(args.in_dim))
        else:
            self.node_emb = None

        self.conv1 = GCNConv(args.in_dim, args.hidden_dim)
        self.conv2 = GCNConv(args.hidden_dim, args.out_dim)

        if expected_layer_dims is None:
            expected_d1 = getattr(args, "expected_layer1_effective_dim", None)
            expected_d2 = getattr(args, "expected_layer2_effective_dim", None)
            if (expected_d1 is None) != (expected_d2 is None):
                raise ValueError(
                    "Both expected layer dimensions must be provided together"
                )
            if expected_d1 is not None:
                expected_layer_dims = (int(expected_d1), int(expected_d2))

        self.validate_runtime_dimensions(expected_layer_dims)

    def effective_layer_dimensions(self) -> Tuple[int, int]:
        """Return dimensions derived from the constructed convolution modules."""
        return int(self.conv1.out_channels), int(self.conv2.out_channels)

    def validate_runtime_dimensions(
        self, expected_layer_dims: Optional[Tuple[int, int]] = None
    ) -> Tuple[int, int]:
        actual = self.effective_layer_dimensions()
        if expected_layer_dims is not None:
            expected = tuple(int(value) for value in expected_layer_dims)
            if len(expected) != 2:
                raise ValueError("expected_layer_dims must contain exactly two values")
            if actual != expected:
                raise ValueError(
                    f"Runtime layer dimensions {actual} do not match expected {expected}"
                )
        return actual

    def common_input_preparation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the explicit CommonHgGcnInputContractV1 input preparation."""
        if self.input_mode == "feature_matrix":
            if self.node_emb is not None:
                raise RuntimeError("feature_matrix mode must not contain node_emb")
            if x.dim() != 2:
                raise ValueError(
                    f"feature_matrix requires x with shape [N,F], got {tuple(x.shape)}"
                )
            if not x.is_floating_point():
                raise TypeError("feature_matrix input must use a floating dtype")
            if x.size(1) != int(self.conv1.in_channels):
                raise ValueError(
                    f"Expected feature dimension {self.conv1.in_channels}, got {x.size(1)}"
                )
            return x

        if self.node_emb is None:
            raise RuntimeError("node_id_embedding mode requires node_emb")
        if x.dim() != 1:
            raise ValueError(
                f"node_id_embedding requires x with shape [N], got {tuple(x.shape)}"
            )
        if x.dtype != torch.int64:
            raise TypeError(
                f"node_id_embedding requires torch.int64 IDs, got {x.dtype}"
            )
        if x.size(0) != int(self.num_nodes):
            raise ValueError(
                "node_id_embedding requires one ID per node: "
                f"expected {self.num_nodes}, got {x.size(0)}"
            )
        if x.numel() > 0:
            minimum = int(x.min().item())
            maximum = int(x.max().item())
            if minimum < 0 or maximum >= int(self.num_nodes):
                raise ValueError(
                    f"Node IDs must be in [0,{self.num_nodes}), got [{minimum},{maximum}]"
                )
        return self.node_emb(x)

    @staticmethod
    def official_intermediate_processing(x: torch.Tensor) -> torch.Tensor:
        """Preserve the official GNNDelete GCN inter-layer processing."""
        return F.relu(x)

    def forward(self, x, edge_index, return_all_emb=False):
        h0 = self.common_input_preparation(x)
        x1 = self.conv1(h0, edge_index)
        x = self.official_intermediate_processing(x1)
        x2 = self.conv2(x, edge_index)

        if return_all_emb:
            return x1, x2
        
        return x2

    def decode(self, z, pos_edge_index, neg_edge_index=None):
        if neg_edge_index is not None:
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

        else:
            edge_index = pos_edge_index
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

        return logits
