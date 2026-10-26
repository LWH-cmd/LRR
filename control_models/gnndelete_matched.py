"""Phase 0B GNNDelete-MatchedGraph-IdentityInit operator."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .gcn import GCN


class IdentityDeletionLayer(nn.Module):
    """Full square deletion matrix, applied only inside the supplied support."""

    def __init__(
        self,
        dim: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        if self.dim <= 0:
            raise ValueError("deletion layer dimension must be positive")
        resolved_dtype = torch.get_default_dtype() if dtype is None else dtype
        self.deletion_weight = nn.Parameter(
            torch.eye(
                self.dim,
                device=device,
                dtype=resolved_dtype,
            )
        )

    def forward(self, x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != self.dim:
            raise ValueError(f"expected x with shape [N,{self.dim}]")
        if support.dim() != 1 or support.shape != (x.size(0),):
            raise ValueError(f"support must have shape [{x.size(0)}]")
        if support.dtype != torch.bool or support.device != x.device:
            raise TypeError("support must be bool on the representation device")
        if x.device != self.deletion_weight.device:
            raise TypeError("deletion matrix device must match representation")
        if x.dtype != self.deletion_weight.dtype:
            raise TypeError("deletion matrix dtype must match representation")
        output = x.clone()
        output[support] = output[support].matmul(self.deletion_weight)
        return output


class GCNDeleteMatched(GCN):
    """Two full-matrix identity-initialized deletion operators on frozen GCN."""

    def __init__(
        self,
        args,
        num_nodes: Optional[int] = None,
        input_mode: Optional[str] = None,
        expected_layer_dims: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            args,
            num_nodes=num_nodes,
            input_mode=input_mode,
            expected_layer_dims=expected_layer_dims,
            require_explicit_input_mode=True,
            **kwargs,
        )
        d1, d2 = self.effective_layer_dimensions()
        reference = next(self.conv1.parameters())
        self.deletion1 = IdentityDeletionLayer(
            d1, device=reference.device, dtype=reference.dtype
        )
        self.deletion2 = IdentityDeletionLayer(
            d2, device=reference.device, dtype=reference.dtype
        )
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        if self.node_emb is not None:
            for parameter in self.node_emb.parameters():
                parameter.requires_grad = False
        for parameter in self.conv1.parameters():
            parameter.requires_grad = False
        for parameter in self.conv2.parameters():
            parameter.requires_grad = False
        for parameter in self.deletion1.parameters():
            parameter.requires_grad = True
        for parameter in self.deletion2.parameters():
            parameter.requires_grad = True

    def matched_trainable_parameter_names(self) -> Tuple[str, ...]:
        return tuple(
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        )

    def identity_init_exact(self) -> bool:
        expected1 = torch.eye(
            self.deletion1.dim,
            device=self.deletion1.deletion_weight.device,
            dtype=self.deletion1.deletion_weight.dtype,
        )
        expected2 = torch.eye(
            self.deletion2.dim,
            device=self.deletion2.deletion_weight.device,
            dtype=self.deletion2.deletion_weight.dtype,
        )
        return bool(
            torch.equal(self.deletion1.deletion_weight, expected1)
            and torch.equal(self.deletion2.deletion_weight, expected2)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        support1: torch.Tensor,
        support2: torch.Tensor,
        return_all_emb: bool = False,
    ):
        h0 = self.common_input_preparation(x)
        h1_base = self.conv1(h0, edge_index)
        h1_student = self.deletion1(h1_base, support1)
        h1_processed = self.official_intermediate_processing(h1_student)
        h2_base = self.conv2(h1_processed, edge_index)
        z_student = self.deletion2(h2_base, support2)
        if return_all_emb:
            return h1_student, z_student
        return z_student
