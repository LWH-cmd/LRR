from __future__ import annotations

import torch
from torch import nn


class FullRankResidual(nn.Module):
    """Dense-square LRR residual: H = H0 + Diag(g) H0 W."""

    def __init__(self, dim: int, *, device=None, dtype=None) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = int(dim)
        self.W = nn.Parameter(torch.zeros(self.dim, self.dim, device=device, dtype=dtype))

    def forward(self, h0: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if h0.ndim != 2 or h0.size(1) != self.dim:
            raise ValueError(f"expected [N,{self.dim}] input")
        if gate.shape != (h0.size(0), 1) or gate.device != h0.device:
            raise ValueError("gate must be [N,1] on the input device")
        if gate.dtype != h0.dtype:
            raise TypeError("gate and representation dtypes must match")
        return h0 + gate * h0.matmul(self.W)
