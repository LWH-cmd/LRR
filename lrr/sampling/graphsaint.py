from __future__ import annotations

import torch
from torch_geometric.loader import GraphSAINTRandomWalkSampler


class WindowsSafeGraphSAINTRandomWalkSampler(GraphSAINTRandomWalkSampler):
    """Windows-compatible backend for the same GraphSAINT random-walk rule."""

    def __init__(self, *args, num_workers: int = 0, **kwargs):
        if num_workers != 0:
            raise ValueError("the safe backend requires num_workers=0")
        super().__init__(*args, num_workers=0, **kwargs)

    def _sample_nodes(self, batch_size: int) -> torch.Tensor:
        start = torch.randint(0, self.N, (batch_size,), dtype=torch.long)
        rowptr = self.adj.storage.rowptr()
        col = self.adj.storage.col()
        walk = torch.empty((batch_size, self.walk_length + 1), dtype=torch.long)
        walk[:, 0] = start
        current = start
        for walk_step in range(self.walk_length):
            left = rowptr[current]
            degree = rowptr[current + 1] - left
            if col.numel() == 0:
                next_node = current
            else:
                safe_degree = degree.clamp_min(1)
                choice = torch.floor(torch.rand(batch_size) * safe_degree.float()).long()
                safe_index = (left + choice).clamp_max(col.numel() - 1)
                next_node = col[safe_index]
                current = torch.where(degree > 0, next_node, current)
            walk[:, walk_step + 1] = current
        return walk.reshape(-1)
