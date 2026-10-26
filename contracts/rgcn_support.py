from __future__ import annotations
from dataclasses import dataclass
import torch
NUM_NODES = 40943


@dataclass(frozen=True)
class SupportGateBundle:
    support1: torch.Tensor
    gate1: torch.Tensor
    support2: torch.Tensor
    gate2: torch.Tensor
    distance: torch.Tensor

def build_support_gates(*, full_d: torch.Tensor, df: torch.Tensor, num_nodes: int = NUM_NODES) -> SupportGateBundle:
    """Frozen full-D undirected hop support with R3 gates."""

    adjacency: list[set[int]] = [set() for _ in range(int(num_nodes))]
    for source, target in full_d.detach().cpu().t().tolist():
        adjacency[source].add(target)
        adjacency[target].add(source)
    endpoints = sorted(set(df.detach().cpu().reshape(-1).tolist()))
    if not endpoints:
        raise ValueError("Df endpoints are empty")
    distance = [-1] * int(num_nodes)
    frontier = endpoints
    for node in endpoints:
        distance[node] = 0
    for hop in (1, 2):
        next_frontier: list[int] = []
        for node in frontier:
            for neighbor in sorted(adjacency[node]):
                if distance[neighbor] == -1:
                    distance[neighbor] = hop
                    next_frontier.append(neighbor)
        frontier = next_frontier
    dist = torch.tensor(distance, dtype=torch.long)
    support1 = (dist >= 0) & (dist <= 1)
    support2 = (dist >= 0) & (dist <= 2)
    gate1 = torch.zeros((num_nodes, 1), dtype=torch.float32)
    gate2 = torch.zeros((num_nodes, 1), dtype=torch.float32)
    gate1[dist == 0] = 1.0
    gate1[dist == 1] = 0.7
    gate2[dist == 0] = 1.0
    gate2[dist == 1] = 0.7
    gate2[dist == 2] = 0.3
    return SupportGateBundle(support1, gate1, support2, gate2, dist)
