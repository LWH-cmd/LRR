"""Frozen support/gate construction from full D, never from Dr."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SupportGateBundle:
    support1: torch.Tensor
    gate1: torch.Tensor
    support2: torch.Tensor
    gate2: torch.Tensor
    distance_from_df: torch.Tensor


def _validate_edge_index(edge_index: torch.Tensor, *, num_nodes: int, name: str) -> None:
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"{name} must have shape [2,E]")
    if edge_index.dtype != torch.long:
        raise TypeError(f"{name} must use torch.long")
    if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes):
        raise ValueError(f"{name} contains a node outside [0,{num_nodes})")


def build_support_gates(
    *,
    num_nodes: int,
    full_d_edge_index: torch.Tensor,
    df_edge_index: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> SupportGateBundle:
    """Build endpoint/strict-hop gates on the undirected full-D topology."""

    num_nodes = int(num_nodes)
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    _validate_edge_index(full_d_edge_index, num_nodes=num_nodes, name="full_d_edge_index")
    _validate_edge_index(df_edge_index, num_nodes=num_nodes, name="df_edge_index")
    if not dtype.is_floating_point:
        raise TypeError("gate dtype must be floating")
    device = full_d_edge_index.device
    if df_edge_index.device != device:
        raise TypeError("full D and Df edge indices must share a device")

    adjacency: list[set[int]] = [set() for _ in range(num_nodes)]
    for source, target in full_d_edge_index.detach().cpu().t().tolist():
        adjacency[source].add(target)
        adjacency[target].add(source)
    endpoints = sorted(set(df_edge_index.detach().cpu().reshape(-1).tolist()))
    if not endpoints:
        raise ValueError("Df must contain at least one endpoint")

    distance = [-1] * num_nodes
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

    distance_tensor = torch.tensor(distance, device=device, dtype=torch.long)
    support1 = (distance_tensor >= 0) & (distance_tensor <= 1)
    support2 = (distance_tensor >= 0) & (distance_tensor <= 2)
    gate1 = torch.zeros((num_nodes, 1), device=device, dtype=dtype)
    gate2 = torch.zeros((num_nodes, 1), device=device, dtype=dtype)
    gate1[distance_tensor == 0] = 1.0
    gate1[distance_tensor == 1] = 0.7
    gate2[distance_tensor == 0] = 1.0
    gate2[distance_tensor == 1] = 0.7
    gate2[distance_tensor == 2] = 0.3
    return SupportGateBundle(
        support1=support1,
        gate1=gate1,
        support2=support2,
        gate2=gate2,
        distance_from_df=distance_tensor,
    )
