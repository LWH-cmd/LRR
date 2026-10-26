from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.utils import k_hop_subgraph


@dataclass(frozen=True)
class GateBundle:
    support1: torch.Tensor
    support2: torch.Tensor
    gate1: torch.Tensor
    gate2: torch.Tensor


def validate_gates(endpoint_nodes: torch.Tensor, support1: torch.Tensor,
                   support2: torch.Tensor, gate1: torch.Tensor,
                   gate2: torch.Tensor) -> None:
    """Check the formal B0/B1/B2 gate contract once before training."""
    if support1.ndim != 1 or support1.dtype != torch.bool or support2.shape != support1.shape or support2.dtype != torch.bool:
        raise TypeError("support tensors must be aligned one-dimensional bool masks")
    if gate1.shape != (support1.numel(), 1) or gate2.shape != gate1.shape:
        raise ValueError("gate tensors must have shape [num_nodes,1]")
    if not (support1.device == support2.device == gate1.device == gate2.device):
        raise ValueError("support and gate tensors must share a device")
    endpoints = torch.zeros_like(support1)
    endpoints[endpoint_nodes.to(device=support1.device, dtype=torch.long).unique()] = True
    if bool((endpoints & ~support1).any()) or bool((support1 & ~support2).any()):
        raise ValueError("formal supports must contain endpoints and be nested")
    expected1 = torch.zeros_like(gate1)
    expected2 = torch.zeros_like(gate2)
    expected1[endpoints, 0] = 1.0
    expected1[support1 & ~endpoints, 0] = 0.7
    expected2[endpoints, 0] = 1.0
    expected2[support1 & ~endpoints, 0] = 0.7
    expected2[support2 & ~support1, 0] = 0.3
    if not torch.equal(gate1, expected1) or not torch.equal(gate2, expected2):
        raise ValueError("gate values do not match the paper-main B0/B1/B2 contract")


def _make_gates(endpoint_mask: torch.Tensor, support1: torch.Tensor,
                support2: torch.Tensor, *, device: torch.device,
                dtype: torch.dtype = torch.float32) -> GateBundle:
    endpoint = endpoint_mask.to(device)
    s1, s2 = support1.to(device), support2.to(device)
    gate1 = torch.zeros((endpoint.numel(), 1), dtype=dtype, device=device)
    gate2 = torch.zeros_like(gate1)
    gate1[endpoint, 0] = 1.0
    gate1[s1 & ~endpoint, 0] = 0.7
    gate2[endpoint, 0] = 1.0
    gate2[s1 & ~endpoint, 0] = 0.7
    gate2[s2 & ~s1, 0] = 0.3
    return GateBundle(s1, s2, gate1, gate2)


def build_gcn_gates(full_train_directed_edge_index: torch.Tensor,
                    df_edge_index: torch.Tensor, num_nodes: int,
                    device: torch.device, dtype: torch.dtype = torch.float32) -> GateBundle:
    """Paper topology support from full training D and the requested endpoints."""
    if df_edge_index.numel() == 0:
        raise ValueError("request must contain at least one directed edge")
    endpoints = df_edge_index.flatten().unique()
    _, one_hop, _, _ = k_hop_subgraph(endpoints, 1, full_train_directed_edge_index,
                                      num_nodes=int(num_nodes))
    _, two_hop, _, _ = k_hop_subgraph(endpoints, 2, full_train_directed_edge_index,
                                      num_nodes=int(num_nodes))
    support1 = torch.zeros(int(num_nodes), dtype=torch.bool)
    support2 = torch.zeros_like(support1)
    support1[one_hop.flatten().unique()] = True
    support2[two_hop.flatten().unique()] = True
    endpoint_mask = torch.zeros(int(num_nodes), dtype=torch.bool)
    endpoint_mask[endpoints] = True
    return _make_gates(endpoint_mask, support1, support2, device=device, dtype=dtype)


def build_rgcn_gates(full_train_edge_index: torch.Tensor, df_edge_index: torch.Tensor,
                     num_nodes: int, device: torch.device,
                     dtype: torch.dtype = torch.float32) -> GateBundle:
    """Paper RGCN gate: two-hop support on undirected entity topology."""
    adjacency: list[set[int]] = [set() for _ in range(int(num_nodes))]
    for source, target in full_train_edge_index.detach().cpu().t().tolist():
        adjacency[source].add(target)
        adjacency[target].add(source)
    endpoints = sorted(set(df_edge_index.detach().cpu().reshape(-1).tolist()))
    if not endpoints:
        raise ValueError("request must contain at least one triple")
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
    endpoint_mask = dist == 0
    return _make_gates(endpoint_mask, support1, support2, device=device, dtype=dtype)
