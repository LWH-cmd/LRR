"""Runtime-only adapters for the frozen homogeneous-GCN LRR contract.

This module does not define an objective or a trainer.  It only reproduces the
official GNNDelete support construction, separates graph domains, and builds
the Phase 0B gate buckets used by runtime audits.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict

import torch
from torch_geometric.utils import is_undirected, k_hop_subgraph, to_undirected


@dataclass(frozen=True)
class OfficialSupport:
    endpoint_mask: torch.Tensor
    support1: torch.Tensor
    support2: torch.Tensor
    official_sdf_mask: torch.Tensor


@dataclass(frozen=True)
class MessagePassingDomains:
    full_train_mp_edge_index: torch.Tensor
    df_mp_mask: torch.Tensor
    dr_mp_mask: torch.Tensor
    dr_mp_edge_index: torch.Tensor
    official_sdf_mp_mask: torch.Tensor
    local_loss_edge_mask: torch.Tensor


@dataclass(frozen=True)
class GateBuckets:
    bucket0: torch.Tensor
    bucket1: torch.Tensor
    bucket2: torch.Tensor
    outside1: torch.Tensor
    outside2: torch.Tensor
    gate1: torch.Tensor
    gate2: torch.Tensor


def tensor_hash_v1(tensor: torch.Tensor) -> str:
    """Return the Phase 0B TensorHashV1 digest for a supported dense tensor."""
    if tensor.is_sparse or tensor.is_quantized or tensor.is_complex():
        raise TypeError("TensorHashV1 requires a dense, non-quantized real tensor")
    value = tensor.detach().cpu().contiguous()
    dtype_name = str(value.dtype).removeprefix("torch.")
    header = {
        "tensor_hash_version": "tensor_hash_v1",
        "dtype": dtype_name,
        "shape": list(value.shape),
        "byte_order": "little",
    }
    header_bytes = json.dumps(
        header, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    raw = value.numpy().tobytes(order="C")
    return hashlib.sha256(header_bytes + b"\n" + raw).hexdigest()


def reproduce_delete_gnn_df_indices(
    candidate_mask: torch.Tensor,
    directed_train_edge_count: int,
    df_size: float,
) -> torch.Tensor:
    """Reproduce delete_gnn.py's seeded randperm selection.

    The caller must seed the global PyTorch RNG exactly as delete_gnn.py does
    before invoking this function.
    """
    if candidate_mask.dtype != torch.bool or candidate_mask.dim() != 1:
        raise TypeError("candidate_mask must be a one-dimensional bool tensor")
    if candidate_mask.numel() != int(directed_train_edge_count):
        raise ValueError("candidate_mask length must equal directed train edge count")
    if float(df_size) >= 100:
        requested_count = int(df_size)
    else:
        requested_count = int(
            float(df_size) / 100 * int(directed_train_edge_count)
        )
    candidate_indices = candidate_mask.nonzero().squeeze()
    if candidate_indices.dim() != 1:
        raise ValueError("official candidate selection requires at least two candidates")
    if requested_count <= 0 or requested_count > candidate_indices.numel():
        raise ValueError(
            f"invalid requested Df count {requested_count} for "
            f"{candidate_indices.numel()} candidates"
        )
    selected = torch.randperm(candidate_indices.shape[0])[:requested_count]
    return candidate_indices[selected]


def _support_from_endpoints(
    endpoint_nodes: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> OfficialSupport:
    if edge_index.dtype != torch.int64 or edge_index.dim() != 2:
        raise TypeError("edge_index must be an int64 tensor with shape [2,E]")
    if edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2,E]")
    if endpoint_nodes.dtype != torch.int64 or endpoint_nodes.dim() != 1:
        raise TypeError("endpoint_nodes must be a one-dimensional int64 tensor")
    if endpoint_nodes.numel() == 0:
        raise ValueError("effective Df must contain at least one endpoint")
    if int(endpoint_nodes.min()) < 0 or int(endpoint_nodes.max()) >= int(num_nodes):
        raise ValueError("Df endpoint node ID is out of range")

    _, two_hop_edge, _, two_hop_mask = k_hop_subgraph(
        endpoint_nodes,
        2,
        edge_index,
        num_nodes=int(num_nodes),
    )
    _, one_hop_edge, _, _ = k_hop_subgraph(
        endpoint_nodes,
        1,
        edge_index,
        num_nodes=int(num_nodes),
    )

    support1 = torch.zeros(int(num_nodes), dtype=torch.bool)
    support2 = torch.zeros(int(num_nodes), dtype=torch.bool)
    support1[one_hop_edge.flatten().unique()] = True
    support2[two_hop_edge.flatten().unique()] = True
    endpoint_mask = torch.zeros(int(num_nodes), dtype=torch.bool)
    endpoint_mask[endpoint_nodes] = True
    return OfficialSupport(
        endpoint_mask=endpoint_mask,
        support1=support1,
        support2=support2,
        official_sdf_mask=two_hop_mask,
    )


def build_official_gnndelete_support(
    full_train_directed_edge_index: torch.Tensor,
    directed_df_mask: torch.Tensor,
    num_nodes: int,
) -> OfficialSupport:
    """Mirror delete_gnn.py's support construction on full directed D."""
    if directed_df_mask.dtype != torch.bool or directed_df_mask.dim() != 1:
        raise TypeError("directed_df_mask must be one-dimensional bool")
    if directed_df_mask.numel() != full_train_directed_edge_index.size(1):
        raise ValueError("directed_df_mask must use the directed edge-column domain")
    directed_df_edge_index = full_train_directed_edge_index[:, directed_df_mask]
    if directed_df_edge_index.size(1) == 0:
        raise ValueError("effective directed Df must be non-empty")
    endpoint_nodes = directed_df_edge_index.flatten().unique()
    return _support_from_endpoints(
        endpoint_nodes, full_train_directed_edge_index, num_nodes
    )


def build_dr_support_nonformal_diagnostic(
    dr_mp_edge_index: torch.Tensor,
    endpoint_mask: torch.Tensor,
    num_nodes: int,
) -> OfficialSupport:
    """Build the prohibited Dr-recomputed comparison for diagnostics only."""
    endpoint_nodes = endpoint_mask.nonzero().flatten()
    return _support_from_endpoints(endpoint_nodes, dr_mp_edge_index, num_nodes)


def build_message_passing_domains(
    full_train_directed_edge_index: torch.Tensor,
    directed_df_mask: torch.Tensor,
    official_sdf_mask: torch.Tensor,
) -> MessagePassingDomains:
    """Reproduce delete_gnn.py's directed-to-undirected mask conversion."""
    edge_count = full_train_directed_edge_index.size(1)
    for name, mask in (
        ("directed_df_mask", directed_df_mask),
        ("official_sdf_mask", official_sdf_mask),
    ):
        if mask.dtype != torch.bool or mask.dim() != 1 or mask.numel() != edge_count:
            raise ValueError(
                f"{name} must be bool with the directed edge-column shape"
            )

    full_train_mp_edge_index, attrs = to_undirected(
        full_train_directed_edge_index,
        [directed_df_mask.int(), official_sdf_mask.int()],
    )
    df_mp_mask = attrs[0].bool()
    official_sdf_mp_mask = attrs[1].bool()
    dr_mp_mask = ~df_mp_mask
    dr_mp_edge_index = full_train_mp_edge_index[:, dr_mp_mask]
    local_loss_edge_mask = official_sdf_mp_mask & dr_mp_mask

    if not is_undirected(full_train_mp_edge_index):
        raise RuntimeError("full_train_mp_edge_index must be undirected")
    if not is_undirected(dr_mp_edge_index):
        raise RuntimeError("dr_mp_edge_index must be undirected")
    if bool((local_loss_edge_mask & df_mp_mask).any()):
        raise RuntimeError("local loss edges must not contain Df")

    return MessagePassingDomains(
        full_train_mp_edge_index=full_train_mp_edge_index,
        df_mp_mask=df_mp_mask,
        dr_mp_mask=dr_mp_mask,
        dr_mp_edge_index=dr_mp_edge_index,
        official_sdf_mp_mask=official_sdf_mp_mask,
        local_loss_edge_mask=local_loss_edge_mask,
    )


def build_gate_buckets(
    endpoint_mask: torch.Tensor,
    support1: torch.Tensor,
    support2: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
) -> GateBuckets:
    """Build the frozen B0/B1/B2 partition and the two mandatory gates."""
    masks = {
        "endpoint_mask": endpoint_mask,
        "support1": support1,
        "support2": support2,
    }
    expected_size = endpoint_mask.numel()
    for name, mask in masks.items():
        if mask.dtype != torch.bool or mask.dim() != 1:
            raise TypeError(f"{name} must be one-dimensional bool")
        if mask.numel() != expected_size:
            raise ValueError("endpoint/support masks must have the same shape")
    if bool((endpoint_mask & ~support1).any()):
        raise ValueError("endpoint_mask must be a subset of support1")
    if bool((support1 & ~support2).any()):
        raise ValueError("support1 must be a subset of support2")
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("gate dtype must be float32 or float64")

    bucket0 = endpoint_mask.clone()
    bucket1 = support1 & ~bucket0
    bucket2 = support2 & ~support1
    outside1 = ~support1
    outside2 = ~support2

    if bool((bucket0 & bucket1).any()):
        raise RuntimeError("B0 and B1 must be disjoint")
    if bool((bucket0 & bucket2).any()):
        raise RuntimeError("B0 and B2 must be disjoint")
    if bool((bucket1 & bucket2).any()):
        raise RuntimeError("B1 and B2 must be disjoint")
    if not torch.equal(bucket0 | bucket1, support1):
        raise RuntimeError("B0 union B1 must equal support1")
    if not torch.equal(bucket0 | bucket1 | bucket2, support2):
        raise RuntimeError("B0 union B1 union B2 must equal support2")

    target_device = torch.device(device)
    b0 = bucket0.to(target_device)
    b1 = bucket1.to(target_device)
    b2 = bucket2.to(target_device)
    gate1 = torch.zeros(expected_size, 1, dtype=dtype, device=target_device)
    gate2 = torch.zeros(expected_size, 1, dtype=dtype, device=target_device)
    gate1[b0, 0] = 1.0
    gate1[b1, 0] = 0.7
    gate2[b0, 0] = 1.0
    gate2[b1, 0] = 0.7
    gate2[b2, 0] = 0.3

    if gate1 is gate2:
        raise RuntimeError("gate1 and gate2 must be distinct tensors")
    if not bool(torch.isfinite(gate1).all() and torch.isfinite(gate2).all()):
        raise RuntimeError("gate values must be finite")
    if torch.count_nonzero(gate1[outside1.to(target_device)]).item() != 0:
        raise RuntimeError("gate1 must be exactly zero outside support1")
    if torch.count_nonzero(gate2[outside2.to(target_device)]).item() != 0:
        raise RuntimeError("gate2 must be exactly zero outside support2")

    return GateBuckets(
        bucket0=bucket0,
        bucket1=bucket1,
        bucket2=bucket2,
        outside1=outside1,
        outside2=outside2,
        gate1=gate1,
        gate2=gate2,
    )


def mask_summary(mask: torch.Tensor) -> Dict[str, object]:
    return {
        "shape": list(mask.shape),
        "dtype": str(mask.dtype),
        "count": int(mask.count_nonzero().item()),
        "tensor_hash_v1": tensor_hash_v1(mask),
    }
