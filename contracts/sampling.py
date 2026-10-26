"""Deterministic shared sampling for matched homogeneous-GCN methods."""

from __future__ import annotations

import hashlib
import json
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch_geometric.utils import negative_sampling

from .lrr_runtime import tensor_hash_v1


MATCHED_SAMPLER_VERSION = (
    "deterministic_index_selection_v1__anchor_child_stream_v1__"
    "pyg_rng_isolation_v1"
)
NEGATIVE_SAMPLING_CONTRACT_VERSION = "pyg_2_6_1_sparse_directed_v1"
STATIC_BUNDLE_SCHEMA_VERSION = "static_sample_bundle_v1"
EVALUATION_BUNDLE_VERSION = "evaluation_candidate_bundle_v1"
MIXED_ANCHOR_POLICY_VERSION = "canonical_dedup_mixed_anchor_v1"
ANCHOR_CHILD_STREAM_VERSION = "anchor_child_stream_v1"
MAX_LOCAL_EDGES = 50_000
MAX_RETAIN_EDGES = 50_000
MAX_ANCHOR_EDGES = 50_000
_SEED_MODULUS = 2**63 - 1
_KEY_FIELDS = (
    "dataset",
    "backbone",
    "deletion_mode",
    "deletion_ratio",
    "run_seed",
    "epoch_or_static_marker",
    "sample_role",
)
_STATIC_ROLES = {"local_static", "retain_positive_static", "anchor_static"}
_EPOCH_ROLES = {"forget_negative", "retain_negative"}


@dataclass(frozen=True)
class SamplingIdentity:
    dataset: str
    backbone: str
    deletion_mode: str
    deletion_ratio: str
    run_seed: int


@dataclass(frozen=True)
class StaticSampleBundle:
    local_static: torch.Tensor
    retain_positive_static: torch.Tensor
    anchor_static: torch.Tensor
    metadata: Dict[str, object]


@dataclass(frozen=True)
class EpochNegativeBundle:
    forget_negative: torch.Tensor
    retain_negative: torch.Tensor
    metadata: Dict[str, object]


@dataclass(frozen=True)
class EvaluationCandidateBundle:
    validation_pos: torch.Tensor
    validation_neg: torch.Tensor
    test_pos: torch.Tensor
    test_neg: torch.Tensor
    df_retained_group_positions: torch.Tensor
    metadata: Dict[str, object]


def _canonical_json(value: Dict[str, object]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def canonical_sampling_key(
    identity: SamplingIdentity,
    *,
    sample_role: str,
    epoch: Optional[int] = None,
) -> str:
    if sample_role in _STATIC_ROLES:
        if epoch is not None:
            raise ValueError("static sampling roles must not receive an epoch")
        marker: object = "static"
    elif sample_role in _EPOCH_ROLES:
        if epoch is None or int(epoch) < 0:
            raise ValueError("per-epoch sampling roles require a non-negative epoch")
        marker = int(epoch)
    else:
        raise ValueError(f"unsupported matched sampling role {sample_role!r}")
    values: Dict[str, object] = {
        "dataset": identity.dataset,
        "backbone": identity.backbone,
        "deletion_mode": identity.deletion_mode,
        "deletion_ratio": identity.deletion_ratio,
        "run_seed": int(identity.run_seed),
        "epoch_or_static_marker": marker,
        "sample_role": sample_role,
    }
    if tuple(values) != _KEY_FIELDS:
        raise RuntimeError("canonical sampling key field order changed")
    return _canonical_json(values)


def seed_from_canonical_key(canonical_key: str) -> Tuple[int, str]:
    digest = hashlib.sha256(canonical_key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False) % _SEED_MODULUS
    return seed, digest.hex()


def anchor_child_seed(parent_key: str, stage_token: str) -> Tuple[int, str]:
    if stage_token not in {"local_select", "global_fill"}:
        raise ValueError("anchor stage token must be local_select or global_fill")
    parent_digest = hashlib.sha256(parent_key.encode("utf-8")).digest()
    child_digest = hashlib.sha256(
        parent_digest + b"\n" + stage_token.encode("utf-8")
    ).digest()
    seed = (
        int.from_bytes(child_digest[:8], byteorder="big", signed=False)
        % _SEED_MODULUS
    )
    return seed, child_digest.hex()


def deterministic_index_selection(
    candidate_count: int, requested_count: int, derived_seed: int
) -> torch.Tensor:
    candidate_count = int(candidate_count)
    requested_count = int(requested_count)
    if candidate_count < 0 or requested_count < 0:
        raise ValueError("candidate/requested counts must be non-negative")
    if requested_count >= candidate_count:
        return torch.arange(candidate_count, dtype=torch.int64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(derived_seed))
    return torch.randperm(
        candidate_count, generator=generator, device="cpu"
    )[:requested_count]


@contextmanager
def _isolated_global_rng(seed: int) -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    cuda_initialized = bool(torch.cuda.is_available() and torch.cuda.is_initialized())
    cuda_states = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if cuda_initialized
        else None
    )
    try:
        random.seed(int(seed))
        np.random.seed(int(seed) % 2**32)
        torch.manual_seed(int(seed))
        if cuda_initialized:
            torch.cuda.manual_seed_all(int(seed))
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_initialized and cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def isolated_pyg_negative_sampling(
    *,
    edge_index: torch.Tensor,
    num_nodes: int,
    num_neg_samples: int,
    derived_seed: int,
) -> torch.Tensor:
    if edge_index.device.type != "cpu":
        raise ValueError("shared negative sampling requires a CPU exclusion graph")
    requested = int(num_neg_samples)
    with _isolated_global_rng(int(derived_seed)):
        sampled = negative_sampling(
            edge_index=edge_index,
            num_nodes=int(num_nodes),
            num_neg_samples=requested,
            method="sparse",
            force_undirected=False,
        )
    if sampled.shape != (2, requested):
        raise RuntimeError(
            f"NEGATIVE_SAMPLE_SHORTAGE: requested {requested}, "
            f"received {sampled.size(1)}"
        )
    if bool((sampled[0] == sampled[1]).any()):
        raise RuntimeError("negative samples must not contain self-loops")
    positive_ids = edge_index[0] * int(num_nodes) + edge_index[1]
    negative_ids = sampled[0] * int(num_nodes) + sampled[1]
    if bool(torch.isin(negative_ids, positive_ids).any()):
        raise RuntimeError("negative sample overlaps the full MP exclusion graph")
    return sampled


def _select_edge_columns(
    edge_index: torch.Tensor, requested_count: int, seed: int
) -> torch.Tensor:
    positions = deterministic_index_selection(
        edge_index.size(1), requested_count, seed
    )
    return edge_index[:, positions]


def _canonical_unique_edges(
    edge_index: torch.Tensor,
    *,
    excluded_identities: Optional[Set[Tuple[int, int]]] = None,
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    excluded = excluded_identities if excluded_identities is not None else set()
    seen: Set[Tuple[int, int]] = set()
    kept_columns: List[int] = []
    kept_identities: List[Tuple[int, int]] = []
    rows = edge_index[0].tolist()
    cols = edge_index[1].tolist()
    for column, (source, target) in enumerate(zip(rows, cols)):
        identity = (
            (source, target) if source <= target else (target, source)
        )
        if identity in excluded or identity in seen:
            continue
        seen.add(identity)
        kept_columns.append(column)
        kept_identities.append(identity)
    if not kept_columns:
        return edge_index[:, :0], []
    positions = torch.tensor(kept_columns, dtype=torch.int64)
    return edge_index[:, positions], kept_identities


def _artifact_hash(fields: Sequence[Tuple[str, object]]) -> str:
    payload = _canonical_json({name: value for name, value in fields})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SharedSampler:
    """One sampler implementation for LRR and future matched GNNDelete."""

    def __init__(
        self,
        identity: SamplingIdentity,
        *,
        full_train_mp_edge_index: torch.Tensor,
        dr_mp_edge_index: torch.Tensor,
        local_mp_mask: torch.Tensor,
        directed_df_edge_index: torch.Tensor,
        num_nodes: int,
    ) -> None:
        self.identity = identity
        self.full_train_mp_edge_index = (
            full_train_mp_edge_index.detach().cpu().long()
        )
        self.dr_mp_edge_index = dr_mp_edge_index.detach().cpu().long()
        self.local_mp_mask = local_mp_mask.detach().cpu().bool()
        self.directed_df_edge_index = directed_df_edge_index.detach().cpu().long()
        self.num_nodes = int(num_nodes)
        if self.local_mp_mask.shape != (
            self.full_train_mp_edge_index.size(1),
        ):
            raise ValueError("local_mp_mask must use the full MP edge-column domain")
        if self.directed_df_edge_index.size(1) == 0:
            raise ValueError("effective Df must be non-empty")

    def _role_seed(
        self, sample_role: str, epoch: Optional[int] = None
    ) -> Tuple[int, str, str]:
        key = canonical_sampling_key(
            self.identity, sample_role=sample_role, epoch=epoch
        )
        seed, digest = seed_from_canonical_key(key)
        return seed, key, digest

    def build_static_bundle(self) -> StaticSampleBundle:
        full_local_pool = self.full_train_mp_edge_index[:, self.local_mp_mask]
        if full_local_pool.size(1) == 0:
            raise RuntimeError("non-zero local objective requires local edges")

        local_seed, local_key, local_key_sha = self._role_seed("local_static")
        local_static = _select_edge_columns(
            full_local_pool, MAX_LOCAL_EDGES, local_seed
        )
        retain_seed, retain_key, retain_key_sha = self._role_seed(
            "retain_positive_static"
        )
        retain_positive_static = _select_edge_columns(
            self.dr_mp_edge_index, MAX_RETAIN_EDGES, retain_seed
        )
        if retain_positive_static.size(1) == 0:
            raise RuntimeError("non-zero retain objective requires retained positives")

        anchor_key = canonical_sampling_key(
            self.identity, sample_role="anchor_static"
        )
        _, anchor_parent_sha = seed_from_canonical_key(anchor_key)
        local_child_seed, local_child_sha = anchor_child_seed(
            anchor_key, "local_select"
        )
        global_child_seed, global_child_sha = anchor_child_seed(
            anchor_key, "global_fill"
        )

        local_unique, local_identities = _canonical_unique_edges(full_local_pool)
        if local_unique.size(1) > MAX_ANCHOR_EDGES:
            local_positions = deterministic_index_selection(
                local_unique.size(1), MAX_ANCHOR_EDGES, local_child_seed
            )
            local_selected = local_unique[:, local_positions]
            selected_local_identities = [
                local_identities[int(position)] for position in local_positions
            ]
        else:
            local_selected = local_unique
            selected_local_identities = local_identities

        remaining = MAX_ANCHOR_EDGES - local_selected.size(1)
        global_unique, global_identities = _canonical_unique_edges(
            self.dr_mp_edge_index,
            excluded_identities=set(selected_local_identities),
        )
        if remaining > 0:
            global_positions = deterministic_index_selection(
                global_unique.size(1), remaining, global_child_seed
            )
            global_selected = global_unique[:, global_positions]
            selected_global_identities = [
                global_identities[int(position)] for position in global_positions
            ]
        else:
            global_selected = global_unique[:, :0]
            selected_global_identities = []
        anchor_static = torch.cat([local_selected, global_selected], dim=1)
        if anchor_static.size(1) == 0:
            raise RuntimeError("non-zero anchor objective requires anchor edges")

        all_identities = selected_local_identities + selected_global_identities
        if len(all_identities) != len(set(all_identities)):
            raise RuntimeError("canonical mixed anchor contains duplicate identities")
        anchor_shortfall = MAX_ANCHOR_EDGES - anchor_static.size(1)
        anchor_post_dedup_count = len(local_identities) + len(global_identities)

        metadata: Dict[str, object] = {
            "static_bundle_schema_version": STATIC_BUNDLE_SCHEMA_VERSION,
            "matched_sampler_version": MATCHED_SAMPLER_VERSION,
            "mixed_anchor_policy_version": MIXED_ANCHOR_POLICY_VERSION,
            "anchor_child_stream_version": ANCHOR_CHILD_STREAM_VERSION,
            "max_local_edges": MAX_LOCAL_EDGES,
            "max_retain_edges": MAX_RETAIN_EDGES,
            "max_anchor_edges": MAX_ANCHOR_EDGES,
            "full_local_candidate_count": int(full_local_pool.size(1)),
            "local_static_count": int(local_static.size(1)),
            "retain_positive_static_count": int(
                retain_positive_static.size(1)
            ),
            "anchor_total_count": int(anchor_static.size(1)),
            "anchor_component_counts": {
                "local_selected": int(local_selected.size(1)),
                "global_fill": int(global_selected.size(1)),
            },
            "anchor_pre_dedup_count": int(
                full_local_pool.size(1) + self.dr_mp_edge_index.size(1)
            ),
            "anchor_post_dedup_count": int(anchor_post_dedup_count),
            "anchor_deduplicated_count": int(
                full_local_pool.size(1)
                + self.dr_mp_edge_index.size(1)
                - anchor_post_dedup_count
            ),
            "anchor_shortfall_count": int(anchor_shortfall),
            "anchor_shortage_status": (
                "SHORTFALL_USE_ALL_AVAILABLE"
                if anchor_shortfall > 0
                else "NONE"
            ),
            "local_static_key": local_key,
            "local_static_key_sha256": local_key_sha,
            "retain_positive_static_key": retain_key,
            "retain_positive_static_key_sha256": retain_key_sha,
            "anchor_static_key": anchor_key,
            "anchor_static_key_sha256": anchor_parent_sha,
            "anchor_local_stage_key_sha256": local_child_sha,
            "anchor_global_stage_key_sha256": global_child_sha,
            "local_static_tensor_sha256": tensor_hash_v1(local_static),
            "retain_positive_static_tensor_sha256": tensor_hash_v1(
                retain_positive_static
            ),
            "anchor_static_tensor_sha256": tensor_hash_v1(anchor_static),
        }
        metadata["static_bundle_sha256"] = _artifact_hash(
            [
                ("static_bundle_schema_version", STATIC_BUNDLE_SCHEMA_VERSION),
                ("local_static", metadata["local_static_tensor_sha256"]),
                (
                    "retain_positive_static",
                    metadata["retain_positive_static_tensor_sha256"],
                ),
                ("anchor_static", metadata["anchor_static_tensor_sha256"]),
                ("anchor_shortfall_count", anchor_shortfall),
            ]
        )
        return StaticSampleBundle(
            local_static=local_static,
            retain_positive_static=retain_positive_static,
            anchor_static=anchor_static,
            metadata=metadata,
        )

    def sample_epoch(self, epoch: int, *, step: int = 0) -> EpochNegativeBundle:
        if int(step) != 0:
            raise ValueError("full-batch matched protocol has exactly step=0 per epoch")
        forget_seed, forget_key, forget_sha = self._role_seed(
            "forget_negative", int(epoch)
        )
        retain_seed, retain_key, retain_sha = self._role_seed(
            "retain_negative", int(epoch)
        )
        forget_negative = isolated_pyg_negative_sampling(
            edge_index=self.full_train_mp_edge_index,
            num_nodes=self.num_nodes,
            num_neg_samples=self.directed_df_edge_index.size(1),
            derived_seed=forget_seed,
        )
        retain_positive_count = min(
            self.dr_mp_edge_index.size(1), MAX_RETAIN_EDGES
        )
        retain_negative = isolated_pyg_negative_sampling(
            edge_index=self.full_train_mp_edge_index,
            num_nodes=self.num_nodes,
            num_neg_samples=retain_positive_count,
            derived_seed=retain_seed,
        )
        metadata = {
            "epoch": int(epoch),
            "step": 0,
            "forget_negative_key": forget_key,
            "forget_negative_key_sha256": forget_sha,
            "retain_negative_key": retain_key,
            "retain_negative_key_sha256": retain_sha,
            "forget_negative_tensor_sha256": tensor_hash_v1(forget_negative),
            "retain_negative_tensor_sha256": tensor_hash_v1(retain_negative),
            "negative_sampling_contract_version": (
                NEGATIVE_SAMPLING_CONTRACT_VERSION
            ),
        }
        return EpochNegativeBundle(
            forget_negative=forget_negative,
            retain_negative=retain_negative,
            metadata=metadata,
        )

    def build_evaluation_bundle(
        self,
        *,
        validation_pos: torch.Tensor,
        validation_neg: torch.Tensor,
        test_pos: torch.Tensor,
        test_neg: torch.Tensor,
        num_groups: int = 500,
    ) -> EvaluationCandidateBundle:
        group_size = self.directed_df_edge_index.size(1)
        candidate_count = self.dr_mp_edge_index.size(1)
        if candidate_count < group_size:
            raise RuntimeError("Dr column count is smaller than directed Df count")
        positions: List[torch.Tensor] = []
        position_hashes: List[str] = []
        edge_hashes: List[str] = []
        for group_index in range(int(num_groups)):
            key_values = {
                "evaluation_candidate_version": EVALUATION_BUNDLE_VERSION,
                "dataset": self.identity.dataset,
                "backbone": self.identity.backbone,
                "deletion_mode": self.identity.deletion_mode,
                "deletion_ratio_canonical": self.identity.deletion_ratio,
                "run_seed": int(self.identity.run_seed),
                "candidate_role": "df_retained_positive_group",
                "group_index": group_index,
            }
            key = _canonical_json(key_values)
            seed, _ = seed_from_canonical_key(key)
            selected = deterministic_index_selection(
                candidate_count, group_size, seed
            )
            positions.append(selected)
            position_hashes.append(tensor_hash_v1(selected))
            edge_hashes.append(
                tensor_hash_v1(self.dr_mp_edge_index[:, selected])
            )
        stacked_positions = torch.stack(positions, dim=0)
        group_list_sha = tensor_hash_v1(stacked_positions)
        metadata: Dict[str, object] = {
            "evaluation_candidate_version": EVALUATION_BUNDLE_VERSION,
            "evaluation_df_control_policy_version": (
                "shared_df_retained_groups_v1"
            ),
            "num_groups": int(num_groups),
            "group_size": int(group_size),
            "validation_dt_candidate_sha256": tensor_hash_v1(
                torch.cat([validation_pos, validation_neg], dim=1)
            ),
            "test_dt_candidate_sha256": tensor_hash_v1(
                torch.cat([test_pos, test_neg], dim=1)
            ),
            "directed_df_order_sha256": tensor_hash_v1(
                self.directed_df_edge_index
            ),
            "group_selected_position_hashes": position_hashes,
            "group_edge_tensor_hashes": edge_hashes,
            "shared_group_order": list(range(int(num_groups))),
            "shared_group_list_sha256": group_list_sha,
        }
        metadata["total_bundle_sha256"] = _artifact_hash(
            [
                (
                    "evaluation_candidate_version",
                    EVALUATION_BUNDLE_VERSION,
                ),
                (
                    "validation_dt_candidate_sha256",
                    metadata["validation_dt_candidate_sha256"],
                ),
                (
                    "test_dt_candidate_sha256",
                    metadata["test_dt_candidate_sha256"],
                ),
                (
                    "directed_df_order_sha256",
                    metadata["directed_df_order_sha256"],
                ),
                ("shared_group_list_sha256", group_list_sha),
            ]
        )
        return EvaluationCandidateBundle(
            validation_pos=validation_pos.detach().cpu().long(),
            validation_neg=validation_neg.detach().cpu().long(),
            test_pos=test_pos.detach().cpu().long(),
            test_neg=test_neg.detach().cpu().long(),
            df_retained_group_positions=stacked_positions,
            metadata=metadata,
        )
