from __future__ import annotations

import random
import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RelationalSamplingIdentity:
    dataset: str
    backbone: str
    deletion_mode: str
    deletion_ratio: str
    run_seed: int


def role_seed(identity: RelationalSamplingIdentity, role: str, update: int) -> int:
    if role not in {"forget_negative", "retain_negative"}:
        raise ValueError(f"unsupported epoch sampling role: {role}")
    key = json.dumps({
        "dataset": identity.dataset,
        "backbone": identity.backbone,
        "deletion_mode": identity.deletion_mode,
        "deletion_ratio": identity.deletion_ratio,
        "run_seed": int(identity.run_seed),
        "epoch_or_static_marker": int(update),
        "sample_role": role,
    }, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


@contextmanager
def isolated_rng(seed: int):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    try:
        random.seed(int(seed))
        np.random.seed(int(seed) % 2**32)
        torch.manual_seed(int(seed))
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def relation_aware_negative(edges: torch.Tensor, relations: torch.Tensor,
                            seed: int) -> torch.Tensor:
    """Paper relation-wise head permutation, preserving each (r,t) pair."""
    result = edges.detach().cpu().long().clone()
    relation_cpu = relations.detach().cpu().long()
    with isolated_rng(seed):
        for relation in relation_cpu.unique(sorted=True):
            mask = relation_cpu == relation
            heads = result[0, mask]
            result[0, mask] = heads[torch.randperm(heads.numel())]
    return result


class RelationalSampler:
    """Step-indexed formal relation-wise head permutation sampler."""

    def __init__(self, identity: RelationalSamplingIdentity) -> None:
        self.identity = identity

    def role_seed(self, role: str, update: int) -> int:
        return role_seed(self.identity, role, update)
