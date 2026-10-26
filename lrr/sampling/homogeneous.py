from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import numpy as np
from torch_geometric.utils import negative_sampling


class RetainNegativeSchedule:
    """Tensor or portable, hash-checked frozen NPY shards; one step on device."""

    def __init__(self, source, *, base_path: str | Path = ".") -> None:
        self.source = source
        self.base_path = Path(base_path)
        self._maps = {}
        self.sampler = None
        if isinstance(source, dict) and source.get("format") == "role_seeded_retain_v1":
            from contracts.sampling import SharedSampler, SamplingIdentity as ContractIdentity
            self.shape = tuple(source["shape"])
            self.sampler = SharedSampler(ContractIdentity(**source["identity"]),
                full_train_mp_edge_index=source["full"], dr_mp_edge_index=source["dr"],
                local_mp_mask=source["local_mask"],directed_df_edge_index=source["df"],num_nodes=source["num_nodes"])
            return
        if torch.is_tensor(source):
            self.shape = tuple(source.shape)
        else:
            if source.get("format") != "lrr_npy_schedule_v1":
                raise ValueError("unsupported retained-negative schedule format")
            self.shape = tuple(source["shape"])
            cursor = 0
            for shard in source["shards"]:
                if shard["start"] != cursor or shard["stop"] <= cursor:
                    raise ValueError("schedule shards must cover contiguous updates")
                cursor = shard["stop"]
            if cursor != self.shape[0]:
                raise ValueError("schedule shard coverage differs from shape")
        if len(self.shape) != 3 or self.shape[1] != 2:
            raise ValueError("retained-negative schedule must be [updates,2,N]")

    def size(self, axis: int) -> int:
        return self.shape[axis]

    def close(self) -> None:
        """Release NPY file mappings, including Windows file handles."""
        for mapped in self._maps.values():
            mapped._mmap.close()
        self._maps.clear()

    def validate(self) -> None:
        """Check every shard before a runtime timer can start."""
        if not torch.is_tensor(self.source) and self.sampler is None:
            for shard in self.source["shards"]:
                self[shard["start"]]

    def __getitem__(self, update: int) -> torch.Tensor:
        if not 0 <= update < self.shape[0]:
            raise IndexError(update)
        if self.sampler is not None:
            return self.sampler.sample_epoch(update).retain_negative
        if torch.is_tensor(self.source):
            return self.source[update].long()
        for shard in self.source["shards"]:
            if shard["start"] <= update < shard["stop"]:
                path = self.base_path / shard["path"]
                key = str(path)
                if key not in self._maps:
                    digest = hashlib.sha256()
                    with path.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                    if digest.hexdigest() != shard["sha256"]:
                        raise ValueError(f"retained-negative schedule hash mismatch: {path}")
                    value = np.load(path, mmap_mode="r", allow_pickle=False)
                    expected = (shard["stop"] - shard["start"], 2, self.shape[2])
                    if value.shape != expected or value.dtype not in (np.int32, np.int64):
                        raise ValueError("invalid schedule shard shape or dtype")
                    self._maps[key] = value
                return torch.from_numpy(np.array(self._maps[key][update - shard["start"]],
                                                 dtype=np.int64, copy=True))
        raise IndexError(update)


@dataclass(frozen=True)
class SamplingIdentity:
    dataset: str
    backbone: str
    deletion_mode: str
    deletion_ratio: str
    run_seed: int


def role_seed(identity: SamplingIdentity, role: str, epoch: int) -> int:
    if role not in {"forget_negative", "retain_negative"}:
        raise ValueError(f"unsupported epoch sampling role: {role}")
    key = json.dumps({
        "dataset": identity.dataset,
        "backbone": identity.backbone,
        "deletion_mode": identity.deletion_mode,
        "deletion_ratio": identity.deletion_ratio,
        "run_seed": int(identity.run_seed),
        "epoch_or_static_marker": int(epoch),
        "sample_role": role,
    }, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


@contextmanager
def isolated_cpu_rng(seed: int):
    python_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    try:
        random.seed(int(seed))
        torch.random.default_generator.manual_seed(int(seed))
        yield
    finally:
        random.setstate(python_state)
        torch.random.set_rng_state(torch_state)


class ForgetNegativeSampler:
    def __init__(self, full_train_mp_edge_index: torch.Tensor, num_nodes: int,
                 requested_count: int, identity: SamplingIdentity) -> None:
        self.edge_index = full_train_mp_edge_index.detach().cpu().long()
        self.num_nodes = int(num_nodes)
        self.requested_count = int(requested_count)
        self.identity = identity

    def sample(self, update: int) -> torch.Tensor:
        seed = role_seed(self.identity, "forget_negative", update)
        with isolated_cpu_rng(seed):
            result = negative_sampling(
                edge_index=self.edge_index, num_nodes=self.num_nodes,
                num_neg_samples=self.requested_count, method="sparse",
                force_undirected=False,
            )
        if result.size(1) != self.requested_count:
            raise RuntimeError("Forget negative sampler returned a short sample")
        return result
