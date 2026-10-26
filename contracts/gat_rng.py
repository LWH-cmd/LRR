from __future__ import annotations
import contextlib, hashlib, json, random
from typing import Iterator
import numpy as np
import torch


def child_seed(dataset: str, setting_seed: int, role: str, epoch: int | None = None) -> int:
    value = {"schema": "phase3b_rng_v1", "dataset": dataset, "setting_seed": int(setting_seed), "role": role, "epoch": epoch}
    return int.from_bytes(hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).digest()[:8], "big") % (2**63 - 1)

@contextlib.contextmanager
def isolated_rng(seed: int) -> Iterator[None]:
    py_state = random.getstate(); np_state = np.random.get_state(); cpu_state = torch.random.get_rng_state().clone()
    cuda_states = [x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() and torch.cuda.is_initialized() else None
    try:
        random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(py_state); np.random.set_state(np_state); torch.random.set_rng_state(cpu_state)
        if cuda_states is not None: torch.cuda.set_rng_state_all(cuda_states)
