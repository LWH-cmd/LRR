"""Minimal FullRank smoke from the proven-success formal execution path.

This script is intentionally not the current crash wrapper.  It imports the
formal source first, uses the historical-success interpreter externally, and
changes only the operator parameterization after the native formal path has
been loaded.
"""

from __future__ import annotations

import argparse
import builtins
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
FORMAL_SOURCE = PROJECT_ROOT / "isolated_experiments/phase4c_wordnet18rr_p2_formal_seed42/src/run_formal_p2_crossed.py"
FULLRANK_SOURCE = ROOT / "runners/fullrank_operators.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module import failed: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach() - b.detach()).abs().max().item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    formal = load_module(FORMAL_SOURCE, "rgcn_proven_success_formal")
    fullrank = load_module(FULLRANK_SOURCE, "rgcn_proven_success_fullrank_operators")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda")
    formal.set_seed(formal.SEED)

    graph, base_relations, counts, train = formal.load_graph()
    if base_relations != formal.BASE_RELATIONS or counts["num_entities"] != formal.NUM_NODES:
        raise RuntimeError(f"metadata mismatch: {counts}")
    assets = formal.build_assets(graph, base_relations, train)
    formal.add_test_bank(graph, assets)
    request = formal.build_request(graph, base_relations, train)
    data = formal.build_data(train, graph.train_edge_type, request["selected_train_positions"])
    config = json.loads(formal.ORIGINAL_CONFIG.read_text(encoding="utf-8"))

    backbone = formal.build_original_model(formal.NUM_NODES, formal.BASE_RELATIONS, config, device)
    payload = torch.load(formal.ORIGINAL_CHECKPOINT, map_location="cpu", weights_only=False)
    backbone.load_state_dict(payload["model_state"], strict=True)
    operator = fullrank.FullRankRGCN(backbone, formal.SEED)
    actual_names = tuple(name for name, value in operator.named_parameters() if value.requires_grad)
    expected_names = tuple(operator.TRAINABLE_ALLOWLIST)
    if actual_names != expected_names:
        raise RuntimeError(f"TRAINABLE_PARAMETER_ALLOWLIST_FAIL: {actual_names}")
    unexpected = [name for name, value in operator.named_parameters()
                  if value.requires_grad and name not in expected_names]
    if unexpected:
        raise RuntimeError(f"UNEXPECTED_TRAINABLE_PARAMETERS: {unexpected}")
    trainable_count = sum(value.numel() for name, value in operator.named_parameters() if name in expected_names)
    if trainable_count != 128 * 128 + 64 * 64:
        raise RuntimeError(f"unexpected parameter count: {trainable_count}")

    loader_cls = builtins.__import__("torch_geometric.loader", fromlist=["GraphSAINTRandomWalkSampler"]).GraphSAINTRandomWalkSampler
    loader = loader_cls(data, batch_size=1024, walk_length=2, num_steps=32, num_workers=0)
    batch = next(iter(loader))
    batch_device = batch.to(device)
    edge = batch_device.edge_index[:, batch_device.dr_mask]
    edge_type = batch_device.edge_type[batch_device.dr_mask]
    gates = formal.p2_gates(assets, batch_device.x, device)

    with torch.no_grad():
        full_reps = operator(batch_device.x, edge, edge_type, gates, return_representations=True)
        original_reps = operator.original_representations(batch_device.x, edge, edge_type)
    identity_diffs = {
        "h0": max_abs(full_reps.h0, original_reps.h0),
        "h1": max_abs(full_reps.h1, original_reps.h1),
        "h1_activated": max_abs(full_reps.h1a, original_reps.h1a),
        "h2": max_abs(full_reps.h2, original_reps.h2),
    }
    if max(identity_diffs.values()) > 1e-5:
        raise RuntimeError(f"FULLRANK_UPDATE0_IDENTITY_FAIL: {identity_diffs}")

    sampler_module = __import__(
        "isolated_experiments.phase4c_rgcn_core_formal_canary.src.relational_core",
        fromlist=["RelationalSampler"],
    )
    sampling = __import__("framework.sampling", fromlist=["SamplingIdentity"])
    sampler = sampler_module.RelationalSampler(
        sampling.SamplingIdentity("WordNet18RR", "rgcn", "in", "2.5", formal.SEED),
        df=assets["df"], df_type=assets["df_type"],
        retain_positive=assets["retain_positive"], retain_type=assets["retain_type"],
    )
    stream = formal.OrderedLeanBRetainStream(sampler, epochs=formal.UPDATES)
    optimizer = torch.optim.Adam(
        [dict(operator.named_parameters())[name] for name in expected_names],
        lr=1e-3, weight_decay=0.0,
    )
    try:
        step_result = formal.p2_batch_step(
            operator, optimizer, batch_device, assets, sampler, stream, 0, 0.05, formal.BASE_RELATIONS
        )
    finally:
        stream.close()
    gradients = {name: float(value.grad.detach().norm().item()) for name, value in operator.named_parameters()
                 if value.requires_grad and value.grad is not None}
    if not gradients or not all(math.isfinite(value) for value in gradients.values()):
        raise RuntimeError(f"BACKWARD_NONFINITE_OR_MISSING: {gradients}")
    operator.assert_original_unchanged()
    result: dict[str, Any] = {
        "status": "PASS",
        "formal_source": str(FORMAL_SOURCE),
        "fullrank_source": str(FULLRANK_SOURCE),
        "python": str(Path(sys.executable).resolve()),
        "device": str(device),
        "graphsaint": {"batch_size": 1024, "walk_length": 2, "num_steps": 32, "num_workers": 0},
        "trainable_parameter_allowlist": list(actual_names),
        "unexpected_trainable_parameters": unexpected,
        "fullrank_parameter_count": int(trainable_count),
        "FULLRANK_UPDATE0_IDENTITY": "PASS",
        "update0_max_abs_diffs": identity_diffs,
        "batch": {"nodes": int(batch.x.numel()), "edges": int(batch.edge_index.size(1)),
                  "df_edges": int(batch.df_mask.sum()), "dr_edges": int(batch.dr_mask.sum())},
        "one_p2_step": {key: value for key, value in step_result.items()
                         if isinstance(value, (int, float, str, bool))},
        "gradient_norms": gradients,
        "BACKBONE_FROZEN": "YES",
        "RGCN_RELATION_SEMANTICS_PRESERVED": "YES",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
