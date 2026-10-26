"""Layered fresh-process probes for the exact WN18RR RGCN native crash.

This diagnostic deliberately imports the authoritative seed-42 formal runner
and uses its native GraphSAINT, asset construction, model, and P2 entry point.
It does not replace or modify any formal component.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch

from family_common import ROOT
from fullrank_operators import FullRankRGCN


PROJECT_ROOT = ROOT.parents[1]
FORMAL_SOURCE = PROJECT_ROOT / "isolated_experiments/phase4c_wordnet18rr_p2_formal_seed42/src/run_formal_p2_crossed.py"
PROBE_ROOT = ROOT / "diagnostics" / "rgcn_crash_probes"


def load_formal():
    spec = importlib.util.spec_from_file_location("rgcn_crash_probe_formal", FORMAL_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("RGCN-EXACT-FORMAL-IMPORT-FAIL")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def construct_model(formal, variant: str, config: dict, device: torch.device):
    if variant == "lowrank":
        return formal.build_operator("LRR-P2", config, device)
    backbone = formal.build_original_model(formal.NUM_NODES, formal.BASE_RELATIONS, config, device)
    payload = torch.load(formal.ORIGINAL_CHECKPOINT, map_location="cpu", weights_only=False)
    backbone.load_state_dict(payload["model_state"], strict=True)
    return FullRankRGCN(backbone, formal.SEED)


def construct_inputs(formal):
    print("PROBE_MARKER=load_graph", flush=True)
    graph, base_relations, counts, train = formal.load_graph()
    if base_relations != formal.BASE_RELATIONS or counts["num_entities"] != formal.NUM_NODES:
        raise RuntimeError(f"WN18RR metadata mismatch: {counts}")
    print("PROBE_MARKER=build_assets", flush=True)
    assets = formal.build_assets(graph, base_relations, train)
    formal.add_test_bank(graph, assets)
    print("PROBE_MARKER=build_request", flush=True)
    request = formal.build_request(graph, base_relations, train)
    data = formal.build_data(train, graph.train_edge_type, request["selected_train_positions"])
    config = json.loads(formal.ORIGINAL_CONFIG.read_text(encoding="utf-8"))
    return graph, assets, data, config


def native_loader(formal, data):
    print("PROBE_MARKER=construct_native_graphsaint", flush=True)
    loader = __import__(
        "torch_geometric.loader", fromlist=["GraphSAINTRandomWalkSampler"]
    ).GraphSAINTRandomWalkSampler(
        data, batch_size=1024, walk_length=2, num_steps=32, num_workers=0
    )
    print("PROBE_MARKER=next_native_graphsaint_batch", flush=True)
    batch = next(iter(loader))
    return loader, batch


def check_batch(formal, batch):
    required = {
        "x": batch.x,
        "edge_index": batch.edge_index,
        "edge_type": batch.edge_type,
        "df_mask": batch.df_mask,
        "dr_mask": batch.dr_mask,
    }
    if batch.edge_index.dim() != 2 or batch.edge_index.size(0) != 2:
        raise RuntimeError("GRAPHSAINT_EDGE_INDEX_SHAPE_FAIL")
    if batch.edge_type.dim() != 1 or batch.edge_type.numel() != batch.edge_index.size(1):
        raise RuntimeError("GRAPHSAINT_EDGE_TYPE_SHAPE_FAIL")
    if int(batch.edge_type.min()) < 0 or int(batch.edge_type.max()) >= 2 * formal.BASE_RELATIONS:
        raise RuntimeError("GRAPHSAINT_EDGE_TYPE_RANGE_FAIL")
    if int(batch.x.min()) < 0 or int(batch.x.max()) >= formal.NUM_NODES:
        raise RuntimeError("GRAPHSAINT_NODE_RANGE_FAIL")
    if batch.df_mask.dtype != torch.bool or batch.dr_mask.dtype != torch.bool:
        raise RuntimeError("GRAPHSAINT_MASK_DTYPE_FAIL")
    if not all(torch.isfinite(value.float()).all() for value in required.values() if value.is_floating_point()):
        raise RuntimeError("GRAPHSAINT_BATCH_NONFINITE")
    return {
        "nodes": int(batch.x.numel()),
        "edges": int(batch.edge_index.size(1)),
        "deleted_edges": int(batch.df_mask.sum()),
        "retained_edges": int(batch.dr_mask.sum()),
        "relation_min": int(batch.edge_type.min()),
        "relation_max": int(batch.edge_type.max()),
    }


def make_runtime(formal, assets, variant: str, config: dict, device: torch.device):
    model = construct_model(formal, variant, config, device)
    names = tuple(name for name, value in model.named_parameters() if value.requires_grad)
    expected = tuple(model.TRAINABLE_ALLOWLIST if hasattr(model, "TRAINABLE_ALLOWLIST") else model.allowlist())
    if names != expected:
        raise RuntimeError(f"TRAINABLE_PARAMETER_ALLOWLIST_FAIL:{names}")
    sampler = __import__(
        "isolated_experiments.phase4c_rgcn_core_formal_canary.src.relational_core",
        fromlist=["RelationalSampler"],
    ).RelationalSampler(
        __import__("framework.sampling", fromlist=["SamplingIdentity"]).SamplingIdentity(
            "WordNet18RR", "rgcn", "in", "2.5", formal.SEED
        ),
        df=assets["df"], df_type=assets["df_type"],
        retain_positive=assets["retain_positive"], retain_type=assets["retain_type"],
    )
    stream = formal.OrderedLeanBRetainStream(sampler, epochs=formal.UPDATES)
    return model, names, sampler, stream


def run_forward(formal, model, assets, batch, device):
    batch = batch.to(device)
    edge = batch.edge_index[:, batch.dr_mask]
    edge_type = batch.edge_type[batch.dr_mask]
    gates = formal.p2_gates(assets, batch.x, device)
    print("PROBE_MARKER=fullrank_or_lowrank_forward", flush=True)
    with torch.no_grad():
        reps = model(batch.x, edge, edge_type, gates, return_representations=True)
    values = (reps.h0, reps.h1, reps.h1a, reps.h2)
    if not all(torch.isfinite(value).all() for value in values):
        raise RuntimeError("MODEL_FORWARD_NONFINITE")
    return {"h0_shape": list(reps.h0.shape), "h1_shape": list(reps.h1.shape), "h2_shape": list(reps.h2.shape)}


def run_p2(formal, model, assets, batch, sampler, stream, device, *, do_step: bool):
    optimizer = torch.optim.Adam(
        [dict(model.named_parameters())[name] for name in model.TRAINABLE_ALLOWLIST],
        lr=1e-3, weight_decay=0.0,
    )
    if not do_step:
        optimizer.step = lambda closure=None: None
    batch = batch.to(device)
    print("PROBE_MARKER=exact_p2_batch_step", flush=True)
    values = formal.p2_batch_step(model, optimizer, batch, assets, sampler, stream, 0, 0.05, formal.BASE_RELATIONS)
    if not all(math.isfinite(float(value)) for value in values.values() if isinstance(value, (int, float))):
        raise RuntimeError("P2_RESULT_NONFINITE")
    gradients = {
        name: float(torch.linalg.vector_norm(value.grad).item())
        for name, value in model.named_parameters()
        if value.requires_grad and value.grad is not None
    }
    if not all(math.isfinite(value) for value in gradients.values()):
        raise RuntimeError("P2_GRADIENT_NONFINITE")
    return {"values": {key: value for key, value in values.items() if isinstance(value, (int, float))}, "gradient_norms": gradients}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("lowrank", "fullrank"), required=True)
    parser.add_argument("--stage", choices=("loader", "forward", "backward", "step"), required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda")
    formal = load_formal()
    graph, assets, data, config = construct_inputs(formal)
    loader, batch = native_loader(formal, data)
    batch_info = check_batch(formal, batch)
    print(json.dumps({"stage": args.stage, "variant": args.variant, "batch": batch_info}, sort_keys=True), flush=True)
    if args.stage == "loader":
        if args.variant == "fullrank":
            model = construct_model(formal, args.variant, config, device)
            print(json.dumps({"construction": "PASS", "trainable": [name for name, value in model.named_parameters() if value.requires_grad]}, sort_keys=True), flush=True)
        result = {"status": "PASS", "variant": args.variant, "stage": args.stage, "batch": batch_info}
    else:
        model, names, sampler, stream = make_runtime(formal, assets, args.variant, config, device)
        try:
            if args.stage == "forward":
                detail = run_forward(formal, model, assets, batch, device)
            else:
                detail = run_p2(formal, model, assets, batch, sampler, stream, device, do_step=args.stage == "step")
        finally:
            stream.close()
        result = {"status": "PASS", "variant": args.variant, "stage": args.stage, "trainable": names, "detail": detail}
    out = PROBE_ROOT / args.variant / f"{args.stage}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
