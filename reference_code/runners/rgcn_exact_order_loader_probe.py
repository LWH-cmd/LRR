"""Loader probe following the formal run_cell construction order exactly."""

from __future__ import annotations

import argparse
import builtins
import ctypes
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
FORMAL_SOURCE = PROJECT_ROOT / "isolated_experiments/phase4c_wordnet18rr_p2_formal_seed42/src/run_formal_p2_crossed.py"
FULLRANK_SOURCE = ROOT / "runners/fullrank_operators.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"import failed: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def loaded_native_modules() -> list[str]:
    """Best-effort read-only enumeration of this process's loaded DLLs."""
    if sys.platform != "win32":
        return []
    try:
        psapi = ctypes.WinDLL("psapi")
        kernel32 = ctypes.WinDLL("kernel32")
        process = kernel32.GetCurrentProcess()
        handles = (ctypes.c_void_p * 2048)()
        needed = ctypes.c_ulong()
        if not psapi.EnumProcessModules(process, ctypes.byref(handles), ctypes.sizeof(handles), ctypes.byref(needed)):
            return []
        count = min(int(needed.value // ctypes.sizeof(ctypes.c_void_p)), len(handles))
        get_name = psapi.GetModuleFileNameExW
        get_name.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_ulong]
        paths: list[str] = []
        for index in range(count):
            buffer = ctypes.create_unicode_buffer(1024)
            if get_name(process, handles[index], buffer, len(buffer)):
                path = buffer.value
                lower = path.lower()
                if any(token in lower for token in ("torch", "cuda", "cudnn", "cublas", "cusparse", "torch_sparse", "torch_scatter", "pyg_lib", "python")):
                    paths.append(path)
        return sorted(set(paths))
    except Exception:
        return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("lowrank", "fullrank"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    formal = load_module(FORMAL_SOURCE, "rgcn_exact_order_formal")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    graph, base_relations, counts, train = formal.load_graph()
    assets = formal.build_assets(graph, base_relations, train)
    formal.add_test_bank(graph, assets)
    request = formal.build_request(graph, base_relations, train)
    data = formal.build_data(train, graph.train_edge_type, request["selected_train_positions"])
    config = json.loads(formal.ORIGINAL_CONFIG.read_text(encoding="utf-8"))
    device = torch.device("cuda")

    # This is the formal run_cell order: seed, model, optimizer, sampler,
    # p2 gates, then GraphSAINT construction and first iteration.
    formal.set_seed(formal.SEED)
    if args.variant == "lowrank":
        model = formal.build_operator("LRR-P2", config, device)
        names = tuple(model.TRAINABLE_ALLOWLIST)
    else:
        fullrank = load_module(FULLRANK_SOURCE, "rgcn_exact_order_fullrank")
        backbone = formal.build_original_model(formal.NUM_NODES, formal.BASE_RELATIONS, config, device)
        payload = torch.load(formal.ORIGINAL_CHECKPOINT, map_location="cpu", weights_only=False)
        backbone.load_state_dict(payload["model_state"], strict=True)
        model = fullrank.FullRankRGCN(backbone, formal.SEED)
        names = tuple(model.TRAINABLE_ALLOWLIST)
    optimizer = torch.optim.Adam([dict(model.named_parameters())[name] for name in names], lr=1e-3, weight_decay=0.0)
    sampler_module = __import__("isolated_experiments.phase4c_rgcn_core_formal_canary.src.relational_core", fromlist=["RelationalSampler"])
    sampling = __import__("framework.sampling", fromlist=["SamplingIdentity"])
    relation_sampler = sampler_module.RelationalSampler(
        sampling.SamplingIdentity("WordNet18RR", "rgcn", "in", "2.5", formal.SEED),
        df=assets["df"], df_type=assets["df_type"],
        retain_positive=assets["retain_positive"], retain_type=assets["retain_type"],
    )
    full_x = torch.arange(formal.NUM_NODES, dtype=torch.long, device=device)
    gates = formal.p2_gates(assets, full_x, device)
    print(f"EXACT_ORDER_MARKER=construct_native_graphsaint variant={args.variant}", flush=True)
    loader_cls = builtins.__import__("torch_geometric.loader", fromlist=["GraphSAINTRandomWalkSampler"]).GraphSAINTRandomWalkSampler
    loader = loader_cls(data, batch_size=1024, walk_length=2, num_steps=32, num_workers=0)
    pre_next = Path(args.out).with_name(Path(args.out).stem + "_pre_next.json")
    pre_next.parent.mkdir(parents=True, exist_ok=True)
    pre_next.write_text(json.dumps({
        "variant": args.variant,
        "pid": os.getpid(),
        "python": str(Path(sys.executable).resolve()),
        "loaded_native_modules": loaded_native_modules(),
        "torch_sparse_rw": str(Path(sys.modules["torch_sparse.rw"].__file__).resolve()) if "torch_sparse.rw" in sys.modules else None,
        "torch_geometric_graph_saint": str(Path(sys.modules["torch_geometric.loader.graph_saint"].__file__).resolve()) if "torch_geometric.loader.graph_saint" in sys.modules else None,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"EXACT_ORDER_MARKER=next_native_graphsaint_batch variant={args.variant}", flush=True)
    batch = next(iter(loader))
    result = {"status": "PASS", "variant": args.variant, "batch_nodes": int(batch.x.numel()),
              "batch_edges": int(batch.edge_index.size(1)), "counts": counts,
              "formal_source": str(FORMAL_SOURCE), "fullrank_source": str(FULLRANK_SOURCE)
              if args.variant == "fullrank" else None,
              "trainable_names": list(names), "sampler": {"batch_size": 1024, "walk_length": 2,
              "num_steps": 32, "num_workers": 0}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
