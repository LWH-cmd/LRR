"""Read-only differential probe for the successful RGCN replay and canary.

The probe deliberately stops before GraphSAINT iteration.  It records runtime,
import binding, graph/request tensors, and the native sampler construction so
that the first divergence is observable without changing the formal protocol.
"""

from __future__ import annotations

import argparse
import builtins
import hashlib
import importlib.util
import json
import multiprocessing
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
FORMAL_SOURCE = PROJECT_ROOT / "isolated_experiments/phase4c_wordnet18rr_p2_formal_seed42/src/run_formal_p2_crossed.py"

ENV_KEYS = sorted({
    "PATH", "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "CUDA_PATH", "CUDA_HOME",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "KMP_AFFINITY", "KMP_DUPLICATE_LIB_OK",
    "KMP_INIT_AT_FORK", "TORCH_HOME", "TORCH_CUDA_ARCH_LIST", "TORCH_SHOW_CPP_STACKTRACES",
    "RGCN_CORRECTED_OUTPUT", "RGCN_CONSTRAINT_DT", "RGCN_PASSIVE_SNAPSHOT_ROOT",
})


def sha256_file(path: str | Path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_import(name: str) -> dict[str, Any]:
    try:
        module = __import__(name, fromlist=["*"])
        path = getattr(module, "__file__", None)
        return {"status": "PASS", "version": getattr(module, "__version__", None),
                "file": str(Path(path).resolve()) if path else None,
                "sha256": sha256_file(path) if path else None}
    except Exception as exc:  # pragma: no cover - diagnostic path
        spec = importlib.util.find_spec(name)
        return {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}",
                "spec_origin": str(spec.origin) if spec and spec.origin else None}


def module_bindings() -> dict[str, dict[str, Any]]:
    prefixes = (
        "isolated_experiments.phase4c_wordnet18rr_p2_formal_seed42",
        "isolated_experiments.phase4c_wordnet18rr_p2_wf_adaptation_seed42",
        "isolated_experiments.phase4c_rgcn_core_formal_canary",
        "framework",
        "torch_geometric",
        "torch_sparse",
        "torch_scatter",
        "pyg_lib",
    )
    result: dict[str, dict[str, Any]] = {}
    for name, module in sorted(sys.modules.items()):
        if not name.startswith(prefixes):
            continue
        path = getattr(module, "__file__", None)
        result[name] = {"file": str(Path(path).resolve()) if path else None,
                        "sha256": sha256_file(path) if path else None}
    return result


def tensor_fingerprint(value: Any) -> dict[str, Any] | None:
    try:
        import torch
        if not isinstance(value, torch.Tensor):
            return None
        cpu = value.detach().cpu().contiguous()
        raw = cpu.numpy().tobytes()
        result: dict[str, Any] = {
            "shape": list(cpu.shape), "dtype": str(cpu.dtype), "device": str(value.device),
            "contiguous": bool(value.is_contiguous()), "sha256": hashlib.sha256(raw).hexdigest(),
            "numel": int(cpu.numel()),
        }
        if cpu.numel():
            if cpu.dtype == torch.bool:
                result["min"] = bool(cpu.min())
                result["max"] = bool(cpu.max())
            elif cpu.is_floating_point():
                result["min"] = float(cpu.min())
                result["max"] = float(cpu.max())
            else:
                result["min"] = int(cpu.min())
                result["max"] = int(cpu.max())
        else:
            result["min"] = None
            result["max"] = None
        return result
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def object_tensors(obj: Any, names: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in names:
        if hasattr(obj, name):
            result[name] = tensor_fingerprint(getattr(obj, name))
    return result


def runtime_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {
        "sys_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd().resolve()),
        "argv": sys.argv,
        "sys_path": [str(Path(p).resolve()) if p else "" for p in sys.path],
        "env": {key: os.environ.get(key) for key in ENV_KEYS},
        "multiprocessing_start_method": None,
        "python_random_state_sha256": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        "module_bindings_before_formal": module_bindings(),
    }
    try:
        result["multiprocessing_start_method"] = multiprocessing.get_start_method(allow_none=True)
    except Exception as exc:
        result["multiprocessing_start_method_error"] = str(exc)
    result["packages"] = {name: safe_import(name) for name in (
        "torch", "torch_geometric", "torch_sparse", "torch_scatter", "pyg_lib", "numpy", "scipy", "sklearn"
    )}
    torch_info = result["packages"]["torch"]
    if torch_info.get("status") == "PASS":
        import torch
        result["torch"] = {
            "version": torch.__version__, "cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "intraop_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "initial_seed": int(torch.initial_seed()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    return result


def load_formal() -> Any:
    spec = importlib.util.spec_from_file_location("rgcn_success_canary_formal", FORMAL_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {FORMAL_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sampler_snapshot(loader: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "class": f"{type(loader).__module__}.{type(loader).__qualname__}",
        "module_file": str(Path(type(loader).__module__ and sys.modules[type(loader).__module__].__file__).resolve())
        if getattr(sys.modules.get(type(loader).__module__), "__file__", None) else None,
        "scalar_attrs": {}, "tensor_attrs": {},
    }
    scalar_names = ("batch_size", "walk_length", "num_steps", "sample_coverage", "save_dir", "num_workers", "log", "N")
    for name in scalar_names:
        if hasattr(loader, name):
            value = getattr(loader, name)
            if isinstance(value, (str, int, float, bool)) or value is None:
                result["scalar_attrs"][name] = value
    for name, value in vars(loader).items():
        fp = tensor_fingerprint(value)
        if fp is not None:
            result["tensor_attrs"][name] = fp
    result["dict_keys"] = sorted(vars(loader).keys())
    return result


def graph_snapshot(formal: Any) -> dict[str, Any]:
    graph, base_relations, counts, train = formal.load_graph()
    assets = formal.build_assets(graph, base_relations, train)
    formal.add_test_bank(graph, assets)
    request = formal.build_request(graph, base_relations, train)
    data = formal.build_data(train, graph.train_edge_type, request["selected_train_positions"])
    result: dict[str, Any] = {
        "base_relations": int(base_relations), "counts": counts,
        "graph": object_tensors(graph, [
            "edge_index", "edge_type", "train_pos_edge_index", "train_edge_type",
            "val_pos_edge_index", "val_edge_type", "test_pos_edge_index", "test_edge_type",
        ]),
        "data": object_tensors(data, ["x", "edge_index", "edge_type", "df_mask", "dr_mask"]),
        "request": {key: tensor_fingerprint(value) for key, value in request.items()
                    if tensor_fingerprint(value) is not None},
        "assets": {key: tensor_fingerprint(value) for key, value in assets.items()
                   if tensor_fingerprint(value) is not None},
        "asset_scalar_keys": {key: type(value).__name__ for key, value in assets.items()
                              if tensor_fingerprint(value) is None},
        "formal_constants": {key: str(getattr(formal, key)) for key in (
            "ORIGINAL_CHECKPOINT", "ORIGINAL_CONFIG", "SEED", "NUM_NODES", "BASE_RELATIONS", "UPDATES", "VALIDATION_EVERY"
        ) if hasattr(formal, key)},
    }
    loader = builtins.__import__("torch_geometric.loader", fromlist=["GraphSAINTRandomWalkSampler"]).GraphSAINTRandomWalkSampler(
        data, batch_size=1024, walk_length=2, num_steps=32, num_workers=0
    )
    result["sampler"] = sampler_snapshot(loader)
    result["module_bindings_after_formal"] = module_bindings()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result: dict[str, Any] = {"label": args.label, "formal_source": str(FORMAL_SOURCE),
                              "formal_source_sha256": sha256_file(FORMAL_SOURCE)}
    try:
        result["runtime"] = runtime_snapshot()
        formal = load_formal()
        result["formal_import"] = "PASS"
        result["graph_pre_sampler"] = graph_snapshot(formal)
        result["status"] = "PASS"
    except Exception as exc:
        result["status"] = "FAIL"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["module_bindings_on_failure"] = module_bindings()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"label": args.label, "status": result["status"], "out": str(out),
                      "error": result.get("error")}, sort_keys=True), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
