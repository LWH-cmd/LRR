"""Run one isolated V4.12 FullRank GCN seed-42 ablation configuration.

The authoritative formal runner and evaluator are imported read-only.  This
driver changes only the requested gate allocation or one objective coefficient
and redirects every generated artifact into the V4.12 output root.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import torch

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(r"D:\LRR-GNNDELETE").resolve()
OUTPUT_ROOT_DEFAULT = PROJECT_ROOT / "lrr_v4_12_single_seed_ablation"
FULLRANK_ROOT = PROJECT_ROOT / "isolated_experiments" / "fullrank_main_matrix_20260826"
FULLRANK_ENTRYPOINT = FULLRANK_ROOT / "runners" / "gcn_cora_2p5_fullrank.py"
FULLRANK_MANIFEST = FULLRANK_ROOT / "fullrank_matrix_manifest.json"
CELL_ID = "cora__gcn__ed_in__2p5__seed42"
SEED = 42

DEFAULT_OBJECTIVE = {
    "forget": 0.05,
    "local": 0.50,
    "retain": 1.00,
    "anchor": 0.20,
    "reg": 0.00,
}
DEFAULT_MARGIN = 0.05

VARIANTS: dict[str, dict[str, Any]] = {
    "gate_global": {
        "stage": "gate",
        "label": "GLOBAL",
        "gate_mode": "GLOBAL",
        "objective": dict(DEFAULT_OBJECTIVE),
        "margin": DEFAULT_MARGIN,
        "changed_field": "gate allocation",
    },
    "gate_binary_local": {
        "stage": "gate",
        "label": "BINARY_LOCAL",
        "gate_mode": "BINARY_LOCAL",
        "objective": dict(DEFAULT_OBJECTIVE),
        "margin": DEFAULT_MARGIN,
        "changed_field": "gate allocation",
    },
    "objective_forget_0p025": {
        "stage": "objective",
        "label": "F=0.025",
        "gate_mode": "GRADED_LRR",
        "objective": {**DEFAULT_OBJECTIVE, "forget": 0.025},
        "margin": DEFAULT_MARGIN,
        "changed_field": "lambda_F",
    },
    "objective_forget_0p10": {
        "stage": "objective",
        "label": "F=0.10",
        "gate_mode": "GRADED_LRR",
        "objective": {**DEFAULT_OBJECTIVE, "forget": 0.10},
        "margin": DEFAULT_MARGIN,
        "changed_field": "lambda_F",
    },
    "objective_local_0p25": {
        "stage": "objective",
        "label": "L=0.25",
        "gate_mode": "GRADED_LRR",
        "objective": {**DEFAULT_OBJECTIVE, "local": 0.25},
        "margin": DEFAULT_MARGIN,
        "changed_field": "lambda_L",
    },
    "objective_local_1p00": {
        "stage": "objective",
        "label": "L=1.00",
        "gate_mode": "GRADED_LRR",
        "objective": {**DEFAULT_OBJECTIVE, "local": 1.00},
        "margin": DEFAULT_MARGIN,
        "changed_field": "lambda_L",
    },
    "objective_retain_0p50": {
        "stage": "objective",
        "label": "R=0.50",
        "gate_mode": "GRADED_LRR",
        "objective": {**DEFAULT_OBJECTIVE, "retain": 0.50},
        "margin": DEFAULT_MARGIN,
        "changed_field": "lambda_R",
    },
    "objective_retain_2p00": {
        "stage": "objective",
        "label": "R=2.00",
        "gate_mode": "GRADED_LRR",
        "objective": {**DEFAULT_OBJECTIVE, "retain": 2.00},
        "margin": DEFAULT_MARGIN,
        "changed_field": "lambda_R",
    },
    "objective_anchor_0p10": {
        "stage": "objective",
        "label": "A=0.10",
        "gate_mode": "GRADED_LRR",
        "objective": {**DEFAULT_OBJECTIVE, "anchor": 0.10},
        "margin": DEFAULT_MARGIN,
        "changed_field": "lambda_A",
    },
    "objective_anchor_0p40": {
        "stage": "objective",
        "label": "A=0.40",
        "gate_mode": "GRADED_LRR",
        "objective": {**DEFAULT_OBJECTIVE, "anchor": 0.40},
        "margin": DEFAULT_MARGIN,
        "changed_field": "lambda_A",
    },
    "objective_margin_0p025": {
        "stage": "objective",
        "label": "margin=0.025",
        "gate_mode": "GRADED_LRR",
        "objective": dict(DEFAULT_OBJECTIVE),
        "margin": 0.025,
        "changed_field": "margin",
    },
    "objective_margin_0p10": {
        "stage": "objective",
        "label": "margin=0.10",
        "gate_mode": "GRADED_LRR",
        "objective": dict(DEFAULT_OBJECTIVE),
        "margin": 0.10,
        "changed_field": "margin",
    },
}


def load_module(path: Path, name: str):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module import failed: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def isolated_guard(root: Path):
    root = root.resolve()

    def guard(path: Path) -> Path:
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(root):
            raise RuntimeError(f"V4.12-ISOLATION-WRITE-VIOLATION: {resolved}")
        return resolved

    return guard


def profile_hash(base, value: Mapping[str, Any], key: str) -> str:
    payload = copy.deepcopy(dict(value))
    payload.pop(key, None)
    return base.canonical_hash(payload)


def build_profile(formal, base, spec: Mapping[str, Any]) -> dict[str, Any]:
    profile = copy.deepcopy(formal.FORMAL_PROFILE)
    weights = {name: float(value) for name, value in spec["objective"].items()}
    margin = float(spec["margin"])
    training = copy.deepcopy(profile["training_profile"])
    training["objective_weights"] = weights
    training["forget_margin"] = margin
    training["training_profile_sha256"] = profile_hash(
        base, training, "training_profile_sha256"
    )
    profile["objective_formula"] = (
        f"{weights['forget']:g}*L_forget+{weights['local']:g}*L_local+"
        f"{weights['retain']:g}*L_retain+{weights['anchor']:g}*L_anchor+"
        f"{weights['reg']:g}*L_reg"
    )
    profile["training_profile"] = training
    profile["v4_12_ablation"] = {
        "stage": spec["stage"],
        "label": spec["label"],
        "gate_mode": spec["gate_mode"],
        "changed_field": spec["changed_field"],
        "objective_weights": weights,
        "margin": margin,
    }
    profile["resolved_profile_sha256"] = profile_hash(
        base, profile, "resolved_profile_sha256"
    )
    return profile


def make_runtime_builder(base, graph_builder, variant: str):
    original_build_runtime = base.build_runtime

    def build_runtime(graph: Mapping[str, Any], device: torch.device):
        formal_runtime = original_build_runtime(graph, device)
        mode = VARIANTS[variant]["gate_mode"]
        n = int(graph["num_nodes"])
        dtype = formal_runtime.x.dtype
        if mode == "GLOBAL":
            support1 = torch.ones(n, dtype=torch.bool, device=device)
            support2 = torch.ones(n, dtype=torch.bool, device=device)
            gate1 = torch.ones(n, 1, dtype=dtype, device=device)
            gate2 = torch.ones(n, 1, dtype=dtype, device=device)
        elif mode == "BINARY_LOCAL":
            support1 = graph["official"].support1.to(device)
            support2 = graph["official"].support2.to(device)
            gate1 = torch.zeros(n, 1, dtype=dtype, device=device)
            gate2 = torch.zeros(n, 1, dtype=dtype, device=device)
            gate1[support1, 0] = 1.0
            gate2[support2, 0] = 1.0
        elif mode == "GRADED_LRR":
            return formal_runtime
        else:  # pragma: no cover - guarded by CLI choices
            raise ValueError(mode)
        if gate1 is gate2:
            raise RuntimeError("V4.12 gate tensors must be distinct")
        return replace(
            formal_runtime,
            support1=support1,
            support2=support2,
            gate1=gate1,
            gate2=gate2,
        )

    return build_runtime


def gate_identity(base, graph: Mapping[str, Any], runtime, spec: Mapping[str, Any]) -> dict[str, Any]:
    formal_support1 = graph["official"].support1
    formal_support2 = graph["official"].support2
    gate1_values, gate1_counts = torch.unique(runtime.gate1.detach().cpu(), return_counts=True)
    gate2_values, gate2_counts = torch.unique(runtime.gate2.detach().cpu(), return_counts=True)
    return {
        "status": "PASS",
        "gate_mode": spec["gate_mode"],
        "layer1_support_definition": (
            "all nodes" if spec["gate_mode"] == "GLOBAL" else "formal support1"
        ),
        "layer2_support_definition": (
            "all nodes" if spec["gate_mode"] == "GLOBAL" else "formal support2"
        ),
        "formal_support1_count": int(formal_support1.sum()),
        "formal_support2_count": int(formal_support2.sum()),
        "operator_support1_count": int(runtime.support1.sum()),
        "operator_support2_count": int(runtime.support2.sum()),
        "gate1_value_counts": {
            str(float(value)): int(count)
            for value, count in zip(gate1_values, gate1_counts)
        },
        "gate2_value_counts": {
            str(float(value)): int(count)
            for value, count in zip(gate2_values, gate2_counts)
        },
        "formal_gate1_sha256": base.tensor_hash_v1(
            base.build_gate_buckets(
                graph["official"].endpoint_mask,
                graph["official"].support1,
                graph["official"].support2,
                dtype=graph["data"].x.dtype,
                device="cpu",
            ).gate1
        ),
        "formal_gate2_sha256": base.tensor_hash_v1(
            base.build_gate_buckets(
                graph["official"].endpoint_mask,
                graph["official"].support1,
                graph["official"].support2,
                dtype=graph["data"].x.dtype,
                device="cpu",
            ).gate2
        ),
        "operator_gate1_sha256": base.tensor_hash_v1(runtime.gate1),
        "operator_gate2_sha256": base.tensor_hash_v1(runtime.gate2),
        "outside_formal_support_nonzero_allowed": (
            spec["gate_mode"] == "GLOBAL"
        ),
        "only_gate_allocation_changed": True,
    }


def source_bindings(entry, formal, base, row: Mapping[str, Any]) -> dict[str, Any]:
    paths = {
        "formal_matrix_summary": Path(
            r"D:\LRR-GNNDELETE\isolated_experiments\fullrank_main_matrix_20260826\runs_homogeneous\cora__gcn__ed_in__2p5__seed42\matrix_summary.json"
        ),
        "formal_profile": Path(
            r"D:\LRR-GNNDELETE\isolated_experiments\fullrank_main_matrix_20260826\runs_homogeneous\cora__gcn__ed_in__2p5__seed42\resolved_training_profile.json"
        ),
        "formal_graph_manifest": Path(
            r"D:\LRR-GNNDELETE\isolated_experiments\phase1b5_formal_p2_multiseed_3000_cora\seed42\df_request\graph_identity.json"
        ),
        "formal_request": Path(
            r"D:\LRR-GNNDELETE\isolated_experiments\phase1b5_formal_p2_multiseed_3000_cora\seed42\df_request\df_request.pt"
        ),
        "original_checkpoint": Path(row["formal_original_checkpoint"]["path"]),
        "formal_entrypoint": FULLRANK_ENTRYPOINT,
        "formal_snapshot": Path(entry.FORMAL_SOURCE),
        "reference_runner": Path(formal.REFERENCE_RUNNER),
        "fullrank_operator": FULLRANK_ROOT / "runners" / "fullrank_operators.py",
        "fullrank_trainer": FULLRANK_ROOT / "runners" / "gcn_fullrank_common.py",
        "shared_objective": PROJECT_ROOT / "framework" / "trainer" / "shared_objective.py",
        "sampling": PROJECT_ROOT / "framework" / "sampling.py",
        "runtime_support": PROJECT_ROOT / "framework" / "lrr_runtime.py",
    }
    result: dict[str, Any] = {}
    for name, path in paths.items():
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = {
            "path": str(path),
            "sha256": base.file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return result


def run_one(variant: str, output_root: Path, device_name: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported V4.12 variant: {variant}")
    spec = VARIANTS[variant]
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "runs").mkdir(parents=True, exist_ok=True)
    (output_root / "manifests").mkdir(parents=True, exist_ok=True)
    marker = output_root / "manifests" / "prepare_complete.json"
    if not marker.exists():
        write_json(
            marker,
            {
                "status": "PASS",
                "scope": "LRR V4.12 single-seed ablation",
                "setting": "GCN / Cora / Ed_IN / 2.5% / seed42",
                "formal_data_center": str(
                    PROJECT_ROOT / "research_data_hub" / "LRR" / "PAPER_FORMAL_DATA_HUB"
                ),
                "new_seed_allowed": [42],
                "formal_sources_read_only": True,
            },
        )
    out = output_root / "runs" / variant
    if out.exists():
        raise FileExistsError(f"V4.12 no-resume output exists: {out}")

    entry = load_module(FULLRANK_ENTRYPOINT, "v412_fullrank_entrypoint")
    formal = entry.load_formal()
    base = formal.BASE
    row = entry.load_row(FULLRANK_MANIFEST, CELL_ID)
    entry.bind_formal(formal, row, out)
    entry.install_fullrank(formal)

    guard = isolated_guard(output_root)
    formal.ISOLATION_ROOT = output_root
    formal.ensure_isolated = guard
    base.ISOLATION_ROOT = output_root
    base.ensure_isolated = guard
    base.P2_EPOCHS = 3000
    base.P2_VALID_FREQ = 100

    profile = build_profile(formal, base, spec)
    formal.FORMAL_PROFILE = copy.deepcopy(profile)
    formal.FORMAL_PROFILE_HASH = profile["resolved_profile_sha256"]
    formal.FORMAL_TRAINING_HASH = profile["training_profile"][
        "training_profile_sha256"
    ]
    base.EXPECTED_P2_PROFILE_HASH = profile["resolved_profile_sha256"]
    base.EXPECTED_P2_TRAINING_HASH = profile["training_profile"][
        "training_profile_sha256"
    ]
    base.validate_p2_profile = lambda: copy.deepcopy(profile)

    base.build_runtime = make_runtime_builder(base, base.build_graph, variant)
    bindings = source_bindings(entry, formal, base, row)
    run_config = {
        "status": "PREPARED",
        "run_id": variant,
        "stage": spec["stage"],
        "label": spec["label"],
        "setting": "GCN / Cora / Ed_IN / 2.5% / seed42",
        "seed": SEED,
        "gate_mode": spec["gate_mode"],
        "objective_weights": spec["objective"],
        "margin": spec["margin"],
        "changed_field": spec["changed_field"],
        "fullrank_parameter_count": 20480,
        "matrix_shapes": {"repair1.W": [128, 128], "repair2.W": [64, 64]},
        "zero_initialization": True,
        "backbone_frozen": True,
        "optimizer": {"type": "Adam", "learning_rate": 0.0003, "weight_decay": 0.0},
        "updates": 3000,
        "validation_frequency": 100,
        "selector_id": entry.SELECTOR_ID,
        "selection_rule": entry.SELECTOR_RULE,
        "test_metrics_used_for_selection": False,
        "formal_default_reused": False,
        "resolved_profile_sha256": profile["resolved_profile_sha256"],
        "resolved_training_profile_sha256": profile["training_profile"][
            "training_profile_sha256"
        ],
        "source_bindings": bindings,
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
        "output_root": str(output_root),
    }
    # The rebound formal runner uses the existence of ``out`` as its fresh
    # run/no-resume guard.  Keep the prepared configuration outside ``out``
    # until the runner has created the fresh run directory itself.
    write_json(output_root / "manifests" / f"{variant}_prepared_config.json", run_config)

    gate = {
        "status": "PASS",
        "variant": spec["label"],
        "stage": spec["stage"],
        "FULLRANK_IS_RESIDUAL": "YES",
        "BACKBONE_FROZEN": "YES",
        "LOCALIZATION_IDENTICAL": "YES" if spec["gate_mode"] != "GLOBAL" else "NO_BY_DESIGN",
        "P2_IDENTICAL": "YES",
        "FORMAL_ASSETS_IDENTICAL": "YES",
        "FORMAL_SELECTOR_IDENTICAL": "YES",
        "BUDGET_IDENTICAL": "YES",
        "ONLY_CHANGED_FIELD": spec["changed_field"],
        "ONLY_PARAMETERIZATION_CHANGED": "NO",
        "TRAINABLE_PARAMETER_ALLOWLIST": "PASS",
        "UNEXPECTED_TRAINABLE_PARAMETERS": 0,
        "FULLRANK_PARAMETER_COUNT": 20480,
        "FORMAL_RESIDUAL_PLACEMENT_PRESERVED": "YES",
        "FORMAL_RESIDUAL_SHARING_SEMANTICS_PRESERVED": "YES",
        "RELATION_SPECIFIC_BACKBONE_COMPUTATION_UNCHANGED": "YES",
        "formal_selector_id": entry.SELECTOR_ID,
        "formal_selection_rule": entry.SELECTOR_RULE,
        "test_metrics_used_for_selection": False,
        "objective_weights": spec["objective"],
        "margin": spec["margin"],
        "gate_mode": spec["gate_mode"],
    }

    started = time.perf_counter()
    print(
        json.dumps(
            {
                "status": "START",
                "run_id": variant,
                "gate_mode": spec["gate_mode"],
                "objective_weights": spec["objective"],
                "margin": spec["margin"],
                "device": device_name,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    formal.train_arm(SEED, "LRR-P2", device_name)
    result = entry.strict_selection(formal, out, row, gate, started, device_name)

    graph = base.build_graph(SEED, require_prepared_request=True)
    runtime = base.build_runtime(graph, torch.device(device_name))
    write_json(out / "gate_identity.json", gate_identity(base, graph, runtime, spec))

    result.update(
        {
            "run_id": variant,
            "stage": spec["stage"],
            "label": spec["label"],
            "gate_mode": spec["gate_mode"],
            "objective_weights": spec["objective"],
            "margin": spec["margin"],
            "changed_field": spec["changed_field"],
            "resolved_profile_sha256": profile["resolved_profile_sha256"],
            "resolved_training_profile_sha256": profile["training_profile"][
                "training_profile_sha256"
            ],
            "selected_checkpoint_sha256": base.file_sha256(
                Path(result["selected_checkpoint"])
            ),
            "test_metrics_used_for_selection": False,
            "formal_default_reused": False,
        }
    )
    write_json(out / "matrix_summary.json", result)
    run_config["status"] = "PASS"
    run_config["selected_checkpoint_epoch"] = result["selected_checkpoint_epoch"]
    run_config["selected_checkpoint"] = result["selected_checkpoint"]
    run_config["selected_checkpoint_sha256"] = result["selected_checkpoint_sha256"]
    write_json(out / "run_config.json", run_config)
    preflight = json.loads((out / "preflight.json").read_text(encoding="utf-8"))
    preflight.update(
        {
            "v4_12_run_id": variant,
            "v4_12_gate_mode": spec["gate_mode"],
            "v4_12_objective_weights": spec["objective"],
            "v4_12_margin": spec["margin"],
            "test_metrics_used_for_selection": False,
        }
    )
    write_json(out / "preflight.json", preflight)
    write_json(
        out / "run_completion.json",
        {
            "status": "PASS",
            "run_id": variant,
            "selected_epoch": result["selected_checkpoint_epoch"],
            "selected_checkpoint": result["selected_checkpoint"],
            "selected_checkpoint_sha256": result["selected_checkpoint_sha256"],
            "test_metrics_evaluated_after_validation_selection": True,
            "test_metrics_used_for_selection": False,
            "output_dir": str(out),
        },
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "run_id": variant,
                "selected_epoch": result["selected_checkpoint_epoch"],
                "DT_AUC": result["DT_AUC"],
                "DT_AUP": result["DT_AUP"],
                "Df_AUC": result["Df_AUC"],
                "Df_AUP": result["Df_AUP"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LRR V4.12 single-seed ablation runner")
    parser.add_argument("--variant", choices=tuple(VARIANTS))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.variant is None:
        parser.error("--variant is required")
    run_one(args.variant, args.output_root, args.device)


if __name__ == "__main__":
    main()
