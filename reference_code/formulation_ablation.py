"""Minimal unified ablation for LRR paper V2.

The only newly trained arm is Variant A: the final full-rank localized
residual operator with GNNDelete's original Deleted Edge Consistency (DEC)
and Neighborhood Influence (NI) objective.  Variants B and C are read-only
reuse of exact completed formal cells.

Scope is intentionally fixed to Cora / GCN / Ed_IN / 2.5% and seeds
13, 21, 42, 87, 100.  No test metric participates in checkpoint selection.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch_geometric.seed import seed_everything


PROJECT_ROOT = Path(r"D:\LRR-GNNDELETE").resolve()
ROOT = (
    PROJECT_ROOT
    / "isolated_experiments"
    / "unified_ablation_gcn_cora_edin2p5_20260829"
).resolve()
RUNS_ROOT = ROOT / "runs" / "variant_a"
SEEDS = (13, 21, 42, 87, 100)
EPOCHS = 3000
VALID_FREQ = 100
VALIDATION_EPOCHS = tuple(range(99, EPOCHS, VALID_FREQ))
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.0
SELECTOR_ID = "S6-GNNDELETE-SUM-v1"
SELECTOR_RULE = "strict max validation DT_AUC + Df_AUC; exact tie -> earlier"
TRAINABLE_NAMES = ("repair1.W", "repair2.W")

DECNI_SRC = (
    PROJECT_ROOT
    / "isolated_experiments"
    / "phase2d_lrr_decni_cora_dblp_edin25_s6"
    / "src"
)
DECNI_RUNNER = DECNI_SRC / "run_lrr_decni.py"
FULLRANK_RUNNERS = (
    PROJECT_ROOT
    / "isolated_experiments"
    / "fullrank_main_matrix_20260826"
    / "runners"
)
FULLRANK_WRAPPER = FULLRANK_RUNNERS / "gcn_cora_2p5_fullrank.py"
FULLRANK_MANIFEST = FULLRANK_RUNNERS.parent / "fullrank_matrix_manifest.json"
FINAL_B_ROOT = FULLRANK_RUNNERS.parent / "runs_homogeneous"
P2_ROOT = (
    PROJECT_ROOT
    / "isolated_experiments"
    / "phase1b5_formal_p2_multiseed_3000_cora"
)
P2_REPORT = P2_ROOT / "reports" / "formal_comparison.json"

for source_dir in (DECNI_SRC, FULLRANK_RUNNERS, PROJECT_ROOT):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DEC = load_module("unified_ablation_frozen_decni", DECNI_RUNNER)
FULLRANK_ENTRY = load_module("unified_ablation_fullrank_entry", FULLRANK_WRAPPER)
from gcn_fullrank_common import FullRankGCN, FullRankGCNTrainer  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise RuntimeError(f"write outside isolated root: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def tensor_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.equal(left.detach().cpu(), right.detach().cpu()))


def manifest_row(seed: int) -> dict[str, Any]:
    value = json.loads(FULLRANK_MANIFEST.read_text(encoding="utf-8"))
    cell_id = f"cora__gcn__ed_in__2p5__seed{seed}"
    matches = [row for row in value["rows"] if row.get("cell_id") == cell_id]
    if len(matches) != 1:
        raise RuntimeError(f"manifest identity mismatch for {cell_id}")
    return matches[0]


def build_fullrank_student(runtime: Mapping[str, Any], seed: int):
    args = runtime["base"].model_args(seed)
    model = FullRankGCN(
        args,
        num_nodes=int(runtime["graph"]["num_nodes"]),
        input_mode="feature_matrix",
        expected_layer_dims=(128, 64),
        run_seed=seed,
    )
    loaded = model.load_state_dict(runtime["original_state"], strict=False)
    if set(loaded.missing_keys) != set(TRAINABLE_NAMES) or loaded.unexpected_keys:
        raise RuntimeError(f"full-rank original load mismatch: {loaded}")
    model.freeze_backbone()
    actual = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    if actual != TRAINABLE_NAMES:
        raise RuntimeError(f"trainable allowlist mismatch: {actual}")
    if sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad) != 20480:
        raise RuntimeError("full-rank parameter count mismatch")
    if torch.count_nonzero(model.repair1.W) or torch.count_nonzero(model.repair2.W):
        raise RuntimeError("full-rank matrices are not exactly zero initialized")
    return model


def forward_with_bases(model, runtime: Mapping[str, Any]):
    h0 = model.common_input_preparation(runtime["x"])
    h1_base = model.conv1(h0, runtime["dr_edge_index"])
    h1 = model.repair1(h1_base, runtime["support1"], runtime["gate1"])
    h2_base = model.conv2(
        model.official_intermediate_processing(h1), runtime["dr_edge_index"]
    )
    h2 = model.repair2(h2_base, runtime["support2"], runtime["gate2"])
    return h1_base, h1, h2_base, h2


def objective_terms(model, runtime, teacher_h1, teacher_h2):
    _h1_base, h1, _h2_base, h2 = forward_with_bases(model, runtime)
    return DEC.layerwise_decni(
        student_h1=h1,
        student_h2=h2,
        original_h1=teacher_h1,
        original_h2=teacher_h2,
        deleted_edge_index=runtime["deleted_mp"],
        random_edge_index=runtime["random_pairs"],
        ni_support1=runtime["ni_support1"],
        ni_support2=runtime["ni_support2"],
    )


def sequential_layerwise_step(
    model,
    runtime,
    teacher_h1,
    teacher_h2,
    optimizer1: torch.optim.Optimizer,
    optimizer2: torch.optim.Optimizer,
) -> dict[str, float]:
    optimizer1.zero_grad(set_to_none=True)
    optimizer2.zero_grad(set_to_none=True)
    terms1 = objective_terms(model, runtime, teacher_h1, teacher_h2)
    (gradient1,) = torch.autograd.grad(terms1.loss1, (model.repair1.W,))
    if not bool(torch.isfinite(gradient1).all()):
        raise RuntimeError("non-finite layer-1 gradient")
    model.repair1.W.grad = gradient1.detach()
    gradient1_norm = float(torch.linalg.vector_norm(gradient1))
    optimizer1.step()
    optimizer1.zero_grad(set_to_none=True)

    terms2 = objective_terms(model, runtime, teacher_h1, teacher_h2)
    (gradient2,) = torch.autograd.grad(terms2.loss2, (model.repair2.W,))
    if not bool(torch.isfinite(gradient2).all()):
        raise RuntimeError("non-finite layer-2 gradient")
    model.repair2.W.grad = gradient2.detach()
    gradient2_norm = float(torch.linalg.vector_norm(gradient2))
    optimizer2.step()
    optimizer2.zero_grad(set_to_none=True)
    if gradient1_norm <= 0.0 or gradient2_norm <= 0.0:
        raise RuntimeError("inactive full-rank DEC/NI gradient")
    return {
        "DEC_layer1": float(terms1.dec1.detach()),
        "NI_layer1": float(terms1.ni1.detach()),
        "layer1_loss": float(terms1.loss1.detach()),
        "DEC_layer2": float(terms2.dec2.detach()),
        "NI_layer2": float(terms2.ni2.detach()),
        "layer2_loss": float(terms2.loss2.detach()),
        "total_loss": float(terms1.loss1.detach() + terms2.loss2.detach()),
        "repair1_W_gradient_norm": gradient1_norm,
        "repair2_W_gradient_norm": gradient2_norm,
    }


def outside_support_exact(model, runtime: Mapping[str, Any]) -> bool:
    with torch.no_grad():
        h1_base, h1, h2_base, h2 = forward_with_bases(model, runtime)
    return bool(
        torch.equal(h1[~runtime["support1"]], h1_base[~runtime["support1"]])
        and torch.equal(h2[~runtime["support2"]], h2_base[~runtime["support2"]])
    )


def state_hash(state: Mapping[str, torch.Tensor], prefixes: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        if not any(name.startswith(prefix) for prefix in prefixes):
            continue
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def checkpoint(model, seed: int, epoch: int, validation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "lrr_unified_ablation_variant_a_v1",
        "variant": "A",
        "seed": seed,
        "epoch": epoch,
        "model_state": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
        "validation": copy.deepcopy(dict(validation)),
        "test_metrics_used_for_selection": False,
        "selector_id": SELECTOR_ID,
    }


def select_s6(events: list[dict[str, Any]]) -> dict[str, Any]:
    if len(events) != 30:
        raise RuntimeError(f"expected 30 validation events, got {len(events)}")
    return max(
        events,
        key=lambda event: (
            float(event["dt_auc"]) + float(event["df_auc"]),
            -int(event["epoch"]),
        ),
    )


def compare_evaluation_bundles(left, right) -> dict[str, bool]:
    names = (
        "validation_pos",
        "validation_neg",
        "test_pos",
        "test_neg",
        "df_retained_group_positions",
    )
    return {name: tensor_equal(getattr(left, name), getattr(right, name)) for name in names}


def preflight(device_name: str) -> None:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    report = json.loads(P2_REPORT.read_text(encoding="utf-8"))
    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        row = manifest_row(seed)
        final = FULLRANK_ENTRY.load_formal()
        FULLRANK_ENTRY.bind_formal(final, row, ROOT / "_read_only_binding" / f"seed{seed}")
        base = final.BASE
        final_graph = base.build_graph(seed, require_prepared_request=True)
        _, _, final_evaluation = base.build_sampling_artifacts(final_graph)
        dec_runtime = DEC.graph_runtime("Cora", seed, torch.device("cpu"))
        evaluation_equal = compare_evaluation_bundles(
            final_evaluation, dec_runtime["evaluation"]
        )
        graph_equal = {
            "graph_identity_sha256": (
                final_graph["graph_identity"]["graph_identity_sha256"]
                == dec_runtime["graph"]["graph_identity"]["graph_identity_sha256"]
            ),
            "Df": tensor_equal(
                final_graph["domains"].df_mp_mask,
                dec_runtime["graph"]["domains"].df_mp_mask,
            ),
            "Dr": tensor_equal(
                final_graph["domains"].dr_mp_edge_index,
                dec_runtime["dr_edge_index"],
            ),
            "features": tensor_equal(final_graph["data"].x, dec_runtime["x"]),
        }
        original_path = Path(row["formal_original_checkpoint"]["path"])
        source_checkpoint_equal = (
            sha256(original_path) == sha256(dec_runtime["original_path"])
        )
        c_summary = json.loads(
            (P2_ROOT / f"seed{seed}" / "matched_p2" / "final_summary.json").read_text(
                encoding="utf-8"
            )
        )
        b_summary = json.loads(
            (
                FINAL_B_ROOT
                / f"cora__gcn__ed_in__2p5__seed{seed}"
                / "matrix_summary.json"
            ).read_text(encoding="utf-8")
        )
        c_history = json.loads(
            (P2_ROOT / f"seed{seed}" / "matched_p2" / "validation_history.json").read_text(
                encoding="utf-8"
            )
        )["events"]
        c_selected = select_s6(c_history)
        c_test = report["diagnostic_checkpoint_test_results"][str(seed)]["arms"][
            "MATCHED-P2"
        ]["best_by_legacy_sum.pt"]
        profile_equal = (
            c_summary["resolved_formal_profile_sha256"]
            == b_summary["protocol_completion"]["resolved_formal_profile_sha256"]
        )
        c_selector_equal = int(c_selected["epoch"]) == int(c_test["epoch"])
        checks = list(evaluation_equal.values()) + list(graph_equal.values()) + [
            source_checkpoint_equal,
            profile_equal,
            c_selector_equal,
            c_summary["epochs_completed"] == EPOCHS,
            c_summary["validation_frequency"] == VALID_FREQ,
            c_summary["test_metrics_used_for_selection"] is False,
        ]
        if not all(checks):
            raise RuntimeError(f"semantic preflight mismatch for seed {seed}")
        per_seed[str(seed)] = {
            "status": "PASS",
            "graph": graph_equal,
            "evaluation_candidates": evaluation_equal,
            "pretrained_checkpoint_sha256": sha256(original_path),
            "pretrained_checkpoint_equal": source_checkpoint_equal,
            "B_C_suppression_preservation_profile_equal": profile_equal,
            "B_C_profile_sha256": c_summary["resolved_formal_profile_sha256"],
            "C_strict_S6_selected_epoch": int(c_selected["epoch"]),
            "C_existing_test_epoch": int(c_test["epoch"]),
            "test_excluded_from_selection": True,
        }

    # One representative, ephemeral update proves the new A path is active.
    seed = 42
    runtime = DEC.graph_runtime("Cora", seed, device)
    original = DEC.build_base(runtime, seed, runtime["original_state"]).to(device).eval()
    teacher = DEC.build_base(runtime, seed, runtime["original_state"]).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student = build_fullrank_student(runtime, seed).to(device)
    with torch.no_grad():
        teacher_h1, teacher_h2 = teacher(
            runtime["x"], runtime["dr_edge_index"], return_all_emb=True
        )
        original_z = original(runtime["x"], runtime["dr_edge_index"])
        student_z = DEC.student_forward(student, runtime)
    update0 = torch.allclose(student_z, original_z, rtol=1e-5, atol=1e-6)
    if not update0 or not outside_support_exact(student, runtime):
        raise RuntimeError("Variant A update-0 identity/locality failure")
    optimizer1 = torch.optim.Adam(
        (student.repair1.W,), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    optimizer2 = torch.optim.Adam(
        (student.repair2.W,), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    step = sequential_layerwise_step(
        student,
        runtime,
        teacher_h1.detach(),
        teacher_h2.detach(),
        optimizer1,
        optimizer2,
    )
    if not outside_support_exact(student, runtime):
        raise RuntimeError("Variant A post-update locality failure")
    write_json(
        ROOT / "preflight.json",
        {
            "status": "PASS",
            "setting": "GCN / Cora / Ed_IN / 2.5%",
            "seeds": list(SEEDS),
            "per_seed": per_seed,
            "shared_budget": {"updates": EPOCHS, "validation_frequency": VALID_FREQ},
            "shared_optimizer": {
                "type": "Adam",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            },
            "shared_selector": {
                "id": SELECTOR_ID,
                "rule": SELECTOR_RULE,
                "validation_only": True,
            },
            "B_C_negative_sampling": "same frozen formal SharedSampler profile",
            "A_negative_sampling": (
                "method-intrinsic objective difference: frozen per-seed DEC random-pair "
                "tensor required by the original DEC/NI objective"
            ),
            "variant_a_update0_identity": "PASS",
            "variant_a_support_outside_exact": "PASS",
            "variant_a_first_step": step,
            "variant_b_status": "REUSE_EXACT_FINAL_FULLRANK_CELL",
            "variant_c_status": "REUSE_EXACT_FORMAL_MATCHED_P2_CELL",
        },
    )
    print(json.dumps({"status": "PASS", "seeds": list(SEEDS)}, indent=2))


def run_variant_a(seed: int, device_name: str) -> None:
    if seed not in SEEDS:
        raise ValueError(seed)
    preflight_path = ROOT / "preflight.json"
    if not preflight_path.is_file() or json.loads(preflight_path.read_text())["status"] != "PASS":
        raise RuntimeError("preflight must pass before training")
    output = RUNS_ROOT / f"seed{seed}"
    if output.exists():
        raise FileExistsError(f"NO-RESUME: {output}")
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=False)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    seed_everything(seed)
    runtime = DEC.graph_runtime("Cora", seed, device)
    original = DEC.build_base(runtime, seed, runtime["original_state"]).to(device).eval()
    retrain = DEC.build_base(runtime, seed, runtime["retrain_state"]).to(device).eval()
    teacher = DEC.build_base(runtime, seed, runtime["original_state"]).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student = build_fullrank_student(runtime, seed).to(device)
    optimizer1 = torch.optim.Adam(
        (student.repair1.W,), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    optimizer2 = torch.optim.Adam(
        (student.repair2.W,), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    with torch.no_grad():
        teacher_h1, teacher_h2 = teacher(
            runtime["x"], runtime["dr_edge_index"], return_all_emb=True
        )
    teacher_h1 = teacher_h1.detach()
    teacher_h2 = teacher_h2.detach()
    backbone_before = state_hash(student.state_dict(), ("node_emb.", "conv1.", "conv2."))
    teacher_before = state_hash(teacher.state_dict(), ("node_emb.", "conv1.", "conv2."))
    initial_operator = {
        name: parameter.detach().cpu().clone()
        for name, parameter in student.named_parameters()
        if name in TRAINABLE_NAMES
    }
    events: list[dict[str, Any]] = []
    first_step = None
    last_step = None
    for epoch in range(EPOCHS):
        student.train()
        values = sequential_layerwise_step(
            student,
            runtime,
            teacher_h1,
            teacher_h2,
            optimizer1,
            optimizer2,
        )
        if first_step is None:
            first_step = copy.deepcopy(values)
        last_step = copy.deepcopy(values)
        if (epoch + 1) % VALID_FREQ == 0:
            student.eval()
            with torch.no_grad():
                terms = objective_terms(student, runtime, teacher_h1, teacher_h2)
                metrics = DEC.evaluate(
                    model=student,
                    original=original,
                    retrain=retrain,
                    runtime=runtime,
                    split="validation",
                )
            event = {
                "epoch": epoch,
                **{key: float(value) for key, value in metrics.items()},
                "validation_total_loss": float(terms.total),
                "DEC_layer1": float(terms.dec1),
                "NI_layer1": float(terms.ni1),
                "DEC_layer2": float(terms.dec2),
                "NI_layer2": float(terms.ni2),
                "test_metrics_used": False,
            }
            checkpoint_path = checkpoints / f"validation_epoch_{epoch:04d}.pt"
            torch.save(checkpoint(student, seed, epoch, event), checkpoint_path)
            event["checkpoint_sha256"] = sha256(checkpoint_path)
            events.append(event)
            print(
                f"seed={seed} update={epoch + 1}/3000 "
                f"val_DT={metrics['dt_auc']:.6f} val_Df={metrics['df_auc']:.6f}",
                flush=True,
            )

    if tuple(int(event["epoch"]) for event in events) != VALIDATION_EPOCHS:
        raise RuntimeError("validation cadence mismatch")
    selected = select_s6(events)
    selected_source = checkpoints / f"validation_epoch_{int(selected['epoch']):04d}.pt"
    selected_path = output / "model_selected.pt"
    shutil.copy2(selected_source, selected_path)
    container = torch.load(selected_path, map_location="cpu", weights_only=True)
    selected_model = build_fullrank_student(runtime, seed).to(device)
    loaded = selected_model.load_state_dict(container["model_state"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("selected checkpoint strict reload failed")
    selected_model.eval()
    with torch.no_grad():
        test_metrics = DEC.evaluate(
            model=selected_model,
            original=original,
            retrain=retrain,
            runtime=runtime,
            split="test",
        )
    if not all(
        math.isfinite(float(test_metrics[name]))
        for name in ("dt_auc", "dt_aup", "df_auc", "df_aup")
    ):
        raise RuntimeError("non-finite test metric")
    if state_hash(student.state_dict(), ("node_emb.", "conv1.", "conv2.")) != backbone_before:
        raise RuntimeError("backbone changed")
    if state_hash(teacher.state_dict(), ("node_emb.", "conv1.", "conv2.")) != teacher_before:
        raise RuntimeError("teacher changed")
    final_operator = {
        name: parameter.detach().cpu()
        for name, parameter in student.named_parameters()
        if name in TRAINABLE_NAMES
    }
    delta = math.sqrt(
        sum(
            float((final_operator[name] - initial_operator[name]).double().pow(2).sum())
            for name in TRAINABLE_NAMES
        )
    )
    if delta <= 0.0 or not outside_support_exact(student, runtime):
        raise RuntimeError("inactive or non-local Variant A result")
    summary = {
        "status": "COMPLETE",
        "variant": "A",
        "paper_label": "LRR + original deletion objective",
        "dataset": "Cora",
        "architecture": "GCN",
        "deletion_type": "Ed_IN",
        "ratio": "2.5",
        "seed": seed,
        "epochs_completed": EPOCHS,
        "optimizer_steps_completed": EPOCHS * 2,
        "validation_frequency": VALID_FREQ,
        "validation_event_count": len(events),
        "selector_id": SELECTOR_ID,
        "selection_rule": SELECTOR_RULE,
        "selected_epoch": int(selected["epoch"]),
        "selected_checkpoint": str(selected_path),
        "selected_checkpoint_sha256": sha256(selected_path),
        "selected_validation_metrics": selected,
        "test_metrics": test_metrics,
        "DT_AUC": float(test_metrics["dt_auc"]),
        "DT_AUP": float(test_metrics["dt_aup"]),
        "Df_AUC": float(test_metrics["df_auc"]),
        "Df_AUP": float(test_metrics["df_aup"]),
        "test_metrics_used_for_selection": False,
        "trainable_parameter_names": list(TRAINABLE_NAMES),
        "trainable_parameter_count": 20480,
        "backbone_frozen": True,
        "teacher_frozen": True,
        "outside_support_residual_exact": True,
        "operator_delta_norm": delta,
        "objective": {
            "public_terms": ["Deleted Edge Consistency", "Neighborhood Influence"],
            "weights": {"DEC": 0.5, "NI": 0.5},
            "source_objective_id": DEC.OBJECTIVE_ID,
            "distance": DEC.DISTANCE,
        },
        "random_pair_tensor_sha256": runtime["random_pair_hash"],
        "graph_identity_sha256": runtime["graph"]["graph_identity"]["graph_identity_sha256"],
        "first_step": first_step,
        "last_step": last_step,
    }
    write_json(output / "validation_history.json", {"events": events})
    write_json(output / "matrix_summary.json", summary)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "seed": seed,
                "selected_epoch": summary["selected_epoch"],
                "DT_AUC": summary["DT_AUC"],
                "DT_AUP": summary["DT_AUP"],
                "Df_AUC": summary["Df_AUC"],
                "Df_AUP": summary["Df_AUP"],
            },
            indent=2,
        ),
        flush=True,
    )


def metric_record(variant: str, seed: int) -> dict[str, Any]:
    if variant == "A":
        value = json.loads((RUNS_ROOT / f"seed{seed}" / "matrix_summary.json").read_text())
        return {
            "seed": seed,
            "selected_epoch": value["selected_epoch"],
            "DT_AUC": value["DT_AUC"],
            "DT_AUP": value["DT_AUP"],
            "Df_AUC": value["Df_AUC"],
            "Df_AUP": value["Df_AUP"],
            "source": str(RUNS_ROOT / f"seed{seed}" / "matrix_summary.json"),
        }
    if variant == "B":
        path = (
            FINAL_B_ROOT
            / f"cora__gcn__ed_in__2p5__seed{seed}"
            / "matrix_summary.json"
        )
        value = json.loads(path.read_text())
        return {
            "seed": seed,
            "selected_epoch": value["selected_checkpoint_epoch"],
            "DT_AUC": value["DT_AUC"],
            "DT_AUP": value["DT_AUP"],
            "Df_AUC": value["Df_AUC"],
            "Df_AUP": value["Df_AUP"],
            "source": str(path),
        }
    report = json.loads(P2_REPORT.read_text())
    path = P2_ROOT / f"seed{seed}" / "matched_p2" / "validation_history.json"
    events = json.loads(path.read_text())["events"]
    selected = select_s6(events)
    value = report["diagnostic_checkpoint_test_results"][str(seed)]["arms"][
        "MATCHED-P2"
    ]["best_by_legacy_sum.pt"]
    if int(value["epoch"]) != int(selected["epoch"]):
        raise RuntimeError(f"C selector mismatch for seed {seed}")
    metrics = value["test_metrics"]
    return {
        "seed": seed,
        "selected_epoch": int(value["epoch"]),
        "DT_AUC": metrics["dt_auc"],
        "DT_AUP": metrics["dt_aup"],
        "Df_AUC": metrics["df_auc"],
        "Df_AUP": metrics["df_aup"],
        "source": str(P2_REPORT),
    }


def aggregate() -> None:
    variants = {variant: [metric_record(variant, seed) for seed in SEEDS] for variant in "ABC"}
    metrics = ("DT_AUC", "DT_AUP", "Df_AUC", "Df_AUP")
    summary: dict[str, Any] = {}
    for variant, records in variants.items():
        aggregate_metrics: dict[str, Any] = {}
        for metric in metrics:
            values = [float(record[metric]) for record in records]
            aggregate_metrics[metric] = {
                "mean": statistics.fmean(values),
                "sample_std": statistics.stdev(values),
                "values": values,
            }
        per_seed_with_uf = []
        for record in records:
            value = dict(record)
            value["U"] = (float(value["DT_AUC"]) + float(value["DT_AUP"])) / 2.0
            value["F"] = (float(value["Df_AUC"]) + float(value["Df_AUP"])) / 2.0
            per_seed_with_uf.append(value)
        aggregate_metrics["U"] = {
            "mean": statistics.fmean(record["U"] for record in per_seed_with_uf),
            "sample_std": statistics.stdev(record["U"] for record in per_seed_with_uf),
        }
        aggregate_metrics["F"] = {
            "mean": statistics.fmean(record["F"] for record in per_seed_with_uf),
            "sample_std": statistics.stdev(record["F"] for record in per_seed_with_uf),
        }
        summary[variant] = {
            "label": {
                "A": "LRR + original deletion objective",
                "B": "LRR + suppression-preservation objective",
                "C": "GNNDelete operator + suppression-preservation objective",
            }[variant],
            "status": "RUN" if variant == "A" else "REUSED",
            "per_seed": per_seed_with_uf,
            "aggregate": aggregate_metrics,
        }
    deltas: dict[str, Any] = {}
    for comparison, left, right in (("B_minus_A", "B", "A"), ("B_minus_C", "B", "C")):
        deltas[comparison] = {
            metric: summary[left]["aggregate"][metric]["mean"]
            - summary[right]["aggregate"][metric]["mean"]
            for metric in (*metrics, "U", "F")
        }
    value = {
        "status": "PASS",
        "setting": "GCN / Cora / Ed_IN / 2.5%",
        "seeds": list(SEEDS),
        "new_run_count": 5,
        "sample_standard_deviation_ddof": 1,
        "variants": summary,
        "mean_deltas": deltas,
    }
    write_json(ROOT / "unified_ablation_results.json", value)
    print(json.dumps(value, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "run-a", "aggregate"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "preflight":
        preflight(args.device)
    elif args.command == "run-a":
        if args.seed is None:
            parser.error("run-a requires --seed")
        run_variant_a(args.seed, args.device)
    else:
        aggregate()


if __name__ == "__main__":
    main()
