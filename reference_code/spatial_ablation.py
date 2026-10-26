"""Isolated Phase-B gate/operator variants for the formal Cora GCN cell.

The Phase 1B-5 formal snapshot remains the owner of graph construction,
sampling, objective, evaluator, optimizer, validation cadence and selector.
This module only injects the frozen Phase-B operator control and writes under
the Phase-5 workspace.
"""

from __future__ import annotations

import argparse
import csv
import copy
import dataclasses
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

PROJECT_ROOT = Path(r"D:\LRR-GNNDELETE").resolve()
PHASE5_ROOT = PROJECT_ROOT / "isolated_experiments" / "phase5_lrr_patent_ablation" / "20260915"
RUNNERS = PROJECT_ROOT / "isolated_experiments" / "fullrank_main_matrix_20260826" / "runners"
MANIFEST = PROJECT_ROOT / "isolated_experiments" / "fullrank_main_matrix_20260826" / "fullrank_matrix_manifest.csv"
FORMAL_SOURCE = RUNNERS / "snapshots" / "gcn_phase1b5_formal_runner.py"

SEEDS = (13, 21, 42, 87, 100)
VARIANTS = (
    "UNIFORM_MASSMATCH",
    "LOCAL_BINARY",
    "UNGATED",
    "LAST_LAYER_ONLY",
    "TIER2_MASSMATCH",
    "DECAY_LAMBDA_0P5",
    "DECAY_LAMBDA_1P25",
)

for path in (PROJECT_ROOT, RUNNERS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fullrank_operators import FullRankGCN  # noqa: E402
from gcn_fullrank_common import FullRankGCNTrainer  # noqa: E402


def load_manifest_row(seed: int) -> dict[str, Any]:
    with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [
        row for row in rows
        if row["cell_id"] == f"cora__gcn__ed_in__2p5__seed{seed}"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"formal Cora GCN row mismatch for seed {seed}")
    row = matches[0]
    row["formal_original_checkpoint"] = json.loads(row["formal_original_checkpoint"])
    row["formal_request_asset"] = json.loads(row["formal_request_asset"])
    row["formal_dr_df_asset"] = json.loads(row["formal_dr_df_asset"])
    return row


def import_formal():
    spec = importlib.util.spec_from_file_location("phase5_gcn_formal_snapshot", FORMAL_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("PHASE5-FORMAL-SNAPSHOT-IMPORT-FAIL")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class VariantResidualRepair(nn.Module):
    """Full-rank W with the frozen variant gate contract checked upstream."""

    def __init__(self, dim: int, *, device, dtype) -> None:
        super().__init__()
        self.dim = int(dim)
        self.W = nn.Parameter(torch.zeros(self.dim, self.dim, device=device, dtype=dtype))

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.W.zero_()

    def forward(self, h: torch.Tensor, support: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if h.dim() != 2 or h.size(1) != self.dim:
            raise ValueError("residual input shape mismatch")
        if support.shape != (h.size(0),) or support.dtype != torch.bool:
            raise TypeError("support must be bool [N]")
        if gate.shape != (h.size(0), 1) or gate.dtype != h.dtype:
            raise TypeError("gate must match h [N,1]")
        if h.device != self.W.device or gate.device != h.device or support.device != h.device:
            raise TypeError("residual tensors must share device")
        if torch.count_nonzero(gate[~support]).item() != 0:
            raise ValueError("gate must be zero outside support")
        return h + gate * h.matmul(self.W)


class VariantTrainer(FullRankGCNTrainer):
    def __init__(self, args, variant: str) -> None:
        super().__init__(args)
        self.variant = variant
        self.ALLOWLIST = ("repair2.W",) if variant == "LAST_LAYER_ONLY" else ("repair1.W", "repair2.W")

    def expected_trainable_parameter_names(self, model):
        return self.ALLOWLIST


def bucket_masks(graph: Mapping[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    endpoint = graph["official"].endpoint_mask.to(device)
    support1 = graph["official"].support1.to(device)
    support2 = graph["official"].support2.to(device)
    return endpoint, support1 & ~endpoint, support2 & ~support1


def make_variant_runtime(formal, graph: Mapping[str, Any], device: torch.device, variant: str):
    base_runtime = formal.BASE.build_runtime.__wrapped__(graph, device) if hasattr(formal.BASE.build_runtime, "__wrapped__") else None
    if base_runtime is None:
        raise RuntimeError("base runtime wrapper missing")
    b0, b1, b2 = bucket_masks(graph, device)
    dtype = base_runtime.gate1.dtype
    s1 = graph["official"].support1.to(device)
    s2 = graph["official"].support2.to(device)
    g1 = torch.zeros((s1.numel(), 1), device=device, dtype=dtype)
    g2 = torch.zeros((s2.numel(), 1), device=device, dtype=dtype)
    if variant == "UNIFORM_MASSMATCH":
        c1 = (b0.sum().float() + 0.7 * b1.sum().float()) / (b0.sum() + b1.sum()).float()
        c2 = (b0.sum().float() + 0.7 * b1.sum().float() + 0.3 * b2.sum().float()) / (b0.sum() + b1.sum() + b2.sum()).float()
        g1[s1, 0] = c1.to(dtype)
        g2[s2, 0] = c2.to(dtype)
    elif variant == "LOCAL_BINARY":
        g1[s1, 0] = 1.0
        g2[s2, 0] = 1.0
    elif variant == "UNGATED":
        s1 = torch.ones_like(s1)
        s2 = torch.ones_like(s2)
        g1[:, 0] = 1.0
        g2[:, 0] = 1.0
    elif variant == "LAST_LAYER_ONLY":
        g1[b0, 0] = 1.0
        g1[b1, 0] = 0.7
        g2[b0, 0] = 1.0
        g2[b1, 0] = 0.7
        g2[b2, 0] = 0.3
    elif variant == "TIER2_MASSMATCH":
        c12 = (0.7 * b1.sum().float() + 0.3 * b2.sum().float()) / (b1.sum() + b2.sum()).float()
        g1[b0, 0] = 1.0
        g1[b1, 0] = 0.7
        g2[b0, 0] = 1.0
        g2[b1, 0] = c12.to(dtype)
        g2[b2, 0] = c12.to(dtype)
    elif variant in ("DECAY_LAMBDA_0P5", "DECAY_LAMBDA_1P25"):
        lam = 0.5 if variant.endswith("0P5") else 1.25
        gamma1 = 1.0 - 0.3 * lam
        gamma2 = 1.0 - 0.7 * lam
        g1[b0, 0] = 1.0
        g1[b1, 0] = gamma1
        g2[b0, 0] = 1.0
        g2[b1, 0] = gamma1
        g2[b2, 0] = gamma2
    else:
        raise ValueError(variant)
    if torch.count_nonzero(g1[~s1]).item() or torch.count_nonzero(g2[~s2]).item():
        raise RuntimeError(f"{variant}: gate outside support")
    return dataclasses.replace(base_runtime, support1=s1, support2=s2, gate1=g1, gate2=g2), {
        "variant": variant,
        "bucket_counts": {"B0": int(b0.sum()), "B1": int(b1.sum()), "B2": int(b2.sum())},
        "support_counts": {"layer1": int(s1.sum()), "layer2": int(s2.sum())},
        "gate1_unique": sorted(float(x) for x in torch.unique(g1).detach().cpu().tolist()),
        "gate2_unique": sorted(float(x) for x in torch.unique(g2).detach().cpu().tolist()),
        "gate1_mass": float(g1.sum()), "gate2_mass": float(g2.sum()),
        "operator_support_override": variant == "UNGATED",
    }


def configure(formal, row: Mapping[str, Any], variant: str, out: Path) -> dict[str, Any]:
    base = formal.BASE
    seed = int(row["seed"])
    original_by_seed = {int(row["seed"]): Path(row["formal_original_checkpoint"]["path"]).resolve()}
    request_by_seed = {seed: Path(row["formal_request_asset"]["path"]).resolve()}
    graph_by_seed = {seed: Path(row["formal_dr_df_asset"]["path"]).resolve()}
    # The full formal graph/request paths for the other four seeds are fixed by
    # the same authoritative Phase 1B-5 family and are resolved from its seed dirs.
    for value in SEEDS:
        if value == seed:
            continue
        root = PROJECT_ROOT / "isolated_experiments" / "phase1b5_formal_p2_multiseed_3000_cora" / f"seed{value}" / "df_request"
        original_by_seed[value] = root.parent / "original" / "checkpoints" / "model_best.pt"
        request_by_seed[value] = root / "df_request.pt"
        graph_by_seed[value] = root / "graph_identity.json"
    run_root = PHASE5_ROOT / "runs" / variant / f"seed{seed}"
    def ensure(path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(PHASE5_ROOT.resolve()):
            raise RuntimeError(f"PHASE5-WRITE-VIOLATION: {resolved}")
        return resolved
    base.ISOLATION_ROOT = PHASE5_ROOT
    formal.ISOLATION_ROOT = PHASE5_ROOT
    base.ensure_isolated = ensure
    formal.ensure_isolated = ensure
    base.P2_EPOCHS = formal.P2_EPOCHS
    base.P2_VALID_FREQ = formal.P2_VALID_FREQ
    base.SENTINEL_EPOCHS = formal.FORMAL_SENTINEL_EPOCHS
    base.EXPECTED_P2_PROFILE_HASH = formal.FORMAL_PROFILE_HASH
    base.EXPECTED_P2_TRAINING_HASH = formal.FORMAL_TRAINING_HASH
    base.request_path = lambda value: request_by_seed[int(value)]
    base.graph_manifest_path = lambda value: graph_by_seed[int(value)]
    base.sampler_manifest_path = lambda value: run_root / "sampler_identity.json"
    base.run_dir = lambda _seed, _kind: run_root
    base.ckpt_dir = lambda _seed, _kind: run_root / "checkpoints"
    base.original_checkpoint = lambda value, final=False: original_by_seed[int(value)]
    base.retrain_checkpoint = lambda value, final=False: original_by_seed[int(value)]
    formal.request_path = base.request_path
    formal.graph_manifest_path = base.graph_manifest_path
    formal.sampler_manifest_path = base.sampler_manifest_path
    formal.run_dir = base.run_dir
    formal.ckpt_dir = base.ckpt_dir
    formal.original_checkpoint = base.original_checkpoint
    formal.retrain_checkpoint = base.retrain_checkpoint
    base.validate_p2_profile = formal.validate_p2_profile
    original_runtime = base.build_runtime
    base.build_runtime = original_runtime
    base._phase5_original_build_runtime = original_runtime
    def wrapped_runtime(graph, device):
        return make_variant_runtime(formal, graph, device, variant)[0]
    wrapped_runtime.__wrapped__ = original_runtime
    base.build_runtime = wrapped_runtime
    base._phase5_original_build_runtime = original_runtime
    return {"run_root": str(run_root), "formal_request_path": str(request_by_seed[seed]), "formal_graph_path": str(graph_by_seed[seed]), "formal_original_checkpoint": str(original_by_seed[seed])}


def install_student(formal, variant: str) -> None:
    base = formal.BASE
    def build_student(seed: int, arm: str, original_state: Mapping[str, torch.Tensor]):
        if arm != "LRR-P2":
            return base._ORIGINAL_BUILD_STUDENT(seed, arm, original_state)
        args = base.model_args(seed)
        model = FullRankGCN(args, num_nodes=19793, input_mode="feature_matrix", expected_layer_dims=(128, 64), run_seed=seed)
        reference = next(model.conv1.parameters())
        d1, d2 = model.effective_layer_dimensions()
        model.repair1 = VariantResidualRepair(d1, device=reference.device, dtype=reference.dtype)
        model.repair2 = VariantResidualRepair(d2, device=reference.device, dtype=reference.dtype)
        model.repair1.reset_parameters(); model.repair2.reset_parameters()
        model.freeze_backbone()
        if variant == "LAST_LAYER_ONLY":
            model.repair1.W.requires_grad_(False)
        loaded = model.load_state_dict(original_state, strict=False)
        expected_missing = {"repair1.W", "repair2.W"}
        if set(loaded.missing_keys) != expected_missing or loaded.unexpected_keys:
            raise RuntimeError(f"PHASE5-ORIGINAL-LOAD-MISMATCH: {loaded}")
        trainer = VariantTrainer(args, variant)
        actual = tuple(name for name, p in model.named_parameters() if p.requires_grad)
        if actual != trainer.ALLOWLIST:
            raise RuntimeError(f"PHASE5-TRAINABLE-ALLOWLIST-MISMATCH: {actual}")
        return model, trainer, {
            "missing_keys": sorted(loaded.missing_keys), "unexpected_keys": list(loaded.unexpected_keys),
            "trainable_parameter_names": list(actual),
            "trainable_parameter_count": sum(int(p.numel()) for p in model.parameters() if p.requires_grad),
            "operator": variant,
        }
    base._ORIGINAL_BUILD_STUDENT = base.build_student
    base.build_student = build_student
    def preflight(*, seed, arm, student, trainer, teacher, original, runtime, graph, sampler, static_bundle, evaluation_bundle, profile):
        expected = tuple(trainer.ALLOWLIST)
        actual = tuple(name for name, p in student.named_parameters() if p.requires_grad)
        if actual != expected:
            raise RuntimeError(f"PHASE5-ALLOWLIST-MISMATCH: {actual}")
        parity = base._arm_parity(arm="LRR-P2", original=original, student=student, runtime=runtime)
        if not trainer.support_outside_residual_exact_zero(student, runtime):
            raise RuntimeError("PHASE5-SUPPORT-OUTSIDE-RESIDUAL-FAIL")
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        targets = trainer.build_teacher_targets(teacher, runtime, static_bundle)
        negatives = sampler.sample_epoch(0, step=0)
        student.train(); student.zero_grad(set_to_none=True)
        z = trainer.student_forward(student, runtime)
        values = profile["training_profile"]
        loss, loss_values = base.shared_lrr_objective(
            model=student, student_z=z, directed_df_edge_index=runtime.directed_df_edge_index,
            forget_negative_edge_index=negatives.forget_negative.to(runtime.x.device),
            local_edges=static_bundle.local_static.to(runtime.x.device), teacher_local_probability=targets.local_probability,
            retain_positive_edges=static_bundle.retain_positive_static.to(runtime.x.device), retain_negative_edges=negatives.retain_negative.to(runtime.x.device),
            anchor_edges=static_bundle.anchor_static.to(runtime.x.device), teacher_anchor_probability=targets.anchor_probability,
            objective_weights=values["objective_weights"], margin=float(values["forget_margin"]),
        )
        loss.backward()
        gradients = {}
        for name, p in student.named_parameters():
            if name in expected:
                if p.grad is None or not bool(torch.isfinite(p.grad).all()):
                    raise RuntimeError(f"PHASE5-GRADIENT-FAIL: {name}")
                gradients[name] = float(torch.linalg.vector_norm(p.grad))
            elif p.grad is not None:
                raise RuntimeError(f"PHASE5-FROZEN-GRADIENT: {name}")
        if set(gradients) != set(expected) or any(v <= 0 for v in gradients.values()):
            raise RuntimeError(f"PHASE5-INACTIVE-GRADIENT: {gradients}")
        student.zero_grad(set_to_none=True)
        optimizer = torch.optim.Adam(tuple(p for p in student.parameters() if p.requires_grad), lr=0.0003, weight_decay=0.0)
        return {
            "status": "PASS", "seed": seed, "arm": variant, "device": str(runtime.x.device),
            "strict_original_load": "PASS", "parity": parity,
            "support_outside_residual_exact_zero": True,
            "trainable_parameter_names": list(expected),
            "trainable_parameter_count": sum(int(p.numel()) for p in student.parameters() if p.requires_grad),
            "optimizer": {"type": "torch.optim.Adam", "learning_rate": 0.0003, "weight_decay": 0.0},
            "autograd": {"loss_values": loss_values, "gradient_norms": gradients, "backbone_gradients_absent": True, "teacher_gradients_absent": True},
            "objective_callable": "framework.trainer.shared_objective.shared_lrr_objective",
            "no_test_selection": True,
        }, expected, tuple(p for p in student.parameters() if p.requires_grad)
    base._arm_preflight = preflight


def gate_metadata(formal, graph, runtime, variant: str) -> dict[str, Any]:
    _, metadata = make_variant_runtime(formal, graph, runtime.x.device, variant)
    metadata["support1_sha256"] = formal.BASE.tensor_hash_v1(runtime.support1.detach().cpu()) if hasattr(formal.BASE, "tensor_hash_v1") else hashlib.sha256(runtime.support1.detach().cpu().numpy().tobytes()).hexdigest()
    metadata["support2_sha256"] = formal.BASE.tensor_hash_v1(runtime.support2.detach().cpu()) if hasattr(formal.BASE, "tensor_hash_v1") else hashlib.sha256(runtime.support2.detach().cpu().numpy().tobytes()).hexdigest()
    return metadata


def select_and_evaluate(formal, row: Mapping[str, Any], variant: str, out: Path, device_name: str, parameter_count: int) -> dict[str, Any]:
    """Apply the final runner's validation-only strict-sum selector."""
    base = formal.BASE
    history = json.loads((out / "validation_history.json").read_text(encoding="utf-8"))["events"]
    if len(history) != 30:
        raise RuntimeError(f"{variant} seed {row['seed']}: validation history count {len(history)}")
    chosen = None
    for event in history:
        candidate = (float(event["dt_auc"]) + float(event["df_auc"]), -int(event["epoch"]), event)
        if chosen is None or candidate[:2] > chosen[:2]:
            chosen = candidate
    if chosen is None:
        raise RuntimeError("empty validation history")
    event = chosen[2]
    epoch = int(event["epoch"])
    checkpoint_path = out / "checkpoints" / f"validation_epoch_{epoch:04d}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    device = torch.device(device_name)
    graph = base.build_graph(int(row["seed"]), require_prepared_request=True)
    _, _, evaluation = base.build_sampling_artifacts(graph)
    runtime = base.build_runtime(graph, device)
    original_state = base.checkpoint_state(base.original_checkpoint(int(row["seed"])))
    original = base.build_base_model(int(row["seed"]), original_state).to(device).eval()
    model_cpu, trainer, _ = base._load_arm_checkpoint_strict(int(row["seed"]), "LRR-P2", checkpoint_path, original_state)
    model = model_cpu.to(device).eval()
    with torch.no_grad():
        original_z = original(runtime.x, runtime.dr_mp_edge_index)
        z = trainer.student_forward(model, runtime)
        test = base.evaluate_full(model=model, z=z, graph=graph, evaluation_bundle=evaluation, split="test", original_model=original, original_z=original_z)
    for key in ("dt_auc", "dt_aup", "df_auc", "df_aup"):
        if not math.isfinite(float(test[key])):
            raise RuntimeError(f"nonfinite selected test metric {key}")
    selector = {"status": "PASS", "selector_id": "S6-GNNDELETE-SUM-v1", "selection_rule": "strict max validation DT_AUC + Df_AUC; exact tie -> earlier", "selected_epoch": epoch, "selected_update": epoch + 1, "validation": event, "test_used_for_selection": False}
    (out / "strict_max_selection.json").write_text(json.dumps(selector, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {
        "status": "COMPLETE", "scientific_status": "COMPLETE", "cell_id": row["cell_id"], "variant": variant,
        "dataset": row["dataset"], "architecture": row["architecture"], "deletion_type": row["deletion_type"], "ratio": row["ratio"], "seed": row["seed"],
        "selector_id": selector["selector_id"], "selection_rule": selector["selection_rule"], "selected_checkpoint": str(checkpoint_path.resolve()), "selected_checkpoint_epoch": epoch,
        "selected_validation_metrics": event, "DT_AUC": test["dt_auc"], "DT_AUP": test["dt_aup"], "Df_AUC": test["df_auc"], "Df_AUP": test["df_aup"],
        "trainable_params": parameter_count, "total_optimizer_updates": 3000, "test_metrics_used_for_selection": False, "retrain_metrics_used_for_selection": False,
        "formal_runner_family": row["formal_runner_family"], "formal_runner": str(FORMAL_SOURCE), "implementation_gate": {"variant": variant, "parameter_count": parameter_count},
        "protocol_completion": json.loads((out / "final_summary.json").read_text(encoding="utf-8")),
    }
    (out / "matrix_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def canary(seed: int, variant: str, device_name: str) -> dict[str, Any]:
    row = load_manifest_row(seed)
    formal = import_formal()
    out = PHASE5_ROOT / "canary" / variant / f"seed{seed}"
    if out.exists():
        raise FileExistsError(out)
    configuration = configure(formal, row, variant, out)
    install_student(formal, variant)
    formal.validate_p2_profile()
    device = torch.device(device_name)
    graph = formal.BASE.build_graph(seed, require_prepared_request=True)
    sampler, static_bundle, evaluation_bundle = formal.BASE.build_sampling_artifacts(graph)
    runtime, gate = make_variant_runtime(formal, graph, device, variant)
    original_state = formal.BASE.checkpoint_state(formal.BASE.original_checkpoint(seed))
    original = formal.BASE.build_base_model(seed, original_state).to(device).eval()
    teacher = formal.BASE.build_base_model(seed, original_state).to(device).eval()
    student, trainer, load_audit = formal.BASE.build_student(seed, "LRR-P2", original_state)
    student = student.to(device)
    preflight, expected, _ = formal.BASE._arm_preflight(seed=seed, arm="LRR-P2", student=student, trainer=trainer, teacher=teacher, original=original, runtime=runtime, graph=graph, sampler=sampler, static_bundle=static_bundle, evaluation_bundle=evaluation_bundle, profile=formal.FORMAL_PROFILE)
    with torch.no_grad():
        original_z = original(runtime.x, runtime.dr_mp_edge_index)
        variant_z = trainer.student_forward(student, runtime)
    identity = torch.allclose(original_z, variant_z, rtol=1e-5, atol=1e-6)
    if not identity:
        raise RuntimeError(f"{variant}: update0 identity failed")
    backbone_frozen = all(not p.requires_grad for n, p in student.named_parameters() if not n.startswith("repair"))
    if not backbone_frozen:
        raise RuntimeError(f"{variant}: backbone not frozen")
    payload = {
        "status": "PASS", "kind": "seed42_canary", "variant": variant, "seed": seed,
        "formal_cell_id": row["cell_id"], "configuration": configuration,
        "formal_original_checkpoint_sha256": formal.BASE.file_sha256(formal.BASE.original_checkpoint(seed)),
        "formal_graph_identity_sha256": graph["graph_identity"]["graph_identity_sha256"],
        "formal_request_path": str(formal.BASE.request_path(seed)),
        "p2": {"status": "PASS", "profile_sha256": formal.FORMAL_PROFILE_HASH, "epochs": 3000, "validation_frequency": 100, "objective": "unchanged", "sampling": "unchanged", "evaluator": "unchanged", "selector": "S6-GNNDELETE-SUM-v1 / strict max validation DT_AUC + Df_AUC; earlier tie", "test_used_for_selection": False},
        "operator": {"trainable_parameter_names": list(expected), "trainable_parameter_count": sum(int(p.numel()) for p in student.parameters() if p.requires_grad), "backbone_frozen": backbone_frozen, "update0_identity": True, "load_audit": load_audit},
        "gate": gate,
        "preflight": preflight,
        "static_bundle_sha256": static_bundle.metadata["static_bundle_sha256"], "evaluation_bundle_sha256": evaluation_bundle.metadata["total_bundle_sha256"],
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "implementation_gate.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def run(seed: int, variant: str, device_name: str) -> dict[str, Any]:
    row = load_manifest_row(seed)
    formal = import_formal()
    out = PHASE5_ROOT / "runs" / variant / f"seed{seed}"
    if out.exists():
        raise FileExistsError(out)
    marker = PHASE5_ROOT / "manifests" / "prepare_complete.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        marker.write_text(json.dumps({
            "status": "PASS", "phase": "Phase5 patent ablation",
            "formal_source": str(FORMAL_SOURCE), "formal_mainline_changed": False,
            "Df_reused": True, "Original_reused": True, "training": "variant runs only",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    configuration = configure(formal, row, variant, out)
    install_student(formal, variant)
    formal.BASE.train_arm(seed, "LRR-P2", device_name)
    formal.postprocess_arm(seed, "LRR-P2")
    selected = select_and_evaluate(formal, row, variant, out, device_name, 20480 if variant != "LAST_LAYER_ONLY" else 4096)
    summary = json.loads((out / "final_summary.json").read_text(encoding="utf-8"))
    history = json.loads((out / "validation_history.json").read_text(encoding="utf-8"))
    gate = json.loads((PHASE5_ROOT / "canary" / variant / f"seed{seed}" / "implementation_gate.json").read_text(encoding="utf-8")) if (PHASE5_ROOT / "canary" / variant / f"seed{seed}" / "implementation_gate.json").is_file() else {}
    selected_metrics = selected
    result = {
        "status": "COMPLETE", "variant": variant, "seed": seed, "cell_id": row["cell_id"],
        "DT_AUC": float(selected_metrics["DT_AUC"]), "Df_AUC": float(selected_metrics["Df_AUC"]),
        "J_AUC": (float(selected_metrics["DT_AUC"]) + float(selected_metrics["Df_AUC"])) / 2.0,
        "DT_AUP": float(selected_metrics["DT_AUP"]), "Df_AUP": float(selected_metrics["Df_AUP"]),
        "J_AUP": (float(selected_metrics["DT_AUP"]) + float(selected_metrics["Df_AUP"])) / 2.0,
        "selected_validation_update": int(selected_metrics["selected_checkpoint_epoch"]) + 1,
        "selected_validation_epoch": int(selected_metrics["selected_checkpoint_epoch"]),
        "trainable_parameter_count": int(selected_metrics["trainable_params"]),
        "selected_checkpoint": selected_metrics["selected_checkpoint"],
        "selected_validation_metrics": selected_metrics["selected_validation_metrics"],
        "total_optimizer_updates": int(summary["optimizer_steps_completed"]),
        "test_used_for_selection": False, "retrain_used_for_selection": False,
        "formal_original_checkpoint": configuration["formal_original_checkpoint"],
        "formal_request_path": configuration["formal_request_path"], "formal_graph_path": configuration["formal_graph_path"],
        "p2_profile_sha256": formal.FORMAL_PROFILE_HASH, "selector_id": "S6-GNNDELETE-SUM-v1",
        "selector_rule": "strict max validation DT_AUC + Df_AUC; exact tie -> earlier",
        "graph_identity_sha256": summary["graph_identity_sha256"],
        "run_summary_path": str(out / "final_summary.json"), "validation_history_path": str(out / "validation_history.json"),
        "checkpoint_dir": str(out / "checkpoints"), "configuration": configuration, "canary": gate,
    }
    (out / "variant_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def finalize(seed: int, variant: str, device_name: str) -> dict[str, Any]:
    """Complete postprocessing for a run whose 3000 updates already exist."""
    row = load_manifest_row(seed)
    formal = import_formal()
    out = PHASE5_ROOT / "runs" / variant / f"seed{seed}"
    if not out.is_dir() or not (out / "final_summary.json").is_file():
        raise FileNotFoundError(f"completed run not found: {out}")
    configuration = configure(formal, row, variant, out)
    install_student(formal, variant)
    formal.postprocess_arm(seed, "LRR-P2")
    parameter_count = 4096 if variant == "LAST_LAYER_ONLY" else 20480
    selected = select_and_evaluate(formal, row, variant, out, device_name, parameter_count)
    summary = json.loads((out / "final_summary.json").read_text(encoding="utf-8"))
    result = {
        "status": "COMPLETE", "variant": variant, "seed": seed, "cell_id": row["cell_id"],
        "DT_AUC": float(selected["DT_AUC"]), "Df_AUC": float(selected["Df_AUC"]),
        "J_AUC": (float(selected["DT_AUC"]) + float(selected["Df_AUC"])) / 2.0,
        "DT_AUP": float(selected["DT_AUP"]), "Df_AUP": float(selected["Df_AUP"]),
        "J_AUP": (float(selected["DT_AUP"]) + float(selected["Df_AUP"])) / 2.0,
        "selected_validation_update": int(selected["selected_checkpoint_epoch"]) + 1,
        "selected_validation_epoch": int(selected["selected_checkpoint_epoch"]),
        "trainable_parameter_count": parameter_count,
        "selected_checkpoint": selected["selected_checkpoint"],
        "selected_validation_metrics": selected["selected_validation_metrics"],
        "total_optimizer_updates": int(summary["optimizer_steps_completed"]),
        "test_used_for_selection": False, "retrain_used_for_selection": False,
        "formal_original_checkpoint": configuration["formal_original_checkpoint"],
        "formal_request_path": configuration["formal_request_path"], "formal_graph_path": configuration["formal_graph_path"],
        "p2_profile_sha256": formal.FORMAL_PROFILE_HASH, "selector_id": "S6-GNNDELETE-SUM-v1",
        "selector_rule": "strict max validation DT_AUC + Df_AUC; exact tie -> earlier",
        "graph_identity_sha256": summary["graph_identity_sha256"],
        "run_summary_path": str(out / "final_summary.json"), "validation_history_path": str(out / "validation_history.json"),
        "checkpoint_dir": str(out / "checkpoints"), "configuration": configuration,
    }
    (out / "variant_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.seed not in SEEDS:
        raise ValueError(args.seed)
    if args.canary and args.finalize:
        raise ValueError("--canary and --finalize are mutually exclusive")
    value = canary(args.seed, args.variant, args.device) if args.canary else (finalize(args.seed, args.variant, args.device) if args.finalize else run(args.seed, args.variant, args.device))
    print(json.dumps({"status": value["status"], "variant": args.variant, "seed": args.seed}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
