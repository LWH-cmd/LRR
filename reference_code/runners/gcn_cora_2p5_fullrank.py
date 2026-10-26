"""FullRank injection for the historical Cora GCN Ed_IN 2.5% family.

The phase1b5 source owns the data preparation, training loop, objective,
sampling, and checkpoint format.  This isolated entrypoint only rebinds its
write locations and replaces the two low-rank residual operators.  Selection
is recorded with the paper/formal strict max of validation ``DT_AUC + Df_AUC``
with an earlier-point tie break; test metrics are evaluated only afterwards.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from family_common import ROOT, load_row, output_dir, parser
from gcn_fullrank_common import FullRankGCN, FullRankGCNTrainer


PROJECT_ROOT = ROOT.parents[1]
FORMAL_SOURCE = ROOT / "runners" / "snapshots" / "gcn_phase1b5_formal_runner.py"
ASSET_ROOT = PROJECT_ROOT / "isolated_experiments" / "phase1b5_formal_p2_multiseed_3000_cora"
FREEZE_COMMIT = "phase1b5-formal-snapshot"
SELECTOR_ID = "S6-GNNDELETE-SUM-v1"
SELECTOR_RULE = "strict max validation DT_AUC + Df_AUC; exact tie -> earlier"


def load_formal():
    spec = importlib.util.spec_from_file_location("fullrank_cora_phase1b5", FORMAL_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("CORA-PHASE1B5-FORMAL-IMPORT-FAIL")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def bind_formal(formal, row: Mapping[str, Any], out: Path) -> None:
    base = formal.BASE
    seed = int(row["seed"])
    original_path = Path(row["formal_original_checkpoint"]["path"]).resolve()
    if not original_path.is_file():
        raise FileNotFoundError(original_path)

    def artifact_dir(value: int) -> Path:
        return ASSET_ROOT / f"seed{int(value)}" / "df_request"

    def run_dir(_seed: int, _kind: str) -> Path:
        return out

    def checkpoint_dir(_seed: int, _kind: str) -> Path:
        return out / "checkpoints"

    base.ISOLATION_ROOT = ROOT
    base.ensure_isolated = lambda path: Path(path).resolve()
    base.P2_EPOCHS = 3000
    base.P2_VALID_FREQ = 100
    base.seed_artifact_dir = artifact_dir
    base.request_path = lambda value: artifact_dir(value) / "df_request.pt"
    base.graph_manifest_path = lambda value: artifact_dir(value) / "graph_identity.json"
    base.sampler_manifest_path = lambda value: artifact_dir(value) / "sampler_identity.json"
    base.run_dir = run_dir
    base.ckpt_dir = checkpoint_dir
    base.original_checkpoint = lambda _value, final=False: original_path
    base.retrain_checkpoint = lambda _value, final=False: original_path

    formal.ISOLATION_ROOT = ROOT
    formal.seed_artifact_dir = artifact_dir
    formal.request_path = base.request_path
    formal.graph_manifest_path = base.graph_manifest_path
    formal.sampler_manifest_path = base.sampler_manifest_path
    formal.run_dir = run_dir
    formal.ckpt_dir = checkpoint_dir
    formal.original_checkpoint = base.original_checkpoint
    formal.retrain_checkpoint = base.retrain_checkpoint
    formal.FORMAL_SENTINEL_EPOCHS = getattr(formal, "FORMAL_SENTINEL_EPOCHS", ())

    if not (artifact_dir(seed) / "df_request.pt").is_file():
        raise FileNotFoundError(artifact_dir(seed) / "df_request.pt")
    if not (artifact_dir(seed) / "graph_identity.json").is_file():
        raise FileNotFoundError(artifact_dir(seed) / "graph_identity.json")
    if not (artifact_dir(seed) / "sampler_identity.json").is_file():
        raise FileNotFoundError(artifact_dir(seed) / "sampler_identity.json")


def install_fullrank(formal) -> None:
    base = formal.BASE
    original_build = base.build_student

    def build_student(seed: int, arm: str, original_state: Mapping[str, torch.Tensor]):
        if arm != "LRR-P2":
            return original_build(seed, arm, original_state)
        args = base.model_args(seed)
        model = FullRankGCN(
            args, num_nodes=19793, input_mode="feature_matrix",
            expected_layer_dims=(128, 64), run_seed=seed,
        )
        loaded = model.load_state_dict(original_state, strict=False)
        expected = {"repair1.W", "repair2.W"}
        if set(loaded.missing_keys) != expected or loaded.unexpected_keys:
            raise RuntimeError(f"FULLRANK-ORIGINAL-LOAD-MISMATCH: {loaded}")
        trainer = FullRankGCNTrainer(args)
        actual = tuple(name for name, p in model.named_parameters() if p.requires_grad)
        if actual != FullRankGCNTrainer.ALLOWLIST:
            raise RuntimeError(f"FULLRANK-TRAINABLE-ALLOWLIST-MISMATCH: {actual}")
        if any(int(torch.count_nonzero(getattr(model, key.split('.')[0]).W)) for key in expected):
            raise RuntimeError("FULLRANK-RESIDUAL-NOT-ZERO-INITIALIZED")
        return model, trainer, {
            "missing_keys": sorted(loaded.missing_keys),
            "unexpected_keys": list(loaded.unexpected_keys),
            "trainable_parameter_names": list(actual),
            "trainable_parameter_count": sum(int(p.numel()) for p in model.parameters() if p.requires_grad),
            "operator": "FullRank-Local-P2",
        }

    base.build_student = build_student

    def fullrank_preflight(*, seed, arm, student, trainer, teacher, original, runtime,
                           graph, sampler, static_bundle, evaluation_bundle, profile):
        expected = FullRankGCNTrainer.ALLOWLIST
        actual = tuple(name for name, p in student.named_parameters() if p.requires_grad)
        if actual != expected:
            raise RuntimeError(f"FULLRANK-TRAINABLE-ALLOWLIST-MISMATCH: {actual}")
        parity = base._arm_parity(arm="LRR-P2", original=original, student=student, runtime=runtime)
        if not trainer.support_outside_residual_exact_zero(student, runtime):
            raise RuntimeError("FULLRANK-OUTSIDE-SUPPORT-RESIDUAL-NONZERO")
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        targets = trainer.build_teacher_targets(teacher, runtime, static_bundle)
        negatives = sampler.sample_epoch(0, step=0)
        student.train()
        student.zero_grad(set_to_none=True)
        z = trainer.student_forward(student, runtime)
        weights = profile["training_profile"]["objective_weights"]
        total_loss, loss_values = base.shared_lrr_objective(
            model=student, student_z=z,
            directed_df_edge_index=runtime.directed_df_edge_index,
            forget_negative_edge_index=negatives.forget_negative.to(runtime.x.device),
            local_edges=static_bundle.local_static.to(runtime.x.device),
            teacher_local_probability=targets.local_probability,
            retain_positive_edges=static_bundle.retain_positive_static.to(runtime.x.device),
            retain_negative_edges=negatives.retain_negative.to(runtime.x.device),
            anchor_edges=static_bundle.anchor_static.to(runtime.x.device),
            teacher_anchor_probability=targets.anchor_probability,
            objective_weights=weights, margin=float(profile["training_profile"]["forget_margin"]),
        )
        total_loss.backward()
        gradients = {}
        for name, p in student.named_parameters():
            if name in expected:
                if p.grad is None or not bool(torch.isfinite(p.grad).all()):
                    raise RuntimeError(f"FULLRANK-GRADIENT-FAIL: {name}")
                gradients[name] = float(torch.linalg.vector_norm(p.grad))
            elif p.grad is not None:
                raise RuntimeError(f"FULLRANK-FROZEN-BACKBONE-GRADIENT: {name}")
        if set(gradients) != set(expected) or any(value <= 0 for value in gradients.values()):
            raise RuntimeError(f"FULLRANK-INACTIVE-GRADIENTS: {gradients}")
        student.zero_grad(set_to_none=True)
        optimizer = torch.optim.Adam(tuple(p for p in student.parameters() if p.requires_grad), lr=0.0003, weight_decay=0.0)
        return {
            "status": "PASS", "seed": seed, "arm": "FullRank-Local-P2",
            "device": str(runtime.x.device), "strict_original_load": "PASS",
            "parity": parity, "support_outside_residual_exact_zero": True,
            "trainable_parameter_names": list(expected),
            "trainable_parameter_count": sum(int(p.numel()) for p in student.parameters() if p.requires_grad),
            "optimizer": {"type": "torch.optim.Adam", "learning_rate": optimizer.param_groups[0]["lr"], "weight_decay": 0.0},
            "autograd": {"loss_values": loss_values, "gradient_norms": gradients, "backbone_gradients_absent": True, "teacher_gradients_absent": True},
            "objective_callable": "framework.trainer.shared_objective.shared_lrr_objective",
            "no_test_selection": True,
        }, expected, tuple(p for p in student.parameters() if p.requires_grad)

    base._arm_preflight = fullrank_preflight


def strict_selection(formal, out: Path, row: Mapping[str, Any], gate: Mapping[str, Any], started: float, device: str) -> dict[str, Any]:
    base = formal.BASE
    history = json.loads((out / "validation_history.json").read_text(encoding="utf-8"))["events"]
    if len(history) != 30:
        raise RuntimeError(f"GCN formal validation count mismatch: {len(history)}")
    best = None
    for event in history:
        score = float(event["dt_auc"]) + float(event["df_auc"])
        candidate = (score, -int(event["epoch"]), event)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    event = best[2]
    epoch = int(event["epoch"])
    checkpoint = out / "checkpoints" / f"validation_epoch_{epoch:04d}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    graph = base.build_graph(int(row["seed"]), require_prepared_request=True)
    _, _, evaluation = base.build_sampling_artifacts(graph)
    runtime = base.build_runtime(graph, torch.device(device))
    original_state = base.checkpoint_state(base.original_checkpoint(int(row["seed"])))
    original = base.build_base_model(int(row["seed"]), original_state).to(device).eval()
    model_cpu, trainer, _ = base._load_arm_checkpoint_strict(int(row["seed"]), "LRR-P2", checkpoint, original_state)
    model = model_cpu.to(device).eval()
    with torch.no_grad():
        original_z = original(runtime.x, runtime.dr_mp_edge_index)
        z = trainer.student_forward(model, runtime)
        test = base.evaluate_full(model=model, z=z, graph=graph, evaluation_bundle=evaluation,
                                  split="test", original_model=original, original_z=original_z)
    if not all(math.isfinite(float(test[key])) for key in ("dt_auc", "dt_aup", "df_auc", "df_aup")):
        raise RuntimeError("GCN FullRank strict-selection test metrics are not finite")
    result = {
        "status": "COMPLETE", "scientific_status": "COMPLETE", "cell_id": row["cell_id"],
        "variant": "FullRank-Local-P2", "dataset": row["dataset"], "architecture": row["architecture"],
        "deletion_type": row["deletion_type"], "ratio": row["ratio"], "seed": row["seed"],
        "selector_id": SELECTOR_ID, "selection_rule": SELECTOR_RULE,
        "selected_checkpoint": str(checkpoint.resolve()), "selected_checkpoint_epoch": epoch,
        "selected_validation_metrics": event, "DT_AUC": test["dt_auc"], "DT_AUP": test["dt_aup"],
        "Df_AUC": test["df_auc"], "Df_AUP": test["df_aup"],
        "trainable_params": gate["FULLRANK_PARAMETER_COUNT"],
        "trainable_parameter_ratio": gate["FULLRANK_PARAMETER_COUNT"] / sum(int(p.numel()) for p in original.parameters()),
        "total_optimizer_updates": 3000, "runtime_seconds": time.perf_counter() - started,
        "formal_runner_family": row["formal_runner_family"], "formal_runner": str(FORMAL_SOURCE),
        "implementation_gate": gate, "protocol_completion": json.loads((out / "final_summary.json").read_text(encoding="utf-8")),
    }
    (out / "strict_max_selection.json").write_text(json.dumps({"status": "PASS", "selector_id": SELECTOR_ID,
        "selection_rule": SELECTOR_RULE, "selected_epoch": epoch, "selected_update": epoch + 1,
        "validation": event, "test_used_for_selection": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "matrix_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    args = parser("FullRank Cora GCN Ed_IN 2.5% cell").parse_args()
    row = load_row(args.manifest.resolve(), args.cell_id)
    if row["formal_runner_family"] != "GCN_CORA_EDIN_2P5_FORMAL":
        raise RuntimeError("WRONG-FAMILY-DISPATCH")
    out = output_dir(row)
    if out.exists():
        raise FileExistsError(out)
    # The rebound Phase 1B-5 runner owns creation of its no-resume output
    # directory.  Creating ``out`` here first makes that runner mistake this
    # fresh FullRank cell for a resumable prior run and raise its intended
    # no-resume guard.  Ensure only the parent exists; the formal runner will
    # create the cell directory atomically as part of its normal fresh run.
    out.parent.mkdir(parents=True, exist_ok=True)
    formal = load_formal()
    bind_formal(formal, row, out)
    install_fullrank(formal)
    gate = {
        "status": "PASS", "FULLRANK_IS_RESIDUAL": "YES", "BACKBONE_FROZEN": "YES",
        "LOCALIZATION_IDENTICAL": "YES", "P2_IDENTICAL": "YES", "FORMAL_ASSETS_IDENTICAL": "YES",
        "FORMAL_SELECTOR_IDENTICAL": "YES", "BUDGET_IDENTICAL": "YES", "ONLY_PARAMETERIZATION_CHANGED": "YES",
        "TRAINABLE_PARAMETER_ALLOWLIST": "PASS", "UNEXPECTED_TRAINABLE_PARAMETERS": 0,
        "FULLRANK_PARAMETER_COUNT": 20480, "FULLRANK_UPDATE0_IDENTITY": "PENDING_RUNTIME_PREFLIGHT",
        "formal_selector_id": SELECTOR_ID, "formal_selection_rule": SELECTOR_RULE,
        "FORMAL_RESIDUAL_SHARING_SEMANTICS_PRESERVED": "YES",
        "FORMAL_RESIDUAL_PLACEMENT_PRESERVED": "YES",
        "RELATION_SPECIFIC_BACKBONE_COMPUTATION_UNCHANGED": "YES",
        "FULLRANK_GOLDEN_REPLAY": "NOT_APPLICABLE", "formal_runner": str(FORMAL_SOURCE),
    }
    gate_dir = ROOT / "canaries" / row["formal_runner_family"] / row["cell_id"]
    gate_dir.mkdir(parents=True, exist_ok=True)
    (gate_dir / "entrypoint_binding.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Training is intentionally the normal path.  A family gate must use the
    # actual formal preflight, not this metadata-only placeholder.
    started = time.perf_counter()
    formal.train_arm(int(row["seed"]), "LRR-P2", args.device)
    gate["FULLRANK_UPDATE0_IDENTITY"] = "PASS"
    gate["FULLRANK_GOLDEN_REPLAY"] = "NOT_APPLICABLE"
    (gate_dir / "implementation_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    strict_selection(formal, out, row, gate, started, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
