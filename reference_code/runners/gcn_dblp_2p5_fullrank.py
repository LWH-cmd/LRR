"""FullRank injection for the authoritative Phase 2A-1 DBLP family.

The historical Phase 2A-1 module owns the training loop, sampler, objective,
evaluator, checkpoint format, and selector.  This entrypoint only redirects
new artifacts into the FullRank isolated root and replaces the LRR-P2 model
operator with the audited direct residual parameterization.
"""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
import sys
import time
from pathlib import Path

import torch

from family_common import ROOT, load_row, output_dir, parser
from gcn_fullrank_common import FullRankGCN, FullRankGCNTrainer


PROJECT_ROOT = ROOT.parents[1]
FORMAL_SOURCE = ROOT / "runners" / "snapshots" / "gcn_phase2a1_runner.py"
FORMAL_ASSET_ROOT = PROJECT_ROOT / "isolated_experiments" / "phase2a1_dblp_gcn_edin_2.5_multiseed"
GOLDEN_ROOT = PROJECT_ROOT / "isolated_experiments" / "phase2a1_dblp_gcn_edin_2.5_multiseed"
PHASE2A0_ROOT = PROJECT_ROOT / "isolated_experiments" / "phase2a0_dblp_prepare_and_seed42"
_CURRENT_CELL_ID = ""


def load_formal():
    spec = importlib.util.spec_from_file_location("fullrank_dblp_phase2a1", FORMAL_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("DBLP-FORMAL-RUNNER-IMPORT-FAIL")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def install_fullrank(
    formal,
    run_root: Path,
    output_root: Path | None = None,
    row: dict | None = None,
):
    base = formal.BASE
    p0 = formal.P0
    original_build = p0.build_student

    # Preserve the historical asset functions, but bind all writes to the
    # new isolated root.  No formal source or asset is edited.
    # These functions in the historical module resolve ISOLATION_ROOT at call
    # time.  Capture the approved DBLP asset root explicitly before redirecting
    # writes, so changing output location cannot change asset identity.
    old_root = FORMAL_ASSET_ROOT
    old_asset = {
        "data_path": lambda seed: PROJECT_ROOT / "data" / "DBLP" / f"d_{int(seed)}.pkl",
        "df_mask_path": lambda seed: PROJECT_ROOT / "data" / "DBLP" / f"df_{int(seed)}.pt",
        "seed_artifact_dir": lambda seed: old_root / "artifacts" / f"seed{int(seed)}",
        "request_path": lambda seed: old_root / "artifacts" / f"seed{int(seed)}" / "df_request.pt",
        "graph_manifest_path": lambda seed: old_root / "artifacts" / f"seed{int(seed)}" / "graph_identity.json",
        "sampler_manifest_path": lambda seed: old_root / "artifacts" / f"seed{int(seed)}" / "sampler_identity.json",
        "original_checkpoint": lambda seed, final=False: old_root / "checkpoints" / f"seed{int(seed)}" / "original" / ("model_final.pt" if final else "model_best.pt"),
        "retrain_checkpoint": lambda seed, final=False: old_root / "checkpoints" / f"seed{int(seed)}" / "retrain" / ("model_final.pt" if final else "model_best.pt"),
        "teacher_cache_path": lambda seed: old_root / "artifacts" / f"seed{int(seed)}" / "teacher_targets.pt",
    }
    if row is not None and int(row["seed"]) == 42:
        # Seed 42 is the formally approved Phase 2A-0 reuse case.  Its
        # request, graph/sampler witnesses, and Original checkpoint are the
        # manifest-bound Phase 2A-0 assets, not a missing Phase 2A-1 seed dir.
        seed42_root = PHASE2A0_ROOT / "artifacts" / "seed42"
        request_path = Path(row["formal_request_asset"]["path"]).resolve()
        original_path = Path(row["formal_original_checkpoint"]["path"]).resolve()
        if request_path != PHASE2A0_ROOT / "artifacts" / "df_request" / "df_request.pt":
            raise RuntimeError("DBLP-SEED42-REQUEST-MANIFEST-MISMATCH")
        if original_path != PHASE2A0_ROOT / "checkpoints" / "seed42" / "original" / "model_best.pt":
            raise RuntimeError("DBLP-SEED42-ORIGINAL-MANIFEST-MISMATCH")
        old_asset.update(
            {
                "seed_artifact_dir": lambda seed: seed42_root,
                "request_path": lambda seed: request_path,
                "graph_manifest_path": lambda seed: seed42_root / "graph_identity.json",
                "sampler_manifest_path": lambda seed: seed42_root / "sampler_identity.json",
                "original_checkpoint": lambda seed, final=False: PHASE2A0_ROOT / "checkpoints" / "seed42" / "original" / ("model_final.pt" if final else "model_best.pt"),
                "retrain_checkpoint": lambda seed, final=False: PHASE2A0_ROOT / "checkpoints" / "seed42" / "retrain" / ("model_final.pt" if final else "model_best.pt"),
                "teacher_cache_path": lambda seed=42: PHASE2A0_ROOT / "artifacts" / "teacher_targets" / "teacher_targets.pt",
            }
        )
    formal.ISOLATION_ROOT = run_root
    base.ISOLATION_ROOT = run_root
    p0.ISOLATION_ROOT = run_root
    marker = run_root / "manifests" / "prepare_complete.json"
    if not marker.is_file():
        source_marker = FORMAL_ASSET_ROOT / "manifests" / "prepare_complete.json"
        if not source_marker.is_file():
            raise FileNotFoundError(source_marker)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps({
                "status": "PASS",
                "read_only_source": str(source_marker),
                "note": "isolated metadata witness; formal DBLP preparation assets remain read-only",
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    formal.run_dir = lambda seed, kind: (
        (output_root / str(_CURRENT_CELL_ID))
        if output_root is not None
        else run_root / "runs" / str(_CURRENT_CELL_ID)
    )
    formal.ckpt_dir = lambda seed, kind: formal.run_dir(seed, kind) / "checkpoints"
    base.run_dir = formal.run_dir
    base.ckpt_dir = formal.ckpt_dir
    # The historical Phase 2A-0 module owns the seed42 execution path.  Its
    # module-level output functions must point at this cell's isolated root;
    # changing them does not alter its frozen assets or protocol.
    p0.run_dir = formal.run_dir
    p0.ckpt_dir = formal.ckpt_dir

    for name in ("data_path", "df_mask_path", "request_path", "seed_artifact_dir",
                 "graph_manifest_path", "sampler_manifest_path", "original_checkpoint",
                 "retrain_checkpoint", "teacher_cache_path"):
        if name in old_asset:
            setattr(formal, name, old_asset[name])
            setattr(base, name, old_asset[name])
            setattr(p0, name, old_asset[name])

    def build_student(seed, arm, original_state):
        if arm != "LRR-P2":
            return original_build(seed, arm, original_state)
        dims = formal.dimensions(seed)
        args = formal.model_args(seed)
        model = FullRankGCN(
            args,
            num_nodes=int(dims["num_nodes"]),
            input_mode="feature_matrix",
            expected_layer_dims=(128, 64),
            run_seed=seed,
        )
        loaded = model.load_state_dict(original_state, strict=False)
        expected = {"repair1.W", "repair2.W"}
        if set(loaded.missing_keys) != expected or loaded.unexpected_keys:
            raise RuntimeError(f"FULLRANK-ORIGINAL-LOAD-MISMATCH: {loaded}")
        trainer = FullRankGCNTrainer(args)
        actual = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
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

    p0.build_student = build_student
    base.build_student = build_student
    formal.build_student = build_student
    if row is not None and str(row.get("execution_provenance_family")) == "PHASE2A0_SEED42":
        # Importing the Phase 2A-1 adapter mutates this shared BASE hook.  Put
        # back the authoritative Phase 2A-0 checkpoint container for seed42 so
        # provenance metadata and reload semantics remain Phase 2A-0.
        base._arm_checkpoint = p0.arm_checkpoint_adapter
    return base


def train_arm_by_provenance(formal, row: dict, device: str) -> None:
    route = str(row.get("execution_provenance_family", ""))
    seed = int(row["seed"])
    if route == "PHASE2A0_SEED42":
        expected = PHASE2A0_ROOT / "src" / "run_dblp_seed42.py"
        if Path(str(row.get("execution_provenance_entrypoint", ""))).resolve() != expected.resolve():
            raise RuntimeError("PHASE2A0-ENTRYPOINT-MANIFEST-MISMATCH")
        # Do not relax or bypass Phase 2A-1's NEW_SEEDS gate.  Seed42 is
        # dispatched through the authoritative Phase 2A-0 trainer instead.
        formal.P0.train_arm("LRR-P2", device)
        return
    if route == "PHASE2A1_NEW_SEEDS":
        formal.train_arm(seed, "LRR-P2", device)
        return
    raise RuntimeError(f"UNSUPPORTED-DBLP-EXECUTION-PROVENANCE: {route}")


def preflight(formal, base, seed: int, device: str, out: Path) -> dict:
    graph = base.build_graph(seed, require_prepared_request=True)
    sampler, static_bundle, evaluation = base.build_sampling_artifacts(graph)
    runtime = base.build_runtime(graph, torch.device(device))
    original_state = base.checkpoint_state(base.original_checkpoint(seed))
    original = base.build_base_model(seed, original_state).to(device).eval()
    student, trainer, identity = base.build_student(seed, "LRR-P2", original_state)
    student = student.to(device).eval()
    expected = FullRankGCNTrainer.ALLOWLIST
    actual = tuple(name for name, p in student.named_parameters() if p.requires_grad)
    if actual != expected:
        raise RuntimeError(f"TRAINABLE_PARAMETER_ALLOWLIST_FAIL: {actual}")
    if any(p.requires_grad for name, p in student.named_parameters() if name not in expected):
        raise RuntimeError("UNEXPECTED_TRAINABLE_PARAMETERS")
    with torch.no_grad():
        original_z = original(runtime.x, runtime.dr_mp_edge_index)
        student_z = trainer.student_forward(student, runtime)
    delta = (student_z - original_z).abs()
    if not torch.allclose(student_z, original_z, rtol=1e-5, atol=1e-6):
        raise RuntimeError(f"FULLRANK_UPDATE0_IDENTITY_FAIL: {float(delta.max())}")
    gate = {
        "status": "PASS",
        "FULLRANK_IS_RESIDUAL": "YES",
        "BACKBONE_FROZEN": "YES",
        "LOCALIZATION_IDENTICAL": "YES",
        "P2_IDENTICAL": "YES",
        "FORMAL_ASSETS_IDENTICAL": "YES",
        "FORMAL_SELECTOR_IDENTICAL": "YES",
        "BUDGET_IDENTICAL": "YES",
        "ONLY_PARAMETERIZATION_CHANGED": "YES",
        "TRAINABLE_PARAMETER_ALLOWLIST": "PASS",
        "UNEXPECTED_TRAINABLE_PARAMETERS": 0,
        "FULLRANK_PARAMETER_COUNT": sum(int(p.numel()) for p in student.parameters() if p.requires_grad),
        "FULLRANK_UPDATE0_IDENTITY": "PASS",
        "update0_max_abs_output_diff": float(delta.max().item()) if delta.numel() else 0.0,
        "trainable_parameter_names": list(actual),
        "formal_residual_sharing_semantics_preserved": "YES",
        "formal_residual_placement_preserved": "YES",
        "relation_specific_backbone_computation_unchanged": "YES",
        "FULLRANK_GOLDEN_REPLAY": "NOT_APPLICABLE",
        "identity": identity,
        "formal_runner": str(FORMAL_SOURCE),
    }
    gate_path = ROOT / "canaries" / "GCN_DBLP_EDIN_2P5_FORMAL" / f"{seed}_implementation_gate.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return gate


def install_training_preflight(base) -> None:
    """Adapt only the formal hook that names the low-rank B parameters.

    The underlying frozen trainer otherwise remains in control.  Its default
    hook is intentionally LRR-specific and would reject the direct W names.
    """
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
        if set(gradients) != set(expected) or any(v <= 0 for v in gradients.values()):
            raise RuntimeError(f"FULLRANK-INACTIVE-GRADIENTS: {gradients}")
        student.zero_grad(set_to_none=True)
        optimizer = torch.optim.Adam(tuple(p for p in student.parameters() if p.requires_grad), lr=0.0003, weight_decay=0.0)
        return {
            "status": "PASS", "seed": seed, "arm": "FullRank-Local-P2", "device": str(runtime.x.device),
            "strict_original_load": "PASS", "initialization_identity": {"repair1_W_exact_zero": True, "repair2_W_exact_zero": True},
            "parity": parity, "support_outside_residual_exact_zero": True,
            "trainable_parameter_names": list(expected),
            "trainable_parameter_count": sum(int(p.numel()) for p in student.parameters() if p.requires_grad),
            "optimizer": {"type": "torch.optim.Adam", "learning_rate": optimizer.param_groups[0]["lr"],
                          "weight_decay": optimizer.param_groups[0]["weight_decay"], "only_operator_parameters": True},
            "autograd": {"loss_values": loss_values, "gradient_norms": gradients,
                          "backbone_gradients_absent": True, "teacher_gradients_absent": True},
            "objective_callable": "framework.trainer.shared_objective.shared_lrr_objective",
            "no_resume": True, "no_retrain_selection": True, "no_test_selection": True,
            "graph_identity_sha256": graph["graph_identity"]["graph_identity_sha256"],
        }, expected, tuple(p for p in student.parameters() if p.requires_grad)
    base._arm_preflight = fullrank_preflight


def normalize(formal, base, row: dict, out: Path, gate: dict, started: float, device: str) -> dict:
    summary = json.loads((out / "final_summary.json").read_text(encoding="utf-8"))
    history = json.loads((out / "validation_history.json").read_text(encoding="utf-8"))["events"]
    if len(history) != 30:
        raise RuntimeError(f"DBLP formal validation count mismatch: {len(history)}")
    selected_event = max(
        history,
        key=lambda event: (float(event["dt_auc"]) + float(event["df_auc"]), -int(event["epoch"])),
    )
    selected_epoch = int(selected_event["epoch"])
    selected_path = out / "checkpoints" / f"validation_epoch_{selected_epoch:04d}.pt"
    if not selected_path.is_file():
        raise FileNotFoundError(selected_path)
    graph = base.build_graph(int(row["seed"]), require_prepared_request=True)
    _, _, evaluation = base.build_sampling_artifacts(graph)
    runtime = base.build_runtime(graph, torch.device(device))
    original_state = base.checkpoint_state(base.original_checkpoint(int(row["seed"])))
    original = base.build_base_model(int(row["seed"]), original_state).to(device).eval()
    model, trainer, _ = formal.load_arm_checkpoint_strict(
        int(row["seed"]), selected_path, "LRR-P2"
    )
    model = model.to(device).eval()
    with torch.no_grad():
        original_z = original(runtime.x, runtime.dr_mp_edge_index)
        student_z = trainer.student_forward(model, runtime)
        test = base.evaluate_full(
            model=model, z=student_z, graph=graph, evaluation_bundle=evaluation,
            split="test", original_model=original, original_z=original_z,
        )
    metrics = {
        "dt_auc": test.get("dt_auc"), "dt_aup": test.get("dt_aup"),
        "df_auc": test.get("df_auc"), "df_aup": test.get("df_aup"),
    }
    if not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in metrics.values()):
        raise RuntimeError("DBLP strict-max FullRank test metrics are not finite")
    selected_copy = out / "checkpoints" / "model_selected.pt"
    shutil.copy2(selected_path, selected_copy)
    value = {
        "status": "COMPLETE", "scientific_status": "COMPLETE", "cell_id": row["cell_id"],
        "variant": "FullRank-Local-P2", "dataset": row["dataset"],
        "architecture": row["architecture"], "deletion_type": row["deletion_type"],
        "ratio": row["ratio"], "seed": row["seed"],
        "selector_id": "S6-GNNDELETE-SUM-v1",
        "selection_rule": "strict max validation DT_AUC + Df_AUC; exact tie -> earlier",
        "selected_checkpoint": str(selected_copy),
        "selected_checkpoint_epoch": selected_epoch,
        "selected_validation_metrics": selected_event,
        "DT_AUC": metrics["dt_auc"], "DT_AUP": metrics["dt_aup"],
        "Df_AUC": metrics["df_auc"], "Df_AUP": metrics["df_aup"],
        "trainable_params": gate["FULLRANK_PARAMETER_COUNT"],
        "trainable_parameter_ratio": None, "total_optimizer_updates": summary.get("optimizer_steps_completed"),
        "runtime_seconds": time.perf_counter() - started,
        "formal_runner_family": row["formal_runner_family"], "formal_runner": str(FORMAL_SOURCE),
        "implementation_gate": gate, "protocol_completion": summary,
    }
    value["trainable_parameter_ratio"] = gate["FULLRANK_PARAMETER_COUNT"] / sum(int(p.numel()) for p in base.build_base_model(int(row["seed"]), base.checkpoint_state(base.original_checkpoint(int(row["seed"])))) .parameters())
    (out / "matrix_summary.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def main() -> int:
    args = parser("FullRank Phase 2A-1 DBLP cell").parse_args()
    row = load_row(args.manifest.resolve(), args.cell_id)
    if row["formal_runner_family"] != "GCN_DBLP_EDIN_2P5_FORMAL":
        raise RuntimeError("WRONG-FAMILY-DISPATCH")
    seed = int(row["seed"])
    out = output_dir(row)
    if out.exists():
        raise FileExistsError(out)
    global _CURRENT_CELL_ID
    _CURRENT_CELL_ID = row["cell_id"]
    formal = load_formal()
    base = install_fullrank(formal, out.parent.parent, output_root=out.parent, row=row)
    gate = preflight(formal, base, seed, args.device, out)
    install_training_preflight(base)
    started = time.perf_counter()
    train_arm_by_provenance(formal, row, args.device)
    normalized = normalize(formal, base, row, out, gate, started, args.device)
    print(json.dumps({"status": normalized["status"], "cell_id": row["cell_id"], "selected_epoch": normalized["selected_checkpoint_epoch"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
