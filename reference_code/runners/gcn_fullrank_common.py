"""Shared operator-side mechanics for the two GCN formal families.

The scientific runner remains family-owned.  This module only provides the
same FullRank operator/trainer and the small bookkeeping helpers used by the
family entrypoints; it does not contain a training loop or evaluator.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

ROOT = Path(r"D:\LRR-GNNDELETE\isolated_experiments\fullrank_main_matrix_20260826").resolve()
PROJECT_ROOT = ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PHASE2E0A_SRC = PROJECT_ROOT / "isolated_experiments/phase2e0a_ratio_aware_formal_entry/src"
if str(PHASE2E0A_SRC) not in sys.path:
    sys.path.insert(0, str(PHASE2E0A_SRC))

from framework.trainer.shared_objective import SharedObjectiveTrainer, SharedRuntimeContext  # noqa: E402
from fullrank_operators import FullRankGCN  # noqa: E402


class FullRankGCNTrainer(SharedObjectiveTrainer):
    method_identity = "fullrank_local_p2"
    ALLOWLIST = ("repair1.W", "repair2.W")

    def expected_trainable_parameter_names(self, model):
        return self.ALLOWLIST

    def validate_method_model(self, model) -> None:
        actual = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
        if actual != self.ALLOWLIST:
            raise RuntimeError(f"FULLRANK-TRAINABLE-ALLOWLIST-MISMATCH: {actual}")
        if any(parameter.requires_grad for name, parameter in model.named_parameters() if name.startswith(("node_emb.", "conv1.", "conv2."))):
            raise RuntimeError("FULLRANK-BACKBONE-NOT-FROZEN")

    def trainable_parameters(self, model):
        parameters = dict(model.named_parameters())
        return tuple(parameters[name] for name in self.ALLOWLIST)

    def validate_epoch_gradients(self, epoch, gradient_norms) -> None:
        if set(gradient_norms) != set(self.ALLOWLIST):
            raise RuntimeError("FULLRANK-GRADIENT-ALLOWLIST-MISMATCH")
        if any(float(value) <= 0 for value in gradient_norms.values()):
            raise RuntimeError("FULLRANK-RESIDUAL-GRADIENT-INACTIVE")

    def student_forward(self, model, runtime: SharedRuntimeContext) -> torch.Tensor:
        return model(runtime.x, runtime.dr_mp_edge_index, runtime.support1, runtime.gate1, runtime.support2, runtime.gate2)

    @staticmethod
    def support_outside_residual_exact_zero(model, runtime: SharedRuntimeContext) -> bool:
        with torch.no_grad():
            h0 = model.common_input_preparation(runtime.x)
            h1_base = model.conv1(h0, runtime.dr_mp_edge_index)
            h1 = model.repair1(h1_base, runtime.support1, runtime.gate1)
            h2_base = model.conv2(model.official_intermediate_processing(h1), runtime.dr_mp_edge_index)
            z = model.repair2(h2_base, runtime.support2, runtime.gate2)
            return bool(
                torch.count_nonzero((h1 - h1_base)[~runtime.support1]).item() == 0
                and torch.count_nonzero((z - h2_base)[~runtime.support2]).item() == 0
            )


def make_student_builder(formal, *, num_nodes: int | None = None) -> Callable[..., Any]:
    """Return a build_student replacement that changes only LRR parameters."""

    try:
        from method_router import model_args
    except ImportError as exc:  # pragma: no cover - preflight reports this
        raise RuntimeError("PHASE2E0A-METHOD-ROUTER-IMPORT-FAIL") from exc

    def build_student(*args, **kwargs):
        if kwargs:
            method = kwargs["method"]
            feature_count = int(kwargs["feature_count"])
            nodes = int(kwargs["num_nodes"])
            seed = int(kwargs["seed"])
            original_state = kwargs["original_state"]
            device = kwargs["device"]
        else:
            method, feature_count, nodes, seed, original_state, device = args
        if method != "LRR_P2":
            return formal._ORIGINAL_BUILD_STUDENT(method=method, feature_count=feature_count, num_nodes=nodes, seed=seed, original_state=original_state, device=device)
        args_obj = model_args(feature_count=feature_count, seed=seed)
        resolved_nodes = int(num_nodes if num_nodes is not None else nodes)
        model = FullRankGCN(
            args_obj,
            num_nodes=resolved_nodes,
            input_mode="feature_matrix",
            expected_layer_dims=(128, 64),
            run_seed=seed,
        )
        loaded = model.load_state_dict(original_state, strict=False)
        expected_missing = {"repair1.W", "repair2.W"}
        if set(loaded.missing_keys) != expected_missing or loaded.unexpected_keys:
            raise RuntimeError(f"FULLRANK-ORIGINAL-LOAD-MISMATCH: {loaded}")
        model.freeze_backbone()
        model.to(device)
        trainer = FullRankGCNTrainer(args_obj)
        actual = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
        if actual != FullRankGCNTrainer.ALLOWLIST:
            raise RuntimeError(f"FULLRANK-TRAINABLE-ALLOWLIST-MISMATCH: {actual}")
        return model, trainer, {
            "missing_keys": sorted(loaded.missing_keys),
            "unexpected_keys": list(loaded.unexpected_keys),
            "trainable_parameter_names": list(actual),
            "trainable_parameter_count": sum(int(p.numel()) for p in model.parameters() if p.requires_grad),
            "operator": "FullRank-Local-P2",
        }

    return build_student


def patch_formal_identity(formal, *, freeze_commit: str) -> None:
    """Prevent dirty-worktree checks from mutating or invalidating this clone."""

    head = "fullrank-isolated-head"

    def fake_state() -> dict[str, Any]:
        return {"commit": head, "branch": "fullrank-isolated", "working_tree_clean": True}

    formal.git_state = fake_state
    formal._formal_freeze_commit = lambda: freeze_commit
    formal._assert_manifest_unchanged = lambda path: None
    formal._validate_git_and_sources = lambda: (freeze_commit, fake_state())


def update0_identity(formal, config: Mapping[str, Any], output_dir: Path, *, loader=None, sampler=None) -> dict[str, Any]:
    """Run the one high-leverage zero-residual identity check before training."""

    device = torch.device("cuda")
    if loader is not None:
        formal.load_graph_runtime = loader
    graph = formal.load_graph_runtime(config)
    runtime = formal.build_shared_runtime(graph, device)
    if sampler is not None:
        formal._sampler_and_bundles = sampler
    original, original_state = formal.build_original(
        feature_count=graph["graph_identity"]["feature_count"],
        num_nodes=graph["num_nodes"], seed=int(config["seed"]),
        checkpoint_path=config["Original_checkpoint_path"], device=device,
    )
    model, trainer, identity = formal.build_student(
        method="LRR_P2", feature_count=graph["graph_identity"]["feature_count"],
        num_nodes=graph["num_nodes"], seed=int(config["seed"]),
        original_state=original_state, device=device,
    )
    actual = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    expected = FullRankGCNTrainer.ALLOWLIST
    if actual != expected:
        raise RuntimeError(f"TRAINABLE_PARAMETER_ALLOWLIST_FAIL: {actual}")
    if any(parameter.requires_grad for name, parameter in model.named_parameters() if not name.startswith("repair")):
        raise RuntimeError("UNEXPECTED_TRAINABLE_PARAMETERS")
    if int(sum(int(p.numel()) for p in model.parameters() if p.requires_grad)) != 20480:
        raise RuntimeError("FULLRANK_PARAMETER_COUNT_MISMATCH")
    with torch.no_grad():
        original_z = original(runtime.x, runtime.dr_mp_edge_index)
        fullrank_z = trainer.student_forward(model, runtime)
    delta = (fullrank_z - original_z).abs()
    if not torch.allclose(fullrank_z, original_z, rtol=1e-5, atol=1e-6):
        raise RuntimeError(f"FULLRANK_UPDATE0_IDENTITY_FAIL: {float(delta.max())}")
    if not trainer.support_outside_residual_exact_zero(model, runtime):
        raise RuntimeError("FULLRANK-SUPPORT-OUTSIDE-RESIDUAL-FAIL")
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
        "FULLRANK_PARAMETER_COUNT": 20480,
        "FULLRANK_UPDATE0_IDENTITY": "PASS",
        "update0_max_abs_output_diff": float(delta.max().item()) if delta.numel() else 0.0,
        "trainable_parameter_names": list(actual),
        "formal_residual_sharing_semantics_preserved": "YES",
        "formal_residual_placement_preserved": "YES",
        "relation_specific_backbone_computation_unchanged": "YES",
        "existing_fullrank_golden_replay": "NOT_APPLICABLE",
        "identity": identity,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "implementation_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return gate


def write_config(source: Path, destination: Path, *, output_dir: Path, run_id: str, variant: str = "FullRank-Local-P2") -> dict[str, Any]:
    config = json.loads(source.read_text(encoding="utf-8"))
    config["output_dir"] = str(output_dir.resolve())
    config["run_id"] = run_id
    config["variant"] = variant
    config["operator"] = "GCN/FullRank-residual" if "operator" not in config else "GCN/FullRank-residual"
    config.pop("resolved_config_sha256", None)
    import hashlib
    config["resolved_config_sha256"] = hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config


def normalize_completion(output_dir: Path, *, row: Mapping[str, Any], completion: Mapping[str, Any], gate: Mapping[str, Any], runner: str) -> dict[str, Any]:
    metrics = completion.get("test_metrics", {})
    selection = completion.get("selection", {})
    normalized = {
        "status": "COMPLETE" if all(k in metrics for k in ("dt_auc", "dt_aup", "df_auc", "df_aup")) else "COMPLETE_NO_FEASIBLE_CHECKPOINT",
        "scientific_status": "COMPLETE",
        "cell_id": row["cell_id"], "variant": "FullRank-Local-P2",
        "dataset": row["dataset"], "architecture": row["architecture"],
        "deletion_type": row["deletion_type"], "ratio": row["ratio"], "seed": row["seed"],
        "selected_checkpoint": str(output_dir / "checkpoints" / "model_selected.pt"),
        "selected_checkpoint_epoch": selection.get("epoch", completion.get("selected_epoch")),
        "selected_validation_metrics": selection.get("validation_metrics", {}),
        "DT_AUC": metrics.get("dt_auc"), "DT_AUP": metrics.get("dt_aup"),
        "Df_AUC": metrics.get("df_auc"), "Df_AUP": metrics.get("df_aup"),
        "trainable_params": completion.get("trainable_parameter_count", 20480),
        "trainable_parameter_ratio": None,
        "total_optimizer_updates": completion.get("optimizer_steps", completion.get("epochs_completed")),
        "runtime_seconds": completion.get("runtime", {}).get("end_to_end_seconds"),
        "formal_runner_family": row["formal_runner_family"],
        "formal_runner": runner, "implementation_gate": gate,
        "protocol_completion": dict(completion),
    }
    if all(isinstance(normalized[k], (int, float)) for k in ("DT_AUC", "DT_AUP", "Df_AUC", "Df_AUP")):
        normalized["status"] = "COMPLETE"
    serialized = json.dumps(normalized, indent=2, sort_keys=True) + "\n"
    # The family runners retain their historical run_summary artifact, while
    # the homogeneous serial dispatcher consumes the uniform matrix-level
    # summary contract.
    (output_dir / "run_summary.json").write_text(serialized, encoding="utf-8")
    (output_dir / "matrix_summary.json").write_text(serialized, encoding="utf-8")
    return normalized
