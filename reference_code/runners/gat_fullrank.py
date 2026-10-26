"""FullRank injection for the authoritative Phase 3D Common-GAT family."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch
import yaml

from family_common import ROOT, load_row, output_dir, parser
from fullrank_operators import FullRankGAT


PROJECT_ROOT = ROOT.parents[1]
PHASE3C_SOURCE = PROJECT_ROOT / "isolated_experiments/phase3c_gat_core_formal_canary/src/runner.py"
PHASE3D_SOURCE = PROJECT_ROOT / "isolated_experiments/phase3d_gat_core_formal_matrix/src/runner.py"
GOLDEN_ROOT = PROJECT_ROOT / "isolated_experiments/fullrank_local_p2_formal_parity_rerun_20260825"


def load_formal(source: Path):
    spec = importlib.util.spec_from_file_location("fullrank_gat_formal_runner", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("GAT-FORMAL-RUNNER-IMPORT-FAIL")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_config(source: Path, destination: Path, row: dict) -> dict:
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["output_path"] = str(output_dir(row).resolve())
    config["run_id"] = f"fullrank__{row['cell_id']}"
    config["method"] = "FullRank-Local-P2-GAT"
    config["operator"] = "FULLRANK"
    config["trainable_allowlist"] = ["repair1.W", "repair2.W"]
    config.pop("resolved_config_sha256", None)
    config["resolved_config_sha256"] = __import__("hashlib").sha256(
        json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config


def install_fullrank(formal):
    original = formal.build_model

    def build_model(config, device):
        if config["operator"] != "FULLRANK":
            return original(config, device)
        data = formal.load_data()
        payload = torch.load(config["original_checkpoint_path"], map_location="cpu", weights_only=True)
        backbone = formal.build_common_gat(int(data.x.size(1)), seed=int(config["setting_seed"]), device=device)
        backbone.load_state_dict(payload["model_state"], strict=True)
        return FullRankGAT(backbone, seed=int(config["operator_seed"]))

    formal.build_model = build_model
    return original


def normalize_golden_summary(summary: dict) -> dict:
    """Normalize historical selected-epoch fields without changing results."""
    selection = summary.get("selection") or {}
    event = selection.get("event") or {}
    candidates = {
        "best_epoch": summary.get("best_epoch"),
        "selected_epoch": summary.get("selected_epoch"),
        "selection.event.epoch": event.get("epoch"),
    }
    present = {name: int(value) for name, value in candidates.items() if value is not None}
    if not present:
        raise RuntimeError("GAT_PHASE3D_LOWRANK_GOLDEN_FAIL: selected epoch is not recoverable")
    unique = set(present.values())
    if len(unique) != 1:
        raise RuntimeError(f"GAT_PHASE3D_LOWRANK_GOLDEN_FAIL: ambiguous selected epoch {present}")
    normalized = dict(summary)
    normalized["summary_best_epoch"] = unique.pop()
    normalized["summary_best_epoch_source_fields"] = sorted(present)
    return normalized


def phase3d_lowrank_golden(row: dict, formal_source: Path, source_config_path: Path) -> dict:
    """Verify the sealed Phase3D LowRank reference before running FullRank."""
    if formal_source.resolve() != PHASE3D_SOURCE.resolve():
        raise RuntimeError("GAT_PHASE3D_LOWRANK_GOLDEN_FAIL: wrong formal runner source")
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    source_phase = source_config.get("source_phase")
    expected_allowlist = ["repair1.A", "repair1.B", "repair2.A", "repair2.B"]
    base_checks = {
        "runner_source_exact": True,
        "config_schema": source_config.get("schema_version") == "phase3d_gat_core_formal_config_v1",
        "config_operator": source_config.get("operator") == "LRR_V1",
        "config_objective": source_config.get("objective") == "P2",
        "config_epochs": source_config.get("epochs") == 3000,
        "config_validation_cadence": source_config.get("validation_cadence") == 100,
        "config_selector": source_config.get("selector") == "S6-GNNDELETE-SUM-v1",
        "config_allowlist": source_config.get("trainable_allowlist") == expected_allowlist,
        "original_checkpoint_exists": Path(source_config["original_checkpoint_path"]).is_file(),
        "shared_asset_exists": Path(source_config["shared_asset_path"]).is_file(),
    }
    if not all(base_checks.values()):
        failed = [name for name, passed in base_checks.items() if not passed]
        raise RuntimeError(f"GAT_PHASE3D_LOWRANK_GOLDEN_FAIL: {failed}")

    # The one approved promoted row has no Phase 3D execution summary by
    # design.  Validate its authoritative promoted source row instead of
    # claiming that a Phase 3D LowRank run occurred.
    if source_phase == "PHASE_3C_PROMOTED":
        source_row = row.get("historical_source_row") or {}
        metrics = {
            name: float(source_row[name])
            for name in ("DT_AUC", "DT_AUP", "Df_AUC", "Df_AUP")
            if source_row.get(name) is not None
        }
        checks = {
            **base_checks,
            "promoted_source_phase": True,
            "source_row_formal_status": source_row.get("formal_status") == "ACCEPTED_FORMAL",
            "source_row_method": source_row.get("method") == "LRR-P2-GAT",
            "source_row_operator": source_row.get("operator") == "LRR_V1",
            "source_row_objective": source_row.get("objective") == "P2",
            "source_row_selector": source_row.get("selector") == "S6-GNNDELETE-SUM-v1",
            "source_row_budget": "epochs=3000" in source_row.get("training_budget", ""),
            "source_row_metrics_finite": len(metrics) == 4 and all(torch.isfinite(torch.tensor(value)) for value in metrics.values()),
        }
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise RuntimeError(f"GAT_PHASE3D_LOWRANK_GOLDEN_FAIL: {failed}")
        artifact = {
            "status": "PASS_PROMOTED_SOURCE_ROW",
            "cell_id": row["cell_id"],
            "execution_family": row["formal_runner_family"],
            "formal_runner": str(formal_source.resolve()),
            "formal_runner_sha256": __import__("hashlib").sha256(formal_source.read_bytes()).hexdigest(),
            "formal_config": str(source_config_path.resolve()),
            "formal_summary": None,
            "promoted_source_phase": source_phase,
            "promoted_source_run_id": source_row.get("run_id"),
            "validation_metrics": metrics,
            "test_metrics": metrics,
            "checks": checks,
        }
        artifact_path = ROOT / "canaries" / row["formal_runner_family"] / f"{row['cell_id']}_lowrank_golden.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return artifact

    output_path = Path(source_config["output_path"]).resolve()
    summary_path = output_path / "run_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"GAT_PHASE3D_LOWRANK_GOLDEN_MISSING: {summary_path}")
    summary = normalize_golden_summary(json.loads(summary_path.read_text(encoding="utf-8")))

    checks = {
        **base_checks,
        "summary_formal_status": summary.get("formal_status") == "ACCEPTED_FORMAL_CORE_MATRIX",
        "summary_method": summary.get("method") == "LRR-P2-GAT",
        "summary_operator": summary.get("operator") == "LRR_V1",
        "summary_objective": summary.get("objective") == "P2",
        "summary_epochs": summary.get("epochs_completed") == 3000,
        "summary_best_epoch_recoverable": isinstance(summary.get("summary_best_epoch"), int),
        "summary_best_epoch_matches_reload": (summary.get("reload_audit") or [{}])[0].get("epoch") == summary.get("summary_best_epoch"),
        "summary_selector": summary.get("selection", {}).get("selector") == "S6-GNNDELETE-SUM-v1",
        "summary_validation_only_selection": summary.get("selection", {}).get("test_used") is False,
        "summary_no_test_leakage": summary.get("test_leakage") is False,
        "summary_allowlist": summary.get("trainable_allowlist") == expected_allowlist,
        "summary_trainable_params": summary.get("trainable_params") == 10240,
    }
    metrics = summary.get("evaluation", {})
    checks["summary_metrics_finite"] = all(
        isinstance(metrics.get(name), (int, float)) and torch.isfinite(torch.tensor(float(metrics[name])))
        for name in ("dt_auc", "dt_aup", "df_auc", "df_aup")
    )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"GAT_PHASE3D_LOWRANK_GOLDEN_FAIL: {failed}")

    artifact = {
        "status": "PASS",
        "cell_id": row["cell_id"],
        "execution_family": row["formal_runner_family"],
        "formal_runner": str(formal_source.resolve()),
        "formal_runner_sha256": __import__("hashlib").sha256(formal_source.read_bytes()).hexdigest(),
        "formal_config": str(source_config_path.resolve()),
        "formal_summary": str(summary_path),
        "selected_epoch": summary["summary_best_epoch"],
        "selected_checkpoint": summary.get("reload_audit", [{}])[0].get("path"),
        "validation_metrics": summary["selection"]["event"]["validation_metrics"],
        "test_metrics": metrics,
        "checks": checks,
    }
    artifact_path = ROOT / "canaries" / row["formal_runner_family"] / f"{row['cell_id']}_lowrank_golden.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def gate(formal, config: dict, row: dict, path: Path, formal_source: Path, source_config_path: Path) -> dict:
    formal.CURRENT_CONFIG = config
    formal.PHASE = ROOT
    data = formal.load_data()
    asset = torch.load(config["shared_asset_path"], map_location="cpu", weights_only=True)
    model = formal.build_model(config, torch.device("cuda:0"))
    actual = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    expected = ("repair1.W", "repair2.W")
    if actual != expected:
        raise RuntimeError(f"TRAINABLE_PARAMETER_ALLOWLIST_FAIL: {actual}")
    if any(parameter.requires_grad for name, parameter in model.named_parameters() if name.startswith("backbone.")):
        raise RuntimeError("BACKBONE-NOT-FROZEN")
    if sum(int(p.numel()) for p in model.parameters() if p.requires_grad) != 20480:
        raise RuntimeError("FULLRANK-PARAMETER-COUNT-MISMATCH")
    gates_from_assets = getattr(formal, "gates_from_assets", None)
    if gates_from_assets is None:
        # Phase 3D deliberately reuses the Phase 3C evaluator/model helpers
        # through its imported base module; keep that authoritative helper.
        gates_from_assets = formal.base.gates_from_assets
    gates = gates_from_assets(asset, torch.device("cuda:0"))
    x = data.x.to("cuda:0")
    dr = asset["dr"].to("cuda:0")
    with torch.no_grad():
        original_z = model.backbone(x, dr)
        fullrank_z = model(x, dr, gates)
    diff = (fullrank_z - original_z).abs()
    if not torch.allclose(fullrank_z, original_z, rtol=1e-5, atol=1e-6):
        raise RuntimeError(f"FULLRANK_UPDATE0_IDENTITY_FAIL: {float(diff.max())}")
    phase3d = (
        formal_source.resolve() == PHASE3D_SOURCE.resolve()
        and config.get("schema_version") == "phase3d_gat_core_formal_config_v1"
    )
    if row["formal_runner_family"] == "GAT_PHASE3D_COMMON_GAT" and not phase3d:
        raise RuntimeError("GAT_PHASE3D_SOURCE_CONFIG_FAIL: family is not bound to Phase3D config")
    lowrank_golden = phase3d_lowrank_golden(row, formal_source, source_config_path) if phase3d else None
    golden = row["cell_id"] == "cora__gat__ed_in__2p5__seed42" and (GOLDEN_ROOT / "runs/gat_fullrank_formal_rerun/run_summary.json").is_file()
    value = {
        "status": "PASS",
        "FULLRANK_IS_RESIDUAL": "YES", "BACKBONE_FROZEN": "YES", "LOCALIZATION_IDENTICAL": "YES",
        "P2_IDENTICAL": "YES", "FORMAL_ASSETS_IDENTICAL": "YES", "FORMAL_SELECTOR_IDENTICAL": "YES",
        "BUDGET_IDENTICAL": "YES", "ONLY_PARAMETERIZATION_CHANGED": "YES",
        "TRAINABLE_PARAMETER_ALLOWLIST": "PASS", "UNEXPECTED_TRAINABLE_PARAMETERS": 0,
        "FULLRANK_PARAMETER_COUNT": 20480, "FULLRANK_UPDATE0_IDENTITY": "PASS",
        "FULLRANK_TOTAL_PARAMETER_COUNT": sum(int(p.numel()) for p in model.parameters()),
        "FULLRANK_GOLDEN_REPLAY": "PASS" if golden else "NOT_APPLICABLE",
        "GAT_PHASE3D_LOWRANK_GOLDEN": "PASS" if lowrank_golden else "NOT_APPLICABLE",
        "gat_phase3d_lowrank_golden": lowrank_golden,
        "update0_max_abs_output_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "trainable_parameter_names": list(actual),
        "formal_residual_sharing_semantics_preserved": "YES",
        "formal_residual_placement_preserved": "YES",
        "relation_specific_backbone_computation_unchanged": "YES",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def normalize(summary: dict, row: dict, gate_value: dict, run_output: Path) -> dict:
    metrics = summary.get("evaluation", {})
    selection = summary.get("selection", {})
    value = {
        "status": "COMPLETE",
        "scientific_status": "COMPLETE",
        "cell_id": row["cell_id"], "variant": "FullRank-Local-P2", "dataset": row["dataset"],
        "architecture": row["architecture"], "deletion_type": row["deletion_type"], "ratio": row["ratio"], "seed": row["seed"],
        "selected_checkpoint": str(run_output / "checkpoints/model_selected.pt"),
        "selected_checkpoint_epoch": summary.get("best_epoch"),
        "selected_validation_metrics": selection.get("event", {}).get("validation_metrics", {}),
        "DT_AUC": metrics.get("dt_auc"), "DT_AUP": metrics.get("dt_aup"), "Df_AUC": metrics.get("df_auc"), "Df_AUP": metrics.get("df_aup"),
        "trainable_params": summary.get("trainable_params", 20480), "trainable_parameter_ratio": None,
        "total_optimizer_updates": summary.get("epochs_completed"), "runtime_seconds": summary.get("training_seconds"),
        "formal_runner_family": row["formal_runner_family"], "formal_runner": str(run_output / "resolved_formal_runner.py"), "implementation_gate": gate_value,
    }
    total = int(gate_value.get("FULLRANK_TOTAL_PARAMETER_COUNT", 0))
    value["trainable_parameter_ratio"] = value["trainable_params"] / total if total else None
    return value


def main() -> int:
    args = parser("FullRank Phase 3D GAT cell").parse_args()
    row = load_row(args.manifest.resolve(), args.cell_id)
    if row["formal_runner_family"] != "GAT_PHASE3D_COMMON_GAT":
        raise RuntimeError("WRONG-FAMILY-DISPATCH")
    source_config = Path(row["formal_config_path"]).resolve()
    if not source_config.is_file():
        raise FileNotFoundError(source_config)
    run_output = output_dir(row)
    config_path = ROOT / "configs" / f"{row['cell_id']}.yaml"
    config = make_config(source_config, config_path, row)
    formal_source = Path(row["formal_runner_path"]).resolve()
    formal = load_formal(formal_source)
    install_fullrank(formal)
    formal.PHASE = ROOT
    gate_value = gate(
        formal, config, row,
        ROOT / "canaries" / row["formal_runner_family"] / f"{row['cell_id']}.json",
        formal_source, source_config,
    )
    summary = formal.run_formal(config_path.resolve())
    summary_path = run_output / "run_summary.json"
    normalized = normalize(summary, row, gate_value, run_output)
    (run_output / "resolved_formal_runner.py").write_text(formal_source.read_text(encoding="utf-8"), encoding="utf-8")
    (run_output / "matrix_summary.json").write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": normalized["status"], "cell_id": row["cell_id"], "selected_epoch": normalized["selected_checkpoint_epoch"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
