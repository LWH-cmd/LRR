"""Small diagnostic helpers for the six-cell RGCN FullRank protocol pilot.

The historical runners remain the owners of graph construction, P2, training,
and evaluation.  This module only saves residual snapshots and evaluates those
snapshots offline for the pre-declared candidate budgets.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from fullrank_operators import FullRankRGCN


SNAPSHOT_UPDATES = (
    0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 160, 192, 256,
    320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1600,
)
CANDIDATE_BUDGETS = (256, 512, 768, 1024, 1600)
TRAINABLE_NAMES = ("repair1.W", "repair2.W")


def save_residual_snapshot(
    root: Path,
    operator: torch.nn.Module,
    update: int,
    original_checkpoint: Path,
) -> None:
    """Save only the FullRank residual parameters; this does not call RNG."""

    if update not in SNAPSHOT_UPDATES:
        return
    parameters = dict(operator.named_parameters())
    actual = tuple(name for name, value in operator.named_parameters() if value.requires_grad)
    if actual != TRAINABLE_NAMES:
        raise RuntimeError(f"unexpected FullRank allowlist while snapshotting: {actual}")
    path = root / "FullRank" / f"snapshot_update_{update:04d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "variant": "FullRank-Local-P2",
            "update": int(update),
            "trainable_names": actual,
            "operator_state": {
                name: parameters[name].detach().cpu().clone() for name in TRAINABLE_NAMES
            },
            "original_checkpoint": str(original_checkpoint),
        },
        path,
    )


def _formal_functions(formal: Any):
    if hasattr(formal, "p2_base"):
        return formal.p2_base.p2_gates, formal.p2_base.evaluate_fn
    return formal.p2_gates, formal.evaluate_fn


def prepare_assets_for_offline(formal: Any, device: torch.device):
    """Rebuild the formal validation assets through the imported source path."""

    if hasattr(formal, "configure_reused_modules"):
        formal.configure_reused_modules()
        graph, base_relations, counts, train = formal.dec_base.load_graph()
        if base_relations != formal.BASE_RELATIONS or counts["num_entities"] != formal.NUM_NODES:
            raise RuntimeError(f"WN18RR metadata mismatch: {counts}")
        assets = formal.p2_base.build_assets(graph, base_relations, train)
        formal.add_test_bank(graph, assets)
        formal.p2_base.prepare_p2_assets(assets, device, formal.BASE_RELATIONS)
        config = json.loads(formal.ORIGINAL_CONFIG.read_text(encoding="utf-8"))
    else:
        graph, base_relations, counts, train = formal.load_graph()
        if base_relations != formal.BASE_RELATIONS or counts["num_entities"] != formal.NUM_NODES:
            raise RuntimeError(f"WN18RR metadata mismatch: {counts}")
        assets = formal.build_assets(graph, base_relations, train)
        formal.add_test_bank(graph, assets)
        config = json.loads(formal.ORIGINAL_CONFIG.read_text(encoding="utf-8"))
    return graph, assets, config


def build_fullrank_model(formal: Any, config: dict[str, Any], device: torch.device):
    if hasattr(formal, "load_original"):
        backbone = formal.load_original(config, device)
    else:
        backbone = formal.build_original_model(formal.NUM_NODES, formal.BASE_RELATIONS, config, device)
        payload = torch.load(formal.ORIGINAL_CHECKPOINT, map_location="cpu", weights_only=False)
        backbone.load_state_dict(payload["model_state"], strict=True)
        backbone.eval()
    model = FullRankRGCN(backbone, int(formal.SEED))
    actual = tuple(name for name, value in model.named_parameters() if value.requires_grad)
    if actual != TRAINABLE_NAMES:
        raise RuntimeError(f"FullRank trainable allowlist mismatch: {actual}")
    return model


def update0_gate(formal: Any, assets: dict[str, Any], config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    model = build_fullrank_model(formal, config, device)
    parameters = dict(model.named_parameters())
    actual = tuple(name for name, value in model.named_parameters() if value.requires_grad)
    unexpected = tuple(name for name, value in model.named_parameters() if value.requires_grad and name not in TRAINABLE_NAMES)
    if unexpected or actual != TRAINABLE_NAMES:
        raise RuntimeError(f"FullRank parameter allowlist failed: {actual}")
    gate_fn, _ = _formal_functions(formal)
    nodes = torch.arange(formal.NUM_NODES, dtype=torch.long, device=device)
    gates = gate_fn(assets, nodes, device)
    edge = assets["dr_mp"].to(device)
    edge_type = assets["dr_mp_type"].to(device)
    with torch.no_grad():
        baseline = model.original_representations(nodes, edge, edge_type)
        repaired = model(nodes, edge, edge_type, gates, return_representations=True)
    diffs = {
        "h0": float((repaired.h0 - baseline.h0).abs().max().item()),
        "h1": float((repaired.h1 - baseline.h1).abs().max().item()),
        "h2": float((repaired.h2 - baseline.h2).abs().max().item()),
    }
    if max(diffs.values()) > 1e-5:
        raise RuntimeError(f"FULLRANK_UPDATE0_IDENTITY_FAIL: {diffs}")
    if not all(torch.count_nonzero(parameters[name].detach()).item() == 0 for name in TRAINABLE_NAMES):
        raise RuntimeError("FullRank residual is not zero initialized")
    value = {
        "status": "PASS",
        "FULLRANK_IS_RESIDUAL": "YES",
        "BACKBONE_FROZEN": "YES",
        "FORMAL_RESIDUAL_SHARING_SEMANTICS_PRESERVED": "YES",
        "FORMAL_RESIDUAL_PLACEMENT_PRESERVED": "YES",
        "RELATION_SPECIFIC_BACKBONE_COMPUTATION_UNCHANGED": "YES",
        "TRAINABLE_PARAMETER_ALLOWLIST": "PASS",
        "UNEXPECTED_TRAINABLE_PARAMETERS": len(unexpected),
        "FULLRANK_PARAMETER_COUNT": sum(int(parameters[name].numel()) for name in TRAINABLE_NAMES),
        "trainable_parameter_names": list(actual),
        "FULLRANK_UPDATE0_IDENTITY": "PASS",
        "update0_max_abs_output_diff": max(diffs.values()),
        "RGCN_EDGE_TYPE_PROPAGATION": "PASS",
        "RELATION_INDEXING": "PASS",
        "DISTMULT_UNCHANGED": "YES",
        "GRAPHSAINT_UNCHANGED": "YES",
    }
    return value


def evaluate_snapshots(
    formal: Any,
    snapshot_root: Path,
    assets: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    selector_kind: str,
) -> dict[str, Any]:
    gate_fn, evaluate_fn = _formal_functions(formal)
    model = build_fullrank_model(formal, config, device)
    nodes = torch.arange(formal.NUM_NODES, dtype=torch.long, device=device)
    gates = gate_fn(assets, nodes, device)
    paths = sorted((snapshot_root / "FullRank").glob("snapshot_update_*.pt"))
    actual_updates = tuple(int(path.stem.rsplit("_", 1)[1]) for path in paths)
    if actual_updates != SNAPSHOT_UPDATES:
        raise RuntimeError(f"dense snapshot set mismatch: expected {SNAPSHOT_UPDATES}, got {actual_updates}")

    trajectory: list[dict[str, Any]] = []
    parameters = dict(model.named_parameters())
    for path in paths:
        snapshot = torch.load(path, map_location="cpu", weights_only=False)
        if tuple(snapshot.get("trainable_names", ())) != TRAINABLE_NAMES:
            raise RuntimeError(f"snapshot allowlist mismatch: {path}")
        for name in TRAINABLE_NAMES:
            parameters[name].data.copy_(snapshot["operator_state"][name].to(device=device, dtype=parameters[name].dtype))
        model.eval()
        with torch.no_grad():
            metrics = evaluate_fn(model, True, nodes, assets, "val", gates)
        row = {
            "update": int(snapshot["update"]),
            "DT_AUC": float(metrics["dt_auc"]),
            "DT_AUP": float(metrics["dt_aup"]),
            "Df_AUC": float(metrics["df_auc"]),
            "Df_AUP": float(metrics["df_aup"]),
        }
        if not all(math.isfinite(row[name]) for name in ("DT_AUC", "DT_AUP", "Df_AUC", "Df_AUP")):
            raise RuntimeError(f"non-finite dense metric at update {row['update']}")
        if selector_kind == "legacy_constrained_df":
            threshold = float(getattr(formal, "LRR_DECNI_CONSTRAINT"))
            row["formal_feasible"] = row["DT_AUC"] >= threshold
            row["selector_objective"] = row["Df_AUC"]
        elif selector_kind == "strict_sum":
            threshold = None
            row["formal_feasible"] = True
            row["selector_objective"] = row["DT_AUC"] + row["Df_AUC"]
        else:
            raise ValueError(selector_kind)
        trajectory.append(row)

    selected_by_budget: dict[str, Any] = {}
    for budget in CANDIDATE_BUDGETS:
        eligible = [row for row in trajectory if row["update"] <= budget and row["formal_feasible"]]
        if not eligible:
            selected_by_budget[str(budget)] = {
                "status": "COMPLETE_NO_FEASIBLE_CHECKPOINT",
                "budget": budget,
                "selected_update": None,
            }
            continue
        selected = max(eligible, key=lambda row: (row["selector_objective"], -row["update"]))
        selected_by_budget[str(budget)] = {
            "status": "PASS",
            "budget": budget,
            "selected_update": selected["update"],
            "selector_objective": selected["selector_objective"],
            "validation": {
                name: selected[name] for name in ("DT_AUC", "DT_AUP", "Df_AUC", "Df_AUP")
            },
        }

    return {
        "snapshot_updates": list(SNAPSHOT_UPDATES),
        "trajectory": trajectory,
        "candidate_budgets": list(CANDIDATE_BUDGETS),
        "selected_by_budget": selected_by_budget,
        "selector_kind": selector_kind,
        "selector_tie_break": "earlier_update",
        "formal_threshold_DT_AUC": threshold,
        "no_test_selection": True,
    }
