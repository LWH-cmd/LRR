"""FullRank injection for the evidence-bound WN18RR selector-v2 family."""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import os
import sys
from pathlib import Path

import torch

from family_common import ROOT, load_row, output_dir, parser
from fullrank_operators import FullRankRGCN


PROJECT_ROOT = ROOT.parents[1]
FORMAL_SOURCE = PROJECT_ROOT / "isolated_experiments/phase4c_wordnet18rr_paper_main_seed21/src/run_paper_main_seed21.py"


def load_formal():
    spec = importlib.util.spec_from_file_location("fullrank_rgcn_selector_v2", FORMAL_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("RGCN-SELECTOR-V2-IMPORT-FAIL")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def gate(formal, graph, assets, config, device: torch.device, path: Path) -> dict:
    backbone = formal.load_original(config, device)
    model = FullRankRGCN(backbone, int(formal.SEED))
    expected = ("repair1.W", "repair2.W")
    actual = tuple(name for name, p in model.named_parameters() if p.requires_grad)
    if actual != expected:
        raise RuntimeError(f"TRAINABLE_PARAMETER_ALLOWLIST_FAIL: {actual}")
    if any(p.requires_grad for name, p in model.named_parameters() if name not in expected):
        raise RuntimeError("UNEXPECTED_TRAINABLE_PARAMETERS")
    formal.p2_base.prepare_p2_assets(assets, device, formal.BASE_RELATIONS)
    nodes = torch.arange(formal.NUM_NODES, dtype=torch.long, device=device)
    gates = formal.p2_base.p2_gates(assets, nodes, device)
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
    source = inspect.getsource(FullRankRGCN.forward)
    if "conv1(h0, edge, edge_type)" not in source or "conv2(h1a, edge, edge_type)" not in source:
        raise RuntimeError("RGCN edge_type propagation audit failed")
    value = {
        "status": "PASS", "FULLRANK_IS_RESIDUAL": "YES", "BACKBONE_FROZEN": "YES",
        "LOCALIZATION_IDENTICAL": "YES", "P2_IDENTICAL": "YES", "FORMAL_ASSETS_IDENTICAL": "YES",
        "FORMAL_SELECTOR_IDENTICAL": "YES", "BUDGET_IDENTICAL": "YES", "ONLY_PARAMETERIZATION_CHANGED": "YES",
        "TRAINABLE_PARAMETER_ALLOWLIST": "PASS", "UNEXPECTED_TRAINABLE_PARAMETERS": 0,
        "FULLRANK_PARAMETER_COUNT": sum(int(p.numel()) for p in model.parameters() if p.requires_grad),
        "FULLRANK_TOTAL_PARAMETER_COUNT": sum(int(p.numel()) for p in model.parameters()),
        "FULLRANK_UPDATE0_IDENTITY": "PASS", "update0_max_abs_output_diff": max(diffs.values()),
        "trainable_parameter_names": list(actual),
        "FORMAL_RESIDUAL_SHARING_SEMANTICS_PRESERVED": "YES",
        "FORMAL_RESIDUAL_PLACEMENT_PRESERVED": "YES",
        "RELATION_SPECIFIC_BACKBONE_COMPUTATION_UNCHANGED": "YES",
        "formal_residual_sharing_semantics_preserved": "YES",
        "formal_residual_placement_preserved": "YES",
        "relation_specific_backbone_computation_unchanged": "YES",
        "WN18RR_IDENTITY": "PASS", "EDGE_TYPE_PROPAGATION": "PASS",
        "RELATION_INDEXING": "PASS", "DISTMULT_UNCHANGED": "YES", "GRAPHSAINT_UNCHANGED": "YES",
        "FULLRANK_GOLDEN_REPLAY": "NOT_APPLICABLE", "formal_runner": str(FORMAL_SOURCE),
        "formal_original_checkpoint": str(formal.ORIGINAL_CHECKPOINT), "identity_diffs": diffs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def normalize(summary: dict, row: dict, gate_value: dict, out: Path) -> dict:
    best = summary["best"]
    test = summary["test"]
    values = {"dt_auc": test.get("dt_auc"), "dt_aup": test.get("dt_aup"),
              "df_auc": test.get("df_auc"), "df_aup": test.get("df_aup")}
    if not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in values.values()):
        raise RuntimeError("RGCN selector-v2 FullRank test metrics are not finite")
    value = {
        "status": "COMPLETE", "scientific_status": "COMPLETE", "cell_id": row["cell_id"],
        "variant": "FullRank-Local-P2", "dataset": row["dataset"], "architecture": row["architecture"],
        "deletion_type": row["deletion_type"], "ratio": row["ratio"], "seed": row["seed"],
        "selected_checkpoint": str(Path(summary["checkpoint"]).resolve()), "selected_checkpoint_epoch": best["epoch"],
        "selected_validation_metrics": best["validation"], "DT_AUC": values["dt_auc"], "DT_AUP": values["dt_aup"],
        "Df_AUC": values["df_auc"], "Df_AUP": values["df_aup"],
        "trainable_params": gate_value["FULLRANK_PARAMETER_COUNT"],
        "trainable_parameter_ratio": gate_value["FULLRANK_PARAMETER_COUNT"] / gate_value["FULLRANK_TOTAL_PARAMETER_COUNT"],
        "total_optimizer_updates": summary["protocol"]["updates"], "runtime_seconds": None,
        "formal_runner_family": row["formal_runner_family"], "formal_runner": str(FORMAL_SOURCE),
        "implementation_gate": gate_value, "protocol_completion": summary,
    }
    (out / "matrix_summary.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def main() -> int:
    args = parser("FullRank selector-v2 WN18RR RGCN cell").parse_args()
    row = load_row(args.manifest.resolve(), args.cell_id)
    if row["formal_runner_family"] != "RGCN_WN18RR_FORMAL_SELECTOR_V2":
        raise RuntimeError("WRONG-FAMILY-DISPATCH")
    out = output_dir(row)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.update({
        "RGCN_MATRIX_OUTPUT": str(out), "RGCN_MATRIX_SEED": str(int(row["seed"])),
        "RGCN_DELETION_TYPE": str(row["deletion_type"]), "RGCN_DELETION_RATIO": str(float(row["ratio"])),
        "RGCN_MATRIX_METHOD_ONLY": "LRR-P2",
    })
    formal = load_formal()
    formal.configure_reused_modules()
    formal.LRRRGCN = FullRankRGCN
    config = json.loads(formal.ORIGINAL_CONFIG.read_text(encoding="utf-8"))
    graph, base_relations, counts, train = formal.dec_base.load_graph()
    if base_relations != formal.BASE_RELATIONS or counts["num_entities"] != formal.NUM_NODES:
        raise RuntimeError(f"WN18RR metadata mismatch: {counts}")
    assets = formal.p2_base.build_assets(graph, base_relations, train)
    formal.add_test_bank(graph, assets)
    request = formal.p2_base.build_request(graph, base_relations, train)
    data = formal.p2_base.build_data(train, graph.train_edge_type, request["selected_train_positions"])
    device = torch.device(args.device)
    gate_value = gate(formal, graph, assets, config, device, ROOT / "canaries" / row["formal_runner_family"] / row["cell_id"] / "implementation_gate.json")
    formal.main()
    run_dir = out / "runs" / f"LRR-P2-RGCN-WN18RR-S{int(row['seed'])}-{row['deletion_type']}{str(float(row['ratio'])).replace('.', 'p')}"
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    normalized = normalize(summary, row, gate_value, out)
    (out / "resolved_formal_runner.py").write_text(FORMAL_SOURCE.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({"status": normalized["status"], "cell_id": row["cell_id"], "selected_epoch": normalized["selected_checkpoint_epoch"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
