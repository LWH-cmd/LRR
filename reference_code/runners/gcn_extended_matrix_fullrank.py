"""FullRank injection for the authoritative Phase 2F Ed_OUT GCN family."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from family_common import ROOT, load_row, output_dir, parser
from gcn_fullrank_common import (
    make_student_builder,
    normalize_completion,
    patch_formal_identity,
    update0_identity,
    write_config,
)


PROJECT_ROOT = ROOT.parents[1]
PHASE2F_SOURCE = PROJECT_ROOT / "isolated_experiments/phase2f_complete_edout_mainline_baselines/src/edout_ratio_runner.py"
PHASE2F_SRC = PHASE2F_SOURCE.parent
PHASE2E1_SRC = PROJECT_ROOT / "isolated_experiments/phase2e1_ed_in_ratio_robustness_formal/src"
PHASE2E0A_SRC = PROJECT_ROOT / "isolated_experiments/phase2e0a_ratio_aware_formal_entry/src"
FREEZE_COMMIT = "32bca7323c5e76b94567dc5d2acc809d57e316a4"
for path in (PHASE2F_SRC,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_phase2f():
    spec = importlib.util.spec_from_file_location("fullrank_phase2f_edout_runner", PHASE2F_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("PHASE2F-RUNNER-IMPORT-FAIL")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = parser("FullRank Phase 2F GCN cell").parse_args()
    row = load_row(args.manifest.resolve(), args.cell_id)
    if row["formal_runner_family"] != "GCN_PHASE2F_EDOUT":
        raise RuntimeError("WRONG-FAMILY-DISPATCH")
    source_config = Path(row["formal_config_path"]).resolve()
    if not source_config.is_file():
        raise FileNotFoundError(source_config)
    run_output = output_dir(row)
    config_path = ROOT / "configs" / f"{row['cell_id']}.json"
    config = write_config(source_config, config_path, output_dir=run_output, run_id=f"fullrank__{row['cell_id']}")
    phase2f = load_phase2f()
    for module_name in ("shared_input_router", "ratio_aware_runner", "campaign_common", "materialize_shared_inputs"):
        sys.modules.pop(module_name, None)
    for path in (PHASE2F_SRC, PHASE2E1_SRC, PHASE2E0A_SRC):
        while str(path) in sys.path:
            sys.path.remove(str(path))
    sys.path[:0] = [str(PHASE2E0A_SRC), str(PHASE2E1_SRC), str(PHASE2F_SRC)]
    import deletion_type_router
    deletion_type_router.ISOLATION_ROOT = ROOT
    frozen = phase2f._load_frozen_runner()
    frozen._ORIGINAL_BUILD_STUDENT = frozen.build_student
    patch_formal_identity(frozen, freeze_commit=FREEZE_COMMIT)
    frozen.ISOLATION_ROOT = ROOT
    frozen.load_graph_runtime = phase2f.load_graph_runtime
    frozen._sampler_and_bundles = phase2f.sampler_and_bundles
    frozen._formal_freeze_commit = lambda: FREEZE_COMMIT
    frozen._load_config = lambda path: phase2f.load_config(path)
    frozen.build_student = make_student_builder(frozen)
    gate_dir = ROOT / "canaries" / row["formal_runner_family"] / row["cell_id"]
    gate = update0_identity(frozen, config, gate_dir, loader=phase2f.load_graph_runtime, sampler=phase2f.sampler_and_bundles)
    gate["FULLRANK_GOLDEN_REPLAY"] = "NOT_APPLICABLE"
    (gate_dir / "implementation_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    completion = frozen.execute(config_path.resolve(), args.device)
    phase2f._postprocess(run_output)
    summary = normalize_completion(run_output, row=row, completion=completion, gate=gate, runner=str(PHASE2F_SOURCE))
    print(json.dumps({"status": summary["status"], "cell_id": row["cell_id"], "selected_epoch": summary["selected_checkpoint_epoch"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
