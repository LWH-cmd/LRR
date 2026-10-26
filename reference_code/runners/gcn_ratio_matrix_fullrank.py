"""FullRank injection for the authoritative Phase 2E-1 GCN family."""

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
FORMAL_SOURCE = PROJECT_ROOT / "isolated_experiments/phase2e1_ed_in_ratio_robustness_formal/src/formal_runner.py"
FREEZE_COMMIT = "791ae4762a34167d1ff35423594330f35a2fc8aa"
FORMAL_SRC = FORMAL_SOURCE.parent
if str(FORMAL_SRC) not in sys.path:
    sys.path.insert(0, str(FORMAL_SRC))


def load_formal():
    spec = importlib.util.spec_from_file_location("fullrank_phase2e1_formal_runner", FORMAL_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("PHASE2E1-FORMAL-RUNNER-IMPORT-FAIL")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = parser("FullRank Phase 2E-1 GCN cell") .parse_args()
    row = load_row(args.manifest.resolve(), args.cell_id)
    if row["formal_runner_family"] != "GCN_PHASE2E1_EDIN_RATIO_AWARE":
        raise RuntimeError("WRONG-FAMILY-DISPATCH")
    source_config = Path(row["formal_config_path"]).resolve()
    if not source_config.is_file():
        raise FileNotFoundError(source_config)
    run_output = output_dir(row)
    config_path = ROOT / "configs" / f"{row['cell_id']}.json"
    config = write_config(source_config, config_path, output_dir=run_output, run_id=f"fullrank__{row['cell_id']}")
    formal = load_formal()
    formal._ORIGINAL_BUILD_STUDENT = formal.build_student
    patch_formal_identity(formal, freeze_commit=FREEZE_COMMIT)
    formal.ISOLATION_ROOT = ROOT
    formal.build_student = make_student_builder(formal)
    gate_dir = ROOT / "canaries" / row["formal_runner_family"] / row["cell_id"]
    gate = update0_identity(formal, config, gate_dir)
    gate["FULLRANK_GOLDEN_REPLAY"] = "NOT_APPLICABLE"
    (gate_dir / "implementation_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    completion = formal.execute(config_path.resolve(), args.device)
    summary = normalize_completion(run_output, row=row, completion=completion, gate=gate, runner=str(FORMAL_SOURCE))
    print(json.dumps({"status": summary["status"], "cell_id": row["cell_id"], "selected_epoch": summary["selected_checkpoint_epoch"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
