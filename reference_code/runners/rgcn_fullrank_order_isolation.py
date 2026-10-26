"""Fresh-process order isolation for the RGCN native sampler crash."""

from __future__ import annotations

import argparse
import builtins
import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
FORMAL_SOURCE = PROJECT_ROOT / "isolated_experiments/phase4c_wordnet18rr_p2_formal_seed42/src/run_formal_p2_crossed.py"
FULLRANK_SOURCE = ROOT / "runners/fullrank_operators.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"import failed: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_inputs(formal):
    formal.set_seed(formal.SEED)
    graph, base_relations, counts, train = formal.load_graph()
    assets = formal.build_assets(graph, base_relations, train)
    formal.add_test_bank(graph, assets)
    request = formal.build_request(graph, base_relations, train)
    data = formal.build_data(train, graph.train_edge_type, request["selected_train_positions"])
    config = json.loads(formal.ORIGINAL_CONFIG.read_text(encoding="utf-8"))
    return graph, assets, data, config, counts


def make_fullrank(formal, fullrank, config, device):
    backbone = formal.build_original_model(formal.NUM_NODES, formal.BASE_RELATIONS, config, device)
    payload = torch.load(formal.ORIGINAL_CHECKPOINT, map_location="cpu", weights_only=False)
    backbone.load_state_dict(payload["model_state"], strict=True)
    return fullrank.FullRankRGCN(backbone, formal.SEED)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("none", "import_only", "construct_before", "construct_after"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    formal = load_module(FORMAL_SOURCE, "rgcn_order_formal")
    device = torch.device("cuda")
    graph, assets, data, config, counts = build_inputs(formal)
    fullrank = None
    operator = None
    if args.case in ("import_only", "construct_before"):
        fullrank = load_module(FULLRANK_SOURCE, "rgcn_order_fullrank")
    if args.case == "construct_before":
        operator = make_fullrank(formal, fullrank, config, device)
    loader_cls = builtins.__import__("torch_geometric.loader", fromlist=["GraphSAINTRandomWalkSampler"]).GraphSAINTRandomWalkSampler
    loader = loader_cls(data, batch_size=1024, walk_length=2, num_steps=32, num_workers=0)
    print(f"ORDER_MARKER=next_batch case={args.case}", flush=True)
    batch = next(iter(loader))
    if args.case == "construct_after":
        fullrank = load_module(FULLRANK_SOURCE, "rgcn_order_fullrank")
        operator = make_fullrank(formal, fullrank, config, device)
    result = {"status": "PASS", "case": args.case, "batch_nodes": int(batch.x.numel()),
              "batch_edges": int(batch.edge_index.size(1)), "counts": counts,
              "formal_source": str(FORMAL_SOURCE), "fullrank_source": str(FULLRANK_SOURCE)
              if fullrank is not None else None,
              "operator_constructed": operator is not None}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
