"""Direct formal-source loader probe without any FullRank import.

This is a native-runtime isolation check.  It constructs the formal graph and
native GraphSAINT loader, then requests exactly one batch.  It does not train
or evaluate a model.
"""

from __future__ import annotations

import argparse
import builtins
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
FORMAL_SOURCE = PROJECT_ROOT / "isolated_experiments/phase4c_wordnet18rr_p2_formal_seed42/src/run_formal_p2_crossed.py"


def load_formal():
    spec = importlib.util.spec_from_file_location("rgcn_direct_formal_loader", FORMAL_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("formal import spec failure")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    formal = load_formal()
    print("DIRECT_MARKER=load_graph", flush=True)
    graph, base_relations, counts, train = formal.load_graph()
    print("DIRECT_MARKER=build_assets", flush=True)
    assets = formal.build_assets(graph, base_relations, train)
    formal.add_test_bank(graph, assets)
    print("DIRECT_MARKER=build_request", flush=True)
    request = formal.build_request(graph, base_relations, train)
    data = formal.build_data(train, graph.train_edge_type, request["selected_train_positions"])
    print("DIRECT_MARKER=construct_native_graphsaint", flush=True)
    loader_cls = builtins.__import__("torch_geometric.loader", fromlist=["GraphSAINTRandomWalkSampler"]).GraphSAINTRandomWalkSampler
    loader = loader_cls(data, batch_size=1024, walk_length=2, num_steps=32, num_workers=0)
    print("DIRECT_MARKER=next_native_graphsaint_batch", flush=True)
    batch = next(iter(loader))
    result = {
        "status": "PASS",
        "batch_nodes": int(batch.x.numel()),
        "batch_edges": int(batch.edge_index.size(1)),
        "batch_edge_type": list(batch.edge_type.shape),
        "num_nodes": int(counts["num_entities"]),
        "formal_source": str(FORMAL_SOURCE),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
