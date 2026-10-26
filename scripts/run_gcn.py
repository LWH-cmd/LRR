from __future__ import annotations

from _common import arguments, load_inputs

from lrr.utils import choose_device, save_result
from training.train_gcn import evaluate_gcn_test, prepare_gcn, train_prepared_gcn


def main() -> None:
    args = arguments("Run formal FullRank LRR with a GCN backbone")
    config, bundle, output = load_inputs(args)
    prepared = prepare_gcn(config, bundle, choose_device(config))
    result = train_prepared_gcn(prepared)
    if bool(config.get("evaluate_test", False)):
        if "test" not in bundle:
            raise KeyError("evaluate_test is enabled but the bundle has no test section")
        result["test_metrics"] = evaluate_gcn_test(prepared, bundle["test"])
    result["selected_repair_state"] = {
        name: value.detach().cpu() for name, value in result["selected_repair_state"].items()
    }
    result.update({"dataset": config["dataset"], "backbone": config.get("backbone", "GCN"),
                   "deletion_mode": config["deletion_mode"],
                   "deletion_ratio": config["deletion_ratio"], "seed": int(config["seed"])})
    save_result(output, result)
    print(f"selected_update={result['selected_update']} output={output}")


if __name__ == "__main__":
    main()
