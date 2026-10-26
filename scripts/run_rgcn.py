from __future__ import annotations

from _common import arguments, load_inputs

from lrr.utils import choose_device, save_result
from training.train_rgcn import evaluate_rgcn_test, prepare_rgcn, train_prepared_rgcn


def main() -> None:
    args = arguments("Run formal FullRank LRR with an RGCN + DistMult backbone")
    config, bundle, output = load_inputs(args)
    from torch_geometric.seed import seed_everything
    seed_everything(int(config["seed"]))
    prepared = prepare_rgcn(config, bundle, choose_device(config))
    result = train_prepared_rgcn(prepared)
    if bool(config.get("evaluate_test", False)):
        if "test" not in bundle:
            raise KeyError("evaluate_test is enabled but the bundle has no test section")
        result["test_metrics"] = evaluate_rgcn_test(prepared, bundle["test"])
    result["selected_repair_state"] = {
        name: value.detach().cpu() for name, value in result["selected_repair_state"].items()
    }
    result.update({"dataset": config["dataset"], "backbone": config.get("backbone", "RGCN"),
                   "deletion_mode": config["deletion_mode"],
                   "deletion_ratio": config["deletion_ratio"], "seed": int(config["seed"])})
    save_result(output, result)
    print(f"selected_update={result['selected_update']} output={output}")


if __name__ == "__main__":
    main()
