from __future__ import annotations

from _common import arguments, load_inputs
from lrr.utils import choose_device, save_result
from training.train_gat import evaluate_gat_test, prepare_gat, train_prepared_gat


def main() -> None:
    args = arguments("Run formal FullRank LRR with COMMON-GAT-v1")
    config, bundle, output = load_inputs(args)
    prepared = prepare_gat(config, bundle, choose_device(config))
    result = train_prepared_gat(prepared)
    if bool(config.get("evaluate_test", False)):
        if "test" not in bundle:
            raise KeyError("evaluate_test is enabled but the bundle has no test section")
        result["test_metrics"] = evaluate_gat_test(prepared, bundle["test"])
    result["selected_repair_state"] = {
        name: value.detach().cpu() for name, value in result["selected_repair_state"].items()}
    result.update({key: config[key] for key in
                   ("dataset", "backbone", "deletion_mode", "deletion_ratio", "seed")})
    save_result(output, result)
    print(f"selected_update={result['selected_update']} output={output}")


if __name__ == "__main__":
    main()
