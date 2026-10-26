from __future__ import annotations

import time

from _common import arguments, load_inputs

from lrr.utils import choose_device, save_json, synchronize
from training.train_gcn import prepare_gcn, train_prepared_gcn


def main() -> None:
    args = arguments("Measure formal GCN LRR training wall-clock")
    config, bundle, output = load_inputs(args)
    device = choose_device(config)
    prepared = prepare_gcn(config, bundle, device)
    synchronize(device)
    started = time.perf_counter()
    result = train_prepared_gcn(prepared)
    synchronize(device)
    elapsed = time.perf_counter() - started
    save_json(output.with_suffix(".json"), {
        "timing_definition": "formal training updates + scheduled validation + selector",
        "runtime_seconds": elapsed,
        "selected_update": result["selected_update"],
        "validation_count": result["validation_count"],
        "updates": result["updates"],
        "trainable_parameters": result["trainable_parameters"],
    })
    print(f"runtime_seconds={elapsed:.6f} selected_update={result['selected_update']}")


if __name__ == "__main__":
    main()
