from __future__ import annotations

import time

from _common import arguments, load_inputs

from lrr.utils import choose_device, save_json, synchronize
from training.train_rgcn import prepare_rgcn, train_prepared_rgcn


def main() -> None:
    args = arguments("Measure formal RGCN LRR training wall-clock")
    config, bundle, output = load_inputs(args)
    device = choose_device(config)
    prepared = prepare_rgcn(config, bundle, device)
    synchronize(device)
    started = time.perf_counter()
    result = train_prepared_rgcn(prepared)
    synchronize(device)
    elapsed = time.perf_counter() - started
    save_json(output.with_suffix(".json"), {
        "timing_definition": "GraphSAINT updates + update-0 and formal selector validations + selector",
        "runtime_seconds": elapsed,
        "selected_update": result["selected_update"],
        "validation_count": len(result["validation_trajectory"]),
        "validation_updates": result["validation_updates"],
        "updates": result["updates"],
        "optimizer_steps": result["optimizer_steps"],
        "trainable_parameters": result["trainable_parameters"],
        "graphsaint_backend": "Windows-safe random-walk backend",
    })
    print(f"runtime_seconds={elapsed:.6f} selected_update={result['selected_update']}")


if __name__ == "__main__":
    main()
