# Implementation sources

| Public location | Role and provenance |
|---|---|
| `lrr/`, `training/` | LRR models, objectives, selectors, evaluators and training implementations |
| `contracts/` | Source-bound graph/sampling helpers; original hashes and extracted symbols are in `manifests/source_bindings.json` |
| `control_models/` | Unchanged native GCN/operator-control classes, bound by `manifests/operator_control_sources.json` |
| `reference_code/` | Unchanged spatial, sensitivity and formulation-control implementation references, bound by `manifests/control_implementation_sources.json` |
| `configs/source_records/` | Original source configurations retained byte-for-byte |
| `results/source_records/` | Original experiment and selection records retained byte-for-byte |

Core runtime imports resolve within this repository and installed dependencies. The `reference_code/` files retain historical source layout assumptions in their contents; they are source references rather than a separately validated unified portable runner. Current commands are documented in the repository README and `docs/reproduction.md`. Source paths and experiment IDs remain intact even where public delivery filenames have changed.

The active preparation entry is `prepare_dataset.py` with `scripts/build_bundle.py`. Historical precomputed schedules and their obsolete packaging helper are excluded; the active Retain representation uses its registered role-seeded identity.
