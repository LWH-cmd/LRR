# Selected checkpoint access

`manifests/checkpoints.json` maps every selected checkpoint to its original source path, paper-cell/control IDs, SHA256 and Release archive. Native checkpoints are stored byte-for-byte; optimizer/metadata fields already inside a selected file are preserved. Periodic snapshots, duplicate backup states and ordinary baseline models are excluded.

The selection includes 138 main LRR cells plus 57 spatial/layer/distance, objective-coefficient, convergence and formulation-control states. Defaults reused by multiple tables share one selected file with multiple bindings. Formulation C includes its five registered GNNDelete-operator controls, with the source-bound classes in `control_models/`; this is separate from the excluded ordinary baseline checkpoint collection.

```bash
python scripts/fetch_assets.py --list
python scripts/fetch_assets.py --all
python scripts/load_published.py --list
python scripts/load_published.py --id cora__gcn__ed_in__2p5__seed42
```

`load_published.load_model(entry)` returns the strictly loaded model and unchanged native payload. RGCN repair-only states load their exact Original backbone/decoder through `manifests/experiments.json`; the Original hash is verified. Main evaluation is available through `scripts/evaluate_published.py`. Control loading preserves its native operator, and evaluating a control must use that control's recorded gates/objective/input protocol rather than the default main configuration.

The default `fetch_assets.py` download remains the Original archive only; `--all` also downloads both selected-state archives. Each archive is independent and smaller than 350 MB. Checksums are verified before extraction. Files extract into `checkpoints/main/`, `checkpoints/ablation_sensitivity/` and `checkpoints/pretrained/`. Their filenames identify the experiment, control or pretrained backbone; content hashes remain in the manifests. `scripts/manage_artifacts.py` can list, verify, or restore native paths into a separate directory; existing different files are not overwritten. Restoration may use hard links, so copy a restored file before editing it.

`manifests/artifact_catalog.json` also indexes the selected native configuration/result records already in Git. `results/paper/` remains the final paper's display population. Native configuration snapshots can retain historical precomputed path strings; the referenced precomputed files and their dedicated sources are not bundled or used by the main role-seeded interface.
