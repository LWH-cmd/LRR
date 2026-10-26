# Paper results

These CSV files retain the exact numerical contents of the final manuscript package. Their release filenames describe their contents; original manuscript member names and SHA256 values are recorded in `manifests/paper_version.json`.

| File | Contents |
|---|---|
| `benchmark_summary.csv` | 282 method-setting summaries with DT, Df and joint components |
| `benchmark_seed_results.csv` | 1,326 benchmark seed records |
| `main_joint_metrics.csv` | 564 main joint-metric summary rows |
| `spatial_control_seed_results.csv` | Spatial/layer/gate control seed records |
| `distance_decay_seed_results.csv` | Distance-decay sensitivity seed records |
| `formulation_control_seed_results.csv` | Objective/operator control seed records |
| `paper_tables.tex` | Literal paper sensitivity, parameter-count and runtime table excerpts |

All published deletion ratios are retained. Original supporting experiment records are in `results/source_records/`, and source configurations are in `configs/source_records/`. The original manuscript data README is retained byte-for-byte at `results/source_records/paper_data_readme.md`; it documents the original manuscript filenames, while this page documents the public repository names. New runs are written to `outputs/`.
