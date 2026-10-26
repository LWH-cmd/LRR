# Benchmark component data

These files contain the existing benchmark results summarized in Appendix B. They do not represent new experiments.

- benchmark_components.csv: 282 observed method-by-setting rows; full-precision means and sample SDs of all four components and both joint metrics. Missing architecture/method combinations are absent, not assigned numbers.
- benchmark_seed_components.csv: 1,326 seed rows supporting those summaries.
- main_joint_metrics_verified_anchor.csv: unchanged four-decimal extract of the frozen main tables (564 rows).
- distance_decay_seed_components.csv: 20 existing seed rows for the four profiles in Appendix C.
- spatial_control_seed_components.csv: all 40 existing seed rows (eight variants) underlying the main spatial/layer controls and Appendix C.
- formulation_control_seed_components.csv: 15 existing seed rows for the three matched formulation/operator variants reported in the main paper.

J is computed within each seed as (DT + Df)/2. Sample SD uses ddof=1. Settings use five seeds (13,21,42,87,100) for Cora/DBLP and three (13,21,42) for WN18RR. Overall summaries weight setting means equally. Ratio is expressed in percent units (e.g., 2.5 means 2.5%). AUP is average precision. Retrain is a retraining reference; GraphEditor is the shared linear reference, repeated under the two homogeneous backbone labels. GIF, GraphEditor, and INPO-S have no RGCN rows.

The source records are the LRR Formal Data Hub final evidence freeze (2026-08-31, including its accepted replacements and fixed-budget update of 2026-09-27) and the complete INPO-S homogeneous matrix (2026-09-26, with documented reuse of 2026-09-24 results). Distance-decay records are from the fixed controlled experiment of 2026-09-15. The editorial provenance and verification report is delivered separately from the manuscript.
