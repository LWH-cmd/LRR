# Repository and package layout

```text
lrr/
  lrr/                     core models, objectives, samplers and evaluators
  training/                training implementations
  scripts/                 preparation, training, loading and evaluation commands
  configs/
    main/                  138 main experiment configurations
    pretraining/           23 Original backbone recipes
    source_records/        unchanged original control/run configurations
  data/
    splits/                fixed train/validation/test edge indices
    deletion_requests/     fixed requested-edge arrays
  results/
    paper/                 final-paper CSVs and literal table data
    source_records/        original experiment/selection records
  checkpoints/             populated by the model download script
    main/                  138 main selected states
    ablation_sensitivity/   57 selected control states
    pretrained/            23 Original backbone states
  control_models/          source-bound operator-control model classes
  reference_code/          original control implementation references
  manifests/               hashes, experiment IDs and provenance mappings
  docs/                    reproduction and protocol explanations
  outputs/                 new run outputs
```

The source ZIP is `lrr-source-code.zip`; extracting it creates one top-level `lrr/` directory. Model archives are `lrr-main-checkpoints.tar.gz`, `lrr-ablation-sensitivity-checkpoints.tar.gz` and `lrr-pretrained-models.tar.gz`. They extract into the repository's `checkpoints/` tree. Checkpoint filenames identify their experiment or control rather than using opaque content hashes; hashes remain in the manifests.

Release versions are represented by the GitHub Release tag and citation/version metadata. Public folders do not use review dates, internal phase numbers or `FINAL` suffixes. Original paths, IDs and historical labels inside provenance records remain unchanged.
