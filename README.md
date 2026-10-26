# LRR: Localized Residual Repair

Paper code, configurations, selected LRR checkpoints and final experimental records. The interface follows [GNNDelete](https://github.com/mims-harvard/GNNDelete): data preparation → Original model training → unlearning/evaluation.

## Install

Use Python 3.10, PyTorch 2.7.0 and PyG 2.6.1 with matching torch-sparse wheels. Install the PyTorch/PyG wheels for your platform, then:

```bash
pip install -r requirements-reproduction.txt
python tests/smoke_pipeline.py
```

## Evaluate the published main checkpoints

After the Release assets have been published, download them and prepare the dataset:

```bash
python scripts/fetch_assets.py --all
python prepare_dataset.py --dataset DBLP --seed 42
python scripts/evaluate_published.py --cell dblp__gcn__ed_in__2p5__seed42
```

The evaluation command constructs a bundle from the paper Original and frozen input identities if needed, loads the published selected state strictly, and evaluates it without training. New output does not overwrite frozen results. To list all checkpoint/control binding IDs:

```bash
python scripts/load_published.py --list
python scripts/load_published.py --id dblp__gcn__ed_in__2p5__seed42
```

## Train LRR

```bash
python prepare_dataset.py --dataset DBLP --seed 42
python train_gnn.py --config configs/pretraining/gcn_dblp_seed42.json
python delete_gnn.py --cell dblp__gcn__ed_in__2p5__seed42 --original retrained
```

Use `--original published` with downloaded Originals for the actual paper backbone lineage. Fresh stochastic pretraining can differ across environments. `python delete_gnn.py --list` lists the 138 main LRR cells. GAT and RGCN recipes are included under `configs/`; spatial controls use `scripts/run_controls.py`.

## Published files

- `reference_code/` retains the original control/sensitivity objective and runner definitions as source-bound references.
- Core LRR, Original training, evaluation and exact paper-cell configurations are in `lrr/`, `training/`, `scripts/`, `contracts/` and `configs/`.
- **195 unchanged selected checkpoints**: 138 main LRR cells and 57 control/sensitivity/convergence/formulation states. Only the state selected for the paper is kept per producing run. The 5 formulation-C states use the registered GNNDelete operator as an LRR paper control; they are not a general baseline-model collection.
- **23 deduplicated Originals** provide the actual backbone/teacher lineage for training and evaluation, including the backbone required by repair-only RGCN states.
- Three Release archives total approximately **914.5 MB**: main states 319.7 MB, controls 249.8 MB and Originals 344.9 MB. Raw graph datasets and large processed pickles are not bundled.
- `data/splits/` and `data/deletion_requests/`: about 4.2 MB of exact homogeneous split/request indices exported from existing final experiment inputs. These are not resampled.
- `results/paper/`: exact final-paper data, including all ratios, 1,326 benchmark seed rows and 282 method-setting component summaries. `results/source_records/` retains the underlying original records; `configs/source_records/` contains source configurations.

The folder/package guide is [docs/repository_layout.md](docs/repository_layout.md). Download sources and checksum instructions are in [data/README.md](data/README.md). Checkpoint/source mappings and native layout access are in [docs/checkpoints.md](docs/checkpoints.md).

Historical precomputed files, temporary caches, periodic checkpoint snapshots, redundant terminal states, ordinary baseline checkpoints, figures and plotting sources are excluded. The final manuscript ZIP SHA256 is `b8aee723bf7cd078ed01dd36e93ccb0cd4ecb84268ce961e74d374c3fcf46bad`.

## Verification boundary

All 195 selected states passed strict model loading, and the original data features/frozen split restoration were checked for Cora and DBLP. Seven synthetic interface checks passed. This is source/file/interface verification; no full-budget formal training or per-group paper metric replay was performed for release preparation. The new evaluation CLI is checked on synthetic homogeneous data; a full formal matrix replay and cross-platform execution are not claimed. Native sensitivity and control configurations retain their paper protocols; controls are not all exposed as one unified evaluation command.

Please cite the LRR manuscript and the inherited [GNNDelete](https://github.com/mims-harvard/GNNDelete) infrastructure. See `LICENSE`, `THIRD_PARTY_NOTICES.md` and `CITATION.cff`.
