# Datasets, splits and deletion requests

| Dataset | Source | Paper variant |
|---|---|---|
| Cora | [Graph2Gauss Cora](https://github.com/abojchevski/graph2gauss/raw/master/data/cora.npz) | CitationFull/Cora |
| DBLP | [Graph2Gauss DBLP](https://github.com/abojchevski/graph2gauss/raw/master/data/dblp.npz) | CitationFull/DBLP |
| WN18RR | [native split files](https://github.com/villmow/datasets_knowledge_embedding/tree/master/WN18RR/original) | 11 base relations |

Use `python prepare_dataset.py --dataset Cora --seed 42`, or select `DBLP` / `WN18RR`. The PyG sources are documented in [CitationFull 2.6.1](https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.datasets.CitationFull.html) and [WordNet18RR 2.6.1](https://pytorch-geometric.readthedocs.io/en/2.6.1/_modules/torch_geometric/datasets/word_net.html). Raw graph files are downloaded and checked against `manifests/raw_datasets.json`; full graphs and large processed pickles are not bundled.

- `data/splits/`: ten exact homogeneous seed splits, exported from the existing final experiment inputs. Filenames follow `cora_seed_13.npz` / `dblp_seed_13.npz`. Features are downloaded and checked against their source tensor hashes.
- `data/deletion_requests/`: 120 exact homogeneous request edge arrays, named by the unchanged experiment ID. Their original column order is retained.
- `manifests/data_splits.json` and `manifests/deletion_requests.json`: exported-file hashes, original sources and source hashes.

WN18RR retains its native split files and the registered relational construction. Historical precomputed sampling files remain excluded. Numerical experiment records belong under `results/`; original configurations are in `configs/source_records/`. Historical paths inside those source records are preserved as provenance.
