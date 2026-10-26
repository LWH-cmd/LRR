# Comparison methods

The final-paper result exports preserve all methods displayed in the paper. Their source/run identities and protocols are in `manifests/result_provenance.csv`, `03_PROTOCOL_REGISTRY.csv` and `06_BASELINE_REGISTRY.csv`. Main command-line training/evaluation interfaces focus on LRR; ordinary baseline checkpoints and the historical baseline workspace are not included.

See [GNNDelete](https://github.com/mims-harvard/GNNDelete) for its official implementation and baseline references. The five formulation-C selected states belong to the LRR paper's controlled operator study and remain included with their two native source classes in `control_models/`. They are not relabeled as LRR main states. Generic upstream defaults do not establish the protocol that produced the supplied paper numbers.
