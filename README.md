# CGA

## Overview

Curvature-Gated Adaptation (CGA) is a node-wise reliability-allocation framework for graph test-time adaptation under structural shift. Matched source–target curvature change controls adaptation strength and source-reference protection. This release provides reference graph integrations of T3A and TENT.

## Repository scope

This repository provides a reference implementation of Curvature-Gated Adaptation (CGA), together with the primary GraphSAGE experiment interfaces and configurations for T3A and TENT.

## Repository structure

- `src/`: dataset loading, GraphSAGE, source training, rewiring, curvature gates, and adapter integrations.
- `configs/primary/`: the two fixed primary configurations.
- `experiments/primary/run.py`: single-case and full-primary entry point.
- `environment.yml`: minimal environment derived from the reference environment.

## Environment

The reference environment uses Python 3.11.15, PyTorch 2.11.0+cu128, PyG 2.7.0, NumPy 2.4.4, SciPy 1.17.1, NetworkX 3.6.1, and PyYAML 6.0.3. The pinned CUDA 12.8 build reflects the reference environment; it is not a promise of compatibility with every GPU or driver.

```bash
conda env create -f environment.yml
conda activate cga
python -c "import torch, torch_geometric, numpy, scipy, networkx, yaml; print(torch.__version__, torch_geometric.__version__)"
```

Run commands from the repository root. The runner resolves its default data, configuration, and output paths relative to the repository, not the current working directory. Use `--device cpu` when required; hardware/runtime differences can affect reproducibility.

## Data

Datasets are not distributed with this repository. The strict loader does **not** automatically download missing raw files: prepare them before running an experiment. The default directory is `data/`; `--data-root` can select another directory. Only obtain serialized files from trusted dataset sources.

**Cora and CiteSeer.** Obtain the eight `ind.<name>.*` files from the [original Planetoid data directory](https://github.com/kimiyoung/planetoid/tree/master/data). For each lowercase name `cora` or `citeseer`, the suffixes are `x`, `tx`, `allx`, `y`, `ty`, `ally`, `graph`, and `test.index`. Place them respectively in:

- `data/Planetoid/Cora/raw/`
- `data/Planetoid/CiteSeer/raw/`

The loader then uses PyG Planetoid with the fixed public split and feature normalization. Local processed caches are generated as needed and ignored by Git.

**Cornell, Texas, Wisconsin, Chameleon, and Squirrel.** Use the [Geom-GCN data snapshot](https://github.com/graphdml-uiuc-jlu/geom-gcn/tree/f1fc0d14b3b019c562737240d06ec83b07d16a8f/new_data) and its [split files](https://github.com/graphdml-uiuc-jlu/geom-gcn/tree/f1fc0d14b3b019c562737240d06ec83b07d16a8f/splits). This historical version is also the source pinned by the [PyG WikipediaNetwork loader](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/datasets/wikipedia_network.py); the current Geom-GCN branch may omit the Wikipedia raw files.

For each lowercase dataset name, obtain `new_data/<name>/out1_node_feature_label.txt`, `new_data/<name>/out1_graph_edges.txt`, and `splits/<name>_split_0.6_0.2_0.npz`. Put all three files together in the matching directory:

| Dataset | Raw directory |
|---|---|
| Cornell | `data/WebKB/Cornell/raw/` |
| Texas | `data/WebKB/Texas/raw/` |
| Wisconsin | `data/WebKB/Wisconsin/raw/` |
| Chameleon | `data/WikipediaNetwork/chameleon/raw/` |
| Squirrel | `data/WikipediaNetwork/squirrel/raw/` |

Use the original Geom-GCN-preprocessed classification graphs, not a regression variant or a filtered replacement. The same split index 0 is used for every seed; the loader does not fall back to a randomly generated split.

## Running T3A

Single-dataset/seed and full seven-dataset, three-seed commands:

```bash
python experiments/primary/run.py --adapter t3a --dataset Cora --seed 0
python experiments/primary/run.py --adapter t3a --all
```

## Running TENT

```bash
python experiments/primary/run.py --adapter tent --dataset Cora --seed 0
python experiments/primary/run.py --adapter tent --all
```

Validate both interfaces without loading data or training:

```bash
python experiments/primary/run.py --adapter all --all --dry-run
```

To execute both interfaces across the complete fixed grid, omit `--dry-run`. This schedules 42 cases. Each case trains a source model on the source graph, then evaluates the source, baseline adapter and CGA integration on the same rewired target graph. This is a full computation, not a quick test.

New per-case CSV files are written under `outputs/<adapter>/`. Accuracy fields are fractions; delta and swing fields are percentage points. Existing result files are never overwritten; use a fresh `--output-dir` for another run. Results are not versioned in this repository.

## Primary protocol

| Component | Fixed setting |
|---|---|
| Datasets | Cora, CiteSeer, Cornell, Texas, Wisconsin, Chameleon, Squirrel |
| Backbone | Two-layer, mean-concatenation GraphSAGE encoder; hidden width 64; ELU; dropout 0.5; linear classifier head |
| Shift | Random structural rewiring of unique undirected edge pairs; severity 0.5; converted to bidirectional edges |
| Seeds | 0, 1, 2 (initialization and rewiring); not three different split indices |
| Splits | Planetoid public splits; Geom-GCN split index 0 |
| Source training | AdamW; learning rate 0.01; weight decay 0.0005; up to 200 epochs; validation patience 30 |
| Gate | Matched node-curvature change; median normalization; alpha 2; epsilon 1e-8 |
| Adaptation mask | Test-node membership without test-label supervision |
| T3A | Logit-space cosine prototype classifier; support quantile 0.7; CGA rank-20/80 gate; confidence × gate² for both support scores and prototype averaging weights; node-wise source-posterior mixing |
| TENT | Classifier-head adaptation in evaluation mode; Adam; 20 steps; learning rate 0.0015; no weight decay; identity gate; entropy exponent 4; anchor exponent 0.5; source-KL coefficient 12 |

The two configuration files are intentionally fixed to this protocol, and the runner rejects settings outside it. Source validation selects the checkpoint. The inherited trainer also reports source-graph test accuracy, but it does not use that quantity for checkpoint selection. Target labels are used for accuracy reporting, not adaptation updates.

The code is provided as a reference implementation with the interfaces and fixed settings documented above. Numerical agreement with individual reported results has not been established for this release.

## Citation

If you use this code, please cite the corresponding manuscript:

*Mitigating Negative Transfer in Graph Test-Time Adaptation via Curvature-Guided Reliability Allocation.*

This is a manuscript code release; no publication DOI, volume, or page range is asserted.

## Acknowledgements and licensing

We acknowledge the original [T3A](https://github.com/matsuolab/T3A) and [TENT](https://github.com/DequanWang/tent) methods. The integrations here are the paper-specific graph implementations, not the upstream image-adaptation packages. No upstream repository source tree or dataset is vendored. Dependencies and datasets remain subject to their respective terms. No additional blanket software license is granted by this release.
