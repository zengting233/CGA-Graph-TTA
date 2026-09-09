# CGA

## Overview

Curvature-Gated Adaptation (CGA) is a node-wise reliability-allocation framework for graph test-time adaptation under structural shift. It uses matched source–target curvature change to regulate target-side adaptation and source-reference protection.

This repository provides a reference implementation of CGA together with the primary GraphSAGE experiment interfaces and configurations for **T3A** and **TENT**.

## Repository structure

```text
src/                  Core CGA and graph TTA implementation
configs/primary/      Primary experiment configurations
experiments/primary/  Primary experiment runner
environment.yml       Conda environment
```

## Environment

Create the reference environment with:

```bash
conda env create -f environment.yml
conda activate cga
```

Run all commands from the repository root.

## Data

Datasets are **not included** in this repository.

The primary evaluation uses:

- Cora
- CiteSeer
- Cornell
- Texas
- Wisconsin
- Chameleon
- Squirrel

Cora and CiteSeer follow the public [Planetoid](https://github.com/kimiyoung/planetoid) setting.

Cornell, Texas, Wisconsin, Chameleon, and Squirrel use the [Geom-GCN](https://github.com/graphdml-uiuc-jlu/geom-gcn) datasets and fixed split files.

Place the prepared datasets under the default `data/` directory, or specify another location with `--data-root`.

## Running the primary experiments

### T3A

Run a single dataset and seed:

```bash
python experiments/primary/run.py --adapter t3a --dataset Cora --seed 0
```

Run the complete seven-dataset, three-seed evaluation:

```bash
python experiments/primary/run.py --adapter t3a --all
```

### TENT

Run a single dataset and seed:

```bash
python experiments/primary/run.py --adapter tent --dataset Cora --seed 0
```

Run the complete seven-dataset, three-seed evaluation:

```bash
python experiments/primary/run.py --adapter tent --all
```

### Configuration check

Validate the primary experiment configuration without running training:

```bash
python experiments/primary/run.py --adapter all --all --dry-run
```

## Primary protocol

| Component | Setting |
|---|---|
| Datasets | Cora, CiteSeer, Cornell, Texas, Wisconsin, Chameleon, Squirrel |
| Backbone | GraphSAGE |
| Structural shift | Random edge rewiring |
| Severity | 0.5 |
| Seeds | 0, 1, 2 |
| CGA gate | Matched curvature change with median normalization |
| T3A | Reliability-aware prototype adaptation with source-reference protection |
| TENT | Reliability-weighted entropy adaptation with source-reference protection |

Detailed fixed settings are provided in `configs/primary/`.

## Citation

If you use this code, please cite the corresponding manuscript:

**Mitigating Negative Transfer in Graph Test-Time Adaptation via Curvature-Guided Reliability Allocation**

## Acknowledgements

This repository contains paper-specific graph integrations of T3A and TENT.

We acknowledge the original implementations:

- [T3A](https://github.com/matsuolab/T3A)
- [TENT](https://github.com/DequanWang/tent)

Datasets and third-party dependencies remain subject to their respective licenses and terms.
