# ProbeRoute

[![Paper PDF](https://img.shields.io/badge/Paper-PDF-B31B1B.svg)](https://github.com/aliuyar1234/proberoute/raw/main/ProbeRoute_paper.pdf)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg)](ENVIRONMENT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-2ea44f.svg)](LICENSE)

ProbeRoute studies a focused question in frozen-backbone language model adaptation: can future-token probes be turned into a useful routing prior for multi-token prediction?

This repository implements a full research pipeline around that idea. The ProbeRoute method probes horizon-specific signal across transformer depth, uses those probe scores to initialize sparse layer routing, and trains lightweight multi-token heads on top of a frozen pretrained backbone.

## Why This Repo Exists
- To test whether probe-informed sparse routing can outperform simpler frozen-backbone baselines for multi-token prediction.
- To provide a reproducible implementation of the full workflow: data preparation, probe training, MTP training, evaluation, registry assembly, and paper-asset generation.
- To keep the public codebase clean and useful: implementation, configs, tests, and stable design docs are versioned; runtime outputs and internal project-management material are not.

## What Is In Scope
- frozen-backbone multi-token prediction
- probe sweeps over hidden-state depth
- dense and sparse layer-mixing baselines
- evaluation, result registries, and paper asset generation

## Repository Layout
- `src/`: implementation code
- `configs/`: experiment entrypoints
- `tests/`: unit and integration tests
- `fixtures/`: offline smoke-test data
- `schemas/`: result and manifest schemas
- `MODULE_SPECS/`: contributor-facing module contracts
- `PAPER_TEMPLATES/`: manuscript templates and paper asset specs

## Reader's Guide
If you only read a few files, read these:
- [README.md](README.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [ENVIRONMENT.md](ENVIRONMENT.md)
- [DATA_SPEC.md](DATA_SPEC.md)
- [RESULTS_SCHEMA.md](RESULTS_SCHEMA.md)

## Quick Start
Create an environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python -m pip install --upgrade -r requirements-torch-cu128.txt
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
```

Run the offline smoke path:

```powershell
python -m src.cli.prepare_data --config configs/smoke_local_tiny.yaml
python -m src.cli.train_probes --config configs/smoke_local_tiny.yaml
python -m src.cli.train_mtp --config configs/smoke_local_tiny.yaml
python -m src.cli.evaluate --run-dir outputs/runs/SMOKE_LOCAL_TINY/local-toy-gpt/seed_1337
```

Run tests:

```powershell
python -m pytest tests/unit tests/integration
```

## Main Commands
Train probes:

```powershell
python -m src.cli.train_probes --config configs/probe_1b.yaml
```

Train an MTP run:

```powershell
python -m src.cli.train_mtp --config configs/screen_sparse_probe_init_1b.yaml
```

Evaluate a finished run:

```powershell
python -m src.cli.evaluate --run-dir <run_dir>
```

Assemble registries:

```powershell
python -m src.cli.collect_results --outputs-root outputs
```

Build paper assets:

```powershell
python -m src.cli.build_paper_assets --outputs-root outputs
```

Write a draft from artifacts:

```powershell
python -m src.cli.write_paper --outputs-root outputs --template PAPER_TEMPLATES/PAPER_TEMPLATE.md
```

## Notes On Reproducibility
- The canonical runtime is a local `.venv`.
- Generated outputs live under `outputs/` and are intentionally not versioned in this public repository.
- The public repository is meant to expose clean code and reproducible structure, not local experiment state.
