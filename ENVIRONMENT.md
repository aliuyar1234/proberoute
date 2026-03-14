# ENVIRONMENT.md

## Runtime assumptions

Preferred real-run profile:

- 1 x NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- CUDA available
- `.venv` as the canonical interpreter

Supported validation profile:

- CPU-only smoke tests
- CPU-only schema and contract tests
- CPU-only doc and registry work while the GPU is busy

## Canonical bootstrap

Canonical interpreter:

```powershell
.\.venv\Scripts\python
```

Preferred bootstrap path:

```powershell
make env
```

Equivalent explicit steps:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python -m pip install --upgrade -r requirements-torch-cu128.txt
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python -m pip freeze > requirements-lock.txt
```

## Canonical package surface

The repo is currently wired around:

- Python 3.12
- `torch==2.9.1+cu128`
- CUDA 12.8
- `transformers>=4.45,<4.50`
- `datasets>=2.20,<3.1`
- `accelerate>=0.34,<1.1`

## Environment policy

- New authoritative runs must be launched from `.venv`.
- The interpreter path is recorded in `run_manifest.json` and `environment_snapshot.json`.
- Runs launched from a non-`.venv` interpreter are treated as non-authoritative for scientific registries.
- Global interpreters may still be used for ad hoc inspection, but not for new authoritative experiment launches.

## Reproducibility policy

Each run must record:

- resolved config path
- environment snapshot path
- interpreter path
- config digest
- micro batch size
- gradient accumulation
- nominal effective batch size
- nominal tokens per optimizer update
- Python / torch / CUDA versions
- seed
- tokenizer name
- backbone name

Each training checkpoint must record:

- optimizer step
- micro step
- consumed tokens
- model state
- optimizer state
- scheduler/scaler state if used
- RNG state
- config digest
- checkpoint timestamp

## Checkpoint and run-control policy

- `last.pt` is the resumable checkpoint.
- `best.pt` is the validation-selected checkpoint.
- `request_pause` creates only the pause marker.
- `paused` means the trainer honored the pause at a checkpoint/validation boundary and wrote `last.pt`.
- `aborted` means the run was stopped outside the clean pause contract.
- `failed` means the run exited due to an exception or contract violation.

## Precision and memory policy

Default precision for required GPU runs:

- `bf16` if supported
- otherwise `fp16`

Allowed exceptions:

- smoke and CPU validation paths may use `fp32`
- frozen-backbone interfaces may cross reduced-precision / fp32 boundaries for numerical stability

`grad_accum` is part of the scientific training methodology even on the 96 GB workstation. Its primary role is to preserve the intended nominal update batch across model scales and fallback hardware profiles. Using it as an OOM recovery lever is secondary, not the core reason it exists.

If OOM occurs, use the recovery ladder:

1. reduce microbatch size
2. increase gradient accumulation
3. reduce sequence length for probe/screening-only recovery paths
4. reduce screening quotas only
5. step down model scale

## Filesystem policy

Canonical writable paths:

- `outputs/data/processed/`
- `outputs/runs/`
- `outputs/registries/`
- `outputs/paper_assets/`
- `outputs/archive/`

Do not hard-code user-specific absolute paths in committed configs.

## Forbidden drift

Do not:

- make a global interpreter the canonical runtime,
- silently treat old non-authoritative runs as evidence,
- switch the project into a distributed-training rewrite,
- or introduce new required CUDA extensions as a hidden prerequisite.
