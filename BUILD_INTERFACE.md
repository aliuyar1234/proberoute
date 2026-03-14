# BUILD_INTERFACE.md

Codex must maintain a minimal, explicit build interface around `.venv`, the smoke path, the experiment CLIs, and the artifact pipeline.

## Canonical interpreter

Use `.\.venv\Scripts\python` for all new real runs.

Acceptance:

- `.venv` is the canonical interpreter for new authoritative runs,
- manifests record the interpreter path,
- non-`.venv` real runs are treated as non-authoritative.

## Required targets

### `make env`

Creates `.venv`, installs dependencies into `.venv`, prints key versions, and refreshes `requirements-lock.txt`.

Acceptance:

- installs `requirements-torch-cu128.txt`, `requirements.txt`, and `requirements-dev.txt` in order,
- prints Python, torch, CUDA, transformers, datasets, and accelerate versions,
- refreshes `requirements-lock.txt`.

### `make smoke`

Runs the local offline smoke path.

Acceptance:

- no internet access required,
- data prep, probe train, MTP train, eval, and schema checks pass,
- smoke artifacts are written to canonical paths,
- pause/resume smoke passes for both `train_probes` and `train_mtp`.

### `make prepare-data`

Builds or validates the processed dataset for the target config.

Acceptance:

- processed splits and split-specific eval subsets exist,
- rerun is idempotent for compatible configs,
- incompatible existing data fails loudly unless force rebuild is explicit.

### `make probes`

Runs the mandatory real probe sweep.

Acceptance:

- `PROBE_410M` and `PROBE_1B` complete,
- each run has periodic validation,
- each run records accumulation metadata and optimizer-step semantics correctly,
- each run records a true validation-selected `best.pt`,
- each run emits the required probe artifacts.

### `make screen`

Runs all five mandatory screening configs and rebuilds registries.

Acceptance:

- all required screening runs complete,
- each completed run has authoritative best-checkpoint eval artifacts for val and test,
- `outputs/registries/screening_results.csv` contains only authoritative completed screening rows.

### `make collect`

Rebuilds registries.

Acceptance:

- deterministic output,
- `run_registry.csv` is operational history,
- scientific registries exclude smoke, stale, non-completed, and non-authoritative runs.

### `make select-finalist`

Runs finalist selection and emits the authoritative generated final baseline config.

Acceptance:

- refuses to run if any required completed real baseline screening rows are missing,
- excludes the sparse main method from candidate baselines,
- writes `outputs/registries/finalist_selection.json`,
- writes the authoritative `configs/generated/final_best_baseline_1b.yaml` with provenance.

### `make final`

Runs the authoritative final baseline and the final sparse main method, then rebuilds registries.

Acceptance:

- both runs complete,
- both have authoritative best-checkpoint val/test eval artifacts,
- `outputs/registries/main_results.csv` is refreshed.

### `make ablations`

Runs required ablations and one confirmatory path, then rebuilds registries.

Acceptance:

- `outputs/registries/ablation_results.csv` reflects only authoritative rows,
- waived work is logged explicitly in `logs/DECISIONS.md`.

### `make eval`

Runs evaluation and paper-asset packaging for completed authoritative runs.

Acceptance:

- split-specific eval artifacts exist where required,
- figures/tables come from authoritative source registries only,
- `paper_asset_manifest.json` exists.

### `make paper`

Builds the paper draft and final delivery note.

Acceptance:

- the claim-evidence matrix is artifact-driven,
- no `pending` claim rows remain,
- referenced figure/table/appendix artifacts exist,
- `paper/paper_draft.md` and `FINAL_DELIVERY.md` exist.

## Required CLIs

```bash
python -m src.cli.prepare_data --config configs/smoke_local_tiny.yaml --dry-run
python -m src.cli.prepare_data --config configs/probe_1b.yaml
python -m src.cli.train_probes --config configs/probe_1b.yaml --resume
python -m src.cli.train_mtp --config configs/screen_sparse_probe_init_1b.yaml --resume
python -m src.cli.request_pause --run-dir outputs/runs/PROBE_1B/EleutherAI-pythia-1b/seed_1337
python -m src.cli.evaluate --run-dir outputs/runs/BASE_LAST_MLP_1B/pythia-1b/seed_1337
python -m src.cli.collect_results --outputs-root outputs
python -m src.cli.select_finalist --outputs-root outputs --emit-config configs/generated/final_best_baseline_1b.yaml
python -m src.cli.build_paper_assets --outputs-root outputs
python -m src.cli.write_paper --outputs-root outputs --template docs/PAPER_TEMPLATES/PAPER_TEMPLATE.md
python -m src.cli.sync_docs
```

## CLI hard rules

- `train_probes` and `train_mtp` must refuse a fresh launch into an existing run directory unless `--resume` or an explicit force/reset path is used.
- `--resume` must resume only from an existing compatible `checkpoints/last.pt`; it must never silently start a fresh run.
- `hardware.grad_accum` must be active in both training CLIs.
- `train_metrics.jsonl.step`, `val_metrics.jsonl.step`, manifest step fields, and checkpoint step fields must mean optimizer step.
- `train.eval_every_steps` and `train.save_every_steps` count microbatches and execute at the next accumulation boundary.
- pause, checkpoint, and resume boundaries must be accumulation boundaries only.
- `request_pause` must only create the pause marker and must never kill the process.
- Clean pause exits with status `paused`.
- Manual interruption outside the pause contract is `aborted`.
- `evaluate` must default to `checkpoints/best.pt`.
- `evaluate` must emit:
  - `eval/val_future_metrics.json`
  - `eval/test_future_metrics.json`
  - `eval/val_acceptance_metrics.json`
  - `eval/test_acceptance_metrics.json`
  - `eval/router_metrics.json`
- Screening/finalist/paper logic must consume `best.pt` evaluation only.
- Smoke runs must be excluded from scientific registries and paper-facing assets by default.
