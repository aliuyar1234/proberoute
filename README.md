# LayerMix-MTP

LayerMix-MTP is a research codebase for adapting frozen autoregressive language models to multi-token prediction via probe-informed routing over hidden states.

This public repo is intentionally slim. It keeps the implementation, stable design docs, configs, tests, and paper-generation templates needed to understand or reproduce the system structure. Runtime artifacts, private handoff notes, transient project-management docs, and generated experiment outputs are excluded from version control.

## What This Repo Contains
- `src/`: training, evaluation, analysis, and CLI code
- `configs/`: canonical experiment entrypoints
- `tests/`: unit and integration coverage
- `fixtures/`: offline smoke-test assets
- `schemas/`: result and manifest schemas
- `MODULE_SPECS/`: module-level design guardrails
- `PAPER_TEMPLATES/`: paper-writing templates used by the writing pipeline

## Stable Design Docs
These docs remain in the public repo because they help keep implementation and build behavior aligned with the intended design:
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [BUILD_INTERFACE.md](BUILD_INTERFACE.md)
- [DATA_SPEC.md](DATA_SPEC.md)
- [ENVIRONMENT.md](ENVIRONMENT.md)
- [RESULTS_SCHEMA.md](RESULTS_SCHEMA.md)
- [VALIDATION_AND_TESTING.md](VALIDATION_AND_TESTING.md)
- [LICENSE_AND_COMPLIANCE.md](LICENSE_AND_COMPLIANCE.md)

## Quick Start
1. Create the environment:
   `make env`
2. Run the offline smoke path:
   `make smoke`
3. Run tests:
   `make test`

## Main CLI Workflow
- Prepare data:
  `python -m src.cli.prepare_data --config configs/probe_1b.yaml`
- Train probes:
  `python -m src.cli.train_probes --config configs/probe_1b.yaml`
- Train MTP runs:
  `python -m src.cli.train_mtp --config configs/screen_sparse_probe_init_1b.yaml`
- Evaluate a finished run:
  `python -m src.cli.evaluate --run-dir <run_dir>`
- Assemble registries:
  `python -m src.cli.collect_results --outputs-root outputs`
- Build paper assets:
  `python -m src.cli.build_paper_assets --outputs-root outputs`
- Write a draft paper:
  `python -m src.cli.write_paper --outputs-root outputs --template PAPER_TEMPLATES/PAPER_TEMPLATE.md`

## Public Repo Policy
- Model weights and datasets are not bundled.
- Generated outputs under `outputs/` are not tracked.
- Session handoff docs, transient planning notes, and internal operational material are intentionally kept out of the public git surface.
- The public repo is for clean implementation, validation, and reproducible structure, not for shipping local runtime state.
