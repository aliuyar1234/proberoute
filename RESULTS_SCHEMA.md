# RESULTS_SCHEMA.md

## Purpose

This file defines the authoritative artifact layout. The key distinction is:

- `run_registry.csv` is operational history.
- scientific registries and paper assets may contain only authoritative evidence.

## Output root

```text
outputs/
  data/
  runs/
  registries/
  paper_assets/
  archive/
```

`outputs/archive/` is reserved for quarantined pre-adjudication or stale artifacts and must not feed authoritative registries.

## Dataset artifacts

Canonical processed dataset layout:

```text
outputs/data/processed/{dataset_id}/
  manifest.json
  dataset_manifest.json
  train.npy
  val.npy
  test.npy
  eval_future_sequences_val.npy
  eval_future_sequences_test.npy
  eval_acceptance_prefixes_val.npy
  eval_acceptance_prefixes_test.npy
  tokenizer_vocab.json            # optional for local toy tokenizer
```

`manifest.json` / `dataset_manifest.json` must include:

- dataset identity fields,
- split and normalization policy,
- token and sequence counts by split,
- requested and realized quotas,
- dropped-tail counts,
- `eval_future_sequence_count`,
- `eval_acceptance_prefix_count`,
- `eval_future_sequence_count_by_split`,
- `eval_acceptance_prefix_count_by_split`,
- creation timestamp.

The unsuffixed legacy eval subset filenames are non-authoritative.

## Probe run layout

```text
outputs/runs/{exp_id}/{model_slug}/seed_{seed}/
  resolved_config.yaml
  environment_snapshot.json
  run_manifest.json
  train_metrics.jsonl
  val_metrics.jsonl
  checkpoints/
    best.pt
    last.pt
  control/
    pause.request                 # optional while requested
  artifacts/
    probe/
      probe_scores.csv
      probe_heatmap_top1.png
      probe_heatmap_top5.png
      probe_heatmap_nll.png
      probe_init.json
      probe_init.pt
      probe_checkpoint.pt
```

Probe runs are authoritative evidence only if they are:

- stage `probe`,
- status `completed`,
- launched from `.venv`,
- equipped with best-checkpoint metadata,
- and complete with the required probe artifacts.

Training-log semantics:

- `train_metrics.jsonl.step` means optimizer step.
- `train_metrics.jsonl.micro_step` is the processed microbatch counter.
- `val_metrics.jsonl.step` means optimizer step.
- `eval_every_steps` / `save_every_steps` count microbatches and execute at the next accumulation boundary.

## Non-probe run layout

```text
outputs/runs/{exp_id}/{model_slug}/seed_{seed}/
  resolved_config.yaml
  environment_snapshot.json
  run_manifest.json
  train_metrics.jsonl
  val_metrics.jsonl
  checkpoints/
    best.pt
    last.pt
  control/
    pause.request                 # optional while requested
  eval/
    val_future_metrics.json
    test_future_metrics.json
    val_acceptance_metrics.json
    test_acceptance_metrics.json
    router_metrics.json
  artifacts/
    traces/
  training_summary.json
```

Non-probe runs are authoritative evidence only if they are:

- non-smoke,
- status `completed`,
- launched from `.venv`,
- equipped with best-checkpoint metadata,
- and evaluated from `best.pt` into the split-specific eval files above.

## `run_manifest.json` contract

Minimum required fields remain in `schemas/run_manifest.schema.json`, and authoritative runs should additionally record:

- `environment_snapshot_path`
- `interpreter_path`
- `resume_count`
- `config_digest`
- `micro_batch_size`
- `grad_accum`
- `nominal_effective_batch_sequences`
- `nominal_tokens_per_optimizer_update`
- `optimizer_step`
- `micro_step`
- `consumed_tokens`
- `realized_train_tokens`
- `last_checkpoint_path`
- `last_checkpoint_step`
- `last_checkpoint_micro_step`
- `best_step`
- `best_checkpoint_path`
- `best_mean_val_nll_h1_h4`
- `best_mean_val_top1_h2_h4`
- `checkpoint_selection_rule`
- `last_evaluated_checkpoint_name` when eval has been run
- `last_evaluated_checkpoint_path` when eval has been run

Status semantics:

- `running`: active training writer owns the run directory
- `paused`: clean pause at a checkpoint/validation boundary
- `completed`: training finished cleanly
- `failed`: exception or contract failure
- `aborted`: manual stop outside the pause contract

## Checkpoint contract

- `last.pt` is the resumable checkpoint.
- `best.pt` is the validation-selected checkpoint.
- checkpoint/resume boundaries are accumulation boundaries only.
- no mid-accumulation resume state is preserved.
- `best.pt` selection rule:
  1. lowest mean validation NLL over horizons 1-4
  2. tie-break: higher mean validation top-1 over horizons 2-4
  3. second tie-break: earlier step
- `last.pt` must carry enough state to resume:
  - model state
  - optimizer state
  - scheduler/scaler state if used
  - RNG state
  - optimizer step / next step
  - micro step
  - consumed tokens
  - config digest
  - checkpoint timestamp

Token-budget semantics:

- requested training budget is `token_budget_train`,
- realized training volume is `realized_train_tokens`,
- `consumed_tokens` counts all tokens actually processed across microbatches,
- realized tokens may exceed the request because stopping happens only at accumulation boundaries.

## Eval payload contract

`val_future_metrics.json` / `test_future_metrics.json`:

- schema-valid future metrics
- `split` = `val` or `test`
- `checkpoint_used` = `best`
- `checkpoint_path` pointing at the evaluated checkpoint

`val_acceptance_metrics.json` / `test_acceptance_metrics.json`:

- schema-valid acceptance metrics
- `checkpoint_used` = `best`
- `checkpoint_path` pointing at the evaluated checkpoint

`router_metrics.json`:

- schema-valid router metrics
- `checkpoint_used` = `best`
- `checkpoint_path` pointing at the evaluated checkpoint

Unsuffixed legacy eval filenames are non-authoritative.

## Registries

Generate:

```text
outputs/registries/run_registry.csv
outputs/registries/probe_registry.csv
outputs/registries/screening_results.csv
outputs/registries/finalist_selection.json
outputs/registries/main_results.csv
outputs/registries/ablation_results.csv
outputs/paper_assets/paper_asset_manifest.json
```

### `run_registry.csv`

Operational history only. It may include smoke, paused, failed, or aborted runs.

Expected fields include:

- run identity and stage
- status
- best-checkpoint metadata
- interpreter path
- evidence eligibility flag
- authority reason

### `probe_registry.csv`

Probe evidence only. Must exclude smoke, incomplete, archived, or non-authoritative runs.

### `screening_results.csv`

Completed authoritative screening evidence only.

### `main_results.csv`

Completed authoritative final comparison evidence only.

### `ablation_results.csv`

Completed authoritative ablation/confirmatory evidence only.

### `finalist_selection.json`

Authoritative only if derived from the completed required real baseline screening rows and their `best.pt` evaluations.

It should record:

- selected baseline exp id
- selected run dir
- selected best checkpoint path
- selection metric and value
- tie-break trace
- source metrics
- `checkpoint_used`
- `authoritative`
- generated config path
- source screening registry
- creation timestamp

## Paper assets

Paper-facing assets must be built only from authoritative evidence.

```text
outputs/paper_assets/
  figures/
  tables/
  appendix/
    claim_evidence_matrix.md
  paper_asset_manifest.json
```

The claim-evidence matrix must be artifact-driven and must not contain scaffold `pending` placeholders.

## Paper consumption rule

The paper may consume only:

- authoritative scientific registries,
- paper assets derived from those registries,
- and the referenced metric JSONs/traces when needed.

Do not manually copy numbers from logs, aborted runs, smoke runs, or archived outputs.
