# VALIDATION_AND_TESTING.md

## Test philosophy

Tests exist to prevent silent scientific invalidation, not just runtime breakage. The main things to guard are:

- wrong-token target alignment,
- broken gradient-accumulation semantics,
- cadence drift between microbatch scheduling and optimizer-step logging,
- broken checkpoint/resume semantics,
- incorrect best-checkpoint selection,
- eval/registry leakage from smoke or stale runs,
- invalid finalist selection,
- unsupported paper claims.

## Required unit tests

### U1. Config validation

Malformed configs and invalid stage/mode combinations must fail, including real-run configs that would produce zero nonterminal validation events once gradient accumulation is applied.

### U2. Config inheritance

`inherit_from` must deep-merge dicts and replace lists/scalars deterministically.

### U3. Split assignment stability

Same normalized text must always map to the same split.

### U4. Target alignment

Horizon-shift indexing must be exact for horizons 1-4.

### U5. Router support semantics

Sparse, dense, and last-layer modes must produce the expected support pattern.

### U6. Probe compatibility

`train_mtp` must reject incompatible `probe_init` artifacts.

### U7. Finalist selection

Selection must be deterministic, must exclude the sparse main method from baseline candidates, and must fail if required baseline rows are missing.

### U8. Results schema

Dataset, run, future-metric, acceptance-metric, router-metric, and finalist-selection payloads must validate against `schemas/`.

## Required integration tests

### I1. Offline data-prep smoke

Verify:

- processed `.npy` arrays,
- dataset manifest,
- split-specific eval subsets for val and test.

### I2. Probe smoke

Verify:

- probe artifacts exist,
- `val_metrics.jsonl` is emitted,
- `best.pt` and `last.pt` exist.

### I3. MTP smoke

Verify all three layer-mix modes:

- train loop runs,
- validation log exists,
- checkpoints exist,
- manifest is coherent.

### I4. Eval smoke

Verify:

- split-specific eval files exist,
- payloads validate against schema,
- `checkpoint_used=best` is present.

### I5. Pause/resume smoke

Verify for both `train_probes` and `train_mtp`:

- `request_pause` causes a clean `paused` exit at a checkpoint/validation boundary,
- with `grad_accum > 1`, the pause happens on an accumulation boundary rather than mid-window,
- `checkpoints/last.pt` exists,
- `--resume` continues the same run,
- final status becomes `completed`.

### I6. Accumulation/cadence semantics

Verify:

- train logs use optimizer-step numbering,
- validation/save cadence thresholds count microbatches,
- cadence actions that are crossed mid-window execute at the next accumulation boundary,
- realized token counts reflect the full accumulation-window stop rule.

### I7. Registry/evidence filtering

Verify:

- `run_registry.csv` keeps operational rows,
- scientific registries exclude smoke, failed, paused, aborted, archived, and non-authoritative runs.

## Required contract checks before the next real run

These are blockers before the next authoritative GPU wave:

1. `best.pt` is validation-selected, not final-state aliasing.
2. Real runs validate periodically during training.
3. `grad_accum` is active in both probe and MTP trainers and is reflected in logs, manifests, and checkpoints.
4. `--resume` works from any compatible interrupted `last.pt`.
5. Split-specific val/test eval artifacts are emitted from `best.pt`.
6. Smoke runs are excluded from scientific registries and paper assets.
7. Finalist selection refuses incomplete baseline screening sets.
8. Claim-evidence generation has no scaffold `pending` rows.
9. Root docs and mirrored `docs/` copies are synchronized.

## Acceptance criteria by phase

### Bootstrap

- `.venv` is healthy
- config validation passes
- doc sync path works

### Smoke

- offline path passes
- schemas validate
- pause/resume smoke passes

### Probe

- `PROBE_410M` and `PROBE_1B` complete
- periodic validation recorded
- true best-checkpoint metadata recorded
- accumulation metadata recorded
- probe artifacts emitted

### Screening

- all five mandatory runs complete
- each has authoritative `best.pt` val/test eval artifacts
- `screening_results.csv` contains exactly the authoritative completed rows

### Finalist selection

- the four required baseline screening rows are present
- the generated config is rebuilt from authoritative rows only
- provenance is recorded

### Final comparison

- both final runs complete
- both have authoritative `best.pt` eval artifacts
- `main_results.csv` is rebuilt

### Ablations / confirmatory

- required runs complete or are explicitly waived in `logs/DECISIONS.md`

### Paper

- paper assets exist
- claim-evidence matrix is artifact-driven
- no `pending` claim rows remain
- placeholder scan passes

## Regression policy

If any smoke, schema, or contract test regresses:

1. stop,
2. fix the regression,
3. rerun the smallest failing test,
4. rerun the affected smoke or contract slice,
5. only then continue to real experiments.

## Completion gate

Do not call the project complete until:

- MUST experiments are completed or explicitly waived by rule,
- authoritative scientific registries exist,
- paper assets exist,
- the claim-evidence matrix is artifact-driven,
- `paper/paper_draft.md` exists,
- `FINAL_DELIVERY.md` exists,
- and no unresolved placeholders remain.
