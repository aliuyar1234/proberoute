# MODULE_SPECS/07_TRAINING_ORCHESTRATION.md

## Objective

Define the authoritative training, checkpointing, validation, and run-control contract after the GPT Pro adjudication pass.

## Required behavior

### 1. Run initialization

Before training:

- create the run directory,
- save `resolved_config.yaml`,
- save `environment_snapshot.json`,
- create `run_manifest.json` with status `running`.

Fresh non-resume launches into an existing run directory are invalid unless an explicit force/reset path is used.

### 2. Logging

`train_metrics.jsonl` must append:

- step
- micro_step
- consumed tokens
- train loss
- per-horizon train losses
- learning rates
- timestamp

`val_metrics.jsonl` must append at every validation event:

- step
- micro_step
- consumed tokens
- `mean_val_nll_h1_h4`
- `mean_val_top1_h2_h4`
- per-horizon val metrics
- timestamp

Semantics:

- `step` means optimizer update.
- `micro_step` means processed microbatch count.
- `consumed_tokens` counts all tokens actually processed across microbatches.

### 3. Validation cadence

Real runs must validate periodically during training.

Use:

- `train.val_every_steps` if present,
- otherwise `train.eval_every_steps`,
- otherwise `train.save_every_steps`.

These cadence fields count microbatches, not optimizer updates.
If a cadence threshold is crossed mid-accumulation window, execute the action at the next accumulation boundary.

The final step must always trigger validation.

### 4. Checkpointing

At checkpoint/validation boundaries, write `checkpoints/last.pt`.

Checkpoint and pause/resume boundaries are accumulation boundaries only.
No mid-accumulation resume state is preserved.

`last.pt` must be resumable and include:

- model state
- optimizer state
- scheduler/scaler state if used
- RNG state
- next step / optimizer step
- micro step
- consumed tokens
- config digest
- checkpoint timestamp

`best.pt` is the validation-selected checkpoint.

Selection rule:

1. lowest mean validation NLL over horizons 1-4
2. tie-break: higher mean validation top-1 over horizons 2-4
3. second tie-break: earlier step

The run manifest must record:

- `best_step`
- `best_checkpoint_path`
- `best_mean_val_nll_h1_h4`
- `best_mean_val_top1_h2_h4`

### 5. Pause / resume

`python -m src.cli.request_pause --run-dir ...` must only create a pause marker.

The trainer must honor that marker at the next checkpoint/validation boundary by:

- writing `last.pt`,
- updating the manifest status to `paused`,
- exiting cleanly.

`--resume` must:

- require an existing compatible `checkpoints/last.pt`,
- refuse to silently start a fresh run,
- restore step/token state,
- restore micro-step state,
- restore RNG state,
- append logs instead of overwriting,
- increment `resume_count`,
- set status back to `running`.

`--resume` is allowed for compatible interrupted runs in states such as:

- `paused`
- `failed`
- `aborted`
- stale `running`

### 6. Status semantics

- `running`: active trainer owns the run directory
- `paused`: clean pause completed at a boundary
- `completed`: training finished cleanly
- `failed`: exception or contract failure
- `aborted`: manual stop outside the clean pause contract

### 7. Evaluation handoff

Non-probe scientific runs are not evidence-ready after training alone. They become authoritative only after evaluation from `best.pt` writes:

- `eval/val_future_metrics.json`
- `eval/test_future_metrics.json`
- `eval/val_acceptance_metrics.json`
- `eval/test_acceptance_metrics.json`
- `eval/router_metrics.json`

Those eval payloads must record `checkpoint_used=best`.

## Minimal training-loop pseudocode

```python
for optimizer_step in range(start_step, total_optimizer_steps):
    zero_grad()
    for _ in range(grad_accum):
        train_one_microbatch(loss_scale=1 / grad_accum)
        micro_step += 1
        consumed_tokens += tokens_in_microbatch
        mark_save_due_if_microbatch_threshold_crossed()
        mark_val_due_if_microbatch_threshold_crossed()

    optimizer.step()
    append_train_metrics(step=optimizer_step + 1, micro_step=micro_step)

    if is_final_step or val_due:
        validation_record = run_validation(step=optimizer_step + 1, micro_step=micro_step)
        save_last_checkpoint(step=optimizer_step + 1, micro_step=micro_step)
        maybe_update_best(validation_record)
    elif save_due:
        save_last_checkpoint(step=optimizer_step + 1, micro_step=micro_step)

    if pause_request_exists(run_dir):
        ensure_boundary_checkpoint_exists()
        finalize_manifest(status="paused")
        return
```

## Finalization

At the end of training:

- finalize status as `completed`, `paused`, `failed`, or `aborted`
- ensure `last.pt` and best-checkpoint metadata are coherent
- record requested vs realized training tokens
- write any training summary artifacts
- leave non-probe runs ready for the separate `evaluate` step
