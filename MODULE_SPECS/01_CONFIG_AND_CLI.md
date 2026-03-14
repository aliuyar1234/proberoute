# MODULE_SPECS/01_CONFIG_AND_CLI.md

## Objective

Define the config contract and CLI interface so every later module plugs into a stable execution surface.

## Canonical config contract

Implement a structured config object with at least these sections:

```yaml
project:
  name:
  exp_id:
  stage:
  priority:
  seed:
  output_root:
  notes:

hardware:
  device:
  precision:
  expected_vram_gb:
  allow_flash_attn:
  grad_accum:            # real accumulation factor; scientific update-batch control
  micro_batch_size:      # per-microbatch sequence count

data:
  dataset_name:
  dataset_config:
  local_path:
  tokenizer_name:
  text_field_priority:
  split_modulus:
  split_ranges:
  seq_len:
  train_token_quota:
  val_token_quota:
  test_token_quota:
  processed_root:
  append_eos_between_docs:
  drop_last_sequence:

model:
  backbone_name:
  freeze_backbone:
  horizons:
  layer_mix_mode:
  router_init_mode:
  probe_init_metric:
  head_type:
  top_m:
  router_temperature:
  probe_rank:
  hidden_expansion:
  hidden_norm:
  entropy_penalty_beta:
  use_deephead_for_far_horizons:
  deephead_horizons:

train:
  optimizer:
  stop_mode:
  lr_head:
  lr_router:
  lr_probe:
  weight_decay:
  warmup_ratio:
  max_steps:
  eval_every_steps:      # cadence in microbatches; executes at next accumulation boundary
  save_every_steps:      # cadence in microbatches; executes at next accumulation boundary
  log_every_steps:
  early_stop_patience:
  grad_clip_norm:
  loss_weights:
  loss_warmup:

eval:
  future_metrics_sequence_count:
  acceptance_prefix_count:
  acceptance_prefix_len:
  acceptance_max_new_tokens:
  bootstrap_samples:
  bootstrap_seed:
```

## Config merge semantics

`inherit_from` resolution rules:
- recursively load the parent first,
- deep-merge dictionaries,
- replace lists completely,
- replace scalar values completely,
- child values always override parent values.

Invalid parent path or cyclic inheritance -> hard fail.

## Config invariants

- `horizons` must be sorted positive integers.
- `max(horizons) < seq_len`.
- `layer_mix_mode ∈ {last_layer, dense_whs, sparse_topm}` for MTP runs.
- `router_init_mode ∈ {none, random, probe_zscore_top5}` for MTP runs.
- `top_m <= num_layers` for sparse mode.
- `top_m == 1` is not sufficient to imply last-layer mode; the mode must be explicit.
- `probe_rank > 0`.
- `micro_batch_size > 0`.
- `grad_accum > 0`.
- token quotas must be positive.
- `hidden_norm == stateless_layer_norm` in the canonical track.
- bootstrap samples must be >= 100 when enabled.
- real non-smoke runs must yield at least one nonterminal validation event after accumulation-boundary scheduling is applied.

## CLI entrypoints

Implement CLIs:
- `prepare_data`
- `train_probes`
- `train_mtp`
- `evaluate`
- `collect_results`
- `select_finalist`
- `build_paper_assets`
- `write_paper`

## CLI behavior rules

- Every CLI saves the resolved config when a config is used.
- Every CLI supports `--dry-run` where practical.
- All CLIs fail with informative errors.
- No CLI silently overwrites completed runs unless `--force` is set.
- `train_probes` and `train_mtp` must treat scientific `step` as optimizer update, not raw microbatch iteration.
- `train_mtp` must verify probe-init compatibility before loading.
- `select_finalist` must emit both JSON selection metadata and a generated YAML config.
- `write_paper` must fail the final pass if placeholders remain.
