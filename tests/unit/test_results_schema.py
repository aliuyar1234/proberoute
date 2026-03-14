from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from tests.helpers import SCHEMA_DIR


def _load_schema(name: str) -> dict:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def test_dataset_manifest_schema_accepts_minimal_example() -> None:
    schema = _load_schema("dataset_manifest.schema.json")
    payload = {
        "dataset_name": "local_fixture_text",
        "dataset_config": "tiny_v1",
        "tokenizer_name": "local_toy_whitespace",
        "tokenizer_revision": None,
        "dataset_id": "local_fixture_text__tiny_v1__local_toy_whitespace__sl32",
        "seq_len": 32,
        "split_policy": {"split_modulus": 1000},
        "normalization_policy": {"newline": "lf", "strip_nul": True},
        "token_counts_by_split": {"train": 96, "val": 32, "test": 32},
        "sequence_counts_by_split": {"train": 3, "val": 1, "test": 1},
        "requested_token_quotas_by_split": {"train": 2048, "val": 512, "test": 512},
        "realized_token_counts_by_split": {"train": 96, "val": 32, "test": 32},
        "dropped_tail_tokens_by_split": {"train": 0, "val": 0, "test": 0},
        "eval_future_sequence_count": 1,
        "eval_acceptance_prefix_count": 1,
        "creation_timestamp": "2026-03-12T00:00:00Z",
    }
    jsonschema.validate(payload, schema)


def test_run_and_eval_schemas_accept_minimal_smoke_examples() -> None:
    run_schema = _load_schema("run_manifest.schema.json")
    future_schema = _load_schema("future_metrics.schema.json")
    acceptance_schema = _load_schema("acceptance_metrics.schema.json")
    router_schema = _load_schema("router_metrics.schema.json")

    run_manifest = {
        "exp_id": "SMOKE_LOCAL_TINY",
        "stage": "smoke",
        "priority": "must",
        "model_id": "local-toy-gpt",
        "seed": 1337,
        "backbone_name": "local_toy_gpt",
        "layer_mix_mode": "sparse_topm",
        "router_init_mode": "random",
        "dataset_name": "local_fixture_text",
        "dataset_config": "tiny_v1",
        "dataset_id": "local_fixture_text__tiny_v1__local_toy_whitespace__sl32",
        "seq_len": 32,
        "token_budget_train": 2048,
        "token_budget_val": 512,
        "token_budget_test": 512,
        "micro_batch_size": 4,
        "grad_accum": 1,
        "nominal_effective_batch_sequences": 4,
        "nominal_tokens_per_optimizer_update": 128,
        "start_time": "2026-03-12T00:00:00Z",
        "optimizer_step": 8,
        "micro_step": 8,
        "consumed_tokens": 1024,
        "realized_train_tokens": 1024,
        "last_checkpoint_path": "outputs/runs/SMOKE_LOCAL_TINY/local-toy-gpt/seed_1337/checkpoints/last.pt",
        "last_checkpoint_step": 8,
        "last_checkpoint_micro_step": 8,
        "status": "completed",
    }
    future_metrics = {
        "exp_id": "SMOKE_LOCAL_TINY",
        "split": "test",
        "num_sequences": 1,
        "horizons": [1, 2, 3, 4],
        "metrics_by_horizon": {"1": {"top1": 1.0, "top5": 1.0, "nll": 0.1}},
        "aggregate_metrics": {"mean_top1_h1_h4": 1.0, "mean_top1_h2_h4": 1.0, "mean_top5_h1_h4": 1.0, "mean_nll_h1_h4": 0.1},
        "bootstrap_ci": None,
        "evaluation_seed": 2026,
    }
    acceptance_metrics = {
        "exp_id": "SMOKE_LOCAL_TINY",
        "num_prefixes": 1,
        "max_horizon": 4,
        "max_new_tokens": 8,
        "prefix_len": 16,
        "greedy_policy": "argmax",
        "advance_policy": "append_one_base_greedy_token",
        "mean_accept_len": 1.0,
        "accept_rate_depth_1": 1.0,
        "accept_rate_depth_2": 0.0,
        "accept_rate_depth_3": 0.0,
        "accept_rate_depth_4": 0.0,
        "accept_len_histogram": {"1": 1},
        "bootstrap_ci": None,
        "evaluation_seed": 2026,
        "trace_path": "outputs/runs/SMOKE_LOCAL_TINY/local-toy-gpt/seed_1337/artifacts/traces/acceptance_trace.jsonl",
    }
    router_metrics = {
        "exp_id": "SMOKE_LOCAL_TINY",
        "layer_mix_mode": "sparse_topm",
        "horizons": [1, 2, 3, 4],
        "top_m": 2,
        "entropy_by_horizon": {"1": 0.5},
        "selected_layers_by_horizon": {"1": [0, 1]},
        "average_weights_by_horizon": {"1": [0.5, 0.5]},
        "overlap_with_probe_topm_by_horizon": None,
    }

    jsonschema.validate(run_manifest, run_schema)
    jsonschema.validate(future_metrics, future_schema)
    jsonschema.validate(acceptance_metrics, acceptance_schema)
    jsonschema.validate(router_metrics, router_schema)


def test_run_manifest_schema_accepts_paused_status() -> None:
    run_schema = _load_schema("run_manifest.schema.json")
    run_manifest = {
        "exp_id": "SMOKE_LOCAL_TINY",
        "stage": "smoke",
        "priority": "must",
        "model_id": "local-toy-gpt",
        "seed": 1337,
        "backbone_name": "local_toy_gpt",
        "layer_mix_mode": "sparse_topm",
        "router_init_mode": "random",
        "dataset_name": "local_fixture_text",
        "dataset_config": "tiny_v1",
        "dataset_id": "local_fixture_text__tiny_v1__local_toy_whitespace__sl32",
        "seq_len": 32,
        "token_budget_train": 2048,
        "token_budget_val": 512,
        "token_budget_test": 512,
        "micro_batch_size": 4,
        "grad_accum": 1,
        "nominal_effective_batch_sequences": 4,
        "nominal_tokens_per_optimizer_update": 128,
        "start_time": "2026-03-12T00:00:00Z",
        "end_time": "2026-03-12T00:01:00Z",
        "optimizer_step": 1,
        "micro_step": 1,
        "consumed_tokens": 128,
        "realized_train_tokens": 128,
        "last_checkpoint_step": 1,
        "last_checkpoint_micro_step": 1,
        "status": "paused",
    }
    jsonschema.validate(run_manifest, run_schema)
