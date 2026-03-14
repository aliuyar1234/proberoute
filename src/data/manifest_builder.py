from __future__ import annotations

from typing import Any

from src.core.manifest import build_dataset_id, utc_now


def build_dataset_manifest(
    config: dict[str, Any],
    *,
    token_counts_by_split: dict[str, int],
    sequence_counts_by_split: dict[str, int],
    realized_token_counts_by_split: dict[str, int],
    dropped_tail_tokens_by_split: dict[str, int],
    eval_future_sequence_count: int,
    eval_acceptance_prefix_count: int,
    eval_future_sequence_count_by_split: dict[str, int] | None = None,
    eval_acceptance_prefix_count_by_split: dict[str, int] | None = None,
    smoke_sequence_rebalance_applied: bool = False,
) -> dict[str, Any]:
    data = config["data"]
    return {
        "dataset_name": data["dataset_name"],
        "dataset_config": data["dataset_config"],
        "tokenizer_name": data["tokenizer_name"],
        "tokenizer_revision": None,
        "local_path": data["local_path"],
        "dataset_id": build_dataset_id(config),
        "seq_len": data["seq_len"],
        "split_policy": {
            "modulus": data["split_modulus"],
            "ranges": data["split_ranges"],
            "smoke_sequence_rebalance_applied": smoke_sequence_rebalance_applied,
        },
        "normalization_policy": {
            "newline_normalization": True,
            "strip_nul_bytes": True,
            "trim_for_emptiness_check_only": True,
        },
        "token_counts_by_split": token_counts_by_split,
        "sequence_counts_by_split": sequence_counts_by_split,
        "requested_token_quotas_by_split": {
            "train": data["train_token_quota"],
            "val": data["val_token_quota"],
            "test": data["test_token_quota"],
        },
        "realized_token_counts_by_split": realized_token_counts_by_split,
        "dropped_tail_tokens_by_split": dropped_tail_tokens_by_split,
        "eval_future_sequence_count": eval_future_sequence_count,
        "eval_acceptance_prefix_count": eval_acceptance_prefix_count,
        "eval_future_sequence_count_by_split": eval_future_sequence_count_by_split or {},
        "eval_acceptance_prefix_count_by_split": eval_acceptance_prefix_count_by_split or {},
        "creation_timestamp": utc_now(),
    }
