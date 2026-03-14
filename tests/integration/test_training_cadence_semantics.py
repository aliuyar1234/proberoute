from __future__ import annotations

import json

import torch

from tests.helpers import load_json, run_dir_from_output_root, run_python_module, write_smoke_config


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_probe_training_uses_optimizer_steps_and_microbatch_cadence(tmp_path) -> None:
    exp_id = "SMOKE_PROBE_ACCUM"
    config_path, _ = write_smoke_config(
        tmp_path,
        exp_id=exp_id,
        overrides={
            "hardware": {"micro_batch_size": 1, "grad_accum": 2},
            "data": {"train_token_quota": 160, "val_token_quota": 64, "test_token_quota": 64},
            "train": {"eval_every_steps": 3, "save_every_steps": 5},
        },
    )
    output_root = tmp_path / "outputs"

    run_python_module("src.cli.prepare_data", "--config", str(config_path))
    run_python_module("src.cli.train_probes", "--config", str(config_path))

    run_dir = run_dir_from_output_root(output_root, exp_id)
    train_rows = _read_jsonl(run_dir / "train_metrics.jsonl")
    val_rows = _read_jsonl(run_dir / "val_metrics.jsonl")
    manifest = load_json(run_dir / "run_manifest.json")
    checkpoint = torch.load(run_dir / "checkpoints" / "last.pt", map_location="cpu")

    assert [row["step"] for row in train_rows] == [1, 2, 3]
    assert [row["micro_step"] for row in train_rows] == [2, 4, 6]
    assert [row["consumed_tokens"] for row in train_rows] == [64, 128, 192]

    assert [row["step"] for row in val_rows] == [2, 3]
    assert [row["micro_step"] for row in val_rows] == [4, 6]
    assert [row["consumed_tokens"] for row in val_rows] == [128, 192]

    assert manifest["micro_batch_size"] == 1
    assert manifest["grad_accum"] == 2
    assert manifest["nominal_effective_batch_sequences"] == 2
    assert manifest["nominal_tokens_per_optimizer_update"] == 64
    assert manifest["optimizer_step"] == 3
    assert manifest["micro_step"] == 6
    assert manifest["consumed_tokens"] == 192
    assert manifest["realized_train_tokens"] == 192
    assert manifest["last_checkpoint_step"] == 3
    assert manifest["last_checkpoint_micro_step"] == 6

    assert checkpoint["optimizer_step"] == 3
    assert checkpoint["next_step"] == 3
    assert checkpoint["micro_step"] == 6
    assert checkpoint["consumed_tokens"] == 192
