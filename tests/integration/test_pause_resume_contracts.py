from __future__ import annotations

import torch

from tests.helpers import load_json, run_python_module, write_smoke_config


def _expected_smoke_run_dir(tmp_path, exp_id: str):
    return tmp_path / "outputs" / "runs" / exp_id / "local-toy-gpt" / "seed_1337"


def _accumulation_pause_overrides() -> dict:
    return {
        "hardware": {"micro_batch_size": 1, "grad_accum": 3},
        "data": {"train_token_quota": 160, "val_token_quota": 64, "test_token_quota": 64},
        "train": {"eval_every_steps": 10, "save_every_steps": 10},
    }


def test_probe_training_can_pause_and_resume_on_smoke(tmp_path) -> None:
    exp_id = "SMOKE_PROBE_RESUME"
    config_path, _ = write_smoke_config(tmp_path, exp_id=exp_id, overrides=_accumulation_pause_overrides())
    run_dir = _expected_smoke_run_dir(tmp_path, exp_id)

    run_python_module("src.cli.prepare_data", "--config", str(config_path))
    run_python_module("src.cli.request_pause", "--run-dir", str(run_dir))
    run_python_module("src.cli.train_probes", "--config", str(config_path))

    paused_manifest = load_json(run_dir / "run_manifest.json")
    assert paused_manifest["status"] == "paused"
    assert (run_dir / "checkpoints" / "last.pt").exists()
    assert (run_dir / "artifacts" / "probe" / "probe_scores.csv").exists() is False
    assert paused_manifest["optimizer_step"] == 1
    assert paused_manifest["micro_step"] == 3

    paused_checkpoint = torch.load(run_dir / "checkpoints" / "last.pt", map_location="cpu")
    assert paused_checkpoint["next_step"] == 1
    assert paused_checkpoint["micro_step"] == 3
    assert paused_checkpoint["micro_step"] == paused_checkpoint["next_step"] * 3

    run_python_module("src.cli.train_probes", "--config", str(config_path), "--resume")

    resumed_manifest = load_json(run_dir / "run_manifest.json")
    assert resumed_manifest["status"] == "completed"
    assert resumed_manifest["optimizer_step"] == 2
    assert resumed_manifest["micro_step"] == 6
    assert (run_dir / "artifacts" / "probe" / "probe_scores.csv").exists()
    assert (run_dir / "artifacts" / "probe" / "probe_init.json").exists()


def test_mtp_training_can_pause_and_resume_on_smoke(tmp_path) -> None:
    exp_id = "SMOKE_MTP_RESUME"
    config_path, _ = write_smoke_config(tmp_path, exp_id=exp_id, overrides=_accumulation_pause_overrides())
    run_dir = _expected_smoke_run_dir(tmp_path, exp_id)

    run_python_module("src.cli.prepare_data", "--config", str(config_path))
    run_python_module("src.cli.request_pause", "--run-dir", str(run_dir))
    run_python_module("src.cli.train_mtp", "--config", str(config_path))

    paused_manifest = load_json(run_dir / "run_manifest.json")
    assert paused_manifest["status"] == "paused"
    assert paused_manifest["optimizer_step"] == 1
    assert paused_manifest["micro_step"] == 3

    paused_checkpoint = torch.load(run_dir / "checkpoints" / "last.pt", map_location="cpu")
    assert paused_checkpoint["next_step"] == 1
    assert paused_checkpoint["micro_step"] == 3
    assert paused_checkpoint["micro_step"] == paused_checkpoint["next_step"] * 3
    assert sorted(paused_checkpoint["model_state"].keys()) == ["heads", "router"]

    run_python_module("src.cli.train_mtp", "--config", str(config_path), "--resume")
    run_python_module("src.cli.evaluate", "--run-dir", str(run_dir))

    resumed_manifest = load_json(run_dir / "run_manifest.json")
    assert resumed_manifest["status"] == "completed"
    assert resumed_manifest["optimizer_step"] == 2
    assert resumed_manifest["micro_step"] == 6
    assert (run_dir / "training_summary.json").exists()
    assert (run_dir / "eval" / "test_future_metrics.json").exists()
