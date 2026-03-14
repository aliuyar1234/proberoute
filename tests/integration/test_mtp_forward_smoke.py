from __future__ import annotations

import json

import pytest
import torch

from tests.helpers import make_fake_run, run_dir_from_output_root, run_python_module, write_json, write_smoke_config


@pytest.mark.parametrize(
    ("layer_mix_mode", "router_init_mode"),
    [
        ("last_layer", "none"),
        ("dense_whs", "random"),
        ("sparse_topm", "random"),
    ],
)
def test_train_mtp_smoke_runs_for_all_layer_mix_modes(tmp_path, layer_mix_mode: str, router_init_mode: str) -> None:
    exp_id = f"SMOKE_{layer_mix_mode.upper()}"
    config_path, _ = write_smoke_config(
        tmp_path,
        exp_id=exp_id,
        overrides={"model": {"layer_mix_mode": layer_mix_mode, "router_init_mode": router_init_mode}},
    )
    output_root = tmp_path / "outputs"

    run_python_module("src.cli.prepare_data", "--config", str(config_path))
    run_python_module("src.cli.train_mtp", "--config", str(config_path))

    run_dir = run_dir_from_output_root(output_root, exp_id)
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "resolved_config.yaml").exists()
    assert (run_dir / "train_metrics.jsonl").exists()
    assert (run_dir / "val_metrics.jsonl").exists()

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_paths = list(checkpoint_dir.glob("best.*")) + list(checkpoint_dir.glob("last.*"))
    assert checkpoint_paths, f"expected smoke checkpoints in {checkpoint_dir}"

    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["layer_mix_mode"] == layer_mix_mode


def test_train_mtp_smoke_runs_for_last_layer_linear_head_baseline(tmp_path) -> None:
    config_path, _ = write_smoke_config(
        tmp_path,
        exp_id="SMOKE_LAST_LAYER_LINEAR",
        overrides={
            "model": {
                "layer_mix_mode": "last_layer",
                "router_init_mode": "none",
                "head_type": "linear",
                "top_m": 1,
            }
        },
    )
    output_root = tmp_path / "outputs"

    run_python_module("src.cli.prepare_data", "--config", str(config_path))
    run_python_module("src.cli.train_mtp", "--config", str(config_path))

    run_dir = run_dir_from_output_root(output_root, "SMOKE_LAST_LAYER_LINEAR")
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "train_metrics.jsonl").exists()
    assert (run_dir / "val_metrics.jsonl").exists()

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_paths = list(checkpoint_dir.glob("best.*")) + list(checkpoint_dir.glob("last.*"))
    assert checkpoint_paths, f"expected smoke checkpoints in {checkpoint_dir}"

    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["layer_mix_mode"] == "last_layer"


def test_train_mtp_smoke_runs_for_sparse_probe_initialized_router(tmp_path) -> None:
    mtp_config_path, _ = write_smoke_config(
        tmp_path,
        exp_id="SMOKE_SPARSE_PROBE_INIT",
        overrides={
            "model": {
                "layer_mix_mode": "sparse_topm",
                "router_init_mode": "probe_zscore_top5",
                "top_m": 2,
            }
        },
    )
    output_root = tmp_path / "outputs"
    probe_run_dir = make_fake_run(
        output_root,
        exp_id="SMOKE_PROBE_INIT_SOURCE",
        stage="probe",
        model_id="local-toy-gpt",
        with_probe=True,
    )
    probe_dir = probe_run_dir / "artifacts" / "probe"
    write_json(
        probe_dir / "probe_init.json",
        {
            "model_id": "local-toy-gpt",
            "backbone_name": "local_toy_gpt",
            "dataset_name": "local_fixture_text",
            "dataset_config": "tiny_v1",
            "dataset_id": "local_fixture_text__tiny_v1__local_toy_whitespace__sl32",
            "seq_len": 32,
            "horizons": [1, 2, 3, 4],
            "num_layers": 2,
            "metric": "top5",
            "init_metric": "top5",
        },
    )
    torch.save({"scores": torch.zeros((4, 2), dtype=torch.float32)}, probe_dir / "probe_init.pt")

    run_python_module("src.cli.prepare_data", "--config", str(mtp_config_path))
    run_python_module("src.cli.train_mtp", "--config", str(mtp_config_path))

    run_dir = run_dir_from_output_root(output_root, "SMOKE_SPARSE_PROBE_INIT")
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "train_metrics.jsonl").exists()
    assert (run_dir / "val_metrics.jsonl").exists()

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_paths = list(checkpoint_dir.glob("best.*")) + list(checkpoint_dir.glob("last.*"))
    assert checkpoint_paths, f"expected smoke checkpoints in {checkpoint_dir}"

    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["layer_mix_mode"] == "sparse_topm"
    assert run_manifest["router_init_mode"] == "probe_zscore_top5"
