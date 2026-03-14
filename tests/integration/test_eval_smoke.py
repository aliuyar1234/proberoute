from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import torch

from tests.helpers import REPO_ROOT, make_fake_run, run_dir_from_output_root, run_python_module, write_json, write_smoke_config


def test_evaluate_smoke_writes_schema_valid_metric_outputs(tmp_path) -> None:
    config_path, _ = write_smoke_config(tmp_path, exp_id="SMOKE_EVAL")
    output_root = tmp_path / "outputs"

    run_python_module("src.cli.prepare_data", "--config", str(config_path))
    run_python_module("src.cli.train_mtp", "--config", str(config_path))

    run_dir = run_dir_from_output_root(output_root, "SMOKE_EVAL")
    run_python_module("src.cli.evaluate", "--run-dir", str(run_dir))

    val_future_path = run_dir / "eval" / "val_future_metrics.json"
    future_path = run_dir / "eval" / "test_future_metrics.json"
    val_acceptance_path = run_dir / "eval" / "val_acceptance_metrics.json"
    acceptance_path = run_dir / "eval" / "test_acceptance_metrics.json"
    router_path = run_dir / "eval" / "router_metrics.json"
    for path in [val_future_path, future_path, val_acceptance_path, acceptance_path, router_path]:
        assert path.exists(), f"missing expected eval output: {path}"

    future = json.loads(future_path.read_text(encoding="utf-8"))
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    router = json.loads(router_path.read_text(encoding="utf-8"))

    future_schema = json.loads((REPO_ROOT / "schemas" / "future_metrics.schema.json").read_text(encoding="utf-8"))
    acceptance_schema = json.loads((REPO_ROOT / "schemas" / "acceptance_metrics.schema.json").read_text(encoding="utf-8"))
    router_schema = json.loads((REPO_ROOT / "schemas" / "router_metrics.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(future, future_schema)
    jsonschema.validate(acceptance, acceptance_schema)
    jsonschema.validate(router, router_schema)

    trace_path = Path(acceptance["trace_path"])
    if not trace_path.is_absolute():
        candidate_paths = [run_dir / trace_path, output_root / trace_path, REPO_ROOT / trace_path]
    else:
        candidate_paths = [trace_path]
    assert any(path.exists() for path in candidate_paths) or (run_dir / "artifacts" / "traces").exists()


def test_evaluate_smoke_supports_probe_initialized_sparse_run(tmp_path) -> None:
    mtp_config_path, _ = write_smoke_config(
        tmp_path,
        exp_id="SMOKE_EVAL_SPARSE_PROBE_INIT",
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
        exp_id="SMOKE_EVAL_PROBE_SOURCE",
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

    run_dir = run_dir_from_output_root(output_root, "SMOKE_EVAL_SPARSE_PROBE_INIT")
    run_python_module("src.cli.evaluate", "--run-dir", str(run_dir))

    for name in [
        "val_future_metrics.json",
        "test_future_metrics.json",
        "val_acceptance_metrics.json",
        "test_acceptance_metrics.json",
        "router_metrics.json",
    ]:
        assert (run_dir / "eval" / name).exists(), f"missing expected eval output: {name}"
