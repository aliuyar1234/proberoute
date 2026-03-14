from __future__ import annotations

import csv
import subprocess

import pytest

from tests.helpers import future_metrics_payload, make_fake_run, run_python_module, write_json


def test_collect_results_orders_rows_deterministically_and_emits_probe_registry(tmp_path) -> None:
    output_root = tmp_path / "outputs"
    make_fake_run(output_root, exp_id="Z_RUN", stage="screen", model_id="model-b", seed=2, with_probe=True)
    make_fake_run(output_root, exp_id="PROBE_RUN", stage="probe", model_id="model-p", seed=5, with_probe=True)
    make_fake_run(output_root, exp_id="A_RUN", stage="final", model_id="model-a", seed=1)
    make_fake_run(output_root, exp_id="A_RUN", stage="screen", model_id="model-a", seed=1)
    make_fake_run(output_root, exp_id="SMOKE_RUN", stage="smoke", model_id="model-s", seed=3)
    make_fake_run(output_root, exp_id="FAILED_RUN", stage="screen", model_id="model-f", seed=4, status="failed")

    run_python_module("src.cli.collect_results", "--outputs-root", str(output_root))

    run_registry = list(csv.DictReader((output_root / "registries" / "run_registry.csv").open(encoding="utf-8")))
    assert run_registry, "run registry should not be empty"
    observed_exp_ids = [row["exp_id"] for row in run_registry]
    assert observed_exp_ids == ["PROBE_RUN", "A_RUN", "FAILED_RUN", "Z_RUN", "A_RUN", "SMOKE_RUN"], "rows should be ordered deterministically by stage then exp_id"
    assert any(row["status"] == "failed" for row in run_registry)
    assert any(row["stage"] == "smoke" for row in run_registry)

    screening_registry = list(csv.DictReader((output_root / "registries" / "screening_results.csv").open(encoding="utf-8")))
    screening_exp_ids = [row["exp_id"] for row in screening_registry]
    assert screening_exp_ids == ["A_RUN", "Z_RUN"], "only authoritative completed non-smoke screening rows belong in screening_results"

    probe_registry = list(csv.DictReader((output_root / "registries" / "probe_registry.csv").open(encoding="utf-8")))
    assert len(probe_registry) == 1
    assert probe_registry[0]["exp_id"] == "PROBE_RUN"
    assert probe_registry[0]["probe_scores_path"].endswith("probe_scores.csv")


def test_collect_results_rejects_schema_invalid_metric_payloads(tmp_path) -> None:
    output_root = tmp_path / "outputs"
    run_dir = make_fake_run(output_root, exp_id="BAD_METRIC", stage="screen", model_id="model-a", seed=1)
    invalid_future = future_metrics_payload(exp_id="BAD_METRIC", split="val", mean_top1_h2_h4=0.2, mean_nll_h1_h4=1.0)
    invalid_future.pop("aggregate_metrics")
    write_json(run_dir / "eval" / "val_future_metrics.json", invalid_future)

    with pytest.raises(subprocess.CalledProcessError):
        run_python_module("src.cli.collect_results", "--outputs-root", str(output_root))
