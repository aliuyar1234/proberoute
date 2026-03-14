from __future__ import annotations

import json

from tests.helpers import run_dir_from_output_root, run_python_module, write_smoke_config


def test_train_probes_smoke_emits_probe_artifacts(tmp_path) -> None:
    config_path, _ = write_smoke_config(tmp_path, exp_id="SMOKE_PROBE")
    output_root = tmp_path / "outputs"

    run_python_module("src.cli.prepare_data", "--config", str(config_path))
    run_python_module("src.cli.train_probes", "--config", str(config_path))

    run_dir = run_dir_from_output_root(output_root, "SMOKE_PROBE")
    artifact_dir = run_dir / "artifacts" / "probe"

    expected = [
        artifact_dir / "probe_scores.csv",
        artifact_dir / "probe_heatmap_top1.png",
        artifact_dir / "probe_heatmap_top5.png",
        artifact_dir / "probe_heatmap_nll.png",
        artifact_dir / "probe_init.json",
        artifact_dir / "probe_init.pt",
        run_dir / "run_manifest.json",
        run_dir / "resolved_config.yaml",
    ]
    for path in expected:
        assert path.exists(), f"missing expected probe artifact: {path}"

    probe_init = json.loads((artifact_dir / "probe_init.json").read_text(encoding="utf-8"))
    for field in ["exp_id", "backbone_name", "dataset_id", "seq_len", "horizons", "init_metric", "seed"]:
        assert field in probe_init
