from __future__ import annotations

import csv
from pathlib import Path

from tests.helpers import load_json, load_yaml, make_fake_run, run_python_module


def test_select_finalist_applies_tie_breaks_and_emits_final_budget_config(tmp_path) -> None:
    output_root = tmp_path / "outputs"
    registries = output_root / "registries"
    registries.mkdir(parents=True, exist_ok=True)
    config_root = tmp_path / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    for name in [
        "screen_dense_whs_probe_init_1b.yaml",
        "screen_last_mlp_1b.yaml",
        "screen_dense_whs_random_1b.yaml",
        "screen_last_linear_1b.yaml",
        "screen_sparse_probe_init_1b.yaml",
    ]:
        (config_root / name).write_text("project: {}\n", encoding="utf-8")

    run_a = make_fake_run(
        output_root,
        exp_id="BASE_LAST_MLP_1B",
        stage="screen",
        model_id="model-a",
        val_mean_top1_h2_h4=0.50,
        val_mean_accept_len=1.0,
        val_mean_nll_h1_h4=1.3,
        layer_mix_mode="last_layer",
        router_init_mode="none",
    )
    run_b = make_fake_run(
        output_root,
        exp_id="BASE_DENSE_WHS_PROBE_1B",
        stage="screen",
        model_id="model-b",
        val_mean_top1_h2_h4=0.50,
        val_mean_accept_len=1.0,
        val_mean_nll_h1_h4=1.2,
        layer_mix_mode="dense_whs",
        router_init_mode="probe_zscore_top5",
    )
    run_c = make_fake_run(
        output_root,
        exp_id="BASE_DENSE_WHS_RANDOM_1B",
        stage="screen",
        model_id="model-d",
        val_mean_top1_h2_h4=0.40,
        val_mean_accept_len=0.8,
        val_mean_nll_h1_h4=1.4,
        layer_mix_mode="dense_whs",
        router_init_mode="random",
    )
    run_d = make_fake_run(
        output_root,
        exp_id="BASE_LAST_LINEAR_1B",
        stage="screen",
        model_id="model-e",
        val_mean_top1_h2_h4=0.35,
        val_mean_accept_len=0.7,
        val_mean_nll_h1_h4=1.5,
        layer_mix_mode="last_layer",
        router_init_mode="none",
    )
    run_main = make_fake_run(
        output_root,
        exp_id="MAIN_SPARSE_PROBE_1B",
        stage="screen",
        model_id="model-c",
        val_mean_top1_h2_h4=0.99,
        val_mean_accept_len=9.9,
        val_mean_nll_h1_h4=0.1,
        layer_mix_mode="sparse_topm",
        router_init_mode="probe_zscore_top5",
    )

    rows = [
        {
            "exp_id": "BASE_DENSE_WHS_PROBE_1B",
            "stage": "screen",
            "priority": "must",
            "run_dir": str(run_b),
            "layer_mix_mode": "dense_whs",
            "status": "completed",
            "source_config_path": str(config_root / "screen_dense_whs_probe_init_1b.yaml"),
            "val_mean_top1_h2_h4": "0.50",
            "val_mean_accept_len": "1.0",
            "val_mean_nll_h1_h4": "1.2",
            "test_mean_top1_h2_h4": "0.50",
            "test_mean_accept_len": "1.0",
            "test_mean_nll_h1_h4": "1.2",
            "best_checkpoint_path": str(run_b / "checkpoints" / "best.pt"),
            "checkpoint_used": "best",
        },
        {
            "exp_id": "BASE_LAST_MLP_1B",
            "stage": "screen",
            "priority": "must",
            "run_dir": str(run_a),
            "layer_mix_mode": "last_layer",
            "status": "completed",
            "source_config_path": str(config_root / "screen_last_mlp_1b.yaml"),
            "val_mean_top1_h2_h4": "0.50",
            "val_mean_accept_len": "1.0",
            "val_mean_nll_h1_h4": "1.3",
            "test_mean_top1_h2_h4": "0.50",
            "test_mean_accept_len": "1.0",
            "test_mean_nll_h1_h4": "1.3",
            "best_checkpoint_path": str(run_a / "checkpoints" / "best.pt"),
            "checkpoint_used": "best",
        },
        {
            "exp_id": "BASE_DENSE_WHS_RANDOM_1B",
            "stage": "screen",
            "priority": "must",
            "run_dir": str(run_c),
            "layer_mix_mode": "dense_whs",
            "status": "completed",
            "source_config_path": str(config_root / "screen_dense_whs_random_1b.yaml"),
            "val_mean_top1_h2_h4": "0.40",
            "val_mean_accept_len": "0.8",
            "val_mean_nll_h1_h4": "1.4",
            "test_mean_top1_h2_h4": "0.40",
            "test_mean_accept_len": "0.8",
            "test_mean_nll_h1_h4": "1.4",
            "best_checkpoint_path": str(run_c / "checkpoints" / "best.pt"),
            "checkpoint_used": "best",
        },
        {
            "exp_id": "BASE_LAST_LINEAR_1B",
            "stage": "screen",
            "priority": "must",
            "run_dir": str(run_d),
            "layer_mix_mode": "last_layer",
            "status": "completed",
            "source_config_path": str(config_root / "screen_last_linear_1b.yaml"),
            "val_mean_top1_h2_h4": "0.35",
            "val_mean_accept_len": "0.7",
            "val_mean_nll_h1_h4": "1.5",
            "test_mean_top1_h2_h4": "0.35",
            "test_mean_accept_len": "0.7",
            "test_mean_nll_h1_h4": "1.5",
            "best_checkpoint_path": str(run_d / "checkpoints" / "best.pt"),
            "checkpoint_used": "best",
        },
        {
            "exp_id": "MAIN_SPARSE_PROBE_1B",
            "stage": "screen",
            "priority": "must",
            "run_dir": str(run_main),
            "layer_mix_mode": "sparse_topm",
            "status": "completed",
            "source_config_path": str(config_root / "screen_sparse_probe_init_1b.yaml"),
            "val_mean_top1_h2_h4": "0.99",
            "val_mean_accept_len": "9.9",
            "val_mean_nll_h1_h4": "0.1",
            "test_mean_top1_h2_h4": "0.99",
            "test_mean_accept_len": "9.9",
            "test_mean_nll_h1_h4": "0.1",
            "best_checkpoint_path": str(run_main / "checkpoints" / "best.pt"),
            "checkpoint_used": "best",
        },
    ]
    fieldnames = list(rows[0].keys())
    with (registries / "screening_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    emitted = tmp_path / "configs" / "generated" / "generated_final.yaml"
    emitted.parent.mkdir(parents=True, exist_ok=True)
    run_python_module("src.cli.select_finalist", "--outputs-root", str(output_root), "--emit-config", str(emitted))

    generated = load_yaml(emitted)
    selection = load_json(output_root / "registries" / "finalist_selection.json")

    assert selection["selected_baseline_exp_id"] == "BASE_DENSE_WHS_PROBE_1B"
    assert generated["project"]["exp_id"] == "FINAL_BEST_BASELINE_1B"
    assert generated["project"]["stage"] == "final"
    assert generated["project"]["priority"] == "must"
    assert generated["data"]["train_token_quota"] == 50000000
    assert generated["data"]["val_token_quota"] == 5000000
    assert generated["data"]["test_token_quota"] == 5000000
    assert generated["inherit_from"] == str(Path("..") / "screen_dense_whs_probe_init_1b.yaml")
    assert generated["model"]["layer_mix_mode"] == "dense_whs"
    assert generated["model"]["router_init_mode"] == "probe_zscore_top5"
    assert generated["selection_provenance"]["authoritative"] is True
    assert selection["authoritative"] is True
