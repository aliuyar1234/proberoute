from __future__ import annotations

import inspect
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "fixtures" / "tiny_corpus.jsonl"
SMOKE_CONFIG_PATH = REPO_ROOT / "configs" / "smoke_local_tiny.yaml"
SCHEMA_DIR = REPO_ROOT / "schemas"


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_smoke_config(
    tmp_path: Path,
    *,
    exp_id: str = "SMOKE_LOCAL_TINY",
    overrides: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    config = load_yaml(SMOKE_CONFIG_PATH)
    output_root = tmp_path / "outputs"
    config["project"]["exp_id"] = exp_id
    config["project"]["output_root"] = str(output_root)
    config["data"]["local_path"] = str(FIXTURE_PATH)
    config["data"]["processed_root"] = str(output_root / "data" / "processed")
    if overrides:
        _deep_update(config, overrides)
    config_path = tmp_path / f"{exp_id.lower()}.yaml"
    write_yaml(config_path, config)
    return config_path, config


def run_python_module(module: str, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=str(cwd or REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )


def dataset_dir_from_output_root(output_root: Path) -> Path:
    manifests = list((output_root / "data" / "processed").glob("*/manifest.json"))
    assert len(manifests) == 1, f"expected exactly one dataset manifest, found {manifests}"
    return manifests[0].parent


def run_dir_from_output_root(output_root: Path, exp_id: str) -> Path:
    candidates = list((output_root / "runs" / exp_id).glob("**/seed_*"))
    assert len(candidates) == 1, f"expected exactly one run dir for {exp_id}, found {candidates}"
    return candidates[0]


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def future_metrics_payload(*, exp_id: str, split: str, mean_top1_h2_h4: float, mean_nll_h1_h4: float) -> dict[str, Any]:
    return {
        "exp_id": exp_id,
        "split": split,
        "num_sequences": 2,
        "horizons": [1, 2, 3, 4],
        "metrics_by_horizon": {
            "1": {"top1": mean_top1_h2_h4, "top5": 1.0, "nll": mean_nll_h1_h4},
            "2": {"top1": mean_top1_h2_h4, "top5": 1.0, "nll": mean_nll_h1_h4},
            "3": {"top1": mean_top1_h2_h4, "top5": 1.0, "nll": mean_nll_h1_h4},
            "4": {"top1": mean_top1_h2_h4, "top5": 1.0, "nll": mean_nll_h1_h4},
        },
        "aggregate_metrics": {
            "mean_top1_h1_h4": mean_top1_h2_h4,
            "mean_top1_h2_h4": mean_top1_h2_h4,
            "mean_top5_h1_h4": 1.0,
            "mean_nll_h1_h4": mean_nll_h1_h4,
        },
        "bootstrap_ci": None,
        "evaluation_seed": 2026,
    }


def acceptance_metrics_payload(*, exp_id: str, mean_accept_len: float, trace_path: str) -> dict[str, Any]:
    return {
        "exp_id": exp_id,
        "num_prefixes": 2,
        "max_horizon": 4,
        "max_new_tokens": 8,
        "prefix_len": 16,
        "greedy_policy": "argmax",
        "advance_policy": "append_one_base_greedy_token",
        "mean_accept_len": mean_accept_len,
        "accept_rate_depth_1": 1.0,
        "accept_rate_depth_2": 0.5,
        "accept_rate_depth_3": 0.0,
        "accept_rate_depth_4": 0.0,
        "accept_len_histogram": {"1": 1, "2": 1},
        "bootstrap_ci": None,
        "evaluation_seed": 2026,
        "trace_path": trace_path,
    }


def router_metrics_payload(*, exp_id: str, layer_mix_mode: str = "sparse_topm", top_m: int = 2) -> dict[str, Any]:
    return {
        "exp_id": exp_id,
        "layer_mix_mode": layer_mix_mode,
        "horizons": [1, 2, 3, 4],
        "top_m": top_m,
        "entropy_by_horizon": {"1": 0.5, "2": 0.5, "3": 0.5, "4": 0.5},
        "selected_layers_by_horizon": {"1": [0, 1], "2": [0, 1], "3": [0, 1], "4": [0, 1]},
        "average_weights_by_horizon": {"1": [0.5, 0.5], "2": [0.5, 0.5], "3": [0.5, 0.5], "4": [0.5, 0.5]},
        "overlap_with_probe_topm_by_horizon": None,
    }


def make_fake_run(
    output_root: Path,
    *,
    exp_id: str,
    stage: str,
    priority: str = "must",
    model_id: str = "pythia-1b",
    seed: int = 1337,
    status: str = "completed",
    layer_mix_mode: str = "sparse_topm",
    router_init_mode: str = "random",
    val_mean_top1_h2_h4: float = 0.2,
    val_mean_accept_len: float = 1.0,
    val_mean_nll_h1_h4: float = 1.2,
    test_mean_top1_h2_h4: float = 0.2,
    test_mean_accept_len: float = 1.0,
    test_mean_nll_h1_h4: float = 1.2,
    with_probe: bool = False,
    with_png: bool = False,
    config_overrides: dict[str, Any] | None = None,
) -> Path:
    run_root = output_root / "runs" / exp_id
    if run_root.exists():
        existing_manifests = list(run_root.glob("**/run_manifest.json"))
        if existing_manifests:
            existing_stage = json.loads(existing_manifests[0].read_text(encoding="utf-8")).get("stage")
            if existing_stage != stage:
                run_root = output_root / "runs" / f"{exp_id}__{stage}"
    run_dir = run_root / model_id / f"seed_{seed}"
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "exp_id": exp_id,
        "stage": stage,
        "priority": priority,
        "model_id": model_id,
        "seed": seed,
        "backbone_name": "EleutherAI/pythia-1b",
        "layer_mix_mode": layer_mix_mode,
        "router_init_mode": router_init_mode,
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
        "start_time": iso_timestamp(),
        "interpreter_path": str(REPO_ROOT / ".venv" / "Scripts" / "python.exe"),
        "status": status,
        "resume_count": 0,
        "optimizer_step": 8 if status == "completed" else 0,
        "micro_step": 8 if status == "completed" else 0,
        "consumed_tokens": 1024 if status == "completed" else 0,
        "realized_train_tokens": 1024 if status == "completed" else None,
        "last_checkpoint_path": str(run_dir / "checkpoints" / "last.pt") if status == "completed" else None,
        "last_checkpoint_step": 8 if status == "completed" else None,
        "last_checkpoint_micro_step": 8 if status == "completed" else None,
        "best_step": 7 if status == "completed" else None,
        "best_checkpoint_path": str(run_dir / "checkpoints" / "best.pt") if status == "completed" else None,
        "best_mean_val_nll_h1_h4": val_mean_nll_h1_h4 if status == "completed" else None,
        "best_mean_val_top1_h2_h4": val_mean_top1_h2_h4 if status == "completed" else None,
    }
    write_json(run_dir / "run_manifest.json", manifest)

    resolved_config = load_yaml(SMOKE_CONFIG_PATH)
    resolved_config["project"]["exp_id"] = exp_id
    resolved_config["project"]["stage"] = stage
    resolved_config["project"]["priority"] = priority
    resolved_config["project"]["seed"] = seed
    resolved_config["project"]["output_root"] = str(output_root)
    resolved_config["model"]["layer_mix_mode"] = layer_mix_mode
    resolved_config["model"]["router_init_mode"] = router_init_mode
    resolved_config["model"]["backbone_name"] = "EleutherAI/pythia-1b"
    if config_overrides:
        _deep_update(resolved_config, config_overrides)
    write_yaml(run_dir / "resolved_config.yaml", resolved_config)

    trace_dir = run_dir / "artifacts" / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / "acceptance_traces.jsonl").write_text("", encoding="utf-8")
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": {"router": {}, "heads": {}}, "next_step": 7, "optimizer_step": 7, "micro_step": 7, "consumed_tokens": 896}, checkpoint_dir / "best.pt")
    torch.save({"model_state": {"router": {}, "heads": {}}, "next_step": 8, "optimizer_step": 8, "micro_step": 8, "consumed_tokens": 1024}, checkpoint_dir / "last.pt")
    (run_dir / "val_metrics.jsonl").write_text(
        '{"step": 7, "micro_step": 7, "consumed_tokens": 896, "mean_val_nll_h1_h4": 1.0, "mean_val_top1_h2_h4": 0.5}\n',
        encoding="utf-8",
    )
    write_json(
        eval_dir / "val_future_metrics.json",
        {
            **future_metrics_payload(exp_id=exp_id, split="val", mean_top1_h2_h4=val_mean_top1_h2_h4, mean_nll_h1_h4=val_mean_nll_h1_h4),
            "checkpoint_used": "best",
        },
    )
    write_json(
        eval_dir / "test_future_metrics.json",
        {
            **future_metrics_payload(exp_id=exp_id, split="test", mean_top1_h2_h4=test_mean_top1_h2_h4, mean_nll_h1_h4=test_mean_nll_h1_h4),
            "checkpoint_used": "best",
        },
    )
    write_json(
        eval_dir / "val_acceptance_metrics.json",
        {
            **acceptance_metrics_payload(exp_id=exp_id, mean_accept_len=val_mean_accept_len, trace_path=str(trace_dir / "acceptance_traces.jsonl")),
            "checkpoint_used": "best",
        },
    )
    write_json(
        eval_dir / "test_acceptance_metrics.json",
        {
            **acceptance_metrics_payload(exp_id=exp_id, mean_accept_len=test_mean_accept_len, trace_path=str(trace_dir / "acceptance_traces.jsonl")),
            "checkpoint_used": "best",
        },
    )
    write_json(
        eval_dir / "router_metrics.json",
        {
            **router_metrics_payload(exp_id=exp_id, layer_mix_mode=layer_mix_mode),
            "checkpoint_used": "best",
        },
    )

    if with_probe:
        probe_dir = run_dir / "artifacts" / "probe"
        probe_dir.mkdir(parents=True, exist_ok=True)
        (probe_dir / "probe_scores.csv").write_text("layer,horizon,top1,top5,nll\n0,1,0.1,0.2,1.0\n", encoding="utf-8")
        (probe_dir / "probe_heatmap_top1.png").write_bytes(
            bytes.fromhex("89504E470D0A1A0A0000000D4948445200000001000000010802000000907724DE0000000C49444154789C6360000002000154A24F5D0000000049454E44AE426082")
        )
        (probe_dir / "probe_heatmap_top5.png").write_bytes(
            bytes.fromhex("89504E470D0A1A0A0000000D4948445200000001000000010802000000907724DE0000000C49444154789C6360000002000154A24F5D0000000049454E44AE426082")
        )
        (probe_dir / "probe_heatmap_nll.png").write_bytes(
            bytes.fromhex("89504E470D0A1A0A0000000D4948445200000001000000010802000000907724DE0000000C49444154789C6360000002000154A24F5D0000000049454E44AE426082")
        )
        write_json(probe_dir / "probe_init.json", {"exp_id": exp_id, "backbone_name": "EleutherAI/pythia-1b"})
        torch.save({"scores": torch.zeros((4, 2), dtype=torch.float32)}, probe_dir / "probe_init.pt")
    if with_png:
        png_dir = run_dir / "artifacts" / "figures"
        png_dir.mkdir(parents=True, exist_ok=True)
        (png_dir / "example_plot.png").write_bytes(
            bytes.fromhex("89504E470D0A1A0A0000000D4948445200000001000000010802000000907724DE0000000C49444154789C6360000002000154A24F5D0000000049454E44AE426082")
        )
    return run_dir


def call_with_known_kwargs(func: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(func)
    accepted: dict[str, Any] = {}
    aliases = {
        "mode": "layer_mix_mode",
        "layer_mix_mode": "layer_mix_mode",
        "init_mode": "router_init_mode",
        "router_init_mode": "router_init_mode",
        "num_layers": "num_layers",
        "layer_count": "num_layers",
        "horizons": "horizons",
        "top_m": "top_m",
        "rank": "rank",
        "probe_rank": "rank",
        "hidden_size": "hidden_size",
        "d_model": "hidden_size",
        "vocab_size": "vocab_size",
        "unembedding_weight": "unembedding_weight",
        "unembed_weight": "unembedding_weight",
        "config": "config",
        "probe_init": "probe_init",
        "probe_metadata": "probe_init",
    }
    reverse_lookup = {param: key for key, param in aliases.items()}
    for parameter_name in signature.parameters:
        canonical_name = aliases.get(parameter_name, parameter_name)
        if canonical_name in kwargs:
            accepted[parameter_name] = kwargs[canonical_name]
        elif parameter_name in reverse_lookup and reverse_lookup[parameter_name] in kwargs:
            accepted[parameter_name] = kwargs[reverse_lookup[parameter_name]]
    return func(**accepted)


def first_existing_attr(obj: Any, names: list[str]) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AssertionError(f"none of the expected attributes exist: {names}")
