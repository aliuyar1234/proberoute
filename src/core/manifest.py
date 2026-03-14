from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .constants import OUTPUTS_ROOT, REPO_ROOT
from .io_utils import ensure_dir, read_json, slugify, write_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_dataset_id(config: dict[str, Any]) -> str:
    data = config["data"]
    return "__".join(
        [
            slugify(str(data["dataset_name"])),
            slugify(str(data["dataset_config"])),
            slugify(str(data["tokenizer_name"])),
            f"sl{data['seq_len']}",
        ]
    )


def processed_dataset_dir(config: dict[str, Any]) -> Path:
    root = Path(config["data"]["processed_root"])
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root / build_dataset_id(config)


def run_dir_for_config(config: dict[str, Any], model_slug: str | None = None) -> Path:
    root = Path(config["project"]["output_root"]) if config["project"]["output_root"] else OUTPUTS_ROOT
    if not root.is_absolute():
        root = REPO_ROOT / root
    exp_id = config["project"]["exp_id"]
    seed = config["project"]["seed"]
    slug = model_slug or slugify(config["model"]["backbone_name"])
    return root / "runs" / exp_id / slug / f"seed_{seed}"


def build_run_manifest(config: dict[str, Any], *, model_id: str, dataset_id: str, status: str) -> dict[str, Any]:
    stage = config["project"]["stage"]
    layer_mix_mode = "probe_only" if stage == "probe" else config["model"]["layer_mix_mode"]
    router_init_mode = "none" if stage == "probe" else config["model"]["router_init_mode"]
    micro_batch_size = int(config["hardware"]["micro_batch_size"])
    grad_accum = int(config["hardware"]["grad_accum"])
    seq_len = int(config["data"]["seq_len"])
    return {
        "exp_id": config["project"]["exp_id"],
        "stage": stage,
        "priority": config["project"]["priority"],
        "model_id": model_id,
        "seed": config["project"]["seed"],
        "backbone_name": config["model"]["backbone_name"],
        "layer_mix_mode": layer_mix_mode,
        "router_init_mode": router_init_mode,
        "dataset_name": config["data"]["dataset_name"],
        "dataset_config": config["data"]["dataset_config"],
        "dataset_id": dataset_id,
        "seq_len": seq_len,
        "token_budget_train": config["data"]["train_token_quota"],
        "token_budget_val": config["data"]["val_token_quota"],
        "token_budget_test": config["data"]["test_token_quota"],
        "micro_batch_size": micro_batch_size,
        "grad_accum": grad_accum,
        "nominal_effective_batch_sequences": micro_batch_size * grad_accum,
        "nominal_tokens_per_optimizer_update": micro_batch_size * grad_accum * seq_len,
        "git_commit": None,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tokenizer_name": config["data"]["tokenizer_name"],
        "resolved_config_path": None,
        "source_config_path": config.get("_config_path"),
        "environment_snapshot_path": None,
        "interpreter_path": sys.executable,
        "start_time": utc_now(),
        "end_time": None,
        "status": status,
        "resume_count": 0,
        "config_digest": None,
        "optimizer_step": 0,
        "micro_step": 0,
        "consumed_tokens": 0,
        "realized_train_tokens": None,
        "last_checkpoint_path": None,
        "last_checkpoint_step": None,
        "last_checkpoint_micro_step": None,
        "best_step": None,
        "best_checkpoint_path": None,
        "best_mean_val_nll_h1_h4": None,
        "best_mean_val_top1_h2_h4": None,
        "checkpoint_selection_rule": "lowest_mean_val_nll_h1_h4_then_highest_mean_val_top1_h2_h4_then_earlier_step",
        "notes": config["project"].get("notes"),
    }


def finalize_run_manifest(path: Path, status: str) -> dict[str, Any]:
    payload = read_json(path)
    payload["status"] = status
    payload["end_time"] = utc_now()
    write_json(path, payload)
    return payload


def update_run_manifest(path: Path, updates: dict[str, Any]) -> dict[str, Any]:
    payload = read_json(path)
    payload.update(updates)
    write_json(path, payload)
    return payload


def write_run_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    ensure_dir(path.parent)
    return write_json(path, manifest)
