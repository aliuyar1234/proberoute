from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .constants import CANONICAL_HIDDEN_NORM, LAYER_MIX_MODES, ROUTER_INIT_MODES, RUN_PRIORITIES, RUN_STAGES, SMOKE_BACKBONE_NAME
from .io_utils import read_yaml, write_yaml


REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "project": ("name", "exp_id", "stage", "priority", "seed", "output_root", "notes"),
    "hardware": ("device", "precision", "expected_vram_gb", "allow_flash_attn", "grad_accum", "micro_batch_size"),
    "data": (
        "dataset_name",
        "dataset_config",
        "local_path",
        "tokenizer_name",
        "text_field_priority",
        "split_modulus",
        "split_ranges",
        "seq_len",
        "train_token_quota",
        "val_token_quota",
        "test_token_quota",
        "processed_root",
        "append_eos_between_docs",
        "drop_last_sequence",
    ),
    "model": (
        "backbone_name",
        "freeze_backbone",
        "horizons",
        "layer_mix_mode",
        "router_init_mode",
        "probe_init_metric",
        "head_type",
        "top_m",
        "router_temperature",
        "probe_rank",
        "hidden_expansion",
        "hidden_norm",
        "entropy_penalty_beta",
        "use_deephead_for_far_horizons",
        "deephead_horizons",
    ),
    "train": (
        "optimizer",
        "stop_mode",
        "lr_head",
        "lr_router",
        "lr_probe",
        "weight_decay",
        "warmup_ratio",
        "max_steps",
        "eval_every_steps",
        "save_every_steps",
        "log_every_steps",
        "early_stop_patience",
        "grad_clip_norm",
        "loss_weights",
        "loss_warmup",
    ),
    "eval": (
        "future_metrics_sequence_count",
        "acceptance_prefix_count",
        "acceptance_prefix_len",
        "acceptance_max_new_tokens",
        "bootstrap_samples",
        "bootstrap_seed",
    ),
}


def deep_merge(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    merged = dict(parent)
    for key, value in child.items():
        if key == "inherit_from":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config_recursive(path: Path, seen: set[Path]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"Cyclic config inheritance detected at {resolved}")
    seen.add(resolved)
    current = read_yaml(resolved)
    parent_ref = current.get("inherit_from")
    if parent_ref is None:
        return current
    parent_path = (resolved.parent / parent_ref).resolve()
    if not parent_path.exists():
        raise FileNotFoundError(f"Missing parent config: {parent_path}")
    parent = _load_config_recursive(parent_path, seen)
    return deep_merge(parent, current)


def load_config(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    config = _load_config_recursive(resolved, set())
    config["_config_path"] = str(resolved)
    return config


def serializable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def config_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(serializable_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    for section, keys in REQUIRED_KEYS.items():
        if section not in config:
            raise ValueError(f"Missing config section: {section}")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Missing required config key: {section}.{key}")

    project = config["project"]
    hardware = config["hardware"]
    data = config["data"]
    model = config["model"]
    train = config["train"]
    evaluation = config["eval"]

    if project["stage"] not in RUN_STAGES:
        raise ValueError(f"Unsupported stage: {project['stage']}")
    if project["priority"] not in RUN_PRIORITIES:
        raise ValueError(f"Unsupported priority: {project['priority']}")
    horizons = list(model["horizons"])
    if horizons != sorted(horizons) or any(h <= 0 for h in horizons):
        raise ValueError("model.horizons must be sorted positive integers")
    if max(horizons) >= int(data["seq_len"]):
        raise ValueError("max(model.horizons) must be less than data.seq_len")
    if model["layer_mix_mode"] not in LAYER_MIX_MODES - {"probe_only"}:
        raise ValueError(f"Unsupported layer_mix_mode: {model['layer_mix_mode']}")
    if model["router_init_mode"] not in ROUTER_INIT_MODES:
        raise ValueError(f"Unsupported router_init_mode: {model['router_init_mode']}")
    if int(model["probe_rank"]) <= 0:
        raise ValueError("model.probe_rank must be positive")
    if int(model["top_m"]) <= 0:
        raise ValueError("model.top_m must be positive")
    if int(hardware["micro_batch_size"]) <= 0:
        raise ValueError("hardware.micro_batch_size must be positive")
    if int(hardware["grad_accum"]) <= 0:
        raise ValueError("hardware.grad_accum must be positive")
    if model["hidden_norm"] != CANONICAL_HIDDEN_NORM:
        raise ValueError(f"model.hidden_norm must be {CANONICAL_HIDDEN_NORM}")
    for key in ("train_token_quota", "val_token_quota", "test_token_quota"):
        if int(data[key]) <= 0:
            raise ValueError(f"data.{key} must be positive")
    if int(evaluation["bootstrap_samples"]) < 100:
        raise ValueError("eval.bootstrap_samples must be >= 100")
    if model["router_init_mode"] == "none" and project["stage"] != "probe" and model["layer_mix_mode"] != "last_layer":
        raise ValueError("router_init_mode=none is only valid for last_layer runs")
    if project["stage"] == "smoke" and model["backbone_name"] != SMOKE_BACKBONE_NAME:
        raise ValueError("smoke runs must use the local toy backbone")
    if project["stage"] not in {"smoke", "template"}:
        val_every = train.get("val_every_steps")
        if val_every in {None, ""}:
            val_every = train.get("eval_every_steps")
        if val_every in {None, ""}:
            val_every = train.get("save_every_steps")
        val_every = int(val_every or 0)
        if val_every <= 0:
            raise ValueError("real runs must configure a positive validation cadence")
        micro_steps = max(1, -(-int(data["train_token_quota"]) // max(1, int(hardware["micro_batch_size"]) * int(data["seq_len"]))))
        final_boundary_micro_step = max(1, -(-micro_steps // int(hardware["grad_accum"]))) * int(hardware["grad_accum"])
        first_due_boundary = max(1, -(-val_every // int(hardware["grad_accum"]))) * int(hardware["grad_accum"])
        if first_due_boundary >= final_boundary_micro_step:
            raise ValueError(
                "real runs must yield at least one nonterminal validation event under the configured token budget, "
                "micro_batch_size, grad_accum, and validation cadence"
            )


def save_resolved_config(config: dict[str, Any], path: Path) -> Path:
    return write_yaml(path, serializable_config(config))
