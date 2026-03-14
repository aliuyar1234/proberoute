from __future__ import annotations

import math
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from src.core.compatibility import dataset_manifest_matches_config
from src.core.config import config_digest, save_resolved_config
from src.core.environment import write_environment_snapshot
from src.core.io_utils import read_json, write_json
from src.core.manifest import build_dataset_id, build_run_manifest, processed_dataset_dir, run_dir_for_config, update_run_manifest, write_run_manifest
from src.core.seeding import seed_everything
from src.data.dataset_stream import load_documents
from src.models.backbone_wrapper import BackboneWrapper
from src.testing.local_toy_tokenizer import LocalToyTokenizer


def build_tokenizer(config: dict) -> Any:
    tokenizer_name = config["data"]["tokenizer_name"]
    if tokenizer_name == "local_toy_whitespace":
        documents = load_documents(config)
        return LocalToyTokenizer.build_from_texts(documents)
    try:
        transformers = import_module("transformers")
    except ImportError as exc:
        raise RuntimeError("Remote tokenizer loading requires the `transformers` package to be installed.") from exc
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_backbone(config: dict, tokenizer: Any) -> BackboneWrapper:
    return BackboneWrapper(
        config["model"]["backbone_name"],
        precision=config["hardware"]["precision"],
        device=config["hardware"]["device"],
        tokenizer=tokenizer,
        smoke_mode=config["project"]["stage"] == "smoke",
    )


def load_dataset_arrays(config: dict) -> tuple[dict[str, np.ndarray], dict]:
    output_dir = processed_dataset_dir(config)
    manifest_path = output_dir / "dataset_manifest.json"
    manifest = read_json(manifest_path)
    if not dataset_manifest_matches_config(manifest, config):
        raise ValueError(f"Incompatible dataset manifest at {manifest_path}")
    val_future_path = output_dir / "eval_future_sequences_val.npy"
    test_future_path = output_dir / "eval_future_sequences_test.npy"
    val_acceptance_path = output_dir / "eval_acceptance_prefixes_val.npy"
    test_acceptance_path = output_dir / "eval_acceptance_prefixes_test.npy"
    arrays = {
        "train": np.load(output_dir / "train.npy"),
        "val": np.load(output_dir / "val.npy"),
        "test": np.load(output_dir / "test.npy"),
        "eval_future_sequences_val": np.load(val_future_path) if val_future_path.exists() else np.load(output_dir / "val.npy"),
        "eval_future_sequences_test": np.load(test_future_path) if test_future_path.exists() else np.load(output_dir / "eval_future_sequences.npy"),
        "eval_acceptance_prefixes_val": np.load(val_acceptance_path)
        if val_acceptance_path.exists()
        else np.load(output_dir / "eval_acceptance_prefixes.npy"),
        "eval_acceptance_prefixes_test": np.load(test_acceptance_path)
        if test_acceptance_path.exists()
        else np.load(output_dir / "eval_acceptance_prefixes.npy"),
    }
    return arrays, manifest


def init_run_dir(config: dict, *, model_slug: str, dataset_id: str, force: bool = False, resume: bool = False) -> Path:
    if force and resume:
        raise ValueError("`force` and `resume` cannot both be enabled for the same run")
    run_dir = run_dir_for_config(config, model_slug=model_slug)
    manifest_path = run_dir / "run_manifest.json"
    if resume:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Cannot resume a run without an existing manifest at {manifest_path}")
        return run_dir
    if manifest_path.exists() and not force:
        raise FileExistsError(f"Run directory already exists at {run_dir}; use --resume or --force")
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = run_dir / "resolved_config.yaml"
    environment_snapshot_path = run_dir / "environment_snapshot.json"
    save_resolved_config(config, resolved_config_path)
    write_environment_snapshot(environment_snapshot_path, config)
    (run_dir / "stdout.log").touch()
    manifest = build_run_manifest(config, model_id=model_slug, dataset_id=dataset_id, status="running")
    manifest["resolved_config_path"] = str(resolved_config_path)
    manifest["environment_snapshot_path"] = str(environment_snapshot_path)
    manifest["config_digest"] = config_digest(config)
    write_run_manifest(manifest_path, manifest)
    return run_dir


def mark_run_resumed(run_dir: Path, *, config: dict) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    manifest = read_json(manifest_path)
    updates = {
        "status": "running",
        "end_time": None,
        "resolved_config_path": str(run_dir / "resolved_config.yaml"),
        "environment_snapshot_path": str(run_dir / "environment_snapshot.json"),
        "config_digest": config_digest(config),
        "interpreter_path": sys.executable,
        "resume_count": int(manifest.get("resume_count", 0)) + 1,
    }
    return update_run_manifest(manifest_path, updates)


def batches_per_pass(array: np.ndarray, batch_size: int) -> int:
    if len(array) == 0:
        raise ValueError("Cannot iterate batches over an empty dataset")
    return max(1, math.ceil(len(array) / batch_size))


def batch_for_step(array: np.ndarray, batch_size: int, step_idx: int) -> torch.Tensor:
    if step_idx < 0:
        raise ValueError("step_idx must be non-negative")
    steps_per_pass = batches_per_pass(array, batch_size)
    batch_idx = step_idx % steps_per_pass
    start = batch_idx * batch_size
    batch = array[start : start + batch_size]
    return torch.tensor(batch, dtype=torch.long)


def iterate_batches(array: np.ndarray, batch_size: int) -> Iterator[torch.Tensor]:
    step_idx = 0
    while True:
        yield batch_for_step(array, batch_size, step_idx)
        step_idx += 1


def validation_interval_steps(config: dict) -> int:
    train_config = config["train"]
    if train_config.get("val_every_steps") not in {None, ""}:
        return int(train_config["val_every_steps"])
    if train_config.get("eval_every_steps") not in {None, ""}:
        return int(train_config["eval_every_steps"])
    return int(train_config.get("save_every_steps") or 0)


def validation_record_is_better(candidate: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None:
        return True
    candidate_key = (
        float(candidate["mean_val_nll_h1_h4"]),
        -float(candidate["mean_val_top1_h2_h4"]),
        int(candidate["step"]),
    )
    best_key = (
        float(best["mean_val_nll_h1_h4"]),
        -float(best["mean_val_top1_h2_h4"]),
        int(best["step"]),
    )
    return candidate_key < best_key


def is_scientific_stage(stage: str) -> bool:
    return stage not in {"smoke", "template"}


def is_venv_interpreter(path: str | None) -> bool:
    if not path:
        return False
    return ".venv" in path.replace("/", "\\").lower()


def gradient_accumulation_steps(config: dict[str, Any]) -> int:
    return max(1, int(config["hardware"].get("grad_accum") or 1))


def planned_micro_steps(token_quota: int, *, batch_size: int, seq_len: int) -> int:
    tokens_per_step = max(1, batch_size * seq_len)
    return max(1, math.ceil(token_quota / tokens_per_step))


def planned_steps(token_quota: int, *, batch_size: int, seq_len: int, grad_accum: int = 1) -> int:
    micro_steps = planned_micro_steps(token_quota, batch_size=batch_size, seq_len=seq_len)
    return max(1, math.ceil(micro_steps / max(1, grad_accum)))


def realized_train_tokens(token_quota: int, *, batch_size: int, seq_len: int, grad_accum: int = 1) -> int:
    optimizer_steps = planned_steps(token_quota, batch_size=batch_size, seq_len=seq_len, grad_accum=grad_accum)
    return optimizer_steps * max(1, batch_size * seq_len) * max(1, grad_accum)


def next_cadence_micro_step(current_micro_step: int, every_steps: int) -> int | None:
    if every_steps <= 0:
        return None
    return ((current_micro_step // every_steps) + 1) * every_steps


def setup_training(config: dict) -> None:
    seed_everything(int(config["project"]["seed"]))
