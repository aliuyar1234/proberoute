from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from src.core.config import config_digest
from src.core.io_utils import read_json, write_json
from src.core.manifest import finalize_run_manifest, update_run_manifest
from src.models.probe_bank import LowRankProbeBank
from src.train.checkpointing import capture_rng_state, consume_pause_request, load_checkpoint, restore_rng_state, save_checkpoint
from .trainer_common import (
    batch_for_step,
    build_backbone,
    build_tokenizer,
    gradient_accumulation_steps,
    init_run_dir,
    is_scientific_stage,
    load_dataset_arrays,
    mark_run_resumed,
    next_cadence_micro_step,
    planned_steps,
    setup_training,
    validation_interval_steps,
    validation_record_is_better,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _probe_scores(
    probe: LowRankProbeBank,
    backbone,
    array: np.ndarray,
    horizons: list[int],
    *,
    batch_size: int,
) -> dict[str, Any]:
    scores = {
        "top1": {h: [0.0] * backbone.num_layers() for h in horizons},
        "top5": {h: [0.0] * backbone.num_layers() for h in horizons},
        "nll": {h: [0.0] * backbone.num_layers() for h in horizons},
    }
    counts = {
        "top1": {h: [0] * backbone.num_layers() for h in horizons},
        "top5": {h: [0] * backbone.num_layers() for h in horizons},
        "nll": {h: [0] * backbone.num_layers() for h in horizons},
    }
    effective_batch_size = max(1, batch_size)
    unembed_weight = backbone.unembedding_weight()
    for start in range(0, len(array), effective_batch_size):
        batch_array = array[start : start + effective_batch_size]
        batch = torch.tensor(batch_array, dtype=torch.long, device=backbone.device)
        hidden = backbone.forward_hidden(batch)
        for layer_idx, horizon_idx, logits in probe.iter_logits(hidden.hidden_states, unembed_weight):
            horizon = horizons[horizon_idx]
            usable_logits = logits[:, :-horizon, :]
            usable_targets = batch[:, horizon:]
            vocab = usable_logits.shape[-1]
            token_count = int(usable_targets.numel())
            scores["top1"][horizon][layer_idx] += float((usable_logits.argmax(dim=-1) == usable_targets).sum().item())
            scores["top5"][horizon][layer_idx] += float(
                usable_logits.topk(k=min(5, vocab), dim=-1).indices.eq(usable_targets.unsqueeze(-1)).any(dim=-1).sum().item()
            )
            scores["nll"][horizon][layer_idx] += float(
                F.cross_entropy(usable_logits.reshape(-1, vocab), usable_targets.reshape(-1), reduction="sum").item()
            )
            counts["top1"][horizon][layer_idx] += token_count
            counts["top5"][horizon][layer_idx] += token_count
            counts["nll"][horizon][layer_idx] += token_count
            del usable_logits, usable_targets, logits
    for horizon in horizons:
        for layer_idx in range(backbone.num_layers()):
            token_count = max(1, counts["nll"][horizon][layer_idx])
            scores["top1"][horizon][layer_idx] /= token_count
            scores["top5"][horizon][layer_idx] /= token_count
            scores["nll"][horizon][layer_idx] /= token_count
    return scores


def _probe_validation_record(scores: dict[str, Any], horizons: list[int], *, step: int, micro_step: int, consumed_tokens: int) -> dict[str, Any]:
    mean_top1_by_horizon = {str(h): float(np.mean(scores["top1"][h])) for h in horizons}
    mean_top5_by_horizon = {str(h): float(np.mean(scores["top5"][h])) for h in horizons}
    mean_nll_by_horizon = {str(h): float(np.mean(scores["nll"][h])) for h in horizons}
    top1_values = [mean_top1_by_horizon[str(h)] for h in horizons]
    nll_values = [mean_nll_by_horizon[str(h)] for h in horizons]
    return {
        "step": step,
        "micro_step": micro_step,
        "consumed_tokens": consumed_tokens,
        "mean_val_nll_h1_h4": float(np.mean(nll_values)),
        "mean_val_top1_h2_h4": float(np.mean(top1_values[1:])) if len(top1_values) > 1 else float(np.mean(top1_values)),
        "per_horizon_val_metrics": {
            str(h): {
                "top1": mean_top1_by_horizon[str(h)],
                "top5": mean_top5_by_horizon[str(h)],
                "nll": mean_nll_by_horizon[str(h)],
            }
            for h in horizons
        },
        "timestamp": _utc_now(),
    }


def _write_probe_csv(path: Path, scores: dict[str, Any], horizons: list[int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["layer", "horizon", "top1", "top5", "nll"])
        writer.writeheader()
        for horizon in horizons:
            for layer_idx, top1 in enumerate(scores["top1"][horizon]):
                writer.writerow(
                    {
                        "layer": layer_idx,
                        "horizon": horizon,
                        "top1": scores["top1"][horizon][layer_idx],
                        "top5": scores["top5"][horizon][layer_idx],
                        "nll": scores["nll"][horizon][layer_idx],
                    }
                )
    return path


def _write_heatmap(path: Path, values: list[list[float]], horizons: list[int], title: str) -> Path:
    data = np.array(values).T
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(data, aspect="auto")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _build_probe_init(scores: dict[str, Any], horizons: list[int], *, config: dict, backbone, dataset_id: str) -> tuple[dict[str, Any], torch.Tensor]:
    raw = {str(h): list(scores["top5"][h]) for h in horizons}
    z_scored: dict[str, list[float]] = {}
    tensors = []
    for horizon in horizons:
        values = np.asarray(scores["top5"][horizon], dtype=np.float32)
        std = float(values.std()) if len(values) > 1 else 0.0
        if std == 0.0:
            normalized = np.zeros_like(values)
        else:
            normalized = (values - values.mean()) / std
        z_scored[str(horizon)] = normalized.tolist()
        tensors.append(torch.tensor(normalized, dtype=torch.float32))
    init_tensor = torch.stack(tensors, dim=0)
    payload = {
        "exp_id": config["project"]["exp_id"],
        "seed": config["project"]["seed"],
        "model_id": backbone.model_slug(),
        "backbone_name": config["model"]["backbone_name"],
        "dataset_name": config["data"]["dataset_name"],
        "dataset_config": config["data"]["dataset_config"],
        "dataset_id": dataset_id,
        "seq_len": config["data"]["seq_len"],
        "horizons": horizons,
        "num_layers": backbone.num_layers(),
        "metric": config["model"]["probe_init_metric"],
        "init_metric": config["model"]["probe_init_metric"],
        "raw_top5_scores": raw,
        "z_scored_top5_scores": z_scored,
    }
    return payload, init_tensor


def _probe_checkpoint_payload(
    probe: LowRankProbeBank,
    optimizer: torch.optim.Optimizer,
    *,
    next_step: int,
    micro_step: int,
    consumed_tokens: int,
    config_digest_value: str,
) -> dict[str, Any]:
    return {
        "checkpoint_type": "probe_train_state",
        "next_step": next_step,
        "optimizer_step": next_step,
        "micro_step": micro_step,
        "consumed_tokens": consumed_tokens,
        "model_state": probe.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None,
        "scaler_state": None,
        "rng_state": capture_rng_state(),
        "config_digest": config_digest_value,
        "checkpoint_timestamp": _utc_now(),
    }


def train_probe_run(config: dict, *, dry_run: bool = False, force: bool = False, resume: bool = False) -> Path:
    setup_training(config)
    tokenizer = build_tokenizer(config)
    backbone = build_backbone(config, tokenizer)
    arrays, manifest = load_dataset_arrays(config)
    dataset_id = manifest["dataset_id"]
    run_dir = init_run_dir(config, model_slug=backbone.model_slug(), dataset_id=dataset_id, force=force, resume=resume)
    if dry_run:
        finalize_run_manifest(run_dir / "run_manifest.json", "completed")
        return run_dir

    artifact_dir = run_dir / "artifacts" / "probe"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_log = run_dir / "train_metrics.jsonl"
    val_log = run_dir / "val_metrics.jsonl"
    manifest_path = run_dir / "run_manifest.json"
    config_digest_value = config_digest(config)

    probe = LowRankProbeBank(
        num_layers=backbone.num_layers(),
        horizons=list(config["model"]["horizons"]),
        hidden_size=backbone.hidden_size(),
        rank=int(config["model"]["probe_rank"]),
        unembedding_weight=backbone.unembedding_weight().detach().cpu(),
    ).to(backbone.device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=float(config["train"]["lr_probe"]), weight_decay=float(config["train"]["weight_decay"]))
    grad_accum = gradient_accumulation_steps(config)
    steps = planned_steps(
        int(config["data"]["train_token_quota"]),
        batch_size=int(config["hardware"]["micro_batch_size"]),
        seq_len=int(config["data"]["seq_len"]),
        grad_accum=grad_accum,
    )
    batch_size = int(config["hardware"]["micro_batch_size"])
    seq_len = int(config["data"]["seq_len"])
    save_every_steps = int(config["train"].get("save_every_steps") or 0)
    val_every_steps = validation_interval_steps(config)
    horizons = list(config["model"]["horizons"])
    consumed_tokens = 0
    micro_step = 0
    start_step = 0
    last_completed_step = -1
    best_record: dict[str, Any] | None = None
    loss_terms = max(1, backbone.num_layers() * len(horizons))
    unembed_weight = backbone.unembedding_weight()

    if resume:
        checkpoint_path = checkpoint_dir / "last.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Cannot resume probe run without a checkpoint at {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint.get("config_digest") != config_digest_value:
            raise ValueError("Cannot resume probe run with a config that does not match the saved checkpoint digest")
        probe.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        restore_rng_state(checkpoint.get("rng_state"))
        start_step = int(checkpoint.get("next_step", 0))
        consumed_tokens = int(checkpoint.get("consumed_tokens", 0))
        micro_step = int(checkpoint.get("micro_step", consumed_tokens // max(1, batch_size * seq_len)))
        last_completed_step = start_step - 1
        manifest_payload = read_json(manifest_path)
        if manifest_payload.get("best_step") is not None:
            best_record = {
                "step": int(manifest_payload["best_step"]),
                "mean_val_nll_h1_h4": float(manifest_payload["best_mean_val_nll_h1_h4"]),
                "mean_val_top1_h2_h4": float(manifest_payload["best_mean_val_top1_h2_h4"]),
            }
        mark_run_resumed(run_dir, config=config)

    def _checkpoint_payload(next_step: int) -> dict[str, Any]:
        return _probe_checkpoint_payload(
            probe,
            optimizer,
            next_step=next_step,
            micro_step=micro_step,
            consumed_tokens=consumed_tokens,
            config_digest_value=config_digest_value,
        )

    def _save_last_checkpoint(next_step: int) -> None:
        checkpoint_path = checkpoint_dir / "last.pt"
        save_checkpoint(checkpoint_path, _checkpoint_payload(next_step))
        update_run_manifest(
            manifest_path,
            {
                "last_checkpoint_path": str(checkpoint_path),
                "last_checkpoint_step": next_step,
                "last_checkpoint_micro_step": micro_step,
                "optimizer_step": next_step,
                "micro_step": micro_step,
                "consumed_tokens": consumed_tokens,
                "interpreter_path": __import__("sys").executable,
            },
        )

    def _run_validation(step_number: int) -> dict[str, Any]:
        scores = _probe_scores(
            probe,
            backbone,
            arrays["eval_future_sequences_val"],
            horizons,
            batch_size=batch_size,
        )
        record = _probe_validation_record(scores, horizons, step=step_number, micro_step=micro_step, consumed_tokens=consumed_tokens)
        _append_jsonl(val_log, record)
        return record

    def _maybe_update_best(validation_record: dict[str, Any]) -> None:
        nonlocal best_record
        if not validation_record_is_better(validation_record, best_record):
            return
        best_record = {
            "step": int(validation_record["step"]),
            "mean_val_nll_h1_h4": float(validation_record["mean_val_nll_h1_h4"]),
            "mean_val_top1_h2_h4": float(validation_record["mean_val_top1_h2_h4"]),
        }
        checkpoint_path = checkpoint_dir / "best.pt"
        save_checkpoint(checkpoint_path, _checkpoint_payload(int(validation_record["step"])))
        update_run_manifest(
            manifest_path,
            {
                "best_step": int(validation_record["step"]),
                "best_checkpoint_path": str(checkpoint_path),
                "best_mean_val_nll_h1_h4": float(validation_record["mean_val_nll_h1_h4"]),
                "best_mean_val_top1_h2_h4": float(validation_record["mean_val_top1_h2_h4"]),
                "optimizer_step": int(validation_record["step"]),
                "micro_step": micro_step,
                "consumed_tokens": consumed_tokens,
                "interpreter_path": __import__("sys").executable,
            },
        )

    next_save_micro_step = next_cadence_micro_step(micro_step, save_every_steps)
    next_val_micro_step = next_cadence_micro_step(micro_step, val_every_steps)
    save_due = False
    val_due = False

    try:
        for step_idx in range(start_step, steps):
            optimizer.zero_grad(set_to_none=True)
            loss_by_horizon: dict[int, list[float]] = {h: [] for h in horizons}
            mean_loss = 0.0
            for _ in range(grad_accum):
                batch = batch_for_step(arrays["train"], batch_size, micro_step).to(backbone.device)
                hidden = backbone.forward_hidden(batch)
                for layer_idx, horizon_idx, layer_logits in probe.iter_logits(hidden.hidden_states, unembed_weight):
                    horizon = horizons[horizon_idx]
                    usable_logits = layer_logits[:, :-horizon, :]
                    usable_targets = batch[:, horizon:]
                    horizon_loss = F.cross_entropy(usable_logits.reshape(-1, layer_logits.shape[-1]), usable_targets.reshape(-1))
                    (horizon_loss / (loss_terms * grad_accum)).backward()
                    horizon_loss_value = float(horizon_loss.detach().cpu().item())
                    mean_loss += horizon_loss_value
                    loss_by_horizon[horizon].append(horizon_loss_value)
                    del horizon_loss, usable_logits, usable_targets, layer_logits
                consumed_tokens += int(batch.numel())
                micro_step += 1
                if next_save_micro_step is not None and micro_step >= next_save_micro_step:
                    save_due = True
                    while next_save_micro_step is not None and micro_step >= next_save_micro_step:
                        next_save_micro_step += save_every_steps
                if next_val_micro_step is not None and micro_step >= next_val_micro_step:
                    val_due = True
                    while next_val_micro_step is not None and micro_step >= next_val_micro_step:
                        next_val_micro_step += val_every_steps
            mean_loss /= loss_terms * grad_accum
            optimizer.step()
            last_completed_step = step_idx
            _append_jsonl(
                train_log,
                {
                    "step": step_idx + 1,
                    "micro_step": micro_step,
                    "consumed_tokens": consumed_tokens,
                    "train_loss": mean_loss,
                    "per_horizon_train_losses": {str(h): float(np.mean(values)) for h, values in loss_by_horizon.items()},
                    "learning_rates": {"probe": float(optimizer.param_groups[0]["lr"])},
                    "timestamp": _utc_now(),
                },
            )
            update_run_manifest(
                manifest_path,
                {
                    "optimizer_step": step_idx + 1,
                    "micro_step": micro_step,
                    "consumed_tokens": consumed_tokens,
                },
            )

            is_final_step = step_idx == steps - 1
            persisted_boundary_state = False
            if is_final_step or val_due:
                validation_record = _run_validation(step_idx + 1)
                _save_last_checkpoint(step_idx + 1)
                _maybe_update_best(validation_record)
                persisted_boundary_state = True
                val_due = False
                save_due = False
            elif save_due:
                _save_last_checkpoint(step_idx + 1)
                persisted_boundary_state = True
                save_due = False

            if not is_final_step and consume_pause_request(run_dir):
                if not persisted_boundary_state:
                    _save_last_checkpoint(step_idx + 1)
                update_run_manifest(
                    manifest_path,
                    {
                        "realized_train_tokens": consumed_tokens,
                    },
                )
                finalize_run_manifest(manifest_path, "paused")
                return run_dir

        artifact_scores = _probe_scores(
            probe,
            backbone,
            arrays["val"],
            horizons,
            batch_size=batch_size,
        )
        _write_probe_csv(artifact_dir / "probe_scores.csv", artifact_scores, horizons)
        _write_heatmap(artifact_dir / "probe_heatmap_top1.png", [artifact_scores["top1"][h] for h in horizons], horizons, "Probe top-1")
        _write_heatmap(artifact_dir / "probe_heatmap_top5.png", [artifact_scores["top5"][h] for h in horizons], horizons, "Probe top-5")
        _write_heatmap(artifact_dir / "probe_heatmap_nll.png", [artifact_scores["nll"][h] for h in horizons], horizons, "Probe NLL")

        probe_init_json, probe_init_tensor = _build_probe_init(artifact_scores, horizons, config=config, backbone=backbone, dataset_id=dataset_id)
        write_json(artifact_dir / "probe_init.json", probe_init_json)
        save_checkpoint(artifact_dir / "probe_init.pt", {"scores": probe_init_tensor})
        save_checkpoint(artifact_dir / "probe_checkpoint.pt", {"model_state": probe.state_dict()})
        if not is_scientific_stage(config["project"]["stage"]) and best_record is None:
            fallback_record = {
                "step": steps,
                "mean_val_nll_h1_h4": float(np.mean([np.mean(artifact_scores["nll"][h]) for h in horizons])),
                "mean_val_top1_h2_h4": float(np.mean([np.mean(artifact_scores["top1"][h]) for h in horizons[1:]]))
                if len(horizons) > 1
                else float(np.mean([np.mean(artifact_scores["top1"][h]) for h in horizons])),
            }
            _maybe_update_best(fallback_record)
        _save_last_checkpoint(steps)
        update_run_manifest(
            manifest_path,
            {
                "optimizer_step": steps,
                "micro_step": micro_step,
                "consumed_tokens": consumed_tokens,
                "realized_train_tokens": consumed_tokens,
            },
        )
        finalize_run_manifest(manifest_path, "completed")
        return run_dir
    except KeyboardInterrupt:
        update_run_manifest(
            manifest_path,
            {
                "optimizer_step": max(last_completed_step + 1, 0),
                "micro_step": micro_step,
                "consumed_tokens": consumed_tokens,
                "realized_train_tokens": consumed_tokens if consumed_tokens else None,
            },
        )
        finalize_run_manifest(manifest_path, "aborted")
        return run_dir
    except Exception:
        update_run_manifest(
            manifest_path,
            {
                "optimizer_step": max(last_completed_step + 1, 0),
                "micro_step": micro_step,
                "consumed_tokens": consumed_tokens,
                "realized_train_tokens": consumed_tokens if consumed_tokens else None,
            },
        )
        finalize_run_manifest(manifest_path, "failed")
        raise
