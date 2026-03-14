from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.core.compatibility import ensure_probe_init_compatible
from src.core.config import config_digest
from src.core.io_utils import read_json, write_json
from src.core.manifest import finalize_run_manifest, update_run_manifest
from src.eval.acceptance import compute_acceptance_metrics
from src.eval.future_metrics import compute_future_metrics
from src.models.layermix_mtp import LayerMixMTPModel
from src.train.checkpointing import capture_rng_state, consume_pause_request, load_checkpoint, restore_rng_state, save_checkpoint
from .trainer_common import (
    batch_for_step,
    build_backbone,
    build_tokenizer,
    gradient_accumulation_steps,
    init_run_dir,
    is_scientific_stage,
    is_venv_interpreter,
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


def _find_probe_init(config: dict, backbone) -> tuple[Path, dict, torch.Tensor]:
    outputs_root = Path(config["project"]["output_root"]) / "runs"
    for probe_json in sorted(outputs_root.glob("**/artifacts/probe/probe_init.json")):
        run_dir = probe_json.parents[2]
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path)
        if manifest.get("stage") != "probe" or manifest.get("status") != "completed":
            continue
        if not is_venv_interpreter(manifest.get("interpreter_path")):
            continue
        payload = read_json(probe_json)
        try:
            ensure_probe_init_compatible(
                payload,
                config,
                model_id=backbone.model_slug(),
                num_layers=backbone.num_layers(),
            )
        except ValueError:
            continue
        tensor_payload = load_checkpoint(probe_json.with_suffix(".pt"))
        return probe_json, payload, tensor_payload["scores"]
    raise FileNotFoundError("No compatible completed real probe_init artifact found for router_init_mode=probe_zscore_top5")


def _mtp_checkpoint_payload(
    model: LayerMixMTPModel,
    optimizer: torch.optim.Optimizer,
    *,
    next_step: int,
    micro_step: int,
    consumed_tokens: int,
    probe_init_source: Path | None,
    config_digest_value: str,
) -> dict:
    return {
        "checkpoint_type": "mtp_train_state",
        "next_step": next_step,
        "optimizer_step": next_step,
        "micro_step": micro_step,
        "consumed_tokens": consumed_tokens,
        "probe_init_source": str(probe_init_source) if probe_init_source else None,
        "model_state": model.checkpoint_state(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None,
        "scaler_state": None,
        "rng_state": capture_rng_state(),
        "config_digest": config_digest_value,
        "checkpoint_timestamp": _utc_now(),
    }


def train_mtp_run(config: dict, *, dry_run: bool = False, force: bool = False, resume: bool = False) -> Path:
    setup_training(config)
    tokenizer = build_tokenizer(config)
    backbone = build_backbone(config, tokenizer)
    arrays, manifest = load_dataset_arrays(config)
    dataset_id = manifest["dataset_id"]
    run_dir = init_run_dir(config, model_slug=backbone.model_slug(), dataset_id=dataset_id, force=force, resume=resume)
    if dry_run:
        finalize_run_manifest(run_dir / "run_manifest.json", "completed")
        return run_dir

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_log = run_dir / "train_metrics.jsonl"
    val_log = run_dir / "val_metrics.jsonl"
    manifest_path = run_dir / "run_manifest.json"
    config_digest_value = config_digest(config)

    probe_init_scores = None
    probe_init_source = None
    if config["model"]["router_init_mode"] == "probe_zscore_top5":
        probe_init_source, _, probe_init_scores = _find_probe_init(config, backbone)

    model = LayerMixMTPModel(
        backbone=backbone,
        horizons=list(config["model"]["horizons"]),
        layer_mix_mode=config["model"]["layer_mix_mode"],
        router_init_mode=config["model"]["router_init_mode"],
        top_m=int(config["model"]["top_m"]),
        router_temperature=float(config["model"]["router_temperature"]),
        head_type=config["model"]["head_type"],
        hidden_expansion=int(config["model"]["hidden_expansion"]),
        entropy_penalty_beta=float(config["model"]["entropy_penalty_beta"]),
        probe_init_scores=probe_init_scores.to(backbone.device) if probe_init_scores is not None else None,
    ).to(backbone.device)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["train"]["lr_head"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    grad_accum = gradient_accumulation_steps(config)
    steps = planned_steps(
        int(config["data"]["train_token_quota"]),
        batch_size=int(config["hardware"]["micro_batch_size"]),
        seq_len=int(config["data"]["seq_len"]),
        grad_accum=grad_accum,
    )
    batch_size = int(config["hardware"]["micro_batch_size"])
    save_every_steps = int(config["train"].get("save_every_steps") or 0)
    val_every_steps = validation_interval_steps(config)
    last_diagnostics = None
    consumed_tokens = 0
    micro_step = 0
    start_step = 0
    last_completed_step = -1
    best_record: dict | None = None

    if resume:
        checkpoint_path = checkpoint_dir / "last.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Cannot resume MTP run without a checkpoint at {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint.get("config_digest") != config_digest_value:
            raise ValueError("Cannot resume MTP run with a config that does not match the saved checkpoint digest")
        model.load_checkpoint_state(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        restore_rng_state(checkpoint.get("rng_state"))
        start_step = int(checkpoint.get("next_step", 0))
        consumed_tokens = int(checkpoint.get("consumed_tokens", 0))
        micro_step = int(checkpoint.get("micro_step", consumed_tokens // max(1, batch_size * int(config["data"]["seq_len"]))))
        probe_source = checkpoint.get("probe_init_source")
        if probe_source:
            probe_init_source = Path(probe_source)
        last_completed_step = start_step - 1
        manifest_payload = read_json(manifest_path)
        if manifest_payload.get("best_step") is not None:
            best_record = {
                "step": int(manifest_payload["best_step"]),
                "mean_val_nll_h1_h4": float(manifest_payload["best_mean_val_nll_h1_h4"]),
                "mean_val_top1_h2_h4": float(manifest_payload["best_mean_val_top1_h2_h4"]),
            }
        mark_run_resumed(run_dir, config=config)

    def _checkpoint_payload(next_step: int) -> dict:
        return _mtp_checkpoint_payload(
            model,
            optimizer,
            next_step=next_step,
            micro_step=micro_step,
            consumed_tokens=consumed_tokens,
            probe_init_source=probe_init_source,
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
                "interpreter_path": sys.executable,
            },
        )

    def _run_validation(step_number: int) -> dict:
        future_metrics = compute_future_metrics(
            model,
            arrays["eval_future_sequences_val"],
            exp_id=config["project"]["exp_id"],
            split="val",
            horizons=list(config["model"]["horizons"]),
            evaluation_seed=int(config["eval"]["bootstrap_seed"]),
            bootstrap_samples=int(config["eval"]["bootstrap_samples"]),
            batch_size=batch_size,
        )
        validation_record = {
            "step": step_number,
            "micro_step": micro_step,
            "consumed_tokens": consumed_tokens,
            "mean_val_nll_h1_h4": float(future_metrics["aggregate_metrics"]["mean_nll_h1_h4"]),
            "mean_val_top1_h2_h4": float(future_metrics["aggregate_metrics"]["mean_top1_h2_h4"]),
            "per_horizon_val_metrics": future_metrics["metrics_by_horizon"],
            "timestamp": _utc_now(),
        }
        with val_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(validation_record) + "\n")
        return validation_record

    def _maybe_update_best(validation_record: dict) -> None:
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
                "interpreter_path": sys.executable,
            },
        )

    next_save_micro_step = next_cadence_micro_step(micro_step, save_every_steps)
    next_val_micro_step = next_cadence_micro_step(micro_step, val_every_steps)
    save_due = False
    val_due = False

    try:
        for step_idx in range(start_step, steps):
            optimizer.zero_grad()
            mean_loss = 0.0
            aggregated_per_horizon_losses: dict[str, list[float]] = {str(h): [] for h in config["model"]["horizons"]}
            boundary_diagnostics = None
            for _ in range(grad_accum):
                batch = batch_for_step(arrays["train"], batch_size, micro_step).to(backbone.device)
                outputs = model(batch, targets=batch, return_diagnostics=True)
                loss = outputs["loss"]
                (loss / grad_accum).backward()
                mean_loss += float(loss.detach().cpu().item())
                boundary_diagnostics = outputs["diagnostics"]
                for horizon, value in boundary_diagnostics["per_horizon_losses"].items():
                    aggregated_per_horizon_losses[horizon].append(float(value))
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
            optimizer.step()
            mean_loss /= grad_accum
            last_diagnostics = dict(boundary_diagnostics or {})
            last_diagnostics["per_horizon_losses"] = {
                horizon: float(sum(values) / len(values)) if values else 0.0 for horizon, values in aggregated_per_horizon_losses.items()
            }
            last_completed_step = step_idx
            with train_log.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "step": step_idx + 1,
                            "micro_step": micro_step,
                            "consumed_tokens": consumed_tokens,
                            "train_loss": mean_loss,
                            "per_horizon_train_losses": last_diagnostics["per_horizon_losses"],
                            "learning_rates": {"head": float(optimizer.param_groups[0]["lr"])},
                            "timestamp": _utc_now(),
                        }
                    )
                    + "\n"
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

        val_acceptance = compute_acceptance_metrics(
            model,
            arrays["eval_acceptance_prefixes_val"],
            exp_id=config["project"]["exp_id"],
            horizons=list(config["model"]["horizons"]),
            max_new_tokens=int(config["eval"]["acceptance_max_new_tokens"]),
            evaluation_seed=int(config["eval"]["bootstrap_seed"]),
            bootstrap_samples=int(config["eval"]["bootstrap_samples"]),
            output_dir=run_dir / "artifacts" / "traces" / "val_train_loop",
        )

        if not is_scientific_stage(config["project"]["stage"]) and best_record is None:
            fallback_record = {
                "step": steps,
                "mean_val_nll_h1_h4": 0.0,
                "mean_val_top1_h2_h4": 0.0,
            }
            _maybe_update_best(fallback_record)
        _save_last_checkpoint(steps)
        write_json(
            run_dir / "training_summary.json",
            {
                "final_loss": mean_loss,
                "probe_init_source": str(probe_init_source) if probe_init_source else None,
                "diagnostics": last_diagnostics,
                "best_checkpoint_path": str(checkpoint_dir / "best.pt"),
                "best_step": best_record["step"] if best_record else None,
                "best_mean_val_nll_h1_h4": best_record["mean_val_nll_h1_h4"] if best_record else None,
                "best_mean_val_top1_h2_h4": best_record["mean_val_top1_h2_h4"] if best_record else None,
                "final_val_acceptance_mean": float(val_acceptance["mean_accept_len"]),
                "micro_step": micro_step,
                "optimizer_step": steps,
                "consumed_tokens": consumed_tokens,
            },
        )
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
