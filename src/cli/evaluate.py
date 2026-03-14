from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.core.io_utils import read_json
from src.core.config import load_config, validate_config
from src.core.io_utils import write_json
from src.core.schema_utils import validate_payload
from src.eval.acceptance import compute_acceptance_metrics
from src.eval.future_metrics import compute_future_metrics
from src.eval.router_analysis import compute_router_metrics
from src.models.layermix_mtp import LayerMixMTPModel
from src.train.checkpointing import load_checkpoint
from src.train.trainer_common import build_backbone, build_tokenizer, load_dataset_arrays


def _probe_init_scores_for_checkpoint(run_dir: Path, checkpoint: dict) -> torch.Tensor | None:
    probe_source = checkpoint.get("probe_init_source")
    if probe_source:
        probe_source_path = Path(probe_source)
        if not probe_source_path.is_absolute():
            probe_source_path = (Path.cwd() / probe_source_path).resolve()
        tensor_payload = load_checkpoint(probe_source_path.with_suffix(".pt"))
        return tensor_payload.get("scores")

    model_state = checkpoint.get("model_state", {})
    router_state = model_state.get("router", {}) if isinstance(model_state, dict) else {}
    router_scores = router_state.get("router_scores") if isinstance(router_state, dict) else None
    if isinstance(router_scores, torch.Tensor):
        return router_scores
    return None


def _load_model(run_dir: Path, *, checkpoint_name: str):
    config = load_config(run_dir / "resolved_config.yaml")
    validate_config(config)
    tokenizer = build_tokenizer(config)
    backbone = build_backbone(config, tokenizer)
    checkpoint = load_checkpoint(run_dir / "checkpoints" / checkpoint_name)
    probe_init_scores = None
    if config["model"]["router_init_mode"] == "probe_zscore_top5":
        probe_init_scores = _probe_init_scores_for_checkpoint(run_dir, checkpoint)
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
    model.load_checkpoint_state(checkpoint["model_state"])
    model.eval()
    return config, model


def _annotate_payload(payload: dict, *, checkpoint_used: str, checkpoint_path: Path) -> dict:
    payload = dict(payload)
    payload["checkpoint_used"] = checkpoint_used.removesuffix(".pt")
    payload["checkpoint_path"] = str(checkpoint_path)
    return payload


def evaluate_run(run_dir: Path, *, checkpoint_name: str = "best.pt") -> Path:
    config, model = _load_model(run_dir, checkpoint_name=checkpoint_name)
    arrays, _ = load_dataset_arrays(config)
    horizons = list(config["model"]["horizons"])
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    trace_root = run_dir / "artifacts" / "traces"
    trace_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoints" / checkpoint_name
    batch_size = int(config["hardware"]["micro_batch_size"])

    val_future = _annotate_payload(
        compute_future_metrics(
            model,
            arrays["eval_future_sequences_val"],
            exp_id=config["project"]["exp_id"],
            split="val",
            horizons=horizons,
            evaluation_seed=int(config["eval"]["bootstrap_seed"]),
            bootstrap_samples=int(config["eval"]["bootstrap_samples"]),
            batch_size=batch_size,
        ),
        checkpoint_used=checkpoint_name,
        checkpoint_path=checkpoint_path,
    )
    test_future = _annotate_payload(
        compute_future_metrics(
            model,
            arrays["eval_future_sequences_test"],
            exp_id=config["project"]["exp_id"],
            split="test",
            horizons=horizons,
            evaluation_seed=int(config["eval"]["bootstrap_seed"]),
            bootstrap_samples=int(config["eval"]["bootstrap_samples"]),
            batch_size=batch_size,
        ),
        checkpoint_used=checkpoint_name,
        checkpoint_path=checkpoint_path,
    )
    val_acceptance = _annotate_payload(
        compute_acceptance_metrics(
            model,
            arrays["eval_acceptance_prefixes_val"],
            exp_id=config["project"]["exp_id"],
            horizons=horizons,
            max_new_tokens=int(config["eval"]["acceptance_max_new_tokens"]),
            evaluation_seed=int(config["eval"]["bootstrap_seed"]),
            bootstrap_samples=int(config["eval"]["bootstrap_samples"]),
            output_dir=trace_root / "val",
        ),
        checkpoint_used=checkpoint_name,
        checkpoint_path=checkpoint_path,
    )
    test_acceptance = _annotate_payload(
        compute_acceptance_metrics(
            model,
            arrays["eval_acceptance_prefixes_test"],
            exp_id=config["project"]["exp_id"],
            horizons=horizons,
            max_new_tokens=int(config["eval"]["acceptance_max_new_tokens"]),
            evaluation_seed=int(config["eval"]["bootstrap_seed"]),
            bootstrap_samples=int(config["eval"]["bootstrap_samples"]),
            output_dir=trace_root / "test",
        ),
        checkpoint_used=checkpoint_name,
        checkpoint_path=checkpoint_path,
    )
    router_metrics = _annotate_payload(
        compute_router_metrics(model, exp_id=config["project"]["exp_id"], top_m=int(config["model"]["top_m"])),
        checkpoint_used=checkpoint_name,
        checkpoint_path=checkpoint_path,
    )

    validate_payload(val_future, "future_metrics")
    validate_payload(test_future, "future_metrics")
    validate_payload(val_acceptance, "acceptance_metrics")
    validate_payload(test_acceptance, "acceptance_metrics")
    validate_payload(router_metrics, "router_metrics")

    write_json(eval_dir / "val_future_metrics.json", val_future)
    write_json(eval_dir / "test_future_metrics.json", test_future)
    write_json(eval_dir / "val_acceptance_metrics.json", val_acceptance)
    write_json(eval_dir / "test_acceptance_metrics.json", test_acceptance)
    write_json(eval_dir / "router_metrics.json", router_metrics)

    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        manifest["last_evaluated_checkpoint_name"] = checkpoint_name
        manifest["last_evaluated_checkpoint_path"] = str(checkpoint_path)
        write_json(manifest_path, manifest)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint-name", default="best.pt")
    args = parser.parse_args()
    run_dir = evaluate_run(Path(args.run_dir).resolve(), checkpoint_name=args.checkpoint_name)
    print(run_dir)


if __name__ == "__main__":
    main()
