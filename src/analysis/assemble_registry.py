from __future__ import annotations

from pathlib import Path

from src.core.io_utils import read_json
from src.core.registry import write_csv
from src.core.schema_utils import validate_payload
from src.train.trainer_common import is_venv_interpreter


def _is_archived_run(run_dir: Path, outputs_root: Path) -> bool:
    try:
        relative = run_dir.relative_to(outputs_root)
    except ValueError:
        return False
    return relative.parts[:1] == ("archive",)


def _fieldnames(rows: list[dict], fallback: list[str]) -> list[str]:
    if not rows:
        return fallback
    ordered: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in ordered:
                ordered.append(key)
    return ordered


def _eval_paths(run_dir: Path) -> dict[str, Path]:
    eval_dir = run_dir / "eval"
    return {
        "val_future": eval_dir / "val_future_metrics.json",
        "test_future": eval_dir / "test_future_metrics.json",
        "val_acceptance": eval_dir / "val_acceptance_metrics.json",
        "test_acceptance": eval_dir / "test_acceptance_metrics.json",
        "router": eval_dir / "router_metrics.json",
    }


def _read_validated_metrics(run_dir: Path) -> dict[str, dict] | None:
    paths = _eval_paths(run_dir)
    if not all(path.exists() for path in paths.values()):
        return None
    payloads = {
        "val_future": read_json(paths["val_future"]),
        "test_future": read_json(paths["test_future"]),
        "val_acceptance": read_json(paths["val_acceptance"]),
        "test_acceptance": read_json(paths["test_acceptance"]),
        "router": read_json(paths["router"]),
    }
    validate_payload(payloads["val_future"], "future_metrics")
    validate_payload(payloads["test_future"], "future_metrics")
    validate_payload(payloads["val_acceptance"], "acceptance_metrics")
    validate_payload(payloads["test_acceptance"], "acceptance_metrics")
    validate_payload(payloads["router"], "router_metrics")
    checkpoint_used = {
        payloads["val_future"].get("checkpoint_used"),
        payloads["test_future"].get("checkpoint_used"),
        payloads["val_acceptance"].get("checkpoint_used"),
        payloads["test_acceptance"].get("checkpoint_used"),
        payloads["router"].get("checkpoint_used"),
    }
    if checkpoint_used != {"best"}:
        return None
    return payloads


def _probe_artifacts_exist(run_dir: Path) -> bool:
    probe_dir = run_dir / "artifacts" / "probe"
    required = [
        probe_dir / "probe_scores.csv",
        probe_dir / "probe_heatmap_top1.png",
        probe_dir / "probe_heatmap_top5.png",
        probe_dir / "probe_heatmap_nll.png",
        probe_dir / "probe_init.json",
        probe_dir / "probe_init.pt",
    ]
    return all(path.exists() for path in required)


def _authoritative_probe(manifest: dict, run_dir: Path, outputs_root: Path) -> tuple[bool, str]:
    if _is_archived_run(run_dir, outputs_root):
        return False, "archived"
    if manifest.get("stage") != "probe":
        return False, "not_probe_stage"
    if manifest.get("status") != "completed":
        return False, "not_completed"
    if not is_venv_interpreter(manifest.get("interpreter_path")):
        return False, "non_canonical_interpreter"
    if manifest.get("best_step") is None or not manifest.get("best_checkpoint_path"):
        return False, "missing_best_checkpoint_metadata"
    if not Path(manifest["best_checkpoint_path"]).exists():
        return False, "missing_best_checkpoint_file"
    if not (run_dir / "val_metrics.jsonl").exists():
        return False, "missing_validation_log"
    if not _probe_artifacts_exist(run_dir):
        return False, "missing_probe_artifacts"
    return True, "authoritative"


def _authoritative_scientific_run(manifest: dict, run_dir: Path, outputs_root: Path) -> tuple[bool, str, dict | None]:
    if _is_archived_run(run_dir, outputs_root):
        return False, "archived", None
    if manifest.get("stage") in {"smoke", "template", "probe"}:
        return False, "non_scientific_stage", None
    if manifest.get("status") != "completed":
        return False, "not_completed", None
    if not is_venv_interpreter(manifest.get("interpreter_path")):
        return False, "non_canonical_interpreter", None
    if manifest.get("best_step") is None or not manifest.get("best_checkpoint_path"):
        return False, "missing_best_checkpoint_metadata", None
    best_checkpoint_path = Path(manifest["best_checkpoint_path"])
    if not best_checkpoint_path.exists():
        return False, "missing_best_checkpoint_file", None
    payloads = _read_validated_metrics(run_dir)
    if payloads is None:
        return False, "missing_authoritative_eval", None
    return True, "authoritative", payloads


def assemble_registries(outputs_root: Path) -> dict[str, Path]:
    run_rows = []
    probe_rows = []
    screening_rows = []
    main_rows = []
    ablation_rows = []
    for manifest_path in outputs_root.glob("runs/**/run_manifest.json"):
        run_dir = manifest_path.parent
        manifest = read_json(manifest_path)
        archived = _is_archived_run(run_dir, outputs_root)
        row = {
            "exp_id": manifest["exp_id"],
            "stage": manifest["stage"],
            "priority": manifest["priority"],
            "model_id": manifest["model_id"],
            "seed": manifest["seed"],
            "run_dir": str(run_dir),
            "layer_mix_mode": manifest["layer_mix_mode"],
            "router_init_mode": manifest["router_init_mode"],
            "status": manifest["status"],
            "token_budget_train": manifest["token_budget_train"],
            "token_budget_val": manifest["token_budget_val"],
            "token_budget_test": manifest["token_budget_test"],
            "best_step": manifest.get("best_step", ""),
            "best_checkpoint_path": manifest.get("best_checkpoint_path", ""),
            "best_mean_val_nll_h1_h4": manifest.get("best_mean_val_nll_h1_h4", ""),
            "best_mean_val_top1_h2_h4": manifest.get("best_mean_val_top1_h2_h4", ""),
            "resume_count": manifest.get("resume_count", 0),
            "interpreter_path": manifest.get("interpreter_path", ""),
            "source_config_path": manifest.get("source_config_path", ""),
            "archived": archived,
            "evidence_eligible": False,
            "authority_reason": "operational_only",
        }

        authoritative_probe, probe_reason = _authoritative_probe(manifest, run_dir, outputs_root)
        authoritative_scientific, scientific_reason, scientific_payloads = _authoritative_scientific_run(manifest, run_dir, outputs_root)

        if authoritative_probe:
            row["evidence_eligible"] = True
            row["authority_reason"] = "authoritative_probe"
            probe_rows.append(
                {
                    "exp_id": manifest["exp_id"],
                    "run_dir": str(run_dir),
                    "best_step": manifest["best_step"],
                    "best_checkpoint_path": manifest["best_checkpoint_path"],
                    "probe_scores_path": str(run_dir / "artifacts" / "probe" / "probe_scores.csv"),
                    "probe_init_json_path": str(run_dir / "artifacts" / "probe" / "probe_init.json"),
                    "probe_init_tensor_path": str(run_dir / "artifacts" / "probe" / "probe_init.pt"),
                }
            )
        elif scientific_reason != "non_scientific_stage":
            row["authority_reason"] = probe_reason

        if authoritative_scientific and scientific_payloads is not None:
            row.update(
                {
                    "val_mean_top1_h2_h4": scientific_payloads["val_future"]["aggregate_metrics"]["mean_top1_h2_h4"],
                    "val_mean_accept_len": scientific_payloads["val_acceptance"]["mean_accept_len"],
                    "val_mean_nll_h1_h4": scientific_payloads["val_future"]["aggregate_metrics"]["mean_nll_h1_h4"],
                    "test_mean_top1_h2_h4": scientific_payloads["test_future"]["aggregate_metrics"]["mean_top1_h2_h4"],
                    "test_mean_accept_len": scientific_payloads["test_acceptance"]["mean_accept_len"],
                    "test_mean_nll_h1_h4": scientific_payloads["test_future"]["aggregate_metrics"]["mean_nll_h1_h4"],
                    "checkpoint_used": scientific_payloads["test_future"]["checkpoint_used"],
                    "evidence_eligible": True,
                    "authority_reason": "authoritative_scientific",
                }
            )
            if manifest["stage"] == "screen":
                screening_rows.append(dict(row))
            elif manifest["stage"] == "final":
                main_rows.append(dict(row))
            elif manifest["stage"] in {"ablation", "confirm"}:
                ablation_rows.append(dict(row))
        elif manifest["stage"] not in {"smoke", "template", "probe"}:
            row["authority_reason"] = scientific_reason

        run_rows.append(row)

    stage_rank = {"probe": 0, "screen": 1, "final": 2, "ablation": 3, "confirm": 4, "optional": 5, "smoke": 6, "template": 7}

    def _sort_rows(rows: list[dict]) -> list[dict]:
        return sorted(rows, key=lambda row: (stage_rank.get(row["stage"], 99), row["exp_id"], row["model_id"], int(row["seed"])))

    run_rows = _sort_rows(run_rows)
    screening_rows = _sort_rows(screening_rows)
    main_rows = _sort_rows(main_rows)
    ablation_rows = _sort_rows(ablation_rows)
    probe_rows = sorted(probe_rows, key=lambda row: (row["exp_id"], row["run_dir"]))
    registries_dir = outputs_root / "registries"
    paths = {
        "run_registry": write_csv(registries_dir / "run_registry.csv", run_rows, _fieldnames(run_rows, ["exp_id", "stage", "status"])),
        "probe_registry": write_csv(
            registries_dir / "probe_registry.csv",
            probe_rows,
            _fieldnames(probe_rows, ["exp_id", "run_dir", "probe_scores_path"]),
        ),
        "screening_results": write_csv(
            registries_dir / "screening_results.csv",
            screening_rows,
            _fieldnames(screening_rows, ["exp_id", "stage"]),
        ),
        "main_results": write_csv(registries_dir / "main_results.csv", main_rows, _fieldnames(main_rows, ["exp_id", "stage"])),
        "ablation_results": write_csv(
            registries_dir / "ablation_results.csv",
            ablation_rows,
            _fieldnames(ablation_rows, ["exp_id", "stage"]),
        ),
    }
    return paths
