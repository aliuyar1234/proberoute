from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import os
from pathlib import Path

from src.core.config import deep_merge, load_config
from src.core.io_utils import read_yaml, write_json, write_yaml
from src.core.schema_utils import validate_payload


BASELINE_PREFERENCE = [
    "BASE_LAST_MLP_1B",
    "BASE_DENSE_WHS_PROBE_1B",
    "BASE_DENSE_WHS_RANDOM_1B",
    "BASE_LAST_LINEAR_1B",
]
REQUIRED_BASELINE_IDS = set(BASELINE_PREFERENCE)


def _parse_float(value: str) -> float:
    return float(value) if value not in {"", None} else float("-inf")


def _select_row(rows: list[dict[str, str]]) -> dict[str, str]:
    candidates = [
        row
        for row in rows
        if row["stage"] == "screen"
        and row["exp_id"] != "MAIN_SPARSE_PROBE_1B"
        and row["status"] == "completed"
        and row["val_mean_top1_h2_h4"] not in {"", None}
        and row.get("checkpoint_used") == "best"
    ]
    if not candidates:
        raise ValueError("No completed screening baselines available for finalist selection")
    observed_ids = {row["exp_id"] for row in candidates}
    missing_ids = sorted(REQUIRED_BASELINE_IDS - observed_ids)
    if missing_ids:
        raise ValueError(f"Cannot select finalist without all required completed real screening baselines: {missing_ids}")

    def sort_key(row: dict[str, str]) -> tuple:
        exp_id = row["exp_id"]
        try:
            preference = BASELINE_PREFERENCE.index(exp_id)
        except ValueError:
            preference = len(BASELINE_PREFERENCE)
        return (
            -_parse_float(row["val_mean_top1_h2_h4"]),
            -_parse_float(row["val_mean_accept_len"]),
            _parse_float(row["val_mean_nll_h1_h4"]),
            preference,
            exp_id,
        )
    return sorted(candidates, key=sort_key)[0]


def select_finalist(outputs_root: Path, emit_config: Path) -> Path:
    screening_path = outputs_root / "registries" / "screening_results.csv"
    rows = list(csv.DictReader(screening_path.open(encoding="utf-8")))
    selected = _select_row(rows)
    selected_run_dir = Path(selected["run_dir"])
    resolved_config = load_config(selected_run_dir / "resolved_config.yaml")
    template = read_yaml(Path("configs/final_best_baseline_template.yaml"))
    source_config_path = selected.get("source_config_path") or read_yaml(selected_run_dir / "resolved_config.yaml").get("_config_path")
    if source_config_path:
        source_path = Path(source_config_path)
        if not source_path.is_absolute():
            source_path = (Path.cwd() / source_path).resolve()
        try:
            inherit_from = os.path.relpath(source_path, start=emit_config.parent.resolve())
        except ValueError:
            inherit_from = str(source_path)
        generated = {"inherit_from": inherit_from}
        generated = deep_merge(generated, template)
        generated["model"] = {
            "layer_mix_mode": resolved_config["model"]["layer_mix_mode"],
            "router_init_mode": resolved_config["model"]["router_init_mode"],
        }
        generated["project"] = {
            "exp_id": "FINAL_BEST_BASELINE_1B",
            "stage": "final",
            "priority": "must",
        }
    else:
        generated = deep_merge(resolved_config, template)
        generated["project"]["exp_id"] = "FINAL_BEST_BASELINE_1B"
        generated["project"]["stage"] = "final"
        generated["project"]["priority"] = "must"
    generated["selection_provenance"] = {
        "authoritative": True,
        "selected_baseline_exp_id": selected["exp_id"],
        "selected_run_dir": selected["run_dir"],
        "selected_best_checkpoint_path": selected.get("best_checkpoint_path"),
        "selection_metrics": {
            "val_mean_top1_h2_h4": float(selected["val_mean_top1_h2_h4"]),
            "val_mean_accept_len": float(selected["val_mean_accept_len"]),
            "val_mean_nll_h1_h4": float(selected["val_mean_nll_h1_h4"]),
        },
        "checkpoint_used": selected.get("checkpoint_used"),
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    write_yaml(emit_config, generated)
    selection = {
        "selected_baseline_exp_id": selected["exp_id"],
        "selected_run_dir": selected["run_dir"],
        "selection_metric": "val_mean_top1_h2_h4",
        "selection_value": float(selected["val_mean_top1_h2_h4"]),
        "selected_best_checkpoint_path": selected.get("best_checkpoint_path"),
        "tie_break_trace": {
            "selected_baseline_preference_rank": BASELINE_PREFERENCE.index(selected["exp_id"]) if selected["exp_id"] in BASELINE_PREFERENCE else None,
            "val_mean_accept_len": selected["val_mean_accept_len"],
            "val_mean_nll_h1_h4": selected["val_mean_nll_h1_h4"],
        },
        "source_metrics": {
            "val_mean_top1_h2_h4": float(selected["val_mean_top1_h2_h4"]),
            "val_mean_accept_len": float(selected["val_mean_accept_len"]),
            "val_mean_nll_h1_h4": float(selected["val_mean_nll_h1_h4"]),
            "test_mean_top1_h2_h4": float(selected["test_mean_top1_h2_h4"]),
            "test_mean_accept_len": float(selected["test_mean_accept_len"]),
            "test_mean_nll_h1_h4": float(selected["test_mean_nll_h1_h4"]),
        },
        "checkpoint_used": selected.get("checkpoint_used"),
        "authoritative": True,
        "generated_config_path": str(emit_config),
        "source_screening_registry": str(screening_path),
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    validate_payload(selection, "finalist_selection")
    write_json(outputs_root / "registries" / "finalist_selection.json", selection)
    return emit_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", required=True)
    parser.add_argument("--emit-config", required=True)
    args = parser.parse_args()
    target = select_finalist(Path(args.outputs_root).resolve(), Path(args.emit_config).resolve())
    print(target)


if __name__ == "__main__":
    main()
