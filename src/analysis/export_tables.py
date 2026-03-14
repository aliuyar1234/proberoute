from __future__ import annotations

import csv
import shutil
from pathlib import Path

from src.core.io_utils import ensure_dir


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_config_summary(outputs_root: Path, target: Path) -> Path:
    rows = []
    for config_path in outputs_root.glob("runs/**/resolved_config.yaml"):
        import yaml

        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if payload["project"]["stage"] in {"smoke", "template"}:
            continue
        rows.append(
            {
                "exp_id": payload["project"]["exp_id"],
                "stage": payload["project"]["stage"],
                "seed": payload["project"]["seed"],
                "backbone_name": payload["model"]["backbone_name"],
                "layer_mix_mode": payload["model"]["layer_mix_mode"],
                "router_init_mode": payload["model"]["router_init_mode"],
                "seq_len": payload["data"]["seq_len"],
            }
        )
    with target.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["exp_id", "stage", "seed", "backbone_name", "layer_mix_mode", "router_init_mode", "seq_len"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda row: (row["stage"], row["exp_id"], row["seed"])):
            writer.writerow(row)
    return target


def _write_resource_summary(registries: Path, target: Path) -> Path:
    rows = [row for row in _read_csv_rows(registries / "run_registry.csv") if row.get("evidence_eligible") in {"True", "true", True}]
    fieldnames = ["exp_id", "stage", "model_id", "seed", "token_budget_train", "best_step", "status"]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return target


def export_tables(outputs_root: Path) -> dict[str, Path]:
    registries = outputs_root / "registries"
    table_dir = outputs_root / "paper_assets" / "tables"
    ensure_dir(table_dir)
    mappings = {
        "table_screening_results.csv": registries / "screening_results.csv",
        "table_main_results.csv": registries / "main_results.csv",
        "table_ablations.csv": registries / "ablation_results.csv",
        "table_resource_summary.csv": None,
        "appendix_table_config_summary.csv": None,
    }
    written = {}
    for name, source in mappings.items():
        target = table_dir / name
        if source is None:
            if name == "table_resource_summary.csv":
                _write_resource_summary(registries, target)
            else:
                _write_config_summary(outputs_root, target)
        elif source.exists():
            shutil.copyfile(source, target)
        else:
            target.write_text("exp_id,stage\n", encoding="utf-8")
        written[name] = target
    return written
