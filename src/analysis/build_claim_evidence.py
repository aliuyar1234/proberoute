from __future__ import annotations

import csv
from pathlib import Path

from src.core.io_utils import ensure_dir


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _existing(paths: list[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def _status_for_paths(paths: list[Path]) -> str:
    return "supported" if paths else "unsupported"


def _format_evidence(paths: list[Path], outputs_root: Path) -> str:
    if not paths:
        return "none"
    return ", ".join(str(path.relative_to(outputs_root)) for path in paths)


def build_claim_evidence_matrix(outputs_root: Path) -> Path:
    appendix_dir = outputs_root / "paper_assets" / "appendix"
    ensure_dir(appendix_dir)
    registries_dir = outputs_root / "registries"
    probe_rows = _read_rows(registries_dir / "probe_registry.csv")
    main_rows = _read_rows(registries_dir / "main_results.csv")
    ablation_rows = _read_rows(registries_dir / "ablation_results.csv")

    probe_paths = _existing([Path(row["probe_scores_path"]) for row in probe_rows if row.get("probe_scores_path")])
    main_paths = _existing([registries_dir / "main_results.csv"]) if main_rows else []
    acceptance_paths = _existing(
        [Path(row["run_dir"]) / "eval" / "test_acceptance_metrics.json" for row in main_rows if row.get("run_dir")]
    )
    ablation_paths = _existing([registries_dir / "ablation_results.csv"]) if ablation_rows else []

    rows = [
        ("C1", probe_paths, _status_for_paths(probe_paths)),
        ("C2", main_paths, _status_for_paths(main_paths)),
        ("C3", acceptance_paths, _status_for_paths(acceptance_paths)),
        ("C4", ablation_paths, _status_for_paths(ablation_paths)),
    ]

    path = appendix_dir / "claim_evidence_matrix.md"
    lines = [
        "# Claim Evidence Matrix",
        "",
        "| Claim | Evidence | Status |",
        "|---|---|---|",
    ]
    for claim_id, evidence_paths, status in rows:
        lines.append(f"| {claim_id} | {_format_evidence(evidence_paths, outputs_root)} | {status} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
