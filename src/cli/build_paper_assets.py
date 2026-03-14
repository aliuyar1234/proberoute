from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.assemble_registry import assemble_registries
from src.analysis.build_claim_evidence import build_claim_evidence_matrix
from src.analysis.export_tables import export_tables
from src.analysis.plot_heatmaps import export_probe_heatmap_figures
from src.analysis.plot_main_results import export_main_results_figure
from src.analysis.plot_router_support import export_router_support_figure
from src.core.io_utils import write_json
from src.core.schema_utils import validate_payload


def _export_system_overview(outputs_root: Path) -> list[dict[str, str | None]]:
    import matplotlib.pyplot as plt

    figure_dir = outputs_root / "paper_assets" / "figures"
    source_dir = figure_dir / "source"
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    target = figure_dir / "fig_system_overview.png"
    fig, ax = plt.subplots(figsize=(8, 1.8))
    ax.axis("off")
    labels = [
        "data prep",
        "probes",
        "probe init",
        "screening",
        "finalist",
        "final runs",
        "eval/assets",
        "paper",
    ]
    for idx, label in enumerate(labels):
        ax.text(idx, 0.5, label, ha="center", va="center", bbox={"boxstyle": "round", "facecolor": "#f2f2f2"})
        if idx < len(labels) - 1:
            ax.annotate("", xy=(idx + 0.7, 0.5), xytext=(idx + 0.3, 0.5), arrowprops={"arrowstyle": "->"})
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)
    source_path = source_dir / "system_overview.txt"
    source_path.write_text(" -> ".join(labels) + "\n", encoding="utf-8")
    return [{"name": target.name, "path": str(target), "source_path": str(source_path), "caption": None}]


def _export_acceptance_distribution(outputs_root: Path) -> list[dict[str, str | None]]:
    import matplotlib.pyplot as plt
    import numpy as np

    figure_dir = outputs_root / "paper_assets" / "figures"
    source_dir = figure_dir / "source"
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    acceptance_paths = []
    for candidate in outputs_root.glob("runs/**/eval/test_acceptance_metrics.json"):
        run_dir = candidate.parent.parent
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("stage") in {"smoke", "template"} or manifest.get("status") != "completed":
            continue
        acceptance_paths.append(candidate)
    if not acceptance_paths:
        return []
    labels = []
    values = []
    for path in acceptance_paths[:4]:
        payload = __import__("json").loads(path.read_text(encoding="utf-8"))
        labels.append(payload["exp_id"])
        histogram = payload.get("accept_len_histogram", {})
        expanded = []
        for key, count in histogram.items():
            expanded.extend([float(key)] * int(count))
        values.append(expanded or [payload["mean_accept_len"]])
        (source_dir / f"{payload['exp_id']}_acceptance.json").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    target = figure_dir / "fig_acceptance_distribution.png"
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(values, bins=np.arange(0, 6) - 0.5, label=labels, alpha=0.6)
    ax.set_xlabel("Accepted prefix length")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("Acceptance distribution")
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)
    return [{"name": target.name, "path": str(target), "source_path": str(source_dir), "caption": None}]


def build_paper_assets(outputs_root: Path) -> Path:
    assemble_registries(outputs_root)
    tables = export_tables(outputs_root)
    claim_matrix = build_claim_evidence_matrix(outputs_root)
    figures = []
    figures.extend(_export_system_overview(outputs_root))
    figures.extend(export_probe_heatmap_figures(outputs_root))
    figures.extend(export_main_results_figure(outputs_root))
    figures.extend(_export_acceptance_distribution(outputs_root))
    figures.extend(export_router_support_figure(outputs_root))
    for figure_path in outputs_root.glob("runs/**/*.png"):
        manifest_path = figure_path
        while manifest_path != outputs_root and not (manifest_path / "run_manifest.json").exists():
            manifest_path = manifest_path.parent
        if not (manifest_path / "run_manifest.json").exists():
            continue
        manifest = __import__("json").loads((manifest_path / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("stage") in {"smoke", "template"} or manifest.get("status") != "completed":
            continue
        figures.append({"name": figure_path.name, "path": str(figure_path), "source_path": str(figure_path), "caption": None})
    payload = {
        "figures": figures,
        "tables": [{"name": name, "path": str(path), "source_path": str(path), "caption": None} for name, path in tables.items()],
        "appendix": [{"name": "claim_evidence_matrix", "path": str(claim_matrix), "source_path": str(claim_matrix)}],
    }
    validate_payload(payload, "paper_asset_manifest")
    manifest_path = outputs_root / "paper_assets" / "paper_asset_manifest.json"
    write_json(manifest_path, payload)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", required=True)
    args = parser.parse_args()
    path = build_paper_assets(Path(args.outputs_root).resolve())
    print(path)


if __name__ == "__main__":
    main()
