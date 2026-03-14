from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def export_main_results_figure(outputs_root: Path) -> list[dict[str, str | None]]:
    main_results = outputs_root / "registries" / "main_results.csv"
    if not main_results.exists():
        return []
    rows = list(csv.DictReader(main_results.open(encoding="utf-8")))
    if not rows:
        return []
    figure_dir = outputs_root / "paper_assets" / "figures"
    source_dir = figure_dir / "source"
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    labels = [row["exp_id"] for row in rows]
    top1 = [float(row["test_mean_top1_h2_h4"]) for row in rows]
    accept = [float(row["test_mean_accept_len"]) for row in rows]
    x = range(len(rows))
    target = figure_dir / "fig_main_results.png"
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar([value - 0.15 for value in x], top1, width=0.3, label="mean top-1 h2-h4")
    ax.bar([value + 0.15 for value in x], accept, width=0.3, label="mean accept len")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    ax.set_title("Final comparison")
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)
    source_copy = source_dir / "main_results.csv"
    source_copy.write_text(main_results.read_text(encoding="utf-8"), encoding="utf-8")
    return [{"name": target.name, "path": str(target), "source_path": str(source_copy), "caption": None}]
