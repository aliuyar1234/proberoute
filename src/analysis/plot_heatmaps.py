from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def export_probe_heatmap_figures(outputs_root: Path) -> list[dict[str, str | None]]:
    figure_dir = outputs_root / "paper_assets" / "figures"
    source_dir = figure_dir / "source"
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, str | None]] = []
    for probe_csv in outputs_root.glob("runs/**/artifacts/probe/probe_scores.csv"):
        rows = list(csv.DictReader(probe_csv.open(encoding="utf-8")))
        if not rows:
            continue
        horizons = sorted({int(row["horizon"]) for row in rows})
        layers = sorted({int(row["layer"]) for row in rows})
        grid = np.zeros((len(layers), len(horizons)), dtype=np.float32)
        for row in rows:
            grid[int(row["layer"]), horizons.index(int(row["horizon"]))] = float(row["top1"])
        run_name = probe_csv.parents[4].name.lower()
        target = figure_dir / f"fig_probe_heatmap_{run_name}_top1.png"
        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(grid, aspect="auto")
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Layer")
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels([str(value) for value in horizons])
        ax.set_title(f"{run_name} probe heatmap")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(target)
        plt.close(fig)
        source_copy = source_dir / f"{run_name}_{probe_csv.name}"
        source_copy.write_text(probe_csv.read_text(encoding="utf-8"), encoding="utf-8")
        entries.append({"name": target.name, "path": str(target), "source_path": str(source_copy), "caption": None})
    return entries
