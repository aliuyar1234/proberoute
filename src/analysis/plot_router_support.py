from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.core.io_utils import read_json


def export_router_support_figure(outputs_root: Path) -> list[dict[str, str | None]]:
    router_files = list(outputs_root.glob("runs/**/eval/router_metrics.json"))
    if not router_files:
        return []
    preferred = None
    for path in router_files:
        metrics = read_json(path)
        if metrics["layer_mix_mode"] == "sparse_topm":
            preferred = path
            break
    if preferred is None:
        preferred = router_files[0]
    payload = read_json(preferred)
    figure_dir = outputs_root / "paper_assets" / "figures"
    source_dir = figure_dir / "source"
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    horizons = [str(value) for value in payload["horizons"]]
    weights = np.asarray([payload["average_weights_by_horizon"][h] for h in horizons], dtype=np.float32)
    target = figure_dir / "fig_router_support.png"
    fig, ax = plt.subplots(figsize=(5, 3))
    im = ax.imshow(weights, aspect="auto")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Horizon")
    ax.set_yticks(range(len(horizons)))
    ax.set_yticklabels(horizons)
    ax.set_title("Router support")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)
    source_copy = source_dir / preferred.name
    source_copy.write_text(preferred.read_text(encoding="utf-8"), encoding="utf-8")
    return [{"name": target.name, "path": str(target), "source_path": str(source_copy), "caption": None}]
