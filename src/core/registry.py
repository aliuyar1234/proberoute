from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .io_utils import ensure_dir


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path

