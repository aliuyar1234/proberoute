from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core.io_utils import ensure_dir


def write_split_arrays(split_name: str, sequences: np.ndarray, output_dir: Path) -> Path:
    ensure_dir(output_dir)
    path = output_dir / f"{split_name}.npy"
    np.save(path, sequences.astype(np.int32))
    return path

