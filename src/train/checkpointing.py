from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.core.io_utils import ensure_dir

PAUSE_REQUEST_FILENAME = "pause.request"


def save_checkpoint(path: Path, payload: dict[str, Any]) -> Path:
    ensure_dir(path.parent)
    temp_path = path.parent / f"{path.name}.tmp"
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return path


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(
            (
                numpy_state["bit_generator"],
                np.asarray(numpy_state["state"], dtype=np.uint32),
                int(numpy_state["pos"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
    torch_state = state.get("torch")
    if torch_state is not None:
        torch.random.set_rng_state(torch_state)
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def pause_request_path(run_dir: Path) -> Path:
    return run_dir / "control" / PAUSE_REQUEST_FILENAME


def request_pause(run_dir: Path) -> Path:
    path = pause_request_path(run_dir)
    ensure_dir(path.parent)
    path.write_text("pause_requested\n", encoding="utf-8")
    return path


def consume_pause_request(run_dir: Path) -> bool:
    path = pause_request_path(run_dir)
    if not path.exists():
        return False
    path.unlink()
    return True
