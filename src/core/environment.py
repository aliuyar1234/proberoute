from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Any

import torch

from .io_utils import write_json
from .manifest import utc_now


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def capture_environment_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "created_at": utc_now(),
        "python_version": platform.python_version(),
        "interpreter_path": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": _package_version("transformers"),
        "datasets_version": _package_version("datasets"),
        "accelerate_version": _package_version("accelerate"),
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": config["project"]["seed"],
        "tokenizer_name": config["data"]["tokenizer_name"],
        "backbone_name": config["model"]["backbone_name"],
    }


def write_environment_snapshot(path: Path, config: dict[str, Any]) -> Path:
    payload = capture_environment_snapshot(config)
    return write_json(path, payload)
