from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any, Iterator

from .split_assign import normalize_text


def resolve_text_field(sample: dict[str, Any], priority: list[str]) -> str:
    for field in priority:
        value = sample.get(field)
        if isinstance(value, str) and value.strip():
            return value
    raise KeyError(f"No non-empty text field found for priorities={priority}")


def load_documents(config: dict[str, Any]) -> list[str]:
    data = config["data"]
    if data["local_path"]:
        path = Path(data["local_path"])
        documents: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            try:
                text = resolve_text_field(raw, list(data["text_field_priority"]))
            except KeyError:
                continue
            normalized = normalize_text(text)
            if normalized.strip():
                documents.append(normalized)
        return documents
    return list(iter_documents(config))


def _iter_remote_documents(config: dict[str, Any]) -> Iterator[str]:
    try:
        datasets = import_module("datasets")
    except ImportError as exc:
        raise RuntimeError("Remote dataset loading requires the `datasets` package to be installed.") from exc

    data = config["data"]
    stream = datasets.load_dataset(
        data["dataset_name"],
        data["dataset_config"],
        split="train",
        streaming=True,
    )
    for sample in stream:
        try:
            text = resolve_text_field(sample, list(data["text_field_priority"]))
        except KeyError:
            continue
        normalized = normalize_text(text)
        if normalized.strip():
            yield normalized


def iter_documents(config: dict[str, Any]) -> Iterator[str]:
    if config["data"]["local_path"]:
        yield from load_documents({"data": {**config["data"], "local_path": config["data"]["local_path"]}})
        return
    yield from _iter_remote_documents(config)
