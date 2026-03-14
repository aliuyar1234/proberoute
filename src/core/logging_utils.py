from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .io_utils import ensure_dir


def append_markdown_log(path: Path, section: str) -> None:
    ensure_dir(path.parent)
    timestamp = datetime.now(timezone.utc).isoformat()
    if path.exists():
        prefix = path.read_text(encoding="utf-8").rstrip() + "\n\n"
    else:
        prefix = ""
    path.write_text(prefix + f"## {timestamp}\n{section.strip()}\n", encoding="utf-8")

