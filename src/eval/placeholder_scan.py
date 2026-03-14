from __future__ import annotations

from pathlib import Path


PLACEHOLDERS = ("TODO", "TBD", "{TITLE}", "{ABSTRACT}", "FIXME")


def scan_placeholders(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [token for token in PLACEHOLDERS if token in text]

