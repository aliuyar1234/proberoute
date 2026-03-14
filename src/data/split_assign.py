from __future__ import annotations

import hashlib


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")


def assign_split(text: str, modulus: int, ranges: dict[str, list[int]]) -> str:
    normalized = normalize_text(text)
    bucket = int(hashlib.sha1(normalized.encode("utf-8")).hexdigest(), 16) % modulus
    for split_name, bounds in ranges.items():
        lo, hi = bounds
        if lo <= bucket <= hi:
            return split_name
    raise ValueError(f"Split bucket {bucket} did not match any configured range")

