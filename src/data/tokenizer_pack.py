from __future__ import annotations

from typing import Iterable


def tokenize_document(text: str, tokenizer, append_eos: bool) -> list[int]:
    if hasattr(tokenizer, "token_to_id"):
        token_ids = tokenizer.encode(text)
    else:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    if append_eos and getattr(tokenizer, "eos_token_id", None) is not None:
        token_ids = token_ids + [int(tokenizer.eos_token_id)]
    return token_ids


def pack_token_buffer(token_buffer: list[int], seq_len: int) -> tuple[list[list[int]], int]:
    sequences: list[list[int]] = []
    usable = len(token_buffer) - (len(token_buffer) % seq_len)
    for start in range(0, usable, seq_len):
        sequences.append(token_buffer[start : start + seq_len])
    dropped_tail = len(token_buffer) - usable
    return sequences, dropped_tail


def flatten_documents(documents: Iterable[list[int]]) -> list[int]:
    buffer: list[int] = []
    for doc in documents:
        buffer.extend(doc)
    return buffer
