from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core.io_utils import ensure_dir


def build_eval_subsets(
    *,
    val_array: np.ndarray,
    test_array: np.ndarray,
    output_dir: Path,
    future_count: int,
    prefix_len: int,
    n_future: int,
    n_prefix: int,
) -> dict[str, Path | int | dict[str, int]]:
    ensure_dir(output_dir)
    if len(val_array) == 0 or len(test_array) == 0:
        raise ValueError("Cannot build eval subsets from empty validation or test arrays")
    if prefix_len >= test_array.shape[1]:
        raise ValueError("acceptance prefix length must be smaller than seq_len")
    payload: dict[str, Path | int | dict[str, int]] = {}
    future_counts: dict[str, int] = {}
    acceptance_counts: dict[str, int] = {}
    for split_name, array in (("val", val_array), ("test", test_array)):
        future_sequences = array[: min(future_count, len(array))]
        prefix_count = min(n_prefix, len(array))
        acceptance_prefixes = array[:prefix_count, :prefix_len]
        future_path = output_dir / f"eval_future_sequences_{split_name}.npy"
        accept_path = output_dir / f"eval_acceptance_prefixes_{split_name}.npy"
        np.save(future_path, future_sequences.astype(np.int32))
        np.save(accept_path, acceptance_prefixes.astype(np.int32))
        payload[f"future_{split_name}"] = future_path
        payload[f"acceptance_{split_name}"] = accept_path
        future_counts[split_name] = len(future_sequences)
        acceptance_counts[split_name] = len(acceptance_prefixes)
    return {
        **payload,
        "future_count": sum(future_counts.values()),
        "acceptance_count": sum(acceptance_counts.values()),
        "future_count_by_split": future_counts,
        "acceptance_count_by_split": acceptance_counts,
    }
