from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.core.compatibility import dataset_manifest_matches_config
from src.core.config import load_config, validate_config
from src.core.io_utils import read_json, write_json
from src.core.manifest import processed_dataset_dir
from src.core.schema_utils import validate_payload
from src.data.dataset_stream import iter_documents, load_documents
from src.data.eval_subset_builder import build_eval_subsets
from src.data.manifest_builder import build_dataset_manifest
from src.data.shard_writer import write_split_arrays
from src.data.split_assign import assign_split
from src.data.tokenizer_pack import pack_token_buffer, tokenize_document
from src.train.trainer_common import build_tokenizer
from src.testing.local_toy_tokenizer import LocalToyTokenizer


def _trim_to_quota(buffer: list[int], quota: int) -> list[int]:
    return buffer[:quota]


def _pack_split(token_buffer: list[int], *, seq_len: int, quota: int) -> tuple[np.ndarray, int, int]:
    token_buffer = _trim_to_quota(token_buffer, quota)
    sequences, dropped = pack_token_buffer(token_buffer, seq_len)
    if sequences:
        array = np.asarray(sequences, dtype=np.int32)
    else:
        array = np.zeros((0, seq_len), dtype=np.int32)
    return array, len(token_buffer), dropped


def _rebalance_smoke_sequences(docs: list[str], tokenizer: LocalToyTokenizer, config: dict) -> dict[str, np.ndarray]:
    tokenized = [tokenize_document(doc, tokenizer, append_eos=bool(config["data"]["append_eos_between_docs"])) for doc in docs]
    flat: list[int] = []
    for doc in tokenized:
        flat.extend(doc)
    combined, _ = pack_token_buffer(flat, int(config["data"]["seq_len"]))
    if len(combined) < 3:
        raise ValueError("Smoke rebalance requires at least three packed sequences")
    array = np.asarray(combined, dtype=np.int32)
    return {
        "test": array[:1],
        "val": array[1:2],
        "train": array[2:],
    }


def prepare_data(config: dict, *, dry_run: bool = False, force_rebuild: bool = False) -> Path:
    output_dir = processed_dataset_dir(config)
    manifest_path = output_dir / "dataset_manifest.json"
    if manifest_path.exists() and not force_rebuild:
        manifest = read_json(manifest_path)
        if not dataset_manifest_matches_config(manifest, config):
            raise ValueError(f"Incompatible existing processed dataset at {output_dir}; pass --force-rebuild to replace it.")
        return output_dir

    tokenizer = build_tokenizer(config)
    if dry_run:
        return output_dir

    seq_len = int(config["data"]["seq_len"])
    split_buffers = {"train": [], "val": [], "test": []}
    raw_token_counts = {"train": 0, "val": 0, "test": 0}
    docs = load_documents(config) if isinstance(tokenizer, LocalToyTokenizer) else None
    document_stream = docs if docs is not None else iter_documents(config)

    quotas = {
        "train": int(config["data"]["train_token_quota"]),
        "val": int(config["data"]["val_token_quota"]),
        "test": int(config["data"]["test_token_quota"]),
    }
    append_eos = bool(config["data"]["append_eos_between_docs"])
    for doc in document_stream:
        split_name = assign_split(doc, int(config["data"]["split_modulus"]), dict(config["data"]["split_ranges"]))
        if len(split_buffers[split_name]) >= quotas[split_name]:
            if all(len(split_buffers[name]) >= quotas[name] for name in split_buffers):
                break
            continue
        token_ids = tokenize_document(doc, tokenizer, append_eos=append_eos)
        split_buffers[split_name].extend(token_ids)
        raw_token_counts[split_name] += len(token_ids)
        if all(len(split_buffers[name]) >= quotas[name] for name in split_buffers):
            break

    arrays = {}
    token_counts = {}
    dropped_counts = {}
    for split_name, token_buffer in split_buffers.items():
        arrays[split_name], token_counts[split_name], dropped_counts[split_name] = _pack_split(
            token_buffer,
            seq_len=seq_len,
            quota=quotas[split_name],
        )

    smoke_rebalanced = False
    if config["project"]["stage"] == "smoke" and docs is not None and (len(arrays["val"]) == 0 or len(arrays["test"]) == 0):
        arrays = _rebalance_smoke_sequences(docs, tokenizer, config)
        smoke_rebalanced = True
        token_counts = {split: int(array.size) for split, array in arrays.items()}
        dropped_counts = {split: 0 for split in arrays}
        raw_token_counts = {split: int(array.size) for split, array in arrays.items()}

    if config["project"]["stage"] != "smoke":
        missing = [split for split, array in arrays.items() if len(array) == 0]
        if missing:
            raise ValueError(f"Processed dataset is missing non-empty splits: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, array in arrays.items():
        write_split_arrays(split_name, array, output_dir)

    subsets = build_eval_subsets(
        val_array=arrays["val"],
        test_array=arrays["test"],
        output_dir=output_dir,
        future_count=int(config["eval"]["future_metrics_sequence_count"]),
        prefix_len=int(config["eval"]["acceptance_prefix_len"]),
        n_future=max(config["model"]["horizons"]),
        n_prefix=int(config["eval"]["acceptance_prefix_count"]),
    )
    manifest = build_dataset_manifest(
        config,
        token_counts_by_split=raw_token_counts,
        sequence_counts_by_split={split: int(len(array)) for split, array in arrays.items()},
        realized_token_counts_by_split={split: int(array.size) for split, array in arrays.items()},
        dropped_tail_tokens_by_split=dropped_counts,
        eval_future_sequence_count=int(subsets["future_count"]),
        eval_acceptance_prefix_count=int(subsets["acceptance_count"]),
        eval_future_sequence_count_by_split=dict(subsets["future_count_by_split"]),
        eval_acceptance_prefix_count_by_split=dict(subsets["acceptance_count_by_split"]),
        smoke_sequence_rebalance_applied=smoke_rebalanced,
    )
    validate_payload(manifest, "dataset_manifest")
    write_json(manifest_path, manifest)
    write_json(output_dir / "manifest.json", manifest)
    if hasattr(tokenizer, "token_to_id"):
        write_json(output_dir / "tokenizer_vocab.json", tokenizer.token_to_id)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    output_dir = prepare_data(config, dry_run=args.dry_run, force_rebuild=args.force_rebuild)
    print(output_dir)


if __name__ == "__main__":
    main()
