from __future__ import annotations

import json

import jsonschema
import numpy as np

from tests.helpers import REPO_ROOT, dataset_dir_from_output_root, run_python_module, write_smoke_config


def test_prepare_data_smoke_writes_processed_arrays_and_manifest(tmp_path) -> None:
    config_path, config = write_smoke_config(tmp_path, exp_id="SMOKE_PREP_ONLY")
    output_root = tmp_path / "outputs"

    run_python_module("src.cli.prepare_data", "--config", str(config_path))

    dataset_dir = dataset_dir_from_output_root(output_root)
    manifest_path = dataset_dir / "manifest.json"
    train_path = dataset_dir / "train.npy"
    val_path = dataset_dir / "val.npy"
    test_path = dataset_dir / "test.npy"
    val_future_path = dataset_dir / "eval_future_sequences_val.npy"
    test_future_path = dataset_dir / "eval_future_sequences_test.npy"
    val_prefix_path = dataset_dir / "eval_acceptance_prefixes_val.npy"
    test_prefix_path = dataset_dir / "eval_acceptance_prefixes_test.npy"

    for path in [manifest_path, train_path, val_path, test_path, val_future_path, test_future_path, val_prefix_path, test_prefix_path]:
        assert path.exists(), f"missing expected data artifact: {path}"

    train = np.load(train_path)
    val = np.load(val_path)
    test = np.load(test_path)
    assert train.ndim == 2 and train.shape[1] == config["data"]["seq_len"]
    assert val.ndim == 2 and val.shape[1] == config["data"]["seq_len"]
    assert test.ndim == 2 and test.shape[1] == config["data"]["seq_len"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = json.loads((REPO_ROOT / "schemas" / "dataset_manifest.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(manifest, schema)
