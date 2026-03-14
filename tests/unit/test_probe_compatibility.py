from __future__ import annotations

from pathlib import Path

import pytest

import torch

from tests.helpers import REPO_ROOT, first_existing_attr, load_yaml, make_fake_run, write_json


def test_probe_compatibility_rejects_mismatched_dataset_id() -> None:
    import src.core.compatibility as compatibility

    checker = first_existing_attr(
        compatibility,
        [
            "assert_probe_init_compatible",
            "validate_probe_init_compatibility",
            "check_probe_init_compatible",
        ],
    )
    config = load_yaml(REPO_ROOT / "configs" / "smoke_local_tiny.yaml")
    config["data"]["dataset_name"] = "local_fixture_text"
    config["data"]["dataset_config"] = "tiny_v1"
    config["data"]["seq_len"] = 32
    config["model"]["horizons"] = [1, 2, 3, 4]
    config["model"]["probe_init_metric"] = "top5"
    config["model"]["backbone_name"] = "local_toy_gpt"
    probe_init = {
        "backbone_name": "local_toy_gpt",
        "dataset_id": "some_other_dataset",
        "seq_len": 32,
        "horizons": [1, 2, 3, 4],
        "init_metric": "top5",
    }

    with pytest.raises((ValueError, AssertionError)):
        checker(config=config, probe_init=probe_init)


def test_probe_compatibility_allows_seq_len_mismatch_when_dataset_metadata_matches() -> None:
    from src.core.compatibility import ensure_probe_init_compatible, validate_probe_init_compatibility
    from src.core.config import load_config

    config = load_config(REPO_ROOT / "configs" / "screen_dense_whs_probe_init_1b.yaml")
    probe_init = {
        "model_id": "EleutherAI-pythia-1b",
        "backbone_name": "EleutherAI/pythia-1b",
        "dataset_name": "HuggingFaceFW/fineweb-edu",
        "dataset_config": "sample-10BT",
        "dataset_id": "huggingfacefw-fineweb-edu__sample-10bt__eleutherai-pythia-1b__sl1024",
        "seq_len": 1024,
        "horizons": [1, 2, 3, 4],
        "num_layers": 16,
        "metric": "top5",
        "init_metric": "top5",
    }

    validate_probe_init_compatibility(config=config, probe_init=probe_init)
    ensure_probe_init_compatible(
        probe_init,
        config,
        model_id="EleutherAI-pythia-1b",
        num_layers=16,
    )


def test_probe_compatibility_still_rejects_dataset_metadata_mismatch() -> None:
    from src.core.compatibility import ensure_probe_init_compatible
    from src.core.config import load_config

    config = load_config(REPO_ROOT / "configs" / "screen_dense_whs_probe_init_1b.yaml")
    probe_init = {
        "model_id": "EleutherAI-pythia-1b",
        "dataset_name": "other_dataset",
        "dataset_config": "sample-10BT",
        "seq_len": 1024,
        "horizons": [1, 2, 3, 4],
        "num_layers": 16,
        "metric": "top5",
    }

    with pytest.raises(ValueError, match="dataset_name"):
        ensure_probe_init_compatible(
            probe_init,
            config,
            model_id="EleutherAI-pythia-1b",
            num_layers=16,
        )


def test_find_probe_init_accepts_authoritative_probe_from_shorter_seq_len(tmp_path: Path) -> None:
    from src.core.config import load_config
    from src.train.mtp_trainer import _find_probe_init

    output_root = tmp_path / "outputs"
    run_dir = make_fake_run(
        output_root,
        exp_id="PROBE_1B",
        stage="probe",
        model_id="EleutherAI-pythia-1b",
        with_probe=True,
    )
    probe_dir = run_dir / "artifacts" / "probe"
    write_json(
        probe_dir / "probe_init.json",
        {
            "model_id": "EleutherAI-pythia-1b",
            "backbone_name": "EleutherAI/pythia-1b",
            "dataset_name": "HuggingFaceFW/fineweb-edu",
            "dataset_config": "sample-10BT",
            "dataset_id": "huggingfacefw-fineweb-edu__sample-10bt__eleutherai-pythia-1b__sl1024",
            "seq_len": 1024,
            "horizons": [1, 2, 3, 4],
            "num_layers": 16,
            "metric": "top5",
            "init_metric": "top5",
        },
    )
    torch.save({"scores": torch.zeros((4, 16), dtype=torch.float32)}, probe_dir / "probe_init.pt")

    config = load_config(REPO_ROOT / "configs" / "screen_dense_whs_probe_init_1b.yaml")
    config["project"]["output_root"] = str(output_root)

    dummy_backbone = type(
        "DummyBackbone",
        (),
        {
            "model_slug": lambda self: "EleutherAI-pythia-1b",
            "num_layers": lambda self: 16,
        },
    )()

    probe_path, payload, scores = _find_probe_init(config, dummy_backbone)

    assert probe_path == probe_dir / "probe_init.json"
    assert payload["seq_len"] == 1024
    assert tuple(scores.shape) == (4, 16)
