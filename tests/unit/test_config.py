from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.helpers import REPO_ROOT, load_yaml, write_smoke_config


def test_load_and_validate_smoke_config() -> None:
    from src.core.config import load_config, validate_config

    config = load_config(REPO_ROOT / "configs" / "smoke_local_tiny.yaml")
    validate_config(config)
    assert config["project"]["exp_id"] == "SMOKE_LOCAL_TINY"
    assert config["model"]["horizons"] == [1, 2, 3, 4]
    assert config["model"]["hidden_norm"] == "stateless_layer_norm"


def test_validate_config_rejects_unsorted_horizons(tmp_path: Path) -> None:
    from src.core.config import load_config, validate_config

    config_path, payload = write_smoke_config(tmp_path)
    payload["model"]["horizons"] = [2, 1, 4, 3]
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises((ValueError, AssertionError)):
        validate_config(load_config(config_path))


def test_validate_config_rejects_short_sequence_length(tmp_path: Path) -> None:
    from src.core.config import load_config, validate_config

    config_path, payload = write_smoke_config(tmp_path)
    payload["data"]["seq_len"] = 4
    payload["model"]["horizons"] = [1, 2, 3, 4]
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises((ValueError, AssertionError)):
        validate_config(load_config(config_path))


def test_load_config_requires_canonical_sections(tmp_path: Path) -> None:
    from src.core.config import load_config, validate_config

    bad_config_path = tmp_path / "bad.yaml"
    smoke = load_yaml(REPO_ROOT / "configs" / "smoke_local_tiny.yaml")
    smoke.pop("eval")
    bad_config_path.write_text(yaml.safe_dump(smoke, sort_keys=False), encoding="utf-8")

    with pytest.raises((KeyError, ValueError, AssertionError)):
        validate_config(load_config(bad_config_path))


def test_validate_config_rejects_real_run_without_nonterminal_validation_under_accumulation(tmp_path: Path) -> None:
    from src.core.config import load_config, validate_config

    config_path, payload = write_smoke_config(tmp_path)
    payload["project"]["stage"] = "probe"
    payload["hardware"]["micro_batch_size"] = 1
    payload["hardware"]["grad_accum"] = 4
    payload["data"]["train_token_quota"] = 64
    payload["train"]["eval_every_steps"] = 5
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="nonterminal validation event"):
        validate_config(load_config(config_path))


def test_validate_config_accepts_real_run_when_microbatch_cadence_hits_before_final_boundary(tmp_path: Path) -> None:
    from src.core.config import load_config, validate_config

    config_path, payload = write_smoke_config(tmp_path)
    payload["project"]["stage"] = "probe"
    payload["hardware"]["micro_batch_size"] = 1
    payload["hardware"]["grad_accum"] = 2
    payload["data"]["train_token_quota"] = 160
    payload["train"]["eval_every_steps"] = 3
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    validate_config(load_config(config_path))
