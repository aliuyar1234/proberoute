from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def test_inherit_from_deep_merges_dicts_and_replaces_lists_and_scalars(tmp_path: Path) -> None:
    from src.core.config import load_config

    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"
    parent.write_text(
        yaml.safe_dump(
            {
                "project": {"name": "parent", "seed": 1},
                "model": {"horizons": [1, 2, 3, 4], "top_m": 4, "nested": {"a": 1, "b": 2}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    child.write_text(
        yaml.safe_dump(
            {
                "inherit_from": "parent.yaml",
                "project": {"seed": 1337},
                "model": {"horizons": [1, 3], "nested": {"b": 99, "c": 100}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = load_config(child)
    assert config["project"]["name"] == "parent"
    assert config["project"]["seed"] == 1337
    assert config["model"]["horizons"] == [1, 3]
    assert config["model"]["top_m"] == 4
    assert config["model"]["nested"] == {"a": 1, "b": 99, "c": 100}


def test_inherit_from_cycle_is_a_hard_failure(tmp_path: Path) -> None:
    from src.core.config import load_config

    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("inherit_from: b.yaml\nproject: {name: a}\n", encoding="utf-8")
    b.write_text("inherit_from: a.yaml\nproject: {name: b}\n", encoding="utf-8")

    with pytest.raises((RecursionError, RuntimeError, ValueError)):
        load_config(a)
