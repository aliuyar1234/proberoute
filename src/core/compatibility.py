from __future__ import annotations

from typing import Any

from .manifest import build_dataset_id

DATASET_COMPATIBILITY_KEYS = (
    "dataset_name",
    "dataset_config",
    "tokenizer_name",
    "seq_len",
)


def dataset_manifest_matches_config(manifest: dict[str, Any], config: dict[str, Any]) -> bool:
    data = config["data"]
    for key in DATASET_COMPATIBILITY_KEYS:
        if manifest.get(key) != data.get(key):
            return False
    split_policy = manifest.get("split_policy", {})
    if split_policy.get("modulus") != data.get("split_modulus"):
        return False
    if split_policy.get("ranges") != data.get("split_ranges"):
        return False
    normalization = manifest.get("normalization_policy", {})
    if not normalization:
        return False
    if manifest.get("local_path") != data.get("local_path"):
        return False
    return True


def ensure_probe_init_compatible(
    probe_init: dict[str, Any],
    config: dict[str, Any],
    *,
    model_id: str,
    num_layers: int,
) -> None:
    expected_horizons = list(config["model"]["horizons"])
    expected = {
        "model_id": model_id,
        "dataset_name": config["data"]["dataset_name"],
        "dataset_config": config["data"]["dataset_config"],
        "horizons": expected_horizons,
        "num_layers": num_layers,
        "metric": config["model"]["probe_init_metric"],
    }
    for key, value in expected.items():
        observed = probe_init.get(key)
        if observed != value:
            raise ValueError(f"Incompatible probe init for {key}: expected {value!r}, observed {observed!r}")


def validate_probe_init_compatibility(*, config: dict[str, Any], probe_init: dict[str, Any]) -> None:
    # Probe runs are intentionally cheaper than screening/final runs in the
    # canonical plan, so seq_len may differ even when the router init is valid.
    expected = {
        "backbone_name": config["model"]["backbone_name"],
        "horizons": list(config["model"]["horizons"]),
        "init_metric": config["model"]["probe_init_metric"],
    }
    for key, value in expected.items():
        observed = probe_init.get(key)
        if observed != value:
            raise ValueError(f"Incompatible probe init for {key}: expected {value!r}, observed {observed!r}")

    if "dataset_name" in probe_init or "dataset_config" in probe_init:
        expected_dataset = {
            "dataset_name": config["data"]["dataset_name"],
            "dataset_config": config["data"]["dataset_config"],
        }
        for key, value in expected_dataset.items():
            observed = probe_init.get(key)
            if observed != value:
                raise ValueError(f"Incompatible probe init for {key}: expected {value!r}, observed {observed!r}")
        return

    observed_dataset_id = probe_init.get("dataset_id")
    if observed_dataset_id != build_dataset_id(config):
        raise ValueError(
            f"Incompatible probe init for dataset_id: expected {build_dataset_id(config)!r}, observed {observed_dataset_id!r}"
        )


assert_probe_init_compatible = validate_probe_init_compatibility
check_probe_init_compatible = validate_probe_init_compatibility
