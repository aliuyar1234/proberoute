from __future__ import annotations

import torch

from tests.helpers import first_existing_attr


def test_future_target_slices_match_horizon_offsets() -> None:
    import src.models.losses as losses

    slicer = first_existing_attr(
        losses,
        ["slice_targets_by_horizon", "build_targets_by_horizon", "prepare_targets_by_horizon"],
    )
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
    result = slicer(input_ids=input_ids, horizons=[1, 2, 3, 4])

    for horizon in [1, 2, 3, 4]:
        payload = result[horizon]
        if isinstance(payload, tuple):
            _, targets = payload
        elif isinstance(payload, dict):
            targets = payload["targets"] if "targets" in payload else payload["target_ids"]
        else:
            targets = payload
        expected = input_ids[:, horizon:]
        assert torch.equal(targets, expected), f"horizon {horizon} targets were misaligned"
