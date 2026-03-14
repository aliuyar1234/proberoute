from __future__ import annotations

import torch


def target_slices(input_ids: torch.Tensor, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    return input_ids[:, :-horizon], input_ids[:, horizon:]


def slice_targets_by_horizon(*, input_ids: torch.Tensor, horizons: list[int]) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    return {horizon: target_slices(input_ids, horizon) for horizon in horizons}
