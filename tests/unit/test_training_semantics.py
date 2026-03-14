from __future__ import annotations

from src.train.trainer_common import next_cadence_micro_step, planned_micro_steps, planned_steps, realized_train_tokens


def test_token_budget_planning_rounds_to_accumulation_boundaries() -> None:
    assert planned_micro_steps(160, batch_size=1, seq_len=32) == 5
    assert planned_steps(160, batch_size=1, seq_len=32, grad_accum=2) == 3
    assert realized_train_tokens(160, batch_size=1, seq_len=32, grad_accum=2) == 192


def test_next_cadence_micro_step_tracks_microbatch_progress() -> None:
    assert next_cadence_micro_step(0, 3) == 3
    assert next_cadence_micro_step(2, 3) == 3
    assert next_cadence_micro_step(3, 3) == 6
    assert next_cadence_micro_step(5, 3) == 6
    assert next_cadence_micro_step(0, 0) is None
