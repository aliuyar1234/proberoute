from __future__ import annotations

import inspect
import types

import numpy as np
import torch

from tests.helpers import call_with_known_kwargs, first_existing_attr


def _instantiate_probe_bank(unembedding_weight: torch.Tensor):
    from src.models.probe_bank import LowRankProbeBank

    return call_with_known_kwargs(
        LowRankProbeBank,
        num_layers=3,
        horizons=[1, 2, 4],
        hidden_size=8,
        rank=2,
        vocab_size=unembedding_weight.shape[0],
        unembedding_weight=unembedding_weight,
    )


def _run_probe_forward(probe_bank, hidden_states: list[torch.Tensor]):
    forward = first_existing_attr(probe_bank, ["forward", "forward_all", "__call__"])
    signature = inspect.signature(forward)
    if "hidden_states" in signature.parameters:
        return forward(hidden_states=hidden_states)
    return forward(hidden_states)


def test_probe_bank_emits_vocab_sized_logits_for_each_layer_and_horizon() -> None:
    batch_size, seq_len, hidden_size, vocab_size = 2, 5, 8, 13
    hidden_states = [torch.randn(batch_size, seq_len, hidden_size) for _ in range(3)]
    unembedding_weight = torch.randn(vocab_size, hidden_size)
    probe_bank = _instantiate_probe_bank(unembedding_weight)
    outputs = _run_probe_forward(probe_bank, hidden_states)

    if isinstance(outputs, dict) and "logits_by_layer_and_horizon" in outputs:
        outputs = outputs["logits_by_layer_and_horizon"]

    for layer_idx in range(3):
        for horizon in [1, 2, 4]:
            payload = outputs[layer_idx][horizon] if isinstance(outputs[layer_idx], dict) else outputs[layer_idx, horizon]
            assert tuple(payload.shape) == (batch_size, seq_len, vocab_size)


def test_probe_bank_iter_logits_matches_layer_horizon_grid() -> None:
    batch_size, seq_len, hidden_size, vocab_size = 2, 5, 8, 13
    hidden_states = [torch.randn(batch_size, seq_len, hidden_size) for _ in range(3)]
    unembedding_weight = torch.randn(vocab_size, hidden_size)
    probe_bank = _instantiate_probe_bank(unembedding_weight)

    seen = []
    for layer_idx, horizon_idx, payload in probe_bank.iter_logits(hidden_states, unembedding_weight):
        seen.append((layer_idx, horizon_idx))
        assert tuple(payload.shape) == (batch_size, seq_len, vocab_size)

    assert seen == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    ]


def test_probe_validation_uses_logit_width_not_reported_backbone_vocab() -> None:
    from src.train.probe_trainer import _probe_scores

    class DummyBackbone:
        device = torch.device("cpu")

        def num_layers(self) -> int:
            return 2

        def vocab_size(self) -> int:
            return 11

        def unembedding_weight(self) -> torch.Tensor:
            return torch.randn(13, 4)

        def forward_hidden(self, batch: torch.Tensor):
            hidden_states = [torch.zeros(batch.shape[0], batch.shape[1], 4) for _ in range(self.num_layers())]
            return types.SimpleNamespace(hidden_states=hidden_states)

    class DummyProbe:
        def iter_logits(self, hidden_states, unembed_weight):
            batch_size, seq_len = hidden_states[0].shape[:2]
            for layer_idx, _ in enumerate(hidden_states):
                yield layer_idx, 0, torch.randn(batch_size, seq_len, 13)

    scores = _probe_scores(
        DummyProbe(),
        DummyBackbone(),
        np.asarray([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.int64),
        [1],
        batch_size=2,
    )

    assert set(scores.keys()) == {"top1", "top5", "nll"}
    assert len(scores["nll"][1]) == 2
