from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F

from .router import stateless_layer_norm


@dataclass
class ProbeScores:
    top1: dict[int, list[float]]
    top5: dict[int, list[float]]
    nll: dict[int, list[float]]


class LowRankProbeBank(nn.Module):
    def __init__(
        self,
        *,
        num_layers: int,
        horizons: list[int],
        hidden_size: int,
        rank: int,
        unembedding_weight: torch.Tensor | None = None,
        vocab_size: int | None = None,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.horizons = list(horizons)
        self.hidden_size = hidden_size
        self.rank = rank
        self.a = nn.Parameter(torch.empty(num_layers, len(horizons), hidden_size, rank))
        self.b = nn.Parameter(torch.empty(num_layers, len(horizons), rank, hidden_size))
        nn.init.xavier_uniform_(self.a)
        nn.init.xavier_uniform_(self.b)
        if unembedding_weight is not None:
            self.register_buffer("_unembedding_weight", unembedding_weight.clone().detach())
        else:
            self._unembedding_weight = None

    def logits_for(self, hidden_state: torch.Tensor, *, layer_idx: int, horizon_idx: int, unembed_weight: torch.Tensor) -> torch.Tensor:
        normalized = stateless_layer_norm(hidden_state).to(self.a.dtype)
        translated = normalized @ self.a[layer_idx, horizon_idx] @ self.b[layer_idx, horizon_idx]
        return F.linear(translated, unembed_weight.to(translated.dtype))

    def iter_logits(self, hidden_states: list[torch.Tensor], unembed_weight: torch.Tensor) -> Iterator[tuple[int, int, torch.Tensor]]:
        for layer_idx, hidden_state in enumerate(hidden_states):
            for horizon_idx, _ in enumerate(self.horizons):
                yield (
                    layer_idx,
                    horizon_idx,
                    self.logits_for(
                        hidden_state,
                        layer_idx=layer_idx,
                        horizon_idx=horizon_idx,
                        unembed_weight=unembed_weight,
                    ),
                )

    def all_logits(self, hidden_states: list[torch.Tensor], unembed_weight: torch.Tensor) -> dict[tuple[int, int], torch.Tensor]:
        outputs: dict[tuple[int, int], torch.Tensor] = {}
        for layer_idx, horizon_idx, logits in self.iter_logits(hidden_states, unembed_weight):
            outputs[(layer_idx, horizon_idx)] = logits
        return outputs

    def forward(self, hidden_states: list[torch.Tensor]) -> dict[int, dict[int, torch.Tensor]]:
        if self._unembedding_weight is None:
            raise ValueError("LowRankProbeBank.forward requires an unembedding weight to be provided at construction time")
        flat = self.all_logits(hidden_states, self._unembedding_weight)
        nested: dict[int, dict[int, torch.Tensor]] = {}
        for layer_idx in range(self.num_layers):
            nested[layer_idx] = {}
            for horizon_idx, horizon in enumerate(self.horizons):
                nested[layer_idx][horizon] = flat[(layer_idx, horizon_idx)]
        return nested
