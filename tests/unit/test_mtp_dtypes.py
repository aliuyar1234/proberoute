from __future__ import annotations

import torch

from src.models.backbone_wrapper import BackboneOutputs
from src.models.layermix_mtp import LayerMixMTPModel


class _BFloat16Backbone:
    def __init__(self, *, num_layers: int = 3, hidden_size: int = 8, vocab_size: int = 32) -> None:
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._unembed_weight = torch.randn(vocab_size, hidden_size, dtype=torch.float32)

    def num_layers(self) -> int:
        return self._num_layers

    def hidden_size(self) -> int:
        return self._hidden_size

    def unembedding_weight(self) -> torch.Tensor:
        return self._unembed_weight

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_base_logits: bool = False,
    ) -> BackboneOutputs:
        batch_size, seq_len = input_ids.shape
        hidden_states = [
            torch.randn(batch_size, seq_len, self._hidden_size, dtype=torch.bfloat16)
            for _ in range(self._num_layers)
        ]
        return BackboneOutputs(hidden_states=hidden_states, base_logits=None, attention_mask=attention_mask)


def test_last_layer_linear_head_accepts_bfloat16_hidden_states() -> None:
    backbone = _BFloat16Backbone()
    model = LayerMixMTPModel(
        backbone=backbone,
        horizons=[1, 2, 3, 4],
        layer_mix_mode="last_layer",
        router_init_mode="none",
        top_m=1,
        router_temperature=1.0,
        head_type="linear",
        hidden_expansion=2,
        entropy_penalty_beta=0.0,
    )
    batch = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=torch.long)

    outputs = model(batch, targets=batch, return_diagnostics=True)

    assert torch.isfinite(outputs["loss"]).item()
    assert set(outputs["logits_by_horizon"].keys()) == {1, 2, 3, 4}
