from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class TinyCausalBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        normed = self.ln_1(x)
        attn_output, _ = self.attn(normed, normed, normed, attn_mask=causal_mask, need_weights=False)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x


class LocalToyBackbone(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int = 32,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.blocks = nn.ModuleList(TinyCausalBlock(hidden_size, num_heads) for _ in range(num_layers))
        self.final_ln = nn.LayerNorm(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_base_logits: bool = False,
    ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden_states: list[torch.Tensor] = []
        for block in self.blocks:
            hidden = block(hidden, attention_mask=attention_mask)
            hidden_states.append(hidden)
        logits = None
        if return_base_logits:
            logits = F.linear(self.final_ln(hidden), self.token_embedding.weight)
        return hidden_states, logits

    @property
    def unembed_weight(self) -> torch.Tensor:
        return self.token_embedding.weight

