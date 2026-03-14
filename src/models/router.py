from __future__ import annotations

import math

import torch
from torch import nn


def stateless_layer_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight=None, bias=None, eps=eps)


class LayerRouter(nn.Module):
    def __init__(
        self,
        *,
        num_layers: int,
        horizons: list[int],
        mode: str,
        top_m: int,
        init_mode: str = "random",
        temperature: float = 1.0,
        init_scores: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.horizons = list(horizons)
        self.mode = mode
        self.top_m = top_m
        self.temperature = temperature
        if self.mode == "last_layer":
            self.router_scores = None
        else:
            scores = torch.zeros(len(horizons), num_layers)
            if init_mode == "random":
                nn.init.normal_(scores, mean=0.0, std=0.02)
            elif init_mode == "probe_zscore_top5":
                if init_scores is None:
                    raise ValueError("probe_zscore_top5 initialization requires probe scores")
                scores.copy_(init_scores)
            self.router_scores = nn.Parameter(scores)

    def mixture(self) -> tuple[torch.Tensor, dict[str, dict[str, float | list[int]]]]:
        if self.mode == "last_layer":
            device = torch.device("cpu")
            weights = torch.zeros(len(self.horizons), self.num_layers, device=device)
            weights[:, -1] = 1.0
        else:
            scores = self.router_scores / max(self.temperature, 1e-6)
            if self.mode == "dense_whs":
                weights = torch.softmax(scores, dim=-1)
            elif self.mode == "sparse_topm":
                topk_scores, topk_idx = torch.topk(scores, k=min(self.top_m, self.num_layers), dim=-1)
                sparse_logits = torch.full_like(scores, float("-inf"))
                sparse_logits.scatter_(dim=-1, index=topk_idx, src=topk_scores)
                weights = torch.softmax(sparse_logits, dim=-1)
            else:
                raise ValueError(f"Unsupported router mode: {self.mode}")
        entropy = -(weights.clamp_min(1e-9) * weights.clamp_min(1e-9).log()).sum(dim=-1)
        selected = torch.topk(weights, k=min(self.top_m, self.num_layers), dim=-1).indices
        diagnostics = {
            "entropy_by_horizon": {str(h): float(entropy[idx].item()) for idx, h in enumerate(self.horizons)},
            "selected_layers_by_horizon": {
                str(h): [int(value) for value in selected[idx].tolist()] for idx, h in enumerate(self.horizons)
            },
            "average_weights_by_horizon": {
                str(h): [float(value) for value in weights[idx].tolist()] for idx, h in enumerate(self.horizons)
            },
            "effective_support_size_by_horizon": {
                str(h): float(torch.exp(entropy[idx]).item()) for idx, h in enumerate(self.horizons)
            },
        }
        return weights, diagnostics

    def get_weights(self) -> torch.Tensor:
        weights, _ = self.mixture()
        return weights

    def forward(self, _hidden_states=None) -> torch.Tensor:
        return self.get_weights()
