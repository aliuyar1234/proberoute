from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .router import stateless_layer_norm


class ResidualMLPHead(nn.Module):
    def __init__(self, hidden_size: int, expansion: int) -> None:
        super().__init__()
        expanded = hidden_size * expansion
        self.linear_1 = nn.Linear(hidden_size, expanded)
        self.linear_2 = nn.Linear(expanded, hidden_size)
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.zeros_(self.linear_1.bias)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.zeros_(self.linear_2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_work = x.to(self.linear_1.weight.dtype)
        y = stateless_layer_norm(x_work)
        y = self.linear_1(y)
        y = F.gelu(y)
        y = self.linear_2(y)
        return x_work + y


class LinearHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_work = x.to(self.linear.weight.dtype)
        return self.linear(x_work)


class HorizonHeadBank(nn.Module):
    def __init__(self, *, hidden_size: int, horizons: list[int], head_type: str, expansion: int) -> None:
        super().__init__()
        if head_type == "residual_mlp":
            factory = lambda: ResidualMLPHead(hidden_size, expansion)
        elif head_type == "linear":
            factory = lambda: LinearHead(hidden_size)
        else:
            raise NotImplementedError(f"Unsupported head type in scaffold: {head_type}")
        self.heads = nn.ModuleDict({str(h): factory() for h in horizons})

    def forward(self, horizon: int, x: torch.Tensor) -> torch.Tensor:
        return self.heads[str(horizon)](x)
