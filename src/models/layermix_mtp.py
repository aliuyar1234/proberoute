from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .heads import HorizonHeadBank
from .router import LayerRouter, stateless_layer_norm


@dataclass
class ProbeInitBundle:
    tensor: torch.Tensor
    metadata: dict


class LayerMixMTPModel(nn.Module):
    def __init__(
        self,
        *,
        backbone,
        horizons: list[int],
        layer_mix_mode: str,
        router_init_mode: str,
        top_m: int,
        router_temperature: float,
        head_type: str,
        hidden_expansion: int,
        entropy_penalty_beta: float,
        probe_init_scores: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.horizons = list(horizons)
        self.router = LayerRouter(
            num_layers=backbone.num_layers(),
            horizons=self.horizons,
            mode=layer_mix_mode,
            top_m=top_m,
            init_mode=router_init_mode,
            temperature=router_temperature,
            init_scores=probe_init_scores,
        )
        self.heads = HorizonHeadBank(
            hidden_size=backbone.hidden_size(),
            horizons=self.horizons,
            head_type=head_type,
            expansion=hidden_expansion,
        )
        self.layer_mix_mode = layer_mix_mode
        self.entropy_penalty_beta = entropy_penalty_beta

    def checkpoint_state(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "router": self.router.state_dict(),
            "heads": self.heads.state_dict(),
        }

    def load_checkpoint_state(self, payload: dict[str, dict[str, torch.Tensor]] | dict[str, torch.Tensor]) -> None:
        if "router" in payload and "heads" in payload:
            self.router.load_state_dict(payload["router"])
            self.heads.load_state_dict(payload["heads"])
            return
        incompatible = self.load_state_dict(payload, strict=False)
        missing_non_backbone = [key for key in incompatible.missing_keys if not key.startswith("backbone.")]
        if missing_non_backbone or incompatible.unexpected_keys:
            raise ValueError(
                f"Incompatible checkpoint payload: missing={missing_non_backbone}, unexpected={list(incompatible.unexpected_keys)}"
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        *,
        return_diagnostics: bool = False,
    ) -> dict[str, object]:
        backbone_outputs = self.backbone.forward_hidden(input_ids, attention_mask=attention_mask, return_base_logits=True)
        normalized_layers = [stateless_layer_norm(state) for state in backbone_outputs.hidden_states]
        stacked = torch.stack(normalized_layers, dim=1)
        weights, router_diag = self.router.mixture()

        logits_by_horizon: dict[int, torch.Tensor] = {}
        per_horizon_losses: dict[int, torch.Tensor] = {}
        total_loss = torch.tensor(0.0, device=stacked.device)

        for horizon_idx, horizon in enumerate(self.horizons):
            router_weights = weights[horizon_idx].to(device=stacked.device, dtype=stacked.dtype)
            mixed = torch.einsum("l,bltd->btd", router_weights, stacked)
            hidden = self.heads.forward(horizon, mixed)
            logits = F.linear(hidden, self.backbone.unembedding_weight().to(hidden.dtype))
            logits_by_horizon[horizon] = logits
            if targets is not None:
                usable_logits = logits[:, :-horizon, :]
                usable_targets = targets[:, horizon:]
                loss = F.cross_entropy(
                    usable_logits.reshape(-1, usable_logits.shape[-1]),
                    usable_targets.reshape(-1),
                )
                per_horizon_losses[horizon] = loss
                total_loss = total_loss + loss

        entropy_term = 0.0
        if self.layer_mix_mode != "last_layer":
            weights_for_entropy, _ = self.router.mixture()
            entropy_term = -(weights_for_entropy.clamp_min(1e-9) * weights_for_entropy.clamp_min(1e-9).log()).sum(dim=-1).mean()
            total_loss = total_loss + self.entropy_penalty_beta * entropy_term

        result: dict[str, object] = {
            "logits_by_horizon": logits_by_horizon,
            "loss": total_loss,
        }
        if return_diagnostics:
            result["diagnostics"] = {
                "per_horizon_losses": {str(k): float(v.detach().cpu().item()) for k, v in per_horizon_losses.items()},
                "router_entropy": router_diag["entropy_by_horizon"],
                "selected_layers": router_diag["selected_layers_by_horizon"],
                "effective_support_size": router_diag["effective_support_size_by_horizon"],
                "average_weights": router_diag["average_weights_by_horizon"],
            }
        return result
