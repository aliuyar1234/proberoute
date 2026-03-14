from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from .bootstrap_ci import percentile_bootstrap


def compute_future_metrics(
    model,
    sequences: np.ndarray,
    *,
    exp_id: str,
    split: str,
    horizons: list[int],
    evaluation_seed: int,
    bootstrap_samples: int,
    batch_size: int | None = None,
) -> dict:
    if len(sequences) == 0:
        raise ValueError("Cannot compute future metrics over an empty sequence array")
    effective_batch_size = max(1, int(batch_size or len(sequences)))
    accumulators = {
        horizon: {"top1_correct": 0.0, "top5_correct": 0.0, "nll_sum": 0.0, "token_count": 0}
        for horizon in horizons
    }
    for start in range(0, len(sequences), effective_batch_size):
        batch_array = sequences[start : start + effective_batch_size]
        batch = torch.tensor(batch_array, dtype=torch.long, device=model.backbone.device)
        with torch.no_grad():
            outputs = model(batch, targets=batch, return_diagnostics=False)
        for horizon in horizons:
            logits = outputs["logits_by_horizon"][horizon]
            usable_logits = logits[:, :-horizon, :]
            usable_targets = batch[:, horizon:]
            vocab = usable_logits.shape[-1]
            token_count = int(usable_targets.numel())
            accumulators[horizon]["top1_correct"] += float((usable_logits.argmax(dim=-1) == usable_targets).sum().item())
            accumulators[horizon]["top5_correct"] += float(
                usable_logits.topk(k=min(5, vocab), dim=-1).indices.eq(usable_targets.unsqueeze(-1)).any(dim=-1).sum().item()
            )
            accumulators[horizon]["nll_sum"] += float(
                F.cross_entropy(usable_logits.reshape(-1, vocab), usable_targets.reshape(-1), reduction="sum").item()
            )
            accumulators[horizon]["token_count"] += token_count

    metrics_by_horizon: dict[str, dict[str, float]] = {}
    top1_values = []
    top5_values = []
    nll_values = []
    for horizon in horizons:
        token_count = max(1, int(accumulators[horizon]["token_count"]))
        top1 = float(accumulators[horizon]["top1_correct"]) / token_count
        top5 = float(accumulators[horizon]["top5_correct"]) / token_count
        nll = float(accumulators[horizon]["nll_sum"]) / token_count
        metrics_by_horizon[str(horizon)] = {"top1": top1, "top5": top5, "nll": nll}
        top1_values.append(top1)
        top5_values.append(top5)
        nll_values.append(nll)
    aggregate = {
        "mean_top1_h1_h4": float(np.mean(top1_values)),
        "mean_top1_h2_h4": float(np.mean(top1_values[1:])) if len(top1_values) > 1 else float(np.mean(top1_values)),
        "mean_top5_h1_h4": float(np.mean(top5_values)),
        "mean_nll_h1_h4": float(np.mean(nll_values)),
    }
    return {
        "exp_id": exp_id,
        "split": split,
        "num_sequences": int(len(sequences)),
        "horizons": horizons,
        "metrics_by_horizon": metrics_by_horizon,
        "aggregate_metrics": aggregate,
        "bootstrap_ci": percentile_bootstrap(top1_values, samples=bootstrap_samples, seed=evaluation_seed),
        "evaluation_seed": evaluation_seed,
    }
