from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .bootstrap_ci import percentile_bootstrap


def _base_greedy_rollout(backbone, prefix: torch.Tensor, max_horizon: int) -> list[int]:
    generated = prefix.clone()
    tokens: list[int] = []
    for _ in range(max_horizon):
        outputs = backbone.forward_hidden(generated, return_base_logits=True)
        next_token = int(outputs.base_logits[:, -1, :].argmax(dim=-1).item())
        tokens.append(next_token)
        generated = torch.cat([generated, torch.tensor([[next_token]], device=generated.device)], dim=1)
    return tokens


def _mtp_block(model, prefix: torch.Tensor, horizons: list[int]) -> list[int]:
    outputs = model(prefix, targets=None, return_diagnostics=False)
    block = []
    for horizon in horizons:
        logits = outputs["logits_by_horizon"][horizon]
        block.append(int(logits[:, -1, :].argmax(dim=-1).item()))
    return block


def compute_acceptance_metrics(
    model,
    prefixes: np.ndarray,
    *,
    exp_id: str,
    horizons: list[int],
    max_new_tokens: int,
    evaluation_seed: int,
    bootstrap_samples: int,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "acceptance_traces.jsonl"
    accept_lengths: list[int] = []
    max_horizon = max(horizons)
    with trace_path.open("w", encoding="utf-8") as handle:
        for index, prefix_array in enumerate(prefixes):
            prefix = torch.tensor(prefix_array[None, :], dtype=torch.long, device=model.backbone.device)
            total_accept = 0
            step_count = 0
            current = prefix
            while step_count < max_new_tokens:
                base_block = _base_greedy_rollout(model.backbone, current, max_horizon=max_horizon)
                mtp_block = _mtp_block(model, current, horizons)
                accept_len = 0
                for expected, predicted in zip(base_block, mtp_block):
                    if expected != predicted:
                        break
                    accept_len += 1
                total_accept += accept_len
                handle.write(
                    json.dumps(
                        {
                            "exp_id": exp_id,
                            "prefix_index": index,
                            "decode_step": step_count,
                            "base_block": base_block,
                            "mtp_block": mtp_block,
                            "accepted_len": accept_len,
                        }
                    )
                    + "\n"
                )
                next_token = base_block[0]
                current = torch.cat([current, torch.tensor([[next_token]], device=current.device)], dim=1)
                step_count += 1
            accept_lengths.append(total_accept / max(step_count, 1))

    histogram = {str(length): int(accept_lengths.count(length)) for length in sorted(set(accept_lengths))}
    depth_rates = {}
    for depth in range(1, max_horizon + 1):
        depth_rates[depth] = float(np.mean([value >= depth for value in accept_lengths])) if accept_lengths else 0.0
    return {
        "exp_id": exp_id,
        "num_prefixes": int(len(prefixes)),
        "max_horizon": int(max_horizon),
        "max_new_tokens": int(max_new_tokens),
        "prefix_len": int(prefixes.shape[1]),
        "greedy_policy": "argmax",
        "advance_policy": "append_one_base_greedy_token",
        "mean_accept_len": float(np.mean(accept_lengths)) if accept_lengths else 0.0,
        "accept_rate_depth_1": depth_rates.get(1, 0.0),
        "accept_rate_depth_2": depth_rates.get(2, 0.0),
        "accept_rate_depth_3": depth_rates.get(3, 0.0),
        "accept_rate_depth_4": depth_rates.get(4, 0.0),
        "accept_len_histogram": histogram,
        "bootstrap_ci": percentile_bootstrap(accept_lengths, samples=bootstrap_samples, seed=evaluation_seed),
        "evaluation_seed": evaluation_seed,
        "trace_path": str(trace_path),
    }
