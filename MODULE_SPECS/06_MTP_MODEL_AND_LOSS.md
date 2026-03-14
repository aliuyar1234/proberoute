# MODULE_SPECS/06_MTP_MODEL_AND_LOSS.md

## Objective

Combine backbone outputs, routing, heads, and loss computation into a single MTP adaptation model.

## Forward signature

```python
class LayerMixMTPModel(nn.Module):
    def forward(self, input_ids, attention_mask=None, targets=None, return_diagnostics=False):
        ...
        return {
            'logits_by_horizon': {1: ..., 2: ..., 3: ..., 4: ...},
            'loss': ...,
            'diagnostics': ...,
        }
```

## Target contract

For each horizon `k`:
- source positions are `0 .. T-k-1`
- targets are `input_ids[:, k:]`
- logits are evaluated only on positions `[:, :-k]`

Per-horizon loss:

```python
loss_k = CE(logits_k[:, :-k, :].reshape(-1, V), input_ids[:, k:].reshape(-1))
```

Total loss:

```python
loss = sum_k lambda_k * loss_k + beta * router_entropy_penalty
```

## Warmup ablation

Default:
- fixed `lambda_k = 1.0`

Warmup ablation:
- `k=2` reaches full weight by 10% of scheduled budget
- `k=3` by 20%
- `k=4` by 30%

## Token-budget stopping

The training loop must count processed input tokens, not optimizer steps only.
Increment `consumed_tokens` by `batch.input_ids.numel()` for every microbatch.
Stop when:
- the train token budget is exhausted,
- or early stop triggers after at least 30% of the budget.

## Diagnostics to return

- per-horizon losses
- router entropy
- selected layers
- effective support size

## Leakage rule

The model must not:
- use future ground-truth tokens as inputs to the horizon heads,
- or feed target tokens back into the computation graph.

Teacher forcing only provides the prefix.
