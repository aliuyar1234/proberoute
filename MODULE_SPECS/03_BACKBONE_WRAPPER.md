# MODULE_SPECS/03_BACKBONE_WRAPPER.md

## Objective

Expose a stable wrapper around the pretrained LM or the local toy smoke model.

## Required interface

```python
@dataclass
class BackboneOutputs:
    hidden_states: list[torch.Tensor]   # transformer-block outputs only
    base_logits: torch.Tensor | None
    attention_mask: torch.Tensor | None

class BackboneWrapper(nn.Module):
    def __init__(self, model_name: str, precision: str, device: str, smoke_mode: bool = False): ...
    def tokenizer(self): ...
    def num_layers(self) -> int: ...
    def hidden_size(self) -> int: ...
    def vocab_size(self) -> int: ...
    def model_slug(self) -> str: ...
    def unembedding_weight(self) -> torch.Tensor: ...
    def forward_hidden(self, input_ids, attention_mask=None, return_base_logits=False) -> BackboneOutputs: ...
```

## Behavior rules

- Main track: backbone parameters have `requires_grad=False`.
- `forward_hidden` runs under `torch.no_grad()` in required runs.
- Return transformer-block hidden states only; do not include embedding output by default.
- Preserve `[batch, seq, hidden]` ordering.
- Expose the exact unembedding matrix used for base logits.

## Smoke-path exception

For `smoke_local_tiny.yaml`, the wrapper may construct a tiny local GPTNeoX-like model and a local toy tokenizer instead of loading a remote checkpoint.
This exception is allowed only for smoke/tests.

## Sanity checks

- hidden-state count equals transformer depth,
- base logits are finite when requested,
- hidden size matches the unembedding input size,
- toy smoke batch produces no NaNs.
