# MODULE_SPECS/05_ROUTER_AND_HEADS.md

## Objective

Define the layer-mixing mechanisms and horizon heads used by baselines and the main method.

## Canonical hidden-state normalization

Before probes or layer mixing, apply:

```python
x = F.layer_norm(h, (hidden_dim,), weight=None, bias=None, eps=1e-5)
```

This is the canonical hidden normalization in the project.

## Layer-mix variants

### Variant A — `last_layer`
- use the final transformer block output only
- no router parameters
- ignore `top_m`

### Variant B — `dense_whs`
- learn a softmax weight over all layers for each horizon
- if `router_init_mode=probe_zscore_top5`, initialize router scores from the probe z-scores but keep all layers active

### Variant C — `sparse_topm`
- learn router scores per horizon
- take top-`m` layers by score
- renormalize over active support only

## Head design

Default head: residual MLP

```python
y = layer_norm(x)
y = linear1(y)
y = gelu(y)
y = linear2(y)
z = x + y
logits = tied_unembed(z)
```

Properties:
- horizon-specific parameters
- same hidden size as backbone
- expansion ratio default 4×
- dropout 0 by default

## Optional ablation head

Deephead for far horizons:
- apply only to horizons 3 and 4
- add one tiny causal attention block before the residual MLP
- do not make this the default

## Initialization rules

- random router init: small normal noise, e.g. `N(0, 0.02)`
- probe-init router: load z-scored probe scores
- last-layer mode: no router tensor
- head layers: standard stable initialization (e.g. Xavier uniform) and log what was used

## Required diagnostics

For every trained run log:
- router entropy per horizon
- selected layers per horizon
- average router weights on validation data
- overlap with probe top-m layers

## Invariants

- sparse top-m yields exactly `m` nonzero weights per horizon
- active weights sum to 1
- last-layer mode always selects the final layer only
- head output dimension matches unembedding input dimension
