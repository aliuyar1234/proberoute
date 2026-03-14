# MODULE_SPECS/04_PROBE_SWEEP.md

## Objective

Train a cheap probe bank that estimates how predictive each backbone layer is for each future horizon.

## Probe design

Use a **tied-unembedding low-rank linear probe**.

For each layer `l` and horizon `k`:
1. normalize hidden state with stateless layer norm,
2. apply a low-rank translator,
3. project with the frozen unembedding matrix.

## Canonical parameterization

```python
u = x @ A[l, k] @ B[l, k]     # rank-r factorization
logits = F.linear(u, unembed_weight, unembed_bias_if_any)
```

No nonlinearity by default.

## Memory policy

Do **not** materialize logits for all layers and all horizons at once.
Process layers sequentially (or in small chunks) so probe training stays memory-bounded.

## Output artefacts

Each probe run must generate:
- `probe_scores.csv`
- `probe_heatmap_top1.png`
- `probe_heatmap_top5.png`
- `probe_heatmap_nll.png`
- `probe_init.json`
- `probe_init.pt`

## Canonical router-init policy

- init metric: validation top-5
- store raw per-layer scores by horizon
- z-score within each horizon across layers
- sparse runs initialize support from the top-`m` z-scored layers
- dense runs initialize scores from the same z-scored vector but keep all layers active

## Validation rules

- k=1 should be materially above random-like behavior
- no horizon may index beyond available positions
- heatmaps must numerically match the CSV
- `probe_init` metadata must identify model, dataset, seq_len, horizons, and init metric
