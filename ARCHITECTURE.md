# ARCHITECTURE.md

## 1. Overview

This project is a **reproducible research pipeline**, not a serving system. The codebase should be optimized for correctness, observability, bounded scope, and paper-ready outputs.

## 2. High-level flow

```text
raw text source or local fixture
  -> deterministic normalization + split assignment
  -> canonical packed .npy datasets + dataset manifest
  -> backbone wrapper (frozen hidden-state extraction)
  -> probe sweep
  -> probe heatmaps + probe_init artefacts
  -> screening baselines + screening sparse method
  -> screening registry + finalist selection
  -> generated final baseline config + final reruns
  -> ablations / confirmatory run
  -> evaluation + bootstrap + router analysis
  -> registries + figures + tables + claim-evidence matrix
  -> paper draft + FINAL_DELIVERY.md
```

## 3. Module boundaries

### A. Data layer
Responsible for:
- dataset loading or fixture loading,
- deterministic normalization,
- split assignment,
- tokenization,
- packing,
- `.npy` shard writing,
- manifest creation,
- deterministic eval subset generation.

Must not:
- perform model logic,
- hide randomness,
- or mutate existing processed data unless compatibility checks fail and `--force-rebuild` is passed.

### B. Backbone wrapper
Responsible for:
- loading Pythia checkpoints or the local toy smoke model,
- exposing tokenizer and unembedding weights,
- returning transformer-block hidden states,
- handling dtype/device/precision.

Must not:
- implement MTP routing,
- or change backbone weights in the main track.

### C. Probe layer
Responsible for:
- the low-rank tied-unembedding probe bank,
- probe training and validation,
- probe scores CSV,
- heatmaps,
- probe-init artefacts with compatibility metadata.

Must not:
- become the main method,
- or assume the final layer is always best.

### D. Routing and heads layer
Responsible for:
- layer-mix mode selection (`last_layer`, `dense_whs`, `sparse_topm`),
- hidden-state normalization,
- router initialization,
- horizon heads,
- MTP logits.

Must not:
- leak future labels,
- or backpropagate through the frozen trunk in the main track.

### E. Training orchestration layer
Responsible for:
- token-budget stopping,
- checkpointing,
- manifests,
- logging,
- resuming,
- controlled evaluation cadence.

Must not:
- embed hard-coded experiment assumptions outside the config,
- or silently overwrite completed runs.

### F. Evaluation and registry layer
Responsible for:
- future-token metrics,
- greedy acceptance proxy,
- bootstrap CIs,
- router diagnostics,
- screening registry,
- finalist selection,
- paper-asset source files.

Must not:
- rerun training,
- or manually transcribe numbers from terminal output.

### G. Paper layer
Responsible for:
- claim-evidence matrix,
- figures and tables,
- manuscript generation,
- final placeholder scan,
- final delivery note.

Must not:
- fabricate numbers,
- or describe unrun experiments.

## 4. Canonical architectural decisions

### 4.1 Hidden-state normalization
Use **stateless layer normalization** before probes and layer mixing:
- operation: `F.layer_norm(x, (hidden_dim,), weight=None, bias=None, eps=1e-5)`
- no learned affine parameters in the normalization step.

Reason: this keeps normalization stable and comparable across layers and avoids dependence on model-family-specific internal LN placements.

### 4.2 Layer-mix modes
- `last_layer`: fixed final transformer block output; no router parameters.
- `dense_whs`: learned softmax over all layers for each horizon.
- `sparse_topm`: learned router scores, top-`m` support, renormalized over selected layers only.

### 4.3 Horizon heads
Default: horizon-specific residual MLP with tied unembedding.
Optional ablation: a tiny causal attention block before the MLP for horizons 3 and 4 only.

### 4.4 Probe initialization
Canonical init metric: **validation top-5**.
`probe_init.json` stores raw and z-scored per-layer scores; `probe_init.pt` stores tensors ready for loading.

## 5. Data flow in the main method

```text
input_ids [B, T]
  -> frozen backbone (output_hidden_states=True, no_grad)
  -> hidden_states list length L, each [B, T, D]
  -> stateless layer_norm per layer
  -> router scores per horizon
  -> selected layer support per horizon
  -> mixed hidden state per horizon
  -> horizon head per horizon
  -> shared unembedding
  -> logits_k [B, T, V]
  -> CE against targets input_ids[:, k:]
  -> total loss = sum(lambda_k * loss_k) + beta * router_entropy
```

## 6. Failure boundaries

The following do **not** justify changing the thesis:
- an OOM on the first attempt,
- a broken optional accelerator,
- a dataset field mismatch,
- weak wall-clock speed,
- a missing plot script,
- or one unstable baseline run.

The following can justify a scoped fallback:
- probe heatmaps remain flat after validated implementation and probe recovery steps,
- all methods remain worse than the robust last-layer baseline after the bounded recovery ladder,
- or Pythia/FineWeb are unavailable beyond documented fallbacks.

## 7. Required invariants

- no future-token leakage,
- canonical hidden-state normalization everywhere,
- backbone frozen in required runs,
- explicit `layer_mix_mode` in every MTP config,
- explicit `router_init_mode` in every MTP config,
- probe-init compatibility check before loading,
- deterministic dataset IDs and manifests,
- generated final baseline config after screening,
- all plots/tables reproducible from registries and raw run artefacts.
