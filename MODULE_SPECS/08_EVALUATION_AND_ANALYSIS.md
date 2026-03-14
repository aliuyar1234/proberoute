# MODULE_SPECS/08_EVALUATION_AND_ANALYSIS.md

## Objective

Define how future metrics, acceptance metrics, router analyses, and bootstrap CIs are computed.

## A. Held-out future metrics

For each horizon `k`:
- compute top-1 accuracy
- compute top-5 accuracy
- compute NLL

Evaluation uses deterministic sequences from `eval_future_sequences_val.npy` and `eval_future_sequences_test.npy`.

## B. Greedy self-verification acceptance

### Purpose
Estimate how useful the MTP proposals are relative to the frozen base model’s greedy continuation.

### Canonical algorithm

For each prefix in `eval_acceptance_prefixes_val.npy` or `eval_acceptance_prefixes_test.npy`:
1. set `current_prefix = prefix`
2. for each decode step up to `max_new_tokens`:
   - run the frozen base model on `current_prefix`
   - greedily roll the base model forward for up to `K_max = max(horizons)` tokens to obtain `base_block`
   - run the MTP model on the same `current_prefix` to obtain `mtp_block`
   - compute `accepted_len = longest_common_prefix(base_block, mtp_block)`
   - record the trace entry
   - append **one base-greedy token** (`base_block[0]`) to `current_prefix`
3. aggregate over all prefixes and steps

This intentionally does not simulate actual speculative jumps.

### Required outputs
- `mean_accept_len`
- acceptance rates at depths 1..4
- accepted-length histogram
- raw traces under `artifacts/traces/acceptance_traces.jsonl`

### Trace row schema

Each trace row should include:
- exp_id
- prefix_index
- decode_step
- accepted_len
- base_block
- mtp_block

## C. Router analysis

For each trained router:
- compute entropy per horizon
- export selected layers per horizon
- compute average router weights on validation data
- compute overlap with probe top-m layers

## D. Bootstrap confidence intervals

Use paired bootstrap over evaluation units:
- sequences for future metrics
- prefixes for acceptance metrics

Store:
- mean
- lower
- upper
- bootstrap seed
- number of samples

## E. Optional synthetic benchmark

If implemented:
- keep it separate from main tables unless needed for a secondary claim
- evaluate exact match and optionally token-level accuracy
