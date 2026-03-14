# MODULE_SPECS/11_SYNTHETIC_STARGRAPH.md

## Objective

Provide an optional, cheap planning-sensitive synthetic benchmark generated locally.

## Required stage order

If the synthetic benchmark is used:
1. run `SYNTH_STARGRAPH_PROBE` first,
2. then run `SYNTH_STARGRAPH`.

It must never block the main open-text study.

## Task concept

Create simple text problems derived from a star graph:
- central hub node `0`
- leaf nodes `1..N`

Example:

```text
Graph: star with hub 0 and leaves 1..8.
Question: shortest path from 3 to 7?
Answer: 3 0 7
```

## Data generation rules

- deterministic seed
- no exact prompt overlap between splits
- generate train/val/test locally
- record generator parameters in a manifest

Suggested defaults:
- `N=8` or `N=16`
- 100k train examples
- 5k val
- 5k test

## Metrics

- exact match on the answer string
- optional token-level accuracy
- optional acceptance proxy using the same evaluator as the main suite
