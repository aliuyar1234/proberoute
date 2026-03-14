# MODULE_SPECS/09_PLOTTING_AND_TABLES.md

## Objective

Define the non-negotiable plots and tables for the paper-asset pipeline.

## Required plots

### P1. System overview diagram
Show the exact pipeline:
- data prep
- probes
- probe-init artefacts
- screening runs
- finalist selection
- final comparison
- evaluation and paper assets

### P2. Probe heatmaps
Inputs:
- `probe_scores.csv`

Outputs:
- 410M top-1 heatmap
- 1B top-1 heatmap
- top-5/NLL variants may also be exported

### P3. Main results plot
Show:
- final best baseline vs final sparse main method
- mean top-1 over horizons 2–4
- mean acceptance length

### P4. Acceptance distribution
Show:
- histogram or CDF of accepted prefix lengths
- final baseline vs final main method

### P5. Router support plot
Show:
- selected layers and average weights for each horizon
- for the final sparse run

## Required tables

### T1. Screening results
All five mandatory screening methods at screening budget.

### T2. Final main comparison
Selected final baseline vs final sparse main method at final budget.

### T3. Ablations
Sparse probe-init, sparse random-init, warmup, deephead.

### T4. Resource summary
Wall-clock, train tokens, seq_len, and hardware notes.

### Appendix table
Resolved config summary.

## Plotting rules

- store source CSV/JSON used for every plot under `outputs/paper_assets/figures/source/`
- include exact experiment IDs in caption notes or metadata
- do not cherry-pick a seed unless the selection rule explicitly says so
- axis labels must be publication-ready

## Table-generation rules

- export CSV and optionally LaTeX-friendly `.tex`
- include a notes column when runs are missing or waived
- never omit a required run silently; mark it `WAIVED` or `FAILED`
