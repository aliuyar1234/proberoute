# Generated config directory

This directory exists so `select_finalist` has a canonical write target.

## Required generated file
- `final_best_baseline_1b.yaml`

## Rule
The shipped file is only a stub. The real pipeline must overwrite it after reading
`outputs/registries/screening_results.csv` and applying the deterministic finalist-selection rule.
