# MODULE_SPECS/12_RESULT_REGISTRY_AND_SELECTION.md

## Objective

Define how completed run artefacts are assembled into registries and how the finalist baseline is selected.

## Registry assembly rules

`collect_results` must:
1. scan `outputs/runs/`
2. read manifests and metric JSONs
3. skip incomplete runs only if status is not `completed`
4. emit canonical CSV registries with deterministic row ordering
5. validate JSON files against schemas before adding rows

Deterministic row ordering:
- stage
- exp_id
- model_id
- seed

## Finalist selection rules

`select_finalist` must:
1. read `outputs/registries/screening_results.csv`
2. filter to completed baseline rows only
3. apply the primary score and tie-break rules from `EXPERIMENT_PLAN.md`
4. emit `outputs/registries/finalist_selection.json`
5. emit `configs/generated/final_best_baseline_1b.yaml`

## Generated config rules

The generated final baseline config must:
- inherit from the selected screening baseline config,
- override train/val/test quotas to final budgets,
- preserve layer-mix mode and router-init mode,
- set `project.exp_id = FINAL_BEST_BASELINE_1B`,
- set `project.stage = final`.

## `finalist_selection.json` minimum fields

- selected_baseline_exp_id
- selected_run_dir
- selection_metric
- tie_break_trace
- generated_config_path
- source_screening_registry
- creation_timestamp

## Anti-drift rule

Do not handcraft the final baseline choice in the paper or the code.
Always derive it from the screening registry.
