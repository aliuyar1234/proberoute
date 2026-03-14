# Configs

These YAML files are the canonical public execution entrypoints for ProbeRoute.

## Merge Rules
- `inherit_from` paths are resolved relative to the child file.
- Dictionaries deep-merge recursively.
- Lists replace entirely.
- Scalars replace entirely.
- Child values always win.
- Cycles or missing parents are hard failures.

## Design Guardrails
The config contract is defined by [01_CONFIG_AND_CLI.md](../MODULE_SPECS/01_CONFIG_AND_CLI.md). The highest-signal anti-drift fields are:
- `project.stage`
- `project.priority`
- `model.layer_mix_mode`
- `model.router_init_mode`
- `model.probe_init_metric`
- `model.hidden_norm`
- `train.stop_mode`

## File Roles
| File | Purpose |
|---|---|
| `base.yaml` | shared defaults for real runs |
| `fallback_48gb.yaml` | fallback overlay for smaller-memory GPUs |
| `smoke_local_tiny.yaml` | offline smoke path |
| `probe_410m.yaml` | 410M probe run |
| `probe_1b.yaml` | 1B probe run |
| `probe_2p8b.yaml` | optional 2.8B probe prerequisite |
| `screen_last_linear_1b.yaml` | simplest last-layer baseline |
| `screen_last_mlp_1b.yaml` | stronger last-layer baseline |
| `screen_dense_whs_random_1b.yaml` | dense weighted-hidden-state baseline |
| `screen_dense_whs_probe_init_1b.yaml` | dense weighted-hidden-state plus probe init |
| `screen_sparse_random_1b.yaml` | sparse random-init ablation |
| `screen_sparse_probe_init_1b.yaml` | screening version of the main sparse method |
| `final_sparse_probe_init_1b.yaml` | final-budget sparse rerun |
| `final_best_baseline_template.yaml` | template used when emitting the finalist baseline config |
| `ablation_sparse_warmup_1b.yaml` | warmup ablation |
| `ablation_sparse_deephead_1b.yaml` | deephead ablation |
| `confirm_best_2p8b.yaml` | 2.8B confirmatory rerun |
| `confirm_best_410m_seed2.yaml` | low-cost confirmatory fallback |
| `synthetic_stargraph_probe.yaml` | optional synthetic probe path |
| `synthetic_stargraph.yaml` | optional synthetic MTP path |
| `generated/` | runtime-generated configs; not tracked except for this README |

## Policy
- Keep configs declarative and reproducible.
- Save resolved configs into each run directory.
- Treat `configs/generated/` as runtime output rather than hand-edited source.
