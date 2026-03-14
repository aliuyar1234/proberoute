# Figure Specs

## Required figures and sources
| Figure | Required output | Required source data |
|---|---|---|
| Fig 1 | `fig_system_overview.png` | hand-authored diagram source or deterministic plotting spec |
| Fig 2 | `fig_probe_heatmap_410m_top1.png` | `probe_registry.csv` filtered to `PROBE_410M` |
| Fig 3 | `fig_probe_heatmap_1b_top1.png` | `probe_registry.csv` filtered to `PROBE_1B` |
| Fig 4 | `fig_main_results.png` | `main_results.csv` |
| Fig 5 | `fig_acceptance_distribution.png` | `test_acceptance_metrics.json` traces / exported CSV |
| Fig 6 | `fig_router_support.png` | `router_metrics.json` / exported CSV |

## Rules
- Every figure must also have a machine-readable source CSV or JSON in `outputs/paper_assets/figures/source/`.
- Every figure must have a caption grounded in actual results.
- Do not fabricate a figure if the source artefact does not exist.
