# Table Specs

## Required tables and sources
| Table | Required output | Required source |
|---|---|---|
| Table 1 | `table_screening_results.csv` | `outputs/registries/screening_results.csv` |
| Table 2 | `table_main_results.csv` | `outputs/registries/main_results.csv` |
| Table 3 | `table_ablations.csv` | `outputs/registries/ablation_results.csv` |
| Table 4 | `table_resource_summary.csv` | `outputs/registries/resource_summary.csv` |
| Appendix A1 | `appendix_table_config_summary.csv` | resolved configs from completed runs |

## Rules
- Export CSV always.
- Export LaTeX only if it does not introduce drift.
- Captions must define any non-obvious metric abbreviations.
- The paper must not cite numbers that are absent from these tables or their underlying source registries.
