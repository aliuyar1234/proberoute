# ProbeRoute: Probes as Routing Priors for Frozen-Backbone Multi-Token Prediction

[![Paper PDF](https://img.shields.io/badge/Paper-PDF-B31B1B?style=flat-square&logo=adobeacrobatreader&logoColor=white)](ProbeRoute_paper.pdf)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19022709-0A7BBB?style=flat-square)](https://doi.org/10.5281/zenodo.19022709)
[![Manuscript Source](https://img.shields.io/badge/LaTeX-source-1D4ED8?style=flat-square&logo=latex&logoColor=white)](paper/incoming_latex/main.tex)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-2ea44f?style=flat-square)](LICENSE)
[![Scope](https://img.shields.io/badge/Scope-Frozen--Backbone%20MTP-5B4B8A?style=flat-square)](#scope)

Ali Uyar
Independent Researcher

**Paper title:** *ProbeRoute: Probes as Routing Priors for Frozen-Backbone Multi-Token Prediction*

This repository accompanies a methods paper on adapting a frozen autoregressive language model for explicit multi-token prediction. The central question is sharper than a generic "better adapter" claim: does probe-initialized sparse routing improve frozen-backbone multi-token prediction relative to the strongest dense baseline selected under the same screening and final-rerun protocol? ProbeRoute probes every layer for future-token predictability, uses the resulting scores to initialize a sparse per-horizon router, and trains lightweight multi-token heads on top of the frozen backbone.

## Abstract

Frozen autoregressive language models are trained for next-token prediction, but their hidden states can still encode information about several future tokens. ProbeRoute tests whether that latent structure can be converted into a better explicit multi-token adapter without unfreezing the backbone. The method first runs a one-time offline probe stage across depth, then uses probe-derived top-5 scores to initialize a sparse top-*m* router over frozen hidden states. Under a stage-gated protocol with mandatory probes, screening baselines, a finalist-selection step, and a final 1B rerun, the resulting sparse adapter beats the strongest selected dense frozen-backbone baseline on the paper's two headline held-out metrics. On the final 1B comparison, ProbeRoute improves test top-1 (h2-4) from 0.1162 to 0.1172 and the speculative draft acceptance proxy from 1.1110 to 1.1188, while changing test NLL (h1-4) only from 5.3769 to 5.3780. Probe heatmaps reveal horizon-dependent depth structure at both 410M and 1B, the learned sparse router concentrates mass on the same depth bands, and the random-initialization ablation is weaker than probe-initialized sparse routing. By contrast, loss warmup and deeper far-horizon heads do not improve over the base sparse configuration at the tested budget. In this frozen-backbone setting, future-token probes are not merely descriptive diagnostics; they are a useful routing prior for explicit multi-token prediction.

## Main Result

The final 1B comparison at the 50M-token budget — both finalists using probe-derived initialization — is the paper's central quantitative result:

| Model              | Test top-1 (h2-4) | Test accept len. | Test NLL (h1-4) |
| ------------------ | ----------------- | ---------------- | --------------- |
| Dense finalist     | 0.1162            | 1.1110           | **5.3769**      |
| ProbeRoute (ours)  | **0.1172**        | **1.1188**       | 5.3780          |
| Delta (sparse-dense) | +0.00099        | +0.00781         | +0.00110        |

The raw deltas are small but directionally consistent: top-1 improves at every horizon, the speculative draft acceptance proxy improves, and aggregate NLL changes only minimally (about 0.02% relative). The practical point is stronger than the raw magnitude — the gain comes from a more *selective* routing interface, not from a heavier backbone adaptation recipe. The random-init ablation weakens the sparse model; loss warmup and deeper far-horizon heads do not improve over the base sparse configuration at the tested budget.

## Contributions

1. We show that future-token probe heatmaps reveal horizon-dependent depth structure at both 410M and 1B, with shorter horizons peaking later and farther horizons shifting earlier in depth.
2. We turn those probe measurements into a concrete frozen-backbone adapter — a sparse top-*m* layer router initialized from probe top-5 scores — and evaluate it against screening-selected last-layer and dense weighted-hidden-state baselines.
3. We find that the final 1B sparse run beats the strongest selected dense baseline on held-out future-token top-1 and speculative-draft acceptance metrics while relying on a more selective routing interface, and the random-init ablation is the one that meaningfully weakens the result.

## Scope

This study is deliberately focused.

- one frozen-backbone setting; the backbone is never unfrozen
- two scales: 410M for probes and screening diagnostics, 1B for the final comparison
- one final-budget comparison against a screening-selected dense finalist, not a broad sweep
- explicit multi-token prediction with held-out top-1 and speculative-draft acceptance as the headline metrics

It does *not* claim that sparse routing universally dominates dense routing, that frozen adapters beat full finetuning, or that the acceptance proxy is a throughput benchmark.

## Paper

- Compiled PDF: [`ProbeRoute_paper.pdf`](ProbeRoute_paper.pdf)
- LaTeX source: [`paper/incoming_latex/main.tex`](paper/incoming_latex/main.tex)
- Figures and tables: [`paper/figures/`](paper/figures/), [`paper/tables/`](paper/tables/), [`paper/appendix/`](paper/appendix/)
- TMLR submission archive: [`paper/layermix_mtp_paper_tmlr_source.zip`](paper/layermix_mtp_paper_tmlr_source.zip)

## Repository Layout

- [`src/`](src/) — implementation code: probes, sparse routing, MTP heads, training, evaluation
- [`configs/`](configs/) — experiment entrypoints (smoke, screening, final-budget runs)
- [`schemas/`](schemas/) — result and manifest schemas
- [`fixtures/`](fixtures/) — offline smoke-test data
- [`tests/`](tests/) — unit and integration tests
- [`paper/`](paper/) — manuscript source, figures, tables, and compiled PDF
- [`outputs/`](outputs/) — run outputs (gitignored; local only)

## Reproducibility

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — system architecture and data flow
- [`DATA_SPEC.md`](DATA_SPEC.md) — data formats and preparation contracts
- [`ENVIRONMENT.md`](ENVIRONMENT.md) — pinned runtime environment
- [`RESULTS_SCHEMA.md`](RESULTS_SCHEMA.md) — structure of run outputs and registries
- [`MODULE_SPECS/`](MODULE_SPECS/) — contributor-facing module contracts
- [`CITATION.cff`](CITATION.cff) — citation metadata

## Citation

```bibtex
@software{uyar2026proberoute,
  author  = {Uyar, Ali},
  title   = {ProbeRoute: Probes as Routing Priors for Frozen-Backbone Multi-Token Prediction},
  year    = {2026},
  doi     = {10.5281/zenodo.19022709},
  url     = {https://github.com/aliuyar1234/proberoute}
}
```
