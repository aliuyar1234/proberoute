# MODULE_SPECS/10_PAPER_AUTOGENERATION.md

## Objective

Define how the manuscript is assembled from artefacts.

## Inputs

Primary sources:
- `outputs/registries/main_results.csv`
- `outputs/registries/screening_results.csv`
- `outputs/registries/ablation_results.csv`
- `outputs/paper_assets/figures/*`
- `outputs/paper_assets/tables/*`
- `outputs/paper_assets/appendix/claim_evidence_matrix.md`

## Required outputs

- `paper/paper_draft.md`
- optional `paper/main.tex`
- `paper/references.bib`
- `FINAL_DELIVERY.md`

## Manuscript assembly rules

1. Write the paper only after registries exist.
2. Pull numbers from registries or metric JSONs, never from terminal logs.
3. Use the title dictated by `CLAIM_CRITERIA.md`.
4. Reflect actual deviations from the original plan.
5. The final paper may not contain unresolved placeholders.

## Claim-evidence matrix

Before finalizing the paper, generate a matrix with columns:
- claim ID
- claim text
- evidence artefact(s)
- support status
- notes

Every nontrivial paper claim must map to this matrix.

## Negative-result handling

If the main method does not win:
- keep the paper,
- update the title and framing,
- emphasize what the study reveals about adaptation difficulty and probe structure,
- keep the implementation and evaluation sections equally complete.
