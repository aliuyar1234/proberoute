# CLAIM_CRITERIA.md

Use this file to decide what the paper is allowed to claim.

## Status labels

- **Strongly supported**: all required artefacts exist and the numeric condition below is met.
- **Partially supported**: artefacts exist but the effect is mixed, weak, or horizon-specific.
- **Unsupported**: artefacts are missing or the condition fails.

## Claim C1 — non-uniform layer × horizon structure

Strongly supported if both are true on at least one mandatory probe run:
1. the best layer index for horizon 1 differs from the best layer index of horizon 3 or 4 by at least 2 layers,
2. for at least two horizons, `max(top5) - min(top5) >= 0.5` absolute percentage points across layers.

Partially supported if heatmaps are visibly non-flat but only one of the numeric conditions is met.

Unsupported if probe scores are essentially flat after validated probe implementation.

Allowed wording:
- Strong: “probe heatmaps show horizon-dependent depth structure.”
- Partial: “probe heatmaps show mild but non-uniform depth structure.”
- Unsupported: “under our budgets, probe heatmaps were largely flat.”

## Claim C2 — main method beats baselines

Use the **final comparison** (`main_results.csv`) for the headline claim.

Strongly supported if all are true on the final-budget comparison:
1. `MAIN_SPARSE_PROBE_1B_FINAL` beats the selected final baseline on `test_mean_top1_h2_h4`,
2. `MAIN_SPARSE_PROBE_1B_FINAL` is not worse on `test_mean_accept_len`,
3. `mean_nll_h1_h4` does not regress by more than 5% relative,
4. both runs have matching budgets and compatible configs.

Partially supported if gains appear on only some horizons or only on acceptance but not on mean top-1.

Unsupported if the final baseline clearly wins after the recovery ladder.

Allowed wording:
- Strong: “sparse probe-init routing improves held-out MTP adaptation over the selected baseline.”
- Partial: “sparse probe-init routing is competitive and improves some horizons / acceptance metrics.”
- Unsupported: “the selected baseline remained stronger under our budgets.”

## Claim C3 — practical relevance via acceptance proxy

Strongly supported if final sparse routing improves `mean_accept_len` and at least one depth-specific acceptance rate beyond depth 1.

Partially supported if acceptance gains are tiny or inconsistent but future metrics improve.

Unsupported if acceptance gains are absent or negative.

Allowed wording must always mention this is a **proxy**, not a production throughput benchmark.

## Claim C4 — adaptation-limit interpretation

Strongly supported if:
- implementation validation passed,
- the bounded recovery ladder was followed,
- and the main method still failed or only weakly helped.

This claim becomes the main framing for the negative-result paper.

## Title selection rule

Use the positive title only if C1 is at least partially supported and C2 is at least partially supported.
Otherwise use the fallback title from `PAPER_PLAN.md`.

## Final placeholder rule

The final manuscript may not contain unresolved tokens such as:
- `TODO`
- `TBD`
- `{TITLE}`
- `{ABSTRACT}`
- `FIXME`

Intermediate drafts may contain placeholders, but the completion gate does not.
