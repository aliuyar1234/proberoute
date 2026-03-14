# MODULE_SPECS/00_SCOPE_GUARDRAILS.md

## Purpose

This file is a boundary document to prevent scope drift during implementation.

## Hard guardrails

1. The core project is **frozen-backbone adaptation to MTP**.
2. The main mechanism is **probe-initialized sparse layer routing**.
3. The main model family is **Pythia**.
4. The main data source is **FineWeb-Edu sample-10BT**.
5. The main practical metric is **greedy self-verification acceptance**, not production throughput.
6. The output is a **paper**, not just code.
7. The implementation repo uses lowercase `configs/` and `schemas/`.
8. The local smoke harness is allowed to use a toy local model and tokenizer, but only for smoke/tests.

## Explicit non-goals

Do not implement these as primary contributions:
- draft-model speculation,
- Medusa reproduction,
- registers / MuToR,
- TOP/FSP replication,
- instruction-tuning transfer,
- code-agent evaluations,
- large-scale joint backbone tuning.

## Allowed flexibility

You may choose among small implementation details if they preserve the hypothesis:
- Typer vs argparse
- logging library
- exact plotting code
- exact checkpoint serialization format

## Not allowed without logging a decision

- changing model family
- changing main dataset
- changing main hypotheses
- increasing horizons beyond 4 in the main track
- making backbone tuning the default
- skipping the finalist-selection step
