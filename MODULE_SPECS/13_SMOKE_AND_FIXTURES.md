# MODULE_SPECS/13_SMOKE_AND_FIXTURES.md

## Objective

Define a local offline smoke path so the implementation can be validated before any remote dependency is touched.

## Required fixture

Use `fixtures/tiny_corpus.jsonl`.
Each line is a JSON object with a `text` field.

## Required smoke tokenizer

Implement a very small local tokenizer for smoke/tests only.
Recommended behavior:
- whitespace split
- reserve token ids for PAD/EOS/UNK
- deterministic vocab construction from the fixture corpus

## Required smoke backbone

Implement a tiny local causal LM for smoke/tests only.
Recommended path:
- a tiny GPTNeoX-like config built locally with `transformers`,
- random initialization,
- 2 layers,
- small hidden size and vocab.

## Smoke config

`configs/smoke_local_tiny.yaml` must:
- use the local fixture dataset,
- use the local toy tokenizer,
- use the local toy backbone,
- run only a few steps,
- write outputs to canonical locations.

## Smoke success criteria

`make smoke` passes if all are true:
- no internet access is needed,
- config validation passes,
- local data prep succeeds,
- probe step succeeds,
- MTP step succeeds,
- evaluation succeeds,
- schema validation succeeds.
