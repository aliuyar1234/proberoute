# DATA_SPEC.md

## 1. Primary dataset choice

Use `HuggingFaceFW/fineweb-edu` with config `sample-10BT` [R8].

Rationale:
- open and public,
- operationally accessible,
- large enough that deterministic token-budget slices are practical.

## 2. Fallback order

If `fineweb-edu/sample-10BT` fails:
1. `HuggingFaceFW/fineweb/sample-10BT`
2. a documented, cached open corpus with deterministic manifest

If fallback is used:
- record it in `logs/ASSUMPTIONS.md`,
- update the run manifest,
- and mention it in the paper.

## 3. Local smoke fixture path

The local offline smoke harness uses:
- `dataset_name: local_fixture_text`
- `data.local_path: fixtures/tiny_corpus.jsonl`

This path is allowed only for the smoke harness and unit/integration tests.
It is not a substitute for the real experiments.

## 4. Expected raw schema

Preferred text field priority:
1. `text`
2. `content`
3. `raw_content`

If none exist:
- fail clearly,
- inspect one sample,
- choose the semantically equivalent field,
- log the choice.

## 5. Canonical normalization before hashing and tokenization

For every document, before hashing and tokenization:
1. replace `
` and `` with `
`,
2. strip NUL bytes,
3. if the result is empty after conservative whitespace trim, skip the document.

Use the normalized text for both:
- SHA1 split assignment,
- tokenization.

## 6. Deterministic split policy

Use SHA1 over normalized UTF-8 text bytes.

Mapping:
- `hash % 1000 in [0, 19]` -> `test`
- `hash % 1000 in [20, 39]` -> `val`
- otherwise -> `train`

## 7. Tokenization policy

Tokenizer:
- use the tokenizer that matches the backbone checkpoint,
- except for the local smoke fixture, which uses the local toy tokenizer.

Rules:
- append EOS between documents if the tokenizer has one,
- do not insert BOS per packed boundary unless the model requires it,
- preserve split-local document order,
- drop the final incomplete tail by default,
- record dropped-tail tokens in the manifest.

## 8. Canonical processed format

Use canonical `.npy` arrays with `dtype=int32` and shape `[num_sequences, seq_len]`.

Per processed dataset directory:

```text
outputs/data/processed/{dataset_id}/
  manifest.json
  train.npy
  val.npy
  test.npy
  eval_future_sequences_val.npy
  eval_future_sequences_test.npy
  eval_acceptance_prefixes_val.npy
  eval_acceptance_prefixes_test.npy
```

### Dataset ID derivation

Derive `dataset_id` deterministically as:

```text
{dataset_name_slug}__{dataset_config_slug}__{tokenizer_slug}__sl{seq_len}
```

This prevents collisions across tokenizers and sequence lengths.

## 9. Token quotas

Use token budgets, not full dataset sweeps.

### Mandatory default quotas

#### Probe runs
- train: 5M
- val: 1M
- test: 1M
- recommended seq_len: 1024

#### Screening runs
- train: 20M
- val: 2M
- test: 2M
- seq_len: 2048 by default

#### Final comparison runs
- train: 50M
- val: 5M
- test: 5M
- seq_len: 2048 by default

#### Confirmatory run
- train: 20–25M
- val/test: 2M each

### Conversion to sequence counts

Actual emitted token counts are `floor(quota / seq_len) * seq_len`.
Store both the requested quota and the realized token count in the manifest.

## 10. Eval subset generation

Create deterministic subsets after processed `val.npy` and `test.npy` exist.

### Future-metrics subsets
- files:
  - `eval_future_sequences_val.npy`
  - `eval_future_sequences_test.npy`
- shape: `[N_future, seq_len]`
- `N_future = min(eval.future_metrics_sequence_count, num_{split}_sequences)`
- choose the first `N_future` packed validation or test sequences for the corresponding split.

### Acceptance-prefix subsets
- files:
  - `eval_acceptance_prefixes_val.npy`
  - `eval_acceptance_prefixes_test.npy`
- shape: `[N_accept, prefix_len]`
- `N_accept = min(eval.acceptance_prefix_count, num_{split}_sequences)`
- use the first `N_accept` validation or test sequences for the corresponding split,
- take the first `prefix_len` tokens from each sequence.

No random resampling is allowed unless a new seed is explicitly recorded and justified.

## 11. Data compatibility checks

Before reusing an existing processed dataset directory, compare the existing `manifest.json` against the resolved config for these fields:
- dataset name/config,
- tokenizer name,
- seq_len,
- split policy,
- normalization policy,
- local fixture path if applicable.

If any required field differs:
- hard fail,
- require `--force-rebuild` to overwrite.

## 12. Data validation rules

Before training, validate:
- nonzero sequence count per split,
- exact shape `[N, seq_len]` for all arrays,
- token counts match manifest,
- eval subsets are prefixes/sequences from `test.npy`,
- no train/val/test document-hash overlap,
- manifest schema validates.

## 13. Synthetic benchmark

If the synthetic benchmark is used, generate it locally rather than downloading it. It does not replace the main open-text data path.
