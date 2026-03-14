# MODULE_SPECS/02_DATA_PIPELINE.md

## Objective

Create a deterministic corpus-preparation pipeline that transforms open raw text or local fixtures into stable packed token arrays.

## Required public interface

Suggested functions/classes:

```python
resolve_text_field(sample: dict, priority: list[str]) -> str
normalize_text(text: str) -> str
assign_split(text: str, modulus: int, ranges: dict) -> str
tokenize_document(text: str, tokenizer, append_eos: bool) -> list[int]
pack_token_buffer(token_buffer: list[int], seq_len: int) -> tuple[list[list[int]], list[int]]
write_split_arrays(split_name: str, sequences: np.ndarray, output_dir: Path) -> Path
build_dataset_manifest(...) -> dict
build_eval_subsets(test_array: np.ndarray, prefix_len: int, n_future: int, n_prefix: int) -> dict
```

## Canonical behavior

### 1. Text-field resolution
Pick the first non-empty field from the priority list.
If none is found, skip the sample and count it.

### 2. Normalization
Apply:
- newline normalization,
- NUL-byte stripping,
- conservative whitespace trim for emptiness check only.

### 3. Split assignment
Use SHA1 over normalized text bytes.
No randomness.

### 4. Tokenization
- use the resolved tokenizer,
- append EOS between documents if available and enabled,
- do not inject BOS per packed boundary unless explicitly required.

### 5. Packing
Concatenate tokenized docs inside each split and emit fixed-length sequences.
Drop the final incomplete tail by default and record its length.

### 6. Output format
Write `np.ndarray` with shape `[N, seq_len]` and `dtype=int32` for each split.
Use `np.save` so the resulting files are portable and easy to reload.

### 7. Manifest compatibility
If the output directory already exists:
- compare compatibility-critical fields,
- fail unless `--force-rebuild` is passed.

## Edge cases

- empty normalized text -> skip and count
- unicode decode issues -> replace conservatively or skip with logging
- missing EOS -> continue without synthetic EOS and document the behavior
- local fixture path -> read JSONL from `data.local_path`

## Validation expectations

- same raw input -> same split
- same config -> same processed outputs
- manifest counts match arrays
- eval subsets are derived from `test.npy`, not an independent random sample
