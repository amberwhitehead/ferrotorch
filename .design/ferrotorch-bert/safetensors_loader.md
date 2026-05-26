# ferrotorch-bert — `safetensors_loader` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - /home/doll/pytorch/torch/ (the safetensors crate is the on-disk
    spec; HF/sentence-transformers conventions provide the key
    layout the loader expects)
-->

## Summary

`ferrotorch-bert/src/safetensors_loader.rs` turns a path-to-safetensors
on disk into a loaded `BertModel` or `SentenceTransformer`. It handles
two upstream-specific quirks: (a) the `embeddings.position_ids` buffer
that ships as `int64` (and therefore cannot decode through the
`f32`-typed `load_safetensors::<f32>` path) and (b) the optional
`pooler.*` keys that sentence-transformers does not consume. Both are
filtered cleanly so the downstream `BertModel::load_hf_state_dict`
sees a clean state dict.

## Requirements

- REQ-1: `load_bert_model<T: Float>(weights_path, cfg, strict)` loads
  the safetensors at `weights_path`, decodes it into a typed
  `StateDict<T>`, drops the int64
  `embeddings.position_ids` buffer at decode time, and calls
  `BertModel::load_hf_state_dict` to populate the model. Returns
  `(BertModel<T>, DropReport)` carrying the full audit trail.
- REQ-2: `load_sentence_transformer<T: Float>(weights_path, cfg,
  normalize, strict)` wraps `load_bert_model` and returns
  `(SentenceTransformer<T>, DropReport)`. `normalize` is the value
  matching the upstream `2_Normalize` module (`true` for
  `sentence-transformers/all-MiniLM-L6-v2`).
- REQ-3: Internal `load_safetensors_filtered<T: Float>` re-serialises
  only the f32-decodable subset into an in-memory safetensors blob
  before invoking the generic `ferrotorch_serialize::load_safetensors::<T>`
  decoder. This avoids re-implementing dtype dispatch on the
  loader side.
- REQ-4: After decoding the filtered state, the loader re-inserts a
  placeholder `embeddings.position_ids` entry (if it was present in
  the upstream file) so the downstream `DropReport` correctly
  records the drop. The placeholder tensor is never consumed by any
  parameter slot.
- REQ-5: `key_is_skippable_at_decode` is the single source of truth
  for "this key cannot pass through `load_safetensors::<f32>`". It
  currently flags only `embeddings.position_ids` (the int64 buffer);
  `pooler.*` keys decode cleanly as f32 and are dropped later by
  `BertModel::load_hf_state_dict`.
- REQ-6: All IO / parse errors map onto
  `FerrotorchError::InvalidArgument` with a contextual message
  including the offending path.

## Acceptance Criteria

- [x] AC-1: Round-trip — `save_safetensors(&model.state_dict(), &path)`
  followed by `load_bert_model::<f32>(&path, tiny_cfg, true)` returns
  a model whose `forward_from_ids` matches the source within `1e-6`.
- [x] AC-2: The `DropReport` from AC-1 has
  `dropped_position_ids == false` and empty `dropped_pooler` (the
  round-tripped state dict has neither).
- [x] AC-3: `load_sentence_transformer::<f32>(&path, tiny_cfg, true,
  true)` returns a `SentenceTransformer` whose `encode` produces a
  unit-norm `[1, hidden]` output.

## Architecture

`load_bert_model` in `safetensors_loader.rs` performs:

1. Read the raw safetensors bytes to capture the upstream key list
   (so the `DropReport` reflects the upstream checkpoint, not the
   post-filter view). The raw `SafeTensors` parser is dropped before
   the filtered decode.
2. Call `load_safetensors_filtered::<T>` to decode only the
   T-decodable keys.
3. If the upstream contained `embeddings.position_ids`, insert a
   `zeros::<T>(&[1])` placeholder so the model's `load_hf_state_dict`
   sees the key and records the drop.
4. Construct `BertModel::<T>::new(cfg)?` then call
   `model.load_hf_state_dict(&state, strict)`.

`load_safetensors_filtered` in `safetensors_loader.rs` re-serialises
the kept tensor views into an in-memory safetensors blob, writes to a
`tempfile::NamedTempFile`, and feeds the file path to the audited
generic decoder. The temp file approach reuses the audited decoder
rather than copy-pasting its dtype dispatch.

`key_is_skippable_at_decode` in `safetensors_loader.rs` returns
`true` only for `embeddings.position_ids`. The `pooler.*` keys go
through the f32 decoder unchanged and are dropped later by
`BertModel::load_hf_state_dict`.

`load_sentence_transformer` in `safetensors_loader.rs` is a thin
wrapper that re-uses `load_bert_model` and wraps the result in a
`SentenceTransformer { bert, normalize }`.

### Non-test production consumers

- `pub use safetensors_loader::{load_bert_model,
  load_sentence_transformer}` at `ferrotorch-bert/src/lib.rs:91`.
- Both helpers are the canonical public entry point for loading a
  real Hub checkpoint into a `BertModel` / `SentenceTransformer`;
  they are the ferrotorch-bert API surface consumed by integration
  tests and pin scripts.

## Parity contract

`parity_ops = []`. The loader composes
`ferrotorch_serialize::load_safetensors` (covered by the
`ferrotorch-serialize` parity surface) and
`BertModel::load_hf_state_dict` (covered by `model.md`).

Numerical / structural edge cases preserved:

- **`embeddings.position_ids` is int64 upstream.** The
  `load_safetensors::<f32>` decoder cannot read it; the filter drops
  it at decode time. A placeholder is then re-inserted so the
  downstream `DropReport` captures the upstream key.
- **`pooler.*` is f32 upstream.** It decodes cleanly and is dropped
  by `BertModel::load_hf_state_dict` (strict mode rejects it;
  non-strict records it in `DropReport.dropped_pooler`).
- **Single-file safetensors only.** Sharded checkpoints (multi-file)
  are not supported — sentence-transformers/all-MiniLM-L6-v2 is a
  single-file checkpoint so this is sufficient.

## Verification

Tests in `mod tests in safetensors_loader.rs`:

- `round_trip_safetensors_into_bert_model`
- `round_trip_into_sentence_transformer`

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-bert --lib safetensors_loader:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn load_bert_model` in `safetensors_loader.rs`; non-test consumer: `pub use` at `ferrotorch-bert/src/lib.rs:91` (the canonical Hub-load entry point used by integration tests + pin scripts). |
| REQ-2 | SHIPPED | impl: `pub fn load_sentence_transformer` in `safetensors_loader.rs`; non-test consumer: `pub use` at `ferrotorch-bert/src/lib.rs:91`. |
| REQ-3 | SHIPPED | impl: `load_safetensors_filtered` in `safetensors_loader.rs`; non-test consumer: `load_bert_model` in `safetensors_loader.rs` invokes it. |
| REQ-4 | SHIPPED | impl: placeholder re-insert block inside `load_bert_model` in `safetensors_loader.rs`; non-test consumer: same call path — `load_bert_model` is the only entry into the helper and uses the placeholder when building the state dict that feeds `BertModel::load_hf_state_dict`. |
| REQ-5 | SHIPPED | impl: `key_is_skippable_at_decode` in `safetensors_loader.rs`; non-test consumer: `load_safetensors_filtered` in `safetensors_loader.rs` invokes it; that helper is itself consumed by `load_bert_model`. |
| REQ-6 | SHIPPED | impl: `.map_err(\|e\| FerrotorchError::InvalidArgument { message: format!(...) })` patterns throughout `safetensors_loader.rs`; non-test consumer: error type surfaces through the `pub use` at `ferrotorch-bert/src/lib.rs:91`. |
