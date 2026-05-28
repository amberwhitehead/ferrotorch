# ferrotorch-bert — `config` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - /home/doll/pytorch/torch/ (foundational ops only; HF transformers
    BertConfig at huggingface/transformers
    src/transformers/models/bert/configuration_bert.py is the
    architecture-shape upstream)
-->

## Summary

`ferrotorch-bert/src/config.rs` carries the typed configuration the BERT
encoder stack consumes. Mirrors HuggingFace's `BertConfig` JSON schema
(`huggingface/transformers/src/transformers/models/bert/configuration_bert.py`)
and exposes a typed `BertConfig` value plus a serde-fed `HfBertConfig`
that parses real `config.json` files from the Hub.

## Requirements

- REQ-1: `pub struct BertConfig` — flat hyperparameter record covering
  every field the encoder forward path consumes (`vocab_size`,
  `hidden_size`, `num_hidden_layers`, `num_attention_heads`,
  `intermediate_size`, `max_position_embeddings`, `type_vocab_size`,
  `layer_norm_eps`, `pad_token_id`). Marked `Copy`, `Clone`,
  `PartialEq`, `Debug` so it round-trips cheaply through builder
  signatures.
- REQ-2: `pub struct HfBertConfig` with `#[derive(Deserialize)]` and
  `#[non_exhaustive]` parses the full HF JSON schema. Unknown fields
  are ignored so future schema growth does not break existing
  callers; missing optional fields default to documented HF defaults
  (`layer_norm_eps=1e-12`, `type_vocab_size=2`, `hidden_act="gelu"`,
  `position_embedding_type="absolute"`, `pad_token_id=0`).
- REQ-3: `BertConfig::validate` enforces invariants the encoder
  forward pass relies on (all sizes positive,
  `hidden_size % num_attention_heads == 0`) and is called by every
  `BertConfig`-consuming constructor.
- REQ-4: `HfBertConfig::validate` enforces the architecture
  invariants ferrotorch-bert implements: only `hidden_act="gelu"`,
  only `position_embedding_type="absolute"`. Unsupported variants
  return `FerrotorchError::InvalidArgument` rather than silently
  proceeding.
- REQ-5: `BertConfig::from_hf` produces a typed `BertConfig` from a
  parsed `HfBertConfig`; round-trip preserves all encoder-relevant
  fields.
- REQ-6: `BertConfig::all_minilm_l6_v2()` exposes the published
  `sentence-transformers/all-MiniLM-L6-v2` shape (vocab 30522,
  hidden 384, 6 layers, 12 heads, intermediate 1536, max-pos 512)
  so the round-trip can be checked against a known reference.
- REQ-7: `HfBertConfig::from_file` / `from_json_str` cleanly map IO
  and serde errors onto `FerrotorchError::InvalidArgument` with a
  contextual message including the file path.
- REQ-8: `BertConfig::head_dim()` returns
  `hidden_size / num_attention_heads` — a derived quantity the
  attention block reads on construction.

## Acceptance Criteria

- [x] AC-1: `HfBertConfig::from_json_str(MINILM_CONFIG)` parses the
  published 30522-vocab MiniLM config.
- [x] AC-2: `BertConfig::from_hf` produces the same value as
  `BertConfig::all_minilm_l6_v2()` when fed the MiniLM JSON.
- [x] AC-3: `BertConfig::validate` rejects zero-valued sizes.
- [x] AC-4: `BertConfig::validate` rejects
  `num_attention_heads` that does not divide `hidden_size`.
- [x] AC-5: `HfBertConfig::validate` rejects `hidden_act="relu"`.
- [x] AC-6: `HfBertConfig::validate` rejects
  `position_embedding_type="relative_key"`.
- [x] AC-7: Unknown JSON fields parse without error
  (forward-compatible schema).

## Architecture

`pub struct BertConfig` in `config.rs` holds the nine encoder-relevant
fields as `usize` / `f64`. It is marked `#[non_exhaustive]`-compatible
in spirit (every field is `pub`; new fields can be added through
`from_hf` without breaking direct construction of `BertConfig`).

`pub struct HfBertConfig` in `config.rs` mirrors the HF JSON schema
field-by-field. `serde::Deserialize` does the parsing; `#[serde(default
= "...")]` annotations on the optional fields cover HF's documented
defaults. The struct is `#[non_exhaustive]` so adding a future field
is non-breaking.

`HfBertConfig::validate` in `config.rs` is invoked by `from_hf` before
the typed `BertConfig` is constructed; it surfaces unsupported
activation / positional-embedding variants as `InvalidArgument`. The
typed `BertConfig::validate` in `config.rs` runs a second, cheaper
shape check called from every `BertEmbeddings::new` / `BertLayer::new`
constructor.

### Non-test production consumers

- `pub use config::{BertConfig, HfBertConfig}` at
  `ferrotorch-bert/src/lib.rs:87` — grandfathered crate-root re-export.
- `BertEmbeddings::new(cfg: &BertConfig)` in
  `ferrotorch-bert/src/embeddings.rs:49` consumes `BertConfig` (calls
  `cfg.validate()`).
- `BertSelfAttention::new` / `BertSelfOutput::new` /
  `BertAttention::new` in `ferrotorch-bert/src/attention.rs` consume
  `BertConfig`.
- `BertIntermediate::new` / `BertOutput::new` / `BertLayer::new` in
  `ferrotorch-bert/src/layer.rs` consume `BertConfig`.
- `BertEncoder::new` / `BertModel::new` /
  `SentenceTransformer::new` in `ferrotorch-bert/src/model.rs`
  consume `BertConfig`.
- `load_bert_model` / `load_sentence_transformer` in
  `ferrotorch-bert/src/safetensors_loader.rs` consume `BertConfig`.

## Parity contract

`parity_ops = []`. Configuration parsing has no parity-sweep
counterpart — its contract is "round-trip the upstream JSON shape
without losing fields" and is exercised by
`parses_minilm_l6_config` / `from_hf_round_trips_minilm_l6` /
`unknown_fields_ignored` in the file's `mod tests` block.

Numerical / structural edge cases preserved:

- **Default `layer_norm_eps = 1e-12`** matches HF BERT (not the
  PyTorch nn-default `1e-5`); checked against MiniLM's `config.json`
  in `parses_minilm_l6_config`.
- **`hidden_act` accepted = `"gelu"` only.** Other variants (`relu`,
  `gelu_new`, `silu`) return `InvalidArgument` rather than proceeding
  with an unimplemented kernel.
- **`position_embedding_type` accepted = `"absolute"` only.**
  `relative_key` / `relative_key_query` (BERT's relative position
  bias variants) return `InvalidArgument`.

## Verification

Tests in `mod tests in config.rs`:

- `parses_minilm_l6_config`
- `from_hf_round_trips_minilm_l6`
- `all_minilm_l6_v2_is_valid`
- `validate_rejects_zero_fields`
- `validate_rejects_non_divisible_heads`
- `validate_rejects_unsupported_activation`
- `validate_rejects_unsupported_position_embedding_type`
- `unknown_fields_ignored`

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-bert --lib config:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct BertConfig` in `config.rs`; non-test consumer: `pub use` at `config in ferrotorch-bert/src/lib.rs` plus `BertEmbeddings::new` at `new in ferrotorch-bert/src/embeddings.rs`. |
| REQ-2 | SHIPPED | impl: `pub struct HfBertConfig` (with `#[derive(Deserialize)]`) in `config.rs`; non-test consumer: `pub use` at `ferrotorch-bert/src/lib.rs:87`. |
| REQ-3 | SHIPPED | impl: `BertConfig::validate` in `config.rs`; non-test consumer: invoked by `BertEmbeddings::new` at `ferrotorch-bert/src/embeddings.rs:49` and `BertSelfAttention::new` at `ferrotorch-bert/src/attention.rs:46`. |
| REQ-4 | SHIPPED | impl: `HfBertConfig::validate` in `config.rs`; non-test consumer: invoked by `BertConfig::from_hf` in `config.rs` (the only construction path through the HF JSON). |
| REQ-5 | SHIPPED | impl: `BertConfig::from_hf` in `config.rs`; non-test consumer: `pub use` at `ferrotorch-bert/src/lib.rs:87` (called by integration tests + pin scripts that load a real Hub checkpoint). |
| REQ-6 | SHIPPED | impl: `BertConfig::all_minilm_l6_v2` in `config.rs`; non-test consumer: `pub use` at `ferrotorch-bert/src/lib.rs:87`. |
| REQ-7 | SHIPPED | impl: `HfBertConfig::from_file` / `from_json_str` in `config.rs`; non-test consumer: `pub use` at `ferrotorch-bert/src/lib.rs:87` (called by Hub-load helpers). |
| REQ-8 | SHIPPED | impl: `BertConfig::head_dim` in `config.rs`; non-test consumer: `BertSelfAttention::new` at `ferrotorch-bert/src/attention.rs:53` reads `cfg.head_dim()`. |
