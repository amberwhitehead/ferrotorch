# ferrotorch-bert — `embeddings` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - /home/doll/pytorch/torch/ (Embedding / LayerNorm at
    torch/nn/modules/sparse.py + torch/nn/modules/normalization.py;
    HF BertEmbeddings at
    huggingface/transformers/src/transformers/models/bert/modeling_bert.py
    is the composition-shape upstream)
-->

## Summary

`ferrotorch-bert/src/embeddings.rs` is the BERT input-embedding block:
three lookups (word + position + token_type) summed and passed through
a LayerNorm. Mirrors HuggingFace's `BertEmbeddings` minus the
inference-time-noop Dropout. The `embeddings.position_ids` buffer that
HF ships in `model.safetensors` is regenerated each forward from the
input length and never stored as a parameter.

## Requirements

- REQ-1: `pub struct BertEmbeddings<T: Float>` holds three
  `ferrotorch_nn::Embedding<T>` tables (`word_embeddings`,
  `position_embeddings`, `token_type_embeddings`) and one
  `ferrotorch_nn::LayerNorm<T>` (`LayerNorm`) so the HF state-dict
  keys map directly onto the field names.
- REQ-2: `BertEmbeddings::new(cfg)` validates the config and
  constructs randomly-initialised tables of the correct shape
  (`[vocab_size, hidden_size]`, `[max_position_embeddings,
  hidden_size]`, `[type_vocab_size, hidden_size]`,
  `LayerNorm([hidden_size], eps=cfg.layer_norm_eps)`).
- REQ-3: `BertEmbeddings::forward_from_ids` runs the three lookups,
  sums them, and applies LayerNorm. Sequence length must satisfy
  `1 <= S <= max_position_embeddings`; positional indices are
  generated as `0..S`; `token_type_ids` defaults to all-zero (the
  sentence-transformers convention).
- REQ-4: Input validation: empty `input_ids` → `InvalidArgument`;
  oversized sequence → `InvalidArgument`; mismatched
  `token_type_ids` length → `InvalidArgument`; out-of-range
  `token_type_id` → `InvalidArgument`.
- REQ-5: `Module<T>` impl exposes
  `parameters` / `parameters_mut` / `named_parameters` /
  `state_dict` / `load_state_dict` / `train` / `eval`
  with the upstream HF key layout
  (`word_embeddings.weight`, `position_embeddings.weight`,
  `token_type_embeddings.weight`, `LayerNorm.weight`,
  `LayerNorm.bias`).
- REQ-6: `Module::forward` accepts a pre-summed `[1, S, hidden]`
  tensor and only applies LayerNorm (the public path is
  `forward_from_ids`; this fallback keeps the `Module` trait
  satisfied for generic tooling that walks sub-module forwards).
- REQ-7: Output shape is `[1, S, hidden_size]` — promoted from the
  `[S, hidden]` 2-D sum by `reshape_to_3d` before LayerNorm.

## Acceptance Criteria

- [x] AC-1: `BertEmbeddings::<f32>::new(&tiny_cfg)` constructs.
- [x] AC-2: `forward_from_ids(&[1, 5, 7, 9], None)` returns a
  `[1, 4, hidden]` tensor with finite values.
- [x] AC-3: `forward_from_ids(&[..too long..], None)` returns
  `InvalidArgument`.
- [x] AC-4: `forward_from_ids(ids, Some(&bad_len_types))` returns
  `InvalidArgument`.
- [x] AC-5: `named_parameters()` exposes the HF-layout keys
  (`word_embeddings.weight`, `position_embeddings.weight`,
  `token_type_embeddings.weight`, `LayerNorm.weight`,
  `LayerNorm.bias`).

## Architecture

`pub struct BertEmbeddings<T: Float>` in `embeddings.rs` owns
the four sub-modules plus three cached shape fields
(`hidden_size`, `max_position_embeddings`, `type_vocab_size`).
`#[derive(Debug)]` is mandatory because the crate denies
`missing_debug_implementations` at the crate root.

The forward path in `BertEmbeddings::forward_from_ids` in
`embeddings.rs` builds the three index tensors via
`float_index_tensor` (a u32→T cast through
`ferrotorch_core::numeric_cast::cast`), calls the three embedding
lookups, sums with `ferrotorch_core::grad_fns::arithmetic::add`,
reshapes to `[1, S, hidden]`, and applies LayerNorm.

`Module::load_state_dict` in `embeddings.rs` accepts strict and
non-strict modes; in strict mode any key whose prefix is not one of
`{word_embeddings, position_embeddings, token_type_embeddings,
LayerNorm}` returns `InvalidArgument` so an unknown field cannot
silently land into a parameter.

### Non-test production consumers

- `pub use embeddings::BertEmbeddings` at
  `ferrotorch-bert/src/lib.rs:88`.
- `BertModel { embeddings: BertEmbeddings<T>, ... }` field at
  `ferrotorch-bert/src/model.rs:144`; `BertModel::new` at
  `ferrotorch-bert/src/model.rs:161` constructs the embeddings;
  `BertModel::forward_from_ids` at
  `ferrotorch-bert/src/model.rs:185` calls
  `self.embeddings.forward_from_ids`.

## Parity contract

`parity_ops = []`. The embedding stack composes upstream
`embedding` (covered by `ferrotorch-nn` parity) and `layer_norm`
(covered by `ferrotorch-nn` parity).

Numerical / structural edge cases preserved:

- **`position_ids` buffer dropped on load.** HF ships
  `embeddings.position_ids` as an `int64` `[1, max_pos]` constant
  buffer; ferrotorch regenerates it each forward and never stores
  it. The drop is visible in `BertModel::load_hf_state_dict`'s
  `DropReport`.
- **`token_type_ids` defaults to all-zero.** Matches the
  sentence-transformers single-sentence path. Callers passing
  two-sentence inputs must supply the segment ids explicitly.
- **Empty `input_ids` rejected.** Upstream would happily run a
  zero-length sequence; ferrotorch surfaces this as
  `InvalidArgument` because the downstream attention shape check
  would fail anyway.

## Verification

Tests in `mod tests in embeddings.rs`:

- `forward_from_ids_produces_correct_shape`
- `forward_from_ids_rejects_too_long_sequence`
- `forward_from_ids_rejects_bad_token_type_length`
- `named_parameters_use_hf_layout`

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-bert --lib embeddings:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct BertEmbeddings<T: Float>` in `embeddings.rs`; non-test consumer: field of `pub struct BertModel` at `ferrotorch-bert/src/model.rs:144`. |
| REQ-2 | SHIPPED | impl: `BertEmbeddings::new` in `embeddings.rs`; non-test consumer: `BertModel::new` at `ferrotorch-bert/src/model.rs:161` calls it. |
| REQ-3 | SHIPPED | impl: `BertEmbeddings::forward_from_ids` in `embeddings.rs`; non-test consumer: `BertModel::forward_from_ids` at `ferrotorch-bert/src/model.rs:185` calls it. |
| REQ-4 | SHIPPED | impl: input-validation branches at the top of `forward_from_ids` in `embeddings.rs`; non-test consumer: same call path as REQ-3 — `BertModel::forward_from_ids` at `ferrotorch-bert/src/model.rs:185` propagates the error. |
| REQ-5 | SHIPPED | impl: `impl<T: Float> Module<T> for BertEmbeddings<T>` in `embeddings.rs`; non-test consumer: `Module` blanket calls from `BertModel`'s `Module` impl at `ferrotorch-bert/src/model.rs:262` (parameters / state_dict / load_state_dict). |
| REQ-6 | SHIPPED | impl: `Module::forward` for `BertEmbeddings` in `embeddings.rs`; non-test consumer: `BertModel::forward` at `ferrotorch-bert/src/model.rs:263` calls `self.embeddings.forward(input)`. |
| REQ-7 | SHIPPED | impl: `reshape_to_3d` call inside `forward_from_ids` in `embeddings.rs`; non-test consumer: `BertModel::forward_from_ids` at `ferrotorch-bert/src/model.rs:185`, which feeds the resulting `[1, S, hidden]` into `BertEncoder::forward`. |
