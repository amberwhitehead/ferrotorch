# ferrotorch-bert — `model` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - /home/doll/pytorch/torch/ (Module / Embedding / LayerNorm
    foundations; HF BertModel /
    sentence_transformers.SentenceTransformer at
    huggingface/transformers/src/transformers/models/bert/modeling_bert.py
    and the sentence-transformers pooling/normalize composition)
-->

## Summary

`ferrotorch-bert/src/model.rs` ships the top-level BERT encoder
(`BertEncoder` = `N × BertLayer`, `BertModel` = embeddings + encoder)
and the `SentenceTransformer` wrapper that adds mask-aware mean pooling
+ optional L2 normalisation. The HF state-dict ingestion path lives
here, including the `DropReport` audit trail recording which upstream
keys were intentionally not consumed.

## Requirements

- REQ-1: `pub struct BertEncoder<T: Float>` carries
  `pub layer: Vec<BertLayer<T>>` (length = `cfg.num_hidden_layers`)
  and runs each layer in sequence in `Module::forward`.
- REQ-2: `pub struct BertModel<T: Float>` carries
  `pub embeddings: BertEmbeddings<T>` + `pub encoder: BertEncoder<T>`
  + a frozen `pub config: BertConfig`. `forward_from_ids` runs
  embeddings → encoder.
- REQ-3: `BertModel::load_hf_state_dict` accepts a `StateDict` whose
  keys use the HuggingFace `BertModel` naming convention and returns
  a `DropReport` recording which upstream keys were intentionally
  dropped (`embeddings.position_ids` and any `pooler.*` keys).
  Strict mode rejects `pooler.*`; non-strict drops them.
- REQ-4: `pub struct DropReport` (returned by
  `load_hf_state_dict`) exposes
  `dropped_position_ids: bool` and
  `dropped_pooler: Vec<String>`. The pin script asserts these match
  the documented upstream-extras list so a silent state-dict drop
  cannot recur (per #1141).
- REQ-5: `pub struct SentenceTransformer<T: Float>` wraps
  `BertModel<T>` plus a `normalize: bool` flag.
  `SentenceTransformer::encode(input_ids, attention_mask,
  token_type_ids)` runs the encoder, mask-aware mean-pools, and
  optionally L2-normalises (per the sentence-transformers
  `2_Normalize` module).
- REQ-6: Input validation in `SentenceTransformer::encode`:
  empty `input_ids` → `InvalidArgument`; mismatched
  `attention_mask` length → `InvalidArgument`; all-zero
  `attention_mask` → `InvalidArgument` (no token to pool). The
  divisor of the mean-pool is the count of `mask == 1` entries.
- REQ-7: L2-normalize matches HF
  `F.normalize(emb, p=2, dim=1, eps=1e-12)` — divides by
  `max(sqrt(sum(x^2)), 1e-12)` so a zero embedding does not divide
  by zero.
- REQ-8: HF state-dict key layout for `BertModel`:
  `embeddings.{...}`, `encoder.layer.{i}.{...}`. Loadable in strict
  mode without rewriting keys.
- REQ-9: Loading a round-tripped `BertModel` state dict reproduces
  the original `forward_from_ids` output (tolerance `1e-6` on f32).

## Acceptance Criteria

- [x] AC-1: `BertModel::<f32>::new(tiny_cfg)` constructs.
- [x] AC-2: `BertModel::forward_from_ids(&[1, 5, 7, 9], None)`
  returns `[1, 4, hidden]` with finite values.
- [x] AC-3: `BertModel::named_parameters()` exposes the HF-layout
  keys (`embeddings.word_embeddings.weight`,
  `embeddings.LayerNorm.weight`,
  `encoder.layer.0.attention.self.query.weight`,
  `encoder.layer.0.attention.output.LayerNorm.weight`,
  `encoder.layer.0.intermediate.dense.weight`,
  `encoder.layer.0.output.LayerNorm.bias`,
  `encoder.layer.1.attention.self.query.weight`).
- [x] AC-4: Round-trip state-dict load reproduces the original
  forward output (tolerance `1e-6`).
- [x] AC-5: `load_state_dict(strict=true)` rejects unknown keys.
- [x] AC-6: `load_hf_state_dict(strict=false)` drops
  `embeddings.position_ids` and `pooler.*` and the `DropReport`
  records both.
- [x] AC-7: `load_hf_state_dict(strict=true)` rejects `pooler.*`.
- [x] AC-8: `SentenceTransformer::encode` with a non-trivial mask
  returns `[1, hidden]` with finite values.
- [x] AC-9: `SentenceTransformer::encode(normalize=true)` returns
  unit-norm output (tolerance `1e-5`).
- [x] AC-10: `SentenceTransformer::encode` with all-zero mask returns
  `InvalidArgument`.

## Architecture

`pub struct BertEncoder<T: Float>` in `model.rs` owns a `Vec<BertLayer<T>>`.
Its `Module::forward` clones the input then iterates through the layers,
forwarding the accumulator through each `BertLayer::forward`. The
`load_state_dict` strict path rejects any non-`layer.*` prefix.

`pub struct BertModel<T: Float>` in `model.rs` owns the embedding stack,
the encoder, and a frozen `BertConfig`. The `forward_from_ids` method
is the canonical entry point (it knows how to materialise the indices
into the embedding stack); `Module::forward` is the fallback that
applies embeddings → encoder on an already-built input tensor.

`BertModel::load_hf_state_dict` in `model.rs` is the HF-aware ingest
path. It walks the input state dict and:

- Drops `embeddings.position_ids` silently (it is a buffer, not a
  parameter; the forward path regenerates it). Records the drop in
  `DropReport.dropped_position_ids`.
- In `strict=true` mode, rejects any `pooler.*` key with
  `FerrotorchError::InvalidArgument`. In `strict=false`, drops them
  and records each in `DropReport.dropped_pooler`. The
  sentence-transformers inference path does NOT use the pooler.

`pub struct SentenceTransformer<T: Float>` in `model.rs` wraps a
`BertModel<T>` and a `normalize: bool` flag. `encode` runs the
encoder then performs:

1. Mask-aware sum over the `[1, S, hidden]` hidden states (positions
   with `mask == 0` contribute zero).
2. Divide by `kept_count` (the count of `mask == 1` entries). HF
   uses `clamp(mask_sum, min=1e-9)`; ferrotorch surfaces the
   all-zero case as `InvalidArgument` instead, so the divisor here
   is always `>= 1`.
3. If `normalize == true`, L2-normalise via
   `inv = 1 / max(sqrt(sum_sq), 1e-12)`. The `f64` accumulation in
   the sum/sq-sum loops matches HF's NumPy fallback path.

The `#[allow(clippy::assign_op_pattern)]` on the division loop is
intentional: `ferrotorch_core::Float` does not impl `DivAssign`, so
`*v /= denom` does not type-check.

### Non-test production consumers

- `pub use model::{BertEncoder, BertModel, DropReport,
  SentenceTransformer}` at `ferrotorch-bert/src/lib.rs:90`.
- `load_bert_model` at `ferrotorch-bert/src/safetensors_loader.rs:103`
  constructs `BertModel::<T>::new(cfg)?` then calls
  `model.load_hf_state_dict(&state, strict)` and returns
  `(BertModel<T>, DropReport)`.
- `load_sentence_transformer` at
  `ferrotorch-bert/src/safetensors_loader.rs:171` wraps the
  `BertModel` in a `SentenceTransformer { bert, normalize }` and
  returns `(SentenceTransformer<T>, DropReport)`.

## Parity contract

`parity_ops = []`. The model composes
`BertEmbeddings` (covered by `embeddings.md`), `BertLayer` (covered
by `layer.md`, which in turn composes attention, linear, gelu,
layer_norm — all covered by `ferrotorch-nn` parity).

Numerical / structural edge cases preserved:

- **Mask-aware mean pool.** Position with `mask == 0` contributes
  zero; divisor is `mask.sum().clamp_min(1)`. Differs from a
  straight average over all positions when padding is present.
- **L2 normalize with `eps = 1e-12`.** Matches HF's
  `F.normalize(emb, p=2, dim=1, eps=1e-12)`. Zero-vector inputs
  return zero vectors (not NaN).
- **`embeddings.position_ids` is a buffer, not a parameter.**
  Dropped silently on load (recorded in `DropReport`).
- **`pooler.*` is optional in HF; required to be absent for
  sentence-transformers.** Strict mode rejects it; non-strict drops
  it. Either way the `DropReport` records the drop so the pin
  script can audit.

## Verification

Tests in `mod tests in model.rs`:

- `tiny_model_forward_shape`
- `tiny_named_parameters_use_hf_layout`
- `round_trip_state_dict`
- `load_state_dict_strict_rejects_unknown_key`
- `load_hf_state_dict_drops_position_ids_and_pooler`
- `load_hf_state_dict_strict_rejects_pooler`
- `sentence_transformer_encode_shape_and_norm_unnormalized`
- `sentence_transformer_encode_l2_normalizes_to_unit`
- `sentence_transformer_rejects_all_zero_mask`
- `sentence_transformer_rejects_mask_length_mismatch`

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-bert --lib model:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct BertEncoder<T: Float>` + its `Module<T>` impl in `model.rs`; non-test consumer: field `encoder` of `pub struct BertModel` in `model.rs`; `BertModel::forward_from_ids` in `model.rs` calls `self.encoder.forward(&h)`. |
| REQ-2 | SHIPPED | impl: `pub struct BertModel<T: Float>` + `BertModel::new` + `BertModel::forward_from_ids` in `model.rs`; non-test consumer: `load_bert_model` at `ferrotorch-bert/src/safetensors_loader.rs:103` constructs and returns it. |
| REQ-3 | SHIPPED | impl: `BertModel::load_hf_state_dict` in `model.rs`; non-test consumer: `load_bert_model` at `ferrotorch-bert/src/safetensors_loader.rs:103` calls it. |
| REQ-4 | SHIPPED | impl: `pub struct DropReport` in `model.rs`; non-test consumer: returned by `load_bert_model` at `ferrotorch-bert/src/safetensors_loader.rs:107` and propagated up through `load_sentence_transformer` at `ferrotorch-bert/src/safetensors_loader.rs:176`. |
| REQ-5 | SHIPPED | impl: `pub struct SentenceTransformer<T: Float>` + `SentenceTransformer::encode` in `model.rs`; non-test consumer: `load_sentence_transformer` at `ferrotorch-bert/src/safetensors_loader.rs:171` constructs and returns it. |
| REQ-6 | SHIPPED | impl: input-validation branches at the top of `SentenceTransformer::encode` in `model.rs`; non-test consumer: same call path as REQ-5 — the loader returns the `SentenceTransformer` whose `encode` propagates the error. |
| REQ-7 | SHIPPED | impl: L2-normalize block at the bottom of `SentenceTransformer::encode` in `model.rs` (uses `sq_sum_f64.sqrt().max(1e-12)`); non-test consumer: same call path as REQ-5. |
| REQ-8 | SHIPPED | impl: `named_parameters` + `load_state_dict` for `BertModel` in `model.rs`; non-test consumer: `BertModel::load_hf_state_dict` in `model.rs` calls `self.load_state_dict(&remapped, strict)` after remapping the HF keys. |
| REQ-9 | SHIPPED | impl: round-trip-tested via `round_trip_state_dict` in `mod tests in model.rs`; non-test consumer: the `load_bert_model` round trip in `ferrotorch-bert/src/safetensors_loader.rs:103` is the production path that exercises the same logic against a real `model.safetensors` file. |
