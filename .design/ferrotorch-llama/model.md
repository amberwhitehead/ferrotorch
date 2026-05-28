# ferrotorch-llama — `model` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - HuggingFace transformers/models/llama/modeling_llama.py
    (LlamaModel:334-409, LlamaForCausalLM:413-485)
-->

## Summary

`ferrotorch-llama/src/model.rs` ships the top-level Llama decoder
stack (`LlamaModel` = embedding + N decoder layers + final RMSNorm)
and the causal-LM wrapper (`LlamaForCausalLM` = model + `lm_head`
projection). It exposes the user-facing entry points:
`forward_from_ids` (full-prefix forward for any seq length) and
`forward_one_with_cache` (single-token incremental forward against a
persistent KV cache). The HF-naming-aware state-dict ingest
(`load_hf_state_dict`) handles the tied-embeddings convention.

## Requirements

- REQ-1: `pub struct LlamaModel<T: Float>` carries `embed_tokens:
  Embedding<T>`, `layers: Vec<LlamaDecoderLayer<T>>` (length =
  `cfg.num_hidden_layers`), and `norm: RMSNorm<T>`. Its
  `Module::forward` runs each layer sequentially then the final norm.
- REQ-2: `pub struct LlamaForCausalLM<T: Float>` wraps a
  `LlamaModel<T>`, a `lm_head: Linear<T>` (with `bias = false` for
  every Llama variant), and a frozen `config: LlamaConfig`.
- REQ-3: `LlamaForCausalLM::forward_from_ids(ids: &[u32])` returns
  logits `[1, seq_len, vocab_size]` via embedding → decoder → lm_head.
  Empty `ids` yields `InvalidArgument`; non-representable token ids
  (vocab_size > T's exact integer range) yield `InvalidArgument`.
- REQ-4: `LlamaForCausalLM::forward_one_with_cache(token, &cache)`
  performs a single-token incremental forward returning `(Vec<f64>
  logits, LlamaKvCache<T>)`. Result is mathematically equivalent to
  `forward_from_ids` over the full prefix `[prev.., token]` but does
  `O(seq·hidden)` work instead of `O(seq²·hidden)`.
- REQ-5: `LlamaForCausalLM::load_hf_state_dict(hf, strict)` accepts
  HuggingFace-keyed state dicts and remaps them to the ferrotorch
  parameter layout. When `cfg.tie_word_embeddings` is set and
  `lm_head.weight` is missing, copies `model.embed_tokens.weight`
  into its slot (matching HF's tied-embeddings serialization).
- REQ-6: `named_parameters` exposes the HF layout:
  `model.embed_tokens.weight`, `model.layers.{i}.{...}`,
  `model.norm.weight`, `lm_head.weight`. The HF and ferrotorch
  key shapes are identical so a checkpoint round-trips byte-for-byte.
- REQ-7: `load_state_dict(strict=true)` rejects any key not under
  `model.` or `lm_head.`. `load_state_dict(strict=false)` silently
  drops them.
- REQ-8: Round-trip identity: a fresh model loaded with another
  model's `state_dict()` produces bit-close logits on the same
  input (tolerance `1e-6` on f32).

## Acceptance Criteria

- [x] AC-1: `LlamaForCausalLM::<f32>::new(tiny_cfg)` constructs.
- [x] AC-2: `forward_from_ids(&[1, 5, 7, 9])` returns shape
  `[1, 4, vocab_size]` with all-finite values.
- [x] AC-3: `forward_one_with_cache` over `[1, 5, 7, 9, 11]`
  matches `forward_from_ids` position-by-position to `< 1e-4`
  (f32, tiny model).
- [x] AC-4: `state_dict` → `load_state_dict(strict=true)` round
  trip reproduces logits to `< 1e-6`.
- [x] AC-5: `load_state_dict(strict=true)` rejects an unknown key
  with `InvalidArgument`.
- [x] AC-6: `load_hf_state_dict` with `tie_word_embeddings=true`
  copies `embed_tokens.weight` into `lm_head.weight` when the
  latter is absent.

## Architecture

`pub struct LlamaModel<T: Float>` in `model.rs` owns the decoder
stack. `Module::forward` clones the input then folds each
`LlamaDecoderLayer::forward` over the accumulator and applies the
final `RMSNorm`. The structural shape contract is `[1, seq_len,
hidden_size]` throughout.

`pub struct LlamaForCausalLM<T: Float>` in `model.rs` carries the
model, the `lm_head` linear, and a frozen `LlamaConfig`. `Module::
forward` chains `model.forward(input)?` into `lm_head.forward(h)?`
and returns logits `[1, S, vocab]`.

`pub fn forward_from_ids` in `model.rs` is the canonical entry
point for full-prefix inference. It validates `ids` is non-empty,
casts each `u32` token id to `T` via
`ferrotorch_core::numeric_cast::cast::<u32, T>` (which surfaces
non-representable ids — vocab_size > 65504 with `T = bf16` — as
`InvalidArgument`), runs the embedding layer over the `[S]` index
tensor, promotes to `[1, S, hidden]`, then chains
`model.forward(&hidden_3d)?` and `lm_head.forward(&h)?`.

`pub fn forward_one_with_cache` in `model.rs` is the per-token
incremental path (issue #1129). It checks the cache's layer count
matches the model, embeds the single new token, then iterates
`LlamaDecoderLayer::forward_with_cache` over each layer
threading the per-layer `LayerKvCache`. After the final norm and
lm_head, it asserts the logits tensor is `[1, 1, V]` and returns
the per-vocab `Vec<f64>` slice plus the grown `LlamaKvCache`.

`pub fn load_hf_state_dict` in `model.rs` is the
HuggingFace-aware loader. The HF key layout maps byte-for-byte
onto ours (`model.embed_tokens.weight`, `model.layers.{i}.{...}`,
`model.norm.weight`, `lm_head.weight`), so the wrapper exists
primarily for the tied-embeddings remap (when
`tie_word_embeddings` is true and the HF safetensors omits
`lm_head.weight`, copy it from `model.embed_tokens.weight`). The
remapped state dict is then passed to `load_state_dict(strict)`.

The `Module<T>::load_state_dict` path splits state by `model.` /
`lm_head.` prefix, recursing into the corresponding sub-module's
`load_state_dict`. Strict mode rejects keys outside this prefix
set before any partial loads happen so the model is never left in
a half-loaded state.

### Non-test production consumers

- `pub use model::{LlamaForCausalLM, LlamaModel}` at
  `ferrotorch-llama/src/lib.rs:174` exposes both types as
  crate-root re-exports.
- `LlamaForCausalLM::<bf16>::new(cfg)` is constructed at
  `ferrotorch-llama/examples/llama3_8b.rs:112`. It calls
  `model.forward_from_ids(&tokens)` at
  `ferrotorch-llama/examples/llama3_8b.rs:134`.
- `LlamaForCausalLM::<f32>::new(cfg)` is constructed at
  `ferrotorch-llama/examples/llm_inference_dump.rs:215`; the
  same example calls `model.forward_from_ids(ids)` at
  `ferrotorch-llama/examples/llm_inference_dump.rs:220`.
- `forward_one_with_cache` is invoked from
  `generation.rs` (the `beam_search` and KV-cache-enabled
  `generate` paths) and from `spec_decode.rs`'s `LlamaHandle`
  implementation — both module-level production callers reachable
  via the `ferrotorch_llama` crate root.

## Parity contract

`parity_ops = []`. `model.rs` composes the per-layer ops
(`RMSNorm`, `Linear`, `Embedding`, attention, MLP) whose parity
is owned by `ferrotorch-nn` and the GPU kernel suite. Numerical
edge cases preserved at this layer:

- **Empty ids**: `InvalidArgument`. HF Python raises
  `ValueError("You must specify exactly one of input_ids or
  inputs_embeds")` at `modeling_llama.py:365`. Rust surfaces the
  equivalent via `FerrotorchError::InvalidArgument`.
- **Tied embeddings**: `cfg.tie_word_embeddings == true` plus
  missing `lm_head.weight` copies from `model.embed_tokens.weight`.
  Matches HF's `_tied_weights_keys = ["lm_head.weight"]` at
  `modeling_llama.py:414`.
- **lm_head bias**: every Llama variant ships with `bias=False`
  (HF `modeling_llama.py:422`: `nn.Linear(config.hidden_size,
  config.vocab_size, bias=False)`). The Rust constructor hard-codes
  this.
- **Final RMSNorm before lm_head**: HF
  `modeling_llama.py:405` applies `self.norm(hidden_states)` once
  outside the layer loop; ferrotorch mirrors the structure with
  `LlamaModel::norm` applied after the layer fold.

## Verification

Tests in `mod tests in model.rs`:

- `tiny_model_constructs_and_parameter_count_sane`
- `tiny_model_named_parameters_use_hf_layout`
- `tiny_model_forward_from_ids_produces_correct_shape`
- `load_state_dict_round_trip_tiny`
- `load_state_dict_strict_rejects_unknown_key`
- `forward_one_with_cache_matches_full_prefix_forward` (discriminating
  test pinned to #1129)
- `load_hf_state_dict_with_tied_embeddings_copies_lm_head`

Plus integration tests in `ferrotorch-llama/tests/`: `conformance_llama.rs`,
`conformance_pretrained_causal_lm.rs`,
`conformance_pretrained_causal_lm_gpu.rs`.

Smoke command:

```bash
cargo test -p ferrotorch-llama --lib model:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct LlamaModel<T: Float>` + its `Module<T>` impl in `model.rs`; non-test consumer: held as the `model` field of `LlamaForCausalLM` in `model.rs` (consumed via `self.model.forward(input)?` in `LlamaForCausalLM`'s `forward`). |
| REQ-2 | SHIPPED | impl: `pub struct LlamaForCausalLM<T: Float>` + `LlamaForCausalLM::new` in `model.rs`; non-test consumer: `LlamaForCausalLM::<bf16>::new(cfg)` at `ferrotorch-llama/examples/llama3_8b.rs:112`. |
| REQ-3 | SHIPPED | impl: `pub fn forward_from_ids` in `model.rs`; non-test consumer: `model.forward_from_ids(&tokens)` at `ferrotorch-llama/examples/llama3_8b.rs:134` and at `ferrotorch-llama/examples/llm_inference_dump.rs:220`. |
| REQ-4 | SHIPPED | impl: `pub fn forward_one_with_cache` in `model.rs`; non-test consumer: invoked in `beam_search` in `generation.rs` (the seed loop and per-beam expansion call `model.forward_one_with_cache`) and in the `LlamaHandle::forward_ids` impl in `spec_decode.rs`. |
| REQ-5 | SHIPPED | impl: `pub fn load_hf_state_dict` (with the tied-embeddings remap branch) in `model.rs`; non-test consumer: HF-style state dicts loaded by examples via the `ferrotorch-serialize` round trip funnel through `load_hf_state_dict` — pattern documented in the `lib.rs` crate-level doc-comment "Loading real weights" section. |
| REQ-6 | SHIPPED | impl: `Module::named_parameters` for `LlamaForCausalLM` in `model.rs`; non-test consumer: the GPU loader `LlamaGpuInferencer::new` at `gpu.rs` reads tensors by the exact HF-shaped keys (`model.embed_tokens.weight`, `model.layers.{i}.self_attn.q_proj.weight`, ...) that `named_parameters` produces. |
| REQ-7 | SHIPPED | impl: the `strict` branch in `Module::load_state_dict` for `LlamaForCausalLM` in `model.rs`; non-test consumer: same loader path as REQ-5/6 — strict-mode rejection surfaces malformed checkpoints to the example callers. |
| REQ-8 | SHIPPED | impl: the same `Module::load_state_dict` path verifies via the round-trip-tested behavior pinned in the `load_state_dict_round_trip_tiny` test; non-test consumer: production loaders (HF state-dict ingest from safetensors) exercise the same code path the test pins. |
