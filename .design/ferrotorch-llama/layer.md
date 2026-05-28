# ferrotorch-llama — `layer` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - HuggingFace transformers/models/llama/modeling_llama.py
    (LlamaDecoderLayer:268-311)
-->

## Summary

`ferrotorch-llama/src/layer.rs` ships `LlamaDecoderLayer`, the
single decoder block of the stack. It composes
`input_layernorm` (RMSNorm) → `self_attn` (LlamaAttention) →
residual → `post_attention_layernorm` (RMSNorm) → `mlp` (LlamaMLP)
→ residual, matching the HuggingFace pre-norm topology. Both the
full-prefix `Module::forward` and the per-token
`forward_with_cache` (KV-cache) paths are implemented.

## Requirements

- REQ-1: `pub struct LlamaDecoderLayer<T: Float>` carries four
  named sub-modules — `input_layernorm: RMSNorm<T>`, `self_attn:
  LlamaAttention<T>`, `post_attention_layernorm: RMSNorm<T>`,
  `mlp: LlamaMLP<T>` — matching the HF naming so HF state dicts
  load directly.
- REQ-2: `Module::forward` implements the pre-norm residual block:
  `x = x + self_attn(input_layernorm(x)); x = x + mlp(post_attention_layernorm(x))`.
- REQ-3: `forward_with_cache(input, cache, seq_offset)` is the
  incremental sibling: applies the same residual pattern to a
  `[1, 1, hidden]` single-token input, threads `cache` through
  `self_attn.forward_with_cache`, and returns the updated hidden
  state plus the new per-layer K/V cache.
- REQ-4: Strict input validation in `forward_with_cache`: the input
  shape must be exactly `[1, 1, hidden]`. Mismatched shape returns
  `InvalidArgument`.
- REQ-5: `named_parameters` exposes the four sub-modules under their
  HF names (`input_layernorm`, `self_attn`, `post_attention_layernorm`,
  `mlp`) so the model-level state dict layout is HF-compatible
  byte-for-byte.
- REQ-6: `load_state_dict(strict=true)` rejects keys outside the
  four known prefixes (`input_layernorm.`, `self_attn.`,
  `post_attention_layernorm.`, `mlp.`).

## Acceptance Criteria

- [x] AC-1: `LlamaDecoderLayer::<f32>::new(&cfg)` constructs for
  the tiny config used by the `model.rs` round-trip tests.
- [x] AC-2: The full-prefix `Module::forward` returns shape
  `[1, seq, hidden]` for input `[1, seq, hidden]` (exercised
  transitively via `LlamaModel::forward`).
- [x] AC-3: `forward_with_cache` returns a `(Tensor[1,1,hidden],
  LayerKvCache)` pair whose downstream composition matches the
  full-prefix forward to `< 1e-4` (exercised via the model-level
  `forward_one_with_cache_matches_full_prefix_forward` test).
- [x] AC-4: HF state-dict round-trip: a `LlamaDecoderLayer` whose
  `named_parameters` produces `input_layernorm.weight`,
  `self_attn.{q,k,v,o}_proj.weight`, `post_attention_layernorm.weight`,
  `mlp.{gate,up,down}_proj.weight` keys can `load_state_dict` a
  matching HF dict in strict mode.

## Architecture

`pub struct LlamaDecoderLayer<T: Float>` in `layer.rs` carries the
four named sub-modules. `LlamaDecoderLayer::new(&cfg)` validates
the config, constructs each sub-module with the matching slice of
the config, and threads a single `training: bool` flag for
mode propagation.

`Module::forward` in `layer.rs` is the pre-norm pattern:

1. `h = self.input_layernorm.forward(input)?`
2. `attn_out = self.self_attn.forward(&h)?`
3. `x = add(input, &attn_out)?` (the residual via
   `ferrotorch_core::grad_fns::arithmetic::add`)
4. `h2 = self.post_attention_layernorm.forward(&x)?`
5. `mlp_out = self.mlp.forward(&h2)?`
6. `add(&x, &mlp_out)` (final residual)

This matches HF `modeling_llama.py:291-310` line-by-line: residual
captured, layernorm applied, sub-block invoked, residual added —
twice (attention block + MLP block).

`pub fn forward_with_cache` in `layer.rs` is the incremental path
(#1129). It rejects any input shape other than `[1, 1, hidden]`
with `InvalidArgument`, calls `self_attn.forward_with_cache(&h,
cache, seq_offset)` to advance the cache, and otherwise mirrors
the full-prefix structure (input_layernorm → attn → residual →
post_attention_layernorm → mlp → residual). The returned
`LayerKvCache` is the K/V slab grown by one position.

The `Module` impl delegates `parameters`, `parameters_mut`,
`train`, `eval`, `state_dict`, `load_state_dict` to the four
sub-modules. The strict-mode load_state_dict path enumerates the
known prefixes (`input_layernorm`, `self_attn`,
`post_attention_layernorm`, `mlp`) and rejects anything else
before any per-sub-module load runs.

### Non-test production consumers

- `pub use layer::LlamaDecoderLayer` at
  `ferrotorch-llama/src/lib.rs` exposes the type.
- `pub layers: Vec<LlamaDecoderLayer<T>>` field of `LlamaModel`
  in `model.rs` is the canonical consumer — every layer in the
  stack is a `LlamaDecoderLayer`.
- `LlamaModel::new` in `model.rs` calls
  `LlamaDecoderLayer::new(cfg)` once per `cfg.num_hidden_layers`
  to populate the `layers` vec.
- `LlamaModel::forward` calls `layer.forward(&h)` (the
  `Module::forward` path) on each layer in sequence.
- `LlamaForCausalLM::forward_one_with_cache` in `model.rs` calls
  `layer.forward_with_cache(&h, prev_layer_cache, seq_offset)` on
  each layer during incremental decoding.

## Parity contract

`parity_ops = []`. The decoder layer composes ops whose parity is
owned by `ferrotorch-nn` (`RMSNorm`, `Linear`) and by the
attention / MLP sub-modules of this crate. Numerical behavior
mirrored exactly from HF:

- **Pre-norm structure**: HF's `modeling_llama.py:291-310` is
  pre-norm (norm applied before attn / mlp, residual added after).
  Llama does NOT use post-norm; ferrotorch matches.
- **Residual via `add`**: the residual additions use
  `ferrotorch_core::grad_fns::arithmetic::add` (the autograd-aware
  binary add), which preserves the gradient graph for training.
- **Two RMSNorm instances per layer**: separately parameterized
  (`input_layernorm.weight` and
  `post_attention_layernorm.weight`), as HF does.
- **`forward_with_cache` shape requirement**: `[1, 1, hidden]` only
  — incremental decoding is one-token-at-a-time. The full-prefix
  `Module::forward` accepts `[1, S, hidden]` for arbitrary `S`.

## Verification

`layer.rs` has no in-file `#[cfg(test)] mod tests`. Its behavior is
exercised transitively via the model-level tests in
`mod tests in model.rs`:

- `tiny_model_forward_from_ids_produces_correct_shape` — drives
  `Module::forward` on a tiny 2-layer stack.
- `forward_one_with_cache_matches_full_prefix_forward` — drives
  `forward_with_cache` on the same 2-layer stack and pins
  numerical equivalence to the full-prefix path.
- `load_state_dict_round_trip_tiny` — exercises the per-layer
  state-dict round trip with the HF-shaped keys.

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-llama --lib model:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct LlamaDecoderLayer<T: Float>` + `LlamaDecoderLayer::new` in `layer.rs`; non-test consumer: `pub layers: Vec<LlamaDecoderLayer<T>>` field of `LlamaModel` in `model.rs`, populated by `LlamaModel::new` calling `LlamaDecoderLayer::new(cfg)` in a loop. |
| REQ-2 | SHIPPED | impl: `Module::forward` for `LlamaDecoderLayer` in `layer.rs`; non-test consumer: `LlamaModel::forward` in `model.rs` iterates `for layer in &self.layers { h = layer.forward(&h)?; }`. |
| REQ-3 | SHIPPED | impl: `pub fn forward_with_cache` in `layer.rs`; non-test consumer: `LlamaForCausalLM::forward_one_with_cache` in `model.rs` calls `layer.forward_with_cache(&h, prev_layer_cache, seq_offset)?` on each layer. |
| REQ-4 | SHIPPED | impl: shape-validation branch at the top of `forward_with_cache` in `layer.rs`; non-test consumer: same call path as REQ-3 — the example driver paths that feed the model also propagate the validation error. |
| REQ-5 | SHIPPED | impl: `Module::named_parameters` for `LlamaDecoderLayer` in `layer.rs`; non-test consumer: `LlamaModel::named_parameters` in `model.rs` walks `for (n, p) in l.named_parameters()` and prefixes with `layers.{i}.`, producing the canonical HF layout consumed by the loader. |
| REQ-6 | SHIPPED | impl: strict-prefix loop in `Module::load_state_dict` for `LlamaDecoderLayer` in `layer.rs`; non-test consumer: `LlamaModel::load_state_dict` in `model.rs` recurses into `layer.load_state_dict(&extract(&format!("layers.{i}")), strict)?` during HF state-dict ingest. |
