# ferrotorch-bert — `attention` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - /home/doll/pytorch/torch/ (scaled_dot_product_attention at
    aten/src/ATen/native/transformers/attention.cpp; the
    multi-head wrapper is HF's
    huggingface/transformers/src/transformers/models/bert/modeling_bert.py
    `BertSelfAttention` + `BertSelfOutput`)
-->

## Summary

`ferrotorch-bert/src/attention.rs` carries the BERT self-attention
sub-block: Q / K / V linear projections, multi-head scaled dot-product
attention (non-causal, bidirectional), an output projection, and the
POST-norm residual `LayerNorm(input + dense(ctx))`. The architecture
mirrors HuggingFace's `BertSelfAttention` + `BertSelfOutput` +
`BertAttention` triple.

## Requirements

- REQ-1: `pub struct BertSelfAttention<T: Float>` holds three
  `ferrotorch_nn::Linear<T>` projections (`query`, `key`, `value`),
  all `[hidden -> hidden]` with bias. Caches `num_heads`, `head_dim`,
  `hidden` so the forward path can reshape without re-reading the
  config.
- REQ-2: `BertSelfAttention::forward` projects QKV, reshapes to
  `[H, S, d]`, calls
  `ferrotorch_nn::standard_attention(q, k, v, causal=false)`,
  transposes back to `[S, H*d]`, and reshapes to `[1, S, hidden]`.
  Bidirectional (non-causal) is hard-wired.
- REQ-3: `pub struct BertSelfOutput<T: Float>` holds an output dense
  `Linear<T>` (`[hidden -> hidden]`, bias) and a `LayerNorm<T>`
  (`[hidden]`, eps from config). `forward_residual(attn, input)` is
  the canonical entry point: it computes
  `LayerNorm(input + dense(attn))` — the POST-norm residual that
  distinguishes BERT from the modern pre-norm convention.
- REQ-4: `pub struct BertAttention<T: Float>` composes
  `BertSelfAttention` + `BertSelfOutput` and exposes
  `Module::forward(input) = LayerNorm(input + dense(self_attn(input)))`
  through the wrapped sub-modules.
- REQ-5: HF state-dict key layout: under `BertAttention` the
  self-attention sub-block is named `self.{query,key,value}.{weight,
  bias}` and the output sub-block is named
  `output.{dense.{weight,bias}, LayerNorm.{weight,bias}}`. Loadable
  in strict mode without rewriting keys.
- REQ-6: Input rank/shape validation: forward returns
  `FerrotorchError::ShapeMismatch` when the input is not
  `[1, S, hidden]`.
- REQ-7: `Module::forward` on `BertSelfOutput` performs ONLY the
  dense projection (the residual partner is unavailable). The full
  sub-block uses `forward_residual`; the `Module::forward` fallback
  keeps the trait satisfied for tooling that walks sub-module
  forwards individually.

## Acceptance Criteria

- [x] AC-1: `BertSelfAttention::<f32>::new(&tiny_cfg)` constructs.
- [x] AC-2: `BertSelfAttention::forward([1, 4, 8])` returns
  `[1, 4, 8]`.
- [x] AC-3: `BertAttention::<f32>::new(&tiny_cfg)` constructs.
- [x] AC-4: `BertAttention::forward([1, 3, 8])` returns `[1, 3, 8]`.
- [x] AC-5: `BertAttention::named_parameters()` includes the
  HF-layout keys (`self.query.weight`, `self.query.bias`,
  `self.key.weight`, `self.value.weight`, `output.dense.weight`,
  `output.dense.bias`, `output.LayerNorm.weight`,
  `output.LayerNorm.bias`).

## Architecture

`pub struct BertSelfAttention<T: Float>` in `attention.rs` holds
three `Linear<T>` plus cached dim fields. The forward path uses
`ferrotorch_nn::{reshape_to_heads, standard_attention,
transpose_heads_to_2d}` so the multi-head plumbing matches the
shared shape-utility surface that `ferrotorch-llama` /
`ferrotorch-whisper` also consume — the only BERT-specific bit is
the `causal=false` argument plus the post-norm residual.

`pub struct BertSelfOutput<T: Float>` in `attention.rs` exposes
both `Module::forward` (dense-only) and `forward_residual` (dense +
residual add + LayerNorm). The split exists because `Module::forward`
has the signature `(&self, input: &Tensor<T>) -> Tensor<T>` and
cannot receive the second `attn` tensor.

`pub struct BertAttention<T: Float>` in `attention.rs` is the public
composition that `BertLayer::new` instantiates. Its `Module::forward`
calls `self.self_attn.forward(input)` then
`self.output.forward_residual(&ctx, input)` so the residual partner
is the un-attended input.

Private helpers `reshape_2d` / `reshape_3d` at the bottom of
`attention.rs` own the data (no view tricks); they are needed because
`reshape_to_heads` consumes a 2-D tensor while the BertAttention
boundary is 3-D `[1, S, hidden]`.

### Non-test production consumers

- `pub use attention::{BertAttention, BertSelfAttention,
  BertSelfOutput}` at `ferrotorch-bert/src/lib.rs:86`.
- `pub attention: BertAttention<T>` field of `pub struct BertLayer`
  at `BertLayer in ferrotorch-bert/src/layer.rs`; `BertLayer::new` at
  `ferrotorch-bert/src/layer.rs:259` constructs it; `BertLayer`'s
  `Module::forward` at `ferrotorch-bert/src/layer.rs:269` invokes
  `self.attention.forward(input)`.

## Parity contract

`parity_ops = []`. The attention sub-block composes
`scaled_dot_product_attention` (covered by
`ferrotorch-nn`'s `standard_attention` parity) and `linear` /
`layer_norm` (covered by `ferrotorch-nn` parity).

Numerical / structural edge cases preserved:

- **Non-causal** (`standard_attention(..., causal=false)`). BERT is
  an encoder; full bidirectional attention is the contract.
- **Q/K/V all have bias.** Upstream HF `BertSelfAttention` constructs
  the three `nn.Linear` with `bias=True`. (Contrast Whisper's
  `k_proj` which has `bias=False`.)
- **POST-norm residual.** `LayerNorm(input + dense(ctx))` matches
  HF BERT; this is the structural divergence from
  `ferrotorch-nn/src/transformer.rs` (pre-norm) and from
  ferrotorch-whisper (also pre-norm).
- **LayerNorm `eps = cfg.layer_norm_eps`** (default `1e-12` for HF
  BERT, vs the PyTorch nn-default `1e-5`).

## Verification

Tests in `mod tests in attention.rs`:

- `self_attention_shape`
- `full_attention_block_shape`
- `named_parameters_match_hf_layout`

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-bert --lib attention:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct BertSelfAttention<T: Float>` in `attention.rs`; non-test consumer: field `self_attn` of `pub struct BertAttention` in `attention.rs` (the post-norm wrapper), grandfathered re-export at `attention in ferrotorch-bert/src/lib.rs`. |
| REQ-2 | SHIPPED | impl: `Module::forward` for `BertSelfAttention` in `attention.rs`; non-test consumer: `BertAttention`'s `Module::forward` in `attention.rs` calls `self.self_attn.forward(input)`. |
| REQ-3 | SHIPPED | impl: `BertSelfOutput::forward_residual` in `attention.rs`; non-test consumer: `BertAttention`'s `Module::forward` in `attention.rs` calls `self.output.forward_residual(&ctx, input)`. |
| REQ-4 | SHIPPED | impl: `pub struct BertAttention<T: Float>` + its `Module<T>` impl in `attention.rs`; non-test consumer: field `attention` of `pub struct BertLayer` at `BertLayer in ferrotorch-bert/src/layer.rs`; `BertLayer::Module::forward` at `forward in ferrotorch-bert/src/layer.rs` invokes `self.attention.forward(input)`. |
| REQ-5 | SHIPPED | impl: `named_parameters` / `load_state_dict` for `BertSelfAttention` / `BertSelfOutput` / `BertAttention` in `attention.rs`; non-test consumer: `BertLayer::load_state_dict` at `ferrotorch-bert/src/layer.rs:344` recurses through `attention.{self,output}.*`. |
| REQ-6 | SHIPPED | impl: rank/shape checks at the top of `BertSelfAttention::Module::forward` in `attention.rs`; non-test consumer: propagated up through `BertAttention::Module::forward` then `BertLayer::Module::forward` at `ferrotorch-bert/src/layer.rs:269`. |
| REQ-7 | SHIPPED | impl: `BertSelfOutput::Module::forward` (dense-only) in `attention.rs`; non-test consumer: kept reachable through the `pub use` at `attention in ferrotorch-bert/src/lib.rs` (the trait surface is required for the `Module` blanket). Real-application path goes through `forward_residual`; this `Module::forward` shim keeps `Module` satisfied. |
