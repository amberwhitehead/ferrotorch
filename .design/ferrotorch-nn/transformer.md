# ferrotorch-nn — `transformer` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/modules/transformer.py
  - aten/src/ATen/native/transformers/
-->

## Summary

`ferrotorch-nn/src/transformer.rs` provides the LLM-critical
transformer building blocks: RoPE (Rotary Position Embedding),
SwiGLU, KVCache, and the canonical encoder/decoder/full
`Transformer` stack. Mirrors `torch.nn.{Transformer,
TransformerEncoder, TransformerDecoder, TransformerEncoderLayer,
TransformerDecoderLayer}` at
`torch/nn/modules/transformer.py:58-1240` for the API surface plus
the modern LLM extensions (RoPE / SwiGLU / KVCache) that mainline
PyTorch does not ship as first-class nn-modules — they are
ferrotorch additions that mirror HuggingFace's `transformers/models/
llama` conventions and the RoFormer paper (Su et al., 2021).

## Requirements

- REQ-1: `pub enum RoPEConvention { Interleaved, HalfRotation }` —
  selects RoPE element-pairing. `Interleaved` (default) pairs
  `(x[2i], x[2i+1])` (original RoFormer); `HalfRotation` pairs
  `(x[i], x[i+d/2])` (Llama / GPT-NeoX / Pythia).

- REQ-2: `pub struct RotaryPositionEmbedding<T: Float>` — RoPE
  applied to queries and keys with precomputed sin/cos tables.
  `apply_rope` rotates a `[batch_dims, seq_len, dim]` tensor at a
  given `seq_offset` (for KV-cache concatenation). Mirrors Su et
  al. (2021) and Llama's implementation.

- REQ-3: `pub enum RoPEScaling` — `None`, `Linear { factor }`,
  `Dynamic { factor }`, `Llama3 { factor, low_freq_factor,
  high_freq_factor, original_max_position_embeddings }` to support
  the Llama-3 long-context extrapolation. Each variant produces a
  different theta-frequency schedule.

- REQ-4: `pub struct SwiGLU<T: Float>` — the gated linear unit used
  in Llama / Mistral / Mixtral: `w3(silu(w1(x)) * w2(x))`. Three
  `Linear<T>` projections; uses `silu` and `mul` from
  `ferrotorch_core::grad_fns` so autograd traces through.
  `impl Module<T> for SwiGLU<T>` exposes the standard surface.

- REQ-5: `pub struct KVCache<T: Float>` — concat-append KV cache for
  decoder inference. `append_kv(key, value)` concatenates new
  `[batch, n_heads, new_seq, head_dim]` along the seq axis;
  `clear()` drops the cache; accessors return the current
  `[batch, n_heads, total_seq, head_dim]` slabs.

- REQ-6: `pub struct TransformerEncoderLayer<T: Float>` — pre-norm
  encoder block: `norm -> self_attn -> residual -> norm -> ffn ->
  residual`. Holds a `MultiheadAttention<T>` plus two `Linear<T>`
  layers for the FFN and two `LayerNorm<T>` layers. Mirrors
  upstream's `TransformerEncoderLayer` at
  `torch/nn/modules/transformer.py:659-980` (with `norm_first=True`
  semantics).

- REQ-7: `pub struct TransformerDecoderLayer<T: Float>` — pre-norm
  decoder block: `norm -> self_attn (causal) -> residual -> norm ->
  cross_attn -> residual -> norm -> ffn -> residual`. Three norms,
  one self-attn, one cross-attn, one FFN. Mirrors upstream
  `TransformerDecoderLayer` at
  `torch/nn/modules/transformer.py:981-1100`.

- REQ-8: `pub struct TransformerEncoder<T: Float>` — stack of N
  `TransformerEncoderLayer<T>` plus a terminal LayerNorm. Mirrors
  `torch/nn/modules/transformer.py:318-553`.

- REQ-9: `pub struct TransformerDecoder<T: Float>` — stack of N
  `TransformerDecoderLayer<T>` plus a terminal LayerNorm. Mirrors
  `torch/nn/modules/transformer.py:554-658`.

- REQ-10: `pub struct Transformer<T: Float>` — full encoder-decoder
  composition. Holds a `TransformerEncoder<T>` and a
  `TransformerDecoder<T>`. Mirrors
  `torch/nn/modules/transformer.py:58-317`.

- REQ-11: `Module<T>` impl for every public stack — `forward`,
  `parameters`, `parameters_mut`, `named_parameters`, `train`,
  `eval`, `is_training`. State-dict keys follow upstream's
  `layers.<i>.self_attn.{q_proj.weight, ...}` /
  `layers.<i>.linear1.weight` / `layers.<i>.norm1.weight` naming.

- REQ-12: RoPE autograd via `RoPEBackward<T>` — applies the inverse
  rotation matrix (swap of cos / -sin) on the saved sin/cos tables.
  Attached when the input requires grad and grad is globally
  enabled.

## Acceptance Criteria

- [x] AC-1: `RotaryPositionEmbedding::new(dim=64, max_seq_len=2048,
  theta=10000.0)` constructs.
- [x] AC-2: `apply_rope(input, seq_offset=0)` for a
  `[1, 32, 64]` input returns `[1, 32, 64]` (shape preserved).
- [x] AC-3: `SwiGLU::new(in_features=512,
  hidden_features=1376)` constructs three Linear layers.
- [x] AC-4: `KVCache::new(...)` then two `append_kv` calls grow the
  cache to the expected `[B, H, total_seq, D]` shape.
- [x] AC-5: `TransformerEncoderLayer::new(d_model=64, nhead=8,
  dim_feedforward=256)` constructs.
- [x] AC-6: `TransformerEncoder::new(layer, num_layers=4)`
  constructs a 4-layer stack.
- [x] AC-7: `forward(src)` for a `[B, S, d_model]` input returns
  `[B, S, d_model]`.
- [x] AC-8: `Transformer::new(d_model, nhead, num_encoder_layers,
  num_decoder_layers, dim_feedforward)` constructs an
  encoder/decoder pair.
- [x] AC-9: `RotaryPositionEmbedding` backward attaches when input
  requires grad.

## Architecture

### RoPE (REQ-1, REQ-2, REQ-3, REQ-12)

`pub enum RoPEConvention` at
`pub enum RoPEConvention in transformer.rs` carries the two
element-pairing modes. `pub struct RotaryPositionEmbedding<T>` at
`pub struct RotaryPositionEmbedding in transformer.rs` precomputes
sin/cos tables sized `[max_seq_len, dim/2]` and applies the
rotation per (batch, seq, head_dim) element in the forward path.
The two conventions diverge in the element-pair index pattern;
otherwise the rotation math is identical.

`pub enum RoPEScaling` at
`pub enum RoPEScaling in transformer.rs` extends the standard
theta schedule with the three known scaling families (linear,
dynamic, Llama-3); each variant computes a different
`per_freq_factor` applied to the base theta sequence.

`RoPEBackward<T>` at
`struct RoPEBackward in transformer.rs` is the autograd node. On
`backward(grad_output)`, it applies the inverse rotation (cos /
-sin swap) to derive `grad_input`. Attached when the input requires
grad.

### SwiGLU (REQ-4)

`pub struct SwiGLU<T: Float>` at
`pub struct SwiGLU in transformer.rs` carries three `Linear<T>`
projections: `w1` (gate), `w2` (up), `w3` (down). The forward
computes `w3(silu(w1(x)) * w2(x))` via
`ferrotorch_core::grad_fns::activation::silu` plus
`grad_fns::arithmetic::mul`. `impl Module<T> for SwiGLU<T>`
exposes the trait surface.

### KVCache (REQ-5)

`pub struct KVCache<T: Float>` at
`pub struct KVCache in transformer.rs` holds two `Option<Tensor<T>>`
slots for cached keys and values. `append_kv` concatenates new
`[batch, n_heads, new_seq, head_dim]` along the seq axis using the
`cat` op from `ferrotorch_core::grad_fns::shape`. `clear()`
resets both slots to `None`.

### TransformerEncoderLayer (REQ-6)

`pub struct TransformerEncoderLayer<T: Float>` at
`pub struct TransformerEncoderLayer in transformer.rs` holds:

- `self_attn: MultiheadAttention<T>` — multi-head self-attention
  from `crate::attention`.
- `linear1: Linear<T>`, `linear2: Linear<T>` — the FFN.
- `norm1: LayerNorm<T>`, `norm2: LayerNorm<T>` — pre-norm.
- `dropout1`, `dropout2`, `dropout: Dropout` — for the residual
  connections and the FFN intermediate.

Forward: `x + dropout1(self_attn(norm1(x)))`, then `x +
dropout2(linear2(dropout(activation(linear1(norm2(x))))))`.

### TransformerDecoderLayer (REQ-7)

`pub struct TransformerDecoderLayer<T: Float>` at
`pub struct TransformerDecoderLayer in transformer.rs` mirrors the
encoder block with an additional cross-attention sub-layer and a
third norm. The forward path applies self-attn (causal) → cross-attn
(against the encoder memory) → FFN, each with pre-norm and a
residual.

### Stacks and full Transformer (REQ-8, REQ-9, REQ-10)

`pub struct TransformerEncoder<T>` /
`pub struct TransformerDecoder<T>` /
`pub struct Transformer<T>` at the corresponding
`pub struct ... in transformer.rs` items hold a `Vec<Layer<T>>`
plus a terminal LayerNorm. Their forwards just iterate the layers
and apply the terminal norm.

### Module trait surface (REQ-11)

Every public struct in this file has an `impl<T: Float> Module<T>
for <Type><T>` block; collectively at the `impl Module<T>`
sites in `transformer.rs`. Each impl traverses sub-modules to
gather parameters (with prefixed keys for `named_parameters`),
forwards `train`/`eval` to all children, and dispatches `forward`
to the layer-specific entry point.

### Non-test production consumers

- `pub use transformer::{KVCache, RoPEConvention, RoPEScaling,
  RotaryPositionEmbedding, SwiGLU, Transformer,
  TransformerDecoder, TransformerDecoderLayer,
  TransformerEncoder, TransformerEncoderLayer}` at
  `ferrotorch-nn/src/lib.rs:248-251` — grandfathered public API
  surface.
- `ferrotorch-llama/src/attention.rs:23` consumes
  `RoPEConvention`, `RoPEScaling`, `RotaryPositionEmbedding` (the
  Llama-3 GQA attention layer wraps RoPE around the projected
  queries/keys).

## Parity contract

`parity_ops = []`. The transformer stack composes
`scaled_dot_product_attention` (covered by `attention.md` /
blocker #1455), `linear` / `layer_norm` / `silu` / `softmax`
(covered by `linear.md` / `norm.md` / activation-side parity).

Numerical edge cases preserved:

- **Pre-norm vs post-norm** — ferrotorch ships pre-norm only
  (`norm_first=True` equivalent). Upstream defaults to
  `norm_first=False`; ferrotorch's default is the modern LLM
  convention. Callers requiring post-norm must wire it manually.
- **RoPE seq_offset** — `apply_rope(x, seq_offset)` shifts into
  the precomputed sin/cos tables, enabling KV-cache-friendly
  incremental rotation without recomputing position embeddings
  per step.
- **SwiGLU vs GLU vs GeGLU** — ferrotorch ships SwiGLU only (SiLU
  gate). Other variants are not exposed; downstream code uses
  `Linear + activation + Linear` for those.
- **KV-cache datatype** — cache stores tensors in the same dtype
  as the input. No quantisation is applied; PagedAttention's
  page-pool is the canonical path for low-precision serving
  workloads.

## Verification

Tests in `mod tests in transformer.rs` (RoPE construction, SwiGLU
forward shape, KVCache append round-trip, encoder/decoder layer
forward shape, full Transformer forward shape, named_parameters
keys).

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-nn --lib transformer:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum RoPEConvention` in `transformer.rs`; non-test consumer: re-export at `ferrotorch-nn/src/lib.rs:248` + `ferrotorch-llama/src/attention.rs:23`. |
| REQ-2 | SHIPPED | impl: `pub struct RotaryPositionEmbedding<T: Float>` with `apply_rope` in `transformer.rs`; non-test consumer: re-export at `lib.rs` + `ferrotorch-llama/src/attention.rs`. |
| REQ-3 | SHIPPED | impl: `pub enum RoPEScaling` in `transformer.rs`; non-test consumer: re-export at `lib.rs` + `ferrotorch-llama/src/attention.rs` (Llama-3 long-context scaling). |
| REQ-4 | SHIPPED | impl: `pub struct SwiGLU<T: Float>` plus `impl Module<T> for SwiGLU<T>` in `transformer.rs`; non-test consumer: re-export at `lib.rs`. (Llama's MLP composes the three projections directly per `ferrotorch-llama/src/mlp.rs` doc-comment.) |
| REQ-5 | SHIPPED | impl: `pub struct KVCache<T: Float>` with `append_kv` / `clear` in `transformer.rs`; non-test consumer: re-export at `lib.rs`. |
| REQ-6 | SHIPPED | impl: `pub struct TransformerEncoderLayer<T: Float>` mirroring upstream `transformer.py:659-980` in `transformer.rs`; non-test consumer: re-export at `lib.rs`. |
| REQ-7 | SHIPPED | impl: `pub struct TransformerDecoderLayer<T: Float>` mirroring upstream `transformer.py:981-1100` in `transformer.rs`; non-test consumer: re-export at `lib.rs`. |
| REQ-8 | SHIPPED | impl: `pub struct TransformerEncoder<T: Float>` mirroring upstream `transformer.py:318-553` in `transformer.rs`; non-test consumer: re-export at `lib.rs`. |
| REQ-9 | SHIPPED | impl: `pub struct TransformerDecoder<T: Float>` mirroring upstream `transformer.py:554-658` in `transformer.rs`; non-test consumer: re-export at `lib.rs`. |
| REQ-10 | SHIPPED | impl: `pub struct Transformer<T: Float>` mirroring upstream `transformer.py:58-317` in `transformer.rs`; non-test consumer: re-export at `lib.rs`. |
| REQ-11 | SHIPPED | impl: `impl<T: Float> Module<T> for ...` blocks for every transformer struct in `transformer.rs`; non-test consumer: re-export at `lib.rs`. |
| REQ-12 | SHIPPED | impl: `struct RoPEBackward<T>` plus `impl GradFn<T>` in `transformer.rs`; non-test consumer: re-export at `lib.rs` — the autograd engine traverses this GradFn on `backward()` of any RoPE-wrapped tensor. |
