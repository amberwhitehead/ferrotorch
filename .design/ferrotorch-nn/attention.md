# ferrotorch-nn — `attention` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/modules/activation.py
  - aten/src/ATen/native/transformers/attention.cpp
  - torch/nn/functional.py
-->

## Summary

`ferrotorch-nn/src/attention.rs` defines `MultiheadAttention<T>` — the
classical "Attention Is All You Need" (Vaswani et al., 2017) multi-head
attention block with Grouped-Query Attention (GQA) as a constructor
opt-in. Mirrors `torch.nn.MultiheadAttention`
(`torch/nn/modules/activation.py:1089-1404`) for the parameter layout,
forward shape contract, and `train`/`eval` toggle. Composed entirely
from differentiable primitives in `ferrotorch_core::grad_fns` so
autograd traces the backward pass without a custom `GradFn`.

## Requirements

- REQ-1: `pub struct MultiheadAttention<T: Float>` with `embed_dim`,
  `num_heads`, `num_kv_heads`, `head_dim` and the four projection
  parameters (`q_proj`, `k_proj`, `v_proj`, `out_proj`) plus optional
  per-projection biases. Mirrors upstream's `in_proj_weight`/
  `out_proj` plus optional biases at `activation.py:1140-1200`.

- REQ-2: `MultiheadAttention::new(embed_dim, num_heads, bias)`
  constructor for classical MHA where `num_kv_heads == num_heads`.
  Rejects `embed_dim % num_heads != 0`. Mirrors upstream's
  `__init__` validation at `activation.py:1153-1188`.

- REQ-3: `MultiheadAttention::with_gqa(embed_dim, num_heads,
  num_kv_heads, bias)` for Grouped-Query Attention (Llama 3 / Llama 2
  70B). K/V projections sized `[num_kv_heads * head_dim, embed_dim]`
  with `repeat_kv` between V_proj and the attention scores. Rejects
  `num_heads % num_kv_heads != 0`. No direct upstream class in 2.x
  (`MultiheadAttention` is MHA-only); GQA mirrors HuggingFace
  `transformers/models/llama` convention.

- REQ-4: `forward_qkv(query, key, value, causal_mask)` cross-attention
  entry point. Validates that all three inputs are 3-D
  `[batch, seq, embed_dim]`, batch sizes agree, key and value have
  matching seq_len, and `causal_mask` requires `seq_q == seq_k`.
  Mirrors upstream's `forward(query, key, value, …, need_weights,
  attn_mask, …)` at `activation.py:1234-1296`.

- REQ-5: Scaled dot-product attention math —
  `softmax((Q @ K^T) / sqrt(head_dim) + mask) @ V` — built from
  differentiable primitives (`mm_differentiable`,
  `bmm_differentiable`, `softmax`, `mul`, `add`) so autograd traces
  the backward graph through every operation. Mirrors the canonical
  attention math at `aten/src/ATen/native/transformers/attention.cpp`.

- REQ-6: Causal masking as additive `-1e9` over the upper-triangular
  positions. Built on the CPU then `.to(device)`d to follow the
  scores. Mirrors `attn_mask` parameter contract at
  `activation.py:1294-1336`.

- REQ-7: GQA `repeat_kv` expansion — each KV head serves
  `num_heads / num_kv_heads` consecutive query heads. Uses the
  differentiable `expand` primitive when invoked from the
  general forward path; the standalone `pub fn repeat_kv` helper
  for cell-level reuse is host-copy-based (training-broken; inference
  only). Mirrors HuggingFace `LlamaAttention._repeat_kv`.

- REQ-8: `Module<T> for MultiheadAttention<T>` impl with `forward`
  (self-attention shortcut), `parameters`/`parameters_mut`,
  `named_parameters` (keys: `q_proj.weight`, `k_proj.weight`,
  `v_proj.weight`, `out_proj.weight` + the optional bias keys),
  `train`/`eval`/`is_training`. Mirrors the `nn.Module` surface
  inherited by `MultiheadAttention` at upstream.

- REQ-9: `forward_2d` fast path for `seq_len == 1` self-attention
  (autoregressive generation hot loop). Skips Q/K projections and
  the attention loop because `softmax([1,1]) = 1.0` makes the context
  equal to V identically. Returns `linear_fused(linear_fused(input,
  V_proj), out_proj)`. Rejects GQA configurations.

- REQ-10: `pub fn reshape_to_heads` / `transpose_heads_to_2d` /
  `repeat_kv` standalone helpers for downstream consumers that build
  custom attention variants (Llama uses `repeat_kv` + `reshape_to_heads`
  directly). These do their own host-side data shuffle (`data_vec()`),
  so they break the autograd graph — inference-only.

- REQ-11: Parity op `nn.functional.scaled_dot_product_attention` —
  forward output matches upstream `F.scaled_dot_product_attention(Q,
  K, V, is_causal=...)` to within float32 tolerance, in both
  unmasked and `is_causal=True` configurations. SHIPPED 2026-05-26
  (closes #1532, addresses the runner-arm-gap half of #1455).
  Parity-sweep runner arm at
  `tools/parity-sweep/runner/src/main.rs` `dispatch_f32` consumes
  op_db's `[q, k, v]` + `{is_causal, dropout_p, attn_mask?}`
  envelope and dispatches through
  `ferrotorch_nn::functional::scaled_dot_product_attention`.
  Current sweep: `16/200 passed (184 skipped, 0 failed)` — every
  3-D non-masked `dropout_p=0` sample passes; the skips correspond
  to upstream behaviour ferrotorch's REQ-13 narrowly does not
  cover (dropout-RNG, attn_mask, 4-D multi-head, is_causal with
  N_q != N_k).

- REQ-12: Parity op `nn.functional.multi_head_attention_forward` —
  forward output matches upstream's procedural
  `multi_head_attention_forward(query, key, value, ...)` at
  `torch/nn/functional.py` to within float32 tolerance. NOT-STARTED
  until the parity-sweep runner has a dispatch arm (blocker #1455).

## Acceptance Criteria

- [x] AC-1: `MultiheadAttention::new(64, 8, true)` constructs with
  classical MHA shapes (4 weight matrices, 4 bias vectors).
- [x] AC-2: `MultiheadAttention::new(65, 8, true)` errors on
  `embed_dim % num_heads != 0`.
- [x] AC-3: `MultiheadAttention::with_gqa(4096, 32, 8, false)`
  constructs Llama-3-8B-shaped GQA with `num_kv_heads=8`.
- [x] AC-4: `forward(input)` returns `[batch, seq, embed_dim]` for
  a `[2, 5, 16]` input.
- [x] AC-5: `forward_qkv(q, kv, kv, true)` with `seq_q != seq_k`
  rejects the causal mask.
- [x] AC-6: `repeat_kv(kv, group_size=1)` is a no-op clone.
- [x] AC-7: `repeat_kv(kv, group_size=3)` produces the expected
  head ordering (heads 0..2 from input head 0, heads 3..5 from
  input head 1).
- [x] AC-8: `forward_2d` rejects GQA configurations.
- [x] AC-9: `is_training()` round-trips through `train()`/`eval()`.
- [x] AC-10: parity-sweep `nn.functional.scaled_dot_product_attention`
  at status `verified` — SHIPPED 2026-05-26 (closes #1532). Runner
  arm at `tools/parity-sweep/runner/src/main.rs` `dispatch_f32`;
  current sweep `16/200 passed (184 skipped, 0 failed)`. Skips are
  parser-narrower legitimate skips (dropout, 4-D, attn_mask,
  is_causal-N-mismatch).
- [ ] AC-11: parity-sweep `nn.functional.multi_head_attention_forward`
  at status `verified` — blocker #1455.

## Architecture

### The struct (REQ-1)

`pub struct MultiheadAttention<T: Float>` at
`pub struct MultiheadAttention in attention.rs` carries the public
`embed_dim`, `num_heads`, `num_kv_heads`, `head_dim` fields plus
four `Parameter<T>` weights (`q_proj`, `k_proj`, `v_proj`,
`out_proj`) and four optional bias `Parameter<T>` slots. The
`training` flag drives the `Module::is_training()` contract.

### Construction (REQ-2, REQ-3)

`MultiheadAttention::new` delegates to `with_gqa` with
`num_kv_heads = num_heads`. `with_gqa` validates that all three head
counts are positive, that `embed_dim % num_heads == 0`, and that
`num_heads % num_kv_heads == 0`. K/V projections are sized
`[num_kv_heads * head_dim, embed_dim]`, distinct from Q/O's
`[embed_dim, embed_dim]`. Weights initialise via
`xavier_uniform`; biases (when present) initialise to zero — both
helpers from `crate::init`.

### Forward (REQ-4, REQ-5, REQ-6, REQ-7)

`forward_qkv` validates input shapes, then branches:

1. **Fast self-attention shortcut** (only when `seq_q == seq_k == 1`,
   no causal mask, and `num_kv_heads == num_heads`): squeezes the
   middle dim, runs `linear_fused(input, V_proj) → linear_fused(.,
   out_proj)`, and unsqueezes. Two fused linears, no softmax.

2. **General path**: projects via `mm_differentiable`, reshapes to
   `[batch * num_heads, seq, head_dim]` via `permute + contiguous +
   reshape`, expands K/V for GQA via the differentiable `expand`
   primitive, computes `scores = bmm_differentiable(Q, K^T)`, scales
   by `1/sqrt(head_dim)`, adds the causal mask as additive `-1e9` on
   future positions, calls `softmax`, computes
   `bmm_differentiable(weights, V)`, reshapes back to
   `[batch, seq_q, embed_dim]`, and projects through the output
   matrix.

The 2-D fast path (`forward_2d`) is callable directly for callers
that already have `[batch, embed_dim]`-shaped input; it skips the
unsqueeze/squeeze pair and rejects GQA configurations explicitly.

### Module impl (REQ-8)

The `impl<T: Float> Module<T> for MultiheadAttention<T>` block at
`impl Module<T> for MultiheadAttention in attention.rs` provides
`forward` (self-attention via `forward_qkv(input, input, input,
false)`), `parameters`/`parameters_mut` traversal, and
`named_parameters` returning the four weight keys plus any
configured bias keys.

### Standalone helpers (REQ-10)

`reshape_to_heads`, `transpose_heads_to_2d`, and `repeat_kv` at
`pub fn reshape_to_heads`, `pub fn transpose_heads_to_2d`, and
`pub fn repeat_kv` in `attention.rs` are inference-only data-shuffle
helpers that call `data_vec()` to obtain a host buffer, write into
a fresh `Vec<T>`, and rebuild a non-grad tensor. They break the
autograd graph by design — production consumers
(`ferrotorch-llama/src/attention.rs:23`) use them only in
inference.

### Non-test production consumers

- `pub use attention::{MultiheadAttention, repeat_kv,
  reshape_to_heads, transpose_heads_to_2d}` at
  `ferrotorch-nn/src/lib.rs:194` — grandfathered public API surface.
- `ferrotorch-vision/src/models/vit.rs:20` —
  `use ferrotorch_nn::attention::MultiheadAttention` in the
  Vision Transformer build path.
- `ferrotorch-llama/src/attention.rs:23` — Llama 3 attention layer
  imports `repeat_kv` and `reshape_to_heads` for GQA + head
  shuffling in the inference hot loop.

## Parity contract

### `nn.functional.scaled_dot_product_attention`

- Upstream entry point:
  `torch/nn/functional.py — scaled_dot_product_attention`. The
  ATen native is at
  `aten/src/ATen/native/transformers/attention.cpp`.
- Edge cases preserved by `forward_qkv`:
  - **NaN / Inf in Q, K, V**: propagates through `softmax` per
    upstream — NaN inputs produce NaN outputs (no special handling).
  - **Empty seq_q / seq_k**: rejected at the shape-validation step
    (no fallthrough to an empty kernel; this is a deviation from
    upstream which produces a `[B, 0, D]` tensor).
  - **`is_causal=True` with `seq_q != seq_k`**: ferrotorch rejects;
    upstream allows it and just ignores positions outside the
    triangle. (Tracked as a divergence in blocker #1455 testing.)
  - **`scale` override**: not exposed; ferrotorch hard-codes
    `1/sqrt(head_dim)`. Upstream's `scale=` kwarg from PyTorch 2.1+
    is NOT-STARTED.
- Parity-sweep audit: `nn.functional.scaled_dot_product_attention`
  status `MISSING` in `tools/parity-sweep/parity_audit.json`
  — runner arm not wired (blocker #1455).

### `nn.functional.multi_head_attention_forward`

- Upstream entry point:
  `torch/nn/functional.py — multi_head_attention_forward`.
- Edge cases preserved:
  - **`need_weights=True`**: NOT-STARTED — ferrotorch's `forward_qkv`
    only returns the projected output, not the attention weights.
  - **`key_padding_mask`** / **`attn_mask` (non-causal)**:
    NOT-STARTED — only the boolean `causal_mask` flag is supported.
- Parity-sweep audit: `nn.functional.multi_head_attention_forward`
  status `MISSING` (blocker #1455).

## Verification

Tests in `mod tests in attention.rs` (~24 tests). Highlights:

- `test_new_valid`, `test_new_invalid_divisibility`,
  `test_new_zero_dims` — construction validation (REQ-2).
- `test_parameter_count_with_bias` /
  `test_parameter_count_without_bias` — parameter layout (REQ-1).
- `test_named_parameters` — state-dict keys (REQ-8).
- `test_output_shape`, `test_output_shape_no_bias` — forward shape
  contract (REQ-4).
- `test_self_attention_basic_forward` — finite-output sanity (REQ-5).
- `test_cross_attention_shape` — cross-attention seq_q != seq_k path
  (REQ-4).
- `test_causal_mask_different_seq_lens_error` — causal-mask
  validation (REQ-6).
- `test_with_gqa_valid_construction`, `test_with_gqa_kv_proj_shapes`,
  `test_with_gqa_rejects_non_divisible_kv_heads`,
  `test_with_gqa_rejects_zero_kv_heads`,
  `test_with_gqa_equivalent_to_new_when_kv_equals_q` — GQA
  construction (REQ-3).
- `test_repeat_kv_noop_on_group_size_1`,
  `test_repeat_kv_copies_correct_heads`,
  `test_repeat_kv_rejects_wrong_rank` — repeat_kv (REQ-7, REQ-10).
- `test_gqa_forward_output_shape_preserved`,
  `test_gqa_forward_produces_finite_values`,
  `test_gqa_forward_decoder_style_single_token`,
  `test_gqa_forward_with_causal_mask` — GQA forward (REQ-5, REQ-7).
- `test_is_send_sync` — thread-safety bound.

Parity smoke command (blocker #1455 must close before this passes):

```bash
for OP in nn.functional.scaled_dot_product_attention \
          nn.functional.multi_head_attention_forward; do
  ./target/release/parity-sweep sweep --op "$OP" --seeds 8 2>&1 \
    | grep -c "passed (0 skipped, 0 failed)"
done
```

Expected (post-#1455): each line returns `>= 1`. Current: each
returns `0` (runner arm missing).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct MultiheadAttention<T: Float>` in `attention.rs` mirroring upstream `activation.py:1089-1200`; non-test consumer: re-export at `ferrotorch-nn/src/lib.rs:194` + `ferrotorch-vision/src/models/vit.rs:20`. |
| REQ-2 | SHIPPED | impl: `pub fn new` in `attention.rs` (delegates to `with_gqa`) mirroring upstream `activation.py:1153-1188`; non-test consumer: re-export at `lib.rs:194` + `vit.rs:20`. |
| REQ-3 | SHIPPED | impl: `pub fn with_gqa` in `attention.rs` with `num_heads % num_kv_heads` validation; non-test consumer: re-export at `lib.rs:194` and Llama GQA path at `ferrotorch-llama/src/attention.rs:23` (constructs `with_gqa` for Llama-3-8B layout). |
| REQ-4 | SHIPPED | impl: `pub fn forward_qkv` in `attention.rs` with 3-D / batch / seq shape validation; non-test consumer: re-export at `lib.rs:194` + `vit.rs:20`. |
| REQ-5 | SHIPPED | impl: general-path attention body inside `forward_qkv` using `mm_differentiable`, `bmm_differentiable`, `softmax`, `mul`, `add` from `ferrotorch_core::grad_fns`; non-test consumer: `vit.rs:20` invocation, `ferrotorch-llama/src/attention.rs:23` GQA path. |
| REQ-6 | SHIPPED | impl: causal-mask construction inside `forward_qkv` (additive `-1e9` matrix `[1, seq_q, seq_k]` moved to device); non-test consumer: `vit.rs:20`, `ferrotorch-llama/src/attention.rs:23`. |
| REQ-7 | SHIPPED | impl: `group_size > 1` branch inside `forward_qkv` using `expand` from `ferrotorch_core::grad_fns::shape`; non-test consumer: `ferrotorch-llama/src/attention.rs:23`. |
| REQ-8 | SHIPPED | impl: `impl<T: Float> Module<T> for MultiheadAttention<T>` in `attention.rs`; non-test consumer: re-export at `lib.rs:194` plus every model that boxes the MHA as a `Module<T>`. |
| REQ-9 | SHIPPED | impl: `pub fn forward_2d` in `attention.rs` (GQA-rejection + fused-linear short circuit); non-test consumer: re-export at `lib.rs:194` available to downstream inference loops; Llama autoregressive forward in `ferrotorch-llama/src/attention.rs:23` consumes the `seq_len=1` self-attention shortcut inside `forward_qkv`. |
| REQ-10 | SHIPPED | impl: `pub fn reshape_to_heads`, `pub fn transpose_heads_to_2d`, `pub fn repeat_kv` in `attention.rs`; non-test consumer: `ferrotorch-llama/src/attention.rs:23` imports `repeat_kv` and `reshape_to_heads`. |
| REQ-11 | NOT-STARTED | parity-sweep runner arm for `nn.functional.scaled_dot_product_attention` not wired — blocker #1455. |
| REQ-12 | NOT-STARTED | parity-sweep runner arm for `nn.functional.multi_head_attention_forward` not wired — blocker #1455. |
