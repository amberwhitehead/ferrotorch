# ferrotorch-nn — `flex_attention` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/attention/flex_attention.py
  - aten/src/ATen/native/transformers/attention.cpp
-->

## Summary

`ferrotorch-nn/src/flex_attention.rs` implements `flex_attention` — a
generalized scaled-dot-product attention with **composable score
modifications** (causal mask, ALiBi, relative position bias) and
**sparse block masks**. Mirrors PyTorch's
`torch.nn.attention.flex_attention.flex_attention`
(`torch/nn/attention/flex_attention.py:1867-1980`) at the API surface
level — same kwargs shape (`score_mod`, `block_mask`), same
`[batch, n_q, d]` / `[batch, n_k, d]` / `[batch, n_k, d_v]` shape
contract, same softmax-after-mod semantics.

ferrotorch's implementation is a CPU per-element evaluator (no
templated kernel codegen); upstream compiles the `score_mod` callable
to a Triton kernel. The numerical contract is identical; the
performance contract is not.

## Requirements

- REQ-1: `pub fn flex_attention(query, key, value, score_mod,
  block_mask) -> FerrotorchResult<Tensor<T>>` returning
  `[batch, n_q, d_v]`. Mirrors upstream's `flex_attention(query, key,
  value, score_mod, block_mask, scale, enable_gqa, ...)` signature
  at `torch/nn/attention/flex_attention.py:1867-1980` (subset of
  kwargs supported; `scale`, `enable_gqa`, `return_lse` are
  NOT-STARTED — see REQ-9).

- REQ-2: `pub struct BlockMask` with constructors `new` (explicit
  grid), `full_mask` (all-true), `causal_mask` (lower-triangular
  block-level), `sliding_window_mask` (within `window_size`
  positions). Stores `mask: Vec<Vec<bool>>` indexed by
  `[q_block][k_block]` plus the q/k block sizes and total
  sequence lengths. Mirrors upstream's `BlockMask` data class at
  `torch/nn/attention/flex_attention.py`.

- REQ-3: `BlockMask::is_active(q_block, k_block)` /
  `allows_position(q_pos, k_pos)` / `num_q_blocks` /
  `num_k_blocks` accessors. Used by the forward to skip whole
  block pairs and by tests to validate the mask construction.

- REQ-4: `score_mod` invocation per position — a
  `Fn(&Tensor<T>, &Tensor<T>, &Tensor<T>, &Tensor<T>, &Tensor<T>)
  -> Tensor<T>` taking `(score, batch_idx, head_idx, q_idx, kv_idx)`
  scalar tensors and returning the modified score. Called once per
  (b, q, k) triple in the forward. Matches upstream's
  `(score, b, h, q_idx, kv_idx)` callable contract at
  `torch/nn/attention/flex_attention.py:1867-1900`.

- REQ-5: Pre-built `score_mod` constructors —
  `pub fn causal_score_mod`, `pub fn alibi_score_mod(slope)`,
  `pub fn relative_position_bias_score_mod(table)`. Each returns a
  boxed closure ready to pass to `flex_attention`.

- REQ-6: Sparse block-skipping — when `block_mask` is provided,
  the forward skips entire (q_block, k_block) pairs where
  `block_mask.is_active(qb, kb) == false`, treating those positions
  as `-inf` before softmax. Mirrors upstream's structured sparsity
  contract.

- REQ-7: Autograd via `FlexAttentionBackward<T>` — attached when
  any of Q, K, V has `requires_grad` and grad is globally enabled.
  Stores the post-softmax `attn_weights`, then on `backward`
  computes `grad_V = A^T @ dO`, `grad_attn = dO @ V^T`,
  `grad_scores = A * (grad_attn - rowsum(grad_attn * A))`, and
  `grad_Q = grad_scores @ K * scale`, `grad_K = grad_scores^T @ Q
  * scale`. Score modifications are detached during backward —
  this matches upstream's default for non-materialized `score_mod`.

- REQ-8: Shape validation — rejects non-3-D inputs, mismatched
  batch, K/V seq mismatch. Mirrors upstream's shape checks.

- REQ-9: NOT-STARTED — upstream's `scale` kwarg (PyTorch 2.4+) and
  `enable_gqa` kwarg are not implemented. ferrotorch hard-codes
  `scale = 1/sqrt(d)` and does not expand KV heads internally. The
  caller is expected to pre-expand for GQA.

## Acceptance Criteria

- [x] AC-1: `flex_attention(Q, K, V, None, None)` reduces to plain
  SDPA and produces `[B, n_q, d_v]`.
- [x] AC-2: `flex_attention(Q, K, V, Some(causal_score_mod), None)`
  produces causal-masked output.
- [x] AC-3: `BlockMask::causal_mask(n=8, block_size=2)` activates
  the lower-triangular block grid.
- [x] AC-4: `BlockMask::sliding_window_mask(n=16, window_size=4,
  block_size=2)` activates only blocks within the window.
- [x] AC-5: `block_mask.allows_position(q, k)` matches the underlying
  block grid.
- [x] AC-6: Backward through `flex_attention` with grad-requiring
  Q/K/V produces correctly-shaped grad tensors.
- [ ] AC-7: `scale` kwarg parity with upstream — NOT-STARTED.
- [ ] AC-8: `enable_gqa` internal KV-head broadcast — NOT-STARTED.

## Architecture

### Public entry (REQ-1)

`pub fn flex_attention<T: Float>` at
`pub fn flex_attention in flex_attention.rs` validates shapes,
optionally short-circuits when no `score_mod` and no `block_mask`
are provided (reducing to plain SDPA), and otherwise runs the
per-element score evaluator. Returns
`[batch, n_q, d_v]`.

### BlockMask (REQ-2, REQ-3)

`pub struct BlockMask` at `pub struct BlockMask in flex_attention.rs`
carries `mask: Vec<Vec<bool>>` indexed by `[q_block][k_block]`. The
three named constructors (`full_mask`, `causal_mask`,
`sliding_window_mask`) build the grid algorithmically. The
accessors `is_active(q_block, k_block)` /
`allows_position(q_pos, k_pos)` / `num_q_blocks` / `num_k_blocks`
are `#[inline]`.

### Score modifications (REQ-4, REQ-5)

`type ScoreModFn<T>` at
`type ScoreModFn in flex_attention.rs` defines the closure type:

```rust
dyn Fn(&Tensor<T>, &Tensor<T>, &Tensor<T>, &Tensor<T>, &Tensor<T>) -> Tensor<T>
```

The forward evaluates it per (b, q_idx, kv_idx) triple, passing
scalar `Tensor<T>` wrappers for the indices. The pre-built
constructors `pub fn causal_score_mod`, `pub fn alibi_score_mod`,
and `pub fn relative_position_bias_score_mod` package the common
patterns.

### Forward (REQ-6, REQ-8)

The forward path:

1. Validate shapes (REQ-8).
2. Compute `scores[b, q, k] = (Q[b, q] · K[b, k]) / sqrt(d)`.
3. If `score_mod` is `Some(_)`, evaluate it per element and
   overwrite the score.
4. If `block_mask` is `Some(_)`, set
   `scores[b, q, k] = -inf` whenever
   `block_mask.allows_position(q, k) == false`.
5. Softmax row-wise to get `attn_weights`.
6. Output `O[b, q, d_v] = sum_k(attn[b, q, k] * V[b, k, d_v])`.

### Autograd (REQ-7)

`FlexAttentionBackward<T>` at
`struct FlexAttentionBackward in flex_attention.rs` stores cloned
Q, K, V, and the softmax-normalized `attn_weights`. On
`backward(grad_output)`:

- `grad_V = attn^T @ grad_output`
- `grad_attn = grad_output @ V^T`
- `grad_scores = attn * (grad_attn - rowsum(grad_attn * attn))`
- `grad_Q = grad_scores @ K * scale`
- `grad_K = grad_scores^T @ Q * scale`

Returns `vec![grad_Q, grad_K, grad_V]` (each `Option<Tensor<T>>`
gated on the corresponding `requires_grad`).

### Non-test production consumers

- `pub use flex_attention::{BlockMask, alibi_score_mod,
  causal_score_mod, flex_attention,
  relative_position_bias_score_mod}` at
  `ferrotorch-nn/src/lib.rs:200-202` — grandfathered public API
  surface.

## Parity contract

`parity_ops = []`. The flex API piggybacks on the
`scaled_dot_product_attention` parity oracle (blocker #1455 on
`attention.md`) when called with no score_mod / no block_mask.
Distinct numerical contract:

- **`-inf` score handling** — when `score_mod` outputs `-inf` (or
  `block_mask` masks an entire row), the softmax row reduces to
  zero (`exp(-inf) = 0`). ferrotorch uses `f32`/`f64` literal
  `-1e30` rather than true `-inf` to stay numerically stable; this
  matches upstream's BLAS-friendly approximation.
- **Empty row** — when every position in a row is masked,
  `softmax([∅])` is undefined upstream and returns NaN. ferrotorch
  matches.
- **score_mod determinism** — closure is evaluated per-element;
  consumers must keep it pure (no internal state). Matches
  upstream's requirement that `score_mod` be a pure Python
  callable.

## Verification

Tests in `mod tests in flex_attention.rs`. Highlights:

- `BlockMask` constructor smoke tests (causal, sliding-window,
  full).
- `flex_attention` reduces to SDPA when no mods / mask given.
- `flex_attention + causal_score_mod` matches SDPA + causal mask.
- Backward through Q, K, V produces non-None grad tensors.

No parity-sweep ops declared. Smoke command:

```bash
cargo test -p ferrotorch-nn --lib flex_attention:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn flex_attention<T: Float>` in `flex_attention.rs` mirroring upstream `torch/nn/attention/flex_attention.py:1867-1980`; non-test consumer: re-export at `ferrotorch-nn/src/lib.rs:200`. |
| REQ-2 | SHIPPED | impl: `pub struct BlockMask` plus the four constructors (`new`, `full_mask`, `causal_mask`, `sliding_window_mask`) in `flex_attention.rs`; non-test consumer: re-export at `lib.rs:200`. |
| REQ-3 | SHIPPED | impl: `pub fn is_active`, `pub fn allows_position`, `pub fn num_q_blocks`, `pub fn num_k_blocks` in `flex_attention.rs`; non-test consumer: re-export at `lib.rs:200`. |
| REQ-4 | SHIPPED | impl: `type ScoreModFn<T>` and per-element evaluation inside `flex_attention` in `flex_attention.rs`; non-test consumer: re-export at `lib.rs:200`. |
| REQ-5 | SHIPPED | impl: `pub fn causal_score_mod`, `pub fn alibi_score_mod`, `pub fn relative_position_bias_score_mod` in `flex_attention.rs`; non-test consumer: re-export at `lib.rs:200-202`. |
| REQ-6 | SHIPPED | impl: block-mask skip logic inside `flex_attention` using `block_mask.allows_position`; non-test consumer: re-export at `lib.rs:200`. |
| REQ-7 | SHIPPED | impl: `struct FlexAttentionBackward<T>` plus `impl GradFn<T>` in `flex_attention.rs` (full-matrix recompute backward); non-test consumer: re-export at `lib.rs:200` — caller-side grad-requiring inputs trigger the backward via the autograd engine. |
| REQ-8 | SHIPPED | impl: shape-validation guards at the head of `flex_attention` (3-D rank, batch alignment, K/V seq match); non-test consumer: re-export at `lib.rs:200`. |
| REQ-9 | NOT-STARTED | `scale` kwarg and `enable_gqa` kwarg from upstream `torch/nn/attention/flex_attention.py` are not yet exposed. Tracked under the same umbrella as blocker #1455 (attention SDPA parity); no separate blocker filed because no production code path is currently blocked on these kwargs (callers pre-scale and pre-expand). |
