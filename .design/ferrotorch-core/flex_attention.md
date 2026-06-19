# Flex Attention

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - torch/nn/attention/flex_attention.py
-->

## Summary

`ferrotorch-core/src/flex_attention.rs` implements `flex_attention`,
a generalized scaled-dot-product attention primitive with an optional
score-modification callback. Mirrors `torch.nn.attention.flex_attention`
(`torch/nn/attention/flex_attention.py`). Q/K/V have shape
`[batch, heads, n, d]`; the forward composes `bmm → mul → softmax →
bmm` via the existing differentiable ops, so the autograd chain
produces correct gradients for Q, K, V automatically — no custom
backward node.

## Requirements

- REQ-1: `flex_attention(query, key, value, score_mod)` — compute
  `softmax((Q@K^T)/sqrt(d) + score_mod(...)) @ V`. Mirrors
  `torch.nn.attention.flex_attention(query, key, value, score_mod)`.
  Shapes: Q `[B,H,nq,d]`, K `[B,H,nk,d]`, V `[B,H,nk,dv]` → output
  `[B,H,nq,dv]`.
- REQ-2: Score-modification callback — `score_mod: Option<Fn(&Tensor,
  batch, head) -> Result<Tensor>>` operates on the full `[nq, nk]`
  per-(batch, head) scores matrix. Mirrors the
  `score_mod: Callable[[score, b, h, q_idx, kv_idx], score]` PyTorch
  signature, with the simplification that the Rust callback receives
  the full matrix (batched per-(b,h)) rather than per-element. The
  per-element form is expressible via the callback's own elementwise
  ops.
- REQ-3: Shape validation — Q/K/V must all be 4-D; `K.shape[3] ==
  Q.shape[3]` (matching d), `V.shape[2] == K.shape[2]` (matching
  nk); errors with `ShapeMismatch`. Device match enforced.
- REQ-4: GPU-resident composition — Q@K^T via `bmm_differentiable`
  (cuBLAS on CUDA), `mul` for `1/sqrt(d)` scaling, `softmax` for the
  row-normalisation, `bmm` for the value contraction. All on-device;
  no `.cpu()` round trip in the forward. The earlier loop-based
  implementation that downloaded to CPU is replaced (see the
  in-line commentary at `flex_attention.rs:146-160`).
- REQ-5: Backward — composes per-op backwards: `bmm`/`mul`/`softmax`/
  `bmm` each carry their own grad_fn, so the chain propagates Q/K/V
  gradients automatically. NO custom `FlexAttentionBackward` node —
  the previous pass-through backward was structurally wrong, replaced
  by autograd composition.
- REQ-6: `d == 0` guard — emits `InvalidArgument` "head dimension d
  must be > 0" to avoid division by zero in the `1/sqrt(d)` scaling.
- REQ-7: Empty-shape parity — local PyTorch 2.11.0+cu130 accepts
  `B == 0` and `nq == 0`, returning empty outputs of shape
  `[B,H,nq,dv]`, including when `score_mod` is present. It rejects
  `H == 0` and `nk == 0`; ferrotorch returns structured
  `InvalidArgument` errors for those PyTorch-invalid shapes.

## Acceptance Criteria

- [x] AC-1: `cargo test -p ferrotorch-core --lib flex_attention::tests`
  passes (12 tests in `flex_attention.rs`).
- [x] AC-2: Basic 4-D attention round-trip — `test_flex_attention_basic`
  at `test_flex_attention_basic in flex_attention.rs` returns shape `[1,1,2,2]`.
- [x] AC-3: Score-mod additive-constant invariance — adding `+1.0` to
  all scores doesn't change softmax output
  (`test_flex_attention_score_mod_additive_bias in flex_attention.rs`).
- [x] AC-4: Backward propagates to Q/K/V — gradients exist after
  `loss.backward()` (`test_flex_attention_grad_propagates_to_qkv`
  at `test_flex_attention_grad_propagates_to_qkv in flex_attention.rs`).
- [x] AC-5: `d == 0` → `InvalidArgument`
  (`test_flex_attention_d_zero in flex_attention.rs`).
- [x] AC-6: Hand-computed numerical reference — `test_flex_attention_numerical_value`
  at `test_flex_attention_numerical_value in flex_attention.rs` checks against `[1.6603, 2.6602, 2.3399, 3.3399]` within
  `1e-3`.
- [x] AC-7: Empty-shape parity probes — `score_mod_zero_batch_returns_empty_like_torch`,
  `score_mod_zero_query_returns_empty_and_visits_existing_batch_heads`,
  `zero_heads_rejected_like_torch`, and `zero_key_length_rejected_like_torch`
  pin the live PyTorch 2.11.0+cu130 behavior for these edge cases (#1790).
  `gpu_score_mod_zero_batch_returns_cuda_empty_like_torch` in
  `tests/conformance_flex_attention.rs` proves the valid zero-batch
  score_mod result remains CUDA-resident.
- [ ] AC-8: Parity-sweep `nn.functional.scaled_dot_product_attention`
  at `--seeds 8` returns ≥1 passed sample — NOT-STARTED, blocked on
  #1532 (runner has no arm; current `0/200 passed (200 skipped)`).

## Architecture

The single `pub fn flex_attention<T, F>` at `flex_attention.rs:81-260`
takes `query`, `key`, `value`, and `Option<F>` where `F: Fn(&Tensor<T>,
usize, usize) -> Result<Tensor<T>> + Send + Sync + 'static`.

Shape + device validation runs at `:91-141`. The implementation:

1. Reshape Q/K/V from `[B,H,N,D]` to `[B*H, N, D]` via
   `grad_fns::shape::reshape` (`reshape in flex_attention.rs`) — differentiable to
   preserve grad flow to Q/K/V.
2. Transpose K to `[B*H, D, NK]` via `transpose(1, 2)` (`transpose in flex_attention.rs`),
   a zero-copy stride view.
3. Q@K^T via `grad_fns::linalg::bmm_differentiable` → `[B*H, NQ, NK]`
   (`:179`). cuBLAS on CUDA.
4. Multiply by `1/sqrt(d)` scalar lifted to the input device
   (`flex_attention.rs`).
5. Reshape to `[B, H, NQ, NK]` (`:188-191`) for score_mod and softmax.
6. If `score_mod` is provided and `B*H > 0`, walk each (b, h),
   `narrow → narrow → squeeze → squeeze` to extract `[NQ, NK]`, invoke
   the callback, validate shape, `unsqueeze → unsqueeze` to lift back to
   `[1, 1, NQ, NK]`, then `cat` along heads and batches. If `B == 0`,
   keep the already-shaped score tensor instead of calling `cat([])`,
   matching PyTorch's empty output behavior.
7. `softmax` along the last (nk) dim (`:245`).
8. Reshape weights to `[B*H, NQ, NK]` (`:249-250`).
9. weights @ V via `bmm_differentiable` → `[B*H, NQ, DV]` (`:253`).
10. Reshape to `[B, H, NQ, DV]` (`:256-259`).

The score-mod callback path's `narrow + squeeze + cat` is a measurable
overhead vs. the kernel-fused approach PyTorch ships in
`torch._dynamo`-compiled mode; this is acceptable for the eager Rust
implementation. The `narrow`/`squeeze`/`cat` are all device-aware (no
silent CPU detour).

The `#[allow(clippy::needless_pass_by_value)]` at `:80` is documented:
the `Option<F>` ergonomic shape requires by-value (re-exported via
`ferrotorch-nn`).

**Non-test consumer**: re-exported at `lib.rs:157` as
`ferrotorch_core::flex_attention`. The intended downstream consumer is
`ferrotorch-nn::attention::FlexAttention` (a module-style wrapper),
which composes `flex_attention` with a learnable score-bias callback.
At this layer the boundary symbol IS the public API per goal.md S5.

## Parity contract

`parity_ops = ["nn.functional.scaled_dot_product_attention"]`. The
parity-sweep oracle treats `flex_attention` with `score_mod=None` as
equivalent to `torch.nn.functional.scaled_dot_product_attention`
(causal=False, dropout=0). Currently the runner has no
`nn.functional.scaled_dot_product_attention` dispatch arm —
`./target/release/parity-sweep sweep --op
nn.functional.scaled_dot_product_attention --seeds 8` reports `0/200
passed (200 skipped, 0 failed)`. Tracked by #1532. No failing
samples; implementation is unverified by parity sweep, only by unit
tests + hand-computed numerical reference.

## Verification

`cargo test -p ferrotorch-core --lib flex_attention::tests` runs 12
tests including hand-computed numerical reference, grad propagation,
score_mod empty-output parity, and PyTorch-invalid zero-head/key
rejection. `cargo test -p ferrotorch-core --features gpu --test
conformance_flex_attention` runs the CPU/GPU fixture suite plus the
CUDA-residency zero-batch score_mod probe. Parity-sweep smoke fires `0/200 passed (200 skipped, 0
failed)` — runner gap, not divergence.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `flex_attention` at `flex_attention.rs:81` mirrors `torch.nn.attention.flex_attention` (`torch/nn/attention/flex_attention.py`); non-test consumer: re-exported as `ferrotorch_core::flex_attention` at `lib.rs:157` (the boundary symbol IS the public API surface, per goal.md S5) |
| REQ-2 | SHIPPED | impl: `score_mod: Option<F>` parameter at `flex_attention.rs:85`; non-test consumer: re-exported public API at `lib.rs:157` |
| REQ-3 | SHIPPED | impl: shape + device validation at `flex_attention.rs:91-141`; non-test consumer: `flex_attention` itself, called via the re-export |
| REQ-4 | SHIPPED | impl: GPU-aware composition at `flex_attention.rs:167-259` using `bmm_differentiable` + `softmax` + `mul`; non-test consumer: `flex_attention` re-export. The earlier loop-based CPU-only implementation is documented as replaced in the in-line comment at `:146-160` |
| REQ-5 | SHIPPED | impl: no custom backward node — autograd composition via the differentiable building-blocks; non-test consumer: `flex_attention` re-export. The grad-propagation test at `:440` pins this contract |
| REQ-6 | SHIPPED | impl: `d == 0` check at `flex_attention.rs:113-117`; non-test consumer: the public `flex_attention` entry |
| REQ-7 | SHIPPED | impl: `heads == 0` / `n_k == 0` structured guards plus `score_mod` `bh == 0` skip-cat path in `flex_attention.rs`; tests: `score_mod_zero_batch_returns_empty_like_torch`, `score_mod_zero_query_returns_empty_and_visits_existing_batch_heads`, `zero_heads_rejected_like_torch`, `zero_key_length_rejected_like_torch`, `gpu_score_mod_zero_batch_returns_cuda_empty_like_torch` |
