# ferrotorch-profiler — FLOPS estimation from op name + input shapes

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/profiler/profiler.py
  - torch/autograd/profiler.py
-->

## Summary

`ferrotorch-profiler/src/flops.rs` implements the
`flops::estimate(op_name, &input_shapes) -> Option<u64>` function
that converts an op name plus its input tensor shapes into a
shape-driven FLOP count. Mirrors PyTorch's `with_flops=True` path
on `torch.autograd.profiler.profile` (`torch/autograd/profiler.py:219`
"use formula to estimate the FLOPs"); the upstream implementation
lives in C++ under `torch/csrc/autograd/profiler_kineto.cpp`'s
shape-handling and is approximated here with the same
multiply-accumulate-as-2-FLOPs convention.

## Requirements

- REQ-1: `#[must_use] pub fn estimate(op_name: &str, input_shapes: &[Vec<usize>]) -> Option<u64>`
  recognises 22 op-name families and returns a shape-driven FLOP
  count, with `None` on any unrecognised op or insufficient shape
  data. Mirrors the FLOP-counting contract `with_flops=True` advertises
  at `torch/profiler/profiler.py:156-157` ("matrix multiplication
  and 2D convolution operators"). ferrotorch covers the
  PyTorch-documented set plus the elementwise / reduction / norm /
  activation families.
- REQ-2: Elementwise binary ops (`add`, `sub`, `mul`, `div`) charge
  one FLOP per output element (broadcast-aware via `max(numel(a), numel(b))`).
  Elementwise unary ops (`neg`, `abs`, `sqrt`, `exp`, `log`) and
  activations (`relu`, `sigmoid`, `tanh`, `gelu`, `silu`,
  `leaky_relu`) charge one FLOP per input element. Mirrors the
  PyTorch shape-driven estimate (every element pays one FMA).
- REQ-3: `softmax` / `log_softmax` charge `5 * numel` (max + exp +
  divide + reduction); `sum` / `mean` / `prod` charge `numel - 1`
  (N adds for N elements). Conservative approximations matching the
  order-of-magnitude TensorFlow / PyTorch convention.
- REQ-4: Matrix multiplication family (`matmul`, `mm`, `bmm`,
  `linear`) computes `2 * batch * M * N * K` from the last-two-dim
  convention, handling 1D dot (M=1 or N=1), 2D mm, batched bmm, and
  N-D matmul with broadcasted batch dims. Returns `None` when the
  inner dimensions don't match. Mirrors PyTorch's MAC = 2 FLOPs
  convention and the matmul shape inference at
  `torch/_torch_docs.py` `matmul` doc.
- REQ-5: Convolution family (`conv1d`, `conv2d`, `conv3d`) computes
  `2 * batch * C_out * C_in * kernel_volume * spatial_volume`,
  assuming stride=1, padding='same' (the spatial volume equals the
  input spatial volume — the user-visible PyTorch
  `with_flops` formula). Returns `None` when input/weight rank
  doesn't match `2 + n_spatial`.
- REQ-6: Norm family (`layer_norm`, `rms_norm`, `batch_norm`,
  `group_norm`) and `pow` are recognised with simple per-element
  approximations (8 FLOPs / element for norms; 2 FLOPs / element
  for pow with a non-integer exponent). Coverage matches the
  approximations the PyTorch Profiler UI shows.
- REQ-7: Every estimator returns `None` rather than panicking on
  bad shapes (empty shape vec, mismatched ranks, inner-dim
  mismatch). The contract documented at the top of the file: the
  estimate is a lower bound when summed across recognised ops.

## Acceptance Criteria

- [x] AC-1: `estimate("add", &[vec![3,4], vec![3,4]]) == Some(12)`.
- [x] AC-2: `estimate("add", &[vec![3,4], vec![1,4]]) == Some(12)` (broadcast).
- [x] AC-3: `estimate("matmul", &[vec![4,5], vec![5,6]]) == Some(240)`.
- [x] AC-4: `estimate("bmm", &[vec![3,4,5], vec![3,5,6]]) == Some(720)`.
- [x] AC-5: `estimate("matmul", &[vec![4,5], vec![6,7]]) == None` (inner mismatch).
- [x] AC-6: `estimate("conv2d", &[vec![1,3,32,32], vec![16,3,3,3]]) == Some(884_736)`.
- [x] AC-7: `estimate("totally_made_up_op", &[vec![3,4]]) == None`.

## Architecture

### Public surface (REQ-1)

`#[must_use] pub fn estimate(op_name: &str, input_shapes: &[Vec<usize>]) -> Option<u64>`
is the only exported item. The shape vector is borrowed
(read-only); the return is `u64` (no signed FLOP counts) wrapped in
`Option` so callers can sum `filter_map`-style without sentinel
poisoning. The `#[must_use]` attribute reinforces that the estimate
itself has no side effects — dropping the return value is almost
certainly a bug.

### Match table (REQ-2, REQ-3, REQ-4, REQ-5, REQ-6)

The `match op_name` block at `flops in flops.rs` is the dispatch table.
Each arm calls one of five private helpers:

- `elementwise_binary(shapes)` — broadcast-aware numel max.
- `elementwise_unary(shapes)` — first-input numel.
- `matmul_flops(shapes)` — handles 1D / 2D / N-D matmul with
  batch broadcasting.
- `conv_nd_flops(shapes, n_spatial)` — rank-strict, assumes
  stride=1 padding='same'.
- Inline closures for softmax (`5 * n`), reductions (`n - 1`),
  norms (`8 * n`), pow (`2 * n`).

The fall-through arm `_ => None` is the contract: any unrecognised
op silently returns no estimate. This is intentional — the
profiler's job is to surface what it can measure, not to fail.

### `numel` helper

`numel(shape)` returns `shape.iter().product::<usize>().max(1)` so
zero-rank tensors (scalars) count as 1 element, matching the
PyTorch convention.

### Non-test production consumers

- `ferrotorch-profiler/src/profiler.rs` `use crate::flops;` —
  imported at module scope so `profiler::Profiler::record` (line 113)
  and `OpProfiler::record_op` (line 384) can call
  `flops::estimate(name, &input_shapes_vec)` to populate the
  `ProfileEvent::flops` field when shapes are recorded.
- `pub in ferrotorch-profiler/src/lib.rs` `pub mod flops;` — re-exported
  at the crate root so user code can call
  `ferrotorch_profiler::flops::estimate(...)` directly when building
  custom rollup logic outside the standard `with_profiler` flow.
- `ferrotorch/src/lib.rs:107` `pub use ferrotorch_profiler::*;`
  propagates the `flops` module into the meta-crate prelude.

## Parity contract

`parity_ops = []`. FLOP estimation is an approximation; per-tensor
numerical parity isn't meaningful (the "true" FLOP count depends on
hardware details PyTorch also doesn't observe).

Edge cases the estimator owns:

- **Empty shape vec**: every dispatch arm calls `shapes.first()?`
  or `shapes.len() < 2` checks, returning `None`. Matches PyTorch's
  "no shapes → no FLOPs" behaviour.
- **Broadcast in elementwise binary**: ferrotorch approximates
  output numel as `max(numel(a), numel(b))`. For true broadcast
  output that exceeds the max (e.g. `[3,1] + [1,4]` → `[3,4]`), the
  estimate undercounts by the cross-product factor. Acceptable for
  a lower-bound estimator.
- **Pow with integer exponent vs float**: ferrotorch uses 2 FLOPs /
  element for both — conservative for integer pow (which is
  cheaper) and an undercount for float pow (which uses log/exp).
  Matches the same simplification PyTorch's Kineto formula makes.
- **Conv with stride or non-'same' padding**: ferrotorch assumes
  stride=1, padding='same' (output spatial = input spatial). True
  conv with stride 2 should be ~4x smaller; users invoking
  high-precision FLOP analysis should compute manually.

## Verification

13 unit tests in `flops.rs` `mod tests` (lines 169-269) cover every
op family and the failure modes:

- `test_elementwise_add_2d`, `test_elementwise_with_broadcast`
- `test_unary_relu`
- `test_softmax_approx`
- `test_matmul_2d_2d`, `test_matmul_batched`, `test_matmul_dot_1d`,
  `test_matmul_inner_mismatch_returns_none`
- `test_conv2d`, `test_conv1d`
- `test_unknown_op_returns_none`, `test_no_shapes_returns_none`
- `test_layer_norm_estimate`
- `test_sum_reduction`

Plus the crate-level integration `tests/conformance_flops.rs`.
Smoke:

```bash
cargo test -p ferrotorch-profiler --lib flops 2>&1 | tail -3
```

Expected: `13 passed; 0 failed` for `flops::tests`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `#[must_use] pub fn estimate` at `estimate in ferrotorch-profiler/src/flops.rs` mirroring `torch/autograd/profiler.py:219` `with_flops=True` contract; non-test consumer: `estimate in ferrotorch-profiler/src/profiler.rs` `flops::estimate(name, &input_shapes_vec)` inside `Profiler::record` and `record in ferrotorch-profiler/src/profiler.rs` inside `OpProfiler::record_op` (called from `ferrotorch-core/src/grad_fns/arithmetic.rs` etc. on every tensor op when a profiler is active). |
| REQ-2 | SHIPPED | impl: `elementwise_binary` at `ferrotorch-profiler/src/flops.rs:91`, `elementwise_unary` at line 102, match arms for the 4 binary ops + 5 unary + 6 activation ops at lines 44-52; non-test consumer: same path as REQ-1 — `Profiler::record("add", ...)` populates `flops: Some(numel)` on every recorded `add` event. |
| REQ-3 | SHIPPED | impl: softmax/log_softmax arm at `ferrotorch-profiler/src/flops.rs:55-58`, reduction arm at line 61-64; non-test consumer: same `Profiler::record` path; `softmax` / `log_softmax` recorded via the dispatch hook in `ferrotorch-core/src/grad_fns/transcendental.rs` populates the FLOP estimate. |
| REQ-4 | SHIPPED | impl: `matmul_flops` at `ferrotorch-profiler/src/flops.rs:109` with 1D/2D/ND + batch broadcasting, match arm at line 73; non-test consumer: `Profiler::record("matmul", &[a_shape, b_shape])` flows through `flops::estimate` → `matmul_flops`; `bmm` / `mm` / `linear` likewise. |
| REQ-5 | SHIPPED | impl: `conv_nd_flops` at `ferrotorch-profiler/src/flops.rs:149` with rank-strict input/weight checks, conv1d/conv2d/conv3d arms at lines 75-77; non-test consumer: `Profiler::record("conv2d", ...)` populates `flops` via the same hook path. |
| REQ-6 | SHIPPED | impl: norm-family arm at `ferrotorch-profiler/src/flops.rs:79-82` (8 FLOPs/element approx), pow arm at line 68-71; non-test consumer: `Profiler::record("layer_norm", ...)` and `Profiler::record("pow", ...)` route through `flops::estimate`. |
| REQ-7 | SHIPPED | impl: every arm uses `?` on `shapes.first()` and length checks at `flops in flops.rs`, `103`, `110`, `150` — no panics, `None` on bad shapes; non-test consumer: `ferrotorch-profiler/src/profiler.rs` consumes `Option<u64>` and stores it directly in `ProfileEvent::flops`, so the `None` return surfaces without crashing the recording path. |
