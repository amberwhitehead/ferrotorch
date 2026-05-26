# Tensor Manipulation Ops

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/
  - c10/
  - torch/_torch_docs.py
  - torch/overrides.py
-->

## Summary

`ferrotorch-core/src/ops/tensor_ops.rs` ships the common tensor
manipulation primitives: `triu` / `tril` (triangular masks),
`diag` / `diagflat` (diagonal extraction and construction), `roll`
(circular shift), and `cdist` (pairwise distance matrix). Each
mirrors the same-named `torch.*` function. `roll` has a GPU f32 fast
path (cumulative-scan-style dispatch). `triu`/`tril`/`diag`/`diagflat`/
`cdist` are CPU-only — CUDA inputs error with
`NotImplementedOnCuda` (GPU lowerings tracked under #1545).

## Requirements

- REQ-1: `triu(input, diagonal)` — 2-D upper-triangular mask. Elements
  below the `diagonal`-th diagonal are zeroed. `diagonal=0` is the
  main diagonal; positive shifts above, negative below. Mirrors
  `torch.triu`.
- REQ-2: `tril(input, diagonal)` — 2-D lower-triangular mask.
  Elements above the `diagonal`-th diagonal are zeroed. Mirrors
  `torch.tril`.
- REQ-3: `diag(input, diagonal)` — extract the `diagonal`-th diagonal
  of a 2-D input (returns 1-D), OR build a 2-D diagonal matrix from
  a 1-D input (returns 2-D). Mirrors `torch.diag`.
- REQ-4: `diagflat(input, diagonal)` — flatten input then build a
  2-D diagonal matrix. Mirrors `torch.diagflat`.
- REQ-5: `roll(input, shifts, dim)` — circular shift along a
  dimension. Wraps elements past the end. Has GPU f32 fast path via
  `backend.roll_f32` (other dtypes / GPU paths error with
  `NotImplementedOnCuda`). Autograd: when `input.requires_grad()`,
  attaches `RollBackward` that pushes gradients back through the
  inverse shift. Mirrors `torch.roll`.
- REQ-6: `cdist(x1, x2, p)` — pairwise Lp distance matrix. Accepts
  2-D `[P, M]` / `[R, M]` (→ `[P, R]`) or 3-D `[B, P, M]` / `[B, R,
  M]` (→ `[B, P, R]`). Mirrors `torch.cdist`.
- REQ-7: `roll_cpu_inner` is `pub(crate)` and shared with
  `RollBackward` so the backward can reuse the same CPU shift loop
  with negated shift.

## Acceptance Criteria

- [x] AC-1: `cargo test -p ferrotorch-core --lib ops::tensor_ops`
  passes (covers triu/tril/diag/diagflat/roll/cdist).
- [x] AC-2: `triu` of a 3x3 with `diagonal=0` zeros the strict
  lower-left.
- [x] AC-3: `tril` of a 3x3 with `diagonal=0` zeros the strict
  upper-right.
- [x] AC-4: `diag` of a 1-D `[1,2,3]` produces a 3x3 matrix with
  `[1,2,3]` on the diagonal.
- [x] AC-5: `roll([1,2,3,4,5], 2, 0)` → `[4,5,1,2,3]`.
- [x] AC-6: `roll([1,2,3,4,5], -1, 0)` → `[2,3,4,5,1]`.
- [x] AC-7: `cdist` with L2 distance from `(0,0)`, `(1,0)`, `(0,1)`
  to `(1,1)` returns `[sqrt(2), 1, 1]`.
- [x] AC-8: `roll` autograd — gradient flows back through
  `RollBackward` (the test in `grad_fns/shape.rs:1545+` pins this).
- [ ] AC-9: GPU paths for `triu`/`tril`/`diag`/`diagflat`/`cdist` —
  NOT-STARTED, blocked on #1545.

## Architecture

`triu` at `ops/tensor_ops.rs:28-55` validates ndim == 2, rejects
CUDA, walks every `(r, c)` slot, emitting `data[r*cols+c]` when
`c >= r + diagonal`, else `T::zero()`. `tril` at `:62-89` is
symmetric with `c <= r + diagonal`.

`diag` at `:98-148` has a 1-D and 2-D branch:
- 1-D → 2-D matrix of size `(n + |diagonal|)`; place `data[i]` at
  `(i, i + diagonal)` for non-negative diag or `(i + |diagonal|, i)`
  for negative.
- 2-D → 1-D vector of length `min(rows - |start_r|, cols -
  |start_c|)`; read along the diagonal.

`diagflat` at `:155-168`: if 1-D, delegate to `diag`. Else,
`data_vec` to flatten, build a 1-D tensor, delegate to `diag`.

`roll` at `:181-250` is the most involved:
1. Validate `dim < shape.len()`.
2. Normalize `shifts`: `shift_norm = ((shifts % dim_size) +
   dim_size) % dim_size`. Handles negative shifts. `dim_size == 0`
   short-circuits to `shift_norm = 0`.
3. `shift_norm == 0` → return `input.clone()` (preserves the
   upstream grad_fn).
4. GPU f32 fast path at `:209-234`: if `input.is_cuda()` and `T ==
   f32`, call `backend.roll_f32(handle, outer, dim_size, inner,
   shift_norm)`. Other GPU dtypes return `NotImplementedOnCuda`.
5. CPU path at `:237-238`: call `roll_cpu_inner(&data, shape,
   shift_norm, dim)`.
6. Autograd at `:240-249`: when `requires_grad() && is_grad_enabled`,
   attach `RollBackward { input, shifts, dim }` via
   `Tensor::from_operation`.

`roll_cpu_inner` at `:259-282` is the shared shift loop —
`pub(crate)` so `RollBackward::backward` can reuse it with negated
shift. Walks `outer × dim_size × inner` and writes `out[..., new_d,
...] = data[..., d, ...]` where `new_d = (d + shift_norm) %
dim_size`.

`cdist` at `:292-381`:
1. Identify batched (3-D) vs unbatched (2-D) input shape.
2. Validate feature dims match across `x1` / `x2`.
3. For each batch, walk `(i, j)` pairs, accumulating
   `|diff|^p` over the feature dim, then take `result^(1/p)`.

**Non-test consumers**:

- `crate::grad_fns::shape::RollBackward::backward` at
  `grad_fns/shape.rs:1006` invokes `crate::ops::tensor_ops::roll_cpu_inner`
  for the backward shift. This is the REQ-7 production consumer of
  the shared inner kernel.
- `crate::grad_fns::shape::RollBackward` documentation at
  `grad_fns/shape.rs:921` references calling back into
  `crate::ops::tensor_ops::roll` for the in-graph backward path.
- Re-exported at `lib.rs:177` as
  `ferrotorch_core::{cdist, diag, diagflat, roll, tril, triu}`.

## Parity contract

`parity_ops = []` (no specific parity op declared). Numeric contract
is byte-for-byte parity with `torch.{triu, tril, diag, diagflat,
roll, cdist}`. Verified through unit tests + the autograd
correctness of `roll` (covered by `RollBackward` tests in
`grad_fns/shape.rs:1551-1670`).

## Verification

`cargo test -p ferrotorch-core --lib ops::tensor_ops` covers the
forward paths; `cargo test -p ferrotorch-core --lib
grad_fns::shape::tests::roll_*` covers the autograd path.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `triu` at `ops/tensor_ops.rs:28`; non-test consumer: re-exported as `ferrotorch_core::triu` at `lib.rs:177` (boundary public API per goal.md S5) |
| REQ-2 | SHIPPED | impl: `tril` at `ops/tensor_ops.rs:62`; non-test consumer: re-exported as `ferrotorch_core::tril` at `lib.rs:177` |
| REQ-3 | SHIPPED | impl: `diag` at `ops/tensor_ops.rs:98`; non-test consumer: re-exported as `ferrotorch_core::diag` at `lib.rs:177` |
| REQ-4 | SHIPPED | impl: `diagflat` at `ops/tensor_ops.rs:155`; non-test consumer: re-exported as `ferrotorch_core::diagflat` at `lib.rs:177` |
| REQ-5 | SHIPPED | impl: `roll` at `ops/tensor_ops.rs:181`; non-test consumer: re-exported as `ferrotorch_core::roll` at `lib.rs:177`. The autograd-attached `RollBackward` is the consumer of REQ-7's shared inner kernel |
| REQ-6 | SHIPPED | impl: `cdist` at `ops/tensor_ops.rs:292`; non-test consumer: re-exported as `ferrotorch_core::cdist` at `lib.rs:177` |
| REQ-7 | SHIPPED | impl: `roll_cpu_inner` at `ops/tensor_ops.rs:259`; non-test consumer: `crate::grad_fns::shape::RollBackward::backward` at `grad_fns/shape.rs:1006` invokes `crate::ops::tensor_ops::roll_cpu_inner` for the backward shift |
