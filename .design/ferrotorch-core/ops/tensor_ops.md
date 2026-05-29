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
path (cumulative-scan-style dispatch). `triu`/`tril` have GPU f32+f64
fast paths (resident triangular-mask kernels in
`ferrotorch-gpu/src/triangular.rs`, crosslink #1545 / sub #1535).
`diag`/`diagflat` have GPU f32+f64 fast paths (resident diagonal
gather/scatter kernels in `ferrotorch-gpu/src/diag.rs`); `cdist` has a
GPU fast path (resident pairwise-distance kernel in
`ferrotorch-gpu/src/distance.rs`: f32 covers `p in {1, 2, inf}` plus
general finite `p`, f64 covers `p in {1, 2, inf}`). The remaining
CPU-only GPU cases (`cdist p == 0` count-norm, general-p f64) surface
as `NotImplementedOnCuda`.

## Requirements

- REQ-1: `triu(input, diagonal)` — upper-triangular mask over the LAST
  TWO DIMS of a tensor with `ndim >= 2`, batching over all leading dims
  (output shape == input shape). Elements below the `diagonal`-th
  diagonal are zeroed. `diagonal=0` is the main diagonal; positive
  shifts above, negative below. Mirrors `torch.triu`
  (`TriangularOps.cpp:31` requires `dim() >= 2`; batched per
  `cuda/TriangularOps.cu:120`). Has a GPU f32+f64 resident fast path via
  `backend.triu_f32`/`triu_f64` (crosslink #1545 / sub #1535); other
  GPU dtypes error with `NotImplementedOnCuda`.
- REQ-2: `tril(input, diagonal)` — lower-triangular mask over the LAST
  TWO DIMS of a tensor with `ndim >= 2`, batching over all leading dims.
  Elements above the `diagonal`-th diagonal are zeroed. Mirrors
  `torch.tril` (`TriangularOps.cpp:25` requires `dim() >= 2`). Has a GPU
  f32+f64 resident fast path via `backend.tril_f32`/`tril_f64`
  (crosslink #1545 / sub #1535).
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
  `RollBackward` (the `roll_*` tests in `grad_fns/shape.rs` pin this).
- [x] AC-9a: GPU paths for `triu`/`tril` (f32+f64) — SHIPPED (crosslink
  #1545 / sub #1535). LIVE GPU-vs-CPU value parity verified by
  `ferrotorch-gpu/tests/test_gpu_triangular.rs` (asserts `is_cuda()` on
  the result AND byte-identical to the CPU `triu`/`tril` reference).
- [x] AC-9b: GPU paths for `diag`/`diagflat`/`cdist` — SHIPPED (crosslink
  #1545 / sub #1535). LIVE GPU-vs-CPU value parity verified by
  `ferrotorch-gpu/tests/test_gpu_diag_cdist.rs` (asserts `is_cuda()` on
  the result AND value-match to the CPU `diag`/`diagflat`/`cdist`
  reference, for both diagonal directions and the L1/L2/Linf/general-p
  norms). `diag` 1-D→2-D scatter + 2-D→1-D gather + `diagflat` flatten
  all stay GPU-resident (NO `.cpu()` round trip).

## Architecture

`triu` in `ops/tensor_ops.rs` validates `ndim >= 2` (mirrors
`TriangularOps.cpp:31` `dim() >= 2`), takes the trailing two dims as the
matrix and batches over all leading dims (mirrors
`cuda/TriangularOps.cu:120`), routes CUDA inputs through the resident
kernel, then for the CPU path walks every `(b, r, c)` slot, emitting
`data[b*rows*cols + r*cols + c]` when `c >= r + diagonal`, else
`T::zero()`. `tril` in `ops/tensor_ops.rs` is symmetric with
`c <= r + diagonal`.

`diag` in `ops/tensor_ops.rs` has a 1-D and 2-D branch:
- 1-D → 2-D matrix of size `(n + |diagonal|)`; place `data[i]` at
  `(i, i + diagonal)` for non-negative diag or `(i + |diagonal|, i)`
  for negative.
- 2-D → 1-D vector of length `min(rows - |start_r|, cols -
  |start_c|)`; read along the diagonal.

`diagflat` in `ops/tensor_ops.rs`: if 1-D, delegate to `diag`. Else,
`data_vec` to flatten, build a 1-D tensor, delegate to `diag`.

`roll` in `ops/tensor_ops.rs` is the most involved:
1. Validate `dim < shape.len()`.
2. Normalize `shifts`: `shift_norm = ((shifts % dim_size) +
   dim_size) % dim_size`. Handles negative shifts. `dim_size == 0`
   short-circuits to `shift_norm = 0`.
3. `shift_norm == 0` → return `input.clone()` (preserves the
   upstream grad_fn).
4. GPU f32 fast path: if `input.is_cuda()` and `T ==
   f32`, call `backend.roll_f32(handle, outer, dim_size, inner,
   shift_norm)`. Other GPU dtypes return `NotImplementedOnCuda`.
5. CPU path: call `roll_cpu_inner(&data, shape,
   shift_norm, dim)`.
6. Autograd: when `requires_grad() && is_grad_enabled`,
   attach `RollBackward { input, shifts, dim }` via
   `Tensor::from_operation`.

`roll_cpu_inner` in `ops/tensor_ops.rs` is the shared shift loop —
`pub(crate)` so `RollBackward::backward` can reuse it with negated
shift. Walks `outer × dim_size × inner` and writes `out[..., new_d,
...] = data[..., d, ...]` where `new_d = (d + shift_norm) %
dim_size`.

`cdist` in `ops/tensor_ops.rs`:
1. Identify batched (3-D) vs unbatched (2-D) input shape.
2. Validate feature dims match across `x1` / `x2`.
3. For each batch, walk `(i, j)` pairs, accumulating
   `|diff|^p` over the feature dim, then take `result^(1/p)`.

**Non-test consumers**:

- `crate::grad_fns::shape::RollBackward::backward` in
  `grad_fns/shape.rs` invokes `crate::ops::tensor_ops::roll_cpu_inner`
  for the backward shift. This is the REQ-7 production consumer of
  the shared inner kernel.
- `crate::grad_fns::shape::RollBackward` documentation at
  `grad_fns/shape.rs` references calling back into
  `crate::ops::tensor_ops::roll` for the in-graph backward path.
- Re-exported in `lib.rs` as
  `ferrotorch_core::{cdist, diag, diagflat, roll, tril, triu}`.

## Parity contract

`parity_ops = []` (no specific parity op declared). Numeric contract
is byte-for-byte parity with `torch.{triu, tril, diag, diagflat,
roll, cdist}`. Verified through unit tests + the autograd
correctness of `roll` (covered by `RollBackward` tests in
`RollBackward in grad_fns/shape.rs`).

## Verification

`cargo test -p ferrotorch-core --lib ops::tensor_ops` covers the
forward paths; `cargo test -p ferrotorch-core --lib
grad_fns::shape::tests::roll_*` covers the autograd path.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `triu` in `ops/tensor_ops.rs` (CPU + the `input.is_cuda()` GPU f32/f64 branch calling `backend.triu_f32`/`triu_f64`); GPU kernel `gpu_triu_f32`/`gpu_triu_f64` in `ferrotorch-gpu/src/triangular.rs`; non-test consumer: re-exported as `ferrotorch_core::triu` in `lib.rs` (boundary public API per goal.md S5); GPU consumer: the `is_cuda()` branch of `triu` in `ops/tensor_ops.rs` dispatches `CudaBackendImpl::triu_f32`/`triu_f64` in `ferrotorch-gpu/src/backend_impl.rs` |
| REQ-2 | SHIPPED | impl: `tril` in `ops/tensor_ops.rs` (CPU + the `input.is_cuda()` GPU f32/f64 branch calling `backend.tril_f32`/`tril_f64`); GPU kernel `gpu_tril_f32`/`gpu_tril_f64` in `ferrotorch-gpu/src/triangular.rs`; non-test consumer: re-exported as `ferrotorch_core::tril` in `lib.rs`; GPU consumer: the `is_cuda()` branch of `tril` in `ops/tensor_ops.rs` dispatches `CudaBackendImpl::tril_f32`/`tril_f64` in `ferrotorch-gpu/src/backend_impl.rs` |
| REQ-3 | SHIPPED | impl: `diag in ops/tensor_ops.rs` (CPU + the `input.is_cuda()` GPU f32/f64 branch calling `backend.diag_embed_f32`/`diag_extract_f32` etc.); GPU kernels `gpu_diag_embed_f32`/`gpu_diag_extract_f32` (+f64) in `ferrotorch-gpu/src/diag.rs`; non-test consumer: re-exported as `ferrotorch_core::diag in lib.rs`; GPU consumer: the `is_cuda()` branch of `diag` in `ops/tensor_ops.rs` dispatches `CudaBackendImpl::diag_embed_f32`/`diag_extract_f32` (+f64) in `ferrotorch-gpu/src/backend_impl.rs` |
| REQ-4 | SHIPPED | impl: `diagflat in ops/tensor_ops.rs` flattens via the device-aware `Tensor::view_reshape` (GPU-resident) then delegates to `diag`, so CUDA inputs ride the `diag` GPU fast path; non-test consumer: re-exported as `ferrotorch_core::diagflat in lib.rs` |
| REQ-5 | SHIPPED | impl: `roll in ops/tensor_ops.rs`; non-test consumer: re-exported as `ferrotorch_core::roll in lib.rs`. The autograd-attached `RollBackward` is the consumer of REQ-7's shared inner kernel |
| REQ-6 | SHIPPED | impl: `cdist in ops/tensor_ops.rs` (CPU + the `is_cuda()` GPU branch calling `backend.cdist_f32`/`cdist_f64`, gated by `gpu_dispatch::cdist_supported_f32`/`_f64`); GPU kernel `gpu_cdist_f32`/`gpu_cdist_f64` in `ferrotorch-gpu/src/distance.rs`; non-test consumer: re-exported as `ferrotorch_core::cdist in lib.rs`; GPU consumer: the `is_cuda()` branch of `cdist` in `ops/tensor_ops.rs` dispatches `CudaBackendImpl::cdist_f32`/`cdist_f64` in `ferrotorch-gpu/src/backend_impl.rs` |
| REQ-7 | SHIPPED | impl: `roll_cpu_inner in ops/tensor_ops.rs`; non-test consumer: `crate::grad_fns::shape::RollBackward::backward in grad_fns/shape.rs` invokes `crate::ops::tensor_ops::roll_cpu_inner` for the backward shift |
