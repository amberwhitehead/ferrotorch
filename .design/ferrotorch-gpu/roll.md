# GPU roll forward kernel (f32 / f64)

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/cuda/TensorTransformations.cu
  - aten/src/ATen/native/TensorTransformations.cpp
-->

## Summary

`ferrotorch-gpu/src/roll.rs` implements the GPU forward path for
`torch.roll(input, shifts, dim)` as a single-axis cyclic shift on a
contiguous f32 buffer. The kernel is hand-written PTX loaded through
the global `module_cache::get_or_compile` cache. Mirrors upstream
`roll_cuda_kernel` in `aten/src/ATen/native/cuda/TensorTransformations.cu:84`.

## Requirements

- REQ-1: Public single-axis f32 entry point `pub fn gpu_roll_f32` taking
  `(input, outer, dim_size, inner, shift_norm, device)` and returning
  `CudaBuffer<f32>` of the same length. Mirrors the
  `(outer, dim_size, inner)` factorisation used by the CPU
  `roll_cpu_inner` helper so the address map is bit-exact across
  backends.
- REQ-2: Hand-written PTX kernel `roll_f32_kernel` (`ROLL_F32_PTX`)
  with the documented ABI `(in_ptr, out_ptr, outer, dim_size, inner,
  shift_norm, total)`. The kernel computes
  `k_src = (k_new + dim_size - shift_norm) mod dim_size` and writes
  `out[(o*dim_size + k_new)*inner + i] = in[(o*dim_size + k_src)*inner + i]`
  per thread.
- REQ-3: Caller-side normalisation precondition: the function takes a
  non-negative `shift_norm` with `0 <= shift_norm < dim_size`. The
  consumer is responsible for normalising the original signed shift
  via `((shifts % n) + n) % n`. Out-of-range or `dim_size == 0` are
  rejected with `GpuError::ShapeMismatch`.
- REQ-4: Buffer device-residency and length validation: `input.len()`
  must equal `outer * dim_size * inner`, all on the same device, with
  `total <= u32::MAX` for the kernel's u32 index arithmetic.
- REQ-5: Non-test production consumer wiring: ferrotorch-core dispatches
  GPU rolls through the `CudaBackendImpl::roll_f32` trait method which
  calls into this kernel.
- REQ-6: f64 roll. `pub fn gpu_roll_f64` + `pub(crate) const ROLL_F64_PTX`
  are the f64 siblings of REQ-1/REQ-2 — identical `(outer, dim_size, inner)`
  factorisation and identical index map, differing only in the element
  register type (`.f64`) and the 8-byte element stride (`shl.b64 .., 3`).
  Because `roll` performs no arithmetic on the loaded values (a pure
  relocating gather), the f64 path is bit-exact (no transcendentals). The
  non-test production consumer is `CudaBackendImpl::roll_f64`, dispatched from
  `ferrotorch_core::ops::tensor_ops::roll`'s f64 CUDA branch (#1545 / sub
  #1535).

## Acceptance Criteria

- [x] AC-1: `pub fn gpu_roll_f32` exists with the documented signature.
- [x] AC-2: `pub(crate) const ROLL_F32_PTX` carries the PTX ABI matching
  the upstream `roll_cuda_kernel`'s `(linear_index, total_dims, sizes,
  strides, shifts)` semantics (factorised to a single axis).
- [x] AC-3: 8 unit tests in `mod tests` exercise positive shift, 2-D
  inner-axis, 2-D outer-axis, negative-shift normalisation,
  multi-axis composition, zero-shift identity, and the two
  precondition rejections.
- [x] AC-4: Non-test consumer exists at
  `ferrotorch-gpu/src/backend_impl.rs:3618` (`CudaBackendImpl::roll_f32`
  GPU trait method).
- [x] AC-5: SAFETY comment on the single `unsafe { ... .launch(cfg) }`
  block documents the buffer lengths, grid sizing, and the in-bounds
  guarantee on the kernel index map.
- [x] AC-6: `pub fn gpu_roll_f64` exists with the f64-typed signature;
  3 f64 unit tests (`roll_f64_1d_positive_shift_matches_cpu`,
  `roll_f64_2d_inner_axis_matches_cpu`, `roll_f64_rejects_shift_at_dim_size`)
  plus the live-GPU consumer tests `roll_f64_1d_positive_and_negative_match_torch`
  / `roll_f64_2d_per_dim_matches_torch` in
  `ferrotorch-gpu/tests/divergence_roll_f64_unique_consecutive_gpu.rs` assert
  the f64 result `is_cuda()` and matches `torch.roll` (and the CPU path).

## Architecture

`pub fn gpu_roll_f32 in roll.rs` does:

1. Validates buffer device-residency and `input.len() == outer * dim_size * inner`.
2. Validates `dim_size > 0` and `shift_norm < dim_size`.
3. Short-circuits the empty (`total == 0`) case by returning an empty
   buffer without launching.
4. Resolves `roll_f32_kernel` via `crate::module_cache::get_or_compile`.
5. Allocates the f32 output via `alloc_zeros_f32(total, device)`.
6. Launches with `block_dim = 256`, `grid_dim = ceil(total / 256)`,
   shared mem = 0. Each thread does one output write.
7. Returns the `CudaBuffer<f32>` output.

The PTX kernel itself (`pub(crate) const ROLL_F32_PTX in roll.rs`)
decomposes the flat output index `out_idx` into `(o, k_new, i)`,
computes `k_src` via the documented modulo, builds the source flat
index, then issues one `ld.global.f32` / `st.global.f32` pair.

Non-test production consumer: `backend_impl.rs` —
`CudaBackendImpl::roll_f32` is the GPU-side trait method that
ferrotorch-core's `ops::tensor_ops::roll` dispatches into when the
tensor is CUDA-resident. The backward path
(`grad_fns::shape::RollBackward`) re-enters this kernel with
`-shifts` (after CPU-side normalisation).

Single-axis design rationale: PyTorch's `torch.roll` accepts multi-axis
`shifts: IntList` and `dims: IntList`, but the upstream CUDA kernel
itself works one dim at a time — multi-axis is implemented by repeated
single-axis calls in the dispatcher. ferrotorch matches that lower
layer 1:1; the per-tensor multi-axis loop is in ferrotorch-core's
`ops::tensor_ops::roll`.

## Parity contract

`parity_ops = []` for this route. The `roll` op's parity is enforced
at the ferrotorch-core layer where the multi-axis dispatcher lives;
this kernel is a single-axis primitive that lower-layer code consumes.

Edge cases preserved:

- **Empty tensor** (`outer * dim_size * inner == 0` with `dim_size > 0`):
  returns a length-0 buffer without launching.
- **Zero shift**: `shift_norm == 0` produces an identity copy
  (verified by `roll_zero_shift_is_identity`).
- **Negative shift after normalisation**: the caller normalises
  `((shifts % n) + n) % n`; the kernel sees only the non-negative
  result (verified by `roll_negative_shift_via_normalization_matches_cpu`).
- **Wrap-around correctness**: `k_src = (k_new + dim_size - shift_norm)
  mod dim_size` is computed in u32 with `rem.u32`, safe against
  `shift_norm == dim_size` defensively (rejected at the wrapper anyway).
- **u32 index overflow**: `total > u32::MAX` is rejected with
  `ShapeMismatch` before launch.
- **NaN / Inf data**: bit-exact load/store — no arithmetic. Pattern
  preservation matches a memcpy.

## Verification

Unit tests in `ferrotorch-gpu/src/roll.rs` `mod tests` (11 tests):

- `roll_1d_positive_shift_matches_cpu`
- `roll_2d_inner_axis_matches_cpu`
- `roll_2d_outer_axis_matches_cpu`
- `roll_negative_shift_via_normalization_matches_cpu`
- `roll_3d_middle_axis_then_inner_axis_matches_cpu_composed`
- `roll_zero_shift_is_identity`
- `roll_rejects_shift_at_dim_size`
- `roll_rejects_wrong_length`
- `roll_f64_1d_positive_shift_matches_cpu`
- `roll_f64_2d_inner_axis_matches_cpu`
- `roll_f64_rejects_shift_at_dim_size`

Live-GPU consumer-path tests in
`ferrotorch-gpu/tests/divergence_roll_f64_unique_consecutive_gpu.rs` assert
the f64 `roll` result `is_cuda()` and matches `torch.roll` / the CPU path
(`roll_f64_1d_positive_and_negative_match_torch`,
`roll_f64_2d_per_dim_matches_torch`, plus an f32 no-regression guard).

Each test that runs on hardware uses the `match GpuDevice::new(0)`
graceful-skip pattern: on a host without CUDA the test returns
early rather than failing.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-gpu --features cuda roll:: 2>&1 | tail -3
```

Expected: ≥ 1 `test result: ok` line (or graceful skip on hosts
without CUDA).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn gpu_roll_f32 in ferrotorch-gpu/src/roll.rs` (line 176) mirrors upstream `roll_cuda_kernel` at `aten/src/ATen/native/cuda/TensorTransformations.cu:84`; non-test consumer: `CudaBackendImpl::roll_f32` body at `ferrotorch-gpu/src/backend_impl.rs:3618` invokes `crate::roll::gpu_roll_f32`. |
| REQ-2 | SHIPPED | impl: `pub(crate) const ROLL_F32_PTX in roll.rs` (line 63) carries the documented ABI; the launch site at line 291 binds args in the matching order. |
| REQ-3 | SHIPPED | impl: precondition checks at `roll.rs` lines 200-213 (`dim_size == 0` rejection, `shift_norm >= dim_size` rejection); negative-shift normalisation contract documented at lines 162-165 and exercised by `roll_negative_shift_via_normalization_matches_cpu`. |
| REQ-4 | SHIPPED | impl: device-ordinal check at `roll in roll.rs`, length check at line 192, u32-overflow check at line 221. |
| REQ-5 | SHIPPED | impl: `pub use roll::gpu_roll_f32` at `backend_impl in ferrotorch-gpu/src/lib.rs`; non-test consumer: `gpu_roll_f32 in backend_impl.rs` (the trait method `CudaBackendImpl::roll_f32` registered via `init_cuda_backend` is what ferrotorch-core dispatches GPU rolls through). |
| REQ-6 | SHIPPED | impl: `pub fn gpu_roll_f64` + `pub(crate) const ROLL_F64_PTX in ferrotorch-gpu/src/roll.rs` (f64 sibling of REQ-1/REQ-2, 8-byte `shl.b64 .., 3` stride); re-export `pub use roll::{gpu_roll_f32, gpu_roll_f64} in ferrotorch-gpu/src/lib.rs`; non-test consumer: `CudaBackendImpl::roll_f64 in ferrotorch-gpu/src/backend_impl.rs` invokes `crate::roll::gpu_roll_f64`, dispatched from the `is_f64::<T>()` arm of `ferrotorch_core::ops::tensor_ops::roll`'s CUDA branch (`roll in ferrotorch-core/src/ops/tensor_ops.rs`). |
