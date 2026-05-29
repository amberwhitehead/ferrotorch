# GPU Triangular Masks (triu / tril)

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/cuda/TriangularOps.cu
  - aten/src/ATen/native/TriangularOps.cpp
-->

## Summary

`ferrotorch-gpu/src/triangular.rs` ships on-device PTX kernels for the
2-D triangular masks `torch.triu` and `torch.tril`. Each is a pure
per-element index-predicate copy: for an `[rows, cols]` C-contiguous
buffer the element at `(row, col)` is preserved when the predicate
holds and zeroed otherwise.

- **triu** keeps `(row, col)` when `col - row >= k` (i.e. `col >= row + k`).
- **tril** keeps `(row, col)` when `col - row <= k`.

This is exactly PyTorch's CUDA kernel predicate at
`aten/src/ATen/native/cuda/TriangularOps.cu:100`
(`mask = upper ? (col + i - row >= k) : (col + i - row <= k)`) and the
ferrotorch CPU path in `ops/tensor_ops.rs` (`triu`: `c >= r + diagonal`,
`tril`: `c <= r + diagonal`). Because the op only copies-or-zeros (no
arithmetic), the GPU result is **bit-for-bit identical** to both the CPU
ferrotorch path and `torch.{triu,tril}` — no float tolerance applies.

`k` (the diagonal offset) is signed (`i64` in PyTorch). The CUDA backend
passes it as a signed 32-bit value to the PTX; the comparison
`col - row` is done in signed 32-bit arithmetic so negative diagonals
work. Rows/cols are bounded by tensor dims that always fit in 32 bits
on a single device (PyTorch itself uses `int32_t` index math when
`canUse32BitIndexMath`, `TriangularOps.cu:125`).

## Requirements

- REQ-1: `gpu_triu_f32` / `gpu_tril_f32` — launch the f32 PTX mask
  kernel over a resident `CudaBuffer<f32>` of `rows*cols` elements,
  returning a fresh resident `CudaBuffer<f32>`. One thread per element;
  `out[t] = pred ? in[t] : 0.0`.
- REQ-2: `gpu_triu_f64` / `gpu_tril_f64` — f64 counterpart over a
  `CudaBuffer<f64>`.
- REQ-3: signed diagonal `k` — the predicate uses signed 32-bit
  `col - row` vs `k`, so `k < 0`, `k == 0`, `k > 0` all match
  `torch.{triu,tril}(diagonal=k)`.
- REQ-4: dispatch wiring — `GpuBackend::{triu_f32,tril_f32,triu_f64,
  tril_f64}` trait slots (default `NotImplementedOnCuda`); CUDA backend
  overrides them; `ops::tensor_ops::{triu,tril}` route CUDA inputs
  through the backend and keep the result GPU-resident (no host
  round-trip).

## Acceptance Criteria

- [x] AC-1: `cargo test -p ferrotorch-gpu --features cuda triangular`
  passes LIVE on the RTX 3090.
- [x] AC-2: `gpu_triu_f32` of a 3x3 with `k=0` zeros the strict
  lower-left and matches the CPU `ferrotorch_core::triu` element-for-
  element.
- [x] AC-3: `gpu_tril_f32` of a 3x3 with `k=0` zeros the strict
  upper-right and matches CPU `tril`.
- [x] AC-4: negative diagonal `k=-1` for triu and positive `k=1` for
  tril match the CPU path bit-for-bit.
- [x] AC-5: a CUDA tensor passed to `ops::tensor_ops::triu`/`tril`
  returns a tensor whose storage `is_cuda()` (NO `.cpu()` round trip)
  and whose `.cpu()` data equals the CPU reference.

## Architecture

`triangular.rs` is `#![cfg(feature = "cuda")]` (mirrors `reduce_arg.rs`).
Two PTX template constants (`TRIANGULAR_F32_PTX`, `TRIANGULAR_F64_PTX`)
carry one entry each: `triangular_f32_kernel` /
`triangular_f64_kernel` with the 6-arg ABI
`(in_ptr, out_ptr, rows, cols, k, op)` where `op` is `0` for triu and
`1` for tril, `k` is an `s32`. Thread `t in [0, rows*cols)` computes
`row = t / cols`, `col = t % cols`, evaluates `diff = col - row` in
signed 32-bit, and writes `out[t] = (op==0 ? diff>=k : diff<=k) ?
in[t] : 0`. `launch_triangular<V>` resolves the entry via
`module_cache::get_or_compile`, allocates the output via
`alloc_zeros_{f32,f64}`, and launches one 1-D grid. A zero-element
buffer short-circuits to an empty allocation.

The backend (`backend_impl.rs`) overrides `triu_f32`/`tril_f32`/
`triu_f64`/`tril_f64`, unwrapping the dtype-tagged handle to the right
`CudaBuffer`, calling the kernel, and re-wrapping. `ops::tensor_ops::
{triu,tril}` gain a CUDA branch (before the autograd CPU fall-through)
that dispatches on `is_f32`/`is_f64` and returns a GPU-resident result;
other GPU dtypes keep the `NotImplementedOnCuda` error.

**Non-test consumer**: `ferrotorch_core::ops::tensor_ops::triu` and
`::tril` (re-exported as `ferrotorch_core::{triu,tril}` at `lib.rs`)
call these backend methods for CUDA-resident inputs.

## Parity contract

`parity_ops = []`. Numeric contract is byte-for-byte parity with
`torch.{triu,tril}` and the ferrotorch CPU `triu`/`tril`. Verified by
the LIVE GPU-vs-CPU unit tests in `triangular.rs` and the
`tensor_ops.rs` CUDA dispatch test.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn gpu_triu_f32` / `gpu_tril_f32 in triangular.rs`; non-test consumer: `CudaBackendImpl::triu_f32`/`tril_f32 in backend_impl.rs` dispatched from `ops::tensor_ops::triu`/`tril` |
| REQ-2 | SHIPPED | impl: `pub fn gpu_triu_f64` / `gpu_tril_f64 in triangular.rs`; non-test consumer: `CudaBackendImpl::triu_f64`/`tril_f64 in backend_impl.rs` dispatched from `ops::tensor_ops::triu`/`tril` |
| REQ-3 | SHIPPED | impl: `setp.ge.s32` / `setp.le.s32 %diff, %k` in the PTX templates in `triangular.rs`; verified by the `triu_f32_negative_diag` / `tril_f32_positive_diag` unit tests |
| REQ-4 | SHIPPED | impl: `fn triu_f32`/`tril_f32`/`triu_f64`/`tril_f64 in backend_impl.rs`; non-test consumer: the `input.is_cuda()` branch of `triu`/`tril in ops/tensor_ops.rs` keeps the result GPU-resident |
