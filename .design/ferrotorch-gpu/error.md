# GpuError — error taxonomy for the CUDA backend

<!--
tier: 3-component
status: draft
baseline-pytorch: 2fa9c68b1 (working tree at /home/doll/pytorch)
upstream-paths:
  - aten/src/ATen/cuda/
  - aten/src/ATen/native/cuda/
  - c10/cuda/
  - torch/cuda/
-->

## Summary

`ferrotorch-gpu/src/error.rs` defines `GpuError`, the canonical fallible-
result type for everything in the crate, and `GpuResult<T>` =
`Result<T, GpuError>`. Variants enumerate every failure mode the crate
produces, with `#[from]` impls for the cudarc-side error types
(`DriverError`, `CublasError`, `CusolverError`, `CufftError`) so the
`?` operator works idiomatically inside the cudarc call chain. This
is the ferrotorch analog of PyTorch's `c10::Error` /
`c10::cuda::CUDAError` plus the `NotImplementedError: not implemented
for 'CUDA'` registry pattern (`aten/src/ATen/native/Registry.h`).

## Requirements

- REQ-1: `GpuError` is `#[derive(Debug, thiserror::Error)]` and
  `#[non_exhaustive]`. Each variant carries a structured `#[error]`
  format string (no `Box<dyn Error>` opaque wrapping). Per R-DEV-7,
  the Rust ecosystem analog (`thiserror`) replaces PyTorch's hand-rolled
  exception strings while preserving the upstream "what failed and
  why" contract.

- REQ-2: cudarc passthrough variants. `Driver(#[from] DriverError)`,
  `Blas(#[from] CublasError)`, `Solver(#[from] CusolverError)`, and
  `Fft(#[from] CufftError)` are each cfg-gated on `feature = "cuda"`
  and use thiserror's `#[from]` derivation so `?` lifts the cudarc
  error directly. Mirrors PyTorch's `C10_CUDA_CHECK` macro that
  converts the C++ exception into the typed C10 error.

- REQ-3: Structured semantic variants for ferrotorch-specific
  conditions: `InvalidDevice { ordinal, count }`, `DeviceMismatch
  { expected, got }`, `OutOfMemory { requested_bytes, free_bytes }`,
  `BudgetExceeded { requested_bytes, budget_bytes, used_bytes }`,
  `LengthMismatch { a, b }`, `ShapeMismatch { op, expected, got }`,
  `PtxCompileFailed { kernel, source }`, `Unsupported { op, dtype }`,
  `InvalidState { message }`. Each carries the typed payload upstream
  emits in the equivalent log line (`"requested {} bytes but only {}
  bytes free"` mirrors `c10::cuda::CUDAOutOfMemoryError`'s message
  body).

- REQ-4: Conditional `NoCudaFeature` variant under
  `#[cfg(not(feature = "cuda"))]`. The host-only build needs an
  error variant to return from every stub fn; this single variant
  serves that purpose. Mirrors PyTorch's
  `TORCH_INTERNAL_ASSERT(false, "CUDA not built")` behaviour but as a
  return value rather than a process abort.

- REQ-5: `GpuResult<T>` type alias = `Result<T, GpuError>`. Public
  alias so downstream crates write `GpuResult<CudaBuffer<f32>>` rather
  than `Result<CudaBuffer<f32>, GpuError>`. Matches the convention
  used throughout the ferrotorch ecosystem (`FerrotorchResult`,
  `CubeResult` are siblings).

## Acceptance Criteria

- [x] AC-1: `GpuError` is `Debug` and implements `std::error::Error`
  (via `thiserror::Error`).
- [x] AC-2: `?` operator lifts a `cudarc::driver::DriverError` into
  `GpuError::Driver` via the `#[from]` impl. Verified by every
  `dev.stream().alloc_zeros::<T>(count)?` call site in the crate.
- [x] AC-3: `GpuError::OutOfMemory { requested_bytes: 1024, free_bytes: 512 }`
  formats as a string containing both numbers and the words "out of
  memory" — verified by `memory_guard.rs::tests::gpu_error_out_of_memory_display`.
- [x] AC-4: Host-only build (`--no-default-features`) compiles — the
  cuda-specific variants disappear and `NoCudaFeature` appears.
- [x] AC-5: `#[non_exhaustive]` is respected: external callers must
  match with `..` rather than relying on the variant set being closed.

## Architecture

### Derives and metadata (REQ-1)

`pub enum GpuError in error.rs` at `error.rs` is annotated with
`#[derive(Debug, thiserror::Error)]` and `#[non_exhaustive]`. The
`thiserror` derivation auto-generates the `Display` impl from each
variant's `#[error]` attribute and the `Error::source` impl from the
`#[from]` / `#[source]` attributes. `#[non_exhaustive]` ensures future
variants can be added without breaking downstream `match` arms — a
hard requirement because the cudarc surface is still evolving.

### cudarc passthroughs (REQ-2)

Four cudarc forwarders, each `#[cfg(feature = "cuda")]`:
- `Driver(#[from] cudarc::driver::DriverError)` at `error.rs`.
- `Blas(#[from] cudarc::cublas::result::CublasError)` at `error.rs`.
- `Solver(#[from] cudarc::cusolver::result::CusolverError)` at
  `error.rs`.
- `Fft(#[from] cudarc::cufft::result::CufftError)` at `error.rs`.

Non-test production consumer: every kernel file uses `?` against
cudarc calls — e.g. `ferrotorch-gpu/src/transfer.rs`
`use crate::error::{GpuError, GpuResult};` followed by
`dev.stream().clone_htod(data)?` (`?` is the cudarc passthrough).

### Structured semantic variants (REQ-3)

Each non-cudarc failure mode is a distinct variant with named fields:
- `InvalidDevice { ordinal, count }` at `error.rs`: ordinal out
  of range; produced by `StreamPool::get_stream` when
  `device_ordinal >= MAX_DEVICES`
  (`stream.rs`).
- `DeviceMismatch { expected, got }` at `error.rs`: cross-device
  buffer use; produced by binary-op preconditions in `kernels.rs`.
- `OutOfMemory { requested_bytes, free_bytes }` at `error.rs`:
  produced by `memory_guard::MemoryGuard::wait_for_memory`
  (`memory_guard.rs`) and the OOM-detection helper.
- `BudgetExceeded { requested_bytes, budget_bytes, used_bytes }` at
  `error.rs`: produced by `memory_guard::check_budget`
  (`memory_guard.rs`).
- `LengthMismatch { a, b }` at `error.rs`: binary-op length
  validation.
- `ShapeMismatch { op, expected, got }` at `error.rs`: matmul/bmm
  shape validation; produced by `blas.rs`'s shape-check helpers.
- `PtxCompileFailed { kernel, source }` at `error.rs`: PTX
  JIT rejection with the cudarc driver error preserved as `#[source]`
  so `Error::source()` returns the underlying cause.
- `Unsupported { op, dtype }` at `error.rs`: mirrors PyTorch's
  `NotImplementedError: <op> not implemented for 'CUDA'` for unsupported
  (op, dtype) tuples.
- `InvalidState { message }` at `error.rs`: catch-all for
  invariant violations (sealed pool capture attempt, cuSOLVER negative
  info, null graph instantiation).

Non-test production consumer: `ferrotorch-llama/src/gpu.rs`
defines `map_gpu_err(e: ferrotorch_gpu::GpuError) -> FerrotorchError`
that match-arms on every variant to map into the cross-crate
`FerrotorchError`. The complete match coverage in that function
demonstrates the variant set is the contract.

### Host-only stub variant (REQ-4)

`#[cfg(not(feature = "cuda"))] NoCudaFeature` at `error.rs`. The
stub `GpuDevice::new`, `CudaAllocator::alloc_zeros`,
`MemoryGuard::safe_alloc`, etc. all return this variant when the
`cuda` feature is off. The variant is gated to the off-feature build
specifically so it doesn't pollute the on-feature error space (a
caller compiling with `--features cuda` shouldn't match against
`NoCudaFeature` — it can never be produced).

Non-test production consumer: `ferrotorch-data/src/transforms.rs`
calls `ferrotorch_gpu::init_cuda_backend()` and falls back to CPU
when the result is `Err`. Under `--no-default-features` the error
is `NoCudaFeature`; under `--features cuda` it can be any cuda-side
variant.

### Type alias (REQ-5)

`pub type GpuResult<T> = Result<T, GpuError>` at `error.rs`. Used
literally everywhere in the crate as the return type for fallible
fns.

Non-test production consumer: `ferrotorch-gpu/src/buffer.rs`,
`device.rs`, `allocator.rs`, `memory_guard.rs`, etc. all
return `GpuResult<T>`.

## Parity contract

`parity_ops = []`. `GpuError` is INFRASTRUCTURE; no parity-sweep op
verifies an error type directly. The contract is enforced
structurally: every fallible kernel uses `GpuResult<T>` and every
caller `match`-arms against `GpuError` to map to `FerrotorchError`.
The cross-crate map in `ferrotorch-llama/src/gpu.rs` is the
audit of completeness; adding a `GpuError` variant breaks the
exhaustiveness check there and forces an update at the consumer.

Edge cases handled:
- A cudarc `DriverError` with the string `"CUDA_ERROR_OUT_OF_MEMORY"`
  is recognized as OOM by `memory_guard::is_oom`
  (`memory_guard.rs`) even when it arrives as
  `GpuError::Driver(...)` rather than `GpuError::OutOfMemory`.
- `PtxCompileFailed` preserves the underlying cudarc `DriverError`
  via `#[source]` so `e.source()` returns the JIT rejection diagnostic.
- `Unsupported { op: "argmax", dtype: "BFloat16" }` formats as
  `"argmax not implemented for 'BFloat16' on CUDA"` — pinned by the
  inline `#[error(...)]` format string at `error.rs`.

## Verification

Tests in `memory_guard.rs::tests` (which transitively use `GpuError`):
- `gpu_error_out_of_memory_display` at `memory_guard.rs`.
- `gpu_error_budget_exceeded_display` at `memory_guard.rs`.

The `?` operator coverage is verified by every cudarc call site in
the crate compiling — a missing `#[from]` impl would surface as a
compile error.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-gpu --features cuda --lib memory_guard::tests::gpu_error 2>&1 | tail -3
cargo build -p ferrotorch-gpu --no-default-features 2>&1 | tail -3
```

Expected: `0 failed`; host-only build succeeds.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `#[derive(Debug, thiserror::Error)] #[non_exhaustive] pub enum GpuError in error.rs` at `error.rs`. Non-test production consumer: `ferrotorch-llama/src/gpu.rs` match-arms on the variant set. |
| REQ-2 | SHIPPED | impl: `Driver(#[from] DriverError)` at `error.rs`, `Blas(#[from] CublasError)` at `error.rs`, `Solver(#[from] CusolverError)` at `error.rs`, `Fft(#[from] CufftError)` at `error.rs`. Non-test production consumer: `ferrotorch-gpu/src/transfer.rs` and every kernel module use `?` against cudarc calls; the `From` impl is the lifting machinery. |
| REQ-3 | SHIPPED | impl: nine structured semantic variants at `error.rs`. Non-test production consumer: `ferrotorch-gpu/src/stream.rs` produces `InvalidDevice`; `memory_guard.rs` produces `BudgetExceeded`; `memory_guard.rs` produces `OutOfMemory`; `graph.rs` produces `InvalidState`. |
| REQ-4 | SHIPPED | impl: `#[cfg(not(feature = "cuda"))] NoCudaFeature` at `error.rs`. Non-test production consumer: `ferrotorch-data/src/transforms.rs` receives the error from `init_cuda_backend()` and falls back to CPU. |
| REQ-5 | SHIPPED | impl: `pub type GpuResult<T> = Result<T, GpuError>` at `error.rs`. Non-test production consumer: every public fn in `device.rs`, `allocator.rs`, `memory_guard.rs`, `transfer.rs`, etc. returns `GpuResult<T>`. |
