# CUDA implementation of the GpuBackend trait

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/cuda/CUDAContext.cpp
  - aten/src/ATen/cuda/CUDAContextLight.h
  - aten/src/ATen/native/cuda/
  - c10/cuda/CUDAStream.cpp
  - torch/cuda/__init__.py
-->

## Summary

`ferrotorch-gpu/src/backend_impl.rs` is the CUDA implementation of
ferrotorch-core's `GpuBackend` dispatch trait. It bridges the
type-erased `GpuBufferHandle` surface to the typed `CudaBuffer<T>`
storage and forwards every trait method to the corresponding
function in `crate::kernels` / `crate::blas` / `crate::transfer` /
`crate::rng` / `crate::masked_kernels` / `crate::gather_int` /
`crate::reduce_arg` / `crate::roll` / etc. The crate also exposes
the singleton lifecycle (`init_cuda_backend`, `get_cuda_device`)
that ferrotorch-core's `gpu_dispatch::register_gpu_backend` consumes.
Mirrors PyTorch's `torch.cuda.init()` + `at::cuda::CUDAContext`
binding role — the file is the "what does CUDA do when you call
`.cuda()`?" implementation.

## Requirements

- REQ-1: `pub struct CudaBackendImpl` carrying
  `devices: Vec<Arc<GpuDevice>>` + lazy cuSPARSE / cuSPARSELt
  handles. Construction via `pub fn new()`. Implements the
  `ferrotorch_core::gpu_dispatch::GpuBackend` trait.
- REQ-2: Type-erasure bridge — `wrap_buffer_*` /
  `unwrap_buffer_*` / `wrap_slice_*` helpers that round-trip
  between `CudaBuffer<T>` / `CudaSlice<T>` and the type-erased
  `GpuBufferHandle` with `DType`-tag validation. One pair per
  supported value dtype (`f32`, `f64`, `i32`, `i64`, `u8`-as-bool,
  `u16`-as-bf16, `u16`-as-f16).
- REQ-3: Lazy cuSPARSE / cuSPARSELt handle caching via
  `OnceLock<CusparseHandle>` / `OnceLock<CusparseLtHandle>` —
  one handle per backend instance, lazily constructed on first
  SpMM / 2:4-sparse matmul. The handle's stream is rebound per
  call via `cusparseSetStream`.
- REQ-4: `pub fn init_cuda_backend() -> FerrotorchResult<()>` —
  idempotent global init that creates a `CudaBackendImpl` and
  registers it via `ferrotorch_core::gpu_dispatch::register_gpu_backend`.
- REQ-5: `pub fn get_cuda_device() -> FerrotorchResult<Arc<GpuDevice>>`
  — accessor that downcasts the registered global `&dyn GpuBackend`
  back to `&CudaBackendImpl` to retrieve the shared `GpuDevice`.
  Ensures consumers reuse the same CUDA context + module cache
  rather than creating a second context via `GpuDevice::new(0)`.
- REQ-6: Implementation of every trait method in the
  `GpuBackend` trait (348 methods covering: elementwise binary /
  unary / broadcast / reductions / indexing / norms / convs /
  matmul / scan / strided ops / RNG / sparse / pooling / activations
  / dtype-conversion / casts / inplace / autograd-bw helpers). Each
  method delegates to the appropriate `crate::kernels::*` or
  sibling-module function.
- REQ-7: Per-op error mapping via `fn map_gpu_err(e:
  GpuError) -> FerrotorchError` — converts the GPU-specific error
  taxonomy (`PtxCompileFailed`, `Driver`, `ShapeMismatch`,
  `DeviceMismatch`, etc.) into the public `FerrotorchError` shape
  that ferrotorch-core surfaces to user code.
- REQ-8: `gather_or_select` boundary method — unified
  `index_select` / `gather` dispatcher (`is_gather: bool` toggle)
  that handles the 5 value dtypes × 2 index dtypes × 2 ops matrix
  via the `crate::gather_int as gi` sibling module.

## Acceptance Criteria

- [x] AC-1: `pub struct CudaBackendImpl` at line 36 with the
  documented fields; `pub fn new` at line 60.
- [x] AC-2: 12 `wrap_*` / `unwrap_*` helper methods on
  `impl CudaBackendImpl` covering the 7 supported dtypes (some
  dtypes have separate `wrap_buffer` / `wrap_slice` and
  `unwrap_buffer` / `unwrap_buffer_mut` shapes).
- [x] AC-3: `cusparse_handle: OnceLock<CusparseHandle>` field at
  line 44 and `cusparselt_handle: OnceLock<CusparseLtHandle>` at
  line 50; lazy-init methods `cusparse` (line 95) / `cusparselt`
  (line 78).
- [x] AC-4: `pub fn init_cuda_backend` at line 7018 with
  idempotent registration; `pub fn get_cuda_device` at line 6992.
- [x] AC-5: `impl GpuBackend for CudaBackendImpl` at line 582 with
  376 method bodies (matches the trait's 348-method surface plus
  some helper / default-override methods).
- [x] AC-6: `fn map_gpu_err` at line 571 covering the
  `GpuError` variant set.
- [x] AC-7: `fn gather_or_select` at line 442 dispatching the 20
  (vty, ity, op) cells through `gi::gather_*` / `gi::isel_*`.
- [x] AC-8: Non-test production consumer — the `init_cuda_backend`
  symbol is called from `ferrotorch/examples/ferrotorch_bench.rs:239`
  (a production example binary), `ferrotorch-data/src/transforms.rs`,
  `ferrotorch-optim/src/{lbfgs,muon,adagrad}.rs`, and
  `ferrotorch-distributions/src/fallback.rs` — all production code
  paths inside the workspace.

## Architecture

### Type-erasure bridge (REQ-2)

`GpuBufferHandle` (defined in `ferrotorch-core::gpu_dispatch`) is a
type-erased `Box<dyn Any + Send + Sync>` carrying:

- The boxed concrete storage (`CudaBuffer<T>` or `CudaSlice<T>`).
- A `DType` tag — the AUTHORITATIVE element-type marker.
- A device ordinal.
- A length (element count).

The `unwrap_buffer*` family validates `dtype` then `downcast_ref` —
the `dtype` check is the fast, authoritative gate (PyTorch parity);
`downcast_ref` is the safety net that catches a tag/storage mismatch.
Stops an `i32` handle (also 4 bytes) being silently read as `f32`.

The 12 helpers cover:
- `wrap_buffer / unwrap_buffer` (f32)
- `wrap_buffer_f64 / unwrap_buffer_f64 / unwrap_buffer_f64_mut`
- `wrap_buffer_bf16 / unwrap_buffer_bf16` (u16-backed)
- `wrap_buffer_f16 / unwrap_buffer_f16` (u16-backed)
- `wrap_buffer_i32 / unwrap_buffer_i32`
- `wrap_buffer_i64 / unwrap_buffer_i64`
- `wrap_buffer_bool / unwrap_buffer_bool` (u8-backed)
- `wrap_slice_*` variants for the `CudaSlice<T>` shape returned by
  the new kernels that return `CudaSlice` directly (rng, gather_int,
  reduce_arg, masked_kernels).

### Lifecycle (REQ-4, REQ-5)

`init_cuda_backend` at line 7018:
1. Checks `gpu_dispatch::has_gpu_backend()` — idempotent: returns
   `Ok(())` if already registered.
2. Constructs `CudaBackendImpl::new()` (creates `GpuDevice::new(0)`
   and the empty handle caches).
3. Registers via `register_gpu_backend(Box::new(backend))`.

`get_cuda_device` at line 6992:
1. Calls `gpu_dispatch::gpu_backend()` to fetch the global
   `&dyn GpuBackend`.
2. Downcasts to `&CudaBackendImpl` via `Any::downcast_ref` (the
   `as_any` trait method enables this).
3. Returns `Arc::clone(cuda_backend.default_device()?)`.

This pattern is critical: creating a second `GpuDevice::new(0)`
elsewhere would create a parallel CUDA context with its own module
cache — kernels compiled in one context can't be launched on the
other. `get_cuda_device` is the only safe accessor for the shared
device handle.

### Trait method dispatch (REQ-6)

The `impl GpuBackend for CudaBackendImpl` block at line 582 is
the bulk of the file (~6400 lines). Each method follows the same
shape:

```rust
fn add_f32(&self, a: &GpuBufferHandle, b: &GpuBufferHandle)
    -> FerrotorchResult<GpuBufferHandle> {
    let dev = self.device(a.device_ordinal())?;
    let a_buf = Self::unwrap_buffer(a)?;
    let b_buf = Self::unwrap_buffer(b)?;
    let result = crate::kernels::gpu_add(a_buf, b_buf, dev)
        .map_err(Self::map_gpu_err)?;
    Ok(Self::wrap_buffer(result, a.device_ordinal()))
}
```

The 348 trait methods cover the entire GPU op surface ferrotorch
needs:

- **Elementwise binary / unary / broadcast** (f32, f64, bf16):
  add, sub, mul, div, neg, relu, exp, log, sqrt, pow, abs,
  sigmoid, tanh, etc.
- **Activations**: gelu, gelu_tanh, gelu_erf, silu, elu, mish,
  clamp + backward variants for each.
- **Normalisation**: layernorm, layernorm_backward, rmsnorm,
  rmsnorm_backward (f32 + f64).
- **Reductions**: sum, prod, min, max, masked_min/max, sum_axis,
  cumsum, cumprod, cummax, cummin, logcumsumexp.
- **Argreduce**: argmax / argmin for f32, f64, f16, bf16, i32, i64.
- **Indexing**: index_select, gather (via `gather_or_select`),
  scatter_add, strided_split, strided_cat, strided_copy,
  strided_scatter.
- **Mask ops**: masked_fill, masked_zero, where_*, masked_select,
  masked_scatter (via `crate::masked_kernels`).
- **Matmul / BLAS**: matmul_f32, matmul_f64, matmul_bf16_*,
  matmul_f16_f32, bmm_f32; conv2d_f32; flash_attention_f32/f64.
- **Softmax family**: softmax, log_softmax, dropout, dropout_philox,
  + backward variants.
- **Shape / transpose / permute**: transpose_2d, permute_0213,
  embedding lookup, slice read/write (KV cache).
- **RNG**: dropout_philox routes through `crate::rng::cuda_rng_manager`.
- **Sparse / pooling**: spmm via cuSPARSE handle, maxpool2d /
  avgpool2d, batchnorm2d.
- **Optimizer steps**: fused Adam step, fused GRU cell.
- **Roll**: `roll_f32` body at line 3618.

### Error mapping (REQ-7)

`fn map_gpu_err` at line 571 is the single conversion point:

```rust
fn map_gpu_err(e: crate::error::GpuError) -> FerrotorchError {
    match e {
        GpuError::PtxCompileFailed { kernel, source } => ...,
        GpuError::Driver(d) => ...,
        GpuError::ShapeMismatch { op, expected, got } => ...,
        GpuError::DeviceMismatch { expected, got } => ...,
        GpuError::OutOfMemory { bytes } => ...,
        GpuError::InvalidState { message } => ...,
        ...
    }
}
```

Every trait method's `.map_err(Self::map_gpu_err)?` call funnels
through here, so a single change to the error surface is one site.

### `gather_or_select` (REQ-8)

Lines 442-568 implement the unified dispatcher for `index_select`
and `gather`. The boundary takes a single `is_gather: bool`
toggle and `match`es on `src.dtype()` × `index.dtype()` to pick
the right `gi::gather_<vty>_<ity>` or `gi::isel_<vty>_<ity>`
function from `crate::gather_int`. The `run!` declarative macro
expands the 4-arm `(is_gather, i32idx) ∈ {(true, true), (true, false),
(false, true), (false, false)}` shape, then wraps the result via the
appropriate `wrap_slice_*` / `wrap_buffer_*` helper.

## Parity contract

`parity_ops = []` for this route. Backend_impl is the dispatch layer
— it routes each trait method to a per-op kernel, and per-op parity
is tracked by the parity-sweep entries on the kernel-owning files
(arithmetic, activation, reductions, etc., in ferrotorch-core).

Edge cases preserved across the dispatch layer:

- **DType tag is authoritative**: every `unwrap_buffer_*` validates
  `handle.dtype()` before downcast, preventing silent type confusion.
- **Device ordinal validation**: every method that takes two
  handles validates they share a device. Cross-device ops
  (currently single-device only) return `DeviceMismatch`.
- **Lazy handle initialisation**: cuSPARSE / cuSPARSELt handles are
  `OnceLock`-deferred to first use. A process that never touches
  sparse ops doesn't pay the init cost.
- **`init_cuda_backend` is idempotent**: safe to call from multiple
  test paths or from multiple example binaries; the second call is
  a no-op `Ok(())`.
- **Cross-context safety**: `get_cuda_device` enforces a single
  shared `GpuDevice`/context; creating a second `GpuDevice::new(0)`
  is documented as breaking the module-cache invariant.

## Verification

Unit tests in `ferrotorch-gpu/src/backend_impl.rs` `mod tests` (9
tests) cover: backend init + idempotency, gpu_backend roundtrip,
CPU→GPU→CPU round-trip via the trait surface, dtype-tag mismatch
rejection, device-ordinal mismatch rejection.

Cross-cutting integration is exercised at the workspace level
through `ferrotorch-core/tests/`, `ferrotorch-nn/tests/`,
`ferrotorch-gpu/tests/conformance_gpu_backend.rs`, and
`ferrotorch-gpu/tests/conformance_gpu_lifecycle.rs`. Every
"GPU-aware" test path in those crates depends on `init_cuda_backend`
being a no-op-on-repeat call.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-gpu --features cuda backend_impl:: 2>&1 | tail -3
```

Expected: ≥ 1 `test result: ok` line.

The full GPU test suite:

```bash
cargo test -p ferrotorch-gpu --features cuda 2>&1 | tail -5
```

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct CudaBackendImpl in ferrotorch-gpu/src/backend_impl.rs` (line 36) with `pub fn new` (line 60); non-test consumer: `init_cuda_backend` (line 7018) constructs it; `ferrotorch/examples/ferrotorch_bench.rs:239` calls `ferrotorch_gpu::init_cuda_backend()`. |
| REQ-2 | SHIPPED | impl: 12 `wrap_*` / `unwrap_*` helpers on `impl CudaBackendImpl` at lines 127-441; non-test consumer: 348+ call sites across the `impl GpuBackend` block (every trait method body). |
| REQ-3 | SHIPPED | impl: `cusparse_handle: OnceLock<CusparseHandle>` at `backend_impl.rs`, `cusparselt_handle: OnceLock<CusparseLtHandle>` at line 50; lazy methods at lines 78, 95; non-test consumer: the SpMM / 2:4-sparse matmul trait method bodies in the same `impl GpuBackend` block. |
| REQ-4 | SHIPPED | impl: `pub fn init_cuda_backend in backend_impl.rs` (line 7018); non-test consumer: `ferrotorch/examples/ferrotorch_bench.rs:239`. Re-exported at `lib.rs:191`. |
| REQ-5 | SHIPPED | impl: `pub fn get_cuda_device in backend_impl.rs` (line 6992); non-test consumer: re-exported at `lib.rs:191`. The downcast-via-`as_any` pattern is the single canonical accessor for the shared `GpuDevice` from any registered-backend caller. |
| REQ-6 | SHIPPED | impl: `impl GpuBackend for CudaBackendImpl` at `gpu_backend in backend_impl.rs` with 348+ method bodies forwarding to `crate::kernels::*` / siblings; non-test consumer: ferrotorch-core's `gpu_dispatch::gpu_backend()` returns the registered global `&dyn GpuBackend`, and every CUDA-aware tensor op in ferrotorch-core (Tensor::add, matmul, softmax, etc.) dispatches through it when the input is GPU-resident. |
| REQ-7 | SHIPPED | impl: `fn map_gpu_err in backend_impl.rs` (line 571); non-test consumer: every `.map_err(Self::map_gpu_err)?` call site in the trait-method bodies (hundreds of sites). |
| REQ-8 | SHIPPED | impl: `fn gather_or_select in backend_impl.rs` (line 442); non-test consumer: it IS itself the `GpuBackend::gather_or_select` trait method body — ferrotorch-core's `Tensor::index_select` and `Tensor::gather` dispatch through it via the trait when the source tensor is CUDA-resident. |
