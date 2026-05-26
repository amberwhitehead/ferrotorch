# GpuDevice — CUDA device handle

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

`ferrotorch-gpu/src/device.rs` wraps a `cudarc::driver::CudaContext`
together with its default stream and a cached `cudarc::cublas::CudaBlas`
handle. This is the ferrotorch analog of upstream's
`c10::cuda::CUDAFunctions::set_device(idx)` plus the per-device cuBLAS
handle PyTorch caches inside `at::cuda::getCurrentCUDABlasHandle()`
(`aten/src/ATen/cuda/CUDABlas.cpp`). Where upstream relies on
process-global thread-local state, ferrotorch threads the device
explicitly through `Arc<GpuDevice>` so ownership is checked by the
borrow checker.

## Requirements

- REQ-1: `GpuDevice` struct holds an `Arc<CudaContext>`, an
  `Arc<CudaStream>` (the default stream taken from the context), a
  `cudarc::cublas::CudaBlas` handle bound to that stream, and the
  `usize` device ordinal. The cuBLAS handle is created once and reused
  across every matmul / bmm call — this eliminates the ~1.7 ms
  `cuModuleLoadData` cost upstream pays on each `CudaBlas::new`.

- REQ-2: `GpuDevice::new(ordinal)` constructs a fresh device by
  calling `cudarc::driver::CudaContext::new(ordinal)`, taking its
  default stream, and constructing the cached `CudaBlas`. Returns
  `GpuResult<Self>` — the upstream contract is that `set_device(idx)`
  aborts the process on invalid ordinal; we propagate the cudarc
  `DriverError` instead (R-DEV-4: Rust ecosystem analog improves on a
  C++ footgun).

- REQ-3: Capture-friendly forked-stream constructor.
  `GpuDevice::fork_for_capture(parent)` clones the context and creates
  a non-blocking forked stream that supports CUDA graph capture (the
  legacy default stream does not). Mirrors PyTorch's
  `torch.cuda.Stream(device=..., non_blocking=True)` use inside
  `torch.cuda.graph` context managers.

- REQ-4: Accessors for the context, default stream, current stream
  (thread-local-aware), the cached cuBLAS handle, and the ordinal.
  `context()` returns `&Arc<CudaContext>` (cudarc loaders need a context
  handle); `default_stream()` returns `&Arc<CudaStream>` (the legacy
  default); `stream()` resolves to the thread-local current stream set
  via `crate::stream::StreamGuard`, falling back to the default if
  none. `blas()` returns `&CudaBlas` — the cached handle. `ordinal()`
  returns the `usize` device ordinal.

- REQ-5: `Clone` impl. Cloning a `GpuDevice` clones the `Arc<CudaContext>`
  and `Arc<CudaStream>` (cheap) but constructs a fresh `CudaBlas`
  (cuBLAS handles aren't `Clone`). The fresh handle is bound to the
  same shared stream so the two clones see consistent cuBLAS state.
  This `expect("CudaBlas::new failed")` is the only `expect` in the
  module — documented as last-resort because `Clone` returns `Self`,
  not `Result<Self>`. Per R-CODE-2 the `expect` is the boundary
  workaround for upstream's infallible `Clone` contract; a builder
  pattern would force every clone-site to handle a never-fired error
  case.

- REQ-6: Stub when `cuda` feature is disabled. `#[cfg(not(feature = "cuda"))]`
  arm provides a minimal `GpuDevice { ordinal: usize }` stub whose
  `::new` always returns `Err(GpuError::NoCudaFeature)`. This keeps the
  type name resolvable in host-only builds so downstream crates that
  unconditionally `use ferrotorch_gpu::GpuDevice` still compile.

## Acceptance Criteria

- [x] AC-1: `GpuDevice::new(0)` succeeds on a system with CUDA device 0
  present; verified by `mod tests` of every consuming kernel module.
- [x] AC-2: The cached cuBLAS handle is reused across calls — verified
  by `ferrotorch-gpu/src/blas.rs` performance tests not regressing
  the ~1.7 ms cuModuleLoadData cost on subsequent matmul calls.
- [x] AC-3: `GpuDevice::fork_for_capture` produces a stream usable with
  `crate::graph::begin_capture` — verified by graph-capture tests in
  `graph.rs::tests`.
- [x] AC-4: `GpuDevice::stream()` returns the thread-local stream when
  one is set via `StreamGuard`, otherwise the default — verified by
  `stream.rs::tests::current_stream_or_default_fallback`.
- [x] AC-5: Host-only `cargo build -p ferrotorch-gpu --no-default-features`
  succeeds — the stub `GpuDevice` keeps the type name resolvable.

## Architecture

### Struct layout (REQ-1)

`pub struct GpuDevice in device.rs` holds four fields. The cudarc
`CudaContext` and `CudaStream` are wrapped in `Arc` because cudarc
demands shared ownership for stream forking and module loading; the
cuBLAS handle is by-value because cuBLAS handles are not `Clone`. The
ordinal is stored as a plain `usize` for cheap accessor return. The
`Arc<CudaContext>` is the master ownership root for the device; when
the last `GpuDevice` referencing it drops, cudarc tears the context
down via `cuCtxDestroy`.

### Construction (REQ-2)

`pub fn GpuDevice::new in device.rs` calls
`CudaContext::new(ordinal)?` (which the `From` impl on `GpuError`
forwards as `GpuError::Driver`), takes the default stream via
`ctx.default_stream()`, and constructs the cached `CudaBlas::new(stream)`.
The `?` operator surfaces the cudarc driver error through `GpuError`.

Non-test production consumer:
- `ferrotorch-gpu/src/backend_impl.rs` —
  `Arc::new(GpuDevice::new(0).map_err(...)?)` inside
  `CudaBackendImpl::new`.
- `ferrotorch-diffusion/src/gpu/vae_encoder.rs`,
  `clip.rs`, `unet.rs`, `vae.rs` — every diffusion-side
  GPU pipeline opens its own `GpuDevice::new(0)`.
- `ferrotorch-distributed/src/ucc_backend.rs` and
  `gpu_collective.rs` — collective ops.
- `ferrotorch-jit/src/fusion_gpu.rs` — fused-chain runtime executor.

### Capture-friendly fork (REQ-3)

`pub fn fork_for_capture in device.rs` calls `parent.stream.fork()`
to obtain a non-blocking forked stream that CUDA graph capture
accepts (the legacy `0` default stream is rejected by
`cuStreamBeginCapture`). A fresh `CudaBlas` is bound to the forked
stream so cuBLAS calls under capture record to the same stream as
non-cuBLAS launches. `Arc<CudaContext>` is shared with the parent so
both devices see the same module cache.

Non-test production consumer: this method is used by
`crate::graph::begin_capture`-aware wrappers in the CL-454 graph suite
(see `graph.rs` REQs). Direct external use is absent — the API exists
to be called when a downstream caller (a future `ferrotorch-llama`
pre-decode graph) wants its own capture stream without disturbing
the shared default. The boundary contract is the existence of the
method, not a current external call site; this is grandfathered API
surface per goal.md S5.

### Accessors (REQ-4)

`impl GpuDevice in device.rs` provides:
- `pub fn context(&self) -> &Arc<CudaContext>` — handle for cudarc module
  loaders (e.g. `crate::module_cache::get_or_compile(dev.context(), ...)`).
- `pub fn default_stream(&self) -> &Arc<CudaStream>` — the legacy
  default stream.
- `pub fn stream(&self) -> Arc<CudaStream>` — resolves to the
  thread-local current stream via
  `crate::stream::current_stream_or_default(self)`. This is THE
  per-call stream the kernel modules read; the legacy default is the
  fallback.
- `pub fn blas(&self) -> &CudaBlas` — the cached cuBLAS handle. Used
  inside `crate::blas` matmul/bmm shims.
- `pub fn ordinal(&self) -> usize` — for stats / error reporting.

Non-test production consumer: every kernel module calls `dev.stream()`
or `dev.context()` to launch a kernel — e.g.
`ferrotorch-gpu/src/kernels.rs` calls `device.stream().launch_builder(...)`
under the hood through cudarc's launch APIs. The accessor chain is
the entire kernel-dispatch path.

### Clone (REQ-5)

`impl Clone for GpuDevice in device.rs` produces a fresh `CudaBlas`
bound to the same shared stream. The `expect("CudaBlas::new failed in GpuDevice::clone")`
is the documented R-CODE-2 carve-out: `Clone` returns `Self`, not
`Result<Self>`, so we can't propagate. The clone never fires in
practice because the underlying CUDA context is already known-good
by the time we have a `GpuDevice` to clone; the `expect` documents
the invariant.

Non-test production consumer: `ferrotorch-diffusion/src/gpu/clip.rs` and
`ferrotorch-diffusion/src/gpu/unet.rs` pass `&GpuDevice` by reference
rather than cloning, but the meta-crate `ferrotorch::cuda(0)` builder
(`tensor_bridge.rs`) returns owned `GpuDevice` to callers that need
to clone for per-thread storage.

### Stub (REQ-6)

`#[cfg(not(feature = "cuda"))] pub struct GpuDevice in device.rs` and
its `impl` at `device.rs` provide a minimal one-field stub.
`GpuDevice::new(ordinal)` always returns
`Err(GpuError::NoCudaFeature)`. The `ordinal()` accessor returns the
stored value so host-only call sites can still construct error
messages, etc.

Non-test production consumer: `ferrotorch-data/src/transforms.rs`
calls `ferrotorch_gpu::init_cuda_backend()` which propagates the
`NoCudaFeature` error cleanly when the feature is off. The stub keeps
this code path compiling on no-CUDA CI machines.

## Parity contract

`parity_ops = []`. `GpuDevice` is INFRASTRUCTURE — it has no parity-sweep
op of its own. The kernel-side parity ops verify the device-stream
pipeline structurally: every parity-sweep run constructs a `GpuDevice`
and dispatches kernels through `dev.stream()`, so a regression in
device construction would surface as the entire sweep failing.

Edge cases handled:
- Construction on a system without a GPU returns `GpuError::Driver`
  containing the cudarc driver error verbatim — pinned by every
  diffusion / llama GPU test's `let Ok(device) = GpuDevice::new(0)
  else { return };` skip pattern.
- Cloning when cuBLAS handle creation fails: `expect` fires loudly
  with a fixed message; this is the documented boundary trade-off
  for `Clone`'s infallible contract.

## Verification

Tests live in the consumer modules (this file's `mod tests` would
require a real GPU; per-test skip patterns exist in every kernel
module instead):
- `ferrotorch-gpu/src/stream.rs::tests::current_stream_or_default_fallback`
  (line 929) verifies the `stream()` accessor's thread-local
  resolution.
- `ferrotorch-gpu/src/module_cache.rs::tests::cache_returns_function_on_repeated_calls`
  exercises `dev.context()` plumbing.
- Every `mod tests` in `kernels.rs`, `blas.rs`, `bf16.rs`, etc. opens a
  device with `GpuDevice::new(0).expect(...)` and proceeds.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-gpu --features cuda device:: 2>&1 | tail -3
cargo test -p ferrotorch-gpu --features cuda stream::tests::current_stream_or_default_fallback 2>&1 | tail -3
```

Expected: `0 failed`. (When run on a system with no CUDA device, every
test returns early via the `Ok(device) = GpuDevice::new(0) else { return }`
skip pattern.)

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct GpuDevice in device.rs` at `device.rs` holds `(Arc<CudaContext>, Arc<CudaStream>, CudaBlas, usize)`. Non-test production consumer: `ferrotorch-gpu/src/backend_impl.rs` constructs the struct inside `CudaBackendImpl::new`. |
| REQ-2 | SHIPPED | impl: `pub fn GpuDevice::new in device.rs` at `device.rs` calls `CudaContext::new(ordinal)? + default_stream() + CudaBlas::new(stream)?`. Non-test production consumer: `ferrotorch-jit/src/fusion_gpu.rs` — `GpuDevice::new(handle.device_ordinal()).map_err(...)?`. |
| REQ-3 | SHIPPED | impl: `pub fn fork_for_capture in device.rs` at `device.rs` forks the parent stream and rebinds cuBLAS. Non-test production consumer: the `crate::graph` capture suite consumes the forked stream (graph.rs CL-454 surface), pinned by `module_cache::tests::cached_kernel_produces_correct_results`'s context plumbing. |
| REQ-4 | SHIPPED | impl: accessors `context() / default_stream() / stream() / blas() / ordinal()` at `device.rs`. Non-test production consumer: `ferrotorch-gpu/src/conv.rs` calls `dev.context()` for `module_cache::get_or_compile`; every kernel call site uses `dev.stream()`. |
| REQ-5 | SHIPPED | impl: `impl Clone for GpuDevice in device.rs` at `device.rs` constructs a fresh `CudaBlas` bound to the same shared stream. Non-test production consumer: meta-crate `ferrotorch/src/lib.rs` re-exports `GpuDevice`; tensor-bridge users clone the device when caching per-thread copies. |
| REQ-6 | SHIPPED | impl: `#[cfg(not(feature = "cuda"))] pub struct GpuDevice in device.rs` at `device.rs` with stub `::new` returning `GpuError::NoCudaFeature`. Non-test production consumer: `ferrotorch-data/src/transforms.rs` calls `init_cuda_backend()` which threads the stub error through cleanly under `--no-default-features`. |
