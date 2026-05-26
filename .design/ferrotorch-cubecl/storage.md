# CubeCL device-resident storage handle

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/cuda/
  - c10/cuda/
-->

## Summary

`ferrotorch-cubecl/src/storage.rs` is the concrete on-device buffer wrapper
that backs every CubeCL-resident tensor. `CubeclStorageHandle` owns a
`cubecl::server::Handle` (the cubecl-side reference into device memory) plus
an `Arc<CubeRuntime>` so the runtime stays alive as long as any tensor holds
the buffer. The trait it implements (`CubeStorageHandle`) lives in
`ferrotorch-core::storage`, keeping the core→cubecl dependency one-way:
core knows the trait, cubecl provides the only implementation, and
`ferrotorch-xpu` consumes both. This mirrors the upstream split where
`c10::DataPtr` (in `c10/core/Allocator.h`) declares the device-buffer abstraction
and `c10::cuda::CUDACachingAllocator` (in `c10/cuda/CUDACachingAllocator.h`)
provides the CUDA-specific implementation; ferrotorch collapses the
allocator concept into the cubecl-managed handle and keeps only the
reference-counted lifetime here.

## Requirements

- REQ-1: `CubeclStorageHandle` struct — fields `handle: cubecl::server::Handle`,
  `runtime: Arc<CubeRuntime>`, `len: usize`, `ordinal: usize`. Holding the
  `Arc<CubeRuntime>` ensures the cubecl `ComputeClient` (and its device
  memory pool) outlives any tensor referencing buffers it allocated. This
  is the Rust-RAII equivalent of upstream's `DataPtr` + `Storage` ref-count
  chain in `c10/core/Storage.h`.

- REQ-2: `CubeStorageHandle` trait impl — implements the four methods core
  declares: `as_any() -> &dyn Any` (for downcast in `cubecl_handle_of`),
  `len() -> usize`, `ordinal() -> usize`, `read_to_host() ->
  FerrotorchResult<Vec<f32>>` (dispatches to `CubeRuntime::read_f32s`),
  `clone_handle() -> Box<dyn CubeStorageHandle>` (bumps cubecl's internal
  ref-count via `handle.clone()`, NOT a buffer copy).

- REQ-3: `from_raw(handle, runtime, len, ordinal)` — public constructor
  used by `wrap_kernel_output` to package a raw kernel-result handle into
  a `CubeclStorageHandle`. The handle comes from inside cubecl
  (`portable_*` returns a `cubecl::server::Handle`); this constructor lets
  callers wrap it without depending on cubecl directly.

- REQ-4: `raw_handle() -> &cubecl::server::Handle` — borrow accessor used by
  `ops.rs` dispatch macros to pass an already-uploaded device handle into
  the kernel launchers without going through the slice-upload path. Key
  invariant for the no-H2D-round-trip optimisation (#673).

- REQ-5: `runtime() -> &Arc<CubeRuntime>` — borrow accessor for the
  underlying runtime, used by `ferrotorch-xpu::make_xpu_tensor` to chain
  the runtime into new handles allocated for downstream tensors.

- REQ-6: `wrap_kernel_output(handle, shape, runtime, ordinal)
  -> CubeclStorageHandle` — wrap the `(Handle, Vec<usize>)` result of a
  `portable_*` kernel into a `CubeclStorageHandle` without an extra H2D
  upload. `numel = shape.iter().product()`. Used at every `ferrotorch-xpu`
  op return site.

- REQ-7: `upload_f32(data, runtime, ordinal) -> FerrotorchResult
  <CubeclStorageHandle>` — single H2D upload point for CPU→XPU tensor
  construction. Cfg-gated `any(wgpu,cuda,rocm)` real-backend variant
  uses `c.create_from_slice(f32::as_bytes(data))`; the no-feature
  fallback returns `DeviceUnavailable`. Mirrors upstream's
  `at::empty(shape, cuda).copy_(cpu_tensor)` H2D path collapsed into one
  call.

- REQ-8: `cubecl_handle_of(t: &Tensor<f32>) -> Option<&CubeclStorageHandle>`
  — downcast helper. Looks up the tensor's storage; if it's a
  `CubeStorageHandle`, downcasts to the concrete `CubeclStorageHandle`
  (the only implementor). Returns `None` for CPU/CUDA tensors. Used by
  `ops.rs` dispatch macros to route handle-direct vs. slice-upload paths.

## Acceptance Criteria

- [x] AC-1: `CubeclStorageHandle` is `Debug` (derive on the struct).
- [x] AC-2: `clone_handle()` does NOT copy device memory — the cubecl
  `Handle` clone is a ref-count bump (documented in the impl).
- [x] AC-3: `upload_f32` errors `DeviceUnavailable` without a backend
  feature compiled in.
- [x] AC-4: `cubecl_handle_of(cpu_tensor)` returns `None`; for an
  XPU tensor it returns `Some(_)`. Verified through xpu integration
  tests in `ferrotorch-xpu/src/lib.rs`.

## Architecture

### Lifetime contract (REQ-1)

`pub struct CubeclStorageHandle in storage.rs` holds four fields. The
`Arc<CubeRuntime>` is the crucial bit: dropping the last tensor
referencing this buffer drops the storage, which drops the
`CubeclStorageHandle`, which drops the `cubecl::server::Handle` (which
returns the device allocation to cubecl's allocator pool), THEN drops
the `Arc<CubeRuntime>` (which may be the last reference, in which case
the runtime itself drops). Order matters: handle drops first so cubecl
can return the memory before the client goes away. Cubecl's server
already enforces this via its internal ref-counting on `Handle`, so the
Rust drop order falls out correctly without explicit `Drop` impl.

### Trait wiring (REQ-2)

`impl CubeStorageHandle for CubeclStorageHandle in storage.rs`. The
trait lives in `ferrotorch-core::storage`; cubecl is the only impl. The
`as_any` method is the type-erasure escape hatch used by
`cubecl_handle_of` to downcast back to the concrete type:

```text
t.inner_storage_arc().cubecl_handle()
    .and_then(|h: &dyn CubeStorageHandle|
              h.as_any().downcast_ref::<CubeclStorageHandle>())
```

The trait is `dyn`-compatible (no `Self`-typed return) because
`ferrotorch_core::TensorStorage::Cubecl` carries
`Box<dyn CubeStorageHandle>` (core can't name `CubeclStorageHandle`
without taking a cubecl dep).

### Raw-handle wrap (REQ-3 / REQ-6)

`pub fn from_raw in storage.rs` is the public constructor that takes a
`cubecl::server::Handle` returned by a kernel launcher and packages it
with an `Arc<CubeRuntime>`. `pub fn wrap_kernel_output in storage.rs` is
the convenience wrapper that computes `numel = shape.product()` and
calls `from_raw`. Non-test production consumer:
`ferrotorch-xpu/src/lib.rs` — every `xpu_binary!`,
`xpu_unary!`, and `xpu_polynomial!` macro expansion calls
`wrap_kernel_output(handle, &shape, Arc::clone(xpu.runtime()),
xpu.ordinal())`.

### Handle-direct dispatch (REQ-4)

`pub fn raw_handle in storage.rs` exposes the inner cubecl `Handle` by
reference. Used by `ops.rs::dispatch_binary!` (etc.) to pass the
already-uploaded handle directly into `kernels::run_*_handle` instead
of materialising a host slice. Non-test production consumer:
`ferrotorch-cubecl/src/ops.rs` — `ha.raw_handle().clone()`
inside the dispatch macros. The `.clone()` is a cubecl-side refcount
bump (no copy); the original handle inside the storage is unaffected.

### H2D upload (REQ-7)

`pub fn upload_f32 in storage.rs` has two cfg arms:

- Feature-on: build a `cubecl::server::Handle` via
  `client.create_from_slice(f32::as_bytes(data))`, then construct a
  `CubeclStorageHandle::new(handle, runtime, data.len(), ordinal)`.
- Feature-off: return `Err(FerrotorchError::DeviceUnavailable)`.

The match in the feature-on arm includes a `CubeClient::Stub =>
unreachable!()` arm — uploading through a Stub runtime would imply a
kernel could consume the buffer, which dispatch macros refuse. The
`unreachable!()` is the documented #1083 contract.

Non-test production consumer: `ferrotorch-xpu/src/lib.rs` —
`let handle = upload_f32(&data, Arc::clone(xpu.runtime()),
xpu.ordinal())?;` inside `make_xpu_tensor`.

### Downcast helper (REQ-8)

`pub fn cubecl_handle_of in storage.rs` is the structural cousin of
`std::any::Any::downcast_ref`. Used by `ops.rs::dispatch_unary!`
(etc.) at the top of every `portable_*` op to ask: is this tensor's
storage already device-resident? If so, route through the handle-direct
kernel; if not, upload its data.

Non-test production consumer: `ferrotorch-cubecl/src/ops.rs`
— `match (cubecl_handle_of($a), cubecl_handle_of($b))` inside the
dispatch macros.

## Parity contract

ferrotorch-cubecl is INFRASTRUCTURE — `parity_ops = []`. The handle's
contract is enforced structurally by the cubecl ref-count machinery and
the `CubeStorageHandle` trait. No parity-sweep op exists for "device
buffer lifecycle"; the verification gauntlet is:

- `cargo test -p ferrotorch-cubecl --no-default-features` — covers the
  no-backend branch of `upload_f32` (returns `DeviceUnavailable`).
- `cargo test -p ferrotorch-xpu --features wgpu` — covers the H2D +
  device-resident round-trip when a wgpu adapter exists.
- The conformance suite at `ferrotorch-core/tests/conformance_*`
  exercises `Tensor::to(Device::Xpu(0))` followed by `tensor.cpu()`,
  which goes through `upload_f32` → kernel → `read_to_host` → `read_f32s`.

Edge cases:
- `CubeclStorageHandle::len() == 0`: legal; `clone_handle` still bumps
  a refcount on a zero-size cubecl handle. Verified by core's
  empty-tensor conformance tests.
- `clone_handle()` is shallow — pinned by the comment "cubecl Handle
  clone is cheap — it bumps an internal ref count, not a buffer copy."
  Mirrors upstream's `c10::Storage::clone()` (also a refcount bump).
- Ordinal mismatch: if a tensor's storage ordinal disagrees with the
  XPU device it's used on, `ferrotorch-xpu::check_xpu_tensor` errors
  `DeviceMismatch` before any kernel dispatch. Not enforced inside
  this module; the consumer does the check.

## Verification

There are no in-module unit tests for `storage.rs` directly — the
storage handle is structurally trivial (four fields, four trait
methods). Coverage is integration-test driven:

- `ferrotorch-xpu/src/lib.rs::tests::make_xpu_tensor_*` — XPU tensor
  construction via `upload_f32`.
- `ferrotorch-cubecl/src/ops.rs::tests::portable_*_runs_on_gpu`
  (cfg `wgpu`) — round-trips through `CubeclStorageHandle` end-to-end.
- `ferrotorch-cubecl/src/storage.rs::cubecl_handle_of` doctest
  (lines 222-227, marked `ignore` because it requires a runtime).

Smoke command:

```bash
cargo test -p ferrotorch-cubecl --no-default-features 2>&1 | tail -3
```

Expected: ≥ 1 `test result: ok` line (21 tests including the storage-
relevant `upload_f32` no-backend path implicitly via `ops::no_backend_tests`).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct CubeclStorageHandle in storage.rs` with `handle/runtime/len/ordinal` fields. Non-test consumer: `ferrotorch-xpu/src/lib.rs` packs the result of `upload_f32` into `TensorStorage::xpu_from_handle(Box::new(handle), xpu.ordinal())`. |
| REQ-2 | SHIPPED | impl: `impl CubeStorageHandle for CubeclStorageHandle in storage.rs` with `as_any/len/ordinal/read_to_host/clone_handle`. Non-test consumer: `ferrotorch-core::TensorStorage::Cubecl` holds `Box<dyn CubeStorageHandle>` and invokes these methods through the trait when reading back via `Tensor::cpu()`. |
| REQ-3 | SHIPPED | impl: `pub fn from_raw in storage.rs`. Non-test consumer: `pub fn wrap_kernel_output in storage.rs` calls it; `wrap_kernel_output` itself is called at `ferrotorch-xpu/src/lib.rs`. |
| REQ-4 | SHIPPED | impl: `pub fn raw_handle in storage.rs`. Non-test consumer: `ferrotorch-cubecl/src/ops.rs` — `ha.raw_handle().clone()` inside `dispatch_binary!`, `dispatch_unary!`, `dispatch_matmul!`. |
| REQ-5 | SHIPPED | impl: `pub fn runtime in storage.rs`. Non-test consumer: indirect — `ferrotorch-xpu/src/lib.rs` `XpuDevice::runtime()` exposes its own `&Arc<CubeRuntime>`; the matching `CubeclStorageHandle::runtime()` is the symmetric accessor used by downstream code that reaches the handle (e.g. through `cubecl_handle_of`) and needs the runtime to chain into the next op's read-back path. |
| REQ-6 | SHIPPED | impl: `pub fn wrap_kernel_output in storage.rs`. Non-test consumer: `ferrotorch-xpu/src/lib.rs` invoked from `xpu_binary!`, `xpu_unary!`, `xpu_polynomial!` macro expansions on every XPU op. |
| REQ-7 | SHIPPED | impl: `pub fn upload_f32 in storage.rs` with feature-on/off arms; `Stub` arm `unreachable!` per #1083. Non-test consumer: `ferrotorch-xpu/src/lib.rs` inside `make_xpu_tensor`. |
| REQ-8 | SHIPPED | impl: `pub fn cubecl_handle_of in storage.rs` using `Any::downcast_ref`. Non-test consumer: `ferrotorch-cubecl/src/ops.rs` inside dispatch macros. |
