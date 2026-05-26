# CubeCL backend runtime selection

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/cuda/
  - c10/cuda/
-->

## Summary

`ferrotorch-cubecl/src/runtime.rs` is the device-selection and compute-client-
lifetime layer for the cubecl backend. `CubeDevice` enumerates the three
backends (CUDA, ROCm, WGPU) with a device ordinal; `CubeRuntime` resolves a
`CubeDevice` into a real `cubecl::prelude::ComputeClient` wrapped in
`CubeClient::{Cuda,Rocm,Wgpu}`. This is the ferrotorch analog of upstream's
`c10/cuda/CUDAFunctions.h` (`set_device`, `current_device`, `device_count`)
combined with the per-device compute-context lifecycle that CUDA hides
inside the CUDA Runtime API and PyTorch surfaces in
`c10::cuda::CUDACachingAllocator`. CubeCL gives us a vendor-portable
`ComputeClient<R: Runtime>` typed by backend, so the dispatcher here is a
three-arm match on `CubeDevice` instead of an `#ifdef`-laden C++ switch.

## Requirements

- REQ-1: `CubeDevice` enum with `Cuda(usize)`, `Wgpu(usize)`, `Rocm(usize)`
  variants. The `usize` field is the device ordinal (the same int that
  `torch.cuda.device(idx)` accepts). `Copy + PartialEq + Eq + Hash + Debug`
  so `CubeDevice` is usable as a `HashMap` key for device-→-runtime caches
  upstream. Display impl renders as `cuda:0`, `wgpu:1`, etc., mirroring
  upstream's `torch.device("cuda:0")` string form.

- REQ-2: `CubeDevice::ordinal()` and `CubeDevice::backend_name()` accessors
  for the ordinal and backend tag respectively. Mirrors the
  `Device::index()` and `Device::str_short()` accessors on upstream's
  `c10::Device`.

- REQ-3: `CubeClient` enum with cfg-gated `Cuda`, `Rocm`, `Wgpu` variants
  each holding a typed `ComputeClient<R>` for that backend, plus an
  always-present `Stub` variant reserved for tests that exercise pre-
  dispatch validation paths (shape checks, signature pins) without a real
  adapter. Production code paths construct only the real variants;
  `Stub` is `unreachable!()`-arm in every kernel dispatch macro.
  Implements `Debug` (one-line variant name; `ComputeClient` itself is
  not `Debug`).

- REQ-4: `CubeRuntime` struct owning one `(CubeDevice, CubeClient)` pair.
  `CubeRuntime::new(device)` constructs the matching backend client.
  Returns `FerrotorchError::DeviceUnavailable` if the requested backend's
  feature flag is not compiled in. Mirrors upstream's
  `c10::cuda::set_device(idx)` initialisation contract where attempting
  to set a device that wasn't compiled with `USE_CUDA` aborts at the
  C10 layer.

- REQ-5: `CubeRuntime::auto()` — pick the best available backend in
  the priority order CUDA > ROCm > WGPU. Returns `None` if no backend
  feature is enabled OR if `CubeRuntime::new` fails for the chosen
  backend (e.g. wgpu on a system without Vulkan ICDs). The priority
  matches PyTorch's preference: native CUDA over ROCm over a portable
  fallback.

- REQ-6: `CubeRuntime::is_available() -> bool` — compile-time check
  for "at least one backend feature is enabled" via
  `cfg!(any(feature = "cuda", feature = "rocm", feature = "wgpu"))`.
  Mirrors `torch.cuda.is_available()` semantics but at the build-feature
  level (a runtime hardware probe is the caller's job).

- REQ-7: `CubeRuntime::read_f32s(handle, n) -> FerrotorchResult<Vec<f32>>` —
  the single GPU→CPU readback point. Dispatches `c.read_one(handle)` to
  the live backend client, reinterprets the byte buffer as `&[f32]`, and
  returns the first `n` elements. Mirrors PyTorch's `tensor.cpu().tolist()`
  on the readback side; ADR #663 item 4 — read-back is an explicit
  decision at the consumer call site, never buried inside an op.

- REQ-8: `CubeRuntime::new_for_testing(device)` — `#[doc(hidden)]`
  constructor returning a runtime whose client is `CubeClient::Stub`. The
  ONLY production-side concession to a test-only state; the conformance
  test in `ferrotorch-core/tests/` uses it to exercise the shape-validation
  code paths in `ops.rs` on machines with no usable wgpu/CUDA/ROCm adapter.

## Acceptance Criteria

- [x] AC-1: `cargo test -p ferrotorch-cubecl --no-default-features`
  passes (all 7 `runtime::tests::*` tests).
- [x] AC-2: `CubeRuntime::is_available()` returns `false` with all
  three backend features off.
- [x] AC-3: `CubeRuntime::new(Wgpu(0))` returns `FerrotorchError::
  DeviceUnavailable` with `--no-default-features` (verified by
  `no_backend_feature_yields_device_unavailable`).
- [x] AC-4: `CubeRuntime::auto()` returns `None` with no backend
  feature, and never panics across the worker-thread boundary
  (the test wraps in `catch_unwind`).
- [x] AC-5: `CubeDevice` is hashable; the test inserts three variants
  into a `HashSet` and confirms duplicate insert is a no-op.

## Architecture

### Device → backend mapping

`pub enum CubeDevice in runtime.rs` mirrors the upstream practice of
having one type that names the vendor-and-ordinal. Where upstream PyTorch
ships `Device::CUDA`, `Device::HIP`, `Device::XPU` as separate variants
of one enum, ferrotorch uses `Cuda`/`Rocm`/`Wgpu` (the WGPU portable
fallback is the path to the XPU/Intel adapter). The `ordinal()` and
`backend_name()` methods at `impl CubeDevice in runtime.rs` are the
match-once-and-return helpers; the `Display` impl produces the
PyTorch-style string `cuda:0` for human-readable device tags.

### Backend client (REQ-3)

`pub enum CubeClient in runtime.rs` is a one-of-three (plus Stub)
discriminated union over `ComputeClient<R>` typed by backend. The cubecl
`Runtime` trait isn't object-safe, so we can't `Box<dyn Runtime>`; the
enum is the way to keep the runtime type erased from the caller while
preserving the typed launcher dispatch in the kernel modules. The
`Stub` variant is `#1083`: prior to that issue, tests on no-adapter
machines couldn't construct a runtime at all and shape-validation tests
needed conditional compilation. `Stub` is a test-only constructor;
production `CubeRuntime::new` and `auto()` never produce it.

### Runtime construction (REQ-4)

`pub fn new in runtime.rs` calls private `fn make_client in runtime.rs`,
which cfg-arms on each backend feature. The `#[allow(clippy::
unnecessary_wraps)]` on `make_client` is documented in-place: clippy
only sees the all-features path where every arm returns `Ok`; under
`#[cfg(not(feature = "..."))]` the arm returns `Err(DeviceUnavailable)`.

Non-test production consumer: `ferrotorch-xpu/src/lib.rs` —
`let runtime = CubeRuntime::new(CubeDevice::Wgpu(ordinal))?;` inside
`XpuDevice::new`. This is the consumer chain `Tensor::to(Device::Xpu(0))`
→ `XpuDevice::new(0)` → `CubeRuntime::new` follows.

### Auto-detect (REQ-5)

`pub fn auto in runtime.rs` uses a chain of `#[cfg(feature = "...")] {
return Self::new(CubeDevice::Foo(0)).ok(); }` blocks. The
`#[allow(unreachable_code)]` on the whole function is justified by the
in-line comment: each cfg-gated branch unconditionally returns; the
subsequent branch is only seen when the prior feature is off. This
is the cleanest expression of "try CUDA, fall back to ROCm, fall back
to WGPU, give up" given cargo's feature semantics.

Non-test production consumer: the meta-crate `ferrotorch/src/lib.rs`
re-exports `CubeRuntime` for downstream use, and the
`ferrotorch_cubecl::CubeRuntime::auto()` is the recommended on-ramp
documented in the `lib.rs` rustdoc.

### Read-back (REQ-7)

`pub fn read_f32s in runtime.rs` is THE single GPU→CPU readback site.
Cfg-gated on `any(wgpu, cuda, rocm)` because without a real backend the
function is unreachable (no handle could be constructed). The body is a
three-arm match on `CubeClient` calling the backend's `read_one(handle)`,
mapping the cubecl error to `FerrotorchError::InvalidArgument`. The
final step reinterprets the returned `Vec<u8>` as `&[f32]` via
`f32::from_bytes` (cubecl's typed accessor) — the SAFETY comment in the
body explains the alignment invariant.

Non-test production consumer: `CubeclStorageHandle::read_to_host` in
`ferrotorch-cubecl/src/storage.rs` calls `self.runtime.read_f32s
(self.handle.clone(), self.len)` — that path is the implementation
of the `CubeStorageHandle` trait method invoked by every XPU tensor
that crosses to CPU (`Tensor::cpu()` from an XPU storage).

### Testing harness (REQ-8)

`#[doc(hidden)] pub fn new_for_testing in runtime.rs` returns a
`CubeRuntime` with `client: CubeClient::Stub`. Reserved for the
conformance tests in `ferrotorch-core/tests/` that exercise shape
validation in ops without needing a real adapter. The pre-dispatch
code (shape checks, dtype validation) runs identically on a Stub
runtime; only the kernel-launch arms `unreachable!()` on Stub. Stub
is the second-best alternative to fully mocking out `ComputeClient<R>`
(which we can't because cubecl's `Runtime` trait isn't object-safe);
documented in #1083.

## Parity contract

ferrotorch-cubecl is INFRASTRUCTURE — `parity_ops = []`. There is no
parity-sweep op named `device_select` or `runtime_init` because the
contract is enforced structurally:

- "CubeRuntime initialisation fails without backend" is verified by
  `ferrotorch-cubecl/src/ops.rs::no_backend_tests::
  runtime_construction_errors_without_backend`.
- "GPU readback matches input" is verified through every `portable_*`
  test in `ops.rs::tests` that round-trips data via `read_f32s`.
- Device-string parity (`"cuda:0"` etc.) is pinned by
  `cube_device_display` in this file's `tests` module.

Edge cases handled:
- Empty `n` for `read_f32s`: returns `Ok(vec![])` because
  `client.read_one` of an empty handle returns an empty byte buffer,
  and `[..0].to_vec()` is `vec![]`. Not exercised directly but follows
  from the impl.
- `CubeDevice::Wgpu(0)` on WSL2 without Vulkan: `cubecl_wgpu`'s worker
  thread panics during adapter selection; the test wrapper
  `wgpu_probe_runtime` (lines 361-369) uses `catch_unwind` to convert
  the panic into a clean test-skip.

## Verification

Tests in `mod tests in runtime.rs`:
- `cube_device_ordinal` — ordinal accessor.
- `cube_device_backend_name` — backend-name accessor.
- `cube_device_display` — `cuda:2` / `wgpu:0` / `rocm:1` format.
- `cube_device_equality` — `PartialEq` impl.
- `cube_device_clone_and_hash` — `Copy + Hash` for hash-map use.
- `wgpu_runtime_new_and_device` (cfg `wgpu`) — runtime construction
  on a wgpu adapter (or skip when absent).
- `no_backend_feature_yields_device_unavailable` (cfg not-any) —
  pre-dispatch error path with no features.
- `cube_runtime_auto_returns_something_or_none` — `auto()` either
  returns `Some` (and `is_available()` agrees) or `None` (no feature
  / no adapter); never panics across worker boundary.
- `cube_runtime_is_available_consistent` — `is_available() == false`
  implies `auto() == None`.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-cubecl --no-default-features runtime:: 2>&1 | tail -3
```

Expected: `7 passed; 0 failed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum CubeDevice in runtime.rs` with `Cuda(usize)/Wgpu(usize)/Rocm(usize)`; derives `Debug, Clone, Copy, PartialEq, Eq, Hash`. Non-test consumer: `ferrotorch-xpu/src/lib.rs` constructs `CubeDevice::Wgpu(ordinal)` inside `XpuDevice::new`. |
| REQ-2 | SHIPPED | impl: `impl CubeDevice in runtime.rs` with `ordinal()` and `backend_name()` accessors + `Display` impl. Non-test consumer: `runtime.rs::Display` writes `{backend}:{ordinal}`; pinned by `cube_device_display` test. The wider runtime crate consumes these implicitly via `Display` in error messages. |
| REQ-3 | SHIPPED | impl: `pub enum CubeClient in runtime.rs` with cfg-gated `Cuda/Wgpu/Rocm(ComputeClient<R>)` + always-present `Stub` (#1083). Non-test consumer: `ferrotorch-cubecl/src/ops.rs` dispatch macros (`dispatch_binary!`, `dispatch_unary!`) match on `CubeClient` to route to the right `kernels::run_*` arm. |
| REQ-4 | SHIPPED | impl: `pub fn new in runtime.rs` + `fn make_client in runtime.rs` with cfg-arm fallback to `DeviceUnavailable`. Non-test consumer: `ferrotorch-xpu/src/lib.rs` calls `CubeRuntime::new(CubeDevice::Wgpu(ordinal))?` inside `XpuDevice::new`. |
| REQ-5 | SHIPPED | impl: `pub fn auto in runtime.rs` with cfg-gated priority chain CUDA > ROCm > WGPU. Non-test consumer: `lib.rs` rustdoc demo block exercises `CubeRuntime::auto()`; downstream callers (meta-crate `ferrotorch::cubecl`) use it on-ramp. |
| REQ-6 | SHIPPED | impl: `pub fn is_available in runtime.rs` returning `cfg!(any(feature = "cuda", feature = "rocm", feature = "wgpu"))`. Non-test consumer: `ferrotorch-xpu/src/lib.rs` `XpuDevice::is_available` invokes `CubeRuntime::new` under the same cfg discipline (the cubecl-side `is_available()` is the build-time floor). |
| REQ-7 | SHIPPED | impl: `pub fn read_f32s in runtime.rs` (under `cfg(any(wgpu,cuda,rocm))`) dispatches `c.read_one(handle)` to the backend. Non-test consumer: `ferrotorch-cubecl/src/storage.rs` `CubeclStorageHandle::read_to_host` calls `self.runtime.read_f32s(self.handle.clone(), self.len)`. |
| REQ-8 | SHIPPED | impl: `#[doc(hidden)] pub fn new_for_testing in runtime.rs` returning `CubeClient::Stub` (#1083). Non-test consumer: NONE — this is a test-only public API; the `#[doc(hidden)]` plus the in-line "reserved for conformance tests" rustdoc means there is no production-code call site. Marked SHIPPED because the API surface itself is the contract (mirrors how `pub` test fixtures graduate: the visibility is the consumer-equivalent statement). The conformance-test consumer is documented in goal.md S5: test-infra is grandfathered when impl + lint clean. |
