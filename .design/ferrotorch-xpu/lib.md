# ferrotorch-xpu — crate root (lib.rs)

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222 (working tree at /home/doll/pytorch)
upstream-paths:
  - aten/src/ATen/xpu/XPUContext.h
  - aten/src/ATen/xpu/XPUDevice.h
  - c10/xpu/XPUFunctions.h
  - c10/xpu/XPUFunctions.cpp
  - torch/xpu/__init__.py
-->

## Summary

`ferrotorch-xpu/src/lib.rs` is the crate root for ferrotorch's Intel XPU
backend (Arc / Data Center GPU Max). It exposes an `XpuDevice` handle
that wraps a `ferrotorch_cubecl::CubeRuntime` configured for the wgpu
backend (Vulkan under the hood), plus a fixed set of f32 elementwise,
binary, matmul, and orthogonal-polynomial ops (`xpu_add`, `xpu_sub`,
`xpu_mul`, `xpu_div`, `xpu_matmul`, `xpu_neg`, `xpu_abs`, `xpu_relu`,
`xpu_exp`, `xpu_ln`, `xpu_sqrt`, `xpu_sin`, `xpu_cos`, `xpu_tanh`,
`xpu_sigmoid`, and the 8 Chebyshev / Hermite / Laguerre / Legendre
families) that mirror the upstream `c10::xpu::device_count` /
`at::xpu::is_available` / `torch.xpu.is_available` lifecycle and the
SYCL kernel surface in `aten/src/ATen/xpu/`. Without the `wgpu` feature
the public surface compiles as `Err(DeviceUnavailable)` stubs so the
workspace builds on hosts without a usable Vulkan adapter.

## Requirements

- REQ-1: Lint baseline and module gating. `#![warn(clippy::all,
  clippy::pedantic)]` + `#![deny(rust_2018_idioms,
  missing_debug_implementations)]`, `#![allow(missing_docs)]` while the
  workspace-wide rustdoc sweep is in flight (matches the
  ferrotorch-cubecl / ferrotorch-gpu / ferrotorch-jit precedent), and a
  per-item `#![allow(...)]` block covering `clippy::doc_markdown`,
  `clippy::must_use_candidate`, `clippy::many_single_char_names`,
  `clippy::cast_precision_loss`. Each `#![allow]` carries an in-source
  one-line rationale. No `unsafe`, no `unwrap`/`expect`/`panic!` outside
  `#[cfg(test)]`, no `todo!`/`unimplemented!`/`unreachable!` in
  production code.

- REQ-2: `XpuDevice` lifecycle handle. `pub struct XpuDevice in lib.rs`
  derives `Debug + Clone`. With the `wgpu` feature it owns
  `Arc<CubeRuntime>` so the wgpu adapter is initialised once per device
  and shared across every op call. `XpuDevice::new(ordinal)` mirrors
  `c10::xpu::set_device` semantics — successful construction means the
  ordinal is usable by the runtime; failure surfaces
  `FerrotorchError::DeviceUnavailable`. `XpuDevice::ordinal` and
  `XpuDevice::device` expose the device-index round trip needed by
  `Tensor::to(Device::Xpu(n))` consumers in `ferrotorch-core`.

- REQ-3: Availability probe. `XpuDevice::is_available` mirrors
  `torch.xpu.is_available()` from upstream `torch/xpu/__init__.py:271`
  ("Return a bool indicating if XPU is currently available... never
  throws"). The compiled-out path returns `false` immediately; the
  compiled-in path probes for a usable wgpu adapter via
  `CubeRuntime::new(CubeDevice::Wgpu(0))` and catches the wgpu
  worker-thread panic that fires on WSL2 hosts without Vulkan ICDs,
  so callers can rely on the same "never throws" contract upstream
  promises.

- REQ-4: Device-resident H2D upload. `pub fn make_xpu_tensor in lib.rs`
  is the single H2D upload point — it allocates a device buffer
  directly via `ferrotorch_cubecl::upload_f32`, builds a
  `TensorStorage::xpu_from_handle`, and returns a `Tensor<f32>` whose
  `device()` is `Device::Xpu(ordinal)` and whose `data()` errors until
  the caller explicitly calls `.cpu()`. This kills the
  CPU-allocate-then-upload double-allocation path (issue #673) and
  matches upstream's `at::empty({...}, device=xpu)` which never
  materialises a host copy.

- REQ-5: Binary kernel surface. `xpu_add`, `xpu_sub`, `xpu_mul`,
  `xpu_div`, `xpu_matmul` dispatch through the
  `ferrotorch_cubecl::ops::portable_*` surface, validate both operands
  live on the same `XpuDevice` (returning
  `FerrotorchError::DeviceMismatch` otherwise), and keep results
  device-resident. The macro-generated bodies all carry `# Errors`
  rustdoc.

- REQ-6: Unary kernel surface. `xpu_neg`, `xpu_abs`, `xpu_relu`,
  `xpu_exp`, `xpu_ln`, `xpu_sqrt`, `xpu_sin`, `xpu_cos`, `xpu_tanh`,
  `xpu_sigmoid` dispatch through the corresponding
  `ferrotorch_cubecl::ops::portable_*` unary kernels, validate the
  operand's device, and keep results device-resident.

- REQ-7: Orthogonal-polynomial surface. `xpu_chebyshev_polynomial_{t,
  u, v, w}`, `xpu_hermite_polynomial_{h, he}`, `xpu_laguerre_polynomial_l`,
  `xpu_legendre_polynomial_p` dispatch through the corresponding
  cubecl three-term recurrence kernels with the integer-degree `n` as
  a scalar parameter (mirrors the
  `torch.special.chebyshev_polynomial_t(input, n)` shape from upstream
  `torch/_torch_docs.py`).

- REQ-8: Stub-mode compile parity. Without the `wgpu` feature the
  crate still defines every public symbol — `XpuDevice::new`,
  `XpuDevice::is_available`, `make_xpu_tensor`, and all 21 op
  wrappers — but they unconditionally return
  `FerrotorchError::DeviceUnavailable` (for fallible items) or `false`
  (for `is_available`). This keeps downstream code that wires through
  the meta-crate's `pub use ferrotorch_xpu::*` compiling on every
  platform.

- REQ-9: Display format `xpu:N`. `impl Display for XpuDevice in lib.rs`
  prints `"xpu:{ordinal}"` — matching the `Device::Xpu(_)` Display in
  `ferrotorch-core/src/device.rs` and PyTorch's
  `torch.device('xpu:0').__str__()`.

## Acceptance Criteria

- [x] AC-1: `cargo check -p ferrotorch-xpu --no-default-features` is
  clean (no-wgpu stub-mode build).
- [x] AC-2: `cargo check -p ferrotorch-xpu` (default features = `wgpu`)
  is clean.
- [x] AC-3: `cargo clippy -p ferrotorch-xpu --no-default-features --
  -D warnings` produces no new warnings beyond the documented
  per-crate allow set.
- [x] AC-4: `cargo test -p ferrotorch-xpu --no-default-features` passes
  (the `no_backend_tests` module's `xpu_device_new_errors_without_wgpu`
  test compiles and runs).
- [x] AC-5: On hosts with a usable wgpu adapter, the
  `cfg(all(test, feature = "wgpu"))` module's 13 unit tests pass; on
  WSL2 / no-adapter hosts they exit early via the `xpu()` probe
  returning `None`. No silent kernel failure.
- [x] AC-6: Every Rust symbol that takes a `Tensor<f32>` validates the
  tensor's device via `check_xpu_tensor` before dispatch (audit by
  grep on the `check_xpu_tensor(` call inside the
  `xpu_binary!`/`xpu_unary!`/`xpu_polynomial!` macros).
- [x] AC-7: `make_xpu_tensor` results return `Err` from `data()` and
  `Ok` from `cpu().data()` — asserted by
  `xpu_tensor_is_device_resident`.

## Architecture

### Lint baseline + per-item allows (REQ-1)

The crate-root header sets `#![warn(clippy::all, clippy::pedantic)]`
plus `#![deny(rust_2018_idioms, missing_debug_implementations)]`. The
`missing_docs` lint is `#![allow]`'d module-wide while the workspace
rustdoc sweep is in flight (the same posture as
`ferrotorch-cubecl/src/lib.rs`, `ferrotorch-gpu/src/lib.rs`,
`ferrotorch-jit/src/lib.rs`). Each remaining `#![allow(...)]` line in
the header block carries an inline justification block — they're
documented exceptions, not noise suppressors:

- `clippy::doc_markdown` — doc prose names `CubeCL`, `XPU`, `H2D`
  without backticking each one.
- `clippy::must_use_candidate` — getter churn for marginal value;
  existing callers already use returned values.
- `clippy::many_single_char_names` — math kernels naturally use
  single-character operands (a, b for binary; m, k, n for matmul).
- `clippy::cast_precision_loss` — small-integer test fixtures convert
  via `as f32` (exact for `0..8`).

Per R-CODE-2 / R-CODE-3 / R-APG-1 the production code holds no
`unwrap`, `expect`, `panic!`, `todo!`, `unreachable!`, or `unsafe`
outside `#[cfg(test)]`. The only `std::panic::catch_unwind` is inside
`XpuDevice::is_available` and is justified inline as the documented
defense against the wgpu worker thread panicking on WSL2 hosts with no
Vulkan ICD (REQ-3).

### `XpuDevice` lifecycle (REQ-2, REQ-3)

`pub struct XpuDevice in lib.rs` carries a `usize` ordinal and (with
the `wgpu` feature) an `Arc<CubeRuntime>`. `XpuDevice::new(ordinal)`
calls `CubeRuntime::new(CubeDevice::Wgpu(ordinal))` and propagates any
runtime error as `FerrotorchError::DeviceUnavailable`. The non-wgpu
arm returns `Err(FerrotorchError::DeviceUnavailable)` immediately.
This mirrors upstream's `c10::xpu::device_count()` from
`c10/xpu/XPUFunctions.cpp:243-246`:

```cpp
DeviceIndex device_count() {
  initDevicePoolCallOnce();
  return static_cast<DeviceIndex>(gDevicePool.devices.size());
}
```

— upstream lazily initialises the device pool and returns 0 when no
SYCL device is present; ferrotorch lazily initialises the wgpu runtime
and surfaces failure as `DeviceUnavailable`.

`XpuDevice::is_available` mirrors `torch.xpu.is_available` from
`torch/xpu/__init__.py:271-274`:

```python
def is_available() -> bool:
    r"""Return a bool indicating if XPU is currently available."""
    # This function never throws.
    return device_count() > 0
```

The Rust version uses `std::panic::catch_unwind` so a wgpu worker
panic (observed on WSL2 hosts without Vulkan ICDs) is converted to
`false` instead of unwinding past the FFI boundary — preserving the
"never throws" contract.

`XpuDevice::ordinal` and `XpuDevice::device` expose the round trip
needed by `Tensor::to(Device::Xpu(n))` and any caller wanting to log
or hash the device. `XpuDevice::runtime` (only under the `wgpu`
feature) borrows the underlying `Arc<CubeRuntime>` so downstream code
can drop down to the portable cubecl API for ops this crate hasn't
wrapped yet.

`XpuDevice::new_for_testing` (only under
`#[cfg(not(feature = "wgpu"))]`, `#[doc(hidden)]`) constructs a
stub-mode device whose every op returns `DeviceUnavailable`. It exists
solely to unblock runtime stub-mode assertions in conformance tests
that previously had to settle for compile-time signature pins (#1076);
it is unavailable under default features so production code cannot
mistake it for the real constructor.

### Device-resident upload (REQ-4)

`pub fn make_xpu_tensor in lib.rs` calls
`ferrotorch_cubecl::upload_f32(&data, Arc::clone(xpu.runtime()),
xpu.ordinal())` to allocate a device buffer in one shot, wraps the
returned handle as `TensorStorage::xpu_from_handle`, and returns a
`Tensor<f32>` via `Tensor::from_storage`. Calling `data()` on the
returned tensor errors — only `.cpu()` followed by `.data()` produces
host data. This is the post-#673 contract: the H2D copy happens
exactly once at construction; subsequent ops chain device-to-device.

### Binary / unary / polynomial macros (REQ-5, REQ-6, REQ-7)

Three internal macros — `xpu_binary!`, `xpu_unary!`, `xpu_polynomial!`
— factor out the shared dispatch shape:

1. `check_xpu_tensor` validates each operand's device.
2. The named `ferrotorch_cubecl::ops::portable_*` function runs the
   kernel, returning `(handle, shape)`.
3. `wrap_kernel_output` boxes the kernel handle into a
   `CubeclStorageHandle`.
4. `TensorStorage::xpu_from_handle` builds the new storage and
   `Tensor::from_storage` finalises the tensor.

The macros expand to a `pub fn` per op:

- 5 binaries: `xpu_add` / `xpu_sub` / `xpu_mul` / `xpu_div` /
  `xpu_matmul`, each forwarding to
  `ferrotorch_cubecl::ops::portable_{add, sub, mul, div, matmul}`.
- 10 unaries: `xpu_neg`, `xpu_abs`, `xpu_relu`, `xpu_exp`, `xpu_ln`,
  `xpu_sqrt`, `xpu_sin`, `xpu_cos`, `xpu_tanh`, `xpu_sigmoid`.
- 8 polynomials: 4 Chebyshev kinds + 2 Hermite kinds + Laguerre +
  Legendre, each taking the integer degree `n: usize`.

### `cfg(not(feature = "wgpu"))` stub family (REQ-8)

Three more macros — `xpu_binary_stub!`, `xpu_unary_stub!`,
`xpu_polynomial_stub!` — emit `pub fn` bodies that
unconditionally return `Err(FerrotorchError::DeviceUnavailable)`. The
same 23 names (5 binary + 10 unary + 8 polynomial) are emitted under
`#[cfg(not(feature = "wgpu"))]`, plus a no-feature
`make_xpu_tensor`. This keeps the public surface ABI-stable across
feature combinations so the meta-crate re-export
(`ferrotorch/src/lib.rs:143 — pub use ferrotorch_xpu::*;`) compiles
on every host regardless of whether the wgpu feature is active.

### Display + Device round trip (REQ-9)

`impl core::fmt::Display for XpuDevice in lib.rs` prints
`"xpu:{ordinal}"`, matching:

- `ferrotorch_core::Device::Xpu(_)` Display
  (`Xpu in ferrotorch-core/src/device.rs` — `Device::Xpu(id) => write!(f,
  "xpu:{id}")`).
- Upstream `torch.device('xpu:0').__str__()`.

### Non-test production consumers

The crate's public API is consumed by:

- `ferrotorch/src/lib.rs:143` — `pub use ferrotorch_xpu::*;` makes
  `XpuDevice`, `make_xpu_tensor`, `xpu_add`, and every other public
  item visible to application code as
  `ferrotorch::xpu::*` (and re-exports them from the meta-crate root
  via the workspace's standard glob layout).
- `ferrotorch-core/src/tensor.rs:981-991` — the `Tensor::to` D2H
  helper text directs CPU→XPU callers to `ferrotorch_xpu::make_xpu_tensor`
  / `ferrotorch_xpu::XpuDevice::upload`. The XPU→CPU readback path
  itself (`tensor.rs:1040`) consumes the `CubeclStorageHandle` that
  this crate produces, completing the round trip.
- `ferrotorch-cubecl/src/runtime.rs` REQ table records
  `ferrotorch-xpu/src/lib.rs::XpuDevice::new` as the non-test consumer
  for `CubeDevice::Wgpu`, and `ferrotorch-xpu/src/lib.rs::XpuDevice::is_available`
  as the consumer for `ferrotorch_cubecl::runtime::is_available`.
- `ferrotorch-cubecl/src/storage.rs` REQ table records
  `ferrotorch-xpu/src/lib.rs::make_xpu_tensor` as the non-test
  consumer for `ferrotorch_cubecl::upload_f32`.
- `ferrotorch-cubecl/src/kernels.rs` REQ table records the matmul
  kernel chain landing in `ferrotorch-xpu::xpu_matmul`.

## Parity contract

`parity_ops = []` for this route. The crate is the wgpu-targeted
backend wrapper around `ferrotorch-cubecl`; per-op numerical parity is
owned by:

- The `ferrotorch_cubecl::ops::portable_*` kernels themselves
  (kernel-level parity is the cubecl crate's contract).
- The `ferrotorch-core` routes for each op
  (`add`, `mul`, `matmul`, `exp`, etc.) which already have parity-sweep
  ops registered.

The `ferrotorch-xpu` wrapper preserves whatever the underlying
portable kernel does. Edge cases preserved at this crate's level:

- **No `wgpu` feature**: every public function returns the no-device
  stub value (`Err(DeviceUnavailable)` for fallible items, `false` for
  `is_available`). No panic, no silent CPU fallback. Same contract on
  every host.
- **WSL2 / no Vulkan ICD**: `XpuDevice::is_available` catches the wgpu
  worker-thread panic and returns `false`. `XpuDevice::new` returns
  `Err(DeviceUnavailable)`. Test arms exit via `let Some(xpu) = xpu()
  else { return };` rather than asserting.
- **Cross-device operand mismatch**: every binary, unary, and
  polynomial op calls `check_xpu_tensor` first, returning
  `FerrotorchError::DeviceMismatch { expected: Device::Xpu(n), got:
  Device::Cpu }` (or any other non-matching device). Verified by
  `xpu_add_rejects_cpu_input` and
  `xpu_polynomial_rejects_cpu_input`.
- **Non-2-D matmul**: `xpu_matmul` propagates the
  `ferrotorch_cubecl::ops::portable_matmul` shape check
  (`check_matmul_shapes`); 1-D × 1-D and 3-D × 3-D both surface
  `FerrotorchError::ShapeMismatch`. Verified by
  `xpu_matmul_rejects_non_2d_inputs`.
- **Chained device-resident ops**: `xpu_add(a, b) → c → xpu_add(c, a)
  → d`, `d.cpu()` is the only readback. Verified by
  `xpu_chained_ops_stay_on_device`.

## Verification

Unit tests in `mod tests in lib.rs` (`cfg(all(test, feature =
"wgpu"))`, 13 tests):

- `xpu_device_init_and_metadata` — ordinal / Device round trip,
  `is_available` true on adapter-available hosts.
- `xpu_tensor_is_device_resident` — post-#673 contract: `data()`
  errors, `.cpu().data()` succeeds.
- `xpu_add_runs_on_gpu_and_tags_xpu_storage` — full `xpu_add` round
  trip with device-resident result.
- `xpu_sub_mul_div_run_on_gpu` — three binaries in one run.
- `xpu_matmul_runs_on_gpu` — 2×3 · 3×2 = 2×2 with the canonical [58,
  64, 139, 154] expected.
- `xpu_matmul_rejects_non_2d_inputs` — 1-D × 1-D and 3-D × 3-D both
  return `ShapeMismatch`.
- `xpu_unary_kernels_run_on_gpu` — `neg`, `abs`, `relu` on signed
  inputs.
- `xpu_transcendentals_run_on_gpu` — `exp`, `ln` with PyTorch-matched
  tolerance.
- `xpu_add_rejects_cpu_input` — `DeviceMismatch` for CPU operand.
- `xpu_add_rejects_cuda_input_against_xpu_device` —
  `DeviceMismatch` for any non-XPU operand.
- `xpu_chained_ops_stay_on_device` — two chained `xpu_add` calls,
  single `.cpu()` readback at the end.
- `xpu_chebyshev_t_runs_on_gpu` — `T_3(x) = 4x^3 - 3x` parity.
- `xpu_legendre_p_runs_on_gpu` — `P_2(x) = (3x^2 - 1)/2` parity.
- `xpu_polynomial_rejects_cpu_input` — `DeviceMismatch` on the
  polynomial path.

`mod no_backend_tests in lib.rs` (`cfg(all(test, not(feature =
"wgpu")))`, 1 test):

- `xpu_device_new_errors_without_wgpu` — stub-mode build returns
  `DeviceUnavailable` and `!XpuDevice::is_available()`.

Integration tests in `ferrotorch-xpu/tests/conformance_xpu.rs` drive
the same surface through a PyTorch-CPU-generated fixture file and use
`cascade_skip` to exit cleanly on hosts without an Intel XPU.

Smoke commands (no parity ops to run):

```bash
cargo check -p ferrotorch-xpu --no-default-features
cargo check -p ferrotorch-xpu
cargo clippy -p ferrotorch-xpu --no-default-features -- -D warnings
cargo test -p ferrotorch-xpu --no-default-features
```

Expected: each command prints `Finished` / `test result: ok` with no
`error:` / `warning:` lines beyond the documented allows.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: crate-root lint header at top of `lib.rs` — `#![warn(clippy::all, clippy::pedantic)]` + `#![deny(rust_2018_idioms, missing_debug_implementations)]` + the per-item `#![allow(...)]` block with inline justifications; non-test consumer: every public symbol in `lib.rs` (e.g. `pub fn make_xpu_tensor`, `pub fn xpu_add`) compiles under this baseline, verified by `cargo clippy -p ferrotorch-xpu --no-default-features -- -D warnings` exiting clean. |
| REQ-2 | SHIPPED | impl: `pub struct XpuDevice in lib.rs` with `pub fn XpuDevice::new`, `XpuDevice::ordinal`, `XpuDevice::device`, `XpuDevice::runtime` mirroring `c10::xpu::device_count` from `c10/xpu/XPUFunctions.cpp:243`; non-test consumer: `ferrotorch/src/lib.rs:143` re-exports the type via `pub use ferrotorch_xpu::*;` as `ferrotorch::xpu::XpuDevice` for application code. |
| REQ-3 | SHIPPED | impl: `pub fn XpuDevice::is_available in lib.rs` with `catch_unwind`-guarded `CubeRuntime::new(CubeDevice::Wgpu(0)).is_ok()` mirroring `torch.xpu.is_available` from `torch/xpu/__init__.py:271-274` ("never throws"); non-test consumer: `ferrotorch_cubecl/src/runtime.rs` REQ-6 row records `ferrotorch-xpu/src/lib.rs::XpuDevice::is_available` as the production caller of `ferrotorch_cubecl::runtime::is_available`. |
| REQ-4 | SHIPPED | impl: `pub fn make_xpu_tensor in lib.rs` calling `ferrotorch_cubecl::upload_f32` and `TensorStorage::xpu_from_handle`; non-test consumer: `ferrotorch_cubecl/src/storage.rs` REQ-7 row records `ferrotorch-xpu/src/lib.rs::make_xpu_tensor` as the production caller of `upload_f32`, and `ferrotorch-core/src/tensor.rs:981-991` directs CPU→XPU users at `ferrotorch_xpu::make_xpu_tensor`. |
| REQ-5 | SHIPPED | impl: macro `xpu_binary! in lib.rs` expanded into `pub fn xpu_add/xpu_sub/xpu_mul/xpu_div/xpu_matmul` dispatching to `ferrotorch_cubecl::ops::portable_{add, sub, mul, div, matmul}`; non-test consumer: `ferrotorch/src/lib.rs:143` `pub use ferrotorch_xpu::*;` re-exports the five binary ops to application code, and the matmul path is cited as the terminal consumer of `kernel_matmul_naive` in `ferrotorch_cubecl/src/kernels.rs` REQ-4. |
| REQ-6 | SHIPPED | impl: macro `xpu_unary! in lib.rs` expanded into `pub fn xpu_neg/xpu_abs/xpu_relu/xpu_exp/xpu_ln/xpu_sqrt/xpu_sin/xpu_cos/xpu_tanh/xpu_sigmoid` dispatching to `ferrotorch_cubecl::ops::portable_*`; non-test consumer: `ferrotorch/src/lib.rs:143` `pub use ferrotorch_xpu::*;` re-exports all 10 unary ops. |
| REQ-7 | SHIPPED | impl: macro `xpu_polynomial! in lib.rs` expanded into `pub fn xpu_chebyshev_polynomial_{t,u,v,w}/xpu_hermite_polynomial_{h,he}/xpu_laguerre_polynomial_l/xpu_legendre_polynomial_p` dispatching to `ferrotorch_cubecl::ops::portable_*_polynomial_*`; non-test consumer: `ferrotorch/src/lib.rs:143` `pub use ferrotorch_xpu::*;` re-exports all 8 polynomial ops. |
| REQ-8 | SHIPPED | impl: `#[cfg(not(feature = "wgpu"))]` arms in `lib.rs` for `XpuDevice::new`, `XpuDevice::is_available`, `make_xpu_tensor`, plus the `xpu_binary_stub!`/`xpu_unary_stub!`/`xpu_polynomial_stub!` macros emitting 23 `Err(DeviceUnavailable)` bodies; non-test consumer: workspace-wide `cargo check -p ferrotorch-xpu --no-default-features` compiles, and `ferrotorch/src/lib.rs:143` `pub use ferrotorch_xpu::*;` propagates the stub surface to downstream crates that build on Linux/WSL without `wgpu`. |
| REQ-9 | SHIPPED | impl: `impl core::fmt::Display for XpuDevice in lib.rs` writing `"xpu:{ordinal}"`; non-test consumer: `ferrotorch-core/src/device.rs:71` (`Device::Xpu(id) => write!(f, "xpu:{id}")`) prints the same string for the device-level enum, so application code that formats either an `XpuDevice` or `Device::Xpu(0)` observes the same `"xpu:0"` and matches `torch.device('xpu:0').__str__()`. |
