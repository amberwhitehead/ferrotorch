# ferrotorch-mps — crate root (lib.rs)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - aten/src/ATen/mps/MPSDevice.h
  - aten/src/ATen/mps/MPSDevice.mm
  - aten/src/ATen/mps/MPSHooks.h
  - torch/mps/__init__.py
-->

## Summary

`ferrotorch-mps/src/lib.rs` is the crate root for ferrotorch's Apple
Metal Performance Shaders backend. It declares the lint baseline, the
`#[cfg(target_os = "macos")]`-gated `backend` module, the always-available
`kernels` MSL-sources module, and a small lifecycle surface
(`MpsDevice`, `is_mps_available`, `mps_device_count`, `init_mps_backend`)
that mirrors `torch.mps.device_count()` / `torch.mps.is_available()`
from upstream `torch/mps/__init__.py`. On every non-macOS host the
lifecycle items return `false` / `0` / `Err(DeviceUnavailable)` so the
workspace compiles cleanly on Linux/WSL.

## Requirements

- REQ-1: Lint baseline. `#![warn(clippy::all, clippy::pedantic)]`,
  `#![deny(rust_2018_idioms, missing_debug_implementations, missing_docs)]`,
  plus a conditional `#![cfg_attr(not(target_os = "macos"), deny(unsafe_code))]`
  — Linux/WSL builds are pure-safe-Rust; only the macOS branch (which
  goes through `objc2-metal`) carries leaf-primitive `unsafe` blocks
  with per-block `SAFETY:` annotations. Every targeted `#![allow]` carries
  an in-source one-line rationale.

- REQ-2: Platform gating. The `backend` module declaration carries
  `#[cfg(target_os = "macos")]`; the `MtlBackend` re-export carries the
  same gate. On every other platform the symbol is absent and downstream
  callers that conditionally consume it must use the same `cfg`.

- REQ-3: Public lifecycle surface. `MpsDevice`, `MpsDevice::new`,
  `MpsDevice::count`, `MpsDevice::ordinal`, the free `is_mps_available`,
  `mps_device_count`, and `init_mps_backend` are unconditionally
  available on every platform. On macOS they reflect
  `MTLCreateSystemDefaultDevice` and Sprint C.7's `init_mps_backend_metal`;
  on every other platform they return `false` / `0` /
  `Err(FerrotorchError::DeviceUnavailable)` immediately.

- REQ-4: `MpsDevice` value semantics. `MpsDevice` is
  `#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]` because it wraps
  a single `usize` ordinal. On macOS with a Metal device present,
  `MpsDevice::new(0)` returns `Ok`; non-zero ordinals return
  `Err(FerrotorchError::InvalidArgument)` (Apple Silicon exposes a single
  integrated GPU, matching `at::mps::MPSDevice::getInstance()` which is
  the only Metal device upstream uses); platforms without Metal return
  `Err(FerrotorchError::DeviceUnavailable)`.

- REQ-5: Display format `mps:N`. `impl Display for MpsDevice` prints
  `"mps:{ordinal}"` — mirroring `torch.device('mps:0').__str__()` and
  matching the `Device::Mps(_)` Display impl in `ferrotorch-core` so
  application code observes the same string regardless of which type it
  formats.

## Acceptance Criteria

- [x] AC-1: `cargo check -p ferrotorch-mps --no-default-features` is
  clean (Linux/WSL host-only build).
- [x] AC-2: `cargo test -p ferrotorch-mps` passes (5 unit tests + 27
  conformance tests + 1 doctest on Linux/WSL; macOS adds the live-MPS
  cascade_skip arms).
- [x] AC-3: `cargo clippy -p ferrotorch-mps --no-default-features -- -D warnings`
  produces no new warnings beyond the documented per-item `#[allow]` set.
- [x] AC-4: Every macOS-only symbol is `#[cfg(target_os = "macos")]`-gated
  (audit by grep on `pub use backend::` lines in `lib.rs`).
- [x] AC-5: `is_mps_available()`, `mps_device_count()`, `MpsDevice::new`,
  `init_mps_backend()` all return the documented stub values on
  non-macOS hosts (verified by the `#[cfg(not(target_os = "macos"))]`
  test arms).

## Architecture

### Lint baseline (REQ-1)

The crate-root lint header at the top of `lib.rs` sets
`#![warn(clippy::all, clippy::pedantic)]` plus
`#![deny(rust_2018_idioms, missing_debug_implementations)]` and
`#![deny(missing_docs)]`. The `unsafe_code` lint is conditionally denied
on non-macOS targets via
`#![cfg_attr(not(target_os = "macos"), deny(unsafe_code))]` — Linux/WSL
builds compile no Metal code, so they have no need for `unsafe`, while
the macOS branch retains the leaf-primitive carve-out for `objc2-metal`
calls. Each `#![allow(...)]` (module_name_repetitions, cast_possible_truncation,
ref_as_ptr, borrow_as_ptr, similar_names, many_single_char_names,
struct_field_names) carries an inline justification — they are not noise
suppressors but documented exceptions for Metal API patterns (e.g.
`setBytes_length_atIndex` requires the `&n_u32 as *const u32 as *mut _`
pointer pattern).

### Platform gating (REQ-2)

`pub mod backend in lib.rs` carries `#[cfg(target_os = "macos")]`, and
`pub use backend::MtlBackend in lib.rs` carries the same gate. The
`kernels` module is NOT gated — its constants are `include_str!`-bound
MSL strings that compile on every platform; they're only consumed by
the macOS `backend` module. This keeps the `kernels::MATMUL_F32` symbol
available to platform-agnostic tests like
`kernel_source_matmul_f32_present` in
`ferrotorch-mps/tests/conformance_mps.rs`.

### Lifecycle surface (REQ-3, REQ-4, REQ-5)

The free functions in `lib.rs` form a thin platform-agnostic facade:

- `pub fn is_mps_available in lib.rs` — `MTLCreateSystemDefaultDevice().is_some()`
  on macOS, `false` elsewhere. Mirrors upstream's
  `at::mps::is_available()` from `aten/src/ATen/mps/MPSDevice.h:79`.
- `pub fn mps_device_count in lib.rs` — `usize::from(is_mps_available())`
  on macOS (Apple Silicon is single-device), `0` elsewhere. Mirrors
  upstream `torch.mps.device_count()` from
  `torch/mps/__init__.py:25-27`.
- `pub fn init_mps_backend in lib.rs` — delegates to
  `backend::init_mps_backend_metal()` on macOS (compiles all 10 MSL
  kernels, registers via
  `ferrotorch_core::gpu_dispatch::register_gpu_backend`); returns
  `Err(DeviceUnavailable)` immediately on every other platform. No
  panic, no silent-no-op.

The `MpsDevice` type is a single-`usize` newtype carrying the device
ordinal. `MpsDevice::new(ordinal)` validates against
`MTLCreateSystemDefaultDevice` on macOS — `Ok` for ordinal 0 when a
device is present, `Err(InvalidArgument)` for any non-zero ordinal,
`Err(DeviceUnavailable)` when no device is found or on non-macOS hosts.
`MpsDevice::count()` delegates to the free `mps_device_count` so the
two spellings (`MpsDevice::count()` vs `mps_device_count()`) cannot
diverge.

`impl Display for MpsDevice in lib.rs` prints `"mps:{ordinal}"`,
matching the `Device::Mps(_)` Display impl in `ferrotorch-core` and
PyTorch's `torch.device('mps:0').__str__()`.

### Non-test production consumers

The crate's public API is re-exported by the meta-crate:
- `ferrotorch/src/lib.rs:137` — `pub use ferrotorch_mps::*;` makes
  `is_mps_available`, `mps_device_count`, `MpsDevice`,
  `init_mps_backend`, and (on macOS) `MtlBackend` visible to every
  downstream model crate and example binary.

The crate's primary lifecycle path is `init_mps_backend` →
`backend::init_mps_backend_metal` → `register_gpu_backend(Box::new(MtlBackend))`,
which makes `ferrotorch_core::gpu_dispatch::gpu_backend()` return
`Some(&dyn GpuBackend)` on macOS. From that point every CUDA/MPS-aware
tensor op in `ferrotorch-core` dispatches through `MtlBackend` instead
of the CPU path. The meta-crate's `pub use` re-export is therefore
the boundary surface where users call `ferrotorch::init_mps_backend()`.

## Parity contract

`parity_ops = []` for this route. The crate root is a lifecycle facade
— it has no parity op of its own. The per-op parity contract lives on
the individual kernel ops in `ferrotorch-core` (add, mul, matmul,
softmax, sum) which `MtlBackend`'s `impl GpuBackend` forwards into.

Edge cases preserved at the crate-root level:

- **Non-macOS host**: every public function returns the
  no-device stub value (`false` / `0` / `Err(DeviceUnavailable)`)
  unconditionally — no panic, no silent CPU fallback, no `cfg_attr`
  trickery that would let a downstream caller observe different
  behaviour on different platforms.
- **macOS without Metal device**: e.g. CI runner without GPU passthrough.
  `is_mps_available()` returns `false`, `mps_device_count()` returns
  `0`, `MpsDevice::new(0)` returns `Err(DeviceUnavailable)`,
  `init_mps_backend()` returns `Err(DeviceUnavailable)`. The same
  contract as non-macOS but reached via the live
  `MTLCreateSystemDefaultDevice` check.
- **macOS with Metal**: single-device contract (`Apple Silicon`).
  `MpsDevice::new(0) == Ok`, `MpsDevice::new(7) == Err(InvalidArgument)`.
  Multi-GPU systems (Intel Mac with discrete + integrated) are not
  exposed; upstream's `at::mps::MPSDevice::getInstance()` is also a
  singleton (`aten/src/ATen/mps/MPSDevice.h:48`).
- **Display string stability**: `format!("{d}", d = MpsDevice {
  ordinal: 0 })` produces exactly `"mps:0"`, matching
  `Device::Mps(0).to_string()` and `torch.device('mps:0').__str__()`.

## Verification

Unit tests in `mod tests in lib.rs` (5 tests):

- `is_mps_available_false_on_non_apple` — non-macOS branch.
- `mps_device_new_non_macos_returns_unavailable` (cfg-gated) —
  non-macOS `MpsDevice::new` returns `DeviceUnavailable`.
- `mps_device_new_macos_returns_ok_when_available` (cfg-gated) —
  macOS branch correlates with `is_mps_available()`.
- `mps_device_new_macos_rejects_nonzero_ordinal` (cfg-gated) — single-device
  contract on Apple Silicon.
- `mps_device_count_is_zero_on_non_macos` (cfg-gated) — count
  stub on non-macOS.
- `mps_device_count_macos_matches_metal_availability` (cfg-gated).
- `init_mps_backend_contract` — single test exercising both
  cfg-gated branches.
- `device_mps_marker_round_trips` — `Device::Mps(0).is_mps()` true,
  Display `mps:0`.

Integration tests in `ferrotorch-mps/tests/conformance_mps.rs` (27
tests) drive the same surface through serialized fixtures and exercise
the macOS-only branches via cascade_skip on non-Apple hosts.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-mps --no-default-features 2>&1 | tail -5
cargo check -p ferrotorch-mps --no-default-features 2>&1 | tail -3
cargo clippy -p ferrotorch-mps --no-default-features -- -D warnings 2>&1 | tail -3
```

Expected: each command's tail prints `Finished` / `test result: ok` /
no `error:` / `warning:` lines beyond the documented allows.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: lint baseline at top of `lib.rs` — `#![warn(clippy::all, clippy::pedantic)]` + `#![deny(rust_2018_idioms, missing_debug_implementations, missing_docs)]` + `#![cfg_attr(not(target_os = "macos"), deny(unsafe_code))]` + per-item `#![allow(...)]` block each with inline justification; non-test consumer: every `ferrotorch-mps/src/*.rs` file compiles under this baseline (`backend.rs` uses the `ref_as_ptr` / `borrow_as_ptr` allowance for Metal `setBytes_length_atIndex` calls). |
| REQ-2 | SHIPPED | impl: `#[cfg(target_os = "macos")] pub mod backend in lib.rs` + matching `#[cfg(target_os = "macos")] pub use backend::MtlBackend in lib.rs`; non-test consumer: `ferrotorch/src/lib.rs:137` does `pub use ferrotorch_mps::*;` and propagates the `MtlBackend` symbol only to consumers building for macOS — Linux/WSL workspace builds (verified by `cargo check -p ferrotorch-mps --no-default-features`) compile clean. |
| REQ-3 | SHIPPED | impl: free `pub fn is_mps_available in lib.rs`, `pub fn mps_device_count in lib.rs`, `pub fn init_mps_backend in lib.rs` mirroring `torch/mps/__init__.py:25-27` (`device_count`) and `aten/src/ATen/mps/MPSDevice.h:79` (`is_available`); non-test consumer: `ferrotorch/src/lib.rs:137` `pub use ferrotorch_mps::*;` re-exports them to application code as `ferrotorch::init_mps_backend()` / `ferrotorch::is_mps_available()`. |
| REQ-4 | SHIPPED | impl: `pub struct MpsDevice in lib.rs` with `Copy + Hash`, `pub fn MpsDevice::new in lib.rs` returning `Ok` only on macOS with Metal at ordinal 0, `Err(InvalidArgument)` for non-zero ordinals (mirrors `at::mps::MPSDevice::getInstance` singleton from `aten/src/ATen/mps/MPSDevice.mm:18-21`); non-test consumer: re-exported via `ferrotorch::MpsDevice` through the meta-crate glob in `ferrotorch/src/lib.rs:137`. |
| REQ-5 | SHIPPED | impl: `impl fmt::Display for MpsDevice in lib.rs` prints `"mps:{}"`; non-test consumer: `ferrotorch_core::Device::Mps(_)` Display impl in `ferrotorch-core/src/device.rs` uses the same format, asserted by `device_mps_marker_round_trips` and by `conformance_mps::mps_device_display_matches_fixture` against the live torch `torch.device('mps:0').__str__()` fixture; meta-crate re-export at `ferrotorch/src/lib.rs:137` propagates the type to application code. |
