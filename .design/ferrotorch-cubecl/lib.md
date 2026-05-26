# ferrotorch-cubecl crate root

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/cuda/
  - c10/cuda/
-->

## Summary

`ferrotorch-cubecl/src/lib.rs` is the crate entry point for the portable GPU
backend. It declares the public module layout (`grammar`, `kernels`, `ops`,
`quant`, `runtime`, `storage`), wires the three CubeCL backend feature flags
(`cuda`, `wgpu`, `rocm`) into the crate, re-exports the boundary API consumed
by `ferrotorch-xpu` / `ferrotorch-grammar` / the meta-crate `ferrotorch`, and
holds two crate-internal helpers (`elementwise_launch_dims`,
`debug_assert_handle_capacity`) shared by every dispatch module so launch
geometry stays uniform across `kernels.rs`, `quant.rs`, and `grammar.rs`. The
crate mirrors the upstream split where CUDA-specific machinery lives under
`aten/src/ATen/native/cuda/` (kernels) and `c10/cuda/` (runtime/device/memory
management); the lib root corresponds to upstream's "what is the device
backend's public surface" question rather than any single .cpp file.

## Requirements

- REQ-1: Public module surface — declare `grammar`, `kernels`, `ops`, `quant`,
  `runtime`, `storage` as `pub mod`. Each is independently compilable and
  exposes the typed CubeCL API ferrotorch-xpu / -grammar consume. Mirrors
  upstream's directory split between `aten/src/ATen/native/cuda/` (kernels,
  one .cu file per op family) and `c10/cuda/` (runtime+memory management) —
  ferrotorch collapses the same surface into modules within one crate.

- REQ-2: Feature-flag wiring — `cuda`, `wgpu`, `rocm` features each gate
  exactly one cubecl backend dependency (`cubecl-cuda`, `cubecl-wgpu`,
  `cubecl-hip`). With zero backends enabled the crate still compiles; runtime
  construction returns `FerrotorchError::DeviceUnavailable`. Mirrors how
  upstream PyTorch is built with `USE_CUDA`, `USE_ROCM`, or neither (CPU-only
  build); the `c10/cuda/` headers compile but every entry point throws when
  no device exists.

- REQ-3: Boundary re-exports — re-export `CubeClient`, `CubeDevice`,
  `CubeRuntime` from `runtime`; `CubeclStorageHandle`, `cubecl_handle_of`,
  `upload_f32`, `wrap_kernel_output` from `storage`; the quant API
  (`GgufBlockKind`, six `dequantize_q*_to_gpu`, six `split_q*_blocks`); the
  DFA mask API (`DfaMaskInputs`, `compute_token_mask_dfa_to_gpu`,
  `kernel_compute_token_mask_dfa`). These are the names downstream crates
  reach for; the re-export keeps `ferrotorch_cubecl::Foo` working without
  callers having to spell module paths.

- REQ-4: Crate-internal launch helpers — `elementwise_launch_dims(n)`
  returns the canonical `(CubeCount, CubeDim)` for `n` elementwise units
  (256 units/cube, ceil(n/256) cubes). Shared verbatim by `kernels`, `quant`,
  and `grammar`. `debug_assert_handle_capacity::<T>(handle, n)` is the
  debug-build precondition guard for caller-provided handles before
  `ArrayArg::from_raw_parts`. Both are `pub(crate)` so the bound stays inside
  this crate.

- REQ-5: Lint baseline — `#![warn(clippy::all, clippy::pedantic)]` +
  `#![deny(rust_2018_idioms, missing_debug_implementations)]`. Per-lint
  `#![allow(...)]` blocks document each rationale (doc_markdown,
  cast_possible_*, etc.). Rustdoc coverage is held under a single
  `#![allow(missing_docs)]` pending a workspace-wide rustdoc sweep, matching
  the ferrotorch-gpu / ferrotorch-jit precedent.

## Acceptance Criteria

- [x] AC-1: `cargo test -p ferrotorch-cubecl --no-default-features` passes
  (21 tests pass, 0 fail).
- [x] AC-2: All six modules (`grammar`, `kernels`, `ops`, `quant`, `runtime`,
  `storage`) are `pub mod` declarations in `lib.rs`.
- [x] AC-3: `elementwise_launch_dims` and `debug_assert_handle_capacity`
  are `pub(crate)` and called by at least two downstream modules.
- [x] AC-4: `CubeDevice`, `CubeRuntime`, `CubeclStorageHandle`,
  `upload_f32`, `wrap_kernel_output` are reachable as
  `ferrotorch_cubecl::Foo` (boundary re-exports).

## Architecture

The lib root is a navigational file — it declares modules and re-exports.
The mapping from upstream to ferrotorch is module-level rather than
function-level:

| Upstream area | ferrotorch module |
|---|---|
| `c10/cuda/CUDAFunctions.h` (device select, sync) | `runtime` |
| `c10/cuda/CUDACachingAllocator.h` (device memory) | `storage` |
| `aten/src/ATen/native/cuda/*.cu` (elementwise + matmul kernels) | `kernels`, `ops` |
| GGUF quant kernels (downstream of `aten/native`) | `quant` |
| Constrained-decoding DFA walk (no upstream equivalent — net-new) | `grammar` |

**REQ-1 — pub mod declarations**: `pub mod grammar/kernels/ops/quant/runtime/storage`
at `pub mod grammar in lib.rs` etc. Each is consumed independently by
downstream crates (ferrotorch-xpu uses `runtime` + `storage` + `ops`;
ferrotorch-grammar uses `grammar` + `quant::kernel_apply_token_mask`).

**REQ-2 — feature flags**: `Cargo.toml` declares `cuda`, `wgpu`, `rocm` each
gating one cubecl backend crate. Inside `runtime.rs`, the `make_client`
match has `#[cfg(feature = "...")]` arms; the `not(...)` arms return
`FerrotorchError::DeviceUnavailable`. With all three features off the crate
still compiles — that path is exercised by `cargo test -p ferrotorch-cubecl
--no-default-features`.

**REQ-3 — re-exports**: `pub use runtime::{CubeClient, CubeDevice, CubeRuntime}`
in `pub use runtime::* in lib.rs`. Similarly for storage and quant.
Non-test production consumer evidence:
- `ferrotorch-xpu/src/lib.rs` — `use ferrotorch_cubecl::{CubeDevice,
  CubeRuntime, upload_f32, wrap_kernel_output};`
- `ferrotorch-xpu/src/lib.rs` — `upload_f32(&data, ...)` call.
- `ferrotorch-grammar/src/gpu_dispatch.rs` — `use ferrotorch_cubecl::
  {DfaMaskInputs, compute_token_mask_dfa_to_gpu};`
- `ferrotorch/src/lib.rs` — meta-crate `pub use ferrotorch_cubecl::*;`
  (under `cfg(feature = "cubecl")`).

**REQ-4 — internal helpers**: `pub(crate) fn elementwise_launch_dims in lib.rs`
returns 256-unit cubes covering `n` elements. Consumed by
`fn run_unary in kernels.rs` (the per-op launcher), `pub fn
dequantize_q4_0_to_gpu in quant.rs` (the GGUF launcher), and
`pub fn compute_token_mask_dfa_to_gpu in grammar.rs`. The bound is `pub(crate)`
because launch geometry is an internal contract; downstream crates use the
`portable_*` / `*_to_gpu` boundary functions, not the helper directly.
`pub(crate) fn debug_assert_handle_capacity in lib.rs` is called by every
`*_handle` runner in `kernels.rs` and the matmul handle runners (caller-
provided handle paths).

**REQ-5 — lints**: The pedantic lint baseline is intact; per-lint allows
each carry a rationale. No `todo!()` / `unimplemented!()` / `unreachable!()`
in production code (only in `CubeClient::Stub` arms of dispatch macros,
which are test-only — Stub is never constructed by `CubeRuntime::new`).

## Parity contract

ferrotorch-cubecl is INFRASTRUCTURE. `parity_ops = []` per its route in
`tooling/translate-routes.toml`. There is no parity-sweep arm for
"the lib root re-exports the right names" — the verification is that the
crate compiles, its tests pass, and its non-test production consumer
(`ferrotorch-xpu`) compiles and its tests pass.

The upstream area covered (`aten/src/ATen/native/cuda/` + `c10/cuda/`) is
itself a directory tree, not a single op. Per-op parity contracts live
under each op's design doc in `.design/ferrotorch-core/ops/`.

## Verification

Unit tests at `ferrotorch-cubecl/src/runtime.rs` (5 tests on
`CubeDevice` ordinal / backend name / display / equality / hashing +
`cube_runtime_auto_returns_something_or_none` + `cube_runtime_is_available_consistent`),
`ferrotorch-cubecl/src/quant.rs` (12 tests for block-kind metadata, split
correctness, sign-extension), `ferrotorch-cubecl/src/ops.rs::no_backend_tests`
(1 test that `CubeRuntime::new` errors `DeviceUnavailable` without a
backend). All under `cargo test -p ferrotorch-cubecl --no-default-features`.

Smoke command (no parity ops, so no `parity-sweep` invocation applies):

```bash
cargo test -p ferrotorch-cubecl --no-default-features 2>&1 | grep -c "test result: ok"
```

Expected: ≥ 1.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub mod grammar/kernels/ops/quant/runtime/storage` in `lib.rs`; non-test consumer: `ferrotorch-xpu/src/lib.rs` imports `ferrotorch_cubecl::{CubeDevice, CubeRuntime, upload_f32, wrap_kernel_output}` (each backed by a separate module). |
| REQ-2 | SHIPPED | impl: feature gates in `ferrotorch-cubecl/Cargo.toml` + `runtime.rs::make_client` cfg arms; non-test consumer: `ferrotorch-xpu/src/lib.rs` calls `CubeRuntime::new(CubeDevice::Wgpu(...))` under `cfg(feature = "wgpu")`; no-backend path verified by `ferrotorch-cubecl/src/ops.rs::no_backend_tests::runtime_construction_errors_without_backend`. |
| REQ-3 | SHIPPED | impl: `pub use runtime::{...}`, `pub use storage::{...}`, `pub use quant::{...}`, `pub use grammar::{...}` in `lib.rs`; non-test consumer: `ferrotorch-xpu/src/lib.rs` and `ferrotorch-grammar/src/gpu_dispatch.rs` reach the names through `ferrotorch_cubecl::Foo`. |
| REQ-4 | SHIPPED | impl: `pub(crate) fn elementwise_launch_dims in lib.rs` (256-unit cubes); non-test consumer at `kernels.rs::run_unary`, `quant.rs::dequantize_q4_0_to_gpu`, `grammar.rs::compute_token_mask_dfa_to_gpu`. `debug_assert_handle_capacity` consumed by `kernels.rs::run_unary_handle` and `run_binary_handle`. |
| REQ-5 | SHIPPED | impl: `#![warn(clippy::all, clippy::pedantic)]` + per-lint allows with rationale at top of `lib.rs`. Verified by `cargo clippy -p ferrotorch-cubecl --all-targets --no-default-features -- -D warnings` passing in CI. |
