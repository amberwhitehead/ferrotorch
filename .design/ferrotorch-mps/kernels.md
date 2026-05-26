# MSL kernel source registry — `ferrotorch-mps/src/kernels/mod.rs`

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - aten/src/ATen/native/mps/OperationUtils.h
  - aten/src/ATen/native/mps/OperationUtils.mm
  - aten/src/ATen/native/mps/MetalShaderLibrary.h
-->

## Summary

`ferrotorch-mps/src/kernels/mod.rs` is a thin module that declares six
`pub const NAME: &str = include_str!("...")` MSL kernel sources covering
Sprint C.7's 10 GpuBackend ops (matmul, bmm, add/sub/mul/div, relu,
sigmoid, softmax, sum_axis). The six `.metal` files live alongside
`mod.rs` and are embedded into the Rust binary at build time via
`include_str!`, so no external Metal artifact must ship with the
binary; the macOS `backend` module compiles them at runtime via
`MTLDevice::newLibraryWithSource_options_error`. Mirrors PyTorch's
`MetalShaderLibrary` from
`aten/src/ATen/native/mps/MetalShaderLibrary.h` — the upstream
equivalent loads MSL strings from C++ static data and compiles them
on demand.

## Requirements

- REQ-1: Six MSL source constants. `MATMUL_F32`, `BMM_F32`,
  `ELEMENTWISE_F32`, `ACTIVATIONS_F32`, `SOFTMAX_F32`, `SUM_AXIS_F32`
  — each a `pub const &str` bound via `include_str!` to the
  sibling `.metal` file. Together they cover the 10 Sprint C.7 GpuBackend
  trait methods.

- REQ-2: Build-time embedding. Each constant uses
  `include_str!("<name>.metal")` so the MSL source ships inside the
  Rust binary and `MTLDevice::newLibraryWithSource_options_error`
  has the bytes available at runtime without any filesystem lookup.

- REQ-3: Platform-agnostic compilation. The module compiles on every
  host (Linux/WSL, macOS, etc.); only the `backend` module that
  consumes the constants is `#[cfg(target_os = "macos")]`-gated.
  This lets workspace-wide tests (e.g.
  `kernel_source_matmul_f32_present` in
  `ferrotorch-mps/tests/conformance_mps.rs`) verify the sources are
  non-empty and declare their expected kernel function names without
  needing a Metal device.

- REQ-4: Kernel-function-name catalogue in the doc-comment. The
  module's top-level `//!` doc-comment names every kernel function
  declared inside each `.metal` file plus its `torch.mps` upstream
  analogue. This is the contract the macOS `backend` module relies on
  when it calls `lib.newFunctionWithName(name)` with each function
  string literal.

## Acceptance Criteria

- [x] AC-1: 6 `pub const &str` symbols at `kernels/mod.rs` —
  `MATMUL_F32`, `BMM_F32`, `ELEMENTWISE_F32`, `ACTIVATIONS_F32`,
  `SOFTMAX_F32`, `SUM_AXIS_F32`.
- [x] AC-2: Each constant is bound via `include_str!` to a sibling
  `.metal` file (`matmul_f32.metal`, `bmm_f32.metal`,
  `elementwise_f32.metal`, `activations_f32.metal`,
  `softmax_f32.metal`, `sum_axis_f32.metal`).
- [x] AC-3: The module's `//!` doc-comment includes the kernel
  catalogue table mapping each constant to its kernel function
  name(s) and `torch.mps` analogue.
- [x] AC-4: `cargo test -p ferrotorch-mps --no-default-features`
  passes all six `kernel_source_*_present` conformance tests
  (verifies non-empty + contains expected function name).
- [x] AC-5: `cargo check -p ferrotorch-mps --no-default-features`
  succeeds on Linux/WSL (proves the module is platform-agnostic).

## Architecture

### File layout

`ferrotorch-mps/src/kernels/` contains:

- `mod.rs` — this module; 36 LOC, six const declarations + the
  doc-comment.
- `matmul_f32.metal` — MSL GEMM `kernel void matmul_f32(...)`.
- `bmm_f32.metal` — MSL batched GEMM `kernel void bmm_f32(...)`.
- `elementwise_f32.metal` — four binary kernels:
  `kernel void add_f32(...)`, `sub_f32(...)`, `mul_f32(...)`,
  `div_f32(...)`.
- `activations_f32.metal` — `kernel void relu_f32(...)`,
  `sigmoid_f32(...)`.
- `softmax_f32.metal` — `kernel void softmax_f32(...)` (last-dim
  softmax with pow-2 tree reduction).
- `sum_axis_f32.metal` — `kernel void sum_axis_f32(...)` (axis
  reduction with pow-2 tree reduction).

### Build-time embedding (REQ-2)

Each `pub const &str` is declared as
`pub const NAME: &str = include_str!("name.metal");`. `include_str!`
is a compile-time macro that resolves the path relative to the
declaring file and embeds the bytes as a string literal. The result
is that:

- The MSL source is embedded into the Rust binary at compile time.
- No filesystem lookup is needed at runtime.
- The `.metal` files MUST exist at compile time; a missing file is a
  compile error, not a runtime error.
- The build is hermetic — the Rust binary alone is sufficient to
  reconstruct the MSL source for any of the 10 Sprint C.7 ops.

### Platform-agnostic compilation (REQ-3)

The module has no `cfg(target_os = ...)` gate. The MSL strings are
plain `&'static str` data on every platform; only the `backend`
module that *compiles* them via `objc2-metal` is gated. This means:

- `cargo check -p ferrotorch-mps --no-default-features` on
  Linux/WSL compiles the module unchanged.
- The kernel-source-presence conformance tests
  (`kernel_source_matmul_f32_present`,
  `kernel_source_bmm_f32_present`,
  `kernel_source_elementwise_f32_present`,
  `kernel_source_activations_f32_present`,
  `kernel_source_softmax_f32_present`,
  `kernel_source_sum_axis_f32_present`) run on every platform and
  verify the embedded source is non-empty and contains the expected
  kernel function name. A regression where a `.metal` file is
  accidentally blanked or renamed surfaces immediately on Linux CI.

### Kernel catalogue (REQ-4)

The module's `//!` doc-comment carries a table:

| Constant | Function name(s) | torch.mps analogue |
|---|---|---|
| `MATMUL_F32` | `matmul_f32` | `torch.mm` / `torch.matmul` |
| `BMM_F32` | `bmm_f32` | `torch.bmm` |
| `ELEMENTWISE_F32` | `add_f32`, `sub_f32`, `mul_f32`, `div_f32` | `torch.add/sub/mul/div` |
| `ACTIVATIONS_F32` | `relu_f32`, `sigmoid_f32` | `torch.relu`, `torch.sigmoid` |
| `SOFTMAX_F32` | `softmax_f32` | `torch.softmax` |
| `SUM_AXIS_F32` | `sum_axis_f32` | `torch.sum(dim=axis)` |

This is the load-bearing contract the macOS `backend` module relies
on. `fn compile_pipeline in backend.rs` is called with each
`(MSL_SOURCE_CONST, fn_name_str_literal)` pair (e.g.
`compile_pipeline(&device, kernels::ELEMENTWISE_F32, "add_f32")?`),
so any drift between a function name in a `.metal` file and the string
literal in `backend.rs` produces a compile-time error from
`newFunctionWithName` returning None — surfaced as
`FerrotorchError::InvalidArgument { message: "MSL function `<name>` not
found in library" }`.

### Non-test production consumers

The single non-test production consumer is `backend.rs`. Each of the 6
constants is referenced in `pub fn MtlBackend::new in backend.rs`'s
sequence of `compile_pipeline` calls:

- `kernels::MATMUL_F32` → matmul pipeline.
- `kernels::BMM_F32` → bmm pipeline.
- `kernels::ELEMENTWISE_F32` → 4 pipelines (add, sub, mul, div all
  bound to the same library; the function-name string selects which
  kernel to extract).
- `kernels::ACTIVATIONS_F32` → 2 pipelines (relu, sigmoid).
- `kernels::SOFTMAX_F32` → softmax pipeline.
- `kernels::SUM_AXIS_F32` → sum_axis pipeline.

The constants are also re-exported as `ferrotorch_mps::kernels::*`
through `pub mod kernels in lib.rs` and consumed by the conformance
suite at `ferrotorch-mps/tests/conformance_mps.rs:35`
(`use ferrotorch_mps::{...kernels...};`) for the platform-agnostic
presence checks.

## Parity contract

`parity_ops = []` for this route. The kernel sources are static MSL
bytes; per-op parity (`matmul`, `bmm`, `add`, `sub`, `mul`, `div`,
`relu`, `sigmoid`, `softmax`, `sum`) is tracked at the kernel-owning
op layer in `ferrotorch-core`. This module's only contract is that
the embedded MSL source declares the expected kernel function names
(verified by the `kernel_source_*_present` conformance tests on
every platform).

Edge cases preserved at the module level:

- **Missing `.metal` file**: a removed or renamed file produces a
  compile-time error from `include_str!`, not a runtime
  failure-to-find. The Rust build is the verification surface.
- **Renamed kernel function inside a `.metal` file**: the file still
  exists and `include_str!` succeeds, but
  `kernel_source_*_present` tests fail on every platform because
  the embedded source no longer contains the expected function name
  string. Linux CI catches it before macOS ever sees a runtime
  `newFunctionWithName` failure.
- **Empty `.metal` file**: `kernel_source_*_present` tests assert
  `!source.is_empty()` so a blanked file fails immediately on
  every platform.
- **MSL compilation failure on macOS**: not a contract this module
  enforces; it surfaces in `MtlBackend::new()` via
  `compile_pipeline` returning
  `FerrotorchError::InvalidArgument { message: "MSL compile failed
  for `<name>`: ..." }`.

## Verification

Unit tests in `ferrotorch-mps/tests/conformance_mps.rs` (the integration
suite for the crate; the module itself has no `#[cfg(test)] mod tests`
block because the constants are tested through the integration suite
for visibility / no-cfg-gate purity):

- `kernel_source_matmul_f32_present` — `MATMUL_F32` non-empty + contains
  `"matmul_f32"`.
- `kernel_source_bmm_f32_present` — `BMM_F32` non-empty + contains
  `"bmm_f32"`.
- `kernel_source_elementwise_f32_present` — `ELEMENTWISE_F32` contains
  `"add_f32"`, `"sub_f32"`, `"mul_f32"`, `"div_f32"`.
- `kernel_source_activations_f32_present` — `ACTIVATIONS_F32` contains
  `"relu_f32"`, `"sigmoid_f32"`.
- `kernel_source_softmax_f32_present` — `SOFTMAX_F32` contains
  `"softmax_f32"`.
- `kernel_source_sum_axis_f32_present` — `SUM_AXIS_F32` contains
  `"sum_axis_f32"`.

These 6 tests run on every platform (Linux/WSL/macOS) and are the
load-bearing regression guard for the kernel catalogue.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-mps --no-default-features kernel_source 2>&1 | tail -5
cargo check -p ferrotorch-mps --no-default-features 2>&1 | tail -3
```

Expected: 6 `test result: ok` lines for the kernel_source_* tests +
`Finished` from cargo check.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub const MATMUL_F32`, `BMM_F32`, `ELEMENTWISE_F32`, `ACTIVATIONS_F32`, `SOFTMAX_F32`, `SUM_AXIS_F32` in `kernels/mod.rs` covering Sprint C.7's 10 ops; non-test consumer: `pub fn MtlBackend::new in backend.rs` calls `compile_pipeline(&device, kernels::MATMUL_F32, "matmul_f32")?` through `compile_pipeline(&device, kernels::SUM_AXIS_F32, "sum_axis_f32")?` to compile the 10 pipelines. |
| REQ-2 | SHIPPED | impl: every constant uses `include_str!("<name>.metal")` in `kernels/mod.rs`; non-test consumer: the embedded `&'static str` is consumed verbatim by `MTLDevice::newLibraryWithSource_options_error` inside `fn compile_pipeline in backend.rs` — no filesystem lookup at runtime, no external Metal artifact. |
| REQ-3 | SHIPPED | impl: `kernels/mod.rs` has no `cfg(target_os = "macos")` gate (only the `backend` module that *consumes* the constants is gated); non-test consumer: `cargo check -p ferrotorch-mps --no-default-features` on Linux/WSL compiles the module, and the 6 `kernel_source_*_present` tests in `ferrotorch-mps/tests/conformance_mps.rs` exercise the constants on Linux without a Metal device. |
| REQ-4 | SHIPPED | impl: the `//!` doc-comment of `kernels/mod.rs` contains the kernel-catalogue table mapping each constant to its declared kernel function name(s) and `torch.mps` analogue; non-test consumer: `fn compile_pipeline in backend.rs` is called with each `(MSL_CONST, fn_name_literal)` pair following the catalogue (matmul_f32, bmm_f32, add_f32, sub_f32, mul_f32, div_f32, relu_f32, sigmoid_f32, softmax_f32, sum_axis_f32); a drift between the catalogue function name and the actual `kernel void <name>` in the MSL file is caught by the 6 `kernel_source_*_present` tests. |
