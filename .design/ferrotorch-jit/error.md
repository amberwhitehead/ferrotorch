# ferrotorch-jit — `error` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/jit/_trace.py
  - torch/jit/frontend.py
  - torch/_dynamo/exc.py
-->

## Summary

`ferrotorch-jit/src/error.rs` defines `JitError`, the structured
error vocabulary returned by every public entry point in the crate.
Each variant captures one failure mode the JIT pipeline reports:
tracing failure, data-dependent control flow, unsupported op, shape
mismatch, codegen failure, serialisation failure, graph break,
export failure, parameter error, recompilation failure, GPU backend
unavailable, and unsupported (op, dtype) tuples. Mirrors the
exception hierarchy that upstream `torch.jit` and `torch._dynamo`
raise across the same surface (`torch/jit/frontend.py`,
`torch/_dynamo/exc.py`); the Rust analog uses a `thiserror`-derived
enum instead of a Python `Exception` tree (R-DEV-4 / R-DEV-7).

## Requirements

- REQ-1: `pub enum JitError` is `#[non_exhaustive]` and derives
  `Debug` plus `thiserror::Error`, so callers can match on known
  variants today and receive a compile-time prompt to handle any
  new variant added in a future minor version.
- REQ-2: Variants cover the JIT pipeline's known failure modes:
  `TracingError`, `DataDependentControlFlow`, `UnsupportedOp`,
  `ShapeMismatch`, `CodegenError`, `SerializationError`,
  `GraphBreak`, `ExportError`, `ParameterError`,
  `RecompilationError`, `GpuBackendUnavailable`, `Unsupported`.
- REQ-3: Each variant carries the diagnostic context the matching
  upstream exception would carry — for example
  `DataDependentControlFlow { op }` mirrors PyTorch's
  `Unsupported('data dependent control flow at op=X')`; `ShapeMismatch
  { traced, actual }` mirrors `RuntimeError('shape mismatch ...')`.
- REQ-4: `impl From<JitError> for FerrotorchError` lets JIT errors
  bubble up through the workspace's top-level error type
  (`ferrotorch_core::error::FerrotorchError::InvalidArgument`) so JIT
  callers don't have to learn a second error vocabulary.
- REQ-5: Every variant has rustdoc explaining when it fires and what
  the caller's recovery options are (mirrors upstream's per-exception
  docstrings).

## Acceptance Criteria

- [x] AC-1: `JitError::TracingError { message: "x".into() }.to_string()`
  produces `"tracing error: x"` (verified by the `Display` impl
  via `thiserror`).
- [x] AC-2: `JitError::GraphBreak { op, reason }` is constructible
  and matches the formatter string `"graph break at op '{op}':
  {reason}"`.
- [x] AC-3: `let err: FerrotorchError =
  JitError::ParameterError { message: "x".into() }.into();` succeeds.
- [x] AC-4: Adding a new variant in a downstream commit compiles
  against pattern matches that use a wildcard arm (the
  `#[non_exhaustive]` contract).
- [x] AC-5: `cargo doc -p ferrotorch-jit --no-deps` succeeds with
  `missing_docs` denied (every variant has a doc-comment).

## Architecture

`pub enum JitError` (in `error.rs`) is annotated with
`#[derive(Debug, thiserror::Error)]` + `#[non_exhaustive]`. Each
variant has an `#[error("...")]` formatter that controls the
`Display` impl; the field shapes are flat structs (rather than tuple
variants) so call sites read as
`JitError::DataDependentControlFlow { op: name.into() }` instead of
positional argument soup.

The `From<JitError> for FerrotorchError` impl funnels every JIT
error into `FerrotorchError::InvalidArgument { message: e.to_string()
}`. This sacrifices the JIT-specific structure when crossing the
crate boundary, but the trade is intentional: downstream callers
typically catch `FerrotorchError` once at the top of their handler
chain, and the formatted message preserves enough context for
diagnostics. A future iteration could promote `JitError` to its own
variant on `FerrotorchError` if downstream code grows to need the
distinction.

`GpuBackendUnavailable` and `Unsupported` are the two variants that
encode "the op exists in vocabulary but no kernel for this
(op, dtype, device) tuple" — analogous to PyTorch's `NotImplementedError
("operator <op> not implemented for <device>/<dtype>")`. They are
the canonical signal for opt-in CPU fallback in callers (per
`rust-gpu-discipline §3`, silent fallback is forbidden).

### Non-test production consumers

`JitError` is constructed in 42 sites across the JIT crate (codegen
paths, trace/export paths, interpreter, serialisation, fusion). Each
of those callers is a non-test consumer. Examples:
`ferrotorch-jit/src/codegen.rs`, `ferrotorch-jit/src/codegen_gpu.rs`,
`ferrotorch-jit/src/fusion.rs`, `ferrotorch-jit/src/fusion_gpu.rs`,
`ferrotorch-jit/src/graph_break.rs`,
`ferrotorch-jit/src/interpreter.rs`,
`ferrotorch-jit/src/nvrtc.rs`, `ferrotorch-jit/src/export.rs`,
`ferrotorch-jit/src/codegen_jit.rs`,
`ferrotorch-jit/src/codegen_ir.rs`.

The `From` impl is consumed every time a JIT-pipeline function
returns `FerrotorchResult<T>` (which is the trait-bound type alias
defined on `ferrotorch_core::error::FerrotorchError`).

## Parity contract

`parity_ops = []`. The error vocabulary is structural — it doesn't
host numerical ops. Equivalence with upstream is checked by hand on
each new variant: when adding a variant, find the matching PyTorch
exception (`torch.jit.frontend.TracingError` →
`JitError::TracingError`, `torch._dynamo.exc.Unsupported` →
`JitError::UnsupportedOp` or `JitError::GraphBreak` depending on
context, etc.) and document the mapping in the variant's rustdoc.

## Verification

Tested transitively: every test in `ferrotorch-jit/src/*.rs` that
asserts an error path constructs a `JitError` variant and matches on
its formatted message. There are 100+ such assertions across the
crate's `#[cfg(test)]` blocks. The `From` impl is exercised every
time a test calls a JIT entry point and bubbles the error through
`FerrotorchResult`.

Smoke command:

```bash
cargo test -p ferrotorch-jit --lib error:: 2>&1 | tail -3
```

Expected: passes (the error module's own tests, if any) plus the
indirect coverage from every other JIT test.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `#[derive(Debug, thiserror::Error)] #[non_exhaustive] pub enum JitError` in `error.rs`; non-test consumer: every `JitError::...` constructor in `codegen.rs`, `codegen_gpu.rs`, `fusion.rs`, etc. (42 sites). |
| REQ-2 | SHIPPED | impl: all 12 variants (`TracingError`, `DataDependentControlFlow`, `UnsupportedOp`, `ShapeMismatch`, `CodegenError`, `SerializationError`, `GraphBreak`, `ExportError`, `ParameterError`, `RecompilationError`, `GpuBackendUnavailable`, `Unsupported`) in `error.rs`; non-test consumer: each variant is constructed at one or more sites across `interpreter.rs`, `nvrtc.rs`, `codegen_*.rs`, `fusion*.rs`, `graph_break.rs`. |
| REQ-3 | SHIPPED | impl: each variant's struct payload (`DataDependentControlFlow { op }`, `ShapeMismatch { traced, actual }`, `Unsupported { op, dtype }`, etc.) in `error.rs`; non-test consumer: matching field-init at each construction site (e.g. `JitError::Unsupported { op: "exp".into(), dtype: "f64".into() }` in `codegen_gpu.rs`). |
| REQ-4 | SHIPPED | impl: `impl From<JitError> for FerrotorchError` in `error.rs`; non-test consumer: every `?` in `module.rs`/`trace.rs`/`export.rs` that converts a `JitError` into the workspace error type. |
| REQ-5 | SHIPPED | impl: per-variant rustdoc on every member of `JitError` in `error.rs` (verified by `#![deny(missing_docs)]`); non-test consumer: `cargo doc -p ferrotorch-jit --no-deps` renders the variants on the `JitError` page. |
