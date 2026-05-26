# ferrotorch-jit — crate root

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/jit/__init__.py
  - torch/fx/__init__.py
  - torch/export/__init__.py
  - torch/csrc/jit/api/module.cpp
-->

## Summary

`ferrotorch-jit/src/lib.rs` is the crate root for the tracing JIT compiler
and graph optimiser. It declares every module the crate exposes (`graph`,
`trace`, `module`, `interpreter`, `export`, `aot_autograd`, `graph_break`,
`symbolic`, `serialize`, `nvrtc`, plus codegen / fusion / optimize /
autotune / memory_plan), re-exports the user-facing surface, and pins
the crate-wide lint baseline. It mirrors the role of upstream
`torch.jit`, `torch.fx`, and `torch.export` package init files
(`torch/jit/__init__.py:1-300`, `torch/fx/__init__.py`,
`torch/export/__init__.py:1-466`), which serve the same purpose: collect
the JIT pipeline's user-visible names into one place.

## Requirements

- REQ-1: Crate-level documentation describing the JIT pipeline's
  high-level shape (trace → IR → optimise → codegen / interpret /
  export) so first-time readers can orient via `cargo doc` (mirrors
  the module docstring at `torch/jit/__init__.py:1-30`).
- REQ-2: A pinned lint baseline that enables `clippy::all +
  clippy::pedantic`, denies `unsafe_code` / `missing_docs` /
  `missing_debug_implementations`, and documents every `#![allow]`
  with a one-line rationale (mirrors the workspace discipline; no
  PyTorch analog).
- REQ-3: `pub mod` declarations for every JIT-pipeline module shipped
  by the crate (`graph`, `trace`, `module`, `interpreter`, `optimize`,
  `fusion`, `dag_fusion`, `codegen*`, `aot_autograd`, `symbolic`,
  `export`, `graph_break`, `serialize`, `nvrtc`, `error`, `autotune`,
  `memory_plan`).
- REQ-4: Feature-gated `fusion_gpu` module (only declared under
  `#[cfg(feature = "cuda")]`) so the default workspace build is
  CUDA-toolkit-free, matching the PyTorch convention of opt-in CUDA
  modules.
- REQ-5: `pub use` re-exports of the user-facing API: `trace`,
  `compile`, `compile_with_config`, `TracedModule`, `AotCompiledModule`,
  `interpret*`, `export`, `ExportedProgram`, `compile_aot`,
  `AotGraphPair`, `compile_symbolic`, `SymbolicTracedModule`,
  `Guard`, `trace_with_breaks`, `SegmentedModule`, `JitError`, plus
  the codegen / fusion / autotune / memory-plan surfaces.

## Acceptance Criteria

- [x] AC-1: `cargo doc -p ferrotorch-jit --no-deps` succeeds with
  `missing_docs` denied.
- [x] AC-2: `cargo check -p ferrotorch-jit` succeeds on the default
  feature set (no `cuda`).
- [x] AC-3: `cargo check -p ferrotorch-jit --features cuda` succeeds
  with the `fusion_gpu` module included.
- [x] AC-4: `cargo clippy -p ferrotorch-jit --lib -- -D warnings`
  succeeds at the documented allow-list baseline.
- [x] AC-5: Every `#![allow]` at crate root carries a comment
  explaining why the alternative is worse.

## Architecture

The crate root sits at `ferrotorch-jit/src/lib.rs`. The crate-level
doc-comment (`//!` block at the top of the file) is the rustdoc
landing page. The lint configuration sits immediately below it:
`#![warn(clippy::all, clippy::pedantic)]`,
`#![deny(unsafe_code, rust_2018_idioms, missing_debug_implementations)]`,
`#![deny(missing_docs)]`, then the documented `#![allow]` block.

The `#![allow]` items at crate root (REQ-2) are exempted from the
anti-pattern-gate rule `R-CODE-3` because each carries an inline
rationale and the allow is genuinely needed at crate scope:
`module_name_repetitions` (IR taxonomy), `cast_possible_*` family
(codegen indices), `format_push_string` (codegen string builders),
`must_use_candidate` (getter noise), `manual_let_else` (match-arm
readability), and the rest documented in the file.

The `pub mod` declarations enumerate every JIT-pipeline component;
the `pub use` block downstream collects the user-facing entry points
into the crate root for ergonomic `use ferrotorch_jit::trace;` style
imports — mirroring PyTorch's `torch.jit.trace`, `torch.compile`,
`torch.export.export` etc.

The CUDA feature gate at the `fusion_gpu` declaration is the single
opt-in path that turns on NVRTC + cudarc dependencies; without the
feature, callers that attempt GPU codegen receive
`JitError::GpuBackendUnavailable` from the affected backends. Mirrors
the way `torch.cuda` is conditionally available based on the CUDA
toolkit's presence.

### Non-test production consumers

Every downstream crate that depends on `ferrotorch-jit` (e.g.
`ferrotorch-serialize` for ONNX export from `ExportedProgram`, the
top-level `ferrotorch` meta-crate's re-exports) imports through this
file. The re-exports at `ferrotorch-jit/src/lib.rs:87-117` are the
non-test consumer surface for every public symbol declared in the
crate.

## Parity contract

`parity_ops = []`. The crate root composes — no parity ops live
directly in `lib.rs`. Its correctness is the union of the modules it
declares.

## Verification

The lint baseline is verified by `cargo clippy -p ferrotorch-jit --lib
-- -D warnings`. Compilation is verified by `cargo check
-p ferrotorch-jit` (default features) and `cargo check
-p ferrotorch-jit --features cuda` (CUDA path). The rustdoc surface
is verified by `cargo doc -p ferrotorch-jit --no-deps`. All three
commands must succeed before a `lib.rs` change can ship.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: crate doc-comment `//! ...` in `lib.rs`; consumer: every downstream crate's `use ferrotorch_jit::...` import resolves through this file. Upstream parallel `torch/jit/__init__.py:1-30`. |
| REQ-2 | SHIPPED | impl: `#![warn(clippy::all, clippy::pedantic)]` + `#![deny(unsafe_code, missing_docs, ...)]` block in `lib.rs`; consumer: the workspace clippy gauntlet picks the baseline up when linting `ferrotorch-jit`. |
| REQ-3 | SHIPPED | impl: every `pub mod` declaration in `lib.rs` (`pub mod aot_autograd; pub mod autotune; ...`); consumer: downstream consumer crates plus the JIT-internal modules importing each other (`use crate::graph::IrGraph;` etc.). |
| REQ-4 | SHIPPED | impl: `#[cfg(feature = "cuda")] pub mod fusion_gpu;` in `lib.rs`; consumer: `ferrotorch-jit/src/fusion.rs:1220` references `crate::nvrtc::compile_cuda_source_to_ptx` and the CUDA feature is the surface that flips the path on. |
| REQ-5 | SHIPPED | impl: `pub use trace::trace;`, `pub use module::{TracedModule, AotCompiledModule, compile, compile_with_config};`, the `pub use export::...`, `pub use graph_break::...`, `pub use aot_autograd::...`, `pub use symbolic::...`, `pub use error::JitError;` blocks in `lib.rs`; consumer: every downstream user of `ferrotorch-jit` imports through these re-exports. |
