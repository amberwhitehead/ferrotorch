# ferrotorch-profiler — crate root (re-exports + feature gating)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/profiler/__init__.py
  - torch/autograd/profiler.py
-->

## Summary

`ferrotorch-profiler/src/lib.rs` is the crate root: it declares the
six submodules (`cuda_timing` behind the `cuda` feature, `event`,
`flops`, `profiler`, `report`, `schedule`), re-exports the public
API surface end users consume, and pins the crate-level lint
configuration (`warn(clippy::pedantic)`, `deny(unsafe_code)`).
Mirrors `torch/profiler/__init__.py`'s role as the public-surface
shim over the internal C++ / Python implementation modules.

## Requirements

- REQ-1: Crate-level lints fixed at the workspace's strict profile —
  `warn(clippy::all, clippy::pedantic)`,
  `warn(missing_debug_implementations, rust_2018_idioms)`,
  `deny(unsafe_code)`, with a documented per-crate
  `allow(clippy::module_name_repetitions)` override for the
  intentional `ProfileEvent` / `ProfileConfig` / `ProfileReport`
  naming. Override carries an inline justification.
- REQ-2: Submodule declarations match the source layout:
  `cuda_timing` is `pub mod` and gated `#[cfg(feature = "cuda")]`
  (it imports `cudarc`); `event`, `profiler`, `report` are private
  modules whose contents are re-exported selectively; `flops` and
  `schedule` are `pub mod` (users may name `flops::estimate`
  directly).
- REQ-3: Public re-export set covers the user-facing API:
  - `CudaKernelScope` (cuda feature only) from `cuda_timing`.
  - `DeviceType`, `GpuTimingPair`, `MemoryCategory`, `ProfileEvent`
    from `event`.
  - `ProfileConfig`, `Profiler`, `with_profiler` from `profiler`.
  - `OpSummary`, `ProfileReport` from `report`.
  - `ProfileSchedule`, `SchedulePhase` from `schedule`.
  Crate-internal `PendingCudaScope` is `pub(crate)` and NOT
  re-exported (verified in `cuda_timing.rs` line 132 docstring).
- REQ-4: Module-level doc-comment with a runnable doctest showing
  the `with_profiler` quick-start (the test harness compiles the
  example as part of `cargo test --doc`). Mirrors the doctring
  example pattern PyTorch uses on
  `torch.profiler.profile` (`torch/profiler/profiler.py:651`).
- REQ-5: Re-exports expose ONLY the items in REQ-3 — no
  `pub use crate::*` blanket and no leaking of `pub(crate)` types.
  The surface is auditable via `cargo public-api` and pinned by
  the `tests/conformance_surface_coverage.rs` integration test.

## Acceptance Criteria

- [x] AC-1: `cargo clippy -p ferrotorch-profiler --lib -- -D warnings`
  passes against the current code.
- [x] AC-2: `CudaKernelScope` is reachable via `ferrotorch_profiler::CudaKernelScope`
  when built with `--features cuda`, and absent without it.
- [x] AC-3: `ProfileEvent`, `Profiler`, `ProfileReport`,
  `ProfileSchedule` are all reachable at the crate root.
- [x] AC-4: The crate-root doctest compiles (`cargo test -p ferrotorch-profiler --doc`).
- [x] AC-5: `PendingCudaScope` is NOT in the public surface
  (verified by absence from `pub use` lines in `lib.rs`).

## Architecture

### Lint pin (REQ-1)

The lint block at the top of `lib.rs:1-4` is the strictest
configuration this workspace runs. `deny(unsafe_code)` rules out
the `unsafe` blocks that would otherwise appear in event-pointer
manipulation; the report module's hostname-env tests are the only
exception and they carry per-item `#[allow(unsafe_code)]` on the
test function. The `clippy::module_name_repetitions` allowance
documents that `ProfileConfig` / `ProfileEvent` / `ProfileReport`
share the crate name on purpose — every type belongs in the
`ferrotorch_profiler::Profile*` namespace and renaming would
fragment the prelude.

### Submodule visibility (REQ-2, REQ-3, REQ-5)

`event` and `profiler` are PRIVATE modules; users reach their
contents only through the explicit `pub use` lines. This means
adding a new field to `ProfileEvent` does not become a public API
break (the source path is hidden). `flops` and `schedule` are
`pub mod` because users legitimately call `flops::estimate("matmul", &shapes)`
and pattern-match on `schedule::SchedulePhase::*`.

The `cuda_timing` module is `pub mod` AND `#[cfg(feature = "cuda")]`.
This means user code that depends on the cuda feature can name
`ferrotorch_profiler::cuda_timing::*` directly when needed, but the
default `cargo build` (no features) does not compile the module at
all, keeping the no-cuda dependency graph free of `cudarc`.

### Doctest quick-start (REQ-4)

The 13-line doctest at `lib.rs:14-25` exercises the canonical
usage: build a `ProfileConfig`, hand it + a closure to
`with_profiler`, record two ops, render a top-10 table. It compiles
as part of `cargo test --doc`, so changing any signature in the
re-export set breaks the build — the doc and the API stay in sync.

### Non-test production consumers

- `ferrotorch/src/lib.rs:107` `pub use ferrotorch_profiler::*;`
  (inside `#[cfg(feature = "profiler")] pub mod profiler { ... }`)
  is the canonical end-user re-export path. Users write
  `ferrotorch::profiler::ProfileConfig::default()` /
  `ferrotorch::profiler::with_profiler(...)`. This makes
  every item in REQ-3 a production consumer of the
  `pub use` lines in `lib.rs:38-43`.
- Internal cross-file consumption: each submodule's `pub use`
  collapses through this file, so when `profiler in profiler.rs` writes
  `use crate::report::ProfileReport;`, the corresponding re-export
  on `lib.rs:42` is what the meta-crate sees.

## Parity contract

`parity_ops = []`. This file is structural — no numerical kernels.
The parity contract it owns:

- **Public-surface stability**: re-exports must not break
  downstream code that already imports
  `ferrotorch_profiler::ProfileConfig`. The surface-coverage
  conformance test (`tests/conformance_surface_coverage.rs:66-`)
  pins every item; removing a re-export fails the test.
- **Feature gating**: enabling `cuda` must not change the
  semantics of any non-cuda type. Verified by building both
  configurations in CI.

## Verification

No unit tests live in `lib.rs` (re-export module). The crate-root
doctest is the only inline test. Smoke:

```bash
cargo test -p ferrotorch-profiler --doc 2>&1 | tail -3
cargo clippy -p ferrotorch-profiler --lib -- -D warnings 2>&1 | tail -3
```

Expected: doc-tests pass (1 doctest in the crate root), clippy
returns 0 warnings.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: lint pin at `ferrotorch-profiler/src/lib.rs:1-4` with the `clippy::module_name_repetitions` allowance justified inline; non-test consumer: `cargo clippy -p ferrotorch-profiler --lib -- -D warnings` passes (verified during this iter's gauntlet); the workspace-wide `cargo clippy --all-targets --all-features -- -D warnings` step in CI consumes this pin. |
| REQ-2 | SHIPPED | impl: submodule decls at `event in ferrotorch-profiler/src/lib.rs` with `#[cfg(feature = "cuda")] pub mod cuda_timing;` at line 27-28; non-test consumer: `event in ferrotorch-profiler/src/profiler.rs` consumes `crate::event`, `crate::flops`, `crate::report`; `event in ferrotorch-profiler/src/cuda_timing.rs` consumes `crate::profiler::Profiler` only when the cuda feature is on. |
| REQ-3 | SHIPPED | impl: `pub use` block at `ferrotorch-profiler/src/lib.rs:38-43` re-exporting `CudaKernelScope` (cuda-gated), `DeviceType`, `GpuTimingPair`, `MemoryCategory`, `ProfileEvent`, `ProfileConfig`, `Profiler`, `with_profiler`, `OpSummary`, `ProfileReport`, `ProfileSchedule`, `SchedulePhase`; non-test consumer: `ferrotorch/src/lib.rs:107` `pub use ferrotorch_profiler::*;` propagates every name to the meta-crate prelude. |
| REQ-4 | SHIPPED | impl: doctest at `with_profiler in ferrotorch-profiler/src/lib.rs` exercising `with_profiler` + `ProfileConfig::default` + `profiler.record` + `report.table(10)`, mirroring the example pattern at `torch/profiler/profiler.py:651-712`; non-test consumer: `cargo test --doc` runs the example as part of CI. |
| REQ-5 | SHIPPED | impl: `pub(crate) struct PendingCudaScope` at `PendingCudaScope in ferrotorch-profiler/src/cuda_timing.rs` is intentionally absent from the `pub use` block at `lib.rs`; non-test consumer: `ferrotorch-profiler/tests/conformance_surface_coverage.rs` enumerates every re-exported symbol — adding or removing a `pub use` line forces a test update, pinning the surface. |
