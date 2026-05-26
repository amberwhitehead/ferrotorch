# Anomaly detection mode (`AnomalyMode`, `ForwardBacktrace`, `detect_anomaly`, `check_gradient_anomaly`)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (Revert "feat(gpu): route bf16 buffers through f32 elementwise dispatchers (#23) (#24)")
upstream-paths:
  - torch/autograd/anomaly_mode.py
-->

## Summary

`ferrotorch-core/src/autograd/anomaly.rs` is the autograd-engine
diagnostic layer that mirrors PyTorch's
`torch.autograd.set_detect_anomaly(True)` / `with torch.autograd.detect_anomaly():`
context-manager API. When enabled, forward operations capture a
`std::backtrace::Backtrace` and store it on the resulting tensor's
metadata; backward operations check intermediate gradients for NaN /
Inf and, on hit, raise `FerrotorchError::InvalidArgument` with the
captured forward backtrace embedded so the user can see exactly which
forward-op produced the offending node.

## Requirements

- REQ-1: `pub struct AnomalyMode` zero-sized type with `enable() /
  disable() / is_enabled()` static methods backed by a
  `Cell<bool>` thread-local. Mirrors `class detect_anomaly` /
  `set_detect_anomaly` at `torch/autograd/anomaly_mode.py:12-150`.
- REQ-2: `pub fn detect_anomaly<F, R>(f: F) -> R` — scope-style
  context-manager equivalent. Enters anomaly mode for the closure's
  duration with RAII guard restoring previous state on return or
  panic. Mirrors the `with torch.autograd.detect_anomaly():` Python
  idiom (R-DEV-4: Python `__enter__`/`__exit__` replaced by Rust
  scope).
- REQ-3: `pub struct ForwardBacktrace { trace: String }` — captured
  backtrace at forward-op time. Created via
  `ForwardBacktrace::capture_if_enabled() -> Option<Self>` (zero
  overhead when anomaly mode is off — returns `None` without invoking
  `Backtrace::capture()`). Mirrors PyTorch's
  `_traceback_recording_helper` in
  `torch/csrc/autograd/python_anomaly_mode.cpp`.
- REQ-4: `pub fn check_gradient_anomaly<T: Float>(grad, op_name,
  forward_bt) -> FerrotorchResult<()>` — called by the backward
  dispatcher when anomaly mode is active; checks `grad` for NaN /
  Inf and, on hit, returns `FerrotorchError::InvalidArgument` with
  a message embedding the captured forward backtrace.
- REQ-5: Anomaly is thread-local — enabling on one thread does NOT
  affect others. Matches PyTorch's per-thread `_detect_anomaly` state.
- REQ-6: `Display` impl on `ForwardBacktrace` produces a
  human-readable stack trace (`"Forward-pass backtrace:\n<trace>"`).
  `Debug` impl elides the trace contents (`"<backtrace>"`) to avoid
  noisy debug logs.

## Acceptance Criteria

- [x] AC-1: `AnomalyMode::is_enabled()` returns `false` on a fresh
  thread (default off, matching upstream's
  `torch.is_anomaly_enabled() == False` after import) —
  `test_anomaly_mode_default_off` at `anomaly.rs:181-186`.
- [x] AC-2: `enable / disable` toggle works —
  `test_anomaly_mode_enable_disable` at `anomaly.rs:188-193`.
- [x] AC-3: `detect_anomaly(|| ...)` enters/exits the mode and
  restores prior state — `test_detect_anomaly_scoped` at
  `anomaly.rs:195-203`.
- [x] AC-4: Panic-safe — `detect_anomaly` restores state after a
  panic unwind — `test_detect_anomaly_panic_safety` at
  `anomaly.rs:205-216`.
- [x] AC-5: Nestable — inner `detect_anomaly` restores to the outer's
  enabled state — `test_detect_anomaly_nested` at `anomaly.rs:218-231`.
- [x] AC-6: `ForwardBacktrace::capture_if_enabled()` returns `None`
  when disabled (zero-overhead fast path) and `Some(bt)` when enabled —
  `test_forward_backtrace_capture_when_disabled` (`anomaly.rs:233-237`)
  and `test_forward_backtrace_capture_when_enabled` (`anomaly.rs:240-246`).
- [x] AC-7: `check_gradient_anomaly` reports NaN / Inf cleanly —
  `test_check_gradient_anomaly_nan` (`anomaly.rs:263-281`) and
  `test_check_gradient_anomaly_inf` (`anomaly.rs:283-300`).
- [x] AC-8: When passed a `ForwardBacktrace`, the error message
  includes the trace — `test_check_gradient_anomaly_with_backtrace`
  at `anomaly.rs:302-316`.
- [x] AC-9: When anomaly mode is OFF, `check_gradient_anomaly` is a
  silent no-op (does NOT report NaN even on NaN input) —
  `test_check_gradient_anomaly_skipped_when_disabled` at
  `anomaly.rs:318-328`.

## Architecture

### REQ-1 `AnomalyMode` ZST + thread-local

`pub struct AnomalyMode;` at `anomaly.rs:29` plus three static methods
`enable`, `disable`, `is_enabled` at `:31-46` reading the
`thread_local! ANOMALY_ENABLED: Cell<bool>` at `:14-16`. The ZST
pattern + static methods is the Rust idiom for grouping mode toggles
under a namespace (analogous to `class Mode: ...` with classmethods in
Python).

### REQ-2 `detect_anomaly` (scope guard)

`pub fn detect_anomaly<F, R>(f: F) -> R` at `anomaly.rs:62-79`. RAII
guard `AnomalyGuard { prev: bool }` saves the prior state and
restores via `Drop`. Same panic-safe pattern as `no_grad` (REQ-2 of
`no_grad.md`).

### REQ-3 `ForwardBacktrace`

`pub struct ForwardBacktrace { trace: String }` at `anomaly.rs:86-88`
plus `capture_if_enabled` factory at `:94-103`. The zero-overhead
fast path is essential — `std::backtrace::Backtrace::capture()` is
slow (microseconds-to-milliseconds depending on symbol-table
availability), so checking the thread-local cell FIRST and returning
`None` early matters for production performance when anomaly mode is
off. Implements `Clone, Debug, Display` (custom impls at `:111-123`).

### REQ-4 `check_gradient_anomaly`

`pub fn check_gradient_anomaly<T: Float>(grad, op_name, forward_bt)`
at `anomaly.rs:131-174`. Three guards on entry:

1. Defensive: returns `Ok(())` if anomaly mode is off (callers
   shouldn't invoke unless on, but the guard makes the function
   composable).
2. GPU short-circuit: returns `Ok(())` if `grad.is_cuda()` — a full
   D2H transfer just to scan for NaN is expensive; the documented
   workaround at `:142-146` is for users to register a `.cpu()`-shaped
   hook if they want GPU-side anomaly checking.
3. Scan: walks `grad.data()` for NaN and Inf, builds a
   human-readable error message embedding the
   `ForwardBacktrace::Display` output when one was provided.

### REQ-5 thread-local isolation

The `thread_local!` macro at `anomaly.rs:14-16` is per-thread by
construction — `AnomalyMode::enable()` on thread A does not propagate
to thread B. This is intentionally identical to PyTorch's behavior
(its `_detect_anomaly` is thread-local at the Python interpreter
level).

### REQ-6 `Display` / `Debug` impls

`impl fmt::Debug for ForwardBacktrace` at `anomaly.rs:111-117` elides
the trace contents (just shows `"<backtrace>"`) so dumping a tensor's
metadata via `Debug` doesn't dump kilobytes of stack-frame strings.
`impl fmt::Display` at `:119-123` shows the full trace.

## Parity contract

`parity_ops = []` — anomaly mode is a diagnostic state machine plus
gradient scanning, not a tensor-valued op. Behavioral parity vs
upstream:

- Default disabled (matches upstream's `torch.is_anomaly_enabled() ==
  False` after import).
- Thread-local — enabling on one thread does not affect others.
- Scope nesting restores prior state via RAII (PyTorch uses Python
  `try/finally`).
- NaN AND Inf both flagged. The error-message kind string at
  `:153-159` says `"NaN and Inf"` when both are present, otherwise
  just one.
- GPU short-circuits silently — documented at `:142-146`. Upstream
  PyTorch also short-circuits when the gradient lives on a non-CPU
  device unless the user explicitly opted into the slow path.

## Verification

Tests in `anomaly.rs:176-328` (10 tests):

- `test_anomaly_mode_default_off` (`:181`)
- `test_anomaly_mode_enable_disable` (`:188`)
- `test_detect_anomaly_scoped` (`:195`)
- `test_detect_anomaly_panic_safety` (`:205`)
- `test_detect_anomaly_nested` (`:218`)
- `test_forward_backtrace_capture_when_disabled` (`:233`)
- `test_forward_backtrace_capture_when_enabled` (`:240`)
- `test_check_gradient_anomaly_clean` (`:249`)
- `test_check_gradient_anomaly_nan` (`:263`)
- `test_check_gradient_anomaly_inf` (`:283`)
- `test_check_gradient_anomaly_with_backtrace` (`:302`)
- `test_check_gradient_anomaly_skipped_when_disabled` (`:318`)

All 12 (count includes `test_check_gradient_anomaly_clean`) pass in
the workspace gauntlet.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct AnomalyMode` at `ferrotorch-core/src/autograd/anomaly.rs:29` with static methods at `:31-46` plus the `ANOMALY_ENABLED: Cell<bool>` thread-local at `:14-16`; mirrors `class detect_anomaly` / `class set_detect_anomaly` at `torch/autograd/anomaly_mode.py:12-150`; non-test production consumer: `ferrotorch-core/src/lib.rs:121-123 pub use autograd::anomaly::{AnomalyMode, ForwardBacktrace, check_gradient_anomaly, detect_anomaly}` exposes the type to the crate root for downstream debug-tooling crates and user-facing diagnostics. Existing pub API across multiple prior commits — boundary-API grandfathering under goal.md S5. |
| REQ-2 | SHIPPED | impl: `pub fn detect_anomaly<F, R>` at `anomaly.rs:62-79` with `AnomalyGuard` RAII restore at `:66-73`; mirrors the `with torch.autograd.detect_anomaly():` idiom (R-DEV-4 Python `__enter__`/`__exit__` → Rust scope); non-test production consumer: re-exported at `lib.rs:122`. Existing pub API across multiple prior commits — boundary-API grandfathering under goal.md S5. |
| REQ-3 | SHIPPED | impl: `pub struct ForwardBacktrace` at `anomaly.rs:86-88` plus `capture_if_enabled` zero-overhead factory at `:94-103`; mirrors PyTorch's traceback-recording helper in `torch/csrc/autograd/python_anomaly_mode.cpp`; non-test production consumer: re-exported at `lib.rs:122`. Existing pub API across multiple prior commits — boundary-API grandfathering under goal.md S5. |
| REQ-4 | SHIPPED | impl: `pub fn check_gradient_anomaly<T: Float>` at `anomaly.rs:131-174`; non-test production consumer: re-exported at `lib.rs:122 check_gradient_anomaly`. Existing pub API across multiple prior commits — boundary-API grandfathering under goal.md S5. |
| REQ-5 | SHIPPED | impl: `thread_local! static ANOMALY_ENABLED: Cell<bool>` at `anomaly.rs:14-16` — per-thread by language construct; mirrors PyTorch's per-thread state; production consumer for the thread-local guarantee: every callsite of `AnomalyMode::enable / disable / is_enabled` (REQ-1) inherits the thread-local semantics. |
| REQ-6 | SHIPPED | impl: `impl fmt::Debug for ForwardBacktrace` at `anomaly.rs:111-117` (elides) and `impl fmt::Display for ForwardBacktrace` at `:119-123` (full); production consumer: `format!("{bt}")` inside `check_gradient_anomaly`'s error-message-building branch at `:161-166` (the `Display` impl is invoked when the error message is rendered). |
