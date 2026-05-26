# Autocast state (`AutocastDtype`, `autocast`, `with_autocast_state`, `AutocastSnapshot`)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (Revert "feat(gpu): route bf16 buffers through f32 elementwise dispatchers (#23) (#24)")
upstream-paths:
  - aten/src/ATen/autocast_mode.cpp
  - torch/amp/autocast_mode.py
-->

## Summary

`ferrotorch-core/src/autograd/autocast.rs` is the thread-local
mixed-precision (autocast) state machine. Two `Cell<...>`
thread-locals (`AUTOCAST_ENABLED`, `AUTOCAST_DTYPE`) plus a debug-event
toggle (`AUTOCAST_DEBUG`) drive whether ops in the `ReducedPrecision`
category cast their inputs to `f16` / `bf16` for the duration of an
enclosing scope. The actual op-by-op policy lookup lives in the sibling
`autocast_ops.rs` module (this file is the state cell; that file is the
policy registry). The `AutocastSnapshot` round-trip primitive is used
by `autograd::checkpoint` to ensure the backward-pass recomputation
runs under the same mixed-precision context that produced the forward
activations.

## Requirements

- REQ-1: `pub enum AutocastDtype { F16, BF16 }` — the reduced-precision
  target. Mirrors PyTorch's `torch.float16` / `torch.bfloat16`
  per-region target.
- REQ-2: `pub fn is_autocast_enabled() -> bool` — read the
  thread-local `AUTOCAST_ENABLED` cell. Default `false`. Mirrors
  `torch.is_autocast_enabled()` from
  `torch/amp/autocast_mode.py:225-265`.
- REQ-3: `pub fn autocast_dtype() -> AutocastDtype` — read the
  thread-local `AUTOCAST_DTYPE` cell. Default `F16`. Only meaningful
  when `is_autocast_enabled()` is true. Mirrors
  `torch.get_autocast_dtype(device_type)`.
- REQ-4: `pub fn autocast<F, R>(dtype: AutocastDtype, f: F) -> R` —
  scope-style enter/exit with RAII guard restoring previous state on
  return or panic. Mirrors `class autocast` at
  `torch/amp/autocast_mode.py:52-340`. Nestable.
- REQ-5: `pub struct AutocastSnapshot { enabled: bool, dtype:
  AutocastDtype }` — point-in-time capture of the autocast state.
  Used by `checkpoint` to defer-then-restore identical autocast
  context for backward recomputation.
- REQ-6: `pub fn current_autocast_snapshot() -> AutocastSnapshot` —
  capture both cells into one round-trip-able value.
- REQ-7: `pub fn with_autocast_state<F, R>(snapshot, f) -> R` —
  more general than `autocast` because it can also restore the
  "disabled" state. The `StateGuard` saves both cells and restores on
  drop (including panic unwind).
- REQ-8: `set_autocast_debug(enabled) / is_autocast_debug()` — event
  recording toggle, per-thread. When debug is on, every call to
  `autocast_guard` (in the sibling `autocast_ops.rs`) appends an
  `AutocastEvent` to the per-thread event log. Default off (zero
  overhead in production).

## Acceptance Criteria

- [x] AC-1: `is_autocast_enabled()` returns `false` outside any
  scope — `test_autocast_default_disabled` at `autocast.rs:167-169`.
- [x] AC-2: `autocast(F16, || ...)` flips enabled to true for the
  scope's duration and restores on exit — `test_autocast_enables` at
  `autocast.rs:172-179`.
- [x] AC-3: Nested `autocast` calls restore each level's prior dtype
  — `test_autocast_nested` at `autocast.rs:181-198`.
- [x] AC-4: dtype selection sticks — `test_autocast_dtype_selection`
  at `autocast.rs:200-209`.
- [x] AC-5: Default dtype is `F16` — `test_default_dtype_is_f16` at
  `autocast.rs:211-215`.
- [x] AC-6: Panic safety — RAII guard restores state after panic —
  `test_autocast_panic_safety` at `autocast.rs:217-229`.
- [x] AC-7: Debug flag toggle works —
  `test_autocast_debug_flag` at `autocast.rs:231-239`.

## Architecture

### REQ-1 `AutocastDtype`

`pub enum AutocastDtype { F16, BF16 }` at `autocast.rs:28-34` with
`Debug, Clone, Copy, PartialEq, Eq` derived. Two variants:

- `F16` — IEEE 754 half-precision (1-5-10).
- `BF16` — brain-float (1-8-7), wider dynamic range.

Matches upstream's `torch.float16` / `torch.bfloat16` selection — the
enum is an R-DEV-2 substitution (Rust-strong-typing in place of
Python's dtype objects, but the user-facing concept matches).

### REQ-2 / REQ-3 readers

`pub fn is_autocast_enabled` at `autocast.rs:37-39` and `pub fn
autocast_dtype` at `:44-46` are 3-line `Cell::get` readers, identical
pattern to `no_grad`. Optimized builds compile each to a single
thread-local load.

### REQ-4 `autocast` (scope guard)

`pub fn autocast<F, R>(dtype: AutocastDtype, f: F) -> R` at
`autocast.rs:137-161`. RAII guard `AutocastGuard { prev_enabled,
prev_dtype }` saves BOTH cells and restores via `Drop`. Same
panic-safe pattern as `no_grad` / `detect_anomaly`.

### REQ-5 / REQ-6 / REQ-7 — snapshot / round-trip

`pub struct AutocastSnapshot { enabled: bool, dtype: AutocastDtype }`
at `autocast.rs:54-63` is `Debug, Clone, Copy, PartialEq, Eq`.
`pub fn current_autocast_snapshot()` at `:71-76` builds one from the
two thread-locals. `pub fn with_autocast_state(snapshot, f)` at
`:88-111` is the inverse — installs the snapshot's `(enabled, dtype)`
for the scope's duration with `StateGuard` RAII restore.

This is more general than `autocast(dtype, f)` because the snapshot
can carry `enabled = false`, which `autocast` cannot represent (it
unconditionally sets enabled=true). Used by
`checkpoint::CheckpointBackward` (`checkpoint.rs:240`) and
`CheckpointMultiBackward` (`:312`) to reproduce the exact autocast
context the forward pass ran under, so recomputation produces
numerically identical activations.

### REQ-8 — debug event toggle

`thread_local! AUTOCAST_DEBUG: Cell<bool>` at `autocast.rs:5-7` plus
`set_autocast_debug` / `is_autocast_debug` readers at `:18-25`. The
debug flag gates the per-event `AutocastEvent` push inside
`autocast_ops::autocast_guard` at `autocast_ops.rs:73-87`. Default
off — zero overhead in production, opt-in for tests.

## Parity contract

`parity_ops = []` — autocast is mode plumbing. Behavioral parity:

- Default disabled (matches upstream `torch.is_autocast_enabled() ==
  False`).
- Default dtype F16 (matches PyTorch's
  `torch.get_autocast_dtype('cuda')` after import — its default is
  `torch.float16`).
- Thread-local — enabling on one thread does not affect others.
- Nestable; inner scope restores to outer's `(enabled, dtype)` pair
  on exit.
- Panic-safe via RAII drop guards.

`AutocastSnapshot` adds expressive power beyond upstream's
`enabled`-only context manager: it can encode the disabled state so
checkpoint recomputation can faithfully reproduce a forward pass that
ran OUTSIDE an autocast region.

## Verification

Tests in `autocast.rs:163-240` (7 tests):

- `test_autocast_default_disabled` (`:167`)
- `test_autocast_enables` (`:172`)
- `test_autocast_nested` (`:181`)
- `test_autocast_dtype_selection` (`:200`)
- `test_default_dtype_is_f16` (`:211`)
- `test_autocast_panic_safety` (`:217`)
- `test_autocast_debug_flag` (`:231`)

All 7 pass in the workspace gauntlet.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum AutocastDtype { F16, BF16 }` at `ferrotorch-core/src/autograd/autocast.rs:28-34`; mirrors `torch.float16` / `torch.bfloat16` per-region dtype selection at `torch/amp/autocast_mode.py:226 enabled: bool = True`; non-test production consumer: `crate::autograd::autocast::AutocastDtype` referenced from `ferrotorch-core/src/autograd/checkpoint.rs:3` (the `with_autocast_state(self.saved_autocast, ...)` recompute path inside `CheckpointBackward::backward` at `checkpoint.rs:240`) and `ferrotorch-core/src/einsum.rs:2216 use crate::autograd::autocast::{AutocastDtype, autocast, set_autocast_debug}` (and `ferrotorch-nn/src/loss.rs:4374 autocast(AutocastDtype::F16, || ...)`). Re-exported at `lib.rs:131-133`. |
| REQ-2 | SHIPPED | impl: `pub fn is_autocast_enabled` at `autocast.rs:37-39`; mirrors `torch.is_autocast_enabled()`; non-test production consumer: `ferrotorch-core/src/autograd/autocast_ops.rs:48 is_autocast_enabled() && autocast_category(op_name) == AutocastCategory::ReducedPrecision` in `pub fn should_cast_to_reduced` (called from every op that supports reduced-precision dispatch). |
| REQ-3 | SHIPPED | impl: `pub fn autocast_dtype` at `autocast.rs:44-46`; mirrors `torch.get_autocast_dtype('cuda')`; non-test production consumer: re-exported at `lib.rs:131`; used inside `with_autocast_state` (REQ-7) at `:104-110` to populate the `StateGuard.prev_dtype` field; field of `AutocastSnapshot` (REQ-5) populated by `current_autocast_snapshot` at `:71-76`. |
| REQ-4 | SHIPPED | impl: `pub fn autocast<F, R>(dtype, f)` at `autocast.rs:137-161` with `AutocastGuard` RAII at `:142-152`; mirrors `class autocast` at `torch/amp/autocast_mode.py:52-340`; non-test production consumer: `ferrotorch-nn/src/loss.rs:4374 autocast(AutocastDtype::F16, || ...)` and `:4404 autocast(AutocastDtype::BF16, || ...)` and `:4433 autocast(AutocastDtype::F16, || ...)` — three call sites in the loss module's autocast-aware loss wrappers; also invoked from `ferrotorch-core/src/einsum.rs:2229 autocast(AutocastDtype::F16, || ...)`. |
| REQ-5 | SHIPPED | impl: `pub struct AutocastSnapshot { enabled: bool, dtype: AutocastDtype }` at `autocast.rs:54-63`; non-test production consumer: `ferrotorch-core/src/autograd/checkpoint.rs:200 saved_autocast: AutocastSnapshot` field on `CheckpointBackward` and `:280 saved_autocast: AutocastSnapshot` field on `CheckpointMultiBackward` — the round-tripped snapshot the backward-pass recompute uses to reproduce the forward-pass mixed-precision context. |
| REQ-6 | SHIPPED | impl: `pub fn current_autocast_snapshot` at `autocast.rs:71-76`; non-test production consumer: `ferrotorch-core/src/autograd/checkpoint.rs:81 let saved_autocast = current_autocast_snapshot();` inside `checkpoint` and `:125 let saved_autocast = current_autocast_snapshot();` inside `checkpoint_multi` — captures the forward-time autocast state. |
| REQ-7 | SHIPPED | impl: `pub fn with_autocast_state<F, R>` at `autocast.rs:88-111` with `StateGuard` RAII; non-test production consumer: `ferrotorch-core/src/autograd/checkpoint.rs:240 with_autocast_state(self.saved_autocast, || ...)` inside `CheckpointBackward::backward` and `:312 with_autocast_state(self.saved_autocast, || ...)` inside `CheckpointMultiBackward::backward` — invoked on every gradient-checkpointed forward + backward pair. |
| REQ-8 | SHIPPED | impl: `pub fn set_autocast_debug` at `autocast.rs:18-20` and `pub fn is_autocast_debug` at `:23-25` plus the `AUTOCAST_DEBUG: Cell<bool>` thread-local at `:5-7`; non-test production consumer: `ferrotorch-core/src/autograd/autocast_ops.rs:78 if is_autocast_debug() { ... }` inside `pub fn autocast_guard` — gates the per-event `AutocastEvent` push that test/debug code drains via `drain_autocast_events`. Re-exported at `lib.rs:125-129`. |
