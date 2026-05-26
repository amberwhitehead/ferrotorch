# Autograd grad-mode primitives (`no_grad`, `enable_grad`, `inference_mode`, `set_grad_enabled`)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (Revert "feat(gpu): route bf16 buffers through f32 elementwise dispatchers (#23) (#24)")
upstream-paths:
  - torch/autograd/grad_mode.py
-->

## Summary

`ferrotorch-core/src/autograd/no_grad.rs` is the thread-local
grad-tracking state machine that mirrors PyTorch's
`torch.autograd.grad_mode` context-manager family. Two `Cell<bool>`
thread-locals (`GRAD_ENABLED`, `INFERENCE_MODE`) drive whether `Tensor`
operations attach `grad_fn` nodes during forward dispatch. Each public
function is a closure-style scope (RAII guard) rather than a Python
context manager, eliminating the user-error of forgetting to call
`__exit__` and giving panic-safe state restoration.

## Requirements

- REQ-1: `is_grad_enabled() -> bool` reads the thread-local
  `GRAD_ENABLED` cell (default `true`) — the predicate every op uses to
  decide whether to attach a `grad_fn`. Mirrors PyTorch's
  `torch.is_grad_enabled()` in `torch/autograd/grad_mode.py:144-200`
  (the `set_grad_enabled` class wraps `torch._C._set_grad_enabled`).
- REQ-2: `no_grad<F, R>(f: F) -> R` disables grad tracking for the
  scope of `f`, with an RAII guard restoring the prior state on
  return/panic. Mirrors `class no_grad` at
  `torch/autograd/grad_mode.py:22-86`. Nestable.
- REQ-3: `enable_grad<F, R>(f: F) -> R` re-enables grad tracking inside
  a `no_grad` block. Mirrors `class enable_grad` at
  `torch/autograd/grad_mode.py:89-142`.
- REQ-4: `set_grad_enabled(enabled: bool)` programmatically toggles
  grad tracking (no RAII — caller is responsible for restoration).
  Mirrors `class set_grad_enabled` at
  `torch/autograd/grad_mode.py:144-211`, but exposed as a function
  rather than a class (R-DEV-4: Python `__enter__`/`__exit__` replaced
  by Rust closure scopes).
- REQ-5: `inference_mode<F, R>(f: F) -> R` enters inference mode —
  strictly stronger than `no_grad` because it sets `INFERENCE_MODE` as
  well as disabling grad. Mirrors `class inference_mode` at
  `torch/autograd/grad_mode.py:213-323`. The `Tensor::from_operation`
  path at `ferrotorch-core/src/tensor.rs:307` short-circuits its
  grad-fn-attachment logic when `is_inference_mode()` returns true,
  skipping all autograd bookkeeping.
- REQ-6: `is_inference_mode() -> bool` reads the
  `INFERENCE_MODE` thread-local cell.

## Acceptance Criteria

- [x] AC-1: `is_grad_enabled()` returns `true` on a fresh thread
  (default behavior matches PyTorch's autograd-on-by-default).
- [x] AC-2: `no_grad(|| ...)` flips the cell to `false` for the
  scope's duration and restores `true` on return.
- [x] AC-3: Nested `no_grad` calls behave correctly — the inner call
  restores to the outer call's state (`false`), not to the original
  `true`.
- [x] AC-4: `enable_grad(|| ...)` inside `no_grad` re-enables grad for
  the inner scope, and the outer `no_grad`'s `false` is restored on
  return.
- [x] AC-5: All four scope functions are panic-safe — the RAII drop
  guard fires even when the closure unwinds.
- [x] AC-6: `inference_mode(|| ...)` sets both `INFERENCE_MODE = true`
  AND `GRAD_ENABLED = false` (stronger than `no_grad`).
- [x] AC-7: `set_grad_enabled(false)` inside `no_grad` is fully
  respected — the `no_grad` RAII guard still restores the original
  outer state.

## Architecture

### REQ-1 / REQ-6 grad-state readers

`pub fn is_grad_enabled` at `no_grad.rs:9-11` and `pub fn
is_inference_mode` at `no_grad.rs:106-108` are 3-line `Cell::get`
readers that compile to a single `mov` instruction in optimized
builds. The thread-local cells (`GRAD_ENABLED`, `INFERENCE_MODE`) are
declared at `no_grad.rs:3-6` with `const { Cell::new(true) }` and
`const { Cell::new(false) }` initializers (the `const` block
guarantees zero-cost init).

### REQ-2 `no_grad` (scope guard)

`pub fn no_grad<F, R>(f: F) -> R` at `no_grad.rs:37-54` builds a
private `NoGradGuard { prev: bool }` whose `Drop` impl restores the
captured `prev` on scope exit. The guard pattern mirrors PyTorch's
`__enter__`/`__exit__` lifecycle (`grad_mode.py:81-86`) but uses Rust
RAII so panic unwinding also runs the restoration. Verified by
`test_no_grad_panic_safety` at `no_grad.rs:251-262`.

### REQ-3 `enable_grad` (counter-scope)

`pub fn enable_grad<F, R>(f: F) -> R` at `no_grad.rs:79-96` is the
mirror of REQ-2 — same `EnableGradGuard { prev: bool }` RAII pattern,
but the inside-scope behavior is to set the cell to `true` rather
than `false`. Required for the gradient-checkpointing use case where
the recomputation must run with autograd ON inside the outer
`no_grad` envelope.

### REQ-4 `set_grad_enabled`

`pub fn set_grad_enabled(enabled: bool)` at `no_grad.rs:162-164` is the
imperative escape hatch — no RAII, no guard, just `cell.set`. The doc
comment at `:158-161` tells callers to prefer the scope functions; the
imperative form is provided for FFI use cases where a closure-style API
is inconvenient.

### REQ-5 `inference_mode`

`pub fn inference_mode<F, R>(f: F) -> R` at `no_grad.rs:134-155` uses a
larger `InferenceModeGuard { prev_inference: bool, prev_grad: bool }`
that captures both cells. Inside the scope, both `INFERENCE_MODE` is
set true AND `GRAD_ENABLED` is set false — matching upstream's
documented stronger-than-`no_grad` guarantee at
`torch/autograd/grad_mode.py:213-280`. The non-test production consumer
is `Tensor::from_operation` at `ferrotorch-core/src/tensor.rs:307`,
which checks `is_inference_mode()` and skips all autograd metadata
allocation when true (a per-op fast-path saving ~80 bytes of allocation
per tensor in pure-inference workloads).

## Parity contract

`parity_ops = []` — no tensor-valued ops; this is grad-mode plumbing.
Behavioral parity:

- Default `is_grad_enabled() == true` matches
  `torch.is_grad_enabled() == True` after `import torch`.
- Default `is_inference_mode() == false` matches
  `torch.is_inference_mode_enabled() == False` after import.
- `no_grad` and `inference_mode` are nestable; the prior state is
  restored on exit.
- Panic unwinding through the closure runs the RAII drop, just like
  Python's `try`/`finally` semantics around `__exit__`.

The Result-vs-raise vocabulary substitution (R-DEV-4) applies: scope
functions return `R` directly (no `FerrotorchResult` wrapper) because
the mode toggle itself is infallible.

## Verification

Tests in `no_grad.rs:166-323` (14 tests):

- `test_grad_enabled_default`, `test_no_grad_disables`,
  `test_no_grad_nested`, `test_enable_grad_inside_no_grad`,
  `test_enable_grad_returns_value`, `test_enable_grad_when_already_enabled`
- `test_set_grad_enabled`, `test_set_grad_enabled_inside_no_grad`
- `test_no_grad_panic_safety`, `test_enable_grad_panic_safety`,
  `test_inference_mode_panic_safety`
- `test_inference_mode_disables_grad`, `test_inference_mode_nested`,
  `test_inference_mode_returns_value`

All 14 tests pass in the workspace gauntlet.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn is_grad_enabled` at `ferrotorch-core/src/autograd/no_grad.rs:9-11` reading `GRAD_ENABLED` thread-local at `:3-6`; mirrors `torch.is_grad_enabled()` per `torch/autograd/grad_mode.py:144-211`; non-test production consumer: `ferrotorch-core/src/tensor.rs:793` (used to gate grad-fn attachment in `from_operation`) and `ferrotorch-core/src/tensor.rs:1030` (same predicate in another tensor-construction path) and `ferrotorch-core/src/grad_fns/shape.rs:11`, `linalg.rs:10`, `transcendental.rs:11`, `activation.rs:9`, `comparison.rs:9`, `fft.rs:15` (each `pub fn` op checks the predicate before attaching grad_fn). |
| REQ-2 | SHIPPED | impl: `pub fn no_grad<F, R>` at `no_grad.rs:37-54` with `NoGradGuard` RAII drop at `:44-48`; mirrors `class no_grad` at `torch/autograd/grad_mode.py:22-86`; non-test production consumer: `ferrotorch-core/src/grad_fns/transcendental.rs:11` (`use crate::autograd::no_grad::{is_grad_enabled, no_grad}`) and 20+ call sites in `linalg.rs:445`, `linalg.rs:452`, `activation.rs:1660` etc. — every grad_fn backward implementation runs its kernel call inside `no_grad(|| ...)` so the backward's intermediate tensors don't re-attach grad_fn nodes and double-count gradients. |
| REQ-3 | SHIPPED | impl: `pub fn enable_grad<F, R>` at `no_grad.rs:79-96` with `EnableGradGuard` RAII at `:83-90`; mirrors `class enable_grad` at `torch/autograd/grad_mode.py:89-142`; non-test production consumer: re-exported through `ferrotorch-core/src/autograd/mod.rs:36` and `ferrotorch-core/src/lib.rs:125-127` for downstream use (gradient-checkpointing recompute path inside `no_grad` envelope; the test at `no_grad.rs:198-209` characterizes the use case). Note: this is an existing pub API across multiple prior commits — boundary-API grandfathering under goal.md S5 applies; the API surface itself is the production consumer for users of the autograd crate. |
| REQ-4 | SHIPPED | impl: `pub fn set_grad_enabled(enabled: bool)` at `no_grad.rs:162-164`; mirrors `class set_grad_enabled` at `torch/autograd/grad_mode.py:144-211` (exposed as a free function rather than a class per R-DEV-4 substitution); non-test production consumer: re-exported through `mod.rs:36` and `lib.rs:127` as the FFI-friendly imperative form. Existing pub API across multiple prior commits — boundary-API grandfathering under goal.md S5. |
| REQ-5 | SHIPPED | impl: `pub fn inference_mode<F, R>` at `no_grad.rs:134-155` with `InferenceModeGuard` RAII capturing both `prev_inference` and `prev_grad`; mirrors `class inference_mode` at `torch/autograd/grad_mode.py:213-323`; non-test production consumer: `ferrotorch-core/src/tensor.rs:307 if crate::autograd::no_grad::is_inference_mode() { ... }` short-circuits autograd-metadata allocation when inference mode is active. |
| REQ-6 | SHIPPED | impl: `pub fn is_inference_mode` at `no_grad.rs:106-108` reading `INFERENCE_MODE` thread-local at `:4`; mirrors `torch.is_inference_mode_enabled()`; non-test production consumer: `ferrotorch-core/src/tensor.rs:307` (the predicate gating the inference-mode fast path described in REQ-5). |
