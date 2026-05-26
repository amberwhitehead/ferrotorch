# Autocast op-policy registry (`AutocastCategory`, `autocast_category`, `autocast_guard`, `should_cast_*`, `AutocastEvent`)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (Revert "feat(gpu): route bf16 buffers through f32 elementwise dispatchers (#23) (#24)")
upstream-paths:
  - aten/src/ATen/autocast_mode.cpp
-->

## Summary

`ferrotorch-core/src/autograd/autocast_ops.rs` is the op-classification
side of ferrotorch's autocast subsystem. It maps op names (`"mm"`,
`"softmax"`, `"add"`, etc.) to one of three policy categories
(`ReducedPrecision`, `FullPrecision`, `Passthrough`) and exposes
`autocast_guard(op_name) -> Option<AutocastCategory>` as the primary
entry point for op implementations to query "should I cast my inputs
under the current autocast context?". The categorization table mirrors
the `KERNEL_PRIVATEUSE1` / `KERNEL_CPU` / `KERNEL_CUDA` registration
lists in `aten/src/ATen/autocast_mode.cpp`. A per-thread
`AutocastEvent` log captures policy decisions when the debug flag is
on, for test inspection.

## Requirements

- REQ-1: `pub enum AutocastCategory { ReducedPrecision, FullPrecision,
  Passthrough }` — the three-way policy classification. Mirrors
  upstream's `kReducedPrecision` / `kHighPrecision` / `kPassthrough`
  kernel-classification labels in
  `aten/src/ATen/autocast_mode.cpp`.
- REQ-2: `pub fn autocast_category(op_name: &str) -> AutocastCategory`
  — pure-function lookup keyed by op name. The match arm at
  `autocast_ops.rs:30-44` lists every op family classified as
  `ReducedPrecision` (matmul-family: `mm`, `matmul`, `bmm`, `linear`,
  `conv1d`, `conv2d`, `conv_transpose2d`, `addmm`, `einsum`) and
  `FullPrecision` (reduction/loss: `sum`, `mean`, `prod`, `softmax`,
  `log_softmax`, `layer_norm`, `batch_norm`, `group_norm`, `rms_norm`,
  `cross_entropy`, `mse_loss`, `bce_with_logits`). Everything else
  falls through to `Passthrough`.
- REQ-3: `pub fn should_cast_to_reduced(op_name: &str) -> bool` —
  returns true iff autocast is on AND the op is `ReducedPrecision`.
  Primary predicate for matmul-family op implementations.
- REQ-4: `pub fn should_keep_full_precision(op_name: &str) -> bool` —
  returns true iff autocast is on AND the op is `FullPrecision`.
  Predicate for reduction / loss ops that need to upcast f16 inputs
  to f32 before computing.
- REQ-5: `pub fn autocast_guard(op_name: &str) ->
  Option<AutocastCategory>` — the unified entry point. Returns `None`
  when autocast is off (zero-overhead opt-out), otherwise returns the
  category. When `is_autocast_debug()` is also on, side-effect of
  appending the lookup to the per-thread event log.
- REQ-6: `pub fn autocast_log(op_name: &str) ->
  Option<AutocastCategory>` — backward-compat alias for
  `autocast_guard`. New code should call `autocast_guard` directly.
- REQ-7: `pub struct AutocastEvent { op: String, category:
  AutocastCategory }` — single record in the per-thread event log.
- REQ-8: `pub fn drain_autocast_events() -> Vec<AutocastEvent>` —
  consume + clear the per-thread event log. Returns an empty vec
  when the debug flag was off or no ops were guarded.

## Acceptance Criteria

- [x] AC-1: Matmul family classifies as `ReducedPrecision` —
  `test_mm_is_reduced_precision`, `test_matmul_is_reduced_precision`,
  `test_bmm_is_reduced_precision`, `test_linear_is_reduced_precision`,
  `test_conv2d_is_reduced_precision` (`autocast_ops.rs:132-165`).
- [x] AC-2: Reduction / norm / loss family classifies as
  `FullPrecision` — `test_softmax_is_full_precision`,
  `test_log_softmax_is_full_precision`,
  `test_layer_norm_is_full_precision`,
  `test_batch_norm_is_full_precision`,
  `test_cross_entropy_is_full_precision`,
  `test_mse_loss_is_full_precision`, `test_sum_is_full_precision`,
  `test_mean_is_full_precision` (`autocast_ops.rs:167-222`).
- [x] AC-3: Elementwise / unknown ops classify as `Passthrough` —
  `test_add_is_passthrough`, `test_mul_is_passthrough`,
  `test_relu_is_passthrough`, `test_unknown_op_is_passthrough`
  (`autocast_ops.rs:225-245`).
- [x] AC-4: `should_cast_to_reduced` returns false when autocast is
  off — `test_should_cast_to_reduced_false_when_disabled` at
  `autocast_ops.rs:252-257`.
- [x] AC-5: `should_cast_to_reduced` returns true inside an `autocast`
  scope for matmul-family ops — `test_should_cast_to_reduced_true_for_mm_when_enabled`
  at `autocast_ops.rs:259-267`.
- [x] AC-6: `should_keep_full_precision` is symmetric — false outside
  autocast, true inside for reduction ops —
  `test_should_keep_full_precision_*` at `autocast_ops.rs:289-310`.
- [x] AC-7: `autocast_guard` returns `None` when autocast is off and
  `Some(category)` when on — `test_autocast_guard_*` at
  `autocast_ops.rs:379-398`.
- [x] AC-8: Debug events fire only when `set_autocast_debug(true)`
  is set — `test_autocast_guard_debug_events` at
  `autocast_ops.rs:400-425` and the inverse
  `test_autocast_guard_no_events_without_debug` at `:427-442`.
- [x] AC-9: Nested autocast contexts still report the same policy —
  `test_nested_autocast_policy_still_works` at
  `autocast_ops.rs:356-373`.

## Architecture

### REQ-1 `AutocastCategory`

`pub enum AutocastCategory { ReducedPrecision, FullPrecision,
Passthrough }` at `autocast_ops.rs:19-26` with `Debug, Clone, Copy,
PartialEq, Eq` derived.

### REQ-2 `autocast_category` — the static policy table

`pub fn autocast_category(op_name: &str) -> AutocastCategory` at
`autocast_ops.rs:29-45`. A single `match` over the op-name string:

- ReducedPrecision arm: `mm`, `matmul`, `bmm`, `linear`, `conv1d`,
  `conv2d`, `conv_transpose2d`, `addmm`, `einsum` at `:32-34`.
- FullPrecision arm: `sum`, `mean`, `prod`, `softmax`, `log_softmax`,
  `layer_norm`, `batch_norm`, `group_norm`, `rms_norm`,
  `cross_entropy`, `mse_loss`, `bce_with_logits` at `:38-40`.
- Default `Passthrough` arm at `:42-44`.

Each arm matches the corresponding upstream kernel-registration list
in `aten/src/ATen/autocast_mode.cpp`'s `TORCH_LIBRARY_IMPL(aten,
Autocast*, m)` blocks. The `addmm` and `einsum` arms are
classified-but-not-yet-wired (comment at `:33-34`); they belong here
for when the matmul-family integration lands their reduced-precision
GPU kernels.

### REQ-3 / REQ-4 op-side predicates

`pub fn should_cast_to_reduced` at `autocast_ops.rs:48-50` and
`pub fn should_keep_full_precision` at `:57-59` are 3-line composites:
`is_autocast_enabled() && autocast_category(op_name) == <expected>`.

### REQ-5 `autocast_guard` (primary entry point)

`pub fn autocast_guard(op_name: &str) -> Option<AutocastCategory>` at
`autocast_ops.rs:73-87`. Early-out on `!is_autocast_enabled()` →
return `None` (zero overhead in production). Otherwise compute the
category, optionally push an `AutocastEvent` to the per-thread log
when debug is on, return `Some(category)`.

### REQ-6 `autocast_log` (backward-compat alias)

`pub fn autocast_log(op_name: &str)` at `autocast_ops.rs:95-97` is a
1-line delegation to `autocast_guard`. New code should call
`autocast_guard`; this alias exists for callers written before the
unified entry point.

### REQ-7 / REQ-8 event log + drain

`pub struct AutocastEvent { op: String, category: AutocastCategory }`
at `autocast_ops.rs:105-108` with `Debug, Clone, PartialEq, Eq`.
Stored in `thread_local! AUTOCAST_EVENTS: RefCell<Vec<AutocastEvent>>`
at `:110-113`. `pub fn drain_autocast_events` at `:119-121` is a
single `RefCell::borrow_mut().drain(..).collect()`.

## Parity contract

`parity_ops = []` — autocast policy is metadata, not a tensor-valued
op. The categorization table is the parity contract; it must match
upstream's kernel-registration lists. If an op is registered as
`autocast` in `aten/src/ATen/autocast_mode.cpp` but absent from this
file's match arm, that's a divergence.

Currently the matmul-family + reduction-family classification matches
upstream. Edge cases:

- `relu` / `add` / `mul` are correctly `Passthrough` (upstream
  `KERNEL_AUTOCAST_PASSTHROUGH`).
- `embedding` is currently `Passthrough` (upstream classifies it as
  `KERNEL_AUTOCAST_PASSTHROUGH` too — embedding lookups should not
  cast).
- Unrecognized op names fall through to `Passthrough` (safe default
  matching upstream's "if not registered, no cast").

## Verification

Tests in `autocast_ops.rs:123-462` (33 tests across the three
categorization buckets, the policy predicates, the guard, and the
debug-event log).

All 33 tests pass in the workspace gauntlet.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum AutocastCategory { ReducedPrecision, FullPrecision, Passthrough }` at `ferrotorch-core/src/autograd/autocast_ops.rs:19-26`; mirrors upstream's three kernel-classification labels in `aten/src/ATen/autocast_mode.cpp`; non-test production consumer: `ferrotorch-core/src/grad_fns/linalg.rs:9 use crate::autograd::autocast_ops::{AutocastCategory, autocast_guard}` (the matmul-family backward path branches on the category); `ferrotorch-core/src/einsum.rs:53 use crate::autograd::autocast_ops::autocast_guard` (einsum forward branches on the category). Re-exported at `lib.rs:125-131`. |
| REQ-2 | SHIPPED | impl: `pub fn autocast_category(op_name: &str) -> AutocastCategory` at `autocast_ops.rs:29-45`; mirrors upstream's per-kernel autocast registration lists in `aten/src/ATen/autocast_mode.cpp`; non-test production consumer: invoked from inside `autocast_guard` (REQ-5) at `:77`; re-exported at `lib.rs:125-127 autocast_category`. |
| REQ-3 | SHIPPED | impl: `pub fn should_cast_to_reduced(op_name: &str) -> bool` at `autocast_ops.rs:48-50`; non-test production consumer: re-exported at `lib.rs:125-129` (boundary API for op-implementation code to branch on). Existing pub API across multiple prior commits — boundary-API grandfathering under goal.md S5. |
| REQ-4 | SHIPPED | impl: `pub fn should_keep_full_precision(op_name: &str) -> bool` at `autocast_ops.rs:57-59`; non-test production consumer: re-exported at `lib.rs:125-129`. Existing pub API across multiple prior commits — boundary-API grandfathering under goal.md S5. |
| REQ-5 | SHIPPED | impl: `pub fn autocast_guard(op_name: &str) -> Option<AutocastCategory>` at `autocast_ops.rs:73-87`; non-test production consumer: `ferrotorch-core/src/grad_fns/linalg.rs:9 use crate::autograd::autocast_ops::{AutocastCategory, autocast_guard}` and call sites inside `matmul_differentiable` / `bmm_differentiable` / `linear_fused` that branch their forward dispatch on `autocast_guard("mm")` → `Some(ReducedPrecision)`; `ferrotorch-core/src/einsum.rs:53 use ... autocast_guard` similarly inside `pub fn einsum`. |
| REQ-6 | SHIPPED | impl: `pub fn autocast_log(op_name: &str)` at `autocast_ops.rs:95-97`; non-test production consumer: re-exported as part of the autocast surface and held for prior callers; new code uses REQ-5 directly. Existing pub API across multiple prior commits — boundary-API grandfathering under goal.md S5. |
| REQ-7 | SHIPPED | impl: `pub struct AutocastEvent { op: String, category: AutocastCategory }` at `autocast_ops.rs:105-108`; non-test production consumer: pushed into `AUTOCAST_EVENTS` log inside `autocast_guard` at `:79-84`; drained by REQ-8. Re-exported at `lib.rs:125-131 AutocastEvent`. |
| REQ-8 | SHIPPED | impl: `pub fn drain_autocast_events() -> Vec<AutocastEvent>` at `autocast_ops.rs:119-121` plus the `AUTOCAST_EVENTS: RefCell<Vec<AutocastEvent>>` thread-local at `:110-113`; non-test production consumer: re-exported at `lib.rs:125-127 drain_autocast_events` — the public diagnostic-drain API for tooling code (and the conformance harness in `ferrotorch-core/src/einsum.rs:2217 use crate::autograd::autocast_ops::{AutocastCategory, drain_autocast_events}` etc.). Existing pub API across multiple prior commits — boundary-API grandfathering under goal.md S5. |
