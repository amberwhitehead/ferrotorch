# ferrotorch-optim — `scheduler::linear_lr` (LinearLR)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/lr_scheduler.py
-->

## Summary

`ferrotorch-optim/src/scheduler/linear_lr.rs` defines
`pub struct LinearLR`, a scheduler that linearly interpolates the
multiplicative factor from `start_factor` to `end_factor` over
`total_iters` steps. Mirrors `class LinearLR(LRScheduler)` at
`torch/optim/lr_scheduler.py:877-1005`.

## Requirements

- REQ-1: `pub struct LinearLR` with `base_lr: f64`, `start_factor:
  f64`, `end_factor: f64`, `total_iters: usize`, `current_step:
  usize`, `current_lr: f64` fields. Mirrors upstream's attribute
  state at `lr_scheduler.py:917-940`.

- REQ-2: `pub fn LinearLR::new(base_lr, start_factor, end_factor,
  total_iters) -> Self` constructor with two `assert!` preconditions:
  `start_factor` in `(0, 1]` and `end_factor` in `[0, 1]`. Mirrors
  upstream's `if start_factor > 1.0 or start_factor <= 0` and
  `if end_factor > 1.0 or end_factor < 0` raises at
  `lr_scheduler.py:917-934`. Initial `current_lr = base_lr *
  start_factor` so `get_lr()` is correct before any step.

- REQ-3: `impl<T: Float> LrScheduler<T> for LinearLR` using
  closed-form interpolation:

  ```text
  clamped = min(step, total_iters)
  factor = start_factor + (end_factor - start_factor) * clamped / total_iters
  lr = base_lr * factor
  ```

  Mirrors upstream's `_get_closed_form_lr`
  (`lr_scheduler.py:982-1005`).

- REQ-4: After `total_iters` steps, LR stays at
  `base_lr * end_factor` (handled by the `step.min(total_iters)`
  clamp in `compute_lr`).

## Acceptance Criteria

- [x] AC-1: `pub struct LinearLR` with the six named fields.
- [x] AC-2: Initial LR == `base_lr * start_factor`
  (`test_linear_lr_initial`).
- [x] AC-3: At `total_iters`, LR == `base_lr * end_factor`
  (`test_linear_lr_ramp_to_end`).
- [x] AC-4: Closed-form formula matches at every step over
  `total_iters` (`test_linear_lr_analytical`).
- [x] AC-5: Past `total_iters`, LR stays at `base_lr * end_factor`
  (`test_linear_lr_stays_after_total_iters`).
- [x] AC-6: Decreasing factor (`start_factor > end_factor`) works
  (`test_linear_lr_decreasing_factor`).
- [x] AC-7: Constructor panics on `start_factor == 0`
  (`test_linear_lr_invalid_start_factor`).

## Architecture

The closed-form schedule is

```text
clamped_step = min(current_step, total_iters)
factor = start_factor + (end_factor - start_factor) *
         clamped_step as f64 / total_iters as f64
current_lr = base_lr * factor
```

`compute_lr` short-circuits to `base_lr * end_factor` when
`total_iters == 0` (otherwise the division would be `0/0`).

`impl LrScheduler<T> for LinearLR` advances `current_step`, then
recomputes `current_lr`, then `optimizer.set_lr`.

### Non-test production consumers

- `LinearLR` re-exported at
  `ferrotorch-optim/src/lib.rs:47-52`.
- `ChainedScheduler` doc example in
  `scheduler/chained_scheduler.rs:29-40` uses `LinearLR` as one
  of two chained schedulers, demonstrating the canonical pattern.
- Production consumer: `Learner::with_scheduler` at
  `ferrotorch-train/src/learner.rs:105` accepts the boxed
  `LinearLR`. Per-epoch `sched.step` at
  `ferrotorch-train/src/learner.rs:306-308` is the call site.

## Parity contract

`parity_ops = []`. Numerical contract:

- **`total_iters == 0`**: `compute_lr` returns `base_lr *
  end_factor` immediately. Upstream raises a div-zero error in
  this case; ferrotorch's short-circuit is an R-DEV-7 deviation
  (cleaner Rust behavior).
- **`start_factor > end_factor`**: monotonically decreasing
  ramp, as in the upstream's "decay from full to fraction"
  pattern. Allowed.
- **`step > total_iters`**: clamped at `end_factor`.
- **Floating-point precision near `clamped == total_iters`**:
  the formula reduces to `start_factor + (end - start) * 1 ==
  end_factor` exactly because of integer division giving `1.0`
  cleanly.

## Verification

Tests in `#[cfg(test)] mod tests` (7 tests):

- `test_linear_lr_initial`
- `test_linear_lr_ramp_to_end`
- `test_linear_lr_analytical`
- `test_linear_lr_stays_after_total_iters`
- `test_linear_lr_decreasing_factor`
- `test_linear_lr_midpoint`
- `test_linear_lr_invalid_start_factor`

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib scheduler::linear_lr 2>&1 | tail -3
```

Expected: `7 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct LinearLR` with `base_lr`, `start_factor`, `end_factor`, `total_iters`, `current_step`, `current_lr` fields in `scheduler/linear_lr.rs` mirrors `torch/optim/lr_scheduler.py:917-940`; non-test consumer: re-exported at `ferrotorch-optim/src/lib.rs:47-52`; user code boxes it for `Learner::with_scheduler` at `ferrotorch-train/src/learner.rs:105`. |
| REQ-2 | SHIPPED | impl: `pub fn LinearLR::new(base_lr, start_factor, end_factor, total_iters) -> Self` with `assert!` preconditions in `scheduler/linear_lr.rs` mirrors `torch/optim/lr_scheduler.py:917-934`; non-test consumer: the `pub use` at `lib.rs:47-52` is the user-call surface. |
| REQ-3 | SHIPPED | impl: `impl<T: Float> LrScheduler<T> for LinearLR` using closed-form in `scheduler/linear_lr.rs` mirrors `torch/optim/lr_scheduler.py:982-1005`; non-test consumer: `Learner` invokes `sched.step(self.optimizer.as_mut())` at `ferrotorch-train/src/learner.rs:306-308`, dispatching to this impl when the boxed scheduler is a `LinearLR`. |
| REQ-4 | SHIPPED | impl: `step.min(total_iters)` clamp in `compute_lr` (`scheduler/linear_lr.rs`) freezes LR at `base_lr * end_factor` after `total_iters`; non-test consumer: `Learner` driving training past `total_iters` epochs observes the frozen LR via the same `sched.step(...)` invocation at `ferrotorch-train/src/learner.rs:306-308`. Tests `test_linear_lr_ramp_to_end` and `test_linear_lr_stays_after_total_iters` pin the boundary. |
