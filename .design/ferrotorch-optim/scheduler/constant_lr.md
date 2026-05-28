# ferrotorch-optim — `scheduler::constant_lr` (ConstantLR)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/lr_scheduler.py
-->

## Summary

`ferrotorch-optim/src/scheduler/constant_lr.rs` defines
`pub struct ConstantLR`, a scheduler that holds LR at `base_lr *
factor` for `total_iters` steps and then restores LR to
`base_lr`. Mirrors `class ConstantLR(LRScheduler)` at
`torch/optim/lr_scheduler.py:773-874`.

## Requirements

- REQ-1: `pub struct ConstantLR` with `base_lr: f64`, `factor:
  f64`, `total_iters: usize`, `current_step: usize`, `current_lr:
  f64` fields. Mirrors the upstream attributes at
  `lr_scheduler.py:806-820`.

- REQ-2: `pub fn ConstantLR::new(base_lr, factor, total_iters)
  -> Self` constructor with the `assert!((0.0..=1.0).contains(&factor))`
  precondition matching upstream's
  `if factor > 1.0 or factor < 0` check (`lr_scheduler.py:813-815`,
  which raises `ValueError`). The Rust version uses `assert!` —
  R-DEV-1 rationale: factor outside `[0, 1]` is a programmer error
  detectable at construction; failing loudly there is safer than
  silent nonsense at training time. Initial `current_lr = base_lr *
  factor` so `get_lr()` reflects the active-phase LR immediately.

- REQ-3: `impl<T: Float> LrScheduler<T> for ConstantLR` using the
  closed-form formula:
  - if `step >= total_iters`: `lr = base_lr`
  - else: `lr = base_lr * factor`
  Mirrors upstream's `_get_closed_form_lr`
  (`lr_scheduler.py:857-874`).

## Acceptance Criteria

- [x] AC-1: `pub struct ConstantLR` with the five named fields.
- [x] AC-2: Constructor panics on `factor` outside `[0, 1]`
  (`test_constant_lr_invalid_factor`).
- [x] AC-3: Initial LR == `base_lr * factor`
  (`test_constant_lr_initial`).
- [x] AC-4: During phase, LR stays at `base_lr * factor`
  (`test_constant_lr_during_phase`).
- [x] AC-5: At `total_iters`, LR restores to `base_lr`
  (`test_constant_lr_restores_after_total_iters`).
- [x] AC-6: Beyond `total_iters`, LR stays at `base_lr`
  (`test_constant_lr_stays_at_base_after_total_iters`).
- [x] AC-7: `factor == 1.0` → LR == `base_lr` everywhere
  (`test_constant_lr_factor_one`).
- [x] AC-8: `factor == 0.0` → LR == `0` during phase, then
  `base_lr` (`test_constant_lr_factor_zero`).

## Architecture

`pub struct ConstantLR` stores the schedule parameters and the
running state. The private `compute_lr`:

```text
if step >= total_iters { base_lr } else { base_lr * factor }
```

The constructor seeds `current_lr = base_lr * factor` so that
`get_lr()` before any `step` call returns the active-phase LR.
This is a small ergonomic difference from upstream — upstream
relies on `_initial_step` calling `step` once at construction
(`lr_scheduler.py:174-180`). Ferrotorch achieves the equivalent by
direct initialization without doing an LR update through the
optimizer.

### Non-test production consumers

- `ConstantLR` re-exported at
  `ferrotorch-optim/src/lib.rs:47-52`.
- `ChainedScheduler` test code in
  `scheduler in scheduler/chained_scheduler.rs` and `scheduler in scheduler/chained_scheduler.rs`
  demonstrates the canonical chaining pattern, but the real
  production consumer is `Learner::with_scheduler` at
  `ferrotorch-train/src/learner.rs:105`, which accepts the boxed
  `ConstantLR`. The per-epoch step at
  `ferrotorch-train/src/learner.rs:306-308` dispatches.

## Parity contract

`parity_ops = []`. Numerical contract:

- **`factor == 1.0`**: identity scheduler — LR == `base_lr`
  forever (the phase is invisible).
- **`factor == 0.0`**: LR is `0` during the phase, then `base_lr`.
- **`total_iters == 0`**: `step >= 0` is true on the first step, so
  the constructor's initial `base_lr * factor` is immediately
  overwritten by `base_lr` on the first `step()` call.
  Functionally identical to "no constant phase". Upstream behavior
  is the same.
- **`current_step` overflow**: `usize` wraparound; unreachable.

## Verification

Tests in `#[cfg(test)] mod tests` (7 tests):

- `test_constant_lr_initial`
- `test_constant_lr_during_phase`
- `test_constant_lr_restores_after_total_iters`
- `test_constant_lr_stays_at_base_after_total_iters`
- `test_constant_lr_factor_one`
- `test_constant_lr_factor_zero`
- `test_constant_lr_invalid_factor`

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib scheduler::constant_lr 2>&1 | tail -3
```

Expected: `7 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct ConstantLR` with `base_lr`, `factor`, `total_iters`, `current_step`, `current_lr` fields in `scheduler/constant_lr.rs` mirrors `torch/optim/lr_scheduler.py:806-820`; non-test consumer: re-exported at `ferrotorch-optim/src/lib.rs:47-52`; user-code boxes it for `Learner::with_scheduler` at `ferrotorch-train/src/learner.rs:105`. |
| REQ-2 | SHIPPED | impl: `pub fn ConstantLR::new(base_lr, factor, total_iters) -> Self` with `assert!((0.0..=1.0).contains(&factor))` in `scheduler/constant_lr.rs` mirrors `torch/optim/lr_scheduler.py:813-820`; non-test consumer: the `pub use` at `lib.rs:47-52` exposes it for user-call construction. |
| REQ-3 | SHIPPED | impl: `impl<T: Float> LrScheduler<T> for ConstantLR` using closed-form in `scheduler/constant_lr.rs` mirrors `torch/optim/lr_scheduler.py:857-874`; non-test consumer: `Learner` invokes `sched.step(self.optimizer.as_mut())` at `ferrotorch-train/src/learner.rs:306-308`, dispatching to this impl when the boxed scheduler is a `ConstantLR`. |
