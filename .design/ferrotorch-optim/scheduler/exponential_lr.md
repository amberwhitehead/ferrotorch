# ferrotorch-optim — `scheduler::exponential_lr` (ExponentialLR)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/lr_scheduler.py
-->

## Summary

`ferrotorch-optim/src/scheduler/exponential_lr.rs` defines
`pub struct ExponentialLR`, a learning-rate scheduler that
multiplies LR by `gamma` once per step. Mirrors
`class ExponentialLR(LRScheduler)` at
`torch/optim/lr_scheduler.py:1007-1079`.

## Requirements

- REQ-1: `pub struct ExponentialLR` with `base_lr: f64`, `gamma:
  f64`, `current_step: usize`, `current_lr: f64` fields.
  Mirrors the upstream attribute state at
  `lr_scheduler.py:1028-1035`.

- REQ-2: `pub fn ExponentialLR::new(base_lr, gamma) -> Self`
  constructor. Mirrors `ExponentialLR.__init__`
  (`lr_scheduler.py:1028-1035`).

- REQ-3: `impl<T: Float> LrScheduler<T> for ExponentialLR` using
  the **closed-form** schedule
  `lr = base_lr * gamma.powi(step as i32)`, not the per-step
  multiplication. The closed form matches upstream's
  `_get_closed_form_lr` (`lr_scheduler.py:1066-1079`) and avoids
  fp drift over thousands of steps.

## Acceptance Criteria

- [x] AC-1: `pub struct ExponentialLR` with the four named fields.
- [x] AC-2: `ExponentialLR::new(0.1, 0.95).get_lr() == 0.1`
  initially (`test_exponential_initial`).
- [x] AC-3: After 1 step with `gamma = 0.95`, LR == `0.095`
  (`test_exponential_one_step`).
- [x] AC-4: Closed form matches at every step for 20 steps
  (`test_exponential_analytical`).
- [x] AC-5: `gamma == 1.0` → no decay
  (`test_exponential_gamma_one`).

## Architecture

`pub struct ExponentialLR` stores the schedule parameters and the
running state. The private `compute_lr` evaluates
`base_lr * gamma.powi(step as i32)`. `impl LrScheduler<T>`
advances `current_step`, recomputes `current_lr`, and pushes to
`optimizer.set_lr`.

`f64::powi` is preferred over `f64::powf` for integer exponents
because `powi` lowers to a square-and-multiply loop and is
typically faster + more accurate at integer exponents than `powf`
(which routes through `exp(y * ln(x))`).

### Non-test production consumers

- `ExponentialLR` re-exported at
  `ferrotorch-optim/src/lib.rs:47-52`.
- `Learner::with_scheduler` at
  `ferrotorch-train/src/learner.rs:105` accepts
  `Box<dyn LrScheduler<T>>`; user code constructs
  `Box::new(ExponentialLR::new(lr, gamma))` and passes it. The
  per-epoch `sched.step(self.optimizer.as_mut())` at
  `ferrotorch-train/src/learner.rs:306-308` is the production
  consumer.
- `ChainedScheduler` example in `scheduler/chained_scheduler.rs:29-40`
  documents the canonical pattern of chaining an
  `ExponentialLR` with another scheduler.

## Parity contract

`parity_ops = []`. Numerical contract:

- **`gamma == 0.0`**: LR jumps to `0` on the first step and stays
  there. Matches `0.powi(n)` returning `0` for `n > 0`.
- **`gamma < 0.0`**: alternating sign on every step. Allowed but
  almost certainly a user error.
- **`gamma > 1.0`**: LR grows exponentially. Allowed.
- **`current_step` overflow at `i32::MAX`**: `step as i32`
  would wrap; in practice unreachable in training timescales.

## Verification

Tests in `#[cfg(test)] mod tests` (5 tests):

- `test_exponential_initial` — initial `get_lr() == base_lr`.
- `test_exponential_one_step` — one step gives `base_lr * gamma`.
- `test_exponential_analytical` — closed form over 20 steps.
- `test_exponential_gamma_one` — no decay when `gamma == 1.0`.
- `test_exponential_rapid_decay` — `gamma == 0.1` decays to
  `1e-3` in 3 steps.

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib scheduler::exponential_lr 2>&1 | tail -3
```

Expected: `5 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct ExponentialLR` with `base_lr`, `gamma`, `current_step`, `current_lr` fields in `scheduler/exponential_lr.rs` mirrors `torch/optim/lr_scheduler.py:1028-1035`; non-test consumer: re-exported at `ferrotorch-optim/src/lib.rs:47-52`; user code constructs `ExponentialLR::new(...)` and hands `Box::new(...)` to `Learner::with_scheduler` at `ferrotorch-train/src/learner.rs:105`. |
| REQ-2 | SHIPPED | impl: `pub fn ExponentialLR::new(base_lr, gamma) -> Self` in `scheduler/exponential_lr.rs` mirrors `torch/optim/lr_scheduler.py:1028-1035`; non-test consumer: the `pub use` at `lib.rs:47-52` is the API surface user-code calls. |
| REQ-3 | SHIPPED | impl: `impl<T: Float> LrScheduler<T> for ExponentialLR` in `scheduler/exponential_lr.rs` uses `f64::powi` closed-form mirroring `torch/optim/lr_scheduler.py:1066-1079`; non-test consumer: `Learner` invokes `sched.step(self.optimizer.as_mut())` at `ferrotorch-train/src/learner.rs:306-308`, dispatching to this impl when the boxed scheduler is an `ExponentialLR`. |
