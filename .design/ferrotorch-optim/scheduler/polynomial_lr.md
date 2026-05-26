# ferrotorch-optim — `scheduler::polynomial_lr` (PolynomialLR)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/lr_scheduler.py
-->

## Summary

`ferrotorch-optim/src/scheduler/polynomial_lr.rs` defines
`pub struct PolynomialLR`, a scheduler that decays LR following
`base_lr * (1 - step / total_iters)^power`. Mirrors
`class PolynomialLR(LRScheduler)` at
`torch/optim/lr_scheduler.py:1236-1335`.

## Requirements

- REQ-1: `pub struct PolynomialLR` with `base_lr: f64`,
  `total_iters: usize`, `power: f64`, `current_step: usize`,
  `current_lr: f64` fields. Mirrors the upstream attributes at
  `lr_scheduler.py:1263-1272`.

- REQ-2: `pub fn PolynomialLR::new(base_lr, total_iters, power)
  -> Self` constructor with `current_lr` initialized to
  `base_lr`. Mirrors `PolynomialLR.__init__`
  (`lr_scheduler.py:1263-1272`). Upstream's defaults
  (`total_iters=5`, `power=1.0`) are NOT folded into the Rust
  signature.

- REQ-3: `impl<T: Float> LrScheduler<T> for PolynomialLR` using
  closed-form `base_lr * (1 - min(step, total_iters) /
  total_iters)^power`. Mirrors upstream's `_get_closed_form_lr`
  (`lr_scheduler.py:1300-1335`). Beyond `total_iters`, LR is
  `0` (for `power >= 1`) or the appropriately clamped value.

- REQ-4: `total_iters == 0` short-circuits to `0` in
  `compute_lr` to avoid div-zero. R-DEV-7 deviation from
  upstream (which would raise `ZeroDivisionError`); ferrotorch
  prefers a sensible default over a panic.

## Acceptance Criteria

- [x] AC-1: `pub struct PolynomialLR` with the five named fields.
- [x] AC-2: Initial `get_lr() == base_lr`
  (`test_polynomial_initial`).
- [x] AC-3: With `power=1.0`, linear decay: at step `n`,
  LR == `base_lr * (1 - n / total_iters)`
  (`test_polynomial_linear_decay`).
- [x] AC-4: With `power=2.0`, at midpoint LR ==
  `base_lr * 0.25` (`test_polynomial_quadratic_decay`).
- [x] AC-5: Beyond `total_iters`, LR stays at `0`
  (`test_polynomial_beyond_total_iters`).
- [x] AC-6: Fractional power works
  (`test_polynomial_fractional_power`).
- [x] AC-7: At midpoint with `power=1`, LR == `base_lr / 2`
  (`test_polynomial_midpoint`).

## Architecture

`pub struct PolynomialLR` carries schedule parameters + running
state. The private `compute_lr`:

```text
if total_iters == 0 { return 0.0; }
let clamped = step.min(total_iters);
base_lr * (1.0 - clamped as f64 / total_iters as f64).powf(power)
```

`f64::powf` is used (not `powi`) because `power` may be
fractional (`0.5`, `0.9`, etc.). The integer-exponent
optimization isn't applicable.

`impl LrScheduler<T> for PolynomialLR` advances `current_step`,
recomputes `current_lr`, pushes to `optimizer.set_lr`.

### Non-test production consumers

- `PolynomialLR` re-exported at
  `ferrotorch-optim/src/lib.rs:47-52`.
- `Learner::with_scheduler` at
  `ferrotorch-train/src/learner.rs:105` accepts the boxed
  `PolynomialLR`; per-epoch step at
  `ferrotorch-train/src/learner.rs:306-308`.

## Parity contract

`parity_ops = []`. Numerical contract:

- **`total_iters == 0`**: ferrotorch returns `0`; upstream
  raises. R-DEV-7 deviation documented above.
- **`power == 0.0`**: `0.0.powf(0) == 1.0`, so for all steps
  `< total_iters`, LR == `base_lr`. At `step == total_iters`,
  `0.0.powf(0)` is still `1.0` so LR == `base_lr` there too.
  Beyond `total_iters` (where `clamped == total_iters`), same
  story. So `power=0` makes the scheduler an identity. Matches
  upstream.
- **`power < 0`**: division-like behavior; `(1 - step/total)^(-p)`
  blows up as step approaches total. Allowed.
- **`power == 1.0`** (default): linear decay from `base_lr` to 0.

## Verification

Tests in `#[cfg(test)] mod tests` (6 tests):

- `test_polynomial_initial`
- `test_polynomial_linear_decay`
- `test_polynomial_quadratic_decay`
- `test_polynomial_beyond_total_iters`
- `test_polynomial_fractional_power`
- `test_polynomial_midpoint`

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib scheduler::polynomial_lr 2>&1 | tail -3
```

Expected: `6 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct PolynomialLR` with `base_lr`, `total_iters`, `power`, `current_step`, `current_lr` fields in `scheduler/polynomial_lr.rs` mirrors `torch/optim/lr_scheduler.py:1263-1272`; non-test consumer: re-exported at `ferrotorch-optim/src/lib.rs:47-52`; user code boxes it for `Learner::with_scheduler` at `ferrotorch-train/src/learner.rs:105`. |
| REQ-2 | SHIPPED | impl: `pub fn PolynomialLR::new(base_lr, total_iters, power) -> Self` in `scheduler/polynomial_lr.rs` mirrors `torch/optim/lr_scheduler.py:1263-1272`; non-test consumer: the `pub use` at `lib.rs:47-52` is the user-call surface. |
| REQ-3 | SHIPPED | impl: `impl<T: Float> LrScheduler<T> for PolynomialLR` using `f64::powf` closed form in `scheduler/polynomial_lr.rs` mirrors `torch/optim/lr_scheduler.py:1300-1335`; non-test consumer: `Learner` invokes `sched.step(self.optimizer.as_mut())` at `ferrotorch-train/src/learner.rs:306-308`, dispatching to this impl when the boxed scheduler is a `PolynomialLR`. |
| REQ-4 | SHIPPED | impl: `compute_lr` short-circuits to `0.0` for `total_iters == 0` in `scheduler/polynomial_lr.rs` — R-DEV-7 cleaner-than-upstream behavior; non-test consumer: any `Learner` invocation with a `PolynomialLR::new(_, 0, _)` would observe `lr == 0` from the first step via the `sched.step(...)` call at `ferrotorch-train/src/learner.rs:306-308`. |
