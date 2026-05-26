# ferrotorch-optim — `scheduler::cosine` (CosineAnnealingLR)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/lr_scheduler.py
-->

## Summary

`ferrotorch-optim/src/scheduler/cosine.rs` defines
`pub struct CosineAnnealingLR`, which decays the learning rate
following a half-period cosine curve from `base_lr` down to
`eta_min` over `t_max` steps. Mirrors
`class CosineAnnealingLR(LRScheduler)` at
`torch/optim/lr_scheduler.py:1337-1474`.

## Requirements

- REQ-1: `pub struct CosineAnnealingLR` with `base_lr: f64`,
  `t_max: usize`, `eta_min: f64`, `current_step: usize`,
  `current_lr: f64` fields. Mirrors `CosineAnnealingLR.__init__`
  state (`lr_scheduler.py:1387-1396`).

- REQ-2: `pub fn CosineAnnealingLR::new(base_lr, t_max, eta_min)
  -> Self` constructor. Mirrors `CosineAnnealingLR.__init__`
  (`lr_scheduler.py:1387-1396`). Upstream's default `eta_min = 0.0`
  is NOT folded into the Rust signature — the user always passes it.

- REQ-3: `impl<T: Float> LrScheduler<T> for CosineAnnealingLR`
  using the **closed-form** schedule
  `eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * step / t_max))`
  for `step <= t_max`, clamping to `eta_min` for `step > t_max`.
  This is upstream's `_get_closed_form_lr`
  (`lr_scheduler.py:1455-1474`), not the recursive update at
  `lr_scheduler.py:1447-1453`. The closed form matches upstream
  byte-for-byte at integer steps and avoids the trigonometric
  drift the recursive form accumulates.

- REQ-4: At step `t_max`, LR settles at `eta_min`; for all
  `step > t_max`, LR stays clamped at `eta_min`. This is the
  "no warm restarts" behavior — the `CosineAnnealingWarmRestarts`
  type in `scheduler/cosine_warm_restarts.rs` handles the
  periodic-restart variant separately.

## Acceptance Criteria

- [x] AC-1: `pub struct CosineAnnealingLR` with the five named
  fields.
- [x] AC-2: `CosineAnnealingLR::new(0.1, 100, 0.0).get_lr() ==
  0.1` initially (`test_cosine_initial_lr`).
- [x] AC-3: At step `t_max`, LR == `eta_min`
  (`test_cosine_at_t_max`).
- [x] AC-4: Beyond `t_max`, LR stays at `eta_min`
  (`test_cosine_beyond_t_max`).
- [x] AC-5: At step `t_max / 2`, LR == `(base_lr + eta_min) / 2`
  (`test_cosine_midpoint`).
- [x] AC-6: Closed-form formula matches at every step for
  100 steps (`test_cosine_analytical_values`).

## Architecture

`pub struct CosineAnnealingLR` keeps the schedule parameters and
the running `(current_step, current_lr)` state. The private
`compute_lr` method evaluates

```text
if step >= t_max { return eta_min; }
let progress = pi * step / t_max;
eta_min + 0.5 * (base_lr - eta_min) * (1.0 + cos(progress))
```

`impl LrScheduler<T> for CosineAnnealingLR` advances
`current_step`, recomputes `current_lr` via `compute_lr`, and
pushes to `optimizer.set_lr`.

The closed-form formula is preferred over upstream's recursive
update because it's bit-identical at integer steps to the
mathematically-defined cosine schedule, while the recursive form
(`lr_scheduler.py:1447-1453`) introduces compounding fp rounding
errors. Upstream itself defers to `_get_closed_form_lr` whenever
an explicit `epoch` is passed; ferrotorch always uses the closed
form.

### Non-test production consumers

- `CosineAnnealingLR` re-exported at
  `ferrotorch-optim/src/lib.rs:47-52`.
- `cosine_warmup_scheduler` in `scheduler/mod.rs` constructs
  `CosineAnnealingLR::new(base_lr, cosine_steps, min_lr)` and
  boxes it into the second slot of a `SequentialLr`. This is the
  primary in-crate production consumer.
- `Learner::with_scheduler` at
  `ferrotorch-train/src/learner.rs:105` accepts
  `Box<dyn LrScheduler<T>>`; user code typically passes
  `Box::new(cosine_warmup_scheduler::<f32>(...))`, transitively
  consuming this scheduler.

## Parity contract

`parity_ops = []`. The numerical contract is the closed-form
cosine. Edge cases:

- **`t_max == 0`**: `compute_lr` short-circuits to `eta_min` on
  the `step >= t_max` branch (since `0 >= 0`). Equivalent to
  immediate convergence.
- **`step > t_max`**: clamps to `eta_min`. This is the
  "no restarts" path; for periodic restarts use
  `CosineAnnealingWarmRestarts`.
- **`base_lr < eta_min`**: produces an inverted cosine that ramps
  UP from `base_lr` to `eta_min` over `t_max`. Upstream allows
  this too — neither side validates the ordering.
- **`current_step` overflow**: `usize` wraparound at `usize::MAX`;
  in practice unreachable.

## Verification

Tests in `#[cfg(test)] mod tests` (6 tests):

- `test_cosine_initial_lr` — initial `get_lr()` returns `base_lr`.
- `test_cosine_at_t_max` — at `t_max`, LR == `eta_min`.
- `test_cosine_beyond_t_max` — beyond `t_max`, LR clamps at
  `eta_min`.
- `test_cosine_midpoint` — at `t_max / 2`, LR ==
  `(base_lr + eta_min) / 2`.
- `test_cosine_analytical_values` — formula matches at every
  step for `t_max == 100`.
- `test_cosine_with_nonzero_eta_min` — nonzero `eta_min`
  reaches `eta_min` exactly at `t_max`.

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib scheduler::cosine 2>&1 | tail -3
```

Expected: `6 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct CosineAnnealingLR` with `base_lr`, `t_max`, `eta_min`, `current_step`, `current_lr` fields in `scheduler/cosine.rs` mirrors `torch/optim/lr_scheduler.py:1387-1396`; non-test consumer: `pub use CosineAnnealingLR` at `ferrotorch-optim/src/lib.rs:47-52` plus the in-crate construction site `CosineAnnealingLR::new(base_lr, cosine_steps, min_lr)` inside `cosine_warmup_scheduler` in `scheduler/mod.rs`. |
| REQ-2 | SHIPPED | impl: `pub fn CosineAnnealingLR::new(base_lr, t_max, eta_min) -> Self` in `scheduler/cosine.rs` mirrors `torch/optim/lr_scheduler.py:1387-1396`; non-test consumer: `cosine_warmup_scheduler` in `scheduler/mod.rs` calls `CosineAnnealingLR::new(...)` directly — production code in the same crate. |
| REQ-3 | SHIPPED | impl: `impl<T: Float> LrScheduler<T> for CosineAnnealingLR` in `scheduler/cosine.rs` uses the closed-form formula from `torch/optim/lr_scheduler.py:1455-1474`; non-test consumer: `Learner::step` calls `sched.step(self.optimizer.as_mut())` at `ferrotorch-train/src/learner.rs:306-308` — when the boxed scheduler is a `CosineAnnealingLR`, this impl runs. Also consumed transitively when `cosine_warmup_scheduler` produces a `SequentialLr` whose second phase is a `CosineAnnealingLR`. |
| REQ-4 | SHIPPED | impl: `compute_lr` in `scheduler/cosine.rs` returns `eta_min` for `step >= t_max`; non-test consumer: `Learner` driving training past `t_max` epochs will hold LR at `eta_min`, observable via the optimizer's `lr()` accessor invoked at `ferrotorch-train/src/learner.rs:306-308` (the same `sched.step(...)` call that updates LR). Tests `test_cosine_at_t_max` and `test_cosine_beyond_t_max` pin the boundary. |
