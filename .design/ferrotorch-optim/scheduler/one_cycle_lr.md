# ferrotorch-optim — `scheduler::one_cycle_lr` (OneCycleLR)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/lr_scheduler.py
-->

## Summary

`ferrotorch-optim/src/scheduler/one_cycle_lr.rs` defines
`pub struct OneCycleLR` and `pub enum AnnealStrategy`, the 1cycle
super-convergence policy from Smith & Topin (2018). LR ramps from
`initial_lr` to `max_lr` over `pct_start * total_steps` steps,
then anneals back down to `min_lr` over the remainder. Mirrors
`class OneCycleLR(LRScheduler)` at
`torch/optim/lr_scheduler.py:2284-2605`.

## Requirements

- REQ-1: `pub enum AnnealStrategy { Cos, Linear }` with `Debug,
  Clone, Copy, PartialEq, Eq` derived. Mirrors upstream's
  `anneal_strategy: Literal["cos", "linear"]` (`lr_scheduler.py:2329`).

- REQ-2: `pub struct OneCycleLR` with `phases: Vec<SchedulePhase>`,
  `total_steps: usize` (`#[allow(dead_code)]` — retained for
  state-dict roundtrip), `anneal_strategy: AnnealStrategy`,
  `current_step: usize`, `current_lr: f64` fields. The private
  `SchedulePhase` struct holds `end_step: f64`, `start_lr: f64`,
  `end_lr: f64`. Mirrors upstream's `_schedule_phases` machinery
  at `lr_scheduler.py:2454-2520`.

- REQ-3: `pub fn OneCycleLR::new(max_lr, total_steps, pct_start,
  anneal_strategy, div_factor, final_div_factor, three_phase) ->
  Self` constructor with `assert!`s on `total_steps > 0` and
  `pct_start in [0, 1]`. Computes `initial_lr = max_lr /
  div_factor` and `min_lr = initial_lr / final_div_factor`, then
  builds the phase table. Mirrors upstream's `__init__` at
  `lr_scheduler.py:2358-2520`.

- REQ-4: `impl<T: Float> LrScheduler<T> for OneCycleLR`
  evaluates the current step against the phase table, picks the
  active phase, computes the within-phase progress `pct`, and
  interpolates between the phase's start and end LRs using
  either cosine or linear annealing. Mirrors upstream's `get_lr`
  at `lr_scheduler.py:2538-2602`.

- REQ-5: Two-phase mode (default): ramp-up + anneal-down.
  Three-phase mode (`three_phase: bool`): ramp-up + anneal-down
  + further-anneal-down. The phase table differs structurally.
  Mirrors `lr_scheduler.py:2454-2520`.

- REQ-6: NOT-STARTED — `cycle_momentum`, `base_momentum`,
  `max_momentum`, `three_phase` momentum handling (upstream
  `lr_scheduler.py:2342-2350, 2391-2453`) are NOT implemented.
  The `Optimizer<T>` trait doesn't expose `set_momentum`.
  Tracked by blocker #1474.

## Acceptance Criteria

- [x] AC-1: `pub enum AnnealStrategy` with two variants.
- [x] AC-2: `pub struct OneCycleLR` with the named fields.
- [x] AC-3: Initial LR == `max_lr / div_factor`
  (`test_one_cycle_initial_lr`).
- [x] AC-4: At phase-1 boundary, LR ≈ `max_lr`
  (`test_one_cycle_reaches_max_lr_cos`).
- [x] AC-5: At `total_steps`, LR ≈ `min_lr`
  (`test_one_cycle_end_lr`).
- [x] AC-6: Linear annealing produces a monotonic ramp in
  phase 1 (`test_one_cycle_linear_monotonic_ramp`).
- [x] AC-7: Three-phase mode reaches `min_lr` at the end
  (`test_one_cycle_three_phase`).
- [x] AC-8: LR is never negative
  (`test_one_cycle_lr_never_negative`).
- [x] AC-9: Constructor panics on `total_steps == 0`
  (`test_one_cycle_zero_steps_panics`).
- [ ] AC-10: Momentum cycling — blocker #1474.

## Architecture

The schedule is built as a sequence of `SchedulePhase` records,
each holding an end-step (relative to the start of the schedule)
and the start/end LR for that phase. Phase selection is a linear
scan; for at most 3 phases this is faster than binary search.

Two-phase mode (`three_phase == false`):
- Phase 1: `[0, pct_start * total_steps - 1]`, LR `initial_lr →
  max_lr`.
- Phase 2: `[pct_start * total_steps - 1, total_steps - 1]`, LR
  `max_lr → min_lr`.

Three-phase mode (`three_phase == true`):
- Phase 1: ramp-up, LR `initial_lr → max_lr`.
- Phase 2: anneal back to `initial_lr`.
- Phase 3: anneal further down to `min_lr`.

Within each phase, `pct ∈ [0, 1]` is computed and either:
- `anneal_cos`: `end + (start - end) / 2 * (cos(pi * pct) + 1)`.
- `anneal_linear`: `(end - start) * pct + start`.

Both are byte-equivalent to upstream's static method versions at
`lr_scheduler.py:2522-2530`.

### Non-test production consumers

- `OneCycleLR` and `AnnealStrategy` re-exported at
  `ferrotorch-optim/src/lib.rs:47-52`.
- `Learner::with_scheduler` at
  `ferrotorch-train/src/learner.rs:105` accepts the boxed
  `OneCycleLR`; per-epoch step at
  `ferrotorch-train/src/learner.rs:306-308`.

## Parity contract

`parity_ops = []`. Numerical contract:

- **Single-phase boundary**: at `step == pct_start *
  total_steps`, `pct == 1.0` exactly via the phase-2 entry,
  reaching `max_lr` exactly.
- **`total_steps == 0`**: rejected at construction with `assert!`.
- **`pct_start == 0.0`**: no ramp-up phase — phase 1 has zero
  length, immediately enters phase 2. Allowed.
- **`pct_start == 1.0`**: the whole schedule is ramp-up; phase 2
  has zero length. Allowed.
- **`div_factor == 0`**: `max_lr / 0` is `Inf`; downstream LR
  setting gets `Inf`. Allowed but a user error.
- **Momentum cycling**: NOT-STARTED, blocker #1474.

## Verification

Tests in `#[cfg(test)] mod tests` (7 tests):

- `test_one_cycle_initial_lr`
- `test_one_cycle_reaches_max_lr_cos`
- `test_one_cycle_end_lr`
- `test_one_cycle_linear_monotonic_ramp`
- `test_one_cycle_three_phase`
- `test_one_cycle_lr_never_negative`
- `test_one_cycle_zero_steps_panics`

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib scheduler::one_cycle_lr 2>&1 | tail -3
```

Expected: `7 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum AnnealStrategy { Cos, Linear }` with `Debug, Clone, Copy, PartialEq, Eq` derived in `scheduler/one_cycle_lr.rs` mirrors `torch/optim/lr_scheduler.py:2329`; non-test consumer: re-exported at `ferrotorch-optim/src/lib.rs:47-52` (`AnnealStrategy` is in the `pub use` list); user code passes `AnnealStrategy::Cos` or `::Linear` to the constructor. |
| REQ-2 | SHIPPED | impl: `pub struct OneCycleLR` with `phases`, `total_steps`, `anneal_strategy`, `current_step`, `current_lr` fields + private `SchedulePhase` in `scheduler/one_cycle_lr.rs` mirrors `torch/optim/lr_scheduler.py:2454-2520`; non-test consumer: re-exported at `ferrotorch-optim/src/lib.rs:47-52`; user code boxes it for `Learner::with_scheduler` at `ferrotorch-train/src/learner.rs:105`. |
| REQ-3 | SHIPPED | impl: `pub fn OneCycleLR::new(max_lr, total_steps, pct_start, anneal_strategy, div_factor, final_div_factor, three_phase) -> Self` with `assert!`s in `scheduler/one_cycle_lr.rs` mirrors `torch/optim/lr_scheduler.py:2358-2520`; non-test consumer: the `pub use` at `lib.rs:47-52` is the user-call surface. |
| REQ-4 | SHIPPED | impl: `impl<T: Float> LrScheduler<T> for OneCycleLR` with phase-aware compute in `scheduler/one_cycle_lr.rs` mirrors `torch/optim/lr_scheduler.py:2538-2602`; non-test consumer: `Learner` invokes `sched.step(self.optimizer.as_mut())` at `ferrotorch-train/src/learner.rs:306-308`, dispatching to this impl when the boxed scheduler is a `OneCycleLR`. |
| REQ-5 | SHIPPED | impl: Two-phase vs three-phase branching in the constructor (`scheduler/one_cycle_lr.rs`) mirrors `torch/optim/lr_scheduler.py:2454-2520`; non-test consumer: any `Learner` constructed with `OneCycleLR::new(..., true)` exercises the three-phase path via the `sched.step(...)` invocation at `ferrotorch-train/src/learner.rs:306-308`. Tests `test_one_cycle_three_phase` pins the three-phase end-of-schedule LR. |
| REQ-6 | NOT-STARTED | blocker #1474 — momentum cycling requires `Optimizer<T>` trait extension exposing `set_momentum`; `cycle_momentum`, `base_momentum`, `max_momentum` upstream features (`torch/optim/lr_scheduler.py:2342-2350, 2391-2453`) cannot be wired until the trait grows the missing accessor. |
