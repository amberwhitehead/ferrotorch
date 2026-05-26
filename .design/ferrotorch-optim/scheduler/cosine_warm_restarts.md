# ferrotorch-optim — `scheduler::cosine_warm_restarts` (CosineAnnealingWarmRestarts)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/lr_scheduler.py
-->

## Summary

`ferrotorch-optim/src/scheduler/cosine_warm_restarts.rs` defines
`pub struct CosineAnnealingWarmRestarts`, the SGDR scheduler from
Loshchilov & Hutter (2016). The LR follows a cosine curve from
`base_lr` to `eta_min` over the current cycle length `t_i`, then
snaps back to `base_lr` and starts a new cycle (with `t_i *=
t_mult`). Mirrors `class CosineAnnealingWarmRestarts(LRScheduler)`
at `torch/optim/lr_scheduler.py:2104-2273`.

## Requirements

- REQ-1: `pub struct CosineAnnealingWarmRestarts` with `base_lr:
  f64`, `t_0: usize` (retained for state introspection but
  unused in `step` — `#[allow(dead_code)]`), `t_mult: usize`,
  `eta_min: f64`, `t_cur: usize`, `t_i: usize`, `current_lr:
  f64` fields. Mirrors the upstream attributes at
  `lr_scheduler.py:2145-2166`.

- REQ-2: `pub fn CosineAnnealingWarmRestarts::new(base_lr, t_0,
  t_mult, eta_min) -> Self` constructor with two `assert!`
  preconditions: `t_0 > 0` and `t_mult >= 1`. Mirrors upstream's
  `ValueError`s at `lr_scheduler.py:2153-2160`. Initial state:
  `t_cur = 0`, `t_i = t_0`, `current_lr = base_lr`.

- REQ-3: `impl<T: Float> LrScheduler<T> for CosineAnnealingWarmRestarts`
  whose `step` advances `t_cur`, triggers a restart (`t_cur = 0`,
  `t_i *= t_mult`) when `t_cur >= t_i`, then recomputes
  `current_lr = eta_min + 0.5 * (base_lr - eta_min) * (1 +
  cos(pi * t_cur / t_i))`. Mirrors upstream's `step` +
  `get_lr` (`lr_scheduler.py:2168-2273`).

- REQ-4: Restart semantics: when `t_cur` reaches `t_i`, the
  cycle length grows by `t_mult` (for `t_mult > 1`). For
  `t_mult == 1`, the cycle length stays constant — pure
  periodic restarts.

## Acceptance Criteria

- [x] AC-1: `pub struct CosineAnnealingWarmRestarts` with the
  seven named fields (including the `#[allow(dead_code)]` `t_0`).
- [x] AC-2: Constructor panics on `t_0 == 0`
  (`test_warm_restarts_zero_t0_panics`).
- [x] AC-3: With `t_mult == 1`, restarts every `t_0` steps
  (`test_warm_restarts_first_cycle_end`).
- [x] AC-4: With `t_mult == 2`, second cycle is twice as long
  (`test_warm_restarts_t_mult_2`).
- [x] AC-5: At cycle midpoint, LR == `(base_lr + eta_min) / 2`
  (`test_warm_restarts_midpoint`).
- [x] AC-6: Nonzero `eta_min` is the floor reached at cycle end
  (`test_warm_restarts_with_eta_min`).
- [x] AC-7: With `t_mult == 1`, the LR pattern repeats every
  `t_0` steps (`test_warm_restarts_multiple_cycles_analytical`).

## Architecture

The cycle-state machine:

```text
t_cur += 1
if t_cur >= t_i:
    t_cur = 0          # snap back to start of new cycle
    t_i *= t_mult      # grow cycle length
current_lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * t_cur / t_i))
optimizer.set_lr(current_lr)
```

This differs from upstream's `step(epoch=None)` path
(`lr_scheduler.py:2237-2245`) in two ways:

1. Upstream uses `t_cur = t_cur % t_i` (modular wraparound)
   when `t_cur >= t_i`; ferrotorch uses `t_cur = 0` (snap). For
   normal `step()` increments of 1, these are equivalent — the
   modulo branch only matters when stepping by more than one
   unit at a time, which the ferrotorch `LrScheduler` trait
   doesn't support.
2. Upstream supports an optional `epoch` arg that can jump
   anywhere in the schedule (with the closed-form math at
   `lr_scheduler.py:2246-2264`). Ferrotorch's `step(_, _)` is
   single-step-only; the jump-anywhere feature is dropped.
   R-DEV-4: the Rust trait can't accept arbitrary kwargs.

The `t_0` field is retained for future state-dict
introspection / load_state_dict; it's not used in the current
`step` body. The `#[allow(dead_code)]` attribute documents this
intentionally.

### Non-test production consumers

- `CosineAnnealingWarmRestarts` re-exported at
  `ferrotorch-optim/src/lib.rs:47-52`.
- `Learner::with_scheduler` at
  `ferrotorch-train/src/learner.rs:105` accepts the boxed
  `CosineAnnealingWarmRestarts`; per-epoch step at
  `ferrotorch-train/src/learner.rs:306-308`.

## Parity contract

`parity_ops = []`. Numerical contract:

- **`t_mult == 1`**: pure periodic cycles, all of length `t_0`.
- **`t_mult > 1`**: cycle lengths grow geometrically (`t_0, t_0
  * t_mult, t_0 * t_mult^2, ...`). The total step count to
  finish cycle `N` is `t_0 * (t_mult^N - 1) / (t_mult - 1)`.
- **`t_0 == 0`**: rejected at construction with `assert!`.
- **Snapshot of `t_cur` immediately after restart**: LR ==
  `base_lr` (since `cos(0) == 1`).
- **`t_mult == 0`**: rejected at construction (allowed range is
  `>= 1`).
- **Floating-point precision at large `t_i`**: division
  precision degrades at `t_i > 2^53` (well beyond practical
  training); no special handling needed.

## Verification

Tests in `#[cfg(test)] mod tests` (6 tests):

- `test_warm_restarts_first_cycle_end`
- `test_warm_restarts_t_mult_2`
- `test_warm_restarts_midpoint`
- `test_warm_restarts_with_eta_min`
- `test_warm_restarts_multiple_cycles_analytical`
- `test_warm_restarts_zero_t0_panics`

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib scheduler::cosine_warm_restarts 2>&1 | tail -3
```

Expected: `6 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct CosineAnnealingWarmRestarts` with `base_lr`, `t_0`, `t_mult`, `eta_min`, `t_cur`, `t_i`, `current_lr` fields in `scheduler/cosine_warm_restarts.rs` mirrors `torch/optim/lr_scheduler.py:2145-2166`; non-test consumer: re-exported at `ferrotorch-optim/src/lib.rs:47-52`; user code boxes it for `Learner::with_scheduler` at `ferrotorch-train/src/learner.rs:105`. |
| REQ-2 | SHIPPED | impl: `pub fn CosineAnnealingWarmRestarts::new(base_lr, t_0, t_mult, eta_min) -> Self` with `assert!(t_0 > 0)` and `assert!(t_mult >= 1)` preconditions in `scheduler/cosine_warm_restarts.rs` mirrors `torch/optim/lr_scheduler.py:2153-2160`; non-test consumer: the `pub use` at `lib.rs:47-52` is the user-call surface. |
| REQ-3 | SHIPPED | impl: `impl<T: Float> LrScheduler<T> for CosineAnnealingWarmRestarts` with the cycle-snap state machine in `scheduler/cosine_warm_restarts.rs` mirrors the simple-step path of `torch/optim/lr_scheduler.py:2237-2245` and the cosine formula at `lr_scheduler.py:2199-2207`; non-test consumer: `Learner` invokes `sched.step(self.optimizer.as_mut())` at `ferrotorch-train/src/learner.rs:306-308`. |
| REQ-4 | SHIPPED | impl: The restart logic (`if t_cur >= t_i { t_cur = 0; t_i *= t_mult; }`) in the `step` body of `scheduler/cosine_warm_restarts.rs`; non-test consumer: any `Learner` running past one `t_0` cycle observes the restart through the `sched.step(...)` invocation. Tests `test_warm_restarts_first_cycle_end` and `test_warm_restarts_t_mult_2` pin both `t_mult == 1` and `t_mult > 1` restart behavior. |
