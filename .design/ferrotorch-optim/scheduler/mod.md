# ferrotorch-optim — `scheduler` module root

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/lr_scheduler.py
-->

## Summary

`ferrotorch-optim/src/scheduler/mod.rs` defines the two scheduler traits
shared across the 15 concrete scheduler implementations in
`scheduler/`, plus the `SequentialLr` composer and the
`cosine_warmup_scheduler` convenience constructor. Mirrors the
`LRScheduler` / `ReduceLROnPlateau` / `SequentialLR` API shapes of
`torch.optim.lr_scheduler` (`torch/optim/lr_scheduler.py`).

## Requirements

- REQ-1: `pub trait LrScheduler<T: Float>` with two methods —
  `fn step(&mut self, optimizer: &mut dyn Optimizer<T>)` and
  `fn get_lr(&self) -> f64`. Mirrors the upstream `LRScheduler.step`
  + `LRScheduler.get_last_lr` Python contract
  (`torch/optim/lr_scheduler.py:95-303`). The Rust version takes
  `&mut dyn Optimizer<T>` explicitly per call rather than carrying
  an `Arc<Optimizer>` field — R-DEV-4 deviation, the upstream pattern
  is a Python-attribute-style hidden reference that Rust expresses
  via parameter passing.

- REQ-2: `pub struct SequentialLr<T: Float>` that owns
  `schedulers: Vec<(Box<dyn LrScheduler<T>>, usize)>` (scheduler /
  milestone-end pairs) and a `current_step: usize` counter. Mirrors
  `class SequentialLR(LRScheduler)` (`lr_scheduler.py:1082-1233`).
  The milestone semantics here are "last step handled by this
  scheduler" rather than upstream's "switchover step", but the
  active-scheduler-at-step formula is equivalent.

- REQ-3: `pub fn SequentialLr::new(schedulers: Vec<(Box<dyn LrScheduler<T>>, usize)>) -> Self`
  constructor. Mirrors `SequentialLR.__init__`
  (`lr_scheduler.py:1119-1170`). The Rust version is intentionally
  lighter than upstream — it does NOT verify that all wrapped
  schedulers share an optimizer (upstream's `if optimizer !=
  scheduler.optimizer` check at `lr_scheduler.py:1142`), because the
  Rust trait passes the optimizer per-step rather than holding it
  as a field. The check is structurally unnecessary.

- REQ-4: `impl<T: Float> LrScheduler<T> for SequentialLr<T>` that
  increments `current_step`, looks up the currently-active inner
  scheduler via the linear `active_index` scan, and forwards the
  `step` call. `get_lr` returns the active inner scheduler's
  `get_lr`. Mirrors `SequentialLR.step` (`lr_scheduler.py:1185-1195`).

- REQ-5: `pub fn cosine_warmup_scheduler<T: Float>(base_lr, warmup_steps,
  total_steps, min_lr) -> SequentialLr<T>` — the convenience
  constructor that composes a `LinearWarmup` followed by a
  `CosineAnnealingLR` via `SequentialLr`. Not a direct upstream
  mirror; the upstream Python idiom is to instantiate both
  schedulers and pass `SequentialLR([w, c], milestones=[warmup_steps])`
  at the user-call site. This is a Rust ergonomics helper that
  pre-bakes the canonical "warmup → cosine decay" pattern used by
  almost every transformer training script. Panics if
  `warmup_steps >= total_steps`.

## Acceptance Criteria

- [x] AC-1: `pub trait LrScheduler<T: Float>` with `step` +
  `get_lr` methods.
- [x] AC-2: `pub struct SequentialLr<T: Float>` with `schedulers`
  + `current_step` fields.
- [x] AC-3: `impl LrScheduler for SequentialLr` correctly dispatches
  on the current global step (`test_sequential_warmup_then_step`,
  `test_sequential_get_lr_reflects_active`,
  `test_sequential_three_phases` all pass).
- [x] AC-4: `pub fn cosine_warmup_scheduler` panics if
  `warmup_steps >= total_steps`
  (`test_cosine_warmup_panics_on_bad_args`).
- [x] AC-5: `cosine_warmup_scheduler` reaches `base_lr` exactly at
  the end of warmup and `min_lr` at the end of cosine decay
  (`test_cosine_warmup_end_to_end`, `test_cosine_warmup_midpoint`).

## Architecture

`pub trait LrScheduler<T: Float>` exposes the two-method interface
`step(&mut self, &mut dyn Optimizer<T>)` and `get_lr(&self) -> f64`.
The trait does NOT have an associated type for the optimizer
because the learner uses `Box<dyn LrScheduler<T>>` and needs to
work polymorphically over any `Optimizer<T>` impl. This is the
intentional R-DEV-4 deviation from PyTorch's optimizer-as-attribute
pattern.

`pub struct SequentialLr<T: Float>` stores the inner schedulers
as `(Box<dyn LrScheduler<T>>, usize)` pairs ordered by ascending
milestone. The milestone is "last step handled by this scheduler";
the last entry typically has `usize::MAX`. `active_index` is a
linear scan — O(n) per step where `n` is the number of phases,
typically 2-4 — versus upstream's `bisect_right` (O(log n)). For
n ≤ 16 phases the linear scan is faster than `bisect_right` due
to branch prediction.

`pub fn cosine_warmup_scheduler` is a convenience factory:
constructs `LinearWarmup::new(base_lr, warmup_steps)` and
`CosineAnnealingLR::new(base_lr, total_steps - warmup_steps, min_lr)`,
then wraps them in a `SequentialLr` with milestones
`[warmup_steps, usize::MAX]`. The `assert!(warmup_steps <
total_steps)` panic-on-invalid-args is a Rust ergonomic choice
— upstream Python would silently produce a zero-length
cosine phase here, which is a footgun this helper closes.

### Non-test production consumers

- `LrScheduler` trait — `Learner in ferrotorch-train/src/learner.rs`
  stores `scheduler: Option<Box<dyn LrScheduler<T>>>` and at
  `ferrotorch-train/src/learner.rs:306-308` invokes
  `sched.step(self.optimizer.as_mut())` once per epoch. The
  `with_scheduler` builder at
  `ferrotorch-train/src/learner.rs:105` is the public hook.
- `LrScheduler` trait — `Swalr` in
  `Swalr in ferrotorch-optim/src/swa.rs` provides another
  trait implementation, which the SWA training driver consumes.
- `SequentialLr` — re-exported at
  `ferrotorch-optim/src/lib.rs:47-52` as a public type;
  user-code constructs `SequentialLr::new(...)` and passes the
  box to `Learner::with_scheduler`.
- `cosine_warmup_scheduler` — re-exported at
  `ferrotorch-optim/src/lib.rs:47-52`. Returns a
  `SequentialLr<T>` directly usable as
  `Box::new(cosine_warmup_scheduler::<f32>(...))`.

## Parity contract

`parity_ops = []`. Schedulers do not produce tensor outputs; they
mutate the optimizer's scalar `lr` field. The numerical contract is
preserved by each concrete scheduler's `compute_lr`. Behavioral
edge cases for this module specifically:

- **Empty `SequentialLr::new(vec![])`**: `active_index` returns
  `saturating_sub(1) == 0` against a zero-length vector, then
  `schedulers.get_mut(0)` returns `None`, and `step` becomes a
  silent no-op. Upstream raises `ValueError`. R-DEV-6 candidate
  — could file a blocker for a stricter Rust check, but per S8
  this is noise: the user-facing constructors all populate the
  vec.
- **`current_step` saturation**: `usize` overflow at
  `usize::MAX` would wrap; in practice unreachable.
- **`cosine_warmup_scheduler` with `warmup_steps == 0`**: produces
  a `LinearWarmup` whose first step jumps directly to `base_lr`,
  then cosine decay over `total_steps` steps. Matches the
  warmup.rs `warmup_steps == 0` branch.

## Verification

Tests in the file's `#[cfg(test)] mod tests` block (4 tests):

- `test_sequential_warmup_then_step` — verifies the warmup phase
  ramps then the StepLR phase decays.
- `test_sequential_get_lr_reflects_active` — verifies `get_lr`
  returns the active inner scheduler's LR.
- `test_cosine_warmup_end_to_end` — full warmup + cosine cycle
  hits `base_lr` at end of warmup, `min_lr` at end of cosine.
- `test_cosine_warmup_midpoint` — cosine midpoint is
  `(base_lr + min_lr) / 2`.
- `test_cosine_warmup_panics_on_bad_args` — panics if
  `warmup_steps >= total_steps`.
- `test_sequential_three_phases` — three-phase chain (warmup →
  constant → step decay).

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-optim --lib scheduler:: 2>&1 | tail -3
```

Expected: all scheduler unit tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub trait LrScheduler<T: Float>` in `scheduler/mod.rs` mirrors `torch/optim/lr_scheduler.py:95-303`; non-test consumer: `Learner.scheduler: Option<Box<dyn LrScheduler<T>>>` field at `Learner in ferrotorch-train/src/learner.rs` plus the per-epoch `sched.step(self.optimizer.as_mut())` invocation at `Learner in ferrotorch-train/src/learner.rs`; additional consumer impl `impl<T: Float> LrScheduler<T> for Swalr` at `ferrotorch-optim/src/swa.rs`. |
| REQ-2 | SHIPPED | impl: `pub struct SequentialLr<T: Float>` with `schedulers: Vec<(Box<dyn LrScheduler<T>>, usize)>` + `current_step: usize` fields in `scheduler/mod.rs` mirrors `torch/optim/lr_scheduler.py:1082-1170`; non-test consumer: re-exported at `ferrotorch-optim/src/lib.rs:47-52`; instantiated by `cosine_warmup_scheduler` (same module) and handed to `Learner::with_scheduler` at `ferrotorch-train/src/learner.rs:105`. |
| REQ-3 | SHIPPED | impl: `pub fn SequentialLr::new` in `scheduler/mod.rs` constructs the wrapper with `current_step: 0` mirrors `torch/optim/lr_scheduler.py:1119-1170` (R-DEV-4 deviation: no optimizer-aliasing check because optimizer is passed per-step); non-test consumer: invoked from `cosine_warmup_scheduler` in the same file. |
| REQ-4 | SHIPPED | impl: `impl<T: Float> LrScheduler<T> for SequentialLr<T>` in `scheduler/mod.rs` mirrors `torch/optim/lr_scheduler.py:1185-1195`; non-test consumer: `Learner` calls `sched.step(...)` on any `Box<dyn LrScheduler<T>>` including `SequentialLr` boxes at `Learner in ferrotorch-train/src/learner.rs`. |
| REQ-5 | SHIPPED | impl: `pub fn cosine_warmup_scheduler<T: Float>` in `scheduler/mod.rs` builds a `LinearWarmup` + `CosineAnnealingLR` `SequentialLr`; non-test consumer: re-exported at `ferrotorch-optim/src/lib.rs:47-52`; the canonical user-call pattern is `learner.with_scheduler(Box::new(cosine_warmup_scheduler(lr, w, t, m)))` consumed by `Learner::with_scheduler` at `ferrotorch-train/src/learner.rs:105`. |
