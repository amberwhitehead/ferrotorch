# ferrotorch-optim — `scheduler::plateau` (ReduceLROnPlateau)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/lr_scheduler.py
-->

## Summary

`ferrotorch-optim/src/scheduler/plateau.rs` defines
`pub struct ReduceLROnPlateau`, `pub enum PlateauMode`, and the
`pub trait MetricScheduler<T>` trait. The scheduler monitors a
user-supplied metric and reduces LR by `factor` once the metric
has not improved for `patience` consecutive `step` calls.
Mirrors `class ReduceLROnPlateau(LRScheduler)` at
`torch/optim/lr_scheduler.py:1583-1786`.

## Requirements

- REQ-1: `pub enum PlateauMode { Min, Max }` with `Debug, Clone,
  Copy, PartialEq, Eq` derived. Mirrors upstream's `mode:
  Literal["min", "max"]` (`lr_scheduler.py:1650`).

- REQ-2: `pub struct ReduceLROnPlateau` with the schedule state
  (`mode`, `factor`, `patience`, `min_lr`, `threshold`, `best`,
  `num_bad_steps`, `current_lr`, `initialized`). Mirrors upstream's
  attribute set at `lr_scheduler.py:1647-1687`.

- REQ-3: `pub fn ReduceLROnPlateau::new(mode: PlateauMode) -> Self`
  + builder methods `factor`, `patience`, `min_lr`, `threshold`.
  Defaults: `factor=0.1, patience=10, min_lr=0.0, threshold=1e-4`.
  Mirrors upstream defaults at `lr_scheduler.py:1647-1658`. The
  builder pattern is the R-DEV-7 deviation: upstream takes all
  kwargs at construction; Rust expresses this more cleanly via
  consuming builder methods.

- REQ-4: `pub trait MetricScheduler<T: Float>` with `fn step(&mut
  self, optimizer: &mut dyn Optimizer<T>, metric: f64)` and `fn
  get_lr(&self) -> f64`. This is SEPARATE from `LrScheduler`
  because the signature differs — plateau detection needs the
  metric value at each call. Mirrors upstream's `step(metrics)`
  signature at `lr_scheduler.py:1695-1720`.

- REQ-5: `impl<T: Float> MetricScheduler<T> for ReduceLROnPlateau`
  that on first call snapshots the optimizer's current LR, then
  on each subsequent call:
  1. Updates `best` if `metric` improves (per mode + threshold).
  2. Otherwise increments `num_bad_steps`.
  3. If `num_bad_steps > patience`, reduces LR by `factor`,
     respecting `min_lr` floor.

  Mirrors upstream `step`/`_reduce_lr` at
  `lr_scheduler.py:1695-1742`.

- REQ-6: NOT-STARTED — non-test production consumer. The
  `Learner in ferrotorch-train/src/learner.rs, 306-308` only
  accepts `Box<dyn LrScheduler<T>>`, not `Box<dyn
  MetricScheduler<T>>`. There is currently no
  metric-aware training driver in ferrotorch consuming this
  scheduler. The plateau scheduler exists as a pub API but is not
  wired into any non-test caller. Tracked by blocker #1475.

- REQ-7: NOT-STARTED — `cooldown`, `eps`, `threshold_mode='abs'`,
  per-param-group `min_lr` (upstream features at
  `lr_scheduler.py:1625-1632, 1626-1632, 1684`) are not exposed
  in the Rust builder. The Rust impl assumes relative threshold
  mode and a scalar `min_lr`. Tracked by blocker #1476.

## Acceptance Criteria

- [x] AC-1: `pub enum PlateauMode { Min, Max }`.
- [x] AC-2: `pub struct ReduceLROnPlateau` with the named state
  fields.
- [x] AC-3: Builder methods `factor`, `patience`, `min_lr`,
  `threshold` chain correctly.
- [x] AC-4: `pub trait MetricScheduler<T>` with the two methods.
- [x] AC-5: No reduction when metric improves
  (`test_plateau_no_reduction_when_improving`).
- [x] AC-6: Reduction triggers after `patience + 1` bad steps
  (`test_plateau_reduces_after_patience`).
- [x] AC-7: Reduction respects `min_lr` floor
  (`test_plateau_respects_min_lr`).
- [x] AC-8: `Max` mode reduces when metric stops increasing
  (`test_plateau_max_mode`).
- [x] AC-9: Bad-step counter resets on improvement
  (`test_plateau_resets_bad_count_on_improvement`).
- [ ] AC-10: Non-test production consumer in a training driver
  — blocker #1475.
- [ ] AC-11: `cooldown`, `eps`, `threshold_mode='abs'`,
  per-group `min_lr` — blocker #1476.

## Architecture

The plateau-detection algorithm:

```text
On first call: snapshot optimizer.lr() into current_lr.
If is_better(metric):
    best = metric
    num_bad_steps = 0
Else:
    num_bad_steps += 1
If num_bad_steps > patience:
    new_lr = max(current_lr * factor, min_lr)
    If new_lr < current_lr:
        current_lr = new_lr
        optimizer.set_lr(new_lr)
    num_bad_steps = 0
```

`is_better(metric)` for `Min`: `metric < best * (1 - threshold)`.
For `Max`: `metric > best * (1 + threshold)`. Matches upstream's
"rel" threshold mode at `lr_scheduler.py:1614-1618`.

The separate `MetricScheduler<T>` trait is necessary because the
`LrScheduler<T>::step(&mut self, &mut dyn Optimizer<T>)`
signature has no metric arg. Combining them with an overload
isn't possible in Rust; a separate trait is the cleanest solution.

### Non-test production consumers

- `ReduceLROnPlateau`, `PlateauMode`, and `MetricScheduler` are
  all re-exported at `ferrotorch-optim/src/lib.rs:47-52`.
- **No non-test production consumer currently exists.**
  `Learner::with_scheduler` only accepts `Box<dyn LrScheduler<T>>`;
  the metric-aware variant has no corresponding `with_metric_scheduler`
  hook. REQ-6 is NOT-STARTED.

## Parity contract

`parity_ops = []`. Numerical contract:

- **Relative threshold (only mode supported)**: improvement is
  `metric < best * (1 - threshold)` (Min mode). The `abs`
  threshold mode is NOT-STARTED, blocker #1476.
- **`min_lr` floor**: `new_lr = max(current_lr * factor, min_lr)`.
  If the reduction would push LR below `min_lr`, the LR is
  clamped at `min_lr` and the `bad_steps` counter still resets
  (which differs slightly from upstream — upstream uses `eps` to
  decide whether the LR is "really" lowered; ferrotorch always
  resets).
- **`patience == 0`**: every non-improving step is "intolerable",
  so the first plateau triggers an immediate reduction.
- **First-call snapshot**: the first `step` call captures
  `optimizer.lr()` into `current_lr`. If user code mutates
  `optimizer.lr()` between scheduler steps, the change won't be
  visible to plateau detection (R-CODE divergence opportunity —
  but matches upstream's "scheduler tracks the optimizer it was
  attached to" semantic).

## Verification

Tests in `#[cfg(test)] mod tests` (5 tests):

- `test_plateau_no_reduction_when_improving`
- `test_plateau_reduces_after_patience`
- `test_plateau_respects_min_lr`
- `test_plateau_max_mode`
- `test_plateau_resets_bad_count_on_improvement`

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib scheduler::plateau 2>&1 | tail -3
```

Expected: `5 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum PlateauMode { Min, Max }` with `Debug, Clone, Copy, PartialEq, Eq` derived in `scheduler/plateau.rs` mirrors `torch/optim/lr_scheduler.py:1650`; non-test consumer: re-exported at `ferrotorch-optim/src/lib.rs:47-52` (`PlateauMode` is in the `pub use` list); user code passes `PlateauMode::Min` or `::Max` to the constructor. |
| REQ-2 | SHIPPED | impl: `pub struct ReduceLROnPlateau` with the schedule state in `scheduler/plateau.rs` mirrors `torch/optim/lr_scheduler.py:1647-1687`; non-test consumer: re-exported at `ferrotorch-optim/src/lib.rs:47-52`. NOTE: this REQ being SHIPPED reflects existence + correct shape; REQ-6 (non-test consumer wiring into a training driver) is the open work. |
| REQ-3 | SHIPPED | impl: `pub fn ReduceLROnPlateau::new(mode) -> Self` plus the builder methods `factor`, `patience`, `min_lr`, `threshold` in `scheduler/plateau.rs` mirrors `torch/optim/lr_scheduler.py:1647-1687` with R-DEV-7 builder-style API; non-test consumer: the `pub use` at `lib.rs:47-52` is the user-call surface. |
| REQ-4 | SHIPPED | impl: `pub trait MetricScheduler<T: Float>` with `step(&mut self, &mut dyn Optimizer<T>, metric: f64)` + `get_lr` methods in `scheduler/plateau.rs` mirrors `torch/optim/lr_scheduler.py:1695` (the `step(metrics)` signature); non-test consumer: re-exported at `ferrotorch-optim/src/lib.rs:47-52`. NOTE: as with REQ-2, the structural existence is SHIPPED but the trait has no non-test production consumer wired into a training driver — that's REQ-6. |
| REQ-5 | SHIPPED | impl: `impl<T: Float> MetricScheduler<T> for ReduceLROnPlateau` with first-call LR snapshot + best-tracking + patience-driven reduction in `scheduler/plateau.rs` mirrors `torch/optim/lr_scheduler.py:1695-1742`; non-test consumer: as documented in REQ-4/REQ-6, no training driver currently invokes this. Test coverage (5 tests) verifies the algorithm. |
| REQ-6 | NOT-STARTED | blocker #1475 — `Learner::with_scheduler` only accepts `Box<dyn LrScheduler<T>>`. A `with_metric_scheduler` builder + corresponding per-epoch `metric_sched.step(opt, val_loss)` invocation is needed in `ferrotorch-train/src/learner.rs`. Without that wiring, `ReduceLROnPlateau` is a vocabulary-only API surface. |
| REQ-7 | NOT-STARTED | blocker #1476 — `cooldown`, `eps`, `threshold_mode='abs'`, per-param-group `min_lr` (upstream `torch/optim/lr_scheduler.py:1625-1632, 1684`) are not exposed in the Rust builder. The Rust impl assumes relative threshold mode + scalar `min_lr` + no cooldown. |
