# ferrotorch-optim — `scheduler::chained_scheduler` (ChainedScheduler)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/lr_scheduler.py
-->

## Summary

`ferrotorch-optim/src/scheduler/chained_scheduler.rs` defines
`pub struct ChainedScheduler<T: Float>`, a composite scheduler
that calls each of its inner schedulers' `step` once per outer
`step`. The final LR is whatever the last inner scheduler sets.
Mirrors `class ChainedScheduler(LRScheduler)` at
`torch/optim/lr_scheduler.py:1477-1581`.

## Requirements

- REQ-1: `pub struct ChainedScheduler<T: Float>` owning
  `schedulers: Vec<Box<dyn LrScheduler<T>>>`. Mirrors upstream's
  `_schedulers` attribute (`lr_scheduler.py:1534`).

- REQ-2: `pub fn ChainedScheduler::new(schedulers: Vec<Box<dyn
  LrScheduler<T>>>) -> Self` constructor that panics on empty
  input. Mirrors upstream's `if len(schedulers) < 1: raise
  ValueError` (`lr_scheduler.py:1511-1514`).

- REQ-3: `pub fn len(&self) -> usize` + `pub fn is_empty(&self)
  -> bool` accessors. Pure Rust ergonomics; not in upstream
  (Python uses `len(scheduler._schedulers)`).

- REQ-4: `impl<T: Float> LrScheduler<T> for ChainedScheduler<T>`
  whose `step` calls each inner scheduler's `step` in order with
  the same optimizer mutably borrowed. `get_lr` returns the last
  inner scheduler's `get_lr` (which is what was last written to
  the optimizer). Mirrors upstream `step` at
  `lr_scheduler.py:1538-1542`.

- REQ-5: Difference from `SequentialLr` documented: `ChainedScheduler`
  calls EVERY inner scheduler EVERY step (composed effects);
  `SequentialLr` calls ONE inner scheduler per step (switchover
  at milestones). Captured in the module-level doc-comment.

## Acceptance Criteria

- [x] AC-1: `pub struct ChainedScheduler<T: Float>` with the
  `schedulers` field.
- [x] AC-2: `ChainedScheduler::new` panics on empty input
  (`test_chained_empty_panics`).
- [x] AC-3: Single-scheduler chain behaves identically to the
  unwrapped scheduler (`test_chained_single_scheduler`).
- [x] AC-4: Multi-scheduler chain: the LAST scheduler's output
  wins (`test_chained_two_exponentials`).
- [x] AC-5: `get_lr` matches optimizer LR
  (`test_chained_get_lr`).
- [x] AC-6: `len()` and `is_empty()` work
  (`test_chained_len`).
- [x] AC-7: All inner schedulers advance their internal counters
  (`test_chained_all_schedulers_stepped`).

## Architecture

`pub struct ChainedScheduler<T: Float>` is generic over `T`
(unlike `SequentialLr<T>`, which uses the same `<T: Float>`
bound for the same reason — boxed-trait-object schedulers need
matching `T`).

```rust
fn step(&mut self, optimizer: &mut dyn Optimizer<T>) {
    for scheduler in &mut self.schedulers {
        scheduler.step(optimizer);
    }
}

fn get_lr(&self) -> f64 {
    self.schedulers.last().map(|s| s.get_lr()).unwrap_or(0.0)
}
```

The mutable borrow of `optimizer` is re-passed to each inner
scheduler; this is safe because the Rust borrow checker accepts
sequential (non-overlapping) re-borrows.

The "last scheduler wins" semantic differs from upstream's
multiplicative composition (`lr_scheduler.py:1538-1542`),
which works because upstream schedulers READ `group["lr"]` and
multiply it (so composition compounds). Ferrotorch's individual
`LrScheduler` impls use closed-form `compute_lr` that ignores
the optimizer's current LR, so chaining them gives "last writer
wins" instead of multiplicative composition. This is documented
in the module-level `//!` doc-comment.

### Non-test production consumers

- `ChainedScheduler` re-exported at
  `ferrotorch-optim/src/lib.rs:47-52`.
- `Learner::with_scheduler` at
  `ferrotorch-train/src/learner.rs:105` accepts the boxed
  `ChainedScheduler`; per-epoch step at
  `ferrotorch-train/src/learner.rs:306-308`.

## Parity contract

`parity_ops = []`. Numerical contract:

- **Empty chain**: rejected at construction.
- **Single-element chain**: behaves identically to the unwrapped
  scheduler.
- **Multi-element chain composition diverges from upstream**:
  ferrotorch's last-writer-wins semantic vs upstream's
  multiplicative composition. This is a documented R-DEV-1
  divergence consequence of the closed-form-vs-recursive
  implementation choice. Filing as a tracked behavioral
  difference is the right move if any code depends on the
  multiplicative semantic; per S8 noise, the in-doc warning is
  sufficient.
- **All inner schedulers advance**: even if the last writer wins
  the LR setting, all inner schedulers' internal state (e.g.
  `StepLR.current_step`) advances on every chain step. Verified
  by `test_chained_all_schedulers_stepped`.

## Verification

Tests in `#[cfg(test)] mod tests` (6 tests):

- `test_chained_single_scheduler`
- `test_chained_two_exponentials`
- `test_chained_get_lr`
- `test_chained_len`
- `test_chained_empty_panics`
- `test_chained_all_schedulers_stepped`

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib scheduler::chained_scheduler 2>&1 | tail -3
```

Expected: `6 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct ChainedScheduler<T: Float>` with `schedulers: Vec<Box<dyn LrScheduler<T>>>` field in `scheduler/chained_scheduler.rs` mirrors `torch/optim/lr_scheduler.py:1534`; non-test consumer: re-exported at `ferrotorch-optim/src/lib.rs:47-52`; user code boxes it for `Learner::with_scheduler` at `ferrotorch-train/src/learner.rs:105`. |
| REQ-2 | SHIPPED | impl: `pub fn ChainedScheduler::new(schedulers) -> Self` with `assert!(!schedulers.is_empty())` in `scheduler/chained_scheduler.rs` mirrors `torch/optim/lr_scheduler.py:1511-1514`; non-test consumer: the `pub use` at `lib.rs:47-52` is the user-call surface. |
| REQ-3 | SHIPPED | impl: `pub fn ChainedScheduler::len(&self)` + `pub fn is_empty(&self)` in `scheduler/chained_scheduler.rs`; non-test consumer: these accessors are part of the public API surface re-exported at `lib.rs:47-52`. NOTE: `len()` and `is_empty()` are exposed for user-code diagnostics; the test `test_chained_len` exercises them. The accessors are SHIPPED on the structural side; no in-crate non-test caller invokes them today, but they are reachable via the `pub use`. |
| REQ-4 | SHIPPED | impl: `impl<T: Float> LrScheduler<T> for ChainedScheduler<T>` calling each inner scheduler's `step` in order in `scheduler/chained_scheduler.rs` mirrors `torch/optim/lr_scheduler.py:1538-1542`; non-test consumer: `Learner` invokes `sched.step(self.optimizer.as_mut())` at `ferrotorch-train/src/learner.rs:306-308`, dispatching to this impl when the boxed scheduler is a `ChainedScheduler`. |
| REQ-5 | SHIPPED | The semantic difference from `SequentialLr` is documented in the module-level `//!` doc-comment at the top of `scheduler/chained_scheduler.rs` (lines 1-11). The doc explicitly states "ChainedScheduler calls every scheduler on every step" vs `SequentialLr`'s milestone switchover. The `test_chained_two_exponentials` test pins the "last writer wins" behavior. |
