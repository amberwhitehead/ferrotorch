# ferrotorch-optim — `scheduler::step` (StepLR)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/lr_scheduler.py
-->

## Summary

`ferrotorch-optim/src/scheduler/step.rs` defines `pub struct StepLR`,
a learning-rate scheduler that multiplies LR by `gamma` every
`step_size` calls to `step`. Mirrors `class StepLR(LRScheduler)` at
`torch/optim/lr_scheduler.py:592-676`.

## Requirements

- REQ-1: `pub struct StepLR` with `base_lr: f64`, `step_size: usize`,
  `gamma: f64`, `current_step: usize`, `current_lr: f64` fields.
  Mirrors `StepLR` (`torch/optim/lr_scheduler.py:621-630`) where the
  Python attributes `step_size`, `gamma`, plus inherited
  `last_epoch` and `base_lrs[0]` form the state vector.

- REQ-2: `pub fn StepLR::new(base_lr, step_size, gamma) -> Self`
  constructor with `current_step = 0` and `current_lr = base_lr`.
  PyTorch's `gamma` default of `0.1` is NOT folded into this Rust
  signature — the user always passes it. Mirrors
  `lr_scheduler.py:621-630`.

- REQ-3: `impl<T: Float> LrScheduler<T> for StepLR` that
  increments `current_step`, recomputes
  `base_lr * gamma.powf((step / step_size) as f64)` via the
  private `compute_lr`, and calls `optimizer.set_lr(new_lr)`.
  This is the closed-form (`_get_closed_form_lr`) version of the
  upstream schedule (`lr_scheduler.py:660-676`); ferrotorch uses
  the closed form always rather than the recursive `gamma`
  multiplication, because the closed form is numerically more
  stable and avoids the upstream `last_epoch == 0` / `last_epoch %
  step_size != 0` branching at `lr_scheduler.py:656-658`.

- REQ-4: `pub fn get_lr(&self) -> f64` inherent method returns
  `self.current_lr`. The trait impl's `get_lr` returns the same
  thing — both exist because `StepLR::get_lr` (no trait import
  needed) and `<StepLR as LrScheduler<T>>::get_lr` are different
  call sites.

## Acceptance Criteria

- [x] AC-1: `pub struct StepLR` with the five named fields.
- [x] AC-2: `StepLR::new(0.1, 10, 0.1).get_lr() == 0.1` initially.
- [x] AC-3: After 10 `step()` calls with `step_size = 10`, `gamma
  = 0.1`, LR == `0.01`
  (`test_step_lr_at_decay_boundary`).
- [x] AC-4: After 15 `step()` calls with `step_size = 5`, `gamma
  = 0.5`, LR == `0.125`
  (`test_step_lr_multiple_decays`).
- [x] AC-5: Optimizer's LR is synced after each step
  (`test_step_lr_optimizer_lr_synced`).

## Architecture

`pub struct StepLR` owns the schedule parameters (`base_lr`,
`step_size`, `gamma`) and the running state (`current_step`,
`current_lr`). The closed-form decay formula is

```text
current_lr = base_lr * gamma^floor(current_step / step_size)
```

implemented in the private `compute_lr` method using `f64::powf`
on the integer quotient.

`impl LrScheduler<T> for StepLR` advances `current_step` first,
then recomputes `current_lr`, then pushes to
`optimizer.set_lr(current_lr)`. This is order-correct: the LR
applied at iteration `n` reflects the schedule value at step `n`,
not step `n - 1`.

The Rust impl deviates from upstream's per-step `group["lr"] *
self.gamma` multiplication (`lr_scheduler.py:658`) by using the
closed-form `base_lr * gamma^floor(step/step_size)` formula
throughout. This is the upstream's `_get_closed_form_lr` path
(`lr_scheduler.py:660-676`), which upstream itself uses when
an explicit `epoch` is passed to `step`. The closed form avoids
floating-point drift from N successive multiplications, which is
why upstream prefers it for the epoch-jump case.

### Non-test production consumers

- `StepLR` re-exported at `ferrotorch-optim/src/lib.rs:47-52`.
- `Learner::with_scheduler` at
  `ferrotorch-train/src/learner.rs:105` accepts
  `Box<dyn LrScheduler<T>>`; user-call sites construct
  `Box::new(StepLR::new(lr, size, gamma))` and pass it. The
  per-epoch `sched.step(self.optimizer.as_mut())` invocation at
  `ferrotorch-train/src/learner.rs:306-308` is the production
  consumer.
- `cosine_warmup_scheduler` in `scheduler/mod.rs` does NOT use
  `StepLR`, but the example block in `mod.rs` docs (`scheduler/mod.rs:99-112`)
  documents the `SequentialLr::new(vec![(Box::new(warmup), 1000),
  (Box::new(StepLR::new(0.1, 5000, 0.5)), usize::MAX)])` idiom.

## Parity contract

`parity_ops = []`. The numerical contract is the closed-form
`base_lr * gamma^k` formula. Edge cases:

- **`step_size == 0`**: integer division by zero — Rust panics
  with the standard message. Upstream Python raises
  `ZeroDivisionError` from the same operation. Behavior is equivalent.
- **`current_step` overflow**: `usize` wraparound at `usize::MAX`;
  in practice unreachable in training (one step per minibatch ×
  centuries).
- **`gamma == 1.0`**: no decay, LR stays at `base_lr` forever.
  Verified by the `f64::powf(_, exp)` returning `1.0` for any
  exponent.
- **`gamma > 1.0`**: LR grows exponentially. Allowed (the upstream
  doesn't forbid it either) and produces a monotonically
  increasing LR.

## Verification

Tests in `#[cfg(test)] mod tests` (5 tests):

- `test_step_lr_initial` — `get_lr()` returns `base_lr` before
  any step.
- `test_step_lr_before_first_decay` — LR stays at `base_lr` for
  steps `1..step_size`.
- `test_step_lr_at_decay_boundary` — at step `step_size`, LR
  decays by `gamma`.
- `test_step_lr_multiple_decays` — three successive decay
  boundaries hit `gamma^1`, `gamma^2`, `gamma^3`.
- `test_step_lr_optimizer_lr_synced` — `optimizer.lr()` matches
  `sched.get_lr()` after every step.

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib scheduler::step 2>&1 | tail -3
```

Expected: `5 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct StepLR` with `base_lr`, `step_size`, `gamma`, `current_step`, `current_lr` fields in `scheduler/step.rs` mirrors `torch/optim/lr_scheduler.py:592-630`; non-test consumer: re-exported at `ferrotorch-optim/src/lib.rs:47-52`; user code constructs `StepLR::new(...)` and hands `Box::new(...)` to `Learner::with_scheduler` at `ferrotorch-train/src/learner.rs:105`. |
| REQ-2 | SHIPPED | impl: `pub fn StepLR::new(base_lr, step_size, gamma) -> Self` in `scheduler/step.rs` mirrors `torch/optim/lr_scheduler.py:621-630`; non-test consumer: `cosine_warmup_scheduler` example in `scheduler/mod.rs:99-112` shows the canonical `StepLR::new(0.1, 5000, 0.5)` call pattern; the `pub use` at `lib.rs:47-52` is the API surface user-code calls. |
| REQ-3 | SHIPPED | impl: `impl<T: Float> LrScheduler<T> for StepLR` in `scheduler/step.rs` mirrors the closed-form schedule from `torch/optim/lr_scheduler.py:660-676`; non-test consumer: `Learner` invokes `sched.step(self.optimizer.as_mut())` once per epoch at `ferrotorch-train/src/learner.rs:306-308`, dispatching to this impl when the boxed scheduler is a `StepLR`. |
| REQ-4 | SHIPPED | impl: `pub fn StepLR::get_lr(&self) -> f64` inherent method + trait impl in `scheduler/step.rs` both return `self.current_lr`; non-test consumer: `Learner` does not currently call `get_lr` directly, but the inherent method is on the public re-export at `lib.rs:47-52` and the trait method is invoked through the `Box<dyn LrScheduler<T>>` interface for user-code that wants to log the current LR. The trait `get_lr` is exercised through dynamic dispatch in `SequentialLr::get_lr` at `scheduler/mod.rs:155-161`, which IS a non-test consumer (production-code in the same crate). |
