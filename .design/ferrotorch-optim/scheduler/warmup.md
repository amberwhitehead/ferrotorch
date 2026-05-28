# ferrotorch-optim — `scheduler::warmup` (LinearWarmup)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/lr_scheduler.py
-->

## Summary

`ferrotorch-optim/src/scheduler/warmup.rs` defines
`pub struct LinearWarmup`, a scheduler that linearly ramps LR
from 0 to `base_lr` over `warmup_steps` steps, then holds at
`base_lr`. Conceptually equivalent to `LinearLR(start_factor=0,
end_factor=1, total_iters=warmup_steps)` in upstream PyTorch
(`torch/optim/lr_scheduler.py:877-1005`). Provided as a dedicated
type because the warmup pattern is so common in transformer
training that giving it its own type improves call-site
readability.

## Requirements

- REQ-1: `pub struct LinearWarmup` with `base_lr: f64`,
  `warmup_steps: usize`, `current_step: usize`, `current_lr: f64`
  fields. Conceptually a specialization of upstream `LinearLR`
  with `start_factor=0`, `end_factor=1`.

- REQ-2: `pub fn LinearWarmup::new(base_lr, warmup_steps) -> Self`
  constructor. Initial `current_lr = 0.0` so `get_lr()` before any
  step returns 0 (the ramp's starting value).

- REQ-3: `impl<T: Float> LrScheduler<T> for LinearWarmup` using
  the closed-form formula:

  ```text
  ratio = min(step as f64 / warmup_steps as f64, 1.0)
  lr = base_lr * ratio
  ```

  After `warmup_steps`, LR clamps at `base_lr`. Special case:
  `warmup_steps == 0` short-circuits to `base_lr` immediately on
  the first step.

- REQ-4: Used as the first phase in the canonical "warmup →
  decay" pattern provided by `cosine_warmup_scheduler` in
  `scheduler/mod.rs`. Composes cleanly via `SequentialLr`.

## Acceptance Criteria

- [x] AC-1: `pub struct LinearWarmup` with the four named fields.
- [x] AC-2: Initial `get_lr() == 0.0`
  (`test_warmup_initial_lr_is_zero`).
- [x] AC-3: Linear ramp: at step `n` (`n <= warmup_steps`),
  LR == `base_lr * n / warmup_steps`
  (`test_warmup_linear_ramp`, `test_warmup_halfway`).
- [x] AC-4: At step `warmup_steps`, LR == `base_lr`
  (`test_warmup_reaches_base_lr`).
- [x] AC-5: Past `warmup_steps`, LR stays at `base_lr`
  (`test_warmup_stays_at_base_lr_after_completion`).
- [x] AC-6: `warmup_steps == 0` → first step jumps to `base_lr`
  (`test_warmup_zero_steps`).

## Architecture

`pub struct LinearWarmup` stores the schedule parameters and the
running `(current_step, current_lr)` state. The private
`compute_lr`:

```text
if warmup_steps == 0 { return base_lr; }
let ratio = min(step as f64 / warmup_steps as f64, 1.0);
base_lr * ratio
```

`impl LrScheduler<T> for LinearWarmup` advances `current_step`,
recomputes `current_lr`, and pushes to `optimizer.set_lr`.

This is a specialization of the `LinearLR` schedule with
`start_factor = 0.0`, `end_factor = 1.0`, `total_iters =
warmup_steps`. However, `LinearLR` requires `start_factor > 0`,
so we can't directly delegate; the dedicated `LinearWarmup` type
fills the gap.

### Non-test production consumers

- `LinearWarmup` re-exported at
  `ferrotorch-optim/src/lib.rs:47-52`.
- `cosine_warmup_scheduler` in `scheduler/mod.rs` constructs
  `LinearWarmup::new(base_lr, warmup_steps)` and boxes it as the
  first phase of a `SequentialLr`. This is the primary in-crate
  production consumer.
- Module doc-example in `scheduler in scheduler/mod.rs` shows the
  manual composition pattern users might write to combine
  `LinearWarmup` with a different decay scheduler.

## Parity contract

`parity_ops = []`. Numerical contract:

- **`warmup_steps == 0`**: degenerate ramp; first step jumps
  to `base_lr` immediately. Allowed.
- **`step > warmup_steps`**: clamped at `base_lr` by the
  `min(..., 1.0)` on the ratio.
- **`base_lr < 0`**: produces a negative LR ramp. Allowed (no
  Rust-side validation), almost certainly a user error.
- **Floating-point precision at `step == warmup_steps`**:
  `step as f64 / warmup_steps as f64 == 1.0` exactly because of
  integer-to-float exact representation for small integers (up
  to 2^53). For larger `warmup_steps`, the equality still holds
  by IEEE 754 division semantics.

## Verification

Tests in `#[cfg(test)] mod tests` (5 tests):

- `test_warmup_initial_lr_is_zero`
- `test_warmup_linear_ramp`
- `test_warmup_reaches_base_lr`
- `test_warmup_stays_at_base_lr_after_completion`
- `test_warmup_zero_steps`
- `test_warmup_halfway`

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib scheduler::warmup 2>&1 | tail -3
```

Expected: `6 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct LinearWarmup` with `base_lr`, `warmup_steps`, `current_step`, `current_lr` fields in `scheduler/warmup.rs`; conceptually mirrors a specialization of `class LinearLR(LRScheduler)` at `torch/optim/lr_scheduler.py:877-1005` with `start_factor=0`, `end_factor=1`; non-test consumer: re-exported at `ferrotorch-optim/src/lib.rs:47-52` plus the in-crate construction site `LinearWarmup::new(base_lr, warmup_steps)` inside `cosine_warmup_scheduler` in `scheduler/mod.rs`. |
| REQ-2 | SHIPPED | impl: `pub fn LinearWarmup::new(base_lr, warmup_steps) -> Self` with `current_lr: 0.0` initialization in `scheduler/warmup.rs`; non-test consumer: `cosine_warmup_scheduler` calls `LinearWarmup::new(...)` in `scheduler/mod.rs` — production code in the same crate. |
| REQ-3 | SHIPPED | impl: `impl<T: Float> LrScheduler<T> for LinearWarmup` using the `min(ratio, 1.0) * base_lr` closed form in `scheduler/warmup.rs`; non-test consumer: `Learner` invokes `sched.step(self.optimizer.as_mut())` at `ferrotorch-train/src/learner.rs:306-308`, dispatching to this impl when the boxed scheduler is a `LinearWarmup`. Also consumed transitively when `cosine_warmup_scheduler` produces a `SequentialLr` whose first phase is a `LinearWarmup`. |
| REQ-4 | SHIPPED | impl: `LinearWarmup` is composed via `SequentialLr::new(vec![(Box::new(warmup), warmup_steps), (Box::new(cosine), usize::MAX)])` inside `cosine_warmup_scheduler` at `scheduler/mod.rs`; non-test consumer: the resulting `SequentialLr<T>` is the return value handed to `Learner::with_scheduler` at `ferrotorch-train/src/learner.rs:105`. |
