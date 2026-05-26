# ferrotorch-optim — `scheduler::lambda_lr` (LambdaLR)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/lr_scheduler.py
-->

## Summary

`ferrotorch-optim/src/scheduler/lambda_lr.rs` defines
`pub struct LambdaLR`, a scheduler that sets LR to
`base_lr * lr_lambda(current_step)` each step using a
user-provided closure. Mirrors `class LambdaLR(LRScheduler)` at
`torch/optim/lr_scheduler.py:343-466`.

## Requirements

- REQ-1: `pub struct LambdaLR` with `base_lr: f64`, `lr_lambda:
  Box<dyn Fn(usize) -> f64>`, `current_step: usize`, `current_lr:
  f64` fields. Mirrors upstream's `lr_lambdas: list[Callable]`
  attribute (`lr_scheduler.py:386-394`) with the
  single-callable-per-scheduler simplification — upstream
  supports a list of lambdas, one per param group; ferrotorch
  uses a single closure because the `Optimizer` trait exposes a
  single `set_lr`. This is the R-DEV-4 deviation: param-group
  granularity is upstream's choice; ferrotorch's single-LR
  optimizer surface doesn't need it.

- REQ-2: `pub fn LambdaLR::new(base_lr, lr_lambda) -> Self`
  generic constructor that boxes the closure. Mirrors
  `LambdaLR.__init__` (`lr_scheduler.py:378-395`). The closure
  is stored as `Box<dyn Fn(usize) -> f64>` rather than borrowed
  because the scheduler outlives any stack closure passed in.

- REQ-3: `impl<T: Float> LrScheduler<T> for LambdaLR` increments
  `current_step`, computes `base_lr * lr_lambda(current_step)`,
  and calls `optimizer.set_lr`. Mirrors upstream's `get_lr`
  (`lr_scheduler.py:441-466`) — the closed form is the same as
  the recurrence in this case because `lr_lambda(step)` is the
  schedule's closed form by construction.

## Acceptance Criteria

- [x] AC-1: `pub struct LambdaLR` with the four named fields.
- [x] AC-2: `LambdaLR::new(base_lr, |_| 1.0)` → LR stays at
  `base_lr` (`test_lambda_lr_constant_factor`).
- [x] AC-3: `LambdaLR::new(base_lr, |epoch| gamma.powi(epoch))`
  reproduces `ExponentialLR` behavior
  (`test_lambda_lr_exponential_decay`).
- [x] AC-4: User can use any `Fn(usize) -> f64` closure including
  step functions (`test_lambda_lr_step_function`).
- [x] AC-5: `get_lr()` and `optimizer.lr()` stay in sync
  (`test_lambda_lr_get_lr_matches_optimizer`).

## Architecture

`pub struct LambdaLR` stores the boxed closure and the running
state. The lambda receives the current step (1-indexed for the
first `step()` call) and returns a multiplicative factor.

`impl LrScheduler<T> for LambdaLR` is a thin wrapper:

```text
self.current_step += 1;
self.current_lr = self.base_lr * (self.lr_lambda)(self.current_step);
optimizer.set_lr(self.current_lr);
```

The closure-storage choice (`Box<dyn Fn>`) means `LambdaLR` is
not `Clone` and not `Debug` (visible via the lack of those
derives on the struct). This is intentional — function pointers
in Rust can't be cloned or formatted, and faking it would create
a divergence from what `dyn Fn` actually guarantees.

### Non-test production consumers

- `LambdaLR` re-exported at
  `ferrotorch-optim/src/lib.rs:47-52`.
- `Learner::with_scheduler` at
  `ferrotorch-train/src/learner.rs:105` accepts
  `Box<dyn LrScheduler<T>>`; user-code constructs
  `Box::new(LambdaLR::new(lr, |e| ...))` and hands it in. The
  per-epoch `sched.step(...)` at
  `ferrotorch-train/src/learner.rs:306-308` is the production
  consumer.

## Parity contract

`parity_ops = []`. Numerical contract:

- **Closure returns NaN or Inf**: `optimizer.set_lr(f64::NAN)`
  propagates, breaking subsequent training. User responsibility.
- **Closure with side effects** (e.g. mutable state inside the
  closure): allowed by `Fn` semantics — must use interior
  mutability if state is needed. The Rust compiler enforces.
- **`current_step` overflow at `usize::MAX`**: wraparound;
  unreachable in practice.

## Verification

Tests in `#[cfg(test)] mod tests` (4 tests):

- `test_lambda_lr_constant_factor`
- `test_lambda_lr_exponential_decay`
- `test_lambda_lr_step_function`
- `test_lambda_lr_get_lr_matches_optimizer`

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib scheduler::lambda_lr 2>&1 | tail -3
```

Expected: `4 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct LambdaLR` with `base_lr`, `lr_lambda: Box<dyn Fn(usize) -> f64>`, `current_step`, `current_lr` fields in `scheduler/lambda_lr.rs` mirrors `torch/optim/lr_scheduler.py:386-394` (R-DEV-4: single closure instead of `list[Callable]` because ferrotorch optimizer surface has one LR); non-test consumer: re-exported at `ferrotorch-optim/src/lib.rs:47-52`; user code boxes it for `Learner::with_scheduler` at `ferrotorch-train/src/learner.rs:105`. |
| REQ-2 | SHIPPED | impl: `pub fn LambdaLR::new(base_lr, lr_lambda) -> Self` taking `impl Fn(usize) -> f64 + 'static` in `scheduler/lambda_lr.rs` mirrors `torch/optim/lr_scheduler.py:378-395`; non-test consumer: the `pub use` at `lib.rs:47-52` is the user-call surface. |
| REQ-3 | SHIPPED | impl: `impl<T: Float> LrScheduler<T> for LambdaLR` in `scheduler/lambda_lr.rs` mirrors `torch/optim/lr_scheduler.py:441-466`; non-test consumer: `Learner` invokes `sched.step(self.optimizer.as_mut())` at `ferrotorch-train/src/learner.rs:306-308`, dispatching to this impl when the boxed scheduler is a `LambdaLR`. |
