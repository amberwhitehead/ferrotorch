# ferrotorch-optim — `scheduler::multiplicative_lr` (MultiplicativeLR)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/lr_scheduler.py
-->

## Summary

`ferrotorch-optim/src/scheduler/multiplicative_lr.rs` defines
`pub struct MultiplicativeLR`, a scheduler that compounds the
current LR by a user-provided factor each step (`lr = lr *
lr_lambda(step)`). Mirrors
`class MultiplicativeLR(LRScheduler)` at
`torch/optim/lr_scheduler.py:469-590`.

## Requirements

- REQ-1: `pub struct MultiplicativeLR` with `lr_lambda: Box<dyn
  Fn(usize) -> f64>`, `current_step: usize`, `current_lr: f64`
  fields. Mirrors upstream's `lr_lambdas` attribute
  (`lr_scheduler.py:501-509`) with the same single-callable
  simplification as `LambdaLR`. Note: no `base_lr` field is
  stored after construction — `current_lr` is initialized to the
  `base_lr` argument and compounded from there.

- REQ-2: `pub fn MultiplicativeLR::new(base_lr, lr_lambda) -> Self`
  generic constructor. Mirrors `MultiplicativeLR.__init__`
  (`lr_scheduler.py:493-520`). `current_lr` starts at `base_lr`;
  the first `step` call multiplies it by `lr_lambda(1)`.

- REQ-3: `impl<T: Float> LrScheduler<T> for MultiplicativeLR`
  that increments `current_step`, evaluates the closure on the
  new step, multiplies `current_lr *= factor`, and pushes to
  `optimizer.set_lr`. Mirrors upstream's `get_lr` at
  `lr_scheduler.py:541-590` (which multiplies `group["lr"] *
  lmbda(last_epoch)` per param group).

- REQ-4: Difference from `LambdaLR` documented: `LambdaLR`
  recomputes from `base_lr` each step (absolute), while
  `MultiplicativeLR` compounds onto the previous LR (relative).
  Captured in the module doc-comment.

## Acceptance Criteria

- [x] AC-1: `pub struct MultiplicativeLR` with the three named
  fields.
- [x] AC-2: Constant factor compounds correctly
  (`test_multiplicative_constant_factor`).
- [x] AC-3: Factor 1.0 leaves LR unchanged
  (`test_multiplicative_factor_one`).
- [x] AC-4: Epoch-dependent factors compound correctly
  (`test_multiplicative_epoch_dependent`).
- [x] AC-5: Constant `gamma` factor matches `ExponentialLR`
  behavior (`test_multiplicative_vs_exponential`).
- [x] AC-6: Sparse halving pattern works
  (`test_multiplicative_halving`).

## Architecture

`pub struct MultiplicativeLR` carries the boxed closure and the
running state. The step body:

```text
self.current_step += 1;
let factor = (self.lr_lambda)(self.current_step);
self.current_lr *= factor;
optimizer.set_lr(self.current_lr);
```

Unlike `LambdaLR`, this is intrinsically a recursive formula —
the closed form would be `base_lr * prod(lambda(i) for i in
1..=step)`, which is more expensive than the per-step
multiplication unless cached. Upstream does the same per-step
multiply (`lr_scheduler.py:541-590`); the closed form is not
provided by upstream either.

### Non-test production consumers

- `MultiplicativeLR` re-exported at
  `ferrotorch-optim/src/lib.rs:47-52`.
- `Learner::with_scheduler` at
  `ferrotorch-train/src/learner.rs:105` accepts the boxed
  `MultiplicativeLR`; per-epoch step at
  `ferrotorch-train/src/learner.rs:306-308`.

## Parity contract

`parity_ops = []`. Numerical contract:

- **Closure returns 0.0**: `current_lr` becomes 0 and stays
  there (multiplications of 0 are 0). Matches upstream.
- **Closure returns negative**: `current_lr` flips sign each
  time. Allowed; almost certainly a user error.
- **Closure returns Inf or NaN**: propagates through subsequent
  multiplications, breaking training. User responsibility.
- **Floating-point drift over many steps**: the recursive
  multiply form accumulates error; for long training runs
  consider `LambdaLR` with a closed-form lambda instead.

## Verification

Tests in `#[cfg(test)] mod tests` (6 tests):

- `test_multiplicative_constant_factor`
- `test_multiplicative_factor_one`
- `test_multiplicative_epoch_dependent`
- `test_multiplicative_get_lr_matches_optimizer`
- `test_multiplicative_vs_exponential`
- `test_multiplicative_halving`

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib scheduler::multiplicative_lr 2>&1 | tail -3
```

Expected: `6 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct MultiplicativeLR` with `lr_lambda: Box<dyn Fn(usize) -> f64>`, `current_step`, `current_lr` fields in `scheduler/multiplicative_lr.rs` mirrors `torch/optim/lr_scheduler.py:501-509` (R-DEV-4: single closure); non-test consumer: re-exported at `ferrotorch-optim/src/lib.rs:47-52`; user code boxes it for `Learner::with_scheduler` at `ferrotorch-train/src/learner.rs:105`. |
| REQ-2 | SHIPPED | impl: `pub fn MultiplicativeLR::new(base_lr, lr_lambda) -> Self` in `scheduler/multiplicative_lr.rs` mirrors `torch/optim/lr_scheduler.py:493-520`; non-test consumer: the `pub use` at `lib.rs:47-52` is the user-call surface. |
| REQ-3 | SHIPPED | impl: `impl<T: Float> LrScheduler<T> for MultiplicativeLR` in `scheduler/multiplicative_lr.rs` mirrors `torch/optim/lr_scheduler.py:541-590`; non-test consumer: `Learner` invokes `sched.step(self.optimizer.as_mut())` at `ferrotorch-train/src/learner.rs:306-308`, dispatching to this impl when the boxed scheduler is a `MultiplicativeLR`. |
| REQ-4 | SHIPPED | The semantic difference from `LambdaLR` is documented in the module-level `//!` doc-comment at the top of `scheduler/multiplicative_lr.rs`; the comparison test `test_multiplicative_vs_exponential` verifies that constant `gamma` produces the same trajectory as `ExponentialLR`, confirming the relative-compound semantics. |
