# ferrotorch-optim — `GradientAccumulator` (large-effective-batch helper)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/optimizer.py
-->

## Summary

`ferrotorch-optim/src/grad_accumulator.rs` defines
`GradientAccumulator`, the helper that lets a training loop simulate
a larger effective batch size than physical memory allows by
accumulating gradients across multiple micro-batches before
stepping the optimizer. Gradients are normalised so the accumulated
batch has the same scale as a single large-batch gradient
(loss is divided by `accumulation_steps` before `backward()`, and
the resulting per-micro-batch `.grad` tensors sum to the
large-batch equivalent gradient through PyTorch / ferrotorch's
gradient-accumulation contract).

PyTorch ships no canonical `GradientAccumulator` class — the
upstream idiom is a manual `if (step + 1) % accumulation_steps ==
0: optimizer.step(); optimizer.zero_grad()` pattern at the user
training-loop level (documented in
`torch/optim/optimizer.py` doctrings and the PyTorch tutorial
`Gradient Accumulation`). ferrotorch packages the bookkeeping as a
reusable struct because R-DEV-7 applies: the Rust ecosystem analog
(a small bookkeeping struct) is materially better than ad-hoc
modular-arithmetic in user code; users get a typed counter with a
panic-on-zero invariant.

## Requirements

- REQ-1: `pub struct GradientAccumulator { accumulation_steps:
  usize, current_step: usize }` — the bookkeeping struct.
  `#[derive(Debug, Clone)]`. Both fields are crate-private (no
  external mutation outside the public methods).
- REQ-2: `pub fn new(steps: usize) -> Self` — panics if
  `steps == 0` (the zero-step accumulator is a programming error;
  the panic is documented in the doc-comment and pinned by a
  `#[should_panic]` test).
- REQ-3: `pub fn should_step(&mut self) -> bool` — increments
  `current_step`, returns `true` and resets to 0 when the counter
  hits `accumulation_steps`, otherwise returns `false`. This is
  the "step gate" pattern callers use as
  `if accum.should_step() { optimizer.step(); optimizer.zero_grad(); }`.
- REQ-4: `pub fn scale_loss<T: Float>(&self, loss: &Tensor<T>) ->
  FerrotorchResult<Tensor<T>>` — returns `loss /
  accumulation_steps` as a fresh tensor (via
  `arithmetic::mul(loss, 1/accumulation_steps)`). Preserves
  autograd edges so `scaled.backward()` distributes the same
  total gradient across micro-batches.
- REQ-5: `pub fn accumulation_steps(&self) -> usize` /
  `pub fn current_step(&self) -> usize` / `pub fn reset(&mut self)`
  — diagnostic accessors and a manual reset (useful when a
  training loop bails out mid-accumulation and wants to restart
  the window cleanly).

## Acceptance Criteria

- [x] AC-1: `GradientAccumulator::new(0)` panics with the message
  `"accumulation_steps must be >= 1"`.
- [x] AC-2: `should_step` cycles correctly: for `steps=3`, returns
  `false, false, true, false, false, true, ...`.
- [x] AC-3: `should_step` with `steps=1` always returns `true`.
- [x] AC-4: `scale_loss` divides by `accumulation_steps`
  (e.g. `8.0 / 4 == 2.0`).
- [x] AC-5: `scale_loss` works on vector-valued tensors.
- [x] AC-6: `scale_loss` with `steps=1` is identity.
- [x] AC-7: `reset()` sets `current_step` to 0.

## Architecture

### `should_step` (REQ-3)

```text
should_step(&mut self):
  self.current_step += 1
  if self.current_step >= self.accumulation_steps:
    self.current_step = 0
    return true
  return false
```

The increment-then-check ordering matters: callers invoke
`accum.should_step()` exactly once per micro-batch, AFTER
`scaled.backward()`. So step `i = 1..N` increments then checks;
when `i == N` the counter is reset and `true` is returned.

### `scale_loss` (REQ-4)

```text
scale_loss(loss):
  factor = 1.0 / accumulation_steps    (as scalar Tensor<T>)
  return mul(loss, factor)
```

`mul` is `ferrotorch_core::grad_fns::arithmetic::mul`, so the
returned tensor carries an autograd edge back to `loss` and a
constant edge for the scalar (which has `requires_grad=false`).
When the caller calls `scaled.backward()`, `loss` accumulates
`1/N * upstream_grad` into the model parameters via the chain
rule — exactly the contract the PyTorch tutorial documents.

The cast `f64 -> T` uses `cast::<f64, T>`, so a non-representable
factor returns `FerrotorchError::InvalidArgument` rather than
panicking. (Practically: `accumulation_steps` is `usize`, so
`1.0/N` is always representable in `f32`/`f64`/`f16`/`bf16` for
realistic `N`.)

### `new` panic invariant (REQ-2)

`assert!(steps > 0, "accumulation_steps must be >= 1");`. This is
the only `assert!` in the file; the alternative ("return
`Result`") would force every caller to handle an error that can
only fire under programmer misuse. The panic is the cleanest
expression of "this is a precondition the type system can't
encode" — R-DEV-5 / typestate-where-it-fits would push toward a
`NonZeroUsize` API, but the unit type ergonomics there are worse
than the panic-on-zero pattern.

### Non-test production consumers

The struct is the public API surface of large-effective-batch
training loops. Per goal.md S5 ("Boundary methods ARE the public
API"), it's SHIPPED via the crate-root re-export at
`ferrotorch-optim/src/lib.rs:36`
(`pub use grad_accumulator::GradientAccumulator;`). The
integration test
`ferrotorch-optim/tests/conformance_optim_advanced.rs:49`
(`use ferrotorch_optim::grad_accumulator::GradientAccumulator;`)
+ fixture-driven cases at lines 1175, 1195, 1216, 1237, 1255
exercise the contract.

No in-tree training-loop integration exists yet; the helper is
intentionally exposed at the crate boundary as a building block
for downstream user training loops. The doctring example in the
module comment shows the standard usage pattern.

## Parity contract

`parity_ops = []`. The accumulator is a bookkeeping helper; no
numerical comparison against an upstream `GradientAccumulator`
class exists because PyTorch's accumulation is user-code-driven.
Edge cases owned:

- **`steps == 0`**: panic at construction time (debug AND release).
- **`should_step` after `reset`**: counter starts at 0; the next
  `should_step` increments to 1 and the cycle resumes from there.
- **`scale_loss` with `steps == 1`**: returns `loss * 1.0`
  (identity modulo `mul`'s graph contribution).
- **`scale_loss` on a `requires_grad=false` tensor**: returns a
  fresh `requires_grad=false` tensor (no autograd edge to record).
- **Overflow / underflow at very large `steps`**: `1.0 / steps`
  underflows to subnormal at `steps > 2^53` for `f64`. Not a
  practical concern (no realistic training loop has 2^53
  micro-batches).

## Verification

Seven unit tests in `mod tests` (grad_accumulator.rs lines 93-173):

- `test_should_step_cycles` — `false, false, true, false, false, true`
  for `steps=3`.
- `test_should_step_one` — `true` every call for `steps=1`.
- `test_scale_loss_divides_by_steps` — `8.0/4 == 2.0`.
- `test_scale_loss_vector` — vector-valued loss.
- `test_scale_loss_steps_one_identity` — identity at `steps=1`.
- `test_reset` — counter cleared.
- `test_zero_steps_panics` — `#[should_panic(expected =
  "accumulation_steps must be >= 1")]`.

Plus fixture-driven integration tests in
`ferrotorch-optim/tests/conformance_optim_advanced.rs` (lines
1175-1257) — `gradient_accumulator_*_matches_reference` — that
compare against captured PyTorch reference behaviour for the
divide-then-accumulate-then-step contract.

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib grad_accumulator:: 2>&1 | tail -3
```

Expected: `7 passed; 0 failed` (the `#[should_panic]` test counts
as passing when it panics).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct GradientAccumulator` at `GradientAccumulator in ferrotorch-optim/src/grad_accumulator.rs` with `#[derive(Debug, Clone)]`; non-test consumer: `pub use grad_accumulator::GradientAccumulator` at `ferrotorch-optim/src/lib.rs` — boundary-method public API per goal.md S5. |
| REQ-2 | SHIPPED | impl: `pub fn new` at `new in ferrotorch-optim/src/grad_accumulator.rs` with `assert!(steps > 0, ...)`; non-test consumer: same `pub use` re-export at `lib.rs`; pinned by `test_zero_steps_panics` (`#[should_panic]`). |
| REQ-3 | SHIPPED | impl: `pub fn should_step` at `should_step in ferrotorch-optim/src/grad_accumulator.rs`; non-test consumer: same `pub use` re-export at `lib.rs`; pinned by `test_should_step_cycles` and `test_should_step_one`. |
| REQ-4 | SHIPPED | impl: `pub fn scale_loss` at `scale_loss in ferrotorch-optim/src/grad_accumulator.rs` invoking `arithmetic::mul(loss, scalar(1.0/steps))`; non-test consumer: same `pub use` re-export at `lib.rs`; pinned by `test_scale_loss_divides_by_steps` and `test_scale_loss_vector`. |
| REQ-5 | SHIPPED | impl: `accumulation_steps` at `ferrotorch-optim/src/grad_accumulator.rs:74`, `current_step` at line 79, `reset` at line 84; non-test consumer: same `pub use` re-export at `lib.rs:36`; `current_step` + `reset` pinned by `test_reset`. |
