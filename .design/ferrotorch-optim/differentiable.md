# ferrotorch-optim — `differentiable` (meta-learning SGD steps)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/optimizer.py
  - torch/optim/sgd.py
-->

## Summary

`ferrotorch-optim/src/differentiable.rs` implements the
meta-learning analog of `Sgd::step()` — a pair of pure functions
(`diff_sgd_step`, `diff_sgd_momentum_step`) that perform an SGD /
SGD-with-momentum update by RETURNING fresh tensors with autograd
edges, instead of mutating the input parameters in place under a
`no_grad()` guard the way the regular `Sgd` optimizer does. This is
the kernel the MAML family (Finn et al. 2017) and other
meta-learning algorithms need: the outer loss of the adapted
parameters must be differentiable back to the original parameters
through the inner-loop update. Mirrors PyTorch's
`@_use_grad_for_differentiable` decorator-driven branch in
`torch.optim.optimizer.Optimizer.step` (`torch/optim/optimizer.py:59-87`)
when `differentiable=True` is set on the optimizer.

Tracked under CL-389. Out-of-scope here: differentiable Adam,
AdamW, etc. — those exist in upstream PyTorch via the same
decorator but are NOT-STARTED in ferrotorch; this file
intentionally ships SGD-only.

## Requirements

- REQ-1: `pub fn diff_sgd_step<T: Float>(params, grads, lr) ->
  FerrotorchResult<Vec<Tensor<T>>>` computes
  `new_param[i] = param[i] - lr * grad[i]` for every paired
  `(param, grad)`, returning fresh tensors. The arithmetic uses
  `add`/`sub`/`mul` from `ferrotorch_core::grad_fns::arithmetic`
  so every step contributes an autograd edge.
- REQ-2: `diff_sgd_step` validates inputs: `params.len() ==
  grads.len()` and per-element `param.shape() == grad.shape()`. On
  mismatch, returns `FerrotorchError::InvalidArgument` (length) or
  `FerrotorchError::ShapeMismatch` (shape).
- REQ-3: `lr` is `f64` (matching the PyTorch kwarg type); the
  function casts to `T` via `T::from(lr)` and returns
  `FerrotorchError::InvalidArgument` when the cast fails (e.g.
  `f64::INFINITY` into `f16`).
- REQ-4: The scalar `lr` tensor is materialised on each parameter's
  own device; mixed-device parameter lists are supported (the
  device is read off `param.device()` inside the loop).
- REQ-5: `pub fn diff_sgd_momentum_step<T: Float>(params, grads,
  prev_velocities, lr, momentum) -> FerrotorchResult<(Vec<Tensor<T>>,
  Vec<Tensor<T>>)>` returns BOTH the adapted parameters AND the
  new velocity buffers so the caller can chain multi-step inner
  loops. The recurrence is `v_new = momentum * v_prev + grad`
  (with `v_new = grad` on the first step), then
  `new_param = param - lr * v_new`.
- REQ-6: `diff_sgd_momentum_step` accepts `prev_velocities` of
  length 0 (first-step shortcut: `v_new == grad`) or of length
  `params.len()` (steady state). Any other length returns
  `FerrotorchError::InvalidArgument`.
- REQ-7: `pub type DiffSgdMomentumOutput<T> = (Vec<Tensor<T>>,
  Vec<Tensor<T>>)` is a type alias for the
  `diff_sgd_momentum_step` return type. Convenience for callers
  that destructure repeatedly.
- REQ-8: The autograd chain is preserved end-to-end:
  differentiating a loss computed from the adapted parameters
  back to the original `theta` yields a non-`None` gradient on
  `theta` (the inner-loop update has not opaqued the graph). This
  is the central correctness property — without it, MAML's
  second-order gradient does not flow.

## Acceptance Criteria

- [x] AC-1: `diff_sgd_step([p], [g], lr)` returns `[p - lr*g]`
  with values matching the closed form.
- [x] AC-2: `diff_sgd_step` length mismatch returns `Err`.
- [x] AC-3: `diff_sgd_step` shape mismatch returns `Err`.
- [x] AC-4: `diff_sgd_step` multi-parameter case computes the
  per-element update.
- [x] AC-5: Autograd flows: `theta.requires_grad=true`,
  `adapted = diff_sgd_step([theta], [g], lr)`,
  `loss = sum(adapted[0])`, `loss.backward()` ⇒
  `theta.grad() == ones_like(theta)`.
- [x] AC-6: Second-order gradient: when the inner-loop `grad`
  itself has `requires_grad=true`, differentiating through the
  adapted parameters back to that `grad` yields `-lr` (i.e. the
  graph correctly tracks `∂(theta - lr*g)/∂g = -lr`).
- [x] AC-7: `diff_sgd_momentum_step` first-step shortcut
  (empty `prev_velocities`) yields `v_new = grad`,
  `p_new = p - lr*grad`.
- [x] AC-8: `diff_sgd_momentum_step` steady-state
  (`prev_velocities.len() == params.len()`) computes
  `v_new = momentum * v_prev + grad`, `p_new = p - lr * v_new`.
- [x] AC-9: `diff_sgd_momentum_step` autograd chain through
  velocity is preserved.

## Architecture

### `diff_sgd_step` (REQ-1..4)

```text
diff_sgd_step(params, grads, lr):
  lr_t = T::from(lr)  // or error
  for each (p, g):
    check p.shape() == g.shape()
    lr_scalar = scalar(lr_t).to(p.device())
    update = mul(g, lr_scalar)
    new_param = sub(p, update)
    push new_param
  return new_params
```

Every arithmetic op (`mul`, `sub`) routes through the
`ferrotorch_core::grad_fns::arithmetic` family, which builds the
autograd `grad_fn` chain. So the returned tensor's
`.grad_fn()` chain walks back through `sub` → `mul` → the input
`p` and `g` (and the constant `lr_scalar`, which has
`requires_grad=false` so no edge contributes).

### `diff_sgd_momentum_step` (REQ-5..9)

```text
diff_sgd_momentum_step(params, grads, prev_v, lr, momentum):
  lr_t, mom_t = casts
  for i, (p, g):
    if prev_v.is_empty():
      v_new = g.clone()                          // first step shortcut
    else:
      check prev_v[i].shape() == g.shape()
      scaled_v = mul(prev_v[i], mom_scalar)
      v_new = add(scaled_v, g)
    update = mul(v_new, lr_scalar)
    new_param = sub(p, update)
  return (new_params, new_velocities)
```

Returning the new velocities matters: a multi-step MAML inner
loop chains
`(params, velocities) = diff_sgd_momentum_step(params,
grads, velocities, lr, mom)` repeatedly, with each iteration's
`velocities` carrying autograd edges back to the original
`grad` tensors. PyTorch's `differentiable=True` SGD impl in
`torch/optim/sgd.py` does the same (`sgd` functional applied with
`differentiable=True` does not detach `momentum_buffer`).

### Type alias `DiffSgdMomentumOutput<T>` (REQ-7)

Cosmetic — saves callers from spelling out
`(Vec<Tensor<T>>, Vec<Tensor<T>>)` everywhere.

### Non-test production consumers

The two functions are the public API surface of meta-learning
inner loops. Per goal.md S5 ("Boundary methods ARE the public API;
they don't need further downstream callers to be SHIPPED"),
they're SHIPPED through the crate-root re-export
(`ferrotorch-optim/src/lib.rs:34` `pub use differentiable::{diff_sgd_momentum_step, diff_sgd_step};`).
The integration test
`ferrotorch-optim/tests/conformance_optim_advanced.rs:46`
(`use ferrotorch_optim::differentiable::{diff_sgd_momentum_step, diff_sgd_step};`)
exercises them against PyTorch reference fixtures.

No in-tree MAML implementation exists yet; the functions are
intentionally exposed as the kernel that a future
`ferrotorch-meta` (out-of-scope per goal.md) or downstream user
code would consume.

## Parity contract

`parity_ops = []`. Numerical contract:

- **NaN propagation**: `param[i]` or `grad[i]` containing NaN
  poisons the corresponding adapted tensor (matches
  `torch.optim.SGD` with `differentiable=True`).
- **Inf**: same propagation behaviour.
- **Empty tensor**: works (every arithmetic op handles
  zero-element tensors; the loop body never iterates).
- **`lr == 0.0`**: returns `params` unchanged (modulo dtype
  round-trip through the scalar tensor).
- **`lr` not representable in `T`**: returns
  `InvalidArgument`.
- **Mixed devices across parameters**: supported — each iteration
  re-materialises the scalar on `p.device()`.
- **`prev_velocities.is_empty()` first-step branch**: `v_new`
  is `g.clone()`, which preserves `g`'s autograd edges. Matches
  `torch.optim.SGD`'s first-step `momentum_buffer = grad` init.

## Verification

Eight unit tests in `mod tests` (differentiable.rs lines 186-347):

- `test_diff_sgd_step_shape_and_values` — closed-form check.
- `test_diff_sgd_step_length_mismatch_errors` /
  `_shape_mismatch_errors` — input validation.
- `test_diff_sgd_step_multiple_params` — multi-parameter case.
- `test_diff_sgd_step_autograd_edge_to_param` — `theta.grad()`
  populated after `backward()`.
- `test_diff_sgd_step_second_order_via_grad_tensor` — `-lr`
  derivative w.r.t. inner-loop `grad`.
- `test_diff_sgd_momentum_first_step_uses_grad_as_velocity` —
  empty `prev_velocities` shortcut.
- `test_diff_sgd_momentum_second_step_uses_velocity` —
  `v = 0.9*v_prev + g`, `p = p - lr*v`.
- `test_diff_sgd_momentum_length_mismatch_errors` /
  `_velocity_length_mismatch_errors` /
  `_shape_mismatch_errors` — input validation.
- `test_diff_sgd_momentum_maintains_autograd_chain` — autograd
  through the velocity computation.

Plus integration tests in
`ferrotorch-optim/tests/conformance_optim_advanced.rs` (around
lines 1264-1340) — `diff_sgd_step_matches_reference`,
`diff_sgd_step_multi_param_matches_reference`,
`diff_sgd_momentum_step_matches_reference` — that compare to
fixtures captured from a live PyTorch
`@_use_grad_for_differentiable`-decorated SGD step.

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib differentiable:: 2>&1 | tail -3
```

Expected: `9 passed; 0 failed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn diff_sgd_step` at `ferrotorch-optim/src/differentiable.rs:55` using `mul`/`sub` from `ferrotorch_core::grad_fns::arithmetic`; non-test consumer: `pub use differentiable::diff_sgd_step` at `ferrotorch-optim/src/lib.rs:34` — boundary-method public API per goal.md S5. |
| REQ-2 | SHIPPED | impl: length check at `ferrotorch-optim/src/differentiable.rs:60` returning `FerrotorchError::InvalidArgument`; per-element shape check at line 75 returning `FerrotorchError::ShapeMismatch`; non-test consumer: same `pub use` re-export at `lib.rs:34`. |
| REQ-3 | SHIPPED | impl: `T::from(lr)` cast at `ferrotorch-optim/src/differentiable.rs:69` returning `FerrotorchError::InvalidArgument` on failure; non-test consumer: same `pub use` at `lib.rs:34`. |
| REQ-4 | SHIPPED | impl: `creation::scalar(lr_t)?.to(p.device())?` at `ferrotorch-optim/src/differentiable.rs:85` (inside the per-parameter loop); non-test consumer: same `pub use` at `lib.rs:34`. |
| REQ-5 | SHIPPED | impl: `pub fn diff_sgd_momentum_step` at `ferrotorch-optim/src/differentiable.rs:110` with the velocity recurrence implemented on lines 159-183; non-test consumer: `pub use differentiable::diff_sgd_momentum_step` at `ferrotorch-optim/src/lib.rs:34`. |
| REQ-6 | SHIPPED | impl: `prev_velocities.is_empty()` first-step branch at `ferrotorch-optim/src/differentiable.rs:161`, plus length-mismatch check at line 126 returning `InvalidArgument`; non-test consumer: same `pub use` at `lib.rs:34`. |
| REQ-7 | SHIPPED | impl: `pub type DiffSgdMomentumOutput<T> = (Vec<Tensor<T>>, Vec<Tensor<T>>)` at `ferrotorch-optim/src/differentiable.rs:96`; non-test consumer: appears in the return type of the immediately-following `diff_sgd_momentum_step` (line 116), which is re-exported via `lib.rs:34`. |
| REQ-8 | SHIPPED | impl: every arithmetic call routes through `ferrotorch_core::grad_fns::arithmetic::{add, mul, sub}` (imports at `ferrotorch-optim/src/differentiable.rs:38`); non-test consumer: pinned by `test_diff_sgd_step_autograd_edge_to_param` and `test_diff_sgd_momentum_maintains_autograd_chain` in the in-file tests, plus the conformance-fixture integration tests in `ferrotorch-optim/tests/conformance_optim_advanced.rs` (lines 1264, 1321) that match against PyTorch `differentiable=True` reference. |
