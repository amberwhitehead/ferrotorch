# ferrotorch-optim — `foreach_utils` (foreach-step boilerplate helpers)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/optim/_functional.py
  - torch/optim/optimizer.py
-->

## Summary

`ferrotorch-optim/src/foreach_utils.rs` is a tiny grab-bag of helpers that
the `step_foreach` paths of the GPU-aware optimizers (`Adam`, `Adamax`,
`Adadelta`, `Asgd`, etc.) share. It exposes `elemwise_max` (a
broadcast-free `max(a, b)` expressed through `add`/`sub`/`abs`, used by
the AMSGrad branch of Adam and by the running-max `u` of Adamax), plus
two scalar-promotion helpers (`scalar_on`, `f64_scalar_on`) that wrap
the recurrent boilerplate of placing a hyperparameter scalar on the
parameter's device before broadcasting into a tensor op. Mirrors
upstream PyTorch's foreach scaffolding (`torch.utils._foreach_utils`)
in spirit but takes the Rust-typestate approach (typed scalar tensor on
the correct device, no `dtype` kwarg flowing through every call).

## Requirements

- REQ-1: `pub fn elemwise_max<T: Float>(a, b, device) -> Tensor<T>`
  computes the elementwise maximum without invoking the autograd
  `max` grad-fn (which `ferrotorch-core` does not currently expose).
  Algorithm is the classic `0.5 * (a + b + |a - b|)` identity, with
  every intermediate built through the existing
  `ferrotorch_core::grad_fns::arithmetic::{add, sub, abs, mul}`
  primitives so the autograd edge is preserved.
- REQ-2: `pub fn scalar_on<T: Float>(value, device) -> Tensor<T>`
  builds a 0-d scalar tensor on the requested device. The whole point
  is to centralise the `scalar(value)?.to(device)` two-step every
  foreach optimiser repeats for every hyperparameter on every step.
- REQ-3: `pub fn f64_scalar_on<T: Float>(value, device) -> Tensor<T>`
  is the `f64 -> T` cast convenience layer over `scalar_on`; the
  optimiser configs store hyperparameters as `f64` (matching PyTorch's
  Python `float` kwargs) but tensors are generic over the element
  type, so the cast happens once at the boundary.
- REQ-4: All three helpers preserve the autograd contract: scalar
  tensors have `requires_grad == false` (they are constants), and the
  grad edge runs through the inputs they multiply with, not through
  the helpers themselves.

## Acceptance Criteria

- [x] AC-1: `pub fn elemwise_max` exists and compiles for any backend
  exposing `add`, `sub`, `abs`, `mul`.
- [x] AC-2: `pub fn scalar_on` and `pub fn f64_scalar_on` exist and
  are `#[inline]`.
- [x] AC-3: `f64_scalar_on` returns `FerrotorchError::InvalidArgument`
  (via `cast::<f64, T>`) when the value is not representable in the
  target dtype.
- [x] AC-4: Both Adam (AMSGrad branch) and Adamax (running-max `u`)
  consume `elemwise_max`.
- [x] AC-5: `f64_scalar_on` is consumed by `adadelta.rs`, `adamax.rs`,
  `asgd.rs` to lift hyperparameters into device-resident scalar
  tensors.

## Architecture

### `elemwise_max` (REQ-1)

```text
elemwise_max(a, b, device) =
    let diff = sub(a, b)
    let abs_diff = abs(diff)
    let sum_ab = add(a, b)
    let sum_plus_abs = add(sum_ab, abs_diff)
    let half = scalar(0.5).to(device)
    mul(sum_plus_abs, half)
```

The identity `max(a, b) == 0.5 * (a + b + |a - b|)` is exact for
finite floats. NaN propagation: any NaN input poisons every
intermediate (matches PyTorch's `torch.maximum` NaN behaviour where
NaN propagates through the comparison). Inf: handled correctly by the
arithmetic (Inf - finite = Inf, |Inf| = Inf, finite + Inf = Inf,
0.5 * Inf = Inf).

### `scalar_on` / `f64_scalar_on` (REQ-2, REQ-3)

```text
scalar_on(value, device) = scalar(value).to(device)
f64_scalar_on(value, device) = scalar_on(cast::<f64, T>(value)?, device)
```

Both are `#[inline]` so the helper call is free at the call site. The
device argument is non-optional — every consumer already has the
parameter's device in hand and the whole rationale for these helpers
is to avoid the `scalar(...)?.to(device)?` two-step that has to repeat
for every hyperparameter.

### Why not `ferrotorch_core::grad_fns::reduction::max`?

PyTorch's `torch.maximum` exists at the autograd surface, but
`ferrotorch_core` (at this commit) does not expose `max` as a
grad-fn. Expressing elementwise max through `add`/`sub`/`abs` keeps
the autograd graph intact (every step contributes a `grad_fn`) and is
correct for all finite inputs.

### Non-test production consumers

- `ferrotorch-optim/src/adam.rs:19` `use crate::foreach_utils::elemwise_max;` — AMSGrad branch (around line 325).
- `ferrotorch-optim/src/adamax.rs:17` `use crate::foreach_utils::{elemwise_max, f64_scalar_on};` — running-max `u` and hyperparameter promotion.
- `ferrotorch-optim/src/adadelta.rs:22` `use crate::foreach_utils::f64_scalar_on;` — `rho`, `eps`, `lr`, `wd` scalars in the foreach step.
- `ferrotorch-optim/src/asgd.rs:19` `use crate::foreach_utils::f64_scalar_on;` — `eta`, `lambda * eta`, `wd` scalars in the foreach step.

## Parity contract

`parity_ops = []`. The helpers are a utility shim; numerical parity is
owned by their callers (the foreach branches of Adam, Adamax,
Adadelta, Asgd — each has its own dedicated `.design/ferrotorch-optim/<name>.md`
once those docs are authored). Edge cases the helpers themselves own:

- **NaN in `elemwise_max`**: propagates (matches `torch.maximum`).
- **Inf in `elemwise_max`**: correct (`max(+inf, x) == +inf`).
- **Mixed device**: undefined; callers ensure `a`, `b`, `device` are
  consistent. The `0.5` scalar is materialised on the requested
  device every call (no caching), so device-mismatch surfaces from
  the `to(device)` step rather than silently producing wrong
  numerics.
- **Non-representable cast**: `f64_scalar_on` returns
  `FerrotorchError::InvalidArgument` when `cast::<f64, T>(value)`
  fails (e.g. `f64::INFINITY` into `f16`).

## Verification

`foreach_utils.rs` itself has no `#[cfg(test)] mod tests` block — the
helpers are exercised end-to-end through their consumers:

- `ferrotorch-optim/src/adam.rs` `mod tests` — AMSGrad numerical
  parity tests.
- `ferrotorch-optim/src/adamax.rs` `mod tests` — running-max behaviour.
- `ferrotorch-optim/src/adadelta.rs` `mod tests` — hyperparameter
  flow through `f64_scalar_on`.
- `ferrotorch-optim/src/asgd.rs` `mod tests` — same.

Smoke command:

```bash
cargo test -p ferrotorch-optim --lib 2>&1 | tail -3
```

Expected: `327 passed; 0 failed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn elemwise_max` in `ferrotorch-optim/src/foreach_utils.rs:15` mirroring the `max(a,b)` identity (PyTorch's `torch.maximum` upstream `aten/src/ATen/native/BinaryOps.cpp` `maximum_kernel`); non-test consumer: `ferrotorch-optim/src/adam.rs:325` invokes it from the AMSGrad branch of `Adam::step_foreach`, `ferrotorch-optim/src/adamax.rs:221` invokes it for `u_new = max(beta2 * u, |g| + eps)`. |
| REQ-2 | SHIPPED | impl: `pub fn scalar_on` in `ferrotorch-optim/src/foreach_utils.rs:31`; non-test consumer: `pub fn f64_scalar_on` immediately below (line 37) calls it; transitively every `f64_scalar_on` consumer (`adadelta.rs:204-237`, `adamax.rs:199-227`, `asgd.rs:223-235`) consumes `scalar_on` underneath. |
| REQ-3 | SHIPPED | impl: `pub fn f64_scalar_on` in `ferrotorch-optim/src/foreach_utils.rs:37`; non-test consumer: `ferrotorch-optim/src/adadelta.rs:204` `let wd_t = f64_scalar_on::<T>(group_wd, device)?;` and four further uses on lines 209-237; `ferrotorch-optim/src/adamax.rs:199-227` (six uses); `ferrotorch-optim/src/asgd.rs:223-235` (three uses). |
| REQ-4 | SHIPPED | impl: scalar tensors built via `ferrotorch_core::creation::scalar` are leaf constants (`requires_grad=false`) per `ferrotorch-core/src/creation.rs`; non-test consumer: same call sites as REQ-1 / REQ-3 — Adam/Adamax/Adadelta/Asgd `step_foreach` paths rely on the scalars NOT introducing autograd edges so that the optimizer step itself stays inside `no_grad()`. |
