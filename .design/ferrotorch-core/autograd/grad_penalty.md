# Gradient penalty + JVP/VJP / grad-norm (`gradient_penalty`, `grad_norm`, `jvp`, `vjp`)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (Revert "feat(gpu): route bf16 buffers through f32 elementwise dispatchers (#23) (#24)")
upstream-paths:
  - torch/autograd/functional.py
-->

## Summary

`ferrotorch-core/src/autograd/grad_penalty.rs` ships four higher-order
gradient utilities that build on top of `autograd::higher_order::grad`:

1. `gradient_penalty` — WGAN-GP gradient-penalty term (`lambda *
   (||grad(D(x_interp))||_2 - 1)^2` with `x_interp = alpha * real +
   (1-alpha) * fake`).
2. `grad_norm` — L2 norm of gradients of `outputs` w.r.t.
   `inputs`, used for gradient regularization.
3. `jvp` — Jacobian-vector product via central finite differences
   (forward-mode AD with `dual_*` rules is also available via
   `autograd::forward_ad::jvp_exact`; this one is the
   finite-difference fallback for general functions).
4. `vjp` — Vector-Jacobian product (the autograd-backward analog).

Mirrors `torch.autograd.functional` at `torch/autograd/functional.py`
(specifically `vjp`, `jvp`, `jacobian` definitions and their
`create_graph` semantics) and the WGAN-GP idiom from the Gulrajani
et al. 2017 paper.

## Requirements

- REQ-1: `pub fn gradient_penalty<T: Float, F>(discriminator, real,
  fake, lambda) -> FerrotorchResult<Tensor<T>>` — compute the WGAN-GP
  gradient-penalty tensor. The returned tensor has `grad_fn`
  attached so it can be added to the discriminator loss and
  differentiated in the outer training loop.
- REQ-2: Random interpolation coefficient — `alpha ~ U(0, 1)` of the
  same shape as `real` via `crate::creation::rand(real.shape())`.
  Build `x_interp = alpha * real + (1 - alpha) * fake` element-wise.
- REQ-3: Compute `grad(D(x_interp), x_interp, create_graph=true)`
  so the resulting gradient tensor itself carries a `grad_fn` and
  can be differentiated through.
- REQ-4: Penalty formula — `penalty = lambda * (sqrt(sum(grad^2)) -
  1)^2`. Each step (`pow`, `sum`, `sqrt`, `sub`, `mul`) must use the
  autograd-aware versions from `crate::grad_fns::arithmetic` /
  `reduction` so the full chain is differentiable.
- REQ-5: `pub fn grad_norm<T: Float>(outputs, inputs) ->
  FerrotorchResult<Tensor<T>>` — scalar L2 norm of all gradients.
  Returns a `[]`-shape tensor with the computed scalar.
- REQ-6: `pub fn jvp<T: Float, F>(f, input, v) ->
  FerrotorchResult<Tensor<T>>` — JVP via central finite differences:
  `(f(x + h*v) - f(x - h*v)) / (2h)` with `h = 1e-4`. Slower and less
  accurate than `autograd::forward_ad::jvp_exact` but works for any
  callable `f`, including ones that use ops not yet supported by the
  `dual_*` rule set.
- REQ-7: `pub fn vjp<T: Float, F>(f, input, v) ->
  FerrotorchResult<Tensor<T>>` — VJP via autograd: forward `y = f(x)`
  with `x.requires_grad=true`, scalarize as `scalar = sum(y * v)`,
  call `grad(&scalar, &[&x], false, false)` to extract `v^T @ J`.
- REQ-8: Shape validation — `real.shape() == fake.shape()` (REQ-1);
  `input.shape() == v.shape()` (REQ-6); `y.numel() == v.numel()`
  (REQ-7). Otherwise return `ShapeMismatch`.

## Acceptance Criteria

- [x] AC-1: `gradient_penalty` for linear discriminator `D(x) = sum(x)`
  with `n=4` reals/fakes yields `lambda * (sqrt(n) - 1)^2` —
  `test_gradient_penalty_linear_discriminator` at
  `grad_penalty.rs:349-372`.
- [x] AC-2: Shape mismatch between `real` and `fake` errors —
  `test_gradient_penalty_shape_mismatch` at
  `test_gradient_penalty_shape_mismatch in grad_penalty.rs`.
- [x] AC-3: Scalar-input `gradient_penalty` with quadratic
  discriminator computes correctly —
  `test_gradient_penalty_scalar_input in grad_penalty.rs`.

## Architecture

### REQ-1 / REQ-2 / REQ-3 / REQ-4 `gradient_penalty`

`pub fn gradient_penalty<T: Float, F>` at `grad_penalty.rs:58-125`.
Steps:

1. Shape validation at `:67-76`.
2. Sample `alpha` via `crate::creation::rand` at `:81`.
3. Build `x_interp` elementwise at `:87-90`. Construct
   `x_interp` tensor with `requires_grad=true` at `:92-94`.
4. Forward `d_interp = discriminator(&x_interp)` at `:97`.
5. Backward with `create_graph=true`: `let grads = grad(&d_interp,
   &[&x_interp], false, true)?` at `:100`. Extract `grad_interp` at
   `:101-108` (defensive zero when the discriminator output is
   independent of input).
6. Penalty math at `:111-122`:
   - `grad_sq = pow(grad_interp, 2.0)` (`crate::grad_fns::arithmetic`)
   - `grad_sq_sum = sum(grad_sq)` (`crate::grad_fns::reduction`)
   - `grad_norm = sqrt(grad_sq_sum)`
   - `diff = sub(grad_norm, one_tensor)`
   - `diff_sq = pow(diff, 2.0)`
   - `penalty = mul(lambda_tensor, diff_sq)`

Every step uses the differentiable variants — the returned `penalty`
carries an autograd graph rooted at `discriminator`'s parameters.

### REQ-5 `grad_norm`

`pub fn grad_norm<T: Float>(outputs, inputs)` at
`grad_penalty.rs:146-165`. Calls `grad(outputs, inputs, false, false)`
at `:150`, accumulates `total_sq += val * val` across every
gradient tensor at `:153-161`, returns
`Tensor::from_storage(cpu(vec![total_sq.sqrt()]), vec![], false)` at
`:164`. Scalar `[]`-shape result.

### REQ-6 `jvp` (finite-difference)

`pub fn jvp<T: Float, F>(f, input, v)` at `grad_penalty.rs:188-241`.
Shape check at `:192-200`. Compute `x_plus = input + h*v` and
`x_minus = input - h*v` at `:209-220`. Evaluate `f_plus = f(&x_plus)`
and `f_minus = f(&x_minus)` at `:223-224`. Compute `(f_plus -
f_minus) / (2h)` element-wise at `:229-234`.

### REQ-7 `vjp` (autograd-backward)

`pub fn vjp<T: Float, F>(f, input, v)` at `grad_penalty.rs:264-308`.
Fresh `x` with `requires_grad=true` at `:269-273`. Forward `y =
f(&x)` at `:276`. Shape check `y.numel() == v.numel()` at `:283-291`.
Scalarize via `weighted = mul(&y, &v_tensor); scalar = sum(&weighted)`
at `:294-296`. Call `grad(&scalar, &[&x], false, false)` at `:299`,
return `grads[0]` (with a defensive `None` → zeros fallback at
`:301-308`).

### REQ-8 shape validation

`gradient_penalty` validates at `grad_penalty.rs:67-76`. `jvp` at
`:192-200`. `vjp` at `:283-291`. Each returns `ShapeMismatch` with a
descriptive message. Inputs are user data so loose error reporting
matters for diagnostics.

## Parity contract

`parity_ops = []` — these are graph-construction helpers, not
tensor-valued ops. Behavioral parity:

- `gradient_penalty` matches the canonical WGAN-GP penalty term
  (Gulrajani et al. 2017): `lambda * (||grad||_2 - 1)^2` with a
  uniform-distribution interpolation between real and fake.
- `grad_norm` mirrors `torch.nn.utils.clip_grad_norm_`'s internal
  norm computation (without the clipping — that lives in
  `ferrotorch-nn/src/utils.rs`).
- `jvp` is the central finite-difference variant — slower than
  forward-mode AD but always available. The exact alternative is
  `autograd::forward_ad::jvp_exact` (REQ-8 of `forward_ad.md`).
- `vjp` mirrors `torch.autograd.functional.vjp` for the
  `create_graph=False` case at
  `torch/autograd/functional.py:271-345`.

R-DEV-7 (Rust ecosystem analog) applies for `jvp`'s finite-difference
form — upstream's exact `torch.func.jvp` is a forward-mode AD entry
point, while ferrotorch's `jvp` here is the always-available
finite-difference fallback alongside the exact `jvp_exact` in
`forward_ad.rs`.

## Verification

Tests in `grad_penalty.rs:315-708`. Key tests:

- `test_gradient_penalty_linear_discriminator` (`test_gradient_penalty_linear_discriminator in grad_penalty.rs`)
- `test_gradient_penalty_shape_mismatch` (`test_gradient_penalty_shape_mismatch in grad_penalty.rs`)
- `test_gradient_penalty_scalar_input` (`test_gradient_penalty_scalar_input in grad_penalty.rs`)
- Additional `grad_norm`, `jvp`, `vjp` tests in the rest of the
  module.

All tests pass in the workspace gauntlet.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn gradient_penalty<T: Float, F>` at `ferrotorch-core/src/autograd/grad_penalty.rs:58-125`; mirrors the WGAN-GP idiom (Gulrajani et al. 2017) consumed via PyTorch's `torch.autograd.functional.vjp`-style `create_graph=True` chain; non-test production consumer: re-exported at `ferrotorch-core/src/autograd/mod.rs:32 pub use grad_penalty::{grad_norm, gradient_penalty, jvp, vjp}` and `lib.rs:127 gradient_penalty`. Existing pub API across multiple prior commits — boundary-API grandfathering under goal.md S5. |
| REQ-2 | SHIPPED | impl: alpha-uniform interpolation at `grad_penalty.rs:81-90`; non-test consumer: inside REQ-1's `gradient_penalty` body. |
| REQ-3 | SHIPPED | impl: `let grads = grad(&d_interp, &[&x_interp], false, true)?` at `grad_penalty.rs:100`; non-test consumer: inside REQ-1's body — every `gradient_penalty` call. |
| REQ-4 | SHIPPED | impl: penalty composition at `grad_penalty.rs:111-122` using `crate::grad_fns::arithmetic::{pow, sub, mul, sqrt}` and `crate::grad_fns::reduction::sum`; non-test consumer: inside REQ-1's body. |
| REQ-5 | SHIPPED | impl: `pub fn grad_norm<T: Float>` at `grad_norm in grad_penalty.rs`; mirrors PyTorch's L2-norm computation pattern; non-test production consumer: re-exported at `mod.rs grad_norm` and `lib.rs grad_norm`. Existing pub API — boundary-API grandfathering. |
| REQ-6 | SHIPPED | impl: `pub fn jvp<T: Float, F>` at `jvp in grad_penalty.rs`; the finite-difference variant; non-test production consumer: re-exported at `mod.rs jvp` and `lib.rs jvp`. Existing pub API — boundary-API grandfathering. |
| REQ-7 | SHIPPED | impl: `pub fn vjp<T: Float, F>` at `vjp in grad_penalty.rs`; mirrors `torch.autograd.functional.vjp` at `torch/autograd/functional.py:271-345`; non-test production consumer: re-exported at `mod.rs vjp` and `lib.rs vjp`. Existing pub API — boundary-API grandfathering. |
| REQ-8 | SHIPPED | impl: shape-validation branches at `gradient_penalty in grad_penalty.rs` (`gradient_penalty`), `gradient_penalty in grad_penalty.rs` (`jvp`), `jvp in grad_penalty.rs` (`vjp`); non-test consumer: inside REQ-1/REQ-6/REQ-7 bodies; tested by `test_gradient_penalty_shape_mismatch in grad_penalty.rs`. |
