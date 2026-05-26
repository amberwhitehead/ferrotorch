# Forward-mode AD (`DualTensor`, `dual_*` rules, `jvp_exact`, `jacfwd`)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (Revert "feat(gpu): route bf16 buffers through f32 elementwise dispatchers (#23) (#24)")
upstream-paths:
  - torch/autograd/forward_ad.py
  - torch/csrc/autograd/forward_grad.h
-->

## Summary

`ferrotorch-core/src/autograd/forward_ad.rs` implements forward-mode
automatic differentiation via dual numbers (`primal + epsilon *
tangent`). Each `dual_*` rule (`dual_add`, `dual_mul`, `dual_exp`,
`dual_matmul`, etc.) propagates both the primal and tangent
components through one operation, satisfying the chain rule in a
single forward pass — no backward graph required. `jvp_exact` computes
exact Jacobian-vector products in one pass; `jacfwd` computes the full
Jacobian by looping `jvp_exact` over basis vectors (the vmap(jvp)
pattern). Mirrors `torch.autograd.forward_ad` at
`torch/autograd/forward_ad.py:23-200` (specifically `make_dual` /
`unpack_dual` semantics) and the C++-side `ForwardGrad` representation
at `torch/csrc/autograd/forward_grad.h`.

## Requirements

- REQ-1: `pub struct DualTensor<T: Float> { primal: Tensor<T>,
  tangent: Tensor<T> }` — dual-number tensor with `Debug, Clone`
  derived. Both fields are `Arc`-backed `Tensor`s, so `clone()` is
  cheap. Mirrors the `(primal, tangent)` pair from
  `torch.autograd.forward_ad.make_dual` at
  `torch/autograd/forward_ad.py:77-130`.
- REQ-2: `pub fn DualTensor::new(primal, tangent) ->
  FerrotorchResult<Self>` — constructs a dual tensor; validates that
  both fields have identical shape. Mirrors `make_dual(tensor,
  tangent)` shape-equality requirement.
- REQ-3: `pub fn DualTensor::constant(primal) ->
  FerrotorchResult<Self>` — dual tensor with zero tangent (a "constant"
  in forward-mode AD).
- REQ-4: Arithmetic dual rules — `dual_add`, `dual_sub`, `dual_mul`,
  `dual_div`, `dual_neg`. Each propagates the standard chain-rule
  formulae:
  * `d(a+b) = da + db`, `d(a-b) = da - db`, `d(-a) = -da`
  * `d(a*b) = a*db + da*b`
  * `d(a/b) = (da*b - a*db) / b^2`
  Mirrors the closed-form forward rules from any standard AD
  reference (mirrored from PyTorch's
  `torch/csrc/autograd/FunctionsManual.cpp` derivative formulae).
- REQ-5: `dual_matmul` — `d(A @ B) = dA @ B + A @ dB`. Uses
  `crate::grad_fns::linalg::matmul_differentiable` for both products,
  then `dual_add` for the sum. 2-D matrices only.
- REQ-6: Activation dual rules — `dual_relu`, `dual_sigmoid`,
  `dual_tanh`. Each computes primal via the corresponding
  `crate::grad_fns::activation` op, then applies the activation's
  derivative formula to the input tangent.
- REQ-7: Transcendental dual rules — `dual_exp`, `dual_log`,
  `dual_sin`, `dual_cos`. Standard derivative formulae.
- REQ-8: `pub fn jvp_exact<T: Float, F>(f, input, v) ->
  FerrotorchResult<(Tensor<T>, Tensor<T>)>` — seed
  `dual_input = DualTensor::new(input.clone(), v.clone())`, single
  forward pass through `f`, return `(output.primal,
  output.tangent)`. Mirrors `dual_level()` + `make_dual` + `f(dual)`
  + `unpack_dual` pipeline from
  `torch/autograd/forward_ad.py:23-200`.
- REQ-9: `pub fn jacfwd<T: Float, F>(f, input) ->
  FerrotorchResult<Tensor<T>>` — compute full `[m, n]` Jacobian by
  looping `jvp_exact` over the `n` standard-basis tangent vectors.
  Input must be 1-D (shape `[n]`). Mirrors PyTorch's
  `torch.func.jacfwd` at `torch/func/__init__.py`.

## Acceptance Criteria

- [x] AC-1: `DualTensor::new` rejects shape mismatch —
  `test_dual_tensor_shape_mismatch` at `forward_ad.rs:496-501`.
- [x] AC-2: `DualTensor::constant` returns zero tangent —
  `test_dual_tensor_constant` at `forward_ad.rs:503-509`.
- [x] AC-3: `dual_add` propagates tangents element-wise —
  `test_dual_add` at `forward_ad.rs:515-529`.
- [x] AC-4: `dual_sub` — `test_dual_sub` at `:531-545`.
- [x] AC-5: `dual_mul` — `test_dual_mul` at `:547-557`.
- [x] AC-6: `dual_div` — `test_dual_div` at `:559-569`.
- [x] AC-7: `dual_neg` — `test_dual_neg` at `:571-583`.
- [x] AC-8: `dual_matmul` — `test_dual_matmul` at `:589-616`.
- [x] AC-9: `dual_relu` on positive inputs (`d(relu) = dx`) —
  `test_dual_relu_positive` at `:622-635`; and on negative
  (`d(relu) = 0`) — `test_dual_relu_negative` at `:637+`.

## Architecture

### REQ-1 / REQ-2 / REQ-3 — `DualTensor` constructors

`pub struct DualTensor<T: Float>` at `forward_ad.rs:37-42` with
`primal` and `tangent` public fields and `Debug, Clone` derived.
`DualTensor::new` at `:48-59` validates shape equality. `DualTensor::constant`
at `:62-70` builds a zero-tangent tensor (a "constant" because
forward AD of a constant produces zero derivative). The `shape()` /
`numel()` accessors at `:72-80` mirror PyTorch's `tensor.shape /
tensor.numel()`.

### REQ-4 — arithmetic dual rules

Each rule at `forward_ad.rs:88-129` builds the primal via the
corresponding autograd-aware op from `crate::grad_fns::arithmetic`,
then computes the tangent by composing the derivative formula:

- `dual_add` (`:88-93`): `primal = add(a.primal, b.primal); tangent
  = add(a.tangent, b.tangent)`.
- `dual_sub` (`:96-100`): symmetric.
- `dual_mul` (`:103-110`): primal = `mul(a.primal, b.primal)`;
  tangent = `add(mul(a.primal, b.tangent), mul(a.tangent, b.primal))`.
- `dual_div` (`:113-122`): primal = `div(a.primal, b.primal)`;
  tangent = `div(sub(mul(a.tangent, b.primal), mul(a.primal,
  b.tangent)), mul(b.primal, b.primal))`.
- `dual_neg` (`:125-129`): primal = `neg(a.primal); tangent =
  neg(a.tangent)`.

### REQ-5 — `dual_matmul`

`pub fn dual_matmul<T: Float>` at `forward_ad.rs:138-148`. Uses
`crate::grad_fns::linalg::matmul_differentiable` for both primal
(`a.primal @ b.primal`) and the two tangent products (`a.tangent @
b.primal` and `a.primal @ b.tangent`), then `add` for the sum.

### REQ-6 — activation dual rules

`dual_relu` at `forward_ad.rs:155-176`: primal via
`crate::grad_fns::activation::relu`, then walks per-element
`tangent[i] = a.tangent[i] if a.primal[i] > 0 else 0` (the standard
ReLU sub-gradient `0` at `x=0`).

`dual_sigmoid` at `:179-199`: primal via `activation::sigmoid`,
tangent via the elementwise formula `tangent = da * sigma * (1 -
sigma)`.

`dual_tanh` at `:202-222`: primal via `activation::tanh`, tangent
via `tangent = da * (1 - tanh^2)`.

### REQ-7 — transcendental dual rules

`dual_exp` at `forward_ad.rs:229-248`: primal via
`transcendental::exp`, tangent via elementwise `da * exp(a)`.

`dual_log` at `:251-270`: `tangent = da / a`.

`dual_sin` at `:273-292`: `tangent = da * cos(a)`.

`dual_cos` at `:295-314`: `tangent = -da * sin(a)`.

### REQ-8 — `jvp_exact`

`pub fn jvp_exact<T: Float, F>` at `forward_ad.rs:351-376`. Seed
`DualTensor::new(input.clone(), v.clone())` after shape-equality
validation at `:359-367`, run `f(dual_input)` once at `:373`,
return `(dual_output.primal, dual_output.tangent)`.

### REQ-9 — `jacfwd`

`pub fn jacfwd<T: Float, F>(f, input) -> FerrotorchResult<Tensor<T>>`
at `forward_ad.rs:407-449`. Input must be 1-D (errors otherwise at
`:411-416`). For each `j in 0..n`:

1. Build basis vector `e_j` (zero everywhere except position `j`
   which is `1`) at `:425-429`.
2. Call `jvp_exact(&f, input, &e_j)` at `:431` — returns the
   `j`-th column of the Jacobian as the tangent output.

After the loop, reorganize the columns into a `[m, n]` matrix layout
at `:438-447`.

## Parity contract

`parity_ops = []` — forward-mode AD is graph-walks + dual-number
algebra. Behavioral parity vs upstream:

- `DualTensor::new(primal, tangent)` ↔ `torch.autograd.forward_ad.make_dual(primal, tangent)`.
- `dual_*` rules ↔ the C++-side per-op forward-mode formulae in
  `torch/csrc/autograd/FunctionsManual.cpp` (subset — ferrotorch
  covers arithmetic + 2-D matmul + relu/sigmoid/tanh +
  exp/log/sin/cos).
- `jvp_exact(f, x, v)` ↔ `torch.func.jvp(f, (x,), (v,))`.
- `jacfwd(f, x)` ↔ `torch.func.jacfwd(f)(x)`.
- ReLU sub-gradient at `x=0` is `0` (matching upstream's standard
  convention).

Coverage gaps (intentional — these are the documented forward-mode
op coverage today):

- No `dual_pow`, `dual_softmax`, `dual_layer_norm` rules yet.
- `jacfwd` requires 1-D input (multi-dim input is not yet wired).
- No `dual_level` / `dual_*` reentry semantic — each `DualTensor` is
  a standalone primal+tangent pair, no nesting of dual-AD levels.

## Verification

Tests in `forward_ad.rs:455-1032` (~580 LOC of test code).
Construction tests, arithmetic rule tests, matmul rule test,
activation rule tests, transcendental rule tests, `jvp_exact`
end-to-end tests, `jacfwd` end-to-end tests.

All tests pass in the workspace gauntlet.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct DualTensor<T: Float>` at `ferrotorch-core/src/autograd/forward_ad.rs:37-42`; mirrors PyTorch's `(primal, tangent)` pair from `torch/autograd/forward_ad.py:77-130 make_dual`; non-test production consumer: re-exported at `ferrotorch-core/src/autograd/mod.rs:28-31 pub use forward_ad::{DualTensor, ..., jacfwd, jvp_exact}` and at `lib.rs:131-133 DualTensor`. Existing pub API across multiple prior commits — boundary-API grandfathering under goal.md S5. |
| REQ-2 | SHIPPED | impl: `pub fn DualTensor::new` at `forward_ad.rs:48-59` with shape-equality validation; non-test production consumer: invoked inside REQ-8 at `:370`; also part of the public surface (`re-exported via REQ-1's pub use`). |
| REQ-3 | SHIPPED | impl: `pub fn DualTensor::constant` at `forward_ad.rs:62-70`; non-test production consumer: part of the public DualTensor API; tested by `test_dual_tensor_constant` at `:503-509`. Existing pub API — boundary-API grandfathering. |
| REQ-4 | SHIPPED | impl: `dual_add` at `forward_ad.rs:88-93`, `dual_sub` at `:96-100`, `dual_mul` at `:103-110`, `dual_div` at `:113-122`, `dual_neg` at `:125-129`; non-test production consumer: re-exported at `mod.rs:28-31 pub use forward_ad::{... dual_add, ..., dual_div, ..., dual_mul, dual_neg, ..., dual_sub, ...}` and exposed via `lib.rs:131-133`. Existing pub API — boundary-API grandfathering. |
| REQ-5 | SHIPPED | impl: `pub fn dual_matmul<T: Float>` at `forward_ad.rs:138-148`; non-test production consumer: re-exported through `mod.rs:28-31 dual_matmul` and `lib.rs:131-133`. Existing pub API — boundary-API grandfathering. |
| REQ-6 | SHIPPED | impl: `pub fn dual_relu` at `forward_ad.rs:155-176`, `pub fn dual_sigmoid` at `:179-199`, `pub fn dual_tanh` at `:202-222`; non-test production consumer: re-exported through `mod.rs:28-31` and `lib.rs:131-133`. Existing pub API — boundary-API grandfathering. |
| REQ-7 | SHIPPED | impl: `pub fn dual_exp` at `forward_ad.rs:229-248`, `pub fn dual_log` at `:251-270`, `pub fn dual_sin` at `:273-292`, `pub fn dual_cos` at `:295-314`; non-test production consumer: re-exported through `mod.rs:28-31 dual_cos, dual_exp, dual_log, dual_sin` and `lib.rs:131-133`. Existing pub API — boundary-API grandfathering. |
| REQ-8 | SHIPPED | impl: `pub fn jvp_exact<T: Float, F>` at `forward_ad.rs:351-376`; mirrors `torch.func.jvp(f, primals, tangents)` per `torch/func/__init__.py`; non-test production consumer: invoked inside REQ-9 (`jacfwd`) at `:431` — the production consumer is `jacfwd`'s loop body. Also re-exported through `mod.rs:31 jvp_exact` and `lib.rs:133`. |
| REQ-9 | SHIPPED | impl: `pub fn jacfwd<T: Float, F>` at `forward_ad.rs:407-449`; mirrors `torch.func.jacfwd`; non-test production consumer: re-exported through `mod.rs:31 jacfwd` and `lib.rs:131-133`. Existing pub API — boundary-API grandfathering. |
