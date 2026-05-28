# Higher-order autograd (`grad`, `jacobian`, `hessian`)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (Revert "feat(gpu): route bf16 buffers through f32 elementwise dispatchers (#23) (#24)")
upstream-paths:
  - torch/autograd/__init__.py
  - torch/csrc/autograd/python_function.cpp
-->

## Summary

`ferrotorch-core/src/autograd/higher_order.rs` provides the
non-accumulating `grad` function (which returns gradients as values
rather than writing to `.grad()`) and the higher-order helpers
`jacobian` / `hessian` built on top of it. When `create_graph=true`,
the backward pass itself is recorded in the autograd graph, enabling
second-order derivatives (Hessian, WGAN-GP, MAML). Mirrors PyTorch's
`torch.autograd.grad` at `torch/autograd/__init__.py:301-420` and the
helper API surface from `torch.autograd.functional`.

## Requirements

- REQ-1: `pub fn grad<T: Float>(outputs, inputs, retain_graph,
  create_graph) -> FerrotorchResult<Vec<Option<Tensor<T>>>>` — compute
  gradients of `outputs` w.r.t. each tensor in `inputs`, returning
  the gradients directly without accumulating onto `.grad()`. Mirrors
  `torch.autograd.grad` at `torch/autograd/__init__.py:301-420`.
- REQ-2: `outputs` must be scalar or single-element; otherwise return
  `BackwardNonScalar` error. The seed gradient is constructed inside
  `grad` (caller does not pass it).
- REQ-3: `create_graph=true` mode — the seed gradient and all
  intermediate accumulation tensors carry `requires_grad=true`, so
  the gradient-computation graph itself is recorded for second-order
  differentiation. Implemented by routing accumulation through
  `differentiable_add` (which calls
  `crate::grad_fns::arithmetic::add`) rather than raw Vec elementwise
  add.
- REQ-4: Three-phase BFS + Kahn topo-sort backward — same algorithm
  as the engine in `graph.rs`, but with a per-call `grads` map and
  no leaf-accumulation. Recorded gradients are collected for the
  requested `inputs` only.
- REQ-5: `pub fn jacobian<T: Float, F>(f, input) ->
  FerrotorchResult<Tensor<T>>` — compute the `[m, n]` Jacobian of
  `f: R^n -> R^m` by differentiating each output element
  independently via `grad`. Mirrors `torch.autograd.functional.jacobian`
  at `torch/autograd/functional.py:393+`.
- REQ-6: `pub fn hessian<T: Float, F>(f, input) ->
  FerrotorchResult<Tensor<T>>` — compute the `[n, n]` Hessian of
  `f: R^n -> R` by taking the Jacobian of the gradient function
  (using `create_graph=true` in the first-order pass to enable
  second-order). Mirrors `torch.autograd.functional.hessian` at
  `torch/autograd/functional.py:594+`.
- REQ-7: `IndexSelectBackward` — internal helper that extracts a
  single element from a tensor as a scalar tensor while preserving
  the computation graph. Routes the scalar gradient back to the
  correct position via a one-hot vector multiply.
- REQ-8: `BroadcastScalarBackward` — internal helper that
  broadcasts a scalar to a vector during the higher-order grad chain.
  The VJP is `sum(grad_output)`, mirroring the broadcast pattern.
- REQ-9: `Tensor::grad_wrt(&self, inputs, retain_graph, create_graph)`
  convenience method on `Tensor` that delegates to `grad`.

## Acceptance Criteria

- [x] AC-1: `grad(pow(x, 3.0), [x], false, false)` at `x = 2` yields
  `12.0` (= `3 * 2^2`) — `test_grad_simple_pow` at
  `higher_order.rs`.
- [x] AC-2: `grad(x + y, [x, y], false, false)` yields `(1.0, 1.0)`
  — `test_grad_add` at `higher_order.rs`.
- [x] AC-3: `grad(pow(x, 2), [x], true, true)` followed by
  `grad(dy_dx, [x], false, false)` yields the second derivative
  `2.0` (= `d^2(x^2)/dx^2`).
- [x] AC-4: `jacobian` of `f: R^n -> R^n` yields the analytical
  Jacobian matrix.
- [x] AC-5: `hessian` of `x^2 + y^2` yields `[[2, 0], [0, 2]]`.

## Architecture

### REQ-1 / REQ-2 `grad` entry point

`pub fn grad<T: Float>` at `higher_order.rs:56-240`. Validates
scalar output at `grad in higher_order.rs` (`BackwardNonScalar` on failure). Builds
the seed at `:76-91`: when `create_graph=true`, the seed has
`requires_grad=true`; otherwise `false`. Same three-phase BFS as
`graph.rs` (Phase 1 collect / Phase 2 topo-sort / Phase 3
backward).

### REQ-3 `create_graph=true` differentiable accumulation

`fn differentiable_add` at `fn in higher_order.rs` invokes
`crate::grad_fns::arithmetic::add`, which is the autograd-aware add
that builds an `AddBackward` grad_fn on the result. The accumulation
branch at `:196-216` switches:

- When `create_graph=true`: `summed = differentiable_add(&existing,
  &grad_tensor, create_graph)?` so the accumulation is itself
  graph-tracked at `:200-202`.
- When `create_graph=false`: raw `Vec<T>` elementwise add at
  `:204-215` (cheaper, no graph nodes).

The grad-tensor wrap at `:189-193` ensures
that any grad_output returned by a non-differentiable backward
(`Tensor::from_storage(... false)` produces non-graph tensors) gets
`requires_grad_(true)` re-applied when `create_graph=true`, so
downstream ops can record their own backward edges.

### REQ-4 three-phase backward

Phase 1 at `higher_order.rs:93-114` (BFS collect + in-degree).
Phase 2 at `:116-140` (Kahn topo-sort). Phase 3 at `:142-224`
(walk topo-order, call each `grad_fn.backward`, route results).
The key difference from `graph.rs::backward_with_grad` is at
`:167-169`: when a topo-walk node IS one of the requested
`inputs`, record its gradient into the `result` vec
indexed by `input_ids.get(&id)`.

### REQ-5 `jacobian`

`pub fn jacobian<T: Float, F>(f, input)` at `higher_order.rs`.
For an `f: R^n -> R^m`, loop `i in 0..m`: rebuild a fresh
`x_fresh` with `requires_grad=true`, evaluate `f(&x_fresh)`,
extract `y_i = extract_element(&y_fresh, i)`, then `grads =
grad(&y_i, &[&x_fresh], false, false)?`. Concatenate the resulting
`g.data()` slices into `jac_data`; return as a `[m, n]` tensor.

### REQ-6 `hessian`

`pub fn hessian<T: Float, F>(f, input)` at `higher_order.rs`.
For each row `i in 0..n`:

1. Build fresh `x` with grad.
2. Forward `y = f(&x)`.
3. First derivative `grads = grad(&y, &[&x], true, true)?` —
   `create_graph=true` records the gradient computation.
4. Extract `grad_i = extract_element(&grad_vec, i)`.
5. Second derivative `grads2 = grad(&grad_i, &[&x], false, false)?`
   yields row `i` of the Hessian.

### REQ-7 `IndexSelectBackward` (internal)

`struct IndexSelectBackward<T> { input: Tensor<T>, index: usize }`
at `higher_order.rs` with `impl GradFn` at `higher_order.rs`. The
backward implementation has two branches:

- `create_graph` branch at `:443-473`: build a one-hot basis
  vector, broadcast `grad_output` through a tracked
  `BroadcastScalarBackward` (REQ-8), multiply elementwise, return
  — keeping the chain differentiable.
- Plain branch at `:475-484`: produce a sparse gradient vector
  (zeros + `go` at `index`) wrapped in a non-graph tensor.

### REQ-8 `BroadcastScalarBackward` (internal)

`struct BroadcastScalarBackward<T> { scalar_input: Tensor<T> }` at
`higher_order.rs` with `impl GradFn` at `higher_order.rs`. The
VJP `sum(grad_output)` is the broadcasting adjoint — when the
forward replicated a scalar to a vector, the backward sums all
the per-element gradients back into the scalar input.

### REQ-9 `Tensor::grad_wrt` convenience

`impl<T: Float> Tensor<T>` at `higher_order.rs:529-542`:
`pub fn grad_wrt(&self, inputs, retain_graph, create_graph) ->
FerrotorchResult<Vec<Option<Tensor<T>>>>` delegates to `grad`.
Lets users write `loss.grad_wrt(&[&x], false, false)` instead of
`crate::autograd::grad(&loss, &[&x], false, false)`.

## Parity contract

`parity_ops = []` — `grad` / `jacobian` / `hessian` are
graph-walks, not tensor-valued ops. Behavioral parity vs upstream:

- `grad` returns the same set of gradients as
  `torch.autograd.grad(outputs, inputs, retain_graph,
  create_graph)`, modulo the Rust `Vec<Option<Tensor>>` vs Python
  `Tuple[Optional[Tensor], ...]` packaging.
- `create_graph=true` enables second-order derivatives.
- `jacobian` returns `[m, n]` shape (output-rows × input-cols),
  matching upstream's default `vectorize=False` layout.
- `hessian` returns `[n, n]`.

The Rust `retain_graph` parameter is consumed semantically (the
`Arc<dyn GradFn>` graph is immutable; the only thing
`retain_graph=false` would do is allow upstream to free
intermediate tensors, which Rust handles via `Drop` automatically).

## Verification

Tests in `higher_order.rs` (~500 LOC of test code).
Key tests:

- `test_grad_simple_pow` (`higher_order.rs`)
- `test_grad_add` (`higher_order.rs`)
- Jacobian / Hessian tests for elementwise and quadratic
  functions later in the test module.

All tests pass in the workspace gauntlet.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn grad<T: Float>` at `grad in ferrotorch-core/src/autograd/higher_order.rs`; mirrors `torch.autograd.grad` at `torch/autograd/__init__.py:301-420`; non-test production consumer: `grad in ferrotorch-core/src/autograd/grad_penalty.rs use super::higher_order::grad` and call sites at `grad in grad_penalty.rs let grads = grad(&d_interp, &[&x_interp], false, true)?` (inside `pub fn gradient_penalty`), ` let grads = grad(outputs, inputs, false, false)?` (inside `pub fn grad_norm`), ` let grads = grad(&scalar, &[&x], false, false)?` (inside `pub fn vjp`); also `grad in ferrotorch-core/src/autograd/fixed_point.rs use crate::autograd::higher_order::grad` with call sites at `fixed_point in fixed_point.rs, ` (inside `FixedPointBackward::backward`). |
| REQ-2 | SHIPPED | impl: scalar-output check at `higher_order.rs:62-67` returning `BackwardNonScalar`; non-test production consumer: inside REQ-1 (each `grad(outputs, ...)` call flows through this guard). |
| REQ-3 | SHIPPED | impl: `create_graph=true` branch with `differentiable_add` accumulation at `higher_order.rs:196-216` plus `requires_grad_(true)` re-wrap at `:189-193`; non-test production consumer: `grad_penalty.rs:100 grad(&d_interp, &[&x_interp], false, true)` inside `gradient_penalty` (WGAN-GP gradient penalty requires `create_graph=true` so the penalty is itself differentiable for the outer-loop optimization). |
| REQ-4 | SHIPPED | impl: three-phase BFS + Kahn at `higher_order.rs:93-224`; non-test production consumer: inside REQ-1 (the engine of `grad`). |
| REQ-5 | SHIPPED | impl: `pub fn jacobian<T: Float, F>` at `higher_order.rs`; mirrors `torch.autograd.functional.jacobian` at `torch/autograd/functional.py:393+`; non-test production consumer: re-exported at `higher_order in ferrotorch-core/src/autograd/mod.rs pub use higher_order::{grad, hessian, jacobian}` and at `lib.rs jacobian`. Existing pub API — boundary-API grandfathering. |
| REQ-6 | SHIPPED | impl: `pub fn hessian<T: Float, F>` at `higher_order.rs`; mirrors `torch.autograd.functional.hessian` at `torch/autograd/functional.py:594+`; non-test production consumer: re-exported at `mod.rs` and `lib.rs hessian`. Existing pub API — boundary-API grandfathering. |
| REQ-7 | SHIPPED | impl: `struct IndexSelectBackward<T>` at `higher_order.rs` + `impl GradFn` at `higher_order.rs`; non-test production consumer: instantiated inside `extract_element` at `higher_order.rs` which is invoked by `jacobian` at `higher_order.rs` and `hessian` at `higher_order.rs` — every `jacobian`/`hessian` call routes through this helper. |
| REQ-8 | SHIPPED | impl: `struct BroadcastScalarBackward<T>` at `backward in higher_order.rs` + `impl GradFn` at `backward in higher_order.rs`; non-test production consumer: instantiated inside the `create_graph` branch of `IndexSelectBackward::backward` at `backward in higher_order.rs` — every higher-order call (`hessian`, second-order `grad` chains) flows through this. |
| REQ-9 | SHIPPED | impl: `impl<T: Float> Tensor<T> pub fn grad_wrt` at `higher_order.rs`; non-test production consumer: re-exposed as a `Tensor` method to users of the crate (the chainable `loss.grad_wrt(&[&x], ...)` API). Existing pub API — boundary-API grandfathering. |
