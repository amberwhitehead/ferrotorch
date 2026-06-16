# Fixed-point implicit differentiation (`fixed_point`)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (Revert "feat(gpu): route bf16 buffers through f32 elementwise dispatchers (#23) (#24)")
upstream-paths:
  - torch/_higher_order_ops/
-->

## Summary

`ferrotorch-core/src/autograd/fixed_point.rs` implements
implicit-function-theorem differentiation for fixed-point equations
`x* = f(x*, params)`. The forward pass iterates `f` until convergence
without recording the graph; the backward pass uses the Neumann series
to solve `(I - J_x^T) v = grad_output` and then distributes `v`
through `df/dp` to produce gradients for each parameter. This avoids
unrolling the iteration (which can be thousands of steps) and is the
foundational primitive for Deep Equilibrium Models (DEQ), long-context
RNNs, Neural ODEs, and Neural Cellular Automata. The Python ecosystem
analog is `torch._higher_order_ops/` and the `torchdeq` reference
package.

## Requirements

- REQ-1: `pub fn fixed_point<T, F>(f, x0, params, max_iter, tol) ->
  FerrotorchResult<Tensor<T>>` — find a fixed point of `f` starting
  from `x0`, then attach a `FixedPointBackward` that uses implicit
  differentiation. `params` are the tensors to differentiate w.r.t.
  Mirrors the pattern of DEQ / `torchdeq.solver.fixed_point`.
- REQ-2: Forward pass iterates `x_{n+1} = f(x_n, params)` inside
  `no_grad`, first rejecting `max_iter == 0` and non-finite or
  negative `tol`. Each iterate must stay on the same device and keep
  the same shape as the previous iterate. The loop stops when
  `||x_{n+1} - x_n||_1 <= tol`; if no iterate converges, it returns an
  error with the last residual instead of a best-effort tensor.
- REQ-3: Skip-attach when no parameter requires grad — return the
  raw fixed point without a `grad_fn`.
- REQ-4: `FixedPointBackward<T>` — internal `GradFn` impl that
  carries the closure `f`, the fixed point `x*`, the parameters,
  and the backward iteration cap + tolerance. The backward
  iteration cap is `min(max_iter, 50)` (the Neumann series
  typically converges much faster than the forward iteration).
- REQ-5: Backward step 1 — solve `(I - J_x^T) v = grad_output` via
  Neumann series:
  * `v_0 = grad_output`
  * `v_{k+1} = grad_output + J_x^T @ v_k`
  computed via VJP through `f(x*, p)` with grad tracking on `x`.
  Converges because `f` is contractive (`spectral_radius(J_x) <
  1`). If the capped Neumann solve does not meet tolerance, backward
  returns a non-convergence error with the last residual.
- REQ-6: Backward step 2 — for each parameter `p_i`, compute
  `grad_p_i = J_{p_i}^T @ v` via VJP through `f(x*, p)` with grad
  tracking on `p`.
- REQ-7: `elementwise_mul_sum<T>` helper — computes `sum(a * b)`
  using `crate::grad_fns::arithmetic::mul` and
  `crate::grad_fns::reduction::sum`, preserving the autograd graph.
  This is the "scalarize a vector via dot product" trick used in
  both the J_x^T VJP and the J_p^T VJP.

## Acceptance Criteria

- [x] AC-1: Affine fixed-point `f(x) = 0.5 * x + 1` converges to
  `x* = 2` — `test_fixed_point_affine` at
  `fixed_point.rs:361-383`.
- [x] AC-2: Contractive fixed-point `f(x, a) = a * x` with `a =
  0.5` from `x0 = 10` converges to `x* = 0` —
  `test_fixed_point_contractive_to_zero` at `:386-402`.
- [x] AC-3: Looser tolerance terminates earlier with a close
  approximation — `test_fixed_point_tolerance` at `:404-430`.

## Architecture

### REQ-1 entry point

`pub fn fixed_point<T, F>` at `fixed_point.rs:79-135`. Steps:

1. Iterate fixed point inside `no_grad` at `:91-109`. Compute L1
   residual with tensor ops and a scalar readback. Early-out when
   `residual <= tol`; reject invalid solver config, shape/device
   mismatches, non-finite residuals, and non-convergence.
2. Skip-attach at `:113-115` if no parameter requires grad.
3. Build storage for `x*` and attach `FixedPointBackward` via
   `Tensor::from_operation` at `:121-131`.

### REQ-2 forward iteration

The forward loop at `fixed_point.rs:92-108` is purely numerical (no
graph). The residual is computed by device-aware tensor operations
(`sub`, `abs`, `sum`) and only the scalar residual is read back for
the convergence decision. This keeps CUDA tensor payloads resident and
prevents shape truncation bugs from host-side `zip` iteration.

### REQ-3 skip-attach

`if params.iter().any(|p| p.requires_grad())` at `fixed_point.rs:113`.
When no parameter requires grad, the function returns `Ok(x_star)`
without attaching a backward node at `:132-134`.

### REQ-4 `FixedPointBackward<T>` struct

`struct FixedPointBackward<T: Float>` at `fixed_point.rs:147-158`:

- `f_closure: FixedPointFn<T>` — `Arc<dyn Fn(&Tensor<T>,
  &[&Tensor<T>]) -> FerrotorchResult<Tensor<T>> + Send + Sync>`
- `x_star: Tensor<T>` — the converged fixed point.
- `params: Vec<Tensor<T>>` — owned clones for backward use.
- `backward_max_iter: usize` — capped at `min(max_iter, 50)`.
- `backward_tol: f64` — convergence tolerance for Neumann.

`impl Debug` at `:160-170` elides the closure as `"<closure>"`.

### REQ-5 — Neumann series solve

`impl<T: Float> GradFn<T> for FixedPointBackward<T>` at
`fixed_point.rs:172-322`. Step 1 at `:177-245`:

For each Neumann iteration:
1. Construct fresh `x_fresh` with `requires_grad=true` at `requires_grad in fixed_point.rs`.
2. Construct detached params (gradients aren't needed in this
   sub-call) at `fixed_point.rs`.
3. Evaluate `y = f(x_fresh, params_detached)` at `:215`.
4. Compute `yv = elementwise_mul_sum(y, v)` at `:222` — scalarize.
5. Call `grad(yv, [x_fresh], false, false)` at `:224` → returns
   `J_x^T @ v` as the gradient of the scalar w.r.t. `x_fresh`.
6. Update `v_new = grad_output + J_x^T @ v` at `:231-239`.
7. Break when L1 norm of `v_new - v` < `backward_tol` at `:242-244`.

### REQ-6 — parameter gradient distribution

Step 2 at `fixed_point.rs:247-313`:

1. Detached `x_star` (no grad on the x leg this time) at `:255-260`.
2. Fresh `params_with_grad` reconstructed with each parameter's
   original `requires_grad` flag at `:263-275`.
3. Evaluate `y = f(x_detached, params_with_grad)` at `:278`.
4. Loss `loss = elementwise_mul_sum(y, v)` at `:282`.
5. Filter `grad_inputs` to only parameters that require grad at
   `:285-288`.
6. Call `grad(loss, &grad_inputs[..], false, false)` at `:297` → the
   gradient w.r.t. each requesting parameter is `J_p^T @ v`.
7. Map back to the full parameter list (None for non-grad
   parameters) at `:301-309`.

### REQ-7 — `elementwise_mul_sum` helper

`fn elementwise_mul_sum<T: Float>(a, b) -> FerrotorchResult<Tensor<T>>`
at `elementwise_mul_sum in fixed_point.rs`: `prod = mul(a, b)?; sum(&prod)`. Used
twice (REQ-5 step 4 and REQ-6 step 4) to scalarize a vector for `grad`
consumption.

## Parity contract

`parity_ops = []` — fixed-point AD is a graph-construction primitive.
Behavioral parity vs upstream:

- Forward iterates `x_{n+1} = f(x_n, params)` until convergence and
  reports non-convergence as an error rather than returning a tensor
  that is not known to be a fixed point.
- Backward uses Neumann series rather than direct linear solve,
  matching the DEQ literature's default and the
  `torch._higher_order_ops/` reference. Converges when `f` is
  contractive.
- Returns gradients only for parameters with `requires_grad=true`,
  matching upstream's `torch.autograd.grad`-shaped output.

R-DEV-7 (Rust ecosystem analog) applies — `FixedPointBackward` uses
Arc-wrapped closures and Rust-native `Vec`/`Result`, but the user-facing
API (`fixed_point(f, x0, params, max_iter, tol)`) matches the upstream
contract.

## Verification

Tests in `fixed_point.rs:337-551`. Key tests:

- `test_fixed_point_affine` (`test_fixed_point_affine in fixed_point.rs`)
- `test_fixed_point_contractive_to_zero` (`test_fixed_point_contractive_to_zero in fixed_point.rs`)
- `test_fixed_point_tolerance` (`test_fixed_point_tolerance in fixed_point.rs`)
- `test_fixed_point_max_iter_reached` validates non-convergence errors.
- Shape/device/config regression tests pin the compatibility checks.
- Backward / gradient verification tests further down in the test
  module.

All tests pass in the workspace gauntlet.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn fixed_point<T, F>` at `ferrotorch-core/src/autograd/fixed_point.rs:79-135`; the API mirrors DEQ / `torchdeq.solver.fixed_point` semantics from upstream's `torch/_higher_order_ops/` package; non-test production consumer: re-exported at `ferrotorch-core/src/autograd/mod.rs:27 pub use fixed_point::fixed_point` and `lib.rs:127 fixed_point`. Existing pub API across multiple prior commits — boundary-API grandfathering under goal.md S5. |
| REQ-2 | SHIPPED | impl: forward-iteration loop at `fixed_point.rs:91-109` with L1-norm convergence check at `:96-105`; non-test consumer: inside REQ-1 (the engine body). |
| REQ-3 | SHIPPED | impl: `if params.iter().any(|p| p.requires_grad())` at `fixed_point.rs:113` with the no-grad return at `:132-134`; non-test consumer: every `fixed_point` call where none of the params require grad — the inference-time fast path. |
| REQ-4 | SHIPPED | impl: `struct FixedPointBackward<T: Float>` at `fixed_point.rs:147-158` + `impl Debug` at `:160-170` + `impl GradFn` at `:172-322`; non-test consumer: instantiated inside REQ-1 at `:124-130` and dispatched from `Tensor::backward` whenever a `fixed_point`-produced output is differentiated. |
| REQ-5 | SHIPPED | impl: Neumann series solve at `fixed_point.rs:177-245`; non-test consumer: inside REQ-4's `backward` impl — invoked on every backward of a `fixed_point`-produced tensor. |
| REQ-6 | SHIPPED | impl: per-parameter gradient distribution at `fixed_point.rs:247-313`; non-test consumer: inside REQ-4's `backward` impl. |
| REQ-7 | SHIPPED | impl: `fn elementwise_mul_sum<T: Float>` at `elementwise_mul_sum in fixed_point.rs`; non-test consumer: called twice inside REQ-5 (at `fixed_point in fixed_point.rs`) and REQ-6 (at `fixed_point in fixed_point.rs`) — the scalarization step every Neumann iteration and every parameter gradient pass relies on. |
