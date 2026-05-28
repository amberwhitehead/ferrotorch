# Numerical gradient check (`gradcheck`)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (Revert "feat(gpu): route bf16 buffers through f32 elementwise dispatchers (#23) (#24)")
upstream-paths:
  - aten/src/ATen/
  - c10/
  - torch/_torch_docs.py
  - torch/overrides.py
  - torch/autograd/gradcheck.py
-->

## Summary

`ferrotorch-core/src/autograd/gradcheck.rs` is the numerical-gradient
verification utility, mirroring `torch.autograd.gradcheck` at
`torch/autograd/gradcheck.py:22-95`. Given a function `f` returning a
scalar, the analytical gradients computed by autograd are compared
against central-finite-difference estimates element-by-element; a
mismatch beyond `atol + rtol * |numerical|` raises a descriptive
error. Used in tests of custom `GradFn` implementations.

## Requirements

- REQ-1: `pub fn gradcheck<T, F>(func, inputs, eps, atol, rtol) ->
  FerrotorchResult<bool>` — scalar-output gradient check. Returns
  `Ok(true)` if every analytical gradient matches the
  finite-difference numerical estimate within `atol + rtol *
  |numerical|`; returns `Err(InvalidArgument)` with a descriptive
  message on first mismatch. Mirrors `torch.autograd.gradcheck` at
  `torch/autograd/gradcheck.py:22-95`.
- REQ-2: Adaptive defaults based on `T`'s element size — for f32
  (size ≤ 4): `eps = 1e-3`, `atol = 1e-3`, `rtol = 1e-2`; for f64:
  `eps = 1e-6`, `atol = 1e-5`, `rtol = 1e-3`. Wider tolerances on
  f32 because float-precision cancellation error in the
  finite-difference numerator grows quickly with `eps` too small.
  Mirrors upstream's per-dtype default scaling.
- REQ-3: Scalar-output validation — `func` MUST return a tensor with
  `numel() == 1`; otherwise return `Err(InvalidArgument)` with the
  observed shape. Mirrors upstream's gradcheck assertion that the
  output be scalar.
- REQ-4: Central finite difference — `numerical = (f(x + eps) - f(x -
  eps)) / (2 * eps)`. More accurate than forward difference (cancels
  the leading O(eps^2) term). Mirrors `_compute_numerical_gradient`
  at `torch/autograd/gradcheck.py:358+`.
- REQ-5: Per-element gradient mismatch error — when `|analytical -
  numerical| > atol + rtol * |numerical|`, return error including
  the input index, element index, both values, the diff, and the
  tolerance threshold. Mirrors upstream's per-element error
  reporting.
- REQ-6: Multi-input support — the function takes a slice of inputs;
  numerical-grad perturbation is applied to one input at a time,
  with the others held fixed (using detached clones).

## Acceptance Criteria

- [x] AC-1: `gradcheck` of `sum(x * x)` w.r.t. `x` succeeds —
  `test_gradcheck_sum_of_squares` at `gradcheck.rs:196-211`.
- [x] AC-2: `gradcheck` of `sum(a * b)` w.r.t. `[a, b]` (linear
  combination) succeeds — `test_gradcheck_linear_combination` at
  `gradcheck.rs:213-228`.
- [x] AC-3: `gradcheck` of `sum(a + b)` succeeds —
  `test_gradcheck_add` at `gradcheck.rs:230-245`.
- [x] AC-4: Non-scalar function returns errors cleanly —
  `test_gradcheck_non_scalar_fails in gradcheck.rs`.

## Architecture

### REQ-1 / REQ-2 entry point + defaults

`pub fn gradcheck<T, F>(func, inputs, eps, atol, rtol) ->
FerrotorchResult<bool>` at `gradcheck.rs:43-184`. The adaptive
defaults branch at `:54-69` switches on `mem::size_of::<T>() <= 4` to
pick f32 vs f64 tolerances. Each of `eps`, `atol`, `rtol` accepts an
`Option<f64>`; `None` falls back to the per-dtype default.

### REQ-3 scalar-output validation

`if output.numel() != 1` at `gradcheck.rs:78-85` → returns
`Err(InvalidArgument)` with the observed `output.shape()`.
`output.backward()?` at `:86` runs the analytical backward.

### REQ-4 central finite difference

The inner loop at `gradcheck.rs:88-181` walks each input tensor and
each element index. For each `(input_idx, elem_idx)`:

1. Build `perturbed_plus` = clone of `input_data` with
   `perturbed_plus[elem_idx] += eps_t` at `:107-109`.
2. Build `perturbed_minus` symmetrically at `:111-113`.
3. Construct new `Tensor`s with `requires_grad = false` at `gradcheck.rs`.
4. Assemble per-input slices for the `+` and `-` evaluations,
   substituting the perturbed tensor and using detached clones for
   the other inputs at `:128-145`.
5. Evaluate `f_plus = func(&plus_inputs)?` and `f_minus =
   func(&minus_inputs)?` at `:147-148`.
6. Compute `numerical = (f_plus_val - f_minus_val) / (2 * eps)` at
   `:155`.

### REQ-5 mismatch reporting

Compute `diff = |analytical - numerical|` at `:159-163` and
`tolerance = atol + rtol * |numerical|` at `:170`. If `diff >
tolerance`, return `Err(InvalidArgument)` with a message embedding
the input index, element index, analytical/numerical values,
diff, and tolerance at `:172-179`.

### REQ-6 multi-input

The outer loop at `gradcheck.rs:89-181` iterates `for (input_idx,
input) in inputs.iter().enumerate()`. The per-element inner loop
substitutes only the current input's perturbed copy; the other
inputs are detached clones with `requires_grad = false` at
`:135-144`.

## Parity contract

`parity_ops = []` — `gradcheck` is a test utility, not a
tensor-valued op. Behavioral parity vs upstream:

- Defaults per-dtype mirror upstream's `eps=1e-6, atol=1e-5,
  rtol=1e-3` for f64 and the looser `1e-3 / 1e-3 / 1e-2` for f32.
- Central finite difference (not forward) matches upstream's
  `_compute_numerical_gradient` at
  `torch/autograd/gradcheck.py:358+`.
- Scalar-output check matches upstream.
- Per-element error reporting matches upstream's format closely
  (input index, element index, analytical, numerical, diff, tol).

Note: ferrotorch's `gradcheck` here is the minimal subset. PyTorch's
`gradcheck` additionally supports complex tensors, sparse tensors,
batched mode, fast-mode (`fast_mode=True`), and a follow-up
`gradgradcheck` for second-order verification — none are wired here.

## Verification

Tests in `gradcheck.rs:186-253` (4 tests):

- `test_gradcheck_sum_of_squares` (`test_gradcheck_sum_of_squares in gradcheck.rs`)
- `test_gradcheck_linear_combination` (`test_gradcheck_linear_combination in gradcheck.rs`)
- `test_gradcheck_add` (`test_gradcheck_add in gradcheck.rs`)
- `test_gradcheck_non_scalar_fails` (`test_gradcheck_non_scalar_fails in gradcheck.rs`)

All 4 pass in the workspace gauntlet.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn gradcheck<T, F>` at `gradcheck in ferrotorch-core/src/autograd/gradcheck.rs`; mirrors `torch.autograd.gradcheck` at `torch/autograd/gradcheck.py:22-95`; non-test production consumer: re-exported at `gradcheck in ferrotorch-core/src/autograd/mod.rs pub use gradcheck::gradcheck` and exposed through `crate::autograd::gradcheck`; the public test-utility API that downstream crates use to verify their custom `GradFn` impls. Existing pub API across multiple prior commits — boundary-API grandfathering under goal.md S5 (gradcheck is a test utility consumed primarily by test code, which is a permitted consumer pattern for diagnostic surfaces). |
| REQ-2 | SHIPPED | impl: adaptive default selection at `gradcheck.rs:54-69` switching on `mem::size_of::<T>() <= 4`; non-test consumer: invoked inside REQ-1's `gradcheck` body — every caller that passes `None` for any of `eps / atol / rtol` flows through here. |
| REQ-3 | SHIPPED | impl: scalar-output validation at `gradcheck in gradcheck.rs`; non-test consumer: inside REQ-1; tested by `test_gradcheck_non_scalar_fails in gradcheck.rs`. |
| REQ-4 | SHIPPED | impl: central finite difference at `gradcheck.rs:88-181` (the perturb-plus / perturb-minus / divide-by-`2*eps` body); mirrors `_compute_numerical_gradient` at `torch/autograd/gradcheck.py:358+`; non-test consumer: inside REQ-1. |
| REQ-5 | SHIPPED | impl: per-element mismatch error at `gradcheck.rs:159-180`; non-test consumer: inside REQ-1; tested implicitly by the four passing tests (they ensure the error path does NOT fire). |
| REQ-6 | SHIPPED | impl: multi-input outer-loop at `gradcheck.rs:89-181` with per-input substitution at `:128-145`; non-test consumer: tested by `test_gradcheck_linear_combination` (`gradcheck.rs:213-228`) and `test_gradcheck_add` (`:230-245`), both exercising 2-input cases. |
