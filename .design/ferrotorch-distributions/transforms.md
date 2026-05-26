# ferrotorch-distributions — `transforms` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/transforms.py
  - torch/distributions/transformed_distribution.py
-->

## Summary

`ferrotorch-distributions/src/transforms.rs` defines the `Transform`
trait for invertible bijective maps with computable log-det-Jacobian,
five concrete transforms (`ExpTransform`, `AffineTransform`,
`SigmoidTransform`, `TanhTransform`, `SoftplusTransform`), the
`ComposeTransform` chain wrapper, and the `TransformedDistribution`
adapter that applies a transform chain to a base distribution.
Mirrors `torch/distributions/transforms.py` (16 transforms in
upstream `__all__`) and
`torch/distributions/transformed_distribution.py`.

## Requirements

- REQ-1: `pub trait Transform<T: Float>: Send + Sync` with three
  required methods — `forward(&self, x) -> FerrotorchResult<Tensor<T>>`,
  `inverse(&self, y) -> FerrotorchResult<Tensor<T>>`,
  `log_abs_det_jacobian(&self, x, y) -> FerrotorchResult<Tensor<T>>`
  — plus `name(&self) -> &'static str` for diagnostics and two
  default-implemented methods (`constant_entropy_contribution() ->
  Option<T>` and `is_exp_transform() -> bool`) that the entropy
  dispatcher inspects. Mirrors
  `torch/distributions/transforms.py:48-200` `class Transform` with
  its `_call` (forward), `_inverse`, and `log_abs_det_jacobian`
  abstract methods.

- REQ-2: `pub struct ExpTransform` (`y = exp(x)`) with
  `log_abs_det_jacobian == x` and `is_exp_transform() -> true`.
  All three op bodies use the device-resident core ops
  (`exp_op`, `log_op`) so the transform preserves device through
  every leg. Mirrors `torch/distributions/transforms.py:ExpTransform`.

- REQ-3: `pub struct AffineTransform<T: Float> { loc: T, scale: T }`
  (`y = loc + scale * x`). `log_abs_det_jacobian` broadcasts
  `log|scale|` onto the input shape using `creation::full` then
  `to(device)`. Overrides `constant_entropy_contribution()` to
  return `Some(abs_scale.ln())` so the entropy dispatcher's
  closed-form path 2 kicks in. Mirrors
  `torch/distributions/transforms.py:AffineTransform`.

- REQ-4: Three "trigonometric-style" transforms — `SigmoidTransform`
  (`y = 1/(1+exp(-x))`), `TanhTransform` (`y = tanh(x)`),
  `SoftplusTransform` (`y = log(1+exp(x))`). Each uses the
  device-resident core ops (`sigmoid_op`, `tanh_op`, `softplus_op`)
  and the inverse uses device-resident algebra (e.g. `atanh` via
  `0.5 * log((1+y)/(1-y))`). All three use numerically-stable
  formulas matching PyTorch: sigmoid LDJ is `-softplus(-x) -
  softplus(x)`, tanh LDJ is `2 * (log(2) - x - softplus(-2x))`,
  softplus LDJ is `-softplus(-x)`. Mirrors
  `torch/distributions/transforms.py:{SigmoidTransform,TanhTransform,SoftplusTransform}`.

- REQ-5: `pub struct ComposeTransform<T: Float>` holding
  `Vec<Box<dyn Transform<T>>>`. Forward applies left-to-right;
  inverse applies right-to-left. `log_abs_det_jacobian` computes
  intermediate values `xs[0..=n]` then sums per-link LDJs. The
  empty-chain case returns a zero tensor on `x.device()`.
  `constant_entropy_contribution()` returns `Some(sum)` iff every
  link is constant-Jacobian. Mirrors
  `torch/distributions/transforms.py:ComposeTransform`.

- REQ-6: `pub struct TransformedDistribution<T: Float>` holding a
  `Box<dyn Distribution<T>>` base and `Vec<Box<dyn Transform<T>>>`.
  `sample`/`rsample` push the base sample through the chain.
  `log_prob` walks transforms in reverse: invert back to the base
  sample, accumulate `sum_ldj` via device-resident `add`, then
  return `base.log_prob(inverted) - sum_ldj`. Mirrors
  `torch/distributions/transformed_distribution.py:TransformedDistribution`.

- REQ-7: `TransformedDistribution::entropy` dispatches three
  closed-form cases:
    1. Empty chain → `base.entropy()` directly.
    2. Every link has `constant_entropy_contribution() -> Some(c)`
       → `base.entropy() + sum(c_i)` (the affine-chain case).
    3. Single `ExpTransform` link + base implements `mean()` →
       `base.entropy() + base.mean()` (since `E_X[log|exp'(X)|] =
       E_X[X] = mean`).
  Anything else surfaces a precise `InvalidArgument` error naming
  the problematic transform(s). Mirrors PyTorch's
  `transformed_distribution.py:entropy` which raises
  `NotImplementedError` for the non-closed-form chains; ferrotorch
  ships the three closed-form cases inline rather than as separate
  `register_kl`-style decorators.

- REQ-8: NOT-STARTED — 11 of 16 upstream transforms are missing
  (`AbsTransform`, `PowerTransform`, `SoftmaxTransform`,
  `StickBreakingTransform`, `CatTransform`, `StackTransform`,
  `CorrCholeskyTransform`, `LowerCholeskyTransform`,
  `PositiveDefiniteTransform`, `ReshapeTransform`,
  `IndependentTransform`, `CumulativeDistributionTransform`).
  Likewise the Constraint linkage (`domain` / `codomain` properties
  per transform) is not implemented.

- REQ-9: NOT-STARTED — Monte-Carlo entropy fallback for non-closed-form
  transform chains (Sigmoid, Tanh, Softplus, multi-Exp,
  Exp-then-Affine, etc.) is not implemented; today these surface
  `InvalidArgument` instead of falling back to sampled estimation.

## Acceptance Criteria

- [x] AC-1: `pub trait Transform<T: Float>: Send + Sync` with the
  three required methods + `name` + two defaults.
- [x] AC-2: `pub struct ExpTransform` with `is_exp_transform() ->
  true` and `log_abs_det_jacobian == x.clone()`.
- [x] AC-3: `pub struct AffineTransform<T>` with
  `constant_entropy_contribution() -> Some(abs_scale.ln())`.
- [x] AC-4: `pub struct SigmoidTransform`, `TanhTransform`,
  `SoftplusTransform` — three device-resident bodies.
- [x] AC-5: `pub struct ComposeTransform<T>` with forward L→R,
  inverse R→L, sum-of-LDJs, empty-chain identity branch.
- [x] AC-6: `pub struct TransformedDistribution<T>` with
  `sample`/`rsample`/`log_prob`/`entropy` impls.
- [x] AC-7: `TransformedDistribution::entropy` three-path dispatch
  with named-transform error message on fall-through.
- [ ] AC-8: 11 missing transforms — blocker #1373.
- [ ] AC-9: Monte-Carlo entropy fallback — blocker #1378.

## Architecture

### `Transform` trait (REQ-1)

The trait is the Rust analog of `class Transform` in
`torch/distributions/transforms.py:48`. Differences:

- **Required `name(&self) -> &'static str`**. PyTorch derives the
  display name from `__class__.__name__`; we make it explicit so
  the trait stays object-safe and so the entropy dispatcher's
  error message can name the offending transform without
  reflection.
- **Two default methods inspect-able by the dispatcher**:
  `constant_entropy_contribution() -> Option<T>` is `None` by
  default — only `AffineTransform` and `ComposeTransform`
  (recursively) override to `Some(...)`; `is_exp_transform() ->
  bool` is `false` by default — only `ExpTransform` overrides to
  `true`. These two flags let the dispatcher in REQ-7 enumerate
  closed-form cases without trait-object downcasting.
- **No `domain` / `codomain` Constraint linkage**. Upstream wires
  each Transform to a `domain: constraints.Constraint` and
  `codomain: constraints.Constraint` for `biject_to` /
  `transform_to`; ferrotorch's `Transform` does not.

### `ExpTransform` (REQ-2)

The simplest non-trivial transform. All three method bodies use
`no_grad(|| ...)` wrapping the device-resident core ops:

```rust
fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    no_grad(|| exp_op(x))
}
fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    no_grad(|| log_op(y))
}
fn log_abs_det_jacobian(&self, x: &Tensor<T>, _y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    no_grad(|| Ok(x.clone()))  // log|exp'(x)| = log(exp(x)) = x
}
```

The `no_grad` wrapper is mandatory: transforms are applied during
both forward and `log_prob` passes; the LDJ contribution to
`log_prob` must NOT flow gradients back to the input value (the
gradient flows through `base.log_prob(inverted)` instead).
`is_exp_transform() -> true` is the flag the entropy dispatcher
inspects for closed-form case 3.

### `AffineTransform` (REQ-3)

`y = loc + scale * x`. Stores scalar `loc` and `scale` of type
`T: Float`; materialises them as 0-D scalar tensors on
`x.device()` at every `forward`/`inverse` call via
`scalar_on(value, device)`. This avoids caching state and
re-uploading state for every device.

The LDJ broadcasts `log|scale|` onto `x.shape()`:
```rust
let log_abs_scale = if self.scale > 0 { scale.ln() } else { (-scale).ln() };
let cpu = creation::full(x.shape(), log_abs_scale)?;
cpu.to(x.device())
```

`constant_entropy_contribution()` returns `Some(abs_scale.ln())` so
the entropy dispatcher's closed-form path 2 (all-constant-Jacobian
chain) finds it.

### `SigmoidTransform`, `TanhTransform`, `SoftplusTransform` (REQ-4)

All three use the numerically stable LDJ formulas PyTorch ships:

- `SigmoidTransform`: LDJ = `-softplus(-x) - softplus(x)`. The
  inverse is logit clamped to `(eps, 1-eps)` to avoid log-of-zero.
- `TanhTransform`: LDJ = `2 * (log(2) - x - softplus(-2x))`. This
  is the TensorFlow Probability formula upstream uses.
- `SoftplusTransform`: LDJ = `-softplus(-x) = log(sigmoid(x))`.
  Inverse is `log(exp(y) - 1)`, valid for the f32 input range
  exercised by tests (exp(20) ~ 4.85e8 << f32::MAX ~ 3.4e38).

Each body chains the device-resident core ops; the test suite
(`{exp,affine,sigmoid,tanh,softplus,compose}_transform_preserves_device_and_value`)
asserts `device()` equality and numerical correctness through
every leg.

### `ComposeTransform` (REQ-5)

Holds `Vec<Box<dyn Transform<T>>>`. The trait-object wrapping is
the Rust analog of PyTorch's heterogeneous-list-of-transforms;
`+ Send + Sync` propagates through the `Box<dyn ...>` because
`Transform<T>: Send + Sync` is the trait's super-trait.

The LDJ computation walks the chain forward to materialise
intermediates `xs[0..=n]` (so each link's LDJ can be computed at
its own input), then sums the per-link LDJs via device-resident
`add`. The empty-chain branch returns a zero tensor on
`x.device()`, matching the identity transform's LDJ.

`constant_entropy_contribution()` returns `Some(sum)` iff EVERY
link returns `Some(c_i)`. Any `None` short-circuits to `None` —
the chain is constant-Jacobian iff every link is.

### `TransformedDistribution` (REQ-6, REQ-7)

`pub struct TransformedDistribution<T: Float>` holds
`Box<dyn Distribution<T>>` + `Vec<Box<dyn Transform<T>>>`. The
`Distribution` impl:

- `sample`/`rsample`: `base.sample(shape) → for each t in transforms: t.forward(x)`.
- `log_prob`: walks transforms IN REVERSE. For each `t` in
  `transforms.iter().rev()`: `let x = t.inverse(&y)?; let ldj =
  t.log_abs_det_jacobian(&x, &y)?; sum_ldj += ldj; y = x`. Then
  `base.log_prob(y) - sum_ldj`. All on `value.device()`.
- `entropy`: the three-case dispatcher (REQ-7).

The three closed-form entropy cases are:

1. **Empty chain** → return `base.entropy()`. The identity has
   zero LDJ, so `H(Y) = H(X)`.
2. **All-constant-Jacobian chain** → `H(Y) = H(X) + sum(c_i)`
   where `c_i = log|scale_i|` per link. This catches arbitrary
   affine chains.
3. **Single ExpTransform** → `H(Y) = H(X) + E_X[log|exp'(X)|] = H(X)
   + E_X[X] = H(X) + base.mean()`. This is the LogNormal entropy
   identity. Requires `base.mean()` to be implemented; if not, the
   error from `base.mean()` propagates.

Any other chain (Sigmoid, Tanh, Softplus, multi-Exp,
Exp-with-non-Affine) surfaces an `InvalidArgument` listing the
problematic transform(s) by name. PyTorch's upstream behaviour is
to raise `NotImplementedError` for the same cases; ferrotorch's
diagnostic is more specific (it names the link).

### Non-test production consumers

- `Transform` + `TransformedDistribution` are part of the public
  API surface (`pub use transforms::TransformedDistribution` in
  `lib.rs:108`; the trait itself is `pub` in `transforms.rs`).
  No internal site of `ferrotorch-distributions/src/` constructs
  a `TransformedDistribution` — `LogNormal` is hand-coded rather
  than built on `TransformedDistribution::new(Normal, [Exp])`.
  Goal.md S5 + R-DEFER-1 grandfather existing public API surface
  ("Boundary methods ARE the public API"); the `pub use` is the
  consumer.
- The `entropy` dispatcher INTERNALLY consumes
  `Transform::constant_entropy_contribution()` and
  `Transform::is_exp_transform()` in production code: see
  `fn TransformedDistribution::entropy in transforms.rs`. Both
  default-method overrides on `AffineTransform` and `ExpTransform`
  are exercised on the production path when a caller composes
  them.
- `LogNormal`-via-`TransformedDistribution` is the canonical
  upstream example (`torch/distributions/log_normal.py`) but
  ferrotorch ships `LogNormal` directly. Wiring `LogNormal` to use
  `TransformedDistribution::new(Normal, [Exp])` would be a
  R-DEV-7 swap (preserve API, reuse the chain machinery) tracked
  separately if desired; not a blocker for `transforms.md`.

## Parity contract

`parity_ops = []`. Transforms are composition-layer infrastructure;
the numerical contract is on the underlying core ops (`exp`,
`log`, `sigmoid`, `softplus`, `tanh`) which are parity-tested in
`ferrotorch-core`. Edge cases preserved:

- **`AffineTransform` with `scale < 0`**: LDJ uses `(-scale).ln()`,
  so `log|scale| = log|-scale|`. Verified by
  `test_affine_negative_scale_log_det`.
- **`AffineTransform` with `scale == 0`**: not a valid bijection;
  `constant_entropy_contribution` returns `Some(0_f64.ln()) =
  Some(-inf)`. Upstream silently produces NaN in the same case.
- **`ExpTransform::inverse(y <= 0)`**: dispatches to
  `log_op(y)`, which returns NaN/-inf for non-positive y. Matches
  upstream `torch.log` behaviour.
- **`SigmoidTransform::inverse(0 or 1)`**: clamped to
  `(eps, 1-eps)` with `eps = 1e-7`. Matches the prior CPU body's
  domain-safety contract.
- **`SoftplusTransform::inverse(y <= 0)`**: `log(exp(y) - 1)` —
  `exp(0) - 1 = 0`, so `log(0) = -inf`. Matches upstream.
- **Device preservation**: every method body wraps in `no_grad` +
  uses device-resident core ops; the
  `*_transform_preserves_device_and_value` tests assert
  `result.device() == input.device()` on every leg.
- **`TransformedDistribution::entropy` fall-through**: surfaces an
  `InvalidArgument` listing problematic transforms by name. The
  message format is `"closed-form contribution is intractable for
  transform(s) [<name>, ...]"`. Tests
  `test_transformed_distribution_entropy_sigmoid_errors` and
  `_exp_then_affine_errors` pin the error shape.

## Verification

Tests in `mod tests in transforms.rs` (~30 tests):

- `test_exp_{forward,inverse,roundtrip,log_det_jacobian}`
- `test_affine_{forward,inverse,roundtrip,log_det_jacobian,negative_scale_log_det}`
- `test_sigmoid_{forward,roundtrip,log_det_jacobian}`
- `test_tanh_{forward,roundtrip,log_det_jacobian}`
- `test_softplus_{forward,roundtrip,log_det_jacobian}`
- `test_compose_{empty_is_identity,exp_then_affine,roundtrip,log_det_jacobian}`
- `test_transformed_distribution_{sample_shape,log_prob,log_prob_general,entropy_empty_chain_matches_base,entropy_affine,entropy_affine_negative_scale,entropy_exp_matches_lognormal,entropy_compose_affine_chain,entropy_sigmoid_errors,entropy_exp_then_affine_errors}`
- `test_transforms_f64` — f64 round-trip for Exp, Affine, Sigmoid.
- `{exp,affine,sigmoid,tanh,softplus,compose}_transform_preserves_device_and_value`
- `transformed_distribution_log_prob_preserves_device`

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-distributions --lib transforms:: 2>&1 | tail -3
```

Expected: `~30 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub trait Transform<T: Float>: Send + Sync` with `forward`/`inverse`/`log_abs_det_jacobian`/`name` + two defaults in `transforms.rs`, mirroring `torch/distributions/transforms.py:48-200`; non-test consumer: `pub trait Transform` is grandfathered public API (per goal.md S5); `fn TransformedDistribution::entropy in transforms.rs` invokes `t.constant_entropy_contribution()` and `t.is_exp_transform()` on the trait in production. |
| REQ-2 | SHIPPED | impl: `pub struct ExpTransform` with `is_exp_transform() -> true` and `log_abs_det_jacobian == x.clone()` in `transforms.rs`, all bodies device-resident, mirroring `torch/distributions/transforms.py:ExpTransform`; non-test consumer: `fn TransformedDistribution::entropy in transforms.rs` inspects `is_exp_transform()` for closed-form case 3; `pub use` is grandfathered public API. |
| REQ-3 | SHIPPED | impl: `pub struct AffineTransform<T: Float> { loc, scale }` with device-resident scalar materialisation via `scalar_on` and `constant_entropy_contribution() -> Some(abs_scale.ln())` in `transforms.rs`, mirroring `torch/distributions/transforms.py:AffineTransform`; non-test consumer: `fn TransformedDistribution::entropy in transforms.rs` consumes `constant_entropy_contribution` for closed-form case 2; grandfathered public API. |
| REQ-4 | SHIPPED | impl: `pub struct SigmoidTransform`, `TanhTransform`, `SoftplusTransform` with numerically-stable device-resident formulas in `transforms.rs`, mirroring `torch/distributions/transforms.py:{SigmoidTransform,TanhTransform,SoftplusTransform}`; non-test consumer: `pub use transforms::*` glob in PyTorch parity (`__all__`) — ferrotorch ships these as public API; tests `sigmoid_transform_preserves_device_and_value` et al. verify the chain. |
| REQ-5 | SHIPPED | impl: `pub struct ComposeTransform<T: Float>` + L→R forward + R→L inverse + sum-of-LDJs + empty-chain identity branch in `transforms.rs`, mirroring `torch/distributions/transforms.py:ComposeTransform`; non-test consumer: `pub use` grandfathered API + the inner `Vec<Box<dyn Transform>>` used by tests `test_compose_*` exercises the production path; the `constant_entropy_contribution()` override is read by `fn TransformedDistribution::entropy`. |
| REQ-6 | SHIPPED | impl: `pub struct TransformedDistribution<T: Float>` with `Box<dyn Distribution<T>>` base + `Vec<Box<dyn Transform<T>>>` chain + `sample`/`rsample`/`log_prob`/`entropy` Distribution impl in `transforms.rs`, mirroring `torch/distributions/transformed_distribution.py:TransformedDistribution`; non-test consumer: `pub use transforms::TransformedDistribution` in `lib.rs:108` — grandfathered public API. |
| REQ-7 | SHIPPED | impl: `fn TransformedDistribution::entropy` three-case dispatcher (empty, all-constant-Jacobian, single-Exp) with named-transform error message on fall-through in `transforms.rs`; non-test consumer: `fn TransformedDistribution::entropy` is itself the dispatcher; the consumer is any downstream code calling `td.entropy()` — `pub use TransformedDistribution` makes the entropy method part of the public Distribution impl and `test_transformed_distribution_entropy_*` tests pin all four branches. |
| REQ-8 | NOT-STARTED | blocker #1373 — 11 of 16 upstream transforms not ported (`AbsTransform`, `PowerTransform`, `SoftmaxTransform`, `StickBreakingTransform`, `CatTransform`, `StackTransform`, `CorrCholeskyTransform`, `LowerCholeskyTransform`, `PositiveDefiniteTransform`, `ReshapeTransform`, `IndependentTransform`, `CumulativeDistributionTransform`); also Constraint domain/codomain linkage missing. |
| REQ-9 | NOT-STARTED | blocker #1378 — Monte-Carlo entropy fallback for non-closed-form chains (Sigmoid/Tanh/Softplus/multi-Exp/Exp-then-Affine) not implemented; PyTorch uses MC fallback, ferrotorch surfaces `InvalidArgument`. |
