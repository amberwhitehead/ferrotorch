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
sixteen concrete transforms (`ExpTransform`, `AffineTransform`,
`SigmoidTransform`, `TanhTransform`, `SoftplusTransform`,
`AbsTransform`, `PowerTransform`, `SoftmaxTransform`,
`StickBreakingTransform`, `LowerCholeskyTransform`,
`CorrCholeskyTransform`, `ReshapeTransform`, `IndependentTransform`,
`CatTransform`, `StackTransform`, `CumulativeDistributionTransform`),
the `ComposeTransform` chain wrapper, and the `TransformedDistribution`
adapter that applies a transform chain to a base distribution.
Mirrors `torch/distributions/transforms.py` (the upstream `__all__` has
17 transform classes; ferrotorch ports 16 — `PositiveDefiniteTransform`
remains NOT-STARTED, tracked under #1373) and
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

- REQ-8: SHIPPED (#1373) — the Constraint domain/codomain linkage plus
  the 11 remaining upstream transforms are now ported.
  `Transform::domain()` / `Transform::codomain()` return object-safe
  `Box<dyn DistConstraint>` accessors, defaulting to `Real`/`Real` and
  overridden per transform to mirror the `domain`/`codomain` class
  attributes in `torch/distributions/transforms.py`. The 11 newly
  ported transforms each mirror upstream `_call`/`_inverse`/
  `log_abs_det_jacobian`/`domain`/`codomain`:
    - `AbsTransform` (`y=|x|`, R→Positive, not bijective, LDJ undefined)
    - `PowerTransform` (`y=x^exp`, scalar `T` exponent, Positive→Positive)
    - `SoftmaxTransform` (RealVector→Simplex, not bijective, LDJ undefined)
    - `StickBreakingTransform` (RealVector→Simplex, event-dim-1)
    - `LowerCholeskyTransform` (→LowerCholesky, event-dim-2)
    - `CorrCholeskyTransform` (RealVector→CorrCholesky, signed stick-breaking)
    - `ReshapeTransform` (unit-Jacobian trailing-dim reshape)
    - `IndependentTransform` (wraps a base transform; LDJ sums rightmost dims)
    - `CatTransform` / `StackTransform` (sub-transforms over a dim)
    - `CumulativeDistributionTransform` (CDF/ICDF/log_prob of a base dist)
  The trait gained `event_dim()`/`bijective()`/`sign()` defaults
  (mirroring the upstream class attributes). New codomain constraints
  `RealVector`/`CorrCholesky`/`LowerCholesky` were added to
  `constraints.rs`. The production consumer is
  `TransformedDistribution::support()` (chain's final codomain) plus the
  `TransformedDistribution` chain machinery exercising each transform's
  forward/inverse/LDJ; each transform is re-exported from `lib.rs` as
  boundary public API (goal.md S5). Still NOT-STARTED: 1 of 17 upstream
  transforms — `PositiveDefiniteTransform` (its `_inverse` composes
  `torch.linalg.cholesky` with `LowerCholeskyTransform.inv` on the
  event-dim-2 path; it is outside the #1373 dispatch manifest). Blocker
  #1373's remaining scope is that single transform.

- REQ-9: SHIPPED — Monte-Carlo entropy fallback for non-closed-form
  transform chains (Sigmoid, Tanh, Softplus, multi-Exp,
  Exp-then-Affine, etc.). `fn TransformedDistribution::entropy_monte_carlo`
  estimates `H(Y) = H(X) + E_X[log|det J_f(X)|]` with
  `MC_ENTROPY_SAMPLES = 20_000` base draws pushed through each link's
  `log_abs_det_jacobian`, averaged over the sample axis and broadcast
  onto `base_entropy`'s shape. The `entropy` dispatcher invokes it as
  path 4 on fall-through, so `td.entropy()` on these chains now returns
  a value instead of `InvalidArgument`. Quadrature-verified by
  `test_transformed_distribution_entropy_{sigmoid,exp_then_affine}_monte_carlo`
  + integration `divergence_wave_k_audit::audit_1378_*`. Closes #1378.

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
- [x] AC-8: Constraint domain/codomain linkage + the 11 remaining
  upstream transforms — SHIPPED (#1373) via `Transform::domain()` /
  `codomain()` + the 11 transform impls + `TransformedDistribution::support()`
  consumer, pinned by `test_transform_domain_codomain_names`,
  `test_compose_domain_codomain_endpoints`,
  `test_transformed_distribution_support_is_last_codomain`,
  `test_{abs,power,softmax,stick_breaking,lower_cholesky,corr_cholesky,reshape,independent,cat,stack,cumulative_distribution}_transform`,
  `test_transformed_distribution_with_power_transform`, and
  `divergence_wave_l_audit::audit_1373_*`. Only
  `PositiveDefiniteTransform` stays NOT-STARTED (out of dispatch scope).
- [x] AC-9: Monte-Carlo entropy fallback for X-dependent-Jacobian
  chains — closes #1378.

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
- **`domain` / `codomain` Constraint linkage** (#1373). The trait
  exposes `domain()` and `codomain()` returning object-safe
  `Box<dyn DistConstraint>` (the `Constraint::check<T>` generic
  method forbids a plain trait object, so the dtype-independent
  `DistConstraint` surface from `lib.rs` is the carrier). Defaults
  are `Real`/`Real`; overrides per transform mirror the
  `domain`/`codomain` class attributes upstream. The production
  consumer is `TransformedDistribution::support()`. Upstream uses
  these for `biject_to` / `transform_to`; ferrotorch surfaces them
  as the support of a transformed distribution.

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
  `lib.rs`; the trait itself is `pub` in `transforms.rs`).
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
| REQ-6 | SHIPPED | impl: `pub struct TransformedDistribution<T: Float>` with `Box<dyn Distribution<T>>` base + `Vec<Box<dyn Transform<T>>>` chain + `sample`/`rsample`/`log_prob`/`entropy` Distribution impl in `transforms.rs`, mirroring `torch/distributions/transformed_distribution.py:TransformedDistribution`; non-test consumer: `pub use transforms::TransformedDistribution` in `lib.rs` — grandfathered public API. |
| REQ-7 | SHIPPED | impl: `fn TransformedDistribution::entropy` three-case dispatcher (empty, all-constant-Jacobian, single-Exp) with named-transform error message on fall-through in `transforms.rs`; non-test consumer: `fn TransformedDistribution::entropy` is itself the dispatcher; the consumer is any downstream code calling `td.entropy()` — `pub use TransformedDistribution` makes the entropy method part of the public Distribution impl and `test_transformed_distribution_entropy_*` tests pin all four branches. |
| REQ-8 | SHIPPED | #1373 — Constraint domain/codomain linkage + the 11 remaining upstream transforms. impl: `fn Transform::domain` / `fn Transform::codomain` (default `Real`/`Real`) with per-transform overrides plus 11 new `pub struct`s in `transforms.rs` — `AbsTransform`, `PowerTransform<T>`, `SoftmaxTransform`, `StickBreakingTransform`, `LowerCholeskyTransform`, `CorrCholeskyTransform`, `ReshapeTransform`, `IndependentTransform<T>`, `CatTransform<T>`, `StackTransform<T>`, `CumulativeDistributionTransform<T>`, each mirroring upstream `_call`/`_inverse`/`log_abs_det_jacobian`/`domain`/`codomain` at `torch/distributions/transforms.py:741-754` (Abs), `:599-639` (Power), `:947-980` (Softmax), `:983-1036` (StickBreaking), `:1039-1058` (LowerCholesky), `:864-944` (CorrCholesky), `:500-573` (Reshape), `:422-497` (Independent), `:1081-1220` (Cat), `:1223-1321` (Stack), `:1324-1367` (CDF); trait gained `event_dim()`/`bijective()`/`sign()` defaults (`transforms.py:93,113-117,133-139`); new constraints `RealVector`/`CorrCholesky`/`LowerCholesky` in `constraints.rs` (`constraints.py:943,947,954`). non-test consumer: `fn TransformedDistribution::support in transforms.rs` returns the chain's final codomain (`transformed_distribution.py:129-137`) and the `TransformedDistribution` chain (sample/log_prob/entropy) drives each boxed transform's forward/inverse/LDJ — reachable via `pub use transforms::{AbsTransform, …, TransformedDistribution}` in `lib.rs`. Pinned by `test_{transform_domain_codomain_names,compose_domain_codomain_endpoints,transformed_distribution_support_is_last_codomain,abs_transform_forward_inverse,power_transform,softmax_transform,stick_breaking_transform,lower_cholesky_transform,corr_cholesky_transform,reshape_transform,independent_transform,cat_transform,stack_transform,cumulative_distribution_transform,transformed_distribution_with_power_transform}` (oracle-derived from live torch 2.11) + `divergence_wave_l_audit::audit_1373_*`. Remaining NOT-STARTED: 1 of 17 upstream transforms — `PositiveDefiniteTransform` (event-dim-2 cholesky composition, outside the #1373 dispatch manifest). |
| REQ-9 | SHIPPED | impl: `fn TransformedDistribution::entropy_monte_carlo` in `transforms.rs` estimates `H(Y) = H(X) + E_X[log|det J_f(X)|]` with `MC_ENTROPY_SAMPLES = 20_000` base draws pushed through each link's `log_abs_det_jacobian`, averaged over the sample axis and broadcast onto the batch shape; non-test consumer: `fn TransformedDistribution::entropy` invokes it as path 4 on fall-through (so a Sigmoid/Tanh chain's `td.entropy()` returns a value instead of erroring) — reachable through `pub use transforms::TransformedDistribution`. Quadrature-verified by `test_transformed_distribution_entropy_{sigmoid,exp_then_affine}_monte_carlo` + integration `divergence_wave_k_audit::audit_1378_{sigmoid,tanh,exp_then_affine}_*`. Closes #1378. |
