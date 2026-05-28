# ferrotorch-distributions — `lognormal` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/log_normal.py
-->

## Summary

`ferrotorch-distributions/src/lognormal.rs` defines `LogNormal<T>` —
the distribution of `Y = exp(X)` where
`X ~ Normal(loc, scale)`. Mirrors `torch.distributions.LogNormal`
(`torch/distributions/log_normal.py:13-75`). Upstream constructs
LogNormal as a `TransformedDistribution` over `Normal` composed with
`ExpTransform`; ferrotorch ships a direct implementation that
performs the exp + log accounting inline.

## Requirements

- REQ-1: `pub struct LogNormal<T: Float>` with `loc: Tensor<T>` and
  `scale: Tensor<T>` fields (the mu and sigma of the *underlying*
  normal). Mirrors the upstream `loc`/`scale` properties at
  `log_normal.py:53-59`.

- REQ-2: `pub fn LogNormal::new(loc, scale) -> FerrotorchResult<Self>`
  with shape-equality precondition + `ShapeMismatch` rejection.
  Mirrors upstream's implicit `Normal(loc, scale)` constructor at
  `log_normal.py:46`.

- REQ-3: `loc()` and `scale()` accessors for parameter introspection.
  Mirrors upstream property access at `log_normal.py:53-59`.

- REQ-4: `impl<T: Float> Distribution<T> for LogNormal<T>` with
  closed-form `sample`, `rsample`, `log_prob`, `entropy`, `mean`,
  `mode`, `variance`. Mirrors the corresponding methods at
  `log_normal.py:53-75`.

- REQ-5: Sampling via the underlying normal — `z = exp(loc + scale * eps)`
  where `eps ~ N(0, 1)`. Mirrors upstream's `ExpTransform(Normal)`
  chain at `log_normal.py:46-47`.

- REQ-6: `rsample` attaches `LogNormalRsampleBackward` for autograd
  flow through `loc` and `scale` with `d(z)/d(loc) = z` and
  `d(z)/d(scale) = z * eps`. Mirrors upstream's autograd-traced
  `ExpTransform(Normal)` chain.

- REQ-7: `log_prob` uses the change-of-variables formula
  `log_prob_lognormal(x) = log_prob_normal(ln(x)) - ln(x)` — the
  `-ln(x)` term is the log-determinant of the `ExpTransform`'s
  Jacobian. Mirrors upstream's `TransformedDistribution.log_prob`
  reduction.

- REQ-8: Closed-form mean / mode / variance:
  - mean = `exp(mu + sigma^2/2)`
  - mode = `exp(mu - sigma^2)`
  - variance = `(exp(sigma^2) - 1) * exp(2*mu + sigma^2)`

  Mirror upstream `log_normal.py:61-72`.

- REQ-9: `entropy = mu + 0.5 + ln(sigma) + 0.5 * ln(2*pi)`. This
  is `Normal(mu, sigma).entropy() + mu` (the additive
  `+ self.loc` term comes from the `ExpTransform`'s log-det
  Jacobian integrated against the base density). Mirrors
  upstream's `self.base_dist.entropy() + self.loc`
  (`log_normal.py:74-75`).

- REQ-10: Device-resident outputs — every method ends with an
  `out.to(device)` if `loc.device().is_cuda()`. Mirrors the
  upstream implicit device contract.

- REQ-11: `mean_value()` / `variance_value()` are public helpers
  returning `Vec<T>` rather than a `Tensor<T>`. These are the
  pre-`mean()`/`variance()` API that downstream code used before
  the `Distribution::mean` trait method was added. Both helpers
  remain part of the public surface (not removed) to preserve the
  prior call sites.

## Acceptance Criteria

- [x] AC-1: `pub struct LogNormal<T: Float>` with `loc`/`scale`.
- [x] AC-2: `pub fn LogNormal::new` rejects shape-mismatched input.
- [x] AC-3: `loc()`/`scale()` accessors + `mean_value()`/`variance_value()`.
- [x] AC-4: `impl Distribution<T> for LogNormal<T>` with the 7
  trait methods.
- [x] AC-5: `test_lognormal_sample_positive` confirms samples are
  positive.
- [x] AC-6: `test_lognormal_log_prob_at_e` confirms log_prob at
  `x = e` matches `-1.5 - 0.5*ln(2*pi)` for `LogNormal(0, 1)`.
- [x] AC-7: `test_lognormal_entropy` confirms
  `entropy(0, 1) == 0.5 + 0.5*ln(2*pi)`.
- [x] AC-8: `test_lognormal_rsample_backward` confirms gradient
  flow through `loc` and `scale`.
- [x] AC-9: `test_lognormal_mean_mode_variance` confirms closed
  forms match.

## Architecture

### The struct (REQ-1, REQ-2, REQ-3, REQ-11)

```rust
pub struct LogNormal<T: Float> {
    loc: Tensor<T>,
    scale: Tensor<T>,
}
```

Defined at `lognormal.rs`. Constructor at `lognormal.rs`.
Accessors at `mean_value in lognormal.rs`. The `mean_value()` /
`variance_value()` helpers at `variance_value in lognormal.rs` return `Vec<T>`
directly for old call sites.

### The Distribution impl (REQ-4, REQ-5, REQ-7, REQ-8, REQ-9, REQ-10)

`sample` (`sample in lognormal.rs`) draws `eps ~ N(0, 1)` via
`creation::randn` then computes `(loc + scale * eps).exp()`
element-wise with cyclic broadcasting.

`rsample` (`rsample in lognormal.rs`) follows the same closed form
and attaches `LogNormalRsampleBackward` when either parameter
has `requires_grad`.

`log_prob` (`log_prob in lognormal.rs`) computes
`-0.5 * z^2 - ln(scale) - 0.5*ln(2*pi) - ln(x)` where
`z = (ln(x) - loc) / scale`. The trailing `- ln(x)` is the
log-det-Jacobian of the `ExpTransform`.

`entropy` (`entropy in lognormal.rs`) computes
`mu + 0.5 + ln(sigma) + 0.5*ln(2*pi)`. Mirrors upstream
`base_dist.entropy() + loc`.

`mean`/`mode`/`variance` (`variance in lognormal.rs`) use the closed
forms enumerated in REQ-8.

### LogNormalRsampleBackward (REQ-6)

Defined at `loc in lognormal.rs`. Holds `loc`, `scale`, and the
saved `eps`. On `backward(grad_output)`:

- `grad_loc = sum(grad_output * z)` where `z = exp(loc + scale*eps)`
- `grad_scale = sum(grad_output * z * eps)`

Both follow directly from `d/d(loc) exp(loc + scale*eps) = z` and
`d/d(scale) exp(loc + scale*eps) = z * eps`.

### Non-test production consumers

- `pub use lognormal::LogNormal` at `lib.rs` — grandfathered
  public API. Downstream VAE / financial-modelling / Bayesian
  inference code constructs `LogNormal::new(mu, sigma)?`.
- `LogNormalRsampleBackward` is consumed by the autograd engine
  via `Tensor::from_operation` — production consumer of the
  `GradFn<T>` trait.

## Parity contract

`parity_ops = []`. LogNormal has no direct parity-sweep oracle;
underlying primitives (`exp`, `ln`, `randn`) are independently
audited. Edge cases preserved:

- **`x <= 0` in log_prob** — yields `NaN`/`-infinity` via
  `x.ln()` returning negative-infinity / NaN. Upstream
  `TransformedDistribution.log_prob` likewise returns `-inf`
  via `validate_args` filtering.
- **Sample positivity** — every output of `exp(...)` is positive
  by construction. Test `test_lognormal_sample_positive` pins
  this.
- **`mu = 0, sigma = 1`** — mean = `exp(0.5)`, mode = `exp(-1)`,
  variance = `(e - 1) * e`. Test
  `test_lognormal_mean_mode_variance` pins this.
- **Mode is below mean** — for any `sigma > 0`,
  `mode = exp(mu - sigma^2) < exp(mu + sigma^2/2) = mean`.
- **Entropy formula** — equals `Normal.entropy() + mu` exactly
  (the `+ mu` term comes from the average log-det of the exp).

## Verification

Tests in `mod tests in lognormal.rs` (10 tests):

- `test_lognormal_sample_shape`,
- `test_lognormal_sample_positive`,
- `test_lognormal_rsample_has_grad`,
- `test_lognormal_log_prob`,
- `test_lognormal_log_prob_at_e`,
- `test_lognormal_entropy`,
- `test_lognormal_shape_mismatch`,
- `test_lognormal_rsample_backward`,
- `test_lognormal_f64`,
- `test_lognormal_mean_mode_variance`.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-distributions --lib lognormal:: 2>&1 | tail -3
```

Expected: `10 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct LogNormal<T: Float>` with `loc`/`scale` `Tensor<T>` fields at `LogNormal in lognormal.rs`, mirroring `torch/distributions/log_normal.py:13-59`; non-test consumer: `pub use lognormal::LogNormal` at `lib.rs`. |
| REQ-2 | SHIPPED | impl: the constructor at `new in lognormal.rs` with shape-equality precondition, mirroring `log_normal.py:40-47`; non-test consumer: the re-export at `lib.rs` exposes `::new` as part of the public API. |
| REQ-3 | SHIPPED | impl: `loc()`/`scale()` accessors at `scale in lognormal.rs`, mirroring `log_normal.py:53-59`; non-test consumer: re-export at `lib.rs`. |
| REQ-4 | SHIPPED | impl: `impl<T: Float> Distribution<T> for LogNormal<T>` at `sample in lognormal.rs` with `sample`/`rsample`/`log_prob`/`entropy`/`mean`/`mode`/`variance`, mirroring `log_normal.py:46-75`; non-test consumer: re-export at `lib.rs` means external Distribution trait callers hit this impl. 10 tests pin behaviour. |
| REQ-5 | SHIPPED | impl: `(l + s * e).exp()` body in `sample in lognormal.rs`, mirroring `ExpTransform(Normal)` chain at `log_normal.py:46-47`; non-test consumer: `Distribution::sample` invocation through the re-export. |
| REQ-6 | SHIPPED | impl: `LogNormalRsampleBackward in lognormal.rs` attached via `Tensor::from_operation` at `from_operation in lognormal.rs`; non-test consumer: the autograd engine in `ferrotorch_core::tensor` traverses this `GradFn<T>` on `backward()`. |
| REQ-7 | SHIPPED | impl: `-(half * z * z) - scale.ln() - half * log_2pi - ln_x` body of `log_prob in lognormal.rs`, mirroring upstream's `TransformedDistribution.log_prob` reduction; non-test consumer: `Distribution::log_prob` via re-export. Tests pin numeric value at `x=1` and `x=e`. |
| REQ-8 | SHIPPED | impl: closed-form `mean`/`mode`/`variance in lognormal.rs` using `(mu + 0.5*sigma^2).exp()` / `(mu - sigma^2).exp()` / `(exp(sigma^2) - 1) * exp(2*mu + sigma^2)`, mirroring `log_normal.py:61-72`; non-test consumer: re-export at `lib.rs` exposes them via the `Distribution` trait. |
| REQ-9 | SHIPPED | impl: `mu + half + sigma.ln() + half * log_2pi` body of `entropy in lognormal.rs`, mirroring upstream's `base_dist.entropy() + self.loc` at `log_normal.py:74-75`; non-test consumer: re-export + `Distribution::entropy` external invocation. |
| REQ-10 | SHIPPED | impl: `out.to(device)` at the tail of every method (e.g. `lognormal.rs`); non-test consumer: every external caller receives device-correct tensors. |
| REQ-11 | SHIPPED | impl: `pub fn mean_value`/`pub fn variance_value` at `variance_value in lognormal.rs` returning `Vec<T>`; non-test consumer: `fn LogNormal::mean` at `LogNormal in lognormal.rs` calls `self.mean_value()?`, and `fn LogNormal::variance` at `LogNormal in lognormal.rs` calls `self.variance_value()?` — both are production sites in the same module. |
| REQ-12 | SHIPPED | impl: `has_rsample`(=true) / `batch_shape` / `support`(`Positive` per `log_normal.py:35`) / `arg_constraints`(`{loc: Real, scale: Positive}` per `log_normal.py:33`) / `expand` overrides at the tail of `impl Distribution for LogNormal` in `lognormal.rs` mirroring `torch/distributions/log_normal.py:33-36`; non-test consumer: trait dispatch through `pub use lognormal::LogNormal` re-export at `lib.rs`; `test_lognormal_surface_overrides` and `test_lognormal_expand` pin the overrides. |
