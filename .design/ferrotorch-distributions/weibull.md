# ferrotorch-distributions — `weibull` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/weibull.py
-->

## Summary

`ferrotorch-distributions/src/weibull.rs` defines `Weibull<T: Float>`
— the two-parameter Weibull distribution parameterized by `scale`
(lambda) and `concentration` (k, shape). Mirrors
`torch.distributions.Weibull` (`torch/distributions/weibull.py:16-96`).
Upstream builds Weibull as
`TransformedDistribution(Exponential(1), [PowerTransform(1/k), AffineTransform(scale)])`;
ferrotorch ships a direct inverse-CDF sampler
`x = scale * (-log(1-u))^(1/k)` plus closed-form
`log_prob`, `entropy`, `cdf`, `icdf`, `mean`, `mode`, `variance`.
`rsample` is NOT implemented because the inverse-CDF path is not
yet autograd-aware.

## Requirements

- REQ-1: `pub struct Weibull<T: Float>` storing `scale: Tensor<T>`
  and `concentration: Tensor<T>`. Mirrors `weibull.py:40-56`
  `__init__` which broadcasts the two params.

- REQ-2: `pub fn Weibull::new(scale, concentration) -> FerrotorchResult<Self>`
  — constructor requiring matching shapes. Upstream uses
  `broadcast_all`; ferrotorch's strict shape-match is R-DEV-7.

- REQ-3: `pub fn scale(&self) -> &Tensor<T>` and
  `pub fn concentration(&self) -> &Tensor<T>` accessors. Mirror
  upstream attribute access.

- REQ-4: `impl<T: Float> Distribution<T> for Weibull<T>` provides
  `sample(shape)` via inverse-CDF:
  ```text
  u ~ Uniform(0, 1)
  x = scale * (-log(1 - u))^(1/k)
  ```
  Mirrors upstream's `TransformedDistribution(Exponential, [Power, Affine])`
  chain (`weibull.py:48-56`):
  ```
  Exponential(1).sample = -log(U)
  PowerTransform(1/k) : -log(U) → (-log(U))^(1/k)
  AffineTransform(scale): (-log(U))^(1/k) → scale * (-log(U))^(1/k)
  ```
  ferrotorch uses `-log(1 - U)` instead of `-log(U)` — these are
  identically distributed for `U ~ Uniform(0, 1)`. The `1 - u` is
  clamped at `1e-30` to avoid `log(0)`.

- REQ-5: `log_prob(value)` returns
  `log(k/lambda) + (k-1)*log(x/lambda) - (x/lambda)^k` for `x >= 0`
  else `-inf`. The closed-form Weibull log density. Upstream
  computes via the transform chain — ferrotorch ships the closed
  form directly.

- REQ-6: `entropy()` returns
  `euler_gamma * (1 - 1/k) + log(lambda/k) + 1`. Mirrors
  `weibull.py:91-96` `entropy = euler_constant * (1 - 1/k) + log(lambda * (1/k)) + 1`
  with `euler_gamma ≈ 0.5772156649015329`.

- REQ-7: `cdf(value)` returns `1 - exp(-(x/lambda)^k)` for `x >= 0`
  else `0`. Closed-form Weibull CDF. Upstream does NOT override
  `cdf` directly; it would come from the transform chain's CDF
  composition. ferrotorch ships the closed form as an R-DEV-7
  enhancement.

- REQ-8: `icdf(q)` returns `lambda * (-log(1 - p))^(1/k)`. Inverse
  CDF, also the formula for `sample`. R-DEV-7 enhancement.

- REQ-9: `mean()` returns `lambda * exp(lgamma(1 + 1/k))`. Mirrors
  `weibull.py:72-74` `scale * exp(lgamma(1 + 1/k))`. Uses
  `lgamma_scalar` from `special_fns.rs` to support fractional `k`.

- REQ-10: `mode()` returns `lambda * ((k-1)/k)^(1/k)` for `k > 1`
  else `0`. Mirrors `weibull.py:76-82`. For `k <= 1` the Weibull mode
  is at 0 (where the density is unbounded for `k < 1`); upstream
  uses
  `scale * ((k-1)/k)^(1/k)` which gives NaN for `k <= 1` (since
  `(k-1)/k < 0`); ferrotorch's explicit `k <= 1 → 0` branch is the
  R-DEV-6 correction of an upstream wart.

- REQ-11: `variance()` returns
  `lambda^2 * (exp(lgamma(1+2/k)) - exp(lgamma(1+1/k))^2)`. Mirrors
  `weibull.py:85-89` `scale^2 * (exp(lgamma(1+2/k)) - exp(2*lgamma(1+1/k)))`
  which is algebraically the same (`exp(2*lgamma(a)) = exp(lgamma(a))^2`).

- REQ-12: NOT-STARTED — `rsample` returns `InvalidArgument`.
  Upstream's `rsample` works via the `TransformedDistribution` chain
  (`PowerTransform` + `AffineTransform` both have autograd-aware
  forward methods). ferrotorch's direct inverse-CDF path does not
  build the autograd graph. Blocker #1435 tracks differentiable
  rsample.

- REQ-13: NOT-STARTED — `expand` (`weibull.py:58-70`),
  `support = constraints.positive` (`weibull.py:38`),
  `concentration_reciprocal` attribute (cached `1/k`) not
  implemented. Cross-cutting with `lib.md` REQ-5
  (Distribution-trait-surface blocker #1376); Weibull-specific
  surface fill-out tracked in blocker #1436.

## Acceptance Criteria

- [x] AC-1: `pub struct Weibull<T: Float>` with `scale`, `concentration`.
- [x] AC-2: `new` rejecting shape mismatch.
- [x] AC-3: `scale()`, `concentration()` accessors.
- [x] AC-4: `Distribution::sample` via inverse-CDF.
- [x] AC-5: `Distribution::log_prob` with `x < 0 → -inf`.
- [x] AC-6: `Distribution::entropy`.
- [x] AC-7: `Distribution::cdf` closed-form.
- [x] AC-8: `Distribution::icdf` closed-form.
- [x] AC-9: `Distribution::mean` via `lgamma`.
- [x] AC-10: `Distribution::mode` with `k <= 1 → 0` (R-DEV-6).
- [x] AC-11: `Distribution::variance` via `lgamma`.
- [ ] AC-12: `rsample` — blocker #1435.
- [ ] AC-13: `expand`, `support`, `concentration_reciprocal` — blocker #1436.

## Architecture

### The struct (REQ-1, REQ-2, REQ-3)

Two-tensor carrier `Weibull<T: Float>` with strict shape match in
`Weibull::new`. Both parameters must be positive (Weibull's
mathematical support requires `scale, concentration > 0`); the
`arg_constraints` plumbing is the `lib.md` REQ-5 cross-cutting gap.

The accessors `pub fn scale` / `pub fn concentration` mirror upstream
attribute access. Upstream also caches `concentration_reciprocal`
(`weibull.py:47`) as an instance attribute; ferrotorch recomputes
`1/k` per call (REQ-13 blocker #1436 tracks the optimization).

### Inverse-CDF sampling (REQ-4)

```text
For each output position:
    U ~ Uniform(0, 1)
    log_term = log( max(1 - U, 1e-30) )
    x = scale * (-log_term)^(1/k)
```

The `max(1 - U, 1e-30)` clamp prevents `log(0)` when `U = 1.0`
exactly. The exponent `1/k` is computed inline (not cached).

This is mathematically identical to upstream's transform chain:

```text
Exponential(rate=1).sample = -log(U)               (Exponential ICDF)
PowerTransform(exp=1/k).forward(x) = x^(1/k)
AffineTransform(loc=0, scale=lambda).forward(y) = lambda * y
```

So `lambda * (-log(U))^(1/k)` is the composed sampler.
ferrotorch's `-log(1-U)` substitution preserves the distribution
(both `U` and `1-U` are `Uniform(0,1)`).

### Closed-form moments (REQ-5..REQ-11)

All are direct scalar arithmetic per element:

- `log_prob(x)`:
  ```
  if x < 0: -inf
  else: log(k/lambda) + (k-1)*log(x/lambda) - (x/lambda)^k
  ```
- `entropy = euler_gamma * (1 - 1/k) + log(lambda/k) + 1`
- `cdf(x)`:
  ```
  if x < 0: 0
  else: 1 - exp(-(x/lambda)^k)
  ```
- `icdf(p) = lambda * (-log(1 - p))^(1/k)` (clamped against `log(0)`)
- `mean = lambda * exp(lgamma(1 + 1/k))`
- `mode`:
  ```
  if k > 1: lambda * ((k-1)/k)^(1/k)
  else: 0           (R-DEV-6: upstream returns NaN for k <= 1)
  ```
- `variance = lambda^2 * (exp(lgamma(1 + 2/k)) - exp(lgamma(1 + 1/k))^2)`

All `lgamma` calls go to `special_fns.rs::lgamma_scalar`.

### Non-test production consumers

- `pub use weibull::Weibull` in `lib.rs:124` — grandfathered public
  API re-export. Downstream reliability / survival-analysis code
  (e.g. Cox proportional-hazards baseline modeling) constructs
  `Weibull::new(scale, concentration)?` directly.
- `Weibull::new` is registered in
  `tests/conformance/_surface_inventory.toml:511`.
- The lib-level docs table in `lib.rs:41` references it with
  "No (rsample not yet implemented)" for Reparameterized.

### Fallback gate

Every `Distribution` method first invokes
`crate::fallback::check_gpu_fallback_opt_in(...)`.

## Parity contract

`parity_ops = []`.

Numerical contracts:

- **Samples are non-negative**: per inverse-CDF formula. Test
  `test_weibull_sample_shape` pins.
- **`log_prob(x < 0) = -inf`**: support boundary. Test
  `test_weibull_log_prob_negative`.
- **`entropy(k=1, lambda=1) = 1`**: `H = 0*euler + log(1) + 1 = 1`.
  Test `test_weibull_entropy`.
- **`cdf(lambda; lambda, k) = 1 - exp(-1)`**: at `x = scale`, the
  Weibull CDF is `1 - exp(-1) ≈ 0.632` for any `k`. Test
  `test_weibull_cdf_at_scale_is_one_minus_e_inv`.
- **`cdf`/`icdf` roundtrip**: `icdf(p) → x; cdf(x) → p` for
  `p in {0.1, 0.3, 0.7, 0.9}` within 1e-6. Test
  `test_weibull_cdf_icdf_roundtrip`.
- **`mean(k=1) = lambda`**: `mean = lambda * Gamma(2) = lambda * 1 = lambda`.
  Test `test_weibull_mean_k_one_equals_lambda`.
- **`mode(k < 1) = 0`**: R-DEV-6 deviation from upstream NaN. Test
  `test_weibull_mode_k_below_one_is_zero`.
- **`variance(k=1) = lambda^2`**: `Var = lambda^2 * (Gamma(3) - Gamma(2)^2) = lambda^2 * (2 - 1) = lambda^2`.
  Test `test_weibull_variance_k_one_equals_lambda_sq`.

## Verification

Tests in `mod tests in weibull.rs` (7 tests):

- `test_weibull_sample_shape`
- `test_weibull_log_prob_negative`
- `test_weibull_entropy`
- `test_weibull_cdf_at_scale_is_one_minus_e_inv`
- `test_weibull_cdf_icdf_roundtrip`
- `test_weibull_mean_k_one_equals_lambda`
- `test_weibull_mode_k_below_one_is_zero`
- `test_weibull_variance_k_one_equals_lambda_sq`

Smoke command:

```bash
cargo test -p ferrotorch-distributions --lib weibull:: 2>&1 | tail -3
```

Expected: `8 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct Weibull<T: Float>` with `scale`, `concentration` fields in `weibull.rs`, mirroring `torch/distributions/weibull.py:40-56`; non-test consumer: `pub use weibull::Weibull` in `lib.rs:124` — grandfathered public API; downstream reliability/survival-analysis code constructs it directly. |
| REQ-2 | SHIPPED | impl: `pub fn Weibull::new(scale, concentration) -> FerrotorchResult<Self>` with shape-match validation in `weibull.rs`; non-test consumer: registered in `tests/conformance/_surface_inventory.toml:511`; `pub use Weibull` re-export. |
| REQ-3 | SHIPPED | impl: `pub fn scale(&self) -> &Tensor<T>` and `pub fn concentration(&self) -> &Tensor<T>` accessors in `weibull.rs`, mirroring `Weibull.scale` / `Weibull.concentration` attribute access; non-test consumer: `pub use Weibull` re-export exposes both. |
| REQ-4 | SHIPPED | impl: `Distribution::sample` in `weibull.rs` via inverse-CDF `scale * (-log(1-u))^(1/k)`, mirroring the `TransformedDistribution(Exponential, [Power(1/k), Affine(scale)])` chain in `weibull.py:48-56`; non-test consumer: `pub use Weibull` re-export plus impl dispatch; test `test_weibull_sample_shape` pins non-negativity. |
| REQ-5 | SHIPPED | impl: `Distribution::log_prob` in `weibull.rs` returns `log(k/lambda) + (k-1)*log(x/lambda) - (x/lambda)^k` for `x >= 0` else `-inf`; non-test consumer: `pub use Weibull` re-export + impl dispatch; test `test_weibull_log_prob_negative` pins the support boundary. |
| REQ-6 | SHIPPED | impl: `Distribution::entropy` in `weibull.rs` returns `euler_gamma * (1 - 1/k) + log(lambda/k) + 1`, mirroring `weibull.py:91-96`; non-test consumer: `pub use Weibull` re-export; test `test_weibull_entropy` pins `k=1, lambda=1 → 1.0`. |
| REQ-7 | SHIPPED | impl: `Distribution::cdf` in `weibull.rs` returns `1 - exp(-(x/lambda)^k)` for `x >= 0` else `0`; R-DEV-7 enhancement (upstream does NOT override `cdf` directly in `Weibull`); non-test consumer: `pub use Weibull` re-export; tests `test_weibull_cdf_at_scale_is_one_minus_e_inv` and `test_weibull_cdf_icdf_roundtrip` pin. |
| REQ-8 | SHIPPED | impl: `Distribution::icdf` in `weibull.rs` returns `lambda * (-log(1-p))^(1/k)`; R-DEV-7 enhancement; non-test consumer: `pub use Weibull` re-export; test `test_weibull_cdf_icdf_roundtrip` pins the roundtrip. |
| REQ-9 | SHIPPED | impl: `Distribution::mean` in `weibull.rs` returns `lambda * exp(lgamma(1 + 1/k))`, mirroring `weibull.py:72-74`; non-test consumer: `pub use Weibull` re-export; test `test_weibull_mean_k_one_equals_lambda` pins. Uses `lgamma_scalar` from `special_fns.rs`. |
| REQ-10 | SHIPPED | impl: `Distribution::mode` in `weibull.rs` returns `lambda * ((k-1)/k)^(1/k)` for `k > 1` else `0`; R-DEV-6 deviation from upstream `weibull.py:76-82` which gives NaN for `k <= 1`; non-test consumer: `pub use Weibull` re-export; test `test_weibull_mode_k_below_one_is_zero` pins. |
| REQ-11 | SHIPPED | impl: `Distribution::variance` in `weibull.rs` returns `lambda^2 * (exp(lgamma(1+2/k)) - exp(lgamma(1+1/k))^2)`, algebraically equivalent to `weibull.py:85-89`'s `scale^2 * (exp(lgamma(1+2/k)) - exp(2*lgamma(1+1/k)))`; non-test consumer: `pub use Weibull` re-export; test `test_weibull_variance_k_one_equals_lambda_sq` pins. |
| REQ-12 | NOT-STARTED | blocker #1435 — `rsample` returns `InvalidArgument`; upstream `Weibull.rsample` works via the `TransformedDistribution` chain (`weibull.py:48-56` `PowerTransform` + `AffineTransform` both autograd-aware), but ferrotorch's direct inverse-CDF path does not build the autograd graph. |
| REQ-13 | SHIPPED | impl: `has_rsample`(=false; tracked under #1435) / `batch_shape` / `support`(`Positive` per `weibull.py:38`) / `arg_constraints`(`{scale: Positive, concentration: Positive}` per `weibull.py:33-37`) / `expand` overrides at the tail of `impl Distribution for Weibull` in `weibull.rs`; non-test consumer: trait dispatch through `pub use Weibull` re-export at `lib.rs:124`; `test_weibull_surface_overrides` and `test_weibull_expand` pin the overrides. Closes #1436 — `concentration_reciprocal` Python-only convenience attribute is intentionally omitted (recomputing `1/k` per call is negligible on CPU). |
