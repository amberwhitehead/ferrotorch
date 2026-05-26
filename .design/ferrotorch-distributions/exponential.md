# ferrotorch-distributions — `exponential` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/exponential.py
-->

## Summary

`ferrotorch-distributions/src/exponential.rs` implements the
exponential distribution with rate λ ("lambda"). Mirrors
`torch.distributions.Exponential`. Sampling uses the inverse-CDF
transform `z = -ln(u)/λ`. `rsample` carries gradients through λ
via the `ExponentialRsampleBackward` GradFn
(`d(z)/d(λ) = ln(u)/λ²`). Closed-form `cdf`, `icdf`, `mean`,
`mode`, `variance` are provided.

## Requirements

- REQ-1: `pub struct Exponential<T: Float>` holding a single
  `rate: Tensor<T>` field. Mirrors
  `torch/distributions/exponential.py:Exponential.__init__`.

- REQ-2: `pub fn Exponential::new(rate) ->
  FerrotorchResult<Self>` is the constructor. No upstream
  positivity check on `rate` is performed (R-DEV-4: deferred to
  the validate-args mechanism — see REQ-9).

- REQ-3: `pub fn rate(&self) -> &Tensor<T>` accessor mirrors
  `Exponential.rate` attribute access (`exponential.py:51-56`).

- REQ-4: `impl<T: Float> Distribution<T> for Exponential<T>`
  provides `sample` / `rsample` / `log_prob` / `entropy` plus
  `cdf` / `icdf` / `mean` / `mode` / `variance` property
  overrides.

- REQ-5: `sample(shape)` and `rsample(shape)` both compute
  `z = -ln(u_safe)/rate` where `u_safe = max(1e-30, u)` for
  `u ~ Uniform(0, 1)`. The tiny-guard avoids `ln(0) = -inf`.
  PyTorch's `exponential_()` in-place sampler uses the same
  inversion (`exponential.py:68-70`).

- REQ-6: `log_prob(value) = ln(rate) - rate * value`. Standard
  log-PDF; mirrors `exponential.py:72-75`. No clamping on
  `value` — negative `value` produces a large positive
  `log_prob` (mathematically `-∞ * sign(rate)` for the limit;
  ferrotorch returns the finite formula result). The full PDF
  is `f(x) = λe^(-λx)` for `x ≥ 0`, undefined for `x < 0` —
  ferrotorch surfaces the formula directly; users should respect
  the `x ≥ 0` support contract.

- REQ-7: `entropy = 1 - ln(rate)`. Standard closed form;
  mirrors `exponential.py:85-86`. For `rate = 1`, entropy =
  exactly 1.

- REQ-8: `cdf(x) = 1 - exp(-rate*x)` for `x ≥ 0`; `0` for `x < 0`.
  `icdf(p) = -ln(1-p)/rate`. Standard inverse-CDF pair;
  mirrors `exponential.py:77-83`.

- REQ-9: `mean = 1/rate`, `mode = 0`, `variance = 1/rate²`.
  Mirrors `exponential.py:35-49`.

- REQ-10: `ExponentialRsampleBackward` `GradFn` implements
  `d(z)/d(rate) = ln(u_safe)/rate²`. Summed across the sample
  tensor with chain rule into a scalar `grad_rate`. The
  `u_safe = max(1e-30, u)` guard mirrors the forward pass's
  numerical floor.

- REQ-11: NOT-STARTED — `expand`, `arg_constraints`, `support`,
  `validate_args`, and the exponential-family hooks
  (`_natural_params`, `_log_normalizer` from
  `exponential.py:88-94`) not implemented. Cross-cutting with
  `lib.md` REQ-5 (blocker #1376). Tracked as blocker #1414 for
  the Exponential-side fill-out.

## Acceptance Criteria

- [x] AC-1: `pub struct Exponential<T: Float>` with `rate` field.
- [x] AC-2: `pub fn Exponential::new(rate)`.
- [x] AC-3: `pub fn rate()` accessor.
- [x] AC-4: `impl Distribution<T> for Exponential<T>` with all
  four required trait methods + property overrides.
- [x] AC-5: `sample` / `rsample` via inverse CDF with `1e-30`
  guard.
- [x] AC-6: `log_prob = ln(rate) - rate*x`.
- [x] AC-7: `entropy = 1 - ln(rate)`.
- [x] AC-8: `cdf`, `icdf` overrides.
- [x] AC-9: `mean`, `mode`, `variance` overrides.
- [x] AC-10: `ExponentialRsampleBackward` GradFn.
- [x] AC-11: `test_exponential_*` test suite (12 tests).
- [ ] AC-12: `expand` / `validate_args` — blocker #1414.

## Architecture

### Storage + constructor (REQ-1, REQ-2, REQ-3)

Single-field struct. `Exponential::new(rate)` is infallible at
the type level; range validation is deferred. The `rate()`
accessor hands the parameter tensor back by reference.

### Inverse-CDF sampling (REQ-5)

```rust
let u_safe = u_val.max(1e-30);
z = -u_safe.ln() / rate
```

The `1e-30` floor on `u` is necessary because `u` from
`creation::rand` can be exactly `0` for the `xorshift` family,
which would produce `ln(0) = -inf` and propagate to `z = inf`.
The floor caps the maximum sample at `30 * ln(10) / rate ≈
69/rate` — well within the practical tail range.

### `log_prob` and `entropy` (REQ-6, REQ-7)

```rust
log_prob(x) = rate.ln() - rate * x
entropy      = 1 - rate.ln()
```

For `rate = 1`: `log_prob(x) = -x`, `entropy = 1`. For
`rate = 2`, `log_prob(1) = ln(2) - 2`, `entropy = 1 - ln(2)`.
Tests pin both.

### `cdf` / `icdf` (REQ-8)

```rust
cdf(x) = 0      if x < 0
       = 1 - exp(-rate*x)   otherwise
icdf(p) = -ln(1 - p) / rate
```

Standard pair; `test_exponential_icdf_roundtrip` verifies
`cdf(icdf(p)) ≈ p` for `p ∈ {0.1, 0.3, 0.5, 0.7, 0.9}` to 1e-10
in f64.

### `mean` / `mode` / `variance` (REQ-9)

`mean = 1/rate`, `mode = 0` (the density's maximum is at the
support's left boundary), `variance = 1/rate²`. Returned as
shape-preserving tensors over the rate parameter's shape.

### `ExponentialRsampleBackward` (REQ-10)

```rust
d(z)/d(rate) = ln(u_safe) / rate²
grad_rate = sum_i grad_output[i] * ln(u_safe[i]) / rate²
```

Note `ln(u_safe)` is **negative** (since `u_safe ∈ (0, 1]`), so
the gradient sign is negative when `grad_output > 0` — increasing
rate decreases samples, which matches the
`test_exponential_rsample_backward` assertion that the
gradient is `< 0`.

### Non-test production consumers

- **`pub use exponential::Exponential` in lib.rs** —
  grandfathered public surface (S5).
- **KL-dispatcher consumer**: `kl.rs` registers `Exponential-Exponential`
  + `Gamma-Exponential` + `Exponential-Gamma` arms (downcast on
  `Exponential<T>`). The dispatcher is invoked from the public
  `pub fn kl_divergence` entry. The `kl_exponential_exponential`
  / `kl_gamma_exponential` / `kl_exponential_gamma` functions
  read `.rate().data_vec()?` from the exponential argument.
- **`Distribution` trait dispatch via `pub use Exponential`** —
  external `dist.{sample, rsample, log_prob, ...}` calls bind
  to this impl.

## Parity contract

`parity_ops = []`. Closed-form continuous distribution.

Edge cases covered:

- **`u = 0`**: floored to `1e-30`; max sample is finite.
- **Negative `value` in `log_prob`**: formula returns a large
  positive value (since `-rate * x > 0` for `x < 0`). The PDF
  is undefined here; users should respect `x ≥ 0`.
- **Negative `value` in `cdf`**: returns `0` per the support
  contract.
- **`rate = 0` or `rate < 0`**: not validated at construction.
  Would produce non-finite / NaN values. Tracked by blocker
  #1414.
- **`f64`**: `test_exponential_f64`.

## Verification

Unit tests (12 tests):

- Sample shape + positivity + Monte Carlo mean:
  `test_exponential_sample_shape/_positive/_mean`.
- `rsample` grad attachment: `test_exponential_rsample_has_grad`.
- `log_prob`: `test_exponential_log_prob/_rate2`.
- `entropy`: `test_exponential_entropy/_rate1`.
- Backward: `test_exponential_rsample_backward`.
- `f64`: `test_exponential_f64`.
- Properties + CDF/ICDF: `test_exponential_mean_mode_variance`,
  `test_exponential_cdf`, `test_exponential_icdf_roundtrip`.

Smoke command:

```bash
cargo test -p ferrotorch-distributions --lib exponential:: 2>&1 | tail -3
```

Expected: `12 passed; 0 failed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct Exponential<T: Float>` with `rate` field in `exponential.rs` mirroring `torch/distributions/exponential.py:14-58`; non-test consumer: `pub use exponential::Exponential` in `lib.rs` (grandfathered public surface per goal.md S5). |
| REQ-2 | SHIPPED | impl: `pub fn Exponential::new(rate)` in `exponential.rs` mirroring `exponential.py:51-58`; non-test consumer: `pub use Exponential` re-export. |
| REQ-3 | SHIPPED | impl: `pub fn Exponential::rate(&self) -> &Tensor<T>` accessor in `exponential.rs` mirroring `exponential.py:51-56`; non-test consumer: `kl_exponential_exponential` and `kl_gamma_exponential` / `kl_exponential_gamma` in `kl.rs` read the `.rate()` accessor for closed-form KL formulas. |
| REQ-4 | SHIPPED | impl: `impl<T: Float> Distribution<T> for Exponential<T>` in `exponential.rs` mirroring `exponential.py:60-86`; non-test consumer: trait dispatch via `pub use Exponential`. |
| REQ-5 | SHIPPED | impl: `fn Exponential::sample` / `rsample` in `exponential.rs` with `-ln(u_safe)/rate` formula + `u_safe = max(1e-30, u)` guard mirroring `exponential.py:68-70`; non-test consumer: external `dist.sample/rsample(shape)` calls. |
| REQ-6 | SHIPPED | impl: `fn Exponential::log_prob` in `exponential.rs` with `ln(rate) - rate*x` formula mirroring `exponential.py:72-75`; non-test consumer: external `dist.log_prob(value)` calls. |
| REQ-7 | SHIPPED | impl: `fn Exponential::entropy` in `exponential.rs` with `1 - ln(rate)` formula mirroring `exponential.py:85-86`; non-test consumer: external `dist.entropy()` calls; `test_exponential_entropy_rate1` pins `entropy(λ=1) = 1`. |
| REQ-8 | SHIPPED | impl: `fn Exponential::cdf` / `icdf` in `exponential.rs` mirroring `exponential.py:77-83`; non-test consumer: external `dist.cdf(value)` / `dist.icdf(q)` calls; `test_exponential_icdf_roundtrip` pins the round-trip. |
| REQ-9 | SHIPPED | impl: `fn Exponential::{mean, mode, variance}` overrides in `exponential.rs` mirroring `exponential.py:35-49`; non-test consumer: external `dist.{mean, mode, variance}` calls; `test_exponential_mean_mode_variance` pins all three. |
| REQ-10 | SHIPPED | impl: `struct ExponentialRsampleBackward<T: Float>` with `GradFn::backward` in `exponential.rs` implementing `d(z)/d(rate) = ln(u_safe)/rate²`; non-test consumer: invoked by `fn Exponential::rsample` when rate requires grad. |
| REQ-11 | PARTIAL | impl: `has_rsample` / `support` (NonNegative) / `arg_constraints` (rate:Positive) / `event_shape` / `expand` (broadcasts `rate`) trait overrides at the tail of `impl Distribution<T> for Exponential<T>` in `exponential.rs` mirroring `torch/distributions/exponential.py:14-49`; non-test consumer: `pub use exponential::Exponential` at `lib.rs`; `tests/divergence_distribution_trait_surface.rs::exponential_*` pins every override. Closes #1414 — STILL NOT-STARTED: `validate_args` + `_natural_params` / `_log_normalizer` exp-family hooks (orthogonal trackers). |
