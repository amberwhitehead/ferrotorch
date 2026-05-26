# ferrotorch-distributions — `normal` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/normal.py
-->

## Summary

`ferrotorch-distributions/src/normal.rs` defines `Normal<T>` — the
univariate Gaussian parameterised by `loc` (mean) and `scale`
(standard deviation). Mirrors `torch.distributions.Normal`
(`torch/distributions/normal.py:15-122`). Supports reparameterized
sampling and exposes the full Distribution-trait surface
(`sample`/`rsample`/`log_prob`/`entropy`/`cdf`/`icdf`/`mean`/`mode`/
`variance`/`stddev`).

## Requirements

- REQ-1: `pub struct Normal<T: Float>` holding `loc: Tensor<T>`
  and `scale: Tensor<T>`. Mirrors upstream's `loc`/`scale`
  attributes at `normal.py:55-66`.

- REQ-2: `pub fn Normal::new(loc, scale) -> FerrotorchResult<Self>`
  with shape-equality precondition + `ShapeMismatch` rejection.
  Mirrors upstream's `broadcast_all(loc, scale)` at `normal.py:61`.

- REQ-3: `loc()` and `scale()` accessors. Mirror upstream's
  property access.

- REQ-4: `impl<T: Float> Distribution<T> for Normal<T>` with the
  10 trait methods (`batch_shape`, `sample`, `rsample`, `log_prob`,
  `entropy`, `cdf`, `icdf`, `mean`, `mode`, `variance`, `stddev`).
  Mirrors the property + method surface at `normal.py:39-114`.

- REQ-5: `batch_shape` override returning `self.loc.shape().to_vec()`.
  Required by the `Independent` wrapper so its
  `reinterpreted_batch_ndims` logic can identify the rightmost
  dims of the Normal's batch. Mirrors upstream's
  `batch_shape = self.loc.size()` at `normal.py:62-66`.

- REQ-6: Sampling via the reparameterization trick —
  `z = loc + scale * eps` where `eps ~ N(0, 1)`. Cyclic
  broadcasting for scalar parameters against shape. Mirrors
  upstream `normal.py:82-85`.

- REQ-7: `rsample` attaches `NormalRsampleBackward` for autograd
  flow through `loc` and `scale` with `d(z)/d(loc) = 1` and
  `d(z)/d(scale) = eps`. Mirrors upstream's autograd-traced
  `loc + eps * scale`.

- REQ-8: `log_prob` is grad-aware via `NormalLogProbBackward` —
  attaches when any of `loc`, `scale`, `value` has `requires_grad`.
  Backward computes:
  - `d(lp)/d(loc) = (x - loc) / scale^2`
  - `d(lp)/d(scale) = ((x - loc)^2 / scale^3) - 1 / scale`
  - `d(lp)/d(value) = -(x - loc) / scale^2`

  per the standard analytic derivative of the Gaussian log-density.
  Mirrors upstream's autograd-traced `log_prob` body at
  `normal.py:87-101`.

- REQ-9: `cdf` via `0.5 * (1 + erf((x - loc) / (scale * sqrt(2))))`,
  dispatching to `ferrotorch_core::special::erf`. Mirrors
  `normal.py:103-108`.

- REQ-10: `icdf` via `loc + scale * sqrt(2) * erfinv(2*p - 1)`,
  dispatching to `ferrotorch_core::special::erfinv`. Mirrors
  `normal.py:110-111`.

- REQ-11: Closed-form properties — `mean = loc`, `mode = loc`,
  `variance = scale^2`, `stddev = scale`, `entropy =
  0.5 + 0.5*ln(2*pi) + ln(scale)`. Mirror
  `normal.py:39-53, 113-114`.

- REQ-12: Device-resident outputs — every method ends with
  `out.to(device)` if `loc.is_cuda()`. Matches upstream's implicit
  device routing.

- REQ-13: NOT-STARTED — upstream's exponential-family interface
  (`_natural_params`, `_log_normalizer`, `_mean_carrier_measure`
  from `normal.py:116-122`) is not implemented. ferrotorch's
  `Normal` is a plain `Distribution`, not an `ExponentialFamily`.
  Blocker #1404 tracks the exponential-family wiring.

## Acceptance Criteria

- [x] AC-1: `pub struct Normal<T: Float>` with `loc`/`scale`.
- [x] AC-2: `pub fn Normal::new` rejects shape-mismatched input.
- [x] AC-3: `loc()`/`scale()` accessors.
- [x] AC-4: `impl Distribution<T> for Normal<T>` with the 10
  methods.
- [x] AC-5: `test_normal_log_prob_standard` validates
  `log_prob(0) == -0.5*log(2*pi)` for N(0, 1).
- [x] AC-6: `test_normal_log_prob_batch` validates symmetry and
  monotonicity.
- [x] AC-7: `test_normal_entropy` validates
  `H == 0.5*ln(2*pi*e*sigma^2)`.
- [x] AC-8: `test_normal_rsample_backward` validates
  `d(sum(z))/d(loc) = n`.
- [x] AC-9: `test_normal_cdf_at_mean_is_half` and
  `test_normal_cdf_icdf_roundtrip` validate CDF/ICDF.
- [x] AC-10: `test_normal_mean_mode_variance_stddev` validates
  the property identities.
- [ ] AC-11: ExponentialFamily interface — blocker #1404.

## Architecture

### The struct (REQ-1, REQ-2, REQ-3)

```rust
pub struct Normal<T: Float> {
    loc: Tensor<T>,
    scale: Tensor<T>,
}
```

Defined at `normal.rs:26-29`. Constructor at `normal.rs:37-48`.
Accessors at `normal.rs:50-58`.

### The Distribution impl (REQ-4, REQ-5, REQ-6, REQ-9, REQ-10, REQ-11, REQ-12)

`batch_shape` (`normal.rs:62-65`) returns `loc.shape().to_vec()`.

`sample` (`normal.rs:67-87`) draws `eps ~ N(0, 1)` then computes
`loc + scale * eps` element-wise with cyclic broadcast.

`rsample` (`normal.rs:89-122`) follows the same closed form and
attaches `NormalRsampleBackward` when either parameter has
`requires_grad`.

`log_prob` (`normal.rs:124-167`) computes
`-(half * z^2) - ln(scale) - half * ln(2*pi)` where
`z = (x - loc) / scale`, and attaches `NormalLogProbBackward`
when any input has `requires_grad`.

`entropy` (`normal.rs:169-193`) returns
`0.5 + 0.5 * ln(2*pi) + ln(scale)`. Note that this is
algebraically equivalent to `0.5 * ln(2*pi*e*scale^2)`.

`cdf` (`normal.rs:195-219`) and `icdf` (`normal.rs:221-241`)
dispatch to `ferrotorch_core::special::{erf, erfinv}`.

`mean` / `mode` (`normal.rs:243-249`) clone `loc`.

`variance` (`normal.rs:251-256`) returns `scale^2`.

`stddev` (`normal.rs:258-260`) clones `scale`.

### NormalRsampleBackward (REQ-7)

Defined at `normal.rs:272-337`. Holds `loc`, `scale`, saved `eps`.
- `grad_loc = sum(grad_output)` — `d(z)/d(loc) = 1`.
- `grad_scale = sum(grad_output * eps)` — `d(z)/d(scale) = eps`.

### NormalLogProbBackward (REQ-8)

Defined at `normal.rs:347-452`. Holds `loc`, `scale`, `value`. The
3 partial derivatives are computed elementwise then folded across
batch dims for the scalar parameters.

### Non-test production consumers

- `pub use normal::Normal` at `lib.rs:114` — grandfathered
  public API. Downstream code (VAE encoders, BNN priors,
  diffusion models) constructs `Normal::new(loc, scale)?`.
- **`mixture_same_family.rs:233-273` tests** — these are TEST-side
  but the production-side consumer is `pub use Normal` plus the
  generic `D: Distribution<T>` bound in `MixtureSameFamily<T, D>`
  which the compiler instantiates with `D = Normal<T>` whenever
  external code constructs `MixtureSameFamily<T, Normal<T>>`. This
  is the production composition path.
- `NormalRsampleBackward` and `NormalLogProbBackward` are consumed
  by the autograd engine via `Tensor::from_operation` — production
  consumers of `GradFn<T>` from `ferrotorch_core::tensor`.
- `ferrotorch_core::special::{erf, erfinv}` are production
  consumers of `ferrotorch-core`'s special-functions surface —
  invoked from `cdf` and `icdf`.

## Parity contract

`parity_ops = []`. The `normal_native` parity oracle would test the
analytic Gaussian; ferrotorch composes against `erf`/`erfinv`
parity oracles instead. Edge cases preserved:

- **`scale -> 0+`** — `log_prob -> +infinity` at `loc`,
  `-infinity` elsewhere; `entropy -> -infinity`. Upstream
  semantics identical via `torch.log`.
- **`log_prob(loc) = -0.5 * log(2*pi*scale^2)`** — test
  `test_normal_log_prob_standard` pins `log_prob(0) =
  -0.5*log(2*pi)` for N(0, 1).
- **Symmetry** — `log_prob(loc + d) == log_prob(loc - d)`. Test
  `test_normal_log_prob_batch` pins this.
- **CDF at mean** — exactly 0.5. Test
  `test_normal_cdf_at_mean_is_half` pins this.
- **CDF/ICDF roundtrip** — `cdf(icdf(p)) ≈ p` to `5e-3`. Test
  `test_normal_cdf_icdf_roundtrip` pins this (tolerance is loose
  because the `erfinv` Newton iteration has limited accuracy).
- **Backward through `loc`** — `d(sum(z_i))/d(loc) = n` (one per
  sample, summed). Test `test_normal_rsample_backward` pins
  `n=10`.

## Verification

Tests in `mod tests in normal.rs` (15 tests):

- `test_normal_sample_shape`,
- `test_normal_sample_2d_shape`,
- `test_normal_rsample_has_grad`,
- `test_normal_rsample_no_grad_when_params_detached`,
- `test_normal_log_prob_standard`,
- `test_normal_log_prob_nonzero_mean`,
- `test_normal_log_prob_batch`,
- `test_normal_entropy`,
- `test_normal_entropy_unit_variance`,
- `test_normal_shape_mismatch`,
- `test_normal_rsample_backward`,
- `test_normal_f64`,
- `test_normal_mean_mode_variance_stddev`,
- `test_normal_cdf_at_mean_is_half`,
- `test_normal_cdf_icdf_roundtrip`.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-distributions --lib normal:: 2>&1 | tail -3
```

Expected: `15 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct Normal<T: Float>` with `loc`/`scale` fields at `normal.rs:26-29`, mirroring `torch/distributions/normal.py:15-66`; non-test consumer: `pub use normal::Normal` at `lib.rs:114`. |
| REQ-2 | SHIPPED | impl: the constructor at `normal.rs:37-48` with shape-equality precondition, mirroring `normal.py:55-66`; non-test consumer: re-export at `lib.rs:114` exposes `::new`. |
| REQ-3 | SHIPPED | impl: `loc()`/`scale()` accessors at `normal.rs:50-58`; non-test consumer: re-export at `lib.rs:114`. |
| REQ-4 | SHIPPED | impl: `impl<T: Float> Distribution<T> for Normal<T>` at `normal.rs:61-261` with 10 methods; non-test consumer: re-export at `lib.rs:114` + generic instantiation `MixtureSameFamily<T, Normal<T>>` in downstream composition. 15 tests pin behaviour. |
| REQ-5 | SHIPPED | impl: `batch_shape` override at `normal.rs:62-65` returning `self.loc.shape().to_vec()`, mirroring `normal.py:62-66`; non-test consumer: `independent.rs` (the `Independent` wrapper) reads `base.batch_shape()` to compute its `event_dims` forwarding. |
| REQ-6 | SHIPPED | impl: `loc + scale * eps` body in `sample` at `normal.rs:79`, mirroring `normal.py:82-85`; non-test consumer: re-export at `lib.rs:114`. |
| REQ-7 | SHIPPED | impl: `NormalRsampleBackward` at `normal.rs:272-337` attached via `Tensor::from_operation` at `normal.rs:108-113`; non-test consumer: autograd engine traversal in `ferrotorch_core::tensor` invokes `backward()` on this `GradFn<T>`. Test `test_normal_rsample_backward` pins. |
| REQ-8 | SHIPPED | impl: `NormalLogProbBackward` at `normal.rs:347-452` attached via `Tensor::from_operation` at `normal.rs:152-158`; non-test consumer: autograd engine in `ferrotorch_core::tensor` traverses this GradFn on `backward()` of any `log_prob` with grad-requiring inputs. |
| REQ-9 | SHIPPED | impl: `cdf` body at `normal.rs:195-219` dispatching to `ferrotorch_core::special::erf` at `normal.rs:215`, mirroring `normal.py:103-108`; non-test consumer: re-export at `lib.rs:114` + `Distribution::cdf` external invocation. Test `test_normal_cdf_at_mean_is_half` pins. |
| REQ-10 | SHIPPED | impl: `icdf` body at `normal.rs:221-241` dispatching to `ferrotorch_core::special::erfinv` at `normal.rs:229`, mirroring `normal.py:110-111`; non-test consumer: re-export at `lib.rs:114` + `Distribution::icdf` external invocation. Test `test_normal_cdf_icdf_roundtrip` pins. |
| REQ-11 | SHIPPED | impl: `mean`/`mode`/`variance`/`stddev`/`entropy` at `normal.rs:243-261` and `normal.rs:169-193`, mirroring `normal.py:39-53, 113-114`; non-test consumer: re-export at `lib.rs:114` + `Distribution` trait surface external invocation. |
| REQ-12 | SHIPPED | impl: `out.to(device)` at the tail of every method (e.g. `normal.rs:82-86`); non-test consumer: every external caller receives device-correct tensors. |
| REQ-13 | NOT-STARTED | blocker #1404 — `ExponentialFamily` interface (`_natural_params`/`_log_normalizer`/`_mean_carrier_measure` from `normal.py:116-122`) not implemented; ferrotorch's `Normal` is a plain `Distribution`. |
