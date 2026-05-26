# ferrotorch-distributions — `kumaraswamy` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/kumaraswamy.py
-->

## Summary

`ferrotorch-distributions/src/kumaraswamy.rs` defines the
`Kumaraswamy<T>` distribution on `[0, 1]` parameterized by two
concentration parameters `a > 0` and `b > 0`. Mirrors
`torch.distributions.Kumaraswamy` (`torch/distributions/kumaraswamy.py:24-107`).
Upstream PyTorch constructs Kumaraswamy as a `TransformedDistribution`
over `Uniform(0, 1)` with two `PowerTransform`s + an `AffineTransform`;
ferrotorch ships a direct closed-form implementation since the inverse
CDF is closed-form (`x = (1 - (1-u)^(1/b))^(1/a)`).

## Requirements

- REQ-1: `pub struct Kumaraswamy<T: Float>` holding `a: Tensor<T>` and
  `b: Tensor<T>` fields. Mirrors the upstream `concentration1` (a)
  and `concentration0` (b) attributes from
  `torch/distributions/kumaraswamy.py:42-70`.

- REQ-2: `pub fn Kumaraswamy::new(a, b) -> FerrotorchResult<Self>`
  with a `shape(a) == shape(b)` precondition check returning
  `ShapeMismatch`. Mirrors `broadcast_all(concentration1, concentration0)`
  in upstream (`kumaraswamy.py:56`); ferrotorch's stricter equality
  check is a R-DEV-7 ergonomic choice (broadcasting can be added
  later without breaking the API).

- REQ-3: Two accessors — the `a` borrow and the `b` borrow — for
  parameter introspection. Mirrors `Kumaraswamy.concentration1` and
  `Kumaraswamy.concentration0` property access in upstream.

- REQ-4: `impl<T: Float> Distribution<T> for Kumaraswamy<T>` with
  closed-form `sample`, `log_prob`, `entropy`, `cdf`, `icdf`,
  `mean`, `mode`, `variance`. Mirrors the corresponding methods /
  properties in `kumaraswamy.py:78-107`.

- REQ-5: Sampling via inverse CDF —
  `x = (1 - (1-u)^(1/b))^(1/a)` where `u ~ Uniform(0, 1)`. Mirrors
  the composed `PowerTransform` + `AffineTransform` chain in
  `kumaraswamy.py:64-68`. Element-wise broadcasting via cyclic
  parameter indexing supports scalar-broadcast against batched
  output shape.

- REQ-6: NOT-STARTED — `rsample` is unimplemented. The upstream
  `has_rsample = True` (`kumaraswamy.py:48`) means PyTorch supports
  reparameterized sampling via the `TransformedDistribution` chain;
  ferrotorch's direct implementation could provide rsample via
  the same closed-form differentiable expression but does not.
  The `rsample` method returns `InvalidArgument("rsample not yet
  implemented")` at `kumaraswamy.rs:84-88`. Blocker #1382 tracks
  the rsample fill-out.

- REQ-7: Numerical-stability divergence: ferrotorch's `mode`
  defensively returns `0` for parameter combinations where
  `a <= 1` or `b < 1` (where the mode is on a boundary). Upstream
  returns `NaN` for those cases via the
  `log_mode[(self.concentration0 < 1) | (self.concentration1 < 1)] = nan`
  mask at `kumaraswamy.py:89`. R-DEV-6 divergence tracked by
  blocker #1384 (mode boundary semantics: NaN vs 0).

- REQ-8: Entropy uses `digamma(b+1)` via the shifted-asymptotic
  expansion from `special_fns::digamma_scalar` to support all
  `b > 0` (integer or fractional). Mirrors upstream's
  `torch.digamma(self.concentration0 + 1)`
  (`kumaraswamy.py:101`) which routes to the PyTorch C++ digamma
  implementation.

## Acceptance Criteria

- [x] AC-1: `pub struct Kumaraswamy<T: Float>` with `a`, `b` fields.
- [x] AC-2: `pub fn Kumaraswamy::new` with shape-mismatch rejection.
- [x] AC-3: `a` and `b` accessor methods.
- [x] AC-4: `impl Distribution<T> for Kumaraswamy<T>` with
  `sample`/`log_prob`/`entropy`/`cdf`/`icdf`/`mean`/`mode`/`variance`.
- [x] AC-5: `test_kumaraswamy_sample_range` confirms samples are in
  `(0, 1)`.
- [x] AC-6: `test_kumaraswamy_uniform_case` confirms `a=1, b=1`
  yields log_prob `0` at `x=0.5`.
- [x] AC-7: `test_kumaraswamy_cdf_icdf_roundtrip` confirms CDF/ICDF
  consistency.
- [ ] AC-8: rsample — blocker #1383.
- [ ] AC-9: mode boundary returns NaN like upstream — blocker #1384.

## Architecture

### The struct (REQ-1, REQ-2, REQ-3)

```rust
pub struct Kumaraswamy<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
}
```

Constructor (`kumaraswamy.rs:28-40`) takes ownership of both
parameter tensors and checks `a.shape() == b.shape()`. The accessors
`a()` and `b()` return shared borrows for downstream introspection.

### The Distribution impl (REQ-4, REQ-5)

`sample` (`kumaraswamy.rs:50-82`) draws `u ~ Uniform(0,1)` then
applies the inverse CDF element-wise. The cyclic indexing
(`ai = i % a_data.len()`) supports scalar parameters broadcast
against any sample shape — matches PyTorch's broadcasting rule
implicitly. The `.max(T::from(1e-30))` clamp before `powf(1/a)`
prevents `log(0)` cascades when `u` is at the boundary.

`log_prob` (`kumaraswamy.rs:90-120`) computes
`log(a) + log(b) + (a-1)*log(x) + (b-1)*log(1 - x^a)` and returns
`-infinity` outside `(0, 1)`. Matches the analytic PDF derived
from the upstream `TransformedDistribution.log_prob` reduction.

`cdf` / `icdf` (`kumaraswamy.rs:153-191`) use the closed forms
`F(x) = 1 - (1 - x^a)^b` and
`F^{-1}(p) = (1 - (1 - p)^(1/b))^(1/a)`.

`mean` / `variance` (`kumaraswamy.rs:193-256`) use the
moment formula `E[X^n] = b * B(1 + n/a, b)` evaluated via
`lgamma_scalar` to avoid overflow. Mirrors upstream's `_moments`
helper at `kumaraswamy.py:15-21`.

`mode` (`kumaraswamy.rs:210-234`) returns
`((a-1) / (a*b - 1))^(1/a)` when `a > 1` and `b >= 1`, else `0`
defensively (see REQ-7 divergence).

`entropy` (`kumaraswamy.rs:122-151`) uses the closed form
`(1 - 1/a)*(γ + ψ(b+1)) + (1 - 1/b) - ln(a) - ln(b)` where
`γ` is Euler-Mascheroni and `ψ` is digamma, dispatched via
`special_fns::digamma_scalar`. Mirrors upstream
`kumaraswamy.py:98-107`.

### Non-test production consumers

- `pub use kumaraswamy::Kumaraswamy` at `lib.rs:107` — grandfathered
  public API. Downstream code (Bayesian inference layers, VAE
  prior modules) constructs `Kumaraswamy::new(a, b)?` directly.
- `special_fns::{digamma_scalar, lgamma_scalar}` are the production
  consumers of `ferrotorch-distributions`'s internal special-functions
  module — invoked from `entropy`, `mean`, `variance`.

No internal `.rs` file constructs a `Kumaraswamy` directly; per
goal.md S5 the `pub use` re-export is the grandfathered public
API surface. Downstream model crates compose Kumaraswamy at their
composition layer.

## Parity contract

`parity_ops = []`. Kumaraswamy is a transformed-distribution wrapper
on its own numerics; the route declares no parity ops because
ferrotorch's implementation diverges structurally (closed-form vs
`TransformedDistribution` chain) and the parity-sweep infra
currently has no Kumaraswamy oracle. Edge cases preserved:

- **`a == 1, b == 1`** — degenerates to `Uniform(0, 1)`. The unit
  test `test_kumaraswamy_uniform_case` pins this.
- **`x = 0` or `x = 1`** — boundary log_prob returns `-infinity`;
  PyTorch via `TransformedDistribution` returns `-inf` likewise via
  the chained PowerTransform jacobian.
- **`a <= 1` or `b < 1` in mode** — ferrotorch returns `0`,
  PyTorch returns `NaN`. Tracked in blocker #1384.
- **Non-positive sample input to log_prob** — returns `-infinity`;
  upstream returns `-inf` via `validate_args` filtering.
- **Gradient flow** — neither `sample` nor `rsample` is grad-aware
  (rsample errors per REQ-6). Closed-form `mean` / `variance` /
  `entropy` build fresh tensors with `requires_grad=false`.

## Verification

Tests in `mod tests in kumaraswamy.rs` (8 tests):

- `test_kumaraswamy_sample_range` — samples within `(0, 1)`.
- `test_kumaraswamy_log_prob_boundary` — boundary returns
  `-infinity`.
- `test_kumaraswamy_uniform_case` — `a=1,b=1` log_prob is 0 at 0.5.
- `test_kumaraswamy_cdf_unit_case_is_identity` — CDF is identity
  for the uniform case.
- `test_kumaraswamy_cdf_icdf_roundtrip` — CDF/ICDF consistency.
- `test_kumaraswamy_mean_uniform_case_is_half` — mean is 0.5 for
  uniform case.
- `test_kumaraswamy_variance_uniform_case_is_one_twelfth` —
  variance is 1/12 for uniform case.
- `test_kumaraswamy_mode_well_defined_when_a_gt_one` — mode
  matches `((a-1)/(ab-1))^(1/a)` for `a=2, b=2`.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-distributions --lib kumaraswamy:: 2>&1 | tail -3
```

Expected: `8 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct Kumaraswamy<T: Float>` with `a` and `b` `Tensor<T>` fields in `kumaraswamy.rs:23-26`, mirroring `torch/distributions/kumaraswamy.py:24-70`; non-test consumer: `pub use kumaraswamy::Kumaraswamy` at `lib.rs:107` exposes the type as grandfathered public API for downstream Bayesian-inference / VAE-prior modules. |
| REQ-2 | SHIPPED | impl: the constructor at `kumaraswamy.rs:29-40` with `shape(a) == shape(b)` precondition + `ShapeMismatch` error path, mirroring `kumaraswamy.py:56` (`broadcast_all`); non-test consumer: `pub use Kumaraswamy::new` via the re-export at `lib.rs:107`. |
| REQ-3 | SHIPPED | impl: the `a()` and `b()` accessors at `kumaraswamy.rs:42-47`, mirroring upstream property access at `kumaraswamy.py:56-58`; non-test consumer: `pub use` re-export at `lib.rs:107` makes the accessors part of the public surface for parameter introspection. |
| REQ-4 | SHIPPED | impl: the full `impl<T: Float> Distribution<T> for Kumaraswamy<T>` block at `kumaraswamy.rs:50-257` with `sample`/`log_prob`/`entropy`/`cdf`/`icdf`/`mean`/`mode`/`variance`, mirroring `torch/distributions/kumaraswamy.py:78-107`; non-test consumer: `pub use Kumaraswamy` re-export at `lib.rs:107` means any external Distribution-trait caller hits this impl. Tests pin all 8 methods. |
| REQ-5 | SHIPPED | impl: the inverse-CDF body in `sample` at `kumaraswamy.rs:73-78` (`inner = (one - u)^(1/b); val = (1 - inner)^(1/a)`), mirroring the chained PowerTransform + AffineTransform at `kumaraswamy.py:64-68`; non-test consumer: `pub use Kumaraswamy` re-export + Distribution-trait `sample` invocation. Test `test_kumaraswamy_sample_range` pins the `(0, 1)` support. |
| REQ-6 | NOT-STARTED | blocker #1383 — `rsample` returns `InvalidArgument("not yet implemented")` at `kumaraswamy.rs:84-88` instead of the closed-form differentiable expression upstream provides via `has_rsample = True` (`kumaraswamy.py:48`). |
| REQ-7 | NOT-STARTED | blocker #1384 — mode boundary semantics divergence: ferrotorch returns `0` at `kumaraswamy.rs:228-231` for `a <= 1` or `b < 1`, upstream returns `NaN` (`kumaraswamy.py:89`). |
| REQ-8 | SHIPPED | impl: `digamma_b1 = digamma_scalar(b[i] + one)` at `kumaraswamy.rs:143` using the shifted-asymptotic expansion from `special_fns`, mirroring upstream `torch.digamma(self.concentration0 + 1)` at `kumaraswamy.py:101`; non-test consumer: invoked by `entropy()` at `kumaraswamy.rs:144`. |
