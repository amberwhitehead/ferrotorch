# ferrotorch-distributions — `cauchy` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/cauchy.py
-->

## Summary

`ferrotorch-distributions/src/cauchy.rs` implements the Cauchy
(Lorentz) distribution with location `loc` and scale `scale`.
Mirrors `torch.distributions.Cauchy`. Sampling uses the
inverse-CDF transform `z = loc + scale * tan(π*(u - 0.5))` where
`u ~ Uniform(0,1)`. The distribution has no defined mean
(returns NaN) and no defined variance (returns +∞), reflecting
its heavy tails. `rsample` carries gradients through `loc` and
`scale` via the `CauchyRsampleBackward` GradFn.

## Requirements

- REQ-1: `pub struct Cauchy<T: Float>` holding `loc: Tensor<T>`
  and `scale: Tensor<T>`. Mirrors
  `torch/distributions/cauchy.py:Cauchy.__init__`.

- REQ-2: `pub fn Cauchy::new(loc, scale) -> FerrotorchResult<Self>`
  with a shape-equality check between `loc` and `scale`. Returns
  `ShapeMismatch` otherwise. PyTorch broadcasts inside
  `__init__` via `broadcast_all`; ferrotorch requires pre-broadcasted
  parameters (R-DEV-4).

- REQ-3: `pub fn loc(&self) -> &Tensor<T>`, `pub fn scale(&self) ->
  &Tensor<T>`, `pub fn median(&self) -> &Tensor<T>` accessors. The
  `median` accessor returns `&self.loc` because the Cauchy
  distribution's median equals its location. Mirrors
  PyTorch's class-level docstring guarantee at
  `cauchy.py:29` ("mode or median of the distribution").

- REQ-4: `impl<T: Float> Distribution<T> for Cauchy<T>` provides
  `sample` / `rsample` / `log_prob` / `entropy` plus the property
  overrides `cdf` / `icdf` / `mean` (NaN) / `mode` /
  `variance` (∞).

- REQ-5: `sample` and `rsample` both compute `z = loc + scale *
  tan(π*(u - 0.5))` with `u` clamped to `[1e-7, 1 - 1e-7]` to
  prevent `tan(±π/2) = ±∞`. PyTorch invokes
  `self.loc.new(shape).cauchy_()` (in-place tensor sampler)
  per `cauchy.py:76-79`; ferrotorch composes via uniform draws +
  inverse CDF, which gives the same distribution.

- REQ-6: `log_prob(x) = -ln(π) - ln(scale) - ln(1 + ((x -
  loc)/scale)²)`. Standard Cauchy log-PDF. Mirrors
  `cauchy.py:81-88` (which uses `.log1p()` for the
  `ln(1 + z²)` term; the ferrotorch form computes
  `ln(1 + z*z)` directly).

- REQ-7: `entropy = ln(4π * scale)`. Standard closed form;
  mirrors `cauchy.py:98-99`.

- REQ-8: `cdf(x) = 1/2 + atan((x - loc)/scale) / π`.
  `icdf(p) = loc + scale * tan(π*(p - 1/2))`. Standard
  inverse-CDF pair; mirrors `cauchy.py:90-96`.

- REQ-9: `mean = NaN`, `variance = +∞`, `mode = loc`.
  Reflects the heavy-tailed nature: Cauchy has no first or
  second moment. Tensor-filled NaN/∞ to keep shape consistent
  with batched-parameter usage. Mirrors `cauchy.py:61-74`.

- REQ-10: `CauchyRsampleBackward` `GradFn` implements:
  ```text
  d(z)/d(loc)   = 1                       ; sum over sample dims
  d(z)/d(scale) = tan(π*(u-0.5))          ; sum, weighted by grad_output
  ```
  This is the standard reparameterization-trick gradient through
  a location-scale transform.

- REQ-11: NOT-STARTED — `expand`, `arg_constraints`, `support`,
  `validate_args` (the full PyTorch `Distribution` surface from
  `cauchy.py:34-36,51-58`) not implemented. Cross-cutting with
  `lib.md` REQ-5 (Distribution-trait blocker #1376). Tracked as
  blocker #1400 for the Cauchy-side fill-out.

## Acceptance Criteria

- [x] AC-1: `pub struct Cauchy<T: Float>` with `loc`, `scale`.
- [x] AC-2: `pub fn Cauchy::new` with shape check.
- [x] AC-3: `pub fn loc` / `scale` / `median` accessors.
- [x] AC-4: `impl Distribution<T> for Cauchy<T>` with all four
  required trait methods + property overrides.
- [x] AC-5: `sample` / `rsample` via clamped inverse CDF.
- [x] AC-6: `log_prob` closed form.
- [x] AC-7: `entropy = ln(4π*scale)`.
- [x] AC-8: `cdf`, `icdf` overrides.
- [x] AC-9: `mean` (NaN), `variance` (∞), `mode` (loc) overrides.
- [x] AC-10: `CauchyRsampleBackward` GradFn.
- [x] AC-11: `test_cauchy_sample_shape`,
  `test_cauchy_rsample_has_grad`,
  `test_cauchy_log_prob_at_loc/at_scale/symmetry/nonunit_scale`,
  `test_cauchy_entropy`, `test_cauchy_shape_mismatch`,
  `test_cauchy_rsample_backward`,
  `test_cauchy_mean_is_nan_variance_is_inf`,
  `test_cauchy_cdf_at_loc_is_half`,
  `test_cauchy_icdf_roundtrip`,
  `test_cauchy_f64` cover the contract.
- [ ] AC-12: `expand` / `validate_args` — blocker #1400.

## Architecture

### Storage + accessors (REQ-1, REQ-2, REQ-3)

Two-field struct. `Cauchy::new` enforces shape equality;
constructor errors are `FerrotorchError::ShapeMismatch` with a
descriptive message. The `median` accessor is a convenience
that returns the location by reference — PyTorch documents the
same equality (`cauchy.py:29`).

### `sample` / `rsample` via inverse CDF (REQ-5)

```rust
let u_clamped = u_val.max(1e-7).min(1.0 - 1e-7);
loc + scale * (π * (u_clamped - 0.5)).tan()
```

The `1e-7` clamp on `u` keeps the `tan` argument inside
`(-π/2, π/2)`, preventing the singularities at the open boundaries.
PyTorch's in-place `cauchy_()` sampler does the same range
restriction internally.

### `log_prob` and `entropy` (REQ-6, REQ-7)

```rust
let z = (x - loc) / scale;
log_prob = -ln(π) - ln(scale) - ln(1 + z*z)
entropy  = ln(4π * scale)
```

For `loc=0, scale=1`: `log_prob(0) = -ln(π)`, `log_prob(±1) =
-ln(π) - ln(2)`, `entropy = ln(4π) ≈ 2.531`. Tests pin all three.
PyTorch uses `.log1p()` for the `1 + z²` term to gain ULP
accuracy near `z ≈ 0`; ferrotorch computes `ln(1 + z²)`
directly. The values diverge by less than 1 ULP for typical
inputs.

### `cdf` / `icdf` (REQ-8)

`cdf(x) = 0.5 + atan((x - loc)/scale) / π`; verify with
`test_cauchy_cdf_at_loc_is_half` (`cdf(loc) = 0.5`).
`icdf(p) = loc + scale * tan(π*(p - 0.5))`; verify via
`test_cauchy_icdf_roundtrip` for `p ∈ {0.1, 0.3, 0.5, 0.7, 0.9}`.

### `mean` / `variance` / `mode` (REQ-9)

- `mean`: returns a tensor filled with `NaN`, shape matching
  `loc.shape()`. PyTorch uses `torch.full(extended_shape, nan)`
  which has the same effect.
- `variance`: returns a tensor filled with `+∞`.
- `mode`: returns `loc.clone()`. The mode of the Cauchy
  distribution is its location parameter.

### `CauchyRsampleBackward` (REQ-10)

The GradFn owns clones of `loc`, `scale`, and `u` (the
realized uniform sample tensor). Backward:

- `grad_loc = sum(grad_output)`: because `d(z)/d(loc) = 1`
  pointwise.
- `grad_scale = sum(grad_output * tan(π*(u - 0.5)))`: the
  tan-multiplied sum is `d(z)/d(scale)` integrated over the
  sample dims.

Both gradient tensors are constructed on CPU then transferred
to the parameter's device.

### Non-test production consumers

- **`pub use cauchy::Cauchy` in lib.rs** — grandfathered
  public surface (S5).
- **`Distribution` trait dispatch via `pub use Cauchy`** — every
  external `dist.sample/rsample/log_prob/entropy/...` call binds
  to this `impl` block; the closed-form `mean/mode/variance/cdf/icdf`
  overrides are exercised by any external introspection layer
  (diagnostic logs, KL approximators, sampling visualizers).

## Parity contract

`parity_ops = []`. Closed-form distribution-level math; the
parity sweep covers tensor ops.

Edge-case coverage:

- **`u ∈ {0, 1}`**: clamped to `[1e-7, 1-1e-7]` before `tan`.
- **Heavy-tail moments**: `mean = NaN`, `variance = +∞`.
  Tests `test_cauchy_mean_is_nan_variance_is_inf` pin these.
- **Symmetry**: `log_prob(loc - δ) == log_prob(loc + δ)`.
  Tested by `test_cauchy_log_prob_symmetry`.
- **CDF roundtrip**: `cdf(icdf(p)) ≈ p` to 1e-9 in f64.
- **`f64`**: dtype-generic via `T: Float`; covered by
  `test_cauchy_f64`.
- **`scale = 0`**: not validated at construction; would produce
  `±∞` in `log_prob` and `NaN` in `cdf`. Tracked by
  blocker #1400 as part of the validate-args fill-out.

## Verification

Unit tests in `mod tests` (15 tests):

- Sample shape + grad: `test_cauchy_sample_shape`,
  `test_cauchy_rsample_has_grad`.
- `log_prob` at known points + symmetry:
  `test_cauchy_log_prob_at_loc/at_scale/symmetry/nonunit_scale`.
- `entropy`: `test_cauchy_entropy`, `test_cauchy_entropy_scale2`.
- Constructor error: `test_cauchy_shape_mismatch`.
- Backward: `test_cauchy_rsample_backward`.
- Properties + CDF/ICDF: `test_cauchy_mean_is_nan_variance_is_inf`,
  `test_cauchy_cdf_at_loc_is_half`, `test_cauchy_icdf_roundtrip`.
- f64: `test_cauchy_f64`.

Smoke command:

```bash
cargo test -p ferrotorch-distributions --lib cauchy:: 2>&1 | tail -3
```

Expected: `15 passed; 0 failed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct Cauchy<T: Float>` with `loc`, `scale` fields in `cauchy.rs` mirroring `torch/distributions/cauchy.py:15-49`; non-test consumer: `pub use cauchy::Cauchy` in `lib.rs` (grandfathered public surface per goal.md S5). |
| REQ-2 | SHIPPED | impl: `pub fn Cauchy::new` in `cauchy.rs` with shape-equality check mirroring `cauchy.py:38-49`; non-test consumer: `pub use Cauchy` re-export. |
| REQ-3 | SHIPPED | impl: `pub fn Cauchy::loc/scale/median` accessors in `cauchy.rs` mirroring `cauchy.py:29` (median ≡ loc); non-test consumer: `pub use Cauchy` re-export — diagnostic/KL-style code uses these. |
| REQ-4 | SHIPPED | impl: `impl<T: Float> Distribution<T> for Cauchy<T>` in `cauchy.rs` mirroring `cauchy.py:51-99`; non-test consumer: trait dispatch through `pub use Cauchy`. |
| REQ-5 | SHIPPED | impl: `fn Cauchy::sample` / `rsample` in `cauchy.rs` with clamped inverse-CDF formula mirroring `cauchy.py:76-79`; non-test consumer: external `dist.{sample, rsample}(shape)` calls. |
| REQ-6 | SHIPPED | impl: `fn Cauchy::log_prob` in `cauchy.rs` with `-ln(π) - ln(scale) - ln(1 + z²)` formula mirroring `cauchy.py:81-88`; non-test consumer: external `dist.log_prob(value)` calls; `test_cauchy_log_prob_at_loc/at_scale/symmetry/nonunit_scale` pin the analytical values. |
| REQ-7 | SHIPPED | impl: `fn Cauchy::entropy` in `cauchy.rs` with `ln(4π * scale)` formula mirroring `cauchy.py:98-99`; non-test consumer: external `dist.entropy()` calls. |
| REQ-8 | SHIPPED | impl: `fn Cauchy::cdf` / `icdf` in `cauchy.rs` mirroring `cauchy.py:90-96`; non-test consumer: external `dist.cdf(value)` / `dist.icdf(q)` calls; `test_cauchy_cdf_at_loc_is_half` and `test_cauchy_icdf_roundtrip` pin the round-trip. |
| REQ-9 | SHIPPED | impl: `fn Cauchy::{mean, mode, variance}` overrides in `cauchy.rs` returning NaN/loc/∞ respectively, mirroring `cauchy.py:61-74`; non-test consumer: external `dist.{mean, mode, variance}` calls; `test_cauchy_mean_is_nan_variance_is_inf` pins all three. |
| REQ-10 | SHIPPED | impl: `struct CauchyRsampleBackward<T: Float>` with `GradFn::backward` in `cauchy.rs` (sum-of-grad and tan-weighted-sum); non-test consumer: invoked by `fn Cauchy::rsample` whenever either parameter requires grad; `test_cauchy_rsample_backward` pins `grad_loc = 10.0` for `rsample([10])`. |
| REQ-11 | NOT-STARTED | blocker #1400 — `expand`, `arg_constraints`, `support`, `validate_args`, scalar-broadcast `__init__` branch from `cauchy.py:34-36,38-49,51-58` not implemented. Cross-cutting with `lib.md` REQ-5 (Distribution-trait-surface blocker #1376). |
