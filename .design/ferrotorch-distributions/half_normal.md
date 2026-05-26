# ferrotorch-distributions — `half_normal` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/half_normal.py
-->

## Summary

`ferrotorch-distributions/src/half_normal.rs` implements the
half-normal distribution — the absolute value of a
`Normal(0, scale)` random variable. Supported on `[0, ∞)`.
Mirrors `torch.distributions.HalfNormal`. Sampling draws
`eps ~ N(0, 1)` then returns `scale * |eps|`. `rsample`
carries gradients through `scale` via the
`HalfNormalRsampleBackward` GradFn. Closed-form `log_prob`,
`entropy`, `mean`, `mode`, `variance`.

## Requirements

- REQ-1: `pub struct HalfNormal<T: Float>` holding a single
  `scale: Tensor<T>` field. Mirrors
  `torch/distributions/half_normal.py:HalfNormal.__init__`,
  which composes `Normal(0, scale)` with `AbsTransform`.
  ferrotorch flattens this into a direct implementation
  (R-DEV-7).

- REQ-2: `pub fn HalfNormal::new(scale) ->
  FerrotorchResult<Self>` is the constructor. No upstream
  positivity check on `scale` is performed (R-DEV-4: deferred to
  validate-args — see REQ-9).

- REQ-3: `pub fn scale(&self) -> &Tensor<T>` accessor mirrors
  `HalfNormal.scale` property (`half_normal.py:52-54`). Plus
  utility helpers `pub fn mean_value() -> FerrotorchResult<Vec<T>>`
  and `pub fn variance_value() -> FerrotorchResult<Vec<T>>`
  returning raw `Vec<T>` for the closed-form mean / variance
  values (used internally by the tensor-returning `mean` /
  `variance` overrides).

- REQ-4: `impl<T: Float> Distribution<T> for HalfNormal<T>`
  provides `sample` / `rsample` / `log_prob` / `entropy` +
  property overrides `mean` / `mode` / `variance`.

- REQ-5: `sample` / `rsample` draw `eps ~ N(0, 1)` via
  `creation::randn` then compute `scale * |eps|` per element.
  Mirrors `HalfNormal = |Normal(0, scale)|` per the class
  docstring (`half_normal.py:15-27`).

- REQ-6: `log_prob(x) = 0.5*ln(2/π) - ln(scale) - x²/(2*scale²)`
  for `x ≥ 0`; returns `-∞` for `x < 0` (outside the support).
  Mirrors PyTorch's `Normal.log_prob(value) + ln(2)` with
  `torch.where(value >= 0, log_prob, -inf)` mask
  (`half_normal.py:68-73`). The half-Normal PDF is `2 *
  Normal(0, scale).pdf(x)` for `x ≥ 0`, so the log_prob is
  `Normal.log_prob + ln(2)`. Ferrotorch's direct formula is
  the algebraic expansion.

- REQ-7: `entropy = 0.5*ln(π/2) + ln(scale) + 0.5`. Mirrors
  PyTorch's `Normal.entropy() - ln(2)`
  (`half_normal.py:83-84`). Both are equivalent algebraically.

- REQ-8: Property overrides: `mean = scale * sqrt(2/π)`,
  `mode = 0`, `variance = scale² * (1 - 2/π)`. Mirrors
  `half_normal.py:56-66`.

- REQ-9: `HalfNormalRsampleBackward` GradFn implements
  `d(z)/d(scale) = |eps|` per-element, summed across the sample
  tensor into a scalar `grad_scale`.

- REQ-10: NOT-STARTED — `expand`, `arg_constraints`, `support`,
  `validate_args`, `cdf` (would require erf), `icdf` (inverse
  erf), the `TransformedDistribution` base hooks from
  `half_normal.py:40-50, 75-81` not implemented. The trait's
  default `cdf` / `icdf` returns `InvalidArgument`. Cross-
  cutting with `lib.md` REQ-5 (blocker #1376). Tracked as
  blocker #1421 for the HalfNormal-side fill-out.

## Acceptance Criteria

- [x] AC-1: `pub struct HalfNormal<T: Float>` with `scale`
  field.
- [x] AC-2: `pub fn HalfNormal::new(scale)`.
- [x] AC-3: `pub fn scale` accessor + `mean_value` /
  `variance_value` utility helpers.
- [x] AC-4: `impl Distribution<T> for HalfNormal<T>` with all
  four required trait methods + property overrides.
- [x] AC-5: `sample` / `rsample` via `scale * |randn|`.
- [x] AC-6: `log_prob` with `x < 0 → -inf` mask.
- [x] AC-7: `entropy = 0.5*ln(π/2) + ln(scale) + 0.5`.
- [x] AC-8: `mean`, `mode`, `variance` overrides.
- [x] AC-9: `HalfNormalRsampleBackward` GradFn.
- [x] AC-10: `test_half_normal_*` test suite (13 tests).
- [ ] AC-11: `expand` / `cdf` / `icdf` / `validate_args` —
  blocker #1421.

## Architecture

### Storage + accessors (REQ-1, REQ-2, REQ-3)

Single-field struct. The two utility helpers `mean_value` and
`variance_value` return raw `Vec<T>` — they exist as a
diagnostic / probing surface separate from the
tensor-returning `mean` / `variance` trait methods (the trait
methods invoke the utility helpers internally to avoid math
duplication).

### `sample` and `rsample` (REQ-5)

```rust
let eps ~ N(0, 1) via creation::randn(shape)
out = scale * eps.abs()
```

`rsample` additionally attaches `HalfNormalRsampleBackward`
when `scale.requires_grad()` AND grad is enabled. The full
`eps` tensor (not `|eps|`) is stored on the backward node so
the backward can recompute `|eps|` if needed (currently it
just uses `eps.abs()` directly).

### `log_prob` with support mask (REQ-6)

```rust
if x < 0 { return -inf; }
0.5 * ln(2/π) - ln(scale) - x² / (2 * scale²)
```

`test_half_normal_log_prob_negative_is_neginf` pins the
support mask. PyTorch uses `torch.where(value >= 0, ..., -inf)`
which materializes a full tensor mask; ferrotorch does the
per-element branch directly inside the `.map()` body.

### `entropy` (REQ-7)

```rust
0.5 * ln(π/2) + ln(scale) + 0.5
```

For `scale = 1`: `entropy = 0.5*ln(π/2) + 0.5 ≈ 0.726`.
For `scale = 2`: `entropy = 0.5*ln(π/2) + ln(2) + 0.5 ≈ 1.419`.
Tests pin both.

### Property overrides (REQ-8)

`mean = scale * sqrt(2/π) ≈ 0.7979 * scale`,
`variance = scale² * (1 - 2/π) ≈ 0.3634 * scale²`,
`mode = 0` (the half-normal density's maximum is at the
support's left boundary `x = 0`).

### `HalfNormalRsampleBackward` (REQ-9)

```rust
d(z)/d(scale) = |eps|
grad_scale = sum_i grad_output[i] * |eps[i]|
```

The gradient is always non-negative (since `|eps| ≥ 0`), so for
`grad_output > 0` (the typical case of `loss = sum(z)`) the
scale gradient is positive — increasing scale increases all
samples. Test
`test_half_normal_rsample_backward` pins
`grad > 0` for `rsample([10]).sum_all().backward()`.

### Non-test production consumers

- **`pub use half_normal::HalfNormal` in lib.rs** —
  grandfathered public surface (S5).
- **`Distribution` trait dispatch via `pub use HalfNormal`** —
  every external `dist.{sample, rsample, log_prob, ...}` call
  hits this impl block; production callers use the half-normal
  as a positive-support prior in Bayesian models and as the
  noise distribution in some reparameterization tricks.

## Parity contract

`parity_ops = []`. Closed-form distribution.

Edge cases:

- **`x < 0` in `log_prob`**: returns `-inf` per the support
  mask. Pinned by
  `test_half_normal_log_prob_negative_is_neginf`.
- **`scale = 0`**: not validated at construction. Would produce
  `ln(0) = -inf` in `log_prob` and a tensor of all zeros in
  `sample` (since `0 * |eps| = 0`). Tracked by blocker #1421.
- **Sample monotonicity in scale**: `sample(scale=2) ≈ 2 *
  sample(scale=1)` in distribution. Implicit in
  `test_half_normal_sample_mean`.
- **`f64`**: `test_half_normal_f64`.

## Verification

Unit tests (13 tests):

- Sample shape + non-negativity + Monte Carlo mean:
  `test_half_normal_sample_shape/_nonnegative/_mean`.
- `rsample` grad: `test_half_normal_rsample_has_grad`.
- `log_prob` analytical + support mask:
  `test_half_normal_log_prob_at_zero/_at_one/_negative_is_neginf/_scale2`.
- `entropy`: `test_half_normal_entropy/_scale2`.
- Backward: `test_half_normal_rsample_backward`.
- `f64`: `test_half_normal_f64`.
- Properties: `test_half_normal_mean_mode_variance`.

Smoke command:

```bash
cargo test -p ferrotorch-distributions --lib half_normal:: 2>&1 | tail -3
```

Expected: `13 passed; 0 failed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct HalfNormal<T: Float>` with `scale` field in `half_normal.rs` mirroring `torch/distributions/half_normal.py:15-46`; non-test consumer: `pub use half_normal::HalfNormal` in `lib.rs` (grandfathered public surface per goal.md S5). |
| REQ-2 | SHIPPED | impl: `pub fn HalfNormal::new(scale)` in `half_normal.rs` mirroring `half_normal.py:40-46`; non-test consumer: `pub use HalfNormal` re-export. |
| REQ-3 | SHIPPED | impl: `pub fn HalfNormal::scale/mean_value/variance_value` in `half_normal.rs` mirroring `half_normal.py:52-54`; non-test consumer: `fn HalfNormal::mean` and `fn HalfNormal::variance` (trait impl) invoke `self.mean_value()?` and `self.variance_value()?` internally. |
| REQ-4 | SHIPPED | impl: `impl<T: Float> Distribution<T> for HalfNormal<T>` in `half_normal.rs` mirroring `half_normal.py:56-84`; non-test consumer: trait dispatch via `pub use HalfNormal`. |
| REQ-5 | SHIPPED | impl: `fn HalfNormal::sample` / `rsample` in `half_normal.rs` with `scale * |randn|` formula mirroring the `|Normal(0, scale)|` decomposition at `half_normal.py:15-27`; non-test consumer: external `dist.sample/rsample(shape)` calls. |
| REQ-6 | SHIPPED | impl: `fn HalfNormal::log_prob` in `half_normal.rs` with closed-form and `x < 0 → -inf` mask mirroring `half_normal.py:68-73`; non-test consumer: external `dist.log_prob(value)` calls; `test_half_normal_log_prob_negative_is_neginf` pins the support mask. |
| REQ-7 | SHIPPED | impl: `fn HalfNormal::entropy` in `half_normal.rs` with `0.5*ln(π/2) + ln(scale) + 0.5` formula mirroring `half_normal.py:83-84`; non-test consumer: external `dist.entropy()` calls. |
| REQ-8 | SHIPPED | impl: `fn HalfNormal::{mean, mode, variance}` overrides in `half_normal.rs` mirroring `half_normal.py:56-66`; non-test consumer: external `dist.{mean, mode, variance}` calls; `test_half_normal_mean_mode_variance` pins all three. |
| REQ-9 | SHIPPED | impl: `struct HalfNormalRsampleBackward<T: Float>` with `GradFn::backward` in `half_normal.rs` computing `sum(grad_output * |eps|)`; non-test consumer: invoked by `fn HalfNormal::rsample` whenever scale requires grad; `test_half_normal_rsample_backward` pins the positive-gradient contract. |
| REQ-10 | NOT-STARTED | blocker #1421 — `expand`, `arg_constraints`, `support`, `validate_args`, `cdf` (requires erf), `icdf` (requires inverse erf), `TransformedDistribution` base hooks (from `half_normal.py:33-50, 75-81`) not implemented. Cross-cutting with `lib.md` REQ-5 (blocker #1376). |
