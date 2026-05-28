# ferrotorch-distributions — `student_t` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/studentT.py
-->

## Summary

`ferrotorch-distributions/src/student_t.rs` defines `StudentT<T: Float>`
— Student's t-distribution parameterized by `df` (degrees of freedom),
`loc` (location / mean for `df > 1`), and `scale`. Mirrors
`torch.distributions.StudentT`
(`torch/distributions/studentT.py:15-127`). Supports reparameterized
sampling via the
`Y = loc + scale * Z * sqrt(df / Chi2)` representation with
`Z ~ Normal(0, 1)`, `Chi2 ~ Chi2(df)`. Ships a hand-rolled
Marsaglia-Tsang Chi-squared sampler and a hand-coded backward node
`StudentTRsampleBackward` that propagates gradients to `loc` and
`scale` (but NOT `df` — see REQ-9 NOT-STARTED).

## Requirements

- REQ-1: `pub struct StudentT<T: Float>` storing `df: Tensor<T>`,
  `loc: Tensor<T>`, `scale: Tensor<T>`. Mirrors `studentT.py:64-74`
  `__init__` which broadcasts the three params via `broadcast_all`.

- REQ-2: `pub fn StudentT::new(df, loc, scale) -> FerrotorchResult<Self>`
  — constructor requiring all three tensors to share the same shape.
  Returns `ShapeMismatch` on mismatch. Upstream uses `broadcast_all`
  which auto-broadcasts; ferrotorch's strict-shape check is R-DEV-7.

- REQ-3: Three accessors: `pub fn df`, `pub fn loc`, `pub fn scale` —
  all returning `&Tensor<T>`. Mirror upstream attribute access.

- REQ-4: Two convenience inherent methods:
  `pub fn mean_value(&self) -> FerrotorchResult<Vec<T>>` (returns
  `loc` if `df > 1` else `NaN`) and
  `pub fn variance_value(&self) -> FerrotorchResult<Vec<T>>`
  (returns `scale^2 * df / (df - 2)` if `df > 2`, `inf` if
  `1 < df <= 2`, `NaN` if `df <= 1` — closed-form Student's t moments).
  Inherent methods that distinguish themselves from the
  `Distribution::mean` trait method (the trait method is only the
  `loc` clone, NaN-masked for `df <= 1`, matching upstream verbatim).
  Mirror `studentT.py:42-62` `mean`, `variance` @property's.

- REQ-5: `impl<T: Float> Distribution<T> for StudentT<T>` provides
  `sample(shape)` via the canonical Student's-t representation:
  `Y = loc + scale * Z * sqrt(df / Chi2)` where `Z ~ Normal(0, 1)`,
  `Chi2 ~ Chi2(df)`. Mirrors `studentT.py:87-99` `rsample`.
  Chi-squared is sampled via private `fn sample_chi2` using
  Marsaglia-Tsang Gamma-rejection.

- REQ-6: `rsample(shape)` is differentiable through `loc` and `scale`.
  Builds the result via `Tensor::from_operation` with a
  `StudentTRsampleBackward` autograd node that captures `df`, `loc`,
  `scale`, `z`, `chi2`. Returns a non-grad tensor if neither `loc` nor
  `scale` requires grad or grad is globally disabled.

- REQ-7: `log_prob(value)` returns the closed-form Student's-t log density:
  ```text
  lgamma((df+1)/2) - lgamma(df/2)
    - 0.5 * ln(df * pi) - ln(scale)
    - (df+1)/2 * ln(1 + ((x - loc)/scale)^2 / df)
  ```
  Equivalent to `studentT.py:101-112`'s
  `-0.5*(df+1)*log1p(y^2/df) - Z` form where `Z` is the
  log-normalization constant. Test `test_student_t_log_prob_at_loc`
  pins the `df=1` Cauchy edge case (`StudentT(1, 0, 1).log_prob(0) =
  -ln(pi)`); `test_student_t_log_prob_high_df_approaches_normal` pins
  the `df → inf` Normal limit.

- REQ-8: `entropy()` returns the Student's-t closed-form entropy:
  ```text
  (df+1)/2 * (digamma((df+1)/2) - digamma(df/2))
    + 0.5*ln(df) + lgamma(df/2) + 0.5*ln(pi) - lgamma((df+1)/2)
    + ln(scale)
  ```
  Mirrors `studentT.py:114-127` `entropy`. Uses `lgamma_scalar` and
  `digamma_scalar` from `special_fns.rs`.

- REQ-9: SHIPPED — `rsample` propagates a gradient to `df`. The
  `StudentTRsampleBackward::backward` populates the df slot with the
  pathwise Chi2 reparameterization gradient combining (1) the explicit
  `df` in `sqrt(df/chi2)` and (2) the implicit channel
  `d(chi2)/d(df) = standard_gamma_grad_one(df/2, chi2/2)`, where
  `standard_gamma_grad_one` is the correct
  `d(sg)/d(alpha) = -(∂_alpha P(alpha,sg)) / pdf(sg)` series
  (PyTorch's `_standard_gamma_grad`, `Distributions.h:302-368`) that
  landed in `special_fns.rs` (commit fae8ca185). `rsample` now attaches
  the autograd node whenever `df.requires_grad()` (previously only when
  loc/scale did). FD-verified against an independent `gammp`-based
  central difference across the small-x / rational / saddle branches by
  `test_student_t_df_gradient_matches_finite_difference`; the end-to-end
  attach + finite-grad path is pinned by
  `test_student_t_df_gradient_attaches_node` and (integration)
  `divergence_wave_k_audit::audit_1427_*`. Closes #1427. The OLD
  high-variance score-function form `sg·(ln sg − digamma(alpha))` is
  NOT the pathwise gradient and is pinned-as-wrong by
  `test_repo_gamma_implicit_grad_formula_is_incorrect` to guard against
  a regression that reuses it.

- REQ-10: NOT-STARTED — `expand`, `support`, `mode`,
  `cdf`, `icdf` not implemented. Mode is `loc` (`studentT.py:48-50`)
  — trivially implementable. Cross-cutting with `lib.md` REQ-5
  (Distribution-trait-surface blocker #1376); StudentT-specific
  surface fill-out tracked in blocker #1428.

## Acceptance Criteria

- [x] AC-1: `pub struct StudentT<T: Float>` with `df`, `loc`, `scale`.
- [x] AC-2: `new` rejecting shape mismatch.
- [x] AC-3: `df`, `loc`, `scale` accessors.
- [x] AC-4: `mean_value`, `variance_value` inherent methods.
- [x] AC-5: `impl Distribution::sample` via Normal/Chi2 composition.
- [x] AC-6: `impl Distribution::rsample` differentiable through
  `loc` and `scale` via `StudentTRsampleBackward`.
- [x] AC-7: `impl Distribution::log_prob` matching upstream.
- [x] AC-8: `impl Distribution::entropy`.
- [x] AC-9: `df` gradient via `standard_gamma_grad_one` — closes #1427.
- [ ] AC-10: `expand`, `support`, `mode`, `cdf`, `icdf` — blocker #1428.

## Architecture

### The struct (REQ-1, REQ-2, REQ-3, REQ-4)

```rust
pub struct StudentT<T: Float> {
    df: Tensor<T>,
    loc: Tensor<T>,
    scale: Tensor<T>,
}
```

Strict shape-match in `new`. The convenience `mean_value` /
`variance_value` methods return `Vec<T>` (not tensors) for direct
introspection by tests and downstream diagnostic code. These coexist
with the `Distribution::mean` / `Distribution::variance` trait
methods; the trait method is `loc.clone()` (NaN-masking pending the
mask infrastructure that upstream does via Python boolean indexing).

### Chi-squared sampling helper

`fn sample_chi2<T: Float>(df_values: &[T], n: usize) -> FerrotorchResult<Vec<T>>`
implements Marsaglia-Tsang Gamma sampling with the
`alpha = df/2 < 1` boost trick:

- Standard Marsaglia-Tsang for `alpha >= 1`:
  ```
  d = alpha - 1/3
  c = 1 / (3 * sqrt(d))
  loop:
      x ~ Normal(0, 1)
      v = (1 + c*x)^3
      reject if v <= 0
      u ~ Uniform(0, 1)
      accept if u < 1 - 0.0331 * x^4
            OR ln(u) < x^2/2 + d*(1 - v + ln(v))
      return d * v
  ```
- Boost for `alpha < 1`: sample `Gamma(alpha + 1)` then multiply by
  `U^(1/alpha)`.
- Chi-squared = `2 * Gamma(df/2)`.

Both `sample` and `rsample` pre-draw `n` Normal samples and `n`
Chi-squared samples (via `sample_chi2`), then combine elementwise.

### Distribution::sample and rsample (REQ-5, REQ-6)

Both compute `Y = loc + scale * Z * sqrt(df / Chi2)` per element.
The difference is autograd wiring:

- `sample`: builds the result via `Tensor::from_storage(_, _, false)`
  — `requires_grad = false`.
- `rsample`: if `loc` or `scale` requires grad AND grad is enabled,
  builds via `Tensor::from_operation` with a
  `StudentTRsampleBackward` GradFn that captures `df`, `loc`,
  `scale`, `z`, `chi2` tensors. Otherwise falls back to a plain
  `from_storage`.

The `StudentTRsampleBackward::backward` computes:
- `grad_loc = sum(grad_output)` (since `d Y / d loc = 1`)
- `grad_scale = sum(grad_output * z * sqrt(df / chi2))`
  (since `d Y / d scale = z * sqrt(df / chi2)`)
- `grad_df = scale·z·[ 0.5/sqrt(df·chi2) - 0.5·sqrt(df)·chi2^(-1.5)·sgg ]`
  where `sgg = standard_gamma_grad_one(df/2, chi2/2)` — REQ-9 SHIPPED
  (#1427); see the `StudentTRsampleBackward` struct doc for the
  two-channel derivation.

The `name() -> "StudentTRsampleBackward"` is the GradFn's debug-only
identifier (no inspection in production code).

### Closed-form log_prob and entropy (REQ-7, REQ-8)

Both forms invoke `lgamma_scalar` and `digamma_scalar` from
`special_fns.rs`. The `log_prob` is mathematically equivalent to
upstream's `-0.5 * (df+1) * log1p(y^2/df) - Z` where
`Z = scale.log() + 0.5*df.log() + 0.5*log(pi) + lgamma(df/2) - lgamma((df+1)/2)`.
ferrotorch's expanded form is the same but with the terms
re-ordered to match the typical density-formula presentation.

The `df=1` Cauchy edge case is implicit in the formula. Test
`test_student_t_log_prob_at_loc` verifies that
`StudentT(1, 0, 1).log_prob(0)` equals `-ln(pi)` (the standard Cauchy
density at zero) to 1e-4. The `df → inf` Normal limit is verified
by `test_student_t_log_prob_high_df_approaches_normal` (df=10000
matches `Normal(0,1).log_prob(1) = -0.5 - 0.5*ln(2pi)` to 0.01).

### Non-test production consumers

- `pub use student_t::StudentT` in `lib.rs` — grandfathered
  public API re-export. Downstream Bayesian hierarchical-model code
  with t-distributed priors constructs `StudentT::new(df, loc, scale)?`
  directly.
- `StudentT::new` is registered in
  `tests/conformance/_surface_inventory.toml:315`.
- The lib-level docs table in `lib.rs:28` references it with
  "Yes" for Reparameterized (limited — `df` excluded per REQ-9).

### Fallback gate

Every `Distribution` method first invokes
`crate::fallback::check_gpu_fallback_opt_in(&[&self.df, &self.loc, &self.scale, ...], "StudentT::<method>")`.

## Parity contract

`parity_ops = []`.

Numerical contracts:

- **`sample` mean ~ `loc`**: for `df > 1`, `E[X] = loc`. Test
  `test_student_t_sample_mean` draws 10000 from
  `StudentT(df=10, loc=2, scale=1)` and checks empirical mean is
  within 0.2 of 2.0.
- **`rsample` has gradient**: with `loc.requires_grad_(true)` and
  `scale.requires_grad_(true)`, `rsample(...)` returns a tensor with
  `requires_grad = true` and a non-None `grad_fn`. Test
  `test_student_t_rsample_has_grad`.
- **`rsample` backward**: `loss = z.sum_all(); loss.backward()` gives
  `loc.grad ≈ n` (linear in `loc`) and finite `scale.grad`. Test
  `test_student_t_rsample_backward` pins `loc_grad = 10.0` (for
  n=10 samples).
- **`log_prob` symmetry around `loc`**: test
  `test_student_t_log_prob_symmetry`.
- **`log_prob` at `df=1` matches Cauchy**: test
  `test_student_t_log_prob_at_loc`.
- **`log_prob` at `df=10000` matches Normal**: test
  `test_student_t_log_prob_high_df_approaches_normal`.
- **`entropy > 0`**: test `test_student_t_entropy_positive`.
- **Shape mismatch in `new`**: test `test_student_t_shape_mismatch`.

## Verification

Tests in `mod tests in student_t.rs` (9 tests):

- `test_student_t_sample_shape`
- `test_student_t_sample_mean`
- `test_student_t_rsample_has_grad`
- `test_student_t_log_prob_at_loc`
- `test_student_t_log_prob_symmetry`
- `test_student_t_log_prob_high_df_approaches_normal`
- `test_student_t_entropy_positive`
- `test_student_t_shape_mismatch`
- `test_student_t_rsample_backward`
- `test_student_t_f64`

Smoke command:

```bash
cargo test -p ferrotorch-distributions --lib student_t:: 2>&1 | tail -3
```

Expected: `10 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct StudentT<T: Float>` with `df`, `loc`, `scale` fields in `student_t.rs`, mirroring `torch/distributions/studentT.py:64-74` (`broadcast_all`-based init); non-test consumer: `pub use student_t::StudentT` in `lib.rs` — grandfathered public API. |
| REQ-2 | SHIPPED | impl: `pub fn StudentT::new(df, loc, scale) -> FerrotorchResult<Self>` with shape-match validation in `student_t.rs`; non-test consumer: registered in `tests/conformance/_surface_inventory.toml:315`; `pub use StudentT` re-export. Test `test_student_t_shape_mismatch` pins the rejection. |
| REQ-3 | SHIPPED | impl: `pub fn df(&self)`, `pub fn loc(&self)`, `pub fn scale(&self)` accessors in `student_t.rs`, mirroring `StudentT.df` / `StudentT.loc` / `StudentT.scale` attribute access; non-test consumer: `pub use StudentT` re-export exposes all three. |
| REQ-4 | SHIPPED | impl: inherent `pub fn mean_value(&self) -> FerrotorchResult<Vec<T>>` (NaN-masked for `df <= 1`) and `pub fn variance_value(&self) -> FerrotorchResult<Vec<T>>` (3-way branch for `df > 2`, `1 < df <= 2`, `df <= 1`) in `student_t.rs`, mirroring `studentT.py:42-62` `mean` / `variance` @property formulas; non-test consumer: `pub use StudentT` re-export. |
| REQ-5 | SHIPPED | impl: `Distribution::sample` in `student_t.rs` via `loc + scale * Z * sqrt(df / Chi2)` composition with `fn sample_chi2` Marsaglia-Tsang Gamma sampler, mirroring `studentT.py:87-99` `rsample`'s representation; non-test consumer: the trait impl is the production dispatch; test `test_student_t_sample_mean` pins empirical mean. |
| REQ-6 | SHIPPED | impl: `Distribution::rsample` in `student_t.rs` builds `Tensor::from_operation` with `Arc<StudentTRsampleBackward>` autograd node capturing `df`, `loc`, `scale`, `z`, `chi2` (lines 250-263); the backward computes `grad_loc = sum(grad_output)` and `grad_scale = sum(grad_output * z * sqrt(df/chi2))`; non-test consumer: tests `test_student_t_rsample_{has_grad, backward}` pin the differentiable path. The production code path is the `impl Distribution::rsample` itself — any external caller invoking `dist.rsample(...)` with `loc.requires_grad_(true)` hits this path and gets a differentiable result, which is the production use case (e.g. Bayesian neural network with t-distributed priors). |
| REQ-7 | SHIPPED | impl: `Distribution::log_prob` in `student_t.rs` returns `lgamma((df+1)/2) - lgamma(df/2) - 0.5*ln(df*pi) - ln(scale) - (df+1)/2 * ln(1 + y^2/df)` for `y = (x-loc)/scale`, mirroring `studentT.py:101-112` (algebraically equivalent expansion of `-0.5*(df+1)*log1p(y^2/df) - Z` form); non-test consumer: `pub use StudentT` re-export + impl dispatch; tests `test_student_t_log_prob_{at_loc, symmetry, high_df_approaches_normal}` pin three behaviours. |
| REQ-8 | SHIPPED | impl: `Distribution::entropy` in `student_t.rs` returns `(df+1)/2 * (digamma((df+1)/2) - digamma(df/2)) + 0.5*ln(df) + lgamma(df/2) + 0.5*ln(pi) - lgamma((df+1)/2) + ln(scale)`, mirroring `studentT.py:114-127`; non-test consumer: `pub use StudentT` re-export; test `test_student_t_entropy_positive` pins. Uses `lgamma_scalar` / `digamma_scalar` from `special_fns.rs`. |
| REQ-9 | SHIPPED | impl: `StudentTRsampleBackward::backward` in `student_t.rs` populates the `df` slot with the pathwise Chi2 reparameterization gradient — explicit `df` in `sqrt(df/chi2)` plus the implicit channel `d(chi2)/d(df) = standard_gamma_grad_one(df/2, chi2/2)` (the correct `-(∂_alpha P(alpha,sg))/pdf(sg)` series from `special_fns.rs`, commit fae8ca185, ported from PyTorch `_standard_gamma_grad` `aten/.../Distributions.h:302-368`); `Distribution::rsample` now attaches the node whenever `df.requires_grad()`. Non-test consumer: `Distribution::rsample` itself — any external `dist.rsample(...)` with `df.requires_grad_(true)` (reachable via `pub use StudentT`, e.g. learning the df of a t-distributed prior) drives this path. FD-verified by `test_student_t_df_gradient_matches_finite_difference` (independent `gammp` oracle, all three branches) + `test_student_t_df_gradient_attaches_node` + integration `divergence_wave_k_audit::audit_1427_*`. Closes #1427. The OLD score-function form `sg·(ln sg − digamma(alpha))` is pinned-as-wrong by `test_repo_gamma_implicit_grad_formula_is_incorrect`. |
| REQ-10 | SHIPPED | impl: `has_rsample`(=true) / `batch_shape` / `support`(`Real` per `studentT.py:39`) / `arg_constraints`(`{df: Positive, loc: Real, scale: Positive}` per `studentT.py:34-38`) / `mean` (=loc if df>1 else NaN per `studentT.py:42-46`) / `mode` (=loc per `studentT.py:48-50`) / `variance` (=scale^2*df/(df-2) if df>2 else inf per `studentT.py:52-62`) / `expand` overrides at the tail of `impl Distribution for StudentT` in `student_t.rs`; non-test consumer: trait dispatch through `pub use StudentT` re-export at `lib.rs`; `test_student_t_mean_mode_variance_df_gt_2` / `test_student_t_mean_df_le_1_is_nan` / `test_student_t_variance_df_le_2_is_inf` / `test_student_t_surface_overrides` / `test_student_t_expand` pin the overrides. Closes #1428 — `cdf` / `icdf` require regularized incomplete-beta (under #1372). |
