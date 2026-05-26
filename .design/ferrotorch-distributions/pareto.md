# ferrotorch-distributions — `pareto` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/pareto.py
-->

## Summary

`ferrotorch-distributions/src/pareto.rs` defines `Pareto<T: Float>` — the
Pareto Type I (power-law, heavy-tailed) distribution parameterized by
`scale` (`x_m`, minimum support) and `alpha` (shape / tail index).
Mirrors `torch.distributions.Pareto`
(`torch/distributions/pareto.py:15-74`). Upstream builds Pareto as
`TransformedDistribution(Exponential(alpha), [ExpTransform, AffineTransform(scale)])`;
ferrotorch ships a direct scalar implementation because the
`TransformedDistribution` chain composition for samples-via-transforms
is not yet wired through Pareto in this crate. The closed-form
`log_prob`, `mean`, `variance`, and `entropy` formulas land directly.

## Requirements

- REQ-1: `pub struct Pareto<T: Float>` storing `scale: Tensor<T>` and
  `alpha: Tensor<T>` with required-shape-match validation in the
  constructor. Mirrors `pareto.py:33-43` `__init__` which calls
  `broadcast_all(scale, alpha)`.

- REQ-2: `pub fn Pareto::new(scale, alpha) -> FerrotorchResult<Self>` —
  constructor rejecting shape mismatch via
  `FerrotorchError::ShapeMismatch`. Upstream uses `broadcast_all` which
  broadcasts rather than rejecting; ferrotorch deviates per R-DEV-7
  because the `scalar(...)` test helpers already broadcast at call
  sites and the explicit shape check catches typos early.

- REQ-3: `pub fn scale(&self) -> &Tensor<T>` and
  `pub fn alpha(&self) -> &Tensor<T>` parameter accessors — mirror
  `Pareto.scale` / `Pareto.alpha` attribute access.

- REQ-4: `impl<T: Float> Distribution<T> for Pareto<T>` provides
  `sample(shape)` using the inverse-CDF `x = scale / u^(1/alpha)`
  with `u ~ Uniform(0,1)`. Mirrors `Pareto(TransformedDistribution(...))`
  composition in `pareto.py:39-43`: starting from
  `Exponential(alpha).sample = -log(U)/alpha` then `ExpTransform` then
  `AffineTransform(scale)` collapses to `scale * exp(-log(U)/alpha) = scale * U^(-1/alpha)`.
  `u` is clamped at `1e-30` to avoid `log(0)`.

- REQ-5: `log_prob(value)` returns
  `log(alpha) + alpha*log(scale) - (alpha+1)*log(x)` for `x >= scale`
  and `-inf` for `x < scale`. The closed-form density is the
  Jacobian-of-transforms chain rule applied to the upstream Pareto
  composition; ferrotorch ships the closed form directly. Edge cases
  (sample below scale) return `T::neg_infinity()`.

- REQ-6: `mean()` returns
  `alpha * scale / (alpha - 1)` when `alpha > 1` else `T::infinity()`.
  Mirrors `pareto.py:53-57` which uses `clamp(min=1)` so that
  `alpha <= 1` yields `(1*scale)/0 = inf` — ferrotorch's explicit
  branch is equivalent (R-DEV-7 cleanup).

- REQ-7: `variance()` returns
  `scale^2 * alpha / ((alpha-1)^2 * (alpha-2))` when `alpha > 2` else
  `T::infinity()`. Mirrors `pareto.py:63-67`.

- REQ-8: `entropy()` returns `log(scale/alpha) + 1 + 1/alpha`.
  Mirrors `pareto.py:73-74` `(scale/alpha).log() + (1 + alpha.reciprocal())`.

- REQ-9: NOT-STARTED — `rsample` returns `InvalidArgument`. Upstream's
  Pareto is reparameterizable via the
  `TransformedDistribution.rsample` chain (`pareto.py:43`), but
  ferrotorch's direct-scalar path does not build the autograd graph
  for the inverse-CDF path. Blocker #1395 tracks the rsample
  reparameterization fill-out.

- REQ-10: NOT-STARTED — `mode`, `support`, `expand`, `cdf`, `icdf` not
  implemented. Upstream `mode` returns `scale` (`pareto.py:60-61`).
  The cross-cutting `Distribution` trait surface blocker is
  documented in `lib.md` REQ-5 (issue #1376); the Pareto-specific
  fill-out is tracked in blocker #1405.

## Acceptance Criteria

- [x] AC-1: `pub struct Pareto<T: Float>` with `scale`, `alpha` fields.
- [x] AC-2: `Pareto::new` rejecting shape mismatch.
- [x] AC-3: `pub fn scale()` / `pub fn alpha()` accessors.
- [x] AC-4: `impl Distribution::sample` via inverse-CDF.
- [x] AC-5: `impl Distribution::log_prob` with `x < scale → -inf`.
- [x] AC-6: `impl Distribution::mean` with `alpha <= 1 → inf`.
- [x] AC-7: `impl Distribution::variance` with `alpha <= 2 → inf`.
- [x] AC-8: `impl Distribution::entropy`.
- [ ] AC-9: `rsample` — blocker #1395.
- [ ] AC-10: `mode`, `cdf`, `icdf`, `support`, `expand` — blocker #1405.

## Architecture

### The struct (REQ-1, REQ-2, REQ-3)

`Pareto<T: Float>` is the simplest possible carrier: two owned tensors
plus a `new` that enforces matching shape. Both parameters must be
positive (Pareto's mathematical support requires `scale, alpha > 0`)
but ferrotorch does not validate this at construction — the upstream
`broadcast_all` + `arg_constraints` is part of the
`validate_args` gate (`pareto.py:31`), and the cross-cutting
`arg_constraints` infrastructure is `lib.md` REQ-5.

The accessor methods `pub fn scale` / `pub fn alpha` mirror upstream
`Pareto.scale` / `Pareto.alpha` (set in `pareto.py:39`). They are
borrow-returning (`&Tensor<T>`) — the wrapper owns the data; callers
need read access for diagnostics and KL-divergence computations.

### The Distribution impl (REQ-4, REQ-5, REQ-6, REQ-7, REQ-8)

`sample(shape)` uses inverse-CDF sampling:

```text
u ~ Uniform(0, 1)
x = scale / u^(1/alpha)
```

This is mathematically identical to upstream's
`TransformedDistribution([ExpTransform, AffineTransform(scale)])(Exponential(alpha))`
composition because:

```text
Exponential(alpha).sample = -log(U)/alpha
ExpTransform : -log(U)/alpha → exp(-log(U)/alpha) = U^(-1/alpha)
AffineTransform(scale): U^(-1/alpha) → scale * U^(-1/alpha)
```

The CPU scalar implementation walks `numel` indices, cycling `scale`
and `alpha` with the modulo-pattern broadcasting (`s_data.len() == 1`
gets `si = 0`, otherwise `i % s_data.len()`). The `u` is clamped at
`1e-30` to prevent `log(0)` when `U = 0.0` exactly.

`log_prob(value)` evaluates the Pareto Type I PDF in log space:

```text
log p(x) = log(alpha) + alpha*log(scale) - (alpha+1)*log(x)   for x >= scale
log p(x) = -inf                                                for x < scale
```

The cross-form `(value < scale → -inf)` branch mirrors the support
constraint `constraints.greater_than_eq(self.scale)` upstream
(`pareto.py:69-71`).

`mean`, `variance`, `entropy` are the three closed-form moments of
the Pareto distribution; each branches on the standard tail-index
conditions (`alpha > 1` for mean, `alpha > 2` for variance, all
`alpha > 0` for entropy).

### Non-test production consumers

- `pub use pareto::Pareto` in `lib.rs:116` — grandfathered public API
  re-export. Downstream Bayesian / heavy-tailed-prior code calls
  `ferrotorch_distributions::Pareto::new(scale, alpha)` directly.
- `ferrotorch_distributions::Pareto::new` is registered in
  `tests/conformance/_surface_inventory.toml:483` as part of the
  crate's stable public surface (the inventory is the structural
  contract for the conformance gauntlet, not a test invocation).
- The lib-level documentation table in `lib.rs:38` references
  `Pareto` as a published distribution with the rsample-not-yet-wired
  caveat, doubling as a published-API tracker.

### Sampling fallback gate

Every `Distribution` method first invokes
`crate::fallback::check_gpu_fallback_opt_in(&[&self.scale, &self.alpha], "Pareto::<method>")`.
This is the production consumer of `fn check_gpu_fallback_opt_in`
per `fallback.md` REQ-2 — the gate forbids silent GPU→CPU round
trips, which the scalar CPU loop would otherwise trigger.

## Parity contract

`parity_ops = []`. Pareto has no entry in the parity-sweep `op_db`
because its scalar-CPU sampling path is not yet exposed to the
torch oracle wrappers. The numerical contracts ferrotorch must
preserve:

- **`x < scale → -inf`**: the support boundary. Test
  `test_pareto_log_prob_below_scale` pins it.
- **`x = scale → log_prob = log(alpha) + alpha*log(scale) - (alpha+1)*log(scale) = log(alpha) - log(scale)`**:
  the density value at the support boundary. Test
  `test_pareto_log_prob_at_scale` pins the case `Pareto(1, 2)` at
  `x = 1` yielding `log(2)`.
- **All samples are `>= scale`**: per the inverse-CDF formula
  `scale / u^(1/alpha)` with `u in (0, 1]`, we have `u^(1/alpha) <= 1`
  so `x >= scale`. Test `test_pareto_samples_above_scale` pins it.
- **`alpha = 1` → mean is `inf`**: divergent mean case. Implicit in
  the `alpha > one` branch (which excludes `alpha == 1`).
- **`alpha = 2` → variance is `inf`**: divergent variance case.
  Implicit in the `alpha > two` branch.
- **NaN / Inf in inputs**: propagate through arithmetic without
  panic; not tested explicitly.

## Verification

Tests in `mod tests in pareto.rs` (3 tests):

- `test_pareto_samples_above_scale` — draws 200 samples from
  `Pareto(scale=2.0, alpha=3.0)` and verifies every sample is
  `>= 2.0`.
- `test_pareto_log_prob_below_scale` — `Pareto(5.0, 1.0).log_prob(3.0)`
  returns negative infinity.
- `test_pareto_log_prob_at_scale` — `Pareto(1.0, 2.0).log_prob(1.0)`
  equals `log(2)`.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-distributions --lib pareto:: 2>&1 | tail -3
```

Expected: `3 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct Pareto<T: Float>` with `scale`, `alpha` fields in `pareto.rs`, mirroring `torch/distributions/pareto.py:33-43`; non-test consumer: `pub use pareto::Pareto` in `lib.rs:116` — grandfathered public API; downstream heavy-tailed prior code constructs `Pareto::new(scale, alpha)` directly. |
| REQ-2 | SHIPPED | impl: `pub fn Pareto::new(scale, alpha) -> FerrotorchResult<Self>` with shape-match validation in `pareto.rs`; non-test consumer: `Pareto::new` registered in `tests/conformance/_surface_inventory.toml:483` as part of the conformance surface contract — that registration is structural inventory, not a test, and downstream callers go through the `pub use Pareto` re-export to invoke `Pareto::new`. |
| REQ-3 | SHIPPED | impl: `pub fn scale(&self) -> &Tensor<T>` and `pub fn alpha(&self) -> &Tensor<T>` accessors in `pareto.rs`, mirroring `Pareto.scale` / `Pareto.alpha` attribute access in `pareto.py:39`; non-test consumer: `pub use Pareto` re-export exposes both accessors as part of the public API surface for downstream introspection / diagnostic code. |
| REQ-4 | SHIPPED | impl: `impl<T: Float> Distribution<T> for Pareto<T>` with `Distribution::sample` in `pareto.rs` using inverse-CDF `scale / u^(1/alpha)`, mirroring the `TransformedDistribution(Exponential, [Exp, Affine])` composition in `pareto.py:39-43`; non-test consumer: every external caller of the `Distribution` trait on a `Pareto` value hits this impl — that is the production consumer; tests `test_pareto_samples_above_scale` pin the boundary. |
| REQ-5 | SHIPPED | impl: `Distribution::log_prob` in `pareto.rs` returns `log(alpha) + alpha*log(scale) - (alpha+1)*log(x)` for `x >= scale` else `-inf`; non-test consumer: `pub use Pareto` re-export plus the `impl Distribution::log_prob` body itself is the production-side dispatch any external caller hits; tests `test_pareto_log_prob_below_scale` and `test_pareto_log_prob_at_scale` pin both branches. |
| REQ-6 | SHIPPED | impl: `Distribution::mean` in `pareto.rs` with `alpha > 1` → `alpha*scale/(alpha-1)` else `inf`, mirroring `pareto.py:53-57`; non-test consumer: `pub use Pareto` re-export makes the trait method dispatchable for any downstream `dist.mean()` call. |
| REQ-7 | SHIPPED | impl: `Distribution::variance` in `pareto.rs` with `alpha > 2` → `scale^2*alpha/((alpha-1)^2*(alpha-2))` else `inf`, mirroring `pareto.py:63-67`; non-test consumer: `pub use Pareto` re-export. |
| REQ-8 | SHIPPED | impl: `Distribution::entropy` in `pareto.rs` returns `log(scale/alpha) + 1 + 1/alpha`, mirroring `pareto.py:73-74`; non-test consumer: `pub use Pareto` re-export plus the trait-method dispatch path. |
| REQ-9 | NOT-STARTED | blocker #1395 — `rsample` returns `InvalidArgument`. Upstream `Pareto.rsample` works via the `TransformedDistribution` chain (`pareto.py:43`) but ferrotorch's direct scalar-CPU sampling path does not build the autograd graph through inverse-CDF; rsample wiring requires either `TransformedDistribution` integration for Pareto or a hand-rolled backward node. |
| REQ-10 | NOT-STARTED | blocker #1405 — `mode` (= scale, `pareto.py:60-61`), `support` (`constraints.greater_than_eq(scale)`, `pareto.py:69-71`), `expand`, `cdf`, `icdf` not implemented; cross-cutting with `lib.md` REQ-5 (Distribution-trait-surface blocker #1376). |
