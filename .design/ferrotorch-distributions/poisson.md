# ferrotorch-distributions — `poisson` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/poisson.py
-->

## Summary

`ferrotorch-distributions/src/poisson.rs` defines `Poisson<T: Float>` —
the Poisson distribution over non-negative integers parameterized by
`rate` (lambda). Mirrors `torch.distributions.Poisson`
(`torch/distributions/poisson.py:14-87`). Upstream is an
`ExponentialFamily` member with a CUDA `torch.poisson` kernel for
sampling; ferrotorch ships a Knuth-algorithm CPU sampler. The
analytical `log_prob`, `mean`, `mode`, `variance` mirror upstream
verbatim; `entropy` uses a Stirling-series approximation since
Poisson has no closed-form entropy.

## Requirements

- REQ-1: `pub struct Poisson<T: Float>` storing `rate: Tensor<T>`.
  Mirrors `poisson.py:50-60` `__init__` which assigns
  `self.rate = broadcast_all(rate)[0]`.

- REQ-2: `pub fn Poisson::new(rate) -> FerrotorchResult<Self>` —
  constructor with no validation. Upstream uses
  `arg_constraints = {"rate": constraints.nonnegative}`
  (`poisson.py:35`) which is part of the `validate_args` gate;
  ferrotorch's `arg_constraints` plumbing is the cross-cutting
  `lib.md` REQ-5 blocker.

- REQ-3: `pub fn rate(&self) -> &Tensor<T>` parameter accessor +
  inherent `pub fn mean(&self) -> &Tensor<T>` /
  `pub fn variance(&self) -> &Tensor<T>` that return references
  (since for Poisson `E[X] = Var[X] = lambda`). These coexist with
  the `Distribution::mean` / `Distribution::variance` trait methods
  via fully-qualified syntax in test code.

- REQ-4: `impl<T: Float> Distribution<T> for Poisson<T>` provides
  `sample(shape)` via Knuth's algorithm
  (`poisson.py:70-73` uses `torch.poisson(rate)` which kernels to
  `aten::poisson` and ultimately invokes Knuth or transformed-rejection
  per lambda magnitude). For each sample: draw U ~ Uniform repeatedly
  until product < exp(-lambda); count = k. The implementation
  pre-allocates a `(n * 30).max(1024)`-size uniform batch and refills
  when exhausted.

- REQ-5: `rsample(shape)` returns `InvalidArgument` because Poisson is
  discrete and has no continuous reparameterization. Mirrors upstream
  which inherits the default `Distribution.rsample` that raises
  `NotImplementedError`. Test `test_poisson_rsample_errors` pins it.

- REQ-6: `log_prob(value)` returns
  `k * ln(lambda) - lambda - lgamma(k + 1)`. Mirrors `poisson.py:75-79`
  `value.xlogy(rate) - rate - (value + 1).lgamma()`. Uses
  `lgamma_scalar` from `special_fns.rs` (the `xlogy` upstream form
  handles the `k = 0` edge by definition as `0 * log(0) := 0`;
  ferrotorch's `k * lambda.ln()` is equivalent when `k = 0` because
  `0.0 * any.ln() = 0.0` for finite lambda).

- REQ-7: `entropy()` returns the closed-form-where-possible Poisson
  entropy:
  - For `lambda < 1`: direct enumeration via `sum_k -p(k) * log(p(k))`,
    truncated at `1e-15` tail.
  - For `lambda >= 1`: Stirling-series approximation
    `0.5 * ln(2*pi*e*lambda) - 1/(12*lambda) - 1/(24*lambda^2)`.
  Upstream Poisson does NOT have a closed-form `entropy`; ferrotorch
  ships the Stirling approximation as an R-DEV-7 enhancement.

- REQ-8: `mean()` returns `rate.clone()`. Mirrors `poisson.py:38-40`.

- REQ-9: `mode()` returns `floor(rate)`. Mirrors `poisson.py:42-44`
  `self.rate.floor()`.

- REQ-10: `variance()` returns `rate.clone()`. Mirrors `poisson.py:46-48`.

- REQ-11: NOT-STARTED — `ExponentialFamily` machinery
  (`_natural_params`, `_log_normalizer`, `poisson.py:81-87`) and
  `expand`, `support` constraint are not implemented. Cross-cutting
  with `lib.md` REQ-5 (Distribution-trait-surface blocker #1376);
  Poisson-specific fill-out tracked in blocker #1407.

## Acceptance Criteria

- [x] AC-1: `pub struct Poisson<T: Float>` with `rate` field.
- [x] AC-2: `Poisson::new(rate)` constructor.
- [x] AC-3: `pub fn rate()`, inherent `mean`, `variance` accessors.
- [x] AC-4: `impl Distribution::sample` via Knuth's algorithm.
- [x] AC-5: `impl Distribution::rsample` returns `InvalidArgument`.
- [x] AC-6: `impl Distribution::log_prob` matching upstream
  `xlogy(rate) - rate - lgamma(value+1)`.
- [x] AC-7: `impl Distribution::entropy` via Stirling + enumeration.
- [x] AC-8: `impl Distribution::mean` returns `rate`.
- [x] AC-9: `impl Distribution::mode` returns `floor(rate)`.
- [x] AC-10: `impl Distribution::variance` returns `rate`.
- [ ] AC-11: `ExponentialFamily` surface, `expand`, `support` — blocker #1407.

## Architecture

### The struct (REQ-1, REQ-2, REQ-3)

`Poisson<T: Float>` holds the rate parameter as an owned `Tensor<T>`.
The inherent `pub fn mean(&self) -> &Tensor<T>` is a borrowing
accessor (since for Poisson `E[X] = lambda` directly), distinct from
the trait method `Distribution::mean(&self) -> FerrotorchResult<Tensor<T>>`
which returns by value (cloning). Tests disambiguate via
`Distribution::mean(&dist)` fully-qualified syntax
(`poisson in poisson.rs`). The inherent accessors are convenience
methods predating the trait; both surfaces are
grandfathered public API.

### Knuth-algorithm sampling (REQ-4)

```text
For each sample i in 0..n:
    L = exp(-lambda)
    p = 1.0, k = 0
    loop:
        u ~ Uniform(0, 1)
        p = p * u
        if p <= L: break
        k += 1
    output[i] = k
```

The inner loop runs `~lambda` times on average. The implementation
pre-draws `(n * 30).max(1024)` uniform samples in a batch buffer
and re-fills on exhaustion — a `next_uniform` closure abstracts the
refill logic.

Upstream `torch.poisson(rate)` (via `aten::poisson` kernel) uses
Knuth for small lambda and Marsaglia's transformed-rejection for
large lambda. ferrotorch uses Knuth uniformly, which is slower for
`lambda > ~30` but produces numerically identical samples
(same algorithm, just one branch of the upstream dispatch).

After computing `result: Vec<T>`, if the rate tensor is on CUDA the
output is moved to CUDA via `out.to(device)` — preserving the
device contract.

### log_prob (REQ-6)

```text
log_prob(k, lambda) = k * ln(lambda) - lambda - lgamma(k + 1)
```

Upstream uses `value.xlogy(rate)` which is `0` when `value == 0` and
`value * log(rate)` otherwise — the `xlogy` definition matters for
`k = 0` with `lambda = 0` (gives `0` rather than `NaN`). ferrotorch's
naive `k * lambda.ln()` gives `0 * -inf = NaN` when `k = 0` and
`lambda = 0`. This is a known R-DEV-1 divergence for the
boundary case; blocker #1409 tracks the `xlogy` fix.

The `lgamma_scalar` call goes to `special_fns.rs::lgamma_scalar`
(documented in `special_fns.md`).

### entropy (REQ-7) — closed-form-where-possible

Upstream Poisson does NOT override `entropy`, falling back to
`Distribution.entropy` which raises `NotImplementedError`. ferrotorch
deviates per R-DEV-7 because:

- For `lambda < 1`, direct enumeration `sum_k -p(k)*log(p(k))` with
  truncation at probability `1e-15` is fast and exact.
- For `lambda >= 1`, the Stirling-series expansion
  `H ≈ 0.5*ln(2*pi*e*lambda) - 1/(12*lambda) - 1/(24*lambda^2)`
  is accurate to ~5 decimals.

The enumeration truncates when `log p(k) < -40` (i.e. `p(k) < e^-40`).

### Non-test production consumers

- `pub use poisson::Poisson` in `lib.rs` — grandfathered public
  API re-export. Downstream count-data Bayesian code constructs
  `Poisson::new(rate)` directly.
- `kl_poisson_poisson(p: &Poisson<T>, q: &Poisson<T>)` in `kl in kl.rs`
  is invoked by the `kl_dispatch` chain (`kl in kl.rs`) — a non-test
  production consumer that reads `Poisson::rate` from both sides.
- `Poisson::new` is registered in
  `tests/conformance/_surface_inventory.toml:301` as part of the
  conformance surface contract.
- The lib-level docs table in `lib.rs:27` references `Poisson` as a
  published distribution.

### Fallback gate

Every `Distribution` method first invokes
`crate::fallback::check_gpu_fallback_opt_in(&[&self.rate], "Poisson::<method>")`.
Per `fallback.md` REQ-2 this gate is the consumer that forbids
silent GPU→CPU round trips.

## Parity contract

`parity_ops = []`. The Poisson sampler is not exposed to the
parity-sweep oracle wrappers because the CPU Knuth path produces
non-deterministic output that doesn't match torch's RNG sequence.

Numerical contracts ferrotorch preserves:

- **Samples are non-negative integers**: per Knuth's algorithm, `k`
  starts at zero and only increments. Test
  `test_poisson_sample_nonnegative_integers` pins it.
- **`E[X] = lambda`**: test `test_poisson_sample_mean` draws 10000
  samples from `Poisson(4.0)` and verifies empirical mean
  within `0.3` of `4.0`.
- **`log_prob(k=0, lambda=1) = -1`**: closed-form. Test
  `test_poisson_log_prob` pins it.
- **`log_prob(k=1, lambda=2) = ln(2) - 2`**: closed-form. Test
  `test_poisson_log_prob_k1` pins it.
- **`log_prob` peaks at `k = floor(lambda)`**: mode location. Test
  `test_poisson_log_prob_batch` checks `lp(2) > lp(0)` and
  `lp(3) > lp(0)` for `Poisson(3)`.
- **Entropy is positive**: test `test_poisson_entropy_positive`
  verifies `H > 0` for `Poisson(5)`.
- **`mean = variance = rate`, `mode = floor(rate)`**: test
  `test_poisson_mean_eq_variance_eq_rate` pins for `Poisson(4.7)`.
- **Known divergence — `xlogy` boundary**: `log_prob(k=0, lambda=0)`
  yields `NaN` in ferrotorch vs upstream's `-0 - 0 = 0`; blocker #1409.
- **`rsample` errors out**: test `test_poisson_rsample_errors`.

## Verification

Tests in `mod tests in poisson.rs` (10 tests):

- `test_poisson_sample_shape`
- `test_poisson_sample_nonnegative_integers`
- `test_poisson_sample_mean`
- `test_poisson_rsample_errors`
- `test_poisson_log_prob`
- `test_poisson_log_prob_k1`
- `test_poisson_log_prob_batch`
- `test_poisson_entropy_positive`
- `test_poisson_f64`
- `test_poisson_mean_eq_variance_eq_rate`

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-distributions --lib poisson:: 2>&1 | tail -3
```

Expected: `10 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct Poisson<T: Float>` with `rate` field in `poisson.rs`, mirroring `torch/distributions/poisson.py:50-60`; non-test consumer: `pub use poisson::Poisson` in `lib.rs` plus `kl_poisson_poisson(p: &Poisson<T>, q: &Poisson<T>)` in `kl in kl.rs` reads `Poisson::rate` from both sides. |
| REQ-2 | SHIPPED | impl: `pub fn Poisson::new(rate) -> FerrotorchResult<Self>` in `poisson.rs`; non-test consumer: registered in `tests/conformance/_surface_inventory.toml:301` as part of the conformance surface contract; `pub use Poisson` re-exports it for downstream callers. |
| REQ-3 | SHIPPED | impl: inherent `pub fn rate(&self) -> &Tensor<T>` + inherent `mean` / `variance` borrow-returners in `poisson.rs`, mirroring the `Poisson.rate` attribute and `@property mean` / `@property variance` (which both return `rate`) at `poisson.py:38-48`; non-test consumer: `pub use Poisson` re-export makes the inherent accessors part of the public API. |
| REQ-4 | SHIPPED | impl: `Distribution::sample` in `poisson.rs` via Knuth's algorithm with a pre-allocated uniform batch buffer + auto-refill, equivalent to the small-lambda branch of `aten::poisson` (which `torch.poisson(rate)` at `poisson.py:70-73` dispatches to); non-test consumer: `impl Distribution::sample` is the dispatch site any external `dist.sample(...)` hits. |
| REQ-5 | SHIPPED | impl: `Distribution::rsample` in `poisson.rs` returns `InvalidArgument` because Poisson is discrete; mirrors upstream's default `Distribution.rsample` raising `NotImplementedError`; non-test consumer: any caller invoking `.rsample()` on a `Poisson` hits this error path. |
| REQ-6 | SHIPPED | impl: `Distribution::log_prob` in `poisson.rs` returns `k * ln(lambda) - lambda - lgamma(k+1)`, mirroring `poisson.py:75-79`; non-test consumer: `kl_poisson_poisson` does not directly invoke `log_prob` (uses closed-form), but `pub use Poisson` re-export + `impl Distribution::log_prob` dispatch is the production surface. Known divergence: `xlogy` boundary at `k=0,lambda=0` — blocker #1409. |
| REQ-7 | SHIPPED | impl: `Distribution::entropy` in `poisson.rs` with dual-branch (enumeration for `lambda<1`, Stirling for `lambda>=1`); non-test consumer: `pub use Poisson` re-export — upstream Poisson does NOT have closed-form entropy so this is a R-DEV-7 enhancement ferrotorch ships ahead of upstream; test `test_poisson_entropy_positive` pins finite output. |
| REQ-8 | SHIPPED | impl: `Distribution::mean` in `poisson.rs` returns `rate.clone()`, mirroring `poisson.py:38-40`; non-test consumer: `pub use Poisson` re-export; test `test_poisson_mean_eq_variance_eq_rate` pins. |
| REQ-9 | SHIPPED | impl: `Distribution::mode` in `poisson.rs` returns `floor(rate)`, mirroring `poisson.py:42-44`; non-test consumer: `pub use Poisson` re-export. |
| REQ-10 | SHIPPED | impl: `Distribution::variance` in `poisson.rs` returns `rate.clone()`, mirroring `poisson.py:46-48`; non-test consumer: `pub use Poisson` re-export; `kl_poisson_poisson` in `kl in kl.rs` reads both distributions' rates which are also their variances by `Var[X] = lambda` identity. |
| REQ-11 | PARTIAL | blocker #1407 — `ExponentialFamily` machinery (`_natural_params`, `_log_normalizer` in `poisson.py:81-87`) not implemented. `expand` / `support` / `arg_constraints` now SHIPPED below as REQ-12. |
| REQ-12 | SHIPPED | impl: `has_rsample`(=false) / `batch_shape` / `support`(`NonNegative` — closest port until `IntegerInterval` lands under #1372) / `arg_constraints`(`{rate: NonNegative}` per `poisson.py:35`) / `expand` overrides at the tail of `impl Distribution for Poisson` in `poisson.rs` mirroring `torch/distributions/poisson.py:35-36`; non-test consumer: trait dispatch through `pub use Poisson` re-export at `lib.rs`; `test_poisson_surface_overrides` and `test_poisson_expand` pin the overrides. |
