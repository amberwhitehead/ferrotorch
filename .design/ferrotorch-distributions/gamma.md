# ferrotorch-distributions — `gamma` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/gamma.py
-->

## Summary

`ferrotorch-distributions/src/gamma.rs` implements the Gamma
distribution with shape parameter α (`concentration`) and rate
parameter β (`rate`). Mirrors `torch.distributions.Gamma`.
Sampling uses Marsaglia-and-Tsang's standard-Gamma method
(with `α < 1` boost via `Gamma(α+1) * U^(1/α)`) then divides by
β. `rsample` carries the implicit-reparameterization gradient
through both parameters via the `GammaRsampleBackward` GradFn.
Closed-form `log_prob` / `entropy` / `mean` / `mode` /
`variance` use the special functions `lgamma_scalar` and
`digamma_scalar` from `crate::special_fns`.

## Requirements

- REQ-1: `pub struct Gamma<T: Float>` holding `concentration:
  Tensor<T>` (α) and `rate: Tensor<T>` (β). Mirrors
  `torch/distributions/gamma.py:Gamma.__init__`.

- REQ-2: `pub fn Gamma::new(concentration, rate) ->
  FerrotorchResult<Self>` validates `concentration.shape() ==
  rate.shape()` and returns `ShapeMismatch` otherwise.
  PyTorch broadcasts internally; ferrotorch requires
  pre-broadcasted parameters (R-DEV-4).

- REQ-3: `pub fn concentration(&self) -> &Tensor<T>` and
  `pub fn rate(&self) -> &Tensor<T>` accessors mirror
  `Gamma.concentration` / `Gamma.rate` properties
  (`gamma.py:45-55`).

- REQ-4: `impl<T: Float> Distribution<T> for Gamma<T>` provides
  `sample` / `rsample` / `log_prob` / `entropy` + closed-form
  property overrides `mean` / `mode` / `variance`.

- REQ-5: `sample_standard_gamma<T>(alphas, n)` is the private
  Marsaglia-and-Tsang scalar Gamma(α, 1) sampler. Batches RNG
  draws via `creation::randn` and `creation::rand` to amortize
  the per-call overhead. For `α < 1`, uses the
  Ahrens-Dieter boost: `Gamma(α+1, 1) * U^(1/α)`. Both the
  squeeze test (`u < 1 - 0.0331*x⁴`) and the full test
  (`ln u < 0.5*x² + d*(1 - v + ln v)`) from the M-T paper are
  implemented for the α ≥ 1 path.

- REQ-6: `Gamma::sample(shape)` invokes
  `sample_standard_gamma(&concentration_data, n)` then divides
  each draw by the corresponding rate (`gamma_sample / rate`).
  Mirrors PyTorch's `_standard_gamma(α) / rate`
  (`gamma.py:79-87`).

- REQ-7: `Gamma::rsample(shape)` mirrors `sample`'s forward path
  but attaches `GammaRsampleBackward` when either parameter
  requires grad. Includes the upstream `clamp(min=tiny)` guard
  to prevent zero-valued samples from blowing up the
  log-gradient (`gamma.py:84-86`). The standard-Gamma draws are
  stored on the GradFn for the implicit-reparam computation.

- REQ-8: `log_prob(value) = α*ln(β) + (α-1)*ln(x) - β*x -
  lgamma(α)`. Standard Gamma log-PDF; mirrors PyTorch's xlogy
  formula (`gamma.py:89-98`). Uses `lgamma_scalar`.

- REQ-9: `entropy = α - ln(β) + lgamma(α) + (1-α)*ψ(α)`. Standard
  closed form; mirrors `gamma.py:100-106`. Uses
  `lgamma_scalar` + `digamma_scalar`.

- REQ-10: Property overrides: `mean = α/β`,
  `mode = (α-1)/β if α >= 1 else NaN`,
  `variance = α/β²`. Mirrors `gamma.py:45-55`. The
  ferrotorch mode returns NaN for `α < 1` instead of PyTorch's
  `clamp(min=0)` — this is a documented R-DEV-6 divergence
  (PyTorch's clamp doesn't match the mathematical definition;
  the mode is undefined / boundary in that regime).

- REQ-11: `GammaRsampleBackward` `GradFn` implements:
  ```text
  output = standard_gamma / rate
  d(out)/d(rate) = -standard_gamma / rate²  = -output/rate
  d(out)/d(α)    = standard_gamma * (ln(standard_gamma) - ψ(α)) / rate  ; implicit
  ```
  Summed across the sample tensor into scalar gradients.

- REQ-12: PARTIAL — `expand`, `arg_constraints`, `support`,
  and `cdf` are SHIPPED. `cdf` is the regularized lower
  incomplete gamma `P(conc, rate*x)` (Numerical-Recipes
  `gammp` in `lower_incomplete_gamma_regularized` /
  `gammp_f64`), verified against `scipy.special.gammainc`
  (closes #1397). STILL NOT-STARTED: `validate_args`,
  `_natural_params` / `_log_normalizer` (from
  `gamma.py:36-43, 108-114`). Cross-cutting with `lib.md`
  REQ-5 (blocker #1376). Blocker #1416 closed for the
  trait-surface fill-out.

## Acceptance Criteria

- [x] AC-1: `pub struct Gamma<T: Float>` with `concentration`,
  `rate`.
- [x] AC-2: `pub fn Gamma::new` with shape-equality check.
- [x] AC-3: `pub fn concentration` / `rate` accessors.
- [x] AC-4: `impl Distribution<T> for Gamma<T>` with all four
  required trait methods + property overrides.
- [x] AC-5: `sample_standard_gamma` private helper with M-T
  and α<1 boost.
- [x] AC-6: `sample` via standard-Gamma division.
- [x] AC-7: `rsample` with tiny-guard + `GammaRsampleBackward`.
- [x] AC-8: `log_prob` closed form.
- [x] AC-9: `entropy` closed form.
- [x] AC-10: `mean`, `mode`, `variance` overrides.
- [x] AC-11: `GammaRsampleBackward` GradFn.
- [x] AC-12: `test_gamma_*` test suite (12 tests).
- [x] AC-13: `expand` / `cdf` (regularized lower incomplete
  gamma, scipy-verified) — closes #1416, #1397. `validate_args`
  remains an orthogonal tracker.

## Architecture

### Marsaglia-Tsang scalar sampler (REQ-5)

The `sample_standard_gamma<T>(alphas, n)` helper:

1. Allocates batched buffers `norm_buf: Vec<T>` and `unif_buf:
   Vec<T>` of size `max(n, 256)` and refills them lazily as the
   rejection loop consumes draws.
2. For each `α[i % len]`:
   - If `α < 1`, uses the boost path: sample
     `Gamma(α+1, 1)` then multiply by `U^(1/α)` (with
     `U = max(U, 1e-30)`).
   - Otherwise, runs the M-T rejection loop with
     `d = α - 1/3` and `c = (1/3) / sqrt(d)`.
3. Returns the full `Vec<T>` of standard-Gamma draws.

The lazy refill of `norm_buf` and `unif_buf` amortizes the
per-call RNG setup overhead, important when `n` is large.

### Constructor + accessors (REQ-1, REQ-2, REQ-3)

`Gamma::new` enforces strict shape equality on its two
parameters. The two accessors are by-reference.

### `sample` (REQ-6) and `rsample` (REQ-7)

```rust
let standard = sample_standard_gamma(&conc, n)?;
let result = standard.zip(rate.cycle()).map(|(g, r)| g / r);
```

`rsample` additionally:

1. Guards each output element to `>= 1e-30` (otherwise
   `ln(x) = -inf` in the implicit-reparam grad).
2. When `concentration.requires_grad() || rate.requires_grad()`,
   stores the **un-divided** standard-Gamma samples on the
   `GammaRsampleBackward` GradFn.

### `log_prob` and `entropy` (REQ-8, REQ-9)

```rust
log_prob = α*ln(β) + (α-1)*ln(x) - β*x - lgamma_scalar(α)
entropy  = α - ln(β) + lgamma_scalar(α) + (1-α)*digamma_scalar(α)
```

Tests pin `log_prob(2) = -2` for `Gamma(1,1) = Exp(1)` and
`log_prob(1) = -1` for `Gamma(2,1)`. Entropy of `Gamma(1,1) = 1`
(same as Exponential(1)).

### Property overrides (REQ-10)

`mean = α/β`, `variance = α/β²`, `mode = (α-1)/β` for `α ≥ 1`
else NaN. Tests:

- `test_gamma_mean_variance` for `Gamma(4, 2)`:
  mean = 2, var = 1, mode = 1.5.
- `test_gamma_mode_nan_for_concentration_below_one`:
  `Gamma(0.5, 2).mode().is_nan()`.

### `GammaRsampleBackward` (REQ-11)

Per-element backward:

- `d(out)/d(rate) = -standard_gamma / rate²`
- `d(out)/d(α) = standard_gamma * (ln(standard_gamma) - ψ(α)) / rate`

The `standard_gamma` factor uses `max(g, 1e-30)` to guard
against `ln(0)` for tiny boost-path samples. Summed into
scalar gradient tensors that match the parameter shapes.

### Non-test production consumers

- **`pub use gamma::Gamma` in lib.rs** — grandfathered public
  surface (S5).
- **`Beta::sample` and `Beta::rsample` in beta.rs** — the
  Beta-via-Gamma-ratio sampler explicitly constructs
  `crate::Gamma::new(self.concentration1.clone(), ones.clone())?`
  and `crate::Gamma::new(self.concentration0.clone(), ones)?`
  inside `fn Beta::sample` and `fn Beta::rsample`. These are
  load-bearing **non-test production consumers** of both the
  `pub fn Gamma::new` constructor AND the `Distribution` trait
  surface (`Beta::rsample` calls `gamma_a.rsample(shape)` and
  `gamma_b.rsample(shape)` on the constructed Gammas).
- **KL-dispatcher consumer**: `kl.rs` registers `Gamma-Gamma`,
  `Gamma-Exponential`, `Exponential-Gamma` arms (downcast on
  `Gamma<T>`). The dispatcher is invoked from the public
  `pub fn kl_divergence` entry; `kl_gamma_gamma` and
  `kl_gamma_exponential` / `kl_exponential_gamma` read
  `.concentration()` and `.rate()` for closed-form formulas.

## Parity contract

`parity_ops = []`. Closed-form distribution with rejection
sampling.

Edge cases covered:

- **`α < 1` (boost path)**: `test_gamma_sample_small_alpha` with
  `α = 0.5` exercises the boost.
- **Sample mean Monte Carlo**:
  `test_gamma_sample_mean` for `Gamma(3, 2)` (E[X] = 1.5).
- **`α = 1, β = 1` (Exponential(1))**:
  `log_prob(2) = -2`, `entropy = 1`.
- **Mode NaN for `α < 1`**: pinned by
  `test_gamma_mode_nan_for_concentration_below_one`.
- **`f64`**: `test_gamma_f64`.
- **Tiny standard-Gamma output**: `1e-30` guard prevents
  gradient blow-up.

## Verification

Unit tests (12 tests):

- Sample shape + positivity + mean Monte Carlo +
  small-alpha boost: `test_gamma_sample_shape/_positive/_mean/_small_alpha`.
- `rsample` grad: `test_gamma_rsample_has_grad`.
- `log_prob`: `test_gamma_log_prob/_alpha2`.
- `entropy`: `test_gamma_entropy`.
- Constructor: `test_gamma_shape_mismatch`.
- `f64`: `test_gamma_f64`.
- Properties: `test_gamma_mean_variance`,
  `test_gamma_mode_nan_for_concentration_below_one`.

Smoke command:

```bash
cargo test -p ferrotorch-distributions --lib gamma:: 2>&1 | tail -3
```

Expected: `12 passed; 0 failed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct Gamma<T: Float>` with `concentration`, `rate` fields in `gamma.rs` mirroring `torch/distributions/gamma.py:18-68`; non-test consumer: `pub use gamma::Gamma` in `lib.rs` (grandfathered public surface per goal.md S5) + `fn Beta::sample` and `fn Beta::rsample` in `beta.rs` construct `crate::Gamma::new(...)` instances. |
| REQ-2 | SHIPPED | impl: `pub fn Gamma::new` in `gamma.rs` with shape-equality check mirroring `gamma.py:57-68`; non-test consumer: `fn Beta::sample` / `fn Beta::rsample` in `beta.rs` invoke `crate::Gamma::new(self.concentration1.clone(), ones.clone())?` — 4 production callsites in `beta.rs`. |
| REQ-3 | SHIPPED | impl: `pub fn Gamma::concentration` / `rate` accessors in `gamma.rs` mirroring `gamma.py:45-55`; non-test consumer: `kl_gamma_gamma` / `kl_gamma_exponential` / `kl_exponential_gamma` in `kl.rs` read `.concentration().data_vec()?` and `.rate().data_vec()?` for the closed-form KL formulas. |
| REQ-4 | SHIPPED | impl: `impl<T: Float> Distribution<T> for Gamma<T>` in `gamma.rs` mirroring `gamma.py:79-106`; non-test consumer: `fn Beta::sample` calls `gamma_a.sample(shape)?` and `fn Beta::rsample` calls `gamma_a.rsample(shape)?` — the entire Beta sampling path flows through the Gamma trait surface. |
| REQ-5 | SHIPPED | impl: `fn sample_standard_gamma<T>` in `gamma.rs` with Marsaglia-Tsang + α<1 boost + lazy RNG buffer refill; non-test consumer: invoked by both `fn Gamma::sample` and `fn Gamma::rsample`; production callers reach it transitively via `fn Beta::sample` / `fn Beta::rsample` (REQ-2 evidence). |
| REQ-6 | SHIPPED | impl: `fn Gamma::sample` in `gamma.rs` invoking `sample_standard_gamma` then dividing by rate, mirroring `gamma.py:79-87`; non-test consumer: `fn Beta::sample` invokes `gamma_a.sample(shape)?` at `beta.rs`. |
| REQ-7 | SHIPPED | impl: `fn Gamma::rsample` in `gamma.rs` with tiny-guard + `GammaRsampleBackward` attachment, mirroring the `value.detach().clamp_(min=tiny)` from `gamma.py:84-86`; non-test consumer: `fn Beta::rsample` invokes `gamma_a.rsample(shape)?` at `beta.rs`. |
| REQ-8 | SHIPPED | impl: `fn Gamma::log_prob` in `gamma.rs` with `α*ln(β) + (α-1)*ln(x) - β*x - lgamma(α)` formula mirroring `gamma.py:89-98`; non-test consumer: external `dist.log_prob(value)` calls. |
| REQ-9 | SHIPPED | impl: `fn Gamma::entropy` in `gamma.rs` mirroring `gamma.py:100-106`; non-test consumer: external `dist.entropy()` calls. |
| REQ-10 | SHIPPED | impl: `fn Gamma::{mean, mode, variance}` overrides in `gamma.rs` mirroring `gamma.py:45-55`; non-test consumer: external `dist.{mean, mode, variance}` calls; `test_gamma_mean_variance` pins all three. |
| REQ-11 | SHIPPED | impl: `struct GammaRsampleBackward<T: Float>` with `GradFn::backward` in `gamma.rs` implementing implicit-reparam through standard-Gamma; non-test consumer: invoked by `fn Gamma::rsample` whenever either parameter requires grad — and that `rsample` is reached transitively by `fn Beta::rsample` (which uses the Gamma trait surface), so `BetaRsampleBackward`'s grad path also depends on `GammaRsampleBackward` indirectly via the autograd graph. |
| REQ-12 | PARTIAL | impl: `has_rsample` / `support` (NonNegative) / `arg_constraints` (concentration:Positive, rate:Positive) / `event_shape` / `expand` (broadcasts both parameters) / `cdf` trait overrides at the tail of `impl Distribution<T> for Gamma<T>` in `gamma.rs` mirroring `torch/distributions/gamma.py:18-119`; `cdf` = regularized lower incomplete gamma `P(conc, rate*x)` via `fn lower_incomplete_gamma_regularized` / `fn gammp_f64` (Numerical-Recipes `gammp`: power series for `x<s+1`, Lentz continued fraction for `x≥s+1`) mirroring `gamma.py:116-119 torch.special.gammainc`, verified to 1e-12 against `scipy.special.gammainc` by `gamma.rs::test_gamma_cdf_*` (7 tests); non-test consumer: `pub use gamma::Gamma` at `lib.rs` + external `dist.cdf(value)` calls. Closes #1416, #1397 — STILL NOT-STARTED: `validate_args`, `_natural_params` / `_log_normalizer` (orthogonal trackers). |
