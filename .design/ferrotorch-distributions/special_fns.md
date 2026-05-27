# ferrotorch-distributions — `special_fns` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/distribution.py
-->

## Summary

`ferrotorch-distributions/src/special_fns.rs` provides two scalar
special-function implementations used internally by distribution
code: `lgamma_scalar` (log-gamma via the Lanczos approximation) and
`digamma_scalar` (psi via recurrence + asymptotic expansion). The
module is `pub(crate)` — only this crate's distribution and KL code
calls into it. PyTorch's tensor-level analogs live in
`torch.special.gammaln` and `torch.special.digamma`; the scalar
forms here exist because per-element distribution formulas
(`kl_gamma_gamma`, Beta parameter ratios, Dirichlet entropy) want
scalars to operate on `Vec<T>` iterators rather than spinning up
1-element tensors.

## Requirements

- REQ-1: `pub(crate) fn lgamma_scalar<T: Float>(x: T) -> T` computes
  `log|Γ(x)|` via the 9-term Lanczos approximation with `g = 7`,
  including a reflection-formula branch (`x < 0.5`) that uses
  `lgamma(1 - x) - ln(π/sin(πx))` to extend to negative arguments.
  Accuracy is ~1e-12 for f64 in `[0.1, 100]`; ~1e-6 for f32. Mirrors
  the standard Lanczos coefficients used by SciPy and PyTorch's
  `aten/src/ATen/native/Math.h:calc_lgamma`.

- REQ-2: `pub(crate) fn digamma_scalar<T: Float>(x: T) -> T`
  computes `ψ(x) = d/dx ln Γ(x)` via the recurrence
  `ψ(x) = ψ(x+1) - 1/x` (shift x ≥ 6) followed by the
  Abramowitz–Stegun 6.3.18 asymptotic expansion. Negative arguments
  use the reflection `ψ(x) = ψ(1 - x) - π·cot(πx)`. NaN propagates.
  Accuracy is ~1e-10 for f64.

- REQ-3: Both functions are generic over `T: Float`, internally
  computing coefficients in f64 and promoting via `T::from(...).unwrap()`.
  This keeps the constants as f64 literals (preserving precision)
  while letting callers operate on f32 tensor element types. The
  `.unwrap()` on `T::from(<f64 literal>)` is sound because every
  literal in the body is a normal-range f64 representable in any
  `Float` type (f32, f64).

- REQ-4: `pub(crate)` visibility — neither function is part of the
  crate's public API. Users who need tensor-level lgamma/digamma
  call into `ferrotorch_core::special::{lgamma, digamma}` directly.
  The scalar forms exist for per-element host-side compute inside
  distribution formulas.

- REQ-5 (#1379): `pub(crate) fn trigamma_scalar<T: Float>(x: T) -> T`
  computes `ψ'(x)` via the recurrence `ψ'(x) = ψ'(x+1) + 1/x²`
  (shift `x ≥ 6`) + the Abramowitz–Stegun 6.4.12 asymptotic series
  `1/y + 1/(2y²) + 1/(6y³) - 1/(30y⁵) + 1/(42y⁷) - 1/(30y⁹)`, with
  reflection `ψ'(x) = π²/sin²(πx) - ψ'(1-x)` for `x < 0`. The general
  entry point `pub(crate) fn polygamma_scalar<T: Float>(n: u32, x: T)
  -> T` covers `n = 0` (digamma), `n = 1` (trigamma), and `n ≥ 2`
  (`(-1)^(n+1)·n!·ζ(n+1, x)` via a Hurwitz-zeta recurrence shift to
  `x ≥ 10` + a 3-Bernoulli Euler–Maclaurin tail). `polygamma_scalar`
  is the production consumer of both `digamma_scalar` (which now
  delegates to `polygamma_scalar(0, x)`) and `trigamma_scalar`
  (invoked from the `n == 1` arm). Mirrors `scipy.special.polygamma`
  / `torch.special.polygamma`. Accuracy: trigamma ~1e-9 rel f64,
  polygamma(n≥2) ~1e-7 rel f64. A crate-internal *scalar*
  `multigammaln_scalar` is intentionally NOT added here — this crate
  has no host-side per-element consumer for it. The PUBLIC tensor-level
  `multigammaln` / `mvlgamma` (mirroring `torch.special.multigammaln`)
  now ships in `ferrotorch_core::special` (REQ-15 of
  `.design/ferrotorch-core/special.md`), where the torch.special public
  surface IS the consumer (goal.md S5). The earlier "NOT-STARTED,
  blocked on a Wishart consumer (R-DEFER-1)" framing for #1379 was an
  over-application: a public op mirroring `torch.special.X` needs no
  further downstream caller to be SHIPPED.

## Acceptance Criteria

- [x] AC-1: `pub(crate) fn lgamma_scalar<T: Float>(x: T) -> T` with
  Lanczos coefficients + reflection branch in `special_fns.rs`.
- [x] AC-2: `pub(crate) fn digamma_scalar<T: Float>(x: T) -> T` with
  recurrence + asymptotic expansion + reflection branch.
- [x] AC-3: Constants stored as `const LANCZOS_COEFFICIENTS: [f64;
  9]` with `#[rustfmt::skip]` to preserve column alignment.
- [x] AC-4: `mod tests` exercises both functions against ≥15 scipy
  reference pairs each at tolerance 1e-11 (f64) / 5e-6 (f32).
- [x] AC-5: `#[allow(clippy::approx_constant)]` on each reference-cases
  helper to silence the lint that fires when scipy reference values
  happen to coincide with mathematical constants like π/2.

## Architecture

### `lgamma_scalar` (REQ-1)

The Lanczos approximation is a 9-coefficient polynomial recipe
that approximates `Γ(z+1) = sqrt(2π) * (z + g + 0.5)^(z+0.5) *
exp(-(z + g + 0.5)) * A(z)` where `A(z) = c_0 + sum_k c_k/(z+k)`.
With `g = 7` and 9 coefficients, the error is ≤ 1e-14 in f64 for
positive z; ferrotorch's coefficients are the standard set used by
SciPy's `gammaln` and CPython's `math.lgamma`.

The reflection formula `Γ(x) * Γ(1-x) = π/sin(πx)` extends the
domain to `x < 0.5`:

```rust
if x < half {
    let pi = T::from(std::f64::consts::PI).unwrap();
    let sin_pi_x = (pi * x).sin();
    if sin_pi_x == zero { return T::infinity(); }
    return (pi / sin_pi_x.abs()).ln() - lgamma_scalar(one - x);
}
```

The recursive call on `lgamma_scalar(one - x)` shifts the argument
to `> 0.5`, hitting the Lanczos branch. The `.abs()` is the source
of the `log|Γ(x)|` (rather than `log Γ(x)`) contract — for negative
half-integers Γ alternates sign but `|Γ|` is monotone.

The main body computes:

```rust
let z = x - one;
let mut sum = coefficients[0];
for (i, &c) in coefficients.iter().enumerate().skip(1) {
    sum += c / (z + i as f64);
}
let t = z + g + half;
half_ln_2pi + t.ln() * (z + half) - t + sum.ln()
```

Constants:
- `LANCZOS_G = 7.0`
- `half_ln_2pi = 0.5 * ln(2π) ≈ 0.918938533204672741780329`

### `digamma_scalar` (REQ-2)

Three branches:

1. **NaN**: return NaN.
2. **Negative**: reflection — `ψ(x) = ψ(1-x) - π·cot(πx)`.
3. **Non-negative**: shift `x` upward via `ψ(x) = ψ(x+1) - 1/x`
   until `x >= 6`, then apply the asymptotic series:

```
ψ(y) ≈ ln(y) - 1/(2y) - 1/(12y²) + 1/(120y⁴) - 1/(252y⁶)
       + 1/(240y⁸) - 1/(132y¹⁰)
```

implemented as a nested Horner expansion in `y² = 1/y²` for
numerical stability. The threshold `y >= 6` keeps the asymptotic
error ≤ 1e-13 in f64.

### Shared `T: Float` parametrisation (REQ-3)

Both functions follow the same pattern:

- Constants are f64 literals declared at module scope.
- Inside the function, every literal is promoted via
  `T::from(<f64>).unwrap()` to the caller's type.
- The arithmetic flows through `T`'s operators.

This dual-precision strategy preserves the f64 precision of the
Lanczos and asymptotic coefficients (which dominate the error
budget) while still producing f32 outputs for callers operating on
f32 tensors. The `.unwrap()` is sound because `T::from(c: f64) ->
Option<T>` returns `Some` for every normal f64 representable in
the target type, and all our literals are in `[-1e9, 1e9]` for which
f32 conversion is always defined.

### `pub(crate)` visibility (REQ-4)

`pub(crate) fn lgamma_scalar` and `pub(crate) fn digamma_scalar`
+ `pub(crate) mod special_fns;` in `lib.rs:77` make the module
crate-internal. The public tensor-level analogs live in
`ferrotorch_core::special::{lgamma, digamma}` (a separate module
in a separate crate). This separation matches the upstream
PyTorch architecture: `torch.special.gammaln` is the public tensor
op, while the C++ implementations live in
`aten/src/ATen/native/Math.h` (private headers). ferrotorch
mirrors that layout — `ferrotorch_core::special::*` is the public
tensor op, `ferrotorch_distributions::special_fns::*` is the
crate-internal scalar helper.

### Non-test production consumers

Confirmed via `grep -rn "special_fns::"
ferrotorch-distributions/src/`:

- `multinomial.rs:19` — `use crate::special_fns::lgamma_scalar;`
- `student_t.rs:16` — `use crate::special_fns::{digamma_scalar,
  lgamma_scalar};`
- `weibull.rs:13` — `use crate::special_fns::lgamma_scalar;`
- `dirichlet.rs:39` — `use crate::special_fns::digamma_scalar;`
- `gamma.rs:18` — `use crate::special_fns::{digamma_scalar,
  lgamma_scalar};`
- `beta.rs:18` — `use crate::special_fns::{digamma_scalar,
  lgamma_scalar};`
- `poisson.rs:14` — `use crate::special_fns::lgamma_scalar;`
- `kumaraswamy.rs:13` — `use crate::special_fns::{digamma_scalar,
  lgamma_scalar};`
- `kl.rs:15` — `use crate::special_fns::{digamma_scalar,
  lgamma_scalar};`

9 production callers across the crate. Every formula that needs
`log|Γ|` or `ψ` invokes these scalar forms during host-side
`data_vec().iter().map(...)` pipelines.

## Parity contract

`parity_ops = []`. The functions are scalar host-side; tensor-level
parity is `ferrotorch_core::special`'s responsibility. Edge cases
preserved:

- **`lgamma(0) = +inf`**: handled by the reflection branch
  detecting `sin(πx) == 0` and returning `T::infinity()`. Matches
  `scipy.special.gammaln(0) = inf`.
- **`lgamma(negative integer) = +inf`**: same reflection-branch
  zero-sine check. Matches scipy / PyTorch.
- **`lgamma(1) == 0` and `lgamma(2) == 0`**: tested by
  `test_ln_gamma_known_values` in `kl.rs:1086` to f64 tolerance
  `1e-12`.
- **`lgamma(-0.5) ≈ 1.265512` and `lgamma(-1.5) ≈ 0.860047`**:
  reflection branch hit; tested by
  `lgamma_negative_reflection_matches_scipy`.
- **`digamma(NaN) = NaN`**: explicit `is_nan` check at the top of
  `digamma_scalar`.
- **`digamma(1) = -γ ≈ -0.577215`** (negative Euler-Mascheroni
  constant): tested by `digamma_matches_scipy_reference`.
- **f32 vs f64 precision**: `lgamma_f32_round_trip` exercises f32
  conversion across the same 15 reference points, expecting
  ~5e-6 relative tolerance.

## Verification

Tests in `mod tests in special_fns.rs` (4 tests):

- `lgamma_matches_scipy_reference` — 15 (x, expected) pairs from
  `scipy.special.gammaln` at `x ∈ {0.1, 0.25, 0.5, 0.75, 1, 1.5, 2,
  3, 4, 5, 6, 7.5, 10, 25, 100}`, tolerance `1e-11 * max(1, |expected|)`.
- `digamma_matches_scipy_reference` — same 15 (x, expected) pairs
  from `scipy.special.digamma`, tolerance `1e-9 * max(1, |expected|)`.
- `lgamma_negative_reflection_matches_scipy` — `(x, expected)` at
  `x ∈ {-0.5, -1.5}`, tolerance `1e-11`.
- `lgamma_f32_round_trip` — same 15 pairs at f32 precision,
  tolerance `5e-6 * max(1, |expected|)`.

Plus 9 indirect consumers via the distributions that import the
scalars — every `kl_gamma_gamma`, `Beta::log_prob`, `Dirichlet::entropy`,
etc. test exercises the scalar special-function path.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-distributions --lib special_fns:: 2>&1 | tail -3
```

Expected: `4 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub(crate) fn lgamma_scalar<T: Float>(x: T) -> T` with 9-coefficient Lanczos + reflection branch in `special_fns.rs`, matching scipy reference values to 1e-11 f64; non-test consumer: `kl_gamma_scalar` in `kl.rs` calls `lgamma_scalar(pa)`, `lgamma_scalar(qa)` on every Gamma-Gamma KL evaluation; `fn Gamma::log_prob in gamma.rs` calls it for the normalising constant; `fn Beta::log_prob in beta.rs` calls it 3 times for `lnB(α,β) = lnΓ(α) + lnΓ(β) - lnΓ(α+β)`. 9 production callers total. |
| REQ-2 | SHIPPED | impl: `pub(crate) fn digamma_scalar<T: Float>(x: T) -> T` with recurrence + Abramowitz-Stegun asymptotic + reflection branch in `special_fns.rs`, matching scipy reference values to 1e-9 f64; non-test consumer: `kl_gamma_scalar` in `kl.rs` calls `digamma_scalar(pa)`; `fn Beta::entropy in beta.rs` calls `digamma_scalar` on each concentration; `fn Dirichlet::entropy in dirichlet.rs` similarly. 5 production callers. |
| REQ-3 | SHIPPED | impl: both functions are `<T: Float>` generic with f64 constants promoted via `T::from(<f64>).unwrap()` in `special_fns.rs` — the `LANCZOS_COEFFICIENTS: [f64; 9]` const is f64 to preserve precision; non-test consumer: `lgamma_f32_round_trip` test exercises the f32-promotion path with 15 reference cases; in production `fn Beta::log_prob` operates on `T = f32` AND `T = f64` (both tested by `_f64` variants) routing through the same generic body. |
| REQ-4 | SHIPPED | impl: `pub(crate) fn lgamma_scalar`, `pub(crate) fn digamma_scalar`, and `pub(crate) mod special_fns;` in `lib.rs:77` together make the module crate-internal; non-test consumer: 9 production sites within `ferrotorch-distributions/src/` (`gamma.rs`, `beta.rs`, `dirichlet.rs`, `kumaraswamy.rs`, `multinomial.rs`, `poisson.rs`, `student_t.rs`, `weibull.rs`, `kl.rs`); `cargo doc -p ferrotorch-distributions` omits these from public docs by virtue of `pub(crate)`. |
| REQ-5 | SHIPPED (#1379) | impl: `pub(crate) fn trigamma_scalar<T: Float>` (recurrence + A&S 6.4.12 asymptotic + reflection) and `pub(crate) fn polygamma_scalar<T: Float>(n: u32, x: T)` (n=0 digamma kernel, n=1 trigamma, n≥2 Hurwitz-zeta) in `special_fns.rs`, matching `scipy.special.polygamma` to 1e-9 (trigamma) / 1e-7 (n≥2) f64; non-test consumer: `polygamma_scalar` calls `trigamma_scalar` from its `n == 1` arm (production caller of trigamma), and `digamma_scalar` now delegates to `polygamma_scalar(0, x)` — so the 5 existing digamma callers (`kl_gamma_scalar` in `kl.rs`, `kl_dirichlet_dirichlet` in `kl.rs`, `Beta::entropy` in `beta.rs`, `Dirichlet::entropy` in `dirichlet.rs`, `StudentT::entropy` in `student_t.rs`) transitively consume `polygamma_scalar`. Pinned by `trigamma_matches_scipy_reference`, `trigamma_negative_reflection_matches_scipy`, `trigamma_finite_difference_of_digamma`, `polygamma_general_matches_scipy_reference`, `polygamma_order0_is_digamma_order1_is_trigamma`. (Public tensor-level `multigammaln`/`mvlgamma` now SHIPPED in `ferrotorch_core::special` — REQ-15 of `.design/ferrotorch-core/special.md`; no crate-internal scalar copy needed here, no Wishart consumer required per goal.md S5.) |
