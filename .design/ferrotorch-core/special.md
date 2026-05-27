# torch.special Functions

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/
  - c10/
  - torch/_torch_docs.py
  - torch/overrides.py
-->

## Summary

`ferrotorch-core/src/special.rs` implements `torch.special.*` —
special-function families including the error functions (`erf`,
`erfc`, `erfinv`), the gamma family (`lgamma`, `digamma`), numerically-
stable identities (`log1p`, `expm1`, `sinc`, `xlogy`), and the
orthogonal-polynomial families (Chebyshev T/U/V/W, Hermite H/He,
Laguerre L, Legendre P, shifted Chebyshev T/U/V/W). The scalar
evaluators sit inside the module (SunPro fdlibm `erf` / `erfc` for
f64, Abramowitz-Stegun 7.1.26 for f32; Lanczos for `lgamma`;
Winitzki rational for `erfinv`; three-term recurrences for the
polynomial families).

## Requirements

- REQ-1: `erf(input)` — error function, ULP-accurate per f64 SunPro
  fdlibm and ~1.5e-7 per f32 Abramowitz-Stegun. Mirrors
  `torch.special.erf` / `torch.erf`.
- REQ-2: `erfc(input)` — complementary error function. f64 path uses
  `erfc_f64_hi` directly (avoids cancellation in `1 - erf(x)` for
  large positive x); f32/bf16 use `1 - erf_scalar`. Mirrors
  `torch.special.erfc`.
- REQ-3: `erfinv(input)` — inverse error function via Winitzki (2008)
  rational. Returns `±inf` for `±1`, `NaN` for `|x| > 1`. Mirrors
  `torch.special.erfinv`.
- REQ-4: `lgamma(input)` — `log(|Gamma(x)|)` via Lanczos (g=7, n=9).
  Mirrors `torch.special.gammaln` / `torch.lgamma`.
- REQ-5: `digamma(input)` — `d/dx ln Gamma(x)` via shift-up-then-
  asymptotic-expansion. Mirrors `torch.special.digamma`.
- REQ-6: `log1p(input)` / `expm1(input)` — numerically-stable
  `log(1+x)` / `exp(x)-1` via `num_traits::Float::ln_1p` /
  `exp_m1`. Mirrors `torch.log1p` / `torch.expm1`.
- REQ-7: `sinc(input)` — normalised sinc: `sin(pi*x)/(pi*x)` with
  `sinc(0) = 1`. Mirrors `torch.special.sinc`.
- REQ-8: `xlogy(x, y)` — `x * log(y)` with `xlogy(0, y) = 0`
  convention (matches the entropy-computation use). Mirrors
  `torch.special.xlogy`.
- REQ-9: Chebyshev polynomial family — `chebyshev_polynomial_{t,u,v,w}`
  via three-term recurrence. Mirrors
  `torch.special.chebyshev_polynomial_{t,u,v,w}`. CPU-only;
  GPU path NOT-STARTED, blocked on #1545.
- REQ-10: Hermite polynomial family — `hermite_polynomial_h`
  (physicist's, `H_{n+1} = 2x H_n - 2n H_{n-1}`),
  `hermite_polynomial_he` (probabilist's, `He_{n+1} = x He_n - n
  He_{n-1}`). Mirrors `torch.special.hermite_polynomial_h` /
  `hermite_polynomial_he`.
- REQ-11: Laguerre + Legendre — `laguerre_polynomial_l`,
  `legendre_polynomial_p` via standard three-term recurrences.
  Mirrors `torch.special.laguerre_polynomial_l` /
  `legendre_polynomial_p`.
- REQ-12: Shifted Chebyshev family — `shifted_chebyshev_polynomial_{t,u,v,w}`
  evaluating `T_n(2x - 1)` etc. Mirrors
  `torch.special.shifted_chebyshev_polynomial_{t,u,v,w}`.
- REQ-13: Incomplete-gamma family — `gammainc(input, other)` (regularized
  lower `P(a, x)`) and `gammaincc(input, other)` (regularized upper
  `Q(a, x) = 1 - P`). Numerical-Recipes `gammp`/`gammq` interior (power series
  for `x < a + 1`, Lentz continued fraction for `x >= a + 1`) wrapped with
  PyTorch's exact boundary contract (`calc_igamma`/`calc_igammac`: NaN for
  negatives, `a=0,x>0 -> 1` / `0`, `x=0 -> 0` / `1`, infinity cases). Mirrors
  `torch.special.gammainc` / `torch.special.gammaincc`.
- REQ-14: Log-beta / beta — `log_beta(a, b) = lgamma(a) + lgamma(b) -
  lgamma(a + b)` and `beta(a, b) = exp(log_beta(a, b))`, computed in f64 then
  narrowed. Mirrors `scipy.special.betaln` / `scipy.special.beta` (the `lbeta`
  PyTorch users compose from `torch.lgamma`).
- REQ-15: Multivariate log-gamma — `multigammaln(input, p)` and its alias
  `mvlgamma(input, p)` computing `(p(p-1)/4) ln(π) + Σ_{i=1}^p lgamma(a +
  (1-i)/2)`. Domain `a > (p-1)/2` (NaN outside); `p >= 1` (error for `p == 0`).
  Mirrors `torch.special.multigammaln` / `torch.mvlgamma`.
- REQ-16: Gamma sign — `gammaln_sign(input)` returning the `±1`/`NaN` factor
  that `lgamma = ln|Γ|` discards: `+1` for `x > 0`, `NaN` at negative-integer
  poles, `(-1)^floor(x)` for `x < 0` non-integer. Mirrors
  `scipy.special.gammasgn`.

## Acceptance Criteria

- [x] AC-1: `cargo test -p ferrotorch-core --lib special::tests` passes
  (covers `erf`/`erfc`/`erfinv` round-trips, `lgamma` known values,
  `digamma`, polynomial recurrence sanity).
- [x] AC-2: `erf(0.0) == 0.0`, `erf(inf) == 1.0`, `erf(-inf) == -1.0`.
- [x] AC-3: `xlogy(0.0, 0.0) == 0.0` (special convention).
- [x] AC-4: `lgamma(1.0) == 0.0` and `lgamma(2.0) == 0.0`.
- [x] AC-5: `chebyshev_polynomial_t(x, 0) == 1` and
  `chebyshev_polynomial_t(x, 1) == x` for all x.
- [x] AC-6: `hermite_polynomial_h(x, 2)` matches `4*x^2 - 2`.
- [ ] AC-7: GPU lowering for the polynomial families — NOT-STARTED,
  blocked on #1545 (CubeCL kernel for on-device three-term
  recurrence).
- [x] AC-8: `gammainc(a, x) + gammaincc(a, x) == 1` for `a, x > 0`, and the
  boundary table matches live `torch.special.gammainc`/`gammaincc`
  (`gammainc(0, x>0) == 1`, `gammainc(a>0, 0) == 0`, NaN for negatives).
- [x] AC-9: `multigammaln(a, 1) == lgamma(a)`; `multigammaln` / `log_beta` /
  `gammaln_sign` match live `scipy.special.multigammaln` / `betaln` /
  `gammasgn` oracle values to f64 tolerance.

## Architecture

The scalar evaluators live in this module:

- `erf_scalar` at `special.rs:400` is `pub(crate)` so
  `grad_fns::activation` can reuse it (the gelu derivative path
  calls `crate::special::erf_scalar`).
- `erfc_scalar` (private, referenced at `:685`): for f64, directly
  evaluates `erfc_f64_hi`; for f32/bf16, returns `1 - erf_scalar(x)`.
- `erfinv_scalar`, `lgamma_scalar`, `digamma_scalar`, `sinc_scalar`,
  `xlogy_scalar`: each private, dispatched through
  `unary_map` / `binary_map` from `crate::ops::elementwise`.
- `hermite_h`, `hermite_he`, `chebyshev_{t,u,v,w}`, `laguerre_l`,
  `legendre_p` (`:918-...`): private three-term-recurrence loops in
  f64. The polynomial public APIs all delegate to a private
  `elementwise_f64` helper at `special.rs:770` that asserts CPU,
  reads `data_vec`, maps each element through the recurrence, and
  builds the result tensor.

The public functions at `special.rs:670-912` are uniformly
single-line dispatches via `unary_map(input, scalar_fn)` or
`elementwise_f64(input, op, |x| poly(n, x))`.

**Non-test consumers**:

- `crate::grad_fns::activation::erf_for_gelu` at `grad_fns/activation.rs:413`
  invokes `crate::special::erf_scalar` directly — this is the gelu
  derivative path's most important consumer of REQ-1.
- Re-exported at `lib.rs:187` as the top-level
  `ferrotorch_core::{digamma, erf, erfc, erfinv, expm1, lgamma,
  log1p, sinc, xlogy}`. The polynomial families are accessed via
  the `special::` module path: `ferrotorch_core::special::chebyshev_polynomial_t`.

## Parity contract

`parity_ops = []` (no specific parity-sweep op declared; the
underlying torch ops are checked by their `unary_map` /
`binary_map` parents). Numerical contract:

- ULP accuracy targets: f64 transcendentals at `1e-10`, f32 at
  `1e-5` (constants `F64_TRANSCENDENTAL` / `F32_TRANSCENDENTAL` in
  the test module).
- Domain edge cases: `erfinv(±1) = ±inf`, `erfinv(|x|>1) = NaN`,
  `xlogy(0, anything) = 0`, `sinc(0) = 1`.
- The polynomial recurrences are exact to the recurrence's
  numerical-stability limits (typically n ≤ 50 for the families
  used in ML; higher n may drift).

## Verification

`cargo test -p ferrotorch-core --lib special::tests` covers all 12
families. The `tests` mod includes known-value comparisons (e.g.
`lgamma(2.0) = 0`, `chebyshev_polynomial_t(0.5, 3) = -0.25`) and
recurrence-identity checks (`H_n(x) - 2x*H_{n-1}(x) + 2(n-1)*H_{n-2}(x) =
0`).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `erf` at `special.rs:675` mirrors `torch.special.erf`; non-test consumer: `crate::grad_fns::activation::erf_for_gelu` at `grad_fns/activation.rs:413` invokes `crate::special::erf_scalar` (the per-element evaluator that `erf` uses internally) — this is the gelu-derivative path's direct dependency |
| REQ-2 | SHIPPED | impl: `erfc` at `special.rs:684`; non-test consumer: re-exported as `ferrotorch_core::erfc` at `lib.rs:187` |
| REQ-3 | SHIPPED | impl: `erfinv` at `special.rs:692`; non-test consumer: re-exported as `ferrotorch_core::erfinv` at `lib.rs:187` |
| REQ-4 | SHIPPED | impl: `lgamma` at `special.rs:699`; non-test consumer: re-exported as `ferrotorch_core::lgamma` at `lib.rs:187` |
| REQ-5 | SHIPPED | impl: `digamma` at `special.rs:707`; non-test consumer: re-exported as `ferrotorch_core::digamma` at `lib.rs:187` |
| REQ-6 | SHIPPED | impl: `log1p`/`expm1` at `special.rs:714,721`; non-test consumer: re-exported as `ferrotorch_core::log1p`/`expm1` at `lib.rs:187` |
| REQ-7 | SHIPPED | impl: `sinc` at `special.rs:726`; non-test consumer: re-exported as `ferrotorch_core::sinc` at `lib.rs:187` |
| REQ-8 | SHIPPED | impl: `xlogy` at `special.rs:733`; non-test consumer: re-exported as `ferrotorch_core::xlogy` at `lib.rs:187` |
| REQ-9 | SHIPPED | impl: `chebyshev_polynomial_{t,u,v,w}` at `special.rs:794-832`; non-test consumer: accessible via `ferrotorch_core::special::chebyshev_polynomial_*`. GPU lowering NOT-STARTED, blocked on #1545 — does NOT block CPU SHIPPED |
| REQ-10 | SHIPPED | impl: `hermite_polynomial_h` / `hermite_polynomial_he` at `special.rs:841,849`; non-test consumer: accessible via `ferrotorch_core::special::hermite_polynomial_*` |
| REQ-11 | SHIPPED | impl: `laguerre_polynomial_l` / `legendre_polynomial_p` at `special.rs:859,867`; non-test consumer: accessible via `ferrotorch_core::special::laguerre_polynomial_l` / `legendre_polynomial_p` |
| REQ-12 | SHIPPED | impl: `shifted_chebyshev_polynomial_{t,u,v,w}` at `special.rs:875-908`; non-test consumer: accessible via `ferrotorch_core::special::shifted_chebyshev_polynomial_*` |
| REQ-13 | SHIPPED | impl: `pub fn gammainc` / `pub fn gammaincc` in `special.rs` (boundary kernels `calc_igamma_f64` / `calc_igammac_f64` over NR core `gammp_core_f64` / `gammq_core_f64`) mirror `torch.special.gammainc` / `gammaincc` (`aten/src/ATen/native/Math.h:1144 calc_igamma`, `:1085 calc_igammac`); non-test consumer: re-exported as top-level `ferrotorch_core::gammainc` / `ferrotorch_core::gammaincc` in `lib.rs` — per goal.md S5 the public `torch.special` surface IS the consumer (boundary API needs no further downstream caller) |
| REQ-14 | SHIPPED | impl: `pub fn log_beta` / `pub fn beta` in `special.rs` (scalar `log_beta_scalar` / `beta_scalar`) mirror `scipy.special.betaln` / `beta`; non-test consumer: re-exported as `ferrotorch_core::log_beta` / `ferrotorch_core::beta` in `lib.rs` (S5 public-surface consumer) |
| REQ-15 | SHIPPED | impl: `pub fn multigammaln` + alias `pub fn mvlgamma` in `special.rs` (scalar `multigammaln_scalar`) mirror `torch.special.multigammaln` / `torch.mvlgamma` (`aten/src/ATen/native/UnaryOps.cpp:887 mvlgamma`, domain `a > (p-1)/2` per `torch/special/__init__.py:862`); non-test consumer: re-exported as `ferrotorch_core::multigammaln` / `ferrotorch_core::mvlgamma` in `lib.rs` (S5 public-surface consumer) |
| REQ-16 | SHIPPED | impl: `pub fn gammaln_sign` in `special.rs` (scalar `gammaln_sign_scalar`) mirrors `scipy.special.gammasgn`; non-test consumer: re-exported as `ferrotorch_core::gammaln_sign` in `lib.rs` (S5 public-surface consumer) |
