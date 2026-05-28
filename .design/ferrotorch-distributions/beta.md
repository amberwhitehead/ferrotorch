# ferrotorch-distributions — `beta` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/beta.py
-->

## Summary

`ferrotorch-distributions/src/beta.rs` implements the Beta
distribution on `[0, 1]` parameterized by two positive
concentration parameters (`concentration1` = alpha,
`concentration0` = beta). Mirrors `torch.distributions.Beta`.
Sampling uses the Gamma-ratio reparameterization
(`Beta(a,b) = Gamma(a,1) / (Gamma(a,1) + Gamma(b,1))`). The
`rsample` path carries gradients through the implicit
reparameterization of both wrapped Gamma draws — a custom
`BetaRsampleBackward` `GradFn` accumulates the per-element
chain-rule contribution and surfaces gradients into both
concentration tensors.

## Requirements

- REQ-1: `pub struct Beta<T: Float>` holding two equally-shaped
  `Tensor<T>` fields, `concentration1` (alpha) and
  `concentration0` (beta). Mirrors the surface that
  `torch/distributions/beta.py:Beta.__init__` exposes via its
  Dirichlet-of-two construction.

- REQ-2: `pub fn Beta::new(concentration1: Tensor<T>,
  concentration0: Tensor<T>) -> FerrotorchResult<Self>` is the
  constructor with a shape-equality check. Upstream broadcasts via
  `broadcast_all(concentration1, concentration0)`; the
  ferrotorch constructor instead requires exact shape match and
  errors with `ShapeMismatch` otherwise. Pre-broadcasting is the
  caller's responsibility (R-DEV-4: Rust's explicit-types
  philosophy moves the broadcast out of construction).

- REQ-3: `pub fn concentration1(&self) -> &Tensor<T>` and
  `pub fn concentration0(&self) -> &Tensor<T>` accessors mirror
  `Beta.concentration1` / `Beta.concentration0` properties
  (`beta.py:96-110`).

- REQ-4: `impl<T: Float> Distribution<T> for Beta<T>` provides
  `sample` / `rsample` / `log_prob` / `entropy` plus the property
  overrides `mean` / `mode` / `variance`. `sample` and `rsample`
  delegate to two wrapped `Gamma::new(concentrationN, ones)`
  instances and form the ratio `xa / (xa + xb)`.

- REQ-5: `sample` uses `crate::Gamma::new` to construct two
  ancillary Gamma(α, 1) and Gamma(β, 1) distributions, draws
  scalar samples from each, then forms the ratio `xa / (xa + xb)`
  per element. The Gamma-of-α-1 + Gamma-of-β-1 ratio is the
  classical Beta(α, β) reparameterization. Mirrors PyTorch's
  approach via `Dirichlet([α, β]).rsample(...)`. The zero-sum
  guard `if sum == 0 { 0.5 }` handles the (numerically-impossible
  but mathematically-defined) degenerate case.

- REQ-6: `rsample` uses `Gamma::rsample` (NOT `Gamma::sample`)
  so the implicit-reparameterization gradient propagates through
  the wrapped Gammas. When either concentration parameter
  `requires_grad`, a `BetaRsampleBackward` node is attached.
  Tiny-guard `if sum < 1e-30 { 1e-30 }` replaces the zero-sum
  branch from `sample` (rsample must avoid `0 / 0` to keep
  gradient signals finite).

- REQ-7: `log_prob(value) = (α-1)*ln(x) + (β-1)*ln(1-x) -
  lbeta(α, β)` where `lbeta(α, β) = lgamma(α) + lgamma(β) -
  lgamma(α+β)`. Mirrors PyTorch's Dirichlet-based formula
  (`beta.py:87-91` invokes `self._dirichlet.log_prob(heads_tails)`
  which expands to the same closed form). Uses
  `crate::special_fns::lgamma_scalar`.

- REQ-8: `entropy = lbeta(α, β) - (α-1)*ψ(α) - (β-1)*ψ(β) +
  (α+β-2)*ψ(α+β)` where `ψ` is the digamma function via
  `crate::special_fns::digamma_scalar`. Standard closed form;
  mirrors PyTorch's `self._dirichlet.entropy()` reduction
  (`beta.py:93-94`).

- REQ-9: Closed-form properties: `mean = α/(α+β)`,
  `mode = (α-1)/(α+β-2) if α>1 ∧ β>1 else NaN`,
  `variance = αβ / ((α+β)² * (α+β+1))`. Mirrors
  `beta.py:71-82`.

- REQ-10: `BetaRsampleBackward` `GradFn` implements the
  implicit-reparameterization gradient through the Gamma ratio.
  For each element:
  ```text
  d(out)/d(γa) = γb / (γa+γb)²
  d(out)/d(γb) = -γa / (γa+γb)²
  d(γa)/d(α)  = standard_gamma_grad_one(α, γa)   ; pathwise
  d(γb)/d(β)  = standard_gamma_grad_one(β, γb)
  ```
  Each per-Gamma term is the PATHWISE implicit-reparameterization
  gradient `d(γ)/d(conc) = -(∂_conc P(conc, γ)) / pdf(γ; conc)`
  computed by `pub(crate) fn standard_gamma_grad_one` in
  `standard_gamma_grad_one in ferrotorch-distributions/src/special_fns.rs` (port of
  PyTorch's `_standard_gamma_grad`). This replaced the prior
  score-function form `γ * (ln(γ) - ψ(conc))`, which is unbiased
  only in expectation and flips sign per-sample (fixed by commit
  fae8ca185, closes #1555). Summed across the output tensor to
  produce scalar `grad_concentration1` and `grad_concentration0`.

- REQ-11: NOT-STARTED — `expand`, `arg_constraints`, `support`,
  `validate_args`, the alternative-stacked-Dirichlet
  parameterization the upstream `__init__` accepts when called
  with scalars, and the `_natural_params` / `_log_normalizer`
  exponential-family hooks (`beta.py:113-118`) are not
  implemented. Cross-cutting with `lib.md` REQ-5
  Distribution-trait surface (blocker #1376). Tracked as
  blocker #1408 for the Beta-side fill-out.

## Acceptance Criteria

- [x] AC-1: `pub struct Beta<T: Float>` with `concentration1` /
  `concentration0` fields.
- [x] AC-2: `pub fn Beta::new` with shape-equality check.
- [x] AC-3: `pub fn concentration1` / `concentration0` accessors.
- [x] AC-4: `impl Distribution<T> for Beta<T>` with all four
  required trait methods + closed-form property overrides.
- [x] AC-5: `sample` / `rsample` via Gamma ratio.
- [x] AC-6: `log_prob` closed form using `lgamma_scalar`.
- [x] AC-7: `entropy` closed form using `digamma_scalar`.
- [x] AC-8: `BetaRsampleBackward` GradFn with implicit reparam.
- [x] AC-9: `test_beta_sample_in_unit_interval`,
  `test_beta_sample_mean`, `test_beta_rsample_has_grad`,
  `test_beta_log_prob_symmetric/known`, `test_beta_entropy`,
  `test_beta_mean_variance_mode`,
  `test_beta_mode_undefined_for_alpha_le_one` cover the contract.
- [ ] AC-10: `expand` / `arg_constraints` / `validate_args` —
  blocker #1408.

## Architecture

### Constructor + shape contract (REQ-1, REQ-2, REQ-3)

The struct stores `concentration1`, `concentration0` directly.
`Beta::new` enforces `shape(concentration1) ==
shape(concentration0)` upfront; this is strictly tighter than
PyTorch's `broadcast_all`-then-stack approach but downstream code
that wants batched Beta with different parameter shapes can
broadcast at the call site.

### Gamma-ratio sampling (REQ-5, REQ-6)

`sample` and `rsample` both:

1. Build `ones = creation::scalar(1.0)`.
2. Construct `Gamma::new(self.concentration1.clone(), ones.clone())?`
   and `Gamma::new(self.concentration0.clone(), ones)?`.
3. Draw `xa`, `xb` of shape `shape` from each.
4. Form `result[i] = xa[i] / (xa[i] + xb[i])`.

The `sample` path uses `Gamma::sample` (no gradient); the
`rsample` path uses `Gamma::rsample` so gradients flow through
each Gamma's implicit reparameterization. The two wrapped Gamma
distributions are constructed inside the method (not stored as
struct fields) because the `concentration` tensors may have
been cloned with `requires_grad` since the `Beta` was first
created, and we always want fresh Gammas that mirror the current
grad state of the parameter tensors.

The `Gamma::new` call is the load-bearing internal production
consumer of the Gamma type — `beta.rs` lines 77, 78, 114, 115
all construct `Gamma` instances. This is consumer evidence for
the Gamma module (cited from there as well).

### `log_prob` (REQ-7) and `entropy` (REQ-8)

The standard Beta log-PDF reads
`log f(x; α, β) = (α-1) ln x + (β-1) ln(1-x) - ln B(α, β)`
where `B(α, β) = Γ(α) Γ(β) / Γ(α+β)` is the Beta function. Both
methods build per-element scalar formulas using
`lgamma_scalar` / `digamma_scalar` from `crate::special_fns`
(internal CPU Lanczos approximation). PyTorch routes the same
math through Dirichlet (since `Beta(α,β) ≡ Dirichlet([α,β])
projected to the first coordinate`); ferrotorch computes the
formula directly to skip the Dirichlet machinery overhead.

### `BetaRsampleBackward` (REQ-10)

The custom `struct BetaRsampleBackward<T: Float>` at
`ferrotorch-distributions/src/beta.rs:362` is the key piece of
the rsample contract. It owns clones of `concentration1`,
`concentration0`, `gamma_a` (the realized Gamma(α, 1) samples),
and `gamma_b` (the realized Gamma(β, 1) samples). On backward:

- Chains the ratio's local Jacobian (`γb/(γa+γb)²`,
  `-γa/(γa+γb)²`) with the per-Gamma PATHWISE implicit-reparam
  gradient `standard_gamma_grad_one(conc, γ)` from
  `ferrotorch-distributions/src/special_fns.rs:134` — NOT the
  former score-function form `γ·(ln γ - ψ(conc))`, which was
  replaced in commit fae8ca185 (#1555).
- Sums into scalar accumulators per concentration parameter,
  yielding a single-element gradient tensor of the same shape as
  the input parameter.
- Returns `None` entries when a parameter doesn't require grad,
  per the standard `GradFn` contract.

### Property overrides (REQ-9)

Standard closed forms — see REQ-9. The mode formula returns NaN
for `α ≤ 1 ∨ β ≤ 1` (where the mode is at the boundary, not in
the interior). Test `test_beta_mode_undefined_for_alpha_le_one`
pins the NaN result for `Beta(1, 1)`.

### Non-test production consumers

- **`pub use beta::Beta` in lib.rs** — grandfathered public
  surface (goal.md S5).
- **`Distribution` trait dispatch via `pub use Beta`** — any
  external caller that holds a `Beta<T>` and calls the trait
  methods hits this impl block. Tests `test_beta_*` exercise this
  path; production callers (downstream VI / Bayesian-NN training
  loops, importance-sampling routines) use the same surface.

## Parity contract

`parity_ops = []`. Beta is a closed-form distribution; the
parity-sweep covers tensor ops, not distribution-level formulas.
Conformance is verified by
`ferrotorch-distributions/tests/conformance_distributions_continuous.rs`.

Edge-case coverage:

- **`α=β=1` (Uniform on [0,1])**: `log_prob(0.5) ≈ 0`,
  `entropy ≈ 0`, mode = NaN. Pinned by
  `test_beta_log_prob_symmetric` and `test_beta_entropy`.
- **`α=2, β=3` (peaked near 1/3)**: `log_prob(0.5) = ln(1.5)`.
  Pinned by `test_beta_log_prob_known`.
- **`α=2, β=5`**: mean = 2/7, var = 10/(49·8), mode = 1/5.
  Pinned by `test_beta_mean_variance_mode`.
- **Sum-of-Gammas zero**: `sample` returns 0.5 (mathematical
  midpoint); `rsample` returns `1e-30 / 1e-30 = 1` after the
  tiny-guard kicks in.
- **`f64`**: `test_beta_f64`.

## Verification

Unit tests in `mod tests` cover 12 scenarios:

- `test_beta_sample_shape`,
  `test_beta_sample_in_unit_interval` — shape + range invariants.
- `test_beta_sample_mean` — Monte Carlo mean check.
- `test_beta_rsample_has_grad` — autograd hook.
- `test_beta_log_prob_symmetric`,
  `test_beta_log_prob_known` — analytical log_prob values.
- `test_beta_entropy` — closed-form entropy.
- `test_beta_shape_mismatch` — constructor error.
- `test_beta_f64` — dtype generic.
- `test_beta_mean_variance_mode`,
  `test_beta_mode_undefined_for_alpha_le_one` — closed-form
  properties + NaN-mode tie-break.

Smoke command:

```bash
cargo test -p ferrotorch-distributions --lib beta:: 2>&1 | tail -3
```

Expected: `12 passed; 0 failed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct Beta<T: Float>` with `concentration1`, `concentration0` in `beta.rs` mirroring `torch/distributions/beta.py:15-61`; non-test consumer: `pub use beta::Beta` in `lib.rs` (grandfathered public surface per goal.md S5). |
| REQ-2 | SHIPPED | impl: `pub fn Beta::new` in `beta.rs` with shape-equality check mirroring `beta.py:41-61`; non-test consumer: `pub use Beta` re-export. |
| REQ-3 | SHIPPED | impl: `pub fn Beta::concentration1` / `concentration0` accessors in `beta.rs` mirroring `beta.py:96-110`; non-test consumer: re-exported via `pub use Beta`; external callers using `dist.concentration1()` for diagnostic or KL-style introspection exercise these. |
| REQ-4 | SHIPPED | impl: `impl<T: Float> Distribution<T> for Beta<T>` in `beta.rs` mirroring `beta.py:84-94, 71-82`; non-test consumer: every external trait invocation through `pub use Beta` re-export. |
| REQ-5 | SHIPPED | impl: `fn Beta::sample` in `beta.rs` constructing two `crate::Gamma::new(...)` instances and forming the ratio, mirroring PyTorch's Dirichlet-based equivalent at `beta.py:84-85`; non-test consumer: external `dist.sample(shape)` calls through the trait dispatch. |
| REQ-6 | SHIPPED | impl: `fn Beta::rsample` in `beta.rs` invoking `Gamma::rsample` (so implicit reparam flows) and attaching `BetaRsampleBackward` when grad enabled; non-test consumer: external `dist.rsample(shape)` calls; `test_beta_rsample_has_grad` pins the grad attachment. |
| REQ-7 | SHIPPED | impl: `fn Beta::log_prob` in `beta.rs` with `(α-1)*ln(x) + (β-1)*ln(1-x) - lbeta(α,β)` formula via `lgamma_scalar`, mirroring `beta.py:87-91`; non-test consumer: external `dist.log_prob(value)` calls. |
| REQ-8 | SHIPPED | impl: `fn Beta::entropy` in `beta.rs` with closed-form using `digamma_scalar`, mirroring `beta.py:93-94`; non-test consumer: external `dist.entropy()` calls. |
| REQ-9 | SHIPPED | impl: `fn Beta::{mean, mode, variance}` overrides in `beta.rs` mirroring `beta.py:71-82`; non-test consumer: external `dist.{mean, mode, variance}()` calls exercise the overrides; `test_beta_mean_variance_mode` and `test_beta_mode_undefined_for_alpha_le_one` pin the closed-forms. |
| REQ-10 | SHIPPED | impl: `struct BetaRsampleBackward<T: Float>` at `BetaRsampleBackward in ferrotorch-distributions/src/beta.rs` whose `GradFn::backward` chains the ratio Jacobian with the per-Gamma PATHWISE implicit-reparam gradient `pub(crate) fn standard_gamma_grad_one` at `standard_gamma_grad_one in ferrotorch-distributions/src/special_fns.rs` (port of `torch._standard_gamma_grad`) — replacing the prior score-function form `γ·(ln γ - ψ(conc))` in commit fae8ca185 (#1555); non-test consumer: invoked by `fn Beta::rsample` (REQ-6) whenever either parameter requires grad. |
| REQ-11 | NOT-STARTED | blocker #1408 — `expand`, `arg_constraints`, `support`, `validate_args`, scalar-broadcast `__init__` branch, `_natural_params` / `_log_normalizer` (from `beta.py:63-69, 113-118`) not implemented. Cross-cutting with `lib.md` REQ-5 (Distribution-trait-surface blocker #1376). |
