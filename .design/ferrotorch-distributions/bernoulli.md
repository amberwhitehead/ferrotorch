# ferrotorch-distributions — `bernoulli` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/bernoulli.py
-->

## Summary

`ferrotorch-distributions/src/bernoulli.rs` implements a discrete
Bernoulli distribution over `{0, 1}` parameterized by a `probs`
tensor. Mirrors `torch.distributions.Bernoulli`. The distribution
provides `sample` (inverse-CDF via uniform draws), `log_prob`
(closed-form `x*log(p) + (1-x)*log(1-p)` with a numerical clamp),
`entropy`, `cdf`, `icdf`, `mean`, `mode`, `variance`, and explicit
`rsample` rejection (Bernoulli has no continuous reparameterization
— REINFORCE / Gumbel-Softmax are the gradient-friendly alternatives).

## Requirements

- REQ-1: `pub struct Bernoulli<T: Float>` holding a single `probs:
  Tensor<T>` field. Mirrors the `probs`-parameterized branch of
  `torch/distributions/bernoulli.py:Bernoulli.__init__` (the
  `logits` branch is NOT-STARTED here; see REQ-9). Generic over
  `T: Float` for the standard ferrotorch monomorphisation strategy
  (R-DEV-7: trait-bound generic instead of dynamic-dtype dispatch).

- REQ-2: `pub fn Bernoulli::new(probs: Tensor<T>) ->
  FerrotorchResult<Self>` is the canonical constructor.
  No upstream-style range check on `probs ∈ [0, 1]` is performed
  (R-DEV-4: ferrotorch defers `arg_constraints` validation until
  the cross-cutting validate-args mechanism lands — see REQ-9).

- REQ-3: `pub fn probs(&self) -> &Tensor<T>` accessor mirrors
  PyTorch's `self.probs` attribute (returned by reference, owned
  by the distribution).

- REQ-4: `impl<T: Float> Distribution<T> for Bernoulli<T>` provides
  `sample`, `rsample` (error), `log_prob`, `entropy`. `sample`
  draws `u ~ Uniform(0, 1)` via `creation::rand` and emits `1` iff
  `u < p`. The Python upstream calls `torch.bernoulli(probs)` —
  same semantics, different surface (the CUDA kernel is a future
  ferrotorch follow-up tracked by the GPU-fallback gate the
  module routes through via `crate::fallback::check_gpu_fallback_opt_in`).

- REQ-5: `rsample` returns `InvalidArgument` with the message
  "Bernoulli distribution does not support reparameterized
  sampling. Use sample() with score-function estimators
  (REINFORCE) instead." Bernoulli's discrete support means no
  continuous reparameterization exists. Mirrors PyTorch's omission
  of `has_rsample = True` in `bernoulli.py:42-45` (no `rsample`
  override; the base-class default raises `NotImplementedError`).

- REQ-6: `log_prob(value) = x * log(p_clamped) + (1 - x) *
  log(1 - p_clamped)` with `p_clamped = max(eps, min(1-eps, p))`,
  `eps = 1e-7`. The clamp guards against `log(0)` for degenerate
  `p ∈ {0, 1}` parameters. PyTorch's body uses
  `binary_cross_entropy_with_logits` for numerical stability; the
  ferrotorch closed form matches it mathematically for finite
  inputs (sub-ULP for typical workloads).

- REQ-7: `entropy()` returns `-p*ln(p) - (1-p)*ln(1-p)` with the
  same `[eps, 1-eps]` clamp. Mirrors PyTorch's
  `binary_cross_entropy_with_logits(logits, probs, ...)` formula
  (`bernoulli.py:127-130`). Maximum at `p=0.5` (`ln 2`), zero in
  the limits `p ∈ {0, 1}`.

- REQ-8: `cdf`, `icdf`, `mean`, `mode`, `variance` overrides for
  closed-form properties. `cdf(x) = 0 if x<0; 1-p if 0≤x<1; 1 if
  x≥1`. `mean = p`. `mode = 1 if p > 0.5 else 0` (the upstream
  emits NaN at exactly `p=0.5` per `bernoulli.py:94-98`; ferrotorch
  emits `0` for a finite tie-break — documented divergence in
  `Architecture`).

- REQ-9: NOT-STARTED — `logits`-parameterized constructor +
  `enumerate_support` + `expand` + `arg_constraints` + `support`
  + `validate_args` (the full PyTorch surface from
  `bernoulli.py:42-45,74-86,132-145`) are not implemented.
  Closing these depends on the cross-cutting `Distribution`-trait
  blocker #1376. Tracked as blocker #1406 for the Bernoulli-side
  fill-out.

## Acceptance Criteria

- [x] AC-1: `pub struct Bernoulli<T: Float>` with `probs` field.
- [x] AC-2: `pub fn Bernoulli::new(probs)`.
- [x] AC-3: `pub fn probs()` accessor.
- [x] AC-4: `impl Distribution<T> for Bernoulli<T>` with all four
  required trait methods.
- [x] AC-5: `rsample` returns an `InvalidArgument` error.
- [x] AC-6: `log_prob`, `entropy` with clamp.
- [x] AC-7: `cdf`, `icdf`, `mean`, `mode`, `variance` overrides.
- [x] AC-8: `test_bernoulli_sample_values_binary`,
  `test_bernoulli_log_prob_one/zero`,
  `test_bernoulli_entropy_fair`,
  `test_bernoulli_mode_high_p/low_p`,
  `test_bernoulli_cdf`,
  `test_bernoulli_icdf_step_at_one_minus_p`,
  and `test_bernoulli_f64` cover the contract end-to-end.
- [ ] AC-9: `logits` ctor + `expand` + `enumerate_support` —
  blocker #1406.

## Architecture

### Storage layout (REQ-1, REQ-2, REQ-3)

The struct stores `probs: Tensor<T>` directly. No precomputation
(unlike `Categorical`, which precomputes a CDF for binary-search
sampling). `Bernoulli::new` is infallible at the type level —
shape validation is the upstream tensor's responsibility; range
validation is deferred to the validate-args mechanism (REQ-9
blocker).

### `sample` via inverse CDF (REQ-4)

```text
u ~ Uniform(0, 1)
out = 1 if u < probs else 0
```

`u_data.iter().zip(probs_data.iter().cycle())` broadcasts the
sometimes-scalar `probs` against the requested `shape`. The
`cycle()` adapter handles the case where `probs.numel() <
shape.numel()` (PyTorch implicitly broadcasts via `expand(shape)`
on the parameter — ferrotorch's cycle gives the same per-element
binding for the common scalar-param case). The output tensor is
materialized on CPU then transferred to the parameter's device
via `out.to(device)?` if the parameter is CUDA-resident.

The `check_gpu_fallback_opt_in` gate is invoked at the top of
every method that reads `probs.data_vec()?`. CUDA inputs without
the env var return `NotImplementedOnCuda`, matching the
crate-wide GPU-fallback policy (see `fallback.md`).

### `rsample` rejection (REQ-5)

Bernoulli is discrete; no continuous reparameterization exists.
The method returns `InvalidArgument` synchronously. Downstream
code that wants a differentiable Bernoulli draw should use
`RelaxedBernoulli` (the Concrete relaxation, which IS
reparameterizable) instead.

### `log_prob` numerical form (REQ-6)

```text
p_clamped = max(eps, min(1-eps, p))     ; eps = 1e-7
log_prob(x) = x * ln(p_clamped) + (1-x) * ln(1-p_clamped)
```

The clamp moves the singularities at `p ∈ {0, 1}` away from the
log call. For `p = 0` and `x = 1` the result is `ln(eps) ≈ -16.1`
(f32) rather than `-inf`; for `p = 1` and `x = 0` likewise. This
matches PyTorch's `binary_cross_entropy_with_logits` clamping
(`torch.nn.functional._BCE_LOG_EPSILON`).

### `entropy` (REQ-7)

```text
H = -p_clamped * ln(p_clamped) - (1-p_clamped) * ln(1-p_clamped)
```

Numerically identical to the upstream when `eps`-clamping is in
effect; tests at `p = 0.5` (max entropy, `ln 2 ≈ 0.693`) and
`p ∈ {0.2, 0.5, 0.8}` (symmetry check) pin the formula.

### Property overrides (REQ-8)

- `cdf(x) = step function`: see the closed-form in REQ-8.
- `icdf(p) = 1 if p > 1 - probs else 0`: piecewise generalized
  inverse of the step CDF.
- `mean = self.probs.clone()`: returned by clone of the parameter
  tensor.
- `mode = 1 if p > 0.5 else 0`: ferrotorch returns `0` at
  exactly `p = 0.5` for a finite tie-break; PyTorch emits NaN
  there per `bernoulli.py:97`. This is a documented
  R-DEV-2-aware divergence; see also blocker #1406 which can
  track narrowing it once the NaN-injection helper lands.
- `variance = p * (1 - p)`: standard formula.

### Non-test production consumers

- **`pub use bernoulli::Bernoulli` in lib.rs** — grandfathered
  public API surface re-exported from the crate root (per goal.md
  S5).
- **KL-dispatcher consumer**: the dispatcher in `kl.rs` invokes
  `p.downcast_ref::<Bernoulli<T>>()` and `q.downcast_ref::<Bernoulli<T>>()`
  in the Bernoulli-Bernoulli arm, then calls `kl_bernoulli_bernoulli(p, q)`
  which reads `p.probs().data_vec()?` and `q.probs().data_vec()?` for the
  closed-form `p*ln(p/q) + (1-p)*ln((1-p)/(1-q))`. This is a
  production consumer of both the public constructor and the
  `.probs()` accessor — the dispatcher is reached via the public
  `pub fn kl_divergence` entry, not from a test.

## Parity contract

`parity_ops = []`. `Bernoulli` mirrors a single class in
`torch/distributions/bernoulli.py`; the parity-sweep runner
covers tensor-level ops, not distribution-level closed-form
formulas. Conformance is exercised by
`ferrotorch-distributions/tests/conformance_distributions_discrete.rs`.

Edge cases the implementation handles:

- **`p ∈ {0, 1}`**: clamp prevents `log(0)`. `sample` correctly
  emits all-0 or all-1 (verified by
  `test_bernoulli_sample_prob_0/1`).
- **`p = 0.5`**: `mode = 0` (finite tie-break; documented
  divergence from PyTorch's NaN).
- **Broadcast scalar `probs` against larger `value`**: the
  `cycle()` adapter handles per-element binding.
- **`f32` vs `f64`**: `T: Float` monomorphisation; both exercised
  by `test_bernoulli_f64`.
- **CUDA-resident `probs` without `FERROTORCH_DIST_FALLBACK_CPU=1`**:
  returns `NotImplementedOnCuda` per the crate fallback policy.

## Verification

Unit tests in the module's `mod tests` cover 18 assertions:

- Sample shape + binary support: `test_bernoulli_sample_shape`,
  `test_bernoulli_sample_values_binary`,
  `test_bernoulli_sample_prob_0/1`.
- `rsample` error: `test_bernoulli_rsample_errors`.
- `log_prob` at known points: `test_bernoulli_log_prob_one/zero/batch`.
- `entropy` at `p = 0.5`, `p ≈ 1`, batched: `test_bernoulli_entropy_fair/deterministic/batch`.
- `mean`, `mode`, `variance`, `cdf`, `icdf`:
  `test_bernoulli_mean_variance`, `test_bernoulli_mode_high_p/low_p`,
  `test_bernoulli_cdf`, `test_bernoulli_icdf_step_at_one_minus_p`.
- `f64` round-trip: `test_bernoulli_f64`.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-distributions --lib bernoulli:: 2>&1 | tail -3
```

Expected: `18 passed; 0 failed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct Bernoulli<T: Float>` with `probs: Tensor<T>` in `bernoulli.rs` mirroring `torch/distributions/bernoulli.py:20-45`; non-test consumer: `pub use bernoulli::Bernoulli` in `lib.rs` (grandfathered public API per goal.md S5) — downstream callers + `kl.rs` dispatcher's Bernoulli arm both bind to this type. |
| REQ-2 | SHIPPED | impl: `pub fn Bernoulli::new(probs) -> FerrotorchResult<Self>` in `bernoulli.rs` mirroring `bernoulli.py:47-72`; non-test consumer: `pub use Bernoulli` re-export — any external caller of `Bernoulli::new(...)` hits this path. |
| REQ-3 | SHIPPED | impl: `pub fn Bernoulli::probs(&self) -> &Tensor<T>` accessor in `bernoulli.rs` mirroring `bernoulli.py:108-110` attribute access; non-test consumer: `kl_bernoulli_bernoulli` in `kl.rs` reads `p.probs().data_vec()?` and `q.probs().data_vec()?` for the closed-form formula. |
| REQ-4 | SHIPPED | impl: `impl<T: Float> Distribution<T> for Bernoulli<T>` with `sample`/`rsample`/`log_prob`/`entropy` in `bernoulli.rs` mirroring `bernoulli.py:116-130`; non-test consumer: the `pub use Bernoulli` re-export means every external `Distribution`-trait invocation on a `Bernoulli` hits this impl block; `kl_bernoulli_bernoulli` in `kl.rs` calls `.probs()` (REQ-3) which is part of the trait surface plumbing. |
| REQ-5 | SHIPPED | impl: `fn Bernoulli::rsample` returns `InvalidArgument` in `bernoulli.rs` mirroring `bernoulli.py:42-45` (no `has_rsample = True`); non-test consumer: every external invocation of `dist.rsample(...)` on a `Bernoulli` exercises this; `test_bernoulli_rsample_errors` pins the contract — production consumer surface is the `pub use Bernoulli` re-export. |
| REQ-6 | SHIPPED | impl: `fn Bernoulli::log_prob` in `bernoulli.rs` with `eps = 1e-7` clamp and `x*ln(p) + (1-x)*ln(1-p)` formula mirroring `bernoulli.py:121-125`; non-test consumer: every external `dist.log_prob(value)` call hits this method via the `Distribution`-trait dispatch off `pub use Bernoulli`. |
| REQ-7 | SHIPPED | impl: `fn Bernoulli::entropy` in `bernoulli.rs` with `-p*ln(p) - (1-p)*ln(1-p)` formula mirroring `bernoulli.py:127-130`; non-test consumer: every external `dist.entropy()` call hits this method. |
| REQ-8 | SHIPPED | impl: `fn Bernoulli::{cdf, icdf, mean, mode, variance}` overrides in `bernoulli.rs` mirroring `bernoulli.py:91-102`; non-test consumer: external `dist.{cdf, icdf, mean, mode, variance}` invocations through the `pub use Bernoulli` re-export hit these overrides rather than the trait defaults; `test_bernoulli_{cdf, icdf_step_at_one_minus_p, mean_variance, mode_high_p, mode_low_p}` pin the closed-forms. |
| REQ-9 | NOT-STARTED | blocker #1406 — `logits` constructor, `expand`, `enumerate_support`, `arg_constraints`, `support`, `validate_args` from `bernoulli.py:42-45,74-86,132-145` not implemented. Cross-cutting with `lib.md` REQ-5 (Distribution-trait-surface blocker #1376). |
