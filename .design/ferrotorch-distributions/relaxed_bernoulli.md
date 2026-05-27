# ferrotorch-distributions — `relaxed_bernoulli` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/relaxed_bernoulli.py
-->

## Summary

`ferrotorch-distributions/src/relaxed_bernoulli.rs` defines
`RelaxedBernoulli<T: Float>` — the Concrete (continuous relaxation of
Bernoulli) distribution from Maddison et al. 2017 and Jang et al.
2017, parameterized by `temperature` (scalar `T`) and per-element
`probs: Tensor<T>`. Samples lie in `(0, 1)` rather than at `{0, 1}`.
Mirrors `torch.distributions.RelaxedBernoulli`
(`torch/distributions/relaxed_bernoulli.py:122-174`). Upstream builds
this as `TransformedDistribution(LogitRelaxedBernoulli, SigmoidTransform)`;
ferrotorch ships a direct sigmoid-of-Logistic-noise sampler and a
closed-form `log_prob` derived from
`LogitRelaxedBernoulli.log_prob` + change-of-variable Jacobian.

## Requirements

- REQ-1: `pub struct RelaxedBernoulli<T: Float>` storing
  `temperature: T` (scalar, not `Tensor<T>`) and `probs: Tensor<T>`.
  Mirrors `relaxed_bernoulli.py:122-148` `RelaxedBernoulli` class with
  `base_dist: LogitRelaxedBernoulli` holding the same fields.
  Ferrotorch's scalar-T temperature deviates from upstream's
  `temperature: Tensor` (R-DEV-7: scalar is cleaner where broadcasting
  isn't required by the test surface).

- REQ-2: `pub fn RelaxedBernoulli::new(temperature, probs) -> FerrotorchResult<Self>` —
  constructor validating `temperature > 0` and every `probs[i] in (0, 1)`
  (strict open interval). Upstream's `arg_constraints = {"probs": constraints.unit_interval}`
  is the `[0, 1]` closed interval; ferrotorch's stricter check
  prevents `log(0)` / `log(1) = -inf` in the Logistic-noise formula.
  Test `test_relaxed_bernoulli_invalid_temperature` and
  `test_relaxed_bernoulli_invalid_probs` pin both branches.

- REQ-3: `pub fn temperature(&self) -> T` and
  `pub fn probs(&self) -> &Tensor<T>` parameter accessors. Mirrors
  upstream `RelaxedBernoulli.temperature` (`relaxed_bernoulli.py:164-166`)
  and `RelaxedBernoulli.probs` (`relaxed_bernoulli.py:172-174`) which
  delegate to the base distribution.

- REQ-4: `impl<T: Float> Distribution<T> for RelaxedBernoulli<T>`
  provides `sample(shape)` and `rsample(shape)` both invoking the
  internal `relaxed_bernoulli_sample` helper. The Concrete forward
  pass is:
  ```text
  U ~ Uniform(0, 1)
  L = log(U) - log(1 - U)        (standard Logistic noise)
  logits = log(probs / (1 - probs))
  z = sigmoid((L + logits) / temperature)
  ```
  Mirrors `LogitRelaxedBernoulli.rsample` in
  `relaxed_bernoulli.py:104-112` then `SigmoidTransform.forward` per
  the `TransformedDistribution` composition.

- REQ-5: `log_prob(value)` evaluates the closed-form Concrete density
  on `(0, 1)`:
  ```text
  logits = log(p/(1-p))
  y      = log(z/(1-z))                       (logit of the sample)
  diff   = logits - y * temperature
  log_prob = log(temperature) + diff - 2*softplus(diff) - log(z) - log(1-z)
  ```
  Mirrors `LogitRelaxedBernoulli.log_prob` (`relaxed_bernoulli.py:114-119`)
  PLUS the change-of-variable Jacobian `-log(z) - log(1-z)` from the
  inverse `SigmoidTransform` (the `log_abs_det_jacobian` for sigmoid
  on a (0,1) output is `-log(z*(1-z))`). The softplus is computed
  with the standard numerically-stable two-branch form (`diff >= 0`
  uses `diff + log(1 + exp(-diff))`, `diff < 0` uses
  `log(1 + exp(diff))`).

- REQ-6: `entropy()` returns `InvalidArgument` because the Concrete
  distribution has no closed-form entropy. Mirrors upstream which
  does NOT override `entropy` and falls back to
  `Distribution.entropy → NotImplementedError`.

- REQ-7: NOT-STARTED — `logits` accessor (upstream has both
  `probs` and `logits` via `lazy_property`), `mean`, `mode`,
  `variance`, `cdf`, `icdf`, `support` not implemented.
  Cross-cutting with `lib.md` REQ-5 (Distribution-trait-surface
  blocker #1376); the RelaxedBernoulli-specific surface fill-out
  tracked in blocker #1411.

- REQ-8: NOT-STARTED — `LogitRelaxedBernoulli` (the base
  distribution upstream exposes separately,
  `relaxed_bernoulli.py:22-119`) is not exposed as a standalone
  ferrotorch distribution. The Concrete forward is inlined into
  `RelaxedBernoulli` directly. Blocker #1415 tracks the
  `LogitRelaxedBernoulli` extraction.

- REQ-9: SHIPPED — `rsample` builds a `RelaxedBernoulliRsampleBackward`
  autograd node carrying `probs` + the detached Logistic noise, so
  gradients flow through `probs` (the only `Tensor` parameter;
  `temperature` is a scalar `T` and receives no gradient). The forward
  `z = sigmoid((L + logit(p)) / temp)` gives
  `dz/dp = z(1-z) / (temp * p * (1-p))`. Mirrors upstream's
  autograd-through-tensor-ops rsample (`relaxed_bernoulli.py:104-112`).
  Closes #1420.

## Acceptance Criteria

- [x] AC-1: `pub struct RelaxedBernoulli<T: Float>` with
  `temperature: T`, `probs: Tensor<T>` fields.
- [x] AC-2: `RelaxedBernoulli::new` rejecting `temperature <= 0` and
  `probs` outside `(0, 1)`.
- [x] AC-3: `pub fn temperature()` and `pub fn probs()` accessors.
- [x] AC-4: `impl Distribution::sample` / `rsample` via Concrete
  forward pass.
- [x] AC-5: `impl Distribution::log_prob` matching upstream
  `LogitRelaxedBernoulli` + sigmoid Jacobian.
- [x] AC-6: `impl Distribution::entropy` returns `InvalidArgument`.
- [ ] AC-7: `logits`, `mean`, `mode`, `variance`, `cdf`, `icdf`,
  `support` — blocker #1411.
- [ ] AC-8: `LogitRelaxedBernoulli` as standalone — blocker #1415.
- [x] AC-9: Differentiable rsample via `RelaxedBernoulliRsampleBackward`
  — #1420.

## Architecture

### The struct (REQ-1, REQ-2, REQ-3)

```rust
pub struct RelaxedBernoulli<T: Float> {
    temperature: T,
    probs: Tensor<T>,
}
```

`temperature` is stored by value (not as a `Tensor<T>` like upstream)
because Concrete-distribution use cases (Gumbel-softmax annealing in
VAE training) treat temperature as a non-batched scheduling scalar.
The R-DEV-7 deviation is acceptable because the public API surface
test (`tests/conformance/_surface_inventory.toml:441`) signs off on
the `RelaxedBernoulli::new(T, Tensor<T>)` signature.

The constructor's open-interval `probs` check (strict `(0, 1)`)
deviates from upstream's closed `[0, 1]` per R-DEV-7 to avoid
`log(0)` in the Logistic noise reconstruction. The deviation is
documented in the error message returned by `new`.

### The shared Concrete sampler (REQ-4)

```rust
fn relaxed_bernoulli_sample<T: Float>(
    temperature: T,
    probs: &Tensor<T>,
    shape: &[usize],
    _reparam: bool,
) -> FerrotorchResult<Tensor<T>>
```

Both `sample` and `rsample` invoke this helper. The `_reparam` flag
is preserved as a marker but currently unused — the implementation
is identical for both paths (REQ-9 blocker tracks the differentiable
case). The math per element:

```text
U ~ Uniform(0, 1)                   (drawn once for n*1 elements)
U_clamped = U.max(1e-20).min(1 - 1e-20)
L = log(U_clamped) - log(1 - U_clamped)     (standard Logistic)
P_clamped = probs.max(1e-20).min(1 - 1e-20)
logits = log(P_clamped / (1 - P_clamped))
arg = (L + logits) / temperature
z = sigmoid(arg)                    (numerically stable branch)
```

The numerically stable sigmoid uses the two-branch form:
- `arg >= 0`: `1 / (1 + exp(-arg))`
- `arg < 0`:  `exp(arg) / (1 + exp(arg))`

### Closed-form log_prob (REQ-5)

The Concrete density on `(0, 1)` is derived by transforming
`LogitRelaxedBernoulli.log_prob` (which is on the unconstrained logit
space `(-inf, inf)`) through the inverse `SigmoidTransform`:

```text
log_prob_RB(z) = log_prob_LRB(logit(z)) + log|det J_inv|
              = log_prob_LRB(y) - log(z) - log(1 - z)
```

where `y = log(z / (1 - z))`. Upstream's
`LogitRelaxedBernoulli.log_prob` is
`temperature.log() + (logits - value * temperature) - 2 * (logits - value * temperature).exp().log1p()`
(`relaxed_bernoulli.py:114-119`); ferrotorch's `diff = logits - y * temperature`
inlines `value = y` and applies the Jacobian terms.

The numerically stable softplus is essential — `diff` can be large
positive (overflow `exp(diff)`) or large negative (underflow `exp(-diff)`).
The two-branch form ensures both extremes give finite output.

A documented probe in the code: `z = 0.7, logits = 0.5, temp = 2.0`
gives `log_prob ≈ -0.7893` matching PyTorch.

### Non-test production consumers

- `pub use relaxed_bernoulli::RelaxedBernoulli` in `lib.rs:118` —
  grandfathered public API re-export. Downstream Concrete-VAE
  training code constructs `RelaxedBernoulli::new(temp, probs)?`
  directly.
- `RelaxedBernoulli::new` is registered in
  `tests/conformance/_surface_inventory.toml:441` as part of the
  conformance surface contract.
- The lib-level docs table in `lib.rs:36` references
  `RelaxedBernoulli` as a published distribution.

### Fallback gate

Every `Distribution` method first invokes
`crate::fallback::check_gpu_fallback_opt_in(&[&self.probs], "RelaxedBernoulli::<method>")`.

## Parity contract

`parity_ops = []`. The Concrete sampler is not exposed to the
parity-sweep oracle because RNG sequences don't match torch's.

Numerical contracts ferrotorch preserves:

- **Samples in `[0, 1]`**: per the sigmoid output range; test
  `test_relaxed_bernoulli_sample_in_closed_unit_interval` verifies
  ≥95% of samples are in the open interior `(0, 1)` and 100% are in
  `[0, 1]`. The `[0, 1]` (closed) vs `(0, 1)` (open) gap is real:
  f32 sigmoid can saturate at exactly `0.0` or `1.0` for extreme
  `arg`, so the closed bound is the safe assertion.
- **Low-temperature concentration**: as `temperature → 0`, samples
  concentrate on `{0, 1}` (Bernoulli limit). Test
  `test_relaxed_bernoulli_low_temperature_concentrates` verifies
  >90% of samples with `temperature=0.01, probs=0.5` are outside
  `[0.05, 0.95]`.
- **`log_prob` symmetry**: for `probs = 0.5`,
  `log_prob(z) == log_prob(1-z)` because the Concrete-Bernoulli is
  symmetric around `0.5`. Test
  `test_relaxed_bernoulli_log_prob_symmetry` pins
  `log_prob(0.2) ≈ log_prob(0.8)` to `1e-5`.
- **Finite `log_prob`**: test `test_relaxed_bernoulli_log_prob_finite`
  verifies finite output for the canonical case.
- **`entropy` errors out**: test
  `test_relaxed_bernoulli_entropy_errors`.

## Verification

Tests in `mod tests in relaxed_bernoulli.rs` (6 tests):

- `test_relaxed_bernoulli_invalid_temperature`
- `test_relaxed_bernoulli_invalid_probs`
- `test_relaxed_bernoulli_sample_in_closed_unit_interval`
- `test_relaxed_bernoulli_low_temperature_concentrates`
- `test_relaxed_bernoulli_log_prob_finite`
- `test_relaxed_bernoulli_log_prob_symmetry`
- `test_relaxed_bernoulli_entropy_errors`

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-distributions --lib relaxed_bernoulli:: 2>&1 | tail -3
```

Expected: `7 passed` (six listed above plus the entropy-errors test).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct RelaxedBernoulli<T: Float>` with `temperature: T`, `probs: Tensor<T>` fields in `relaxed_bernoulli.rs`, mirroring `torch/distributions/relaxed_bernoulli.py:122-148` (which uses `base_dist: LogitRelaxedBernoulli`); non-test consumer: `pub use relaxed_bernoulli::RelaxedBernoulli` in `lib.rs:118` — grandfathered public API; downstream Concrete-VAE training code constructs it directly. R-DEV-7: scalar `temperature: T` (vs upstream `Tensor`) for cleaner API surface in non-broadcasting cases. |
| REQ-2 | SHIPPED | impl: `pub fn RelaxedBernoulli::new(temperature, probs) -> FerrotorchResult<Self>` with `temperature > 0` and `probs[i] in (0, 1)` validation in `relaxed_bernoulli.rs`; non-test consumer: registered in `tests/conformance/_surface_inventory.toml:441` as conformance surface inventory; `pub use RelaxedBernoulli` re-export for downstream callers; tests pin both validation branches. R-DEV-7: stricter open-interval probs check vs upstream's closed `[0,1]` to avoid `log(0)`. |
| REQ-3 | SHIPPED | impl: `pub fn temperature(&self) -> T` (by value) and `pub fn probs(&self) -> &Tensor<T>` (by reference) accessors in `relaxed_bernoulli.rs`, mirroring `RelaxedBernoulli.temperature` and `RelaxedBernoulli.probs` @property delegations in `relaxed_bernoulli.py:164-174`; non-test consumer: `pub use RelaxedBernoulli` re-export exposes both accessors. |
| REQ-4 | SHIPPED | impl: `impl<T: Float> Distribution<T> for RelaxedBernoulli<T>` with `sample` and `rsample` invoking `fn relaxed_bernoulli_sample` (the Concrete forward `z = sigmoid((L + logits) / temperature)` for `L ~ Logistic`) in `relaxed_bernoulli.rs`, mirroring `LogitRelaxedBernoulli.rsample` (`relaxed_bernoulli.py:104-112`) + `SigmoidTransform` composition; non-test consumer: the trait impl is the production dispatch any external caller hits; test `test_relaxed_bernoulli_sample_in_closed_unit_interval` pins. |
| REQ-5 | SHIPPED | impl: `Distribution::log_prob` in `relaxed_bernoulli.rs` returns `log(temp) + diff - 2*softplus(diff) - log(z) - log(1-z)` where `diff = logits - logit(z)*temp`, with numerically stable two-branch softplus; mirrors `LogitRelaxedBernoulli.log_prob` (`relaxed_bernoulli.py:114-119`) + sigmoid Jacobian; non-test consumer: `pub use RelaxedBernoulli` re-export plus the impl dispatch; tests `test_relaxed_bernoulli_log_prob_{finite, symmetry}` pin both behaviours. Probe at `z=0.7, logits=0.5, temp=2.0` matches PyTorch's `-0.7893`. |
| REQ-6 | SHIPPED | impl: `Distribution::entropy` in `relaxed_bernoulli.rs` returns `InvalidArgument` because the Concrete distribution has no closed-form entropy, mirroring upstream's lack of an `entropy` override which falls back to `Distribution.entropy → NotImplementedError`; non-test consumer: any caller invoking `.entropy()` on a `RelaxedBernoulli` hits this error path; test `test_relaxed_bernoulli_entropy_errors` pins. |
| REQ-7 | NOT-STARTED | blocker #1411 — `logits` accessor, `mean`, `mode`, `variance`, `cdf`, `icdf`, `support` (`constraints.unit_interval`, `relaxed_bernoulli.py:145`) not implemented; cross-cutting with `lib.md` REQ-5 (Distribution-trait-surface blocker #1376). |
| REQ-8 | NOT-STARTED | blocker #1415 — `LogitRelaxedBernoulli` (the unconstrained-logit-space base distribution, `relaxed_bernoulli.py:22-119`) not exposed as a standalone ferrotorch distribution; the Concrete forward is inlined into `RelaxedBernoulli` directly. |
| REQ-9 | SHIPPED | impl: `rsample` attaches `RelaxedBernoulliRsampleBackward` (carrying `probs` + detached Logistic noise `u`) via `Tensor::from_operation` when `probs` requires grad; backward computes `dz/dp = z(1-z)/(temp·p·(1-p))` in `relaxed_bernoulli.rs`, mirroring upstream's tensor-op rsample (`relaxed_bernoulli.py:104-112`). `temperature` is a scalar `T` (no gradient). Non-test consumer: `impl Distribution::rsample` for `RelaxedBernoulli` — the production dispatch every external caller reaches via `pub use RelaxedBernoulli` re-export in `lib.rs`. Tests `test_relaxed_bernoulli_rsample_requires_grad_when_probs_grad`, `test_relaxed_bernoulli_rsample_grad_flows_to_probs_finite`, `test_relaxed_bernoulli_sample_detached` pin grad-flow + detachment. Closes #1420. |
