# ferrotorch-distributions — `binomial` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/binomial.py
-->

## Summary

`ferrotorch-distributions/src/binomial.rs` implements a discrete
Binomial distribution `Binomial(total_count, probs)` over the integer
interval `{0, 1, ..., n}`, where `n = total_count` is the number of
Bernoulli trials and `probs` the per-trial success probability.
Mirrors `torch.distributions.Binomial`
(`torch/distributions/binomial.py:Binomial`). It is parameterised by
either `probs` (canonical storage) or `logits`
(`Binomial::from_logits`, converted to `probs` via the binary sigmoid
`logits_to_probs`). The distribution provides `sample` (sum of `n`
Bernoulli draws), `log_prob` (the `lgamma` binomial-coefficient form),
`entropy` (finite-support enumeration), `mean = n·p`,
`variance = n·p·(1-p)`, `mode`, `support` (`integer_interval(0, n)`),
`arg_constraints`, `enumerate_support`, and explicit `rsample`
rejection (Binomial has no continuous reparameterisation).

## Requirements

- REQ-1: `pub struct Binomial<T: Float>` holding `total_count:
  Tensor<T>` and `probs: Tensor<T>`. Mirrors the `probs` branch of
  `torch/distributions/binomial.py:55-85` (`__init__`), which stores
  `self.total_count` and `self.probs` after `broadcast_all`. Generic
  over `T: Float` for the standard ferrotorch monomorphisation
  strategy (R-DEV-7).

- REQ-2: `pub fn Binomial::new(total_count: Tensor<T>, probs:
  Tensor<T>) -> FerrotorchResult<Self>` is the canonical
  (probs-parameterised) constructor, and `pub fn
  Binomial::from_logits(total_count: Tensor<T>, logits: Tensor<T>) ->
  FerrotorchResult<Self>` is the logits-parameterised constructor
  that converts via the binary sigmoid `p = 1/(1+exp(-logits))`
  (`logits_to_probs(., is_binary=True)`, `torch/distributions/utils.py`).
  Mirrors the "either `probs` or `logits`" contract of `binomial.py:62-85`.

- REQ-3: `pub fn total_count(&self) -> &Tensor<T>`, `pub fn probs(&self)
  -> &Tensor<T>`, and `pub fn logits(&self) -> FerrotorchResult<Tensor<T>>`
  accessors. `total_count`/`probs` mirror the stored attributes;
  `logits` recomputes `probs_to_logits(probs, is_binary=True) =
  ln(p) - ln(1-p)` on demand, mirroring the `@lazy_property logits`
  at `binomial.py:121-123`.

- REQ-4: `impl<T: Float> Distribution<T> for Binomial<T>` provides
  `sample`, `rsample` (error), `log_prob`, `entropy`. `sample` draws
  `n` Bernoulli outcomes per output element and sums them (mirrors the
  result distribution of `torch.binomial(total_count, probs)` at
  `binomial.py:133-138`; ferrotorch uses the explicit
  sum-of-Bernoulli construction since it has no fused `aten::binomial`
  kernel). The output is materialised on CPU then moved to the
  parameter device via `out.to(device)?` if CUDA-resident.

- REQ-5: `rsample` returns `InvalidArgument` ("Binomial distribution
  does not support reparameterized sampling. Use sample() with
  score-function estimators (REINFORCE) instead."). Binomial is
  discrete; no continuous reparameterisation exists. Mirrors the
  absence of `has_rsample = True` in `binomial.py`.

- REQ-6: `log_prob(value)` computes the exact binomial log-mass via
  `lgamma`, mirroring `binomial.py:140-158`:
  ```text
  log C(n, k) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
  log_prob(k) = log C(n, k) + k·ln(p) + (n-k)·ln(1-p)
  ```
  ferrotorch evaluates the mathematically-equivalent direct form
  (`log C(n,k) + k·ln p + (n-k)·ln(1-p)`) rather than PyTorch's
  logit-stable `normalize_term` rearrangement; the two agree to
  sub-ULP for finite `p ∈ (0, 1)` and `0 ≤ k ≤ n`. `lgamma` is the
  per-element `crate::special_fns::lgamma_scalar` (Lanczos, < 1e-12
  for x > 0.5).

- REQ-7: `entropy()` enumerates the finite support `{0..n}`, evaluates
  `log_prob` at each value, and returns `-Σ_k exp(log_prob_k)·log_prob_k`,
  mirroring `binomial.py:160-168`. Requires a homogeneous
  `total_count` across the batch (PyTorch raises `NotImplementedError`
  for "inhomogeneous total count"; ferrotorch returns `InvalidArgument`
  in that case, `binomial.py:161-165`).

- REQ-8: `mean`, `variance`, `mode` overrides for the closed-form
  properties: `mean = n·p` (`binomial.py:109-111`),
  `variance = n·p·(1-p)` (`binomial.py:117-119`),
  `mode = clamp(floor((n+1)·p), max=n)` (`binomial.py:113-115`).

- REQ-9: Full PyTorch surface flags — `has_rsample = false`,
  `has_enumerate_support = true` (`binomial.py:53`),
  `support = integer_interval(0, n)` (`binomial.py:104-107`),
  `arg_constraints = {total_count: NonnegativeInteger, probs:
  UnitInterval, logits: Real}` (`binomial.py:48-52`),
  `enumerate_support` yields `{0..n}` along dim 0 (`binomial.py:170-182`).

## Acceptance Criteria

- [x] AC-1: `pub struct Binomial<T: Float>` with `total_count`, `probs`.
- [x] AC-2: `Binomial::new(total_count, probs)` + `Binomial::from_logits`.
- [x] AC-3: `total_count()`, `probs()`, `logits()` accessors.
- [x] AC-4: `impl Distribution<T> for Binomial<T>` with the four
  required trait methods.
- [x] AC-5: `rsample` returns an `InvalidArgument` error.
- [x] AC-6: `log_prob` via `lgamma` binomial coefficient.
- [x] AC-7: `mean`, `variance`, `mode` overrides.
- [x] AC-8: `entropy` via finite-support enumeration.
- [x] AC-9: `support` / `arg_constraints` / `enumerate_support` /
  `has_enumerate_support`.
- [x] AC-10: oracle tests against live torch 2.11
  (`test_binomial_log_prob_*`, `test_binomial_mean_variance`,
  `test_binomial_entropy_*`, `test_binomial_from_logits`,
  `test_binomial_sample_*`) pin the contract end-to-end.

## Architecture

### Storage layout (REQ-1, REQ-2)

The struct stores `total_count: Tensor<T>` and `probs: Tensor<T>`.
The logits constructor converts up-front via the binary sigmoid so
the rest of the surface reads a single canonical `probs` field
(matching how PyTorch's `@lazy_property probs` materialises probs
from logits). Storing `total_count` as a tensor (not a scalar
`usize`) preserves PyTorch's per-element `total_count` broadcasting
(`binomial.py:36-39` shows a `[[5.],[10.]]` total_count example).

### `sample` as a sum of Bernoulli draws (REQ-4)

```text
for each output element (n_i, p_i):
    out_i = Σ_{t=1..n_i} [ u_t < p_i ]   ; u_t ~ Uniform(0, 1)
```

This produces exactly `Binomial(n_i, p_i)`-distributed counts (the
sum of `n_i` iid `Bernoulli(p_i)` is `Binomial(n_i, p_i)`). PyTorch's
`binomial.py:133-138` calls the fused `torch.binomial` kernel;
ferrotorch has no such leaf primitive, so the sum-of-Bernoulli
construction is the faithful CPU equivalent. `total_count` is rounded
to the nearest non-negative integer count before the draw loop.

### `log_prob` via `lgamma` (REQ-6)

```text
log_prob(k) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
              + k·ln(p) + (n-k)·ln(1-p)
```

`p` is clamped to `[eps, 1-eps]` where `eps = T::epsilon()`
(`finfo(dtype).eps`: 1.19e-7 for f32, 2.22e-16 for f64) so the
`ln(p)` / `ln(1-p)` terms stay finite for degenerate `p ∈ {0, 1}`
parameters. This matches torch's `clamp_probs`
(`torch/distributions/utils.py:124` — `eps = torch.finfo(probs.dtype).eps`),
which both the `logits` accessor (`probs_to_logits`) and `log_prob`
route through. A hardcoded dtype-independent `1e-7` over-clamps f64 by
~9 orders of magnitude and was the source of a 100× `log_prob` error
near `p = 1` (fixed under #1569). For `k ∉ {0..n}` the `lgamma(n-k+1)`
term diverges to `+∞`, giving `log_prob = -∞`, matching the
zero-mass-outside-support contract.

### `entropy` via enumeration (REQ-7)

The finite support `{0..n}` makes the closed sum
`H = -Σ_k p(k)·ln p(k)` exact. ferrotorch builds the value vector
`[0, 1, ..., n]`, calls `log_prob` on each, then folds
`-Σ exp(lp)·lp`. A non-homogeneous `total_count` (different `n`
across the batch) returns `InvalidArgument`, mirroring PyTorch's
`NotImplementedError` (`binomial.py:161-165`).

### Property overrides (REQ-8)

- `mean = n·p` (element-wise).
- `variance = n·p·(1-p)`.
- `mode = clamp(floor((n+1)·p), max=n)` — `binomial.py:113-115`.

### `from_logits` conversion (REQ-2)

```text
p = sigmoid(logit) = 1 / (1 + exp(-logit))
```

This is `logits_to_probs(logit, is_binary=True)`
(`torch/distributions/utils.py`). The inverse used by the `logits()`
accessor is `probs_to_logits(p, is_binary=True) = ln(p) - ln(1-p)`.

### Non-test production consumers

- **`pub use binomial::Binomial` in lib.rs** — `Binomial` is the
  boundary public API mirroring `torch.distributions.Binomial`; the
  re-export is the consumer per goal.md S5 (boundary methods ARE the
  public API).
- **KL-dispatcher consumer**: the dispatcher in `kl.rs` invokes
  `p.downcast_ref::<Binomial<T>>()` and `q.downcast_ref::<Binomial<T>>()`
  in the Binomial-Binomial arm, then calls `kl_binomial_binomial(p, q)`
  which reads `p.total_count().data_vec()?`, `p.probs().data_vec()?`,
  and the recomputed logits. A second arm (`Poisson, Binomial`) routes
  through `kl_infinite_like` and reads only `Poisson::rate`. Both are
  reached via the public `pub fn kl_divergence` entry, not from tests —
  they are in-crate production consumers of the `Binomial` constructor
  and accessors.

## Parity contract

`parity_ops = []`. `Binomial` mirrors a single class in
`torch/distributions/binomial.py`; the parity-sweep runner covers
tensor-level ops, not distribution-level closed-form formulas.
Conformance is exercised by the module's `mod tests` (oracle values
from live `torch.distributions.Binomial`, torch 2.11) and by the
KL gap regression tests in `kl.rs`.

Edge cases the implementation handles:

- **`p ∈ {0, 1}`**: `log_prob` clamp prevents `ln(0)`.
- **`k ∉ {0..n}`**: `lgamma(n-k+1) = +∞` ⇒ `log_prob = -∞`.
- **inhomogeneous `total_count`**: `entropy` / `enumerate_support`
  return `InvalidArgument`, mirroring PyTorch's `NotImplementedError`.
- **`f32` vs `f64`**: `T: Float` monomorphisation; both exercised by
  `test_binomial_f64`.
- **CUDA-resident params without `FERROTORCH_DIST_FALLBACK_CPU=1`**:
  return `NotImplementedOnCuda` per the crate fallback policy.

## Verification

Unit tests in the module's `mod tests` cover the contract:

- `log_prob` at known points vs live torch:
  `test_binomial_log_prob_known`, `test_binomial_log_prob_batch`.
- `mean` / `variance`: `test_binomial_mean_variance`.
- `mode`: `test_binomial_mode`.
- `entropy` vs live torch: `test_binomial_entropy_known`.
- `from_logits` round-trip: `test_binomial_from_logits`.
- `sample` shape + range: `test_binomial_sample_shape`,
  `test_binomial_sample_in_support`, `test_binomial_sample_prob_0/1`.
- `rsample` error: `test_binomial_rsample_errors`.
- `enumerate_support`: `test_binomial_enumerate_support`.
- `f64` round-trip: `test_binomial_f64`.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-distributions --lib binomial:: 2>&1 | tail -3
```

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct Binomial<T: Float>` with `total_count`/`probs`/`batch_shape` (the `broadcast_shapes(total_count, probs)` batch shape, `binomial.py:66-85`) in `binomial.rs` mirroring `torch/distributions/binomial.py:55-85`; non-test consumer: `pub use binomial::Binomial` in `lib.rs` (boundary public API per goal.md S5) — `kl.rs` dispatcher's Binomial arm binds to this type. |
| REQ-2 | SHIPPED | impl: `pub fn Binomial::new` + `pub fn Binomial::from_logits` in `binomial.rs` mirroring `binomial.py:55-85`; non-test consumer: `kl_binomial_binomial` in `kl.rs` reaches `Binomial` instances via the public `kl_divergence` dispatch; `pub use Binomial` re-export. |
| REQ-3 | SHIPPED | impl: `pub fn Binomial::{total_count, probs, logits}` in `binomial.rs` mirroring `binomial.py:109-127`; non-test consumer: `kl_binomial_binomial` in `kl.rs` reads `p.total_count().data_vec()?`, `p.probs().data_vec()?`, and recomputes per-element logits. |
| REQ-4 | SHIPPED | impl: `impl<T: Float> Distribution<T> for Binomial<T>` with `sample`/`rsample`/`log_prob`/`entropy` in `binomial.rs` mirroring `binomial.py:133-168`; `sample` returns `_extended_shape = sample_shape ++ batch_shape` (`distribution.py:266-278`) via the `broadcast_flat_index` helper; non-test consumer: `pub use Binomial` re-export — every external `Distribution`-trait call on a `Binomial` hits this impl; `tests/divergence_binomial_3d1dd0881_batch_shape.rs::divergence_binomial_sample_extends_batch_shape` pins the `[4,2]` shape (#1569). |
| REQ-5 | SHIPPED | impl: `fn Binomial::rsample` returns `InvalidArgument` in `binomial.rs` (Binomial is discrete, no `has_rsample`); non-test consumer: `pub use Binomial` re-export; `test_binomial_rsample_errors` pins it. |
| REQ-6 | SHIPPED | impl: `fn Binomial::log_prob` in `binomial.rs` with `lgamma(n+1)-lgamma(k+1)-lgamma(n-k+1)+k·ln(p)+(n-k)·ln(1-p)` mirroring `binomial.py:140-158`; `value` broadcasts against `batch_shape` (output = `broadcast(value.shape, batch_shape)`) and the clamp uses `T::epsilon()` = `clamp_probs`'s `finfo(dtype).eps` (`torch/distributions/utils.py:124`); non-test consumer: external `dist.log_prob` via the trait dispatch off `pub use Binomial`; `tests/divergence_binomial_3d1dd0881_batch_shape.rs::{divergence_binomial_log_prob_batched_probs_scalar_value, divergence_binomial_f64_clamp_eps_too_coarse}` pin batched-probs broadcast and the f64 eps (#1569). |
| REQ-7 | SHIPPED | impl: `fn Binomial::entropy` in `binomial.rs` enumerating `{0..n}` and folding `-Σ exp(lp)·lp` mirroring `binomial.py:160-168`; non-test consumer: external `dist.entropy()` via the trait dispatch. |
| REQ-8 | SHIPPED | impl: `fn Binomial::{mean, variance, mode}` overrides in `binomial.rs` mirroring `binomial.py:109-119`; non-test consumer: external `dist.{mean, variance, mode}` invocations through `pub use Binomial` hit these overrides; `test_binomial_{mean_variance, mode}` pin the closed-forms. |
| REQ-9 | SHIPPED | impl: `has_rsample`/`has_enumerate_support`/`support` (`IntegerInterval`)/`arg_constraints`/`enumerate_support` trait overrides at the tail of `impl Distribution<T> for Binomial<T>` in `binomial.rs` mirroring `binomial.py:48-53,104-107,170-182`; `enumerate_support` views values `{0..n}` as `(-1,)+(1,)*ndim(batch)` (no-expand) / `(-1,)+batch_shape` (expand) per `binomial.py:179-182`; non-test consumer: `pub use binomial::Binomial` at `lib.rs`; `test_binomial_enumerate_support` + `tests/divergence_binomial_3d1dd0881_batch_shape.rs::divergence_binomial_enumerate_support_batch_and_expand` pin the `(5,1)`/`(5,2)` shapes (#1569). |
</content>
