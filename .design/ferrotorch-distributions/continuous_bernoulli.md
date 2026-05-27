# ferrotorch-distributions — `continuous_bernoulli` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/continuous_bernoulli.py
  - torch/distributions/kl.py
-->

## Summary

`ferrotorch-distributions/src/continuous_bernoulli.rs` translates
`torch.distributions.ContinuousBernoulli`
(`torch/distributions/continuous_bernoulli.py`). The Continuous
Bernoulli is a **continuous** distribution supported on the closed unit
interval `[0, 1]`, parameterized by `probs` (in `(0, 1)`) or `logits`
(real-valued). Despite the names, `probs` is NOT a probability and
`logits` is NOT a log-odds — they are the natural parameter of an
exponential-family density whose normalizing constant is the crux of the
distribution (see Loaiza-Ganem & Cunningham, NeurIPS 2019,
arXiv:1907.06845).

The density is `p(x; λ) = C(λ) · λ^x · (1-λ)^(1-x)` where `λ = probs` and
`C(λ)` is the continuous-Bernoulli normalizing constant. ferrotorch
matches PyTorch's `log_prob` (the BCE-with-logits + log-normalizer form),
`mean`, `variance`, `entropy`, `cdf`, `icdf`, `sample`, and `rsample`
(CB is reparameterizable via inverse-CDF).

### The numerical-stability cutoff (the whole point of CB)

Every CB closed form is singular at `probs = 0.5` (the density becomes a
flat uniform there, and the `λ/(2λ-1)`, `1/(log1p(-λ) - log(λ))` factors
hit `0/0`). PyTorch guards this with `_lims = (0.499, 0.501)`: for
`probs` inside `(0.499, 0.501]` the exact formula is replaced by a Taylor
expansion about `0.5`. ferrotorch matches `_lims` and every Taylor series
byte-for-byte (R-DEV-1: numerical contract). The `_outside_unstable_region`
predicate is `probs <= 0.499 OR probs > 0.501`; inside that band the cut
value `_cut_probs = 0.499` is substituted before the exact branch is
computed (so the exact branch never divides by zero), and the Taylor
branch is selected by `torch.where`.

## Requirements

- REQ-1: `pub struct ContinuousBernoulli<T: Float>` storing the clamped
  `probs` (clamped via `clamp_probs`, `eps = finfo(dtype).eps`, matching
  `continuous_bernoulli.py:76`) and the `batch_shape` (= `probs.size()`,
  `continuous_bernoulli.py:84-87`). Mirrors
  `torch/distributions/continuous_bernoulli.py:22-89`.

- REQ-2: constructors `pub fn ContinuousBernoulli::new(probs)` and
  `pub fn ContinuousBernoulli::from_logits(logits)`. `new` clamps the
  probs (`clamp_probs`, `continuous_bernoulli.py:76`); `from_logits`
  recovers `probs = sigmoid(logits) = 1/(1+exp(-logit))`
  (`logits_to_probs(is_binary=True)`, `utils.py:97-98`) then clamps.
  Mirrors `continuous_bernoulli.py:55-89`.

- REQ-3: accessors `pub fn probs()` and `pub fn logits()`
  (`logits = probs_to_logits(probs, is_binary=True) = ln(p) - log1p(-p)`,
  the `@lazy_property logits` at `continuous_bernoulli.py:164-166`).

- REQ-4: `impl<T: Float> Distribution<T> for ContinuousBernoulli<T>` with
  `sample` / `rsample` / `log_prob` / `entropy`. `sample`/`rsample` both
  draw `u ~ Uniform(0,1)` over `_extended_shape = sample_shape ++
  batch_shape` and return `icdf(u)` (`continuous_bernoulli.py:176-185`);
  CB is reparameterizable (`has_rsample = True`,
  `continuous_bernoulli.py:53`). `log_prob` is `value·logits +
  _cont_bern_log_norm()` (the `-BCE_with_logits(logits, value) +
  log_norm` form, `continuous_bernoulli.py:187-194`). `value` is
  right-aligned broadcast against `batch_shape` (NOT `iter().cycle()`).

- REQ-5: `mean` / `variance` / `entropy` closed forms with the
  `_lims = (0.499, 0.501)` cutoff + Taylor fallback, matching
  `continuous_bernoulli.py:140-162,224-231` exactly:
    - `mean`: `λ/(2λ-1) + 1/(log1p(-λ) - log(λ))`; Taylor about 0.5:
      `0.5 + (1/3 + 16/45·(λ-0.5)²)·(λ-0.5)`.
    - `variance`: `λ(λ-1)/(1-2λ)² + 1/(log1p(-λ)-log(λ))²`; Taylor:
      `1/12 - (1/15 - 128/945·x)·x`, `x = (λ-0.5)²`.
    - `entropy`: `mean·(log1p(-λ) - log(λ)) - _cont_bern_log_norm() -
      log1p(-λ)`.
  The `_cont_bern_log_norm` helper itself
  (`continuous_bernoulli.py:120-138`) carries its own
  `log(abs(log1p(-cut) - log(cut))) - (...)` exact branch + Taylor
  `ln(2) + (4/3 + 104/45·x)·x`, `x = (λ-0.5)²`.

- REQ-6: `cdf` / `icdf` with the cutoff. `cdf`
  (`continuous_bernoulli.py:196-210`):
  `(λ^v·(1-λ)^(1-v) + λ - 1)/(2λ-1)` clamped to `[0,1]` at the support
  ends, Taylor branch `= v`; `icdf`
  (`continuous_bernoulli.py:212-222`):
  `(log1p(-λ + v·(2λ-1)) - log1p(-λ))/(log(λ) - log1p(-λ))`, Taylor
  branch `= v`. `value` broadcasts against `batch_shape`.

- REQ-7: full surface — `has_rsample` (`true`), `support`
  (`UnitInterval`), `arg_constraints` (`{probs: unit_interval, logits:
  real}`), `event_shape` (`[]`), `batch_shape`, `mode` (CB has no
  `mode`; the trait default `InvalidArgument` is kept since
  `continuous_bernoulli.py` declares none). Mirrors
  `continuous_bernoulli.py:49-53`.

- REQ-8: the CB KL pairs in `kl.rs` (`kl.md` REQ-7). PyTorch registers 13
  pairs touching `ContinuousBernoulli` (`kl.py`):
    - finite: `(CB, CB)` (`kl.py:255-260`), `(Beta, CB)`
      (`kl.py:518-525`), `(CB, Exponential)` (`kl.py:586-588`),
      `(CB, Normal)` (`kl.py:595-604`), `(CB, Uniform)`
      (`kl.py:607-617`, where-mask `+inf` when the Uniform support
      contains `[0,1]`), `(Uniform, CB)` (`kl.py:871-886`, where-mask).
    - `+inf` (`_infinite_like`): `(CB, Pareto)` (`kl.py:581-583`),
      `(Exponential, CB)` (`kl.py:621`), `(Gamma, CB)` (`kl.py:666`),
      `(Gumbel, CB)` (`kl.py:719`), `(Laplace, CB)` (`kl.py:741`),
      `(Normal, CB)` (`kl.py:762`), `(Pareto, CB)` (`kl.py:796`).
  Note PyTorch deliberately does NOT register `(CB, Beta)`, `(CB, Gamma)`,
  or `(CB, Laplace)` — they have no closed form (`kl.py:578,591,592`).

## Acceptance Criteria

- [x] AC-1: `pub struct ContinuousBernoulli<T: Float>` + `new` +
  `from_logits` ship in `continuous_bernoulli.rs`; re-exported from
  `lib.rs`.
- [x] AC-2: `mean`/`variance`/`entropy`/`cdf`/`icdf` honour the
  `_lims = (0.499, 0.501)` Taylor cutoff and match live torch f64 at and
  near `probs = 0.5`.
- [x] AC-3: `log_prob` = `value·logits + log_norm_const`, broadcasting
  `value` against `batch_shape`; matches live torch f64.
- [x] AC-4: `has_rsample = true`; `sample`/`rsample` produce values in
  `[0, 1]`.
- [x] AC-5: 13 CB KL pairs (6 finite + 7 `+inf`) ship in `kl.rs`,
  `KL_SUPPORTED_PAIR_COUNT` bumped 71 → 84, doc table + dispatcher arms +
  the two wave-audit pins moved in lockstep, `kl_doc_table_matches_dispatcher`
  green.

## Architecture

### The cutoff predicate + cut-probs

A single private `outside_unstable_region(p) = p <= 0.499 || p > 0.501`
and `cut_probs(p) = if outside { p } else { 0.499 }` mirror
`continuous_bernoulli.py:108-118`. Every exact branch is evaluated on
`cut_probs` (so it never divides by zero in the band) and the Taylor
branch on the raw `probs`; `torch.where(outside, exact, taylor)` becomes
a Rust `if outside { exact } else { taylor }` per element.

### Log-normalizing constant

`fn cont_bern_log_norm_scalar(p)` mirrors
`continuous_bernoulli.py:120-138`: exact branch
`log(|log1p(-cut) - log(cut)|) - (cut<=0.5 ? log1p(-2·cut_below) :
log(2·cut_above - 1))`, Taylor `ln(2) + (4/3 + 104/45·x)·x`, `x =
(p-0.5)²`. This scalar is the production consumer reused by `log_prob`,
`entropy`, and the CB KL formulas in `kl.rs`.

### Sampling

`sample`/`rsample` both build the `_extended_shape` (`sample_shape ++
batch_shape`), draw `u = creation::rand(shape)` (Uniform[0,1)), and apply
`icdf(u)` element-wise (`continuous_bernoulli.py:176-185`). `sample`
returns `requires_grad = false`; `rsample` builds the same values (CB is
reparameterizable — the inverse-CDF is differentiable in `probs`).

### Non-test production consumers

- `pub use continuous_bernoulli::ContinuousBernoulli` in `lib.rs` is the
  boundary public API (goal.md S5).
- The 6 finite CB KL formulas in `kl.rs` (`kl_continuous_bernoulli_*`,
  `kl_beta_continuous_bernoulli`, `kl_uniform_continuous_bernoulli`) call
  `ContinuousBernoulli::probs()` and the crate-visible scalar helpers, and
  are themselves invoked by `fn kl_dispatch` on every `kl_divergence`
  call — the in-crate production consumers of the CB surface.

## Parity contract

`parity_ops = []`. CB is a distribution surface; the parity contract is
on the elementary ops its closed forms compose (`log`, `log1p`, `pow`,
`where`), not on the CB formula itself.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | `pub struct ContinuousBernoulli<T: Float>` with `probs`/`batch_shape` fields in `continuous_bernoulli.rs` mirroring `continuous_bernoulli.py:22-89`; consumer: `pub use continuous_bernoulli::ContinuousBernoulli` in `lib.rs` + the CB KL arms in `kl.rs`. |
| REQ-2 | SHIPPED | `pub fn ContinuousBernoulli::new` (clamps via `clamp_probs`) + `pub fn ContinuousBernoulli::from_logits` (sigmoid + clamp) in `continuous_bernoulli.rs` mirroring `continuous_bernoulli.py:55-89`; consumer: `kl_continuous_bernoulli_continuous_bernoulli` reaches instances via `kl_divergence`; `pub use` re-export. |
| REQ-3 | SHIPPED | `pub fn ContinuousBernoulli::{probs, logits}` in `continuous_bernoulli.rs` mirroring `continuous_bernoulli.py:164-174`; consumer: `kl_beta_continuous_bernoulli` / `kl_uniform_continuous_bernoulli` read `q.probs()` in `kl.rs`. |
| REQ-4 | SHIPPED | `impl<T: Float> Distribution<T> for ContinuousBernoulli<T>` (`sample`/`rsample`/`log_prob`/`entropy`) in `continuous_bernoulli.rs` mirroring `continuous_bernoulli.py:176-231`; `log_prob` broadcasts `value` against `batch_shape`; consumer: trait surface via `pub use`; `test_cb_log_prob_*`. |
| REQ-5 | SHIPPED | `fn ContinuousBernoulli::{mean, variance, entropy}` with the `_lims` Taylor cutoff in `continuous_bernoulli.rs` mirroring `continuous_bernoulli.py:140-162,224-231`; consumer: trait overrides via `pub use` + CB KL formulas in `kl.rs`; `test_cb_{mean,variance,entropy}_*` (incl. near-0.5). |
| REQ-6 | SHIPPED | `fn ContinuousBernoulli::{cdf, icdf}` with the cutoff in `continuous_bernoulli.rs` mirroring `continuous_bernoulli.py:196-222`; consumer: `sample`/`rsample` call `icdf` (in-module production consumer); `test_cb_{cdf,icdf}_*`. |
| REQ-7 | SHIPPED | `has_rsample`/`support`(`UnitInterval`)/`arg_constraints`/`event_shape`/`batch_shape` overrides in `continuous_bernoulli.rs` mirroring `continuous_bernoulli.py:49-53`; consumer: `pub use`; `test_cb_{support,arg_constraints,has_rsample}`. |
| REQ-8 | SHIPPED | 13 CB KL pairs in `kl.rs` (6 finite `fn kl_continuous_bernoulli_*`/`kl_beta_continuous_bernoulli`/`kl_uniform_continuous_bernoulli` + 7 `+inf` via `kl_infinite_like`) mirroring `kl.py:255,518,581,586,595,607,621,666,719,741,762,796,871`; consumer: each invoked by its `fn kl_dispatch` downcast arm reached via `pub fn kl_divergence`; `test_kl_cb_*` (live-torch 2.11 f64). |
