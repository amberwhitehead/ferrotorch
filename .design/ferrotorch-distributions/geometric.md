# ferrotorch-distributions — `geometric` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/geometric.py
  - torch/distributions/utils.py
  - torch/distributions/kl.py
-->

## Summary

`ferrotorch-distributions/src/geometric.rs` translates
`torch.distributions.Geometric`. A `Geometric(probs)` is the distribution of
the **number of failures** `k ∈ {0, 1, 2, ...}` before the first success in a
sequence of independent Bernoulli trials with per-trial success probability
`probs` (so `P(X=k) = (1-p)^k · p`, `geometric.py:26-27`). It is discrete and
has no reparameterization (`rsample` errors).

The implementation mirrors the FIXED `binomial.rs` batch-broadcast contract:
`batch_shape` is recorded at construction and `sample`/`log_prob` route through
right-aligned broadcast indexing (`row_major_strides` + `broadcast_flat_index`),
NOT the `iter().cycle()` pattern that the Binomial critic (#1569) flagged as
dropping batched params / failing to broadcast `value`.

## Requirements

- REQ-1: `pub struct Geometric<T: Float>` with `probs: Tensor<T>` and
  `batch_shape: Vec<usize>` (= `probs.shape()`, `geometric.py:74`). Mirrors
  `torch/distributions/geometric.py:20-87`.

- REQ-2: Constructors. `pub fn Geometric::new(probs)` (probs branch,
  `geometric.py:60-62`) and `pub fn Geometric::from_logits(logits)` using the
  binary sigmoid `p = 1/(1+exp(-l))` (`logits_to_probs(is_binary=True) =
  torch.sigmoid`, `utils.py:97-98`; `geometric.py:63-67`).

- REQ-3: Accessors. `pub fn Geometric::probs(&self) -> &Tensor<T>` and
  `pub fn Geometric::logits(&self) -> FerrotorchResult<Tensor<T>>` where
  `logits = probs_to_logits(probs, is_binary=True) = ln(p) - log1p(-p)`
  (`utils.py:135-137`, `geometric.py:112-114`), clamped to `[eps, 1-eps]` with
  `eps = finfo(dtype).eps = T::epsilon()` (`utils.py:124`).

- REQ-4: `impl<T: Float> Distribution<T> for Geometric<T>`:
    - `sample` = inverse-CDF `(u.log() / (-p).log1p()).floor()`, `u ~
      Uniform(tiny, 1)`, `tiny = finfo(dtype).tiny = T::min_positive_value()`
      (`geometric.py:120-130`). Output shape = `_extended_shape(sample_shape) =
      sample_shape ++ batch_shape` (`distribution.py:266-278`).
    - `log_prob(k)` = `k·log1p(-p) + ln(p)` with the `probs==1 & value==0 -> 0`
      clamp that forces `log1p(-p)` to 0 there to dodge the `0·(-inf)` NaN
      (`geometric.py:132-138`). `value` is right-aligned broadcast against
      `batch_shape`.
    - `entropy` = `binary_cross_entropy_with_logits(logits, probs)/probs`
      (`geometric.py:140-144`), computed via the numerically stable
      `max(ℓ,0) - ℓ·t + log1p(exp(-|ℓ|))` BCE-with-logits form (target `t = p`).

- REQ-5: `rsample` returns `InvalidArgument` — Geometric is discrete, no
  `has_rsample` attr at `geometric.py`.

- REQ-6: `log_prob` as in REQ-4 (broadcast-correct).

- REQ-7: `entropy` as in REQ-4.

- REQ-8: Property overrides `mean = (1-p)/p` (= `1/p - 1`,
  `geometric.py:100-102`), `variance = (1-p)/p²` (`geometric.py:108-110`),
  `mode = 0` (`geometric.py:104-106`).

- REQ-9: Full surface: `has_rsample = false`, `support =
  nonnegative_integer`, `arg_constraints = {probs: unit_interval, logits:
  real}`, `event_shape = []`, `batch_shape = probs.shape()`
  (`geometric.py:46-48,74`).

KL wiring (in `kl.rs`, REQ-7 of `kl.md`):

- `fn kl_geometric_geometric` = `-p.entropy() - log1p(-q.probs)/p.probs -
  q.logits` mirroring `torch/distributions/kl.py:320-322`
  (`@register_kl(Geometric, Geometric)`). `q.logits = ln(q) - log1p(-q)`.
  Wired as a dispatcher arm; `KL_SUPPORTED_PAIR_COUNT` bumps 70 → 71 with the
  doc-table row + the two wave audit count-pins in lockstep.

## Acceptance Criteria

- [x] AC-1: `pub struct Geometric<T: Float>` + `new`/`from_logits`.
- [x] AC-2: `probs`/`logits` accessors with the binary sigmoid/logit
  conventions.
- [x] AC-3: `Distribution` impl (`sample`/`rsample`/`log_prob`/`entropy`).
- [x] AC-4: `mean`/`variance`/`mode`/`support`/`arg_constraints`/`batch_shape`
  overrides.
- [x] AC-5: batch-broadcast correctness — `log_prob` of a scalar value against
  batched `probs=[0.3,0.5]` returns one row per param
  (`test_geometric_log_prob_batched_probs`), and `sample` of a batched
  Geometric returns `sample_shape ++ batch_shape`
  (`test_geometric_sample_batched_shape`).
- [x] AC-6: `kl_geometric_geometric` wired + `kl_divergence(Geometric,
  Geometric)` matches live torch f64.

## Architecture

### Batch-broadcast (the critical contract)

`batch_shape = probs.shape()` (`geometric.py:74`). `sample` draws uniforms over
`out_shape = sample_shape ++ batch_shape` and broadcasts `probs` into it via
`broadcast_flat_index`. `log_prob` broadcasts `value` against `batch_shape`
right-aligned via `broadcast_shapes` + the same index helper — so a scalar
`value` against `probs=[0.3,0.5]` yields a `[2]` result, and `value=[1,3]`
against `probs=[0.3,0.5]` pairs element-wise. This is the corrected pattern
from the FIXED `binomial.rs` (NOT `bernoulli.rs`'s `iter().cycle()`, which the
#1569 critic flagged for dropping batched params and not broadcasting `value`).

### Non-test production consumers

- `pub use geometric::Geometric` in `lib.rs` — the boundary public API
  (grandfathered per goal.md S5).
- `fn kl_geometric_geometric` in `kl.rs` consumes `p.probs()` / `q.probs()` and
  recomputes `q.logits()` off the `Geometric` accessors; reached through the
  public `kl_divergence` dispatcher arm.

## Parity contract

`parity_ops = []`. Geometric is a `torch.distributions` composite of `rand` +
`log`/`log1p`/`floor`; the parity contract is on those underlying tensor ops,
not on the distribution wrapper. Edge cases preserved:

- `log_prob(p=1, k=0) == 0` (the `geometric.py:137` NaN-dodge clamp).
- `sample` always non-negative integer; `p=1 -> all 0`.
- `mode == 0` for every `p`.
- `KL(Geometric(p) || Geometric(p)) ≈ 0`.

## Verification

Tests in `mod tests in geometric.rs` (live-torch 2.11.0+cu130 f64 references,
R-CHAR-3 non-tautological): `test_geometric_log_prob_{known,k0,p1_k0,
batched_probs,batched_value_and_probs}`, `test_geometric_mean_variance{,_batched}`,
`test_geometric_mode`, `test_geometric_entropy_{known,batched}`,
`test_geometric_{from_logits,logits_accessor}`,
`test_geometric_sample_{shape,batched_shape,in_support,prob_1}`,
`test_geometric_rsample_errors`, `test_geometric_{support,arg_constraints,
batch_shape,f32}`. KL: `test_kl_geometric_geometric_{same_is_zero,known_value,
batched}` in `kl.rs`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | `pub struct Geometric<T: Float>` with `probs`/`batch_shape` fields in `geometric.rs` mirroring `torch/distributions/geometric.py:20-87`; non-test consumer: `pub use geometric::Geometric` in `lib.rs` (boundary API, goal.md S5) + `fn kl_geometric_geometric` arm in `kl.rs`. |
| REQ-2 | SHIPPED | `pub fn Geometric::new` + `pub fn Geometric::from_logits` in `geometric.rs` mirroring `geometric.py:50-74` (`from_logits` uses the binary sigmoid `utils.py:97-98`); non-test consumer: `kl_geometric_geometric` reaches `Geometric` instances via `kl_divergence`. Pinned by `test_geometric_from_logits`. |
| REQ-3 | SHIPPED | `pub fn Geometric::probs` + `pub fn Geometric::logits` in `geometric.rs` mirroring `geometric.py:112-118` (`probs_to_logits`/`logits_to_probs` binary forms `utils.py:135-137,97-98`); non-test consumer: `fn kl_geometric_geometric in kl.rs` reads `p.probs()` / `q.probs()` + recomputes `q.logits()`. Pinned by `test_geometric_logits_accessor`. |
| REQ-4 | SHIPPED | `impl<T: Float> Distribution<T> for Geometric<T>` in `geometric.rs` mirroring `geometric.py:120-144`; `sample` returns `_extended_shape` (`distribution.py:266-278`); non-test consumer: trait surface via `pub use Geometric`. Pinned by `test_geometric_sample_{shape,batched_shape,in_support}` + `test_geometric_log_prob_*`. |
| REQ-5 | SHIPPED | `fn Geometric::rsample` returns `InvalidArgument` in `geometric.rs` (no reparam; `geometric.py` declares no `has_rsample`); non-test consumer: trait surface. Pinned by `test_geometric_rsample_errors`. |
| REQ-6 | SHIPPED | `fn Geometric::log_prob` = `k·log1p(-p) + ln(p)` with the `probs==1 & value==0 -> 0` clamp in `geometric.rs` mirroring `geometric.py:132-138`; `value` broadcasts against `batch_shape` right-aligned; non-test consumer: trait surface. Pinned by `test_geometric_log_prob_{known,k0,p1_k0,batched_probs,batched_value_and_probs}`. |
| REQ-7 | SHIPPED | `fn Geometric::entropy` = `BCE_with_logits(logits, probs)/probs` in `geometric.rs` mirroring `geometric.py:140-144`; non-test consumer: trait surface. Pinned by `test_geometric_entropy_{known,batched}`. |
| REQ-8 | SHIPPED | `fn Geometric::{mean, variance, mode}` = `(1-p)/p` / `(1-p)/p²` / `0` in `geometric.rs` mirroring `geometric.py:100-110`; non-test consumer: trait-default overrides via `pub use Geometric`. Pinned by `test_geometric_mean_variance{,_batched}` + `test_geometric_mode`. |
| REQ-9 | SHIPPED | `has_rsample`/`support` (`NonNegativeInteger`)/`arg_constraints` (`{probs: unit_interval, logits: real}`)/`event_shape`/`batch_shape` overrides in `geometric.rs` mirroring `geometric.py:46-48,74`; non-test consumer: `pub use Geometric`. Pinned by `test_geometric_{support,arg_constraints,batch_shape}`. |
