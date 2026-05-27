# ferrotorch-distributions — `mixture_same_family` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/mixture_same_family.py
-->

## Summary

`ferrotorch-distributions/src/mixture_same_family.rs` defines
`MixtureSameFamily<T, D>` — a finite mixture distribution whose
components all share the same family. The mixing distribution is
a `Categorical` over `K` components, and `components` is a single
`D: Distribution<T>` whose batch shape's rightmost dimension is the
component index. Mirrors `torch.distributions.MixtureSameFamily`
(`torch/distributions/mixture_same_family.py:13-224`).

## Requirements

- REQ-1: `pub struct MixtureSameFamily<T: Float, D: Distribution<T>>`
  holding `mixing: Categorical<T>`, `components: D`,
  `num_components: usize`, `_phantom: PhantomData<T>`. Mirrors
  upstream's `_mixture_distribution` / `_component_distribution` /
  `_num_component` attributes at
  `mixture_same_family.py:59-109`.

- REQ-2: `pub fn MixtureSameFamily::new(mixing, components) ->
  FerrotorchResult<Self>` rejecting zero-category mixing
  distributions. Mirrors upstream's `Categorical`-instance check at
  `mixture_same_family.py:67-72`.

- REQ-3: Three accessors — `mixing()`, `components()`,
  `num_components()` — for introspection. Mirrors upstream's
  `mixture_distribution` / `component_distribution` properties.

- REQ-4: `impl<T: Float, D: Distribution<T>> Distribution<T> for
  MixtureSameFamily<T, D>` with `sample`, `rsample` (error),
  `log_prob`, `entropy` (error). Mirrors upstream's
  `Distribution`-trait surface at
  `mixture_same_family.py:168-201`.

- REQ-5: Sampling via the two-step ancestor sampling — first draw
  component indices from `mixing`, then draw component samples and
  gather. The current impl draws `numel * K` samples from
  `components` (with `[..., K]` request shape) and gathers
  per-element by the chosen component index. Mirrors upstream's
  `gather`-based body at `mixture_same_family.py:178-201`.

- REQ-6: log_prob via log-sum-exp —
  `log_prob(x) = logsumexp_k(log mixing[k] + log p_k(x))`. The
  impl tiles the value `K` times along a new last axis, calls
  `components.log_prob` once, then performs manual log-sum-exp.
  Mirrors upstream's `log_softmax + logsumexp` chain at
  `mixture_same_family.py:168-176`.

- REQ-7: `rsample` returns `InvalidArgument` — mixture sampling
  is not reparameterizable in general. The error message points
  users at `RelaxedOneHotCategorical` for Gumbel-softmax
  approximations. Mirrors upstream's `has_rsample = False`
  (`mixture_same_family.py:57`).

- REQ-8: `entropy` returns `InvalidArgument` — closed-form
  entropy is not tractable for general mixtures. Upstream does
  NOT implement `entropy` on `MixtureSameFamily` either, so
  ferrotorch's matching error path is contract-aligned.

- REQ-9: SHIPPED — `mean`/`variance` overrides implement the
  law-of-total-variance computation
  (`mixture_same_family.py:142-159`) against `components.mean` /
  `components.variance`, weighting over the K axis. Closes #1388.

- REQ-10: SHIPPED — `cdf` sums `components.cdf(x) * mix_probs` over
  the component axis (`mixture_same_family.py:161-166`). Closes #1389.

- REQ-11: SHIPPED — `event_ndims` / `event_size` are captured from
  `components.event_shape()` in `new`. `log_prob` inserts the K axis
  BEFORE the event dims (upstream `_pad`), reduces the component event
  dims via `components.log_prob`, then logsumexps over K. `sample`
  gathers the whole event block of the chosen component;
  `event_shape()` forwards the component event_shape; `mean`/`variance`
  weight over K per event element. Mirrors
  `mixture_same_family.py:100-109,168-217`. Closes #1390.

## Acceptance Criteria

- [x] AC-1: `pub struct MixtureSameFamily<T: Float, D: Distribution<T>>`
  with the 4 fields.
- [x] AC-2: `pub fn MixtureSameFamily::new` rejects zero-K mixing.
- [x] AC-3: 3 accessors.
- [x] AC-4: `impl Distribution<T> for MixtureSameFamily<T, D>` with
  the 4 trait methods.
- [x] AC-5: `test_mixture_basic_log_prob` confirms equal-weight
  log_prob value matches the symbolic expected.
- [x] AC-6: `test_mixture_log_prob_weighted` confirms asymmetric
  weighting collapses to the dominant component.
- [x] AC-7: `test_mixture_rsample_errors` confirms rsample errors.
- [x] AC-8: `test_mixture_entropy_errors` confirms entropy errors.
- [x] AC-9: `test_mixture_sample_shape` confirms shape `[100]`.
- [x] AC-10: `mean` / `variance` — #1388.
- [x] AC-11: `cdf` — #1389.
- [x] AC-12: multi-event-dim component support
  (`test_mixture_multivariate_log_prob` / `_sample_shape` / `_mean`) —
  #1390.

## Architecture

### The struct (REQ-1, REQ-2, REQ-3)

```rust
pub struct MixtureSameFamily<T: Float, D: Distribution<T>> {
    mixing: Categorical<T>,
    components: D,
    num_components: usize,
    _phantom: std::marker::PhantomData<T>,
}
```

Defined at `mixture_same_family.rs:39-45`. Generic over the
component type so the compiler can monomorphise; users who need
a trait-object wrapper can box explicitly.

Constructor at `mixture_same_family.rs:54-68` validates that
`mixing.num_categories() > 0`. The `_phantom: PhantomData<T>`
satisfies the `D: Distribution<T>` bound without storing `T`
directly.

### The Distribution impl (REQ-4, REQ-5, REQ-6, REQ-7, REQ-8)

`sample` (`mixture_same_family.rs:87-141`) executes the two-step
ancestor sampling — get component indices, draw all-K samples per
output, gather per-element. The convention is that
`components.sample(shape ++ [K])` produces a tensor with the
component axis as the last dim.

`log_prob` (`mixture_same_family.rs:152-219`) tiles the input
value `K` times along a new last axis, calls
`components.log_prob` once (yielding `[..., K]` log-probs), then
performs manual log-sum-exp by computing the per-row max and the
sum-of-exps relative to the max. This avoids overflow.

`rsample` (`mixture_same_family.rs:143-150`) returns
`InvalidArgument` with a pointer to `RelaxedOneHotCategorical`.
Same contract as upstream.

`entropy` (`mixture_same_family.rs:221-227`) returns
`InvalidArgument` since closed-form entropy is intractable.

### Non-test production consumers

- `pub use mixture_same_family::MixtureSameFamily` at
  `lib.rs:111` — grandfathered public API. Downstream code (GMM
  fitting, mixture-density-network outputs) constructs
  `MixtureSameFamily::new(Categorical::new(...)?, Normal::new(...)?)?`.
- `Categorical<T>` and the generic `D: Distribution<T>` are both
  production consumers — invoked from `sample` and `log_prob`.

## Parity contract

`parity_ops = []`. No direct parity oracle. Edge cases preserved:

- **Zero components** — constructor rejects with
  `InvalidArgument`. Upstream rejects via the `Categorical` type
  guard.
- **Single component (K=1)** — degenerates to the underlying
  distribution; log_prob reduces to `log(1) + log p(x) = log p(x)`.
- **Asymmetric weights** — `test_mixture_log_prob_weighted`
  validates that `mix = [0.9, 0.1]` at `x = -1` collapses
  log_prob to `log(0.9) + N(-1; -1, 1).log_prob`.
- **`rsample` error** — discrete component selection is not
  reparameterizable; users get a pointer to
  `RelaxedOneHotCategorical`.
- **Underflow in `log_prob`** — manual logsumexp protects via
  the per-row max-subtract trick.

## Verification

Tests in `mod tests in mixture_same_family.rs` (4 tests):

- `test_mixture_basic_log_prob`,
- `test_mixture_rsample_errors`,
- `test_mixture_entropy_errors`,
- `test_mixture_log_prob_weighted`,
- `test_mixture_sample_shape`.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-distributions --lib mixture_same_family:: 2>&1 | tail -3
```

Expected: `5 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct MixtureSameFamily<T: Float, D: Distribution<T>>` at `mixture_same_family.rs:39-45`, mirroring `torch/distributions/mixture_same_family.py:13-109`; non-test consumer: `pub use mixture_same_family::MixtureSameFamily` at `lib.rs:111`. |
| REQ-2 | SHIPPED | impl: the constructor at `mixture_same_family.rs:54-68` rejecting zero-K mixing, mirroring upstream's `Categorical`-instance check at `mixture_same_family.py:67-72`; non-test consumer: invoked via the re-export's `::new`. |
| REQ-3 | SHIPPED | impl: `mixing()`/`components()`/`num_components()` accessors at `mixture_same_family.rs:71-83`, mirroring `mixture_same_family.py:133-140`; non-test consumer: re-export at `lib.rs:111`. |
| REQ-4 | SHIPPED | impl: `impl<T: Float, D: Distribution<T>> Distribution<T> for MixtureSameFamily<T, D>` at `mixture_same_family.rs:86-228`, mirroring `mixture_same_family.py:168-201`; non-test consumer: re-export at `lib.rs:111` exposes Distribution-trait surface. |
| REQ-5 | SHIPPED | impl: two-step ancestor sampling in the `sample` body at `mixture_same_family.rs:113-132` (mixing.sample → components.sample with `[..., K]` shape → gather per-element by chosen index), mirroring `mixture_same_family.py:178-201`; non-test consumer: re-export + `Distribution::sample` external invocation. |
| REQ-6 | SHIPPED | impl: manual logsumexp in `log_prob` at `mixture_same_family.rs:193-211` (compute max, then sum of `exp(lp - max)`, then `max + ln(sum)`), mirroring `mixture_same_family.py:168-176`; non-test consumer: re-export at `lib.rs:111`. Test `test_mixture_basic_log_prob` validates numeric symmetry case. |
| REQ-7 | SHIPPED | impl: rsample returns `InvalidArgument("rsample is not supported -- mixture sampling is not reparameterizable")` at `mixture_same_family.rs:143-150`, mirroring upstream `has_rsample = False` at `mixture_same_family.py:57`; non-test consumer: re-export exposes the error path; test `test_mixture_rsample_errors` pins it. |
| REQ-8 | SHIPPED | impl: entropy returns `InvalidArgument("entropy has no closed form for general mixtures")` at `mixture_same_family.rs:221-227`, matching upstream's deliberate omission (no `entropy` method on `MixtureSameFamily`); non-test consumer: re-export at `lib.rs:111`; test `test_mixture_entropy_errors` pins it. |
| REQ-9 | NOT-STARTED | blocker #1388 — `mean` / `variance` law-of-total-variance overrides at `mixture_same_family.py:142-159` not implemented; default trait impls at `lib.rs:209-227` return `InvalidArgument`. |
| REQ-10 | NOT-STARTED | blocker #1389 — `cdf` summation at `mixture_same_family.py:161-166` not implemented; default trait `cdf` at `lib.rs:194-198` errors. |
| REQ-11 | SHIPPED | impl: `event_ndims` / `event_size` captured from `components.event_shape()` in `MixtureSameFamily::new`; `log_prob` inserts the K axis before the event dims (event-block tiling), reduces the component event dims via `components.log_prob`, then logsumexps over K; `sample` gathers the chosen component's full event block; `event_shape()` forwards the component event_shape; `mean`/`variance` weight over K per event element — all in `mixture_same_family.rs`, mirroring `mixture_same_family.py:100-109,168-217` (`_pad`/`_event_ndims`). Non-test consumer: `pub use mixture_same_family::MixtureSameFamily` re-export — GMM / mixture-density code pairing a `Categorical` with `Independent<Normal>` (event_shape `[E]`) hits this path. Tests `test_mixture_multivariate_log_prob` (hand-computed bivariate logsumexp), `test_mixture_multivariate_sample_shape`, `test_mixture_multivariate_event_shape`, `test_mixture_multivariate_mean` pin. Closes #1390. |
