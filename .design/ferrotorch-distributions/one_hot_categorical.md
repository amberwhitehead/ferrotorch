# ferrotorch-distributions — `one_hot_categorical` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/one_hot_categorical.py
-->

## Summary

`ferrotorch-distributions/src/one_hot_categorical.rs` defines
`OneHotCategorical<T>` — a categorical distribution that returns
samples as one-hot vectors of shape `[..., K]` instead of integer
indices. Mirrors `torch.distributions.OneHotCategorical`
(`torch/distributions/one_hot_categorical.py:14-126`). Discrete:
no reparameterized sampling (the
`OneHotCategoricalStraightThrough` variant which adds rsample
via a straight-through estimator is NOT in this module).

## Requirements

- REQ-1: `pub struct OneHotCategorical<T: Float>` holding
  `probs: Tensor<T>` (the original 1-D probability tensor),
  `normalized: Vec<T>` (cache of the normalised probability vector),
  `cdf: Vec<T>` (precomputed CDF for inverse-CDF sampling), and
  `num_categories: usize`. Mirrors upstream's `_categorical` cached
  state at `one_hot_categorical.py:49-72`.

- REQ-2: `pub fn OneHotCategorical::new(probs) ->
  FerrotorchResult<Self>` enforcing 1-D, non-empty, sum > 0
  preconditions. Normalises internally. Mirrors upstream's
  `Categorical(probs, logits)` construction at
  `one_hot_categorical.py:49-59`.

- REQ-3: `probs()` and `num_categories()` accessors. Mirror
  upstream's `probs` / `param_shape` properties at
  `one_hot_categorical.py:78-102`.

- REQ-4: `impl<T: Float> Distribution<T> for OneHotCategorical<T>`
  with `sample`, `rsample` (error), `log_prob`, `entropy`. Mirrors
  `one_hot_categorical.py:104-118`.

- REQ-5: Sampling via inverse-CDF + one-hot scatter — draw `u ~
  Uniform(0, 1)`, binary-search the CDF for the category, then
  set `out[i*K + cat] = 1`. Output shape is `shape ++ [K]`.
  Mirrors upstream's `Categorical.sample + one_hot` chain at
  `one_hot_categorical.py:104-109`.

- REQ-6: `log_prob` accepts a `[..., K]` value tensor and returns
  `sum_k value[k] * log(probs[k])` over the trailing K dim,
  collapsing it. This is the generalised form that works for both
  strict one-hot inputs (returns `log(probs[picked])`) and arbitrary
  non-negative weights. Upstream restricts the value to a strict
  one-hot via `indices = value.max(-1)[1]; categorical.log_prob(indices)`
  at `one_hot_categorical.py:111-115`; ferrotorch's broader
  contract is a R-DEV-7 ergonomic generalisation.

- REQ-7: `entropy` via `H = -sum_k p_k * log(p_k)`. Same as the
  underlying `Categorical.entropy`. Mirrors
  `one_hot_categorical.py:117-118` which delegates to
  `self._categorical.entropy()`.

- REQ-8: `rsample` returns `InvalidArgument` — discrete sampling
  is not reparameterizable in the strict sense. The error message
  points users at `RelaxedOneHotCategorical`. Upstream's
  `OneHotCategorical` (not `OneHotCategoricalStraightThrough`)
  inherits the default `rsample` which also errors.

- REQ-9: NOT-STARTED — `mean` / `mode` / `variance` properties
  from `one_hot_categorical.py:87-98` not implemented:
  - mean = `probs`
  - mode = one_hot(`argmax(probs)`)
  - variance = `probs * (1 - probs)`

  Blocker #1413 tracks these property overrides.

- REQ-10: NOT-STARTED — `enumerate_support` from
  `one_hot_categorical.py:120-126` not implemented; the trait
  has no `enumerate_support` method. Blocker #1417 tracks
  the trait-level fill-out + this distribution's override.

- REQ-11: NOT-STARTED — `OneHotCategoricalStraightThrough` from
  `one_hot_categorical.py:129-143` (the straight-through-estimator
  variant with `has_rsample = True`) is not exposed. Blocker
  #1418 tracks the straight-through variant.

## Acceptance Criteria

- [x] AC-1: `pub struct OneHotCategorical<T: Float>` with the 4
  fields.
- [x] AC-2: `pub fn OneHotCategorical::new` with 3 preconditions.
- [x] AC-3: `probs()` and `num_categories()` accessors.
- [x] AC-4: `impl Distribution<T> for OneHotCategorical<T>` with
  `sample`/`rsample`/`log_prob`/`entropy`.
- [x] AC-5: `test_one_hot_categorical_sample_shape` validates
  output shape and one-hotness.
- [x] AC-6: `test_one_hot_categorical_log_prob_pure_one_hot`
  validates `log_prob([0,1,0]) == log(0.3)` for `probs=[0.2,0.3,0.5]`.
- [x] AC-7: `test_one_hot_categorical_log_prob_batch` validates
  batched log_prob.
- [x] AC-8: `test_one_hot_categorical_entropy` validates
  `H([0.5, 0.5]) == log(2)`.
- [x] AC-9: `test_one_hot_categorical_rsample_errors` confirms
  rsample errors.
- [x] AC-10: `test_one_hot_categorical_wrong_shape_errors`
  validates wrong-K rejection.
- [ ] AC-11: `mean` / `mode` / `variance` — blocker #1413.
- [ ] AC-12: `enumerate_support` — blocker #1417.
- [ ] AC-13: `OneHotCategoricalStraightThrough` variant — blocker
  #1418.

## Architecture

### The struct (REQ-1, REQ-2, REQ-3)

```rust
pub struct OneHotCategorical<T: Float> {
    probs: Tensor<T>,
    normalized: Vec<T>,
    cdf: Vec<T>,
    num_categories: usize,
}
```

Defined at `one_hot_categorical.rs`. Constructor at
`one_hot_categorical.rs`. Accessors at
`one_hot_categorical.rs`.

### The Distribution impl (REQ-4, REQ-5, REQ-6, REQ-7, REQ-8)

`sample` (`sample in one_hot_categorical.rs`) draws `n = prod(shape)`
uniform samples, binary-searches the CDF for each, and writes a
`1.0` at the selected one-hot position in a `[n, K]` zero buffer.
Output reshape is `shape ++ [K]`.

`log_prob` (`log_prob in one_hot_categorical.rs`) validates that
`value.shape[-1] == K`, precomputes `log(normalized + eps)`, then
computes `sum_k value[k] * log_p[k]` per row, returning shape
`value.shape[..-1]`.

`entropy` (`entropy in one_hot_categorical.rs`) returns
`-sum p_k * log(p_k)` as a 0-D scalar tensor.

`rsample` (`rsample in one_hot_categorical.rs`) returns
`InvalidArgument` with a pointer to `RelaxedOneHotCategorical`.

### Non-test production consumers

- `pub use one_hot_categorical::OneHotCategorical` at `lib.rs`
  — grandfathered public API. Downstream code (Gumbel-softmax
  baselines, discrete-action policy networks, structured
  prediction layers) constructs
  `OneHotCategorical::new(probs)?`.
- `RelaxedOneHotCategorical` (in a separate module) is the
  continuous-relaxation counterpart; users move from
  `OneHotCategorical` to `RelaxedOneHotCategorical` when they
  need gradient flow.

## Parity contract

`parity_ops = []`. No direct parity oracle. Edge cases preserved:

- **Non-1D probs** — constructor errors. Same as upstream's
  `simplex` constraint indirectly.
- **Empty probs** — constructor errors.
- **`sum(probs) <= 0`** — constructor errors.
- **Wrong K in log_prob** — last dim of value must equal `K`.
  Test `test_one_hot_categorical_wrong_shape_errors` pins this.
- **Uniform case** — `OneHotCategorical([0.5, 0.5])` yields
  entropy `log(2)`. Test pins this.
- **`probs[k] == 0`** — the `(p + 1e-30).ln()` clamp in
  `log_prob` / `entropy` prevents `-infinity` propagation;
  upstream PyTorch likewise clamps via `torch.where(p == 0, ...)`
  guards in the underlying `Categorical`.

## Verification

Tests in `mod tests in one_hot_categorical.rs` (6 tests):

- `test_one_hot_categorical_sample_shape`,
- `test_one_hot_categorical_log_prob_pure_one_hot`,
- `test_one_hot_categorical_log_prob_batch`,
- `test_one_hot_categorical_entropy`,
- `test_one_hot_categorical_rsample_errors`,
- `test_one_hot_categorical_wrong_shape_errors`.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-distributions --lib one_hot_categorical:: 2>&1 | tail -3
```

Expected: `6 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct OneHotCategorical<T: Float>` with `probs`/`normalized`/`cdf`/`num_categories in one_hot_categorical.rs`, mirroring `torch/distributions/one_hot_categorical.py:49-72`; non-test consumer: `pub use one_hot_categorical::OneHotCategorical` at `lib.rs`. |
| REQ-2 | SHIPPED | impl: the constructor at `new in one_hot_categorical.rs` with the 3 preconditions + normalisation + CDF precompute, mirroring `one_hot_categorical.py:49-59`; non-test consumer: invoked via `::new` through the re-export. |
| REQ-3 | SHIPPED | impl: `probs()` and `num_categories()` accessors at `num_categories in one_hot_categorical.rs`; non-test consumer: re-export at `lib.rs`. |
| REQ-4 | SHIPPED | impl: `impl<T: Float> Distribution<T> for OneHotCategorical<T>` at `one_hot_categorical.rs` with the 4 trait methods, mirroring `one_hot_categorical.py:104-118`; non-test consumer: re-export at `lib.rs`. 6 tests pin behaviour. |
| REQ-5 | SHIPPED | impl: inverse-CDF + scatter body in `sample in one_hot_categorical.rs`, mirroring `one_hot_categorical.py:104-109`; non-test consumer: re-export + `Distribution::sample` invocation. Test `test_one_hot_categorical_sample_shape` validates one-hotness. |
| REQ-6 | SHIPPED | impl: `sum_k value[k] * log_p[k]` body of `log_prob in one_hot_categorical.rs`, generalising upstream's `value.max(-1)[1]` indexing at `one_hot_categorical.py:111-115`; non-test consumer: re-export + `Distribution::log_prob` invocation. Test `test_one_hot_categorical_log_prob_pure_one_hot` validates `log(0.3)` value. |
| REQ-7 | SHIPPED | impl: `-sum p * log(p)` body of `entropy in one_hot_categorical.rs`, mirroring `one_hot_categorical.py:117-118`; non-test consumer: re-export + `Distribution::entropy` invocation. Test `test_one_hot_categorical_entropy` validates uniform case. |
| REQ-8 | SHIPPED | impl: rsample returns `InvalidArgument("not supported -- discrete distribution")` at `test_one_hot_categorical_rsample_errors in one_hot_categorical.rs`; non-test consumer: re-export at `lib.rs`. Test `test_one_hot_categorical_rsample_errors` pins it. |
| REQ-9 | NOT-STARTED | blocker #1413 — `mean` / `mode` / `variance` overrides at `one_hot_categorical.py:87-98` not implemented; default trait impls at `lib.rs:209-227` return `InvalidArgument`. |
| REQ-10 | NOT-STARTED | blocker #1417 — `enumerate_support` at `one_hot_categorical.py:120-126` not implemented; the `Distribution` trait has no `enumerate_support` method (cross-cutting with the trait-fill-out task). |
| REQ-11 | NOT-STARTED | blocker #1418 — `OneHotCategoricalStraightThrough` straight-through-estimator variant at `one_hot_categorical.py:129-143` not exposed as a separate type. |
