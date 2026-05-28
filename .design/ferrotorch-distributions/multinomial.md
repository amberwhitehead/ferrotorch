# ferrotorch-distributions — `multinomial` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/multinomial.py
-->

## Summary

`ferrotorch-distributions/src/multinomial.rs` defines
`Multinomial<T>` — a distribution over `K`-vectors of non-negative
integer counts summing to `total_count`. Each trial selects a
category by `probs`. Mirrors `torch.distributions.Multinomial`
(`torch/distributions/multinomial.py:14-148`). This is a discrete
distribution and does not support reparameterized sampling.

## Requirements

- REQ-1: `pub struct Multinomial<T: Float>` holding `total_count:
  usize`, `probs: Tensor<T>` (the original, normalised internally
  per call site), `cdf: Vec<T>` (precomputed for inverse-CDF
  sampling), `log_probs: Vec<T>` (precomputed log-softmax for
  log_prob), `num_categories: usize`. Mirrors upstream's
  `total_count` / `_categorical` / `_binomial` cached state at
  `multinomial.py:54-79`.

- REQ-2: `pub fn Multinomial::new(total_count, probs) ->
  FerrotorchResult<Self>` enforcing:
  - `probs.ndim() == 1` (a 1-D vector of category probabilities);
  - `probs.shape()[0] > 0` (at least one category);
  - `sum(probs) > 0`.

  Internally normalises probs to sum to 1, builds the CDF and
  the log-probs lookup table. Mirrors upstream's `Categorical(probs)`
  + `Binomial(total_count, probs)` construction at
  `multinomial.py:64-79`.

- REQ-3: Three accessors — `total_count()`, `probs()`,
  `num_categories()` — for introspection. Mirrors upstream's
  property accessors at `multinomial.py:100-110`.

- REQ-4: `impl<T: Float> Distribution<T> for Multinomial<T>` with
  `sample`, `rsample` (error), `log_prob`, `mean`, `variance`,
  `entropy`. Mirrors `multinomial.py:112-148`.

- REQ-5: Sampling via `total_count` independent categorical draws
  per output position — each draw uses inverse CDF + binary search
  on the precomputed CDF, accumulated into a `[K]` count vector.
  Mirrors upstream's `_categorical.sample` + `scatter_add_` chain
  at `multinomial.py:112-124`.

- REQ-6: `log_prob` via the multinomial PMF
  `log_factorial_n - sum(log_factorial_x_k) + sum(x_k * log_p_k)`
  using `lgamma_scalar` for the factorials. Mirrors upstream
  `multinomial.py:139-148` (which uses `torch.lgamma`).

- REQ-7: Closed-form `mean` (`total_count * probs`) and
  `variance` (`total_count * probs * (1 - probs)`). Mirror
  upstream properties at `multinomial.py:56-62`.

- REQ-8: `rsample` returns `InvalidArgument` — discrete sampling
  is not reparameterizable. The error message points users at
  REINFORCE / relaxation methods. Mirrors upstream's omission of
  `has_rsample` (defaults to False).

- REQ-9: NOT-STARTED — `entropy` uses a Stirling-based
  approximation rather than the exact formula upstream computes
  via `Binomial.enumerate_support` + `Binomial.log_prob`. The
  current `multinomial.rs` is a closed-form approximation
  (`H ≈ n * H_cat + correction(lgamma)`) good for large `n` but
  not exact. Upstream `multinomial.py:126-137` does the exact
  enumeration. Blocker #1391 tracks the exact-entropy fill-in.

- REQ-10: NOT-STARTED — upstream supports `logits` as an
  alternative parameterisation; ferrotorch only accepts `probs`.
  The `logits` setter at `multinomial.py:101-102` is unimplemented.
  Blocker #1392 tracks logits parameterisation.

## Acceptance Criteria

- [x] AC-1: `pub struct Multinomial<T: Float>` with the 5 fields.
- [x] AC-2: `pub fn Multinomial::new` with the 3 preconditions.
- [x] AC-3: 3 accessors.
- [x] AC-4: `impl Distribution<T> for Multinomial<T>` with the 6
  trait methods.
- [x] AC-5: `test_multinomial_sample_sums_to_total_count`
  validates the count constraint.
- [x] AC-6: `test_multinomial_sample_deterministic` validates
  degenerate-probs case.
- [x] AC-7: `test_multinomial_log_prob` validates exact symbolic
  log_prob for `Multinomial(10, [0.5, 0.5]).log_prob([5, 5])`.
- [x] AC-8: `test_multinomial_rsample_errors` confirms rsample
  errors.
- [ ] AC-9: exact entropy via Binomial.enumerate_support —
  blocker #1391.
- [ ] AC-10: `logits` parameterisation — blocker #1392.

## Architecture

### The struct (REQ-1, REQ-2, REQ-3)

```rust
pub struct Multinomial<T: Float> {
    total_count: usize,
    probs: Tensor<T>,
    cdf: Vec<T>,
    log_probs: Vec<T>,
    num_categories: usize,
}
```

Defined at `multinomial.rs`. Constructor at
`multinomial.rs` performs all validation, normalisation,
CDF construction, and log-probs precomputation. Accessors at
`multinomial.rs`.

### The Distribution impl (REQ-4, REQ-5, REQ-6, REQ-7, REQ-8)

`sample` (`sample in multinomial.rs`) draws `n * total_count` uniform
samples and bucket-counts them via binary search on the precomputed
CDF. Output shape is `shape ++ [K]`.

`log_prob` (`log_prob in multinomial.rs`) iterates per-batch
computing `lgamma(n+1) - sum lgamma(x_k+1) + sum x_k * log_p_k`.
Output shape is `shape[..-1]` (the `K` dim is collapsed).

`mean` (`mean in multinomial.rs`) returns
`total_count * probs / sum(probs)` (normalising on the fly).

`variance` (`variance in multinomial.rs`) returns
`total_count * p_k * (1 - p_k)`.

`entropy` (`entropy in multinomial.rs`) uses the Stirling-based
approximation (REQ-9 divergence).

`rsample` (`rsample in multinomial.rs`) returns `InvalidArgument`.

### Non-test production consumers

- `pub use multinomial::Multinomial` at `lib.rs` —
  grandfathered public API. Downstream language-model / topic-model
  / RL discrete-action code constructs
  `Multinomial::new(total_count, probs)?`.
- `special_fns::lgamma_scalar` is the production consumer of
  ferrotorch-distributions' special-functions module — invoked
  from `log_prob` and `entropy`.

## Parity contract

`parity_ops = []`. No direct multinomial parity oracle. Edge cases
preserved:

- **`probs` not 1-D** — constructor errors. Test
  `test_multinomial_not_1d_errors` pins this.
- **Empty probs** — constructor errors. Test
  `test_multinomial_empty_errors` pins this.
- **`sum(probs) <= 0`** — constructor errors. Same as upstream's
  `simplex` constraint.
- **`probs = [0, 0, 1]`** — degenerates to all-counts-at-category-2.
  Test `test_multinomial_sample_deterministic` pins this.
- **`total_count == 1`** — output is a one-hot per row. Test
  `test_multinomial_single_trial` pins this.
- **Sum invariant** — every sample row sums to `total_count`.
  Test `test_multinomial_sample_sums_to_total_count` pins this.

## Verification

Tests in `mod tests in multinomial.rs` (12 tests):

- `test_multinomial_sample_shape`,
- `test_multinomial_sample_2d_shape`,
- `test_multinomial_sample_sums_to_total_count`,
- `test_multinomial_sample_nonnegative`,
- `test_multinomial_sample_deterministic`,
- `test_multinomial_rsample_errors`,
- `test_multinomial_log_prob`,
- `test_multinomial_log_prob_batch`,
- `test_multinomial_not_1d_errors`,
- `test_multinomial_empty_errors`,
- `test_multinomial_num_categories`,
- `test_multinomial_f64`,
- `test_multinomial_single_trial`.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-distributions --lib multinomial:: 2>&1 | tail -3
```

Expected: `13 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct Multinomial<T: Float>` with the 5 fields at `Multinomial in multinomial.rs`, mirroring `torch/distributions/multinomial.py:54-79`; non-test consumer: `pub use multinomial::Multinomial` at `lib.rs`. |
| REQ-2 | SHIPPED | impl: the constructor at `new in multinomial.rs` with 3 preconditions + normalisation + CDF + log-probs precompute, mirroring `multinomial.py:64-79`; non-test consumer: invoked via `::new` through the re-export. |
| REQ-3 | SHIPPED | impl: `total_count()`/`probs()`/`num_categories()` accessors at `num_categories in multinomial.rs`, mirroring `multinomial.py:100-110`; non-test consumer: re-export at `lib.rs`. |
| REQ-4 | SHIPPED | impl: `impl<T: Float> Distribution<T> for Multinomial<T>` at `multinomial.rs` with 6 methods, mirroring `multinomial.py:112-148`; non-test consumer: re-export at `lib.rs`. Tests pin behaviour. |
| REQ-5 | SHIPPED | impl: the binary-search-on-CDF body in `sample in multinomial.rs`, mirroring `_categorical.sample + scatter_add_` at `multinomial.py:112-124`; non-test consumer: re-export at `lib.rs` + `Distribution::sample` external invocation. Test `test_multinomial_sample_sums_to_total_count` pins the invariant. |
| REQ-6 | SHIPPED | impl: `lgamma(n+1) - sum lgamma(x_k+1) + sum x_k * log_p_k` body of `log_prob in multinomial.rs` using `lgamma_scalar`, mirroring `multinomial.py:139-148`; non-test consumer: re-export at `lib.rs` + `Distribution::log_prob` invocation. |
| REQ-7 | SHIPPED | impl: `mean` and `variance in multinomial.rs` returning `total_count * p_k` and `total_count * p_k * (1 - p_k)`, mirroring `multinomial.py:56-62`; non-test consumer: re-export at `lib.rs` exposes them through `Distribution::mean` / `Distribution::variance`. |
| REQ-8 | SHIPPED | impl: rsample returns `InvalidArgument("does not support reparameterized sampling")` at `test_multinomial_rsample_errors in multinomial.rs`; non-test consumer: re-export at `lib.rs`; test `test_multinomial_rsample_errors` pins it. |
| REQ-9 | NOT-STARTED | blocker #1391 — `entropy in multinomial.rs` is a Stirling-based approximation, not the exact `Binomial.enumerate_support` + `Binomial.log_prob` enumeration from `multinomial.py:126-137`. |
| REQ-10 | NOT-STARTED | blocker #1392 — `logits` parameterisation at `multinomial.py:101-102` not supported; `Multinomial::new` only accepts `probs`. |
