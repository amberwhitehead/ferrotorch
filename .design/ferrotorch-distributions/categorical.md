# ferrotorch-distributions — `categorical` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/distributions/categorical.py
-->

## Summary

`ferrotorch-distributions/src/categorical.rs` implements a
discrete Categorical distribution over `{0, 1, ..., K-1}`
parameterized by a 1-D `probs` tensor. Mirrors
`torch.distributions.Categorical`. Sampling uses inverse-CDF
via a precomputed cumulative-sum table + binary search on
`u ~ Uniform(0, 1)` per element. The distribution is discrete
and rejects `rsample` (use Gumbel-Softmax / `RelaxedOneHotCategorical`
for continuous relaxation). Samples are returned as float
tensors holding integer class indices (e.g. `0.0, 1.0, ...`).

## Requirements

- REQ-1: `pub struct Categorical<T: Float>` holding the
  parameter tensor `probs: Tensor<T>`, a precomputed `cdf:
  Vec<T>` of length `K` for fast inverse-CDF sampling, and
  `num_categories: usize`. The precomputation is a Rust-side
  optimization (R-DEV-7: trade O(K) construction time for O(log K)
  per-sample lookup, vs PyTorch's `torch.multinomial` which uses
  O(K) per sample on CPU). Mirrors
  `torch/distributions/categorical.py:Categorical.__init__`.

- REQ-2: `pub fn Categorical::new(probs: Tensor<T>) ->
  FerrotorchResult<Self>` validates: `probs.ndim() == 1`,
  `probs.shape()[0] > 0`, `sum(probs) > 0`. Returns
  `InvalidArgument` otherwise. Internally normalizes to sum to
  1 and builds the CDF table (last entry forced exactly to 1 to
  avoid floating-point CDF-tail edge cases). Upstream allows
  N-D `probs` with batched semantics; ferrotorch restricts to
  1-D for the initial impl. Tracked by REQ-11.

- REQ-3: `pub fn probs(&self) -> &Tensor<T>` and
  `pub fn num_categories(&self) -> usize` accessors. Mirrors
  `Categorical.probs` / `Categorical._num_events` attribute
  access (`categorical.py:114-117`).

- REQ-4: `impl<T: Float> Distribution<T> for Categorical<T>`
  provides `sample` / `rsample` (error) / `log_prob` / `entropy`.

- REQ-5: `sample(shape)` draws `u ~ Uniform(0, 1)` of shape
  `shape`, then binary-searches the precomputed CDF table for
  each draw. The output tensor contains float-typed integer
  indices (e.g. `2.0`, not `2`). Mirrors the contract of
  `torch.multinomial(probs, n, True)` then `.long()`.
  Conversion-to-float preserves the `T: Float` generic surface
  ferrotorch uses; downstream callers cast to `i64` at the use
  site if needed.

- REQ-6: `rsample` returns `InvalidArgument` with the message
  "Categorical distribution does not support reparameterized
  sampling. Use sample() with REINFORCE or the Gumbel-Softmax
  trick instead." Mirrors PyTorch's omission of
  `has_rsample = True` on `Categorical` (`categorical.py:13-54`).

- REQ-7: `log_prob(value) = log(probs[idx] / sum(probs))` for
  each integer-valued element of `value`. Out-of-range indices
  return `-inf`. The internal normalization-by-sum makes the
  function robust to unnormalized `probs` inputs (matching
  PyTorch's behaviour: `probs / probs.sum(-1, keepdim=True)` is
  done at `__init__`-time in upstream too). Uses
  `eps = 1e-7` clamp on the normalized probability before `ln`
  to prevent `log(0)` for zero-mass categories.

- REQ-8: `entropy = -sum(p * ln(p))` over normalized probs.
  Returns a scalar (0-D) tensor — for the 1-D-only ferrotorch
  impl, the entropy collapses to a single value. PyTorch returns
  a tensor of shape `batch_shape` (which is empty for 1-D probs
  → scalar) per `categorical.py:159-163`.

- REQ-9: Numerical guards: CDF's last entry is forced exactly
  to 1 to avoid `u = 0.999999...` returning index `K`
  (out-of-range); `log_prob` uses `max(eps, p)` before `ln` to
  cap `-inf` from zero-probability categories.

- REQ-10: NOT-STARTED — `logits`-parameterized constructor +
  N-D probs (batched Categorical) + `expand` +
  `enumerate_support` + `arg_constraints` + `support` +
  `mean` (NaN per PyTorch) + `mode` (`argmax` of probs) +
  `variance` (NaN). Cross-cutting with `lib.md` REQ-5
  (blocker #1376). Tracked as blocker #1410 for the
  Categorical-side fill-out.

## Acceptance Criteria

- [x] AC-1: `pub struct Categorical<T: Float>` with `probs`,
  `cdf`, `num_categories`.
- [x] AC-2: `pub fn Categorical::new` with the three validation
  checks (ndim, nonempty, positive-sum) + CDF precomputation.
- [x] AC-3: `pub fn probs` / `num_categories` accessors.
- [x] AC-4: `impl Distribution<T> for Categorical<T>` with all
  four required trait methods.
- [x] AC-5: `sample` via inverse-CDF binary search.
- [x] AC-6: `rsample` returns `InvalidArgument`.
- [x] AC-7: `log_prob` with index lookup + `-inf` for OOR + eps
  clamp.
- [x] AC-8: `entropy` as scalar 0-D tensor.
- [x] AC-9: `test_categorical_*` test suite (16 tests) covers
  the contract end-to-end.
- [ ] AC-10: `logits` ctor / N-D probs / `expand` /
  `enumerate_support` / `mean` / `mode` / `variance` —
  blocker #1410.

## Architecture

### Storage layout (REQ-1)

The struct holds three fields:

1. `probs: Tensor<T>` — the parameter tensor (kept unnormalized,
   accessible via `probs()`).
2. `cdf: Vec<T>` — host-side precomputed cumulative normalized
   probabilities. Length `K`.
3. `num_categories: usize` — equals `probs.shape()[0]`.

The CDF table is built once in `Categorical::new`. Each
`sample(shape)` call binary-searches into it `numel(shape)`
times. For `K=1000, shape=[1000]`, this is `1e6` comparisons
total vs ~5e5 if we walked the CDF linearly — both are fine for
CPU, but binary search ensures O(log K) per draw regardless of
K.

### Constructor validation (REQ-2)

```rust
if probs.ndim() != 1 { return InvalidArgument; }
if probs.shape()[0] == 0 { return InvalidArgument; }
let total = sum(probs);
if total <= 0 { return InvalidArgument; }
```

CDF construction normalizes by `total` and forces the last entry
exactly to 1 to avoid the tail-edge case (`u = 0.999999...`
should map to category `K-1`, not OOR).

### Inverse-CDF sampling (REQ-5)

```rust
let u ~ Uniform(0, 1) of shape `shape`;
for u_val in u_data:
    binary_search(cdf, u_val) -> idx in [0, K-1]
    push T::from(idx)
```

The output tensor has shape `shape` and dtype `T`. Integer
indices are stored as float values — downstream code that wants
`i64` can cast at the use site, as ferrotorch's tensor
ecosystem is dtype-generic by the `T: Float` bound.

### `log_prob` (REQ-7)

For each element `x` of the `value` tensor:

```rust
let idx = x as usize;
if idx < K {
    log_prob = max(eps, probs[idx] / total).ln()
} else {
    log_prob = -inf
}
```

This matches PyTorch's behaviour of `log_pmf.gather(-1, value)`
where `value` is broadcast and clamped via `.long()` casting at
the upstream call site (`categorical.py:151-157`).

### `entropy` (REQ-8)

```rust
entropy = -sum(p_norm[k] * ln(p_norm[k]))   for k in 0..K
```

Returns a 0-D tensor. For uniform K-class probs, entropy = ln(K).
For a deterministic distribution (all mass on one category),
entropy approaches 0 (limited by `eps` clamp: `eps * ln(eps) ≈ -1.6e-6`
for `eps = 1e-7`).

### Non-test production consumers

- **`pub use categorical::Categorical` in lib.rs** — grandfathered
  public surface (S5).
- **`MixtureSameFamily<T, D>` in mixture_same_family.rs** — the
  `MixtureSameFamily` struct has a field `mixing: Categorical<T>`
  at `mixture_same_family.rs`. Its constructor `pub fn
  MixtureSameFamily::new(mixing: Categorical<T>, ...)` at
  `mixture_same_family.rs` takes a `Categorical<T>` by value;
  its accessor `pub fn mixing(&self) -> &Categorical<T>` at
  `mixing in mixture_same_family.rs` hands the reference back. Every
  internal `MixtureSameFamily` instantiation across ferrotorch
  flows through the `Categorical` type — this is a non-test
  production consumer of both the struct and its `Distribution`
  impl (the mixing distribution's `sample` and `log_prob` are
  invoked from `MixtureSameFamily::sample` / `log_prob`).
- **KL-dispatcher consumer**: `kl.rs` registers a
  `Categorical-Categorical` arm at the dispatcher (downcast on
  `Categorical<T>`); the consumer is the public
  `pub fn kl_divergence` entry point. `kl_categorical_categorical`
  in `kl.rs` reads each `.probs()` for the closed-form
  `sum_k p_k * ln(p_k/q_k)`.

## Parity contract

`parity_ops = []`. Closed-form distribution mathematics;
parity sweep doesn't cover this layer.

Edge cases the implementation handles:

- **Unnormalized `probs` (e.g. `[1, 2, 3]`)**: normalized to
  `[1/6, 2/6, 3/6]` internally; `log_prob(2) = ln(3/6) = ln(0.5)`.
  Pinned by `test_categorical_unnormalized_probs`.
- **Zero-mass category**: clamped to `eps`; `log_prob ≈ -16.1`
  (f32) rather than `-inf`. The
  `test_categorical_sample_deterministic` confirms sampling
  always hits the high-prob class.
- **Out-of-range `value` in log_prob**: `-inf` (PyTorch raises,
  ferrotorch surfaces -inf — this is a documented soft
  divergence; blocker #1410 tracks tightening if needed).
- **CDF tail edge**: last entry forced to 1 exactly to avoid
  binary-search escape.
- **`f64`**: `test_categorical_f64`.

## Verification

Unit tests in `mod tests` (16 tests):

- Sample shape + 2-D shape forwarding + range:
  `test_categorical_sample_shape/_2d_shape/_valid_range/_deterministic`.
- `rsample` error: `test_categorical_rsample_errors`.
- `log_prob` at known indices + batch + OOR:
  `test_categorical_log_prob/_first_class/_batch/_out_of_range`.
- `entropy`: `test_categorical_entropy_uniform/_deterministic/_binary`.
- Constructor errors: `test_categorical_not_1d_errors`,
  `test_categorical_empty_errors`.
- Unnormalized probs: `test_categorical_unnormalized_probs`.
- `num_categories` accessor: `test_categorical_num_categories`.
- `f64`: `test_categorical_f64`.

Smoke command:

```bash
cargo test -p ferrotorch-distributions --lib categorical:: 2>&1 | tail -3
```

Expected: `16 passed; 0 failed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct Categorical<T: Float>` with `probs`, `cdf: Vec<T>`, `num_categories` fields in `categorical.rs` mirroring `torch/distributions/categorical.py:13-85`; non-test consumer: `pub use categorical::Categorical` in `lib.rs` + `MixtureSameFamily` holds `mixing: Categorical<T>` field at `mixture_same_family.rs`. |
| REQ-2 | SHIPPED | impl: `pub fn Categorical::new` in `categorical.rs` with ndim/empty/positive-sum validation + CDF precomputation mirroring `categorical.py:56-85`; non-test consumer: `MixtureSameFamily::new(mixing, components)` at `mixture_same_family.rs` takes a `Categorical<T>` constructed via `Categorical::new`. |
| REQ-3 | SHIPPED | impl: `pub fn Categorical::probs/num_categories` accessors in `categorical.rs` mirroring `categorical.py:114-117`; non-test consumer: `MixtureSameFamily::mixing` returns `&Categorical<T>` at `mixture_same_family.rs` — downstream introspection uses `.probs()` / `.num_categories()`. |
| REQ-4 | SHIPPED | impl: `impl<T: Float> Distribution<T> for Categorical<T>` in `categorical.rs` mirroring `categorical.py:144-163`; non-test consumer: `MixtureSameFamily::sample` / `log_prob` invoke the mixing categorical's trait methods at `mixture_same_family.rs`. |
| REQ-5 | SHIPPED | impl: `fn Categorical::sample` in `categorical.rs` with binary-search inverse-CDF lookup mirroring `categorical.py:144-149`; non-test consumer: `MixtureSameFamily::sample` calls `self.mixing.sample(...)` to pick a component index per draw. |
| REQ-6 | SHIPPED | impl: `fn Categorical::rsample` returns `InvalidArgument` in `categorical.rs` mirroring PyTorch's no-`has_rsample` design at `categorical.py:13-54`; non-test consumer: any external `rsample` call on a `Categorical` exercises this — `test_categorical_rsample_errors` pins the contract; `MixtureSameFamily::rsample` propagates this error. |
| REQ-7 | SHIPPED | impl: `fn Categorical::log_prob` in `categorical.rs` with eps-clamp + OOR-to-neg-inf mirroring `categorical.py:151-157`; non-test consumer: `kl_categorical_categorical` in `kl.rs` invokes log_prob-style math via `.probs()` (a sibling path), and `MixtureSameFamily::log_prob` invokes `self.mixing.log_prob(...)`. |
| REQ-8 | SHIPPED | impl: `fn Categorical::entropy` in `categorical.rs` with `-sum(p*ln(p))` formula mirroring `categorical.py:159-163`; non-test consumer: external `dist.entropy()` calls through `pub use Categorical`. |
| REQ-9 | SHIPPED | impl: CDF-last-entry-forced-to-1 + eps-clamp in `fn Categorical::new` and `log_prob` / `entropy` bodies in `categorical.rs`; non-test consumer: the inverse-CDF binary search in `fn Categorical::sample` relies on this guarantee — `test_categorical_sample_deterministic` (which sets `probs=[0,0,1]`) exercises the tail-edge case. |
| REQ-10 | PARTIAL | impl: `has_enumerate_support` (true) / `support` (NonNegative discrete-non-negative proxy — tight integer-interval awaits #1372) / `arg_constraints` (probs:Simplex with event_dim=1) / `event_shape` / `enumerate_support` (yields `[0..K-1]` along dim 0 mirroring `categorical.py:172-182`) / `expand` (returns `InvalidArgument` for 1-D-only ferrotorch Categorical) trait overrides at the tail of `impl Distribution<T> for Categorical<T>` in `categorical.rs`; non-test consumer: `pub use categorical::Categorical` at `lib.rs`; `tests/divergence_distribution_trait_surface.rs::categorical_*` pins. STILL NOT-STARTED (blocker #1410 remains open for these): `logits` constructor, N-D batched probs (limits `expand`), `mean`/`mode`/`variance` (currently trait-default `InvalidArgument`; PyTorch returns NaN). |
