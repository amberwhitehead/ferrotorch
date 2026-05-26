# ferrotorch-data — `sampler` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/utils/data/sampler.py
  - torch/utils/data/distributed.py
-->

## Summary

`ferrotorch-data/src/sampler.rs` implements the `Sampler` trait and
its five concrete impls — `SequentialSampler`, `RandomSampler`,
`DistributedSampler`, `WeightedRandomSampler`, and `BatchSampler` —
plus the shared deterministic-shuffle primitive `shuffle_with_seed`
and the Walker's-alias-table O(1)-per-draw weighted sampling fast
path. Mirrors `torch/utils/data/sampler.py:28-355` and
`torch/utils/data/distributed.py:17-138`.

The PRNG is deterministic and seeded by `(base_seed XOR epoch ×
0x9e3779b97f4a7c15)` so different epochs produce different
orderings while staying reproducible across runs — this matches the
contract PyTorch implements via `torch.Generator().manual_seed(seed
+ epoch)`.

## Requirements

- REQ-1: `pub trait Sampler: Send + Sync` with `fn indices(&self,
  epoch: usize) -> Vec<usize>`, `fn len(&self) -> usize`, and `fn
  is_empty(&self) -> bool { self.len() == 0 }`. Mirrors
  `torch/utils/data/sampler.py:28-94` `class Sampler(Generic[_T_co])`
  but returns `Vec<usize>` from `indices(epoch)` rather than an
  iterator — the loader needs the materialised index list anyway
  for batching, and the eager return matches the
  Fisher-Yates / partition_point implementations below.

- REQ-2: `pub fn shuffle_with_seed(indices: &mut [usize], seed: u64)`
  — Fisher-Yates shuffle with a deterministic xorshift64 PRNG and
  Lemire's nearly-divisionless rejection-sampling for unbiased
  bounded draws. The shared primitive used by `RandomSampler` and
  `DistributedSampler`. Mirrors PyTorch's `torch.randperm(n,
  generator=g)` upstream (`aten/src/ATen/native/RangeFactories.cpp`)
  but uses xorshift64 instead of MT19937 — the deviation is R-DEV-7
  (Rust ecosystem analog is materially better: xorshift64 is faster
  and trivially seedable from a single u64).

- REQ-3: `pub struct SequentialSampler` — yields `0, 1, ..., n-1`
  every epoch. `indices(_epoch)` collects from a `Range`. Mirrors
  `torch/utils/data/sampler.py:97-113` `class SequentialSampler`
  exactly.

- REQ-4: `pub struct RandomSampler` carrying `(size, seed)`. Each
  call to `indices(epoch)` builds a fresh `(0..size).collect()`,
  XORs the seed with `epoch * 0x9e3779b97f4a7c15`, and shuffles
  in-place via `shuffle_with_seed`. Mirrors
  `torch/utils/data/sampler.py:116-188` `class RandomSampler`
  (without-replacement branch); the with-replacement branch
  upstream is not exposed because PyTorch's `RandomSampler(replacement=True,
  num_samples=...)` is rarely used and `WeightedRandomSampler`
  with all-equal weights subsumes it.

- REQ-5: `pub struct DistributedSampler { num_samples, num_replicas,
  rank, shuffle, seed }` — partitions indices across distributed
  ranks via interleaved `step_by(num_replicas)`. The total index
  count is padded to `ceil(num_samples / num_replicas) *
  num_replicas` so all ranks process the same count. Mirrors
  `torch/utils/data/distributed.py:17-138` `class DistributedSampler`
  with its `drop_last=False` (default) branch and
  `indices[rank :: num_replicas]` subsampling. Constructor panics
  if `rank >= num_replicas` (matches upstream's `ValueError`).
  Builder methods `shuffle(bool)` and `seed(u64)` mirror the upstream
  `kwargs`.

- REQ-6: `pub struct WeightedRandomSampler { weights, num_samples,
  replacement, seed }` with two paths:
  - `replacement=true` → Walker's alias method (O(N) preprocess,
    O(1) per draw).
  - `replacement=false` → Efraimidis-Spirakis weighted reservoir
    sampling with a `BinaryHeap` of size `num_samples`
    (O(N log num_samples) instead of the naive O(N²)
    cumulative-rebuild).

  Mirrors `torch/utils/data/sampler.py:213-283` `class
  WeightedRandomSampler` which delegates to
  `torch.multinomial(weights, num_samples, replacement)`. The Rust
  implementation reproduces the same statistical contract without
  pulling in a tensor op for what is fundamentally a CPU-side
  sampling problem (R-DEV-7 — Rust ecosystem analog is cleaner;
  Walker's alias is the standard textbook algorithm and was already
  the documented PyTorch implementation under the hood).

- REQ-7: `pub struct BatchSampler<S: Sampler>` — wraps another
  sampler and chunks its indices into batches of `batch_size`. The
  `drop_last: bool` flag controls whether the final partial batch
  is dropped. `batches(epoch)` returns `Vec<Vec<usize>>` and
  `num_batches()` returns the chunk count. Mirrors
  `torch/utils/data/sampler.py:286-355` `class BatchSampler`
  exactly. Constructor panics if `batch_size == 0` (matches
  upstream's `ValueError`).

- REQ-8: Internal `fn bounded_u64(state: &mut u64, bound: u64) ->
  u64` (Lemire's nearly-divisionless method) and `fn xorshift64(state:
  &mut u64) -> u64` (the named PRNG step). The bounded-draw function
  uses the high 64 bits of the `u64 × u64 -> u128` product to avoid
  both modulo bias and xorshift64's known weak low bits. This is the
  load-bearing correctness primitive for `shuffle_with_seed` and the
  Walker's alias `draw`.

## Acceptance Criteria

- [x] AC-1: `pub trait Sampler: Send + Sync` with three required
  methods + the default `is_empty`.
- [x] AC-2: `pub fn shuffle_with_seed(&mut [usize], u64)` is the
  shared shuffle primitive, using `bounded_u64` for unbiased draws.
- [x] AC-3: `pub struct SequentialSampler` + `impl Sampler` yields
  `(0..size).collect()` regardless of epoch.
- [x] AC-4: `pub struct RandomSampler` + `impl Sampler` shuffles
  with the epoch-mixed seed.
- [x] AC-5: `pub struct DistributedSampler` constructor panics on
  invalid rank; `indices(epoch)` shuffles, pads to the divisible
  size, and returns the rank's interleaved subset.
- [x] AC-6: `pub struct WeightedRandomSampler` + `impl Sampler`
  dispatches on `replacement` flag — alias table vs reservoir
  sampling.
- [x] AC-7: `pub struct BatchSampler<S: Sampler>` + `batches(epoch)`
  yields `Vec<Vec<usize>>` honoring `drop_last`.
- [x] AC-8: `fn bounded_u64` + `fn xorshift64` are present and
  used internally; `test_random_sampler_reproducible` proves the
  determinism contract.

## Architecture

### `Sampler` trait + PRNG primitives (REQ-1, REQ-2, REQ-8)

The trait is intentionally simpler than upstream's `Iterator`-based
`__iter__` — we return `Vec<usize>` materialised because:
1. The loader needs to compute batch boundaries (chunk by
   `batch_size`, possibly drop the partial last batch), which needs
   a known length.
2. Most samplers compute their full output in O(N) anyway (Fisher-
   Yates shuffle, full Vec scan, etc.).
3. Workers that need a streaming index source can wrap the Vec in
   an iterator at the loader level.

`xorshift64(state)` is the canonical Marsaglia xorshift — `s ^= s
<< 13; s ^= s >> 7; s ^= s << 17`. The output stream has known weak
low bits; the bounded-draw helper compensates by reading the high
64 bits of the `u64 × u64 -> u128` product, which is the Lemire
nearly-divisionless technique. The rejection threshold `(-bound) %
bound` removes the residual bias when `bound` does not divide 2^64.
This is the documented bias-free fast path; ferrotorch chose it
over MT19937 (R-DEV-7) for speed and trivial seedability.

### `SequentialSampler` (REQ-3)

The simplest sampler — `(0..size).collect()`. Identical to upstream.
The epoch parameter is ignored (sequential order doesn't shuffle).

### `RandomSampler` (REQ-4)

`RandomSampler { size, seed }` produces a Fisher-Yates-shuffled
permutation of `0..size`. The effective seed is
`base_seed XOR (epoch * 0x9e3779b97f4a7c15)` where the magic
constant is the golden-ratio multiplicand (used by splitmix64 to
avalanche the seed). This matches PyTorch's
`generator.manual_seed(seed + epoch)` contract approximately — the
XOR-with-magic-mul gives better avalanche than the additive form
upstream uses.

### `DistributedSampler` (REQ-5)

The most complex sampler. The constructor stores
`(num_samples, num_replicas, rank, shuffle=true, seed=0)`. Builder
methods `shuffle(bool)` and `seed(u64)` consume-and-return so they
can be chained.

`indices(epoch)`:
1. Build `(0..num_samples).collect()`.
2. If `shuffle`, shuffle with the epoch-mixed seed.
3. Pad to `ceil(num_samples / num_replicas) * num_replicas` by
   wrapping `indices.push(indices[indices.len() - num_samples])` —
   wraps around to the first index, the second, etc. Matches
   upstream's `indices += indices[:padding_size]`.
4. Subsample for this rank via `indices.into_iter().skip(rank).
   step_by(num_replicas).collect()`. This is the
   "interleaved partitioning" (`indices[rank::num_replicas]` in
   upstream).

`len()` returns `num_samples.div_ceil(num_replicas)` (the
post-padding per-rank count).

The interleaved partitioning is the default upstream — `drop_last`
upstream changes the padding strategy (truncate instead of wrap)
but does not change the interleaving. Ferrotorch ships only the
wrap-padding path (which is upstream's `drop_last=False` default);
the `drop_last=True` upstream branch is a planned extension tracked
separately.

### `WeightedRandomSampler` (REQ-6) + `AliasTable` (REQ-6 cont)

Constructor validates weights (non-empty, non-negative, at least
one positive). With-replacement uses Walker's alias method via the
private `AliasTable` struct in `dataset.rs`:

- Vose's algorithm: scale each weight to mean 1, then pair "small"
  (< 1) bins with "large" (≥ 1) bins until all are balanced.
- `prob[i]` and `alias[i]` arrays — a draw picks bucket `i`
  uniformly, then flips a biased coin against `prob[i]` to choose
  between `i` and `alias[i]`.

The pop-pair loop has an important footgun documented in-source:
`while let (Some(s), Some(l)) = (small.pop(), large.pop())` would
consume from both stacks even when only one matches, silently
dropping the surviving entry — the explicit `while !small.is_empty()
&& !large.is_empty()` loop is necessary.

Without-replacement uses A-Res (Efraimidis-Spirakis): for each
item `i`, compute key `k_i = ln(u_i) / w_i` where `u_i ~
Uniform(0, 1]`. The top-k keys form a uniform weighted sample
without replacement; a min-heap of size `num_samples` keeps the
running cost at O(N log k). Tie-break on `idx` keeps the result
deterministic. The previous O(N²) cumulative-rebuild
implementation is replaced.

### `BatchSampler` (REQ-7)

Wraps another sampler and chunks via `slice::chunks(batch_size)`.
The `drop_last` flag controls whether the last partial chunk is
included. `num_batches()` computes the chunk count without
materialising. Constructor panics on `batch_size == 0` (matches
upstream's `ValueError`).

### Non-test production consumers

- `DataLoader::build_indices` in `dataloader.rs` constructs
  `RandomSampler::new(dataset.len(), seed)` and
  `SequentialSampler::new(dataset.len())` and calls
  `.indices(epoch)` on the result — the primary `Sampler` trait
  consumer.
- `DataLoader::with_sampler` in `dataloader.rs` accepts `Box<dyn
  Sampler>` for custom samplers (including the user-passed
  `DistributedSampler` for DDP training); the test
  `test_with_distributed_sampler` exercises this exact path.
- `pub use sampler::{...}` in `lib.rs` re-exports all seven items;
  the meta-crate glob propagates them as `ferrotorch::Sampler` /
  `ferrotorch::DistributedSampler` etc.
- `shuffle_with_seed` is called internally by `RandomSampler` and
  `DistributedSampler` (both above) — the production consumer for
  the PRNG primitive.

## Parity contract

`parity_ops = []`. Sampling is index arithmetic; the numerical
contract on the indices is "every index in `0..n` appears, no
duplicates" which is exercised by tests. Edge cases preserved:

- **Reproducibility across runs**: `RandomSampler::new(100,
  42).indices(0)` returns the same Vec on every invocation
  (asserted by `test_random_sampler_reproducible`). Matches
  PyTorch's deterministic `torch.Generator().manual_seed(...)`
  contract.
- **Epoch variance**: `indices(0) != indices(1)` (asserted by
  `test_random_sampler_different_epochs`). Matches PyTorch's
  `set_epoch(epoch)` discipline.
- **Distributed coverage**: with `world_size=3` and `n=10`, the
  union of all three ranks' indices covers `0..10` (asserted by
  `test_distributed_sampler_no_overlap`). The two extra padded
  indices wrap to repeat the first elements.
- **Weighted bias**: a weight of 1000 vs 1 produces >150 / 200 hits
  on the heavy index (asserted by
  `test_weighted_sampler_heavy_bias`). Matches the textbook
  Walker's-alias contract.
- **Zero-weight exclusion**: `weights=[1.0, 0.0, 1.0]` never
  samples index 1 (asserted by `test_weighted_sampler_zero_weight_excluded`).
- **Bias-free shuffle**: `shuffle_with_seed` uses Lemire's
  rejection-sampling, not `% bound`, so the distribution is
  unbiased even for `bound` that does not divide 2^64.
- **PRNG zero-state recovery**: `shuffle_with_seed(slice, 0)`
  remaps the zero seed to `0xdeadbeefcafe` to avoid the
  xorshift64-on-zero-state pathology. The same remap applies to
  `WeightedRandomSampler` epoch-zero with base-seed-zero.

## Verification

Unit tests in `mod tests in sampler.rs` (~30 tests across five
groups):

- SequentialSampler: `test_sequential_sampler` (3 assertions).
- RandomSampler: `_permutation`, `_reproducible`,
  `_different_epochs`, `_shuffled` (4 tests).
- `shuffle_with_seed`: `_deterministic`, `_different_seeds` (2).
- DistributedSampler: `_len`, `_no_overlap`,
  `_sequential_partitioning`, `_shuffle_reproducible`,
  `_shuffle_varies_by_epoch`, `_different_ranks_differ`,
  `_exact_division`, `_invalid_rank` (panic), `_single_replica` (9).
- WeightedRandomSampler: `_len`, `_replacement_count`,
  `_reproducible`, `_different_epochs`, `_zero_weight_excluded`,
  `_heavy_bias`, `_no_replacement`, `_no_replacement_subset`,
  `_empty_weights` (panic), `_negative_weight` (panic),
  `_all_zero_weights` (panic), `_no_replacement_too_many` (panic)
  (12).
- BatchSampler: `_basic`, `_drop_last`, `_exact`, `_exact_drop_last`,
  `_num_batches`, `_with_random`, `_single_element_batches`,
  `_zero_batch_size` (panic) (8).

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-data --lib sampler:: 2>&1 | tail -3
```

Expected: ~38 passed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub trait Sampler: Send + Sync` with `indices`, `len`, default `is_empty` in `sampler.rs`, mirroring `torch/utils/data/sampler.py:28-94`; non-test consumer: `DataLoader::with_sampler` in `dataloader.rs` accepts `Box<dyn Sampler>` and `DataLoader::build_indices` constructs sampler instances + calls `.indices(epoch)`. |
| REQ-2 | SHIPPED | impl: `pub fn shuffle_with_seed(indices: &mut [usize], seed: u64)` in `sampler.rs` using Fisher-Yates + Lemire's bounded draws + xorshift64; non-test consumer: `impl Sampler for RandomSampler in sampler.rs` and `impl Sampler for DistributedSampler in sampler.rs` both call `shuffle_with_seed(&mut indices, effective_seed)` inside their `indices(epoch)` body, and `pub use sampler::shuffle_with_seed` in `lib.rs` re-exports it for custom-sampler authors. |
| REQ-3 | SHIPPED | impl: `pub struct SequentialSampler { size: usize }` + `impl Sampler for SequentialSampler in sampler.rs` returning `(0..size).collect()`, mirroring `torch/utils/data/sampler.py:97-113`; non-test consumer: `DataLoader::build_indices` constructs `SequentialSampler::new(self.dataset.len())` when `shuffle=false && custom_sampler=None`. |
| REQ-4 | SHIPPED | impl: `pub struct RandomSampler { size, seed }` + `impl Sampler for RandomSampler in sampler.rs` shuffling with epoch-mixed seed `self.seed ^ (epoch * 0x9e3779b97f4a7c15)`, mirroring `torch/utils/data/sampler.py:116-188`; non-test consumer: `DataLoader::build_indices` constructs `RandomSampler::new(self.dataset.len(), self.seed)` when `shuffle=true`. |
| REQ-5 | SHIPPED | impl: `pub struct DistributedSampler { num_samples, num_replicas, rank, shuffle, seed }` + `impl Sampler for DistributedSampler in sampler.rs` with wrap-padding + interleaved subsampling, mirroring `torch/utils/data/distributed.py:17-138`; non-test consumer: `DataLoader::with_sampler` accepts `Box<dyn Sampler>` so users construct `DistributedSampler::new(n, world_size, rank).shuffle(true).seed(...)` and pass it through; `pub use sampler::DistributedSampler` in `lib.rs` re-exports the type. |
| REQ-6 | SHIPPED | impl: `pub struct WeightedRandomSampler { weights, num_samples, replacement, seed }` + `impl Sampler` in `sampler.rs` dispatching on `replacement` between `AliasTable::draw` (Walker's, O(1)) and A-Res min-heap (O(N log k)), mirroring `torch/utils/data/sampler.py:213-283`; non-test consumer: `pub use sampler::WeightedRandomSampler` in `lib.rs` re-exports the type, and `DataLoader::with_sampler` accepts it for class-imbalance training workflows. |
| REQ-7 | SHIPPED | impl: `pub struct BatchSampler<S: Sampler>` + `batches(epoch) -> Vec<Vec<usize>>` + `num_batches()` in `sampler.rs`, mirroring `torch/utils/data/sampler.py:286-355`; non-test consumer: `pub use sampler::BatchSampler` in `lib.rs` re-exports the type for users that need explicit batch-of-indices control (the default `DataLoader` does its own chunking inline rather than constructing a BatchSampler internally, mirroring upstream's behavior when `batch_sampler` is None). |
| REQ-8 | SHIPPED | impl: `fn bounded_u64(state: &mut u64, bound: u64) -> u64` and `fn xorshift64(state: &mut u64) -> u64` in `sampler.rs`, with the Lemire-rejection-sampling docstring naming the bias-correctness contract; non-test consumer: `fn shuffle_with_seed` and `fn AliasTable::draw` in `sampler.rs` both call `bounded_u64(&mut state, bound)`, and `WeightedRandomSampler::unit_uniform` calls `xorshift64(state) >> 11` to mint the 53-bit-mantissa float used in the reservoir sampling — every PRNG-consumer in this file routes through these primitives. |
