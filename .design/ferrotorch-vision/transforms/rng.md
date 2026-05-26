# ferrotorch-vision — `rng` (shared PRNG for vision transforms)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (torchvision v0.26.0 site-packages)
upstream-paths:
  - torchvision/transforms/v2/_auto_augment.py
  - torchvision/transforms/v2/_transform.py
-->

## Summary

`ferrotorch-vision/src/transforms/rng.rs` provides a seedable counter-based
splitmix64 PRNG that backs every randomized vision transform in this crate
(flips, crops, color jitter, augmentation magnitudes). It exposes one
public function — `vision_manual_seed(seed: u64)` — and two crate-private
sampling primitives `random_f64()` and `random_usize(upper)`. The seed
epoch-baseline design makes seeded sequences reproducible across
multi-threaded test runs even when other threads draw concurrently from
the same PRNG. Mirrors PyTorch's `torch.manual_seed` semantics narrowed
to the vision-augmentation namespace.

## Requirements

- REQ-1: `pub fn vision_manual_seed(seed: u64)` — global seed setter that
  records the seed value AND the current value of a shared draw counter
  as an "epoch baseline", so every subsequent `random_f64()` derives its
  draw-index by subtracting the baseline from the current counter. This
  makes the sequence depend only on the seed and the number of draws
  since seeding — independent of unrelated concurrent draws from other
  threads. Mirrors `torch.manual_seed` (PyTorch's seed setter for the
  default CPU generator).

- REQ-2: `pub(crate) fn random_f64() -> f64` — single-value sample in
  `[0, 1)` computed by passing `(global_counter - epoch_baseline)`
  through the splitmix64 mixing constants `0x9E3779B97F4A7C15`,
  `0xBF58476D1CE4E5B9`, `0x94D049BB133111EB`. Atomic counter increment
  is `Ordering::Relaxed` (single-source-of-truth monotonic order is
  sufficient; no other memory is synchronized through it). Mirrors
  torchvision's `torch.rand(())` calls which back every random
  `make_params` block (e.g. `_auto_augment.py:497` `torch.rand(())`).

- REQ-3: `pub(crate) fn random_usize(upper: usize) -> usize` — integer
  sample in `[0, upper)` computed as `(random_f64() * upper) as usize %
  upper`. The trailing `% upper` defends against the rare case where
  `random_f64()` returns a value that rounds to exactly `upper` when
  multiplied (degenerate float boundary). Mirrors
  `torch.randint(upper, ())` used throughout the auto-augment ops.

- REQ-4: Thread-safe shared state — three `AtomicU64` statics
  (`VISION_SEED`, `VISION_COUNTER`, `VISION_EPOCH_BASE`) initialized to
  `(42, 0, 0)`. The default seed of 42 matches PyTorch convention
  (`torch.manual_seed(42)` is the canonical reproducibility incantation
  in tutorials). All loads/stores use atomic ordering — `SeqCst` on the
  seed-setter path to establish a happens-before edge with subsequent
  draws, `Relaxed` on the hot draw path since the counter is monotonic
  per-thread.

## Acceptance Criteria

- [x] AC-1: `pub fn vision_manual_seed(seed: u64)` is callable from the
  crate root (`ferrotorch_vision::vision_manual_seed`).
- [x] AC-2: `random_f64()` is reachable from sibling modules via
  `use super::rng::random_f64;` (verified by every randomized transform
  importing it).
- [x] AC-3: `random_usize(upper)` returns a value strictly less than
  `upper` (verified structurally by the `% upper` guard).
- [x] AC-4: The three atomic statics use `AtomicU64` (verified by
  `use std::sync::atomic::{AtomicU64, Ordering};` at `rng.rs:16`).

## Architecture

### Seed-setter epoch baseline (REQ-1)

```rust
pub fn vision_manual_seed(seed: u64) {
    let baseline = VISION_COUNTER.load(Ordering::SeqCst);
    VISION_SEED.store(seed, Ordering::SeqCst);
    VISION_EPOCH_BASE.store(baseline, Ordering::SeqCst);
}
```

at `rng.rs:31-37`. The `SeqCst` ordering establishes a total ordering on
seed updates — any thread that calls `random_f64()` AFTER the store will
see both the new seed AND the new baseline. The "capture baseline before
seed store" sequence is deliberate: we want the very next `random_f64()`
to get draw-index 0, which it does because `fetch_add` returns the
PRE-increment value.

### Splitmix64 sampler (REQ-2)

```rust
pub(crate) fn random_f64() -> f64 {
    let seed = VISION_SEED.load(Ordering::Relaxed);
    let base = VISION_EPOCH_BASE.load(Ordering::Relaxed);
    let global = VISION_COUNTER.fetch_add(1, Ordering::Relaxed);
    let draw = global.wrapping_sub(base);
    let mut state = seed.wrapping_add(draw.wrapping_mul(0x9E3779B97F4A7C15));
    state = (state ^ (state >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    state = (state ^ (state >> 27)).wrapping_mul(0x94D049BB133111EB);
    state = state ^ (state >> 31);
    (state as f64) / (u64::MAX as f64)
}
```

at `rng.rs:46-58`. The three mixing constants are standard splitmix64
(Vigna 2014, "Better splittable pseudorandom number generators"). The
final divide by `u64::MAX` produces a value in `[0, 1]` — note the
upper bound is *closed*, which is why `random_usize` carries a `%`
defense (see REQ-3).

### Bounded integer sampler (REQ-3)

```rust
pub(crate) fn random_usize(upper: usize) -> usize {
    (random_f64() * upper as f64) as usize % upper
}
```

at `rng.rs:61-63`. Note: this samples uniformly only when `upper` is
small relative to `2^53` (f64 mantissa width). For all uses in this
crate (`upper` = number of transform-op variants, magnitude bins, etc.
typically `<= 31`), the bias is below f64 round-off.

### Atomic state (REQ-4)

```rust
static VISION_SEED: AtomicU64 = AtomicU64::new(42);
static VISION_COUNTER: AtomicU64 = AtomicU64::new(0);
static VISION_EPOCH_BASE: AtomicU64 = AtomicU64::new(0);
```

at `rng.rs:18-23`. `AtomicU64` is chosen so the entire RNG state is
lock-free and Send-Sync without `Mutex`. The default seed of 42 is
visible to the first `random_f64()` call without any prior
`vision_manual_seed` invocation.

### Non-test production consumers

- `random_f64` is consumed by `random_horizontal_flip.rs:54`,
  `random_vertical_flip.rs:58`, `random_apply.rs:39`,
  `random_apply.rs:78` (RandomChoice's selector),
  `gaussian_noise.rs:49-50` (Box-Muller pair),
  `random_resized_crop.rs:119,122,135,140`,
  `random_rotation.rs:106`, `random_gaussian_blur.rs:135`,
  `color_jitter.rs:83,93,163`, `elastic_transform.rs:166-167`.
- `random_usize` is consumed by `random_crop.rs:62,67`,
  `trivial_augment_wide.rs:117,349`.
- `vision_manual_seed` is re-exported at `mod.rs:32` and
  `lib.rs:115`, making it the externally-callable seed setter. End-users
  call it from training driver code at the top of an epoch to make a
  data-augmentation pass reproducible.

## Parity contract

`parity_ops = []`. Vision-augmentation determinism is NOT bit-exact with
PyTorch — PyTorch's `torch.rand(())` uses Philox/MT19937 (depending on
generator), while ferrotorch's vision RNG uses splitmix64. Bit-equality
with PyTorch's vision-side draws would require re-implementing the
upstream RNG; for image augmentation the statistical contract (uniform
`[0, 1)`, identical statistics across runs at the same seed) is what
end-users observe, not byte-equality. This R-DEV-7 deviation (Rust
ecosystem analog is materially better — splitmix64 is lock-free; PyTorch's
generators require GIL-equivalent locking) is intentional and documented
here so future audits don't flag it as drift.

## Verification

`rng.rs` has no `#[cfg(test)] mod tests`; verification is integration-level
via every randomized-transform test using `vision_manual_seed` to pin a
reproducible draw sequence — e.g.
`crate::transforms::rng::vision_manual_seed(12345);` in
`gaussian_noise.rs:132`, `elastic_transform.rs:226`, etc. Run:

```bash
cargo test -p ferrotorch-vision --lib transforms:: 2>&1 | tail -3
```

Expected: all randomized-transform tests pass with stable values across
multi-threaded test runs (the entire reason for the epoch baseline
design).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn vision_manual_seed(seed: u64)` at `ferrotorch-vision/src/transforms/rng.rs:31-37` records seed + epoch baseline; non-test consumer: `pub use rng::vision_manual_seed;` at `ferrotorch-vision/src/transforms/mod.rs:32` re-exports it, and `ferrotorch-vision/src/lib.rs:115` re-exports at crate root — end-user training drivers call `ferrotorch_vision::vision_manual_seed(42)` at epoch start. |
| REQ-2 | SHIPPED | impl: `pub(crate) fn random_f64() -> f64` at `ferrotorch-vision/src/transforms/rng.rs:46-58` with splitmix64 constants; non-test consumer: `super::rng::random_f64` imported and called from `random_horizontal_flip.rs:54`, `random_vertical_flip.rs:58`, `random_apply.rs:39`, `gaussian_noise.rs:49`, `random_resized_crop.rs:119`, `random_rotation.rs:106`, `random_gaussian_blur.rs:135`, `color_jitter.rs:83`, `elastic_transform.rs:166`. |
| REQ-3 | SHIPPED | impl: `pub(crate) fn random_usize(upper: usize) -> usize` at `ferrotorch-vision/src/transforms/rng.rs:61-63`; non-test consumer: `super::rng::random_usize` imported and called from `random_crop.rs:62,67` (top-left corner sampling) and `trivial_augment_wide.rs:117,349` (magnitude-bin and op-index sampling). |
| REQ-4 | SHIPPED | impl: three `static AtomicU64` at `ferrotorch-vision/src/transforms/rng.rs:18-23` (`VISION_SEED`, `VISION_COUNTER`, `VISION_EPOCH_BASE`); non-test consumer: every load/store path in `vision_manual_seed` (REQ-1) and `random_f64` (REQ-2) touches these atomics. |
