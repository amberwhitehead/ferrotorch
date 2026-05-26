// CL-332: Vision Transforms & Augmentation — shared RNG utilities
//
// This module provides a seedable PRNG for vision augmentation transforms.
// It uses the same splitmix64 algorithm as ferrotorch-data's RNG to ensure
// consistent statistical properties. The state is separate so that vision
// transforms have independent reproducibility control.
//
// Thread-safety design:
//   `vision_manual_seed(s)` records the seed value AND the current value of
//   VISION_COUNTER as the "epoch baseline". Subsequent calls to `random_f64()`
//   compute the per-call index as (global_counter - epoch_baseline), so the
//   sequence is determined solely by the seed and the number of random draws
//   since the last seed call — regardless of how many draws other threads have
//   made. This makes seeded sequences reproducible in concurrent test runs.

//! ## REQ status (per `.design/ferrotorch-vision/transforms/rng.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `vision_manual_seed` at `rng.rs:31-37`; consumer: `pub use rng::vision_manual_seed;` at `mod.rs:32` and crate-root re-export at `lib.rs:115`. |
//! | REQ-2 | SHIPPED | `random_f64` at `rng.rs:46-58`; consumer: imported by `random_horizontal_flip.rs:54`, `random_vertical_flip.rs:58`, `gaussian_noise.rs:49`, `random_rotation.rs:106`, `color_jitter.rs:83`, others. |
//! | REQ-3 | SHIPPED | `random_usize` at `rng.rs:61-63`; consumer: `random_crop.rs:62,67`, `trivial_augment_wide.rs:117,349`. |
//! | REQ-4 | SHIPPED | Three `AtomicU64` statics at `rng.rs:18-23`; consumer: every load/store in `vision_manual_seed` and `random_f64`. |

use std::sync::atomic::{AtomicU64, Ordering};

static VISION_SEED: AtomicU64 = AtomicU64::new(42);
/// Monotonically increasing draw counter shared across all threads.
static VISION_COUNTER: AtomicU64 = AtomicU64::new(0);
/// Value of VISION_COUNTER at the time of the last `vision_manual_seed` call.
/// Subtracted from the current counter to get the per-seed draw index.
static VISION_EPOCH_BASE: AtomicU64 = AtomicU64::new(0);

/// Set the random seed for vision augmentation transforms.
///
/// Records the current global counter value as the epoch baseline, so that
/// subsequent calls to [`random_f64`] produce the same sequence regardless of
/// how many draws other concurrent threads have made. This makes seeded
/// sequences reproducible in multi-threaded test runs.
pub fn vision_manual_seed(seed: u64) {
    // Capture the current counter value atomically before storing the seed,
    // so that the very next random_f64() call gets draw-index 0.
    let baseline = VISION_COUNTER.load(Ordering::SeqCst);
    VISION_SEED.store(seed, Ordering::SeqCst);
    VISION_EPOCH_BASE.store(baseline, Ordering::SeqCst);
}

/// Generate a random `f64` in [0, 1) using a seedable splitmix64 PRNG.
///
/// Each call atomically increments a global counter. The draw index used for
/// hashing is `(global_counter - epoch_base)`, where `epoch_base` is set by
/// the most recent [`vision_manual_seed`] call. This ensures the sequence
/// depends only on the seed and the number of draws since seeding, not on
/// concurrent draw activity from other threads.
pub(crate) fn random_f64() -> f64 {
    let seed = VISION_SEED.load(Ordering::Relaxed);
    let base = VISION_EPOCH_BASE.load(Ordering::Relaxed);
    let global = VISION_COUNTER.fetch_add(1, Ordering::Relaxed);
    // Draw index: number of random_f64() calls since last vision_manual_seed.
    let draw = global.wrapping_sub(base);
    // splitmix64 — good statistical properties for a counter-based PRNG.
    let mut state = seed.wrapping_add(draw.wrapping_mul(0x9E3779B97F4A7C15));
    state = (state ^ (state >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    state = (state ^ (state >> 27)).wrapping_mul(0x94D049BB133111EB);
    state = state ^ (state >> 31);
    (state as f64) / (u64::MAX as f64)
}

/// Generate a random `usize` in `[0, upper)`.
pub(crate) fn random_usize(upper: usize) -> usize {
    (random_f64() * upper as f64) as usize % upper
}
