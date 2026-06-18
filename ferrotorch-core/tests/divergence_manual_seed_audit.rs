//! Divergence audit for the uncommitted RNG / manual_seed work (#1542 / #1537).
//!
//! These probes pin the contract end-to-end (NOT just locally). Each one
//! corresponds to one of the five claims in the dispatch prompt:
//!
//! 1. `manual_seed` actually mutates state -> `reproducibility_probe`
//! 2. `rand`/`randn` consume the process default -> `reproducibility_probe`,
//!    `randn_reproducibility_probe`
//! 3. Reproducibility across in-process calls -> `reproducibility_probe`
//! 4. Torch byte-exact parity -> `torch_parity_probe` (live cross-check)
//! 5. `nn` init helpers thread the explicit generator -> covered by the
//!    `ferrotorch-nn` companion test (`tests/divergence_manual_seed_audit.rs`
//!    in that crate; see `nn_init_generator_threading_probe` below for the
//!    in-crate Generator-stream isolation probe).
//!
//! REFERENCES (live-call captured 2026-05-26 from a stock PyTorch CPU build,
//! see header of `tests/divergence_manual_seed_parity.rs` for the capture
//! script):
//!
//! `torch.manual_seed(42); torch.rand(10)` first 10 f32 bit patterns:
//!   0x3f61dc66 0x3f6a3db3 0x3ec406b8 0x3f75950e 0x3ec7e8d4
//!   0x3f19d447 0x3e835d78 0x3f4b2c14 0x3f70d666 0x3e0861e4

use ferrotorch_core::{Generator, manual_seed, rand, randn};
use std::sync::{Mutex, MutexGuard};

fn default_rng_test_lock() -> MutexGuard<'static, ()> {
    static TEST_LOCK: Mutex<()> = Mutex::new(());
    TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Fixture: `python3 -c "import torch, struct; torch.manual_seed(42);
/// [print(hex(struct.unpack('<I', struct.pack('<f', x))[0])) for x in
/// torch.rand(10).tolist()]"`. Re-verified 2026-05-26.
///
/// This array MUST match
/// `aten/src/ATen/core/TransformationHelper.h:84-89` `uniform_real<float>`
/// composed with `MT19937RNGEngine.h:139-150` tempering. Any drift means the
/// MT19937 state walk or the uniform-real transform desynced from upstream.
const TORCH_RAND_SEED_42_F32_BITS: [u32; 10] = [
    0x3f61_dc66,
    0x3f6a_3db3,
    0x3ec4_06b8,
    0x3f75_950e,
    0x3ec7_e8d4,
    0x3f19_d447,
    0x3e83_5d78,
    0x3f4b_2c14,
    0x3f70_d666,
    0x3e08_61e4,
];

/// Probe 1: `manual_seed(s).unwrap(); rand(...)` must be reproducible IN-PROCESS
/// across calls. If `manual_seed` is a no-op (or stashes the seed into a
/// field that `rand` never reads), calling `manual_seed(42)` twice and
/// then `rand` will return DIFFERENT bit patterns the second time because
/// the underlying default generator has been advanced by the first call.
///
/// Failure mode caught: vocab-only `manual_seed` (the `_seed: u64` rename
/// red flag) — i.e. function compiles, type-checks, but doesn't actually
/// reseed the engine.
#[test]
fn reproducibility_probe() {
    let _guard = default_rng_test_lock();
    manual_seed(42).unwrap();
    let a = rand::<f32>(&[10]).unwrap();
    manual_seed(42).unwrap();
    let b = rand::<f32>(&[10]).unwrap();
    let ad = a.data().unwrap();
    let bd = b.data().unwrap();
    for i in 0..10 {
        assert_eq!(
            ad[i].to_bits(),
            bd[i].to_bits(),
            "i={i}: re-seeded run differs: a=0x{:08x} b=0x{:08x}",
            ad[i].to_bits(),
            bd[i].to_bits(),
        );
    }
}

/// Probe 2: `manual_seed(s).unwrap(); randn(...)` must also be reproducible
/// in-process. Catches a default generator that resets the uniform engine on
/// `manual_seed` but forgets to drain the Box-Muller cache: the first
/// post-seed `randn` would consume the leftover cached normal sample
/// instead of a freshly-drawn pair.
#[test]
fn randn_reproducibility_probe() {
    let _guard = default_rng_test_lock();
    manual_seed(123).unwrap();
    let _ = rand::<f32>(&[1]).unwrap(); // advance state so cache differs
    let _ = randn::<f32>(&[1]).unwrap(); // poison Box-Muller cache
    manual_seed(123).unwrap();
    let a = randn::<f32>(&[10]).unwrap();
    manual_seed(123).unwrap();
    let b = randn::<f32>(&[10]).unwrap();
    let ad = a.data().unwrap();
    let bd = b.data().unwrap();
    for i in 0..10 {
        assert_eq!(
            ad[i].to_bits(),
            bd[i].to_bits(),
            "randn i={i}: re-seeded randn differs — Box-Muller cache not drained?"
        );
    }
}

/// Probe 3: `manual_seed(42).unwrap(); rand(&[10])` must agree BYTE-EXACT with
/// `torch.manual_seed(42); torch.rand(10)`. This is the headline claim of
/// #1537. The fixture is captured from live PyTorch.
///
/// Failure mode caught: vocab-only RNG (e.g. xorshift dressed up as
/// "MT19937" but with wrong state walk), or a wrong uniform-real
/// transform divisor, or wrong mantissa mask.
#[test]
fn torch_parity_probe() {
    let _guard = default_rng_test_lock();
    manual_seed(42).unwrap();
    let t = rand::<f32>(&[10]).unwrap();
    let data = t.data().unwrap();
    for (i, (&got, &expected_bits)) in data
        .iter()
        .zip(TORCH_RAND_SEED_42_F32_BITS.iter())
        .enumerate()
    {
        assert_eq!(
            got.to_bits(),
            expected_bits,
            "rand[{i}]: got 0x{:08x} ({got:.17}), expected 0x{expected_bits:08x} (torch.manual_seed(42); torch.rand(10))",
            got.to_bits()
        );
    }
}

/// Probe 4: an explicit `Generator` must produce an INDEPENDENT stream
/// from the process-default generator. Advancing the default through `rand`
/// must not affect a freshly-constructed `Generator::new(seed)`.
///
/// This also pins the "explicit generator threading" claim end-to-end:
/// two `Generator::new(s)` instances with the same seed produce identical
/// streams, and that stream is byte-exact with torch's seed-`s` stream.
///
/// Failure mode caught: an init helper that takes `_generator: &mut
/// Generator` but secretly samples from the process-default generator instead.
#[test]
fn nn_init_generator_threading_probe() {
    let _guard = default_rng_test_lock();

    // First: process-default stream advancement does not pollute an explicit
    // Generator.
    manual_seed(7).unwrap();
    let _ = rand::<f32>(&[100]).unwrap();
    let mut g_a = Generator::new(42);
    let v_a: Vec<u32> = (0..10).map(|_| g_a.next_uniform_f32().to_bits()).collect();
    assert_eq!(
        v_a.as_slice(),
        &TORCH_RAND_SEED_42_F32_BITS,
        "explicit Generator(42) stream polluted by default-generator advancement"
    );

    // Second: two explicit Generators with the same seed produce
    // identical streams. This is the "I really am consuming bits from
    // the passed-in generator" check that an init helper would need to
    // honour to be a true ferrotorch_core::Generator consumer.
    let mut g_b = Generator::new(42);
    let mut g_c = Generator::new(42);
    for i in 0..50 {
        let a = g_b.next_uniform_f32();
        let b = g_c.next_uniform_f32();
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "i={i}: two Generator::new(42) streams diverged"
        );
    }

    // Third: re-seeding the SAME generator via `manual_seed` (method on
    // &mut Generator) restarts the stream. Catches the case where the
    // method-level `manual_seed` is a no-op even though the top-level
    // one isn't.
    let mut g_d = Generator::new(42);
    let pre = g_d.next_uniform_f32().to_bits();
    let _ = g_d.next_uniform_f32();
    let _ = g_d.next_uniform_f32();
    g_d.manual_seed(42);
    let post = g_d.next_uniform_f32().to_bits();
    assert_eq!(
        pre, post,
        "Generator::manual_seed did not reset the engine: pre=0x{pre:08x} post=0x{post:08x}"
    );
    assert_eq!(
        post, TORCH_RAND_SEED_42_F32_BITS[0],
        "Generator::manual_seed(42); next_uniform_f32() != torch.manual_seed(42); torch.rand(1)[0]"
    );
}

/// Probe 5 (covers claim 3, cross-process determinism approximation):
/// the SECOND test in this binary, running AFTER probes 1-4 have
/// advanced the default state arbitrarily many times, must still
/// see byte-exact torch parity once it calls `manual_seed(42)`. This is
/// "cross-test isolation" — the default generator's previous state must not
/// matter once `manual_seed` is called.
///
/// Note: test execution order in Rust is non-deterministic by default,
/// so we cannot assert "run me last"; we instead assert the invariant
/// that holds regardless of order: every `manual_seed(42)` call must
/// produce the canonical seed-42 stream.
#[test]
fn cross_test_isolation_probe() {
    let _guard = default_rng_test_lock();

    // Burn arbitrary default-generator state.
    manual_seed(0xdead_beef).unwrap();
    for _ in 0..7 {
        let _ = rand::<f32>(&[13]).unwrap();
        let _ = randn::<f32>(&[5]).unwrap();
    }
    // After re-seeding, the canonical stream must reappear.
    manual_seed(42).unwrap();
    let t = rand::<f32>(&[10]).unwrap();
    let data = t.data().unwrap();
    for (i, (&got, &expected_bits)) in data
        .iter()
        .zip(TORCH_RAND_SEED_42_F32_BITS.iter())
        .enumerate()
    {
        assert_eq!(
            got.to_bits(),
            expected_bits,
            "post-burn rand[{i}]: default generator leaked across tests"
        );
    }
}
