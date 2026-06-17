//! Process-global default random number generator state, mirroring
//! `torch.manual_seed` / `torch.default_generator`.
//!
//! ## REQ status (per `.design/ferrotorch-core/rng.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (MT19937 engine) | SHIPPED | `Mt19937` engine mirrors `aten/src/ATen/core/MT19937RNGEngine.h:110-150` (state array of 624 uint32, twist/temper bits identical); non-test consumer: `creation::rand`/`creation::randn` source bits from `with_thread_rng`. |
//! | REQ-2 (`Generator` newtype) | SHIPPED | `pub struct Generator` exposes `new(seed)`, `manual_seed(seed)`, `next_uniform_f32/f64`, `next_normal_f32/f64`; non-test consumer: process-default `Generator` used by `creation::rand`. |
//! | REQ-3 (`manual_seed` top-level) | SHIPPED | `pub fn manual_seed(seed)` sets the process-global default CPU `Generator`, mirroring `torch.manual_seed` at `torch/random.py:46-86`. Non-test consumer: re-exported at `lib.rs` as `ferrotorch_core::manual_seed`. |
//! | REQ-4 (default-generator state) | SHIPPED | `DEFAULT_RNG: Mutex<Generator>` is initialised once from entropy and serialized like PyTorch's `default_generator` mutex; `manual_seed` reaches all threads. Non-test consumer: `with_thread_rng` invoked by `creation::rand`/`randn`. |
//! | REQ-5 (byte-exact parity for f32 rand) | SHIPPED | `Mt19937` reproduces `torch.manual_seed(42); torch.rand(10)` byte-for-byte; pinned by `ferrotorch-core/tests/divergence_manual_seed_parity.rs`. |

use std::cell::Cell;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{LazyLock, Mutex, MutexGuard};
use std::time::SystemTime;

// MT19937 constants — mirror `aten/src/ATen/core/MT19937RNGEngine.h:19-23`.
const MERSENNE_STATE_N: usize = 624;
const MERSENNE_STATE_M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UMASK: u32 = 0x8000_0000;
const LMASK: u32 = 0x7fff_ffff;

/// Mersenne Twister (MT19937) 32-bit engine — byte-identical to PyTorch CPU's
/// `at::mt19937_engine` in `aten/src/ATen/core/MT19937RNGEngine.h:110-150`.
#[derive(Clone)]
struct Mt19937 {
    state: [u32; MERSENNE_STATE_N],
    next: usize,
    left: i32,
    seed: u64,
}

impl Mt19937 {
    /// Seed using the upstream `init_with_uint32` algorithm at
    /// `MT19937RNGEngine.h:155-164` so the state array is byte-identical
    /// to PyTorch's after `torch.manual_seed(seed)`.
    fn new(seed: u64) -> Self {
        let mut state = [0u32; MERSENNE_STATE_N];
        state[0] = (seed & 0xffff_ffff) as u32;
        for j in 1..MERSENNE_STATE_N {
            // (1812433253 * (state[j-1] ^ (state[j-1] >> 30)) + j) wrapping u32.
            let prev = state[j - 1];
            state[j] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(j as u32);
        }
        Self {
            state,
            next: 0,
            left: 1,
            seed,
        }
    }

    fn mix_bits(u: u32, v: u32) -> u32 {
        (u & UMASK) | (v & LMASK)
    }

    fn twist(u: u32, v: u32) -> u32 {
        let mixed = Self::mix_bits(u, v) >> 1;
        if v & 1 != 0 { mixed ^ MATRIX_A } else { mixed }
    }

    /// Reload — mirrors `next_state` at `MT19937RNGEngine.h:174-188`.
    fn next_state(&mut self) {
        self.left = MERSENNE_STATE_N as i32;
        self.next = 0;
        // First loop: indices [0, N-M).
        // For j in (N - M + 1)..1 stepping down (loop body runs N-M times):
        //   state[p] = state[p + M] ^ twist(state[p], state[p+1])
        // We translate the pointer-walking C++ loop to indexed Rust.
        let n = MERSENNE_STATE_N;
        let m = MERSENNE_STATE_M;
        for p in 0..(n - m) {
            self.state[p] = self.state[p + m] ^ Self::twist(self.state[p], self.state[p + 1]);
        }
        for p in (n - m)..(n - 1) {
            // state[p] = state[p + M - N] ^ twist(state[p], state[p+1])
            self.state[p] = self.state[p + m - n] ^ Self::twist(self.state[p], self.state[p + 1]);
        }
        // Last: p = N - 1. state[N-1] = state[M-1] ^ twist(state[N-1], state[0])
        self.state[n - 1] = self.state[m - 1] ^ Self::twist(self.state[n - 1], self.state[0]);
    }

    /// Return one uint32 from the engine — `operator()` at
    /// `MT19937RNGEngine.h:139-150`.
    fn random_u32(&mut self) -> u32 {
        self.left -= 1;
        if self.left == 0 {
            self.next_state();
        }
        let mut y = self.state[self.next];
        self.next += 1;
        // Tempering.
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// Two u32 -> one u64 — `make64BitsFrom32Bits` at
    /// `CPUGeneratorImpl.cpp:71-73`. Order: hi = first call, lo = second.
    fn random_u64(&mut self) -> u64 {
        let hi = self.random_u32();
        let lo = self.random_u32();
        ((hi as u64) << 32) | (lo as u64)
    }
}

/// Explicit seeded RNG state, mirroring `torch.Generator`.
///
/// Construct with [`Generator::new`] (deterministic seed) or
/// [`Generator::seed_from_entropy`] (`SystemTime` + thread id, matches the
/// pre-#1537 default behaviour of `creation::rand`).
///
/// Holds an MT19937 engine plus the Box-Muller cache slots so the
/// normal-distribution stream agrees byte-for-byte with PyTorch
/// (`aten/src/ATen/core/DistributionsHelper.h:171-202`).
#[derive(Clone)]
pub struct Generator {
    engine: Mt19937,
    next_float_normal: Option<f32>,
    next_double_normal: Option<f64>,
}

impl std::fmt::Debug for Generator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Generator")
            .field("seed", &self.engine.seed)
            .field("has_cached_f32_normal", &self.next_float_normal.is_some())
            .field("has_cached_f64_normal", &self.next_double_normal.is_some())
            .finish()
    }
}

impl Generator {
    /// Construct a generator seeded with the given `u64` — byte-identical to
    /// `torch.Generator().manual_seed(seed)`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            engine: Mt19937::new(seed),
            next_float_normal: None,
            next_double_normal: None,
        }
    }

    /// Seed from `SystemTime::now()` + the current thread id — matches the
    /// pre-#1537 default behaviour of `creation::rand` when `manual_seed` is
    /// never called.
    #[must_use]
    pub fn seed_from_entropy() -> Self {
        let mut hasher = DefaultHasher::new();
        SystemTime::now().hash(&mut hasher);
        std::thread::current().id().hash(&mut hasher);
        let mut seed = hasher.finish();
        if seed == 0 {
            seed = 0xdead_beef_cafe;
        }
        Self::new(seed)
    }

    /// Reseed this generator in place. Drops any cached normal samples so
    /// the post-reseed Box-Muller stream restarts cleanly.
    pub fn manual_seed(&mut self, seed: u64) {
        self.engine = Mt19937::new(seed);
        self.next_float_normal = None;
        self.next_double_normal = None;
    }

    /// Underlying seed value.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.engine.seed
    }

    /// Raw uint32 from the engine — mirrors `CPUGeneratorImpl::random()` at
    /// `aten/src/ATen/CPUGeneratorImpl.cpp:226-228`.
    pub fn random_u32(&mut self) -> u32 {
        self.engine.random_u32()
    }

    /// Raw uint64 from the engine — mirrors `CPUGeneratorImpl::random64()` at
    /// `aten/src/ATen/CPUGeneratorImpl.cpp:235-239`.
    pub fn random_u64(&mut self) -> u64 {
        self.engine.random_u64()
    }

    /// Uniform-`[0,1)` f32, byte-identical to `at::uniform_real_distribution<float>(0,1)(gen)`
    /// from `aten/src/ATen/core/DistributionsHelper.h:106-113`:
    /// `((random() & ((1u<<24)-1)) as f32) * (1.0 / (1u<<24) as f32)`.
    pub fn next_uniform_f32(&mut self) -> f32 {
        const MASK: u32 = (1u32 << 24) - 1;
        const DIVISOR: f32 = 1.0f32 / ((1u32 << 24) as f32);
        let v = self.engine.random_u32() & MASK;
        (v as f32) * DIVISOR
    }

    /// Uniform-`[0,1)` f64, byte-identical to `at::uniform_real_distribution<double>(0,1)(gen)`:
    /// `((random64() & ((1u<<53)-1)) as f64) * (1.0 / (1u<<53) as f64)`.
    pub fn next_uniform_f64(&mut self) -> f64 {
        const MASK: u64 = (1u64 << 53) - 1;
        const DIVISOR: f64 = 1.0f64 / ((1u64 << 53) as f64);
        let v = self.engine.random_u64() & MASK;
        (v as f64) * DIVISOR
    }

    /// Standard-normal f32, byte-identical to `at::normal_distribution<float>(0,1)(gen)`
    /// from `aten/src/ATen/core/DistributionsHelper.h:172-201`. Box-Muller pairs
    /// are computed in `f32` acctype with `r * cos(theta)` returned and
    /// `r * sin(theta)` cached for the next call. `r = sqrt(-2 * log1p(-u2))`,
    /// `theta = 2 * pi * u1`.
    pub fn next_normal_f32(&mut self) -> f32 {
        if let Some(cached) = self.next_float_normal.take() {
            return cached;
        }
        let u1 = self.next_uniform_f32();
        let u2 = self.next_uniform_f32();
        let r = (-2.0f32 * (-u2).ln_1p()).sqrt();
        let theta = 2.0f32 * std::f32::consts::PI * u1;
        let (sin_t, cos_t) = theta.sin_cos();
        self.next_float_normal = Some(r * sin_t);
        r * cos_t
    }

    /// Standard-normal f64, byte-identical to `at::normal_distribution<double>(0,1)(gen)`.
    pub fn next_normal_f64(&mut self) -> f64 {
        if let Some(cached) = self.next_double_normal.take() {
            return cached;
        }
        let u1 = self.next_uniform_f64();
        let u2 = self.next_uniform_f64();
        let r = (-2.0f64 * (-u2).ln_1p()).sqrt();
        let theta = 2.0f64 * std::f64::consts::PI * u1;
        let (sin_t, cos_t) = theta.sin_cos();
        self.next_double_normal = Some(r * sin_t);
        r * cos_t
    }
}

impl Default for Generator {
    fn default() -> Self {
        Self::seed_from_entropy()
    }
}

static DEFAULT_RNG: LazyLock<Mutex<Generator>> =
    LazyLock::new(|| Mutex::new(Generator::seed_from_entropy()));

thread_local! {
    /// Protects the public closure accessor from recursive use on one thread.
    /// `std::sync::Mutex` is intentionally not reentrant; without this guard a
    /// nested default-RNG call would hang instead of reporting the API misuse.
    static DEFAULT_RNG_ACTIVE: Cell<bool> = const { Cell::new(false) };
}

struct DefaultRngAccessGuard;

impl DefaultRngAccessGuard {
    fn enter(operation: &'static str) -> Self {
        DEFAULT_RNG_ACTIVE.with(|active| {
            assert!(
                !active.get(),
                "{operation}: default RNG is already mutably borrowed; use an explicit Generator \
                 for nested random generation"
            );
            active.set(true);
        });
        Self
    }
}

impl Drop for DefaultRngAccessGuard {
    fn drop(&mut self) {
        DEFAULT_RNG_ACTIVE.with(|active| active.set(false));
    }
}

fn lock_default_rng() -> MutexGuard<'static, Generator> {
    DEFAULT_RNG
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(debug_assertions)]
struct DefaultRngTestSerialGuard {
    guard: Option<MutexGuard<'static, ()>>,
}

#[cfg(debug_assertions)]
impl Drop for DefaultRngTestSerialGuard {
    fn drop(&mut self) {
        if self.guard.is_some() {
            DEFAULT_RNG_TEST_SERIAL_ACTIVE.with(|active| active.set(false));
        }
    }
}

#[cfg(debug_assertions)]
thread_local! {
    static DEFAULT_RNG_TEST_SERIAL_ACTIVE: Cell<bool> = const { Cell::new(false) };
}

#[cfg(debug_assertions)]
fn default_rng_test_serial_guard() -> DefaultRngTestSerialGuard {
    if DEFAULT_RNG_TEST_SERIAL_ACTIVE.with(|active| active.get()) {
        return DefaultRngTestSerialGuard { guard: None };
    }

    static TEST_SERIAL_LOCK: Mutex<()> = Mutex::new(());
    let guard = TEST_SERIAL_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    DEFAULT_RNG_TEST_SERIAL_ACTIVE.with(|active| active.set(true));
    DefaultRngTestSerialGuard { guard: Some(guard) }
}

/// Run a debug/test-only critical section over the process-global default RNG.
///
/// This is for tests that need `manual_seed(seed)` and one or more following
/// random operations to be observed as one deterministic transaction under the
/// parallel Rust test harness. Release builds do not expose or pay for this
/// helper.
#[cfg(debug_assertions)]
#[doc(hidden)]
pub fn with_default_rng_test_lock<R>(f: impl FnOnce() -> R) -> R {
    let _guard = default_rng_test_serial_guard();
    f()
}

/// Set the process-global default CPU RNG seed — mirrors `torch.manual_seed`
/// at `torch/random.py:46-86`.
///
/// # Production consumer
///
/// `crate::creation::rand`/`randn` consume bits from this shared default
/// generator. Calling `manual_seed(42)` in any thread reseeds the stream seen
/// by subsequently scheduled random creation on any thread, matching PyTorch's
/// process-global CPU default generator.
pub fn manual_seed(seed: u64) {
    #[cfg(debug_assertions)]
    let _serial = default_rng_test_serial_guard();

    {
        let _access = DefaultRngAccessGuard::enter("manual_seed");
        lock_default_rng().manual_seed(seed);
    }
    // Mirror `torch.manual_seed`, which seeds BOTH the CPU and all CUDA
    // generators: `torch/random.py:67` calls `torch.cuda.manual_seed_all(seed)`
    // (`torch/cuda/random.py:112`). When a GPU backend is registered, forward
    // the seed to its per-device Philox manager so that
    // `creation::rand_on_device(.., Cuda)` after `manual_seed` is reproducible.
    // No-op (and no error surfaced) when CUDA is unavailable, matching torch's
    // "silently ignored if CUDA is not available" contract.
    if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
        let _ = backend.manual_seed_gpu(seed);
    }
}

/// Run a closure with mutable access to the process-global default generator.
///
/// The historical name is retained for API compatibility. The semantics match
/// PyTorch's default CPU generator: one process-wide stream serialized behind a
/// mutex, not a per-thread RNG. Used by `creation::rand` / `creation::randn`
/// and by `ferrotorch-nn` initialisers that don't take an explicit
/// [`Generator`].
pub fn with_thread_rng<R>(f: impl FnOnce(&mut Generator) -> R) -> R {
    #[cfg(debug_assertions)]
    let _serial = default_rng_test_serial_guard();

    let _access = DefaultRngAccessGuard::enter("with_thread_rng");
    let mut rng = lock_default_rng();
    f(&mut rng)
}

/// Clone the process-global CPU RNG state, including cached normal samples.
/// Checkpointing uses this to mirror `torch.get_rng_state()`.
pub(crate) fn thread_rng_state() -> Generator {
    #[cfg(debug_assertions)]
    let _serial = default_rng_test_serial_guard();

    let _access = DefaultRngAccessGuard::enter("thread_rng_state");
    lock_default_rng().clone()
}

/// Restore the process-global CPU RNG state. Checkpointing uses this
/// inside a fork-style guard so stochastic recomputation sees the same stream
/// as the original forward while the caller's surrounding stream is restored.
pub(crate) fn set_thread_rng_state(state: Generator) {
    #[cfg(debug_assertions)]
    let _serial = default_rng_test_serial_guard();

    let _access = DefaultRngAccessGuard::enter("set_thread_rng_state");
    *lock_default_rng() = state;
}

#[cfg(test)]
pub(crate) fn default_rng_test_lock() -> MutexGuard<'static, ()> {
    static TEST_LOCK: Mutex<()> = Mutex::new(());
    TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    /// `torch.manual_seed(42); torch.rand(10)` reference — captured from
    /// PyTorch 2.x CPU MT19937 as exact f32 bit patterns (see
    /// `ferrotorch-core/tests/divergence_manual_seed_parity.rs` for the
    /// live-call cross-check).
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

    #[test]
    fn mt19937_seed_42_matches_torch_rand_f32() {
        let mut g = Generator::new(42);
        for (i, &expected_bits) in TORCH_RAND_SEED_42_F32_BITS.iter().enumerate() {
            let got = g.next_uniform_f32();
            assert_eq!(
                got.to_bits(),
                expected_bits,
                "i={i}: got=0x{:08x} ({got:.17}), expected=0x{expected_bits:08x}",
                got.to_bits()
            );
        }
    }

    #[test]
    fn manual_seed_resets_default_generator() {
        let _guard = default_rng_test_lock();
        manual_seed(42);
        let a: Vec<u32> = (0..5)
            .map(|_| with_thread_rng(|g| g.random_u32()))
            .collect();
        manual_seed(42);
        let b: Vec<u32> = (0..5)
            .map(|_| with_thread_rng(|g| g.random_u32()))
            .collect();
        assert_eq!(a, b);
    }

    #[test]
    fn manual_seed_distinct_seeds_distinct_streams() {
        let _guard = default_rng_test_lock();
        manual_seed(42);
        let a = with_thread_rng(|g| g.random_u32());
        manual_seed(43);
        let b = with_thread_rng(|g| g.random_u32());
        assert_ne!(a, b);
    }

    fn draw_default_uniform_bits(n: usize) -> Vec<u32> {
        with_thread_rng(|g| (0..n).map(|_| g.next_uniform_f32().to_bits()).collect())
    }

    #[test]
    fn manual_seed_reaches_fresh_worker_thread() {
        let _guard = default_rng_test_lock();
        manual_seed(42);

        let worker = std::thread::spawn(|| draw_default_uniform_bits(10));
        let got = worker.join().expect("fresh RNG worker should not panic");

        assert_eq!(
            got.as_slice(),
            &TORCH_RAND_SEED_42_F32_BITS,
            "manual_seed must seed the process default generator seen by a fresh worker thread"
        );
    }

    #[test]
    fn manual_seed_reaches_existing_worker_thread() {
        let _guard = default_rng_test_lock();
        let (ready_tx, ready_rx) = mpsc::sync_channel::<()>(0);
        let (go_tx, go_rx) = mpsc::sync_channel::<()>(0);
        let (out_tx, out_rx) = mpsc::sync_channel::<Vec<u32>>(0);

        let worker = std::thread::spawn(move || {
            let _ = with_thread_rng(|g| g.random_u32());
            ready_tx.send(()).expect("main should wait for ready");
            go_rx.recv().expect("main should signal draw");
            out_tx
                .send(draw_default_uniform_bits(10))
                .expect("main should receive output");
        });

        ready_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("worker should initialize the default RNG");
        manual_seed(42);
        go_tx.send(()).expect("worker should still be waiting");
        let got = out_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("worker should return seeded output");
        worker.join().expect("existing RNG worker should not panic");

        assert_eq!(
            got.as_slice(),
            &TORCH_RAND_SEED_42_F32_BITS,
            "manual_seed must reseed the shared default generator even for an already-running worker"
        );
    }

    #[test]
    fn default_generator_stream_is_shared_across_threads() {
        let _guard = default_rng_test_lock();
        manual_seed(42);

        let first_half = std::thread::spawn(|| draw_default_uniform_bits(5))
            .join()
            .expect("worker should draw first half");
        let second_half = draw_default_uniform_bits(5);

        let mut got = first_half;
        got.extend(second_half);
        assert_eq!(
            got.as_slice(),
            &TORCH_RAND_SEED_42_F32_BITS,
            "worker and caller must advance one shared default stream, not independent streams"
        );
    }

    #[test]
    fn generator_clone_preserves_stream() {
        let mut g = Generator::new(12345);
        let _ = g.random_u32();
        let mut g2 = g.clone();
        assert_eq!(g.random_u32(), g2.random_u32());
        assert_eq!(g.random_u32(), g2.random_u32());
    }

    #[test]
    fn normal_box_muller_cache_used() {
        let mut g = Generator::new(42);
        // Two consecutive normal draws should consume exactly two uniform draws
        // total (one Box-Muller pair). The second draw must come from the cache.
        let n1 = g.next_normal_f32();
        let n2 = g.next_normal_f32();
        assert!(n1.is_finite() && n2.is_finite());
        assert!(g.next_float_normal.is_none(), "cache must be drained");
    }

    #[test]
    fn random_u64_concatenates_two_u32_in_order() {
        let mut g = Generator::new(7);
        let mut g2 = Generator::new(7);
        let hi = g2.random_u32();
        let lo = g2.random_u32();
        let expected = ((hi as u64) << 32) | (lo as u64);
        assert_eq!(g.random_u64(), expected);
    }
}
