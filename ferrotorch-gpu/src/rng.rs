//! CUDA RNG state management with Philox 4x32-10 counter-based generator.
//!
//! Provides deterministic, parallelizable random number generation for GPU
//! operations. The Philox algorithm is the same one used by CUDA's cuRAND
//! library: it maps a (counter, key) pair through 10 rounds of bijective
//! mixing to produce 4 uniform `u32` values per invocation.
//!
//! # Key types
//!
//! - [`PhiloxGenerator`] — stateful generator that tracks counter/offset
//! - [`PhiloxState`] — serializable snapshot for checkpoint save/restore
//! - [`CudaRngManager`] — per-device generator registry (one per GPU)
//! - [`cuda_rng_manager`] — global singleton accessor
//!
//! # GPU kernels
//!
//! PTX kernels generate random numbers directly on device without CPU-to-GPU
//! transfer:
//!
//! - `philox_uniform_kernel` — fills a buffer with uniform f32 in [0, 1)
//! - `philox_normal_kernel` — fills with standard normal f32 (Box-Muller)
//! - f64 uniform/normal kernels generate double-precision samples directly on
//!   device; f16/bf16 conversion kernels narrow CUDA-resident f32 streams
//!   without staging through host memory.
//!
//! # Fork/join for data parallelism
//!
//! [`fork_rng`] and [`join_rng`] snapshot and restore RNG states across
//! multiple devices, ensuring each DDP rank gets independent RNG streams.
//!
//! ## REQ status (per `.design/ferrotorch-gpu/rng.md`)
//!
//! Full evidence rows (impl + non-test production consumer + upstream
//! cites) live in the design doc; this synopsis is a one-line summary per
//! REQ.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`PhiloxState`) | SHIPPED | `pub struct PhiloxState in rng.rs` with documented constructors; consumer `CudaRngManager::state / restore in rng.rs` for save/restore; re-exported at `lib.rs` |
//! | REQ-2 (`PhiloxGenerator`) | SHIPPED | `pub struct PhiloxGenerator in rng.rs` with Philox 4x32-10 constants; consumer `CudaRngManager.generators in rng.rs` map stores instances; production callers via `backend_impl.rs` |
//! | REQ-3 (`CudaRngManager` + global accessor) | SHIPPED | `pub struct CudaRngManager in rng.rs` + `pub fn cuda_rng_manager` singleton; consumer four `crate::rng::cuda_rng_manager().lock()` sites in `backend_impl.rs` |
//! | REQ-4 (`fork_rng` / `join_rng`) | SHIPPED | `pub fn fork_rng / pub fn join_rng in rng.rs`; consumer re-exported at `lib.rs`; `ferrotorch-core/src/quantize.rs` defines parallel `cuda_rng::fork_rng / join_rng` Python-API surface wrapping these |
//! | REQ-5 (`gpu_philox_uniform`) | SHIPPED | `build_philox_uniform_f32_ptx` / `build_philox_uniform_f64_ptx` emit resident PyTorch-layout Philox kernels; consumers include `CudaBackendImpl::rand_uniform_*` and the resident dropout Philox kernels in this module |
//! | REQ-6 (`gpu_philox_normal`) | SHIPPED | `build_philox_normal_f32_ptx` / `build_philox_normal_f64_ptx` emit resident PyTorch-layout Philox kernels; consumer re-exported through `lib.rs`, consumed by `ferrotorch-distributions` Normal sampling path on GPU |
//! | REQ-7 (Manager↔backend wiring) | SHIPPED | `use crate::rng` import sites in `backend_impl.rs` inside dropout-philox / stochastic-rounding `CudaBackendImpl` methods; ferrotorch-core dispatches `Tensor::dropout` through `GpuBackend::dropout_philox_f32` |

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

#[cfg(feature = "cuda")]
use cudarc::driver::LaunchConfig;

use crate::buffer::CudaBuffer;
use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};
#[cfg(feature = "cuda")]
use crate::transfer::alloc_zeros_f32;
#[cfg(feature = "cuda")]
use crate::transfer::alloc_zeros_f64;

// ---------------------------------------------------------------------------
// Philox 4x32-10 constants
// ---------------------------------------------------------------------------

/// Philox multiplier constants (from the original Salmon et al. paper).
const PHILOX_M0: u32 = 0xD2511F53;
const PHILOX_M1: u32 = 0xCD9E8D57;

/// Philox Weyl sequence constants for key advancement.
const PHILOX_W0: u32 = 0x9E3779B9; // golden ratio
const PHILOX_W1: u32 = 0xBB67AE85; // sqrt(3) - 1

// ---------------------------------------------------------------------------
// PhiloxState — serializable snapshot
// ---------------------------------------------------------------------------

/// Serializable snapshot of a [`PhiloxGenerator`]'s state.
///
/// Used for checkpoint save/restore and fork/join in data parallelism.
///
/// # Construction
///
/// Use [`PhiloxState::new`] when starting from a fresh `(counter, seed)`
/// pair (offset starts at zero). Use [`PhiloxState::from_parts`] when
/// reconstructing from a checkpoint that captured a non-zero offset; that
/// constructor validates the offset is in the legal range `0..4`.
///
/// `counter` and `seed` remain public fields because they are legitimate
/// snapshot values that callers commonly read (and may compare). `offset`
/// is `pub(crate)` because external code can put it in an invalid state:
/// values `>= 4` produce a generator state the algorithm cannot represent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct PhiloxState {
    /// Counter value — incremented for each 4-tuple generated.
    pub counter: u64,
    /// Key/seed — set by the user via `manual_seed`.
    pub seed: u64,
    /// Offset into the current 4-tuple (0..4). Tracks how many of the
    /// 4 values from the last Philox round have been consumed.
    pub(crate) offset: u64,
}

impl PhiloxState {
    /// Create a new snapshot starting at counter `counter` with seed `seed`
    /// and a zero offset (no values consumed from the current 4-tuple).
    #[must_use]
    pub fn new(counter: u64, seed: u64) -> Self {
        Self {
            counter,
            seed,
            offset: 0,
        }
    }

    /// Reconstruct a snapshot from raw parts, validating the offset.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::InvalidState`] if `offset >= 4`. The Philox
    /// 4x32-10 algorithm produces 4 `u32` values per counter step, so the
    /// offset cursor must be in `0..4`.
    pub fn from_parts(counter: u64, seed: u64, offset: u64) -> GpuResult<Self> {
        if offset >= 4 {
            return Err(GpuError::InvalidState {
                message: format!("invalid Philox offset {offset}; must be < 4"),
            });
        }
        Ok(Self {
            counter,
            seed,
            offset,
        })
    }

    /// Read the offset cursor (`0..4`).
    #[must_use]
    pub fn offset(&self) -> u64 {
        self.offset
    }
}

// ---------------------------------------------------------------------------
// PhiloxGenerator
// ---------------------------------------------------------------------------

/// Philox 4x32-10 counter-based random number generator.
///
/// This is a CBRNG (counter-based RNG): given a counter and key, it
/// deterministically produces 4 uniform `u32` values. The counter is
/// incremented after each group of 4 values is consumed, and the offset
/// tracks which of the 4 values in the current group has been consumed.
///
/// The algorithm is:
/// 1. Split the 64-bit counter into two 32-bit halves (counter_lo, counter_hi)
/// 2. Split the 64-bit seed/key into two 32-bit halves (key_lo, key_hi)
/// 3. Run 10 rounds of Philox mixing (multiply + xor + key advance)
/// 4. Output the 4 mixed 32-bit values
pub struct PhiloxGenerator {
    /// Counter — incremented for each group of 4 random numbers generated.
    counter: u64,
    /// Key/seed — set by the user.
    seed: u64,
    /// Offset into the current 4-tuple (0..4).
    offset: u64,
    /// Cached output from the last Philox round. When offset is 0, this is
    /// invalid and needs to be regenerated.
    cached: [u32; 4],
}

impl PhiloxGenerator {
    /// Create a new Philox generator with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            counter: 0,
            seed,
            offset: 0,
            cached: [0; 4],
        }
    }

    /// Set the seed, resetting the counter and offset.
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
        self.counter = 0;
        self.offset = 0;
        self.cached = [0; 4];
    }

    /// Snapshot the generator state for checkpoint save.
    pub fn get_state(&self) -> PhiloxState {
        PhiloxState {
            counter: self.counter,
            seed: self.seed,
            offset: self.offset,
        }
    }

    /// Restore the generator from a previously saved state.
    pub fn set_state(&mut self, state: PhiloxState) {
        self.seed = state.seed;
        self.counter = state.counter;
        self.offset = state.offset;
        self.cached = [0; 4];
        // If offset is non-zero, we need to regenerate the cached tuple
        // so that subsequent next_u32() calls return the correct values.
        if self.offset > 0 {
            self.cached = philox_4x32_10(self.counter, self.seed);
        }
    }

    /// Advance the generator by `n_counters` counter steps, resetting the
    /// offset and cached values. Used when a GPU kernel consumes random
    /// numbers directly and we need to keep the CPU-side state in sync.
    pub fn advance(&mut self, n_counters: u64) {
        self.counter += n_counters;
        self.offset = 0;
        self.cached = [0; 4];
    }

    /// Generate the next uniform `u32` value.
    pub fn next_u32(&mut self) -> u32 {
        if self.offset == 0 {
            self.cached = philox_4x32_10(self.counter, self.seed);
        }
        let val = self.cached[self.offset as usize];
        self.offset += 1;
        if self.offset >= 4 {
            self.offset = 0;
            self.counter += 1;
        }
        val
    }

    /// Generate a uniform f32 value in [0, 1).
    ///
    /// Uses the standard conversion: `(u32 >> 8) * 2^-24`, which produces
    /// all representable floats in [0, 1) with uniform probability.
    pub fn next_f32(&mut self) -> f32 {
        let bits = self.next_u32();
        // Use the upper 24 bits for the mantissa (f32 has 23-bit mantissa + 1 implicit).
        // This gives 2^24 equally spaced values in [0, 1).
        (bits >> 8) as f32 * (1.0 / 16777216.0) // 2^-24
    }

    /// Generate `n` uniform f32 values in [0, 1).
    pub fn generate_uniform(&mut self, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.next_f32());
        }
        out
    }

    /// Generate `n` standard normal f32 values using the Box-Muller transform.
    ///
    /// Generates pairs of normal values from pairs of uniform values:
    ///   z0 = sqrt(-2 * ln(u1)) * cos(2 * pi * u2)
    ///   z1 = sqrt(-2 * ln(u1)) * sin(2 * pi * u2)
    ///
    /// If `n` is odd, the last unpaired value is still generated correctly
    /// (we just discard the second value of the final pair).
    pub fn generate_normal(&mut self, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        let two_pi = 2.0 * std::f32::consts::PI;

        while out.len() < n {
            // Generate u1 in (0, 1] to avoid ln(0).
            let mut u1 = self.next_f32();
            while u1 == 0.0 {
                u1 = self.next_f32();
            }
            let u2 = self.next_f32();

            let r = (-2.0 * u1.ln()).sqrt();
            let theta = two_pi * u2;

            out.push(r * theta.cos());
            if out.len() < n {
                out.push(r * theta.sin());
            }
        }

        out
    }
}

impl std::fmt::Debug for PhiloxGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhiloxGenerator")
            .field("counter", &self.counter)
            .field("seed", &self.seed)
            .field("offset", &self.offset)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Philox 4x32-10 core algorithm
// ---------------------------------------------------------------------------

/// Single Philox round: multiply-xor-swap.
///
/// Takes 4 state values and 2 key values, produces 4 mixed state values.
#[inline]
fn philox_round(c0: u32, c1: u32, c2: u32, c3: u32, k0: u32, k1: u32) -> (u32, u32, u32, u32) {
    // hi/lo decomposition of 32x32 -> 64 multiply
    let prod0 = (PHILOX_M0 as u64) * (c0 as u64);
    let hi0 = (prod0 >> 32) as u32;
    let lo0 = prod0 as u32;

    let prod1 = (PHILOX_M1 as u64) * (c2 as u64);
    let hi1 = (prod1 >> 32) as u32;
    let lo1 = prod1 as u32;

    // Feistel-like swap and xor with key
    let new_c0 = hi1 ^ c1 ^ k0;
    let new_c1 = lo1;
    let new_c2 = hi0 ^ c3 ^ k1;
    let new_c3 = lo0;

    (new_c0, new_c1, new_c2, new_c3)
}

/// Philox 4x32-10: 10 rounds of mixing a (counter, key) pair.
///
/// Takes a 64-bit counter and 64-bit key, returns 4 uniform u32 values.
fn philox_4x32_10(counter: u64, key: u64) -> [u32; 4] {
    // Split counter and key into 32-bit halves
    let mut c0 = counter as u32;
    let mut c1 = (counter >> 32) as u32;
    let mut c2 = 0u32; // Second counter half (we use a single 64-bit counter)
    let mut c3 = 0u32;

    let mut k0 = key as u32;
    let mut k1 = (key >> 32) as u32;

    // 10 rounds of mixing
    // Round 1
    (c0, c1, c2, c3) = philox_round(c0, c1, c2, c3, k0, k1);
    k0 = k0.wrapping_add(PHILOX_W0);
    k1 = k1.wrapping_add(PHILOX_W1);

    // Round 2
    (c0, c1, c2, c3) = philox_round(c0, c1, c2, c3, k0, k1);
    k0 = k0.wrapping_add(PHILOX_W0);
    k1 = k1.wrapping_add(PHILOX_W1);

    // Round 3
    (c0, c1, c2, c3) = philox_round(c0, c1, c2, c3, k0, k1);
    k0 = k0.wrapping_add(PHILOX_W0);
    k1 = k1.wrapping_add(PHILOX_W1);

    // Round 4
    (c0, c1, c2, c3) = philox_round(c0, c1, c2, c3, k0, k1);
    k0 = k0.wrapping_add(PHILOX_W0);
    k1 = k1.wrapping_add(PHILOX_W1);

    // Round 5
    (c0, c1, c2, c3) = philox_round(c0, c1, c2, c3, k0, k1);
    k0 = k0.wrapping_add(PHILOX_W0);
    k1 = k1.wrapping_add(PHILOX_W1);

    // Round 6
    (c0, c1, c2, c3) = philox_round(c0, c1, c2, c3, k0, k1);
    k0 = k0.wrapping_add(PHILOX_W0);
    k1 = k1.wrapping_add(PHILOX_W1);

    // Round 7
    (c0, c1, c2, c3) = philox_round(c0, c1, c2, c3, k0, k1);
    k0 = k0.wrapping_add(PHILOX_W0);
    k1 = k1.wrapping_add(PHILOX_W1);

    // Round 8
    (c0, c1, c2, c3) = philox_round(c0, c1, c2, c3, k0, k1);
    k0 = k0.wrapping_add(PHILOX_W0);
    k1 = k1.wrapping_add(PHILOX_W1);

    // Round 9
    (c0, c1, c2, c3) = philox_round(c0, c1, c2, c3, k0, k1);
    k0 = k0.wrapping_add(PHILOX_W0);
    k1 = k1.wrapping_add(PHILOX_W1);

    // Round 10 (final — no key advance needed)
    (c0, c1, c2, c3) = philox_round(c0, c1, c2, c3, k0, k1);

    [c0, c1, c2, c3]
}

// ---------------------------------------------------------------------------
// CudaRngManager — per-device generator registry
// ---------------------------------------------------------------------------

/// Per-device RNG state manager.
///
/// Maintains one [`PhiloxGenerator`] per GPU device, initialized lazily with
/// a default seed. The manager is accessed through the global singleton
/// [`cuda_rng_manager`].
pub struct CudaRngManager {
    /// One generator per GPU device, keyed by device ordinal.
    generators: HashMap<usize, PhiloxGenerator>,
    /// Default seed used when a device's generator is first accessed.
    default_seed: u64,
}

impl CudaRngManager {
    /// Create a new manager with the given default seed.
    fn new(default_seed: u64) -> Self {
        Self {
            generators: HashMap::new(),
            default_seed,
        }
    }

    /// Set the seed for a specific device, resetting its counter and offset.
    pub fn manual_seed(&mut self, device: usize, seed: u64) {
        let rng_gen = self
            .generators
            .entry(device)
            .or_insert_with(|| PhiloxGenerator::new(seed));
        rng_gen.set_seed(seed);
    }

    /// Set the seed for all currently-initialized devices.
    ///
    /// Also updates the default seed so that any future devices will use it.
    pub fn manual_seed_all(&mut self, seed: u64) {
        self.default_seed = seed;
        for rng_gen in self.generators.values_mut() {
            rng_gen.set_seed(seed);
        }
    }

    /// Get the RNG state for a specific device.
    ///
    /// Initializes the generator with the default seed if not already present.
    pub fn get_rng_state(&mut self, device: usize) -> PhiloxState {
        let default_seed = self.default_seed;
        self.generators
            .entry(device)
            .or_insert_with(|| PhiloxGenerator::new(default_seed))
            .get_state()
    }

    /// Set the RNG state for a specific device from a snapshot.
    pub fn set_rng_state(&mut self, device: usize, state: PhiloxState) {
        let rng_gen = self
            .generators
            .entry(device)
            .or_insert_with(|| PhiloxGenerator::new(state.seed));
        rng_gen.set_state(state);
    }

    /// Get a mutable reference to the generator for a specific device.
    ///
    /// Initializes the generator with the default seed if not already present.
    pub fn generator(&mut self, device: usize) -> &mut PhiloxGenerator {
        let default_seed = self.default_seed;
        self.generators
            .entry(device)
            .or_insert_with(|| PhiloxGenerator::new(default_seed))
    }
}

impl std::fmt::Debug for CudaRngManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaRngManager")
            .field("num_devices", &self.generators.len())
            .field("default_seed", &self.default_seed)
            .finish()
    }
}

/// Global singleton for the CUDA RNG manager.
///
/// The default seed is 0 (matching PyTorch's default). Use `manual_seed`
/// or `manual_seed_all` to set deterministic seeds before training.
static CUDA_RNG_MANAGER: LazyLock<Mutex<CudaRngManager>> =
    LazyLock::new(|| Mutex::new(CudaRngManager::new(0)));

/// Access the global CUDA RNG manager.
///
/// # Example
///
/// ```rust,no_run
/// use ferrotorch_gpu::rng::cuda_rng_manager;
///
/// let mut mgr = cuda_rng_manager().lock().unwrap();
/// mgr.manual_seed(0, 42);
/// let val = mgr.generator(0).next_f32();
/// ```
pub fn cuda_rng_manager() -> &'static Mutex<CudaRngManager> {
    &CUDA_RNG_MANAGER
}

// ---------------------------------------------------------------------------
// Fork/join for data parallelism
// ---------------------------------------------------------------------------

/// Snapshot the RNG state of multiple devices.
///
/// Used by DDP (distributed data parallel) to save each rank's RNG state
/// before a training step, ensuring reproducibility when resuming.
///
/// # Arguments
///
/// * `devices` — slice of device ordinals to snapshot
///
/// # Returns
///
/// A vector of `PhiloxState` in the same order as `devices`.
///
/// # Errors
///
/// Returns [`GpuError::InvalidState`] if the global RNG manager mutex is
/// poisoned (would only happen if a prior caller panicked while holding
/// the lock).
pub fn fork_rng(devices: &[usize]) -> GpuResult<Vec<PhiloxState>> {
    let mut mgr = CUDA_RNG_MANAGER
        .lock()
        .map_err(|e| GpuError::InvalidState {
            message: format!("CUDA RNG manager mutex poisoned: {e}"),
        })?;
    Ok(devices.iter().map(|&d| mgr.get_rng_state(d)).collect())
}

/// Restore RNG states for multiple devices from a previous [`fork_rng`] call.
///
/// # Arguments
///
/// * `devices` — slice of device ordinals (must match the `fork_rng` call)
/// * `states` — vector of `PhiloxState` to restore, in device order
///
/// # Errors
///
/// - Returns [`GpuError::ShapeMismatch`] if `devices.len() != states.len()`.
/// - Returns [`GpuError::InvalidState`] if the global RNG manager mutex is
///   poisoned.
pub fn join_rng(devices: &[usize], states: Vec<PhiloxState>) -> GpuResult<()> {
    if devices.len() != states.len() {
        return Err(GpuError::ShapeMismatch {
            op: "join_rng",
            expected: vec![devices.len()],
            got: vec![states.len()],
        });
    }
    let mut mgr = CUDA_RNG_MANAGER
        .lock()
        .map_err(|e| GpuError::InvalidState {
            message: format!("CUDA RNG manager mutex poisoned: {e}"),
        })?;
    for (&device, state) in devices.iter().zip(states) {
        mgr.set_rng_state(device, state);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// PTX kernels for Philox RNG on GPU
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
static PHILOX_UNIFORM_F64_PTX: LazyLock<String> = LazyLock::new(build_philox_uniform_f64_ptx);

#[cfg(feature = "cuda")]
static PHILOX_NORMAL_F64_PTX: LazyLock<String> = LazyLock::new(build_philox_normal_f64_ptx);

#[cfg(feature = "cuda")]
static PHILOX_UNIFORM_F32_PTX: LazyLock<String> = LazyLock::new(build_philox_uniform_f32_ptx);

#[cfg(feature = "cuda")]
static PHILOX_NORMAL_F32_PTX: LazyLock<String> = LazyLock::new(build_philox_normal_f32_ptx);

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
enum DropoutStorage {
    F32,
    F64,
    F16,
    BF16,
}

#[cfg(feature = "cuda")]
impl DropoutStorage {
    fn entry_prefix(self) -> &'static str {
        match self {
            Self::F32 => "philox_dropout_f32",
            Self::F64 => "philox_dropout_f64",
            Self::F16 => "philox_dropout_f16",
            Self::BF16 => "philox_dropout_bf16",
        }
    }

    fn max_vector_size(self) -> u32 {
        match self {
            Self::F32 => 4,
            Self::F64 => 2,
            Self::F16 | Self::BF16 => 8,
        }
    }

    fn element_shift(self) -> u32 {
        match self {
            Self::F32 => 2,
            Self::F64 => 3,
            Self::F16 | Self::BF16 => 1,
        }
    }

    fn uses_f64_accumulator(self) -> bool {
        matches!(self, Self::F64)
    }
}

#[cfg(feature = "cuda")]
fn torch_dropout_vector_size(n: usize, storage: DropoutStorage) -> u32 {
    let mut vector_size = storage.max_vector_size();
    while vector_size > 1 && !n.is_multiple_of(vector_size as usize) {
        vector_size /= 2;
    }
    vector_size
}

#[cfg(feature = "cuda")]
fn push_philox_rounds_ptx(ptx: &mut String) {
    for round in 0..10 {
        ptx.push_str(
            r"
    mul.wide.u32 %prod, %c0, 0xD2511F53;
    cvt.u32.u64 %lo_val, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %hi_val, %prod;
    mul.wide.u32 %prod, %c2, 0xCD9E8D57;
    cvt.u32.u64 %t2, %prod;
    shr.u64 %prod, %prod, 32;
    cvt.u32.u64 %t3, %prod;
    xor.b32 %t0, %t3, %c1;
    xor.b32 %t0, %t0, %k0;
    mov.u32 %t1, %t2;
    xor.b32 %t2, %hi_val, %c3;
    xor.b32 %t2, %t2, %k1;
    mov.u32 %t3, %lo_val;
    mov.u32 %c0, %t0;
    mov.u32 %c1, %t1;
    mov.u32 %c2, %t2;
    mov.u32 %c3, %t3;
",
        );
        if round != 9 {
            ptx.push_str(
                r"
    add.u32 %k0, %k0, 0x9E3779B9;
    add.u32 %k1, %k1, 0xBB67AE85;
",
            );
        }
    }
}

#[cfg(feature = "cuda")]
fn push_curand_uniform_f32_ptx(ptx: &mut String, word: &str, out: &str) {
    ptx.push_str("    cvt.rn.f32.u32 ");
    ptx.push_str(out);
    ptx.push_str(", ");
    ptx.push_str(word);
    ptx.push_str(";\n");
    ptx.push_str("    fma.rn.f32 ");
    ptx.push_str(out);
    ptx.push_str(", ");
    ptx.push_str(out);
    ptx.push_str(", 0f2F800000, 0f2F000000;\n");
}

#[cfg(feature = "cuda")]
fn push_store_f32_lane_ptx(ptx: &mut String, lane: usize, word: &str) {
    let label = format!("SKIP_UNIFORM_F32_LANE_{lane}");
    if lane == 0 {
        ptx.push_str("    mov.u64 %store_idx, %linear;\n");
    } else {
        ptx.push_str("    add.u64 %store_idx, %linear, %stride_reg;\n");
        for _ in 1..lane {
            ptx.push_str("    add.u64 %store_idx, %store_idx, %stride_reg;\n");
        }
    }
    ptx.push_str("    setp.ge.u64 %p_store, %store_idx, %n_reg;\n");
    ptx.push_str("    @%p_store bra ");
    ptx.push_str(&label);
    ptx.push_str(";\n");
    push_curand_uniform_f32_ptx(ptx, word, "%fval");
    ptx.push_str("    setp.eq.f32 %p_one, %fval, 0f3F800000;\n");
    ptx.push_str("    @%p_one mov.f32 %fval, 0f00000000;\n");
    ptx.push_str("    shl.b64 %off, %store_idx, 2;\n");
    ptx.push_str("    add.u64 %off, %out, %off;\n");
    ptx.push_str("    st.global.f32 [%off], %fval;\n");
    ptx.push_str(&label);
    ptx.push_str(":\n");
}

#[cfg(feature = "cuda")]
fn push_dropout_store_index_ptx(ptx: &mut String, lane: usize, vector_size: u32) {
    if vector_size == 1 {
        if lane == 0 {
            ptx.push_str("    mov.u64 %store_idx, %linear;\n");
        } else {
            ptx.push_str("    add.u64 %store_idx, %linear, %stride_reg;\n");
            for _ in 1..lane {
                ptx.push_str("    add.u64 %store_idx, %store_idx, %stride_reg;\n");
            }
        }
    } else if lane == 0 {
        ptx.push_str("    mov.u64 %store_idx, %linear;\n");
    } else {
        ptx.push_str("    add.u64 %store_idx, %linear, ");
        ptx.push_str(&lane.to_string());
        ptx.push_str(";\n");
    }
}

#[cfg(feature = "cuda")]
fn push_bf16_store_from_f32_ptx(ptx: &mut String, f32_reg: &str, addr_reg: &str) {
    ptx.push_str("    mov.b32 %bits, ");
    ptx.push_str(f32_reg);
    ptx.push_str(";\n");
    ptx.push_str(
        r"    shr.u32 %lsb, %bits, 16;
    and.b32 %lsb, %lsb, 1;
    add.u32 %round, %bits, 0x7FFF;
    add.u32 %round, %round, %lsb;
    shr.u32 %raw32, %round, 16;
    st.global.u16 [",
    );
    ptx.push_str(addr_reg);
    ptx.push_str("], %raw32;\n");
}

#[cfg(feature = "cuda")]
fn push_store_dropout_lane_ptx(
    ptx: &mut String,
    storage: DropoutStorage,
    vector_size: u32,
    lane: usize,
    word: &str,
) {
    let label = format!("SKIP_DROPOUT_{}_LANE_{lane}", storage.entry_prefix());
    push_dropout_store_index_ptx(ptx, lane, vector_size);
    ptx.push_str("    setp.ge.u64 %p_store, %store_idx, %n_reg;\n");
    ptx.push_str("    @%p_store bra ");
    ptx.push_str(&label);
    ptx.push_str(";\n");
    push_curand_uniform_f32_ptx(ptx, word, "%u");

    if storage.uses_f64_accumulator() {
        ptx.push_str(
            r"    cvt.f64.f32 %u64f, %u;
    setp.lt.f64 %p_keep, %u64f, %keep64;
",
        );
    } else {
        ptx.push_str("    setp.lt.f32 %p_keep, %u, %keep32;\n");
    }

    ptx.push_str("    shl.b64 %off, %store_idx, ");
    ptx.push_str(&storage.element_shift().to_string());
    ptx.push_str(
        r";
    add.u64 %addr_in, %in, %off;
    add.u64 %addr_out, %out, %off;
    add.u64 %addr_mask, %mask, %off;
",
    );

    match storage {
        DropoutStorage::F32 => {
            ptx.push_str(
                r"    ld.global.f32 %val32, [%addr_in];
    mul.f32 %out32, %val32, %scale32;
    selp.f32 %out32, %out32, %zero32, %p_keep;
    selp.f32 %mask32, %scale32, %zero32, %p_keep;
    st.global.f32 [%addr_out], %out32;
    st.global.f32 [%addr_mask], %mask32;
",
            );
        }
        DropoutStorage::F64 => {
            ptx.push_str(
                r"    ld.global.f64 %val64, [%addr_in];
    mul.rn.f64 %out64, %val64, %scale64;
    selp.f64 %out64, %out64, %zero64, %p_keep;
    selp.f64 %mask64, %scale64, %zero64, %p_keep;
    st.global.f64 [%addr_out], %out64;
    st.global.f64 [%addr_mask], %mask64;
",
            );
        }
        DropoutStorage::F16 => {
            ptx.push_str(
                r"    ld.global.b16 %hraw, [%addr_in];
    cvt.f32.f16 %val32, %hraw;
    mul.f32 %out32, %val32, %scale32;
    selp.f32 %out32, %out32, %zero32, %p_keep;
    selp.f32 %mask32, %scale32, %zero32, %p_keep;
    cvt.rn.f16.f32 %hout, %out32;
    cvt.rn.f16.f32 %hmask, %mask32;
    st.global.b16 [%addr_out], %hout;
    st.global.b16 [%addr_mask], %hmask;
",
            );
        }
        DropoutStorage::BF16 => {
            ptx.push_str(
                r"    ld.global.u16 %raw16, [%addr_in];
    cvt.u32.u16 %bits, %raw16;
    shl.b32 %bits, %bits, 16;
    mov.b32 %val32, %bits;
    mul.f32 %out32, %val32, %scale32;
    selp.f32 %out32, %out32, %zero32, %p_keep;
    selp.f32 %mask32, %scale32, %zero32, %p_keep;
",
            );
            push_bf16_store_from_f32_ptx(ptx, "%out32", "%addr_out");
            push_bf16_store_from_f32_ptx(ptx, "%mask32", "%addr_mask");
        }
    }

    ptx.push_str(&label);
    ptx.push_str(":\n");
}

#[cfg(feature = "cuda")]
fn build_philox_dropout_ptx(storage: DropoutStorage, vector_size: u32, entry_name: &str) -> String {
    let keep_param = if storage.uses_f64_accumulator() {
        ".f64"
    } else {
        ".f32"
    };
    let mut ptx = format!(
        r".version 7.0
.target sm_52
.address_size 64

.visible .entry {entry_name}(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u64 mask_ptr,
    .param .u64 n,
    .param {keep_param} keep_probability,
    .param {keep_param} scale,
    .param .u32 seed_lo,
    .param .u32 seed_hi,
    .param .u32 counter_lo,
    .param .u32 counter_hi,
    .param .u64 stride
) {{
    .reg .u32 %ltid, %bid, %bdim, %gid;
    .reg .u32 %slo, %shi, %clo, %chi;
    .reg .u32 %ctr_add_lo, %ctr_add_hi;
    .reg .u32 %c0, %c1, %c2, %c3, %k0, %k1;
    .reg .u32 %hi_val, %lo_val, %t0, %t1, %t2, %t3;
    .reg .u32 %bits, %round, %lsb, %raw32;
    .reg .u16 %raw16;
    .reg .b16 %hraw, %hout, %hmask;
    .reg .u64 %gid64, %tmp64, %ctr_add, %n_reg, %stride_reg, %linear, %step, %store_idx;
    .reg .u64 %prod, %in, %out, %mask, %off, %addr_in, %addr_out, %addr_mask;
    .reg .f32 %u, %keep32, %scale32, %zero32, %val32, %out32, %mask32;
    .reg .f64 %u64f, %keep64, %scale64, %zero64, %val64, %out64, %mask64;
    .reg .pred %p_done, %p_store, %p_keep;

    ld.param.u64 %in, [input_ptr];
    ld.param.u64 %out, [output_ptr];
    ld.param.u64 %mask, [mask_ptr];
    ld.param.u64 %n_reg, [n];
",
    );

    if storage.uses_f64_accumulator() {
        ptx.push_str(
            r"    ld.param.f64 %keep64, [keep_probability];
    ld.param.f64 %scale64, [scale];
    mov.f64 %zero64, 0d0000000000000000;
",
        );
    } else {
        ptx.push_str(
            r"    ld.param.f32 %keep32, [keep_probability];
    ld.param.f32 %scale32, [scale];
    mov.f32 %zero32, 0f00000000;
",
        );
    }

    ptx.push_str(
        r"    ld.param.u32 %slo, [seed_lo];
    ld.param.u32 %shi, [seed_hi];
    ld.param.u32 %clo, [counter_lo];
    ld.param.u32 %chi, [counter_hi];
    ld.param.u64 %stride_reg, [stride];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %ltid, %tid.x;
    mad.lo.u32 %gid, %bid, %bdim, %ltid;
    cvt.u64.u32 %gid64, %gid;
",
    );

    if vector_size == 1 {
        ptx.push_str(
            r"    mov.u64 %linear, %gid64;
    mul.lo.u64 %step, %stride_reg, 4;
",
        );
    } else {
        ptx.push_str("    mul.lo.u64 %linear, %gid64, ");
        ptx.push_str(&vector_size.to_string());
        ptx.push_str(";\n    mul.lo.u64 %step, %stride_reg, ");
        ptx.push_str(&vector_size.to_string());
        ptx.push_str(";\n");
    }

    let groups_per_iter = if vector_size == 1 {
        1
    } else {
        vector_size.div_ceil(4)
    };
    let lanes_per_iter = if vector_size == 1 { 4 } else { vector_size };

    ptx.push_str(
        r"    mov.u64 %ctr_add, 0;

LOOP_DROPOUT:
    setp.ge.u64 %p_done, %linear, %n_reg;
    @%p_done bra DONE;
",
    );

    for group in 0..groups_per_iter {
        if group == 0 {
            ptx.push_str("    mov.u64 %tmp64, %ctr_add;\n");
        } else {
            ptx.push_str("    add.u64 %tmp64, %ctr_add, ");
            ptx.push_str(&group.to_string());
            ptx.push_str(";\n");
        }
        ptx.push_str(
            r"    cvt.u32.u64 %ctr_add_lo, %tmp64;
    shr.u64 %tmp64, %tmp64, 32;
    cvt.u32.u64 %ctr_add_hi, %tmp64;
    add.cc.u32 %c0, %clo, %ctr_add_lo;
    addc.u32 %c1, %chi, %ctr_add_hi;
    cvt.u32.u64 %c2, %gid64;
    shr.u64 %tmp64, %gid64, 32;
    cvt.u32.u64 %c3, %tmp64;
    mov.u32 %k0, %slo;
    mov.u32 %k1, %shi;
",
        );
        push_philox_rounds_ptx(&mut ptx);

        let first_lane = (group * 4) as usize;
        let last_lane = lanes_per_iter.min((group + 1) * 4) as usize;
        let words = ["%c0", "%c1", "%c2", "%c3"];
        for lane in first_lane..last_lane {
            push_store_dropout_lane_ptx(
                &mut ptx,
                storage,
                vector_size,
                lane,
                words[lane - first_lane],
            );
        }
    }

    ptx.push_str(
        r"    add.u64 %linear, %linear, %step;
    add.u64 %ctr_add, %ctr_add, ",
    );
    ptx.push_str(&groups_per_iter.to_string());
    ptx.push_str(
        r";
    bra LOOP_DROPOUT;

DONE:
    ret;
}
",
    );

    ptx
}

#[cfg(feature = "cuda")]
fn build_philox_uniform_f32_ptx() -> String {
    let mut ptx = String::from(
        r".version 7.0
.target sm_52
.address_size 64

.visible .entry philox_uniform_kernel(
    .param .u64 out_ptr,
    .param .u64 n,
    .param .u32 seed_lo,
    .param .u32 seed_hi,
    .param .u32 counter_lo,
    .param .u32 counter_hi,
    .param .u64 stride
) {
    .reg .u32 %ltid, %bid, %bdim, %gid;
    .reg .u32 %slo, %shi, %clo, %chi;
    .reg .u32 %ctr_add_lo, %ctr_add_hi;
    .reg .u32 %c0, %c1, %c2, %c3, %k0, %k1;
    .reg .u32 %hi_val, %lo_val, %t0, %t1, %t2, %t3;
    .reg .u64 %gid64, %tmp64, %ctr_add, %n_reg, %stride_reg, %linear, %step, %store_idx;
    .reg .u64 %prod, %out, %off;
    .reg .f32 %fval;
    .reg .pred %p_done, %p_store, %p_one;

    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %n_reg, [n];
    ld.param.u32 %slo, [seed_lo];
    ld.param.u32 %shi, [seed_hi];
    ld.param.u32 %clo, [counter_lo];
    ld.param.u32 %chi, [counter_hi];
    ld.param.u64 %stride_reg, [stride];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %ltid, %tid.x;
    mad.lo.u32 %gid, %bid, %bdim, %ltid;
    cvt.u64.u32 %gid64, %gid;
    mov.u64 %linear, %gid64;
    mov.u64 %ctr_add, 0;
    mul.lo.u64 %step, %stride_reg, 4;

LOOP_UNIFORM_F32:
    setp.ge.u64 %p_done, %linear, %n_reg;
    @%p_done bra DONE;

    cvt.u32.u64 %ctr_add_lo, %ctr_add;
    shr.u64 %tmp64, %ctr_add, 32;
    cvt.u32.u64 %ctr_add_hi, %tmp64;
    add.cc.u32 %c0, %clo, %ctr_add_lo;
    addc.u32 %c1, %chi, %ctr_add_hi;
    cvt.u32.u64 %c2, %gid64;
    shr.u64 %tmp64, %gid64, 32;
    cvt.u32.u64 %c3, %tmp64;
    mov.u32 %k0, %slo;
    mov.u32 %k1, %shi;
",
    );
    push_philox_rounds_ptx(&mut ptx);
    push_store_f32_lane_ptx(&mut ptx, 0, "%c0");
    push_store_f32_lane_ptx(&mut ptx, 1, "%c1");
    push_store_f32_lane_ptx(&mut ptx, 2, "%c2");
    push_store_f32_lane_ptx(&mut ptx, 3, "%c3");
    ptx.push_str(
        r"
    add.u64 %linear, %linear, %step;
    add.u64 %ctr_add, %ctr_add, 1;
    bra LOOP_UNIFORM_F32;

DONE:
    ret;
}
",
    );
    ptx
}

#[cfg(feature = "cuda")]
fn push_store_normal_f32_pair_ptx(
    ptx: &mut String,
    pair: usize,
    x_word: &str,
    y_word: &str,
    base_lane: usize,
) {
    let skip_sin = format!("SKIP_NORMAL_F32_PAIR_{pair}_SIN");
    let skip_cos = format!("SKIP_NORMAL_F32_PAIR_{pair}_COS");
    push_curand_uniform_f32_ptx(ptx, x_word, "%u1");
    ptx.push_str("    cvt.rn.f32.u32 %u2, ");
    ptx.push_str(y_word);
    ptx.push_str(";\n");
    ptx.push_str("    fma.rn.f32 %theta, %u2, 0f30C90FDB, 0f30490FDB;\n");
    push_libdevice_logf_ptx(ptx, "%u1", "%ln_u1", pair);
    ptx.push_str("    mul.f32 %radius, %ln_u1, 0fC0000000;\n");
    ptx.push_str("    sqrt.rn.f32 %radius, %radius;\n");
    ptx.push_str("    sin.approx.f32 %sin_val, %theta;\n");
    ptx.push_str("    cos.approx.f32 %cos_val, %theta;\n");
    ptx.push_str("    mul.f32 %sin_val, %radius, %sin_val;\n");
    ptx.push_str("    mul.f32 %cos_val, %radius, %cos_val;\n");

    if base_lane == 0 {
        ptx.push_str("    mov.u64 %store_idx, %linear;\n");
    } else {
        ptx.push_str("    add.u64 %store_idx, %linear, %stride_reg;\n");
        for _ in 1..base_lane {
            ptx.push_str("    add.u64 %store_idx, %store_idx, %stride_reg;\n");
        }
    }
    ptx.push_str("    setp.ge.u64 %p_store, %store_idx, %n_reg;\n");
    ptx.push_str("    @%p_store bra ");
    ptx.push_str(&skip_sin);
    ptx.push_str(";\n");
    ptx.push_str("    shl.b64 %off, %store_idx, 2;\n");
    ptx.push_str("    add.u64 %off, %out, %off;\n");
    ptx.push_str("    st.global.f32 [%off], %sin_val;\n");
    ptx.push_str(&skip_sin);
    ptx.push_str(":\n");

    ptx.push_str("    add.u64 %store_idx, %store_idx, %stride_reg;\n");
    ptx.push_str("    setp.ge.u64 %p_store, %store_idx, %n_reg;\n");
    ptx.push_str("    @%p_store bra ");
    ptx.push_str(&skip_cos);
    ptx.push_str(";\n");
    ptx.push_str("    shl.b64 %off, %store_idx, 2;\n");
    ptx.push_str("    add.u64 %off, %out, %off;\n");
    ptx.push_str("    st.global.f32 [%off], %cos_val;\n");
    ptx.push_str(&skip_cos);
    ptx.push_str(":\n");
}

#[cfg(feature = "cuda")]
fn push_libdevice_logf_ptx(ptx: &mut String, input: &str, output: &str, label_suffix: usize) {
    // Transcribed from libdevice `__nv_logf`. The RNG module loads raw PTX
    // through `module_cache`, so keeping this inline avoids CUDA C++/NVRTC while
    // matching cuRAND's normal transform.
    let normal = format!("LOGF_NORMAL_{label_suffix}");
    let finite = format!("LOGF_FINITE_{label_suffix}");
    let done = format!("LOGF_DONE_{label_suffix}");

    ptx.push_str("    mov.f32 %f38, ");
    ptx.push_str(input);
    ptx.push_str(
        r";
    setp.geu.f32 %p1, %f38, 0f00800000;
    mov.f32 %f39, 0f00000000;
    @%p1 bra ",
    );
    ptx.push_str(&normal);
    ptx.push_str(
        r";
    mul.rn.f32 %f38, %f38, 0f4B000000;
    mov.f32 %f39, 0fC1B80000;
",
    );
    ptx.push_str(&normal);
    ptx.push_str(
        r":
    mov.b32 %r1, %f38;
    add.s32 %r2, %r1, -1059760811;
    and.b32 %r3, %r2, -8388608;
    sub.s32 %r4, %r1, %r3;
    mov.b32 %f11, %r4;
    cvt.rn.f32.s32 %f12, %r3;
    mov.f32 %f13, 0f34000000;
    fma.rn.f32 %f14, %f12, %f13, %f39;
    add.rn.f32 %f15, %f11, 0fBF800000;
    mov.f32 %f16, 0f3E1039F6;
    mov.f32 %f17, 0fBE055027;
    fma.rn.f32 %f18, %f17, %f15, %f16;
    mov.f32 %f19, 0fBDF8CDCC;
    fma.rn.f32 %f20, %f18, %f15, %f19;
    mov.f32 %f21, 0f3E0F2955;
    fma.rn.f32 %f22, %f20, %f15, %f21;
    mov.f32 %f23, 0fBE2AD8B9;
    fma.rn.f32 %f24, %f22, %f15, %f23;
    mov.f32 %f25, 0f3E4CED0B;
    fma.rn.f32 %f26, %f24, %f15, %f25;
    mov.f32 %f27, 0fBE7FFF22;
    fma.rn.f32 %f28, %f26, %f15, %f27;
    mov.f32 %f29, 0f3EAAAA78;
    fma.rn.f32 %f30, %f28, %f15, %f29;
    mov.f32 %f31, 0fBF000000;
    fma.rn.f32 %f32, %f30, %f15, %f31;
    mul.rn.f32 %f33, %f32, %f15;
    fma.rn.f32 %f34, %f33, %f15, %f15;
    mov.f32 %f35, 0f3F317218;
    fma.rn.f32 %f41, %f14, %f35, %f34;
    setp.lt.u32 %p2, %r1, 2139095040;
    @%p2 bra ",
    );
    ptx.push_str(&finite);
    ptx.push_str(
        r";
    mov.f32 %f36, 0f7F800000;
    fma.rn.f32 %f41, %f38, %f36, %f36;
",
    );
    ptx.push_str(&finite);
    ptx.push_str(
        r":
    setp.neu.f32 %p3, %f38, 0f00000000;
    @%p3 bra ",
    );
    ptx.push_str(&done);
    ptx.push_str(
        r";
    mov.f32 %f41, 0fFF800000;
",
    );
    ptx.push_str(&done);
    ptx.push_str(
        r":
    mov.f32 ",
    );
    ptx.push_str(output);
    ptx.push_str(", %f41;\n");
}

#[cfg(feature = "cuda")]
fn push_libdevice_log_f64_ptx(ptx: &mut String, input: &str, output: &str) {
    // Transcribed from libdevice `__nv_log`; emitted once in the f64 normal
    // kernel to match `curand_normal2_double`.
    ptx.push_str("    mov.f64 %fd56, ");
    ptx.push_str(input);
    ptx.push_str(
        r";
    {
    .reg .b32 %temp;
    mov.b64 {%temp, %r23}, %fd56;
    }
    {
    .reg .b32 %temp;
    mov.b64 {%r24, %temp}, %fd56;
    }
    setp.gt.s32 %p1, %r23, 1048575;
    mov.b32 %r25, -1023;
    @%p1 bra LOG_F64_NORMAL_EXP;
    mul.rn.f64 %fd56, %fd56, 0d4350000000000000;
    {
    .reg .b32 %temp;
    mov.b64 {%temp, %r23}, %fd56;
    }
    {
    .reg .b32 %temp;
    mov.b64 {%r24, %temp}, %fd56;
    }
    mov.b32 %r25, -1077;
LOG_F64_NORMAL_EXP:
    setp.lt.s32 %p2, %r23, 1;
    @%p2 bra LOG_F64_SPECIAL;
    setp.gt.s32 %p3, %r23, 2146435071;
    @%p3 bra LOG_F64_SPECIAL;
    shr.u32 %r14, %r23, 20;
    add.s32 %r26, %r25, %r14;
    and.b32 %r15, %r23, -2146435073;
    or.b32 %r16, %r15, 1072693248;
    mov.b64 %fd57, {%r24, %r16};
    setp.lt.s32 %p5, %r16, 1073127583;
    @%p5 bra LOG_F64_REDUCED;
    {
    .reg .b32 %temp;
    mov.b64 {%r17, %temp}, %fd57;
    }
    {
    .reg .b32 %temp;
    mov.b64 {%temp, %r18}, %fd57;
    }
    add.s32 %r19, %r18, -1048576;
    mov.b64 %fd57, {%r17, %r19};
    add.s32 %r26, %r26, 1;
LOG_F64_REDUCED:
    sub.rn.f64 %fd12, %fd57, 0d3FF0000000000000;
    add.rn.f64 %fd13, %fd57, 0d3FF0000000000000;
    rcp.approx.ftz.f64 %fd14, %fd13;
    neg.f64 %fd15, %fd13;
    mov.f64 %fd16, 0d3FF0000000000000;
    fma.rn.f64 %fd17, %fd15, %fd14, %fd16;
    fma.rn.f64 %fd18, %fd17, %fd17, %fd17;
    fma.rn.f64 %fd19, %fd18, %fd14, %fd14;
    mul.rn.f64 %fd20, %fd12, %fd19;
    add.rn.f64 %fd21, %fd20, %fd20;
    mul.rn.f64 %fd22, %fd21, %fd21;
    mov.f64 %fd23, 0d3ED0EE258B7A8B04;
    mov.f64 %fd24, 0d3EB1380B3AE80F1E;
    fma.rn.f64 %fd25, %fd24, %fd22, %fd23;
    mov.f64 %fd26, 0d3EF3B2669F02676F;
    fma.rn.f64 %fd27, %fd25, %fd22, %fd26;
    mov.f64 %fd28, 0d3F1745CBA9AB0956;
    fma.rn.f64 %fd29, %fd27, %fd22, %fd28;
    mov.f64 %fd30, 0d3F3C71C72D1B5154;
    fma.rn.f64 %fd31, %fd29, %fd22, %fd30;
    mov.f64 %fd32, 0d3F624924923BE72D;
    fma.rn.f64 %fd33, %fd31, %fd22, %fd32;
    mov.f64 %fd34, 0d3F8999999999A3C4;
    fma.rn.f64 %fd35, %fd33, %fd22, %fd34;
    mov.f64 %fd36, 0d3FB5555555555554;
    fma.rn.f64 %fd37, %fd35, %fd22, %fd36;
    sub.rn.f64 %fd38, %fd12, %fd21;
    add.rn.f64 %fd39, %fd38, %fd38;
    neg.f64 %fd40, %fd21;
    fma.rn.f64 %fd41, %fd40, %fd12, %fd39;
    mul.rn.f64 %fd42, %fd19, %fd41;
    mul.rn.f64 %fd43, %fd37, %fd22;
    fma.rn.f64 %fd44, %fd43, %fd21, %fd42;
    xor.b32 %r20, %r26, -2147483648;
    mov.b32 %r21, 1127219200;
    mov.b64 %fd45, {%r20, %r21};
    mov.b32 %r22, -2147483648;
    mov.b64 %fd46, {%r22, %r21};
    sub.rn.f64 %fd47, %fd45, %fd46;
    mov.f64 %fd48, 0d3FE62E42FEFA39EF;
    fma.rn.f64 %fd49, %fd47, %fd48, %fd21;
    neg.f64 %fd50, %fd47;
    fma.rn.f64 %fd51, %fd50, %fd48, %fd49;
    sub.rn.f64 %fd52, %fd51, %fd21;
    sub.rn.f64 %fd53, %fd44, %fd52;
    mov.f64 %fd54, 0d3C7ABC9E3B39803F;
    fma.rn.f64 %fd55, %fd47, %fd54, %fd53;
    add.rn.f64 %fd58, %fd49, %fd55;
    bra.uni LOG_F64_DONE;
LOG_F64_SPECIAL:
    mov.f64 %fd10, 0d7FF0000000000000;
    fma.rn.f64 %fd58, %fd56, %fd10, %fd10;
    {
    .reg .b32 %temp;
    mov.b64 {%temp, %r13}, %fd56;
    }
    mov.b32 %f1, %r13;
    setp.neu.f32 %p4, %f1, 0f00000000;
    @%p4 bra LOG_F64_DONE;
    mov.f64 %fd58, 0dFFF0000000000000;
LOG_F64_DONE:
    mov.f64 ",
    );
    ptx.push_str(output);
    ptx.push_str(", %fd58;\n");
}

#[cfg(feature = "cuda")]
fn push_libdevice_sincospi_f64_ptx(ptx: &mut String, input: &str, sin_out: &str, cos_out: &str) {
    // Transcribed from libdevice `__nv_sincospi`; emitted once in the f64
    // normal kernel because cuRAND computes sin/cos on a pi-scaled argument.
    ptx.push_str("    mov.f64 %fd55, ");
    ptx.push_str(input);
    ptx.push_str(
        r";
    {
    .reg .b32 %temp;
    mov.b64 {%temp, %r2}, %fd55;
    }
    add.s32 %r3, %r2, %r2;
    setp.lt.u32 %p1, %r3, -2038431743;
    mov.f64 %fd16, 0d0000000000000000;
    @%p1 bra SINCOSPI_REDUCE;
    mul.rn.f64 %fd55, %fd55, %fd16;
SINCOSPI_REDUCE:
    {
    .reg .b32 %temp;
    mov.b64 {%r4, %temp}, %fd55;
    }
    {
    .reg .b32 %temp;
    mov.b64 {%temp, %r5}, %fd55;
    }
    add.s32 %r6, %r5, 1048576;
    mov.b64 %fd17, {%r4, %r6};
    cvt.rni.f64.f64 %fd18, %fd17;
    cvt.rzi.s64.f64 %rd3, %fd18;
    cvt.u32.u64 %r1, %rd3;
    neg.f64 %fd19, %fd18;
    mov.f64 %fd20, 0d3FE0000000000000;
    fma.rn.f64 %fd21, %fd19, %fd20, %fd55;
    mul.rn.f64 %fd22, %fd21, 0d3CA1A62633145C07;
    mov.f64 %fd23, 0d400921FB54442D18;
    fma.rn.f64 %fd24, %fd21, %fd23, %fd22;
    mul.rn.f64 %fd25, %fd24, %fd24;
    mov.f64 %fd26, 0d3E21EEA7C1EF8528;
    mov.f64 %fd27, 0dBDA8FF8320FD8164;
    fma.rn.f64 %fd28, %fd27, %fd25, %fd26;
    mov.f64 %fd29, 0dBE927E4F8E06E6D9;
    fma.rn.f64 %fd30, %fd28, %fd25, %fd29;
    mov.f64 %fd31, 0d3EFA01A019DDBCE9;
    fma.rn.f64 %fd32, %fd30, %fd25, %fd31;
    mov.f64 %fd33, 0dBF56C16C16C15D47;
    fma.rn.f64 %fd34, %fd32, %fd25, %fd33;
    mov.f64 %fd35, 0d3FA5555555555551;
    fma.rn.f64 %fd36, %fd34, %fd25, %fd35;
    mov.f64 %fd37, 0dBFE0000000000000;
    fma.rn.f64 %fd38, %fd36, %fd25, %fd37;
    mov.f64 %fd39, 0d3FF0000000000000;
    fma.rn.f64 %fd59, %fd38, %fd25, %fd39;
    mov.f64 %fd40, 0dBE5AE5F12CB0D246;
    mov.f64 %fd41, 0d3DE5DB65F9785EBA;
    fma.rn.f64 %fd42, %fd41, %fd25, %fd40;
    mov.f64 %fd43, 0d3EC71DE369ACE392;
    fma.rn.f64 %fd44, %fd42, %fd25, %fd43;
    mov.f64 %fd45, 0dBF2A01A019DB62A1;
    fma.rn.f64 %fd46, %fd44, %fd25, %fd45;
    mov.f64 %fd47, 0d3F81111111110818;
    fma.rn.f64 %fd48, %fd46, %fd25, %fd47;
    mov.f64 %fd49, 0dBFC5555555555554;
    fma.rn.f64 %fd50, %fd48, %fd25, %fd49;
    fma.rn.f64 %fd52, %fd50, %fd25, %fd16;
    fma.rn.f64 %fd58, %fd52, %fd24, %fd24;
    and.b64 %rd4, %rd3, 1;
    setp.eq.b64 %p2, %rd4, 1;
    not.pred %p3, %p2;
    @%p3 bra SINCOSPI_NO_ODD_SWAP;
    {
    .reg .b32 %temp;
    mov.b64 {%temp, %r7}, %fd58;
    }
    {
    .reg .b32 %temp;
    mov.b64 {%r8, %temp}, %fd58;
    }
    xor.b32 %r9, %r7, -2147483648;
    mov.b64 %fd5, {%r8, %r9};
    mov.f64 %fd58, %fd59;
    mov.f64 %fd59, %fd5;
SINCOSPI_NO_ODD_SWAP:
    and.b32 %r10, %r1, 2;
    setp.eq.s32 %p4, %r10, 0;
    @%p4 bra SINCOSPI_NO_SIGN_FLIP;
    {
    .reg .b32 %temp;
    mov.b64 {%temp, %r11}, %fd58;
    }
    {
    .reg .b32 %temp;
    mov.b64 {%r12, %temp}, %fd58;
    }
    xor.b32 %r13, %r11, -2147483648;
    mov.b64 %fd58, {%r12, %r13};
    {
    .reg .b32 %temp;
    mov.b64 {%temp, %r14}, %fd59;
    }
    {
    .reg .b32 %temp;
    mov.b64 {%r15, %temp}, %fd59;
    }
    xor.b32 %r16, %r14, -2147483648;
    mov.b64 %fd59, {%r15, %r16};
SINCOSPI_NO_SIGN_FLIP:
    mov.f64 %fd53, 0d0000000000000000;
    add.rn.f64 %fd12, %fd59, %fd53;
    cvt.rzi.f64.f64 %fd54, %fd55;
    setp.neu.f64 %p5, %fd55, %fd54;
    @%p5 bra SINCOSPI_DONE;
    mul.rn.f64 %fd58, %fd55, %fd53;
SINCOSPI_DONE:
    mov.f64 ",
    );
    ptx.push_str(sin_out);
    ptx.push_str(", %fd58;\n");
    ptx.push_str("    mov.f64 ");
    ptx.push_str(cos_out);
    ptx.push_str(", %fd12;\n");
}

#[cfg(feature = "cuda")]
fn build_philox_normal_f32_ptx() -> String {
    let mut ptx = String::from(
        r".version 7.0
.target sm_52
.address_size 64

.visible .entry philox_normal_kernel(
    .param .u64 out_ptr,
    .param .u64 n,
    .param .u32 seed_lo,
    .param .u32 seed_hi,
    .param .u32 counter_lo,
    .param .u32 counter_hi,
    .param .u64 stride
) {
    .reg .u32 %ltid, %bid, %bdim, %gid;
    .reg .u32 %slo, %shi, %clo, %chi;
    .reg .u32 %ctr_add_lo, %ctr_add_hi;
    .reg .u32 %c0, %c1, %c2, %c3, %k0, %k1;
    .reg .u32 %hi_val, %lo_val, %t0, %t1, %t2, %t3;
    .reg .u32 %r<5>;
    .reg .u64 %gid64, %tmp64, %ctr_add, %n_reg, %stride_reg, %linear, %step, %store_idx;
    .reg .u64 %prod, %out, %off;
    .reg .f32 %f<42>;
    .reg .f32 %u1, %u2, %theta, %ln_u1, %radius, %sin_val, %cos_val;
    .reg .pred %p<4>;
    .reg .pred %p_done, %p_store;

    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %n_reg, [n];
    ld.param.u32 %slo, [seed_lo];
    ld.param.u32 %shi, [seed_hi];
    ld.param.u32 %clo, [counter_lo];
    ld.param.u32 %chi, [counter_hi];
    ld.param.u64 %stride_reg, [stride];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %ltid, %tid.x;
    mad.lo.u32 %gid, %bid, %bdim, %ltid;
    cvt.u64.u32 %gid64, %gid;
    mov.u64 %linear, %gid64;
    mov.u64 %ctr_add, 0;
    mul.lo.u64 %step, %stride_reg, 4;

LOOP_NORMAL_F32:
    setp.ge.u64 %p_done, %linear, %n_reg;
    @%p_done bra DONE;

    cvt.u32.u64 %ctr_add_lo, %ctr_add;
    shr.u64 %tmp64, %ctr_add, 32;
    cvt.u32.u64 %ctr_add_hi, %tmp64;
    add.cc.u32 %c0, %clo, %ctr_add_lo;
    addc.u32 %c1, %chi, %ctr_add_hi;
    cvt.u32.u64 %c2, %gid64;
    shr.u64 %tmp64, %gid64, 32;
    cvt.u32.u64 %c3, %tmp64;
    mov.u32 %k0, %slo;
    mov.u32 %k1, %shi;
",
    );
    push_philox_rounds_ptx(&mut ptx);
    push_store_normal_f32_pair_ptx(&mut ptx, 0, "%c0", "%c1", 0);
    push_store_normal_f32_pair_ptx(&mut ptx, 1, "%c2", "%c3", 2);
    ptx.push_str(
        r"
    add.u64 %linear, %linear, %step;
    add.u64 %ctr_add, %ctr_add, 1;
    bra LOOP_NORMAL_F32;

DONE:
    ret;
}
",
    );
    ptx
}

#[cfg(feature = "cuda")]
fn push_f64_from_two_u32_ptx(
    ptx: &mut String,
    x: &str,
    y: &str,
    out: &str,
    reverse_one_to_zero: bool,
) {
    ptx.push_str("    cvt.u64.u32 %lo64, ");
    ptx.push_str(x);
    ptx.push_str(";\n");
    ptx.push_str("    cvt.u64.u32 %hi64, ");
    ptx.push_str(y);
    ptx.push_str(";\n");
    ptx.push_str("    shl.b64 %hi64, %hi64, 21;\n");
    ptx.push_str("    xor.b64 %mant, %lo64, %hi64;\n");
    ptx.push_str("    cvt.rn.f64.u64 ");
    ptx.push_str(out);
    ptx.push_str(", %mant;\n");
    ptx.push_str("    fma.rn.f64 ");
    ptx.push_str(out);
    ptx.push_str(", ");
    ptx.push_str(out);
    ptx.push_str(", 0d3CA0000000000000, 0d3C90000000000000;\n");
    if reverse_one_to_zero {
        ptx.push_str("    setp.eq.f64 %p_one, ");
        ptx.push_str(out);
        ptx.push_str(", 0d3FF0000000000000;\n");
        ptx.push_str("    @%p_one mov.f64 ");
        ptx.push_str(out);
        ptx.push_str(", 0d0000000000000000;\n");
    }
}

#[cfg(feature = "cuda")]
fn push_box_muller_f64_uniform_ptx(ptx: &mut String, x: &str, y: &str, out: &str) {
    push_f64_from_two_u32_ptx(ptx, x, y, out, false);
}

#[cfg(feature = "cuda")]
fn build_philox_uniform_f64_ptx() -> String {
    let mut ptx = String::from(
        r".version 7.0
.target sm_52
.address_size 64

.visible .entry philox_uniform_f64_kernel(
    .param .u64 out_ptr,
    .param .u64 n,
    .param .u32 seed_lo,
    .param .u32 seed_hi,
    .param .u32 counter_lo,
    .param .u32 counter_hi,
    .param .u64 stride
) {
    .reg .u32 %ltid, %bid, %bdim, %gid;
    .reg .u32 %slo, %shi, %clo, %chi;
    .reg .u32 %ctr_add_lo, %ctr_add_hi;
    .reg .u32 %c0, %c1, %c2, %c3, %k0, %k1;
    .reg .u32 %hi_val, %lo_val, %t0, %t1, %t2, %t3;
    .reg .u64 %gid64, %tmp64, %ctr_add, %n_reg, %stride_reg, %linear, %step, %store_idx;
    .reg .u64 %prod, %out, %off, %hi64, %lo64, %mant;
    .reg .f64 %fval;
    .reg .pred %p_done, %p_store, %p_one;

    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %n_reg, [n];
    ld.param.u32 %slo, [seed_lo];
    ld.param.u32 %shi, [seed_hi];
    ld.param.u32 %clo, [counter_lo];
    ld.param.u32 %chi, [counter_hi];
    ld.param.u64 %stride_reg, [stride];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %ltid, %tid.x;
    mad.lo.u32 %gid, %bid, %bdim, %ltid;
    cvt.u64.u32 %gid64, %gid;
    mov.u64 %linear, %gid64;
    mov.u64 %ctr_add, 0;
    mul.lo.u64 %step, %stride_reg, 2;

LOOP_UNIFORM_F64:
    setp.ge.u64 %p_done, %linear, %n_reg;
    @%p_done bra DONE;

    cvt.u32.u64 %ctr_add_lo, %ctr_add;
    shr.u64 %tmp64, %ctr_add, 32;
    cvt.u32.u64 %ctr_add_hi, %tmp64;
    add.cc.u32 %c0, %clo, %ctr_add_lo;
    addc.u32 %c1, %chi, %ctr_add_hi;
    cvt.u32.u64 %c2, %gid64;
    shr.u64 %tmp64, %gid64, 32;
    cvt.u32.u64 %c3, %tmp64;
    mov.u32 %k0, %slo;
    mov.u32 %k1, %shi;
",
    );
    push_philox_rounds_ptx(&mut ptx);
    ptx.push_str("    mov.u64 %store_idx, %linear;\n");
    ptx.push_str("    setp.ge.u64 %p_store, %store_idx, %n_reg;\n");
    ptx.push_str("    @%p_store bra SKIP_UNIFORM_F64_LANE_0;\n");
    push_f64_from_two_u32_ptx(&mut ptx, "%c0", "%c1", "%fval", true);
    ptx.push_str(
        r"
    shl.b64 %off, %store_idx, 3;
    add.u64 %off, %out, %off;
    st.global.f64 [%off], %fval;
SKIP_UNIFORM_F64_LANE_0:
    add.u64 %store_idx, %linear, %stride_reg;
    setp.ge.u64 %p_store, %store_idx, %n_reg;
    @%p_store bra SKIP_UNIFORM_F64_LANE_1;
",
    );
    push_f64_from_two_u32_ptx(&mut ptx, "%c2", "%c3", "%fval", true);
    ptx.push_str(
        r"
    shl.b64 %off, %store_idx, 3;
    add.u64 %off, %out, %off;
    st.global.f64 [%off], %fval;
SKIP_UNIFORM_F64_LANE_1:
    add.u64 %linear, %linear, %step;
    add.u64 %ctr_add, %ctr_add, 1;
    bra LOOP_UNIFORM_F64;

DONE:
    ret;
}
",
    );
    ptx
}

#[cfg(feature = "cuda")]
fn build_philox_normal_f64_ptx() -> String {
    let mut ptx = String::from(
        r".version 7.0
.target sm_52
.address_size 64

.visible .entry philox_normal_f64_kernel(
    .param .u64 out_ptr,
    .param .u64 n,
    .param .u32 seed_lo,
    .param .u32 seed_hi,
    .param .u32 counter_lo,
    .param .u32 counter_hi,
    .param .u64 stride
) {
    .reg .u32 %ltid, %bid, %bdim, %gid;
    .reg .u32 %slo, %shi, %clo, %chi;
    .reg .u32 %ctr_add_lo, %ctr_add_hi;
    .reg .u32 %c0, %c1, %c2, %c3, %k0, %k1;
    .reg .u32 %hi_val, %lo_val, %t0, %t1, %t2, %t3;
    .reg .u32 %r<27>;
    .reg .u64 %gid64, %tmp64, %ctr_add, %n_reg, %stride_reg, %linear, %step, %store_idx;
    .reg .u64 %prod, %out, %off, %xbits, %mantissa_bits, %bias_bits, %q;
    .reg .u64 %rd<5>;
    .reg .u64 %hi64, %lo64, %mant;
    .reg .s64 %exp64, %n_i;
    .reg .b64 %n_bits;
    .reg .f32 %f<2>;
    .reg .f64 %fd<61>;
    .reg .f64 %u1, %u2, %u_tmp0, %u_tmp1, %ln_u1, %radius, %theta, %z0, %z1;
    .reg .f64 %m, %log_f, %log_f2, %log_s, %log_poly, %ln2_hi, %ln2_lo;
    .reg .f64 %one, %two, %sqrt2, %half_const, %nf;
    .reg .f64 %theta_y, %theta_nf, %theta_r, %theta_r2, %sin_poly, %cos_poly;
    .reg .f64 %sin_r, %cos_r, %neg_sin, %neg_cos, %sin_theta, %cos_theta;
    .reg .pred %p<6>;
    .reg .pred %p_done, %p_store, %p_shift, %p_q;

    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %n_reg, [n];
    ld.param.u32 %slo, [seed_lo];
    ld.param.u32 %shi, [seed_hi];
    ld.param.u32 %clo, [counter_lo];
    ld.param.u32 %chi, [counter_hi];
    ld.param.u64 %stride_reg, [stride];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %ltid, %tid.x;
    mad.lo.u32 %gid, %bid, %bdim, %ltid;
    cvt.u64.u32 %gid64, %gid;
    mov.u64 %linear, %gid64;
    mov.u64 %ctr_add, 0;
    mul.lo.u64 %step, %stride_reg, 2;

LOOP_NORMAL_F64:
    setp.ge.u64 %p_done, %linear, %n_reg;
    @%p_done bra DONE;

    cvt.u32.u64 %ctr_add_lo, %ctr_add;
    shr.u64 %tmp64, %ctr_add, 32;
    cvt.u32.u64 %ctr_add_hi, %tmp64;
    add.cc.u32 %c0, %clo, %ctr_add_lo;
    addc.u32 %c1, %chi, %ctr_add_hi;
    cvt.u32.u64 %c2, %gid64;
    shr.u64 %tmp64, %gid64, 32;
    cvt.u32.u64 %c3, %tmp64;
    mov.u32 %k0, %slo;
    mov.u32 %k1, %shi;
",
    );
    push_philox_rounds_ptx(&mut ptx);
    push_box_muller_f64_uniform_ptx(&mut ptx, "%c0", "%c1", "%u1");
    push_box_muller_f64_uniform_ptx(&mut ptx, "%c2", "%c3", "%u2");
    push_libdevice_log_f64_ptx(&mut ptx, "%u1", "%ln_u1");
    ptx.push_str(
        r"
    mul.rn.f64 %radius, 0dC000000000000000, %ln_u1;
    sqrt.rn.f64 %radius, %radius;
    add.rn.f64 %theta, %u2, %u2;
",
    );
    push_libdevice_sincospi_f64_ptx(&mut ptx, "%theta", "%sin_theta", "%cos_theta");
    ptx.push_str(
        r"

    mul.rn.f64 %z0, %radius, %sin_theta;
    mul.rn.f64 %z1, %radius, %cos_theta;

    mov.u64 %store_idx, %linear;
    setp.ge.u64 %p_store, %store_idx, %n_reg;
    @%p_store bra SKIP_NORMAL_F64_LANE_0;
    shl.b64 %off, %store_idx, 3;
    add.u64 %off, %out, %off;
    st.global.f64 [%off], %z0;

SKIP_NORMAL_F64_LANE_0:
    add.u64 %store_idx, %linear, %stride_reg;
    setp.ge.u64 %p_store, %store_idx, %n_reg;
    @%p_store bra SKIP_NORMAL_F64_LANE_1;
    shl.b64 %off, %store_idx, 3;
    add.u64 %off, %out, %off;
    st.global.f64 [%off], %z1;

SKIP_NORMAL_F64_LANE_1:
    add.u64 %linear, %linear, %step;
    add.u64 %ctr_add, %ctr_add, 1;
    bra LOOP_NORMAL_F64;

DONE:
    ret;
}
",
    );
    ptx
}

#[cfg(feature = "cuda")]
const RNG_UNIFORM_F32_TO_F16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry rng_uniform_f32_to_f16_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %in, %out, %ioff, %ooff;
    .reg .f32 %v, %max;
    .reg .b16 %r;
    .reg .pred %p, %too_high;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %nr, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr;
    @%p bra DONE;

    cvt.u64.u32 %ioff, %idx;
    shl.b64 %ioff, %ioff, 2;
    add.u64 %in, %in, %ioff;
    ld.global.f32 %v, [%in];

    mov.f32 %max, 0f3F7FE000;
    setp.gt.f32 %too_high, %v, %max;
    @%too_high mov.f32 %v, %max;

    cvt.rn.f16.f32 %r, %v;

    cvt.u64.u32 %ooff, %idx;
    shl.b64 %ooff, %ooff, 1;
    add.u64 %out, %out, %ooff;
    st.global.b16 [%out], %r;

DONE:
    ret;
}
";

#[cfg(feature = "cuda")]
const RNG_UNIFORM_F32_TO_BF16_PTX: &str = "\
.version 7.8
.target sm_80
.address_size 64

.visible .entry rng_uniform_f32_to_bf16_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %in, %out, %ioff, %ooff;
    .reg .f32 %v, %max;
    .reg .b16 %r;
    .reg .pred %p, %too_high;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %nr, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr;
    @%p bra DONE;

    cvt.u64.u32 %ioff, %idx;
    shl.b64 %ioff, %ioff, 2;
    add.u64 %in, %in, %ioff;
    ld.global.f32 %v, [%in];

    mov.f32 %max, 0f3F7F0000;
    setp.gt.f32 %too_high, %v, %max;
    @%too_high mov.f32 %v, %max;

    cvt.rn.bf16.f32 %r, %v;

    cvt.u64.u32 %ooff, %idx;
    shl.b64 %ooff, %ooff, 1;
    add.u64 %out, %out, %ooff;
    st.global.b16 [%out], %r;

DONE:
    ret;
}
";

// ---------------------------------------------------------------------------
// GPU kernel launch functions
// ---------------------------------------------------------------------------

/// Standard 1-D launch config for `n` elements.
#[cfg(feature = "cuda")]
fn rng_launch_cfg(n: usize) -> GpuResult<LaunchConfig> {
    if n > u32::MAX as usize {
        return Err(GpuError::ShapeMismatch {
            op: "rng_kernel_launch",
            expected: vec![u32::MAX as usize],
            got: vec![n],
        });
    }
    const BLOCK: u32 = 256;
    let grid = ((n as u32).saturating_add(BLOCK - 1)) / BLOCK;
    Ok(LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    })
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
struct TorchDistributionPolicy {
    launch: LaunchConfig,
    stride: u64,
    /// Number of Philox 4x32 counter groups reserved per logical CUDA thread.
    ///
    /// PyTorch's `calc_execution_policy` returns `calls_per_thread * 4` because
    /// `curand_init(..., offset, ...)` measures offset in 32-bit curand
    /// elements. `PhiloxGenerator::advance` stores the same position in
    /// 4x32 counter groups, so the internal increment is `calls_per_thread`.
    counter_advance: u64,
}

#[cfg(feature = "cuda")]
fn torch_distribution_policy(
    n: usize,
    unroll_factor: u64,
    device: &GpuDevice,
) -> GpuResult<TorchDistributionPolicy> {
    if n == 0 {
        return Ok(TorchDistributionPolicy {
            launch: LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            stride: 256,
            counter_advance: 0,
        });
    }
    if n > u32::MAX as usize {
        return Err(GpuError::ShapeMismatch {
            op: "rng_kernel_launch",
            expected: vec![u32::MAX as usize],
            got: vec![n],
        });
    }

    const BLOCK: u32 = 256;
    let numel = n as u64;
    let mut grid = numel.div_ceil(BLOCK as u64);

    let max_threads_per_sm = device.context().attribute(
        cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
    )? as u64;
    let sm_count = device.context().attribute(
        cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    )? as u64;
    let blocks_per_sm = (max_threads_per_sm / BLOCK as u64).max(1);
    let grid_cap = (sm_count * blocks_per_sm).max(1);
    grid = grid.min(grid_cap).max(1);

    let stride = grid * BLOCK as u64;
    let calls_per_thread = ((numel - 1) / (stride * unroll_factor)) + 1;

    Ok(TorchDistributionPolicy {
        launch: LaunchConfig {
            grid_dim: (grid as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        },
        stride,
        counter_advance: calls_per_thread,
    })
}

#[cfg(feature = "cuda")]
fn cast_uniform_rng_f32_to_u16(
    input: &CudaBuffer<f32>,
    device: &GpuDevice,
    ptx: &'static str,
    kernel: &'static str,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    use cudarc::driver::PushKernelArg;

    let n = input.len();
    if n == 0 {
        return device.stream().alloc_zeros::<u16>(0).map_err(Into::into);
    }

    let f =
        crate::module_cache::get_or_compile(device.context(), ptx, kernel, device.ordinal() as u32)
            .map_err(|e| GpuError::PtxCompileFailed { kernel, source: e })?;

    let mut out = device.stream().alloc_zeros::<u16>(n)?;
    let n_u32 = n as u32;
    let cfg = rng_launch_cfg(n)?;

    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(input.inner())
            .arg(&mut out)
            .arg(&n_u32)
            .launch(cfg)?;
    }

    Ok(out)
}

#[cfg(feature = "cuda")]
fn gpu_philox_f64(
    n: usize,
    device: &GpuDevice,
    kernel_name: &'static str,
    ptx_src: String,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;

    if n == 0 {
        return alloc_zeros_f64(0, device);
    }
    let policy = torch_distribution_policy(n, 2, device)?;

    let state = {
        let mut mgr = CUDA_RNG_MANAGER
            .lock()
            .map_err(|e| GpuError::InvalidState {
                message: format!("CUDA RNG manager mutex poisoned: {e}"),
            })?;
        let rng_gen = mgr.generator(device.ordinal());
        let state = rng_gen.get_state();
        rng_gen.advance(policy.counter_advance);
        state
    };

    let f = crate::module_cache::get_or_compile_owned(
        device.context(),
        ptx_src,
        kernel_name.to_string(),
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: kernel_name,
        source: e,
    })?;

    let mut out = alloc_zeros_f64(n, device)?;
    let n_u64 = n as u64;
    let seed_lo = state.seed as u32;
    let seed_hi = (state.seed >> 32) as u32;
    let counter_lo = state.counter as u32;
    let counter_hi = (state.counter >> 32) as u32;
    let stride = policy.stride;

    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(out.inner_mut())
            .arg(&n_u64)
            .arg(&seed_lo)
            .arg(&seed_hi)
            .arg(&counter_lo)
            .arg(&counter_hi)
            .arg(&stride)
            .launch(policy.launch)?;
    }

    Ok(out)
}

#[cfg(feature = "cuda")]
fn validate_dropout_keep_probability(keep_probability: f64) -> GpuResult<()> {
    if keep_probability.is_finite() && keep_probability > 0.0 && keep_probability <= 1.0 {
        Ok(())
    } else {
        Err(GpuError::InvalidState {
            message: format!(
                "dropout keep_probability must be finite and in (0, 1], got {keep_probability}"
            ),
        })
    }
}

#[cfg(feature = "cuda")]
fn reserve_dropout_philox_state(
    n: usize,
    device: &GpuDevice,
) -> GpuResult<(PhiloxState, TorchDistributionPolicy)> {
    let policy = torch_distribution_policy(n, 4, device)?;
    let state = {
        let mut mgr = CUDA_RNG_MANAGER
            .lock()
            .map_err(|e| GpuError::InvalidState {
                message: format!("CUDA RNG manager mutex poisoned: {e}"),
            })?;
        let rng_gen = mgr.generator(device.ordinal());
        let state = rng_gen.get_state();
        if state.offset() != 0 {
            return Err(GpuError::InvalidState {
                message: format!(
                    "CUDA dropout requires a 4x32-aligned Philox state, got offset {}",
                    state.offset()
                ),
            });
        }
        rng_gen.advance(policy.counter_advance);
        state
    };
    Ok((state, policy))
}

#[cfg(feature = "cuda")]
fn dropout_kernel_error_name(storage: DropoutStorage) -> &'static str {
    match storage {
        DropoutStorage::F32 => "philox_dropout_f32_kernel",
        DropoutStorage::F64 => "philox_dropout_f64_kernel",
        DropoutStorage::F16 => "philox_dropout_f16_kernel",
        DropoutStorage::BF16 => "philox_dropout_bf16_kernel",
    }
}

#[cfg(feature = "cuda")]
fn dropout_entry_name(storage: DropoutStorage, vector_size: u32) -> String {
    format!("{}_v{vector_size}_kernel", storage.entry_prefix())
}

#[cfg(feature = "cuda")]
fn compile_dropout_kernel(
    storage: DropoutStorage,
    vector_size: u32,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaFunction> {
    let entry = dropout_entry_name(storage, vector_size);
    let ptx = build_philox_dropout_ptx(storage, vector_size, &entry);
    crate::module_cache::get_or_compile_owned(device.context(), ptx, entry, device.ordinal() as u32)
        .map_err(|e| GpuError::PtxCompileFailed {
            kernel: dropout_kernel_error_name(storage),
            source: e,
        })
}

#[cfg(feature = "cuda")]
fn launch_dropout_f32_like(
    input: &CudaBuffer<f32>,
    keep_probability: f64,
    device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, CudaBuffer<f32>, PhiloxState)> {
    use cudarc::driver::PushKernelArg;

    validate_dropout_keep_probability(keep_probability)?;
    let n = input.len();
    if n == 0 {
        let state = {
            let mut mgr = CUDA_RNG_MANAGER
                .lock()
                .map_err(|e| GpuError::InvalidState {
                    message: format!("CUDA RNG manager mutex poisoned: {e}"),
                })?;
            mgr.generator(device.ordinal()).get_state()
        };
        return Ok((
            alloc_zeros_f32(0, device)?,
            alloc_zeros_f32(0, device)?,
            state,
        ));
    }

    let (state, policy) = reserve_dropout_philox_state(n, device)?;
    let vector_size = torch_dropout_vector_size(n, DropoutStorage::F32);
    let f = compile_dropout_kernel(DropoutStorage::F32, vector_size, device)?;

    let mut out = alloc_zeros_f32(n, device)?;
    let mut mask = alloc_zeros_f32(n, device)?;
    let n_u64 = n as u64;
    let keep = keep_probability as f32;
    let scale = (1.0 / keep_probability) as f32;
    let seed_lo = state.seed as u32;
    let seed_hi = (state.seed >> 32) as u32;
    let counter_lo = state.counter as u32;
    let counter_hi = (state.counter >> 32) as u32;
    let stride = policy.stride;

    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(mask.inner_mut())
            .arg(&n_u64)
            .arg(&keep)
            .arg(&scale)
            .arg(&seed_lo)
            .arg(&seed_hi)
            .arg(&counter_lo)
            .arg(&counter_hi)
            .arg(&stride)
            .launch(policy.launch)?;
    }

    Ok((out, mask, state))
}

#[cfg(feature = "cuda")]
fn launch_dropout_f64_like(
    input: &CudaBuffer<f64>,
    keep_probability: f64,
    device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f64>, CudaBuffer<f64>, PhiloxState)> {
    use cudarc::driver::PushKernelArg;

    validate_dropout_keep_probability(keep_probability)?;
    let n = input.len();
    if n == 0 {
        let state = {
            let mut mgr = CUDA_RNG_MANAGER
                .lock()
                .map_err(|e| GpuError::InvalidState {
                    message: format!("CUDA RNG manager mutex poisoned: {e}"),
                })?;
            mgr.generator(device.ordinal()).get_state()
        };
        return Ok((
            alloc_zeros_f64(0, device)?,
            alloc_zeros_f64(0, device)?,
            state,
        ));
    }

    let (state, policy) = reserve_dropout_philox_state(n, device)?;
    let vector_size = torch_dropout_vector_size(n, DropoutStorage::F64);
    let f = compile_dropout_kernel(DropoutStorage::F64, vector_size, device)?;

    let mut out = alloc_zeros_f64(n, device)?;
    let mut mask = alloc_zeros_f64(n, device)?;
    let n_u64 = n as u64;
    let keep = keep_probability;
    let scale = 1.0 / keep_probability;
    let seed_lo = state.seed as u32;
    let seed_hi = (state.seed >> 32) as u32;
    let counter_lo = state.counter as u32;
    let counter_hi = (state.counter >> 32) as u32;
    let stride = policy.stride;

    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(mask.inner_mut())
            .arg(&n_u64)
            .arg(&keep)
            .arg(&scale)
            .arg(&seed_lo)
            .arg(&seed_hi)
            .arg(&counter_lo)
            .arg(&counter_hi)
            .arg(&stride)
            .launch(policy.launch)?;
    }

    Ok((out, mask, state))
}

#[cfg(feature = "cuda")]
fn launch_dropout_u16_like(
    input: &cudarc::driver::CudaSlice<u16>,
    logical_len: usize,
    keep_probability: f64,
    storage: DropoutStorage,
    device: &GpuDevice,
) -> GpuResult<(
    cudarc::driver::CudaSlice<u16>,
    cudarc::driver::CudaSlice<u16>,
    PhiloxState,
)> {
    use cudarc::driver::PushKernelArg;

    validate_dropout_keep_probability(keep_probability)?;
    if logical_len > input.len() {
        return Err(GpuError::ShapeMismatch {
            op: "philox_dropout_u16",
            expected: vec![input.len()],
            got: vec![logical_len],
        });
    }
    if logical_len == 0 {
        let state = {
            let mut mgr = CUDA_RNG_MANAGER
                .lock()
                .map_err(|e| GpuError::InvalidState {
                    message: format!("CUDA RNG manager mutex poisoned: {e}"),
                })?;
            mgr.generator(device.ordinal()).get_state()
        };
        return Ok((
            device.stream().alloc_zeros::<u16>(0)?,
            device.stream().alloc_zeros::<u16>(0)?,
            state,
        ));
    }

    let (state, policy) = reserve_dropout_philox_state(logical_len, device)?;
    let vector_size = torch_dropout_vector_size(logical_len, storage);
    let f = compile_dropout_kernel(storage, vector_size, device)?;

    let mut out = device.stream().alloc_zeros::<u16>(logical_len)?;
    let mut mask = device.stream().alloc_zeros::<u16>(logical_len)?;
    let n_u64 = logical_len as u64;
    let keep = keep_probability as f32;
    let scale = (1.0 / keep_probability) as f32;
    let seed_lo = state.seed as u32;
    let seed_hi = (state.seed >> 32) as u32;
    let counter_lo = state.counter as u32;
    let counter_hi = (state.counter >> 32) as u32;
    let stride = policy.stride;

    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(input)
            .arg(&mut out)
            .arg(&mut mask)
            .arg(&n_u64)
            .arg(&keep)
            .arg(&scale)
            .arg(&seed_lo)
            .arg(&seed_hi)
            .arg(&counter_lo)
            .arg(&counter_hi)
            .arg(&stride)
            .launch(policy.launch)?;
    }

    Ok((out, mask, state))
}

/// Apply PyTorch-layout Philox dropout to a contiguous f32 CUDA buffer.
#[cfg(feature = "cuda")]
pub fn gpu_philox_dropout_f32(
    input: &CudaBuffer<f32>,
    keep_probability: f64,
    device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, CudaBuffer<f32>, PhiloxState)> {
    launch_dropout_f32_like(input, keep_probability, device)
}

/// Apply PyTorch-layout Philox dropout to a contiguous f64 CUDA buffer.
#[cfg(feature = "cuda")]
pub fn gpu_philox_dropout_f64(
    input: &CudaBuffer<f64>,
    keep_probability: f64,
    device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f64>, CudaBuffer<f64>, PhiloxState)> {
    launch_dropout_f64_like(input, keep_probability, device)
}

/// Apply PyTorch-layout Philox dropout to a contiguous IEEE f16 CUDA buffer.
#[cfg(feature = "cuda")]
pub fn gpu_philox_dropout_f16(
    input: &cudarc::driver::CudaSlice<u16>,
    logical_len: usize,
    keep_probability: f64,
    device: &GpuDevice,
) -> GpuResult<(
    cudarc::driver::CudaSlice<u16>,
    cudarc::driver::CudaSlice<u16>,
    PhiloxState,
)> {
    launch_dropout_u16_like(
        input,
        logical_len,
        keep_probability,
        DropoutStorage::F16,
        device,
    )
}

/// Apply PyTorch-layout Philox dropout to a contiguous bf16 CUDA buffer.
#[cfg(feature = "cuda")]
pub fn gpu_philox_dropout_bf16(
    input: &cudarc::driver::CudaSlice<u16>,
    logical_len: usize,
    keep_probability: f64,
    device: &GpuDevice,
) -> GpuResult<(
    cudarc::driver::CudaSlice<u16>,
    cudarc::driver::CudaSlice<u16>,
    PhiloxState,
)> {
    launch_dropout_u16_like(
        input,
        logical_len,
        keep_probability,
        DropoutStorage::BF16,
        device,
    )
}

/// Fill a GPU buffer with uniform random f32 values in [0, 1) using the
/// Philox 4x32-10 algorithm.
///
/// The values are generated entirely on device — no CPU-to-GPU transfer.
/// The per-device RNG state is advanced with PyTorch's CUDA distribution
/// policy: each logical CUDA thread reserves enough Philox 4x32 counter groups
/// for its grid-stride `float4` lane consumption.
///
/// # Arguments
///
/// * `n` — number of f32 values to generate
/// * `device` — the GPU device
///
#[cfg(feature = "cuda")]
pub fn gpu_philox_uniform(n: usize, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    if n == 0 {
        return alloc_zeros_f32(0, device);
    }
    let policy = torch_distribution_policy(n, 4, device)?;

    // Get the current RNG state and advance it.
    let state = {
        let mut mgr = CUDA_RNG_MANAGER
            .lock()
            .map_err(|e| GpuError::InvalidState {
                message: format!("CUDA RNG manager mutex poisoned: {e}"),
            })?;
        let rng_gen = mgr.generator(device.ordinal());
        let state = rng_gen.get_state();
        rng_gen.advance(policy.counter_advance);
        state
    };

    let ctx = device.context();
    let stream = device.stream();

    let f = crate::module_cache::get_or_compile_owned(
        ctx,
        PHILOX_UNIFORM_F32_PTX.as_str().to_owned(),
        "philox_uniform_kernel".to_string(),
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "philox_uniform_kernel",
        source: e,
    })?;

    let mut out = alloc_zeros_f32(n, device)?;
    let n_u64 = n as u64;
    let seed_lo = state.seed as u32;
    let seed_hi = (state.seed >> 32) as u32;
    let counter_lo = state.counter as u32;
    let counter_hi = (state.counter >> 32) as u32;
    let stride = policy.stride;

    unsafe {
        stream
            .launch_builder(&f)
            .arg(out.inner_mut())
            .arg(&n_u64)
            .arg(&seed_lo)
            .arg(&seed_hi)
            .arg(&counter_lo)
            .arg(&counter_hi)
            .arg(&stride)
            .launch(policy.launch)?;
    }

    Ok(out)
}

/// Fill a GPU buffer with standard normal f32 values using the Philox 4x32-10
/// algorithm and Box-Muller transform.
///
/// # Arguments
///
/// * `n` — number of f32 values to generate
/// * `device` — the GPU device
#[cfg(feature = "cuda")]
pub fn gpu_philox_normal(n: usize, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    if n == 0 {
        return alloc_zeros_f32(0, device);
    }
    let policy = torch_distribution_policy(n, 4, device)?;

    // Get the current RNG state and advance it.
    let state = {
        let mut mgr = CUDA_RNG_MANAGER
            .lock()
            .map_err(|e| GpuError::InvalidState {
                message: format!("CUDA RNG manager mutex poisoned: {e}"),
            })?;
        let rng_gen = mgr.generator(device.ordinal());
        let state = rng_gen.get_state();
        rng_gen.advance(policy.counter_advance);
        state
    };

    let ctx = device.context();
    let stream = device.stream();

    let f = crate::module_cache::get_or_compile_owned(
        ctx,
        PHILOX_NORMAL_F32_PTX.as_str().to_owned(),
        "philox_normal_kernel".to_string(),
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "philox_normal_kernel",
        source: e,
    })?;

    let mut out = alloc_zeros_f32(n, device)?;
    let n_u64 = n as u64;
    let seed_lo = state.seed as u32;
    let seed_hi = (state.seed >> 32) as u32;
    let counter_lo = state.counter as u32;
    let counter_hi = (state.counter >> 32) as u32;
    let stride = policy.stride;

    unsafe {
        stream
            .launch_builder(&f)
            .arg(out.inner_mut())
            .arg(&n_u64)
            .arg(&seed_lo)
            .arg(&seed_hi)
            .arg(&counter_lo)
            .arg(&counter_hi)
            .arg(&stride)
            .launch(policy.launch)?;
    }

    Ok(out)
}

/// Fill a GPU buffer with uniform f64 values without host staging.
#[cfg(feature = "cuda")]
pub fn gpu_philox_uniform_f64(n: usize, device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    gpu_philox_f64(
        n,
        device,
        "philox_uniform_f64_kernel",
        PHILOX_UNIFORM_F64_PTX.clone(),
    )
}

/// Fill a GPU buffer with standard-normal f64 values without host staging.
#[cfg(feature = "cuda")]
pub fn gpu_philox_normal_f64(n: usize, device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    gpu_philox_f64(
        n,
        device,
        "philox_normal_f64_kernel",
        PHILOX_NORMAL_F64_PTX.clone(),
    )
}

/// Fill a GPU buffer with uniform f16 values without host staging.
#[cfg(feature = "cuda")]
pub fn gpu_philox_uniform_f16(
    n: usize,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    let f32_vals = gpu_philox_uniform(n, device)?;
    cast_uniform_rng_f32_to_u16(
        &f32_vals,
        device,
        RNG_UNIFORM_F32_TO_F16_PTX,
        "rng_uniform_f32_to_f16_kernel",
    )
}

/// Fill a GPU buffer with standard-normal f16 values without host staging.
#[cfg(feature = "cuda")]
pub fn gpu_philox_normal_f16(
    n: usize,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    let f32_vals = gpu_philox_normal(n, device)?;
    crate::kernels::gpu_f32_to_f16(&f32_vals, device)
}

/// Fill a GPU buffer with uniform bf16 values without host staging.
#[cfg(feature = "cuda")]
pub fn gpu_philox_uniform_bf16(
    n: usize,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    let f32_vals = gpu_philox_uniform(n, device)?;
    cast_uniform_rng_f32_to_u16(
        &f32_vals,
        device,
        RNG_UNIFORM_F32_TO_BF16_PTX,
        "rng_uniform_f32_to_bf16_kernel",
    )
}

/// Fill a GPU buffer with standard-normal bf16 values without host staging.
#[cfg(feature = "cuda")]
pub fn gpu_philox_normal_bf16(
    n: usize,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    let f32_vals = gpu_philox_normal(n, device)?;
    crate::kernels::gpu_f32_to_bf16(&f32_vals, device)
}

// ---------------------------------------------------------------------------
// Stubs when `cuda` feature is disabled
// ---------------------------------------------------------------------------

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_philox_uniform(_n: usize, _device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_philox_normal(_n: usize, _device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_philox_uniform_f64(_n: usize, _device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_philox_normal_f64(_n: usize, _device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Philox core algorithm tests
    // -----------------------------------------------------------------------

    #[test]
    fn philox_deterministic() {
        // Same counter + key must produce the same output.
        let a = philox_4x32_10(0, 0);
        let b = philox_4x32_10(0, 0);
        assert_eq!(a, b);
    }

    #[test]
    fn philox_different_counters_differ() {
        let a = philox_4x32_10(0, 42);
        let b = philox_4x32_10(1, 42);
        assert_ne!(a, b);
    }

    #[test]
    fn philox_different_keys_differ() {
        let a = philox_4x32_10(0, 0);
        let b = philox_4x32_10(0, 1);
        assert_ne!(a, b);
    }

    #[test]
    fn philox_outputs_nonzero() {
        // With high probability, at least some outputs are non-zero.
        let out = philox_4x32_10(1, 1);
        assert!(
            out.iter().any(|&x| x != 0),
            "all Philox outputs are zero — very unlikely"
        );
    }

    #[test]
    fn philox_avalanche_effect() {
        // Changing a single bit in the counter should change many output bits.
        let a = philox_4x32_10(0, 42);
        let b = philox_4x32_10(1, 42); // counter differs by 1
        let mut total_differing_bits = 0u32;
        for (&x, &y) in a.iter().zip(b.iter()) {
            total_differing_bits += (x ^ y).count_ones();
        }
        // With 128 output bits, roughly half should differ (64 +/- some).
        // We accept anything in [20, 108] as a sanity check.
        assert!(
            total_differing_bits > 20 && total_differing_bits < 108,
            "poor avalanche: {total_differing_bits} bits differ out of 128"
        );
    }

    // -----------------------------------------------------------------------
    // PhiloxGenerator tests
    // -----------------------------------------------------------------------

    #[test]
    fn generator_deterministic_with_same_seed() {
        let mut g1 = PhiloxGenerator::new(42);
        let mut g2 = PhiloxGenerator::new(42);

        let vals1: Vec<u32> = (0..100).map(|_| g1.next_u32()).collect();
        let vals2: Vec<u32> = (0..100).map(|_| g2.next_u32()).collect();
        assert_eq!(vals1, vals2);
    }

    #[test]
    fn generator_different_seeds_differ() {
        let mut g1 = PhiloxGenerator::new(42);
        let mut g2 = PhiloxGenerator::new(43);

        let vals1: Vec<u32> = (0..10).map(|_| g1.next_u32()).collect();
        let vals2: Vec<u32> = (0..10).map(|_| g2.next_u32()).collect();
        assert_ne!(vals1, vals2);
    }

    #[test]
    fn generator_produces_4_values_per_counter() {
        let mut rng_gen = PhiloxGenerator::new(12345);

        // First 4 values should come from counter 0
        let _ = rng_gen.next_u32();
        assert_eq!(rng_gen.counter, 0);
        assert_eq!(rng_gen.offset, 1);

        let _ = rng_gen.next_u32();
        let _ = rng_gen.next_u32();
        let _ = rng_gen.next_u32();
        // After 4 values, counter should advance to 1
        assert_eq!(rng_gen.counter, 1);
        assert_eq!(rng_gen.offset, 0);
    }

    #[test]
    fn generator_set_seed_resets_state() {
        let mut rng_gen = PhiloxGenerator::new(42);
        let first_val = rng_gen.next_u32();

        // Advance a bunch
        for _ in 0..100 {
            rng_gen.next_u32();
        }

        // Reset
        rng_gen.set_seed(42);
        let after_reset = rng_gen.next_u32();
        assert_eq!(first_val, after_reset);
    }

    #[test]
    fn generator_state_save_restore() {
        let mut rng_gen = PhiloxGenerator::new(42);

        // Advance partway
        for _ in 0..7 {
            rng_gen.next_u32();
        }

        let state = rng_gen.get_state();

        // Generate 20 more values
        let vals1: Vec<u32> = (0..20).map(|_| rng_gen.next_u32()).collect();

        // Restore and generate the same 20
        rng_gen.set_state(state);
        let vals2: Vec<u32> = (0..20).map(|_| rng_gen.next_u32()).collect();

        assert_eq!(vals1, vals2);
    }

    #[test]
    fn generator_state_save_restore_at_offset() {
        // Save state when offset is non-zero (mid-tuple)
        let mut rng_gen = PhiloxGenerator::new(99);

        // Consume 2 of 4 values from counter 0
        rng_gen.next_u32();
        rng_gen.next_u32();
        assert_eq!(rng_gen.offset, 2);

        let state = rng_gen.get_state();

        let vals1: Vec<u32> = (0..10).map(|_| rng_gen.next_u32()).collect();

        rng_gen.set_state(state);
        let vals2: Vec<u32> = (0..10).map(|_| rng_gen.next_u32()).collect();

        assert_eq!(vals1, vals2);
    }

    // -----------------------------------------------------------------------
    // next_f32 tests
    // -----------------------------------------------------------------------

    #[test]
    fn f32_in_unit_interval() {
        let mut rng_gen = PhiloxGenerator::new(42);
        for _ in 0..10000 {
            let v = rng_gen.next_f32();
            assert!((0.0..1.0).contains(&v), "f32 value {v} outside [0, 1)");
        }
    }

    #[test]
    fn f32_not_all_same() {
        let mut rng_gen = PhiloxGenerator::new(42);
        let vals: Vec<f32> = (0..100).map(|_| rng_gen.next_f32()).collect();
        let first = vals[0];
        assert!(
            vals.iter().any(|&v| v != first),
            "all f32 values are the same: {first}"
        );
    }

    // -----------------------------------------------------------------------
    // generate_uniform tests
    // -----------------------------------------------------------------------

    #[test]
    fn generate_uniform_correct_length() {
        let mut rng_gen = PhiloxGenerator::new(42);
        let vals = rng_gen.generate_uniform(1000);
        assert_eq!(vals.len(), 1000);
    }

    #[test]
    fn generate_uniform_values_in_range() {
        let mut rng_gen = PhiloxGenerator::new(42);
        let vals = rng_gen.generate_uniform(10000);
        for &v in &vals {
            assert!((0.0..1.0).contains(&v), "uniform value {v} outside [0, 1)");
        }
    }

    #[test]
    fn generate_uniform_reasonable_mean() {
        let mut rng_gen = PhiloxGenerator::new(42);
        let vals = rng_gen.generate_uniform(100_000);
        let mean: f64 = vals.iter().map(|&v| v as f64).sum::<f64>() / vals.len() as f64;
        assert!(
            (mean - 0.5).abs() < 0.01,
            "uniform mean = {mean}, expected ~0.5"
        );
    }

    // -----------------------------------------------------------------------
    // generate_normal tests
    // -----------------------------------------------------------------------

    #[test]
    fn generate_normal_correct_length() {
        let mut rng_gen = PhiloxGenerator::new(42);
        assert_eq!(rng_gen.generate_normal(1000).len(), 1000);
        // Odd count
        assert_eq!(rng_gen.generate_normal(999).len(), 999);
    }

    #[test]
    fn generate_normal_reasonable_mean_and_std() {
        let mut rng_gen = PhiloxGenerator::new(42);
        let vals = rng_gen.generate_normal(100_000);

        let n = vals.len() as f64;
        let mean: f64 = vals.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var: f64 = vals
            .iter()
            .map(|&v| {
                let d = v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        let std = var.sqrt();

        assert!(mean.abs() < 0.02, "normal mean = {mean}, expected ~0.0");
        assert!(
            (std - 1.0).abs() < 0.02,
            "normal std = {std}, expected ~1.0"
        );
    }

    #[test]
    fn generate_normal_no_nan_or_inf() {
        let mut rng_gen = PhiloxGenerator::new(42);
        let vals = rng_gen.generate_normal(100_000);
        for &v in &vals {
            assert!(v.is_finite(), "normal value is not finite: {v}");
        }
    }

    // -----------------------------------------------------------------------
    // CudaRngManager tests
    // -----------------------------------------------------------------------

    #[test]
    fn manager_initializes_device_on_access() {
        let mut mgr = CudaRngManager::new(42);
        let state = mgr.get_rng_state(0);
        assert_eq!(state.seed, 42);
        assert_eq!(state.counter, 0);
        assert_eq!(state.offset, 0);
    }

    #[test]
    fn manager_manual_seed() {
        let mut mgr = CudaRngManager::new(0);
        mgr.manual_seed(0, 12345);

        let rng_gen = mgr.generator(0);
        assert_eq!(rng_gen.seed, 12345);
        assert_eq!(rng_gen.counter, 0);
    }

    #[test]
    fn manager_manual_seed_all() {
        let mut mgr = CudaRngManager::new(0);
        // Initialize a few devices
        mgr.manual_seed(0, 100);
        mgr.manual_seed(1, 200);
        mgr.manual_seed(2, 300);

        // Now set all to the same seed
        mgr.manual_seed_all(42);

        assert_eq!(mgr.get_rng_state(0).seed, 42);
        assert_eq!(mgr.get_rng_state(1).seed, 42);
        assert_eq!(mgr.get_rng_state(2).seed, 42);

        // Newly-created device should also get the new default
        assert_eq!(mgr.get_rng_state(3).seed, 42);
    }

    #[test]
    fn manager_set_rng_state() {
        let mut mgr = CudaRngManager::new(0);
        let custom_state = PhiloxState::from_parts(100, 999, 2).expect("offset 2 is in 0..4");
        mgr.set_rng_state(0, custom_state);

        let state = mgr.get_rng_state(0);
        assert_eq!(state, custom_state);
    }

    #[test]
    fn manager_independent_devices() {
        let mut mgr = CudaRngManager::new(0);
        mgr.manual_seed(0, 42);
        mgr.manual_seed(1, 43);

        let v0 = mgr.generator(0).next_u32();
        let v1 = mgr.generator(1).next_u32();
        // Different seeds should produce different values
        assert_ne!(v0, v1);
    }

    // -----------------------------------------------------------------------
    // Fork/join tests
    // -----------------------------------------------------------------------

    #[test]
    fn fork_join_roundtrip() {
        // Set up known state via the global manager
        {
            let mut mgr = CUDA_RNG_MANAGER.lock().unwrap();
            mgr.manual_seed(10, 42);
            mgr.manual_seed(11, 43);
        }

        let devices = &[10, 11];
        let states = fork_rng(devices).expect("fork_rng must succeed in test");

        // Advance the generators
        {
            let mut mgr = CUDA_RNG_MANAGER.lock().unwrap();
            for _ in 0..100 {
                mgr.generator(10).next_u32();
                mgr.generator(11).next_u32();
            }
        }

        // Restore
        join_rng(devices, states).expect("join_rng must succeed in test");

        // Verify restoration
        {
            let mut mgr = CUDA_RNG_MANAGER.lock().unwrap();
            assert_eq!(mgr.get_rng_state(10).counter, 0);
            assert_eq!(mgr.get_rng_state(10).seed, 42);
            assert_eq!(mgr.get_rng_state(11).counter, 0);
            assert_eq!(mgr.get_rng_state(11).seed, 43);
        }
    }

    #[test]
    fn fork_join_length_mismatch_returns_shape_mismatch() {
        let states = vec![PhiloxState::new(0, 0)];
        let result = join_rng(&[0, 1], states);
        assert!(matches!(
            result,
            Err(GpuError::ShapeMismatch { op: "join_rng", .. })
        ));
    }

    // -----------------------------------------------------------------------
    // Global singleton test
    // -----------------------------------------------------------------------

    #[test]
    fn global_singleton_accessible() {
        let mgr = cuda_rng_manager();
        let mut guard = mgr.lock().unwrap();
        guard.manual_seed(99, 12345);
        let state = guard.get_rng_state(99);
        assert_eq!(state.seed, 12345);
    }
}
