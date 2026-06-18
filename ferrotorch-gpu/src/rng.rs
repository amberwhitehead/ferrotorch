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
//! | REQ-5 (`gpu_philox_uniform`) | SHIPPED | `build_philox_uniform_f32_ptx` / `build_philox_uniform_f64_ptx` emit resident PyTorch-layout Philox kernels; consumer dropout-philox path in `CudaBackendImpl::dropout_philox_f32 in backend_impl.rs` derives a seed from the manager and launches the dropout kernel |
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
    ptx.push_str("    lg2.approx.f32 %ln_u1, %u1;\n");
    ptx.push_str("    mul.f32 %ln_u1, %ln_u1, 0f3F317218;\n");
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
    .reg .u64 %gid64, %tmp64, %ctr_add, %n_reg, %stride_reg, %linear, %step, %store_idx;
    .reg .u64 %prod, %out, %off;
    .reg .f32 %u1, %u2, %theta, %ln_u1, %radius, %sin_val, %cos_val;
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
    .reg .u64 %gid64, %tmp64, %ctr_add, %n_reg, %stride_reg, %linear, %step, %store_idx;
    .reg .u64 %prod, %out, %off, %xbits, %mantissa_bits, %bias_bits, %q;
    .reg .u64 %hi64, %lo64, %mant;
    .reg .s64 %exp64, %n_i;
    .reg .b64 %n_bits;
    .reg .f64 %u1, %u2, %u_tmp0, %u_tmp1, %ln_u1, %radius, %theta, %z0, %z1;
    .reg .f64 %m, %log_f, %log_f2, %log_s, %log_poly, %ln2_hi, %ln2_lo;
    .reg .f64 %one, %two, %sqrt2, %half_const, %nf;
    .reg .f64 %theta_y, %theta_nf, %theta_r, %theta_r2, %sin_poly, %cos_poly;
    .reg .f64 %sin_r, %cos_r, %neg_sin, %neg_cos, %sin_theta, %cos_theta;
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
    ptx.push_str(
        r"
    // ln(u1), matching the f64 log kernel's half-step argument reduction.
    mov.f64 %ln2_hi, 0d3FE62E42FEFA3800;
    mov.f64 %ln2_lo, 0d3D2EF35793C76730;
    mov.f64 %one, 0d3FF0000000000000;
    mov.f64 %two, 0d4000000000000000;
    mov.b64 %xbits, %u1;
    shr.u64 %exp64, %xbits, 52;
    and.b64 %exp64, %exp64, 2047;
    sub.s64 %exp64, %exp64, 1023;
    cvt.rn.f64.s64 %nf, %exp64;

    mov.u64 %bias_bits, 0x3FF0000000000000;
    and.b64 %mantissa_bits, %xbits, 0x000FFFFFFFFFFFFF;
    or.b64 %mantissa_bits, %mantissa_bits, %bias_bits;
    mov.b64 %m, %mantissa_bits;

    mov.f64 %sqrt2, 0d3FF6A09E667F3BCD;
    mov.f64 %half_const, 0d3FE0000000000000;
    setp.gt.f64 %p_shift, %m, %sqrt2;
    @%p_shift mul.f64 %m, %m, %half_const;
    @%p_shift add.f64 %nf, %nf, %one;

    sub.f64 %log_f, %m, %one;
    add.f64 %log_s, %m, %one;
    div.rn.f64 %log_f, %log_f, %log_s;
    mul.f64 %log_f2, %log_f, %log_f;
    mov.f64 %log_poly, 0d3FB1111111111111;
    fma.rn.f64 %log_poly, %log_poly, %log_f2, 0d3FB3B13B13B13B14;
    fma.rn.f64 %log_poly, %log_poly, %log_f2, 0d3FB745D1745D1746;
    fma.rn.f64 %log_poly, %log_poly, %log_f2, 0d3FBC71C71C71C71C;
    fma.rn.f64 %log_poly, %log_poly, %log_f2, 0d3FC2492492492492;
    fma.rn.f64 %log_poly, %log_poly, %log_f2, 0d3FC999999999999A;
    fma.rn.f64 %log_poly, %log_poly, %log_f2, 0d3FD5555555555555;
    fma.rn.f64 %log_poly, %log_poly, %log_f2, %one;
    mul.f64 %log_poly, %log_poly, %log_f;
    add.f64 %log_poly, %log_poly, %log_poly;
    fma.rn.f64 %ln_u1, %nf, %ln2_hi, %log_poly;
    fma.rn.f64 %ln_u1, %nf, %ln2_lo, %ln_u1;

    mul.rn.f64 %radius, 0dC000000000000000, %ln_u1;
    sqrt.rn.f64 %radius, %radius;

    mul.rn.f64 %theta, 0d401921FB54442D18, %u2;

    // sincos(theta) with the same f64 polynomial/reduction path used by
    // the standalone CUDA sin/cos kernels.
    mul.rn.f64 %theta_y, %theta, 0d3FE45F306DC9C883;
    cvt.rni.s64.f64 %n_i, %theta_y;
    cvt.rn.f64.s64 %theta_nf, %n_i;
    fma.rn.f64 %theta_r, %theta_nf, 0dBFF921FB54400000, %theta;
    fma.rn.f64 %theta_r, %theta_nf, 0dBDD0B46000000000, %theta_r;
    mul.rn.f64 %theta_r2, %theta_r, %theta_r;

    mov.f64 %sin_poly, 0d3DE6124613A86D09;
    fma.rn.f64 %sin_poly, %sin_poly, %theta_r2, 0dBE5AE64567F544E4;
    fma.rn.f64 %sin_poly, %sin_poly, %theta_r2, 0d3EC71DE3A556C734;
    fma.rn.f64 %sin_poly, %sin_poly, %theta_r2, 0dBF2A01A01A01A01A;
    fma.rn.f64 %sin_poly, %sin_poly, %theta_r2, 0d3F81111111111111;
    fma.rn.f64 %sin_poly, %sin_poly, %theta_r2, 0dBFC5555555555555;
    mul.rn.f64 %sin_poly, %sin_poly, %theta_r2;
    fma.rn.f64 %sin_r, %sin_poly, %theta_r, %theta_r;

    mov.f64 %cos_poly, 0d3E21EED8EFF8D898;
    fma.rn.f64 %cos_poly, %cos_poly, %theta_r2, 0dBE927E4FB7789F5C;
    fma.rn.f64 %cos_poly, %cos_poly, %theta_r2, 0d3EFA01A01A01A01A;
    fma.rn.f64 %cos_poly, %cos_poly, %theta_r2, 0dBF56C16C16C16C17;
    fma.rn.f64 %cos_poly, %cos_poly, %theta_r2, 0d3FA5555555555555;
    fma.rn.f64 %cos_poly, %cos_poly, %theta_r2, 0dBFE0000000000000;
    fma.rn.f64 %cos_r, %cos_poly, %theta_r2, %one;

    neg.f64 %neg_sin, %sin_r;
    neg.f64 %neg_cos, %cos_r;
    mov.f64 %sin_theta, %sin_r;
    mov.f64 %cos_theta, %cos_r;
    mov.b64 %n_bits, %n_i;
    and.b64 %q, %n_bits, 3;
    setp.eq.u64 %p_q, %q, 1;
    @%p_q mov.f64 %sin_theta, %cos_r;
    @%p_q mov.f64 %cos_theta, %neg_sin;
    setp.eq.u64 %p_q, %q, 2;
    @%p_q mov.f64 %sin_theta, %neg_sin;
    @%p_q mov.f64 %cos_theta, %neg_cos;
    setp.eq.u64 %p_q, %q, 3;
    @%p_q mov.f64 %sin_theta, %neg_cos;
    @%p_q mov.f64 %cos_theta, %sin_r;

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
