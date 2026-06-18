//! Process-global default random number generator state, mirroring
//! `torch.manual_seed` / `torch.default_generator`.
//!
//! ## REQ status (per `.design/ferrotorch-core/rng.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (MT19937 engine) | SHIPPED | `Mt19937` engine mirrors `aten/src/ATen/core/MT19937RNGEngine.h:110-150` (state array of 624 uint32, twist/temper bits identical); non-test consumer: `creation::rand`/`creation::randn` source bits from `with_thread_rng`. |
//! | REQ-2 (`Generator` newtype) | SHIPPED | `pub struct Generator` exposes `new(seed)`, `manual_seed(seed)`, dtype-specific uniform helpers, scalar `next_normal_f32/f64`, and PyTorch tensor `randn` fill helpers; non-test consumer: process-default `Generator` used by `creation::rand` / `creation::randn`. |
//! | REQ-3 (`manual_seed` top-level) | SHIPPED | `pub fn manual_seed(seed) -> FerrotorchResult<()>` seeds registered GPU generators before reseeding the process-global CPU `Generator`, mirroring `torch.manual_seed` at `torch/random.py:49-86`; real backend failures are surfaced. Non-test consumer: re-exported at `lib.rs` as `ferrotorch_core::manual_seed`. |
//! | REQ-4 (default-generator state) | SHIPPED | `DEFAULT_RNG: Mutex<Generator>` is initialised once from entropy and serialized like PyTorch's `default_generator` mutex; `manual_seed` reaches all threads and nested default-RNG access returns `InvalidArgument` instead of panicking or deadlocking. Non-test consumer: `with_thread_rng` invoked by `creation::rand`/`randn`. |
//! | REQ-5 (byte-exact parity for f32 rand) | SHIPPED | `Mt19937` reproduces `torch.manual_seed(42); torch.rand(10)` byte-for-byte; pinned by `ferrotorch-core/tests/divergence_manual_seed_parity.rs`. |

use crate::error::{FerrotorchError, FerrotorchResult};
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
/// Holds an MT19937 engine plus the Box-Muller cache slots used by PyTorch's
/// scalar normal distributions (`aten/src/ATen/core/DistributionsHelper.h`).
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

    /// Uniform-`[0,1)` f16, byte-identical to
    /// `at::uniform_real_distribution<Half>(0,1)(gen)`: PyTorch masks to
    /// `numeric_limits<Half>::digits == 11` random bits, scales in f32
    /// distribution accumulator precision, then casts to Half.
    pub fn next_uniform_f16(&mut self) -> half::f16 {
        const MASK: u32 = (1u32 << 11) - 1;
        const DIVISOR: f32 = 1.0f32 / ((1u32 << 11) as f32);
        let v = self.engine.random_u32() & MASK;
        half::f16::from_f32((v as f32) * DIVISOR)
    }

    /// Uniform-`[0,1)` bf16, byte-identical to
    /// `at::uniform_real_distribution<BFloat16>(0,1)(gen)`: PyTorch masks to
    /// `numeric_limits<BFloat16>::digits == 8` random bits, scales in f32
    /// distribution accumulator precision, then casts to BFloat16.
    pub fn next_uniform_bf16(&mut self) -> half::bf16 {
        const MASK: u32 = (1u32 << 8) - 1;
        const DIVISOR: f32 = 1.0f32 / ((1u32 << 8) as f32);
        let v = self.engine.random_u32() & MASK;
        half::bf16::from_f32((v as f32) * DIVISOR)
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

    /// Values for `torch.randn(..., dtype=float32, device=cpu)`.
    ///
    /// PyTorch's CPU `normal_kernel` has two distinct paths:
    ///
    /// - `numel < 16`: scalar `normal_distribution<double>` and cast to f32.
    /// - `numel >= 16`: `normal_fill<float>`, which fills uniform f32 values
    ///   into the output buffer, applies 16-wide Box-Muller blocks, and
    ///   recomputes the final 16 values for non-multiple-of-16 tails.
    ///
    /// The large path intentionally does not read or clear cached scalar
    /// normal samples, matching `DistributionTemplates.h:168-235`.
    pub(crate) fn torch_randn_f32_values(&mut self, numel: usize) -> Vec<f32> {
        if numel < 16 {
            return (0..numel).map(|_| self.next_normal_f64() as f32).collect();
        }

        let mut data: Vec<f32> = (0..numel).map(|_| self.next_uniform_f32()).collect();
        for start in (0..=(numel - 16)).step_by(16) {
            normal_fill16_f32(&mut data[start..start + 16]);
        }
        if !numel.is_multiple_of(16) {
            let start = numel - 16;
            for slot in &mut data[start..start + 16] {
                *slot = self.next_uniform_f32();
            }
            normal_fill16_f32(&mut data[start..start + 16]);
        }
        data
    }

    /// Values for `torch.randn(..., dtype=float64, device=cpu)`.
    ///
    /// Mirrors the same scalar-vs-`normal_fill<double>` split described in
    /// [`Self::torch_randn_f32_values`], but using f64 uniforms and scalar
    /// libm for the 16-wide transform.
    pub(crate) fn torch_randn_f64_values(&mut self, numel: usize) -> Vec<f64> {
        if numel < 16 {
            return (0..numel).map(|_| self.next_normal_f64()).collect();
        }

        let mut data: Vec<f64> = (0..numel).map(|_| self.next_uniform_f64()).collect();
        for start in (0..=(numel - 16)).step_by(16) {
            normal_fill16_f64(&mut data[start..start + 16]);
        }
        if !numel.is_multiple_of(16) {
            let start = numel - 16;
            for slot in &mut data[start..start + 16] {
                *slot = self.next_uniform_f64();
            }
            normal_fill16_f64(&mut data[start..start + 16]);
        }
        data
    }

    /// Values for `torch.randn(..., dtype=float16, device=cpu)`.
    ///
    /// PyTorch 2.11 routes non-float CPU dtypes through scalar
    /// `normal_fill<scalar_t>` for contiguous tensors with at least 16 elements:
    /// uniforms are first rounded to f16 and every named intermediate in the
    /// 16-wide Box-Muller transform is assigned back to f16 before the final
    /// store. Small tensors still use scalar `normal_distribution<double>` and
    /// cast through f32 into f16.
    pub(crate) fn torch_randn_f16_values(&mut self, numel: usize) -> Vec<half::f16> {
        if numel < 16 {
            return (0..numel)
                .map(|_| half::f16::from_f32(self.next_normal_f64() as f32))
                .collect();
        }

        let mut data: Vec<half::f16> = (0..numel).map(|_| self.next_uniform_f16()).collect();
        for start in (0..=(numel - 16)).step_by(16) {
            normal_fill16_f16(&mut data[start..start + 16]);
        }
        if !numel.is_multiple_of(16) {
            let start = numel - 16;
            for slot in &mut data[start..start + 16] {
                *slot = self.next_uniform_f16();
            }
            normal_fill16_f16(&mut data[start..start + 16]);
        }
        data
    }

    /// Values for `torch.randn(..., dtype=bfloat16, device=cpu)`.
    ///
    /// Mirrors [`Self::torch_randn_f16_values`] with bfloat16 uniform and
    /// intermediate rounding (`numeric_limits<BFloat16>::digits == 8`).
    pub(crate) fn torch_randn_bf16_values(&mut self, numel: usize) -> Vec<half::bf16> {
        if numel < 16 {
            return (0..numel)
                .map(|_| half::bf16::from_f32(self.next_normal_f64() as f32))
                .collect();
        }

        let mut data: Vec<half::bf16> = (0..numel).map(|_| self.next_uniform_bf16()).collect();
        for start in (0..=(numel - 16)).step_by(16) {
            normal_fill16_bf16(&mut data[start..start + 16]);
        }
        if !numel.is_multiple_of(16) {
            let start = numel - 16;
            for slot in &mut data[start..start + 16] {
                *slot = self.next_uniform_bf16();
            }
            normal_fill16_bf16(&mut data[start..start + 16]);
        }
        data
    }
}

fn normal_fill16_f64(data: &mut [f64]) {
    debug_assert_eq!(data.len(), 16);
    for j in 0..8 {
        let u1 = 1.0 - data[j];
        let u2 = data[j + 8];
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        data[j] = radius * theta.cos();
        data[j + 8] = radius * theta.sin();
    }
}

fn normal_fill16_f32(data: &mut [f32]) {
    debug_assert_eq!(data.len(), 16);
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: runtime feature detection proves AVX2/FMA support, and
            // the debug assertion above pins the 16-f32 block size required by
            // the unaligned loads/stores below.
            unsafe {
                normal_fill16_f32_avx2(data);
            }
            return;
        }
    }

    for j in 0..8 {
        let u1 = 1.0 - data[j];
        let u2 = data[j + 8];
        let radius = (-2.0 * avx_mathfun_log_f32(u1)).sqrt();
        let theta = (2.0 * std::f64::consts::PI) as f32 * u2;
        let (sin_t, cos_t) = avx_mathfun_sincos_f32(theta);
        data[j] = radius * cos_t;
        data[j + 8] = radius * sin_t;
    }
}

fn normal_fill16_f16(data: &mut [half::f16]) {
    debug_assert_eq!(data.len(), 16);
    let mean = half::f16::from_f32(0.0);
    let std = half::f16::from_f32(1.0);
    let one = half::f16::from_f32(1.0);
    let minus_two = half::f16::from_f32(-2.0);
    for j in 0..8 {
        let u1 = half::f16::from_f32(one.to_f32() - data[j].to_f32());
        let u2 = data[j + 8];
        let log_u1 = half::f16::from_f32(u1.to_f32().ln());
        let scaled_log = half::f16::from_f32(minus_two.to_f32() * log_u1.to_f32());
        let radius = half::f16::from_f32(scaled_log.to_f32().sqrt());
        let theta =
            half::f16::from_f32((2.0_f64 * std::f64::consts::PI * (u2.to_f32() as f64)) as f32);
        let (sin_theta, cos_theta) = theta.to_f32().sin_cos();
        let sin_theta = half::f16::from_f32(sin_theta);
        let cos_theta = half::f16::from_f32(cos_theta);

        let out1 = half::f16::from_f32(radius.to_f32() * cos_theta.to_f32());
        let out1 = half::f16::from_f32(out1.to_f32() * std.to_f32());
        data[j] = half::f16::from_f32(out1.to_f32() + mean.to_f32());

        let out2 = half::f16::from_f32(radius.to_f32() * sin_theta.to_f32());
        let out2 = half::f16::from_f32(out2.to_f32() * std.to_f32());
        data[j + 8] = half::f16::from_f32(out2.to_f32() + mean.to_f32());
    }
}

fn normal_fill16_bf16(data: &mut [half::bf16]) {
    debug_assert_eq!(data.len(), 16);
    let mean = half::bf16::from_f32(0.0);
    let std = half::bf16::from_f32(1.0);
    let one = half::bf16::from_f32(1.0);
    let minus_two = half::bf16::from_f32(-2.0);
    for j in 0..8 {
        let u1 = half::bf16::from_f32(one.to_f32() - data[j].to_f32());
        let u2 = data[j + 8];
        let log_u1 = half::bf16::from_f32(u1.to_f32().ln());
        let scaled_log = half::bf16::from_f32(minus_two.to_f32() * log_u1.to_f32());
        let radius = half::bf16::from_f32(scaled_log.to_f32().sqrt());
        let theta =
            half::bf16::from_f32((2.0_f64 * std::f64::consts::PI * (u2.to_f32() as f64)) as f32);
        let (sin_theta, cos_theta) = theta.to_f32().sin_cos();
        let sin_theta = half::bf16::from_f32(sin_theta);
        let cos_theta = half::bf16::from_f32(cos_theta);

        let out1 = half::bf16::from_f32(radius.to_f32() * cos_theta.to_f32());
        let out1 = half::bf16::from_f32(out1.to_f32() * std.to_f32());
        data[j] = half::bf16::from_f32(out1.to_f32() + mean.to_f32());

        let out2 = half::bf16::from_f32(radius.to_f32() * sin_theta.to_f32());
        let out2 = half::bf16::from_f32(out2.to_f32() * std.to_f32());
        data[j + 8] = half::bf16::from_f32(out2.to_f32() + mean.to_f32());
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::wildcard_imports)]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn normal_fill16_f32_avx2(data: &mut [f32]) {
    use std::arch::x86_64::*;

    let ptr = data.as_mut_ptr();
    let one = _mm256_set1_ps(1.0);
    let minus_two = _mm256_set1_ps(-2.0);
    let mean = _mm256_setzero_ps();
    let std = _mm256_set1_ps(1.0);
    let two_pi = _mm256_set1_ps((2.0 * std::f64::consts::PI) as f32);

    let u1 = _mm256_sub_ps(one, unsafe { _mm256_loadu_ps(ptr) });
    let u2 = unsafe { _mm256_loadu_ps(ptr.add(8)) };
    let radius = _mm256_sqrt_ps(_mm256_mul_ps(minus_two, avx2_log256_ps(u1)));
    let theta = _mm256_mul_ps(two_pi, u2);
    let (sin_theta, cos_theta) = avx2_sincos256_ps(theta);
    let out1 = _mm256_fmadd_ps(_mm256_mul_ps(radius, cos_theta), std, mean);
    let out2 = _mm256_fmadd_ps(_mm256_mul_ps(radius, sin_theta), std, mean);
    unsafe {
        _mm256_storeu_ps(ptr, out1);
        _mm256_storeu_ps(ptr.add(8), out2);
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::excessive_precision, clippy::wildcard_imports)]
#[inline(never)]
#[target_feature(enable = "avx2")]
fn avx2_log256_ps(mut x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    let one = _mm256_set1_ps(1.0);
    let invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);
    x = _mm256_max_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x0080_0000)));

    let mut imm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);
    x = _mm256_and_ps(
        x,
        _mm256_castsi256_ps(_mm256_set1_epi32(!0x7f80_0000u32 as i32)),
    );
    x = _mm256_or_ps(x, _mm256_set1_ps(0.5));

    imm0 = _mm256_sub_epi32(imm0, _mm256_set1_epi32(0x7f));
    let mut e = _mm256_cvtepi32_ps(imm0);
    e = _mm256_add_ps(e, one);

    let mask = _mm256_cmp_ps(x, _mm256_set1_ps(0.707_106_77), _CMP_LT_OS);
    let tmp = _mm256_and_ps(x, mask);
    x = _mm256_sub_ps(x, one);
    e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
    x = _mm256_add_ps(x, tmp);

    let z = _mm256_mul_ps(x, x);
    let mut y = _mm256_set1_ps(7.037_683_629_2e-2);
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(-1.151_461_031_0e-1));
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(1.167_699_874_0e-1));
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(-1.242_014_084_6e-1));
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(1.424_932_278_7e-1));
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(-1.666_805_766_5e-1));
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(2.000_071_476_5e-1));
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(-2.499_999_399_3e-1));
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(3.333_333_117_4e-1));
    y = _mm256_mul_ps(y, x);
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, _mm256_mul_ps(e, _mm256_set1_ps(-2.121_944_4e-4)));
    y = _mm256_sub_ps(y, _mm256_mul_ps(z, _mm256_set1_ps(0.5)));
    x = _mm256_add_ps(x, y);
    x = _mm256_add_ps(x, _mm256_mul_ps(e, _mm256_set1_ps(0.693_359_4)));
    _mm256_or_ps(x, invalid_mask)
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::wildcard_imports)]
#[inline(never)]
#[target_feature(enable = "avx2", enable = "fma")]
fn avx2_sincos256_ps(
    mut x: std::arch::x86_64::__m256,
) -> (std::arch::x86_64::__m256, std::arch::x86_64::__m256) {
    use std::arch::x86_64::*;

    let inv_sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(!0x8000_0000u32 as i32));
    let sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x8000_0000u32 as i32));
    let mut sign_bit_sin = _mm256_and_ps(x, sign_mask);
    x = _mm256_and_ps(x, inv_sign_mask);

    let mut y = _mm256_mul_ps(x, _mm256_set1_ps(1.273_239_5));
    let mut imm2 = _mm256_cvttps_epi32(y);
    imm2 = _mm256_add_epi32(imm2, _mm256_set1_epi32(1));
    imm2 = _mm256_and_si256(imm2, _mm256_set1_epi32(!1));
    y = _mm256_cvtepi32_ps(imm2);
    let mut imm4 = imm2;

    let mut imm0 = _mm256_and_si256(imm2, _mm256_set1_epi32(4));
    imm0 = _mm256_slli_epi32(imm0, 29);
    imm2 = _mm256_and_si256(imm2, _mm256_set1_epi32(2));
    imm2 = _mm256_cmpeq_epi32(imm2, _mm256_setzero_si256());

    let swap_sign_bit_sin = _mm256_castsi256_ps(imm0);
    let poly_mask = _mm256_castsi256_ps(imm2);

    let xmm1 = _mm256_mul_ps(y, _mm256_set1_ps(-0.785_156_25));
    let xmm2 = _mm256_mul_ps(y, _mm256_set1_ps(-2.418_756_5e-4));
    let xmm3 = _mm256_mul_ps(y, _mm256_set1_ps(-3.774_895e-8));
    x = _mm256_add_ps(x, xmm1);
    x = _mm256_add_ps(x, xmm2);
    x = _mm256_add_ps(x, xmm3);

    imm4 = _mm256_sub_epi32(imm4, _mm256_set1_epi32(2));
    imm4 = _mm256_andnot_si256(imm4, _mm256_set1_epi32(4));
    imm4 = _mm256_slli_epi32(imm4, 29);
    let sign_bit_cos = _mm256_castsi256_ps(imm4);
    sign_bit_sin = _mm256_xor_ps(sign_bit_sin, swap_sign_bit_sin);

    let z = _mm256_mul_ps(x, x);
    let mut y_cos = _mm256_set1_ps(2.443_315_7e-5);
    y_cos = _mm256_fmadd_ps(y_cos, z, _mm256_set1_ps(-1.388_731_6e-3));
    y_cos = _mm256_fmadd_ps(y_cos, z, _mm256_set1_ps(4.166_664_6e-2));
    y_cos = _mm256_mul_ps(y_cos, z);
    y_cos = _mm256_mul_ps(y_cos, z);
    y_cos = _mm256_sub_ps(y_cos, _mm256_mul_ps(z, _mm256_set1_ps(0.5)));
    y_cos = _mm256_add_ps(y_cos, _mm256_set1_ps(1.0));

    let mut y_sin = _mm256_set1_ps(-1.951_529_6e-4);
    y_sin = _mm256_fmadd_ps(y_sin, z, _mm256_set1_ps(8.332_161e-3));
    y_sin = _mm256_fmadd_ps(y_sin, z, _mm256_set1_ps(-1.666_665_5e-1));
    y_sin = _mm256_mul_ps(y_sin, z);
    y_sin = _mm256_fmadd_ps(y_sin, x, x);

    let ysin2 = _mm256_and_ps(poly_mask, y_sin);
    let ysin1 = _mm256_andnot_ps(poly_mask, y_cos);
    y_sin = _mm256_sub_ps(y_sin, ysin2);
    y_cos = _mm256_sub_ps(y_cos, ysin1);

    let sin_v = _mm256_xor_ps(_mm256_add_ps(ysin1, ysin2), sign_bit_sin);
    let cos_v = _mm256_xor_ps(_mm256_add_ps(y_cos, y_sin), sign_bit_cos);
    (sin_v, cos_v)
}

/// Scalarized lane of PyTorch's AVX2 `log256_ps` (`avx_mathfun.h`).
///
/// `normal_fill<float>` uses this approximation on AVX2 builds. Each lane is
/// independent for the Box-Muller inputs, so preserving the polynomial and
/// operation order is enough to reproduce PyTorch's per-element bits on the
/// local AVX2 CPU without introducing C/C++ code.
#[allow(clippy::excessive_precision)]
fn avx_mathfun_log_f32(mut x: f32) -> f32 {
    let invalid = x <= 0.0;
    x = x.max(f32::from_bits(0x0080_0000));

    let bits = x.to_bits();
    let exponent = ((bits >> 23) as i32) - 0x7f;
    x = f32::from_bits((bits & !0x7f80_0000) | 0x3f00_0000);

    let mut e = exponent as f32;
    e += 1.0;

    let mask = x < 0.707_106_77_f32;
    let tmp = if mask { x } else { 0.0 };
    x -= 1.0;
    if mask {
        e -= 1.0;
    }
    x += tmp;

    let z = x * x;
    let mut y = 7.037_683_629_2e-2_f32;
    y = y * x + -1.151_461_031_0e-1_f32;
    y = y * x + 1.167_699_874_0e-1_f32;
    y = y * x + -1.242_014_084_6e-1_f32;
    y = y * x + 1.424_932_278_7e-1_f32;
    y = y * x + -1.666_805_766_5e-1_f32;
    y = y * x + 2.000_071_476_5e-1_f32;
    y = y * x + -2.499_999_399_3e-1_f32;
    y = y * x + 3.333_333_117_4e-1_f32;
    y *= x;
    y *= z;
    y += e * -2.121_944_4e-4_f32;
    y -= z * 0.5;
    x += y;
    x += e * 0.693_359_4_f32;

    if invalid { f32::NAN } else { x }
}

/// Scalarized lane of PyTorch's AVX2 `sincos256_ps` (`avx_mathfun.h`).
fn avx_mathfun_sincos_f32(x: f32) -> (f32, f32) {
    let sign_bit_sin = x.to_bits() & 0x8000_0000;
    let mut x = f32::from_bits(x.to_bits() & !0x8000_0000);

    let y = x * 1.273_239_5_f32;
    let mut imm2 = y as i32;
    imm2 = (imm2 + 1) & !1;
    let y = imm2 as f32;
    let imm4 = imm2;

    let swap_sign_bit_sin = ((imm2 & 4) as u32) << 29;
    let poly_mask = (imm2 & 2) == 0;

    x += y * -0.785_156_25_f32;
    x += y * -2.418_756_5e-4_f32;
    x += y * -3.774_895e-8_f32;

    let imm4 = imm4.wrapping_sub(2);
    let sign_bit_cos = (((!imm4) & 4) as u32) << 29;
    let sign_bit_sin = sign_bit_sin ^ swap_sign_bit_sin;

    let z = x * x;
    let mut y_cos = 2.443_315_7e-5_f32;
    y_cos = y_cos * z + -1.388_731_6e-3_f32;
    y_cos = y_cos * z + 4.166_664_6e-2_f32;
    y_cos *= z;
    y_cos *= z;
    y_cos -= z * 0.5;
    y_cos += 1.0;

    let mut y_sin = -1.951_529_6e-4_f32;
    y_sin = y_sin * z + 8.332_161e-3_f32;
    y_sin = y_sin * z + -1.666_665_5e-1_f32;
    y_sin *= z;
    y_sin *= x;
    y_sin += x;

    let ysin2 = if poly_mask { y_sin } else { 0.0 };
    let ysin1 = if poly_mask { 0.0 } else { y_cos };
    y_sin -= ysin2;
    y_cos -= ysin1;

    let sin_v = f32::from_bits((ysin1 + ysin2).to_bits() ^ sign_bit_sin);
    let cos_v = f32::from_bits((y_cos + y_sin).to_bits() ^ sign_bit_cos);
    (sin_v, cos_v)
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
    fn enter(operation: &'static str) -> FerrotorchResult<Self> {
        DEFAULT_RNG_ACTIVE.with(|active| -> FerrotorchResult<()> {
            if active.get() {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "{operation}: default RNG is already mutably borrowed; use an explicit \
                         Generator for nested random generation"
                    ),
                });
            }
            active.set(true);
            Ok(())
        })?;
        Ok(Self)
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

struct DefaultRngTestSerialGuard {
    guard: Option<MutexGuard<'static, ()>>,
}

impl Drop for DefaultRngTestSerialGuard {
    fn drop(&mut self) {
        if self.guard.is_some() {
            DEFAULT_RNG_TEST_SERIAL_ACTIVE.with(|active| active.set(false));
        }
    }
}

thread_local! {
    static DEFAULT_RNG_TEST_SERIAL_ACTIVE: Cell<bool> = const { Cell::new(false) };
}

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

/// Run a test-only critical section over the process-global default RNG.
///
/// This is for tests that need `manual_seed(seed)` and one or more following
/// random operations to be observed as one deterministic transaction under the
/// parallel Rust test harness. The normal release RNG paths still avoid this
/// extra serialization unless a test calls this hidden helper directly.
#[doc(hidden)]
pub fn with_default_rng_test_lock<R>(f: impl FnOnce() -> R) -> R {
    let _guard = default_rng_test_serial_guard();
    f()
}

/// Set the process-global default RNG seed — mirrors `torch.manual_seed`
/// at `torch/random.py:49-86`.
///
/// # Production consumer
///
/// `crate::creation::rand`/`randn` consume bits from this shared default
/// generator. Calling `manual_seed(42)` in any thread seeds registered GPU
/// generators, then reseeds the CPU default stream seen by subsequently
/// scheduled random creation on any thread, matching PyTorch's all-device
/// ordering. A registered backend seeding failure is returned before CPU state
/// is changed.
pub fn manual_seed(seed: u64) -> FerrotorchResult<()> {
    #[cfg(debug_assertions)]
    let _serial = default_rng_test_serial_guard();

    let _access = DefaultRngAccessGuard::enter("manual_seed")?;
    // Mirror `torch.manual_seed`, which seeds BOTH the CPU and all CUDA
    // generators: `torch/random.py:67` calls `torch.cuda.manual_seed_all(seed)`
    // (`torch/cuda/random.py:112`). When a GPU backend is registered, forward
    // the seed to its per-device Philox manager so that
    // `creation::rand_on_device(.., Cuda)` after `manual_seed` is reproducible.
    // No-op when CUDA is unavailable, matching torch's "silently ignored if
    // CUDA is not available" contract. If a registered backend reports a real
    // seeding failure, propagate it instead of claiming deterministic parity.
    if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
        backend.manual_seed_gpu(seed)?;
    }
    lock_default_rng().manual_seed(seed);
    Ok(())
}

/// Run a closure with mutable access to the process-global default generator.
///
/// The historical name is retained for API compatibility. The semantics match
/// PyTorch's default CPU generator: one process-wide stream serialized behind a
/// mutex, not a per-thread RNG. Used by `creation::rand` / `creation::randn`
/// and by `ferrotorch-nn` initialisers that don't take an explicit
/// [`Generator`].
pub fn with_thread_rng<R>(f: impl FnOnce(&mut Generator) -> R) -> FerrotorchResult<R> {
    #[cfg(debug_assertions)]
    let _serial = default_rng_test_serial_guard();

    let _access = DefaultRngAccessGuard::enter("with_thread_rng")?;
    let mut rng = lock_default_rng();
    Ok(f(&mut rng))
}

/// Clone the process-global CPU RNG state, including cached normal samples.
/// Checkpointing uses this to mirror `torch.get_rng_state()`.
pub(crate) fn thread_rng_state() -> FerrotorchResult<Generator> {
    #[cfg(debug_assertions)]
    let _serial = default_rng_test_serial_guard();

    let _access = DefaultRngAccessGuard::enter("thread_rng_state")?;
    Ok(lock_default_rng().clone())
}

/// Restore the process-global CPU RNG state. Checkpointing uses this
/// inside a fork-style guard so stochastic recomputation sees the same stream
/// as the original forward while the caller's surrounding stream is restored.
pub(crate) fn set_thread_rng_state(state: Generator) -> FerrotorchResult<()> {
    #[cfg(debug_assertions)]
    let _serial = default_rng_test_serial_guard();

    let _access = DefaultRngAccessGuard::enter("set_thread_rng_state")?;
    *lock_default_rng() = state;
    Ok(())
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
        manual_seed(42).unwrap();
        let a: Vec<u32> = (0..5)
            .map(|_| with_thread_rng(|g| g.random_u32()).unwrap())
            .collect();
        manual_seed(42).unwrap();
        let b: Vec<u32> = (0..5)
            .map(|_| with_thread_rng(|g| g.random_u32()).unwrap())
            .collect();
        assert_eq!(a, b);
    }

    #[test]
    fn manual_seed_distinct_seeds_distinct_streams() {
        let _guard = default_rng_test_lock();
        manual_seed(42).unwrap();
        let a = with_thread_rng(|g| g.random_u32()).unwrap();
        manual_seed(43).unwrap();
        let b = with_thread_rng(|g| g.random_u32()).unwrap();
        assert_ne!(a, b);
    }

    fn draw_default_uniform_bits(n: usize) -> Vec<u32> {
        with_thread_rng(|g| (0..n).map(|_| g.next_uniform_f32().to_bits()).collect()).unwrap()
    }

    #[test]
    fn manual_seed_reaches_fresh_worker_thread() {
        let _guard = default_rng_test_lock();
        manual_seed(42).unwrap();

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
            let _ = with_thread_rng(|g| g.random_u32()).unwrap();
            ready_tx.send(()).expect("main should wait for ready");
            go_rx.recv().expect("main should signal draw");
            out_tx
                .send(draw_default_uniform_bits(10))
                .expect("main should receive output");
        });

        ready_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("worker should initialize the default RNG");
        manual_seed(42).unwrap();
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
        manual_seed(42).unwrap();

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

    fn assert_reentrant_error(err: FerrotorchError, operation: &str) {
        match err {
            FerrotorchError::InvalidArgument { message } => {
                assert!(
                    message.contains(operation),
                    "error should name {operation}, got {message}"
                );
                assert!(
                    message.contains("already mutably borrowed"),
                    "error should explain the active default RNG borrow, got {message}"
                );
                assert!(
                    message.contains("explicit Generator"),
                    "error should direct callers to explicit Generator, got {message}"
                );
            }
            other => panic!("expected InvalidArgument for reentrant RNG access, got {other:?}"),
        }
    }

    #[test]
    fn nested_with_thread_rng_returns_structured_error() {
        let _guard = default_rng_test_lock();
        manual_seed(7).unwrap();

        let nested = with_thread_rng(|g| {
            let first = g.random_u32();
            let nested = with_thread_rng(|inner| inner.random_u32());
            (first, nested)
        })
        .expect("outer access should succeed");

        assert_reentrant_error(
            nested.1.expect_err("nested access must fail"),
            "with_thread_rng",
        );
        let after = with_thread_rng(|g| g.random_u32()).expect("guard should release after error");
        assert_ne!(nested.0, after, "outer draw must still advance the stream");
    }

    #[test]
    fn rand_inside_with_thread_rng_returns_structured_error() {
        let _guard = default_rng_test_lock();
        manual_seed(8).unwrap();

        let nested = with_thread_rng(|_| crate::creation::rand::<f32>(&[1]))
            .expect("outer access should succeed");

        assert_reentrant_error(
            nested.expect_err("nested rand must fail"),
            "with_thread_rng",
        );
        let after = crate::creation::rand::<f32>(&[1]).expect("guard should release after error");
        assert_eq!(after.shape(), &[1]);
    }

    #[test]
    fn manual_seed_inside_with_thread_rng_returns_structured_error() {
        let _guard = default_rng_test_lock();
        manual_seed(9).unwrap();

        let nested = with_thread_rng(|_| manual_seed(10)).expect("outer access should succeed");

        assert_reentrant_error(
            nested.expect_err("nested manual_seed must fail"),
            "manual_seed",
        );
        manual_seed(10).expect("guard should release after error");
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
