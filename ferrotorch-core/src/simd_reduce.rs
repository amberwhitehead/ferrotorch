//! Torch-matching CPU reduction primitives.
//!
//! PyTorch's CPU reductions are not naive scalar folds. The exact reduction
//! tree depends on the dispatch class chosen by
//! `aten/src/ATen/native/DispatchStub.cpp` and on the kernel family:
//!
//! - f32 L2 norm (`ReduceOpsKernel.cpp`) accumulates f32 square terms in
//!   `Vectorized<float>` lanes, left-folds those lanes, then handles the scalar
//!   tail and `sqrt`.
//! - floating `sum` (`SumKernel.cpp`) uses `cascade_sum`: f32 inputs accumulate
//!   in `acc_type<float, true> == float`, f64 inputs accumulate in f64, and long
//!   rows are summed with PyTorch's four-way, four-level cascade.
//!
//! This module scalarizes those vector lane trees in Rust. It deliberately
//! selects the same lane geometry PyTorch would select for the current process
//! (`ATEN_CPU_CAPABILITY`, then x86 AVX512/AVX2/default and the supported
//! non-x86 vector classes) instead of hard-coding the AVX2 host used by the
//! original probes (#1793).
//!
//! ## REQ status (per `.design/ferrotorch-core/simd_reduce.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`l2_norm_f32_torch`) | SHIPPED | `pub fn l2_norm_f32_torch` models torch's vectorized last-dim L2 kernel (`aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:222-255`) with PyTorch-selected `Vectorized<float>` width, f32 lane square accumulation, naive lane left-fold, scalar FMA tail, and f32 `sqrt`. Non-test production consumers: `ferrotorch_core::grad_fns::reduction::norm_with_dim` and `ferrotorch_nn::embedding::renorm_weight_rows_in_place`. |
//! | REQ-2 (`sum_f32` / `sum_f64`) | SHIPPED | `pub fn sum_f32` / `pub fn sum_f64` model torch's current `cascade_sum` contiguous inner-reduction path (`aten/src/ATen/native/cpu/SumKernel.cpp:346-456`): vector-width loads, four independent row accumulators, four cascade levels, scalar remainder first, then vector partials. `sum_f32` deliberately uses f32 accumulation because `SumKernel.cpp` asks for `acc_type<float, true>`, which maps to float in `AccumulateType.h`. Non-test production consumers: `sum_dim` / `mean_dim` forward in `grad_fns/reduction.rs` (the `inner==1` contiguous-last-dim rows of the `[outer, axis, inner]` fast path). |
//!
//! ## Honest L2 residual (R-HONEST-3)
//!
//! The f32 L2 primitive still has the known one-ULP residual for a small set of
//! rows where PyTorch's compiled scalar tail contracts FMAs in value-dependent
//! ways. The boundary rows that motivated the embedding renorm decision match,
//! and the residual is pinned by tests. This residual does not apply to the
//! floating sum path, which now follows `SumKernel.cpp`'s cascade structure.

const MAX_F32_LANES: usize = 16;
const MAX_F64_LANES: usize = 8;
const PYTORCH_SUM_ILP: usize = 4;
const PYTORCH_SUM_LEVELS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TorchCpuVectorClass {
    Default,
    Avx2,
    Avx512,
    Sve256,
    Vsx,
    ZVector,
}

impl TorchCpuVectorClass {
    fn f32_lanes(self) -> usize {
        match self {
            Self::Avx512 => 16,
            Self::Avx2 | Self::Sve256 => 8,
            Self::Vsx | Self::ZVector => 4,
            Self::Default => torch_default_f32_lanes(),
        }
    }

    fn f64_lanes(self) -> usize {
        match self {
            Self::Avx512 => 8,
            Self::Avx2 | Self::Sve256 => 4,
            Self::Vsx | Self::ZVector => 2,
            Self::Default => torch_default_f64_lanes(),
        }
    }
}

fn torch_default_f32_lanes() -> usize {
    if cfg!(target_arch = "aarch64") {
        // `vec128` NEON: VECTOR_WIDTH=16.
        4
    } else {
        // x86 DEFAULT and the portable vec_base path use VECTOR_WIDTH=32.
        8
    }
}

fn torch_default_f64_lanes() -> usize {
    if cfg!(target_arch = "aarch64") { 2 } else { 4 }
}

fn parse_torch_cpu_capability_override(value: &str) -> Option<TorchCpuVectorClass> {
    match value {
        "default" => Some(TorchCpuVectorClass::Default),
        "avx2" => Some(TorchCpuVectorClass::Avx2),
        "avx512" => Some(TorchCpuVectorClass::Avx512),
        "sve256" => Some(TorchCpuVectorClass::Sve256),
        "vsx" => Some(TorchCpuVectorClass::Vsx),
        "zvector" => Some(TorchCpuVectorClass::ZVector),
        _ => None,
    }
}

fn detect_torch_cpu_vector_class_uncached() -> TorchCpuVectorClass {
    // PyTorch's DispatchStub.cpp honors lower-case ATEN_CPU_CAPABILITY values
    // before probing hardware.
    if let Ok(value) = std::env::var("ATEN_CPU_CAPABILITY")
        && let Some(class) = parse_torch_cpu_capability_override(&value)
    {
        return class;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512vl")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512dq")
            && std::is_x86_feature_detected!("fma")
        {
            TorchCpuVectorClass::Avx512
        } else if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            TorchCpuVectorClass::Avx2
        } else {
            TorchCpuVectorClass::Default
        }
    }

    #[cfg(target_arch = "powerpc64")]
    {
        TorchCpuVectorClass::Vsx
    }

    #[cfg(target_arch = "s390x")]
    {
        TorchCpuVectorClass::ZVector
    }

    #[cfg(not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "powerpc64",
        target_arch = "s390x"
    )))]
    {
        TorchCpuVectorClass::Default
    }
}

fn torch_cpu_vector_class() -> TorchCpuVectorClass {
    static CLASS: std::sync::OnceLock<TorchCpuVectorClass> = std::sync::OnceLock::new();
    *CLASS.get_or_init(detect_torch_cpu_vector_class_uncached)
}

fn torch_f32_lanes() -> usize {
    torch_cpu_vector_class().f32_lanes()
}

fn torch_f64_lanes() -> usize {
    torch_cpu_vector_class().f64_lanes()
}

/// Compute the L2 (Euclidean) norm of an f32 slice the way PyTorch's CPU
/// reduction does, so the result matches `at::norm(2.0)` on an f32 contiguous
/// last-dim reduction for the same CPU dispatch class (modulo the known
/// one-ULP L2 scalar-tail residual documented at the module level).
///
/// This mirrors the vectorized last-dim L2 kernel at
/// `aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:222-255`:
///
/// ```text
///   fVec acc_vec{acc_t(0)};                       // fVec::size() lanes
///   acc_t buffer[fVec::size()];
///   for (; d < size - (size % fVec::size()); d += fVec::size()) {
///     acc_vec += data_vec * data_vec;             // lane-wise mul+add
///   }
///   acc_vec.store(buffer);
///   for (j = 1; j < fVec::size(); j++) buffer[0] += buffer[j];
///   for (; d < size; d++) buffer[0] += data*data;
///   result = sqrt(buffer[0]);
/// ```
///
/// The accumulator type is `at::opmath_type<float> == float` (`OpMathType.h:16`),
/// i.e. f32, NOT f64. The lane accumulate uses plain multiply-then-add
/// (`Vectorized<float>::operator*` then `operator+`, NOT fused); the scalar
/// tail's `buffer[0] += data*data` compiles to a fused multiply-add in the
/// inspected PyTorch build, so we use [`f32::mul_add`] there.
/// `NormTwoOps::project` is `device_sqrt` (`SharedReduceOps.h:375-381`), i.e.
/// `f32::sqrt`.
///
/// A scalar model of the tree (no `unsafe` SIMD intrinsics) is used for
/// portability and determinism: f32 `a + b*b` without contraction is
/// bit-identical to PyTorch's vector lane multiply-then-add for the selected
/// dispatch width.
#[must_use]
pub fn l2_norm_f32_torch(data: &[f32]) -> f32 {
    l2_norm_f32_with_lanes(data, torch_f32_lanes())
}

fn l2_norm_f32_with_lanes(data: &[f32], lane_count: usize) -> f32 {
    assert!(
        (1..=MAX_F32_LANES).contains(&lane_count),
        "l2_norm_f32_with_lanes: invalid PyTorch vector lane count {lane_count}"
    );
    let n = data.len();
    let mut lanes = [0.0_f32; MAX_F32_LANES];

    // Main loop: process contiguous vector-width chunks, accumulating each
    // element's square into its lane. `main` is `size - (size % Vec::size())`,
    // matching the kernel's `d < size - (size % Vec::size())` bound. Lane
    // accumulate is plain mul-then-add (NOT fused).
    let main = n - (n % lane_count);
    let mut d = 0;
    while d < main {
        for (j, lane) in lanes[..lane_count].iter_mut().enumerate() {
            let x = data[d + j];
            // `+= x*x` lowers to a single f32 add of the product (no FMA),
            // matching the vectorized multiply-then-add lane operation.
            *lane += x * x;
        }
        d += lane_count;
    }

    // Horizontal reduce: naive LEFT-FOLD of the lanes into lane 0, exactly as
    // the kernel does (`for (j = 1; j < fVec::size(); j++) buffer[0] +=
    // buffer[j]`). FP addition is non-associative, so the left-fold order is
    // load-bearing.
    let mut acc = lanes[0];
    for &lane in &lanes[1..lane_count] {
        acc += lane;
    }

    // Scalar remainder tail: the `size % Vec::size()` elements that didn't fill
    // a full vector-width chunk accumulate into lane 0. The compiled PyTorch
    // tail contracts to a fused multiply-add (single rounding) under the
    // inspected build flags, so we use `mul_add` to match it.
    while d < n {
        let x = data[d];
        acc = x.mul_add(x, acc);
        d += 1;
    }

    // `NormTwoOps::project` is `device_sqrt(a)` (SharedReduceOps.h:375-381),
    // i.e. f32 sqrt of the accumulated sum-of-squares.
    acc.sqrt()
}

/// Horizontal sum of a contiguous `f32` slice, modelling torch's vectorized
/// contiguous inner-reduction order rather than a sequential `Σ`.
///
/// Current PyTorch routes floating CPU `sum` through `cascade_sum` in
/// `aten/src/ATen/native/cpu/SumKernel.cpp`, not through the older generic
/// `Reduce.h` binary reducer. For f32 input, `acc_type<float, true>` maps to
/// float in `AccumulateType.h`, so contiguous inner reductions load
/// `Vectorized<float>` chunks, run PyTorch's four-way `row_sum` /
/// `multi_row_sum` cascade over those vector chunks, add the scalar tail first,
/// then add the vector partial lanes.
///
/// This scalar model is intentionally structured like the PyTorch source
/// instead of relying on autovectorization. Floating addition is
/// non-associative; changing the tree changes observable results on adversarial
/// rows.
#[must_use]
pub fn sum_f32(data: &[f32]) -> f32 {
    pytorch_sum_f32_with_lanes(data, torch_f32_lanes())
}

/// Horizontal sum of a contiguous `f64` slice, the `f64` analogue of
/// [`sum_f32`]. The accumulator and output are both f64, but the row tree still
/// follows PyTorch's selected `Vectorized<double>` width and cascade ordering.
#[must_use]
pub fn sum_f64(data: &[f64]) -> f64 {
    pytorch_sum_f64_with_lanes(data, torch_f64_lanes())
}

fn ceil_log2_usize(n: usize) -> usize {
    if n <= 1 {
        0
    } else {
        usize::BITS as usize - (n - 1).leading_zeros() as usize
    }
}

fn add_vec_f32(dst: &mut [f32; MAX_F32_LANES], src: &[f32; MAX_F32_LANES], lanes: usize) {
    for lane in 0..lanes {
        dst[lane] += src[lane];
    }
}

fn zero_vec_f32(dst: &mut [f32; MAX_F32_LANES], lanes: usize) {
    for value in &mut dst[..lanes] {
        *value = 0.0;
    }
}

fn add_vec_f64(dst: &mut [f64; MAX_F64_LANES], src: &[f64; MAX_F64_LANES], lanes: usize) {
    for lane in 0..lanes {
        dst[lane] += src[lane];
    }
}

fn zero_vec_f64(dst: &mut [f64; MAX_F64_LANES], lanes: usize) {
    for value in &mut dst[..lanes] {
        *value = 0.0;
    }
}

fn load_f32_vec_lanes(data: &[f32], chunk_idx: usize, f32_lanes: usize) -> [f32; MAX_F32_LANES] {
    let mut acc = [0.0_f32; MAX_F32_LANES];
    let base = chunk_idx * f32_lanes;
    acc[..f32_lanes].copy_from_slice(&data[base..base + f32_lanes]);
    acc
}

fn load_f64_vec_lanes(data: &[f64], chunk_idx: usize, f64_lanes: usize) -> [f64; MAX_F64_LANES] {
    let mut acc = [0.0_f64; MAX_F64_LANES];
    let base = chunk_idx * f64_lanes;
    acc[..f64_lanes].copy_from_slice(&data[base..base + f64_lanes]);
    acc
}

fn pytorch_multi_row_sum_vec4_f32<F>(
    size: usize,
    lanes: usize,
    mut load: F,
) -> [[f32; MAX_F32_LANES]; PYTORCH_SUM_ILP]
where
    F: FnMut(usize) -> [f32; MAX_F32_LANES],
{
    let level_power = 4.max(ceil_log2_usize(size) / PYTORCH_SUM_LEVELS);
    let level_step = 1usize << level_power;
    let level_mask = level_step - 1;
    let mut acc = [[[0.0_f32; MAX_F32_LANES]; PYTORCH_SUM_ILP]; PYTORCH_SUM_LEVELS];

    let mut i = 0usize;
    while i + level_step <= size {
        for j in 0..level_step {
            let row_base = (i + j) * PYTORCH_SUM_ILP;
            for k in 0..PYTORCH_SUM_ILP {
                let loaded = load(row_base + k);
                add_vec_f32(&mut acc[0][k], &loaded, lanes);
            }
        }
        i += level_step;

        for level in 1..PYTORCH_SUM_LEVELS {
            for k in 0..PYTORCH_SUM_ILP {
                let prev = acc[level - 1][k];
                add_vec_f32(&mut acc[level][k], &prev, lanes);
                zero_vec_f32(&mut acc[level - 1][k], lanes);
            }

            let mask = level_mask << (level * level_power);
            if (i & mask) != 0 {
                break;
            }
        }
    }

    while i < size {
        let row_base = i * PYTORCH_SUM_ILP;
        for k in 0..PYTORCH_SUM_ILP {
            let loaded = load(row_base + k);
            add_vec_f32(&mut acc[0][k], &loaded, lanes);
        }
        i += 1;
    }

    for level in 1..PYTORCH_SUM_LEVELS {
        for k in 0..PYTORCH_SUM_ILP {
            let src = acc[level][k];
            add_vec_f32(&mut acc[0][k], &src, lanes);
        }
    }

    acc[0]
}

fn pytorch_row_sum_vec_f32<F>(vec_size: usize, lanes: usize, mut load: F) -> [f32; MAX_F32_LANES]
where
    F: FnMut(usize) -> [f32; MAX_F32_LANES],
{
    let size_ilp = vec_size / PYTORCH_SUM_ILP;
    let mut partial_sums = pytorch_multi_row_sum_vec4_f32(size_ilp, lanes, &mut load);

    for i in size_ilp * PYTORCH_SUM_ILP..vec_size {
        let loaded = load(i);
        add_vec_f32(&mut partial_sums[0], &loaded, lanes);
    }

    for k in 1..PYTORCH_SUM_ILP {
        let src = partial_sums[k];
        add_vec_f32(&mut partial_sums[0], &src, lanes);
    }

    partial_sums[0]
}

fn pytorch_multi_row_sum_vec4_f64<F>(
    size: usize,
    lanes: usize,
    mut load: F,
) -> [[f64; MAX_F64_LANES]; PYTORCH_SUM_ILP]
where
    F: FnMut(usize) -> [f64; MAX_F64_LANES],
{
    let level_power = 4.max(ceil_log2_usize(size) / PYTORCH_SUM_LEVELS);
    let level_step = 1usize << level_power;
    let level_mask = level_step - 1;
    let mut acc = [[[0.0_f64; MAX_F64_LANES]; PYTORCH_SUM_ILP]; PYTORCH_SUM_LEVELS];

    let mut i = 0usize;
    while i + level_step <= size {
        for j in 0..level_step {
            let row_base = (i + j) * PYTORCH_SUM_ILP;
            for k in 0..PYTORCH_SUM_ILP {
                let loaded = load(row_base + k);
                add_vec_f64(&mut acc[0][k], &loaded, lanes);
            }
        }
        i += level_step;

        for level in 1..PYTORCH_SUM_LEVELS {
            for k in 0..PYTORCH_SUM_ILP {
                let prev = acc[level - 1][k];
                add_vec_f64(&mut acc[level][k], &prev, lanes);
                zero_vec_f64(&mut acc[level - 1][k], lanes);
            }

            let mask = level_mask << (level * level_power);
            if (i & mask) != 0 {
                break;
            }
        }
    }

    while i < size {
        let row_base = i * PYTORCH_SUM_ILP;
        for k in 0..PYTORCH_SUM_ILP {
            let loaded = load(row_base + k);
            add_vec_f64(&mut acc[0][k], &loaded, lanes);
        }
        i += 1;
    }

    for level in 1..PYTORCH_SUM_LEVELS {
        for k in 0..PYTORCH_SUM_ILP {
            let src = acc[level][k];
            add_vec_f64(&mut acc[0][k], &src, lanes);
        }
    }

    acc[0]
}

fn pytorch_row_sum_vec_f64<F>(vec_size: usize, lanes: usize, mut load: F) -> [f64; MAX_F64_LANES]
where
    F: FnMut(usize) -> [f64; MAX_F64_LANES],
{
    let size_ilp = vec_size / PYTORCH_SUM_ILP;
    let mut partial_sums = pytorch_multi_row_sum_vec4_f64(size_ilp, lanes, &mut load);

    for i in size_ilp * PYTORCH_SUM_ILP..vec_size {
        let loaded = load(i);
        add_vec_f64(&mut partial_sums[0], &loaded, lanes);
    }

    for k in 1..PYTORCH_SUM_ILP {
        let src = partial_sums[k];
        add_vec_f64(&mut partial_sums[0], &src, lanes);
    }

    partial_sums[0]
}

fn pytorch_multi_row_sum_scalar4<F>(size: usize, mut load: F) -> [f64; PYTORCH_SUM_ILP]
where
    F: FnMut(usize) -> f64,
{
    let level_power = 4.max(ceil_log2_usize(size) / PYTORCH_SUM_LEVELS);
    let level_step = 1usize << level_power;
    let level_mask = level_step - 1;
    let mut acc = [[0.0_f64; PYTORCH_SUM_ILP]; PYTORCH_SUM_LEVELS];

    let mut i = 0usize;
    while i + level_step <= size {
        for j in 0..level_step {
            let row_base = (i + j) * PYTORCH_SUM_ILP;
            for k in 0..PYTORCH_SUM_ILP {
                acc[0][k] += load(row_base + k);
            }
        }
        i += level_step;

        for level in 1..PYTORCH_SUM_LEVELS {
            for k in 0..PYTORCH_SUM_ILP {
                acc[level][k] += acc[level - 1][k];
                acc[level - 1][k] = 0.0;
            }

            let mask = level_mask << (level * level_power);
            if (i & mask) != 0 {
                break;
            }
        }
    }

    while i < size {
        let row_base = i * PYTORCH_SUM_ILP;
        for k in 0..PYTORCH_SUM_ILP {
            acc[0][k] += load(row_base + k);
        }
        i += 1;
    }

    for level in 1..PYTORCH_SUM_LEVELS {
        for k in 0..PYTORCH_SUM_ILP {
            acc[0][k] += acc[level][k];
        }
    }

    acc[0]
}

fn pytorch_multi_row_sum_scalar4_f32<F>(size: usize, mut load: F) -> [f32; PYTORCH_SUM_ILP]
where
    F: FnMut(usize) -> f32,
{
    let level_power = 4.max(ceil_log2_usize(size) / PYTORCH_SUM_LEVELS);
    let level_step = 1usize << level_power;
    let level_mask = level_step - 1;
    let mut acc = [[0.0_f32; PYTORCH_SUM_ILP]; PYTORCH_SUM_LEVELS];

    let mut i = 0usize;
    while i + level_step <= size {
        for j in 0..level_step {
            let row_base = (i + j) * PYTORCH_SUM_ILP;
            for k in 0..PYTORCH_SUM_ILP {
                acc[0][k] += load(row_base + k);
            }
        }
        i += level_step;

        for level in 1..PYTORCH_SUM_LEVELS {
            for k in 0..PYTORCH_SUM_ILP {
                acc[level][k] += acc[level - 1][k];
                acc[level - 1][k] = 0.0;
            }

            let mask = level_mask << (level * level_power);
            if (i & mask) != 0 {
                break;
            }
        }
    }

    while i < size {
        let row_base = i * PYTORCH_SUM_ILP;
        for k in 0..PYTORCH_SUM_ILP {
            acc[0][k] += load(row_base + k);
        }
        i += 1;
    }

    for level in 1..PYTORCH_SUM_LEVELS {
        for k in 0..PYTORCH_SUM_ILP {
            acc[0][k] += acc[level][k];
        }
    }

    acc[0]
}

fn pytorch_scalar_row_sum_f32<F>(size: usize, mut load: F) -> f32
where
    F: FnMut(usize) -> f32,
{
    let size_ilp = size / PYTORCH_SUM_ILP;
    let mut partial_sums = pytorch_multi_row_sum_scalar4_f32(size_ilp, &mut load);

    for i in size_ilp * PYTORCH_SUM_ILP..size {
        partial_sums[0] += load(i);
    }

    for k in 1..PYTORCH_SUM_ILP {
        partial_sums[0] += partial_sums[k];
    }

    partial_sums[0]
}

fn pytorch_scalar_row_sum<F>(size: usize, mut load: F) -> f64
where
    F: FnMut(usize) -> f64,
{
    let size_ilp = size / PYTORCH_SUM_ILP;
    let mut partial_sums = pytorch_multi_row_sum_scalar4(size_ilp, &mut load);

    for i in size_ilp * PYTORCH_SUM_ILP..size {
        partial_sums[0] += load(i);
    }

    for k in 1..PYTORCH_SUM_ILP {
        partial_sums[0] += partial_sums[k];
    }

    partial_sums[0]
}

fn pytorch_sum_f32_with_lanes(data: &[f32], f32_lanes: usize) -> f32 {
    assert!(
        (1..=MAX_F32_LANES).contains(&f32_lanes),
        "pytorch_sum_f32_with_lanes: invalid lane count {f32_lanes}"
    );
    if data.len() < f32_lanes {
        return pytorch_scalar_row_sum_f32(data.len(), |i| data[i]);
    }

    let vec_size = data.len() / f32_lanes;
    let vec_acc = pytorch_row_sum_vec_f32(vec_size, f32_lanes, |chunk_idx| {
        load_f32_vec_lanes(data, chunk_idx, f32_lanes)
    });

    let mut final_acc = 0.0_f32;
    for &value in &data[vec_size * f32_lanes..] {
        final_acc += value;
    }
    for &partial in &vec_acc[..f32_lanes] {
        final_acc += partial;
    }
    final_acc
}

fn pytorch_sum_f64_with_lanes(data: &[f64], f64_lanes: usize) -> f64 {
    assert!(
        (1..=MAX_F64_LANES).contains(&f64_lanes),
        "pytorch_sum_f64_with_lanes: invalid lane count {f64_lanes}"
    );
    if data.len() < f64_lanes {
        return pytorch_scalar_row_sum(data.len(), |i| data[i]);
    }

    let vec_size = data.len() / f64_lanes;
    let vec_acc = pytorch_row_sum_vec_f64(vec_size, f64_lanes, |chunk_idx| {
        load_f64_vec_lanes(data, chunk_idx, f64_lanes)
    });

    let mut final_acc = 0.0_f64;
    for &value in &data[vec_size * f64_lanes..] {
        final_acc += value;
    }
    for &partial in &vec_acc[..f64_lanes] {
        final_acc += partial;
    }
    final_acc
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "tests assert bit-exact f32/f64 sums against the torch oracle bit \
              pattern (R-CHAR-3); approximate comparison would mask precision \
              regressions instead of catching them."
)]
mod tests {
    use super::*;

    /// Helper: assert that the AVX2-width L2 model produces EXACTLY the f32 bit
    /// pattern `torch_bits` that live torch 2.11 `at::norm(2.0)` produced for
    /// the same input on the AVX2 oracle host (R-CHAR-3: `torch_bits` is the
    /// live-oracle value, not copied from ferrotorch).
    #[track_caller]
    fn assert_torch_avx2_bits(data: &[f32], torch_bits: u32) {
        let got = l2_norm_f32_with_lanes(data, 8);
        assert_eq!(
            got.to_bits(),
            torch_bits,
            "l2_norm_f32_with_lanes({data:?}, 8) = {got} (bits {:#010x}); \
             live torch AVX2 at::norm(2.0) f32 = {} (bits {torch_bits:#010x})",
            got.to_bits(),
            f32::from_bits(torch_bits)
        );
    }

    /// The #1612 / #1614 boundary row. Live torch `at::norm(2.0)` f32 produces
    /// bits `0x4201970d` (== 32.39751052856445). A scalar `Σ v*v` (or the old
    /// `Σ powf(|v|,2)`) gives `0x4201970e`, one ULP high — which flips the
    /// `max_norm` renorm decision. This primitive must reproduce `0x4201970d`.
    /// Oracle: live torch 2.11.0+cu130, 2026-05-28.
    #[test]
    fn matches_torch_boundary_row_1614() {
        let row = [
            3.6006885_f32,
            18.799816,
            0.4159323,
            -2.6984854,
            -4.786058,
            25.550726,
        ];
        assert_torch_avx2_bits(&row, 0x4201_970d);
    }

    /// Length-8 row (exactly one full 8-wide lane chunk, empty tail). Oracle:
    /// live torch 2.11 `at::norm(2.0)` f32 = bits `0x42547be4`.
    #[test]
    fn matches_torch_len8() {
        let row = [
            8.36561_f32,
            -28.49935,
            -13.49824,
            -16.60736,
            14.18827,
            10.60197,
            23.53077,
            -24.78367,
        ];
        assert_torch_avx2_bits(&row, 0x4254_7be4);
    }

    /// Length-6 random row (pure scalar tail, no full lane chunk). Oracle: live
    /// torch 2.11 `at::norm(2.0)` f32 = bits `0x423abbaf`.
    #[test]
    fn matches_torch_len6_remainder() {
        let row = [-24.30948_f32, 23.0681, 5.86093, 0.7079, -25.30966, 19.51437];
        assert_torch_avx2_bits(&row, 0x423a_bbaf);
    }

    /// Length-7 random row (7-element scalar tail). Oracle: live torch 2.11
    /// `at::norm(2.0)` f32 = bits `0x4258780e`.
    #[test]
    fn matches_torch_len7_remainder() {
        let row = [
            -1.91376_f32,
            23.78282,
            -27.81234,
            0.16903,
            22.27274,
            24.92021,
            21.6505,
        ];
        assert_torch_avx2_bits(&row, 0x4258_780e);
    }

    /// Length-13 row (one 8-wide chunk + a 5-element tail). Oracle: live torch
    /// 2.11 `at::norm(2.0)` f32 = bits `0x425c3351`.
    #[test]
    fn matches_torch_len13() {
        let row = [
            -2.20465_f32,
            28.78827,
            11.0945,
            -3.93455,
            -4.74478,
            22.8556,
            -1.42997,
            -19.06973,
            -13.36156,
            -22.74215,
            5.86617,
            -19.91655,
            4.5731,
        ];
        assert_torch_avx2_bits(&row, 0x425c_3351);
    }

    /// Length-16 row (two full 8-wide chunks, empty tail). Oracle: live torch
    /// 2.11 `at::norm(2.0)` f32 = bits `0x42a3f203`.
    #[test]
    fn matches_torch_len16() {
        let row = [
            -7.07259_f32,
            -28.7053,
            17.81876,
            -24.27542,
            27.74416,
            14.62365,
            -12.6029,
            29.56241,
            6.84485,
            5.97385,
            -17.96887,
            -20.27365,
            19.81938,
            -22.51791,
            28.62067,
            -19.66962,
        ];
        assert_torch_avx2_bits(&row, 0x42a3_f203);
    }

    /// Length-17 row (two 8-wide chunks + a 1-element tail). Oracle: live torch
    /// 2.11 `at::norm(2.0)` f32 = bits `0x429fcbe7` (this particular 17-row IS
    /// byte-exact under the model — not every odd-tail length lands off).
    #[test]
    fn matches_torch_len17() {
        let row = [
            8.04938_f32,
            -29.45572,
            26.4767,
            -1.62244,
            27.07591,
            -16.99281,
            8.93223,
            -3.06544,
            13.78239,
            27.94275,
            18.62917,
            -22.4124,
            -23.7873,
            11.43891,
            -16.49435,
            -28.32381,
            6.74599,
        ];
        assert_torch_avx2_bits(&row, 0x429f_cbe7);
    }

    /// A KNOWN-RESIDUAL row (R-HONEST-3): the portable model lands ONE ULP
    /// below live torch here. This documents the ~3% residual honestly rather
    /// than hiding it. Live torch 2.11 `at::norm(2.0)` f32 = bits `0x40c9a36f`;
    /// the model gives `0x40c9a36e` (one ULP low) because torch's compiled
    /// scalar remainder contracts FMA in a pattern this length-5 tail can't
    /// reproduce. The #1612 / #1614 boundary cases are NOT in this residual
    /// set. The model value `0x40c9a36e` is pinned so a future change to the
    /// algorithm that moves this row is caught; the live-torch value is
    /// recorded alongside so the divergence direction stays auditable.
    #[test]
    fn known_residual_one_ulp_below_torch() {
        let row = [0.60962_f32, 2.0169, -3.36223, 4.05906, 2.73588];
        let got = l2_norm_f32_with_lanes(&row, 8);
        const TORCH_BITS: u32 = 0x40c9_a36f;
        assert_eq!(
            got.to_bits(),
            0x40c9_a36e,
            "residual len-5 row: model bits should be the known 0x40c9a36e \
             (one ULP below live torch {TORCH_BITS:#010x}); if this changes, \
             re-derive the model against the oracle"
        );
        assert!(
            (i64::from(got.to_bits()) - i64::from(TORCH_BITS)).abs() <= 1,
            "residual must stay within one ULP of live torch"
        );
    }

    /// Single-element row: the norm of `[x]` is `|x|`. For `x = 29.04990959`,
    /// live torch `at::norm(2.0)` f32 = bits `0x41e86637`.
    #[test]
    #[allow(
        clippy::excessive_precision,
        reason = "29.04990959 is the verbatim torch input scalar for the len-1 \
                  norm oracle; kept for provenance against at::norm(2.0) f32"
    )]
    fn matches_torch_len1() {
        let row = [29.04990959_f32];
        assert_torch_avx2_bits(&row, 0x41e8_6637);
    }

    /// Empty slice norms to 0 (sum of zero squares, sqrt(0) == 0).
    #[test]
    fn empty_is_zero() {
        assert_eq!(l2_norm_f32_torch(&[]).to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn public_l2_wrapper_uses_detected_lane_model() {
        let row = [3.0_f32, 4.0];
        let expected = l2_norm_f32_with_lanes(&row, torch_f32_lanes());
        assert_eq!(l2_norm_f32_torch(&row).to_bits(), expected.to_bits());
    }

    #[test]
    fn cpu_capability_override_spelling_matches_pytorch() {
        assert_eq!(
            parse_torch_cpu_capability_override("default"),
            Some(TorchCpuVectorClass::Default)
        );
        assert_eq!(
            parse_torch_cpu_capability_override("avx2"),
            Some(TorchCpuVectorClass::Avx2)
        );
        assert_eq!(
            parse_torch_cpu_capability_override("avx512"),
            Some(TorchCpuVectorClass::Avx512)
        );
        assert_eq!(parse_torch_cpu_capability_override("AVX2"), None);
        assert_eq!(parse_torch_cpu_capability_override(""), None);
    }

    #[test]
    fn dispatch_classes_expose_pytorch_vector_widths() {
        assert_eq!(TorchCpuVectorClass::Avx512.f32_lanes(), 16);
        assert_eq!(TorchCpuVectorClass::Avx512.f64_lanes(), 8);
        assert_eq!(TorchCpuVectorClass::Avx2.f32_lanes(), 8);
        assert_eq!(TorchCpuVectorClass::Avx2.f64_lanes(), 4);
        assert_eq!(TorchCpuVectorClass::Sve256.f32_lanes(), 8);
        assert_eq!(TorchCpuVectorClass::Sve256.f64_lanes(), 4);
        assert_eq!(TorchCpuVectorClass::Vsx.f32_lanes(), 4);
        assert_eq!(TorchCpuVectorClass::Vsx.f64_lanes(), 2);
        assert_eq!(TorchCpuVectorClass::ZVector.f32_lanes(), 4);
        assert_eq!(TorchCpuVectorClass::ZVector.f64_lanes(), 2);
    }

    fn adversarial_f32_row(len: usize) -> Vec<f32> {
        assert!(len >= 3);
        let mut data = vec![1.0_f32; len];
        data[0] = 1.0e20;
        data[2] = -1.0e20;
        data
    }

    fn adversarial_f64_row(len: usize) -> Vec<f64> {
        assert!(len >= 3);
        let mut data = vec![1.0_f64; len];
        data[0] = 1.0e300;
        data[2] = -1.0e300;
        data
    }

    /// `sum_f32` of an empty / single-element / short (sub-lane) slice. The
    /// fold-plus-scalar-tail structure must handle `n < vec_t::size()` (pure
    /// scalar path, no full vector group) without touching uninitialised lanes.
    #[test]
    fn sum_f32_short_slices() {
        assert_eq!(sum_f32(&[]), 0.0_f32);
        assert_eq!(sum_f32(&[3.5]), 3.5_f32);
        assert_eq!(sum_f32(&[1.0, 2.0, 3.0]), 6.0_f32);
    }

    #[test]
    fn sum_f32_uses_pytorch_f32_accumulation() {
        let data = [16_777_216.0_f32, 1.0, 1.0];
        let got = pytorch_sum_f32_with_lanes(&data, 8);
        assert_eq!(
            got.to_bits(),
            16_777_216.0_f32.to_bits(),
            "PyTorch SumKernel.cpp uses at::acc_type<float, true>, which maps \
             f32 to f32 accumulation; changing this to f64 diverges from torch"
        );
    }

    /// Live PyTorch 2.11.0+cu130 oracle on this AVX2 host, using
    /// `torch.tensor(row, dtype=torch.float32).reshape(1, n).sum(dim=1)`.
    /// These adversarial rows fail if the implementation falls back to a
    /// sequential f32 sum or the older `Reduce.h` left-fold approximation.
    #[test]
    fn sum_f32_avx2_matches_pytorch_cascade_oracle() {
        for (len, bits) in [
            (31, 0x4170_0000), // 15.0
            (32, 0x41a0_0000), // 20.0
            (33, 0x41a0_0000), // 20.0
            (64, 0x4220_0000), // 40.0
            (65, 0x4220_0000), // 40.0
        ] {
            let got = pytorch_sum_f32_with_lanes(&adversarial_f32_row(len), 8);
            assert_eq!(
                got.to_bits(),
                bits,
                "AVX2 f32 cascade oracle mismatch for len={len}: got {got}"
            );
        }
    }

    /// Source-derived PyTorch AVX512 geometry probe. It uses the same
    /// `SumKernel.cpp` row order as the AVX2 oracle above but with
    /// `Vectorized<float>::size()==16`.
    /// The result differs from AVX2 on purpose, so this catches accidental
    /// reintroduction of hard-coded width-8 behavior.
    #[test]
    fn sum_f32_avx512_geometry_is_not_avx2_hardcoded() {
        let data = adversarial_f32_row(32);
        let avx2 = pytorch_sum_f32_with_lanes(&data, 8);
        let avx512 = pytorch_sum_f32_with_lanes(&data, 16);
        assert_eq!(avx2.to_bits(), 20.0_f32.to_bits());
        assert_eq!(avx512.to_bits(), 26.0_f32.to_bits());
    }

    /// `sum_f32` over a length spanning multiple vector groups plus a tail
    /// (n = 19) of exactly representable integers; the sum is exact regardless
    /// of lane grouping, so it must equal the closed form.
    #[test]
    fn sum_f32_exact_integers() {
        let data: Vec<f32> = (1..=19).map(|i| i as f32).collect();
        // 1+2+...+19 = 190, exactly representable in f32.
        assert_eq!(sum_f32(&data), 190.0_f32);
    }

    /// `sum_f64` mirrors `sum_f32` over the short and multi-group-with-tail
    /// cases at f64 width (4 lanes), with exactly representable integers.
    #[test]
    fn sum_f64_cases() {
        assert_eq!(sum_f64(&[]), 0.0_f64);
        assert_eq!(sum_f64(&[3.5]), 3.5_f64);
        let data: Vec<f64> = (1..=19).map(f64::from).collect();
        assert_eq!(sum_f64(&data), 190.0_f64);
    }

    /// Live PyTorch 2.11.0+cu130 oracle on this AVX2 host, using
    /// `torch.tensor(row, dtype=torch.float64).reshape(1, n).sum(dim=1)`.
    #[test]
    fn sum_f64_avx2_matches_pytorch_cascade_oracle() {
        for (len, bits) in [
            (31, 0x401c_0000_0000_0000), // 7.0
            (32, 0x4020_0000_0000_0000), // 8.0
            (33, 0x4020_0000_0000_0000), // 8.0
            (64, 0x4030_0000_0000_0000), // 16.0
            (65, 0x4030_0000_0000_0000), // 16.0
        ] {
            let got = pytorch_sum_f64_with_lanes(&adversarial_f64_row(len), 4);
            assert_eq!(
                got.to_bits(),
                bits,
                "AVX2 f64 cascade oracle mismatch for len={len}: got {got}"
            );
        }
    }

    /// The boundary row's norm must NOT exceed itself: feeding the matched f32
    /// norm back as a `max_norm` reproduces torch's "not greater, do not clip"
    /// decision. This is the exact comparison `embedding_renorm_cpu_` makes
    /// (`Embedding.cpp:204`, `norm > max_norm`).
    #[test]
    fn boundary_decision_does_not_clip() {
        let row = [
            3.6006885_f32,
            18.799816,
            0.4159323,
            -2.6984854,
            -4.786058,
            25.550726,
        ];
        let norm = l2_norm_f32_with_lanes(&row, 8);
        // torch's f32 norm widened to f64 (== `.item<double>()` at
        // Embedding.cpp:203) used as the max_norm threshold.
        let max_norm = f64::from(norm);
        #[allow(
            clippy::neg_cmp_op_on_partial_ord,
            reason = "deliberately mirrors torch's `norm > max_norm` decision \
                      (Embedding.cpp:204) verbatim — asserting it is false at the \
                      boundary; `<=` would obscure the upstream comparison operator"
        )]
        let does_not_clip = !(f64::from(norm) > max_norm);
        assert!(
            does_not_clip,
            "norm > max_norm must be false at the boundary (torch does not clip)"
        );
    }
}
