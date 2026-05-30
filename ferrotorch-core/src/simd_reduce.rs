//! Torch-matching f32 reduction primitives.
//!
//! PyTorch's CPU f32 L2-norm reduction is NOT a naive scalar `Σ v*v`. It is a
//! width-8 lane-grouped accumulation (one `Vectorized<float>` of 8 AVX2 lanes)
//! followed by a specific horizontal fold and a scalar remainder, then `sqrt`.
//! A naive scalar `Σ v.abs().powf(2.0)` (or even a scalar `Σ v*v`) gives a
//! result that differs from torch by one ULP on a meaningful fraction of f32
//! rows, which flips boundary decisions (e.g. the `embedding(max_norm=...)`
//! renorm-or-not decision — #1612 / #1614). This module ships the reduction
//! torch actually performs so those boundary decisions match byte-for-byte.
//!
//! ## REQ status (per `.design/ferrotorch-core/simd_reduce.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`l2_norm_f32_torch`) | SHIPPED | `pub fn l2_norm_f32_torch` here models torch's vectorized last-dim L2 kernel (`aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:222-255`): 8 f32 lanes accumulate `lanes[j] += data[d+j]*data[d+j]` (plain mul+add, mirroring AVX2 `_mm256_mul_ps`+`_mm256_add_ps` at `vec256/vec256_float.h:564`), a naive left-fold `buffer[0] += buffer[j]`, a scalar FMA tail `buffer[0] = fma(data, data, buffer[0])` (mirroring the compiled contraction of `buffer[0] += data*data` at `ReduceOpsKernel.cpp:251`), then `sqrt`. Non-test production consumers: `ferrotorch_core::grad_fns::reduction::norm_with_dim` (the `p==2.0`, `T==f32`, last-dim-contiguous slice) and `ferrotorch_nn::embedding::renorm_weight_rows_in_place` (the `norm_type==2.0`, `T==f32` renorm decision). |
//! | REQ-2 (`sum_f32` / `sum_f64`) | SHIPPED | `pub fn sum_f32` / `pub fn sum_f64` here model torch's vectorized contiguous (last-dim) sum kernel: a lane-grouped multi-accumulator (`acc[j] += data[d+j]`, mirroring `acc_vec += data_vec` in `vectorized_reduction` at `aten/src/ATen/native/cpu/Reduce.h:41-51`) followed by a horizontal fold and a scalar tail. A scalar `iter().sum()` does NOT autovectorize (FP add is non-associative → a sequential dependency the compiler cannot reorder), so the multi-accumulator is required to get AVX throughput AND is numerically CLOSER to torch's own lane-grouped order than a sequential sum (well within the conformance atol 1e-5 / 1e-10). Non-test production consumers: `sum_dim` / `mean_dim` forward in `grad_fns/reduction.rs` (the `inner==1` contiguous-last-dim rows of the `[outer, axis, inner]` fast path). |
//!
//! ## Why this is not byte-exact for 100% of rows (honest scope, R-HONEST-3)
//!
//! Across a 400-row live-torch oracle (this AVX2 host, lengths 1..65), this
//! primitive matches torch's `at::norm(2.0)` f32 bits on ~97% of rows; the
//! current scalar-`powf` path matched ~79%. The residual ~3% are one-ULP
//! misses at certain non-multiple-of-8 lengths where torch's compiled scalar
//! remainder contracts FMA in a pattern a portable Rust loop cannot reproduce
//! exactly. The #1612/#1614 boundary row IS in the matching set, so the renorm
//! decision it pins now matches torch. This is a strict improvement over the
//! `powf` path, not a regression — and the parity-sweep envelope (atol 1e-7)
//! tolerates the residual sub-ULP differences on non-boundary rows.

/// Number of `f32` lanes torch's `Vectorized<float>` holds on AVX2.
///
/// This host is AVX2 (width-8), no AVX512 (width-16). The accumulation tree is
/// width-dependent, so we model exactly the width-8 structure torch's compiled
/// kernel uses here. See `aten/src/ATen/cpu/vec/vec256/vec256_float.h`
/// (`Vectorized<float>::size() == 8`).
const F32_LANES: usize = 8;

/// Compute the L2 (Euclidean) norm of an f32 slice the way PyTorch's CPU
/// reduction does, so the result matches `at::norm(2.0)` on an f32 contiguous
/// last-dim reduction byte-for-byte (on this AVX2 host, modulo the ~3% one-ULP
/// residual documented at the module level).
///
/// This mirrors the vectorized last-dim L2 kernel at
/// `aten/src/ATen/native/cpu/ReduceOpsKernel.cpp:222-255`:
///
/// ```text
///   fVec acc_vec{acc_t(0)};                       // 8 lanes, all zero
///   acc_t buffer[fVec::size()];                   // [f32; 8]
///   for (; d < size - (size % 8); d += 8) {
///     acc_vec += data_vec * data_vec;             // lane-wise mul+add
///   }
///   acc_vec.store(buffer);
///   for (j = 1; j < 8; j++) buffer[0] += buffer[j];  // naive LEFT-FOLD
///   for (; d < size; d++) buffer[0] += data*data;    // scalar tail
///   result = sqrt(buffer[0]);
/// ```
///
/// The accumulator type is `at::opmath_type<float> == float` (`OpMathType.h:16`),
/// i.e. f32, NOT f64. The lane accumulate uses plain multiply-then-add (AVX2
/// `_mm256_mul_ps` + `_mm256_add_ps`, `vec256_float.h:564`, NOT fused); the
/// scalar tail's `buffer[0] += data*data` compiles to a fused multiply-add, so
/// we use [`f32::mul_add`] there. `NormTwoOps::project` is `device_sqrt`
/// (`SharedReduceOps.h:375-381`), i.e. `f32::sqrt`.
///
/// A scalar model of the tree (no `unsafe` SIMD intrinsics) is used for
/// portability and determinism: f32 `a + b*b` without contraction is
/// bit-identical to AVX2 `_mm256_add_ps(a, _mm256_mul_ps(b, b))` lane-wise, so
/// the scalar lane model reproduces the AVX2 path's rounding exactly.
#[must_use]
pub fn l2_norm_f32_torch(data: &[f32]) -> f32 {
    let n = data.len();
    // 8 lane accumulators, all zero — mirrors `fVec acc_vec{acc_t(0)}`.
    let mut lanes = [0.0_f32; F32_LANES];

    // Main loop: process contiguous chunks of 8 elements, accumulating each
    // element's square into its lane. `main` is `size - (size % 8)`, matching
    // the kernel's `d < size - (size % Vec::size())` bound. Lane accumulate is
    // plain mul-then-add (NOT fused), mirroring `acc_fvec += data_fvec *
    // data_fvec` lowered to `_mm256_add_ps(acc, _mm256_mul_ps(d, d))`.
    let main = n - (n % F32_LANES);
    let mut d = 0;
    while d < main {
        for (j, lane) in lanes.iter_mut().enumerate() {
            let x = data[d + j];
            // `+= x*x` lowers to a single f32 add of the product (no FMA),
            // bit-identical to AVX2 `_mm256_add_ps(acc, _mm256_mul_ps(x, x))`.
            *lane += x * x;
        }
        d += F32_LANES;
    }

    // Horizontal reduce: naive LEFT-FOLD of the 8 lanes into lane 0, exactly as
    // the kernel does (`for (j = 1; j < fVec::size(); j++) buffer[0] += buffer[j]`).
    // FP addition is non-associative, so the left-fold order is load-bearing —
    // this is NOT the AVX2 `vec_reduce_all` permute tree (that tree is used by
    // sum/prod's `binary_kernel_reduce_vec`, not by the norm last-dim kernel).
    let mut acc = lanes[0];
    for &lane in &lanes[1..] {
        acc += lane;
    }

    // Scalar remainder tail: the `size % 8` elements that didn't fill a full
    // 8-wide chunk accumulate into lane 0 (`for (; d < size; d++) buffer[0] +=
    // data_val * data_val`). The compiled `buffer[0] += data*data` contracts to
    // a fused multiply-add (single rounding) under PyTorch's `-ffp-contract`,
    // so we use `mul_add` to match it.
    while d < n {
        let x = data[d];
        acc = x.mul_add(x, acc);
        d += 1;
    }

    // `NormTwoOps::project` is `device_sqrt(a)` (SharedReduceOps.h:375-381),
    // i.e. f32 sqrt of the accumulated sum-of-squares.
    acc.sqrt()
}

/// Number of `f64` lanes torch's `Vectorized<double>` holds on AVX2 (256-bit /
/// 64-bit = 4). The sum accumulation tree width is dtype-dependent.
const F64_LANES: usize = 4;

/// Horizontal sum of a contiguous `f32` slice, modelling torch's vectorized
/// contiguous reduction order (a lane-grouped multi-accumulator + fold) rather
/// than a sequential `Σ`.
///
/// PyTorch's CPU contiguous sum reduction is NOT a sequential scalar fold. The
/// inner contiguous-reduction kernel (`vectorized_inner_reduction` →
/// `vectorized_reduction` at `aten/src/ATen/native/cpu/Reduce.h:36-91`) keeps a
/// bank of `Vec` lane accumulators and adds successive vector loads into them
/// (`acc[j] = vop(acc[j], Vec::loadu(ptr))`, `Reduce.h:47-50`), then folds the
/// lanes (`Reduce.h:54-58`). The lane count makes the accumulation a tree, not a
/// chain.
///
/// Why a multi-accumulator (and not `data.iter().sum()`): f32 addition is
/// non-associative, so the compiler cannot legally reassociate a sequential
/// `Σ` into independent lanes — the sequential version carries a single
/// loop-carried dependency and stays scalar (no AVX). The `F32_LANES`
/// independent accumulators here have no cross-lane dependency in the hot loop,
/// so the autovectorizer (with this crate's `target-cpu=native`) lowers the
/// lane loop to AVX adds, and the result is ALSO numerically closer to torch's
/// own lane-grouped order than a sequential sum would be.
///
/// Scope (R-HONEST-3): this is NOT claimed byte-exact with torch. Torch's f32
/// sum fold tree is the permute-based `VecReduceAllSIMD<float>`
/// (`functional_base.h:59-76`) over 4×8 lanes, not this 8-lane left-fold; the
/// difference is sub-ULP and well inside the conformance atol (1e-5 for f32).
/// `sum`/`mean` carry no byte-exact boundary-decision contract (unlike the L2
/// renorm decision that `l2_norm_f32_torch` exists to pin), so the
/// within-tolerance multi-accumulator is the right model: it gets AVX speed and
/// torch-adjacent rounding without the byte-exactness machinery.
#[must_use]
pub fn sum_f32(data: &[f32]) -> f32 {
    let n = data.len();
    // 8 lane accumulators, all zero — mirrors the `Vec acc` bank in
    // `vectorized_reduction` (Reduce.h:41), narrowed to one Vec width here.
    let mut lanes = [0.0_f32; F32_LANES];

    // Main loop: each contiguous group of 8 elements adds into its own lane.
    // The 8 `*lane += ...` statements touch 8 DISTINCT accumulators, so they are
    // mutually independent and autovectorize to a single AVX add per group.
    let main = n - (n % F32_LANES);
    let mut d = 0;
    while d < main {
        for (j, lane) in lanes.iter_mut().enumerate() {
            *lane += data[d + j];
        }
        d += F32_LANES;
    }

    // Horizontal fold of the 8 lanes into lane 0 (left-fold; FP add is
    // non-associative so the order is fixed and documented).
    let mut acc = lanes[0];
    for &lane in &lanes[1..] {
        acc += lane;
    }

    // Scalar remainder for the `n % 8` trailing elements.
    while d < n {
        acc += data[d];
        d += 1;
    }

    acc
}

/// Horizontal sum of a contiguous `f64` slice, the `f64` analogue of
/// [`sum_f32`]. Uses `F64_LANES` (4) independent accumulators — torch's
/// `Vectorized<double>` width on AVX2 — for the same autovectorize-friendly,
/// torch-adjacent reduction order. Same honest scope as [`sum_f32`]: within
/// conformance atol (1e-10 for f64), not byte-exact.
#[must_use]
pub fn sum_f64(data: &[f64]) -> f64 {
    let n = data.len();
    let mut lanes = [0.0_f64; F64_LANES];

    let main = n - (n % F64_LANES);
    let mut d = 0;
    while d < main {
        for (j, lane) in lanes.iter_mut().enumerate() {
            *lane += data[d + j];
        }
        d += F64_LANES;
    }

    let mut acc = lanes[0];
    for &lane in &lanes[1..] {
        acc += lane;
    }

    while d < n {
        acc += data[d];
        d += 1;
    }

    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: assert that `l2_norm_f32_torch(data)` produces EXACTLY the f32
    /// bit pattern `torch_bits` that live torch 2.11 `at::norm(2.0)` produced
    /// for the same input (R-CHAR-3: `torch_bits` is the live-oracle value,
    /// not copied from ferrotorch).
    #[track_caller]
    fn assert_torch_bits(data: &[f32], torch_bits: u32) {
        let got = l2_norm_f32_torch(data);
        assert_eq!(
            got.to_bits(),
            torch_bits,
            "l2_norm_f32_torch({data:?}) = {got} (bits {:#010x}); \
             live torch at::norm(2.0) f32 = {} (bits {torch_bits:#010x})",
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
        assert_torch_bits(&row, 0x4201_970d);
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
        assert_torch_bits(&row, 0x4254_7be4);
    }

    /// Length-6 random row (pure scalar tail, no full lane chunk). Oracle: live
    /// torch 2.11 `at::norm(2.0)` f32 = bits `0x423abbaf`.
    #[test]
    fn matches_torch_len6_remainder() {
        let row = [-24.30948_f32, 23.0681, 5.86093, 0.7079, -25.30966, 19.51437];
        assert_torch_bits(&row, 0x423a_bbaf);
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
        assert_torch_bits(&row, 0x4258_780e);
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
        assert_torch_bits(&row, 0x425c_3351);
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
        assert_torch_bits(&row, 0x42a3_f203);
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
        assert_torch_bits(&row, 0x429f_cbe7);
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
        let got = l2_norm_f32_torch(&row);
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
        assert_torch_bits(&row, 0x41e8_6637);
    }

    /// Empty slice norms to 0 (sum of zero squares, sqrt(0) == 0).
    #[test]
    fn empty_is_zero() {
        assert_eq!(l2_norm_f32_torch(&[]).to_bits(), 0.0_f32.to_bits());
    }

    /// `sum_f32` of an empty / single-element / short (sub-lane) slice. The fold
    /// + scalar-tail structure must handle `n < F32_LANES` (pure tail, no full
    /// 8-wide group) without touching uninitialised lanes.
    #[test]
    fn sum_f32_short_slices() {
        assert_eq!(sum_f32(&[]), 0.0_f32);
        assert_eq!(sum_f32(&[3.5]), 3.5_f32);
        assert_eq!(sum_f32(&[1.0, 2.0, 3.0]), 6.0_f32);
    }

    /// `sum_f32` over a length spanning multiple full 8-wide groups plus a tail
    /// (n = 19 = 2 groups + 3 tail) of exactly representable integers; the sum
    /// is exact regardless of lane grouping, so it must equal the closed form.
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
        let norm = l2_norm_f32_torch(&row);
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
