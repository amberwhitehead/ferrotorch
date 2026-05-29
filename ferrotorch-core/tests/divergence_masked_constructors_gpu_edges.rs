//! Discriminator audit (#1545 / #1534): edge-case coverage for the on-device
//! `masked_invalid` / `masked_equal` predicate kernels added in commit
//! `0dbcb4a5a` (`ferrotorch-gpu/src/masked_kernels.rs`) and wired through
//! `ferrotorch-core/src/masked.rs`.
//!
//! The author's tests (`conformance_masked.rs::gpu::*` and the in-lib
//! `masked_kernels::tests::*`) cover:
//!   - isfinite_mask f32/f64 for {normal, NaN, +inf, -inf}
//!   - ne_scalar_mask f32 for value=5.0 over {equal, unequal} (NO NaN)
//!   - ne_scalar_mask f64 for value=5.0 over {equal, NaN, unequal}
//!
//! GAPS this file pins (each asserted against an IEEE / numpy-style reference
//! constructed from `f64::is_finite()` and the CPU `v != value` walk at
//! `ferrotorch-core/src/masked.rs:527`, NOT copied from the GPU side — see
//! R-CHAR-3):
//!
//!   (A) ne_scalar_mask **f32** NaN case. The builder claims the
//!       `setp.ne -> setp.neu` fix matters for BOTH widths, but only the f64
//!       NaN case has a test. `NaN != value` must be TRUE (mask=valid=true)
//!       under the CPU walk; an ordered `setp.ne.f32` regression would give 0.
//!       Mirrors `numpy.ma.masked_equal` (NaN is never == value, so never
//!       masked) and the ferrotorch torch-convention mask = `v != value`.
//!
//!   (B) ne_scalar_mask with value = +inf / -inf / 0.0, including ±0 vs 0.0
//!       (IEEE: -0.0 == 0.0, so `-0.0 != 0.0` is FALSE -> masked out) and
//!       inf == inf (so `+inf != +inf` is FALSE -> masked out). Untested.
//!
//!   (C) isfinite_mask subnormal + ±0 (TensorCompare.cpp:484-486:
//!       `(v==v) && (|v| != inf)` -> subnormals and ±0 are finite -> valid).
//!       The author's f32 test omits subnormal and ±0.
//!
//! All assertions use the consumer path (`ferrotorch_core::masked::*`) on a
//! genuinely CUDA-resident input, and additionally assert the data tensor
//! stays `is_cuda()` after the op (no value round trip; R-CODE-4).
//!
//! Reference math:
//!   - isfinite: `aten/src/ATen/native/TensorCompare.cpp:484-486`
//!     `(self == self) * (self.abs() != infinity)`.
//!   - masked_equal valid mask: `ferrotorch-core/src/masked.rs:527`
//!     `mask: Vec<bool> = data_vec.iter().map(|&v| v != value).collect()`.
//!     numpy.ma.masked_equal masks where `x == value`; ferrotorch torch
//!     convention stores the complement (valid = `v != value`).

#![cfg(feature = "gpu")]

use ferrotorch_core::masked::{masked_equal, masked_invalid};
use ferrotorch_core::{Device, Tensor, TensorStorage};
use std::sync::Once;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the GPU masked-edge audit");
    });
}

fn cpu_f32(data: &[f32]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
        .expect("cpu f32 tensor")
}

fn cpu_f64(data: &[f64]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
        .expect("cpu f64 tensor")
}

fn to_cuda_f32(t: Tensor<f32>) -> Tensor<f32> {
    t.to(Device::Cuda(0)).expect("upload f32 to cuda")
}

fn to_cuda_f64(t: Tensor<f64>) -> Tensor<f64> {
    t.to(Device::Cuda(0)).expect("upload f64 to cuda")
}

// ── (A) ne_scalar_mask f32 NaN: the untested half of the setp.neu fix ───────

/// Divergence guard: `ferrotorch_core::masked_equal::<f32>` on CUDA must mirror
/// the CPU walk `v != value` (`ferrotorch-core/src/masked.rs:527`) for a NaN
/// element. IEEE / numpy.ma.masked_equal: `NaN != value` is TRUE, so NaN is
/// never masked (mask = valid = true). An ordered `setp.ne.f32` regression in
/// `masked_kernels.rs::ne_scalar_ptx` would return 0 (NaN ordered-compare
/// false) and mask the NaN — diverging from the CPU/numpy reference.
#[test]
fn gpu_masked_equal_f32_nan_is_valid() {
    ensure_cuda_backend();
    let host = [5.0_f32, f32::NAN, 5.0, 2.0, f32::NAN];
    let value = 5.0_f32;
    // Named reference: the CPU walk at masked.rs:527.
    let expected: Vec<bool> = host.iter().map(|&v| v != value).collect();
    assert_eq!(
        expected,
        vec![false, true, false, true, true],
        "reference: NaN != 5.0 is true (valid); equal is masked"
    );

    let gpu_in = to_cuda_f32(cpu_f32(&host));
    assert!(gpu_in.is_cuda(), "input must be CUDA-resident");
    let mt = masked_equal(gpu_in, value).expect("gpu masked_equal f32");
    assert!(mt.data().is_cuda(), "data must stay on device (R-CODE-4)");
    assert_eq!(
        mt.mask(),
        expected.as_slice(),
        "f32 NaN ne_scalar: GPU mask must match CPU `v != value` walk"
    );
}

// ── (B) ne_scalar_mask special values: ±inf, ±0 ─────────────────────────────

/// `masked_equal(x, +inf)`: IEEE `+inf == +inf` is true, so `+inf != +inf` is
/// FALSE (masked out); `-inf != +inf` is TRUE (valid); finite != +inf is TRUE.
/// Mirrors the CPU walk at masked.rs:527.
#[test]
fn gpu_masked_equal_f32_value_pos_inf() {
    ensure_cuda_backend();
    let host = [f32::INFINITY, f32::NEG_INFINITY, 1.0, f32::NAN];
    let value = f32::INFINITY;
    let expected: Vec<bool> = host.iter().map(|&v| v != value).collect();
    assert_eq!(
        expected,
        vec![false, true, true, true],
        "reference: +inf==+inf masked; -inf, finite, NaN all != +inf -> valid"
    );

    let gpu_in = to_cuda_f32(cpu_f32(&host));
    let mt = masked_equal(gpu_in, value).expect("gpu masked_equal f32 +inf");
    assert!(mt.data().is_cuda());
    assert_eq!(mt.mask(), expected.as_slice());
}

/// `masked_equal(x, 0.0)`: IEEE `-0.0 == 0.0` is true, so both ±0 are masked
/// out (`!= 0.0` is FALSE). A subnormal is `!= 0.0` -> valid. Mirrors the CPU
/// walk at masked.rs:527 (Rust `-0.0_f64 != 0.0` is false).
#[test]
fn gpu_masked_equal_f64_value_zero_signed_zero_and_subnormal() {
    ensure_cuda_backend();
    let subnormal = f64::from_bits(1); // smallest positive subnormal
    let host = [0.0_f64, -0.0_f64, subnormal, 1.0];
    let value = 0.0_f64;
    let expected: Vec<bool> = host.iter().map(|&v| v != value).collect();
    assert_eq!(
        expected,
        vec![false, false, true, true],
        "reference: +0 and -0 both == 0.0 (masked); subnormal and 1.0 valid"
    );

    let gpu_in = to_cuda_f64(cpu_f64(&host));
    let mt = masked_equal(gpu_in, value).expect("gpu masked_equal f64 zero");
    assert!(mt.data().is_cuda());
    assert_eq!(
        mt.mask(),
        expected.as_slice(),
        "f64 ne_scalar value=0.0: -0.0 must compare equal to 0.0 (masked)"
    );
}

// ── (C) isfinite_mask subnormal + ±0 ────────────────────────────────────────

/// `masked_invalid` on CUDA must mark subnormals and ±0 as finite (valid),
/// matching `at::isfinite` (`TensorCompare.cpp:484-486`: `(v==v)&&(|v|!=inf)`).
/// `f64::is_finite()` is the named reference.
#[test]
fn gpu_masked_invalid_f64_subnormal_and_signed_zero_are_finite() {
    ensure_cuda_backend();
    let subnormal = f64::from_bits(1);
    let host = [
        subnormal,
        0.0_f64,
        -0.0_f64,
        f64::MIN_POSITIVE,
        f64::NAN,
        f64::INFINITY,
    ];
    // Named reference: IEEE is_finite, matching TensorCompare.cpp:484-486.
    let expected: Vec<bool> = host.iter().map(|v| v.is_finite()).collect();
    assert_eq!(
        expected,
        vec![true, true, true, true, false, false],
        "reference: subnormal/±0/min-positive finite; NaN/inf not"
    );

    let gpu_in = to_cuda_f64(cpu_f64(&host));
    assert!(gpu_in.is_cuda());
    let mt = masked_invalid(gpu_in).expect("gpu masked_invalid f64");
    assert!(mt.data().is_cuda(), "data must stay on device (R-CODE-4)");
    assert_eq!(
        mt.mask(),
        expected.as_slice(),
        "isfinite GPU mask must match IEEE is_finite for subnormal/±0"
    );
}

/// Same for f32: subnormal + ±0 finite, -inf not.
#[test]
fn gpu_masked_invalid_f32_subnormal_and_signed_zero_are_finite() {
    ensure_cuda_backend();
    let subnormal = f32::from_bits(1);
    let host = [
        subnormal,
        -0.0_f32,
        f32::MIN_POSITIVE,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let expected: Vec<bool> = host.iter().map(|v| v.is_finite()).collect();
    assert_eq!(
        expected,
        vec![true, true, true, false, false],
        "reference: subnormal/-0/min-positive finite; -inf/NaN not"
    );

    let gpu_in = to_cuda_f32(cpu_f32(&host));
    let mt = masked_invalid(gpu_in).expect("gpu masked_invalid f32");
    assert!(mt.data().is_cuda());
    assert_eq!(mt.mask(), expected.as_slice());
}
