//! Audit CORE-107 (crosslink #1801): `ComplexTensor::abs` must not overflow
//! or underflow for representable magnitudes.
//!
//! At HEAD, `abs` computes `(re*re + im*im).sqrt()`: squaring overflows to
//! `inf` for finite magnitudes (re ≈ 1e200 in f64, ≈ 1e20 in f32),
//! flushes subnormal components to zero, and poisons `(inf, nan)` inputs to
//! `nan` (torch: `inf` — the magnitude of an infinite component is infinite
//! regardless of the other component, C99 hypot semantics).
//!
//! Live PyTorch oracle (torch 2.11.0+cu130) — R-ORACLE-1 path (b):
//!
//! ```text
//! >>> t = torch.tensor([complex(1e200, 1e200), complex(1e-308, 1e-308),
//! ...                   complex(3e-320, 4e-320), complex(float('inf'), float('nan')),
//! ...                   complex(float('nan'), 1.0), complex(0.0, 0.0),
//! ...                   complex(-3.0, 4.0)], dtype=torch.complex128)
//! >>> torch.abs(t).tolist()
//! [1.414213562373095e+200, 1.414213562373095e-308, 5e-320, inf, nan, 0.0, 5.0]
//! >>> t32 = torch.tensor([complex(1e20, 1e20), complex(1e-30, 1e-30),
//! ...                     complex(float('inf'), float('nan')),
//! ...                     complex(float('nan'), 2.0)], dtype=torch.complex64)
//! >>> torch.abs(t32).tolist()
//! [1.414213581995256e+20, 1.4142135555081815e-30, inf, nan]
//! ```
//!
//! All expectations are asserted bit-exactly: the decimal literals above are
//! Python shortest-round-trip reprs (they identify the IEEE values uniquely),
//! and a standalone Rust `hypot` cross-check reproduced every one of them
//! bit-for-bit (glibc libm on this runner; e.g. f64 `hypot(1e200,1e200)`
//! bits `0x697d8f9811335b57`, f32 `hypot(1e20,1e20)` bits `0x60f553b3`).
//! No tolerance is needed (R-ORACLE-5: exact equality is the justification).
//!
//! `ComplexTensor` is CPU-only by type design (SoA `Arc<Vec<T>>` buffers, no
//! device residency) — there is no GPU lane for this type.

use ferrotorch_core::complex_tensor::ComplexTensor;

fn abs64(re: &[f64], im: &[f64]) -> Vec<f64> {
    ComplexTensor::<f64>::from_re_im(re.to_vec(), im.to_vec(), vec![re.len()])
        .unwrap()
        .abs()
        .unwrap()
        .data()
        .unwrap()
        .to_vec()
}

fn abs32(re: &[f32], im: &[f32]) -> Vec<f32> {
    ComplexTensor::<f32>::from_re_im(re.to_vec(), im.to_vec(), vec![re.len()])
        .unwrap()
        .abs()
        .unwrap()
        .data()
        .unwrap()
        .to_vec()
}

// ── f64 ─────────────────────────────────────────────────────────────────────

#[test]
fn abs_f64_near_max_does_not_overflow() {
    // torch: |1e200 + 1e200j| = 1.414213562373095e+200 (finite).
    // HEAD: re*re = 1e400 = inf -> sqrt(inf) = inf.
    let r = abs64(&[1e200], &[1e200]);
    assert!(r[0].is_finite(), "magnitude must be finite, got {}", r[0]);
    assert_eq!(r[0].to_bits(), 1.414213562373095e200_f64.to_bits());
}

#[test]
fn abs_f64_tiny_does_not_underflow() {
    // torch: |1e-308 + 1e-308j| = 1.414213562373095e-308.
    // HEAD: re*re = 1e-616 -> 0.0 -> sqrt(0) = 0.
    let r = abs64(&[1e-308], &[1e-308]);
    assert_eq!(r[0].to_bits(), 1.414213562373095e-308_f64.to_bits());
}

#[test]
fn abs_f64_subnormal_3_4_5() {
    // torch: |3e-320 + 4e-320j| = 5e-320 exactly (subnormal 3-4-5 triangle).
    let r = abs64(&[3e-320], &[4e-320]);
    assert_eq!(r[0].to_bits(), 5e-320_f64.to_bits());
}

#[test]
fn abs_f64_inf_with_nan_component_is_inf() {
    // torch: |inf + nanj| = inf (C99 hypot: an infinite component dominates).
    // HEAD: inf*inf + nan*nan = nan -> sqrt(nan) = nan.
    let r = abs64(&[f64::INFINITY], &[f64::NAN]);
    assert_eq!(r[0], f64::INFINITY);
    // Symmetric case: |nan + infj| = inf.
    let r = abs64(&[f64::NAN], &[f64::INFINITY]);
    assert_eq!(r[0], f64::INFINITY);
}

#[test]
fn abs_f64_nan_with_finite_component_is_nan() {
    // torch: |nan + 1j| = nan.
    let r = abs64(&[f64::NAN], &[1.0]);
    assert!(r[0].is_nan(), "expected nan, got {}", r[0]);
}

#[test]
fn abs_f64_zero_and_pythagorean() {
    let r = abs64(&[0.0, -3.0], &[0.0, 4.0]);
    assert_eq!(r[0].to_bits(), 0.0_f64.to_bits());
    assert_eq!(r[1].to_bits(), 5.0_f64.to_bits());
}

// ── f32 ─────────────────────────────────────────────────────────────────────

#[test]
fn abs_f32_near_max_does_not_overflow() {
    // torch: |1e20 + 1e20j| (complex64) = 1.414213581995256e+20
    // (= f32 bits 0x60f553b3). HEAD: (1e20)^2 = 1e40 > f32::MAX -> inf.
    let r = abs32(&[1e20], &[1e20]);
    assert!(r[0].is_finite(), "magnitude must be finite, got {}", r[0]);
    assert_eq!(r[0].to_bits(), 0x60f5_53b3_u32);
}

#[test]
// The literal is torch's shortest-round-trip f64 repr of the expected f32,
// quoted verbatim from the oracle (header) — keep it bit-faithful rather
// than clippy's truncation (probe bits: 0x0de57822).
#[allow(clippy::excessive_precision)]
fn abs_f32_tiny_does_not_underflow() {
    // torch: |1e-30 + 1e-30j| = 1.4142135555081815e-30.
    // HEAD: (1e-30)^2 = 1e-60 -> 0.0f32 -> 0.
    let r = abs32(&[1e-30], &[1e-30]);
    assert_eq!(r[0].to_bits(), 1.4142135555081815e-30_f32.to_bits());
}

#[test]
fn abs_f32_inf_with_nan_component_is_inf() {
    // torch: |inf + nanj| = inf.
    let r = abs32(&[f32::INFINITY], &[f32::NAN]);
    assert_eq!(r[0], f32::INFINITY);
}

#[test]
fn abs_f32_nan_with_finite_component_is_nan() {
    // torch: |nan + 2j| = nan.
    let r = abs32(&[f32::NAN], &[2.0]);
    assert!(r[0].is_nan(), "expected nan, got {}", r[0]);
}
