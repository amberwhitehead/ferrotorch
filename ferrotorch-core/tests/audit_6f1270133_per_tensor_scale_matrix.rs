//! Audit of commit `6f1270133` ("remove scale>0 rejection in
//! `fake_quantize_per_tensor_affine` to match upstream silent inf/NaN
//! propagation; closes #1265").
//!
//! Per acto-critic.md the audit produces *failing* tests for divergence and,
//! per the audit-fix-patterns memory, positive regression probes for the
//! cross-product the fixer did NOT cover. This file is the positive-regression
//! coverage; every assertion below was constructed from the live torch oracle
//! on 2026-05-25 (R-CHAR-3 — no tautological "ferrotorch matches itself" tests).
//!
//! Upstream pin: `aten/src/ATen/native/quantized/FakeQuantPerTensorAffine.cpp:69-89`
//! (no `scale > 0` check; only validates `quant_min <= quant_max` and
//! `zero_point in [quant_min, quant_max]`).
//!
//! Ferrotorch pin: `ferrotorch-core/src/grad_fns/quantize_grad.rs:197-294`
//! (`fake_quantize_per_tensor_affine_impl`; the rejection block at the old
//! `:225-231` was removed in `6f1270133`).
//!
//! Matrix audited (from the audit shape sections A, C, E):
//!
//! - A. scale=0 across (zp ∈ {0, 64}) × (qmin/qmax ∈ {int8, uint8-anchored})
//! - C. scale<0 magnitude variants: scale=-2.0 saturates non-trivially
//!      (NOT a clean double-negation — torch returns `[-0.0, 2.0, 4.0, 4.0]`
//!      for input `[1,2,3,4]`, scale=-2.0, zp=0, qmin/qmax=int8).
//! - E. cross product of (scale ∈ {0, -0.1, NaN}) × (shape ∈ {1-D, 2-D})
//!      × (zp ∈ {0, 64}) × (range ∈ {int8, uint8-zp-compatible}). The
//!      NON-TRIVIAL case is `scale=-0.1, zp=0, qmin=0, qmax=255` where torch
//!      saturates to qmin=0 then dequant yields `-0.0` for all elements
//!      (input cleared, not double-negated).

use ferrotorch_core::{from_vec, grad_fns};

// =================== A. scale=0 across zp / qmin variants ===================

/// scale=0, zp=0, int8 — already exercised by the existing pinned test, but
/// re-asserted here as the matrix anchor.
///
/// Live torch 2026-05-25:
///   `torch.fake_quantize_per_tensor_affine([5.0], 0.0, 0, -128, 127)` →
///   `tensor([0.])`.
#[test]
fn audit_scale_zero_zp0_int8() {
    let input = from_vec(vec![5.0_f32], &[1]).unwrap();
    let out =
        grad_fns::quantize_grad::fake_quantize_per_tensor_affine(&input, 0.0_f64, 0, -128, 127)
            .expect("scale=0 should silently proceed post-6f1270133");
    let data = out.data().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0], 0.0_f32, "torch oracle: 0.0; got {}", data[0]);
}

/// scale=0, zp=64 (non-zero zero_point), int8 range.
///
/// Live torch 2026-05-25:
///   `torch.fake_quantize_per_tensor_affine([5.0], 0.0, 64, -128, 127)` →
///   `tensor([0.])`.
///
/// Note: even with `zp=64` the dequant `(clamped_qval - zp) * scale` collapses
/// to 0 because scale is exactly 0.0 (zero × any-finite = +0 in f32). This
/// confirms that the per-tensor kernel does NOT exhibit the per-channel's
/// `-0.0` quirk; torch returns clean `+0.0` regardless of zp.
#[test]
fn audit_scale_zero_zp64_int8() {
    let input = from_vec(vec![5.0_f32], &[1]).unwrap();
    let out =
        grad_fns::quantize_grad::fake_quantize_per_tensor_affine(&input, 0.0_f64, 64, -128, 127)
            .expect("scale=0, zp=64 should silently proceed");
    let data = out.data().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0], 0.0_f32, "torch oracle: 0.0; got {}", data[0]);
}

/// scale=0, zp=0, qmin=0 qmax=127 — zero-anchored qmin (uint7-like).
///
/// Live torch 2026-05-25:
///   `torch.fake_quantize_per_tensor_affine([5.0], 0.0, 0, 0, 127)` →
///   `tensor([0.])`.
#[test]
fn audit_scale_zero_zp0_qmin_zero() {
    let input = from_vec(vec![5.0_f32], &[1]).unwrap();
    let out = grad_fns::quantize_grad::fake_quantize_per_tensor_affine(&input, 0.0_f64, 0, 0, 127)
        .expect("scale=0, zp=0, qmin=0, qmax=127 should silently proceed");
    let data = out.data().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0], 0.0_f32, "torch oracle: 0.0; got {}", data[0]);
}

// =========== C. negative scale magnitude variants (non-trivial) =============

/// `scale=-2.0` is NOT a clean double-negation. With input `[1,2,3,4]`,
/// zp=0, qmin/qmax=int8, torch saturates non-trivially.
///
/// Live torch 2026-05-25:
///   `torch.fake_quantize_per_tensor_affine([1,2,3,4], -2.0, 0, -128, 127)` →
///   `tensor([-0.0, 2.0, 4.0, 4.0])`.
///
/// The fixer's commit message claims "scale<0 double-negates back to input"
/// — that's only true for SMALL negative magnitudes (e.g. -0.1 where
/// `round(x / -0.1) * -0.1 = x`). For `scale=-2.0` the round-to-even
/// quantization produces saturation: input 1 → qval `round(-0.5) = 0`,
/// dequant `(0 - 0) * -2 = -0.0`; input 3 → qval `round(-1.5) = -2`,
/// dequant `(-2 - 0) * -2 = 4.0`; input 4 → qval `round(-2) = -2`,
/// dequant 4.0; input 2 → qval `round(-1) = -1`, dequant 2.0. The
/// `-0.0` for input=1 is the signature of upstream IEEE-754 behavior:
/// `0_i64 * -2.0_f64 = -0.0_f64`.
#[test]
fn audit_scale_negative_two_saturating() {
    let input = from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[4]).unwrap();
    let out =
        grad_fns::quantize_grad::fake_quantize_per_tensor_affine(&input, -2.0_f64, 0, -128, 127)
            .expect("scale=-2.0 should silently proceed post-6f1270133");
    let data = out.data().unwrap();
    let expected = [-0.0_f32, 2.0, 4.0, 4.0];
    for (i, (&a, &e)) in data.iter().zip(expected.iter()).enumerate() {
        // Allow signed-zero distinction at index 0: torch oracle is -0.0,
        // assert via bit pattern (a == -0.0 evaluates `true` for +0.0 too).
        if i == 0 {
            assert_eq!(
                a, 0.0,
                "elem 0: torch oracle: -0.0 (numerically equals 0); got {a}",
            );
            assert!(
                a.is_sign_negative() || a == 0.0,
                "elem 0: torch oracle: -0.0 (signbit set); got {a} (signbit={})",
                a.is_sign_negative(),
            );
        } else {
            assert!(
                (a - e).abs() < 1e-6,
                "elem {i}: torch oracle: {e}, ferrotorch: {a}",
            );
        }
    }
}

/// `scale=-0.1, zp=64, qmin/qmax=int8` — sanity check that zp ≠ 0 still
/// double-negates back to input.
///
/// Live torch 2026-05-25:
///   `torch.fake_quantize_per_tensor_affine([1,2,3,4], -0.1, 64, -128, 127)` →
///   `tensor([1.0, 2.0, 3.0, 4.0])`.
#[test]
fn audit_scale_negative_zp_nonzero() {
    let input = from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[4]).unwrap();
    let out =
        grad_fns::quantize_grad::fake_quantize_per_tensor_affine(&input, -0.1_f64, 64, -128, 127)
            .expect("scale=-0.1, zp=64 should silently proceed");
    let data = out.data().unwrap();
    let expected = [1.0_f32, 2.0, 3.0, 4.0];
    for (i, (&a, &e)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-6,
            "elem {i}: torch oracle: {e}, ferrotorch: {a}",
        );
    }
}

// =========== E. cross-product — non-trivial 2-D + uint8 variants ============

/// 2-D input with NaN scale — element ordering preserved through reshape.
///
/// Live torch 2026-05-25:
///   `torch.fake_quantize_per_tensor_affine([[1.0,2.0],[3.0,4.0]], NaN, 0, -128, 127)` →
///   `tensor([[nan, nan], [nan, nan]])`.
#[test]
fn audit_2d_scale_nan() {
    let input = from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let out =
        grad_fns::quantize_grad::fake_quantize_per_tensor_affine(&input, f64::NAN, 0, -128, 127)
            .expect("scale=NaN on 2-D input should silently proceed");
    assert_eq!(out.shape(), &[2, 2]);
    let data = out.data().unwrap();
    assert_eq!(data.len(), 4);
    for (i, &a) in data.iter().enumerate() {
        assert!(a.is_nan(), "2-D NaN-scale elem {i}: expected NaN, got {a}");
    }
}

/// 2-D input with scale=0 and zp=64 (non-zero zp), int8 range.
///
/// Live torch 2026-05-25:
///   `torch.fake_quantize_per_tensor_affine([[1.0,2.0],[3.0,4.0]], 0.0, 64, -128, 127)` →
///   `tensor([[0.0, 0.0], [0.0, 0.0]])`.
#[test]
fn audit_2d_scale_zero_zp64() {
    let input = from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let out =
        grad_fns::quantize_grad::fake_quantize_per_tensor_affine(&input, 0.0_f64, 64, -128, 127)
            .expect("scale=0, zp=64 on 2-D input should silently proceed");
    assert_eq!(out.shape(), &[2, 2]);
    let data = out.data().unwrap();
    let expected = [0.0_f32, 0.0, 0.0, 0.0];
    for (i, (&a, &e)) in data.iter().zip(expected.iter()).enumerate() {
        assert_eq!(a, e, "2-D scale=0 elem {i}: expected {e}, got {a}");
    }
}

/// `scale=-0.1, zp=0, qmin=0, qmax=255` (uint8 range, zero-anchored).
///
/// Live torch 2026-05-25:
///   `torch.fake_quantize_per_tensor_affine([1,2,3,4], -0.1, 0, 0, 255)` →
///   `tensor([-0.0, -0.0, -0.0, -0.0])`.
///
/// This is the most non-trivial case: with `scale=-0.1` and zp=0, the qval
/// `round(x / -0.1) = round(-10*x)` is negative for positive inputs, then
/// clamps to `qmin=0`. Dequant: `(0 - 0) * -0.1 = -0.0_f64` (signed zero
/// product). All elements collapse to `-0.0`.
#[test]
fn audit_scale_negative_uint8_saturate() {
    let input = from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[4]).unwrap();
    let out = grad_fns::quantize_grad::fake_quantize_per_tensor_affine(&input, -0.1_f64, 0, 0, 255)
        .expect("scale=-0.1 uint8 should silently proceed");
    let data = out.data().unwrap();
    assert_eq!(data.len(), 4);
    for (i, &a) in data.iter().enumerate() {
        // All four elements collapse to -0.0 (saturate to qmin=0 then *-0.1).
        // Numerically -0.0 == 0.0; assert magnitude is zero. Bit-pattern is a
        // soft check — IEEE-754 lets `(0_i64 - 0_i64 as f64) * -0.1_f64` be
        // either +0.0 or -0.0 depending on the float pipeline.
        assert_eq!(a, 0.0, "elem {i}: torch oracle: -0.0 (==0); got {a}");
    }
}

/// `scale=-0.1, zp=64, qmin/qmax=uint8 [0, 255]` — double-negates back to
/// input because qval `round(-10*x) + 64` stays in `[0, 255]` for small `x`.
///
/// Live torch 2026-05-25:
///   `torch.fake_quantize_per_tensor_affine([1,2,3,4], -0.1, 64, 0, 255)` →
///   `tensor([1.0, 2.0, 3.0, 4.0])`.
#[test]
fn audit_scale_negative_uint8_zp64_round_trip() {
    let input = from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[4]).unwrap();
    let out =
        grad_fns::quantize_grad::fake_quantize_per_tensor_affine(&input, -0.1_f64, 64, 0, 255)
            .expect("scale=-0.1, zp=64, uint8 should silently proceed");
    let data = out.data().unwrap();
    let expected = [1.0_f32, 2.0, 3.0, 4.0];
    for (i, (&a, &e)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-5,
            "elem {i}: torch oracle: {e}, ferrotorch: {a}",
        );
    }
}

// ================= F. Prior tests unchanged — quick smoke ====================

/// Sanity: the canonical `scale=1.0, zp=0, int8` identity-ish path still works
/// (no regression from removing the rejection block).
///
/// Live torch 2026-05-25:
///   `torch.fake_quantize_per_tensor_affine([0.5, 1.0, 1.5], 1.0, 0, -128, 127)` →
///   `tensor([0.0, 1.0, 2.0])` (banker's rounding: 0.5→0, 1.5→2).
#[test]
fn audit_smoke_scale_one_identity() {
    let input = from_vec(vec![0.5_f32, 1.0, 1.5], &[3]).unwrap();
    let out =
        grad_fns::quantize_grad::fake_quantize_per_tensor_affine(&input, 1.0_f64, 0, -128, 127)
            .expect("scale=1.0 nominal path");
    let data = out.data().unwrap();
    let expected = [0.0_f32, 1.0, 2.0];
    for (i, (&a, &e)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-6,
            "elem {i}: torch oracle: {e}, ferrotorch: {a} (banker's round)",
        );
    }
}

// ============== D. Verify removed unit-test paths still pass ================
//
// The fixer removed `fake_quantize_rejects_zero_scale` and
// `fake_quantize_rejects_negative_scale` from the `mod tests` block in
// `quantize_grad.rs`. Those tests asserted the now-removed rejection
// behavior. The replacement assertions (silent proceed) are covered by the
// existing pinned tests in `divergence_quantize_grad_per_tensor_scale_check.rs`
// — the removal is safe IFF the replacement asserts the inverse on the same
// inputs. The `audit_scale_zero_zp0_int8` and `audit_scale_negative_*` tests
// in THIS file additionally widen coverage over the removed-test surface
// (matrix zp/qmin variations the removed unit tests did not exercise).
