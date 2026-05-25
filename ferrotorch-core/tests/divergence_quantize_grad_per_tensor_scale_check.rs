//! Divergence: `fake_quantize_per_tensor_affine` in ferrotorch ALSO rejects
//! `scale<=0` / `scale==NaN` (at `quantize_grad.rs:225-231`), but upstream
//! `aten/src/ATen/native/quantized/FakeQuantPerTensorAffine.cpp:69-89`
//! performs no such check. The same fix applied to per-channel by
//! commit 36b245151 (closes #1261) must also apply to per-tensor; the
//! fixer's claim "Per-tensor has no analogous `scale > 0` check" is
//! true of UPSTREAM but FALSE of ferrotorch's `fake_quantize_per_tensor_affine`.
//!
//! Upstream source pin (no scale check):
//! `/home/doll/pytorch/aten/src/ATen/native/quantized/FakeQuantPerTensorAffine.cpp:69-89`
//!   ```cpp
//!   std::tuple<Tensor, Tensor> fake_quantize_per_tensor_affine_cachemask(
//!       const Tensor& self, double scale, int64_t zero_point,
//!       int64_t quant_min, int64_t quant_max) {
//!     TORCH_CHECK(quant_min <= quant_max, ...);
//!     TORCH_CHECK(zero_point >= quant_min && zero_point <= quant_max, ...);
//!     // NO `scale > 0` check.
//!     fake_quant_tensor_cachemask_stub(... scale ...);
//!   }
//!   ```
//!
//! Ferrotorch source pin (REJECTS scale<=0):
//! `/home/doll/ferrotorch/ferrotorch-core/src/grad_fns/quantize_grad.rs:225-231`
//!   ```rust
//!   // 3. scale > 0 (ferrotorch superset of upstream — upstream would
//!   //    silently produce inf/NaN; we surface it as a clear error).
//!   if scale.is_nan() || scale <= 0.0 {
//!       return Err(FerrotorchError::InvalidArgument {
//!           message: format!("fake_quantize_per_tensor_affine: `scale` must be > 0, got {scale}"),
//!       });
//!   }
//!   ```
//!
//! Per R-CHAR-3 every expected value below comes from live torch oracle
//! 2026-05-25, NOT from ferrotorch's output.

use ferrotorch_core::{from_vec, grad_fns};

/// Divergence: per-tensor `scale=0` rejected by ferrotorch but upstream
/// silently proceeds and returns +0.0.
///
/// Live torch 2026-05-25:
/// ```python
/// >>> torch.fake_quantize_per_tensor_affine(torch.tensor([5.0]), 0.0, 0, -128, 127)
/// tensor([0.])    # signbit False (positive zero, unlike per-channel)
/// ```
///
/// Note: per-tensor returns +0.0 (signbit=False) while per-channel
/// returns -0.0 (signbit=True). Distinct upstream codepaths; both
/// are valid IEEE-754 zeros consistent with their respective f32/f64
/// reduction patterns.
///
/// Tracking: #1265
#[test]
fn divergence_per_tensor_scale_zero_silently_proceeds() {
    let input = from_vec(vec![5.0_f32], &[1]).unwrap();
    let result =
        grad_fns::quantize_grad::fake_quantize_per_tensor_affine(&input, 0.0_f64, 0, -128, 127);
    let out = result.expect(
        "upstream torch.fake_quantize_per_tensor_affine(5.0, scale=0.0, zp=0, -128, 127) \
         returns tensor([0.]) (live oracle 2026-05-25); ferrotorch rejects with \
         `scale must be > 0` at quantize_grad.rs:227. Parallel divergence to #1261; \
         per-tensor scope. Tracking #1265.",
    );
    let actual = out.data().unwrap();
    assert_eq!(actual.len(), 1);
    // Torch returns +0.0; allow either sign of zero (the f32-tensor-args path
    // returned +0.0 in our probe; per-tensor cast-first may differ from
    // per-channel's -0.0). The KEY divergence is that ferrotorch errored at all.
    assert_eq!(
        actual[0], 0.0,
        "expected 0.0 (torch live oracle), got {}",
        actual[0]
    );
}

/// Divergence: per-tensor `scale<0` rejected by ferrotorch but upstream
/// silently proceeds via the same double-negation cancellation as per-channel.
///
/// Live torch 2026-05-25:
/// ```python
/// >>> torch.fake_quantize_per_tensor_affine(torch.tensor([1.0, 2.0, 3.0, 4.0]), -0.1, 0, -128, 127)
/// tensor([1., 2., 3., 4.])
/// ```
///
/// Tracking: #1265
#[test]
fn divergence_per_tensor_scale_negative_silently_proceeds() {
    let input = from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[4]).unwrap();
    let result =
        grad_fns::quantize_grad::fake_quantize_per_tensor_affine(&input, -0.1_f64, 0, -128, 127);
    let out = result.expect(
        "upstream torch.fake_quantize_per_tensor_affine(..., scale=-0.1, ...) \
         returns tensor([1., 2., 3., 4.]) (live oracle 2026-05-25); ferrotorch \
         rejects at quantize_grad.rs:227. Parallel divergence to #1261; per-tensor scope. \
         Tracking #1265.",
    );
    let actual = out.data().unwrap();
    let expected = [1.0_f32, 2.0, 3.0, 4.0];
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-6,
            "per-tensor scale<0 element {i}: torch returns {e}, ferrotorch returns {a}.",
        );
    }
}

/// Divergence: per-tensor `scale=NaN` rejected by ferrotorch but upstream
/// silently propagates the NaN.
///
/// Live torch 2026-05-25:
/// ```python
/// >>> torch.fake_quantize_per_tensor_affine(torch.tensor([1.0, 2.0]), float('nan'), 0, -128, 127)
/// tensor([nan, nan])
/// ```
///
/// Tracking: #1265
#[test]
fn divergence_per_tensor_scale_nan_silently_propagates() {
    let input = from_vec(vec![1.0_f32, 2.0], &[2]).unwrap();
    let result =
        grad_fns::quantize_grad::fake_quantize_per_tensor_affine(&input, f64::NAN, 0, -128, 127);
    let out = result.expect(
        "upstream torch.fake_quantize_per_tensor_affine(..., scale=NaN, ...) \
         returns tensor([nan, nan]) (live oracle 2026-05-25); ferrotorch rejects \
         with `scale must be > 0, got NaN`. Parallel divergence to #1261. \
         Tracking #1265.",
    );
    let actual = out.data().unwrap();
    assert!(
        actual[0].is_nan(),
        "per-tensor NaN scale element 0: expected NaN, got {}",
        actual[0]
    );
    assert!(
        actual[1].is_nan(),
        "per-tensor NaN scale element 1: expected NaN, got {}",
        actual[1]
    );
}
