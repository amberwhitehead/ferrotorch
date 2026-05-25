//! Divergence-coverage tests for #1239 (fake_quantize_per_channel_affine
//! REQ-2) audit of commit `2a7c5040a`.
//!
//! The commit ships `grad_fns::quantize_grad::fake_quantize_per_channel_affine`
//! at `ferrotorch-core/src/grad_fns/quantize_grad.rs:421` with
//! `FakeQuantizePerChannelBackward` at `:582` and the helper
//! `per_channel_dequantize_f64` at `:308`. The builder's 13 new unit tests
//! at `quantize_grad.rs:1019-1259` cover (a) per-tensor parity per row, (b)
//! axis=0 vs axis=1 dispatch, (c) per-channel STE mask, (d) empty channel
//! dim, and 8 validation-rejection paths. The parity-sweep reports
//! `[fake_quantize_per_channel_affine] 72/72 passed (0 skipped, 0 failed)`.
//!
//! HOWEVER, three concrete behavioral divergences from upstream PyTorch
//! `torch.fake_quantize_per_channel_affine` are unpinned and lie outside
//! the parity-sweep oracle's sample-space, masked by the 9 hand-crafted
//! samples at `tools/parity-sweep/oracle.py:247-331` (each sample uses a
//! safe positive scale; none probe the f32-vs-f64 banker-rounding gap nor
//! a non-Float scale dtype).
//!
//! ## Divergence 1 — f32-vs-f64 inv_scale precision (tracking #1260)
//!
//! Upstream CPU kernel at
//! `/home/doll/pytorch/aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp:2837-2843`:
//!
//! ```cpp
//! cpu_kernel(iter, [=](SelfType self, float scale, int32_t zero_point) -> SelfType {
//!   float inv_scale = 1.0f / scale;
//!   return (std::fmin(std::fmax(static_cast<int64_t>(
//!              zero_point + std::nearbyint(self * inv_scale)),
//!              quant_min), quant_max) - zero_point) * scale;
//! });
//! ```
//!
//! `inv_scale` and `self * inv_scale` are both **float32**. The builder's
//! `per_channel_dequantize_f64` at `quantize_grad.rs:308-329` widens the
//! input to f64 via `x.to_f64()` and computes `inv_scale = 1.0 / scale_f64`
//! in **float64**. For inputs where the f32 product lands exactly on the
//! .5 banker boundary while the f64 product lands a fraction-of-a-ULP
//! past it (or vice versa), `round_ties_even` snaps to a different
//! integer and the dequantized output differs by one scale-step.
//!
//! Concrete probe: `x = 0.35_f32`, `scale = 0.1_f32`, `zp=0`,
//! `quant_min/max = -128/127`. Upstream live torch on 2026-05-25:
//!
//! ```python
//! >>> torch.fake_quantize_per_channel_affine(
//! ...     torch.tensor([[0.35]], dtype=torch.float32),
//! ...     torch.tensor([0.1], dtype=torch.float32),
//! ...     torch.tensor([0], dtype=torch.int32), 0, -128, 127)
//! tensor([[0.4]])
//! ```
//!
//! Numpy reproduction of both paths (verified 2026-05-25):
//! - **Upstream f32 path**: `0.35_f32 = 0x3eb33333 ≈ 0.34999999404f`.
//!   `inv_scale_f32 = 1.0f / 0.1f = 0x41200000 = 10.0f exactly` (since
//!   `0.1f * 10.0f = 1.0f` in round-to-nearest). Product `0.35f * 10.0f`
//!   in f32 rounds to `3.5f` exactly (the round-bit raises). Banker
//!   rounds 3.5 → **4**. Dequant `4 * 0.1f = 0.4f`.
//! - **Ferrotorch f64 path**: `0.35_f32.to_f64() = 0.3499999940395355`,
//!   `scale_f64 = 0.10000000149011612` (the f32-widened literal),
//!   `inv_scale_f64 = 9.99999985098839`, product `=
//!   3.4999998882412924` — strictly less than `3.5` — banker rounds to
//!   **3**. Dequant `3 * 0.1 = 0.3`.
//!
//! (Note: `x=0.05_f32` is the inverse case: in f32 the product is `0.5`
//! exactly (banker → 0), but the ferrotorch f64-widening cascade also
//! produces `0.5` exactly (the rounding errors cancel) so this specific
//! boundary happens to MATCH upstream — see the control test
//! `divergence_per_channel_f32_vs_f64_banker_rounding_control_passes` below.
//! The divergence is real but boundary-sensitive; 0.35 exposes it.)
//!
//! ## Divergence 2 — scale=0 / scale<0 rejection (tracking #1261)
//!
//! Upstream `FakeQuantPerChannelAffine.cpp:32-77` performs NO `scale > 0`
//! check. The kernel proceeds with `inv_scale = 1.0f / 0.0f = +Inf` (per
//! channel where scale=0); `self * +Inf` is `+Inf | -Inf | NaN`, cast to
//! `int64_t` saturates to `INT64_MIN`, clamps to `quant_min`, dequant is
//! `(quant_min - zp) * 0.0 = 0.0` (or `-0.0` for negative
//! `quant_min - zp`).
//!
//! Live torch on 2026-05-25:
//!
//! ```python
//! >>> torch.fake_quantize_per_channel_affine(
//! ...     torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
//! ...     torch.tensor([0.0, 1.0], dtype=torch.float32),
//! ...     torch.tensor([0, 0], dtype=torch.int32), 0, -128, 127)
//! tensor([[-0., -0.],
//!         [ 3.,  4.]])
//! ```
//!
//! Ferrotorch's `quantize_grad.rs:511-523` rejects with
//! `FerrotorchError::InvalidArgument: "scale must be > 0 per channel"`.
//! This is a strict-superset divergence: PyTorch returns a tensor; ferrotorch
//! returns an `Err`. The design doc at
//! `.design/ferrotorch-core/grad_fns/quantize_grad.md:418-422` acknowledges
//! the divergence as a "ferrotorch superset of upstream" but R-DEV-1 says
//! numerical-contract divergences MUST match upstream byte-for-byte.
//!
//! ## Divergence 3 — f64 scale acceptance (tracking #1262)
//!
//! Upstream `FakeQuantPerChannelAffine.cpp:51-52`:
//!
//! ```cpp
//! TORCH_CHECK(scale.scalar_type() == ScalarType::Float
//!     || scale.scalar_type() == at::kBFloat16,
//!     "Scale must be Float or BFloat16, found ", scale.scalar_type());
//! ```
//!
//! Live torch with `f64` scale rejects with
//! `"Scale must be Float or BFloat16, found Double"`. Ferrotorch's
//! `fake_quantize_per_channel_affine<T: Float>` accepts ANY `T:
//! num_traits::Float` — including `f64` — and silently produces a
//! result. This is a vocabulary divergence (R-DEV-2 API-shape match):
//! a strict superset of upstream's accepted dtypes, but a real divergence
//! from PyTorch's user-visible contract.
//!
//! Per R-CHAR-3 every expected value below traces to either the live
//! `torch.fake_quantize_per_channel_affine` output cited above or to the
//! upstream cpp formula at `QuantizedOpKernels.cpp:2837-2843`, never to
//! ferrotorch's own output.

use ferrotorch_core::{from_vec, grad_fns, IntTensor};

/// Control test for Divergence 1: at `x=0.05_f32, scale=0.1_f32` the
/// f32 vs f64 paths happen to produce the SAME banker-rounding result
/// (both 0). This is the rounding-error-cancellation case — included so
/// future regressions in the f32 path don't accidentally show up as a
/// "newly passing" test on this boundary.
///
/// Per R-CHAR-3: expected value 0.0 sourced from live torch oracle
/// 2026-05-25 (`tensor([[0.]])`), NOT from ferrotorch's output.
#[test]
fn divergence_per_channel_f32_vs_f64_banker_rounding_control_passes() {
    // Live oracle 2026-05-25:
    //   torch.fake_quantize_per_channel_affine(
    //       torch.tensor([[0.05]], dtype=torch.float32),
    //       torch.tensor([0.1], dtype=torch.float32),
    //       torch.tensor([0], dtype=torch.int32), 0, -128, 127)
    //   -> tensor([[0.]])  bits 0x00000000
    let expected: f32 = 0.0;

    let input = from_vec(vec![0.05f32], &[1, 1]).unwrap();
    let scale = from_vec(vec![0.1f32], &[1]).unwrap();
    let zp = IntTensor::<i64>::from_vec(vec![0], vec![1]).unwrap();

    let out = grad_fns::quantize_grad::fake_quantize_per_channel_affine(
        &input, &scale, &zp, 0, -128, 127,
    )
    .expect("fake_quantize_per_channel_affine should not error on valid inputs");
    let actual = out.data().unwrap();
    let a = actual[0];
    assert_eq!(
        a.to_bits(),
        expected.to_bits(),
        "control: x=0.05f32, scale=0.1f32 — both f32 and f64 paths happen to \
         banker-round to 0 here; torch returns 0.0 (per live oracle 2026-05-25), \
         ferrotorch returns {a}."
    );
}

/// Divergence 1 — f32-vs-f64 inv_scale precision divergence (REAL).
///
/// `x=0.35f32`, `scale=0.1f32` exposes the precision gap:
/// - upstream f32 math: product is exactly `3.5f`, banker → 4, dequant 0.4
/// - ferrotorch f64 math: product is `3.4999998882...`, banker → 3,
///   dequant 0.3
///
/// Live torch confirmation 2026-05-25:
/// ```python
/// >>> torch.fake_quantize_per_channel_affine(
/// ...     torch.tensor([[0.35]], dtype=torch.float32),
/// ...     torch.tensor([0.1], dtype=torch.float32),
/// ...     torch.tensor([0], dtype=torch.int32), 0, -128, 127)
/// tensor([[0.4]])
/// ```
///
/// The f32 vs f64 cast-ordering is set by the upstream CPU kernel at
/// `/home/doll/pytorch/aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp:2838`:
/// `float inv_scale = 1.0f / scale;` — `inv_scale` is f32 throughout the
/// kernel. R-DEV-1 numerical-contract divergence.
///
/// Tracking: #1260
#[test]
fn divergence_per_channel_f32_vs_f64_banker_rounding() {
    // R-CHAR-3: expected value 0.4 sourced from the LIVE torch oracle
    // 2026-05-25 (`tensor([[0.4]])`), NOT from ferrotorch's output. Also
    // independently derivable from upstream's f32 math at
    // QuantizedOpKernels.cpp:2837-2843: 4 * 0.1f = 0.4f.
    let expected_dq_target: f32 = 0.4;

    let input = from_vec(vec![0.35f32], &[1, 1]).unwrap();
    let scale = from_vec(vec![0.1f32], &[1]).unwrap();
    let zp = IntTensor::<i64>::from_vec(vec![0], vec![1]).unwrap();

    let out = grad_fns::quantize_grad::fake_quantize_per_channel_affine(
        &input, &scale, &zp, 0, -128, 127,
    )
    .expect("fake_quantize_per_channel_affine should not error on valid inputs");
    let actual = out.data().unwrap();
    let a = actual[0];
    // Standard parity-sweep tolerance (rtol=1e-5, atol=1e-7). The actual
    // divergence is `|0.4 - 0.3| = 0.1`, two orders of magnitude beyond
    // the bound, so any reasonable tolerance fails.
    let diff = (a - expected_dq_target).abs();
    let bound = 1e-7_f32 + 1e-5_f32 * expected_dq_target.abs();
    assert!(
        diff <= bound,
        "f32-vs-f64 banker divergence: torch returns {expected_dq_target} \
         for x=0.35f32, scale=0.1f32 (live oracle 2026-05-25 + upstream \
         f32-math QuantizedOpKernels.cpp:2837-2843), ferrotorch returns \
         {a} (diff={diff} > bound={bound}). The builder's \
         per_channel_dequantize_f64 at quantize_grad.rs:316 uses \
         `1.0 / scale_f64` where upstream uses `1.0f / scale`. \
         Tracking #1260."
    );
}

/// Divergence 2a — `scale=0` per-channel must NOT error; upstream
/// silently proceeds.
///
/// Upstream `FakeQuantPerChannelAffine.cpp:32-77` performs NO `scale > 0`
/// check. With scale=0.0 the kernel computes `1.0f/0.0f = +Inf`, the
/// per-element loop produces `int64_t(+Inf) = INT64_MIN` (the x86 cvttsd2si
/// invalid-operation result), `std::fmax(INT64_MIN, quant_min) = quant_min`,
/// dequant = `(quant_min - zp) * 0.0 = 0.0` (or `-0.0` for negative
/// `quant_min - zp`).
///
/// Live torch on 2026-05-25:
/// ```python
/// >>> torch.fake_quantize_per_channel_affine(
/// ...     torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
/// ...     torch.tensor([0.0, 1.0], dtype=torch.float32),
/// ...     torch.tensor([0, 0], dtype=torch.int32), 0, -128, 127)
/// tensor([[-0., -0.],
///         [ 3.,  4.]])
/// ```
///
/// Ferrotorch's `quantize_grad.rs:511-523` rejects with
/// `FerrotorchError::InvalidArgument: "scale must be > 0 per channel"`.
///
/// Tracking: #1261
#[test]
fn divergence_per_channel_scale_zero_silently_proceeds() {
    // Expected (live torch oracle 2026-05-25): tensor([[-0., -0.], [3., 4.]]).
    // R-CHAR-3 honored: derived from running torch directly, NOT from
    // ferrotorch's behavior.
    let expected: [f32; 4] = [-0.0, -0.0, 3.0, 4.0];

    let input = from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let scale = from_vec(vec![0.0f32, 1.0], &[2]).unwrap();
    let zp = IntTensor::<i64>::from_vec(vec![0, 0], vec![2]).unwrap();

    let out_result =
        grad_fns::quantize_grad::fake_quantize_per_channel_affine(&input, &scale, &zp, 0, -128, 127);

    // Upstream returns an OK result; ferrotorch must mirror that, not Err.
    let out = out_result.expect(
        "upstream torch returns tensor([[-0., -0.], [3., 4.]]) for scale=[0., 1.] \
         (no scale>0 check at FakeQuantPerChannelAffine.cpp:32-77); ferrotorch \
         currently rejects with 'scale must be > 0 per channel'. Tracking #1261.",
    );
    let actual = out.data().unwrap();
    assert_eq!(actual.len(), 4);
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if i < 2 {
            // bit-exact compare for -0.0 from channel 0.
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "scale=0 channel 0 element {i}: torch returns -0.0 \
                 (bits 0x80000000), ferrotorch returns {a} (bits 0x{:08x}). \
                 Tracking #1261.",
                a.to_bits()
            );
        } else {
            assert!(
                (a - e).abs() < 1e-6,
                "scale=0 channel 1 element {i}: torch returns {e}, \
                 ferrotorch returns {a}. Tracking #1261."
            );
        }
    }
}

/// Divergence 2b — `scale<0` per-channel must NOT error; upstream
/// silently proceeds with the negative-scale math.
///
/// Live torch on 2026-05-25:
/// ```python
/// >>> torch.fake_quantize_per_channel_affine(
/// ...     torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
/// ...     torch.tensor([-0.1, 1.0], dtype=torch.float32),
/// ...     torch.tensor([0, 0], dtype=torch.int32), 0, -128, 127)
/// tensor([[1., 2.],
///         [3., 4.]])
/// ```
///
/// With scale=-0.1, `inv_scale=-10`, `1.0 * -10 = -10`, `int64_t(-10)=-10`,
/// clamps to [-128,127] passes, dequant `(-10 - 0) * -0.1 = 1.0`. The
/// result is *consistent* with the negative scale (the double-negation
/// cancels).
///
/// Ferrotorch's `quantize_grad.rs:511-523` rejects with
/// `FerrotorchError::InvalidArgument: "scale must be > 0 per channel"`.
///
/// Tracking: #1261
#[test]
fn divergence_per_channel_scale_negative_silently_proceeds() {
    // Expected from LIVE torch oracle 2026-05-25.
    let expected: [f32; 4] = [1.0, 2.0, 3.0, 4.0];

    let input = from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let scale = from_vec(vec![-0.1f32, 1.0], &[2]).unwrap();
    let zp = IntTensor::<i64>::from_vec(vec![0, 0], vec![2]).unwrap();

    let out_result =
        grad_fns::quantize_grad::fake_quantize_per_channel_affine(&input, &scale, &zp, 0, -128, 127);

    let out = out_result.expect(
        "upstream torch returns tensor([[1., 2.], [3., 4.]]) for scale=[-0.1, 1.0]; \
         ferrotorch rejects with 'scale must be > 0 per channel' \
         at quantize_grad.rs:515. Tracking #1261.",
    );
    let actual = out.data().unwrap();
    assert_eq!(actual.len(), 4);
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-6,
            "scale<0 element {i}: torch returns {e}, ferrotorch returns {a}. \
             Tracking #1261."
        );
    }
}

/// Divergence 3 — `scale` dtype `f64` must be REJECTED by ferrotorch
/// (vocabulary-level R-DEV-2 divergence).
///
/// Upstream `FakeQuantPerChannelAffine.cpp:51-52`:
/// ```cpp
/// TORCH_CHECK(scale.scalar_type() == ScalarType::Float
///     || scale.scalar_type() == at::kBFloat16,
///     "Scale must be Float or BFloat16, found ", scale.scalar_type());
/// ```
///
/// Live torch with `f64` scale rejects 2026-05-25:
/// ```python
/// >>> torch.fake_quantize_per_channel_affine(
/// ...     torch.tensor([[1.0]], dtype=torch.float32),
/// ...     torch.tensor([1.0], dtype=torch.float64),
/// ...     torch.tensor([0], dtype=torch.int32), 0, -128, 127)
/// RuntimeError: Scale must be Float or BFloat16, found Double
/// ```
///
/// Ferrotorch's generic `T: Float` admits any `num_traits::Float`,
/// including `f64`, and silently produces a result.
///
/// Tracking: #1262
#[test]
fn divergence_per_channel_f64_scale_silently_accepted() {
    // Contract: with `Tensor<f64>` input + `Tensor<f64>` scale, ferrotorch
    // MUST return Err whose message indicates the unsupported scale dtype,
    // mirroring upstream's TORCH_CHECK at
    // FakeQuantPerChannelAffine.cpp:51-52. R-CHAR-3: expected behavior
    // sourced from upstream's TORCH_CHECK message verified live 2026-05-25.
    let input = from_vec(vec![1.0f64], &[1, 1]).unwrap();
    let scale = from_vec(vec![1.0f64], &[1]).unwrap();
    let zp = IntTensor::<i64>::from_vec(vec![0], vec![1]).unwrap();

    let result =
        grad_fns::quantize_grad::fake_quantize_per_channel_affine(&input, &scale, &zp, 0, -128, 127);

    // Expected: Err containing "Scale must be Float or BFloat16" or
    // equivalent dtype-rejection message.
    assert!(
        result.is_err(),
        "f64 scale must be rejected per upstream \
         FakeQuantPerChannelAffine.cpp:51-52 'Scale must be Float or \
         BFloat16, found Double' (verified live 2026-05-25). Ferrotorch's \
         T: Float accepts f64 silently. Tracking #1262."
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("Float") || err_msg.contains("BFloat16") || err_msg.contains("dtype"),
        "rejection message should mention the dtype constraint per upstream \
         FakeQuantPerChannelAffine.cpp:51-52; got '{err_msg}'. Tracking #1262."
    );
}
