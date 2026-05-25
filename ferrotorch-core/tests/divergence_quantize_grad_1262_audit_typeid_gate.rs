//! Audit probes for commit `68f29f183` — TypeId-based dtype gate for
//! `fake_quantize_per_channel_affine` claimed to reject `f64` and `f16`
//! scale per upstream `FakeQuantPerChannelAffine.cpp:51-52`.
//!
//! Each probe traces its expected value to live `torch.fake_quantize_per_channel_affine`
//! verified 2026-05-25 (R-CHAR-3). The probes here are POSITIVE coverage
//! around the new behavior — none should fail if the commit's claims hold.
//! A failure here is a divergence the fixer missed.
//!
//! Upstream cite, exact text from
//! `/home/doll/pytorch/aten/src/ATen/native/quantized/FakeQuantPerChannelAffine.cpp:51-52`:
//!
//! ```cpp
//! TORCH_CHECK(scale.scalar_type() == ScalarType::Float || scale.scalar_type() == at::kBFloat16,
//!             "Scale must be Float or BFloat16, found ", scale.scalar_type());
//! ```
//!
//! Live torch error messages observed 2026-05-25:
//! - f64 scale  -> `'Scale must be Float or BFloat16, found Double'`
//! - f16 scale  -> `'Scale must be Float or BFloat16, found Half'`
//! - bf16 scale -> passes the scalar_type check; later TensorIterator
//!   `needs_dynamic_casting` internal assert fires for the input/scale
//!   dtype combinations exercised by typical Python callers. The scalar
//!   type *gate* (which is what #1262 fixes) admits bf16.

use ferrotorch_core::{from_vec, grad_fns, IntTensor};

/// Audit A — exact substring of upstream's error message for f64 scale.
///
/// Upstream torch live 2026-05-25:
/// ```python
/// >>> torch.fake_quantize_per_channel_affine(
/// ...     torch.tensor([[1.0]], dtype=torch.float32),
/// ...     torch.tensor([1.0], dtype=torch.float64),
/// ...     torch.tensor([0], dtype=torch.int32), 0, -128, 127)
/// RuntimeError: Scale must be Float or BFloat16, found Double
/// ```
///
/// The commit message promises ferrotorch returns the same phrasing.
/// This test pins the EXACT substring `"Scale must be Float or BFloat16, found Double"`,
/// not just "Float or BFloat16". If ferrotorch's phrasing drifts from
/// upstream's, this fails — pin commit 68f29f183.
#[test]
fn audit_per_channel_f64_error_message_exact_phrasing() {
    let input = from_vec(vec![1.0f64], &[1, 1]).unwrap();
    let scale = from_vec(vec![1.0f64], &[1]).unwrap();
    let zp = IntTensor::<i64>::from_vec(vec![0], vec![1]).unwrap();
    let result = grad_fns::quantize_grad::fake_quantize_per_channel_affine(
        &input, &scale, &zp, 0, -128, 127,
    );
    assert!(result.is_err(), "expected f64 rejection per #1262");
    let msg = format!("{}", result.unwrap_err());
    // Expected upstream substring sourced from live `torch` error string.
    // R-CHAR-3: this constant is the upstream-emitted phrasing, not a
    // ferrotorch literal copy.
    let expected_upstream_substring = "Scale must be Float or BFloat16, found Double";
    assert!(
        msg.contains(expected_upstream_substring),
        "ferrotorch error message does not contain upstream phrasing.\n\
         expected substring: {expected_upstream_substring:?}\n\
         got: {msg:?}\n\
         upstream cite: FakeQuantPerChannelAffine.cpp:51-52"
    );
}

/// Audit C — f16 scale must be rejected with `"found Half"`.
///
/// Upstream torch live 2026-05-25:
/// ```python
/// >>> torch.fake_quantize_per_channel_affine(
/// ...     torch.tensor([[1.0]], dtype=torch.float32),
/// ...     torch.tensor([1.0], dtype=torch.float16),
/// ...     torch.tensor([0], dtype=torch.int32), 0, -128, 127)
/// RuntimeError: Scale must be Float or BFloat16, found Half
/// ```
#[test]
fn audit_per_channel_f16_scale_rejected_with_half_phrasing() {
    let input = from_vec(
        vec![half::f16::from_f32(1.0)],
        &[1, 1],
    )
    .unwrap();
    let scale = from_vec(vec![half::f16::from_f32(1.0)], &[1]).unwrap();
    let zp = IntTensor::<i64>::from_vec(vec![0], vec![1]).unwrap();
    let result = grad_fns::quantize_grad::fake_quantize_per_channel_affine(
        &input, &scale, &zp, 0, -128, 127,
    );
    assert!(
        result.is_err(),
        "f16 scale must be rejected per upstream \
         FakeQuantPerChannelAffine.cpp:51-52"
    );
    let msg = format!("{}", result.unwrap_err());
    let expected_upstream_substring = "Scale must be Float or BFloat16, found Half";
    assert!(
        msg.contains(expected_upstream_substring),
        "ferrotorch error message does not contain upstream phrasing for f16.\n\
         expected substring: {expected_upstream_substring:?}\n\
         got: {msg:?}"
    );
}

/// Audit B — f32 scale must be ACCEPTED (no false positive).
///
/// Upstream torch live 2026-05-25:
/// ```python
/// >>> torch.fake_quantize_per_channel_affine(
/// ...     torch.tensor([[1.0]], dtype=torch.float32),
/// ...     torch.tensor([1.0], dtype=torch.float32),
/// ...     torch.tensor([0], dtype=torch.int32), 0, -128, 127)
/// tensor([[1.]])
/// ```
#[test]
fn audit_per_channel_f32_scale_accepted() {
    let input = from_vec(vec![1.0f32], &[1, 1]).unwrap();
    let scale = from_vec(vec![1.0f32], &[1]).unwrap();
    let zp = IntTensor::<i64>::from_vec(vec![0], vec![1]).unwrap();
    let result = grad_fns::quantize_grad::fake_quantize_per_channel_affine(
        &input, &scale, &zp, 0, -128, 127,
    );
    assert!(
        result.is_ok(),
        "f32 scale must be ACCEPTED per upstream \
         FakeQuantPerChannelAffine.cpp:51-52; the new TypeId gate must \
         not produce a false positive on f32. Error: {:?}",
        result.err()
    );
    // Upstream output for this case is tensor([[1.]]).
    let out = result.unwrap();
    let data = out.data_vec().unwrap();
    assert_eq!(data.len(), 1, "shape: {:?}", out.shape());
    assert!(
        (data[0] - 1.0f32).abs() < 1e-6,
        "expected upstream value 1.0, got {}",
        data[0]
    );
}

/// Audit B — bf16 scale must be ACCEPTED past the dtype gate.
///
/// Upstream torch passes the `TORCH_CHECK(scale.scalar_type() == Float
/// || scale.scalar_type() == BFloat16, ...)` gate at line 51-52 for
/// bf16 scale. (A separate TensorIterator-internal-cast assert later
/// trips on Python's calling pattern, but that is downstream of the
/// scalar_type gate and is not what #1262 was about.)
///
/// Ferrotorch's TypeId check at quantize_grad.rs:488 explicitly admits
/// `half::bf16` — verify it does.
#[test]
fn audit_per_channel_bf16_scale_passes_typeid_gate() {
    // Pure bf16 path: input is bf16, scale is bf16 (the ferrotorch
    // signature ties input.dtype == scale.dtype via the shared `T`).
    let input = from_vec(
        vec![half::bf16::from_f32(1.0), half::bf16::from_f32(2.0)],
        &[1, 2],
    )
    .unwrap();
    let scale = from_vec(
        vec![half::bf16::from_f32(1.0)],
        &[1],
    )
    .unwrap();
    let zp = IntTensor::<i64>::from_vec(vec![0], vec![1]).unwrap();
    let result = grad_fns::quantize_grad::fake_quantize_per_channel_affine(
        &input, &scale, &zp, 0, -128, 127,
    );
    // The dtype gate from #1262 MUST NOT reject bf16. Any other error
    // (e.g. a downstream NumCast issue) is separate; here we verify
    // specifically that the error — if any — is NOT a "Scale must be
    // Float or BFloat16" message.
    if let Err(e) = &result {
        let msg = format!("{e}");
        assert!(
            !msg.contains("Scale must be Float or BFloat16"),
            "bf16 scale was rejected by the #1262 TypeId gate, but upstream \
             admits BFloat16 at FakeQuantPerChannelAffine.cpp:51-52. \
             Error: {msg:?}"
        );
    }
}

/// Audit F — coexistence with #1261's removal of the `scale > 0` check.
///
/// The two parallel-agent commits (#1262 TypeId gate, then #1261 scale>0
/// removal) MUST coexist. Pin both behaviors on a single call:
///   - f32 scale (passes the #1262 gate)
///   - scale value 0.0 (would have been rejected pre-#1261, must not be now)
///
/// Upstream torch live 2026-05-25:
/// ```python
/// >>> torch.fake_quantize_per_channel_affine(
/// ...     torch.tensor([[1.0, 2.0]], dtype=torch.float32),
/// ...     torch.tensor([0.0, 1.0], dtype=torch.float32),
/// ...     torch.tensor([0, 0], dtype=torch.int32), 1, -128, 127)
/// tensor([[-0.,  2.]])
/// ```
/// (channel 0 has scale=0; the (-128 - 0) * 0.0 = -0.0 dequant lands;
///  channel 1 round-trips 2.0 unchanged.)
#[test]
fn audit_per_channel_1261_and_1262_coexist() {
    let input = from_vec(vec![1.0f32, 2.0], &[1, 2]).unwrap();
    let scale = from_vec(vec![0.0f32, 1.0], &[2]).unwrap();
    let zp = IntTensor::<i64>::from_vec(vec![0, 0], vec![2]).unwrap();
    let result = grad_fns::quantize_grad::fake_quantize_per_channel_affine(
        &input, &scale, &zp, 1, -128, 127,
    );
    assert!(
        result.is_ok(),
        "f32 scale + scale[0]=0 must be accepted: #1262 admits f32, \
         #1261 removed the scale>0 reject. Got error: {:?}",
        result.err()
    );
    let out = result.unwrap();
    let data = out.data_vec().unwrap();
    // Upstream-cited expected output. R-CHAR-3: from live torch above.
    // channel 0: scale=0 -> dequant -0.0; we tolerate ±0 via abs check.
    assert_eq!(data.len(), 2, "shape: {:?}", out.shape());
    assert!(
        data[0] == 0.0f32 || data[0] == -0.0f32,
        "channel 0 (scale=0) expected ±0.0 (upstream tensor([[-0., 2.]])), got {}",
        data[0]
    );
    assert!(
        (data[1] - 2.0f32).abs() < 1e-6,
        "channel 1 expected 2.0, got {}",
        data[1]
    );
}

/// Audit D — the TypeId check runs at function entry, BEFORE any other
/// validation. An f64 caller with otherwise-invalid args (e.g. ndim
/// mismatch) must trip the dtype gate first, not a downstream error.
///
/// Rationale: upstream's TORCH_CHECK on scalar_type is at line 51-52,
/// BEFORE the scale.dim()==1 check at line 55. Ferrotorch must mirror
/// this ordering so the user sees the dtype error first.
#[test]
fn audit_per_channel_typeid_check_runs_before_shape_check() {
    // Deliberately also-invalid: 2D scale (would fail the dim==1 check),
    // but the dtype check should fire first.
    let input = from_vec(vec![1.0f64, 2.0], &[1, 2]).unwrap();
    let scale = from_vec(vec![1.0f64, 1.0], &[1, 2]).unwrap(); // 2D, invalid
    let zp = IntTensor::<i64>::from_vec(vec![0, 0], vec![2]).unwrap();
    let result = grad_fns::quantize_grad::fake_quantize_per_channel_affine(
        &input, &scale, &zp, 1, -128, 127,
    );
    assert!(result.is_err(), "must reject (either dtype or shape)");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("Scale must be Float or BFloat16"),
        "expected dtype rejection to FIRE FIRST (upstream order: \
         FakeQuantPerChannelAffine.cpp:51-52 before :55); got {msg:?}"
    );
}
