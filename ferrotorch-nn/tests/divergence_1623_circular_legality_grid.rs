//! Divergence audit for commit `51f103d35` (#1623 "reflect legality SIGNED;
//! circular over-crop Errs", refs #1621 #1620 #1611) — the FINAL audit of the
//! negative-pad legality chain.
//!
//! METHOD (R-CHAR-3): brute-forced the full accept/reject + value grid for
//! `torch.nn.functional.pad(x, [lo, hi], mode=<m>)` over sizes 2..6 and all
//! `lo,hi` in `-size-1..=size+1` against LIVE torch 2.11.0+cu130, then compared
//! ferrotorch's `functional_pad_1d_signed` accept/reject + value at every grid
//! point. Result per mode:
//!
//! reflect : 0 accept-mismatch, 0 value-mismatch, 0 panic  -> CLEAN (matches torch)
//! constant: 0 accept-mismatch, 0 value-mismatch, 0 panic  -> CLEAN (matches torch)
//! circular: 86 accept-mismatch + 70 PANICS                -> 3 DIVERGENCE CLASSES
//!
//! The #1623 fix correctly made the REFLECT legality SIGNED (verified: reflect
//! grid is byte-for-byte torch-matching, incl. net-zero -> torch errors
//! `output_w >= 1`, ferrotorch errors). CONSTANT net-zero is also clean (torch
//! returns empty `[...,0]`, ferrotorch returns empty `[...,0]`). But CIRCULAR
//! still diverges from torch in three distinct ways below.
//!
//! Upstream contracts (pytorch 6710f8ebc). The circular kernel
//! `_pad_circular_symint` enforces TWO checks. First, `PadNd.cpp:142`
//! `TORCH_CHECK(pad_l <= size && pad_r <= size, "Padding value causes wrapping
//! around more than once.")` — a POSITIVE pad strictly greater than `size` is
//! REJECTED. Second, `PadNd.cpp:144-145` `TORCH_CHECK(out_shape[...] >= 0,
//! "Negative padding value is resulting in an empty dimension")` — circular
//! ALLOWS a net output size of EXACTLY 0 (empty dim), only rejecting a negative
//! net size. `constant_pad_nd:76` has the identical `new_dim >= 0` allow-zero
//! check; `ReflectionPad.cpp:59-60` instead demands `output_w >= 1`, so reflect
//! REJECTS net-zero — that's why reflect/constant differ from circular at the
//! net-zero boundary.
//!
//! Every `assert_*` expected value/acceptance below is from the live torch
//! oracle (reproducing Python inlined). NONE are copied from the ferrotorch
//! side (ferrotorch Errs or PANICs on every divergence case).
//!
//! Tracking: #1624 (blocker filed for the surviving circular legality residual;
//! #1623 is the audited commit that fixed reflect but left circular diverging).

use ferrotorch_core::Tensor;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_nn::padding::{PaddingMode, functional_pad_1d_signed};

fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

fn tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

// ===========================================================================
// DIVERGENCE CLASS A — circular NET-ZERO crop (THE FLAGGED EDGE)
// torch returns an empty `[...,0]` tensor; ferrotorch's `new_size < 1` guard
// (padding.rs:1628, shared with reflect) Errs. 31 grid points diverge this way.
// ===========================================================================

/// THE FLAGGED EDGE. Live torch 2.11.0+cu130:
/// ```python
/// x = torch.tensor([[1., 2., 3., 4.]])      # size 4
/// y = F.pad(x, [-4, 0], mode="circular")    # shape [1, 0], data []  (NOT an error)
/// ```
/// Upstream `PadNd.cpp:144-145` checks `out_shape >= 0`, so a net output size of
/// 0 is a VALID empty dim — not an error. ferrotorch's `pad_nd_signed_reflect_
/// circular` rejects it via `new_size < 1` (padding.rs:1628), Err'ing instead of
/// returning the empty tensor. (The constant-mode signed path #1611 already
/// returns `[...,0]` here — circular should match.)
///
/// Tracking: #1624
#[test]
#[ignore = "divergence: circular net-zero crop returns empty [..,0] in torch but ferrotorch new_size<1 Errs; tracking #1624"]
fn divergence_circular_netzero_crop_left() {
    let x = tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let y = functional_pad_1d_signed(&x, -4, 0, PaddingMode::Circular, 0.0)
        .expect("torch circular [-4,0] on size 4 returns empty [1,0]; ferrotorch must not Err");
    assert_eq!(
        y.shape(),
        &[1, 0],
        "torch circular [-4,0] on size 4 -> empty dim shape [1,0]"
    );
    assert_eq!(
        y.data().unwrap().len(),
        0,
        "torch circular [-4,0] on size 4 -> empty data"
    );
}

/// Mirror: right-side net-zero crop. Live torch 2.11.0+cu130:
/// ```python
/// x = torch.tensor([[1., 2., 3., 4.]])
/// F.pad(x, [0, -4], mode="circular")   # shape [1, 0], data []
/// ```
/// Tracking: #1624
#[test]
#[ignore = "divergence: circular net-zero crop (right) returns empty [..,0] in torch but ferrotorch Errs; tracking #1624"]
fn divergence_circular_netzero_crop_right() {
    let x = tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let y = functional_pad_1d_signed(&x, 0, -4, PaddingMode::Circular, 0.0)
        .expect("torch circular [0,-4] on size 4 returns empty [1,0]; ferrotorch must not Err");
    assert_eq!(y.shape(), &[1, 0]);
    assert_eq!(y.data().unwrap().len(), 0);
}

/// Both-side net-zero crop. Live torch 2.11.0+cu130:
/// ```python
/// x = torch.tensor([[1., 2., 3., 4.]])
/// F.pad(x, [-2, -2], mode="circular")  # shape [1, 0], data []
/// ```
/// Tracking: #1624
#[test]
#[ignore = "divergence: circular net-zero crop (both sides) returns empty [..,0] in torch but ferrotorch Errs; tracking #1624"]
fn divergence_circular_netzero_crop_both() {
    let x = tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let y = functional_pad_1d_signed(&x, -2, -2, PaddingMode::Circular, 0.0)
        .expect("torch circular [-2,-2] on size 4 returns empty [1,0]; ferrotorch must not Err");
    assert_eq!(y.shape(), &[1, 0]);
    assert_eq!(y.data().unwrap().len(), 0);
}

/// Net-zero crop where the LEFT crop alone exceeds the dim but the right pad is
/// positive (`lo=-3 < -size=-2`, net `2-3+1=0`). torch STILL returns empty —
/// its only checks are `pad_l <= size` (`-3 <= 2` true) and `out_shape >= 0`
/// (`0 >= 0` true). ferrotorch rejects via BOTH `lo <= -size` (the #1623 guard)
/// AND `new_size < 1`. Live torch 2.11.0+cu130:
/// ```python
/// x = torch.tensor([[1., 2.]])         # size 2
/// F.pad(x, [-3, 1], mode="circular")   # shape [1, 0], data []
/// ```
/// Tracking: #1624
#[test]
#[ignore = "divergence: circular net-zero with deep one-side crop returns empty [..,0] in torch but ferrotorch Errs; tracking #1624"]
fn divergence_circular_netzero_deep_left_crop() {
    let x = tensor(&[1.0, 2.0], &[1, 2]);
    let y = functional_pad_1d_signed(&x, -3, 1, PaddingMode::Circular, 0.0)
        .expect("torch circular [-3,1] on size 2 returns empty [1,0]; ferrotorch must not Err");
    assert_eq!(y.shape(), &[1, 0]);
    assert_eq!(y.data().unwrap().len(), 0);
}

/// Backward for the net-zero crop: an empty output contributes no gradient, so
/// `x.grad` is all-zero (no positions read). Live torch 2.11.0+cu130:
/// ```python
/// x = torch.tensor([[1., 2., 3., 4.]], requires_grad=True)
/// y = F.pad(x, [-4, 0], mode="circular"); y.sum().backward()
/// x.grad   # [0., 0., 0., 0.]
/// ```
/// Tracking: #1624
#[test]
#[ignore = "divergence: circular net-zero crop forward Errs so backward unreachable in ferrotorch; tracking #1624"]
fn divergence_circular_netzero_crop_backward() {
    let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let y = functional_pad_1d_signed(&x, -4, 0, PaddingMode::Circular, 0.0)
        .expect("torch circular [-4,0] returns empty [1,0]; ferrotorch must not Err");
    let sum = ferrotorch_core::grad_fns::reduction::sum(&y).unwrap();
    ferrotorch_core::backward(&sum).unwrap();
    let g = x.grad().unwrap().expect("grad must be populated");
    assert_eq!(
        g.data().unwrap(),
        &[0.0, 0.0, 0.0, 0.0],
        "torch circular [-4,0] grad is all-zero (empty output reads nothing)"
    );
}

// ===========================================================================
// DIVERGENCE CLASS B — circular POSITIVE over-wrap
// torch REJECTS `pad > size` ("wrapping around more than once", PadNd.cpp:142),
// but ferrotorch's all-non-negative circular path (`functional_pad_nd_positive`
// -> `src_index_1d` rem_euclid, padding.rs:642) has NO legality check and wraps
// repeatedly. 55 grid points diverge this way (ferrotorch accepts what torch
// rejects). The #1623 guard `lo > size || hi > size` lives in `pad_nd_signed_
// reflect_circular`, which is NEVER reached for all-non-negative pads.
// ===========================================================================

/// Live torch 2.11.0+cu130:
/// ```python
/// x = torch.tensor([[1., 2.]])         # size 2
/// F.pad(x, [0, 3], mode="circular")
/// # RuntimeError: Padding value causes wrapping around more than once.
/// ```
/// `hi=3 > size=2` -> upstream `PadNd.cpp:142` rejects. ferrotorch accepts and
/// returns `[1,2,1,2,1]` (repeated wrap) because the all-positive circular path
/// skips the legality guard entirely.
///
/// Tracking: #1624
#[test]
#[ignore = "divergence: circular positive over-wrap (pad>size) rejected by torch but ferrotorch wraps repeatedly; tracking #1624"]
fn divergence_circular_positive_overwrap_right() {
    let x = tensor(&[1.0, 2.0], &[1, 2]);
    let r = functional_pad_1d_signed(&x, 0, 3, PaddingMode::Circular, 0.0);
    assert!(
        r.is_err(),
        "torch rejects circular [0,3] on size 2 ('wrapping around more than once'); \
         ferrotorch must reject too, but it returned {:?}",
        r.ok()
            .map(|t| (t.shape().to_vec(), t.data().unwrap().to_vec()))
    );
}

/// Left-side positive over-wrap. Live torch 2.11.0+cu130:
/// ```python
/// x = torch.tensor([[1., 2., 3.]])     # size 3
/// F.pad(x, [4, 0], mode="circular")
/// # RuntimeError: Padding value causes wrapping around more than once.
/// ```
/// `lo=4 > size=3`. ferrotorch accepts (returns `[3,1,2,3,1,2,3]`).
///
/// Tracking: #1624
#[test]
#[ignore = "divergence: circular positive over-wrap (left, pad>size) rejected by torch but ferrotorch wraps; tracking #1624"]
fn divergence_circular_positive_overwrap_left() {
    let x = tensor(&[1.0, 2.0, 3.0], &[1, 3]);
    let r = functional_pad_1d_signed(&x, 4, 0, PaddingMode::Circular, 0.0);
    assert!(
        r.is_err(),
        "torch rejects circular [4,0] on size 3; ferrotorch must reject too, got {:?}",
        r.ok()
            .map(|t| (t.shape().to_vec(), t.data().unwrap().to_vec()))
    );
}

// ===========================================================================
// DIVERGENCE CLASS C — circular MIXED-SIGN over-crop -> OOB PANIC (R-CODE-2)
// torch REJECTS these (crop leaves fewer elements than the opposite-side wrap
// needs -> "wrapping around more than once" / "unsupported operation"). The
// #1623 guard catches `lo <= -size || hi <= -size`, but NOT cases like
// `[-1, 2]` on size 2 (lo=-1 > -size=-2, hi=2 not > size=2) — these slip past
// every guard and `circular_axis_src` indexes OOB into `data`, causing a Rust
// `index out of bounds` PANIC. 70 grid points panic this way.
// ===========================================================================

/// Live torch 2.11.0+cu130 REJECTS (no defined value):
/// ```python
/// x = torch.tensor([[1., 2.]])         # size 2
/// F.pad(x, [-1, 2], mode="circular")
/// # RuntimeError: unsupported operation: ... input and written-to tensor overlap
/// ```
/// After the `lo=-1` crop only 1 element remains, but the `hi=+2` wrap needs 2
/// -> torch rejects. ferrotorch's guards all pass (`-1 > -2`, `2 <= 2`,
/// `new_size = 2-1+2 = 3 >= 1`) so it reaches the gather and `circular_axis_src`
/// returns an OOB source index -> `data[..]` PANIC (padding.rs:1667). A panic is
/// never an acceptable substitute for torch's clean reject (R-CODE-2).
///
/// Tracking: #1624
#[test]
#[ignore = "divergence: circular mixed-sign over-crop ([-1,2] size2) PANICs OOB in ferrotorch; torch rejects cleanly (R-CODE-2); tracking #1624"]
fn divergence_circular_mixed_sign_overcrop_panics() {
    let x = tensor(&[1.0, 2.0], &[1, 2]);
    // torch rejects; ferrotorch must NOT panic — it must return Err. This test
    // FAILS today via an `index out of bounds` PANIC inside
    // `pad_nd_signed_reflect_circular` (padding.rs:1667).
    let r = functional_pad_1d_signed(&x, -1, 2, PaddingMode::Circular, 0.0);
    assert!(
        r.is_err(),
        "torch rejects circular [-1,2] on size 2; ferrotorch must return Err (not panic / not garbage), got {:?}",
        r.ok()
            .map(|t| (t.shape().to_vec(), t.data().unwrap().to_vec()))
    );
}

/// Mirror of Class C: positive-left / negative-right over-crop. Live torch:
/// ```python
/// x = torch.tensor([[1., 2.]])         # size 2
/// F.pad(x, [2, -1], mode="circular")
/// # RuntimeError (unsupported / wrapping)
/// ```
/// ferrotorch PANICs OOB (lo=2 not > size=2, hi=-1 > -size=-2, net=3>=1).
///
/// Tracking: #1624
#[test]
#[ignore = "divergence: circular mixed-sign over-crop ([2,-1] size2) PANICs OOB in ferrotorch; torch rejects; tracking #1624"]
fn divergence_circular_mixed_sign_overcrop_mirror_panics() {
    let x = tensor(&[1.0, 2.0], &[1, 2]);
    let r = functional_pad_1d_signed(&x, 2, -1, PaddingMode::Circular, 0.0);
    assert!(
        r.is_err(),
        "torch rejects circular [2,-1] on size 2; ferrotorch must return Err, got {:?}",
        r.ok()
            .map(|t| (t.shape().to_vec(), t.data().unwrap().to_vec()))
    );
}

// ===========================================================================
// REGRESSION GUARDS — confirm the parts the #1623 chain DID fix stay matching
// torch. These PASS today and pin the boundary of the surviving divergence.
// All expected values from the same live torch 2.11.0+cu130 oracle.
// ===========================================================================

/// Live torch: reflect net-zero crop ERRORS (`output_w >= 1`, ReflectionPad.cpp
/// :59-60), unlike circular/constant which return empty. ferrotorch also Errs.
/// This pins that reflect's net-zero rejection (`new_size < 1`) is CORRECT for
/// reflect (the shared guard is only wrong for circular).
/// ```python
/// x = torch.tensor([[1., 2., 3., 4.]])
/// F.pad(x, [-4, 0], mode="reflect")
/// # RuntimeError: input (W: 4) is too small. Calculated output W: 0
/// ```
#[test]
fn regression_reflect_netzero_crop_both_reject() {
    let x = tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let r = functional_pad_1d_signed(&x, -4, 0, PaddingMode::Reflect, 0.0);
    assert!(
        r.is_err(),
        "torch reflect [-4,0] on size 4 errors (output_w must be >= 1); ferrotorch must also Err"
    );
}

/// Live torch: constant net-zero crop returns an EMPTY `[1,0]` tensor (the
/// #1611 signed-constant path). ferrotorch matches. Pins that the constant path
/// already does what circular should.
/// ```python
/// x = torch.tensor([[1., 2., 3., 4.]])
/// F.pad(x, [-4, 0], mode="constant")   # shape [1, 0], data []
/// ```
#[test]
fn regression_constant_netzero_crop_returns_empty() {
    let x = tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let y = functional_pad_1d_signed(&x, -4, 0, PaddingMode::Zeros, 0.0)
        .expect("torch constant [-4,0] returns empty [1,0]; ferrotorch matches");
    assert_eq!(y.shape(), &[1, 0], "constant net-zero -> empty dim");
    assert_eq!(y.data().unwrap().len(), 0);
}

/// Live torch: a valid circular crop+wrap inside the legal range matches torch
/// byte-for-byte (this is the #1620/#1621 case the chain DID fix). Pins the
/// surviving divergence is only at the over-wrap / net-zero boundaries.
/// ```python
/// x = torch.tensor([[1., 2., 3., 4., 5.]])  # size 5
/// F.pad(x, [-1, 0], mode="circular")        # shape [1, 4], data [2,3,4,5]
/// ```
#[test]
fn regression_circular_inrange_crop_matches() {
    let x = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 5]);
    let y = functional_pad_1d_signed(&x, -1, 0, PaddingMode::Circular, 0.0)
        .expect("circular [-1,0] on size 5 is in-range; ferrotorch accepts");
    assert_eq!(y.shape(), &[1, 4]);
    let d = y.data().unwrap();
    for (a, b) in d.iter().zip([2.0_f32, 3.0, 4.0, 5.0].iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "circular [-1,0] size5 -> [2,3,4,5], got {d:?}"
        );
    }
}

/// Live torch: a valid in-range circular wrap (positive `pad <= size`) matches.
/// ```python
/// x = torch.tensor([[1., 2., 3.]])          # size 3
/// F.pad(x, [1, 2], mode="circular")         # shape [1, 6], data [3,1,2,3,1,2]
/// ```
#[test]
fn regression_circular_inrange_wrap_matches() {
    let x = tensor(&[1.0, 2.0, 3.0], &[1, 3]);
    let y = functional_pad_1d_signed(&x, 1, 2, PaddingMode::Circular, 0.0)
        .expect("circular [1,2] on size 3 is in-range (pad<=size); ferrotorch accepts");
    assert_eq!(y.shape(), &[1, 6]);
    let d = y.data().unwrap();
    for (a, b) in d.iter().zip([3.0_f32, 1.0, 2.0, 3.0, 1.0, 2.0].iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "circular [1,2] size3 -> [3,1,2,3,1,2], got {d:?}"
        );
    }
}
