//! Divergence audit for commit `0358c166a` (#1621 "closes", refs #1620 #1611):
//! reflect/circular signed (crop+pad) padding via a UNIFIED INDEX MAP against
//! the ORIGINAL input window, with `PadNdSignedModeBackward` as the scatter-add
//! transpose.
//!
//! The forward index maps and the backward scatter-add are CORRECT (verified
//! against live torch 2.11.0+cu130 over a brute grid of sizes 2..6 x all
//! lo/hi in -size..=size — 0 value mismatches and 0 grad mismatches on every
//! input torch can evaluate).
//!
//! DIVERGENCE FOUND in the reflect LEGALITY CHECK
//! (`ferrotorch-nn/src/padding.rs:1590-1597`, `pad_nd_signed_reflect_circular`):
//!
//! ```ignore
//!     if mode == PaddingMode::Reflect && (lo.abs() >= size || hi.abs() >= size) {
//!         return Err(InvalidArgument { ... });
//!     }
//! ```
//!
//! The check uses `lo.abs()` / `hi.abs()`. Upstream torch checks the SIGNED pad:
//!   `pytorch aten/src/ATen/native/ReflectionPad.cpp:48-49`
//!     `TORCH_CHECK(pad_l < input_w && pad_r < input_w, ...)`
//! and again identically for the backward meta at `ReflectionPad.cpp:88-89`.
//! Because a NEGATIVE pad is always `< input_w`, torch only rejects a pad whose
//! POSITIVE magnitude reaches `>= input_w`. ferrotorch's `abs()` additionally
//! rejects every NEGATIVE (crop) pad whose magnitude is `>= size`, which torch
//! ACCEPTS. So `F.pad(x, [-3, 2], mode="reflect")` on a size-3 axis is legal in
//! torch (returns `[2., 1.]`) but ferrotorch returns `Err`.
//!
//! This is the SAME class of bug the #1620/#1621 series set out to fix
//! (rejecting a signed-pad case torch accepts), just relocated from the
//! crop-then-pad guard into the new unified-map legality guard. The reflect
//! index map itself computes valid indices for these inputs (verified: `[-3,2]`
//! on size 3 maps output j=0,1 -> input idx 1,0 -> `[2,1]`); only the `abs()`
//! guard wrongly rejects.
//!
//! R-CHAR-3: every expected forward + grad below is from a live PyTorch
//! 2.11.0+cu130 oracle (the reproducing Python is inlined in each doc comment).
//! NONE are copied from the ferrotorch side (ferrotorch `Err`s on every
//! divergence case).
//!
//! Tracking: #1621. `#[ignore]`d so the tracked issue, not a red CI bar, drives
//! the generator's fix: the reflect legality guard must compare the SIGNED pad
//! (`lo < size && hi < size`), matching `ReflectionPad.cpp:48-49`, not its
//! absolute value.

use ferrotorch_core::Tensor;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_nn::padding::{PaddingMode, functional_pad_1d_signed, functional_pad_2d_signed};

fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

fn tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, msg: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{msg}: length mismatch actual={} expected={}\n actual={actual:?}\n expected={expected:?}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{msg}: element {i} differs: actual={a} expected={e}\n actual={actual:?}\n expected={expected:?}"
        );
    }
}

/// DIVERGENCE (forward + acceptance): reflect with left crop `-3` on a size-3
/// axis. `|lo| = 3 >= size = 3`, so ferrotorch's `abs()` legality guard rejects
/// with `Err`. torch's SIGNED check `pad_l(-3) < input_w(3)` passes, so it
/// returns the reflection of the original window.
///
/// Live torch 2.11.0+cu130:
/// ```python
/// x = torch.tensor([[1., 2., 3.]])     # shape [1, 3]
/// F.pad(x, [-3, 2], mode="reflect")    # shape [1, 2], data [2., 1.]
/// ```
/// Upstream: `ReflectionPad.cpp:48-49` `pad_l(-3) < 3 && pad_r(2) < 3` passes;
/// offset gather (`PaddingKernel.cpp:63-80`) reads original idx 1,0 -> [2,1].
///
/// Tracking: #1621
#[test]
fn divergence_reflect_negpad_eq_size_legality() {
    let x = tensor(&[1.0, 2.0, 3.0], &[1, 3]);
    let y = functional_pad_1d_signed(&x, -3, 2, PaddingMode::Reflect, 0.0).expect(
        "torch F.pad reflect [-3,2] on size 3 returns [2,1] (pad_l<size signed); \
         ferrotorch rejects because |-3| >= 3 (abs guard, ReflectionPad.cpp:48-49 is signed)",
    );
    assert_eq!(
        y.shape(),
        &[1, 2],
        "torch reflect [-3,2] on size 3 -> shape [1,2]"
    );
    assert_close(
        y.data().unwrap(),
        &[2.0, 1.0],
        1e-7,
        "torch reflect [-3,2] on size 3 forward",
    );
}

/// DIVERGENCE (forward): the mirror — right crop `-3` on size 3, left reflect
/// pad `+1`. `|hi| = 3 >= 3` so ferrotorch rejects. torch accepts (`pad_r(-3) <
/// 3`) and returns `[2.]`.
///
/// Live torch 2.11.0+cu130:
/// ```python
/// x = torch.tensor([[1., 2., 3.]])
/// F.pad(x, [1, -3], mode="reflect")    # shape [1, 1], data [2.]
/// ```
///
/// Tracking: #1621
#[test]
fn divergence_reflect_negpad_mirror_legality() {
    let x = tensor(&[1.0, 2.0, 3.0], &[1, 3]);
    let y = functional_pad_1d_signed(&x, 1, -3, PaddingMode::Reflect, 0.0)
        .expect("torch F.pad reflect [1,-3] on size 3 returns [2]; ferrotorch rejects |−3| >= 3");
    assert_eq!(
        y.shape(),
        &[1, 1],
        "torch reflect [1,-3] on size 3 -> shape [1,1]"
    );
    assert_close(
        y.data().unwrap(),
        &[2.0],
        1e-7,
        "torch reflect [1,-3] on size 3 forward",
    );
}

/// DIVERGENCE (forward, size 4): left crop `-4`, right reflect `+3`.
/// `|lo| = 4 >= size = 4` so ferrotorch rejects; torch accepts (`-4 < 4`).
///
/// Live torch 2.11.0+cu130:
/// ```python
/// x = torch.tensor([[1., 2., 3., 4.]])
/// F.pad(x, [-4, 3], mode="reflect")    # shape [1, 3], data [3., 2., 1.]
/// ```
///
/// Tracking: #1621
#[test]
fn divergence_reflect_negpad4_size4_legality() {
    let x = tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let y = functional_pad_1d_signed(&x, -4, 3, PaddingMode::Reflect, 0.0).expect(
        "torch F.pad reflect [-4,3] on size 4 returns [3,2,1]; ferrotorch rejects |−4| >= 4",
    );
    assert_eq!(
        y.shape(),
        &[1, 3],
        "torch reflect [-4,3] on size 4 -> shape [1,3]"
    );
    assert_close(
        y.data().unwrap(),
        &[3.0, 2.0, 1.0],
        1e-7,
        "torch reflect [-4,3] on size 4 forward",
    );
}

/// DIVERGENCE (backward): grad of the wrongly-rejected `reflect [-3,2]` on size
/// 3 must accumulate onto the original elements the reflection read. ferrotorch
/// cannot even produce the forward (Err), so it cannot produce this grad.
///
/// Live torch 2.11.0+cu130:
/// ```python
/// x = torch.tensor([[1., 2., 3.]], requires_grad=True)
/// y = F.pad(x, [-3, 2], mode="reflect"); y.sum().backward()
/// x.grad   # [1., 1., 0.]   (output [2,1] reads idx 1,0; idx 2 unused)
/// ```
///
/// Tracking: #1621
#[test]
fn divergence_reflect_negpad_eq_size_backward() {
    let x = leaf(&[1.0, 2.0, 3.0], &[1, 3]);
    let y = functional_pad_1d_signed(&x, -3, 2, PaddingMode::Reflect, 0.0)
        .expect("torch F.pad reflect [-3,2] on size 3 returns [2,1]; ferrotorch must not Err");
    let sum = ferrotorch_core::grad_fns::reduction::sum(&y).unwrap();
    ferrotorch_core::backward(&sum).unwrap();
    let g = x.grad().unwrap().expect("grad must be populated");
    assert_close(
        g.data().unwrap(),
        &[1.0, 1.0, 0.0],
        1e-7,
        "torch reflect [-3,2] on size 3 grad",
    );
}

/// DIVERGENCE (2-D): a per-axis mix where the W axis (size 4) has a `-4` crop.
/// `|lo_w| = 4 >= 4` so ferrotorch rejects the whole call; torch accepts and
/// reflects each axis against its original window.
///
/// Live torch 2.11.0+cu130:
/// ```python
/// x = torch.arange(1, 13, dtype=torch.float32).reshape(1, 3, 4)  # H=3, W=4
/// F.pad(x, [-4, 3, 0, 0], mode="reflect")   # shape [1, 3, 3]
/// # data [3,2,1, 7,6,5, 11,10,9]
/// ```
///
/// Tracking: #1621
#[test]
fn divergence_reflect_2d_negpad_w_legality() {
    let data: Vec<f32> = (1..=12).map(|v| v as f32).collect();
    let x = tensor(&data, &[1, 3, 4]);
    // functional_pad_2d_signed(input, pad_left, pad_right, pad_top, pad_bottom, ...)
    let y = functional_pad_2d_signed(&x, -4, 3, 0, 0, PaddingMode::Reflect, 0.0).expect(
        "torch F.pad reflect [-4,3,0,0] on W=4 returns shape [1,3,3]; ferrotorch rejects |−4| >= 4",
    );
    assert_eq!(
        y.shape(),
        &[1, 3, 3],
        "torch reflect [-4,3] on W=4 -> shape [1,3,3]"
    );
    assert_close(
        y.data().unwrap(),
        &[3.0, 2.0, 1.0, 7.0, 6.0, 5.0, 11.0, 10.0, 9.0],
        1e-7,
        "torch reflect [-4,3,0,0] 2d forward",
    );
}

// ===========================================================================
// REGRESSION GUARDS — cases where ferrotorch's reflect legality DOES match
// torch. These should PASS today and pin the exact boundary of the bug.
// Every expected value / acceptance is from the same live torch 2.11.0+cu130
// oracle.
// ===========================================================================

/// Live torch: a POSITIVE reflect pad equal to the size IS rejected by torch
/// (`pad_l(4) < input_w(4)` is false), and ferrotorch also rejects
/// (`|4| >= 4`). Acceptance matches here — both Err.
/// ```python
/// x = torch.tensor([[1., 2., 3., 4.]])
/// F.pad(x, [4, 0], mode="reflect")
/// # RuntimeError: Padding size should be less than the corresponding input dimension
/// ```
#[test]
fn regression_reflect_pospad_eq_size_both_reject() {
    let x = tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let r = functional_pad_1d_signed(&x, 4, 0, PaddingMode::Reflect, 0.0);
    assert!(
        r.is_err(),
        "torch rejects positive reflect pad 4 == input_w 4; ferrotorch must also reject"
    );
}

/// Live torch: the #1621 motivating case `reflect [-3,2]` on size 4 — here
/// `|−3| = 3 < 4`, so ferrotorch's abs guard does NOT misfire and the case
/// works (this is why the prior #1620 regression suite passed). Pins that the
/// bug only surfaces when `|neg pad| >= size`.
/// ```python
/// x = torch.tensor([[1., 2., 3., 4.]])
/// F.pad(x, [-3, 2], mode="reflect")   # shape [1, 3], data [4., 3., 2.]
/// ```
#[test]
fn regression_reflect_negpad3_size4_accepts() {
    let x = tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let y = functional_pad_1d_signed(&x, -3, 2, PaddingMode::Reflect, 0.0)
        .expect("|-3| < 4: ferrotorch accepts and torch returns [4,3,2]");
    assert_eq!(y.shape(), &[1, 3]);
    assert_close(
        y.data().unwrap(),
        &[4.0, 3.0, 2.0],
        1e-7,
        "reflect [-3,2] size 4 forward (boundary regression)",
    );
}
