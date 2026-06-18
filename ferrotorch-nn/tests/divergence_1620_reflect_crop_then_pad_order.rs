//! Divergence audit for commit `08ab5a598` (#1620 "closes", refs #1611):
//! negative/crop pad under reflect/replicate/circular via a CROP-FIRST then
//! mode-pad composition.
//!
//! DIVERGENCE FOUND in `functional_pad_nd_signed`
//! (`ferrotorch-nn/src/padding.rs:1531-1561`). The fixer replaced the (also
//! wrong) #1611 false-rejection guard with:
//!
//! ```ignore
//!     let crop_pads = pads.map(|(lo,hi)| (lo.min(0), hi.min(0)));   // negative only
//!     let pad_pads  = pads.map(|(lo,hi)| (lo.max(0), hi.max(0)));   // positive only
//!     let cropped   = functional_pad_nd_signed(input, &crop_pads, Zeros, value)?; // narrow
//!     return functional_pad_nd_positive(&cropped, &pad_pads, mode, value);        // mode-pad
//! ```
//!
//! and the commit message + `.design/ferrotorch-nn/padding.md` REQ-2 (line 49)
//! BOTH claim this crop-then-pad composition is "byte-identical" / reproduces
//! torch's native kernels "byte-for-byte". That claim is FALSE for `reflect`
//! whenever a side is cropped (negative pad) AND the opposite side has a
//! positive reflect pad that is `>=` the post-crop dimension size.
//!
//! WHY (upstream): torch's reflect kernel does NOT crop first. It reflects
//! against a window of the ORIGINAL input, using a gather offset:
//!   `pytorch aten/src/ATen/native/cpu/PaddingKernel.cpp:63-65`
//!     `i_start = max(0,-pad); o_start = max(0,pad); offset = i_start - o_start;`
//!   `pytorch aten/src/ATen/native/cpu/PaddingKernel.cpp:71-80` (ReflectionPad::index)
//!     reflected index `i` is then read as `i + offset` from the ORIGINAL input.
//! Crucially the reflect pad-size legality is checked against the ORIGINAL
//! `input_w`, not a cropped one:
//!   `pytorch aten/src/ATen/native/ReflectionPad.cpp:46-48`
//!     `int64_t output_w = input_w + pad_l + pad_r;`
//!     `TORCH_CHECK(pad_l < input_w && pad_r < input_w, ...)`
//! So `F.pad([1,2,3,4], [-3, 2], mode="reflect")` is LEGAL (`pad_r=2 < input_w=4`)
//! and the reflected values come from elements (3,2) that a crop-first pass
//! (`[-3,0] -> [4]`) has ALREADY DISCARDED. ferrotorch crops to a size-1 tensor
//! and then asks its reflect kernel to reflect-pad 2 on size 1, whose own
//! `pad < input_dim` guard (against the CROPPED dim) fails -> it ERRORS, while
//! torch returns `[4,3,2]`. A hard observable divergence (Err vs valid tensor),
//! plus wrong values / wrong grad in the cases that do not error.
//!
//! R-CHAR-3: every expected forward + grad below is from a live PyTorch
//! 2.11.0+cu130 oracle (reproducing Python in each test doc comment). NONE are
//! copied from the ferrotorch side (ferrotorch errors or differs on every one).
//!
//! Tracking: #1621. These are `#[ignore]`d so the tracked issue, not a red CI
//! bar, drives the generator's fix; the composition cannot be a simple
//! crop-then-pad — the reflect (and possibly circular) gather must run against
//! the original input window with torch's offset, per PaddingKernel.cpp:63-80.

use ferrotorch_core::Tensor;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_nn::padding::{PaddingMode, functional_pad_1d_signed};

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

/// DIVERGENCE (forward): reflect with a left crop `-3` and a right reflect pad
/// `+2` that exceeds the post-crop dim (1). Crop-first reduces `[1,2,3,4]` to
/// `[4]`, then reflect-pad 2 on a size-1 tensor — ferrotorch's reflect guard
/// (`pad < input_dim`, against the CROPPED dim=1) rejects `2 < 1`, so the call
/// ERRORS. torch reflects against the ORIGINAL 4-wide window and returns the
/// cropped-away values.
///
/// Live torch 2.11.0+cu130:
/// ```python
/// x = torch.tensor([[[1.,2.,3.,4.]]])
/// F.pad(x, [-3, 2], mode="reflect")  # shape [1,1,3], data [4., 3., 2.]
/// ```
/// Upstream: pad_r=2 < input_w=4 passes (`ReflectionPad.cpp:46-48`); offset
/// gather (`PaddingKernel.cpp:63-80`) reads original indices 3,2,1 -> [4,3,2].
///
/// Tracking: #1621
#[test]
fn divergence_reflect_crop3_pad2_forward() {
    let x = tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
    let y = functional_pad_1d_signed(&x, -3, 2, PaddingMode::Reflect, 0.0).expect(
        "torch F.pad reflect [-3,2] returns [4,3,2] (reflects original window); \
         ferrotorch crop-then-pad errors because reflect pad 2 >= cropped dim 1",
    );
    assert_eq!(
        y.shape(),
        &[1, 1, 3],
        "torch reflect [-3,2] -> shape [1,1,3]"
    );
    assert_close(
        y.data().unwrap(),
        &[4.0, 3.0, 2.0],
        1e-7,
        "torch reflect [-3,2] forward",
    );
}

/// DIVERGENCE (forward): reflect left crop `-2`, right reflect pad `+3` that
/// exceeds the post-crop dim (2). Crop-first gives `[3,4]`; reflect pad 3 on
/// size 2 fails `3 < 2` -> ferrotorch errors. torch returns the original-window
/// reflection.
///
/// Live torch 2.11.0+cu130:
/// ```python
/// x = torch.tensor([[[1.,2.,3.,4.]]])
/// F.pad(x, [-2, 3], mode="reflect")  # shape [1,1,5], data [3., 4., 3., 2., 1.]
/// ```
///
/// Tracking: #1621
#[test]
fn divergence_reflect_crop2_pad3_forward() {
    let x = tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
    let y = functional_pad_1d_signed(&x, -2, 3, PaddingMode::Reflect, 0.0).expect(
        "torch F.pad reflect [-2,3] returns [3,4,3,2,1]; ferrotorch crop-then-pad \
         errors because reflect pad 3 >= cropped dim 2",
    );
    assert_eq!(
        y.shape(),
        &[1, 1, 5],
        "torch reflect [-2,3] -> shape [1,1,5]"
    );
    assert_close(
        y.data().unwrap(),
        &[3.0, 4.0, 3.0, 2.0, 1.0],
        1e-7,
        "torch reflect [-2,3] forward",
    );
}

/// DIVERGENCE (forward): the mirror — right crop `-2`, left reflect pad `+3`.
/// Crop-first gives `[1,2]`; reflect pad 3 on size 2 fails -> error. torch
/// returns `[4,3,2,1,2]` — reflecting the original window from the left.
///
/// Live torch 2.11.0+cu130:
/// ```python
/// x = torch.tensor([[[1.,2.,3.,4.]]])
/// F.pad(x, [3, -2], mode="reflect")  # shape [1,1,5], data [4., 3., 2., 1., 2.]
/// ```
///
/// Tracking: #1621
#[test]
fn divergence_reflect_pad3_crop2_mirror_forward() {
    let x = tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
    let y = functional_pad_1d_signed(&x, 3, -2, PaddingMode::Reflect, 0.0).expect(
        "torch F.pad reflect [3,-2] returns [4,3,2,1,2]; ferrotorch crop-then-pad \
         errors because reflect pad 3 >= cropped dim 2",
    );
    assert_eq!(
        y.shape(),
        &[1, 1, 5],
        "torch reflect [3,-2] -> shape [1,1,5]"
    );
    assert_close(
        y.data().unwrap(),
        &[4.0, 3.0, 2.0, 1.0, 2.0],
        1e-7,
        "torch reflect [3,-2] forward",
    );
}

/// DIVERGENCE (backward): the grad of `reflect [-3,2]` accumulates onto the
/// ORIGINAL elements that the reflection read (indices 1,2,3), NOT just the
/// surviving crop element. Crop-then-pad cannot even produce the forward, so it
/// cannot produce this grad.
///
/// Live torch 2.11.0+cu130:
/// ```python
/// x = torch.tensor([[[1.,2.,3.,4.]]], requires_grad=True)
/// y = F.pad(x, [-3, 2], mode="reflect"); y.sum().backward()
/// x.grad  # [0., 1., 1., 1.]   (each of elements 2,3,4 hit once)
/// ```
///
/// Tracking: #1621
#[test]
fn divergence_reflect_crop3_pad2_backward() {
    let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
    let y = functional_pad_1d_signed(&x, -3, 2, PaddingMode::Reflect, 0.0)
        .expect("torch F.pad reflect [-3,2] returns [4,3,2]; ferrotorch must not error");
    let sum = ferrotorch_core::grad_fns::reduction::sum(&y).unwrap();
    ferrotorch_core::backward(&sum).unwrap();
    let g = x.grad().unwrap().expect("grad must be populated");
    assert_close(
        g.data().unwrap(),
        &[0.0, 1.0, 1.0, 1.0],
        1e-7,
        "torch reflect [-3,2] grad",
    );
}

/// DIVERGENCE (backward): grad of `reflect [-2,3]` -> `[1,1,2,1]` (element 3 is
/// read twice: once as the body, once as a reflected source).
///
/// Live torch 2.11.0+cu130:
/// ```python
/// x = torch.tensor([[[1.,2.,3.,4.]]], requires_grad=True)
/// y = F.pad(x, [-2, 3], mode="reflect"); y.sum().backward()
/// x.grad  # [1., 1., 2., 1.]
/// ```
///
/// Tracking: #1621
#[test]
fn divergence_reflect_crop2_pad3_backward() {
    let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
    let y = functional_pad_1d_signed(&x, -2, 3, PaddingMode::Reflect, 0.0)
        .expect("torch F.pad reflect [-2,3] returns [3,4,3,2,1]; ferrotorch must not error");
    let sum = ferrotorch_core::grad_fns::reduction::sum(&y).unwrap();
    ferrotorch_core::backward(&sum).unwrap();
    let g = x.grad().unwrap().expect("grad must be populated");
    assert_close(
        g.data().unwrap(),
        &[1.0, 1.0, 2.0, 1.0],
        1e-7,
        "torch reflect [-2,3] grad",
    );
}

// ===========================================================================
// REGRESSION GUARDS — cases where crop-then-pad DOES coincide with torch
// (reflect pad fits within the post-crop dim). These should PASS today; they
// document the exact boundary at which the composition starts diverging.
// Every expected value is from the same live torch 2.11.0+cu130 oracle.
// ===========================================================================

/// Live torch: `F.pad([[[1.,2.,3.,4.]]], [-1,2], mode="reflect")`
/// -> shape [1,1,5], data [2,3,4,3,2], grad [0,2,2,1]. Crop `[-1,0]`->`[2,3,4]`
/// (dim 3), reflect pad 2 < 3 fits, so crop-then-pad coincides.
#[test]
fn regression_reflect_crop1_pad2_matches() {
    let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
    let y = functional_pad_1d_signed(&x, -1, 2, PaddingMode::Reflect, 0.0).unwrap();
    assert_eq!(y.shape(), &[1, 1, 5]);
    assert_close(
        y.data().unwrap(),
        &[2.0, 3.0, 4.0, 3.0, 2.0],
        1e-7,
        "reflect [-1,2] fwd",
    );
    let sum = ferrotorch_core::grad_fns::reduction::sum(&y).unwrap();
    ferrotorch_core::backward(&sum).unwrap();
    let g = x
        .grad()
        .unwrap()
        .expect("input gradient should be populated");
    assert_close(
        g.data().unwrap(),
        &[0.0, 2.0, 2.0, 1.0],
        1e-7,
        "reflect [-1,2] grad",
    );
}

/// Live torch: `F.pad([[[1.,2.,3.,4.]]], [-2,3], mode="replicate")`
/// -> shape [1,1,5], data [3,4,4,4,4], grad [0,0,1,4]. replicate always reads
/// the boundary element (preserved by crop), so crop-then-pad coincides.
#[test]
fn regression_replicate_crop2_pad3_matches() {
    let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
    let y = functional_pad_1d_signed(&x, -2, 3, PaddingMode::Replicate, 0.0).unwrap();
    assert_eq!(y.shape(), &[1, 1, 5]);
    assert_close(
        y.data().unwrap(),
        &[3.0, 4.0, 4.0, 4.0, 4.0],
        1e-7,
        "replicate [-2,3] fwd",
    );
    let sum = ferrotorch_core::grad_fns::reduction::sum(&y).unwrap();
    ferrotorch_core::backward(&sum).unwrap();
    let g = x
        .grad()
        .unwrap()
        .expect("input gradient should be populated");
    assert_close(
        g.data().unwrap(),
        &[0.0, 0.0, 1.0, 4.0],
        1e-7,
        "replicate [-2,3] grad",
    );
}
