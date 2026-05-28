//! Divergence audit for commit f512e20a3 (#1611): negative (crop) functional
//! pad for `mode="constant"`.
//!
//! DIVERGENCE FOUND. The commit message, the design doc
//! (`.design/ferrotorch-nn/padding.md` REQ-2 / parity contract), and the impl
//! (`functional_pad_nd_signed` in `ferrotorch-nn/src/padding.rs`) all claim that
//! NEGATIVE (crop) pads are a `mode="constant"`-only capability and that
//! reflect / replicate / circular REJECT a negative pad, "matching torch's
//! reflection_pad*/replication_pad* kernels". That claim is FALSE against live
//! PyTorch 2.11.0+cu130.
//!
//! Live torch `F.pad(x, [-1, 0], mode="reflect")` (and replicate / circular)
//! does NOT raise — it CROPS that side and returns a smaller tensor, with the
//! adjoint passing the grad through to the surviving positions. ferrotorch's
//! `functional_pad_1d_signed(.., PaddingMode::Reflect, ..)` returns
//! `Err(InvalidArgument)` for the same input. ferrotorch errors where torch
//! produces a valid cropped tensor — a hard observable divergence.
//!
//! WHY the commit got it wrong: it cited only the C++ `_pad_enum`
//! (`aten/src/ATen/native/PadNd.cpp:207-242`), which dispatches reflect/
//! replicate/circular straight to the native `reflection_pad*` /
//! `replication_pad*` kernels — and concluded those kernels "do not accept a
//! negative pad". But the native kernels DO narrow for negative pads; the
//! observable end-to-end `F.pad(...)` behavior (verified below by a live oracle)
//! crops. The C++ source the commit read does not show a rejection path for
//! negative pads under non-constant modes; the builder assumed one exists.
//!
//! R-CHAR-3: every expected value below is from a live PyTorch 2.11.0+cu130
//! oracle. The exact reproducing Python is in each test's doc comment. NONE of
//! these expected values are copied from the ferrotorch side (the ferrotorch
//! side returns an error for every one of these calls).
//!
//! Tracking: #1620. The divergence is real production work for the generator
//! (`functional_pad_nd_signed` must crop, not error, under reflect/replicate/
//! circular); these tests are `#[ignore]`d so the tracked issue, not a red CI
//! bar, drives the fix.

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
        "{msg}: length mismatch actual={} expected={}",
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

/// DIVERGENCE: reflect mode with a left crop.
///
/// Live torch 2.11:
/// ```python
/// x = torch.tensor([[[1.,2.,3.,4.,5.]]])
/// y = F.pad(x, [-1, 0], mode="reflect")   # shape [1,1,4], out [2,3,4,5]
/// ```
/// torch CROPS the leftmost element. ferrotorch returns Err(InvalidArgument).
///
/// Tracking: #1620
#[test]
fn divergence_reflect_negative_pad_crops_not_errors() {
    let x = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5]);
    let y = functional_pad_1d_signed(&x, -1, 0, PaddingMode::Reflect, 0.0)
        .expect("torch F.pad reflect [-1,0] CROPS (returns [2,3,4,5]); ferrotorch must not error");
    assert_eq!(y.shape(), &[1, 1, 4], "torch crops left -> shape [1,1,4]");
    assert_close(
        y.data().unwrap(),
        &[2.0, 3.0, 4.0, 5.0],
        1e-7,
        "torch reflect [-1,0]",
    );
}

/// DIVERGENCE: replicate mode with mixed signs (replicate-pad left, crop right).
///
/// Live torch 2.11:
/// ```python
/// x = torch.tensor([[[1.,2.,3.,4.,5.]]]).clone().requires_grad_(True)
/// y = F.pad(x, [1, -1], mode="replicate")  # shape [1,1,5], out [1,1,2,3,4]
/// y.sum().backward()                        # grad [2,1,1,1,0]
/// ```
/// torch replicate-pads the left side (extra reference to element 0 -> grad 2)
/// and crops the right. ferrotorch returns Err(InvalidArgument).
///
/// Tracking: #1620
#[test]
fn divergence_replicate_mixed_sign_crops_not_errors() {
    let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5]);
    let y = functional_pad_1d_signed(&x, 1, -1, PaddingMode::Replicate, 0.0)
        .expect("torch F.pad replicate [1,-1] returns [1,1,2,3,4]; ferrotorch must not error");
    assert_eq!(
        y.shape(),
        &[1, 1, 5],
        "torch replicate [1,-1] -> shape [1,1,5]"
    );
    assert_close(
        y.data().unwrap(),
        &[1.0, 1.0, 2.0, 3.0, 4.0],
        1e-7,
        "torch replicate [1,-1] forward",
    );

    let sum = ferrotorch_core::grad_fns::reduction::sum(&y).unwrap();
    ferrotorch_core::backward(&sum).unwrap();
    let g = x.grad().unwrap().expect("grad must be populated");
    assert_close(
        g.data().unwrap(),
        &[2.0, 1.0, 1.0, 1.0, 0.0],
        1e-7,
        "torch replicate [1,-1] grad",
    );
}

/// DIVERGENCE: circular mode with a left crop.
///
/// Live torch 2.11:
/// ```python
/// x = torch.tensor([[[1.,2.,3.,4.,5.]]]).clone().requires_grad_(True)
/// y = F.pad(x, [-1, 0], mode="circular")   # shape [1,1,4], out [2,3,4,5]
/// y.sum().backward()                        # grad [0,1,1,1,1]
/// ```
/// torch CROPS the leftmost element. ferrotorch returns Err(InvalidArgument).
///
/// Tracking: #1620
#[test]
fn divergence_circular_negative_pad_crops_not_errors() {
    let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5]);
    let y = functional_pad_1d_signed(&x, -1, 0, PaddingMode::Circular, 0.0)
        .expect("torch F.pad circular [-1,0] CROPS (returns [2,3,4,5]); ferrotorch must not error");
    assert_eq!(
        y.shape(),
        &[1, 1, 4],
        "torch circular [-1,0] -> shape [1,1,4]"
    );
    assert_close(
        y.data().unwrap(),
        &[2.0, 3.0, 4.0, 5.0],
        1e-7,
        "torch circular [-1,0] forward",
    );

    let sum = ferrotorch_core::grad_fns::reduction::sum(&y).unwrap();
    ferrotorch_core::backward(&sum).unwrap();
    let g = x.grad().unwrap().expect("grad must be populated");
    assert_close(
        g.data().unwrap(),
        &[0.0, 1.0, 1.0, 1.0, 1.0],
        1e-7,
        "torch circular [-1,0] grad",
    );
}

/// DIVERGENCE: 2-D reflect with a per-dim mixed crop/pad on the last dim.
///
/// Live torch 2.11:
/// ```python
/// x = torch.arange(1,10,dtype=torch.float32).reshape(1,1,3,3)
/// y = F.pad(x, [-1, 1, 0, 0], mode="reflect")
/// # shape [1,1,3,3], out [[2,3,2],[5,6,5],[8,9,8]]
/// ```
/// torch crops the leftmost column then reflection-pads one on the right.
/// ferrotorch's 2d signed path returns Err(InvalidArgument) the moment any pad
/// is negative under a non-constant mode.
///
/// Tracking: #1620
#[test]
fn divergence_reflect2d_mixed_sign_crops_not_errors() {
    let x = tensor(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[1, 1, 3, 3],
    );
    let y = functional_pad_2d_signed(&x, -1, 1, 0, 0, PaddingMode::Reflect, 0.0)
        .expect("torch F.pad reflect2d [-1,1,0,0] crops+reflects; ferrotorch must not error");
    assert_eq!(
        y.shape(),
        &[1, 1, 3, 3],
        "torch reflect2d [-1,1,0,0] keeps W=3"
    );
    assert_close(
        y.data().unwrap(),
        &[2.0, 3.0, 2.0, 5.0, 6.0, 5.0, 8.0, 9.0, 8.0],
        1e-7,
        "torch reflect2d [-1,1,0,0] forward",
    );
}
