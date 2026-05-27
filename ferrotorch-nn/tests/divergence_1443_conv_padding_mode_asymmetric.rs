//! Divergence audit for commit 69525542c (#1443): padding_mode threading
//! through Conv1d/Conv3d + ConvTranspose validation.
//!
//! The shipped tests in `conv.rs` only exercise SYMMETRIC, single-dim padding
//! (Conv1d padding=1; Conv3d padding=(1,1,1)). The prompt-flagged risk classes
//! that a symmetric test cannot catch:
//!   1. `_reversed_padding_repeated_twice` per-dim ordering — a transposed pad
//!      mapping passes symmetric kernels but fails asymmetric per-dim padding.
//!   2. circular wrap-around direction (easy to get backwards).
//!   3. the backward adjoint VALUES (not just `grad_fn().is_some()`) for all
//!      three non-zero modes, including asymmetric cases — a wrong scatter-add
//!      transpose passes the presence check but yields wrong gradients.
//!
//! R-CHAR-3: every expected value below is from a live PyTorch 2.11.0+cu130
//! oracle (`torch.nn.Conv1d`/`Conv3d` forward + `out.sum().backward()`), NOT
//! copied from the ferrotorch side. The oracle scripts are reproduced inline in
//! each test's doc comment.
//!
//! Upstream sites mirrored:
//!   - torch/nn/modules/conv.py:367-378  Conv1d._conv_forward (F.pad branch)
//!   - torch/nn/modules/conv.py:716-732  Conv3d._conv_forward (F.pad branch)
//!   - torch/nn/modules/conv.py:137-159  _reverse_repeat_tuple (pad ordering)
//!   - torch/nn/modules/conv.py:755-758  _ConvTransposeNd.__init__ (ValueError)

use ferrotorch_core::Tensor;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_nn::module::Module;
use ferrotorch_nn::padding::PaddingMode;
use ferrotorch_nn::{Conv1d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d};

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

/// Build a Conv1d via the public `from_parts` API with explicit weight, no bias.
fn conv1d(
    weight: &[f32],
    wshape: &[usize],
    padding: usize,
    mode: PaddingMode,
) -> Conv1d<f32> {
    Conv1d::<f32>::from_parts(tensor(weight, wshape), None, 1, padding)
        .unwrap()
        .with_padding_mode(mode)
}

/// Build a Conv3d via the public `from_parts` API with explicit weight, no bias.
fn conv3d(
    weight: &[f32],
    wshape: &[usize],
    padding: (usize, usize, usize),
    mode: PaddingMode,
) -> Conv3d<f32> {
    Conv3d::<f32>::from_parts(tensor(weight, wshape), None, (1, 1, 1), padding)
        .unwrap()
        .with_padding_mode(mode)
}

// =====================================================================
// Conv1d — ASYMMETRIC padding (k=2, padding=2) — left/right pad regions
// pull different source elements; an asymmetric weight [1,10] makes the
// tap order observable. This catches a transposed/mis-ordered pad amount
// that a symmetric kernel hides.
// =====================================================================

/// torch (2.11.0+cu130):
///   c = nn.Conv1d(1,1,2,padding=2,padding_mode='reflect',bias=False)
///   c.weight = [[[1.,10.]]]; x = [[[1,2,3,4,5,6]]]
///   c(x) = [23,12,21,32,43,54,65,56,45]
#[test]
fn divergence_conv1d_reflect_asymmetric_forward() {
    let conv = conv1d(&[1.0, 10.0], &[1, 1, 2], 2, PaddingMode::Reflect);
    let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 6]);
    let y = conv.forward(&x).unwrap();
    assert_close(
        &y.data().unwrap(),
        &[23.0, 12.0, 21.0, 32.0, 43.0, 54.0, 65.0, 56.0, 45.0],
        1e-3,
        "Conv1d reflect padding=2 k=2 forward",
    );
}

/// torch: same layer, mode='replicate'
///   c(x) = [11,11,21,32,43,54,65,66,66]
#[test]
fn divergence_conv1d_replicate_asymmetric_forward() {
    let conv = conv1d(&[1.0, 10.0], &[1, 1, 2], 2, PaddingMode::Replicate);
    let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 6]);
    let y = conv.forward(&x).unwrap();
    assert_close(
        &y.data().unwrap(),
        &[11.0, 11.0, 21.0, 32.0, 43.0, 54.0, 65.0, 66.0, 66.0],
        1e-3,
        "Conv1d replicate padding=2 k=2 forward",
    );
}

/// torch: same layer, mode='circular'
///   c(x) = [65,16,21,32,43,54,65,16,21]  (the left pad wraps from the RIGHT
///   edge of the input). Direction-sensitive — catches a reversed wrap.
#[test]
fn divergence_conv1d_circular_asymmetric_forward() {
    let conv = conv1d(&[1.0, 10.0], &[1, 1, 2], 2, PaddingMode::Circular);
    let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 6]);
    let y = conv.forward(&x).unwrap();
    assert_close(
        &y.data().unwrap(),
        &[65.0, 16.0, 21.0, 32.0, 43.0, 54.0, 65.0, 16.0, 21.0],
        1e-3,
        "Conv1d circular padding=2 k=2 forward",
    );
}

// ---- Conv1d asymmetric BACKWARD (adjoint VALUE check, all 3 modes) ----

/// torch: out.sum().backward(), reflect padding=2 k=2 -> x.grad
///   = [11,22,12,21,22,11]
#[test]
fn divergence_conv1d_reflect_asymmetric_backward() {
    let conv = conv1d(&[1.0, 10.0], &[1, 1, 2], 2, PaddingMode::Reflect);
    let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 6]);
    let y = conv.forward(&x).unwrap();
    y.sum_all().unwrap().backward().unwrap();
    let g = x
        .grad()
        .unwrap()
        .expect("Conv1d reflect input grad must be populated");
    assert_close(
        &g.data().unwrap(),
        &[11.0, 22.0, 12.0, 21.0, 22.0, 11.0],
        1e-3,
        "Conv1d reflect padding=2 k=2 backward input grad",
    );
}

/// torch: replicate padding=2 k=2 -> x.grad = [23,11,11,11,11,32]
#[test]
fn divergence_conv1d_replicate_asymmetric_backward() {
    let conv = conv1d(&[1.0, 10.0], &[1, 1, 2], 2, PaddingMode::Replicate);
    let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 6]);
    let y = conv.forward(&x).unwrap();
    y.sum_all().unwrap().backward().unwrap();
    let g = x
        .grad()
        .unwrap()
        .expect("Conv1d replicate input grad must be populated");
    assert_close(
        &g.data().unwrap(),
        &[23.0, 11.0, 11.0, 11.0, 11.0, 32.0],
        1e-3,
        "Conv1d replicate padding=2 k=2 backward input grad",
    );
}

/// torch: circular padding=2 k=2 -> x.grad = [22,21,11,11,12,22]
#[test]
fn divergence_conv1d_circular_asymmetric_backward() {
    let conv = conv1d(&[1.0, 10.0], &[1, 1, 2], 2, PaddingMode::Circular);
    let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 6]);
    let y = conv.forward(&x).unwrap();
    y.sum_all().unwrap().backward().unwrap();
    let g = x
        .grad()
        .unwrap()
        .expect("Conv1d circular input grad must be populated");
    assert_close(
        &g.data().unwrap(),
        &[22.0, 21.0, 11.0, 11.0, 12.0, 22.0],
        1e-3,
        "Conv1d circular padding=2 k=2 backward input grad",
    );
}

// =====================================================================
// Conv3d — ASYMMETRIC per-dim padding (1,0,1) with a k=(1,1,1) weight=2.
// This is the per-dim ordering trap: padding (pd,ph,pw)=(1,0,1) maps to
// functional_pad_3d(pw=1,pw=1, ph=0,ph=0, pd=1,pd=1). If D and W pads are
// swapped, the output shape [1,1,4,2,5] still holds but values differ.
// =====================================================================

const X3D: [f32; 12] = [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
];

/// torch: Conv3d(1,1,(1,1,1),padding=(1,0,1),padding_mode='reflect',bias=False)
///   weight=[2.], x=arange(1..=12).reshape(1,1,2,2,3)
///   out shape [1,1,4,2,5], 40 elements (oracle vector below).
#[test]
fn divergence_conv3d_reflect_asymmetric_forward() {
    let conv = conv3d(&[2.0], &[1, 1, 1, 1, 1], (1, 0, 1), PaddingMode::Reflect);
    let x = leaf(&X3D, &[1, 1, 2, 2, 3]);
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.shape(), &[1, 1, 4, 2, 5]);
    assert_close(
        &y.data().unwrap(),
        &[
            16.0, 14.0, 16.0, 18.0, 16.0, 22.0, 20.0, 22.0, 24.0, 22.0, 4.0, 2.0, 4.0, 6.0, 4.0,
            10.0, 8.0, 10.0, 12.0, 10.0, 16.0, 14.0, 16.0, 18.0, 16.0, 22.0, 20.0, 22.0, 24.0,
            22.0, 4.0, 2.0, 4.0, 6.0, 4.0, 10.0, 8.0, 10.0, 12.0, 10.0,
        ],
        1e-3,
        "Conv3d reflect padding=(1,0,1) forward",
    );
}

/// torch: same layer mode='replicate' -> 40-element output below.
#[test]
fn divergence_conv3d_replicate_asymmetric_forward() {
    let conv = conv3d(&[2.0], &[1, 1, 1, 1, 1], (1, 0, 1), PaddingMode::Replicate);
    let x = leaf(&X3D, &[1, 1, 2, 2, 3]);
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.shape(), &[1, 1, 4, 2, 5]);
    assert_close(
        &y.data().unwrap(),
        &[
            2.0, 2.0, 4.0, 6.0, 6.0, 8.0, 8.0, 10.0, 12.0, 12.0, 2.0, 2.0, 4.0, 6.0, 6.0, 8.0,
            8.0, 10.0, 12.0, 12.0, 14.0, 14.0, 16.0, 18.0, 18.0, 20.0, 20.0, 22.0, 24.0, 24.0,
            14.0, 14.0, 16.0, 18.0, 18.0, 20.0, 20.0, 22.0, 24.0, 24.0,
        ],
        1e-3,
        "Conv3d replicate padding=(1,0,1) forward",
    );
}

/// torch: same layer mode='circular' -> 40-element output below.
#[test]
fn divergence_conv3d_circular_asymmetric_forward() {
    let conv = conv3d(&[2.0], &[1, 1, 1, 1, 1], (1, 0, 1), PaddingMode::Circular);
    let x = leaf(&X3D, &[1, 1, 2, 2, 3]);
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.shape(), &[1, 1, 4, 2, 5]);
    assert_close(
        &y.data().unwrap(),
        &[
            18.0, 14.0, 16.0, 18.0, 14.0, 24.0, 20.0, 22.0, 24.0, 20.0, 6.0, 2.0, 4.0, 6.0, 2.0,
            12.0, 8.0, 10.0, 12.0, 8.0, 18.0, 14.0, 16.0, 18.0, 14.0, 24.0, 20.0, 22.0, 24.0,
            20.0, 6.0, 2.0, 4.0, 6.0, 2.0, 12.0, 8.0, 10.0, 12.0, 8.0,
        ],
        1e-3,
        "Conv3d circular padding=(1,0,1) forward",
    );
}

// ---- Conv3d asymmetric BACKWARD (adjoint VALUE check, all 3 modes) ----

/// torch: out.sum().backward(), reflect padding=(1,0,1) ->
///   x.grad = [4,12,4, 4,12,4, 4,12,4, 4,12,4]
#[test]
fn divergence_conv3d_reflect_asymmetric_backward() {
    let conv = conv3d(&[2.0], &[1, 1, 1, 1, 1], (1, 0, 1), PaddingMode::Reflect);
    let x = leaf(&X3D, &[1, 1, 2, 2, 3]);
    let y = conv.forward(&x).unwrap();
    y.sum_all().unwrap().backward().unwrap();
    let g = x
        .grad()
        .unwrap()
        .expect("Conv3d reflect input grad must be populated");
    assert_close(
        &g.data().unwrap(),
        &[4.0, 12.0, 4.0, 4.0, 12.0, 4.0, 4.0, 12.0, 4.0, 4.0, 12.0, 4.0],
        1e-3,
        "Conv3d reflect padding=(1,0,1) backward input grad",
    );
}

/// torch: replicate padding=(1,0,1) ->
///   x.grad = [8,4,8, 8,4,8, 8,4,8, 8,4,8]
#[test]
fn divergence_conv3d_replicate_asymmetric_backward() {
    let conv = conv3d(&[2.0], &[1, 1, 1, 1, 1], (1, 0, 1), PaddingMode::Replicate);
    let x = leaf(&X3D, &[1, 1, 2, 2, 3]);
    let y = conv.forward(&x).unwrap();
    y.sum_all().unwrap().backward().unwrap();
    let g = x
        .grad()
        .unwrap()
        .expect("Conv3d replicate input grad must be populated");
    assert_close(
        &g.data().unwrap(),
        &[8.0, 4.0, 8.0, 8.0, 4.0, 8.0, 8.0, 4.0, 8.0, 8.0, 4.0, 8.0],
        1e-3,
        "Conv3d replicate padding=(1,0,1) backward input grad",
    );
}

/// torch: circular padding=(1,0,1) ->
///   x.grad = [8,4,8, 8,4,8, 8,4,8, 8,4,8]
#[test]
fn divergence_conv3d_circular_asymmetric_backward() {
    let conv = conv3d(&[2.0], &[1, 1, 1, 1, 1], (1, 0, 1), PaddingMode::Circular);
    let x = leaf(&X3D, &[1, 1, 2, 2, 3]);
    let y = conv.forward(&x).unwrap();
    y.sum_all().unwrap().backward().unwrap();
    let g = x
        .grad()
        .unwrap()
        .expect("Conv3d circular input grad must be populated");
    assert_close(
        &g.data().unwrap(),
        &[8.0, 4.0, 8.0, 8.0, 4.0, 8.0, 8.0, 4.0, 8.0, 8.0, 4.0, 8.0],
        1e-3,
        "Conv3d circular padding=(1,0,1) backward input grad",
    );
}

// =====================================================================
// ConvTranspose: non-zeros padding_mode rejected with byte-identical
// torch ValueError text, for all 3 layers x 3 non-zero modes.
// torch (2.11.0): str(ValueError) ==
//   'Only "zeros" padding mode is supported for ConvTranspose{N}d'
// =====================================================================

#[test]
fn divergence_conv_transpose_rejection_message_exact() {
    let modes = [
        PaddingMode::Reflect,
        PaddingMode::Replicate,
        PaddingMode::Circular,
    ];
    for &m in &modes {
        let e1 = ConvTranspose1d::<f32>::new(1, 1, 3, 1, 0, 0, false)
            .unwrap()
            .with_padding_mode(m)
            .unwrap_err();
        assert!(
            e1.to_string()
                .contains(r#"Only "zeros" padding mode is supported for ConvTranspose1d"#),
            "ConvTranspose1d msg mismatch: {e1}"
        );
        let e2 = ConvTranspose2d::<f32>::new(1, 1, (3, 3), (1, 1), (0, 0), (0, 0), false)
            .unwrap()
            .with_padding_mode(m)
            .unwrap_err();
        assert!(
            e2.to_string()
                .contains(r#"Only "zeros" padding mode is supported for ConvTranspose2d"#),
            "ConvTranspose2d msg mismatch: {e2}"
        );
        let e3 =
            ConvTranspose3d::<f32>::new(1, 1, (3, 3, 3), (1, 1, 1), (0, 0, 0), (0, 0, 0), false)
                .unwrap()
                .with_padding_mode(m)
                .unwrap_err();
        assert!(
            e3.to_string()
                .contains(r#"Only "zeros" padding mode is supported for ConvTranspose3d"#),
            "ConvTranspose3d msg mismatch: {e3}"
        );
    }
}
