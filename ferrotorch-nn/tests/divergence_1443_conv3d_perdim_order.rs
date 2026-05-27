//! Divergence audit for commit 69525542c (#1443): Conv3d per-dim pad ORDERING.
//!
//! The companion file's asymmetric Conv3d used padding (1,0,1) where pd == pw,
//! so a D<->W pad-amount swap (the `_reversed_padding_repeated_twice` ordering
//! trap, torch/nn/modules/conv.py:137-159) would be INVISIBLE. This file uses
//! padding (pd,ph,pw) = (1,0,2) where pd != pw, so the output shape
//! [1,1,4,2,7] AND the values uniquely pin which spatial dim received which
//! pad amount. A transposed mapping would change the shape/values.
//!
//! R-CHAR-3: expected values from a live PyTorch 2.11.0+cu130 oracle
//!   conv = nn.Conv3d(1,1,(1,1,1),padding=(1,0,2),padding_mode=M,bias=False)
//!   conv.weight = [2.]; x = arange(1..=12).reshape(1,1,2,2,3)
//! NOT copied from ferrotorch.

use ferrotorch_core::Tensor;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_nn::Conv3d;
use ferrotorch_nn::module::Module;
use ferrotorch_nn::padding::PaddingMode;

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
        "{msg}: length actual={} expected={}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{msg}: elem {i} actual={a} expected={e}\n actual={actual:?}\n expected={expected:?}"
        );
    }
}

fn conv3d(mode: PaddingMode) -> Conv3d<f32> {
    Conv3d::<f32>::from_parts(tensor(&[2.0], &[1, 1, 1, 1, 1]), None, (1, 1, 1), (1, 0, 2))
        .unwrap()
        .with_padding_mode(mode)
}

const X3D: [f32; 12] = [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
];

#[test]
fn divergence_conv3d_reflect_perdim_order_forward() {
    let y = conv3d(PaddingMode::Reflect)
        .forward(&leaf(&X3D, &[1, 1, 2, 2, 3]))
        .unwrap();
    assert_eq!(y.shape(), &[1, 1, 4, 2, 7]);
    assert_close(
        &y.data().unwrap(),
        &[
            18.0, 16.0, 14.0, 16.0, 18.0, 16.0, 14.0, 24.0, 22.0, 20.0, 22.0, 24.0, 22.0, 20.0,
            6.0, 4.0, 2.0, 4.0, 6.0, 4.0, 2.0, 12.0, 10.0, 8.0, 10.0, 12.0, 10.0, 8.0, 18.0, 16.0,
            14.0, 16.0, 18.0, 16.0, 14.0, 24.0, 22.0, 20.0, 22.0, 24.0, 22.0, 20.0, 6.0, 4.0, 2.0,
            4.0, 6.0, 4.0, 2.0, 12.0, 10.0, 8.0, 10.0, 12.0, 10.0, 8.0,
        ],
        1e-3,
        "Conv3d reflect padding=(1,0,2) forward",
    );
}

#[test]
fn divergence_conv3d_replicate_perdim_order_forward() {
    let y = conv3d(PaddingMode::Replicate)
        .forward(&leaf(&X3D, &[1, 1, 2, 2, 3]))
        .unwrap();
    assert_eq!(y.shape(), &[1, 1, 4, 2, 7]);
    assert_close(
        &y.data().unwrap(),
        &[
            2.0, 2.0, 2.0, 4.0, 6.0, 6.0, 6.0, 8.0, 8.0, 8.0, 10.0, 12.0, 12.0, 12.0, 2.0, 2.0,
            2.0, 4.0, 6.0, 6.0, 6.0, 8.0, 8.0, 8.0, 10.0, 12.0, 12.0, 12.0, 14.0, 14.0, 14.0, 16.0,
            18.0, 18.0, 18.0, 20.0, 20.0, 20.0, 22.0, 24.0, 24.0, 24.0, 14.0, 14.0, 14.0, 16.0,
            18.0, 18.0, 18.0, 20.0, 20.0, 20.0, 22.0, 24.0, 24.0, 24.0,
        ],
        1e-3,
        "Conv3d replicate padding=(1,0,2) forward",
    );
}

#[test]
fn divergence_conv3d_circular_perdim_order_forward() {
    let y = conv3d(PaddingMode::Circular)
        .forward(&leaf(&X3D, &[1, 1, 2, 2, 3]))
        .unwrap();
    assert_eq!(y.shape(), &[1, 1, 4, 2, 7]);
    assert_close(
        &y.data().unwrap(),
        &[
            16.0, 18.0, 14.0, 16.0, 18.0, 14.0, 16.0, 22.0, 24.0, 20.0, 22.0, 24.0, 20.0, 22.0,
            4.0, 6.0, 2.0, 4.0, 6.0, 2.0, 4.0, 10.0, 12.0, 8.0, 10.0, 12.0, 8.0, 10.0, 16.0, 18.0,
            14.0, 16.0, 18.0, 14.0, 16.0, 22.0, 24.0, 20.0, 22.0, 24.0, 20.0, 22.0, 4.0, 6.0, 2.0,
            4.0, 6.0, 2.0, 4.0, 10.0, 12.0, 8.0, 10.0, 12.0, 8.0, 10.0,
        ],
        1e-3,
        "Conv3d circular padding=(1,0,2) forward",
    );
}

/// Backward adjoint values for the per-dim-distinct case. reflect (1,0,2):
///   x.grad = [8,12,8, 8,12,8, 8,12,8, 8,12,8]
#[test]
fn divergence_conv3d_reflect_perdim_order_backward() {
    let x = leaf(&X3D, &[1, 1, 2, 2, 3]);
    let y = conv3d(PaddingMode::Reflect).forward(&x).unwrap();
    y.sum_all().unwrap().backward().unwrap();
    let g = x.grad().unwrap().expect("reflect input grad populated");
    assert_close(
        &g.data().unwrap(),
        &[
            8.0, 12.0, 8.0, 8.0, 12.0, 8.0, 8.0, 12.0, 8.0, 8.0, 12.0, 8.0,
        ],
        1e-3,
        "Conv3d reflect padding=(1,0,2) backward",
    );
}

/// replicate (1,0,2): x.grad = [12,4,12, 12,4,12, 12,4,12, 12,4,12]
#[test]
fn divergence_conv3d_replicate_perdim_order_backward() {
    let x = leaf(&X3D, &[1, 1, 2, 2, 3]);
    let y = conv3d(PaddingMode::Replicate).forward(&x).unwrap();
    y.sum_all().unwrap().backward().unwrap();
    let g = x.grad().unwrap().expect("replicate input grad populated");
    assert_close(
        &g.data().unwrap(),
        &[
            12.0, 4.0, 12.0, 12.0, 4.0, 12.0, 12.0, 4.0, 12.0, 12.0, 4.0, 12.0,
        ],
        1e-3,
        "Conv3d replicate padding=(1,0,2) backward",
    );
}

/// circular (1,0,2): x.grad = [8,12,8, 8,12,8, 8,12,8, 8,12,8]
#[test]
fn divergence_conv3d_circular_perdim_order_backward() {
    let x = leaf(&X3D, &[1, 1, 2, 2, 3]);
    let y = conv3d(PaddingMode::Circular).forward(&x).unwrap();
    y.sum_all().unwrap().backward().unwrap();
    let g = x.grad().unwrap().expect("circular input grad populated");
    assert_close(
        &g.data().unwrap(),
        &[
            8.0, 12.0, 8.0, 8.0, 12.0, 8.0, 8.0, 12.0, 8.0, 8.0, 12.0, 8.0,
        ],
        1e-3,
        "Conv3d circular padding=(1,0,2) backward",
    );
}
