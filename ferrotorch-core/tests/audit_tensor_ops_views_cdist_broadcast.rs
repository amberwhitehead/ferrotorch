//! PyTorch-parity audit for tensor_ops view handling and `cdist` broadcasting.
//!
//! Live torch 2.11.0+cu130 oracles:
//! - `base = torch.arange(12.).reshape(3,4); view = base.t()`
//! - `torch.triu(view,0).flatten()` ->
//!   `[0,4,8, 0,5,9, 0,0,10, 0,0,0]`
//! - `torch.tril(view,0).flatten()` ->
//!   `[0,0,0, 1,5,0, 2,6,10, 3,7,11]`
//! - `torch.diag(view,0)` -> `[0,5,10]`
//! - `torch.roll(view,1,0).flatten()` ->
//!   `[3,7,11, 0,4,8, 1,5,9, 2,6,10]`
//! - `torch.cdist(arange(24).reshape(2,3,4), arange(20).reshape(1,5,4))`
//!   has shape `[2,3,5]` and the values asserted below.

use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::grad_fns::shape::transpose_2d;
use ferrotorch_core::{Tensor, TensorStorage, cdist, diag, roll, tril, triu};

fn t(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("tensor")
}

fn assert_close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-5,
            "idx {i}: got {g}, want {w}; got={got:?}, want={want:?}"
        );
    }
}

#[test]
fn cpu_tensor_ops_accept_noncontiguous_transpose_views() {
    let base = t(
        &(0..12).map(|x| x as f32).collect::<Vec<_>>(),
        &[3, 4],
        false,
    );
    let view = transpose_2d(&base).expect("transpose");
    assert_eq!(view.shape(), &[4, 3]);
    assert!(!view.is_contiguous());

    assert_eq!(
        triu(&view, 0).expect("triu").data_vec().expect("data"),
        vec![0.0, 4.0, 8.0, 0.0, 5.0, 9.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(
        tril(&view, 0).expect("tril").data_vec().expect("data"),
        vec![0.0, 0.0, 0.0, 1.0, 5.0, 0.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0]
    );
    assert_eq!(
        diag(&view, 0).expect("diag").data_vec().expect("data"),
        vec![0.0, 5.0, 10.0]
    );
    assert_eq!(
        roll(&view, 1, 0).expect("roll").data_vec().expect("data"),
        vec![3.0, 7.0, 11.0, 0.0, 4.0, 8.0, 1.0, 5.0, 9.0, 2.0, 6.0, 10.0]
    );
}

#[test]
fn cpu_cdist_broadcasts_leading_batch_dims_like_torch() {
    let x1 = t(
        &(0..24).map(|x| x as f32).collect::<Vec<_>>(),
        &[2, 3, 4],
        false,
    );
    let x2 = t(
        &(0..20).map(|x| x as f32).collect::<Vec<_>>(),
        &[1, 5, 4],
        false,
    );
    let out = cdist(&x1, &x2, 2.0).expect("cdist");
    assert_eq!(out.shape(), &[2, 3, 5]);
    assert_close(
        &out.data_vec().expect("data"),
        &[
            0.0, 8.0, 16.0, 24.0, 32.0, 8.0, 0.0, 8.0, 16.0, 24.0, 16.0, 8.0, 0.0, 8.0, 16.0, 24.0,
            16.0, 8.0, 0.0, 8.0, 32.0, 24.0, 16.0, 8.0, 0.0, 40.0, 32.0, 24.0, 16.0, 8.0,
        ],
    );
}

#[test]
fn cpu_cdist_broadcast_backward_reduces_to_original_shapes() {
    let x1 = t(
        &(0..24).map(|x| x as f32).collect::<Vec<_>>(),
        &[2, 3, 4],
        true,
    );
    let x2 = t(
        &(0..20).map(|x| x as f32).collect::<Vec<_>>(),
        &[1, 5, 4],
        true,
    );
    let out = cdist(&x1, &x2, 2.0).expect("cdist");
    assert_eq!(out.shape(), &[2, 3, 5]);
    let loss = sum(&out).expect("sum");
    loss.backward().expect("backward");

    let gx1 = x1.grad().expect("grad slot").expect("x1 grad");
    let gx2 = x2.grad().expect("grad slot").expect("x2 grad");
    assert_eq!(gx1.shape(), &[2, 3, 4]);
    assert_eq!(gx2.shape(), &[1, 5, 4]);
    assert_close(
        &gx1.data_vec().expect("gx1"),
        &[
            -2.0, -2.0, -2.0, -2.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5, 2.5,
        ],
    );
    assert_close(
        &gx2.data_vec().expect("gx2"),
        &[
            -2.5, -2.5, -2.5, -2.5, -1.5, -1.5, -1.5, -1.5, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5,
            0.5, 1.5, 1.5, 1.5, 1.5,
        ],
    );
}
