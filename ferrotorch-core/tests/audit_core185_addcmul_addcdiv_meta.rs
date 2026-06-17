//! CORE-185 / crosslink #1879: `addcmul` and `addcdiv` must follow
//! PyTorch's ternary TensorIterator meta contract.
//!
//! PyTorch references inspected locally:
//! - `/home/doll/pytorch/aten/src/ATen/native/PointwiseOps.cpp`
//!   `TORCH_META_FUNC(addcmul)` and `TORCH_META_FUNC(addcdiv)` build a
//!   three-input TensorIterator.
//! - `/home/doll/pytorch/torch/_meta_registrations.py` registers both ops as
//!   ternary `elementwise_meta` calls.
//!
//! Live oracle on this machine (torch 2.11.0+cu130) confirmed:
//! - `(5,1,7), (3,1), (1,3,1) -> (5,3,7)`.
//! - scalar input with `(2,0,3), (1,3) -> (2,0,3)`.
//! - `requires_grad=True` meta inputs produce `AddcmulBackward0` /
//!   `AddcdivBackward0`.
//! - mixed meta/CPU inputs are rejected.

use ferrotorch_core::device::Device;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::grad_fns::arithmetic::{addcdiv, addcmul};
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::{Tensor, creation};

fn meta(shape: &[usize]) -> Tensor<f32> {
    creation::zeros_meta(shape).expect("meta tensor")
}

fn meta_leaf(shape: &[usize]) -> Tensor<f32> {
    meta(shape).requires_grad_(true)
}

fn cpu(shape: &[usize]) -> Tensor<f32> {
    creation::zeros(shape).expect("cpu tensor")
}

fn assert_plain_meta(t: &Tensor<f32>, shape: &[usize]) {
    assert!(t.is_meta(), "output must stay on meta");
    assert_eq!(t.shape(), shape);
    assert!(!t.requires_grad(), "untracked meta op should stay detached");
    assert!(t.is_leaf(), "untracked meta op should be a leaf");
    assert!(t.grad_fn().is_none(), "untracked meta op has grad_fn");
}

fn assert_meta_nonleaf(t: &Tensor<f32>, shape: &[usize], grad_fn: &str) {
    assert!(t.is_meta(), "{grad_fn} output must stay on meta");
    assert_eq!(t.shape(), shape);
    assert!(t.requires_grad(), "{grad_fn} output should require grad");
    assert!(!t.is_leaf(), "{grad_fn} output should be a non-leaf");
    assert_eq!(t.grad_fn().as_ref().map(|f| f.name()), Some(grad_fn));
}

fn assert_meta_grad(t: &Tensor<f32>, shape: &[usize]) {
    let grad = t.grad().expect("grad slot").expect("leaf gradient");
    assert!(grad.is_meta(), "leaf grad must stay on meta");
    assert_eq!(grad.shape(), shape);
    assert!(!grad.requires_grad(), "first-order leaf grad is detached");
}

fn assert_mixed_meta_cpu_rejected(result: FerrotorchResult<Tensor<f32>>, op: &str) {
    match result.expect_err("mixed meta/CPU inputs must error") {
        FerrotorchError::DeviceMismatch { expected, got } => {
            assert_eq!(expected, Device::Meta, "{op}: expected device");
            assert_eq!(got, Device::Cpu, "{op}: got device");
        }
        other => panic!("{op}: expected DeviceMismatch, got {other:?}"),
    }
}

#[test]
fn addcmul_addcdiv_meta_forward_uses_three_way_broadcast_without_data() {
    let input = meta(&[5, 1, 7]);
    let t1 = meta(&[3, 1]);
    let t2 = meta(&[1, 3, 1]);

    let cmul = addcmul(&input, &t1, &t2, 0.5).expect("addcmul meta");
    assert_plain_meta(&cmul, &[5, 3, 7]);

    let cdiv = addcdiv(&input, &t1, &t2, 0.5).expect("addcdiv meta");
    assert_plain_meta(&cdiv, &[5, 3, 7]);
}

#[test]
fn addcmul_meta_backward_reduces_broadcast_axes_shape_only() {
    let input = meta_leaf(&[5, 1, 7]);
    let t1 = meta_leaf(&[3, 1]);
    let t2 = meta_leaf(&[1, 3, 1]);

    let y = addcmul(&input, &t1, &t2, 0.5).expect("addcmul meta");
    assert_meta_nonleaf(&y, &[5, 3, 7], "AddcmulBackward");

    let loss = sum(&y).expect("sum addcmul meta");
    assert_meta_nonleaf(&loss, &[], "SumBackward");
    loss.backward().expect("addcmul meta backward");

    assert_meta_grad(&input, &[5, 1, 7]);
    assert_meta_grad(&t1, &[3, 1]);
    assert_meta_grad(&t2, &[1, 3, 1]);
}

#[test]
fn addcdiv_meta_backward_reduces_broadcast_axes_shape_only() {
    let input = meta_leaf(&[5, 1, 7]);
    let t1 = meta_leaf(&[3, 1]);
    let t2 = meta_leaf(&[1, 3, 1]);

    let y = addcdiv(&input, &t1, &t2, 0.5).expect("addcdiv meta");
    assert_meta_nonleaf(&y, &[5, 3, 7], "AddcdivBackward");

    let loss = sum(&y).expect("sum addcdiv meta");
    assert_meta_nonleaf(&loss, &[], "SumBackward");
    loss.backward().expect("addcdiv meta backward");

    assert_meta_grad(&input, &[5, 1, 7]);
    assert_meta_grad(&t1, &[3, 1]);
    assert_meta_grad(&t2, &[1, 3, 1]);
}

#[test]
fn addcmul_addcdiv_meta_zero_size_broadcast_matches_torch_shape() {
    let input = meta(&[]);
    let t1 = meta(&[2, 0, 3]);
    let t2 = meta(&[1, 3]);

    let cmul = addcmul(&input, &t1, &t2, 1.0).expect("addcmul zero-size meta");
    assert_plain_meta(&cmul, &[2, 0, 3]);
    assert_eq!(cmul.numel(), 0);

    let cdiv = addcdiv(&input, &t1, &t2, 1.0).expect("addcdiv zero-size meta");
    assert_plain_meta(&cdiv, &[2, 0, 3]);
    assert_eq!(cdiv.numel(), 0);
}

#[test]
fn addcmul_addcdiv_mixed_meta_cpu_reject_before_kernel_or_data_access() {
    let input = meta(&[2, 3]);
    let t1 = cpu(&[2, 3]);
    let t2 = meta(&[2, 3]);

    assert_mixed_meta_cpu_rejected(addcmul(&input, &t1, &t2, 1.0), "addcmul");
    assert_mixed_meta_cpu_rejected(addcdiv(&input, &t1, &t2, 1.0), "addcdiv");
}

#[test]
fn addcmul_addcdiv_meta_shape_mismatch_errors() {
    let input = meta(&[2, 3]);
    let t1 = meta(&[4, 3]);
    let t2 = meta(&[2, 3]);

    assert!(
        addcmul(&input, &t1, &t2, 1.0).is_err(),
        "addcmul incompatible meta shapes must error"
    );
    assert!(
        addcdiv(&input, &t1, &t2, 1.0).is_err(),
        "addcdiv incompatible meta shapes must error"
    );
}
