//! CORE-019 (#1713): binary trailing-underscore ops must not silently detach
//! tracking source operands.

use ferrotorch_core::grad_fns::arithmetic::{add, sqrt};
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Tensor, TensorStorage};

fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

fn plain(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

#[cfg(feature = "gpu")]
fn plain64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn backward_sum(t: &Tensor<f32>) -> FerrotorchResult<()> {
    sum(t)?.backward()
}

fn assert_close(got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len());
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!((g - e).abs() < 1e-6, "at {i}: got {g}, expected {e}");
    }
}

fn assert_version_error(err: FerrotorchError) {
    let msg = err.to_string();
    assert!(
        msg.contains("modified by an inplace operation") && msg.contains("expected version"),
        "unexpected error: {msg}"
    );
}

#[test]
fn add_inplace_tracks_requires_grad_source() {
    let dest = plain(&[1.0, 2.0], &[2]);
    let source = leaf(&[3.0, 4.0], &[2]);

    dest.add_(&source).unwrap();

    assert!(dest.requires_grad());
    assert!(!dest.is_leaf());
    assert_eq!(dest.grad_fn().unwrap().name(), "AddBackward");
    assert_eq!(dest.data().unwrap(), &[4.0, 6.0]);

    backward_sum(&dest).unwrap();
    let grad = source.grad().unwrap().unwrap();
    assert_eq!(grad.data().unwrap(), &[1.0, 1.0]);
}

#[test]
fn sub_inplace_tracks_requires_grad_source() {
    let dest = plain(&[1.0, 2.0], &[2]);
    let source = leaf(&[3.0, 4.0], &[2]);

    dest.sub_(&source).unwrap();
    assert_eq!(dest.data().unwrap(), &[-2.0, -2.0]);

    backward_sum(&dest).unwrap();
    let grad = source.grad().unwrap().unwrap();
    assert_eq!(grad.data().unwrap(), &[-1.0, -1.0]);
}

#[test]
fn add_scaled_inplace_tracks_alpha_gradient_to_source() {
    let dest = plain(&[1.0, 2.0], &[2]);
    let source = leaf(&[3.0, 4.0], &[2]);

    dest.add_scaled_(&source, 2.0).unwrap();
    assert_eq!(dest.data().unwrap(), &[7.0, 10.0]);

    backward_sum(&dest).unwrap();
    let grad = source.grad().unwrap().unwrap();
    assert_eq!(grad.data().unwrap(), &[2.0, 2.0]);
}

#[test]
fn mul_inplace_uses_old_destination_value_for_source_gradient() {
    let dest = plain(&[1.0, 2.0], &[2]);
    let source = leaf(&[3.0, 4.0], &[2]);

    dest.mul_(&source).unwrap();
    assert_eq!(dest.data().unwrap(), &[3.0, 8.0]);

    backward_sum(&dest).unwrap();
    let grad = source.grad().unwrap().unwrap();
    assert_eq!(grad.data().unwrap(), &[1.0, 2.0]);
}

#[test]
fn div_inplace_uses_old_destination_value_for_source_gradient() {
    let dest = plain(&[1.0, 2.0], &[2]);
    let source = leaf(&[2.0, 4.0], &[2]);

    dest.div_(&source).unwrap();
    assert_eq!(dest.data().unwrap(), &[0.5, 0.5]);

    backward_sum(&dest).unwrap();
    let grad = source.grad().unwrap().unwrap();
    assert_close(grad.data().unwrap(), &[-0.25, -0.125]);
}

#[test]
fn binary_inplace_reduces_broadcast_source_gradient() {
    let dest = plain(&[1.0, 2.0], &[2]);
    let source = leaf(&[3.0], &[1]);

    dest.add_(&source).unwrap();
    assert_eq!(dest.data().unwrap(), &[4.0, 5.0]);

    backward_sum(&dest).unwrap();
    let grad = source.grad().unwrap().unwrap();
    assert_eq!(grad.data().unwrap(), &[2.0]);
}

#[test]
fn clone_alias_sees_rebased_autograd_metadata() {
    let dest = plain(&[1.0, 2.0], &[2]);
    let alias = dest.clone();
    let source = leaf(&[3.0, 4.0], &[2]);

    dest.add_(&source).unwrap();

    assert!(alias.requires_grad());
    assert!(!alias.is_leaf());
    assert_eq!(alias.grad_fn().unwrap().name(), "AddBackward");

    backward_sum(&alias).unwrap();
    let grad = source.grad().unwrap().unwrap();
    assert_eq!(grad.data().unwrap(), &[1.0, 1.0]);
}

#[test]
fn nonleaf_destination_rebases_through_previous_graph_and_source() {
    let x = leaf(&[4.0, 9.0], &[2]);
    let offset = plain(&[1.0, 1.0], &[2]);
    let dest = add(&x, &offset).unwrap();
    let source = leaf(&[2.0, 3.0], &[2]);

    dest.mul_(&source).unwrap();
    assert_eq!(dest.data().unwrap(), &[10.0, 30.0]);

    backward_sum(&dest).unwrap();
    let x_grad = x.grad().unwrap().unwrap();
    let source_grad = source.grad().unwrap().unwrap();
    assert_eq!(x_grad.data().unwrap(), &[2.0, 3.0]);
    assert_eq!(source_grad.data().unwrap(), &[5.0, 10.0]);
}

#[test]
fn output_saved_nonleaf_inplace_mutation_errors_like_pytorch() {
    let x = leaf(&[4.0, 9.0], &[2]);
    let dest = sqrt(&x).unwrap();
    let source = leaf(&[2.0, 3.0], &[2]);

    dest.mul_(&source).unwrap();

    let err = backward_sum(&dest).expect_err("sqrt saved output must be version-checked");
    assert_version_error(err);
}

#[test]
fn div_rounding_inplace_tracks_zero_gradient_to_source() {
    let dest = plain(&[5.0, -5.0, -7.0], &[3]);
    let source = leaf(&[2.0, 2.0, 0.7], &[3]);

    dest.div_rounding_(&source, "floor").unwrap();
    assert_eq!(dest.data().unwrap(), &[2.0, -3.0, -11.0]);

    backward_sum(&dest).unwrap();
    let grad = source.grad().unwrap().unwrap();
    assert_eq!(grad.data().unwrap(), &[0.0, 0.0, 0.0]);
}

#[cfg(feature = "gpu")]
#[test]
fn cuda_add_inplace_tracks_requires_grad_source_without_cpu_gradient() {
    use ferrotorch_core::Device;

    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend().expect("CUDA backend must initialise");
    });

    let dest = plain(&[1.0, 2.0], &[2]).to(Device::Cuda(0)).unwrap();
    let source = plain(&[3.0, 4.0], &[2])
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(true);

    dest.add_(&source).unwrap();
    assert!(dest.requires_grad());
    assert!(dest.is_cuda());

    backward_sum(&dest).unwrap();
    let grad = source.grad().unwrap().unwrap();
    assert!(grad.is_cuda(), "source gradient must stay on CUDA");
    assert_eq!(grad.data_vec().unwrap(), &[1.0, 1.0]);
}

#[cfg(feature = "gpu")]
#[test]
fn cuda_div_rounding_floor_tracks_zero_gradient_without_cpu_roundtrip() {
    use ferrotorch_core::Device;

    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend().expect("CUDA backend must initialise");
    });

    let dest = plain(&[5.0, -5.0, -7.0], &[3]).to(Device::Cuda(0)).unwrap();
    let source = plain(&[2.0, 2.0, 0.7], &[3])
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(true);

    dest.div_rounding_(&source, "floor").unwrap();
    assert!(dest.is_cuda(), "rounded result must stay on CUDA");
    assert_eq!(dest.data_vec().unwrap(), &[2.0, -3.0, -11.0]);

    backward_sum(&dest).unwrap();
    let grad = source.grad().unwrap().unwrap();
    assert!(grad.is_cuda(), "source gradient must stay on CUDA");
    assert_eq!(grad.data_vec().unwrap(), &[0.0, 0.0, 0.0]);
}

#[cfg(feature = "gpu")]
#[test]
fn cuda_div_rounding_f64_trunc_stays_resident() {
    use ferrotorch_core::Device;
    use ferrotorch_core::grad_fns::reduction::sum;

    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend().expect("CUDA backend must initialise");
    });

    let dest = plain64(&[5.0, -5.0], &[2]).to(Device::Cuda(0)).unwrap();
    let source = plain64(&[2.0, 2.0], &[2])
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(true);

    dest.div_rounding_(&source, "trunc").unwrap();
    assert!(dest.is_cuda(), "rounded f64 result must stay on CUDA");
    assert_eq!(dest.data_vec().unwrap(), &[2.0, -2.0]);

    sum(&dest).unwrap().backward().unwrap();
    let grad = source.grad().unwrap().unwrap();
    assert!(grad.is_cuda(), "f64 source gradient must stay on CUDA");
    assert_eq!(grad.data_vec().unwrap(), &[0.0, 0.0]);
}
