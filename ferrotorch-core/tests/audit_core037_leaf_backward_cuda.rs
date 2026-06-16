#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::Tensor;
use ferrotorch_core::device::Device;
use ferrotorch_core::storage::TensorStorage;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for CORE-037 leaf-backward probes");
    });
}

fn cuda_leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap()
        .detach()
        .requires_grad_(true)
}

fn cuda_const(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap()
}

fn cpu_values(t: &Tensor<f32>) -> Vec<f32> {
    t.cpu().unwrap().data_vec().unwrap()
}

#[test]
fn cuda_scalar_leaf_root_backward_accumulates_cuda_grad() {
    ensure_cuda_backend();

    let x = cuda_leaf(&[2.0], &[]);

    x.backward().unwrap();

    let grad = x.grad().unwrap().expect("CUDA scalar leaf root grad");
    assert_eq!(grad.device(), Device::Cuda(0));
    assert_eq!(grad.shape(), &[] as &[usize]);
    assert!((cpu_values(&grad)[0] - 1.0).abs() < 1e-6);
}

#[test]
fn cuda_leaf_root_repeated_backward_accumulates_on_device() {
    ensure_cuda_backend();

    let x = cuda_leaf(&[2.0], &[]);

    x.backward().unwrap();
    x.backward().unwrap();

    let grad = x.grad().unwrap().expect("CUDA scalar leaf root grad");
    assert_eq!(grad.device(), Device::Cuda(0));
    assert!((cpu_values(&grad)[0] - 2.0).abs() < 1e-6);
}

#[test]
fn cuda_vector_leaf_root_external_gradient_stays_cuda() {
    ensure_cuda_backend();

    let x = cuda_leaf(&[1.0, 2.0], &[2]);
    let grad_seed = cuda_const(&[3.0, 4.0], &[2]);

    x.backward_with_gradient(&grad_seed).unwrap();

    let grad = x.grad().unwrap().expect("CUDA vector leaf root grad");
    assert_eq!(grad.device(), Device::Cuda(0));
    assert_eq!(grad.shape(), &[2]);
    assert_eq!(cpu_values(&grad), vec![3.0, 4.0]);
}
