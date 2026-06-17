#![cfg(feature = "gpu")]

//! CORE-166 / #1860: repeated-index single-input einsum backward must keep
//! CUDA gradients on CUDA and match PyTorch's diagonal-backward semantics.

use std::sync::Once;

use ferrotorch_core::Tensor;
use ferrotorch_core::device::Device;
use ferrotorch_core::einsum::einsum_differentiable;
use ferrotorch_core::storage::TensorStorage;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for CORE-166 CUDA probes");
    });
}

fn cuda_leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu tensor")
        .to(Device::Cuda(0))
        .expect("upload")
        .detach()
        .requires_grad_(true)
}

fn cuda_seed(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu tensor")
        .to(Device::Cuda(0))
        .expect("upload")
}

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (idx, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= 1e-5,
            "{label}[{idx}]: got {a}, expected {e}, abs diff {}",
            (a - e).abs()
        );
    }
}

#[test]
fn cuda_diagonal_vector_backward_matches_torch_and_stays_cuda() {
    ensure_cuda_backend();

    let x = cuda_leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);
    let y = einsum_differentiable("ii->i", &[&x]).expect("einsum ii->i");
    assert_eq!(y.device(), Device::Cuda(0));
    assert_eq!(y.shape(), &[3]);

    // PyTorch 2026-06-17:
    // seed = [2, -3, 5]
    // torch.einsum("ii->i", torch.arange(1, 10, device="cuda").reshape(3,3))
    // .backward(seed) produces a CUDA grad with this flattened value.
    let seed = cuda_seed(&[2.0, -3.0, 5.0], &[3]);
    y.backward_with_gradient(&seed)
        .expect("backward with CUDA seed");

    let grad = x.grad().expect("grad result").expect("x grad");
    assert_eq!(grad.device(), Device::Cuda(0));
    assert_eq!(grad.shape(), &[3, 3]);
    assert_close(
        &grad.data_vec().expect("grad readback"),
        &[2.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 5.0],
        "ii->i grad",
    );
}

#[test]
fn cuda_mixed_repeat_backward_matches_torch_and_stays_cuda() {
    ensure_cuda_backend();

    let data: Vec<f32> = (1..=18).map(|v| v as f32).collect();
    let x = cuda_leaf(&data, &[3, 3, 2]);
    let y = einsum_differentiable("iij->j", &[&x]).expect("einsum iij->j");
    assert_eq!(y.device(), Device::Cuda(0));
    assert_eq!(y.shape(), &[2]);

    // PyTorch 2026-06-17 with seed [7, -11] gives the flattened grad below.
    let seed = cuda_seed(&[7.0, -11.0], &[2]);
    y.backward_with_gradient(&seed)
        .expect("backward with CUDA seed");

    let grad = x.grad().expect("grad result").expect("x grad");
    assert_eq!(grad.device(), Device::Cuda(0));
    assert_eq!(grad.shape(), &[3, 3, 2]);
    assert_close(
        &grad.data_vec().expect("grad readback"),
        &[
            7.0, -11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, -11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            7.0, -11.0,
        ],
        "iij->j grad",
    );
}

#[test]
fn cuda_separated_repeat_backward_matches_torch_and_stays_cuda() {
    ensure_cuda_backend();

    let data: Vec<f32> = (1..=48).map(|v| v as f32).collect();
    let seed_data: Vec<f32> = (1..=12).map(|v| v as f32).collect();
    let x = cuda_leaf(&data, &[2, 3, 4, 2]);
    let y = einsum_differentiable("ijki->jk", &[&x]).expect("einsum ijki->jk");
    assert_eq!(y.device(), Device::Cuda(0));
    assert_eq!(y.shape(), &[3, 4]);

    // PyTorch 2026-06-17 with seed torch.arange(1,13).reshape(3,4).
    let seed = cuda_seed(&seed_data, &[3, 4]);
    y.backward_with_gradient(&seed)
        .expect("backward with CUDA seed");

    let grad = x.grad().expect("grad result").expect("x grad");
    assert_eq!(grad.device(), Device::Cuda(0));
    assert_eq!(grad.shape(), &[2, 3, 4, 2]);
    assert_close(
        &grad.data_vec().expect("grad readback"),
        &[
            1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0, 0.0, 9.0,
            0.0, 10.0, 0.0, 11.0, 0.0, 12.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0,
            0.0, 6.0, 0.0, 7.0, 0.0, 8.0, 0.0, 9.0, 0.0, 10.0, 0.0, 11.0, 0.0, 12.0,
        ],
        "ijki->jk grad",
    );
}
