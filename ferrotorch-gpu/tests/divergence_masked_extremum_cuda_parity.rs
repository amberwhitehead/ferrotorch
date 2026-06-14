#![cfg(feature = "cuda")]

//! CUDA lane for the `torch.masked.amin/amax` all-masked and empty contracts.
//! PyTorch returns identity payloads on the input device for all-masked
//! non-empty inputs and raises on empty inputs.

use ferrotorch_core::masked::{MaskedTensor, masked_max, masked_min};
use ferrotorch_core::{Device, Tensor, TensorStorage};
use ferrotorch_gpu::init_cuda_backend;
use std::sync::Once;

static INIT: Once = Once::new();

fn ensure_cuda() {
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn cuda_f32(data: &[f32], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
        .expect("cpu tensor")
        .to(Device::Cuda(0))
        .expect("to cuda")
        .requires_grad_(requires_grad)
}

fn cuda_f64(data: &[f64], requires_grad: bool) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
        .expect("cpu tensor")
        .to(Device::Cuda(0))
        .expect("to cuda")
        .requires_grad_(requires_grad)
}

fn host_f32(t: &Tensor<f32>) -> Vec<f32> {
    t.cpu().expect("to cpu").data_vec().expect("data")
}

fn host_f64(t: &Tensor<f64>) -> Vec<f64> {
    t.cpu().expect("to cpu").data_vec().expect("data")
}

#[test]
fn cuda_all_masked_extrema_return_identity_payloads_on_device() {
    ensure_cuda();

    let x32 = cuda_f32(&[1.0, 2.0], false);
    let mt32 = MaskedTensor::new(x32, vec![false, false]).expect("masked tensor");
    let mn32 = masked_min(&mt32).expect("amin f32");
    let mx32 = masked_max(&mt32).expect("amax f32");
    assert!(
        mn32.is_cuda() && mx32.is_cuda(),
        "f32 outputs must stay CUDA"
    );
    assert_eq!(host_f32(&mn32), vec![f32::INFINITY]);
    assert_eq!(host_f32(&mx32), vec![f32::NEG_INFINITY]);

    let x64 = cuda_f64(&[1.0, 2.0], false);
    let mt64 = MaskedTensor::new(x64, vec![false, false]).expect("masked tensor");
    let mn64 = masked_min(&mt64).expect("amin f64");
    let mx64 = masked_max(&mt64).expect("amax f64");
    assert!(
        mn64.is_cuda() && mx64.is_cuda(),
        "f64 outputs must stay CUDA"
    );
    assert_eq!(host_f64(&mn64), vec![f64::INFINITY]);
    assert_eq!(host_f64(&mx64), vec![f64::NEG_INFINITY]);
}

#[test]
fn cuda_empty_extrema_error_like_torch() {
    ensure_cuda();
    let x = cuda_f32(&[], false);
    let mt = MaskedTensor::new(x, vec![]).expect("empty masked tensor");

    assert!(
        masked_min(&mt).is_err(),
        "torch.masked.amin raises on empty CUDA input"
    );
    assert!(
        masked_max(&mt).is_err(),
        "torch.masked.amax raises on empty CUDA input"
    );
}

#[test]
fn cuda_all_masked_extrema_backward_zero_grad_stays_device() {
    ensure_cuda();
    let x = cuda_f32(&[1.0, 2.0], true);
    let mt = MaskedTensor::new(x.clone(), vec![false, false]).expect("masked tensor");
    let y = masked_max(&mt).expect("amax");
    assert!(y.is_cuda(), "forward must stay CUDA");
    assert_eq!(host_f32(&y), vec![f32::NEG_INFINITY]);

    y.backward().expect("backward");
    let grad = x.grad().expect("grad access").expect("leaf grad");
    assert!(grad.is_cuda(), "grad must stay CUDA");
    assert_eq!(host_f32(&grad), vec![0.0, 0.0]);
}
