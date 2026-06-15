#![cfg(feature = "gpu")]

//! CUDA triangular/diag dtype-parity probes.
//!
//! PyTorch 2.11 CUDA oracle for this segment:
//! - `torch.triu` / `torch.tril` support float16, bfloat16, float32, float64.
//! - `torch.diag` supports CUDA float16/bfloat16 for both 1-D construction and
//!   2-D extraction, including empty out-of-range diagonals.
//! - VJPs are structural: triangular backward reapplies the same mask; 1-D
//!   `diag` construction gathers grad's diagonal; 2-D `diag` extraction scatters
//!   grad onto a zero matrix. All remain CUDA-resident until explicit readback.

use std::sync::Once;

use ferrotorch_core::autograd::graph::backward;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::device::Device;
use ferrotorch_core::linalg;
use ferrotorch_core::ops::tensor_ops::{diag, tril, triu};
use ferrotorch_core::tensor::Tensor;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for triangular/diag probes");
    });
}

fn cuda_f16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<half::f16> {
    let values = data.iter().copied().map(half::f16::from_f32).collect();
    from_vec::<half::f16>(values, shape)
        .expect("f16 CPU tensor")
        .to(Device::Cuda(0))
        .expect("upload f16")
        .requires_grad_(requires_grad)
}

fn cuda_bf16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<half::bf16> {
    let values = data.iter().copied().map(half::bf16::from_f32).collect();
    from_vec::<half::bf16>(values, shape)
        .expect("bf16 CPU tensor")
        .to(Device::Cuda(0))
        .expect("upload bf16")
        .requires_grad_(requires_grad)
}

fn cuda_f32(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    from_vec::<f32>(data.to_vec(), shape)
        .expect("f32 CPU tensor")
        .to(Device::Cuda(0))
        .expect("upload f32")
        .requires_grad_(requires_grad)
}

fn host_f16(t: &Tensor<half::f16>) -> Vec<f32> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "tensor must stay CUDA-resident until explicit readback"
    );
    t.cpu()
        .expect("D2H f16")
        .data_vec()
        .expect("f16 data")
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

fn host_bf16(t: &Tensor<half::bf16>) -> Vec<f32> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "tensor must stay CUDA-resident until explicit readback"
    );
    t.cpu()
        .expect("D2H bf16")
        .data_vec()
        .expect("bf16 data")
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

fn host_f32(t: &Tensor<f32>) -> Vec<f32> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "tensor must stay CUDA-resident until explicit readback"
    );
    t.cpu().expect("D2H f32").data_vec().expect("f32 data")
}

#[test]
fn triu_cuda_f16_forward_backward_resident() {
    ensure_cuda_backend();
    let x = cuda_f16(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
        true,
    );

    let out = triu(&x, 0).expect("CUDA f16 triu");
    assert!(out.is_cuda(), "forward output must stay CUDA-resident");
    assert_eq!(
        host_f16(&out),
        vec![1.0, 2.0, 3.0, 4.0, 0.0, 6.0, 7.0, 8.0, 0.0, 0.0, 11.0, 12.0]
    );

    backward(&out.sum_all().expect("f16 triu sum")).expect("f16 triu backward");
    let grad = x.grad().expect("grad read").expect("leaf grad");
    assert!(grad.is_cuda(), "triu grad must stay CUDA-resident");
    assert_eq!(
        host_f16(&grad),
        vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    );
}

#[test]
fn tril_cuda_bf16_forward_backward_resident() {
    ensure_cuda_backend();
    let x = cuda_bf16(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
        true,
    );

    let out = tril(&x, 1).expect("CUDA bf16 tril");
    assert!(out.is_cuda(), "forward output must stay CUDA-resident");
    assert_eq!(
        host_bf16(&out),
        vec![
            1.0, 2.0, 0.0, 0.0, 5.0, 6.0, 7.0, 0.0, 9.0, 10.0, 11.0, 12.0
        ]
    );

    backward(&out.sum_all().expect("bf16 tril sum")).expect("bf16 tril backward");
    let grad = x.grad().expect("grad read").expect("leaf grad");
    assert!(grad.is_cuda(), "tril grad must stay CUDA-resident");
    assert_eq!(
        host_bf16(&grad),
        vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    );
}

#[test]
fn triu_cuda_bf16_noncontiguous_view_packs_on_device() {
    ensure_cuda_backend();
    let base = cuda_bf16(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[4, 3],
        false,
    );
    let view = base.transpose(0, 1).expect("transpose CUDA view");
    assert!(
        !view.is_contiguous(),
        "probe must exercise non-contiguous packing"
    );

    let out = triu(&view, 0).expect("CUDA bf16 triu transposed view");
    assert!(out.is_cuda(), "view output must stay CUDA-resident");
    assert_eq!(
        host_bf16(&out),
        vec![
            1.0, 4.0, 7.0, 10.0, 0.0, 5.0, 8.0, 11.0, 0.0, 0.0, 9.0, 12.0
        ]
    );
}

#[test]
fn diag_cuda_f16_construct_forward_backward_resident() {
    ensure_cuda_backend();
    let x = cuda_f16(&[1.0, 2.0, 3.0], &[3], true);

    let out = diag(&x, 1).expect("CUDA f16 diag construct");
    assert!(
        out.is_cuda(),
        "diag construct output must stay CUDA-resident"
    );
    assert_eq!(out.shape(), &[4, 4]);
    assert_eq!(
        host_f16(&out),
        vec![
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0
        ]
    );

    backward(&out.sum_all().expect("f16 diag construct sum")).expect("f16 diag construct backward");
    let grad = x.grad().expect("grad read").expect("leaf grad");
    assert!(
        grad.is_cuda(),
        "diag construct grad must stay CUDA-resident"
    );
    assert_eq!(host_f16(&grad), vec![1.0, 1.0, 1.0]);
}

#[test]
fn diag_cuda_bf16_extract_forward_backward_rectangular_resident() {
    ensure_cuda_backend();
    let x = cuda_bf16(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
        true,
    );

    let out = diag(&x, 1).expect("CUDA bf16 diag extract");
    assert!(out.is_cuda(), "diag extract output must stay CUDA-resident");
    assert_eq!(out.shape(), &[3]);
    assert_eq!(host_bf16(&out), vec![2.0, 7.0, 12.0]);

    backward(&out.sum_all().expect("bf16 diag extract sum")).expect("bf16 diag extract backward");
    let grad = x.grad().expect("grad read").expect("leaf grad");
    assert!(grad.is_cuda(), "diag extract grad must stay CUDA-resident");
    assert_eq!(
        host_bf16(&grad),
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    );
}

#[test]
fn diag_cuda_f16_empty_extract_backward_is_zero_resident() {
    ensure_cuda_backend();
    let x = cuda_f16(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
        true,
    );

    let out = diag(&x, 99).expect("CUDA f16 empty diag extract");
    assert!(out.is_cuda(), "empty diag output must stay CUDA-resident");
    assert_eq!(out.shape(), &[0]);
    assert!(host_f16(&out).is_empty());

    backward(&out.sum_all().expect("f16 empty diag sum")).expect("empty diag backward");
    let grad = x.grad().expect("grad read").expect("leaf grad");
    assert!(grad.is_cuda(), "empty diag grad must stay CUDA-resident");
    assert_eq!(host_f16(&grad), vec![0.0; 12]);
}

#[test]
fn linalg_diagonal_cuda_f32_forward_backward_resident() {
    ensure_cuda_backend();
    let x = cuda_f32(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
        true,
    );

    let out = linalg::diagonal(&x, -1).expect("CUDA f32 linalg diagonal");
    assert!(
        out.is_cuda(),
        "linalg diagonal output must stay CUDA-resident"
    );
    assert_eq!(out.shape(), &[2]);
    assert_eq!(host_f32(&out), vec![5.0, 10.0]);

    backward(&out.sum_all().expect("f32 linalg diagonal sum")).expect("linalg diagonal backward");
    let grad = x.grad().expect("grad read").expect("leaf grad");
    assert!(
        grad.is_cuda(),
        "linalg diagonal grad must stay CUDA-resident"
    );
    assert_eq!(
        host_f32(&grad),
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    );
}
