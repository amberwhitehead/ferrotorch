//! CUDA determinant-family parity probes.
//!
//! `torch.linalg.det` / `torch.linalg.slogdet` are LU-backed on CUDA. These
//! probes pin the same contract for ferrotorch: matrix values remain resident
//! on device, singular forward results match PyTorch, and invertible backward
//! uses resident `inv(A).T` formulas without CPU gradient construction.

#![cfg(feature = "gpu")]

use ferrotorch_core::autograd::graph::backward;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::device::Device;
use ferrotorch_core::grad_fns::linalg::{det_differentiable, slogdet_differentiable};
use ferrotorch_core::linalg::{det, slogdet};
use ferrotorch_core::tensor::Tensor;
use std::sync::Once;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for determinant probes");
    });
}

fn cuda_f32(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    from_vec::<f32>(data.to_vec(), shape)
        .expect("CPU tensor")
        .to(Device::Cuda(0))
        .expect("upload")
        .requires_grad_(requires_grad)
}

fn cuda_f64(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f64> {
    from_vec::<f64>(data.to_vec(), shape)
        .expect("CPU tensor")
        .to(Device::Cuda(0))
        .expect("upload")
        .requires_grad_(requires_grad)
}

fn host_f32(t: &Tensor<f32>) -> Vec<f32> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "tensor must stay CUDA-resident until explicit readback"
    );
    t.cpu().expect("D2H").data_vec().expect("data")
}

fn host_f64(t: &Tensor<f64>) -> Vec<f64> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "tensor must stay CUDA-resident until explicit readback"
    );
    t.cpu().expect("D2H").data_vec().expect("data")
}

fn assert_close_f32(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff <= tol,
            "value {idx}: actual {a}, expected {e}, diff {diff}, tol {tol}"
        );
    }
}

fn assert_close_f64(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff <= tol,
            "value {idx}: actual {a}, expected {e}, diff {diff}, tol {tol}"
        );
    }
}

fn assert_infinite_sign_f32(actual: f32, negative: bool, label: &str) {
    assert!(
        actual.is_infinite(),
        "{label}: expected infinity, got {actual}"
    );
    assert_eq!(
        actual.is_sign_negative(),
        negative,
        "{label}: wrong infinity sign for {actual}"
    );
}

#[test]
fn det_cuda_f32_forward_handles_pivot_sign_and_singular_resident() {
    ensure_cuda_backend();
    let a = cuda_f32(&[0.0, 2.0, 3.0, 4.0], &[2, 2], false);
    let d = det(&a).expect("CUDA det");
    assert!(d.is_cuda(), "det result must stay CUDA-resident");
    assert_close_f32(&host_f32(&d), &[-6.0], 1.0e-6);

    let singular = cuda_f32(&[1.0, 0.0, 0.0, 0.0], &[2, 2], false);
    let d0 = det(&singular).expect("singular CUDA det");
    assert!(d0.is_cuda(), "singular det result must stay CUDA-resident");
    assert_close_f32(&host_f32(&d0), &[0.0], 0.0);

    let negative_zero = cuda_f32(&[1.0, 2.0, 2.0, 4.0], &[2, 2], false);
    let d_neg0 = det(&negative_zero).expect("rank-1 CUDA det");
    let value = host_f32(&d_neg0)[0];
    assert!(
        value == 0.0 && value.is_sign_negative(),
        "rank-1 det should preserve PyTorch's signed -0.0, got {value:?}"
    );
}

#[test]
fn slogdet_cuda_f64_forward_uses_logabs_lu_diagonal_resident() {
    ensure_cuda_backend();
    let a = cuda_f64(&[0.0, 2.0, 3.0, 4.0], &[2, 2], false);
    let (sign, logabsdet) = slogdet(&a).expect("CUDA slogdet");
    assert!(sign.is_cuda(), "sign result must stay CUDA-resident");
    assert!(
        logabsdet.is_cuda(),
        "logabsdet result must stay CUDA-resident"
    );
    assert_close_f64(&host_f64(&sign), &[-1.0], 0.0);
    assert_close_f64(&host_f64(&logabsdet), &[6.0_f64.ln()], 1.0e-12);

    let singular = cuda_f64(&[1.0, 0.0, 0.0, 0.0], &[2, 2], false);
    let (s0, l0) = slogdet(&singular).expect("singular CUDA slogdet");
    assert!(s0.is_cuda(), "singular sign must stay CUDA-resident");
    assert!(l0.is_cuda(), "singular logabsdet must stay CUDA-resident");
    assert_close_f64(&host_f64(&s0), &[0.0], 0.0);
    let logabs = host_f64(&l0)[0];
    assert!(
        logabs.is_infinite() && logabs.is_sign_negative(),
        "singular slogdet logabsdet must be -inf, got {logabs}"
    );

    let negative_zero = cuda_f64(&[1.0, 2.0, 2.0, 4.0], &[2, 2], false);
    let (s_neg0, l_neg0) = slogdet(&negative_zero).expect("rank-1 CUDA slogdet");
    let sign = host_f64(&s_neg0)[0];
    assert!(
        sign == 0.0 && sign.is_sign_negative(),
        "rank-1 slogdet sign should preserve PyTorch's signed -0.0, got {sign:?}"
    );
    let logabs = host_f64(&l_neg0)[0];
    assert!(
        logabs.is_infinite() && logabs.is_sign_negative(),
        "rank-1 slogdet logabsdet must be -inf, got {logabs}"
    );
}

#[test]
fn det_differentiable_cuda_f32_backward_matches_pytorch_formula() {
    ensure_cuda_backend();
    let a = cuda_f32(&[2.0, 1.0, 3.0, 4.0], &[2, 2], true);

    let d = det_differentiable(&a).expect("CUDA det differentiable");
    assert!(
        d.is_cuda(),
        "det_differentiable output must stay CUDA-resident"
    );
    backward(&d).expect("CUDA det backward");

    let grad = a.grad().expect("grad handle").expect("a grad");
    assert!(grad.is_cuda(), "det gradient must stay CUDA-resident");
    assert_close_f32(&host_f32(&grad), &[4.0, -3.0, -1.0, 2.0], 1.0e-5);
}

#[test]
fn slogdet_cuda_f32_backward_matches_pytorch_formula_for_invertible_input() {
    ensure_cuda_backend();
    let a = cuda_f32(&[2.0, 1.0, 3.0, 4.0], &[2, 2], true);

    let (sign, logabsdet) = slogdet(&a).expect("CUDA slogdet differentiable");
    assert!(
        sign.grad_fn().is_none(),
        "real slogdet sign output is non-differentiable"
    );
    assert!(
        logabsdet.is_cuda(),
        "tracked logabsdet output must stay CUDA-resident"
    );
    backward(&logabsdet).expect("CUDA slogdet backward");

    let grad = a.grad().expect("grad handle").expect("a grad");
    assert!(grad.is_cuda(), "slogdet gradient must stay CUDA-resident");
    assert_close_f32(&host_f32(&grad), &[0.8, -0.6, -0.2, 0.4], 1.0e-5);
}

#[test]
fn slogdet_cuda_f32_singular_backward_matches_pytorch_nan_and_inf_contract() {
    ensure_cuda_backend();

    let rank_one_negative_zero = cuda_f32(&[1.0, 2.0, 2.0, 4.0], &[2, 2], true);
    let (_, logabsdet) =
        slogdet_differentiable(&rank_one_negative_zero).expect("rank-1 CUDA slogdet");
    backward(&logabsdet).expect("rank-1 CUDA slogdet backward");
    let grad = rank_one_negative_zero
        .grad()
        .expect("grad handle")
        .expect("rank-1 grad");
    assert!(
        grad.is_cuda(),
        "rank-1 slogdet grad must stay CUDA-resident"
    );
    let grad = host_f32(&grad);
    assert_infinite_sign_f32(grad[0], true, "grad[0,0]");
    assert_infinite_sign_f32(grad[1], false, "grad[0,1]");
    assert_infinite_sign_f32(grad[2], false, "grad[1,0]");
    assert_infinite_sign_f32(grad[3], true, "grad[1,1]");

    let diagonal_singular = cuda_f32(&[1.0, 0.0, 0.0, 0.0], &[2, 2], true);
    let (_, logabsdet) = slogdet(&diagonal_singular).expect("diagonal singular CUDA slogdet");
    backward(&logabsdet).expect("diagonal singular CUDA slogdet backward");
    let grad = diagonal_singular
        .grad()
        .expect("grad handle")
        .expect("diagonal singular grad");
    assert!(
        grad.is_cuda(),
        "diagonal singular slogdet grad must stay CUDA-resident"
    );
    let grad = host_f32(&grad);
    assert!(grad[0].is_nan(), "grad[0,0] should be NaN, got {}", grad[0]);
    assert!(grad[1].is_nan(), "grad[0,1] should be NaN, got {}", grad[1]);
    assert!(grad[2].is_nan(), "grad[1,0] should be NaN, got {}", grad[2]);
    assert_infinite_sign_f32(grad[3], false, "grad[1,1]");
}

#[test]
fn det_cuda_singular_backward_matches_pytorch_zero_and_one_by_one_cases() {
    ensure_cuda_backend();
    let singular = cuda_f32(&[1.0, 0.0, 0.0, 0.0], &[2, 2], true);
    let d0 = det(&singular).expect("tracked singular CUDA det");
    backward(&d0).expect("singular CUDA det backward");
    let grad = singular
        .grad()
        .expect("grad handle")
        .expect("singular grad");
    assert!(grad.is_cuda(), "singular det grad must stay CUDA-resident");
    assert_close_f32(&host_f32(&grad), &[0.0, 0.0, 0.0, 0.0], 0.0);

    let one = cuda_f32(&[0.0], &[1, 1], true);
    let d1 = det(&one).expect("tracked 1x1 singular CUDA det");
    backward(&d1).expect("1x1 singular CUDA det backward");
    let grad = one.grad().expect("grad handle").expect("1x1 grad");
    assert!(grad.is_cuda(), "1x1 det grad must stay CUDA-resident");
    assert_close_f32(&host_f32(&grad), &[1.0], 0.0);
}
