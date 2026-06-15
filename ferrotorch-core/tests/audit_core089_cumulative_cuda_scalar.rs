#![cfg(feature = "gpu")]

//! CUDA scalar cumulative parity.
//!
//! PyTorch 2.11.0+cu130 keeps scalar cumulative outputs on CUDA and handles
//! 0-D views by reading the logical element at the view's storage offset:
//! `torch.as_strided(torch.tensor([2,-3.5,5], device="cuda"), (), (), 1)`.

use std::sync::Once;

use ferrotorch_core::creation::from_vec;
use ferrotorch_core::device::Device;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::grad_fns::cumulative::{cummax, cummin, cumprod, cumsum, logcumsumexp};
use ferrotorch_core::tensor::Tensor;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for cumulative CUDA tests");
    });
}

fn cuda<T: Float>(values: Vec<T>, shape: &[usize], requires_grad: bool) -> Tensor<T> {
    from_vec::<T>(values, shape)
        .expect("CPU tensor")
        .to(Device::Cuda(0))
        .expect("upload tensor")
        .requires_grad_(requires_grad)
}

fn scalar_offset_view<T: Float>(base: &Tensor<T>) -> Tensor<T> {
    base.as_strided(&[], &[], Some(1))
        .expect("0-D offset scalar view")
}

fn f32_values(t: &Tensor<f32>) -> Vec<f32> {
    assert_eq!(t.device(), Device::Cuda(0), "tensor must stay on CUDA");
    t.cpu()
        .expect("assertion readback")
        .data()
        .expect("data")
        .to_vec()
}

fn f64_values(t: &Tensor<f64>) -> Vec<f64> {
    assert_eq!(t.device(), Device::Cuda(0), "tensor must stay on CUDA");
    t.cpu()
        .expect("assertion readback")
        .data()
        .expect("data")
        .to_vec()
}

fn f16_bits(t: &Tensor<half::f16>) -> Vec<u16> {
    assert_eq!(t.device(), Device::Cuda(0), "f16 tensor must stay CUDA");
    t.cpu()
        .expect("assertion readback f16")
        .data()
        .expect("f16 data")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn bf16_bits(t: &Tensor<half::bf16>) -> Vec<u16> {
    assert_eq!(t.device(), Device::Cuda(0), "bf16 tensor must stay CUDA");
    t.cpu()
        .expect("assertion readback bf16")
        .data()
        .expect("bf16 data")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn assert_f32_scalar_grad(base: &Tensor<f32>) {
    let grad = base
        .grad()
        .expect("grad slot")
        .expect("base gradient must exist");
    assert_eq!(f32_values(&grad), vec![0.0, 1.0, 0.0]);
}

#[test]
fn cumulative_cuda_scalar_offset_views_match_torch_autograd() {
    ensure_cuda_backend();

    let base = cuda(vec![2.0_f32, -3.5, 5.0], &[3], true);
    let out = cumsum(&scalar_offset_view(&base), 0).expect("scalar cumsum");
    assert_eq!(out.shape(), &[]);
    assert_eq!(f32_values(&out), vec![-3.5]);
    out.backward().expect("cumsum backward");
    assert_f32_scalar_grad(&base);

    let base = cuda(vec![2.0_f32, -3.5, 5.0], &[3], true);
    let out = cumprod(&scalar_offset_view(&base), 0).expect("scalar cumprod");
    assert_eq!(out.shape(), &[]);
    assert_eq!(f32_values(&out), vec![-3.5]);
    out.backward().expect("cumprod backward");
    assert_f32_scalar_grad(&base);

    let base = cuda(vec![2.0_f32, -3.5, 5.0], &[3], true);
    let out = logcumsumexp(&scalar_offset_view(&base), 0).expect("scalar logcumsumexp");
    assert_eq!(out.shape(), &[]);
    assert_eq!(f32_values(&out), vec![-3.5]);
    out.backward().expect("logcumsumexp backward");
    assert_f32_scalar_grad(&base);

    let base = cuda(vec![2.0_f32, -3.5, 5.0], &[3], true);
    let result = cummax(&scalar_offset_view(&base), 0).expect("scalar cummax");
    assert_eq!(result.values.shape(), &[]);
    assert_eq!(f32_values(&result.values), vec![-3.5]);
    assert_eq!(result.indices, vec![0]);
    assert_eq!(result.indices_tensor.device(), Device::Cuda(0));
    result.values.backward().expect("cummax backward");
    assert_f32_scalar_grad(&base);

    let base = cuda(vec![2.0_f32, -3.5, 5.0], &[3], true);
    let result = cummin(&scalar_offset_view(&base), 0).expect("scalar cummin");
    assert_eq!(result.values.shape(), &[]);
    assert_eq!(f32_values(&result.values), vec![-3.5]);
    assert_eq!(result.indices, vec![0]);
    assert_eq!(result.indices_tensor.device(), Device::Cuda(0));
    result.values.backward().expect("cummin backward");
    assert_f32_scalar_grad(&base);
}

#[test]
fn cumulative_cuda_scalar_offset_views_cover_all_float_storage_dtypes() {
    ensure_cuda_backend();

    let base = cuda(vec![2.0_f64, -3.5, 5.0], &[3], false);
    let scalar = scalar_offset_view(&base);
    assert_eq!(
        f64_values(&cumsum(&scalar, 0).expect("f64 cumsum")),
        vec![-3.5]
    );
    assert_eq!(
        f64_values(&cumprod(&scalar, 0).expect("f64 cumprod")),
        vec![-3.5]
    );
    assert_eq!(
        f64_values(&logcumsumexp(&scalar, 0).expect("f64 logcumsumexp")),
        vec![-3.5]
    );
    let max = cummax(&scalar, 0).expect("f64 cummax");
    assert_eq!(f64_values(&max.values), vec![-3.5]);
    assert_eq!(max.indices_tensor.device(), Device::Cuda(0));
    let min = cummin(&scalar, 0).expect("f64 cummin");
    assert_eq!(f64_values(&min.values), vec![-3.5]);
    assert_eq!(min.indices_tensor.device(), Device::Cuda(0));

    let base = cuda(
        vec![
            half::f16::from_f32(2.0),
            half::f16::from_f32(-3.5),
            half::f16::from_f32(5.0),
        ],
        &[3],
        false,
    );
    let scalar = scalar_offset_view(&base);
    let neg_three_half = half::f16::from_f32(-3.5).to_bits();
    assert_eq!(
        f16_bits(&cumsum(&scalar, 0).expect("f16 cumsum")),
        vec![neg_three_half]
    );
    assert_eq!(
        f16_bits(&cumprod(&scalar, 0).expect("f16 cumprod")),
        vec![neg_three_half]
    );
    assert_eq!(
        f16_bits(&logcumsumexp(&scalar, 0).expect("f16 logcumsumexp")),
        vec![neg_three_half]
    );
    let max = cummax(&scalar, 0).expect("f16 cummax");
    assert_eq!(f16_bits(&max.values), vec![neg_three_half]);
    assert_eq!(max.indices_tensor.device(), Device::Cuda(0));
    let min = cummin(&scalar, 0).expect("f16 cummin");
    assert_eq!(f16_bits(&min.values), vec![neg_three_half]);
    assert_eq!(min.indices_tensor.device(), Device::Cuda(0));

    let base = cuda(
        vec![
            half::bf16::from_f32(2.0),
            half::bf16::from_f32(-3.5),
            half::bf16::from_f32(5.0),
        ],
        &[3],
        false,
    );
    let scalar = scalar_offset_view(&base);
    let neg_three_half = half::bf16::from_f32(-3.5).to_bits();
    assert_eq!(
        bf16_bits(&cumsum(&scalar, 0).expect("bf16 cumsum")),
        vec![neg_three_half]
    );
    assert_eq!(
        bf16_bits(&cumprod(&scalar, 0).expect("bf16 cumprod")),
        vec![neg_three_half]
    );
    assert_eq!(
        bf16_bits(&logcumsumexp(&scalar, 0).expect("bf16 logcumsumexp")),
        vec![neg_three_half]
    );
    let max = cummax(&scalar, 0).expect("bf16 cummax");
    assert_eq!(bf16_bits(&max.values), vec![neg_three_half]);
    assert_eq!(max.indices_tensor.device(), Device::Cuda(0));
    let min = cummin(&scalar, 0).expect("bf16 cummin");
    assert_eq!(bf16_bits(&min.values), vec![neg_three_half]);
    assert_eq!(min.indices_tensor.device(), Device::Cuda(0));
}
