#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::creation::from_vec;
use ferrotorch_core::device::Device;
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::grad_fns::transcendental::{clamp, clamp_max, clamp_min, clamp_opt};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use half::{bf16, f16};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend().expect("CUDA backend must initialize");
    });
}

fn cuda<T: ferrotorch_core::dtype::Float>(values: Vec<T>, shape: &[usize]) -> Tensor<T> {
    from_vec::<T>(values, shape)
        .expect("CPU tensor")
        .to(Device::Cuda(0))
        .expect("upload tensor")
}

fn tracked_cuda<T: ferrotorch_core::dtype::Float>(
    values: Vec<T>,
    shape: &[usize],
) -> (Tensor<T>, Tensor<T>) {
    let leaf =
        Tensor::from_storage(TensorStorage::cpu(values), shape.to_vec(), true).expect("CPU leaf");
    let gpu = leaf.to(Device::Cuda(0)).expect("upload tracked tensor");
    (leaf, gpu)
}

fn cpu_data<T: ferrotorch_core::dtype::Float>(tensor: &Tensor<T>) -> Vec<T> {
    assert_eq!(tensor.device(), Device::Cuda(0), "tensor must stay CUDA");
    tensor
        .to(Device::Cpu)
        .expect("download tensor")
        .data_vec()
        .expect("CPU data")
}

#[test]
fn inplace_clamp_cuda_min_greater_than_max_stays_resident() {
    ensure_cuda_backend();
    let x = cuda(vec![-1.0_f32, 0.5, 2.0], &[3]);

    x.clamp_(5.0, 1.0)
        .expect("torch scalar clamp accepts min > max");

    assert_eq!(x.device(), Device::Cuda(0));
    assert_eq!(cpu_data(&x), vec![1.0, 1.0, 1.0]);
}

#[test]
fn inplace_clamp_cuda_nan_bound_fills_resident_f32_and_f64() {
    ensure_cuda_backend();
    let x = cuda(vec![1.0_f32, 2.0, 3.0], &[3]);
    x.clamp_(0.0, f32::NAN)
        .expect("torch scalar clamp accepts NaN bound");
    assert!(cpu_data(&x).iter().all(|v| v.is_nan()));

    let y = cuda(vec![1.0_f64, 2.0, 3.0], &[3]);
    y.clamp_opt_(Some(f64::NAN), Some(10.0))
        .expect("torch scalar clamp accepts NaN bound");
    assert!(cpu_data(&y).iter().all(|v| v.is_nan()));
}

#[test]
fn out_of_place_clamp_cuda_degenerate_bounds_matches_torch_and_stays_resident() {
    ensure_cuda_backend();
    let x = cuda(vec![-1.0_f32, 0.5, 2.0], &[3]);

    let out = clamp(&x, 5.0, 1.0).expect("torch scalar clamp accepts min > max");

    assert_eq!(out.device(), Device::Cuda(0));
    assert_eq!(cpu_data(&out), vec![1.0, 1.0, 1.0]);
}

#[test]
fn out_of_place_clamp_cuda_nan_bound_fills_and_stays_resident() {
    ensure_cuda_backend();
    let x = cuda(vec![1.0_f32, 2.0, 3.0], &[3]);

    let out = clamp(&x, f32::NAN, 10.0).expect("torch scalar clamp accepts NaN bound");

    assert_eq!(out.device(), Device::Cuda(0));
    assert!(cpu_data(&out).iter().all(|v| v.is_nan()));
}

#[test]
fn out_of_place_clamp_opt_cuda_one_sided_stays_resident() {
    ensure_cuda_backend();
    let x = cuda(vec![-1.0_f32, 0.0, 0.5, 2.0], &[4]);

    let lower = clamp_opt(&x, Some(0.0), None).expect("torch clamp(min=...) accepts one bound");
    assert_eq!(lower.device(), Device::Cuda(0));
    assert_eq!(cpu_data(&lower), vec![0.0, 0.0, 0.5, 2.0]);

    let upper = clamp_opt(&x, None, Some(1.0)).expect("torch clamp(max=...) accepts one bound");
    assert_eq!(upper.device(), Device::Cuda(0));
    assert_eq!(cpu_data(&upper), vec![-1.0, 0.0, 0.5, 1.0]);
}

#[test]
fn out_of_place_clamp_opt_cuda_one_sided_nan_preserves_forward_values() {
    ensure_cuda_backend();
    let x = cuda(vec![-1.0_f32, 0.5, 2.0], &[3]);

    let out =
        clamp_opt(&x, Some(f32::NAN), None).expect("torch clamp(min=nan) accepts one-sided NaN");

    assert_eq!(out.device(), Device::Cuda(0));
    assert_eq!(cpu_data(&out), vec![-1.0, 0.5, 2.0]);
}

#[test]
fn out_of_place_clamp_min_max_cuda_nan_fill_split() {
    ensure_cuda_backend();
    let x = cuda(vec![-1.0_f32, 0.5, 2.0], &[3]);

    let min_out = clamp_min(&x, f32::NAN).expect("torch.clamp_min accepts NaN");
    assert_eq!(min_out.device(), Device::Cuda(0));
    assert!(cpu_data(&min_out).iter().all(|v| v.is_nan()));

    let max_out = clamp_max(&x, f32::NAN).expect("torch.clamp_max accepts NaN");
    assert_eq!(max_out.device(), Device::Cuda(0));
    assert!(cpu_data(&max_out).iter().all(|v| v.is_nan()));
}

#[test]
fn out_of_place_clamp_cuda_half_and_bfloat_stay_resident() {
    ensure_cuda_backend();
    let h = cuda(
        vec![f16::from_f32(-1.0), f16::from_f32(0.5), f16::from_f32(2.0)],
        &[3],
    );
    let h_out = clamp_opt(&h, Some(f16::from_f32(0.0)), Some(f16::from_f32(1.0)))
        .expect("f16 CUDA clamp must use resident kernel");
    assert_eq!(h_out.device(), Device::Cuda(0));
    assert_eq!(
        cpu_data(&h_out),
        vec![f16::from_f32(0.0), f16::from_f32(0.5), f16::from_f32(1.0)]
    );

    let b = cuda(
        vec![
            bf16::from_f32(-1.0),
            bf16::from_f32(0.5),
            bf16::from_f32(2.0),
        ],
        &[3],
    );
    let b_out = clamp_opt(&b, Some(bf16::from_f32(0.0)), Some(bf16::from_f32(1.0)))
        .expect("bf16 CUDA clamp must use resident kernel");
    assert_eq!(b_out.device(), Device::Cuda(0));
    assert_eq!(
        cpu_data(&b_out),
        vec![
            bf16::from_f32(0.0),
            bf16::from_f32(0.5),
            bf16::from_f32(1.0)
        ]
    );
}

#[test]
fn inplace_clamp_cuda_half_and_bfloat_stay_resident() {
    ensure_cuda_backend();
    let h = cuda(
        vec![f16::from_f32(-1.0), f16::from_f32(0.5), f16::from_f32(2.0)],
        &[3],
    );
    h.clamp_opt_(Some(f16::from_f32(0.0)), Some(f16::from_f32(1.0)))
        .expect("f16 in-place clamp must use resident kernel");
    assert_eq!(h.device(), Device::Cuda(0));
    assert_eq!(
        cpu_data(&h),
        vec![f16::from_f32(0.0), f16::from_f32(0.5), f16::from_f32(1.0)]
    );

    let b = cuda(
        vec![
            bf16::from_f32(-1.0),
            bf16::from_f32(0.5),
            bf16::from_f32(2.0),
        ],
        &[3],
    );
    b.clamp_opt_(None, Some(bf16::from_f32(1.0)))
        .expect("bf16 in-place clamp_max path must use resident kernel");
    assert_eq!(b.device(), Device::Cuda(0));
    assert_eq!(
        cpu_data(&b),
        vec![
            bf16::from_f32(-1.0),
            bf16::from_f32(0.5),
            bf16::from_f32(1.0)
        ]
    );
}

#[test]
fn clamp_backward_cuda_half_and_bfloat_match_boundary_mask() {
    ensure_cuda_backend();
    let (h_leaf, h) = tracked_cuda(
        vec![
            f16::from_f32(-1.0),
            f16::from_f32(0.0),
            f16::from_f32(0.5),
            f16::from_f32(1.0),
            f16::from_f32(2.0),
        ],
        &[5],
    );
    let h_out =
        clamp_opt(&h, Some(f16::from_f32(0.0)), Some(f16::from_f32(1.0))).expect("f16 CUDA clamp");
    assert_eq!(h_out.device(), Device::Cuda(0));
    sum(&h_out).expect("sum").backward().expect("backward");
    assert_eq!(
        h_leaf.grad().unwrap().unwrap().data_vec().unwrap(),
        vec![
            f16::from_f32(0.0),
            f16::from_f32(1.0),
            f16::from_f32(1.0),
            f16::from_f32(1.0),
            f16::from_f32(0.0),
        ]
    );

    let (b_leaf, b) = tracked_cuda(
        vec![
            bf16::from_f32(-1.0),
            bf16::from_f32(0.0),
            bf16::from_f32(0.5),
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
        ],
        &[5],
    );
    let b_out = clamp_opt(&b, Some(bf16::from_f32(0.0)), Some(bf16::from_f32(1.0)))
        .expect("bf16 CUDA clamp");
    assert_eq!(b_out.device(), Device::Cuda(0));
    sum(&b_out).expect("sum").backward().expect("backward");
    assert_eq!(
        b_leaf.grad().unwrap().unwrap().data_vec().unwrap(),
        vec![
            bf16::from_f32(0.0),
            bf16::from_f32(1.0),
            bf16::from_f32(1.0),
            bf16::from_f32(1.0),
            bf16::from_f32(0.0),
        ]
    );
}
