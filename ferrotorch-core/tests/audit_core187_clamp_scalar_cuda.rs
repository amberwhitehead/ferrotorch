#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::creation::from_vec;
use ferrotorch_core::device::Device;
use ferrotorch_core::grad_fns::transcendental::clamp;
use ferrotorch_core::tensor::Tensor;

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
