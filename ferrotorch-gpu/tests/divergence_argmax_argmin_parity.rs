//! Regression probes for PyTorch-parity `argmax` / `argmin`.
//!
//! PyTorch 2.11.0+cu130 oracle:
//! - global empty input errors unless a dimension is supplied;
//! - reducing an empty selected dimension errors only when the output would
//!   contain elements;
//! - scalar dim accepts `0` and `-1` but rejects out-of-range dims;
//! - ties keep the first index;
//! - the first NaN in a slice wins for both argmax and argmin;
//! - CUDA results remain GPU-resident until explicitly copied to CPU.

#![cfg(feature = "cuda")]

use ferrotorch_core::grad_fns::reduction::{argmax, argmax_dim, argmin, argmin_dim};
use ferrotorch_core::{Device, Tensor, TensorStorage};
use ferrotorch_gpu::init_cuda_backend;

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn cpu_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu f32 tensor")
}

fn cpu_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu f64 tensor")
}

fn cuda_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    cpu_f32(data, shape).to(Device::Cuda(0)).expect("to cuda")
}

fn cuda_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    cpu_f64(data, shape).to(Device::Cuda(0)).expect("to cuda")
}

fn host_i64(t: &ferrotorch_core::IntTensor<i64>) -> Vec<i64> {
    t.to(Device::Cpu)
        .expect("indices to cpu")
        .data()
        .expect("index data")
        .to_vec()
}

#[test]
fn cpu_argmax_argmin_nan_first_and_scalar_dim_edges_match_torch() {
    let x = cpu_f64(&[1.0, f64::NAN, 3.0, f64::NAN], &[4]);
    assert_eq!(argmax(&x).expect("argmax").data().unwrap(), &[1]);
    assert_eq!(argmin(&x).expect("argmin").data().unwrap(), &[1]);

    let x = cpu_f64(&[f64::NAN, 1.0, f64::NAN], &[3]);
    assert_eq!(argmax(&x).expect("argmax").data().unwrap(), &[0]);
    assert_eq!(argmin(&x).expect("argmin").data().unwrap(), &[0]);

    let scalar = cpu_f32(&[5.0], &[]);
    assert_eq!(argmax_dim(&scalar, 0, false).unwrap().data().unwrap(), &[0]);
    assert_eq!(argmin_dim(&scalar, -1, true).unwrap().data().unwrap(), &[0]);
    assert!(argmax_dim(&scalar, 1, false).is_err());
    assert!(argmin_dim(&scalar, -2, false).is_err());
}

#[test]
fn cpu_argmax_argmin_empty_selected_axis_edges_match_torch() {
    let selected_empty = cpu_f32(&[], &[2, 0, 3]);
    assert!(argmax_dim(&selected_empty, 1, false).is_err());
    assert!(argmin_dim(&selected_empty, 1, true).is_err());

    let empty_output = cpu_f32(&[], &[0, 2]);
    let y = argmax_dim(&empty_output, 1, false).expect("argmax zero-output");
    assert_eq!(y.shape(), &[0]);
    assert!(y.data().unwrap().is_empty());
}

#[test]
fn cuda_argmax_argmin_global_and_dim_stay_device_and_match_torch() {
    ensure_cuda();

    let x = cuda_f32(&[1.0, f32::NAN, 3.0, f32::NAN], &[4]);
    let y = argmax(&x).expect("cuda argmax");
    assert!(y.is_cuda(), "argmax result must stay CUDA-resident");
    assert_eq!(host_i64(&y), vec![1]);
    let y = argmin(&x).expect("cuda argmin");
    assert!(y.is_cuda(), "argmin result must stay CUDA-resident");
    assert_eq!(host_i64(&y), vec![1]);

    let x = cuda_f64(
        &[1.0, f64::NAN, 3.0, 5.0, 5.0, 4.0, -1.0, -1.0, 0.0],
        &[3, 3],
    );
    let y = argmax_dim(&x, 1, false).expect("cuda argmax_dim");
    assert!(y.is_cuda());
    assert_eq!(y.shape(), &[3]);
    assert_eq!(host_i64(&y), vec![1, 0, 2]);

    let y = argmin_dim(&x, -1, true).expect("cuda argmin_dim");
    assert!(y.is_cuda());
    assert_eq!(y.shape(), &[3, 1]);
    assert_eq!(host_i64(&y), vec![1, 2, 0]);
}

#[test]
fn cuda_argmax_argmin_empty_selected_axis_edges_match_torch() {
    ensure_cuda();

    let selected_empty = cuda_f32(&[], &[2, 0, 3]);
    assert!(argmax_dim(&selected_empty, 1, false).is_err());
    assert!(argmin_dim(&selected_empty, 1, true).is_err());

    let empty_output = cuda_f32(&[], &[0, 2]);
    let y = argmax_dim(&empty_output, 1, false).expect("argmax zero-output");
    assert!(y.is_cuda());
    assert_eq!(y.shape(), &[0]);
    assert!(host_i64(&y).is_empty());
}
