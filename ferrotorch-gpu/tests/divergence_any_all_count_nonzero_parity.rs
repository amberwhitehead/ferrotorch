//! Regression probes for PyTorch-parity `any` / `all` / `count_nonzero`.
//!
//! PyTorch 2.11.0+cu130 oracle:
//! - `any(empty) == false`, `all(empty) == true`, `count_nonzero(empty) == 0`;
//! - NaN is nonzero for all three operations;
//! - dim reductions over zero-length selected axes return identities;
//! - scalar dim accepts `0` and `-1` but rejects out-of-range dims;
//! - CUDA results remain resident until explicitly copied to CPU.

#![cfg(feature = "cuda")]

use ferrotorch_core::grad_fns::reduction::{
    all, all_dim, any, any_dim, count_nonzero, count_nonzero_dim,
};
use ferrotorch_core::{BoolTensor, Device, IntTensor, Tensor, TensorStorage};
use ferrotorch_gpu::init_cuda_backend;
use half::{bf16, f16};

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

fn cuda_f16(data: &[f32], shape: &[usize]) -> Tensor<f16> {
    let values: Vec<f16> = data.iter().copied().map(f16::from_f32).collect();
    Tensor::from_storage(TensorStorage::cpu(values), shape.to_vec(), false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap()
}

fn cuda_bf16(data: &[f32], shape: &[usize]) -> Tensor<bf16> {
    let values: Vec<bf16> = data.iter().copied().map(bf16::from_f32).collect();
    Tensor::from_storage(TensorStorage::cpu(values), shape.to_vec(), false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap()
}

fn host_bool(t: &BoolTensor) -> Vec<bool> {
    t.to(Device::Cpu)
        .expect("bool to cpu")
        .data()
        .unwrap()
        .to_vec()
}

fn host_i64(t: &IntTensor<i64>) -> Vec<i64> {
    t.to(Device::Cpu)
        .expect("int to cpu")
        .data()
        .unwrap()
        .to_vec()
}

#[test]
fn cpu_any_all_count_nonzero_edge_cases_match_torch() {
    let x = cpu_f64(&[], &[0]);
    assert_eq!(any(&x).unwrap().data().unwrap(), &[false]);
    assert_eq!(all(&x).unwrap().data().unwrap(), &[true]);
    assert_eq!(count_nonzero(&x).unwrap().data().unwrap(), &[0]);

    let x = cpu_f64(&[0.0, f64::NAN], &[2]);
    assert_eq!(any(&x).unwrap().data().unwrap(), &[true]);
    assert_eq!(all(&x).unwrap().data().unwrap(), &[false]);
    assert_eq!(count_nonzero(&x).unwrap().data().unwrap(), &[1]);

    let scalar = cpu_f32(&[5.0], &[]);
    assert_eq!(any_dim(&scalar, 0, false).unwrap().data().unwrap(), &[true]);
    assert_eq!(all_dim(&scalar, -1, true).unwrap().data().unwrap(), &[true]);
    assert_eq!(
        count_nonzero_dim(&scalar, -1).unwrap().data().unwrap(),
        &[1]
    );
    assert!(any_dim(&scalar, 1, false).is_err());
    assert!(all_dim(&scalar, -2, false).is_err());
    assert!(count_nonzero_dim(&scalar, 1).is_err());

    let selected_empty = cpu_f32(&[], &[2, 0, 3]);
    assert_eq!(
        any_dim(&selected_empty, 1, false).unwrap().data().unwrap(),
        &[false; 6]
    );
    assert_eq!(
        all_dim(&selected_empty, 1, false).unwrap().data().unwrap(),
        &[true; 6]
    );
    assert_eq!(
        count_nonzero_dim(&selected_empty, 1)
            .unwrap()
            .data()
            .unwrap(),
        &[0; 6]
    );
}

#[test]
fn cuda_any_all_count_nonzero_full_stay_device_and_match_torch() {
    ensure_cuda();

    let x = cuda_f32(&[], &[0]);
    let y = any(&x).unwrap();
    assert!(y.is_cuda());
    assert_eq!(host_bool(&y), vec![false]);
    let y = all(&x).unwrap();
    assert!(y.is_cuda());
    assert_eq!(host_bool(&y), vec![true]);
    let y = count_nonzero(&x).unwrap();
    assert!(y.is_cuda());
    assert_eq!(host_i64(&y), vec![0]);

    let x = cuda_f64(&[0.0, f64::NAN, -0.0, 3.0], &[4]);
    assert_eq!(host_bool(&any(&x).unwrap()), vec![true]);
    assert_eq!(host_bool(&all(&x).unwrap()), vec![false]);
    assert_eq!(host_i64(&count_nonzero(&x).unwrap()), vec![2]);
}

#[test]
fn cuda_any_all_count_nonzero_dim_edges_match_torch() {
    ensure_cuda();

    let x = cuda_f32(&[0.0, 1.0, 0.0, f32::NAN, 2.0, 0.0], &[2, 3]);
    let y = any_dim(&x, 1, false).unwrap();
    assert!(y.is_cuda());
    assert_eq!(y.shape(), &[2]);
    assert_eq!(host_bool(&y), vec![true, true]);

    let y = all_dim(&x, -1, true).unwrap();
    assert!(y.is_cuda());
    assert_eq!(y.shape(), &[2, 1]);
    assert_eq!(host_bool(&y), vec![false, false]);

    let y = count_nonzero_dim(&x, 0).unwrap();
    assert!(y.is_cuda());
    assert_eq!(y.shape(), &[3]);
    assert_eq!(host_i64(&y), vec![1, 2, 0]);

    let selected_empty = cuda_f32(&[], &[2, 0, 3]);
    assert_eq!(
        host_bool(&any_dim(&selected_empty, 1, false).unwrap()),
        vec![false; 6]
    );
    assert_eq!(
        host_bool(&all_dim(&selected_empty, 1, true).unwrap()),
        vec![true; 6]
    );
    assert_eq!(
        all_dim(&selected_empty, 1, true).unwrap().shape(),
        &[2, 1, 3]
    );
    assert_eq!(
        host_i64(&count_nonzero_dim(&selected_empty, 1).unwrap()),
        vec![0; 6]
    );

    let empty_output = cuda_f32(&[], &[0, 2]);
    let y = any_dim(&empty_output, 1, false).unwrap();
    assert!(y.is_cuda());
    assert_eq!(y.shape(), &[0]);
    assert!(host_bool(&y).is_empty());
}

#[test]
fn cuda_reduced_precision_any_all_count_nonzero_stay_device() {
    ensure_cuda();

    let x = cuda_f16(&[0.0, 1.0, 0.0, f32::NAN], &[2, 2]);
    let y = any_dim(&x, 1, false).unwrap();
    assert!(y.is_cuda());
    assert_eq!(host_bool(&y), vec![true, true]);
    let y = all_dim(&x, 1, false).unwrap();
    assert!(y.is_cuda());
    assert_eq!(host_bool(&y), vec![false, false]);
    let y = count_nonzero_dim(&x, 0).unwrap();
    assert!(y.is_cuda());
    assert_eq!(host_i64(&y), vec![0, 2]);

    let x = cuda_bf16(&[0.0, 2.0, 3.0, 4.0], &[2, 2]);
    let y = all(&x).unwrap();
    assert!(y.is_cuda());
    assert_eq!(host_bool(&y), vec![false]);
    let y = count_nonzero(&x).unwrap();
    assert!(y.is_cuda());
    assert_eq!(host_i64(&y), vec![3]);
}
