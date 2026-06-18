#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::grad_fns::reduction::sum as reduce_sum;
use ferrotorch_core::{Device, Tensor, TensorStorage, erf, erfinv};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for CORE-2023 erfinv tests");
    });
}

fn cpu_f32(data: Vec<f32>, shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), requires_grad)
        .expect("cpu f32 tensor")
}

fn cuda_f32(data: Vec<f32>, shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    cpu_f32(data, shape, false)
        .to(Device::Cuda(0))
        .expect("upload f32")
        .requires_grad_(requires_grad)
}

fn cpu_f64(data: Vec<f64>, shape: &[usize], requires_grad: bool) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), requires_grad)
        .expect("cpu f64 tensor")
}

fn cuda_f64(data: Vec<f64>, shape: &[usize], requires_grad: bool) -> Tensor<f64> {
    cpu_f64(data, shape, false)
        .to(Device::Cuda(0))
        .expect("upload f64")
        .requires_grad_(requires_grad)
}

fn read_cuda_f32(t: &Tensor<f32>, label: &str) -> Vec<f32> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "{label}: expected CUDA-resident tensor, got {:?}",
        t.device()
    );
    t.to(Device::Cpu)
        .expect("download")
        .data_vec()
        .expect("logical data")
}

fn read_cuda_f64(t: &Tensor<f64>, label: &str) -> Vec<f64> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "{label}: expected CUDA-resident tensor, got {:?}",
        t.device()
    );
    t.to(Device::Cpu)
        .expect("download")
        .data_vec()
        .expect("logical data")
}

fn assert_close_or_special_f32(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "{label}[{i}]: expected NaN, got {a:?}");
        } else if e.is_infinite() {
            assert_eq!(a, e, "{label}[{i}]: expected {e:?}, got {a:?}");
        } else {
            assert!(
                (a - e).abs() <= tol,
                "{label}[{i}]: expected {e:?}, got {a:?}, diff={:?}",
                (a - e).abs()
            );
        }
    }
}

fn assert_close_or_special_f64(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "{label}[{i}]: expected NaN, got {a:?}");
        } else if e.is_infinite() {
            assert_eq!(a, e, "{label}[{i}]: expected {e:?}, got {a:?}");
        } else {
            assert!(
                (a - e).abs() <= tol,
                "{label}[{i}]: expected {e:?}, got {a:?}, diff={:?}",
                (a - e).abs()
            );
        }
    }
}

#[test]
fn cuda_erfinv_f32_matches_torch_domain_and_interior_values() {
    ensure_cuda_backend();

    let x = cuda_f32(
        vec![
            -1.0,
            -0.999_999,
            -0.9,
            -0.7,
            -0.5,
            -0.0,
            0.0,
            0.5,
            0.7,
            0.9,
            0.999_999,
            1.0,
            2.0,
            -2.0,
            f32::NAN,
        ],
        &[15],
        false,
    );

    let out = erfinv(&x).expect("cuda erfinv f32");
    let expected = vec![
        f32::NEG_INFINITY,
        -3.457_074_6,
        -1.163_087_1,
        -0.732_869_1,
        -0.476_936_28,
        -0.0,
        0.0,
        0.476_936_28,
        0.732_869_1,
        1.163_087_1,
        3.457_074_6,
        f32::INFINITY,
        f32::NAN,
        f32::NAN,
        f32::NAN,
    ];
    assert_close_or_special_f32(
        &read_cuda_f32(&out, "erfinv f32"),
        &expected,
        5e-5,
        "erfinv f32",
    );
}

#[test]
fn cuda_erfinv_f64_matches_torch_domain_and_interior_values() {
    ensure_cuda_backend();

    let x = cuda_f64(
        vec![
            -1.0,
            -0.999_999,
            -0.9,
            -0.7,
            -0.5,
            -0.0,
            0.0,
            0.5,
            0.7,
            0.9,
            0.999_999,
            1.0,
            2.0,
            -2.0,
            f64::NAN,
        ],
        &[15],
        false,
    );

    let out = erfinv(&x).expect("cuda erfinv f64");
    let expected = vec![
        f64::NEG_INFINITY,
        -3.458_910_737_275_499,
        -1.163_087_153_676_674_3,
        -0.732_869_077_959_216_7,
        -0.476_936_276_204_469_9,
        -0.0,
        0.0,
        0.476_936_276_204_469_9,
        0.732_869_077_959_216_7,
        1.163_087_153_676_674_3,
        3.458_910_737_275_499,
        f64::INFINITY,
        f64::NAN,
        f64::NAN,
        f64::NAN,
    ];
    assert_close_or_special_f64(
        &read_cuda_f64(&out, "erfinv f64"),
        &expected,
        1e-10,
        "erfinv f64",
    );
}

#[test]
fn cuda_erfinv_roundtrip_and_backward_are_resident() {
    ensure_cuda_backend();

    let x = cuda_f32(vec![-0.8, -0.25, 0.0, 0.25, 0.8], &[5], true);
    let inv = erfinv(&x).expect("cuda erfinv forward");
    let roundtrip = erf(&inv).expect("cuda erf(erfinv(x))");
    assert_close_or_special_f32(
        &read_cuda_f32(&roundtrip, "erfinv roundtrip"),
        &[-0.8, -0.25, 0.0, 0.25, 0.8],
        2e-5,
        "erfinv roundtrip",
    );

    reduce_sum(&inv)
        .expect("sum erfinv")
        .backward()
        .expect("erfinv backward");
    let grad = x.grad().expect("grad lookup").expect("erfinv grad");
    let inv_vals = read_cuda_f32(&inv, "erfinv output for grad oracle");
    let expected: Vec<f32> = inv_vals
        .iter()
        .map(|&v| 0.5 * std::f32::consts::PI.sqrt() * (v * v).exp())
        .collect();
    assert_close_or_special_f32(
        &read_cuda_f32(&grad, "erfinv grad"),
        &expected,
        2e-4,
        "erfinv grad",
    );
}
