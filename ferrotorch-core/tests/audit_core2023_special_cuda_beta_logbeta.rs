#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::grad_fns::reduction::sum as reduce_sum;
use ferrotorch_core::{Device, Tensor, TensorStorage, beta, log_beta};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for CORE-2023 beta/log_beta tests");
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
            let diff = (a - e).abs();
            assert!(
                diff <= tol,
                "{label}[{i}]: expected {e:?}, got {a:?}, diff={diff:?}"
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
            let diff = (a - e).abs();
            assert!(
                diff <= tol,
                "{label}[{i}]: expected {e:?}, got {a:?}, diff={diff:?}"
            );
        }
    }
}

#[test]
fn cuda_log_beta_and_beta_f32_broadcast_and_match_scipy_values() {
    ensure_cuda_backend();

    let a = cuda_f32(vec![1.0, 2.5], &[2, 1], false);
    let b = cuda_f32(vec![2.0, 3.5], &[1, 2], false);

    let lb = log_beta(&a, &b).expect("log_beta f32");
    assert_eq!(lb.shape(), &[2, 2]);
    assert_close_or_special_f32(
        &read_cuda_f32(&lb, "log_beta f32"),
        &[
            -std::f32::consts::LN_2,
            -1.252_763,
            -2.169_053_8,
            -3.301_835_3,
        ],
        3e-4,
        "log_beta f32",
    );

    let be = beta(&a, &b).expect("beta f32");
    assert_eq!(be.shape(), &[2, 2]);
    assert_close_or_special_f32(
        &read_cuda_f32(&be, "beta f32"),
        &[0.5, 0.285_714_3, 0.114_285_715, 0.036_815_54],
        3e-4,
        "beta f32",
    );
}

#[test]
fn cuda_beta_f64_preserves_cephes_sign_poles_and_infinite_limits() {
    ensure_cuda_backend();

    let a = cuda_f64(
        vec![
            -0.5,
            -1.5,
            -0.5,
            -0.5,
            -2.5,
            -1.0,
            -3.0,
            -3.0,
            -4.0,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
        ],
        &[14],
        false,
    );
    let b = cuda_f64(
        vec![
            1.5,
            2.5,
            0.5,
            -0.5,
            0.5,
            0.5,
            2.0,
            1.0,
            2.0,
            0.5,
            -0.5,
            -1.5,
            f64::NAN,
            f64::INFINITY,
        ],
        &[14],
        false,
    );
    let out = read_cuda_f64(&beta(&a, &b).expect("beta f64"), "beta f64");

    assert_close_or_special_f64(
        &out,
        &[
            -std::f64::consts::PI,
            std::f64::consts::PI,
            -0.0,
            0.0,
            -0.0,
            f64::INFINITY,
            1.0 / 6.0,
            -1.0 / 3.0,
            1.0 / 12.0,
            0.0,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            f64::NAN,
        ],
        2e-10,
        "beta f64",
    );
    assert!(out[2].is_sign_negative(), "beta(-0.5, 0.5) must be -0.0");
    assert!(out[3].is_sign_positive(), "beta(-0.5, -0.5) must be +0.0");
    assert!(out[4].is_sign_negative(), "beta(-2.5, 0.5) must be -0.0");
}

#[test]
fn cuda_log_beta_f64_infinite_ladder_matches_scipy_betaln() {
    ensure_cuda_backend();

    let a = cuda_f64(
        vec![
            f64::INFINITY,
            1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
        ],
        &[7],
        false,
    );
    let b = cuda_f64(
        vec![1.0, f64::INFINITY, f64::INFINITY, f64::NAN, -0.5, -1.5, 0.0],
        &[7],
        false,
    );
    assert_close_or_special_f64(
        &read_cuda_f64(&log_beta(&a, &b).expect("log_beta f64"), "log_beta f64"),
        &[
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
        ],
        0.0,
        "log_beta f64",
    );
}

#[test]
fn cuda_log_beta_and_beta_backward_are_resident() {
    ensure_cuda_backend();

    let a = cuda_f32(vec![1.0, 2.5], &[2], true);
    let b = cuda_f32(vec![2.0, 3.5], &[2], true);
    reduce_sum(&log_beta(&a, &b).expect("log_beta backward f32"))
        .expect("sum log_beta f32")
        .backward()
        .expect("log_beta backward f32");
    let ga = a.grad().expect("a grad lookup").expect("a grad");
    let gb = b.grad().expect("b grad lookup").expect("b grad");
    assert_close_or_special_f32(
        &read_cuda_f32(&ga, "log_beta grad a f32"),
        &[-1.5, -1.002_961],
        5e-4,
        "log_beta grad a f32",
    );
    assert_close_or_special_f32(
        &read_cuda_f32(&gb, "log_beta grad b f32"),
        &[-0.5, -0.602_961],
        5e-4,
        "log_beta grad b f32",
    );

    let a = cuda_f64(vec![1.0, 2.5], &[2], true);
    let b = cuda_f64(vec![2.0, 3.5], &[2], true);
    reduce_sum(&beta(&a, &b).expect("beta backward f64"))
        .expect("sum beta f64")
        .backward()
        .expect("beta backward f64");
    let ga = a.grad().expect("a grad lookup").expect("a grad");
    let gb = b.grad().expect("b grad lookup").expect("b grad");
    assert_close_or_special_f64(
        &read_cuda_f64(&ga, "beta grad a f64"),
        &[-0.75, -0.036_924_550_742_942_78],
        5e-10,
        "beta grad a f64",
    );
    assert_close_or_special_f64(
        &read_cuda_f64(&gb, "beta grad b f64"),
        &[-0.25, -0.022_198_335_179_240_625],
        5e-10,
        "beta grad b f64",
    );
}
