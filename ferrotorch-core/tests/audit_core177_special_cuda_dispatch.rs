#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::grad_fns::reduction::sum as reduce_sum;
use ferrotorch_core::{
    Device, Tensor, TensorStorage, expm1, gammainc, gammaincc, log1p, sinc, xlogy,
};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for CORE-177 audit tests");
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

fn assert_close_or_nan(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "{label}[{i}]: expected NaN, got {a:?}");
        } else {
            assert!(
                (a - e).abs() <= tol,
                "{label}[{i}]: expected {e:?}, got {a:?}"
            );
        }
    }
}

fn assert_close_or_nan_f64(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "{label}[{i}]: expected NaN, got {a:?}");
        } else {
            assert!(
                (a - e).abs() <= tol,
                "{label}[{i}]: expected {e:?}, got {a:?}"
            );
        }
    }
}

fn assert_not_implemented<T>(label: &str, result: FerrotorchResult<T>, expected_op: &'static str) {
    match result {
        Err(FerrotorchError::NotImplementedOnCuda { op }) => {
            assert_eq!(op, expected_op, "{label}: wrong CUDA op name");
        }
        Err(other) => panic!("{label}: expected NotImplementedOnCuda, got {other:?}"),
        Ok(_) => panic!("{label}: expected NotImplementedOnCuda"),
    }
}

#[test]
fn cuda_xlogy_f32_is_resident_broadcasted_and_matches_torch_branch_order() {
    ensure_cuda_backend();

    let x = cuda_f32(vec![0.0, 2.0], &[2, 1], false);
    let y = cuda_f32(vec![-1.0, 1.0, f32::NAN, 4.0], &[1, 4], false);

    let out = xlogy(&x, &y).expect("cuda xlogy");
    assert_eq!(out.shape(), &[2, 4]);

    let got = read_cuda_f32(&out, "xlogy forward");
    let expected = vec![
        0.0,
        0.0,
        f32::NAN,
        0.0,
        f32::NAN,
        0.0,
        f32::NAN,
        2.0 * 4.0_f32.ln(),
    ];
    assert_close_or_nan(&got, &expected, 1e-4, "xlogy forward");
}

#[test]
fn cuda_xlogy_f64_is_resident_and_uses_named_kernel() {
    ensure_cuda_backend();

    let x = cuda_f64(vec![0.0, 2.0, 3.0], &[3], false);
    let y = cuda_f64(vec![f64::NAN, 4.0, 0.25], &[3], false);

    let out = xlogy(&x, &y).expect("cuda xlogy f64");
    let got = read_cuda_f64(&out, "xlogy f64 forward");
    let expected = vec![f64::NAN, 2.0 * 4.0_f64.ln(), 3.0 * 0.25_f64.ln()];
    assert_close_or_nan_f64(&got, &expected, 1e-4, "xlogy f64 forward");
}

#[test]
fn cuda_xlogy_f32_backward_broadcasts_and_reduces_without_host_fallback() {
    ensure_cuda_backend();

    let x = cuda_f32(vec![1.0, 2.0], &[2, 1], true);
    let y = cuda_f32(vec![2.0, 4.0], &[1, 2], true);

    let out = xlogy(&x, &y).expect("cuda xlogy forward");
    assert_eq!(out.device(), Device::Cuda(0));
    assert_eq!(out.shape(), &[2, 2]);

    reduce_sum(&out)
        .expect("sum xlogy")
        .backward()
        .expect("xlogy backward");

    let gx = x.grad().expect("x grad lookup").expect("x grad");
    let gy = y.grad().expect("y grad lookup").expect("y grad");

    let expected_gx = vec![2.0_f32.ln() + 4.0_f32.ln(); 2];
    let expected_gy = vec![(1.0 + 2.0) / 2.0, (1.0 + 2.0) / 4.0];
    assert_close_or_nan(&read_cuda_f32(&gx, "x grad"), &expected_gx, 1e-4, "x grad");
    assert_close_or_nan(&read_cuda_f32(&gy, "y grad"), &expected_gy, 1e-4, "y grad");
}

#[test]
fn cuda_special_log1p_expm1_sinc_aliases_use_resident_transcendental_paths() {
    ensure_cuda_backend();

    let x = cuda_f32(vec![-0.5, 0.0, 0.25, 1.0], &[4], false);

    let log1p_out = log1p(&x).expect("special log1p cuda");
    let expm1_out = expm1(&x).expect("special expm1 cuda");
    let sinc_out = sinc(&x).expect("special sinc cuda");

    let expected_log1p: Vec<f32> = [-0.5_f32, 0.0, 0.25, 1.0]
        .into_iter()
        .map(f32::ln_1p)
        .collect();
    let expected_expm1: Vec<f32> = [-0.5_f32, 0.0, 0.25, 1.0]
        .into_iter()
        .map(f32::exp_m1)
        .collect();
    let expected_sinc = vec![
        (-0.5_f32 * std::f32::consts::PI).sin() / (-0.5_f32 * std::f32::consts::PI),
        1.0,
        (0.25_f32 * std::f32::consts::PI).sin() / (0.25_f32 * std::f32::consts::PI),
        0.0,
    ];

    assert_close_or_nan(
        &read_cuda_f32(&log1p_out, "log1p alias"),
        &expected_log1p,
        1e-4,
        "log1p alias",
    );
    assert_close_or_nan(
        &read_cuda_f32(&expm1_out, "expm1 alias"),
        &expected_expm1,
        1e-4,
        "expm1 alias",
    );
    assert_close_or_nan(
        &read_cuda_f32(&sinc_out, "sinc alias"),
        &expected_sinc,
        1e-4,
        "sinc alias",
    );
}

#[test]
fn cuda_unimplemented_special_ops_return_named_notimplemented_not_storage_errors() {
    ensure_cuda_backend();

    let x = cuda_f32(vec![0.5, 1.5], &[2], false);
    let y = cuda_f32(vec![1.0, 2.0], &[2], false);

    assert_not_implemented("gammainc", gammainc(&x, &y), "gammainc");
    assert_not_implemented("gammaincc", gammaincc(&x, &y), "gammaincc");
}
