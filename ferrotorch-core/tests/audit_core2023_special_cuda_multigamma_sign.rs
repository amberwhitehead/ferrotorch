#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::error::FerrotorchError;
use ferrotorch_core::grad_fns::reduction::sum as reduce_sum;
use ferrotorch_core::{Device, Tensor, TensorStorage, gammaln_sign, multigammaln, mvlgamma};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for CORE-2023 multigamma/sign tests");
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
fn cuda_multigammaln_f32_matches_torch_composite_values() {
    ensure_cuda_backend();

    let x = cuda_f32(vec![0.3, 0.5, 1.0, 2.5], &[4], false);

    let p1 = multigammaln(&x, 1).expect("multigammaln p1 f32");
    assert_close_or_special_f32(
        &read_cuda_f32(&p1, "multigammaln p1 f32"),
        &[1.095_797_9, 0.572_364_9, 0.0, 0.284_682_87],
        2e-4,
        "multigammaln p1 f32",
    );

    let p3 = multigammaln(&x, 3).expect("multigammaln p3 f32");
    assert_close_or_special_f32(
        &read_cuda_f32(&p3, "multigammaln p3 f32"),
        &[6.026_863_6, f32::INFINITY, f32::INFINITY, 1.880_995_4],
        3e-4,
        "multigammaln p3 f32",
    );
}

#[test]
fn cuda_multigammaln_f64_matches_torch_composite_values() {
    ensure_cuda_backend();

    let x = cuda_f64(vec![0.3, 0.5, 1.0, 2.5], &[4], false);

    let p2 = multigammaln(&x, 2).expect("multigammaln p2 f64");
    assert_close_or_special_f64(
        &read_cuda_f64(&p2, "multigammaln p2 f64"),
        &[
            3.429_660_528_576_713_7,
            f64::INFINITY,
            1.144_729_885_849_400_2,
            0.857_047_813_397_619_3,
        ],
        3e-10,
        "multigammaln p2 f64",
    );

    let alias = mvlgamma(&x, 4).expect("mvlgamma p4 f64");
    assert_close_or_special_f64(
        &read_cuda_f64(&alias, "mvlgamma p4 f64"),
        &[
            9.323_134_215_997_005,
            f64::INFINITY,
            f64::INFINITY,
            3.598_090_290_385_874_5,
        ],
        5e-10,
        "mvlgamma p4 f64",
    );
}

#[test]
fn cuda_multigammaln_backward_sums_shifted_digamma_terms() {
    ensure_cuda_backend();

    let x = cuda_f32(vec![2.0, 3.0, 4.0], &[3], true);
    reduce_sum(&multigammaln(&x, 3).expect("multigammaln p3 f32"))
        .expect("sum multigammaln f32")
        .backward()
        .expect("multigammaln backward f32");
    let grad = x
        .grad()
        .expect("f32 grad lookup")
        .expect("f32 multigammaln grad");
    assert_close_or_special_f32(
        &read_cuda_f32(&grad, "multigammaln grad f32"),
        &[-0.117_941_8, 2.048_725_1, 3.282_058_7],
        3e-4,
        "multigammaln grad f32",
    );

    let x = cuda_f64(vec![2.0, 3.0, 4.0], &[3], true);
    reduce_sum(&multigammaln(&x, 3).expect("multigammaln p3 f64"))
        .expect("sum multigammaln f64")
        .backward()
        .expect("multigammaln backward f64");
    let grad = x
        .grad()
        .expect("f64 grad lookup")
        .expect("f64 multigammaln grad");
    assert_close_or_special_f64(
        &read_cuda_f64(&grad, "multigammaln grad f64"),
        &[
            -0.117_941_355_824_489_5,
            2.048_725_310_842_177,
            3.282_058_644_175_510_4,
        ],
        3e-10,
        "multigammaln grad f64",
    );
}

#[test]
fn cuda_multigammaln_rejects_p_zero_before_dispatch() {
    ensure_cuda_backend();

    let x = cuda_f32(vec![1.0], &[1], false);
    let err = multigammaln(&x, 0).expect_err("p=0 must reject");
    assert!(
        matches!(err, FerrotorchError::InvalidArgument { ref message } if message.contains("p has to be greater than or equal to 1")),
        "unexpected p=0 error: {err:?}"
    );
}

#[test]
fn cuda_gammaln_sign_f32_matches_scipy_gammasgn_edges() {
    ensure_cuda_backend();

    let x = cuda_f32(
        vec![
            f32::NEG_INFINITY,
            -3.0,
            -2.5,
            -1.5,
            -0.5,
            -0.0,
            0.0,
            0.5,
            f32::INFINITY,
            f32::NAN,
        ],
        &[10],
        false,
    );
    let out = gammaln_sign(&x).expect("gammaln_sign f32");
    assert_close_or_special_f32(
        &read_cuda_f32(&out, "gammaln_sign f32"),
        &[
            f32::NAN,
            f32::NAN,
            -1.0,
            1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            f32::NAN,
        ],
        0.0,
        "gammaln_sign f32",
    );
}

#[test]
fn cuda_gammaln_sign_f64_matches_scipy_gammasgn_edges() {
    ensure_cuda_backend();

    let x = cuda_f64(
        vec![
            f64::NEG_INFINITY,
            -3.0,
            -2.5,
            -1.5,
            -0.5,
            -0.0,
            0.0,
            0.5,
            f64::INFINITY,
            f64::NAN,
        ],
        &[10],
        false,
    );
    let out = gammaln_sign(&x).expect("gammaln_sign f64");
    assert_close_or_special_f64(
        &read_cuda_f64(&out, "gammaln_sign f64"),
        &[
            f64::NAN,
            f64::NAN,
            -1.0,
            1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            f64::NAN,
        ],
        0.0,
        "gammaln_sign f64",
    );
}
