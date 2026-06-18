#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::grad_fns::reduction::sum as reduce_sum;
use ferrotorch_core::{Device, Tensor, TensorStorage, digamma, gammaln, lgamma};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for CORE-2023 lgamma/digamma tests");
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
fn cuda_lgamma_f32_matches_torch_edges_and_reference_values() {
    ensure_cuda_backend();

    let x = cuda_f32(
        vec![
            f32::NEG_INFINITY,
            -100.5,
            -2.0,
            -1.5,
            -1.0,
            -0.5,
            -0.0,
            0.0,
            0.1,
            0.5,
            1.0,
            2.0,
            10.0,
            f32::INFINITY,
            f32::NAN,
        ],
        &[15],
        false,
    );

    let out = lgamma(&x).expect("cuda lgamma f32");
    let alias = gammaln(&x).expect("cuda gammaln alias f32");
    let expected = vec![
        f32::INFINITY,
        -364.900_97,
        f32::INFINITY,
        0.860_047_04,
        f32::INFINITY,
        1.265_512_1,
        f32::INFINITY,
        f32::INFINITY,
        2.252_712_7,
        0.572_364_9,
        0.0,
        0.0,
        12.801_827,
        f32::INFINITY,
        f32::NAN,
    ];
    assert_close_or_special_f32(
        &read_cuda_f32(&out, "lgamma f32"),
        &expected,
        2e-4,
        "lgamma f32",
    );
    assert_close_or_special_f32(
        &read_cuda_f32(&alias, "gammaln f32"),
        &expected,
        2e-4,
        "gammaln f32",
    );
}

#[test]
fn cuda_lgamma_f64_matches_torch_edges_and_reference_values() {
    ensure_cuda_backend();

    let x = cuda_f64(
        vec![
            f64::NEG_INFINITY,
            -100.5,
            -2.0,
            -1.5,
            -1.0,
            -0.5,
            -0.0,
            0.0,
            0.1,
            0.5,
            1.0,
            2.0,
            10.0,
            f64::INFINITY,
            f64::NAN,
        ],
        &[15],
        false,
    );

    let out = lgamma(&x).expect("cuda lgamma f64");
    let expected = vec![
        f64::INFINITY,
        -364.900_968_309_427_36,
        f64::INFINITY,
        0.860_047_015_376_481,
        f64::INFINITY,
        1.265_512_123_484_645_4,
        f64::INFINITY,
        f64::INFINITY,
        2.252_712_651_734_206,
        0.572_364_942_924_700_1,
        0.0,
        0.0,
        12.801_827_480_081_469,
        f64::INFINITY,
        f64::NAN,
    ];
    assert_close_or_special_f64(
        &read_cuda_f64(&out, "lgamma f64"),
        &expected,
        2e-10,
        "lgamma f64",
    );
}

#[test]
fn cuda_digamma_f32_matches_torch_edges_and_reference_values() {
    ensure_cuda_backend();

    let x = cuda_f32(
        vec![
            f32::NEG_INFINITY,
            -100.5,
            -2.0,
            -1.5,
            -1.0,
            -0.5,
            -0.0,
            0.0,
            0.1,
            0.5,
            1.0,
            2.0,
            10.0,
            f32::INFINITY,
            f32::NAN,
        ],
        &[15],
        false,
    );

    let out = digamma(&x).expect("cuda digamma f32");
    let expected = vec![
        f32::NAN,
        4.615_124_7,
        f32::NAN,
        0.703_156_77,
        f32::NAN,
        0.036_489_915,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -10.423_756,
        -1.963_510_9,
        -0.577_215_9,
        0.422_784_2,
        2.251_752_6,
        f32::INFINITY,
        f32::NAN,
    ];
    assert_close_or_special_f32(
        &read_cuda_f32(&out, "digamma f32"),
        &expected,
        2e-4,
        "digamma f32",
    );
}

#[test]
fn cuda_digamma_f64_matches_torch_edges_and_reference_values() {
    ensure_cuda_backend();

    let x = cuda_f64(
        vec![
            f64::NEG_INFINITY,
            -100.5,
            -2.0,
            -1.5,
            -1.0,
            -0.5,
            -0.0,
            0.0,
            0.1,
            0.5,
            1.0,
            2.0,
            10.0,
            f64::INFINITY,
            f64::NAN,
        ],
        &[15],
        false,
    );

    let out = digamma(&x).expect("cuda digamma f64");
    let expected = vec![
        f64::NAN,
        4.615_124_601_338_064,
        f64::NAN,
        0.703_156_640_645_243_3,
        f64::NAN,
        0.036_489_973_978_576_39,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -10.423_754_940_411_076,
        -1.963_510_026_021_422_9,
        -0.577_215_664_901_532_8,
        0.422_784_335_098_467,
        2.251_752_589_066_721,
        f64::INFINITY,
        f64::NAN,
    ];
    assert_close_or_special_f64(
        &read_cuda_f64(&out, "digamma f64"),
        &expected,
        2e-10,
        "digamma f64",
    );
}

#[test]
fn cuda_lgamma_and_digamma_backward_are_resident() {
    ensure_cuda_backend();

    let x = cuda_f32(vec![0.5, 1.0, 2.0, 10.0], &[4], true);
    reduce_sum(&lgamma(&x).expect("lgamma forward"))
        .expect("sum lgamma")
        .backward()
        .expect("lgamma backward");
    let grad = x.grad().expect("lgamma grad lookup").expect("lgamma grad");
    assert_close_or_special_f32(
        &read_cuda_f32(&grad, "lgamma grad f32"),
        &[-1.963_510_9, -0.577_215_9, 0.422_784_2, 2.251_752_6],
        2e-4,
        "lgamma grad f32",
    );

    let y = cuda_f32(vec![0.5, 1.0, 2.0, 10.0], &[4], true);
    reduce_sum(&digamma(&y).expect("digamma forward"))
        .expect("sum digamma")
        .backward()
        .expect("digamma backward");
    let grad = y
        .grad()
        .expect("digamma grad lookup")
        .expect("digamma grad");
    assert_close_or_special_f32(
        &read_cuda_f32(&grad, "digamma grad f32"),
        &[4.934_802, 1.644_934, 0.644_934_06, 0.105_166_33],
        2e-4,
        "digamma grad f32",
    );

    let x64 = cuda_f64(vec![0.5, 1.0, 2.0, 10.0], &[4], true);
    reduce_sum(&lgamma(&x64).expect("lgamma f64 forward"))
        .expect("sum lgamma f64")
        .backward()
        .expect("lgamma f64 backward");
    let grad64 = x64
        .grad()
        .expect("lgamma f64 grad lookup")
        .expect("lgamma f64 grad");
    assert_close_or_special_f64(
        &read_cuda_f64(&grad64, "lgamma grad f64"),
        &[
            -1.963_510_026_021_422_9,
            -0.577_215_664_901_532_8,
            0.422_784_335_098_467,
            2.251_752_589_066_721,
        ],
        2e-10,
        "lgamma grad f64",
    );

    let y64 = cuda_f64(vec![0.5, 1.0, 2.0, 10.0], &[4], true);
    reduce_sum(&digamma(&y64).expect("digamma f64 forward"))
        .expect("sum digamma f64")
        .backward()
        .expect("digamma f64 backward");
    let grad64 = y64
        .grad()
        .expect("digamma f64 grad lookup")
        .expect("digamma f64 grad");
    assert_close_or_special_f64(
        &read_cuda_f64(&grad64, "digamma grad f64"),
        &[
            4.934_802_202_073_678,
            1.644_934_067_638_337_5,
            0.644_934_067_088_189_9,
            0.105_166_335_682_166_57,
        ],
        2e-10,
        "digamma grad f64",
    );
}
