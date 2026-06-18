#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::grad_fns::reduction::sum as reduce_sum;
use ferrotorch_core::{Device, Tensor, TensorStorage, erf, erfc};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for CORE-2023 audit tests");
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

fn assert_close_or_nan_f32(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "{label}[{i}]: expected NaN, got {a:?}");
        } else if e.is_infinite() {
            assert_eq!(a, e, "{label}[{i}]: expected {e:?}, got {a:?}");
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
        } else if e.is_infinite() {
            assert_eq!(a, e, "{label}[{i}]: expected {e:?}, got {a:?}");
        } else {
            assert!(
                (a - e).abs() <= tol,
                "{label}[{i}]: expected {e:?}, got {a:?}"
            );
        }
    }
}

#[test]
#[allow(
    clippy::excessive_precision,
    reason = "hard-coded f32 libm/PyTorch reference values document the rounded oracle outputs"
)]
fn cuda_erf_erfc_f32_are_resident_and_match_reference_edges() {
    ensure_cuda_backend();

    let x = cuda_f32(
        vec![
            f32::NEG_INFINITY,
            -3.0,
            -2.0,
            -1.0,
            -0.0,
            0.0,
            0.5,
            1.0,
            2.0,
            3.0,
            f32::INFINITY,
            f32::NAN,
        ],
        &[12],
        false,
    );

    let erf_out = erf(&x).expect("cuda erf f32");
    let erfc_out = erfc(&x).expect("cuda erfc f32");

    let expected_erf = vec![
        -1.0,
        -0.999_977_9,
        -0.995_322_3,
        -0.842_700_8,
        -0.0,
        0.0,
        0.520_499_9,
        0.842_700_8,
        0.995_322_3,
        0.999_977_9,
        1.0,
        f32::NAN,
    ];
    let expected_erfc = vec![
        2.0,
        1.999_977_9,
        1.995_322_2,
        1.842_700_8,
        1.0,
        1.0,
        0.479_500_1,
        0.157_299_2,
        0.004_677_735,
        0.000_022_090_497,
        0.0,
        f32::NAN,
    ];

    assert_close_or_nan_f32(
        &read_cuda_f32(&erf_out, "erf f32"),
        &expected_erf,
        3e-6,
        "erf f32",
    );
    assert_close_or_nan_f32(
        &read_cuda_f32(&erfc_out, "erfc f32"),
        &expected_erfc,
        3e-6,
        "erfc f32",
    );
}

#[test]
fn cuda_erf_erfc_f64_are_resident_and_preserve_erfc_tail() {
    ensure_cuda_backend();

    let x = cuda_f64(
        vec![
            f64::NEG_INFINITY,
            -3.0,
            -1.0,
            -0.0,
            0.0,
            0.5,
            1.0,
            2.0,
            3.0,
            5.0,
            10.0,
            26.0,
            28.0,
            f64::INFINITY,
            f64::NAN,
        ],
        &[15],
        false,
    );

    let erf_out = erf(&x).expect("cuda erf f64");
    let erfc_out = erfc(&x).expect("cuda erfc f64");

    let expected_erf = vec![
        -1.0,
        -0.999_977_909_503_001_4,
        -0.842_700_792_949_714_9,
        -0.0,
        0.0,
        0.520_499_877_813_046_5,
        0.842_700_792_949_714_9,
        0.995_322_265_018_952_7,
        0.999_977_909_503_001_4,
        0.999_999_999_998_462_6,
        1.0,
        1.0,
        1.0,
        1.0,
        f64::NAN,
    ];
    let expected_erfc = vec![
        2.0,
        1.999_977_909_503_001_5,
        1.842_700_792_949_715,
        1.0,
        1.0,
        0.479_500_122_186_953_5,
        0.157_299_207_050_285_13,
        0.004_677_734_981_047_265,
        0.000_022_090_496_998_585_438,
        0.000_000_000_001_537_459_794_428_035_1,
        2.088_487_583_762_545e-45,
        5.663_192_408_856_143e-296,
        0.0,
        0.0,
        f64::NAN,
    ];

    let got_erf = read_cuda_f64(&erf_out, "erf f64");
    let got_erfc = read_cuda_f64(&erfc_out, "erfc f64");

    assert_close_or_nan_f64(&got_erf, &expected_erf, 1e-12, "erf f64");
    assert_close_or_nan_f64(&got_erfc, &expected_erfc, 1e-12, "erfc f64");
    assert!(
        got_erfc[10] > 0.0,
        "erfc f64 tail at x=10 must not collapse through 1 - erf(x)"
    );
    assert!(
        got_erfc[11] > 0.0,
        "erfc f64 subnormal-adjacent tail at x=26 must remain positive"
    );
}

#[test]
fn cuda_erf_erfc_backward_stays_resident() {
    ensure_cuda_backend();

    let x = cuda_f32(vec![-1.0, 0.5, 2.0], &[3], true);
    let erf_out = erf(&x).expect("cuda erf forward");
    reduce_sum(&erf_out)
        .expect("sum erf")
        .backward()
        .expect("erf backward");
    let grad = x.grad().expect("grad lookup").expect("erf grad");
    let expected: Vec<f32> = [-1.0_f32, 0.5, 2.0]
        .into_iter()
        .map(|v| 2.0 / std::f32::consts::PI.sqrt() * (-(v * v)).exp())
        .collect();
    assert_close_or_nan_f32(
        &read_cuda_f32(&grad, "erf grad"),
        &expected,
        2e-5,
        "erf grad",
    );

    let y = cuda_f32(vec![-1.0, 0.5, 2.0], &[3], true);
    let erfc_out = erfc(&y).expect("cuda erfc forward");
    reduce_sum(&erfc_out)
        .expect("sum erfc")
        .backward()
        .expect("erfc backward");
    let grad = y.grad().expect("grad lookup").expect("erfc grad");
    let expected: Vec<f32> = [-1.0_f32, 0.5, 2.0]
        .into_iter()
        .map(|v| -2.0 / std::f32::consts::PI.sqrt() * (-(v * v)).exp())
        .collect();
    assert_close_or_nan_f32(
        &read_cuda_f32(&grad, "erfc grad"),
        &expected,
        2e-5,
        "erfc grad",
    );
}
