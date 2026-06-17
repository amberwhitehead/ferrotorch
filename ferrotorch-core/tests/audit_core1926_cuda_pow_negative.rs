//! Regression coverage for #1926: CUDA `pow(tensor, scalar)` must match
//! PyTorch's scalar-exponent CUDA semantics for negative bases.
//!
//! Live torch oracle (2.11.0+cu130, cuda:0):
//! ```text
//! x = [-inf, -2, -1, -0, 0, 1, 2, inf]
//! x.pow(3)   = [-inf, -8, -1, -0, 0, 1, 8, inf]
//! x.pow(.5)  = [nan, nan, nan, -0, 0, 1, sqrt(2), inf]
//! x.pow(1.5) = [inf, nan, nan, 0, 0, 1, 2.828..., inf]
//! ```
//!
//! The f32 PTX used to evaluate `2^(e*log2(x))`, producing NaN for every
//! negative finite base. The f64 PTX used the magnitude bits while reconstructing
//! `ln(x)`, producing plausible positive values for odd integral exponents.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::grad_fns::arithmetic::pow;
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::{Device, Tensor, TensorStorage};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the GPU lane of this suite");
    });
}

fn cuda_f32(data: &[f32], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        vec![data.len()],
        requires_grad,
    )
    .expect("cpu f32")
    .to(Device::Cuda(0))
    .expect("upload f32")
    .detach()
    .requires_grad_(requires_grad)
}

fn cuda_f64(data: &[f64], requires_grad: bool) -> Tensor<f64> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        vec![data.len()],
        requires_grad,
    )
    .expect("cpu f64")
    .to(Device::Cuda(0))
    .expect("upload f64")
    .detach()
    .requires_grad_(requires_grad)
}

fn host_f32(t: &Tensor<f32>) -> Vec<f32> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "result must stay CUDA-resident"
    );
    t.cpu().expect("D2H f32").data().expect("f32 data").to_vec()
}

fn host_f64(t: &Tensor<f64>) -> Vec<f64> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "result must stay CUDA-resident"
    );
    t.cpu().expect("D2H f64").data().expect("f64 data").to_vec()
}

fn assert_f32_bits(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: len");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "{label}[{i}]: expected NaN, got {a:?}");
        } else {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{label}[{i}]: got {a:?} bits={:#x}, expected {e:?} bits={:#x}",
                a.to_bits(),
                e.to_bits()
            );
        }
    }
}

fn assert_f64_bits(actual: &[f64], expected: &[f64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: len");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "{label}[{i}]: expected NaN, got {a:?}");
        } else {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{label}[{i}]: got {a:?} bits={:#x}, expected {e:?} bits={:#x}",
                a.to_bits(),
                e.to_bits()
            );
        }
    }
}

fn assert_f32_close_or_nan(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: len");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "{label}[{i}]: expected NaN, got {a:?}");
        } else if e == 0.0 || e.is_infinite() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{label}[{i}]: signed/inf mismatch"
            );
        } else {
            assert!(
                (a - e).abs() <= tol,
                "{label}[{i}]: got {a}, expected {e}, tol {tol}"
            );
        }
    }
}

fn assert_f64_close_or_nan(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: len");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "{label}[{i}]: expected NaN, got {a:?}");
        } else if e == 0.0 || e.is_infinite() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{label}[{i}]: signed/inf mismatch"
            );
        } else {
            assert!(
                (a - e).abs() <= tol,
                "{label}[{i}]: got {a}, expected {e}, tol {tol}"
            );
        }
    }
}

#[test]
fn cuda_pow_negative_integral_forward_and_backward_match_torch() {
    ensure_cuda_backend();

    let x = cuda_f32(&[1.0, 2.0, -1.5, -0.0, 0.0], true);
    let y = pow(&x, 3.0).expect("f32 pow3");
    assert_f32_bits(
        &host_f32(&y),
        &[1.0, 8.0, -3.375, -0.0, 0.0],
        "f32 pow3 fwd",
    );
    sum(&y).expect("sum f32").backward().expect("backward f32");
    assert_f32_bits(
        &host_f32(&x.grad().expect("grad slot").expect("f32 grad")),
        &[3.0, 12.0, 6.75, 0.0, 0.0],
        "f32 pow3 grad",
    );

    let x = cuda_f64(&[1.0, 2.0, -1.5, -0.0, 0.0], true);
    let y = pow(&x, 3.0).expect("f64 pow3");
    assert_f64_bits(
        &host_f64(&y),
        &[1.0, 8.0, -3.375, -0.0, 0.0],
        "f64 pow3 fwd",
    );
    sum(&y).expect("sum f64").backward().expect("backward f64");
    assert_f64_bits(
        &host_f64(&x.grad().expect("grad slot").expect("f64 grad")),
        &[3.0, 12.0, 6.75, 0.0, 0.0],
        "f64 pow3 grad",
    );
}

#[test]
fn cuda_pow_half_exponent_uses_sqrt_domain_and_signed_zero_like_torch() {
    ensure_cuda_backend();

    let xf = cuda_f32(&[f32::NEG_INFINITY, -2.0, -0.0, 0.0, 4.0], false);
    assert_f32_bits(
        &host_f32(&pow(&xf, 0.5).expect("f32 pow half")),
        &[f32::NAN, f32::NAN, -0.0, 0.0, 2.0],
        "f32 pow 0.5",
    );
    assert_f32_bits(
        &host_f32(&pow(&xf, -0.5).expect("f32 pow neg half")),
        &[f32::NAN, f32::NAN, f32::NEG_INFINITY, f32::INFINITY, 0.5],
        "f32 pow -0.5",
    );

    let xd = cuda_f64(&[f64::NEG_INFINITY, -2.0, -0.0, 0.0, 4.0], false);
    assert_f64_bits(
        &host_f64(&pow(&xd, 0.5).expect("f64 pow half")),
        &[f64::NAN, f64::NAN, -0.0, 0.0, 2.0],
        "f64 pow 0.5",
    );
    assert_f64_bits(
        &host_f64(&pow(&xd, -0.5).expect("f64 pow neg half")),
        &[f64::NAN, f64::NAN, f64::NEG_INFINITY, f64::INFINITY, 0.5],
        "f64 pow -0.5",
    );
}

#[test]
fn cuda_pow_generic_fractional_negative_domain_matches_torch() {
    ensure_cuda_backend();

    let xf = cuda_f32(&[f32::NEG_INFINITY, -2.0, -0.0, 0.0, 4.0], false);
    assert_f32_close_or_nan(
        &host_f32(&pow(&xf, 1.5).expect("f32 pow 1.5")),
        &[f32::INFINITY, f32::NAN, 0.0, 0.0, 8.0],
        2e-5,
        "f32 pow 1.5",
    );

    let xd = cuda_f64(&[f64::NEG_INFINITY, -2.0, -0.0, 0.0, 4.0], false);
    assert_f64_close_or_nan(
        &host_f64(&pow(&xd, 1.5).expect("f64 pow 1.5")),
        &[f64::INFINITY, f64::NAN, 0.0, 0.0, 8.0],
        1e-10,
        "f64 pow 1.5",
    );
}

#[test]
fn cuda_pow_special_value_table_matches_torch_scalar_exponents() {
    ensure_cuda_backend();

    let xf = cuda_f32(
        &[
            f32::NEG_INFINITY,
            -2.0,
            -1.0,
            -0.0,
            0.0,
            1.0,
            2.0,
            f32::INFINITY,
            f32::NAN,
        ],
        false,
    );
    let f32_cases: &[(f32, &[f32])] = &[
        (
            -3.0,
            &[
                -0.0,
                -0.125,
                -1.0,
                f32::NEG_INFINITY,
                f32::INFINITY,
                1.0,
                0.125,
                0.0,
                f32::NAN,
            ],
        ),
        (
            4.0,
            &[
                f32::INFINITY,
                16.0,
                1.0,
                0.0,
                0.0,
                1.0,
                16.0,
                f32::INFINITY,
                f32::NAN,
            ],
        ),
        (
            f32::INFINITY,
            &[
                f32::INFINITY,
                f32::INFINITY,
                1.0,
                0.0,
                0.0,
                1.0,
                f32::INFINITY,
                f32::INFINITY,
                f32::NAN,
            ],
        ),
        (
            f32::NEG_INFINITY,
            &[
                0.0,
                0.0,
                1.0,
                f32::INFINITY,
                f32::INFINITY,
                1.0,
                0.0,
                0.0,
                f32::NAN,
            ],
        ),
        (
            f32::NAN,
            &[
                f32::NAN,
                f32::NAN,
                f32::NAN,
                f32::NAN,
                f32::NAN,
                1.0,
                f32::NAN,
                f32::NAN,
                f32::NAN,
            ],
        ),
    ];
    for &(exponent, expected) in f32_cases {
        assert_f32_close_or_nan(
            &host_f32(&pow(&xf, f64::from(exponent)).expect("f32 table pow")),
            expected,
            2e-5,
            &format!("f32 pow {exponent:?}"),
        );
    }

    let xd = cuda_f64(
        &[
            f64::NEG_INFINITY,
            -2.0,
            -1.0,
            -0.0,
            0.0,
            1.0,
            2.0,
            f64::INFINITY,
            f64::NAN,
        ],
        false,
    );
    let f64_cases: &[(f64, &[f64])] = &[
        (
            -3.0,
            &[
                -0.0,
                -0.125,
                -1.0,
                f64::NEG_INFINITY,
                f64::INFINITY,
                1.0,
                0.125,
                0.0,
                f64::NAN,
            ],
        ),
        (
            4.0,
            &[
                f64::INFINITY,
                16.0,
                1.0,
                0.0,
                0.0,
                1.0,
                16.0,
                f64::INFINITY,
                f64::NAN,
            ],
        ),
        (
            f64::INFINITY,
            &[
                f64::INFINITY,
                f64::INFINITY,
                1.0,
                0.0,
                0.0,
                1.0,
                f64::INFINITY,
                f64::INFINITY,
                f64::NAN,
            ],
        ),
        (
            f64::NEG_INFINITY,
            &[
                0.0,
                0.0,
                1.0,
                f64::INFINITY,
                f64::INFINITY,
                1.0,
                0.0,
                0.0,
                f64::NAN,
            ],
        ),
        (
            f64::NAN,
            &[
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                1.0,
                f64::NAN,
                f64::NAN,
                f64::NAN,
            ],
        ),
    ];
    for &(exponent, expected) in f64_cases {
        assert_f64_close_or_nan(
            &host_f64(&pow(&xd, exponent).expect("f64 table pow")),
            expected,
            1e-10,
            &format!("f64 pow {exponent:?}"),
        );
    }
}
