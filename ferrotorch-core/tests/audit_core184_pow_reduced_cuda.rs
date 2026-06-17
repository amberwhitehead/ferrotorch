#![cfg(feature = "gpu")]

//! Regression coverage for CORE-184 / crosslink #1878: CUDA tensor-scalar
//! `pow` must support reduced precision dtypes without CPU detours.
//!
//! PyTorch source target:
//! - `/home/doll/pytorch/aten/src/ATen/native/cuda/PowKernel.cu:170-203`
//!   dispatches tensor-scalar CUDA pow through
//!   `AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, ...)`.
//! - `/home/doll/pytorch/aten/src/ATen/native/cuda/Pow.cuh:20-25` computes
//!   Half/BFloat16 pow in f32 opmath and casts back to the storage dtype.
//!
//! Live oracle on this machine (torch 2.11.0+cu130, cuda:0) was used for
//! every expected table below. The oracle also confirms result and gradient
//! dtypes are `torch.float16` / `torch.bfloat16` and device is CUDA.

use std::sync::Once;

use ferrotorch_core::creation::from_vec;
use ferrotorch_core::device::Device;
use ferrotorch_core::grad_fns::arithmetic::pow;
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::tensor::Tensor;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for reduced pow CUDA tests");
    });
}

fn bf16_cuda(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<half::bf16> {
    let values: Vec<half::bf16> = data.iter().copied().map(half::bf16::from_f32).collect();
    from_vec::<half::bf16>(values, shape)
        .expect("bf16 CPU tensor")
        .to(Device::Cuda(0))
        .expect("upload bf16")
        .detach()
        .requires_grad_(requires_grad)
}

fn f16_cuda(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<half::f16> {
    let values: Vec<half::f16> = data.iter().copied().map(half::f16::from_f32).collect();
    from_vec::<half::f16>(values, shape)
        .expect("f16 CPU tensor")
        .to(Device::Cuda(0))
        .expect("upload f16")
        .detach()
        .requires_grad_(requires_grad)
}

fn host_bf16(t: &Tensor<half::bf16>) -> Vec<f32> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "bf16 result must stay CUDA-resident"
    );
    t.cpu()
        .expect("bf16 D2H")
        .data()
        .expect("bf16 CPU data")
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

fn host_f16(t: &Tensor<half::f16>) -> Vec<f32> {
    assert_eq!(
        t.device(),
        Device::Cuda(0),
        "f16 result must stay CUDA-resident"
    );
    t.cpu()
        .expect("f16 D2H")
        .data()
        .expect("f16 CPU data")
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

fn assert_exact_or_nan(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: len");
    for (i, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        if want.is_nan() {
            assert!(got.is_nan(), "{label}[{i}]: got {got:?}, want NaN");
        } else {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{label}[{i}]: got {got:?} bits={:#x}, want {want:?} bits={:#x}",
                got.to_bits(),
                want.to_bits()
            );
        }
    }
}

fn assert_close_or_nan(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: len");
    for (i, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        if want.is_nan() {
            assert!(got.is_nan(), "{label}[{i}]: got {got:?}, want NaN");
        } else if want == 0.0 || want.is_infinite() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{label}[{i}]: signed-zero/inf mismatch"
            );
        } else {
            assert!(
                (got - want).abs() <= tol,
                "{label}[{i}]: got {got}, want {want}, tol {tol}"
            );
        }
    }
}

fn assert_grad_bf16(leaf: &Tensor<half::bf16>, expected: &[f32], label: &str) {
    let grad = leaf
        .grad()
        .expect("bf16 grad slot")
        .unwrap_or_else(|| panic!("{label}: no gradient reached bf16 CUDA leaf"));
    assert_eq!(grad.device(), Device::Cuda(0), "{label}: gradient device");
    assert_exact_or_nan(&host_bf16(&grad), expected, label);
}

fn assert_grad_f16(leaf: &Tensor<half::f16>, expected: &[f32], label: &str) {
    let grad = leaf
        .grad()
        .expect("f16 grad slot")
        .unwrap_or_else(|| panic!("{label}: no gradient reached f16 CUDA leaf"));
    assert_eq!(grad.device(), Device::Cuda(0), "{label}: gradient device");
    assert_exact_or_nan(&host_f16(&grad), expected, label);
}

#[test]
fn reduced_cuda_pow_special_exponents_match_torch_table() {
    ensure_cuda_backend();

    let values = [
        f32::NEG_INFINITY,
        -2.0,
        -1.0,
        -0.0,
        0.0,
        0.25,
        0.5,
        1.0,
        2.0,
        f32::INFINITY,
        f32::NAN,
    ];
    let cases: &[(f64, &[f32])] = &[
        (
            0.0,
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        (
            0.5,
            &[
                f32::NAN,
                f32::NAN,
                f32::NAN,
                -0.0,
                0.0,
                0.5,
                0.70703125,
                1.0,
                1.4140625,
                f32::INFINITY,
                f32::NAN,
            ],
        ),
        (
            -0.5,
            &[
                f32::NAN,
                f32::NAN,
                f32::NAN,
                f32::NEG_INFINITY,
                f32::INFINITY,
                2.0,
                1.4140625,
                1.0,
                0.70703125,
                0.0,
                f32::NAN,
            ],
        ),
        (
            -1.0,
            &[
                -0.0,
                -0.5,
                -1.0,
                f32::NEG_INFINITY,
                f32::INFINITY,
                4.0,
                2.0,
                1.0,
                0.5,
                0.0,
                f32::NAN,
            ],
        ),
        (
            2.0,
            &[
                f32::INFINITY,
                4.0,
                1.0,
                0.0,
                0.0,
                0.0625,
                0.25,
                1.0,
                4.0,
                f32::INFINITY,
                f32::NAN,
            ],
        ),
        (
            3.0,
            &[
                f32::NEG_INFINITY,
                -8.0,
                -1.0,
                -0.0,
                0.0,
                0.015625,
                0.125,
                1.0,
                8.0,
                f32::INFINITY,
                f32::NAN,
            ],
        ),
        (
            -2.0,
            &[
                0.0,
                0.25,
                1.0,
                f32::INFINITY,
                f32::INFINITY,
                16.0,
                4.0,
                1.0,
                0.25,
                0.0,
                f32::NAN,
            ],
        ),
    ];

    for &(exp, expected) in cases {
        let bf = bf16_cuda(&values, &[values.len()], false);
        assert_exact_or_nan(&host_bf16(&bf), &values, "bf16 input transport");
        assert_exact_or_nan(
            &host_bf16(&pow(&bf, exp).expect("bf16 CUDA pow")),
            expected,
            &format!("bf16 pow({exp})"),
        );

        let hf = f16_cuda(&values, &[values.len()], false);
        assert_exact_or_nan(&host_f16(&hf), &values, "f16 input transport");
        assert_exact_or_nan(
            &host_f16(&pow(&hf, exp).expect("f16 CUDA pow")),
            expected,
            &format!("f16 pow({exp})"),
        );
    }
}

#[test]
fn reduced_cuda_pow_generic_fractional_uses_dtype_rounded_exponent() {
    ensure_cuda_backend();

    let values = [0.25, 0.5, 1.0, 2.0];

    let bf = bf16_cuda(&values, &[values.len()], false);
    assert_close_or_nan(
        &host_bf16(&pow(&bf, 1.7).expect("bf16 CUDA pow 1.7")),
        &[0.094_238_28, 0.306_640_63, 1.0, 3.25],
        5e-4,
        "bf16 pow 1.7",
    );

    let hf = f16_cuda(&values, &[values.len()], false);
    assert_close_or_nan(
        &host_f16(&pow(&hf, 1.7).expect("f16 CUDA pow 1.7")),
        &[0.094_726_56, 0.307_861_33, 1.0, 3.25],
        5e-4,
        "f16 pow 1.7",
    );
}

#[test]
fn reduced_cuda_pow_backward_stays_cuda_and_matches_torch() {
    ensure_cuda_backend();

    let bf = bf16_cuda(&[0.25, 0.5, 1.5, 2.0], &[4], true);
    sum(&pow(&bf, 2.0).expect("bf16 tracked pow2"))
        .expect("bf16 sum")
        .backward()
        .expect("bf16 pow2 backward");
    assert_grad_bf16(&bf, &[0.5, 1.0, 3.0, 4.0], "bf16 pow2 grad");

    let hf = f16_cuda(&[0.25, 0.5, 1.5, 2.0], &[4], true);
    sum(&pow(&hf, 2.0).expect("f16 tracked pow2"))
        .expect("f16 sum")
        .backward()
        .expect("f16 pow2 backward");
    assert_grad_f16(&hf, &[0.5, 1.0, 3.0, 4.0], "f16 pow2 grad");

    let bf = bf16_cuda(&[0.25, 0.5, 1.0, 4.0], &[4], true);
    sum(&pow(&bf, 0.5).expect("bf16 tracked sqrt-pow"))
        .expect("bf16 sum")
        .backward()
        .expect("bf16 pow0.5 backward");
    assert_grad_bf16(&bf, &[1.0, 0.70703125, 0.5, 0.25], "bf16 pow0.5 grad");

    let hf = f16_cuda(&[0.25, 0.5, 1.0, 4.0], &[4], true);
    sum(&pow(&hf, 0.5).expect("f16 tracked sqrt-pow"))
        .expect("f16 sum")
        .backward()
        .expect("f16 pow0.5 backward");
    assert_grad_f16(&hf, &[1.0, 0.70703125, 0.5, 0.25], "f16 pow0.5 grad");
}

#[test]
fn reduced_cuda_pow_handles_empty_and_noncontiguous_views() {
    ensure_cuda_backend();

    let empty_bf = bf16_cuda(&[], &[0], false);
    let empty_bf_out = pow(&empty_bf, 2.0).expect("empty bf16 CUDA pow");
    assert_eq!(empty_bf_out.shape(), &[0]);
    assert_eq!(host_bf16(&empty_bf_out), Vec::<f32>::new());

    let empty_hf = f16_cuda(&[], &[0], false);
    let empty_hf_out = pow(&empty_hf, 2.0).expect("empty f16 CUDA pow");
    assert_eq!(empty_hf_out.shape(), &[0]);
    assert_eq!(host_f16(&empty_hf_out), Vec::<f32>::new());

    let base_bf = bf16_cuda(&[1.0, 4.0, 9.0, 16.0, 25.0, 36.0], &[2, 3], false);
    let view_bf = base_bf.transpose(0, 1).expect("bf16 transpose view");
    assert_eq!(view_bf.shape(), &[3, 2]);
    assert_exact_or_nan(
        &host_bf16(&pow(&view_bf, 0.5).expect("bf16 noncontig pow")),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        "bf16 noncontiguous pow",
    );

    let base_hf = f16_cuda(&[1.0, 4.0, 9.0, 16.0, 25.0, 36.0], &[2, 3], false);
    let view_hf = base_hf.transpose(0, 1).expect("f16 transpose view");
    assert_eq!(view_hf.shape(), &[3, 2]);
    assert_exact_or_nan(
        &host_f16(&pow(&view_hf, 0.5).expect("f16 noncontig pow")),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        "f16 noncontiguous pow",
    );
}
