//! CORE-194 (#1946): `lgamma` must return `+inf` at every non-positive
//! integer pole, not a large finite value leaked through the rounded-pi
//! reflection formula.
//!
//! PyTorch source oracle:
//! - CPU `lgamma` dispatches through `IMPLEMENT_FLOAT_KERNEL_WITH_AVX512(lgamma)`
//!   in `aten/src/ATen/native/cpu/UnaryOpsKernel.cpp`, i.e. the platform
//!   `std::lgamma` family.
//! - CUDA dispatches through `lgamma_kernel_cuda` in
//!   `aten/src/ATen/native/cuda/UnaryGammaKernels.cu`.
//!
//! Live PyTorch 2.11.0+cu130 probes on this machine:
//! ```python
//! torch.lgamma(torch.tensor([-0.0, 0.0, -1.0, -2.0, -100.0],
//!                           dtype=torch.float64))
//! # tensor([inf, inf, inf, inf, inf], dtype=torch.float64)
//! torch.lgamma(torch.tensor([-0.5, -1.5], dtype=torch.float64))
//! # tensor([1.2655121234846454, 0.860047015376481], dtype=torch.float64)
//! torch.special.gammaln(torch.tensor([-1.0], dtype=torch.float64)).item()
//! # inf
//! torch.special.multigammaln(torch.tensor([0.5], dtype=torch.float64), 3)
//! # tensor([inf], dtype=torch.float64)
//! ```

use ferrotorch_core::special::{lgamma, multigammaln};
use ferrotorch_core::{Tensor, TensorStorage, gammaln};

fn t64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("f64 tensor")
}

fn t32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("f32 tensor")
}

#[test]
fn lgamma_non_positive_integer_poles_are_positive_infinity_f64() {
    let out = lgamma(&t64(&[-0.0, 0.0, -1.0, -2.0, -100.0], &[5])).expect("lgamma");
    let data = out.data().expect("cpu data");
    for (i, &value) in data.iter().enumerate() {
        assert!(
            value.is_infinite() && value.is_sign_positive(),
            "f64 lgamma pole lane {i} must be +inf like PyTorch, got {value:e}"
        );
    }
}

#[test]
fn lgamma_non_positive_integer_poles_are_positive_infinity_f32() {
    let out = lgamma(&t32(&[-0.0, 0.0, -1.0, -2.0, -100.0], &[5])).expect("lgamma");
    let data = out.data().expect("cpu data");
    for (i, &value) in data.iter().enumerate() {
        assert!(
            value.is_infinite() && value.is_sign_positive(),
            "f32 lgamma pole lane {i} must be +inf like PyTorch, got {value:e}"
        );
    }
}

#[test]
fn lgamma_negative_non_integer_reflection_values_remain_finite() {
    let out = lgamma(&t64(&[-0.5, -1.5], &[2])).expect("lgamma");
    let data = out.data().expect("cpu data");
    let expected = [1.2655121234846454, 0.860047015376481];
    for (i, (&actual, &want)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - want).abs() < 1e-12,
            "f64 lgamma non-pole lane {i}: expected {want}, got {actual}"
        );
    }

    let out32 = lgamma(&t32(&[-0.5, -1.5], &[2])).expect("lgamma");
    let data32 = out32.data().expect("cpu data");
    let expected32 = [1.2655121_f32, 0.86004704_f32];
    for (i, (&actual, &want)) in data32.iter().zip(expected32.iter()).enumerate() {
        assert!(
            (actual - want).abs() < 1e-5,
            "f32 lgamma non-pole lane {i}: expected {want}, got {actual}"
        );
    }
}

#[test]
fn special_gammaln_alias_path_uses_same_pole_contract() {
    let out = gammaln(&t64(&[-1.0], &[1])).expect("gammaln");
    let value = out.data().expect("cpu data")[0];
    assert!(
        value.is_infinite() && value.is_sign_positive(),
        "special::gammaln(-1.0) must be +inf like torch.special.gammaln, got {value:e}"
    );
}

#[test]
fn multigammaln_propagates_lgamma_poles_as_positive_infinity() {
    let out = multigammaln(&t64(&[0.5], &[1]), 3).expect("multigammaln");
    let value = out.data().expect("cpu data")[0];
    assert!(
        value.is_infinite() && value.is_sign_positive(),
        "multigammaln(0.5, 3) must be +inf when one lgamma argument is a pole, got {value:e}"
    );
}
