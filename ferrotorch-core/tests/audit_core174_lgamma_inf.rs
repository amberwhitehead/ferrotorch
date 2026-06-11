//! Red-then-green regression tests for audit finding CORE-174 (crosslink
//! #1868): `lgamma(±inf)` returns NaN — for `+inf` the Lanczos tail computes
//! `inf - inf = NaN`; for `-inf` the reflection computes `sin(π·-inf) = NaN`,
//! which evades the pole test. C99 `lgamma` and torch return `+inf` for both.
//!
//! Oracle (R-ORACLE-1 path (b)) — live torch 2.11.0+cu130, 2026-06-11,
//! this machine:
//!
//! ```python
//! >>> t = lambda v: torch.tensor(v, dtype=torch.float64)
//! >>> torch.lgamma(t(float('inf'))).item()    # inf
//! >>> torch.lgamma(t(float('-inf'))).item()   # inf
//! >>> torch.lgamma(t(float('nan'))).item()    # nan
//! >>> torch.lgamma(torch.tensor(float('-inf'))).item()  # inf (f32)
//! >>> torch.special.multigammaln(t(float('inf')), 2).item()  # inf
//! ```
//!
//! (scipy.special.gammaln(-inf) is -inf — the two oracles diverge at -inf;
//! torch + C99 are the documented contract for this op and win.)
//!
//! Tolerance justification (R-ORACLE-5): all pins are exact (infinity /
//! NaN-ness); the finite regression rows reuse the in-module 1e-8 bound of
//! the pre-existing `lgamma_known_values` tests (Lanczos g=7 kernel,
//! ~1e-15 relative for O(1) args; 1e-8 is the historical file gate).

use ferrotorch_core::special::{lgamma, multigammaln};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn t64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}
fn t32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

#[test]
fn core174_lgamma_infinite_inputs_are_plus_inf_f64() {
    let r = lgamma(&t64(&[f64::INFINITY, f64::NEG_INFINITY], &[2])).unwrap();
    let d = r.data().unwrap();
    assert!(
        d[0].is_infinite() && d[0] > 0.0,
        "lgamma(+inf) must be +inf per C99/torch, got {}",
        d[0]
    );
    assert!(
        d[1].is_infinite() && d[1] > 0.0,
        "lgamma(-inf) must be +inf per C99/torch, got {}",
        d[1]
    );
}

#[test]
fn core174_lgamma_infinite_inputs_are_plus_inf_f32() {
    let r = lgamma(&t32(&[f32::INFINITY, f32::NEG_INFINITY], &[2])).unwrap();
    let d = r.data().unwrap();
    assert!(
        d[0].is_infinite() && d[0] > 0.0,
        "lgamma f32 (+inf) must be +inf per C99/torch, got {}",
        d[0]
    );
    assert!(
        d[1].is_infinite() && d[1] > 0.0,
        "lgamma f32 (-inf) must be +inf per C99/torch, got {}",
        d[1]
    );
}

#[test]
fn core174_lgamma_nan_propagates_and_finite_unchanged() {
    let r = lgamma(&t64(&[f64::NAN, 1.0, 0.5, 6.0], &[4])).unwrap();
    let d = r.data().unwrap();
    assert!(d[0].is_nan(), "lgamma(NaN) must be NaN, got {}", d[0]);
    assert!(d[1].abs() < 1e-8, "lgamma(1) moved: got {}", d[1]);
    assert!(
        (d[2] - 0.5723649429247001).abs() < 1e-8,
        "lgamma(0.5) moved: got {}",
        d[2]
    );
    assert!(
        (d[3] - (120.0f64).ln()).abs() < 1e-8,
        "lgamma(6) moved: got {}",
        d[3]
    );
}

#[test]
fn core174_multigammaln_inf_propagation() {
    // The finding names multigammaln as a propagation site: with the lgamma
    // guard, multigammaln(inf, 2) = C + lgamma(inf) + lgamma(inf - 1/2) must
    // be +inf (torch live: inf), not NaN.
    let r = multigammaln(&t64(&[f64::INFINITY], &[1]), 2).unwrap();
    let v = r.data().unwrap()[0];
    assert!(
        v.is_infinite() && v > 0.0,
        "multigammaln(+inf, 2) must be +inf per torch, got {v}"
    );
}
