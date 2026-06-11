//! Red-then-green regression tests for audit finding CORE-173 (crosslink
//! #1867): `digamma` computes the reflection `cot(π·x)` directly, and since
//! π is rounded, `sin(π·-1.0)` is ~-1.22e-16 rather than 0 — so
//! `digamma(-1.0)` returns ~-2.57e16: large finite garbage at every
//! negative-integer pole, in both the f64 and the f32 paths.
//!
//! Upstream contract: `pytorch/aten/src/ATen/native/Math.h:375-398
//! calc_digamma(double)` (and `:433-456` float):
//!   - `x == 0` → `copysign(INFINITY, -x)` (C99/SciPy: ±0 → ∓∞)
//!   - `x < 0` and integer → NaN
//!
//! Oracle (R-ORACLE-1 path (b)) — live torch 2.11.0+cu130, 2026-06-11,
//! this machine:
//!
//! ```python
//! >>> t = lambda v: torch.tensor(v, dtype=torch.float64)
//! >>> torch.digamma(t(-1.0)).item()    # nan
//! >>> torch.digamma(t(-2.0)).item()    # nan
//! >>> torch.digamma(t(-100.0)).item()  # nan
//! >>> torch.digamma(t(0.0)).item()     # -inf
//! >>> torch.digamma(t(-0.0)).item()    # inf
//! >>> torch.digamma(t(-0.5)).item()    # 0.03648997397857639
//! >>> torch.digamma(t(-1.5)).item()    # 0.7031566406452433
//! >>> torch.digamma(torch.tensor(-1.0)).item()  # nan (f32 lane)
//! >>> torch.digamma(torch.tensor(-0.0)).item()  # inf (f32 lane)
//! ```
//!
//! Tolerance justification (R-ORACLE-5): pole/zero pins are exact
//! (NaN-ness / signed infinity). The negative non-integer regression pins
//! use 1e-10 absolute in f64 (the documented F64 transcendental gate of the
//! `digamma_f64_hi` Bernoulli-Stirling kernel, residual ≤ ~4e-16) and 1e-5
//! in f32 (F32 transcendental gate of the legacy 4-term expansion); both
//! sit ≥10 orders below the ~1e16 pole garbage being pinned away.

use ferrotorch_core::special::digamma;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn t64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}
fn t32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

#[test]
fn core173_digamma_negative_integer_poles_are_nan_f64() {
    let r = digamma(&t64(&[-1.0, -2.0, -100.0], &[3])).unwrap();
    let d = r.data().unwrap();
    for (i, v) in d.iter().enumerate() {
        assert!(
            v.is_nan(),
            "digamma(negative integer) must be NaN per torch calc_digamma, lane {i} got {v:e}"
        );
    }
}

#[test]
fn core173_digamma_negative_integer_poles_are_nan_f32() {
    let r = digamma(&t32(&[-1.0, -2.0, -100.0], &[3])).unwrap();
    let d = r.data().unwrap();
    for (i, v) in d.iter().enumerate() {
        assert!(
            v.is_nan(),
            "digamma f32 (negative integer) must be NaN per torch calc_digamma, lane {i} got {v:e}"
        );
    }
}

#[test]
fn core173_digamma_zero_keeps_signed_infinity() {
    // x == 0 is NOT a NaN case: torch returns copysign(inf, -x).
    let r = digamma(&t64(&[0.0, -0.0], &[2])).unwrap();
    let d = r.data().unwrap();
    assert!(
        d[0].is_infinite() && d[0] < 0.0,
        "digamma(+0.0) must be -inf, got {}",
        d[0]
    );
    assert!(
        d[1].is_infinite() && d[1] > 0.0,
        "digamma(-0.0) must be +inf, got {}",
        d[1]
    );
    let r32 = digamma(&t32(&[0.0, -0.0], &[2])).unwrap();
    let d32 = r32.data().unwrap();
    assert!(
        d32[0].is_infinite() && d32[0] < 0.0,
        "digamma f32 (+0.0) must be -inf, got {}",
        d32[0]
    );
    assert!(
        d32[1].is_infinite() && d32[1] > 0.0,
        "digamma f32 (-0.0) must be +inf, got {}",
        d32[1]
    );
}

#[test]
fn core173_digamma_negative_noninteger_unchanged() {
    let r = digamma(&t64(&[-0.5, -1.5], &[2])).unwrap();
    let d = r.data().unwrap();
    assert!(
        (d[0] - 0.03648997397857639).abs() < 1e-10,
        "digamma(-0.5) moved: got {}",
        d[0]
    );
    assert!(
        (d[1] - 0.7031566406452433).abs() < 1e-10,
        "digamma(-1.5) moved: got {}",
        d[1]
    );
    let r32 = digamma(&t32(&[-0.5], &[1])).unwrap();
    assert!(
        (r32.data().unwrap()[0] - 0.036489974).abs() < 1e-5,
        "digamma f32 (-0.5) moved: got {}",
        r32.data().unwrap()[0]
    );
}
