//! Divergence pins for commit 8577b8acd (#1379) — `torch.special` public ops
//! in `ferrotorch-core::special`.
//!
//! All expected values below are constructed from LIVE torch 2.11 + scipy 1.17
//! on this machine (2026-05-27), NOT copied from the ferrotorch side (R-CHAR-3).
//!
//! Oracle session (`python3`):
//!   >>> import torch, scipy.special as sp
//!   >>> torch.special.multigammaln(torch.tensor([0.3], dtype=torch.float64), 3).item()
//!   6.026863353182922
//!   >>> torch.special.multigammaln(torch.tensor([-0.4], dtype=torch.float64), 2).item()
//!   4.244962700260123
//!   >>> torch.special.multigammaln(torch.tensor([0.5], dtype=torch.float64), 3).item()
//!   inf
//!
//! torch's `mvlgamma` kernel (aten/src/ATen/native/UnaryOps.cpp:887-905) performs
//! NO domain check beyond `p >= 1`. It computes
//!   args = arange(-p*0.5 + 0.5, 0.5, 0.5); args.add(self).lgamma_().sum(-1) + C
//! verbatim, so out-of-domain inputs `a <= (p-1)/2` yield the ordinary (finite,
//! real) value of `Σ lgamma(a - (i-1)/2)` (the docstring at
//! torch/special/__init__.py:862 only says the result is "undefined", i.e.
//! mathematically meaningless — it does NOT promise NaN). The only way torch
//! produces a non-finite result here is when one of the lgamma arguments lands
//! exactly on a non-positive integer pole (→ +inf), as for `a=0.5, p=3`
//! (args 0.5, 0.0, -0.5 → lgamma(0.0) = +inf).
//!
//! These tests are intentionally left UN-`#[ignore]`d: the op silently returns
//! NaN where torch returns a real (or +inf) value, which is a release-blocker
//! for #1379. The failing test IS the block. Tracking: #1571.

use ferrotorch_core::{multigammaln, Tensor, TensorStorage};

fn t(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

/// Divergence: `ferrotorch_core::multigammaln` diverges from
/// `pytorch aten/src/ATen/native/UnaryOps.cpp:887` (`mvlgamma`) for an
/// out-of-domain input `a=0.3, p=3` (a <= (p-1)/2 = 1.0).
/// Upstream returns 6.026863353182922 (the ordinary finite value of
/// `Σ_{i=1}^3 lgamma(a - (i-1)/2)` — no domain guard exists in the kernel).
/// ferrotorch returns NaN due to a fabricated `af <= (pf-1.0)/2.0 → NaN` guard
/// at special.rs:891 that has no upstream counterpart.
/// Tracking: #1571
#[test]
fn divergence_multigammaln_out_of_domain_returns_finite_p3() {
    // Oracle: torch.special.multigammaln(0.3, 3) == 6.026863353182922
    let expected = 6.026863353182922_f64;
    let a = t(&[0.3], &[1]);
    let r = multigammaln(&a, 3).unwrap();
    let got = r.data().unwrap()[0];
    assert!(
        (got - expected).abs() < 1e-12,
        "multigammaln(0.3, 3): torch returns {expected}, ferrotorch returns {got}"
    );
}

/// Divergence: same fabricated guard, negative non-integer input.
/// `a=-0.4, p=2` (a <= (p-1)/2 = 0.5).
/// Upstream `torch.special.multigammaln(-0.4, 2)` returns 4.244962700260123.
/// ferrotorch returns NaN.
/// Tracking: #1571
#[test]
fn divergence_multigammaln_out_of_domain_returns_finite_p2() {
    // Oracle: torch.special.multigammaln(-0.4, 2) == 4.244962700260123
    let expected = 4.244962700260123_f64;
    let a = t(&[-0.4], &[1]);
    let r = multigammaln(&a, 2).unwrap();
    let got = r.data().unwrap()[0];
    assert!(
        (got - expected).abs() < 1e-12,
        "multigammaln(-0.4, 2): torch returns {expected}, ferrotorch returns {got}"
    );
}

/// Divergence: at `a=0.5, p=3` torch returns +inf (one lgamma argument is the
/// integer pole 0.0 → lgamma = +inf), NOT NaN.
/// Upstream `torch.special.multigammaln(0.5, 3)` returns inf.
/// ferrotorch returns NaN (the domain guard fires before the pole is reached).
/// Tracking: #1571
#[test]
fn divergence_multigammaln_pole_argument_is_pos_inf_not_nan() {
    // Oracle: torch.special.multigammaln(0.5, 3) == inf
    let a = t(&[0.5], &[1]);
    let r = multigammaln(&a, 3).unwrap();
    let got = r.data().unwrap()[0];
    assert!(
        got == f64::INFINITY,
        "multigammaln(0.5, 3): torch returns +inf, ferrotorch returns {got}"
    );
}
