//! Divergence audit of commit `3d1dd0881` (#1374, Binomial distribution).
//!
//! The new `ferrotorch-distributions/src/binomial.rs` claims to mirror
//! `torch/distributions/binomial.py`. Its SCALAR-input behavior
//! (log_prob/mean/variance/mode/entropy/logits conversion, KL Binomial-Binomial
//! and Poisson-Binomial) matches torch 2.11 — those committed unit tests pass.
//!
//! The implementation had collapsed every batched/broadcast shape contract that
//! `binomial.py` inherits from `torch.distributions.Distribution`. The four
//! tests below pin divergences where ferrotorch's OUTPUT SHAPE (and, for
//! `enumerate_support`, the entire `expand` semantics) disagreed with torch for
//! a non-scalar `probs` batch, plus a dtype-independent clamp eps.
//!
//! Reference shapes/values from live `torch.distributions.Binomial` at float64
//! (torch 2.11.0+cu130, this machine, 2026-05-27). Non-tautological per
//! R-CHAR-3: every expected value is either a live-torch oracle output or a
//! named closed-form binomial-pmf bit, never copied from the ferrotorch side.
//!
//! Tracking: #1569 (closed). The fix landed in `binomial.rs` (batch_shape +
//! broadcast indexing in sample/log_prob/enumerate_support + `T::epsilon()`
//! clamp); these tests are now un-ignored permanent regression coverage.

use ferrotorch_core::creation::{from_slice, scalar};
use ferrotorch_distributions::{Binomial, Distribution};

/// Divergence: ferrotorch's `Binomial::log_prob` diverges from
/// `pytorch torch/distributions/binomial.py:140-158` for a scalar `value`
/// against a *batched* `probs`.
///
/// torch:  `Binomial(10, [0.3, 0.5]).log_prob(tensor(3.0))`
///   broadcasts the scalar value against the batch and returns shape `[2]` =
///   `[-1.321151277766889, -2.1439800628174055]` (live torch 2.11, f64).
/// ferrotorch: iterates over the *value* tensor (length 1), indexing `probs`
///   by value position, so it emits a single element of shape `[]` and never
///   sees `probs[1]`.
/// Tracking: #1569
#[test]
fn divergence_binomial_log_prob_batched_probs_scalar_value() {
    let dist = Binomial::new(
        scalar(10.0f64).unwrap(),
        from_slice(&[0.3f64, 0.5], &[2]).unwrap(),
    )
    .unwrap();
    let lp = dist.log_prob(&scalar(3.0f64).unwrap()).unwrap();

    // torch returns the per-batch log_prob, shape [2].
    assert_eq!(
        lp.shape(),
        &[2],
        "torch broadcasts scalar value against batched probs -> shape [2]"
    );
    let d = lp.data().unwrap();
    // Live torch 2.11 f64 oracle values.
    assert!(
        (d[0] - (-1.321_151_277_766_889)).abs() < 1e-10,
        "log_prob for p=0.3 should be torch's -1.321151277766889, got {}",
        d[0]
    );
    assert!(
        (d[1] - (-2.143_980_062_817_405_5)).abs() < 1e-10,
        "log_prob for p=0.5 should be torch's -2.1439800628174055, got {}",
        d[1]
    );
}

/// Divergence: ferrotorch's `Binomial::sample` diverges from
/// `pytorch torch/distributions/binomial.py:133-138` for a *batched* `probs`.
///
/// torch:  `Binomial(10, [0.3, 0.5]).sample((4,))` returns
///   `self._extended_shape(sample_shape) = sample_shape + batch_shape`, i.e.
///   shape `[4, 2]` (live torch 2.11).
/// ferrotorch: treats the argument as the *entire* output shape and produces
///   `[4]`, never appending the batch dim. The two batch elements (p=0.3 vs
///   p=0.5) are folded into the same flat draw via `i % probs_data.len()`.
/// Tracking: #1569
#[test]
fn divergence_binomial_sample_extends_batch_shape() {
    let dist = Binomial::new(
        scalar(10.0f64).unwrap(),
        from_slice(&[0.3f64, 0.5], &[2]).unwrap(),
    )
    .unwrap();
    let s = dist.sample(&[4]).unwrap();
    assert_eq!(
        s.shape(),
        &[4, 2],
        "torch sample(sample_shape) -> sample_shape + batch_shape = [4, 2]"
    );
}

/// Divergence: ferrotorch's `Binomial::enumerate_support` diverges from
/// `pytorch torch/distributions/binomial.py:170-182` for a *batched* `probs`:
/// it ignores both the batch shape AND the `expand` flag.
///
/// torch (n=4, batch=[2]):
///   `enumerate_support(False)` -> shape `(5, 1)` (values viewed as
///       `(-1,) + (1,)*len(batch_shape)`),
///   `enumerate_support(True)`  -> shape `(5, 2)` (expanded over the batch).
/// ferrotorch: returns shape `[5]` for *both* expand values.
/// Tracking: #1569
#[test]
fn divergence_binomial_enumerate_support_batch_and_expand() {
    let dist = Binomial::new(
        scalar(4.0f64).unwrap(),
        from_slice(&[0.3f64, 0.5], &[2]).unwrap(),
    )
    .unwrap();

    let es_no_expand = dist.enumerate_support(false).unwrap();
    assert_eq!(
        es_no_expand.shape(),
        &[5, 1],
        "torch enumerate_support(False) on batch [2] -> shape (5, 1)"
    );

    let es_expand = dist.enumerate_support(true).unwrap();
    assert_eq!(
        es_expand.shape(),
        &[5, 2],
        "torch enumerate_support(True) on batch [2] -> shape (5, 2)"
    );
}

/// Divergence: ferrotorch's `Binomial` logits/probs conversion + `log_prob`
/// hardcode a dtype-independent clamp `eps = 1e-7`, diverging from
/// `pytorch torch/distributions/utils.py clamp_probs` which clamps with
/// `torch.finfo(dtype).eps` (= 2.22e-16 for float64), referenced by
/// `binomial.py:121-127,140-158`.
///
/// For f64 `p = 1 - 1e-9` (inside torch's f64 clamp window, NOT inside
/// ferrotorch's 1e-7 window):
///   torch:  `Binomial(3, tensor(1-1e-9)).logits` = 20.723265864228342
///           `Binomial(10, tensor(1-1e-9)).log_prob(10.)` = -1.0000007932831068e-08
///   ferrotorch: clamps p to 1-1e-7, giving logits 16.118... and
///           log_prob ~ -1.0e-6 (100x off).
/// Tracking: #1569
#[test]
fn divergence_binomial_f64_clamp_eps_too_coarse() {
    let p = 1.0f64 - 1e-9;

    // logits accessor.
    let d_logit = Binomial::new(scalar(3.0f64).unwrap(), scalar(p).unwrap()).unwrap();
    let l = d_logit.logits().unwrap().item().unwrap();
    assert!(
        (l - 20.723_265_864_228_342).abs() < 1e-6,
        "torch f64 logits(1-1e-9) = 20.723265864228342, got {l}"
    );

    // log_prob at k = n (only the ln(p) term survives).
    let d_lp = Binomial::new(scalar(10.0f64).unwrap(), scalar(p).unwrap()).unwrap();
    let lp = d_lp
        .log_prob(&scalar(10.0f64).unwrap())
        .unwrap()
        .item()
        .unwrap();
    // torch oracle: -1.0000007932831068e-08 (~ 10 * ln(1-1e-9)).
    assert!(
        (lp - (-1.000_000_793_283_106_8e-8)).abs() < 1e-12,
        "torch f64 log_prob(k=10, p=1-1e-9) = -1.0000007932831068e-08, got {lp}"
    );
}
