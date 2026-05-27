//! Affirmative cross-checks for commit `0d61d0bdd` (#1374, recursion-based KL).
//!
//! These probe the focus areas where a type-erasure / sum-rightmost bug WOULD
//! hide but — verified against live torch 2.11.0 (this machine, 2026-05-27) —
//! ferrotorch matches. They are EXPECTED TO PASS; if any flips red, the audit
//! verdict changes. Reference values are live-torch `kl_divergence` outputs at
//! float64 (R-CHAR-3: traced to `_kl_independent_independent`
//! `torch/distributions/kl.py:944-949` recursing into the base pair formula,
//! NOT copied from the ferrotorch side).

use ferrotorch_core::creation::from_slice;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_distributions::kl::kl_divergence;
use ferrotorch_distributions::{Independent, Laplace, Normal};

fn vec2(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    from_slice(data, shape).unwrap()
}

fn assert_close(got: &[f64], want: &[f64], tol: f64, ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "{ctx}[{i}]: got {g}, want {w} (tol {tol})"
        );
    }
}

/// Type-erased downcast hook routes to the CORRECT concrete pair: an
/// `Independent` wrapping `Laplace` (NOT Normal) must dispatch the base to the
/// `Laplace-Laplace` arm, then `_sum_rightmost(.., 1)`. A type-erasure bug that
/// routed to a wrong/default arm would produce a different value.
/// torch: `kl_divergence(Independent(Laplace,1), Independent(Laplace,1))`
/// -> [0.6243445787531372, 1.3448632609419877], shape [2].
#[test]
fn affirm_kl_independent_laplace_routes_to_laplace_arm() {
    let p = Independent::new(
        Laplace::new(
            vec2(&[0.0, 1.0, 2.0, 3.0], &[2, 2]),
            vec2(&[1.0, 2.0, 0.5, 1.5], &[2, 2]),
        )
        .unwrap(),
        1,
    )
    .unwrap();
    let q = Independent::new(
        Laplace::new(
            vec2(&[0.5, 0.0, 1.0, 2.0], &[2, 2]),
            vec2(&[1.2, 1.0, 1.0, 0.8], &[2, 2]),
        )
        .unwrap(),
        1,
    )
    .unwrap();
    let kl = kl_divergence(&p, &q).unwrap();
    assert_eq!(kl.shape(), &[2], "Independent(.,1) reduces the last dim");
    assert_close(
        kl.data().unwrap(),
        &[0.624_344_578_753_137_2, 1.344_863_260_941_987_7],
        1e-9,
        "ind(laplace) ndims=1",
    );
}

/// reinterpreted_batch_ndims=1 sum-rightmost over the task's exact [2,2] case.
/// torch value [0.125, 0.5], shape [2].
#[test]
fn affirm_kl_independent_normal_ndims1_value_and_shape() {
    let p = Independent::new(
        Normal::new(
            vec2(&[0.0, 1.0, 2.0, 3.0], &[2, 2]),
            vec2(&[1.0, 1.0, 1.0, 1.0], &[2, 2]),
        )
        .unwrap(),
        1,
    )
    .unwrap();
    let q = Independent::new(
        Normal::new(
            vec2(&[0.5, 1.0, 2.0, 2.0], &[2, 2]),
            vec2(&[1.0, 1.0, 1.0, 1.0], &[2, 2]),
        )
        .unwrap(),
        1,
    )
    .unwrap();
    let kl = kl_divergence(&p, &q).unwrap();
    assert_eq!(kl.shape(), &[2]);
    assert_close(kl.data().unwrap(), &[0.125, 0.5], 1e-12, "ndims1");
}

/// reinterpreted_batch_ndims=2 reduces BOTH dims -> scalar 0.625 (torch).
#[test]
fn affirm_kl_independent_normal_ndims2_scalar() {
    let p = Independent::new(
        Normal::new(
            vec2(&[0.0, 1.0, 2.0, 3.0], &[2, 2]),
            vec2(&[1.0, 1.0, 1.0, 1.0], &[2, 2]),
        )
        .unwrap(),
        2,
    )
    .unwrap();
    let q = Independent::new(
        Normal::new(
            vec2(&[0.5, 1.0, 2.0, 2.0], &[2, 2]),
            vec2(&[1.0, 1.0, 1.0, 1.0], &[2, 2]),
        )
        .unwrap(),
        2,
    )
    .unwrap();
    let kl = kl_divergence(&p, &q).unwrap();
    assert_eq!(kl.shape(), [] as [usize; 0]);
    assert_close(&[kl.item().unwrap()], &[0.625], 1e-12, "ndims2");
}

/// Mismatched reinterpreted_batch_ndims must error (torch raises
/// NotImplementedError at `kl.py:946-947`). p: ndims=1 over [2]; q: ndims=2
/// over [2,2].
#[test]
fn affirm_kl_independent_ndims_mismatch_errs() {
    let p = Independent::new(
        Normal::new(vec2(&[0.0, 0.0], &[2]), vec2(&[1.0, 1.0], &[2])).unwrap(),
        1,
    )
    .unwrap();
    let q = Independent::new(
        Normal::new(vec2(&[0.0, 0.0, 0.0, 0.0], &[2, 2]), vec2(&[1.0; 4], &[2, 2])).unwrap(),
        2,
    )
    .unwrap();
    assert!(kl_divergence(&p, &q).is_err());
}
