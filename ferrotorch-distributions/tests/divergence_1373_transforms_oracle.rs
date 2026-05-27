//! Adversarial parity audit of the 11 transforms shipped in commit `a2ab04347`
//! (#1373). Every expected value below is OracleDerived from LIVE torch 2.11.0
//! (`torch.distributions.transforms`) — see the `cd /tmp && python3` traces in
//! the critic session. No expected value is copied from the ferrotorch side
//! (R-CHAR-3).
//!
//! Each test runs the *actual* ferrotorch transform and compares forward /
//! inverse / log_abs_det_jacobian against the torch reference. A failing test
//! here pins a real divergence; a passing test confirms parity for that leg.

use ferrotorch_core::creation::from_slice;
use ferrotorch_distributions::{
    CorrCholeskyTransform, CumulativeDistributionTransform, ExpTransform, IndependentTransform,
    Normal, PowerTransform, SoftmaxTransform, StickBreakingTransform, Transform,
};

fn approx(a: &[f32], b: &[f32], tol: f32, ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: length mismatch {a:?} vs {b:?}");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < tol,
            "{ctx}: idx {i} got {x} expected {y} (|d|={})",
            (x - y).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// StickBreakingTransform — R^{K-1} -> simplex. The classic off-by-one in the
// stick-breaking Jacobian lives here.
// torch: StickBreakingTransform(); x=[0.5,-0.3]
//   forward y = [0.45186275, 0.23326391, 0.31487334]  (sum 1.0)
//   inverse  = [0.5, -0.3]
//   ldj      = -3.405546188354492
// ---------------------------------------------------------------------------
#[test]
fn divergence_stick_breaking_forward_inverse_ldj() {
    let x = from_slice(&[0.5f32, -0.3], &[2]).unwrap();
    let sb = StickBreakingTransform;
    let y = sb.forward(&x).unwrap();
    approx(
        y.data().unwrap(),
        &[0.45186275f32, 0.23326391, 0.31487334],
        1e-5,
        "SB forward",
    );
    // sum-to-one property.
    let s: f32 = y.data().unwrap().iter().sum();
    assert!((s - 1.0).abs() < 1e-5, "SB simplex sum = {s}");

    let xi = sb.inverse(&y).unwrap();
    approx(xi.data().unwrap(), &[0.5f32, -0.3], 1e-4, "SB inverse");

    // The subtle one: torch log_abs_det_jacobian.
    let ldj = sb.log_abs_det_jacobian(&x, &y).unwrap();
    approx(ldj.data().unwrap(), &[-3.4055462f32], 1e-4, "SB ldj");
}

#[test]
fn divergence_stick_breaking_batched_ldj() {
    // torch batched x=[[0.5,-0.3],[1.0,0.2]]:
    //   ldj = [-3.405546188, -3.664317607]
    let x = from_slice(&[0.5f32, -0.3, 1.0, 0.2], &[2, 2]).unwrap();
    let sb = StickBreakingTransform;
    let y = sb.forward(&x).unwrap();
    approx(
        y.data().unwrap(),
        &[
            0.45186275f32,
            0.23326391,
            0.31487334,
            0.5761169,
            0.23306534,
            0.1908178,
        ],
        1e-5,
        "SB batched forward",
    );
    let ldj = sb.log_abs_det_jacobian(&x, &y).unwrap();
    approx(
        ldj.data().unwrap(),
        &[-3.4055462f32, -3.6643176],
        1e-4,
        "SB batched ldj",
    );
}

// ---------------------------------------------------------------------------
// CorrCholeskyTransform — tanh-based signed stick-breaking + log(1-tanh^2).
// torch: x=[0.2,-0.5,0.7] (D=3):
//   forward y = [[1,0,0],
//                [0.19737533, 0.98032802, 0],
//                [-0.46211717, 0.53596479, 0.70653343]]
//   inverse  = [0.2, -0.5, 0.7]
//   ldj      = -0.8546200394630432
// ---------------------------------------------------------------------------
#[test]
fn divergence_corr_cholesky_d3_forward_inverse_ldj() {
    let x = from_slice(&[0.2f32, -0.5, 0.7], &[3]).unwrap();
    let cc = CorrCholeskyTransform;
    let y = cc.forward(&x).unwrap();
    assert_eq!(y.shape(), &[3, 3], "CC forward shape");
    approx(
        y.data().unwrap(),
        &[
            1.0f32, 0.0, 0.0, 0.19737533, 0.98032802, 0.0, -0.46211717, 0.53596479, 0.70653343,
        ],
        1e-5,
        "CC forward",
    );
    let xi = cc.inverse(&y).unwrap();
    approx(xi.data().unwrap(), &[0.2f32, -0.5, 0.7], 1e-4, "CC inverse");

    let ldj = cc.log_abs_det_jacobian(&x, &y).unwrap();
    approx(ldj.data().unwrap(), &[-0.85462004f32], 1e-4, "CC ldj");
}

#[test]
fn divergence_corr_cholesky_d4_ldj() {
    // torch x=[0.2,-0.5,0.7,0.1,-0.3,0.6] (D=4):
    //   ldj = -1.3478797674179077
    let x = from_slice(&[0.2f32, -0.5, 0.7, 0.1, -0.3, 0.6], &[6]).unwrap();
    let cc = CorrCholeskyTransform;
    let y = cc.forward(&x).unwrap();
    assert_eq!(y.shape(), &[4, 4], "CC4 forward shape");
    let ldj = cc.log_abs_det_jacobian(&x, &y).unwrap();
    approx(ldj.data().unwrap(), &[-1.3478798f32], 1e-4, "CC4 ldj");
}

// ---------------------------------------------------------------------------
// PowerTransform — y = x^exp, ldj = (exp*y/x).abs().log().
// torch: PowerTransform(2.0); x=[1.5,2.0,0.5]:
//   forward = [2.25, 4.0, 0.25]
//   ldj     = [1.0986123, 1.3862944, 0.0]
// ---------------------------------------------------------------------------
#[test]
fn divergence_power_transform_forward_ldj() {
    let x = from_slice(&[1.5f32, 2.0, 0.5], &[3]).unwrap();
    let pt = PowerTransform::new(2.0f32);
    let y = pt.forward(&x).unwrap();
    approx(y.data().unwrap(), &[2.25f32, 4.0, 0.25], 1e-5, "Power fwd");
    let ldj = pt.log_abs_det_jacobian(&x, &y).unwrap();
    approx(
        ldj.data().unwrap(),
        &[1.0986123f32, 1.3862944, 0.0],
        1e-5,
        "Power ldj",
    );
}

// ---------------------------------------------------------------------------
// SoftmaxTransform forward — torch [1,2,3] -> [0.09003057,0.24472848,0.66524094]
// ---------------------------------------------------------------------------
#[test]
fn divergence_softmax_forward() {
    let x = from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
    let sm = SoftmaxTransform;
    let y = sm.forward(&x).unwrap();
    approx(
        y.data().unwrap(),
        &[0.09003057f32, 0.24472848, 0.66524094],
        1e-6,
        "Softmax forward",
    );
}

// ---------------------------------------------------------------------------
// IndependentTransform(Exp, 1) ldj sums over the rightmost dim.
// torch xi=[[0.1,0.2,0.3],[0.4,0.5,0.6]] -> ldj = [0.6, 1.5]  (Exp ldj = x).
// ---------------------------------------------------------------------------
#[test]
fn divergence_independent_exp_ldj_sums_rightmost() {
    let x = from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]).unwrap();
    let it: IndependentTransform<f32> = IndependentTransform::new(Box::new(ExpTransform), 1);
    let y = it.forward(&x).unwrap();
    assert_eq!(y.shape(), &[2, 3], "Indep forward shape");
    let ldj = it.log_abs_det_jacobian(&x, &y).unwrap();
    assert_eq!(ldj.shape(), &[2], "Indep ldj shape after sum_rightmost");
    approx(ldj.data().unwrap(), &[0.6f32, 1.5], 1e-5, "Indep ldj");
}

// ---------------------------------------------------------------------------
// CumulativeDistributionTransform(Normal(0,1)):
//   forward = cdf  = [0.15865526, 0.5, 0.69146246, 0.93319285]
//   inverse = icdf = [-1.0, 0.0, 0.5, 1.5]
//   ldj     = log_prob = [-1.4189385, -0.9189385, -1.0439385, -2.0439386]
// ---------------------------------------------------------------------------
#[test]
fn divergence_cdf_transform_normal() {
    let loc = from_slice(&[0.0f32], &[1]).unwrap();
    let scale = from_slice(&[1.0f32], &[1]).unwrap();
    let n = Normal::new(loc, scale).unwrap();
    let cdf: CumulativeDistributionTransform<f32> =
        CumulativeDistributionTransform::new(Box::new(n));

    let x = from_slice(&[-1.0f32, 0.0, 0.5, 1.5], &[4]).unwrap();
    let y = cdf.forward(&x).unwrap();
    approx(
        y.data().unwrap(),
        &[0.15865526f32, 0.5, 0.69146246, 0.93319285],
        1e-5,
        "CDF forward",
    );
    let xi = cdf.inverse(&y).unwrap();
    approx(xi.data().unwrap(), &[-1.0f32, 0.0, 0.5, 1.5], 1e-4, "CDF inv");
    let ldj = cdf.log_abs_det_jacobian(&x, &y).unwrap();
    approx(
        ldj.data().unwrap(),
        &[-1.4189385f32, -0.9189385, -1.0439385, -2.0439386],
        1e-4,
        "CDF ldj (= base log_prob)",
    );
}
