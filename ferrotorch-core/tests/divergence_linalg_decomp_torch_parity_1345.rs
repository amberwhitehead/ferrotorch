//! Live-torch parity pins for the qr/cholesky/slogdet decomposition backwards
//! shipped in `9332cb285` (#1345).
//!
//! The builder verified each VJP by FINITE DIFFERENCE against the op's OWN
//! forward. FD checks internal self-consistency (analytic VJP vs numeric
//! gradient of ferrotorch's own forward) but NOT torch-convention parity: a
//! VJP that is internally consistent with a forward whose Q/R sign convention,
//! cholesky lower/upper choice, or gradient symmetrisation differs from torch
//! would pass FD yet silently corrupt training against torch-trained weights.
//!
//! These tests close that gap. Each expected `A.grad` is constructed by a LIVE
//! `torch 2.11.0 float64` call (R-CHAR-3 (a)):
//!
//!     A = torch.tensor(..., dtype=torch.float64).requires_grad_(True)
//!     out = torch.linalg.<op>(A)
//!     <scalar-loss>.backward()
//!     A.grad   # the literal asserted below
//!
//! The torch invocation that produced each constant is named in the per-test
//! doc comment. None of these values is copied from the ferrotorch side.
//!
//! Convention conclusions (verified live, 2026-05-27, torch 2.11.0+cu130):
//!   * ferray's QR forward uses the SAME LAPACK Householder sign convention as
//!     torch (negative R-diagonal where geqrf gives it) for BOTH square and
//!     tall (m>n) inputs — so the backward A.grad matches torch exactly. (The
//!     commit message's "positive-diagonal R" remark is inaccurate but the
//!     observable behavior is torch-correct.)
//!   * cholesky is LOWER (A = L Lᵀ) and the A.grad is symmetric, matching
//!     torch's `0.5*(gA + gA.tril(-1).mH())` symmetrisation.
//!   * slogdet's `sign` output carries no grad_fn (non-diff), and the
//!     logabsdet grad is inv(A)ᵀ regardless of det sign (verified on a
//!     negative-determinant matrix).

use ferrotorch_core::Tensor;
use ferrotorch_core::grad_fns::reduction::sum as reduce_sum;
use ferrotorch_core::linalg as linalg_fwd;
use ferrotorch_core::storage::TensorStorage;

fn leaf(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

fn assert_close(actual: &[f64], torch: &[f64], tol: f64, label: &str) {
    assert_eq!(
        actual.len(),
        torch.len(),
        "{label}: length mismatch {} vs {}",
        actual.len(),
        torch.len()
    );
    for (i, (&a, &t)) in actual.iter().zip(torch.iter()).enumerate() {
        assert!(
            (a - t).abs() < tol,
            "{label} grad[{i}]: ferrotorch={a}, torch={t}, diff={}",
            (a - t).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// QR — square, full (sum(Q)+sum(R) loss)
//
// torch 2.11.0:
//   A = torch.tensor([1.0,2.0,0.5, 0.3,1.5,2.0, 1.0,0.2,3.0],
//                    dtype=torch.float64).reshape(3,3).requires_grad_(True)
//   Q,R = torch.linalg.qr(A, mode='reduced'); (Q.sum()+R.sum()).backward()
// ---------------------------------------------------------------------------
#[test]
fn qr_square_full_grad_matches_torch() {
    let d = [1.0, 2.0, 0.5, 0.3, 1.5, 2.0, 1.0, 0.2, 3.0];
    let a = leaf(&d, &[3, 3]);
    let (q, r) = linalg_fwd::qr(&a).unwrap();
    let loss = reduce_sum(&q)
        .unwrap()
        .add_t(&reduce_sum(&r).unwrap())
        .unwrap();
    loss.backward().unwrap();
    let g = a.grad().unwrap().unwrap();
    let g = g.data().unwrap();

    let torch = [
        2.27825084526534161,
        -0.14293125682669797,
        -1.70130483885535111,
        -0.26915291609833719,
        -2.08218105669482467,
        -0.13421407915900810,
        -3.64318819991593656,
        -0.67809765564495084,
        0.29588583312295774,
    ];
    assert_close(g, &torch, 1e-9, "qr square full vs torch");
}

// ---------------------------------------------------------------------------
// QR — square, Q-only and R-only paths (each path's A.grad in isolation).
//
// torch 2.11.0, same A as above:
//   Q-only: Q.sum().backward()
//   R-only: R.sum().backward()
// Verifies the split QrBackwardQ / QrBackwardR nodes each reproduce torch's
// single-output joint backward when the other grad is undefined (zero).
// ---------------------------------------------------------------------------
#[test]
fn qr_square_q_only_and_r_only_match_torch() {
    let d = [1.0, 2.0, 0.5, 0.3, 1.5, 2.0, 1.0, 0.2, 3.0];

    let aq = leaf(&d, &[3, 3]);
    let (q, _r) = linalg_fwd::qr(&aq).unwrap();
    reduce_sum(&q).unwrap().backward().unwrap();
    let gq = aq.grad().unwrap().unwrap();
    let torch_q = [
        0.763691571549592663,
        0.328755666169269101,
        0.0,
        -0.0229606235543893256,
        -0.410944582711586237,
        0.0,
        -0.756803384483275998,
        -0.205472291355793257,
        0.0,
    ];
    assert_close(
        gq.data().unwrap(),
        &torch_q,
        1e-9,
        "qr square Q-only vs torch",
    );

    let ar = leaf(&d, &[3, 3]);
    let (_q, r) = linalg_fwd::qr(&ar).unwrap();
    reduce_sum(&r).unwrap().backward().unwrap();
    let gr = ar.grad().unwrap().unwrap();
    let torch_r = [
        1.51455927371574894,
        -0.47168692299596687,
        -1.70130483885535111,
        -0.24619229254394773,
        -1.67123647398323838,
        -0.13421407915900802,
        -2.88638481543266101,
        -0.47262536428915752,
        0.29588583312295769,
    ];
    assert_close(
        gr.data().unwrap(),
        &torch_r,
        1e-9,
        "qr square R-only vs torch",
    );
}

// ---------------------------------------------------------------------------
// QR — TALL (m>n) reduced mode, full backward. This is the case the task
// flags as easy to get wrong (the copyltu/M formula). 4x2 input.
//
// torch 2.11.0:
//   A = torch.tensor([1.0,2.0, 0.5,1.5, 2.0,1.0, 0.2,3.0],
//                    dtype=torch.float64).reshape(4,2).requires_grad_(True)
//   Q,R = torch.linalg.qr(A, mode='reduced'); (Q.sum()+R.sum()).backward()
// ---------------------------------------------------------------------------
#[test]
fn qr_tall_full_grad_matches_torch() {
    let d = [1.0, 2.0, 0.5, 1.5, 2.0, 1.0, 0.2, 3.0];
    let a = leaf(&d, &[4, 2]);
    let (q, r) = linalg_fwd::qr(&a).unwrap();
    assert_eq!(q.shape(), &[4, 2], "tall Q must be reduced [m,n]");
    assert_eq!(r.shape(), &[2, 2], "tall R must be [n,n]");
    let loss = reduce_sum(&q)
        .unwrap()
        .add_t(&reduce_sum(&r).unwrap())
        .unwrap();
    loss.backward().unwrap();
    let g = a.grad().unwrap().unwrap();

    let torch = [
        -0.494546132662350,
        -0.722256395892993,
        -0.321901346687065,
        -0.612212548318316,
        -0.796521907771106,
        -0.545482497585453,
        -0.257296892259527,
        -0.903361673884713,
    ];
    assert_close(g.data().unwrap(), &torch, 1e-9, "qr tall full vs torch");
}

// ---------------------------------------------------------------------------
// QR — TALL Q-only and R-only (same 4x2 A).
// ---------------------------------------------------------------------------
#[test]
fn qr_tall_q_only_and_r_only_match_torch() {
    let d = [1.0, 2.0, 0.5, 1.5, 2.0, 1.0, 0.2, 3.0];

    let aq = leaf(&d, &[4, 2]);
    let (q, _r) = linalg_fwd::qr(&aq).unwrap();
    reduce_sum(&q).unwrap().backward().unwrap();
    let torch_q = [
        0.066388883175318,
        0.012819074428266,
        0.022365992377206,
        -0.092805861150090,
        -0.057450862968388,
        0.013454731011486,
        0.186649232864277,
        0.033371970619039,
    ];
    assert_close(
        aq.grad().unwrap().unwrap().data().unwrap(),
        &torch_q,
        1e-9,
        "qr tall Q-only vs torch",
    );

    let ar = leaf(&d, &[4, 2]);
    let (_q, r) = linalg_fwd::qr(&ar).unwrap();
    reduce_sum(&r).unwrap().backward().unwrap();
    let torch_r = [
        -0.560935015837669,
        -0.735075470321258,
        -0.344267339064271,
        -0.519406687168225,
        -0.739071044802718,
        -0.558937228596939,
        -0.443946125123804,
        -0.936733644503753,
    ];
    assert_close(
        ar.grad().unwrap().unwrap().data().unwrap(),
        &torch_r,
        1e-9,
        "qr tall R-only vs torch",
    );
}

// ---------------------------------------------------------------------------
// CHOLESKY — symmetric A.grad matching torch's symmetrisation. The task flags
// a wrong symmetrisation as the prime suspect (passes FD-against-self, fails
// vs torch). LOWER triangular L; A = L Lᵀ.
//
// torch 2.11.0:
//   A = torch.tensor([4.0,1.0,0.5, 1.0,3.0,0.8, 0.5,0.8,2.5],
//                    dtype=torch.float64).reshape(3,3).requires_grad_(True)
//   L = torch.linalg.cholesky(A); L.sum().backward()
// ---------------------------------------------------------------------------
#[test]
fn cholesky_grad_matches_torch_symmetrisation() {
    let d = [4.0, 1.0, 0.5, 1.0, 3.0, 0.8, 0.5, 0.8, 2.5];
    let a = leaf(&d, &[3, 3]);
    let l = linalg_fwd::cholesky(&a).unwrap();
    // torch's L is LOWER triangular: strict-upper entries are exactly zero.
    let ld = l.data().unwrap();
    let n = 3usize;
    for r in 0..n {
        for c in (r + 1)..n {
            assert_eq!(
                ld[r * n + c],
                0.0,
                "cholesky L must be lower-triangular; entry ({r},{c}) nonzero"
            );
        }
    }
    reduce_sum(&l).unwrap().backward().unwrap();
    let g = a.grad().unwrap().unwrap();

    // torch returns the full symmetric matrix (NOT triangular).
    let torch = [
        0.19065682463446851,
        0.16061662780736519,
        0.15351214730952165,
        0.16061662780736519,
        0.24748999124901561,
        0.22008699504304774,
        0.15351214730952165,
        0.22008699504304771,
        0.33172883143773141,
    ];
    assert_close(g.data().unwrap(), &torch, 1e-9, "cholesky grad vs torch");
}

// ---------------------------------------------------------------------------
// SLOGDET — positive-det 3x3. logabsdet grad = inv(A)ᵀ; sign is non-diff.
//
// torch 2.11.0:
//   A = torch.tensor([2.0,1.0,0.0, 0.5,3.0,1.0, 0.0,1.0,2.5],
//                    dtype=torch.float64).reshape(3,3).requires_grad_(True)
//   sign,logabsdet = torch.linalg.slogdet(A); logabsdet.backward()
// ---------------------------------------------------------------------------
#[test]
fn slogdet_pos_det_grad_matches_torch_and_sign_nondiff() {
    let d = [2.0, 1.0, 0.0, 0.5, 3.0, 1.0, 0.0, 1.0, 2.5];
    let a = leaf(&d, &[3, 3]);
    let (sign, logabsdet) = linalg_fwd::slogdet(&a).unwrap();
    // sign is non-differentiable: torch's output_differentiability[0]=False.
    assert!(
        sign.grad_fn().is_none(),
        "slogdet sign must carry no grad_fn"
    );
    assert_eq!(sign.item().unwrap(), 1.0, "sign(det>0) must be +1");
    logabsdet.backward().unwrap();
    let g = a.grad().unwrap().unwrap();

    let torch = [
        0.55319148936170215,
        -0.10638297872340426,
        0.04255319148936170,
        -0.21276595744680851,
        0.42553191489361702,
        -0.17021276595744680,
        0.08510638297872340,
        -0.17021276595744680,
        0.46808510638297873,
    ];
    assert_close(g.data().unwrap(), &torch, 1e-9, "slogdet pos vs torch");
}

// ---------------------------------------------------------------------------
// SLOGDET — NEGATIVE-det 3x3 (sign = -1). The logabsdet grad is still inv(A)ᵀ
// regardless of det sign — the task flags this as a place a buggy impl might
// inject the sign into the gradient.
//
// torch 2.11.0:
//   A = torch.tensor([0.5,3.0,1.0, 2.0,1.0,0.0, 0.0,1.0,2.5],
//                    dtype=torch.float64).reshape(3,3).requires_grad_(True)
//   sign,logabsdet = torch.linalg.slogdet(A)   # sign == -1, det == -11.75
//   logabsdet.backward()
// ---------------------------------------------------------------------------
#[test]
fn slogdet_neg_det_grad_matches_torch() {
    let d = [0.5, 3.0, 1.0, 2.0, 1.0, 0.0, 0.0, 1.0, 2.5];
    let a = leaf(&d, &[3, 3]);
    let (sign, logabsdet) = linalg_fwd::slogdet(&a).unwrap();
    assert!(
        sign.grad_fn().is_none(),
        "slogdet sign must carry no grad_fn"
    );
    assert_eq!(sign.item().unwrap(), -1.0, "sign(det<0) must be -1");
    logabsdet.backward().unwrap();
    let g = a.grad().unwrap().unwrap();

    // torch A.grad (= inv(A)ᵀ, unaffected by the negative sign):
    let torch = [
        -0.21276595744680854,
        0.42553191489361702,
        -0.17021276595744683,
        0.55319148936170226,
        -0.10638297872340426,
        0.04255319148936171,
        0.08510638297872340,
        -0.17021276595744680,
        0.46808510638297873,
    ];
    assert_close(g.data().unwrap(), &torch, 1e-9, "slogdet neg vs torch");
}
