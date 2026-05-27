//! Adversarial finite-difference audit of the UNCOMMITTED fused-affine +
//! structural-autograd VJPs in `ferrotorch-core/src/grad_fns/linalg.rs`
//! (#1344 / #1345): addmm, addmv, addr, baddbmm, addbmm, kron, diagonal,
//! diag, tril, triu.
//!
//! The builder that wrote these +1264 LOC died (transient 529) before writing
//! ANY verification — the doc comments even reference a
//! `tests/divergence_linalg_fused_audit.rs` that does not exist. A matrix VJP
//! with a sign error, a transpose error, or a missing beta/alpha factor
//! compiles, passes a shape check, and passes a *uniform-weight* `sum()`
//! finite-difference check (because grad_output = all-ones hides transposes on
//! square / symmetric inputs) — the #1555 plausible-but-wrong-gradient class.
//!
//! To make the FD check adversarial, every test here:
//!   * uses NON-SQUARE shapes (m != k != n) so a transpose changes the shape
//!     of the answer, not just its values;
//!   * uses a NON-UNIFORM, NON-SYMMETRIC weight `W` as the upstream gradient
//!     (`out.backward_with_gradient(W)`), so grad_output is not all-ones and a
//!     swapped factor produces a numerically different gradient;
//!   * uses beta != 1 and alpha != 1 so a dropped scalar factor diverges;
//!   * reconstructs the numeric reference from the op's OWN forward (the
//!     `*_differentiable` fn evaluated under no-grad at perturbed inputs) — NOT
//!     a constant copied from the ferrotorch backward (R-CHAR-3: no tautology).
//!
//! `numeric[i] = (f(x + eps e_i) - f(x - eps e_i)) / (2 eps)` where
//! `f(x) = sum_j W[j] * forward(x)[j]`. Analytic VJP = grad of the same `f`.
//! Match to ~1e-3 (central FD on f64, eps=1e-6) ⇒ VERIFIED; mismatch ⇒
//! DIVERGENCE (wrong gradient; pin + report for the fixer).

use ferrotorch_core::Tensor;
use ferrotorch_core::grad_fns::linalg as glin;
use ferrotorch_core::storage::TensorStorage;

fn leaf(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

fn nog(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

/// Weighted scalar reduction of `out`: `sum_j W[j] * out[j]`.
fn weighted_loss(out: &Tensor<f64>, w: &[f64]) -> f64 {
    let d = out.data().unwrap();
    assert_eq!(d.len(), w.len(), "weight length mismatch");
    d.iter().zip(w).map(|(&o, &wj)| o * wj).sum()
}

/// Central finite-difference gradient of `f(x)` w.r.t. each element of `x`.
fn fd_grad<F>(x: &[f64], eps: f64, f: F) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let mut g = vec![0.0; x.len()];
    for i in 0..x.len() {
        let mut xp = x.to_vec();
        xp[i] += eps;
        let mut xm = x.to_vec();
        xm[i] -= eps;
        g[i] = (f(&xp) - f(&xm)) / (2.0 * eps);
    }
    g
}

#[track_caller]
fn assert_close(analytic: &[f64], numeric: &[f64], tol: f64, label: &str) {
    assert_eq!(
        analytic.len(),
        numeric.len(),
        "{label}: grad length {} vs {}",
        analytic.len(),
        numeric.len()
    );
    let mut worst = 0.0_f64;
    let mut worst_i = 0usize;
    for (i, (&a, &n)) in analytic.iter().zip(numeric.iter()).enumerate() {
        let d = (a - n).abs();
        if d > worst {
            worst = d;
            worst_i = i;
        }
    }
    assert!(
        worst < tol,
        "{label}: DIVERGENCE at grad[{worst_i}]: analytic={}, numeric={}, diff={worst} (tol={tol})\n  analytic={analytic:?}\n  numeric ={numeric:?}",
        analytic[worst_i],
        numeric[worst_i]
    );
}

// ===========================================================================
// addmm(bias, m1, m2, beta, alpha) = beta*bias + alpha*(m1 @ m2)
// Non-square: m1 (2x3), m2 (3x4) -> out (2x4). beta=0.5, alpha=2.5.
// VJP: d_bias=beta*g; d_m1=alpha*(g @ m2^T); d_m2=alpha*(m1^T @ g).
// derivatives.yaml:256-259.
// ===========================================================================
#[test]
fn addmm_fd() {
    let m1d = vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6]; // 2x3
    let m2d = vec![
        1.0, -1.1, 1.2, -1.3, 0.7, 0.8, -0.9, 1.0, -0.4, 0.5, -0.6, 0.3,
    ]; // 3x4
    let bd = vec![0.2, -0.3, 0.4, -0.1, 0.5, -0.6, 0.7, -0.8]; // 2x4
    let beta = 0.5;
    let alpha = 2.5;
    let w = vec![1.0, -2.0, 0.5, 3.0, -1.5, 2.0, -0.5, 1.0]; // 2x4 non-symmetric weight

    let bias = leaf(&bd, &[2, 4]);
    let m1 = leaf(&m1d, &[2, 3]);
    let m2 = leaf(&m2d, &[3, 4]);
    let out = glin::addmm_differentiable(&bias, &m1, &m2, beta, alpha).unwrap();
    let wt = nog(&w, &[2, 4]);
    out.backward_with_gradient(&wt).unwrap();
    let a_bias = bias.grad().unwrap().unwrap().data().unwrap().to_vec();
    let a_m1 = m1.grad().unwrap().unwrap().data().unwrap().to_vec();
    let a_m2 = m2.grad().unwrap().unwrap().data().unwrap().to_vec();

    let f_bias = |x: &[f64]| {
        let o = glin::addmm_differentiable(
            &nog(x, &[2, 4]),
            &nog(&m1d, &[2, 3]),
            &nog(&m2d, &[3, 4]),
            beta,
            alpha,
        )
        .unwrap();
        weighted_loss(&o, &w)
    };
    let f_m1 = |x: &[f64]| {
        let o = glin::addmm_differentiable(
            &nog(&bd, &[2, 4]),
            &nog(x, &[2, 3]),
            &nog(&m2d, &[3, 4]),
            beta,
            alpha,
        )
        .unwrap();
        weighted_loss(&o, &w)
    };
    let f_m2 = |x: &[f64]| {
        let o = glin::addmm_differentiable(
            &nog(&bd, &[2, 4]),
            &nog(&m1d, &[2, 3]),
            &nog(x, &[3, 4]),
            beta,
            alpha,
        )
        .unwrap();
        weighted_loss(&o, &w)
    };
    assert_close(&a_bias, &fd_grad(&bd, 1e-6, f_bias), 1e-3, "addmm d_bias");
    assert_close(&a_m1, &fd_grad(&m1d, 1e-6, f_m1), 1e-3, "addmm d_mat1");
    assert_close(&a_m2, &fd_grad(&m2d, 1e-6, f_m2), 1e-3, "addmm d_mat2");
}

// ===========================================================================
// addmv(bias, mat, vec, beta, alpha) = beta*bias + alpha*(mat @ vec)
// mat (3x2), vec (2) -> out (3). beta=0.5, alpha=1.5.
// VJP: d_bias=beta*g; d_mat=alpha*outer(g,vec); d_vec=alpha*(mat^T @ g).
// derivatives.yaml:267-270.
// ===========================================================================
#[test]
fn addmv_fd() {
    let matd = vec![0.5, -0.6, 0.7, -0.8, 0.9, -1.0]; // 3x2
    let vecd = vec![1.2, -0.4]; // 2
    let bd = vec![0.3, -0.2, 0.1]; // 3
    let beta = 0.5;
    let alpha = 1.5;
    let w = vec![1.0, -2.0, 0.5]; // 3

    let bias = leaf(&bd, &[3]);
    let mat = leaf(&matd, &[3, 2]);
    let vecn = leaf(&vecd, &[2]);
    let out = glin::addmv_differentiable(&bias, &mat, &vecn, beta, alpha).unwrap();
    out.backward_with_gradient(&nog(&w, &[3])).unwrap();
    let a_bias = bias.grad().unwrap().unwrap().data().unwrap().to_vec();
    let a_mat = mat.grad().unwrap().unwrap().data().unwrap().to_vec();
    let a_vec = vecn.grad().unwrap().unwrap().data().unwrap().to_vec();

    let f_bias = |x: &[f64]| {
        weighted_loss(
            &glin::addmv_differentiable(
                &nog(x, &[3]),
                &nog(&matd, &[3, 2]),
                &nog(&vecd, &[2]),
                beta,
                alpha,
            )
            .unwrap(),
            &w,
        )
    };
    let f_mat = |x: &[f64]| {
        weighted_loss(
            &glin::addmv_differentiable(
                &nog(&bd, &[3]),
                &nog(x, &[3, 2]),
                &nog(&vecd, &[2]),
                beta,
                alpha,
            )
            .unwrap(),
            &w,
        )
    };
    let f_vec = |x: &[f64]| {
        weighted_loss(
            &glin::addmv_differentiable(
                &nog(&bd, &[3]),
                &nog(&matd, &[3, 2]),
                &nog(x, &[2]),
                beta,
                alpha,
            )
            .unwrap(),
            &w,
        )
    };
    assert_close(&a_bias, &fd_grad(&bd, 1e-6, f_bias), 1e-3, "addmv d_bias");
    assert_close(&a_mat, &fd_grad(&matd, 1e-6, f_mat), 1e-3, "addmv d_mat");
    assert_close(&a_vec, &fd_grad(&vecd, 1e-6, f_vec), 1e-3, "addmv d_vec");
}

// ===========================================================================
// addr(bias, v1, v2, beta, alpha) = beta*bias + alpha*outer(v1, v2)
// v1 (3), v2 (4) -> out (3x4). beta=0.5, alpha=2.0.
// VJP: d_bias=beta*g; d_v1=alpha*(g @ v2); d_v2=alpha*(g^T @ v1).
// derivatives.yaml:273-276.
// ===========================================================================
#[test]
fn addr_fd() {
    let v1d = vec![0.4, -0.5, 0.6]; // 3
    let v2d = vec![1.1, -0.2, 0.3, -0.7]; // 4
    let bd = vec![
        0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 0.2, -0.3,
    ]; // 3x4
    let beta = 0.5;
    let alpha = 2.0;
    let w = vec![
        1.0, -2.0, 0.5, 3.0, -1.5, 2.0, -0.5, 1.0, 0.3, -0.7, 1.2, -0.9,
    ]; // 3x4

    let bias = leaf(&bd, &[3, 4]);
    let v1 = leaf(&v1d, &[3]);
    let v2 = leaf(&v2d, &[4]);
    let out = glin::addr_differentiable(&bias, &v1, &v2, beta, alpha).unwrap();
    out.backward_with_gradient(&nog(&w, &[3, 4])).unwrap();
    let a_bias = bias.grad().unwrap().unwrap().data().unwrap().to_vec();
    let a_v1 = v1.grad().unwrap().unwrap().data().unwrap().to_vec();
    let a_v2 = v2.grad().unwrap().unwrap().data().unwrap().to_vec();

    let f_bias = |x: &[f64]| {
        weighted_loss(
            &glin::addr_differentiable(
                &nog(x, &[3, 4]),
                &nog(&v1d, &[3]),
                &nog(&v2d, &[4]),
                beta,
                alpha,
            )
            .unwrap(),
            &w,
        )
    };
    let f_v1 = |x: &[f64]| {
        weighted_loss(
            &glin::addr_differentiable(
                &nog(&bd, &[3, 4]),
                &nog(x, &[3]),
                &nog(&v2d, &[4]),
                beta,
                alpha,
            )
            .unwrap(),
            &w,
        )
    };
    let f_v2 = |x: &[f64]| {
        weighted_loss(
            &glin::addr_differentiable(
                &nog(&bd, &[3, 4]),
                &nog(&v1d, &[3]),
                &nog(x, &[4]),
                beta,
                alpha,
            )
            .unwrap(),
            &w,
        )
    };
    assert_close(&a_bias, &fd_grad(&bd, 1e-6, f_bias), 1e-3, "addr d_bias");
    assert_close(&a_v1, &fd_grad(&v1d, 1e-6, f_v1), 1e-3, "addr d_vec1");
    assert_close(&a_v2, &fd_grad(&v2d, 1e-6, f_v2), 1e-3, "addr d_vec2");
}

// ===========================================================================
// baddbmm(bias, b1, b2, beta, alpha) = beta*bias + alpha*bmm(b1, b2)  (3D)
// b1 (2,2,3), b2 (2,3,4) -> out (2,2,4). beta=0.5, alpha=1.5.
// VJP per batch: d_b1=alpha*(g @ b2^T); d_b2=alpha*(b1^T @ g).
// derivatives.yaml:359-362.
// ===========================================================================
#[test]
fn baddbmm_fd() {
    let b1d: Vec<f64> = (0..12).map(|i| (i as f64) * 0.1 - 0.5).collect(); // 2x2x3
    let b2d: Vec<f64> = (0..24).map(|i| ((i as f64) * 0.07 - 0.3).sin()).collect(); // 2x3x4
    let bd: Vec<f64> = (0..16).map(|i| (i as f64) * 0.05 - 0.2).collect(); // 2x2x4
    let beta = 0.5;
    let alpha = 1.5;
    let w: Vec<f64> = (0..16).map(|i| ((i as f64) * 0.31).cos() * 1.7).collect(); // 2x2x4

    let bias = leaf(&bd, &[2, 2, 4]);
    let b1 = leaf(&b1d, &[2, 2, 3]);
    let b2 = leaf(&b2d, &[2, 3, 4]);
    let out = glin::baddbmm_differentiable(&bias, &b1, &b2, beta, alpha).unwrap();
    out.backward_with_gradient(&nog(&w, &[2, 2, 4])).unwrap();
    let a_bias = bias.grad().unwrap().unwrap().data().unwrap().to_vec();
    let a_b1 = b1.grad().unwrap().unwrap().data().unwrap().to_vec();
    let a_b2 = b2.grad().unwrap().unwrap().data().unwrap().to_vec();

    let f_bias = |x: &[f64]| {
        weighted_loss(
            &glin::baddbmm_differentiable(
                &nog(x, &[2, 2, 4]),
                &nog(&b1d, &[2, 2, 3]),
                &nog(&b2d, &[2, 3, 4]),
                beta,
                alpha,
            )
            .unwrap(),
            &w,
        )
    };
    let f_b1 = |x: &[f64]| {
        weighted_loss(
            &glin::baddbmm_differentiable(
                &nog(&bd, &[2, 2, 4]),
                &nog(x, &[2, 2, 3]),
                &nog(&b2d, &[2, 3, 4]),
                beta,
                alpha,
            )
            .unwrap(),
            &w,
        )
    };
    let f_b2 = |x: &[f64]| {
        weighted_loss(
            &glin::baddbmm_differentiable(
                &nog(&bd, &[2, 2, 4]),
                &nog(&b1d, &[2, 2, 3]),
                &nog(x, &[2, 3, 4]),
                beta,
                alpha,
            )
            .unwrap(),
            &w,
        )
    };
    assert_close(&a_bias, &fd_grad(&bd, 1e-6, f_bias), 1e-3, "baddbmm d_bias");
    assert_close(&a_b1, &fd_grad(&b1d, 1e-6, f_b1), 1e-3, "baddbmm d_batch1");
    assert_close(&a_b2, &fd_grad(&b2d, 1e-6, f_b2), 1e-3, "baddbmm d_batch2");
}

// ===========================================================================
// addbmm(bias, b1, b2, beta, alpha) = beta*bias + alpha*sum_b(b1[b] @ b2[b])
// b1 (3,2,3), b2 (3,3,4) -> out (2,4)  (batch reduced). beta=0.5, alpha=1.5.
// VJP: grad (2,4) broadcast over batch:
//   d_b1[b]=alpha*(g @ b2[b]^T); d_b2[b]=alpha*(b1[b]^T @ g).
// derivatives.yaml:238-241.
// ===========================================================================
#[test]
fn addbmm_fd() {
    let b1d: Vec<f64> = (0..18).map(|i| (i as f64) * 0.11 - 0.7).collect(); // 3x2x3
    let b2d: Vec<f64> = (0..36).map(|i| ((i as f64) * 0.05 - 0.4).cos()).collect(); // 3x3x4
    let bd: Vec<f64> = (0..8).map(|i| (i as f64) * 0.07 - 0.25).collect(); // 2x4
    let beta = 0.5;
    let alpha = 1.5;
    let w: Vec<f64> = (0..8)
        .map(|i| ((i as f64) * 0.41).sin() * 1.3 + 0.2)
        .collect(); // 2x4

    let bias = leaf(&bd, &[2, 4]);
    let b1 = leaf(&b1d, &[3, 2, 3]);
    let b2 = leaf(&b2d, &[3, 3, 4]);
    let out = glin::addbmm_differentiable(&bias, &b1, &b2, beta, alpha).unwrap();
    out.backward_with_gradient(&nog(&w, &[2, 4])).unwrap();
    let a_bias = bias.grad().unwrap().unwrap().data().unwrap().to_vec();
    let a_b1 = b1.grad().unwrap().unwrap().data().unwrap().to_vec();
    let a_b2 = b2.grad().unwrap().unwrap().data().unwrap().to_vec();

    let f_bias = |x: &[f64]| {
        weighted_loss(
            &glin::addbmm_differentiable(
                &nog(x, &[2, 4]),
                &nog(&b1d, &[3, 2, 3]),
                &nog(&b2d, &[3, 3, 4]),
                beta,
                alpha,
            )
            .unwrap(),
            &w,
        )
    };
    let f_b1 = |x: &[f64]| {
        weighted_loss(
            &glin::addbmm_differentiable(
                &nog(&bd, &[2, 4]),
                &nog(x, &[3, 2, 3]),
                &nog(&b2d, &[3, 3, 4]),
                beta,
                alpha,
            )
            .unwrap(),
            &w,
        )
    };
    let f_b2 = |x: &[f64]| {
        weighted_loss(
            &glin::addbmm_differentiable(
                &nog(&bd, &[2, 4]),
                &nog(&b1d, &[3, 2, 3]),
                &nog(x, &[3, 3, 4]),
                beta,
                alpha,
            )
            .unwrap(),
            &w,
        )
    };
    assert_close(&a_bias, &fd_grad(&bd, 1e-6, f_bias), 1e-3, "addbmm d_bias");
    assert_close(&a_b1, &fd_grad(&b1d, 1e-6, f_b1), 1e-3, "addbmm d_batch1");
    assert_close(&a_b2, &fd_grad(&b2d, 1e-6, f_b2), 1e-3, "addbmm d_batch2");
}

// ===========================================================================
// kron(A, B): A (2x3), B (2x2) -> K (4x6).  K[i*r+u, j*s+v] = A[i,j]*B[u,v].
// VJP: dA[i,j] = sum_{u,v} g[i*r+u, j*s+v]*B[u,v]; dB symmetric.
// LinearAlgebra.cpp:3530 (KronImpl).
// ===========================================================================
#[test]
fn kron_fd() {
    let ad = vec![0.3, -0.4, 0.5, -0.6, 0.7, -0.8]; // 2x3
    let bdat = vec![1.2, -0.5, 0.9, -1.1]; // 2x2
    let w: Vec<f64> = (0..24)
        .map(|i| ((i as f64) * 0.23).sin() * 1.4 - 0.3)
        .collect(); // 4x6

    let a = leaf(&ad, &[2, 3]);
    let b = leaf(&bdat, &[2, 2]);
    let out = glin::kron_differentiable(&a, &b).unwrap();
    out.backward_with_gradient(&nog(&w, &[4, 6])).unwrap();
    let a_a = a.grad().unwrap().unwrap().data().unwrap().to_vec();
    let a_b = b.grad().unwrap().unwrap().data().unwrap().to_vec();

    let f_a = |x: &[f64]| {
        weighted_loss(
            &glin::kron_differentiable(&nog(x, &[2, 3]), &nog(&bdat, &[2, 2])).unwrap(),
            &w,
        )
    };
    let f_b = |x: &[f64]| {
        weighted_loss(
            &glin::kron_differentiable(&nog(&ad, &[2, 3]), &nog(x, &[2, 2])).unwrap(),
            &w,
        )
    };
    assert_close(&a_a, &fd_grad(&ad, 1e-6, f_a), 1e-3, "kron dA");
    assert_close(&a_b, &fd_grad(&bdat, 1e-6, f_b), 1e-3, "kron dB");
}

// ===========================================================================
// diagonal(A, offset): A (3x4) -> 1D diag. Test offset 0, +1, -1.
// VJP: scatter grad onto the offset-th diagonal. derivatives.yaml:573.
// ===========================================================================
fn diagonal_case(offset: i64) {
    let ad: Vec<f64> = (0..12).map(|i| (i as f64) * 0.13 - 0.5).collect(); // 3x4
    // length of the offset-diagonal of a 3x4 matrix
    let dlen = glin::diagonal_differentiable(&nog(&ad, &[3, 4]), offset)
        .unwrap()
        .data()
        .unwrap()
        .len();
    let w: Vec<f64> = (0..dlen)
        .map(|i| ((i as f64) * 0.7).cos() * 1.5 + 0.4)
        .collect();

    let a = leaf(&ad, &[3, 4]);
    let out = glin::diagonal_differentiable(&a, offset).unwrap();
    out.backward_with_gradient(&nog(&w, &[dlen])).unwrap();
    let a_a = a.grad().unwrap().unwrap().data().unwrap().to_vec();

    let f = |x: &[f64]| {
        weighted_loss(
            &glin::diagonal_differentiable(&nog(x, &[3, 4]), offset).unwrap(),
            &w,
        )
    };
    assert_close(
        &a_a,
        &fd_grad(&ad, 1e-6, f),
        1e-3,
        &format!("diagonal offset={offset}"),
    );
}

#[test]
fn diagonal_fd_offset0() {
    diagonal_case(0);
}
#[test]
fn diagonal_fd_offset_pos() {
    diagonal_case(1);
}
#[test]
fn diagonal_fd_offset_neg() {
    diagonal_case(-1);
}

// ===========================================================================
// diag(A, diagonal): extract (2D->1D) and construct (1D->2D). offsets tested.
// VJP is the adjoint selection. (composite gradient)
// ===========================================================================
#[test]
fn diag_extract_fd() {
    for offset in [0_i64, 1, -1] {
        let ad: Vec<f64> = (0..12).map(|i| (i as f64) * 0.17 - 0.4).collect(); // 3x4
        let dlen = glin::diag_differentiable(&nog(&ad, &[3, 4]), offset)
            .unwrap()
            .data()
            .unwrap()
            .len();
        let w: Vec<f64> = (0..dlen)
            .map(|i| ((i as f64) * 0.9).sin() * 1.2 - 0.3)
            .collect();

        let a = leaf(&ad, &[3, 4]);
        let out = glin::diag_differentiable(&a, offset).unwrap();
        out.backward_with_gradient(&nog(&w, &[dlen])).unwrap();
        let a_a = a.grad().unwrap().unwrap().data().unwrap().to_vec();

        let f = |x: &[f64]| {
            weighted_loss(
                &glin::diag_differentiable(&nog(x, &[3, 4]), offset).unwrap(),
                &w,
            )
        };
        assert_close(
            &a_a,
            &fd_grad(&ad, 1e-6, f),
            1e-3,
            &format!("diag extract offset={offset}"),
        );
    }
}

#[test]
fn diag_construct_fd() {
    for offset in [0_i64, 1, -1] {
        let ad = vec![0.5, -0.6, 0.7, -0.8]; // 1D len 4
        let out0 = glin::diag_differentiable(&nog(&ad, &[4]), offset).unwrap();
        let osh = out0.shape().to_vec();
        let olen: usize = osh.iter().product();
        let w: Vec<f64> = (0..olen)
            .map(|i| ((i as f64) * 0.37).cos() * 1.3 + 0.5)
            .collect();

        let a = leaf(&ad, &[4]);
        let out = glin::diag_differentiable(&a, offset).unwrap();
        out.backward_with_gradient(&nog(&w, &osh)).unwrap();
        let a_a = a.grad().unwrap().unwrap().data().unwrap().to_vec();

        let f = |x: &[f64]| {
            weighted_loss(
                &glin::diag_differentiable(&nog(x, &[4]), offset).unwrap(),
                &w,
            )
        };
        assert_close(
            &a_a,
            &fd_grad(&ad, 1e-6, f),
            1e-3,
            &format!("diag construct offset={offset}"),
        );
    }
}

// ===========================================================================
// tril / triu(A, diagonal): A (3x4). Backward applies the same mask to grad.
// derivatives.yaml:1805-1811.
// ===========================================================================
fn tri_case(lower: bool, diagonal: i64) {
    let ad: Vec<f64> = (0..12).map(|i| (i as f64) * 0.19 - 0.6).collect(); // 3x4
    let w: Vec<f64> = (0..12)
        .map(|i| ((i as f64) * 0.53).sin() * 1.6 - 0.2)
        .collect();

    let a = leaf(&ad, &[3, 4]);
    let out = if lower {
        glin::tril_differentiable(&a, diagonal).unwrap()
    } else {
        glin::triu_differentiable(&a, diagonal).unwrap()
    };
    out.backward_with_gradient(&nog(&w, &[3, 4])).unwrap();
    let a_a = a.grad().unwrap().unwrap().data().unwrap().to_vec();

    let f = |x: &[f64]| {
        let o = if lower {
            glin::tril_differentiable(&nog(x, &[3, 4]), diagonal).unwrap()
        } else {
            glin::triu_differentiable(&nog(x, &[3, 4]), diagonal).unwrap()
        };
        weighted_loss(&o, &w)
    };
    let name = if lower { "tril" } else { "triu" };
    assert_close(
        &a_a,
        &fd_grad(&ad, 1e-6, f),
        1e-3,
        &format!("{name} diagonal={diagonal}"),
    );
}

#[test]
fn tril_fd() {
    tri_case(true, 0);
    tri_case(true, 1);
    tri_case(true, -1);
}
#[test]
fn triu_fd() {
    tri_case(false, 0);
    tri_case(false, 1);
    tri_case(false, -1);
}
