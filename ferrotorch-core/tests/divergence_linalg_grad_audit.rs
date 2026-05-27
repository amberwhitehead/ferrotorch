//! Finite-difference audit for the closed-form linalg VJPs shipped under
//! blocker #1345: `trace`, `outer`, `linalg.det`, `linalg.inv`,
//! `linalg.solve`.
//!
//! Each backward in `ferrotorch-core/src/grad_fns/linalg.rs` is a matrix
//! differential grounded in a named PyTorch `file:line` (cited per-test).
//! A matrix VJP that "looks right" but transposes the wrong factor passes a
//! shape check and a forward check yet silently corrupts training — the
//! #1555 failure class. The only honest guard is a central finite-difference
//! comparison of the analytic gradient against the numeric one, which is what
//! every test here does:
//!
//!     numeric[i] = (loss(x + eps*e_i) - loss(x - eps*e_i)) / (2*eps)
//!
//! and asserts `|analytic[i] - numeric[i]|` is within a tolerance scaled to
//! the conditioning of the op. The reference value is NOT a hard-coded
//! constant copied from a torch run (that would be a tautology if the impl is
//! wrong in the same way); it is reconstructed from the op's own forward
//! evaluated at perturbed inputs, so the test is self-consistent against the
//! analytic backward independent of any cached oracle.

use ferrotorch_core::Tensor;
use ferrotorch_core::grad_fns::linalg as glin;
use ferrotorch_core::grad_fns::reduction::sum as reduce_sum;
use ferrotorch_core::linalg as linalg_fwd;
use ferrotorch_core::storage::TensorStorage;

fn leaf(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

fn no_grad_leaf(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

/// Central finite-difference gradient of `f(x)` (a scalar-valued reduction of
/// the op output) with respect to each element of `x`.
fn fd_grad<F>(x: &[f64], shape: &[usize], eps: f64, f: F) -> Vec<f64>
where
    F: Fn(&Tensor<f64>) -> f64,
{
    let mut g = vec![0.0; x.len()];
    for i in 0..x.len() {
        let mut xp = x.to_vec();
        xp[i] += eps;
        let mut xm = x.to_vec();
        xm[i] -= eps;
        let lp = f(&no_grad_leaf(&xp, shape));
        let lm = f(&no_grad_leaf(&xm, shape));
        g[i] = (lp - lm) / (2.0 * eps);
    }
    g
}

fn assert_grad_close(analytic: &[f64], numeric: &[f64], tol: f64, label: &str) {
    assert_eq!(
        analytic.len(),
        numeric.len(),
        "{label}: grad length mismatch {} vs {}",
        analytic.len(),
        numeric.len()
    );
    for (i, (&a, &n)) in analytic.iter().zip(numeric.iter()).enumerate() {
        assert!(
            (a - n).abs() < tol,
            "{label} grad[{i}]: analytic={a}, numeric={n}, diff={}",
            (a - n).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// trace — VJP: dA = grad * I  (derivatives.yaml:1785 trace_backward_symint)
// ---------------------------------------------------------------------------

#[test]
fn trace_backward_matches_finite_difference() {
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.5];
    let shape = [3, 3];

    let a = leaf(&a_data, &shape);
    let s = glin::trace_differentiable(&a).unwrap();
    // trace is already scalar; backward directly.
    s.backward().unwrap();
    let analytic = a.grad().unwrap().unwrap().data().unwrap().to_vec();

    let numeric = fd_grad(&a_data, &shape, 1e-6, |x| {
        linalg_fwd::trace(x).unwrap().item().unwrap()
    });

    // dA should be exactly the identity matrix (grad_s = 1).
    let expected_identity = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    assert_grad_close(&analytic, &expected_identity, 1e-9, "trace vs I");
    assert_grad_close(&analytic, &numeric, 1e-5, "trace vs FD");
}

// ---------------------------------------------------------------------------
// outer — VJP: da = grad @ b, db = grad^T @ a  (derivatives.yaml:275-276)
// ---------------------------------------------------------------------------

#[test]
fn outer_backward_matches_finite_difference() {
    let a_data = vec![1.0, -2.0, 0.5];
    let b_data = vec![3.0, 4.0, -1.0, 2.0];
    let a_shape = [3];
    let b_shape = [4];

    let a = leaf(&a_data, &a_shape);
    let b = leaf(&b_data, &b_shape);
    let c = glin::outer_differentiable(&a, &b).unwrap();
    assert_eq!(c.shape(), &[3, 4]);
    let loss = reduce_sum(&c).unwrap();
    loss.backward().unwrap();

    let ga = a.grad().unwrap().unwrap().data().unwrap().to_vec();
    let gb = b.grad().unwrap().unwrap().data().unwrap().to_vec();

    // FD wrt a (b fixed).
    let num_a = fd_grad(&a_data, &a_shape, 1e-6, |x| {
        let bb = no_grad_leaf(&b_data, &b_shape);
        let c = linalg_fwd::outer(x, &bb).unwrap();
        reduce_sum(&c).unwrap().item().unwrap()
    });
    // FD wrt b (a fixed).
    let num_b = fd_grad(&b_data, &b_shape, 1e-6, |x| {
        let aa = no_grad_leaf(&a_data, &a_shape);
        let c = linalg_fwd::outer(&aa, x).unwrap();
        reduce_sum(&c).unwrap().item().unwrap()
    });

    assert_grad_close(&ga, &num_a, 1e-5, "outer da vs FD");
    assert_grad_close(&gb, &num_b, 1e-5, "outer db vs FD");
}

// ---------------------------------------------------------------------------
// det — VJP: dA = det * grad * inv(A)^T
//   (FunctionsManual.cpp:4373 linalg_det_backward, invertible branch)
// ---------------------------------------------------------------------------

#[test]
fn det_backward_matches_finite_difference() {
    // Well-conditioned, non-symmetric 3x3 with det far from 0.
    let a_data = vec![2.0, 1.0, 0.0, 0.5, 3.0, 1.0, 0.0, 1.0, 2.5];
    let shape = [3, 3];

    let a = leaf(&a_data, &shape);
    let d = glin::det_differentiable(&a).unwrap();
    assert!(d.is_scalar());
    d.backward().unwrap();
    let analytic = a.grad().unwrap().unwrap().data().unwrap().to_vec();

    let numeric = fd_grad(&a_data, &shape, 1e-6, |x| {
        linalg_fwd::det(x).unwrap().item().unwrap()
    });

    // det conditioning: the gradient magnitude ~ O(det). Use a relative-ish
    // absolute tolerance commensurate with the FD truncation at eps=1e-6.
    assert_grad_close(&analytic, &numeric, 1e-4, "det vs FD");
}

// ---------------------------------------------------------------------------
// inv — VJP: dA = -inv^T @ grad @ inv^T  (derivatives.yaml:917 linalg_inv_ex)
// ---------------------------------------------------------------------------

#[test]
fn inv_backward_matches_finite_difference() {
    let a_data = vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
    let shape = [3, 3];

    let a = leaf(&a_data, &shape);
    let y = glin::inv_differentiable(&a).unwrap();
    assert_eq!(y.shape(), &[3, 3]);
    let loss = reduce_sum(&y).unwrap();
    loss.backward().unwrap();
    let analytic = a.grad().unwrap().unwrap().data().unwrap().to_vec();

    let numeric = fd_grad(&a_data, &shape, 1e-6, |x| {
        let inv = linalg_fwd::inv(x).unwrap();
        reduce_sum(&inv).unwrap().item().unwrap()
    });

    assert_grad_close(&analytic, &numeric, 1e-4, "inv vs FD");
}

// ---------------------------------------------------------------------------
// solve — VJP: gB = A^{-T} @ gX, gA = -gB @ X^T
//   (FunctionsManual.cpp:6160 linalg_solve_backward)
// ---------------------------------------------------------------------------

#[test]
fn solve_backward_matrix_rhs_matches_finite_difference() {
    let a_data = vec![3.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 4.0];
    let b_data = vec![1.0, 2.0, -1.0, 0.5, 2.0, 1.0];
    let a_shape = [3, 3];
    let b_shape = [3, 2];

    let a = leaf(&a_data, &a_shape);
    let b = leaf(&b_data, &b_shape);
    let x = glin::solve_differentiable(&a, &b).unwrap();
    assert_eq!(x.shape(), &[3, 2]);
    let loss = reduce_sum(&x).unwrap();
    loss.backward().unwrap();

    let ga = a.grad().unwrap().unwrap().data().unwrap().to_vec();
    let gb = b.grad().unwrap().unwrap().data().unwrap().to_vec();

    let num_a = fd_grad(&a_data, &a_shape, 1e-6, |xa| {
        let bb = no_grad_leaf(&b_data, &b_shape);
        let xx = linalg_fwd::solve(xa, &bb).unwrap();
        reduce_sum(&xx).unwrap().item().unwrap()
    });
    let num_b = fd_grad(&b_data, &b_shape, 1e-6, |xb| {
        let aa = no_grad_leaf(&a_data, &a_shape);
        let xx = linalg_fwd::solve(&aa, xb).unwrap();
        reduce_sum(&xx).unwrap().item().unwrap()
    });

    assert_grad_close(&ga, &num_a, 1e-4, "solve gA (matrix rhs) vs FD");
    assert_grad_close(&gb, &num_b, 1e-4, "solve gB (matrix rhs) vs FD");
}

#[test]
fn solve_backward_vector_rhs_matches_finite_difference() {
    let a_data = vec![3.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 4.0];
    let b_data = vec![1.0, -2.0, 0.5];
    let a_shape = [3, 3];
    let b_shape = [3];

    let a = leaf(&a_data, &a_shape);
    let b = leaf(&b_data, &b_shape);
    let x = glin::solve_differentiable(&a, &b).unwrap();
    assert_eq!(x.shape(), &[3]);
    let loss = reduce_sum(&x).unwrap();
    loss.backward().unwrap();

    let ga = a.grad().unwrap().unwrap().data().unwrap().to_vec();
    let gb = b.grad().unwrap().unwrap().data().unwrap().to_vec();

    let num_a = fd_grad(&a_data, &a_shape, 1e-6, |xa| {
        let bb = no_grad_leaf(&b_data, &b_shape);
        let xx = linalg_fwd::solve(xa, &bb).unwrap();
        reduce_sum(&xx).unwrap().item().unwrap()
    });
    let num_b = fd_grad(&b_data, &b_shape, 1e-6, |xb| {
        let aa = no_grad_leaf(&a_data, &a_shape);
        let xx = linalg_fwd::solve(&aa, xb).unwrap();
        reduce_sum(&xx).unwrap().item().unwrap()
    });

    assert_eq!(ga.len(), 9, "solve gA (vector rhs) must be 3x3");
    assert_eq!(gb.len(), 3, "solve gB (vector rhs) must be length 3");
    assert_grad_close(&ga, &num_a, 1e-4, "solve gA (vector rhs) vs FD");
    assert_grad_close(&gb, &num_b, 1e-4, "solve gB (vector rhs) vs FD");
}
