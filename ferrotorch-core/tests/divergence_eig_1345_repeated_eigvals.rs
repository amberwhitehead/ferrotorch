//! R-BUILD-4 adversarial audit of commit `97ebfdf16` (#1345 REQ-14 eigvals
//! backward) — REPEATED but DIAGONALIZABLE eigenvalues.
//!
//! The builder's design note claims the eig/eigvals VJP is "EXACT for
//! DIAGONALIZABLE A (distinct eigenvalues)" and that a repeated eigenvalue makes
//! `V` singular / `Econj` diverge. That is TRUE for the eig (eigenVECTOR) path
//! (the `1/(L_j - L_i)` Econj term). But the EIGVALS-ONLY path
//! `gA = V^{-H} diag(gL) V^H` (FunctionsManual.cpp:3857-3862) does NOT touch
//! Econj — it has a perfectly well-defined gradient whenever `V` is invertible,
//! which holds for any DIAGONALIZABLE matrix even with REPEATED eigenvalues
//! (e.g. `2*I`, whose `V` is the identity).
//!
//! torch confirms (LIVE torch 2.11.0+cu130):
//!   A = torch.tensor([2.,0.,0.,2.]).reshape(2,2).requires_grad_(True)
//!   L = torch.linalg.eigvals(A)
//!   ((L.real*[1.3,-0.7]).sum()+(L.imag*[0.4,0.6]).sum()).backward()
//!   A.grad -> [1.3, 0.0, 0.0, -0.7]    (NO error)
//!
//!   S=[[1,.5,.2],[0,1,.3],[0,0,1]]; A=S@diag(2,2,5)@inv(S) (repeated 2,2 + 5)
//!   L=torch.linalg.eigvals(A); (L.real*[1.3,-0.7,0.9]).sum().backward()
//!   A.grad -> [1.3,0,0, 0,-0.7,0, -0.08,0.48,0.9]   (NO error)
//!
//! These assert ferrotorch's `EigvalsBackward` matches torch on repeated
//! diagonalizable inputs. If ferray's `eig` returns a singular/non-invertible V
//! for `2*I` (or orders/normalizes the degenerate column badly so `c_solve`
//! reports SingularMatrix), this FAILS — exposing a regime the builder declared
//! out-of-scope but torch supports.

use ferrotorch_core::Tensor;
use ferrotorch_core::grad_fns::arithmetic::mul;
use ferrotorch_core::grad_fns::reduction::sum as reduce_sum;
use ferrotorch_core::linalg as linalg_fwd;
use ferrotorch_core::storage::TensorStorage;

fn leaf(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

fn no_grad_leaf(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "{label}[{i}]: ferrotorch={a}, torch={e}, diff={}",
            (a - e).abs()
        );
    }
}

fn eigval_linear_loss(w: &Tensor<f64>, cr: &[f64], ci: &[f64]) -> Tensor<f64> {
    let n = cr.len();
    let mut wt = vec![0.0; n * 2];
    for k in 0..n {
        wt[2 * k] = cr[k];
        wt[2 * k + 1] = ci[k];
    }
    let wts = no_grad_leaf(&wt, &[n, 2]);
    reduce_sum(&mul(w, &wts).unwrap()).unwrap()
}

fn eigvals_grad(a_data: &[f64], n: usize, cr: &[f64], ci: &[f64]) -> Result<Vec<f64>, String> {
    let a = leaf(a_data, &[n, n]);
    let w = linalg_fwd::eigvals(&a).map_err(|e| format!("forward: {e:?}"))?;
    let loss = eigval_linear_loss(&w, cr, ci);
    loss.backward().map_err(|e| format!("backward: {e:?}"))?;
    Ok(a.grad().unwrap().unwrap().data().unwrap().to_vec())
}

/// `2*I` has repeated eigenvalue 2 but is DIAGONALIZABLE (V = I, invertible), so
/// torch returns a finite A.grad = [1.3, 0, 0, -0.7] for the eigvals path.
#[test]
fn eigvals_backward_repeated_2i_matches_torch() {
    let g = eigvals_grad(&[2.0, 0.0, 0.0, 2.0], 2, &[1.3, -0.7], &[0.4, 0.6])
        .expect("eigvals on 2*I (diagonalizable, repeated eig) must not error");
    assert_close(
        &g,
        &[1.3, 0.0, 0.0, -0.7],
        1e-6,
        "eigvals 2I A.grad vs torch",
    );
}

/// Diagonalizable with a repeated pair (2,2) + a distinct (5):
/// A = S diag(2,2,5) S^{-1}, S upper-unitriangular. torch returns a finite grad.
#[test]
fn eigvals_backward_repeated_pair_3x3_matches_torch() {
    let a = [2.0, 0.0, 0.6, 0.0, 2.0, 0.9, 0.0, 0.0, 5.0];
    let g = eigvals_grad(&a, 3, &[1.3, -0.7, 0.9], &[0.0, 0.0, 0.0])
        .expect("eigvals on diagonalizable repeated-pair 3x3 must not error");
    // torch A.grad (L.real*[1.3,-0.7,0.9]).sum() only.
    let torch = [1.3, 0.0, 0.0, 0.0, -0.7, 0.0, -0.08, 0.48, 0.9];
    assert_close(
        &g,
        &torch,
        1e-6,
        "eigvals repeated-pair 3x3 A.grad vs torch",
    );
}
