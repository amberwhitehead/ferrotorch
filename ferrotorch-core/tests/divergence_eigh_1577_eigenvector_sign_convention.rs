//! Reframed divergence pin (#1584): `ferrotorch_core::linalg::eigh`'s
//! eigenvector output uses a DIFFERENT per-column SIGN CONVENTION than
//! `torch.linalg.eigh` (LAPACK `syevd`). The eigenVALUES match torch exactly;
//! per-eigenvector column signs differ matrix-by-matrix.
//!
//! ## Why the ORIGINAL `<wv, U>` assertion was ILL-POSED (gauge freedom)
//!
//! This file originally asserted a torch-exact `A.grad` for the loss
//! `<wv, U>` (a fixed-seed linear functional of the raw eigenvector entries).
//! That loss is NOT invariant under eigenvector sign flips: `U` and
//! `U·diag(±1)` are both valid eigenvector matrices (eigenvectors are defined
//! only up to a sign — upstream documents this at
//! `torch/csrc/autograd/FunctionsManual.cpp:3877-3880`: "The eigenvectors ...
//! are specified up to multiplication by e^{i phi}. The specified loss function
//! depends on this quantity, so it is ill-defined."). Because LAPACK `syevd`
//! (torch) and faer (ferray) each emit their own arbitrary column signs,
//! `<wv, U>` is GAUGE-DEPENDENT: torch's answer depends on LAPACK's signs,
//! ferrotorch's on faer's. Neither is "more correct", and matching torch's
//! arbitrary signs would require replicating `syevd` (impractical). The
//! `EighBackwardV` VJP formula is CORRECT — forcing torch to ferray's signs
//! makes the gradients match exactly. So the original assertion pinned an
//! ill-posed quantity, not a backward-formula bug.
//!
//! ## What this file asserts now (WELL-POSED quantities)
//!
//! (a) The gradient of a SIGN-INVARIANT loss `sum((U*U)*M)` (each `U_ij^2` is
//!     unchanged under any column sign flip — the canonical class of well-posed
//!     objectives on eigenvectors: PCA, whitening, `U @ diag(f(w)) @ U^T`
//!     reconstructions) matches LIVE `torch 2.11.0+cu130` float64 `A.grad`.
//!     This is the STRONGER, well-posed correctness check on the VJP.
//! (b) Eigenvector DETERMINISM + canonical sign: two `eigh` calls on the same
//!     input return identical eigenvectors, and the canonical convention
//!     (`canonicalize_eigenvector_signs` in `ferrotorch-core/src/linalg.rs`:
//!     largest-abs-value component of each column non-negative) holds.
//!
//! ferrotorch's eigh signs still differ from torch's (gauge freedom is real),
//! but ferrotorch now has a deterministic, defined sign contract, and the VJP
//! is verified against torch on the only mathematically well-posed kind of
//! eigenvector loss.
//!
//! R-CHAR-3 (a): every expected value here is a LIVE `torch 2.11.0+cu130`
//! float64 call, NOT copied from the ferrotorch side. Reproduce with:
//!
//!   import torch; torch.set_default_dtype(torch.float64)
//!   A = torch.tensor(<a_data>).reshape(n,n).clone().requires_grad_(True)
//!   w, U = torch.linalg.eigh(A)
//!   M = torch.tensor(<m_data>).reshape(n,n)
//!   ((U * U) * M).sum().backward()
//!   A.grad   # the literal asserted below

use ferrotorch_core::Tensor;
use ferrotorch_core::grad_fns::arithmetic::mul;
use ferrotorch_core::grad_fns::reduction::sum as reduce_sum;
use ferrotorch_core::linalg as linalg_fwd;
use ferrotorch_core::storage::TensorStorage;

fn leaf(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}
fn ng(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}
fn assert_close(actual: &[f64], torch: &[f64], tol: f64, label: &str) {
    assert_eq!(actual.len(), torch.len(), "{label}: length mismatch");
    for (i, (&a, &t)) in actual.iter().zip(torch.iter()).enumerate() {
        assert!(
            (a - t).abs() < tol,
            "{label} grad[{i}]: ferrotorch={a}, torch={t}, diff={}",
            (a - t).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// (a) SIGN-INVARIANT loss gradient vs LIVE torch, 3x3.
//
// Loss = sum((U*U) * M): each U_ij^2 is invariant under column sign flips, so
// the gradient is gauge-INDEPENDENT and both torch and ferrotorch must agree
// regardless of their differing eigenvector sign conventions.
//
// torch 2.11.0+cu130:
//   A = torch.tensor([4.0,0.5,0.3, 0.5,2.5,0.2, 0.3,0.2,1.0]).reshape(3,3).clone().requires_grad_(True)
//   w,U = torch.linalg.eigh(A)
//   M = torch.tensor([0.2,-0.5,0.7, 0.3,0.1,-0.4, -0.6,0.8,0.25]).reshape(3,3)
//   ((U*U)*M).sum().backward()
//   A.grad  (row-major) below.
// (Verified sign-invariant: flipping columns of U by diag(±1) leaves A.grad
//  bit-identical.)
// ---------------------------------------------------------------------------
#[test]
fn eigh_sign_invariant_recon_grad_3x3_matches_torch() {
    let d = [4.0, 0.5, 0.3, 0.5, 2.5, 0.2, 0.3, 0.2, 1.0];
    let m = [0.2, -0.5, 0.7, 0.3, 0.1, -0.4, -0.6, 0.8, 0.25];
    let a = leaf(&d, &[3, 3]);
    let (_w, u) = linalg_fwd::eigh(&a).unwrap();
    // Sign-invariant loss: sum((U*U) * M).
    let usq = mul(&u, &u).unwrap();
    let weighted = mul(&usq, &ng(&m, &[3, 3])).unwrap();
    reduce_sum(&weighted).unwrap().backward().unwrap();
    let g = a.grad().unwrap().unwrap();

    // LIVE torch.linalg.eigh A.grad for the sign-invariant loss (R-CHAR-3 (a)).
    let torch = [
        0.153_351_874_733,
        -0.223_150_307_465,
        -0.011_353_963_581,
        -0.223_150_307_465,
        -0.164_767_557_695,
        0.042_645_994_893,
        -0.011_353_963_581,
        0.042_645_994_893,
        0.011_415_682_962,
    ];
    assert_close(
        g.data().unwrap(),
        &torch,
        1e-4,
        "eigh sign-invariant recon A.grad 3x3 vs torch",
    );
}

// ---------------------------------------------------------------------------
// (a) SIGN-INVARIANT loss gradient vs LIVE torch, 4x4.
//
// torch 2.11.0+cu130:
//   A = torch.tensor([5.0,0.4,0.3,0.1, 0.4,3.5,0.2,0.15, 0.3,0.2,2.0,0.25,
//                     0.1,0.15,0.25,0.8]).reshape(4,4).clone().requires_grad_(True)
//   w,U = torch.linalg.eigh(A)
//   M = torch.tensor([0.7,-0.3,1.1,0.2, -0.5,0.9,0.4,-1.2, 0.6,-0.8,0.3,1.5,
//                     -0.1,0.45,-0.65,0.25]).reshape(4,4)
//   ((U*U)*M).sum().backward()
//   A.grad (row-major) below.
// ---------------------------------------------------------------------------
#[test]
fn eigh_sign_invariant_recon_grad_4x4_matches_torch() {
    let d = [
        5.0, 0.4, 0.3, 0.1, 0.4, 3.5, 0.2, 0.15, 0.3, 0.2, 2.0, 0.25, 0.1, 0.15, 0.25, 0.8,
    ];
    let m = [
        0.7, -0.3, 1.1, 0.2, -0.5, 0.9, 0.4, -1.2, 0.6, -0.8, 0.3, 1.5, -0.1, 0.45, -0.65, 0.25,
    ];
    let a = leaf(&d, &[4, 4]);
    let (_w, u) = linalg_fwd::eigh(&a).unwrap();
    let usq = mul(&u, &u).unwrap();
    let weighted = mul(&usq, &ng(&m, &[4, 4])).unwrap();
    reduce_sum(&weighted).unwrap().backward().unwrap();
    let g = a.grad().unwrap().unwrap();

    // LIVE torch.linalg.eigh A.grad for the sign-invariant loss (R-CHAR-3 (a)).
    let torch = [
        0.033_787_639_518,
        -0.088_745_417_758,
        0.043_696_184_003,
        -0.013_761_821_568,
        -0.088_745_417_758,
        -0.070_639_818_616,
        0.122_614_799_814,
        -0.036_818_587_873,
        0.043_696_184_003,
        0.122_614_799_814,
        -0.066_224_629_876,
        0.275_639_249_385,
        -0.013_761_821_568,
        -0.036_818_587_873,
        0.275_639_249_385,
        0.103_076_808_974,
    ];
    assert_close(
        g.data().unwrap(),
        &torch,
        1e-4,
        "eigh sign-invariant recon A.grad 4x4 vs torch",
    );
}

// ---------------------------------------------------------------------------
// (b) Eigenvector DETERMINISM + canonical sign convention.
//
// `eigh` is deterministic (same input -> identical eigenvectors), and the
// canonical sign convention (`canonicalize_eigenvector_signs`: the
// largest-absolute-value component of each eigenvector column is made
// non-negative) holds. This is ferrotorch's STABLE contract — it does NOT
// match torch's LAPACK signs (gauge freedom), but it is reproducible.
// ---------------------------------------------------------------------------
#[test]
fn eigh_eigenvectors_are_deterministic_and_canonically_signed() {
    let d = [
        5.0, 0.4, 0.3, 0.1, 0.4, 3.5, 0.2, 0.15, 0.3, 0.2, 2.0, 0.25, 0.1, 0.15, 0.25, 0.8,
    ];
    let n = 4usize;
    let a1 = ng(&d, &[n, n]);
    let a2 = ng(&d, &[n, n]);
    let (_w1, u1) = linalg_fwd::eigh(&a1).unwrap();
    let (_w2, u2) = linalg_fwd::eigh(&a2).unwrap();
    let u1d = u1.data().unwrap();
    let u2d = u2.data().unwrap();

    // Determinism: identical input -> bit-identical eigenvectors.
    assert_eq!(u1d.len(), n * n);
    for (i, (&x, &y)) in u1d.iter().zip(u2d.iter()).enumerate() {
        assert!(
            (x - y).abs() == 0.0,
            "eigh non-deterministic at [{i}]: {x} vs {y}"
        );
    }

    // Canonical sign: in each column, the largest-abs-value entry is >= 0.
    for col in 0..n {
        let mut best_row = 0usize;
        let mut best_abs = 0.0f64;
        for row in 0..n {
            let v = u1d[row * n + col].abs();
            if v > best_abs {
                best_abs = v;
                best_row = row;
            }
        }
        let pivot = u1d[best_row * n + col];
        assert!(
            pivot >= 0.0,
            "eigh column {col} not canonically signed: pivot at row {best_row} = {pivot} < 0"
        );
    }
}
