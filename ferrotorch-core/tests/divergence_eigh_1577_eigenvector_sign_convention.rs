//! Divergence pin: `ferrotorch_core::linalg::eigh`'s eigenvector output (and
//! therefore the eigenvector-path `A.grad` newly shipped in `bbf40a490`,
//! #1577) uses a DIFFERENT eigenvector SIGN CONVENTION than
//! `torch.linalg.eigh` (LAPACK `syevd`). The eigenvalues match torch exactly,
//! but per-eigenvector column signs differ matrix-by-matrix.
//!
//! The shipped `EighBackwardV` VJP formula is internally CORRECT — it
//! self-consistently mirrors ferray's forward (the builder's in-file FD test
//! `eigh_public_forward_is_grad_aware_and_matches_fd` perturbs the SAME forward
//! and so passes). But a loss that is NOT invariant under eigenvector sign
//! flips (e.g. any `<W, U>` linear functional of the eigenvector entries, as
//! every real training objective on eigenvectors is) produces a DIFFERENT
//! `A.grad` than torch. A model trained against torch-produced eigenvector
//! gradients receives the wrong gradient from ferrotorch.
//!
//! Upstream PyTorch site for the convention: `torch.linalg.eigh` returns the
//! LAPACK `syevd` eigenvectors directly (the column-sign convention LAPACK
//! emits). The eigenvector-gauge invariance documented at
//! `torch/csrc/autograd/FunctionsManual.cpp:3880` ("The eigenvectors ... are
//! specified up to multiplication by e^{i phi}") is the property the loss must
//! respect for the gradient to be convention-independent; a plain `<W, U>` loss
//! does NOT respect it, so the sign convention is observable.
//!
//! ferrotorch site: `ferrotorch-core/src/grad_fns/linalg.rs` `eigh_differentiable`
//! / `EighBackwardV` (commit `bbf40a490`), driven by ferray's `eigh` forward in
//! `ferrotorch-core/src/linalg.rs::eigh`.
//!
//! Tracking: filed below.
//!
//! R-CHAR-3 (a): every expected value here is a LIVE `torch 2.11.0+cu130`
//! float64 call, NOT copied from the ferrotorch side. Reproduce with:
//!
//!   import torch; torch.set_default_dtype(torch.float64)
//!   A = torch.tensor(<a_data>).reshape(n,n).requires_grad_(True)
//!   w, U = torch.linalg.eigh(A)
//!   (U * torch.tensor(<wv>).reshape(n,n)).sum().backward()
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
// eigh eigenvector-path A.grad, 3x3 — diverges from torch.
//
// Input is the builder's own `eigh_public_forward_is_grad_aware_and_matches_fd`
// symmetric 3x3 with distinct eigenvalues. Loss = <wv, U> (eigenvector-only).
//
// torch 2.11.0+cu130:
//   A = torch.tensor([4.0,0.5,0.3, 0.5,2.5,0.2, 0.3,0.2,1.0]).reshape(3,3)
//   w,U = torch.linalg.eigh(A.requires_grad_(True))
//   (U * torch.tensor([0.2,-0.5,0.7, 0.3,0.1,-0.4, -0.6,0.8,0.25]).reshape(3,3)).sum().backward()
//   A.grad = [-0.124404055, 0.204859773, -0.1249883448,
//              0.204859773,  0.1018044171, 0.2265969334,
//             -0.1249883448, 0.2265969334, 0.0225996379]
//
// ferrotorch (commit bbf40a490) produces, on the same input/loss:
//   [ 0.0405387382, -0.0723754336, -0.0474470197,
//    -0.0723754336, -0.1063679218,  0.3469482841,
//    -0.0474470197,  0.3469482841,  0.0658291836]
// — the sign-flipped eigenvector convention (ferray flips columns 0 and 2 vs
// torch on this matrix) makes the eigenvector-path gradient diverge.
// ---------------------------------------------------------------------------
#[test]
fn divergence_eigh_eigenvector_path_grad_3x3_vs_torch() {
    let d = [4.0, 0.5, 0.3, 0.5, 2.5, 0.2, 0.3, 0.2, 1.0];
    let wv = [0.2, -0.5, 0.7, 0.3, 0.1, -0.4, -0.6, 0.8, 0.25];
    let a = leaf(&d, &[3, 3]);
    let (_w, u) = linalg_fwd::eigh(&a).unwrap();
    reduce_sum(&mul(&u, &ng(&wv, &[3, 3])).unwrap())
        .unwrap()
        .backward()
        .unwrap();
    let g = a.grad().unwrap().unwrap();

    // LIVE torch.linalg.eigh A.grad (R-CHAR-3 (a)).
    let torch = [
        -0.124_404_055_0,
        0.204_859_773_0,
        -0.124_988_344_8,
        0.204_859_773_0,
        0.101_804_417_1,
        0.226_596_933_4,
        -0.124_988_344_8,
        0.226_596_933_4,
        0.022_599_637_9,
    ];
    assert_close(
        g.data().unwrap(),
        &torch,
        1e-7,
        "eigh eigenvector-path A.grad 3x3 vs torch",
    );
}

// ---------------------------------------------------------------------------
// eigh BOTH-path (eigenvalues + eigenvectors) A.grad, 3x3 — diverges from torch.
//
// torch 2.11.0+cu130, same A:
//   loss = (w*[0.4,-0.9,1.5]).sum() + (U*[0.2,-0.5,0.7,0.3,0.1,-0.4,-0.6,0.8,0.25]).sum()
//   loss.backward()
//   A.grad = [1.1488500074, 0.8856110128, 0.0164118546,
//             0.8856110128, -0.5772559374, 0.1709648953,
//             0.0164118546, 0.1709648953, 0.42840593]
// The eigenvalue contribution matches torch; the eigenvector contribution does
// not, so the combined gradient diverges.
// ---------------------------------------------------------------------------
#[test]
fn divergence_eigh_combined_path_grad_3x3_vs_torch() {
    let d = [4.0, 0.5, 0.3, 0.5, 2.5, 0.2, 0.3, 0.2, 1.0];
    let ww = [0.4, -0.9, 1.5];
    let wv = [0.2, -0.5, 0.7, 0.3, 0.1, -0.4, -0.6, 0.8, 0.25];
    let a = leaf(&d, &[3, 3]);
    let (w, u) = linalg_fwd::eigh(&a).unwrap();
    let lw = reduce_sum(&mul(&w, &ng(&ww, &[3])).unwrap()).unwrap();
    let lv = reduce_sum(&mul(&u, &ng(&wv, &[3, 3])).unwrap()).unwrap();
    ferrotorch_core::grad_fns::arithmetic::add(&lw, &lv)
        .unwrap()
        .backward()
        .unwrap();
    let g = a.grad().unwrap().unwrap();

    // LIVE torch.linalg.eigh A.grad (R-CHAR-3 (a)).
    let torch = [
        1.148_850_007_4,
        0.885_611_012_8,
        0.016_411_854_6,
        0.885_611_012_8,
        -0.577_255_937_4,
        0.170_964_895_3,
        0.016_411_854_6,
        0.170_964_895_3,
        0.428_405_930_0,
    ];
    assert_close(
        g.data().unwrap(),
        &torch,
        1e-7,
        "eigh combined-path A.grad 3x3 vs torch",
    );
}

// ---------------------------------------------------------------------------
// eigh eigenvector-path A.grad, 4x4 — diverges from torch (different per-matrix
// flip pattern: ferray flips columns 0 and 3 on this 4x4).
//
// torch 2.11.0+cu130:
//   M = [[5,0.4,0.3,0.1],[0.4,3.5,0.2,0.15],[0.3,0.2,2,0.25],[0.1,0.15,0.25,0.8]]
//   w,U = torch.linalg.eigh(torch.tensor(M).requires_grad_(True))
//   (U * torch.tensor([0.1*i-0.5 for i in range(16)]).reshape(4,4)).sum().backward()
//   A.grad (row-major) below.
// ---------------------------------------------------------------------------
#[test]
fn divergence_eigh_eigenvector_path_grad_4x4_vs_torch() {
    let d = [
        5.0, 0.4, 0.3, 0.1, 0.4, 3.5, 0.2, 0.15, 0.3, 0.2, 2.0, 0.25, 0.1, 0.15, 0.25, 0.8,
    ];
    let wv: Vec<f64> = (0..16).map(|i| 0.1 * (i as f64) - 0.5).collect();
    let a = leaf(&d, &[4, 4]);
    let (_w, u) = linalg_fwd::eigh(&a).unwrap();
    reduce_sum(&mul(&u, &ng(&wv, &[4, 4])).unwrap())
        .unwrap()
        .backward()
        .unwrap();
    let g = a.grad().unwrap().unwrap();

    // LIVE torch.linalg.eigh A.grad (R-CHAR-3 (a)).
    let torch = [
        0.068_014_96,
        -0.034_533_19,
        -0.227_633_46,
        -0.083_409_48,
        -0.034_533_19,
        -0.098_122_08,
        0.213_768_44,
        0.210_578_16,
        -0.227_633_46,
        0.213_768_44,
        0.185_166_16,
        -0.421_449_83,
        -0.083_409_48,
        0.210_578_16,
        -0.421_449_83,
        -0.155_059_04,
    ];
    assert_close(
        g.data().unwrap(),
        &torch,
        1e-6,
        "eigh eigenvector-path A.grad 4x4 vs torch",
    );
}
