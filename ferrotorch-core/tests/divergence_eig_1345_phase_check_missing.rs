//! R-BUILD-4 adversarial audit of commit `97ebfdf16` (#1345 REQ-12 eig backward).
//!
//! DIVERGENCE: torch's `linalg_eig_backward` GUARDS against ill-defined losses.
//! Because non-Hermitian eigenvectors are only defined up to a per-column phase
//! `V_j -> V_j e^{i phi}`, torch checks that the loss is phase-invariant and
//! RAISES a RuntimeError otherwise:
//!
//!   torch/csrc/autograd/FunctionsManual.cpp:3867-3879
//!     if (V.is_complex() && !at::isTensorSubclassLike(diag_VhgV)) {
//!       const auto imdiag_VhgV = at::imag(diag_VhgV);
//!       TORCH_CHECK(
//!           at::allclose(imdiag_VhgV, at::zeros_like(imdiag_VhgV),
//!                        /*rtol=*/1e-2, /*atol=*/1e-2),
//!           ...
//!           ": The eigenvectors in the complex case are specified up to "
//!           "multiplication by e^{i phi}. The specified loss function depends "
//!           "on this quantity, so it is ill-defined.");
//!     }
//!
//! ferrotorch's `EigBackwardV::grad_a_from_gv`
//! (ferrotorch-core/src/grad_fns/linalg.rs:6055-6096) computes
//! `VhgV = V^H @ gV` and the unit-norm tangent projection, but has NO
//! `imag(diag(VhgV)) ≈ 0` guard. For a non-phase-invariant loss it therefore
//! SILENTLY returns a gauge-dependent (ill-defined / garbage) `A.grad` where
//! torch errors. This is a behavioral divergence: torch -> RuntimeError,
//! ferrotorch -> a finite number.
//!
//! LIVE torch 2.11.0+cu130 (verified):
//!   A = torch.tensor([1.,-1.,1.,1.]).reshape(2,2).requires_grad_(True)
//!   L,V = torch.linalg.eig(A)
//!   V.real.sum().backward()    # RuntimeError: ... ill-defined
//!   V.imag.sum().backward()    # RuntimeError: ... ill-defined
//! ferrotorch at HEAD silently returns A.grad = [0.354, -1.06, 0.354, -0.354].
//!
//! This test asserts ferrotorch ALSO rejects (returns Err from backward) the
//! ill-defined loss, matching torch. It FAILS at HEAD because ferrotorch
//! silently succeeds. Left UN-`#[ignore]`d: a silent ill-defined gradient where
//! torch errors is a release-blocker (the failing test IS the block).
//!
//! Tracking: crosslink #1590 (blocker).

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

/// Build a loss that weights ONLY the real slot of the complex `[n,n,2]`
/// eigenvectors: `sum(V.real)`. This is the loss torch rejects (it depends on
/// the per-column phase, so `imag(diag(V^H gV)) != 0`).
fn eigvec_real_only_loss(v: &Tensor<f64>, n: usize) -> Tensor<f64> {
    // weight = 1 on every re-slot, 0 on every im-slot.
    let mut wt = vec![0.0; n * n * 2];
    for idx in 0..n * n {
        wt[2 * idx] = 1.0; // re slot
        wt[2 * idx + 1] = 0.0; // im slot
    }
    let wts = no_grad_leaf(&wt, &[n, n, 2]);
    reduce_sum(&mul(v, &wts).unwrap()).unwrap()
}

/// torch RAISES `RuntimeError: ... ill-defined` on a phase-DEPENDENT eig V loss
/// (FunctionsManual.cpp:3867-3879). ferrotorch's EigBackwardV has no such guard,
/// so the backward SUCCEEDS and produces a gauge-dependent A.grad. To match
/// torch, `loss.backward()` must return an `Err`. FAILS at HEAD: backward is Ok.
/// Tracking: #1590.
#[test]
fn eig_backward_rejects_phase_dependent_loss_like_torch() {
    let a = leaf(&[1.0, -1.0, 1.0, 1.0], &[2, 2]);
    let (_w, v) = linalg_fwd::eig(&a).unwrap();
    assert!(v.grad_fn().is_some(), "eig V output must carry a grad_fn");
    let loss = eigvec_real_only_loss(&v, 2);
    let result = loss.backward();
    assert!(
        result.is_err(),
        "torch RAISES RuntimeError for a phase-dependent eig loss \
         (FunctionsManual.cpp:3867-3879, imag(diag(V^H gV)) != 0), but \
         ferrotorch's EigBackwardV silently returned A.grad = {:?} — \
         no phase-invariance guard. Tracking #1590.",
        a.grad()
            .ok()
            .flatten()
            .and_then(|g| g.data().ok().map(|d| d.to_vec()))
    );
}
