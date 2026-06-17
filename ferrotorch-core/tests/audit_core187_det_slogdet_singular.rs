//! CORE-187: tracked singular det/slogdet forward parity.
//!
//! PyTorch's `_linalg_det` / `_linalg_slogdet` forward stores LU metadata and
//! does not compute an inverse at forward time. A singular input with
//! `requires_grad=True` therefore has the same forward surface as an untracked
//! input: `det -> 0`, `slogdet -> (0, -inf)`. The inverse/solve belongs to the
//! backward formula.

#![allow(
    clippy::float_cmp,
    reason = "these tests pin exact PyTorch zero/signed-infinity singular surfaces"
)]

use ferrotorch_core::linalg;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn leaf(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

fn grad_data(t: &Tensor<f64>) -> Vec<f64> {
    t.grad()
        .expect("grad lookup")
        .expect("grad must be present")
        .data_vec()
        .expect("grad data")
}

fn assert_all_exact_zero(data: &[f64], label: &str) {
    for (idx, &value) in data.iter().enumerate() {
        assert!(
            value == 0.0,
            "{label}[{idx}] expected zero from PyTorch ordinary singular det backward, got {value}"
        );
    }
}

#[test]
fn det_singular_tracked_forward_matches_untracked_and_backward_zero() {
    let a = leaf(&[1.0, 0.0, 0.0, 0.0], &[2, 2]);

    let det = linalg::det(&a).expect("tracked singular det forward");
    assert!(
        det.requires_grad(),
        "tracked det output must retain grad_fn"
    );
    assert_eq!(det.item().expect("det scalar"), 0.0);

    det.backward().expect("singular det ordinary backward");
    assert_all_exact_zero(&grad_data(&a), "det grad");
}

#[test]
fn det_singular_one_by_one_backward_is_seed_like_pytorch() {
    let a = leaf(&[0.0], &[1, 1]);

    let det = linalg::det(&a).expect("tracked 1x1 singular det forward");
    assert_eq!(det.item().expect("det scalar"), 0.0);

    det.backward().expect("1x1 singular det backward");
    assert_eq!(
        grad_data(&a),
        vec![1.0],
        "PyTorch special-cases d(det([[x]]))/dx = 1 even at x=0"
    );
}

#[test]
fn slogdet_singular_tracked_forward_returns_zero_and_negative_infinity() {
    let a = leaf(&[1.0, 0.0, 0.0, 0.0], &[2, 2]);

    let (sign, logabsdet) = linalg::slogdet(&a).expect("tracked singular slogdet forward");
    assert!(
        sign.grad_fn().is_none(),
        "real slogdet sign output is non-differentiable"
    );
    assert!(
        logabsdet.requires_grad(),
        "logabsdet output must retain grad_fn"
    );
    assert_eq!(sign.item().expect("sign scalar"), 0.0);
    let logabs = logabsdet.item().expect("logabsdet scalar");
    assert!(
        logabs.is_infinite() && logabs.is_sign_negative(),
        "singular slogdet logabsdet should be -inf, got {logabs}"
    );
}

#[test]
fn slogdet_singular_ordinary_backward_matches_lu_solve_nan_inf_surface() {
    let a = leaf(&[1.0, 0.0, 0.0, 0.0], &[2, 2]);
    let (_sign, logabsdet) = linalg::slogdet(&a).expect("tracked singular slogdet forward");

    logabsdet
        .backward()
        .expect("ordinary singular slogdet backward");
    let grad = grad_data(&a);
    assert!(
        grad[0].is_nan() && grad[1].is_nan() && grad[2].is_nan(),
        "PyTorch ordinary LU-solve slogdet backward yields NaN off the nonsmooth branch, got {grad:?}"
    );
    assert!(
        grad[3].is_infinite() && grad[3].is_sign_positive(),
        "PyTorch ordinary LU-solve slogdet backward yields +inf on the zero pivot, got {grad:?}"
    );
}
