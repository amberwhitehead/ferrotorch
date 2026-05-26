//! Divergence: commit `2847407e6` (#1272) added a 0-d forward path to
//! `grad_fns::indexing::index_fill` that successfully constructs an
//! `IndexFillBackward` graph node from a 0-d input — but the **backward
//! impl at `grad_fns/indexing.rs:1418-1422` was NOT updated to handle the
//! 0-d shape**.
//!
//! The forward stores `dim: 0` and the original 0-d input. On backward
//! the impl computes:
//!     let input_shape = self.input.shape();   // = []
//!     let dim = self.dim;                     // = 0
//!     let outer = input_shape[..dim].iter().product();        // = 1 (ok)
//!     let inner = input_shape[dim + 1..].iter().product();    // PANIC
//!     let dim_size = input_shape[dim];                        // PANIC
//!
//! `input_shape[1..]` panics for a 0-d tensor because `input_shape.len() == 0`:
//!
//!     range start index 1 out of range for slice of length 0
//!
//! Live torch oracle confirms the upstream behavior the impl should match:
//!
//!     >>> x = torch.tensor(5.0, requires_grad=True)
//!     >>> y = torch.index_fill(x, 0, torch.tensor([0]), -1.0)
//!     >>> y.sum().backward()
//!     >>> x.grad
//!     tensor(0.)
//!
//! The Sonnet 4.6 fixer who landed #1272 wrote a 0-d forward path AND
//! recorded a saved_index of `vec![0]` for it, but never wrote a test that
//! actually calls .backward() through the 0-d path — so the backward
//! breakage shipped unnoticed.
//!
//! Root cause: the backward path needs the same unsqueeze-to-1-d treatment
//! the forward applies (or an explicit ndim==0 short-circuit). The Sonnet
//! fixer dropped this upstream edge case.
//!
//! Tracking: blocker (filed by acto-critic against #1272 SHIPPED claim).

use ferrotorch_core::grad_fns::indexing::index_fill;
use ferrotorch_core::int_tensor::IntTensor;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::{GradFn, Tensor};

fn idx_i64(d: Vec<i64>, s: Vec<usize>) -> IntTensor<i64> {
    IntTensor::from_vec(d, s).unwrap()
}

/// Drive `.backward()` through the 0-d path. Must return `tensor(0.)` per
/// upstream oracle. Currently panics in indexing.rs:1418-1422.
#[test]
fn index_fill_zero_d_backward_filled_must_return_zero_grad() {
    let x = Tensor::from_storage(TensorStorage::cpu(vec![5.0_f32]), vec![], true).unwrap();
    let y = index_fill(&x, 0, &idx_i64(vec![0], vec![1]), -1.0)
        .expect("0-d forward must succeed per #1272");
    let gf = y.grad_fn().expect("grad_fn present");
    assert_eq!(GradFn::name(&*gf), "IndexFillBackward");
    let go = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f32]), vec![], false).unwrap();
    let grads = gf
        .backward(&go)
        .expect("0-d backward must not panic — currently fails");
    let g = grads[0].as_ref().expect("Some(grad)");
    assert_eq!(g.shape(), &[] as &[usize], "grad shape must be 0-d");
    assert_eq!(
        g.data().unwrap(),
        &[0.0_f32],
        "filled 0-d scalar must have grad=0 per derivatives.yaml:884-887"
    );
}

/// 0-d + empty-index backward — the saved index is empty, so backward
/// should be the identity (no zeroing). Live torch oracle: grad=tensor(1.)
/// (empty index ⇒ no fill ⇒ d(y)/d(x) = 1).
#[test]
fn index_fill_zero_d_backward_empty_index_must_return_identity_grad() {
    let x = Tensor::from_storage(TensorStorage::cpu(vec![5.0_f32]), vec![], true).unwrap();
    let y = index_fill(&x, 0, &idx_i64(vec![], vec![0]), -1.0)
        .expect("0-d + empty-idx forward must succeed");
    let gf = y.grad_fn().expect("grad_fn present");
    let go = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f32]), vec![], false).unwrap();
    let grads = gf
        .backward(&go)
        .expect("0-d empty-idx backward must not panic");
    let g = grads[0].as_ref().expect("Some(grad)");
    assert_eq!(g.shape(), &[] as &[usize]);
    assert_eq!(
        g.data().unwrap(),
        &[1.0_f32],
        "empty-index 0-d must pass grad through (identity)"
    );
}

/// 0-d + negative index — combines the #1272 (0-d forward) and #1273 (wrap)
/// fixes. The backward path that goes through IndexFillBackward with `dim=0`
/// on a 0-d input is broken for ALL non-empty index lists.
#[test]
fn index_fill_zero_d_backward_negative_index_must_return_zero_grad() {
    let x = Tensor::from_storage(TensorStorage::cpu(vec![5.0_f32]), vec![], true).unwrap();
    let y = index_fill(&x, 0, &idx_i64(vec![-1], vec![1]), -1.0)
        .expect("0-d + idx=-1 forward must succeed (wraps to 0)");
    let gf = y.grad_fn().expect("grad_fn present");
    let go = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f32]), vec![], false).unwrap();
    let grads = gf
        .backward(&go)
        .expect("0-d + wrapped-neg backward must not panic");
    let g = grads[0].as_ref().expect("Some(grad)");
    assert_eq!(g.shape(), &[] as &[usize]);
    assert_eq!(
        g.data().unwrap(),
        &[0.0_f32],
        "wrap-filled 0-d must have grad=0"
    );
}
