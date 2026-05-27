//! Wave-E audit (#1542): #1418 OneHotCategoricalStraightThrough re-export +
//! straight-through gradient path.

#![allow(clippy::approx_constant)]

use ferrotorch_core::autograd::backward;
use ferrotorch_core::{Tensor, TensorStorage};
use ferrotorch_distributions::{Distribution, OneHotCategoricalStraightThrough};

/// Audit: the new symbol is reachable via the crate root re-export.
#[test]
fn audit_1418_ste_reexported_at_crate_root() {
    let probs = Tensor::<f32>::from_storage(
        TensorStorage::cpu(vec![0.25_f32, 0.25, 0.25, 0.25]),
        vec![4],
        false,
    )
    .unwrap();
    let dist = OneHotCategoricalStraightThrough::new(probs).unwrap();
    assert_eq!(dist.num_categories(), 4);
}

/// Audit: `rsample` returns the discrete one-hot sample shape `[..., K]`
/// (same shape as `sample`). Value contract: every drawn row sums to 1 and
/// is one-hot (exactly one 1.0, rest 0.0) — the straight-through forward
/// must NOT alter the value, only the autograd graph.
#[test]
fn audit_1418_ste_rsample_is_discrete_onehot() {
    let probs = Tensor::<f32>::from_storage(
        TensorStorage::cpu(vec![0.1_f32, 0.2, 0.3, 0.4]),
        vec![4],
        false,
    )
    .unwrap();
    let dist = OneHotCategoricalStraightThrough::new(probs).unwrap();
    let r = dist.rsample(&[6]).expect("rsample must succeed");
    assert_eq!(r.shape(), &[6, 4]);
    let data = r.data_vec().unwrap();
    for row in data.chunks(4) {
        let s: f32 = row.iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-5,
            "rsample row must sum to 1 (one-hot), got {row:?}"
        );
        let ones = row.iter().filter(|x| (**x - 1.0).abs() < 1e-5).count();
        let zeros = row.iter().filter(|x| x.abs() < 1e-5).count();
        assert_eq!(ones, 1, "row must have exactly one 1.0: {row:?}");
        assert_eq!(zeros, 3, "row must have exactly three 0.0: {row:?}");
    }
}

/// Audit: the straight-through estimator must actually wire the gradient
/// back to `probs`. After `loss = sum(rsample * weights); loss.backward()`,
/// `probs.grad()` must be Some(non-zero).
#[test]
fn audit_1418_ste_gradient_flows_to_probs() {
    let probs = Tensor::<f32>::from_storage(
        TensorStorage::cpu(vec![0.1_f32, 0.2, 0.3, 0.4]),
        vec![4],
        true, // requires_grad
    )
    .unwrap();
    let dist = OneHotCategoricalStraightThrough::new(probs.clone()).expect("dist construction");

    let r = dist.rsample(&[4]).expect("rsample");
    assert_eq!(r.shape(), &[4, 4]);

    let weights = Tensor::<f32>::from_storage(
        TensorStorage::cpu((0..16).map(|i| (i as f32) * 0.1).collect()),
        vec![4, 4],
        false,
    )
    .unwrap();
    let prod = ferrotorch_core::grad_fns::arithmetic::mul(&r, &weights).expect("mul");
    let loss = ferrotorch_core::grad_fns::reduction::sum(&prod).expect("sum");

    backward(&loss).expect("backward must succeed");

    let g_opt = probs.grad().expect("grad() call must succeed");
    let g =
        g_opt.expect("probs.grad() must be Some after STE backward (straight-through path active)");
    let gd = g.data_vec().unwrap();
    assert_eq!(gd.len(), 4);
    let max_abs = gd.iter().fold(0.0_f32, |m, x| m.max(x.abs()));
    assert!(
        max_abs > 0.0,
        "probs.grad() must have at least one non-zero entry after STE backward, got {gd:?}"
    );
}
