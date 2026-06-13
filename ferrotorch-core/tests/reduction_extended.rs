//! Library-level tests for the reduction-cluster impls landed in the
//! grad_fns/reduction.rs build that closes the umbrella runner-arm
//! issue (#1314) + per-op blockers (#1301 std/var, #1304 argmax/argmin,
//! #1310 logsumexp autograd, #1312 any/all/count_nonzero).
//!
//! These tests pin: forward shape + value parity to hand-computed
//! upstream-aligned references, backward gradients against the closed-
//! form VJPs in `tools/autograd/derivatives.yaml:1052-1054` (logsumexp),
//! `:1673-1676` (std), `:1924-1925` (var). The non-differentiable arms
//! (argmax/argmin/any/all/count_nonzero) only assert forward correctness
//! since they carry no `*Backward` node.

use ferrotorch_core::Tensor;
use ferrotorch_core::grad_fns::reduction::{
    all as red_all, any as red_any, argmax, argmax_dim, argmin, argmin_dim, count_nonzero,
    logsumexp, logsumexp_dim, logsumexp_dims, mean_dims, std as red_std, std_dims, sum_dims,
    var as red_var, var_dims,
};
use ferrotorch_core::storage::TensorStorage;

fn leaf(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f64> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("leaf construction")
}

// ----- logsumexp -----

#[test]
fn logsumexp_forward_1d_matches_handcomputed() {
    let x = leaf(&[1.0, 2.0, 3.0], &[3], false);
    let r = logsumexp(&x).expect("logsumexp forward");
    let v = r.item().expect("scalar item");
    let expected = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();
    assert!((v - expected).abs() < 1e-10, "got {v} expected {expected}");
}

#[test]
fn logsumexp_backward_softmax_routing_sums_to_one() {
    // VJP `grad * exp(input - result)` is the softmax (sums to 1 with grad=1).
    let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
    let r = logsumexp(&x).expect("logsumexp forward");
    r.backward().expect("backward");
    let g = x.grad().expect("grad query").expect("grad set");
    let gd = g.data().expect("grad data");
    let s: f64 = gd.iter().sum();
    assert!((s - 1.0).abs() < 1e-10, "softmax should sum to 1, got {s}");
    assert!(gd[0] < gd[1] && gd[1] < gd[2]);
}

#[test]
fn logsumexp_dim_forward_per_row() {
    let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let r = logsumexp_dim(&x, 1, false).expect("logsumexp_dim forward");
    assert_eq!(r.shape(), &[2]);
    let rd = r.data().expect("data");
    let e0 = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();
    let e1 = (4.0_f64.exp() + 5.0_f64.exp() + 6.0_f64.exp()).ln();
    assert!((rd[0] - e0).abs() < 1e-10);
    assert!((rd[1] - e1).abs() < 1e-10);
}

#[test]
fn logsumexp_dim_keepdim_preserves_axis() {
    let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let r = logsumexp_dim(&x, 1, true).expect("forward");
    assert_eq!(r.shape(), &[2, 1]);
}

#[test]
fn logsumexp_dim_negative_dim_normalizes() {
    let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let r0 = logsumexp_dim(&x, 1, false).expect("dim=1");
    let r1 = logsumexp_dim(&x, -1, false).expect("dim=-1");
    let d0 = r0.data().expect("d0");
    let d1 = r1.data().expect("d1");
    for (a, b) in d0.iter().zip(d1.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn logsumexp_dim_backward_softmax_sums_to_one_per_row() {
    // Per-row VJP is per-row softmax — each row sums to 1.
    let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    let r = logsumexp_dim(&x, 1, false).expect("forward");
    // upstream grad = ones([2])
    let go =
        Tensor::from_storage(TensorStorage::cpu(vec![1.0_f64, 1.0]), vec![2], false).expect("go");
    r.backward_with_gradient(&go)
        .expect("backward_with_gradient");
    let g = x.grad().expect("grad query").expect("grad set");
    let gd = g.data().expect("grad data");
    let row0: f64 = gd[0..3].iter().sum();
    let row1: f64 = gd[3..6].iter().sum();
    assert!((row0 - 1.0).abs() < 1e-10);
    assert!((row1 - 1.0).abs() < 1e-10);
}

// ----- argmax / argmin -----

#[test]
fn argmax_full_flat() {
    let x = leaf(&[1.0, 5.0, 3.0, 2.0], &[4], false);
    let r = argmax(&x).expect("argmax");
    assert_eq!(r.data().expect("data"), &[1]);
    assert_eq!(r.shape(), &[] as &[usize]);
}

#[test]
fn argmin_full_flat() {
    let x = leaf(&[5.0, 1.0, 3.0, 0.5], &[4], false);
    let r = argmin(&x).expect("argmin");
    assert_eq!(r.data().expect("data"), &[3]);
}

#[test]
fn argmax_dim_2d() {
    // [[1, 5, 3], [4, 2, 6]] argmax dim=1 -> [1, 2]
    let x = leaf(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3], false);
    let r = argmax_dim(&x, 1, false).expect("argmax_dim");
    assert_eq!(r.shape(), &[2]);
    assert_eq!(r.data().expect("data"), &[1, 2]);
}

#[test]
fn argmin_dim_2d_keepdim() {
    let x = leaf(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3], false);
    let r = argmin_dim(&x, 0, true).expect("argmin_dim");
    assert_eq!(r.shape(), &[1, 3]);
    assert_eq!(r.data().expect("data"), &[0, 1, 0]);
}

#[test]
fn argmax_negative_dim() {
    let x = leaf(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3], false);
    let r = argmax_dim(&x, -1, false).expect("argmax_dim");
    assert_eq!(r.data().expect("data"), &[1, 2]);
}

#[test]
fn argmax_scalar_input_returns_zero() {
    // Upstream `:1789-1792 fill_(0)` for sizes[dim]==1.
    let x = leaf(&[7.0], &[], false);
    let r = argmax_dim(&x, 0, false).expect("argmax_dim scalar");
    assert_eq!(r.data().expect("data"), &[0]);
}

// ----- std / var -----

#[test]
fn var_unbiased_handcomputed() {
    // var([1,2,3,4], unbiased=true) = sum((x-2.5)^2)/3 = (2.25+0.25+0.25+2.25)/3 = 5/3
    let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], false);
    let r = red_var(&x, true).expect("var");
    let v = r.item().expect("item");
    let expected = 5.0 / 3.0;
    assert!((v - expected).abs() < 1e-10, "got {v}");
}

#[test]
fn var_biased_handcomputed() {
    let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], false);
    let r = red_var(&x, false).expect("var");
    let v = r.item().expect("item");
    let expected = 5.0 / 4.0;
    assert!((v - expected).abs() < 1e-10, "got {v}");
}

#[test]
fn std_matches_sqrt_var() {
    let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5], false);
    let v = red_var(&x, true).expect("var").item().expect("v");
    let s = red_std(&x, true).expect("std").item().expect("s");
    assert!((s - v.sqrt()).abs() < 1e-12);
}

#[test]
fn var_backward_handcomputed() {
    // d(var)/d(x_i) = 2*(x_i - mean)/denom for full reduction.
    let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], true);
    let r = red_var(&x, true).expect("var");
    r.backward().expect("backward");
    let g = x.grad().expect("grad query").expect("grad");
    let gd = g.data().expect("grad data");
    let scale = 2.0 / 3.0;
    let expected = [-1.5 * scale, -0.5 * scale, 0.5 * scale, 1.5 * scale];
    for (a, b) in gd.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-10, "got {a} expected {b}");
    }
}

#[test]
fn std_backward_handcomputed() {
    // d(std)/d(x_i) = (x_i - mean) / (denom * std)
    let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], true);
    let r = red_std(&x, true).expect("std");
    r.backward().expect("backward");
    let g = x.grad().expect("grad query").expect("grad");
    let s = (5.0_f64 / 3.0).sqrt();
    let scale = 1.0 / (3.0 * s);
    let expected = [-1.5 * scale, -0.5 * scale, 0.5 * scale, 1.5 * scale];
    let gd = g.data().expect("grad data");
    for (a, b) in gd.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-10, "got {a} expected {b}");
    }
}

#[test]
fn sum_mean_dims_reduce_non_adjacent_axes_and_keepdim() {
    let data: Vec<f64> = (0..24).map(|v| v as f64).collect();
    let x = leaf(&data, &[2, 3, 4], false);

    let s = sum_dims(&x, &[0, -1], false).expect("sum_dims");
    assert_eq!(s.shape(), &[3]);
    assert_eq!(s.data().expect("sum data"), &[60.0, 92.0, 124.0]);

    let m = mean_dims(&x, &[0, 2], true).expect("mean_dims keepdim");
    assert_eq!(m.shape(), &[1, 3, 1]);
    assert_eq!(m.data().expect("mean data"), &[7.5, 11.5, 15.5]);
}

#[test]
fn std_var_dims_reduce_combined_axis_not_chained_std() {
    let data: Vec<f64> = (0..24).map(|v| v as f64).collect();
    let x = leaf(&data, &[2, 3, 4], true);

    let v = var_dims(&x, &[1, 2], 1.0, false).expect("var_dims");
    assert_eq!(v.shape(), &[2]);
    assert_eq!(v.data().expect("var data"), &[13.0, 13.0]);

    let s = std_dims(&x, &[1, 2], 1.0, true).expect("std_dims");
    assert_eq!(s.shape(), &[2, 1, 1]);
    let sd = s.data().expect("std data");
    let expected = 13.0_f64.sqrt();
    assert!((sd[0] - expected).abs() < 1e-12);
    assert!((sd[1] - expected).abs() < 1e-12);

    s.sum_all().expect("sum").backward().expect("backward");
    let grad = x.grad().expect("grad access").expect("grad");
    assert_eq!(grad.shape(), &[2, 3, 4]);
    let gd = grad.data().expect("grad data");
    let denom_std = 11.0 * expected;
    assert!((gd[0] - ((0.0 - 5.5) / denom_std)).abs() < 1e-12);
    assert!((gd[11] - ((11.0 - 5.5) / denom_std)).abs() < 1e-12);
    assert!((gd[12] - ((12.0 - 17.5) / denom_std)).abs() < 1e-12);
    assert!((gd[23] - ((23.0 - 17.5) / denom_std)).abs() < 1e-12);
}

#[test]
fn reduction_dims_duplicate_rejected_and_empty_reduces_all() {
    let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    let err = sum_dims(&x, &[1, -1], false).expect_err("duplicate dims must error");
    assert!(format!("{err}").contains("appears multiple times"));

    let s = sum_dims(&x, &[], true).expect("empty dims reduce all");
    assert_eq!(s.shape(), &[1, 1]);
    assert_eq!(s.data().expect("data"), &[10.0]);

    let l = logsumexp_dims(&x, &[], true).expect("logsumexp empty dims keepdim");
    assert_eq!(l.shape(), &[1, 1]);
    let expected = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp() + 4.0_f64.exp()).ln();
    assert!((l.data().expect("data")[0] - expected).abs() < 1e-12);
}

// ----- any / all / count_nonzero -----

#[test]
fn any_all_count_nonzero_mixed() {
    let x_mixed = leaf(&[0.0, 1.0, 0.0, 2.0], &[4], false);
    let x_all_zero = leaf(&[0.0, 0.0, 0.0], &[3], false);
    let x_all_nz = leaf(&[1.0, 2.0, 3.0], &[3], false);

    assert!(red_any(&x_mixed).expect("any").data().expect("d")[0]);
    assert!(!red_any(&x_all_zero).expect("any").data().expect("d")[0]);
    assert!(red_any(&x_all_nz).expect("any").data().expect("d")[0]);

    assert!(!red_all(&x_mixed).expect("all").data().expect("d")[0]);
    assert!(!red_all(&x_all_zero).expect("all").data().expect("d")[0]);
    assert!(red_all(&x_all_nz).expect("all").data().expect("d")[0]);

    assert_eq!(
        count_nonzero(&x_mixed).expect("cnz").data().expect("d"),
        &[2]
    );
    assert_eq!(
        count_nonzero(&x_all_zero).expect("cnz").data().expect("d"),
        &[0]
    );
    assert_eq!(
        count_nonzero(&x_all_nz).expect("cnz").data().expect("d"),
        &[3]
    );
}

#[test]
fn count_nonzero_nan_is_nonzero() {
    // NaN != 0.0 in IEEE-754 → counts as non-zero (matches upstream
    // `at::native::nonzero_count` predicate).
    let x = leaf(&[0.0, f64::NAN, 1.0, 0.0], &[4], false);
    let c = count_nonzero(&x).expect("cnz");
    assert_eq!(c.data().expect("d"), &[2]);
}
