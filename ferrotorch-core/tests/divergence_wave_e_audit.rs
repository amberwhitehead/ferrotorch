//! Wave-E audit (uncommitted work from 2 rate-limited builders, issue #1542).
//!
//! Probes every claimed blocker in the production diff:
//!
//! - #1212 mul_ broadcast
//! - #1213 div_ broadcast + rounding_mode (`div_rounding_`)
//! - #1214 clamp_ Option-bounds (`clamp_opt_`) + NaN
//! - #1256 0-d input allow in `gather`
//!
//! Each probe is a host-side test asserting upstream PyTorch behaviour.
//! Tests that PASS confirm the wiring is genuine; tests that FAIL pin the
//! divergence the builder did not actually fix.

#![allow(clippy::approx_constant)]

use ferrotorch_core::{Tensor, TensorStorage, gather};

fn t<T: ferrotorch_core::Float>(data: Vec<T>, shape: Vec<usize>) -> Tensor<T> {
    Tensor::from_storage(TensorStorage::cpu(data), shape, false).expect("cpu storage construction")
}

// ---------------------------------------------------------------------------
// #1212 — mul_ broadcast (scalar shape `[1]` broadcast to `[2, 2]`).
// Upstream: `torch.Tensor.mul_(other)` accepts broadcastable `other`. See
// `aten/src/ATen/native/BinaryOps.cpp:441 TORCH_IMPL_FUNC(mul_out)` which
// inherits `TensorIterator` broadcasting.
// ---------------------------------------------------------------------------
#[test]
fn audit_1212_mul_inplace_broadcasts_scalar_shape_1() {
    let a = t::<f32>(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2]);
    let b = t::<f32>(vec![10.0], vec![1]);
    a.mul_(&b)
        .expect("mul_ must accept broadcast `[1]` against `[2,2]`");
    let got = a.data_vec().unwrap();
    assert_eq!(got, vec![20.0, 40.0, 60.0, 80.0]);
}

#[test]
fn audit_1212_mul_inplace_broadcasts_row_to_matrix() {
    // shape `[2]` broadcast across rows of `[2, 2]`
    let a = t::<f32>(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = t::<f32>(vec![10.0, 100.0], vec![2]);
    a.mul_(&b)
        .expect("mul_ must accept broadcast `[2]` against `[2,2]`");
    let got = a.data_vec().unwrap();
    // a * b broadcasts b over rows: [[1*10, 2*100], [3*10, 4*100]]
    assert_eq!(got, vec![10.0, 200.0, 30.0, 400.0]);
}

#[test]
fn audit_1212_mul_inplace_rejects_non_broadcastable() {
    // shape `[2,2]` mul_ `[3]` is NOT broadcast-compatible (PyTorch rejects).
    let a = t::<f32>(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = t::<f32>(vec![1.0, 2.0, 3.0], vec![3]);
    assert!(a.mul_(&b).is_err(), "non-broadcastable must error");
}

#[test]
fn audit_1212_mul_inplace_rejects_resizing_broadcast() {
    // shape `[1]` self mul_ `[2,2]` would broadcast `self` to `[2,2]` — but
    // in-place ops cannot resize the target tensor (PyTorch invariant).
    let a = t::<f32>(vec![5.0], vec![1]);
    let b = t::<f32>(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    assert!(
        a.mul_(&b).is_err(),
        "in-place mul_ cannot resize self via broadcast"
    );
}

// ---------------------------------------------------------------------------
// #1213 — div_ broadcast + `div_rounding_` modes.
// ---------------------------------------------------------------------------
#[test]
fn audit_1213_div_inplace_broadcasts_scalar() {
    let a = t::<f32>(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
    let b = t::<f32>(vec![10.0], vec![1]);
    a.div_(&b)
        .expect("div_ must accept broadcast `[1]` against `[2,2]`");
    let got = a.data_vec().unwrap();
    assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn audit_1213_div_rounding_floor() {
    // div_rounding_(self, other, "floor"):
    //   floor(7 / 2) = 3, floor(-7 / 2) = -4
    let a = t::<f32>(vec![7.0, -7.0], vec![2]);
    let b = t::<f32>(vec![2.0, 2.0], vec![2]);
    a.div_rounding_(&b, "floor")
        .expect("div_rounding_(\"floor\") must succeed");
    let got = a.data_vec().unwrap();
    assert_eq!(got, vec![3.0, -4.0]);
}

#[test]
fn audit_1213_div_rounding_trunc() {
    // div_rounding_(self, other, "trunc"):
    //   trunc(7 / 2) = 3, trunc(-7 / 2) = -3  (rounds toward zero)
    let a = t::<f32>(vec![7.0, -7.0], vec![2]);
    let b = t::<f32>(vec![2.0, 2.0], vec![2]);
    a.div_rounding_(&b, "trunc")
        .expect("div_rounding_(\"trunc\") must succeed");
    let got = a.data_vec().unwrap();
    assert_eq!(got, vec![3.0, -3.0]);
}

#[test]
fn audit_1213_div_rounding_unknown_mode_errors() {
    let a = t::<f32>(vec![1.0], vec![1]);
    let b = t::<f32>(vec![1.0], vec![1]);
    assert!(
        a.div_rounding_(&b, "round").is_err(),
        "unknown rounding mode must error (upstream `BinaryOps.cpp:186`)"
    );
}

// ---------------------------------------------------------------------------
// #1214 — clamp_opt_ Option bounds + NaN-bound parity.
// Upstream: `torch.Tensor.clamp_(min=None, max=None)` per
// `torch/_tensor_docs.py:1141` + `TORCH_IMPL_FUNC(clamp_out)` at
// `aten/src/ATen/native/TensorCompare.cpp:831`.
// ---------------------------------------------------------------------------
#[test]
fn audit_1214_clamp_opt_lower_only() {
    let a = t::<f32>(vec![-1.0, 0.5, 2.0], vec![3]);
    a.clamp_opt_(Some(0.0_f32), None)
        .expect("clamp_opt_(Some, None) must succeed");
    assert_eq!(a.data_vec().unwrap(), vec![0.0, 0.5, 2.0]);
}

#[test]
fn audit_1214_clamp_opt_upper_only() {
    let a = t::<f32>(vec![-1.0, 0.5, 2.0], vec![3]);
    a.clamp_opt_(None, Some(1.0_f32))
        .expect("clamp_opt_(None, Some) must succeed");
    assert_eq!(a.data_vec().unwrap(), vec![-1.0, 0.5, 1.0]);
}

#[test]
fn audit_1214_clamp_opt_both_none_rejected() {
    let a = t::<f32>(vec![1.0], vec![1]);
    assert!(
        a.clamp_opt_(None, None).is_err(),
        "clamp_opt_(None, None) must error (upstream `TensorCompare.cpp:106`)"
    );
}

#[test]
fn audit_1214_clamp_opt_nan_bound_fills_with_nan() {
    // Per `TensorCompare.cpp:844`: if either supplied bound is NaN, the
    // entire output is filled with NaN.
    let a = t::<f32>(vec![1.0, 2.0, 3.0], vec![3]);
    a.clamp_opt_(Some(f32::NAN), Some(10.0))
        .expect("clamp_opt_ with NaN-bound must not error");
    let got = a.data_vec().unwrap();
    assert!(
        got.iter().all(|x| x.is_nan()),
        "all outputs must be NaN, got {got:?}"
    );
}

#[test]
fn audit_1214_clamp_opt_min_greater_than_max_rejected() {
    let a = t::<f32>(vec![1.0], vec![1]);
    assert!(
        a.clamp_opt_(Some(5.0_f32), Some(1.0_f32)).is_err(),
        "clamp_opt_ must reject min > max"
    );
}

// ---------------------------------------------------------------------------
// #1256 — 0-D input to `gather`.
// Upstream: `aten/src/ATen/native/ScatterGatherChecks.h:44`
// `ensure_nonempty_dim` treats 0-D self as effective ndim=1 with size 1.
// ---------------------------------------------------------------------------
#[test]
fn audit_1256_gather_0d_input_scalar_index() {
    // Per the user's brief: 0-d input + 0-d index, single element.
    let input = t::<f32>(vec![5.0], vec![]);
    // 0-d index_shape (`&[]`), one index value `[0]`.
    let result =
        gather(&input, 0, &[0_usize], &[]).expect("gather must accept 0-D input + 0-D index");
    assert_eq!(result.shape(), &[] as &[usize]);
    assert_eq!(result.data_vec().unwrap(), vec![5.0]);
}

#[test]
fn audit_1256_gather_0d_input_1d_index() {
    // 0-d input, 1-d index of length 3 — upstream behaviour: result shape ==
    // index shape, every element pulls from the lone scalar.
    let input = t::<f32>(vec![5.0], vec![]);
    // With effective_input_shape=[1] and index_shape=[3], validate_gather_shapes
    // requires `input_ndim == index_ndim` → 1 == 1 (OK).
    let result =
        gather(&input, 0, &[0_usize, 0, 0], &[3]).expect("gather 0-D input + 1-D index must work");
    assert_eq!(result.shape(), &[3]);
    assert_eq!(result.data_vec().unwrap(), vec![5.0, 5.0, 5.0]);
}
