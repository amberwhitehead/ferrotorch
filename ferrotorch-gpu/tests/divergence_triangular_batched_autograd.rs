//! Discriminator re-audit of commit `67057b8f6` — batched N-D triu/tril (#1644).
//!
//! DIVERGENCE: #1644 fixed the FORWARD of `triu`/`tril` to batch the mask over
//! the last two dims of an N-D tensor, but left the AUTOGRAD BACKWARD 2-D-only.
//!
//! When grad is tracked, `crate::ops::tensor_ops::triu` delegates to
//! `crate::grad_fns::linalg::triu_differentiable`, which builds a
//! `TriangularBackward { rows: shape[0], cols: shape[1], .. }`
//! (`ferrotorch-core/src/grad_fns/linalg.rs:5410-5411` for tril,
//! `:5435-5436` for triu). `TriangularBackward::backward`
//! (`linalg.rs:5348-5365`) then allocates `vec![zero; rows*cols]`, walks only
//! `rows*cols` slots, and returns a gradient of shape `vec![rows, cols]`.
//!
//! For a batched input shaped `[2, 3, 5]`, this captures `rows = 2`, `cols = 3`
//! and produces a `[2, 3]` gradient (6 elements) instead of torch's `[2, 3, 5]`
//! gradient (30 elements), masked per trailing matrix.
//!
//! Upstream gradient (LIVE torch 2.11.0+cu130):
//!   x = arange(30,f64).reshape(2,3,5).requires_grad_(); torch.triu(x,0).sum().backward()
//!   x.grad.shape == [2,3,5]; x.grad == triu-mask-per-matrix of ones.
//! Per `tools/autograd/derivatives.yaml:1809` `triu -> grad.triu_symint(diagonal)`
//! (and `:1805` `tril -> grad.tril_symint`) the VJP is `grad.triu(k)` — i.e.
//! the SAME batched mask the forward applies, with the gradient KEEPING the
//! input shape.
//!
//! These tests are CPU-only (autograd is a CPU path) and FAIL against commit
//! 67057b8f6 (gradient shape `[2,3]` instead of `[2,3,5]`).
//! Tracking: #1646 (blocker).

use ferrotorch_core::{Tensor, TensorStorage, tril, triu};

/// Leaf tensor `0,1,2,...` with `requires_grad = true`.
fn arange_grad_f64(shape: Vec<usize>) -> Tensor<f64> {
    let n: usize = shape.iter().product();
    let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
    Tensor::from_storage(TensorStorage::cpu(data), shape, true).expect("cpu leaf tensor")
}

/// Divergence: `triu` backward on a batched `[2,3,5]` input produces a `[2,3]`
/// gradient (`rows=shape[0]=2, cols=shape[1]=3`) instead of torch's `[2,3,5]`.
/// Upstream `x.grad` (LIVE torch) is the per-matrix triu mask of ones.
/// Tracking: #1646
#[test]
fn divergence_triu_batched_3d_backward_shape_and_values() {
    let x = arange_grad_f64(vec![2, 3, 5]);
    let y = triu(&x, 0).expect("triu forward");
    // forward shape is correct (the #1644 fix); the backward is the divergence.
    assert_eq!(y.shape(), &[2, 3, 5], "forward shape");

    let loss = y.sum_all().expect("sum_all");
    loss.backward().expect("backward");

    let grad = x.grad().expect("grad query").expect("x has grad");
    assert_eq!(
        grad.shape(),
        &[2, 3, 5],
        "torch x.grad.shape == [2,3,5]; ferrotorch must match the input shape"
    );

    // LIVE torch: torch.triu(arange(30,f64).reshape(2,3,5),0).sum().backward() -> x.grad
    let expected: Vec<f64> = vec![
        1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
    ];
    assert_eq!(grad.data().unwrap().to_vec(), expected);
}

/// Divergence: `tril` backward on a batched `[2,3,5]` input.
/// Tracking: #1646
#[test]
fn divergence_tril_batched_3d_backward_shape_and_values() {
    let x = arange_grad_f64(vec![2, 3, 5]);
    let y = tril(&x, 0).expect("tril forward");
    assert_eq!(y.shape(), &[2, 3, 5], "forward shape");

    let loss = y.sum_all().expect("sum_all");
    loss.backward().expect("backward");

    let grad = x.grad().expect("grad query").expect("x has grad");
    assert_eq!(grad.shape(), &[2, 3, 5], "torch x.grad.shape == [2,3,5]");

    // LIVE torch: torch.tril(arange(30,f64).reshape(2,3,5),0).sum().backward() -> x.grad
    let expected: Vec<f64> = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0,
    ];
    assert_eq!(grad.data().unwrap().to_vec(), expected);
}
