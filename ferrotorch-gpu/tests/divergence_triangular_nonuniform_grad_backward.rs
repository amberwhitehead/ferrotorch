//! Discriminator re-audit of commit `0dcdc4266` — batched `TriangularBackward`
//! (#1646) under a NON-UNIFORM upstream gradient.
//!
//! The existing `divergence_triangular_batched_autograd.rs` tests drive backward
//! via `sum_all()`, which feeds an all-ones upstream gradient. With an all-ones
//! grad_output every kept position holds the SAME value (1.0), so a backward that
//! used a wrong batch stride, read the wrong matrix, or flipped `k` could still
//! pass (the masked-position values are indistinguishable). These tests inject a
//! DISTINCT-PER-ELEMENT ramp gradient (`1.0, 2.0, ... n.0`) via
//! `backward_with_gradient`, so:
//!   - a wrong batch stride permutes which ramp value lands at each slot,
//!   - a wrong/flipped `k` masks the wrong positions,
//!   - a non-square trailing-dim stride error misplaces values,
//!
//! all become observable failures.
//!
//! VJP (`tools/autograd/derivatives.yaml:1805,1809`):
//!   `tril -> grad.tril_symint(diagonal)`, `triu -> grad.triu_symint(diagonal)`
//! i.e. `x.grad == triu(grad_output, k)` / `tril(grad_output, k)`, the SAME
//! batched mask the forward applies, with `x.grad` keeping the input shape.
//!
//! All expected values are LIVE torch 2.11.0+cu130:
//!   x = arange(n,f64).reshape(shape).requires_grad_()
//!   y = torch.triu(x,k) [or tril]; g = arange(1,n+1,f64).reshape(shape)
//!   y.backward(gradient=g); x.grad.flatten().tolist()
//! Tracking: #1646, #1644.

use ferrotorch_core::{Tensor, TensorStorage, tril, triu};

/// Leaf `0,1,2,...` with `requires_grad = true`.
fn arange_grad_f64(shape: Vec<usize>) -> Tensor<f64> {
    let n: usize = shape.iter().product();
    let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
    Tensor::from_storage(TensorStorage::cpu(data), shape, true).expect("cpu leaf tensor")
}

/// Non-uniform ramp `1.0,2.0,...,n.0` with `requires_grad = false` — used as the
/// injected upstream gradient (grad_output) so every slot carries a distinct value.
fn ramp_f64(shape: Vec<usize>) -> Tensor<f64> {
    let n: usize = shape.iter().product();
    let data: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    Tensor::from_storage(TensorStorage::cpu(data), shape, false).expect("cpu ramp tensor")
}

/// Drives `op(x,k)` forward, injects `ramp` as grad_output, returns `x.grad` data.
fn run(upper: bool, shape: Vec<usize>, k: i64) -> (Vec<usize>, Vec<f64>) {
    let x = arange_grad_f64(shape.clone());
    let y = if upper {
        triu(&x, k).expect("triu forward")
    } else {
        tril(&x, k).expect("tril forward")
    };
    assert_eq!(
        y.shape(),
        &shape[..],
        "forward shape must equal input shape"
    );
    let g = ramp_f64(shape.clone());
    y.backward_with_gradient(&g)
        .expect("backward_with_gradient");
    let grad = x.grad().expect("grad query").expect("x has grad");
    (grad.shape().to_vec(), grad.data().unwrap().to_vec())
}

// --- triu, square trailing [2,3,5], k in {-1,0,1} ---------------------------

#[test]
fn divergence_triu_235_km1_nonuniform() {
    let (sh, grad) = run(true, vec![2, 3, 5], -1);
    assert_eq!(sh, vec![2, 3, 5]);
    let expected: Vec<f64> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
        18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 0.0, 27.0, 28.0, 29.0, 30.0,
    ];
    assert_eq!(grad, expected);
}

#[test]
fn divergence_triu_235_k0_nonuniform() {
    let (sh, grad) = run(true, vec![2, 3, 5], 0);
    assert_eq!(sh, vec![2, 3, 5]);
    let expected: Vec<f64> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0, 10.0, 0.0, 0.0, 13.0, 14.0, 15.0, 16.0, 17.0,
        18.0, 19.0, 20.0, 0.0, 22.0, 23.0, 24.0, 25.0, 0.0, 0.0, 28.0, 29.0, 30.0,
    ];
    assert_eq!(grad, expected);
}

#[test]
fn divergence_triu_235_kp1_nonuniform() {
    let (sh, grad) = run(true, vec![2, 3, 5], 1);
    assert_eq!(sh, vec![2, 3, 5]);
    let expected: Vec<f64> = vec![
        0.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 8.0, 9.0, 10.0, 0.0, 0.0, 0.0, 14.0, 15.0, 0.0, 17.0,
        18.0, 19.0, 20.0, 0.0, 0.0, 23.0, 24.0, 25.0, 0.0, 0.0, 0.0, 29.0, 30.0,
    ];
    assert_eq!(grad, expected);
}

// --- tril, square trailing [2,3,5], k in {-1,0,1} ---------------------------

#[test]
fn divergence_tril_235_km1_nonuniform() {
    let (sh, grad) = run(false, vec![2, 3, 5], -1);
    assert_eq!(sh, vec![2, 3, 5]);
    let expected: Vec<f64> = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 11.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 21.0, 0.0, 0.0, 0.0, 0.0, 26.0, 27.0, 0.0, 0.0, 0.0,
    ];
    assert_eq!(grad, expected);
}

#[test]
fn divergence_tril_235_k0_nonuniform() {
    let (sh, grad) = run(false, vec![2, 3, 5], 0);
    assert_eq!(sh, vec![2, 3, 5]);
    let expected: Vec<f64> = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 6.0, 7.0, 0.0, 0.0, 0.0, 11.0, 12.0, 13.0, 0.0, 0.0, 16.0, 0.0,
        0.0, 0.0, 0.0, 21.0, 22.0, 0.0, 0.0, 0.0, 26.0, 27.0, 28.0, 0.0, 0.0,
    ];
    assert_eq!(grad, expected);
}

#[test]
fn divergence_tril_235_kp1_nonuniform() {
    let (sh, grad) = run(false, vec![2, 3, 5], 1);
    assert_eq!(sh, vec![2, 3, 5]);
    let expected: Vec<f64> = vec![
        1.0, 2.0, 0.0, 0.0, 0.0, 6.0, 7.0, 8.0, 0.0, 0.0, 11.0, 12.0, 13.0, 14.0, 0.0, 16.0, 17.0,
        0.0, 0.0, 0.0, 21.0, 22.0, 23.0, 0.0, 0.0, 26.0, 27.0, 28.0, 29.0, 0.0,
    ];
    assert_eq!(grad, expected);
}

// --- 4-D [2,2,3,3] (two leading batch dims) ---------------------------------

#[test]
fn divergence_triu_2233_k0_nonuniform() {
    let (sh, grad) = run(true, vec![2, 2, 3, 3], 0);
    assert_eq!(sh, vec![2, 2, 3, 3]);
    let expected: Vec<f64> = vec![
        1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0, 10.0, 11.0, 12.0, 0.0, 14.0, 15.0, 0.0, 0.0,
        18.0, 19.0, 20.0, 21.0, 0.0, 23.0, 24.0, 0.0, 0.0, 27.0, 28.0, 29.0, 30.0, 0.0, 32.0, 33.0,
        0.0, 0.0, 36.0,
    ];
    assert_eq!(grad, expected);
}

#[test]
fn divergence_tril_2233_kp1_nonuniform() {
    let (sh, grad) = run(false, vec![2, 2, 3, 3], 1);
    assert_eq!(sh, vec![2, 2, 3, 3]);
    let expected: Vec<f64> = vec![
        1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 0.0, 13.0, 14.0, 15.0, 16.0, 17.0,
        18.0, 19.0, 20.0, 0.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 0.0, 31.0, 32.0,
        33.0, 34.0, 35.0, 36.0,
    ];
    assert_eq!(grad, expected);
}

// --- non-square trailing matrices -------------------------------------------

#[test]
fn divergence_triu_253_kp1_nonuniform() {
    // trailing [5,3]: rows > cols. A stride bug that swaps rows/cols misplaces values.
    let (sh, grad) = run(true, vec![2, 5, 3], 1);
    assert_eq!(sh, vec![2, 5, 3]);
    let expected: Vec<f64> = vec![
        0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 17.0, 18.0,
        0.0, 0.0, 21.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    assert_eq!(grad, expected);
}

#[test]
fn divergence_tril_253_km1_nonuniform() {
    let (sh, grad) = run(false, vec![2, 5, 3], -1);
    assert_eq!(sh, vec![2, 5, 3]);
    let expected: Vec<f64> = vec![
        0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 7.0, 8.0, 0.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 0.0, 0.0,
        0.0, 19.0, 0.0, 0.0, 22.0, 23.0, 0.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
    ];
    assert_eq!(grad, expected);
}

// --- 2-D unregressed ---------------------------------------------------------

#[test]
fn divergence_triu_44_k0_2d_unregressed_nonuniform() {
    let (sh, grad) = run(true, vec![4, 4], 0);
    assert_eq!(sh, vec![4, 4]);
    let expected: Vec<f64> = vec![
        1.0, 2.0, 3.0, 4.0, 0.0, 6.0, 7.0, 8.0, 0.0, 0.0, 11.0, 12.0, 0.0, 0.0, 0.0, 16.0,
    ];
    assert_eq!(grad, expected);
}

#[test]
fn divergence_tril_44_kp1_2d_unregressed_nonuniform() {
    let (sh, grad) = run(false, vec![4, 4], 1);
    assert_eq!(sh, vec![4, 4]);
    let expected: Vec<f64> = vec![
        1.0, 2.0, 0.0, 0.0, 5.0, 6.0, 7.0, 0.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    assert_eq!(grad, expected);
}
