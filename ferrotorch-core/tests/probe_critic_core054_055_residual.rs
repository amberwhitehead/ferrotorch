//! Critic re-audit probes for the CORE-054 (#1748) repeat-zero and
//! CORE-055 (#1749) cat staging/device-validation fixes (uncommitted working
//! tree). Goal: surface residual semantic divergence vs PyTorch in the
//! changed regions ONLY (repeat zero branch, cat forward, CatBackward).
//!
//! Oracle (R-ORACLE-1b): live torch 2.11.0+cu130, 2026-06-11. Expected values
//! below are produced by the inlined torch snippets, never copied from the
//! ferrotorch side (R-CHAR-3).
//!
//! ```python
//! import torch
//! # P1 zero-count repeat over a NON-contiguous (transpose) view, forward.
//! x = torch.arange(24.).reshape(4,6)
//! x.t().repeat(0,3).shape                 # torch.Size([0, 12])
//! # P2 zero-axis FOLLOWED by a >=2 axis (cat of zero-numel staged views).
//! torch.arange(6.).reshape(2,3).repeat(0,3).shape   # torch.Size([0, 9])
//! # P3 leading-new-dim + interior zero count.
//! torch.arange(6.).reshape(2,3).repeat(3,0,2).shape # torch.Size([3, 0, 6])
//! # P4 grad of a zero-count repeat over a transpose VIEW reaches the leaf as
//! #    input-shaped zeros (torch RepeatBackward0 + TransposeBackward chain).
//! a = torch.arange(12., requires_grad=True); A = a.reshape(3,4)
//! o = A.t().repeat(0,2)                    # (0,6)
//! o.backward(torch.empty(0,6)); a.grad     # zeros(12)
//! # P5 cat of two views over the SAME storage (transpose), forward.
//! x = torch.arange(24.).reshape(4,6); w = x.t()
//! torch.cat([w, w.narrow(0,1,2)], 0).flatten()
//! # P6 cat empty list errors.
//! ```

use ferrotorch_core::cat;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn plain(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

fn arange(n: usize) -> Vec<f32> {
    (0..n).map(|v| v as f32).collect()
}

#[allow(clippy::float_cmp)]
fn assert_exact(got: &Tensor<f32>, want_shape: &[usize], want: &[f32], label: &str) {
    assert_eq!(got.shape(), want_shape, "{label}: shape");
    let host = got.data_vec().expect("readback");
    assert_eq!(host.len(), want.len(), "{label}: numel");
    for (i, (a, e)) in host.iter().zip(want).enumerate() {
        assert!(
            a == e,
            "{label}[{i}]: got {a}, torch oracle {e} (full: {host:?})"
        );
    }
}

/// P1 — zero-count repeat over a non-contiguous transpose view.
/// torch: x.t().repeat(0,3).shape == (0, 12).
#[test]
fn p1_repeat_zero_over_transpose_view_forward() {
    let x = plain(&arange(24), &[4, 6]);
    let v = x.transpose(0, 1).expect("transpose"); // (6,4) non-contiguous
    assert!(!v.is_contiguous());
    let out = v.repeat_t(&[0, 3]).expect("repeat over view must succeed");
    assert_eq!(out.shape(), &[0, 12], "vt.repeat(0,3) shape");
    assert_eq!(out.numel(), 0);
    assert_eq!(out.data_vec().unwrap(), Vec::<f32>::new());
}

/// P2 — zero axis followed by a >=2 axis: cat of zero-numel staged views.
/// torch: (2,3).repeat(0,3).shape == (0, 9).
#[test]
fn p2_repeat_zero_then_positive_axis() {
    let x = plain(&arange(6), &[2, 3]);
    let out = x.repeat_t(&[0, 3]).expect("repeat([0,3]) must succeed");
    assert_eq!(out.shape(), &[0, 9], "(2,3).repeat(0,3)");
    assert_eq!(out.numel(), 0);
}

/// P3 — leading new dim + interior zero count.
/// torch: (2,3).repeat(3,0,2).shape == (3, 0, 6).
#[test]
fn p3_repeat_leading_newdim_interior_zero() {
    let x = plain(&arange(6), &[2, 3]);
    let out = x
        .repeat_t(&[3, 0, 2])
        .expect("repeat([3,0,2]) must succeed");
    assert_eq!(out.shape(), &[3, 0, 6], "(2,3).repeat(3,0,2)");
    assert_eq!(out.numel(), 0);
}

/// P4 — gradient of a zero-count repeat over a transpose view must reach the
/// original leaf as input-shaped zeros (torch RepeatBackward0 chained through
/// the transpose view's own backward).
#[test]
#[allow(clippy::float_cmp)]
fn p4_repeat_zero_grad_reaches_leaf_through_view() {
    let a = leaf(&arange(12), &[12]);
    let aa = a.reshape_t(&[3, 4]).expect("reshape");
    let vt = aa.transpose(0, 1).expect("t"); // (4,3) non-contiguous
    let o = vt.repeat_t(&[0, 2]).expect("repeat");
    assert_eq!(o.shape(), &[0, 6], "shape");
    assert!(o.requires_grad(), "zero-repeat output must stay tracked");
    let go = plain(&[], &[0, 6]);
    o.backward_with_gradient(&go)
        .expect("backward through zero-count repeat over a view must succeed");
    let g = a.grad().unwrap().expect("grad must reach leaf a");
    assert_eq!(g.shape(), &[12], "a.grad shape");
    assert_exact(&g, &[12], &[0.0; 12], "a.grad");
}

/// P5 — cat of two views over the SAME storage. torch concatenates logical
/// values; CatBackward attaches to the originals.
/// torch: w=x.t() (6,4); cat([w, w.narrow(0,1,2)],0).
#[test]
fn p5_cat_same_storage_views_forward() {
    let x = plain(&arange(24), &[4, 6]);
    let w = x.transpose(0, 1).expect("t"); // (6,4)
    let wn = w.narrow(0, 1, 2).expect("narrow"); // (2,4)
    let out = cat(&[w.clone(), wn], 0).expect("cat of same-storage views must succeed");
    let want: Vec<f32> = {
        // w.flatten() = column-major read of x: w[r][c] = x[c][r] = c*6 + r
        let wflat: Vec<f32> = (0..6)
            .flat_map(|r| (0..4).map(move |c| (c * 6 + r) as f32))
            .collect();
        // w.narrow(0,1,2) = rows 1,2 of w => wflat[4..12]
        let mut v = wflat.clone();
        v.extend_from_slice(&wflat[4..12]);
        v
    };
    assert_exact(&out, &[8, 4], &want, "cat([w, w.narrow],0)");
}

/// P6 — cat empty tensor list must error (torch: RuntimeError).
#[test]
fn p6_cat_empty_list_errors() {
    let empty: Vec<Tensor<f32>> = vec![];
    assert!(cat(&empty, 0).is_err(), "cat([]) must error");
}

/// P7 — cat of two DISJOINT non-contiguous-offset views of the SAME tracked
/// leaf; gradient must scatter back into the right rows of the leaf.
/// torch (see header): a.grad.reshape(4,6) == arange(24)+1 reshaped.
#[test]
fn p7_cat_disjoint_views_of_same_leaf_grad() {
    let a = leaf(&arange(24), &[24]);
    let aa = a.reshape_t(&[4, 6]).expect("reshape");
    let v1 = aa.narrow(0, 0, 2).expect("v1"); // rows 0,1 (offset 0)
    let v2 = aa.narrow(0, 2, 2).expect("v2"); // rows 2,3 (offset 12)
    let out = cat(&[v1, v2], 0).expect("cat disjoint views");
    assert_eq!(out.shape(), &[4, 6]);
    let go = plain(&(1..=24).map(|v| v as f32).collect::<Vec<_>>(), &[4, 6]);
    out.backward_with_gradient(&go).expect("backward");
    let g = a.grad().unwrap().expect("grad to leaf");
    let want: Vec<f32> = (1..=24).map(|v| v as f32).collect();
    assert_exact(&g, &[24], &want, "disjoint a.grad");
}

/// P8 — cat of two OVERLAPPING views of the SAME tracked leaf. torch
/// ACCUMULATES the gradient where the views overlap (row 1).
/// torch (see python): b.grad.reshape(4,6) ==
///   [[1..6],[20,22,24,26,28,30],[19..24],[0,0,0,0,0,0]].
#[test]
fn p8_cat_overlapping_views_of_same_leaf_grad_accumulates() {
    let b = leaf(&arange(24), &[24]);
    let bb = b.reshape_t(&[4, 6]).expect("reshape");
    let o1 = bb.narrow(0, 0, 2).expect("o1"); // rows 0,1
    let o2 = bb.narrow(0, 1, 2).expect("o2"); // rows 1,2 (OVERLAP row 1)
    let out = cat(&[o1, o2], 0).expect("cat overlapping views");
    assert_eq!(out.shape(), &[4, 6]);
    let go = plain(&(1..=24).map(|v| v as f32).collect::<Vec<_>>(), &[4, 6]);
    out.backward_with_gradient(&go).expect("backward");
    let g = b.grad().unwrap().expect("grad to leaf");
    let want: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // row 0
        20.0, 22.0, 24.0, 26.0, 28.0, 30.0, // row 1 = o1.row1 (7..12) + o2.row0 (13..18)
        19.0, 20.0, 21.0, 22.0, 23.0, 24.0, // row 2 = o2.row1
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // row 3 (untouched)
    ];
    assert_exact(&g, &[24], &want, "overlap b.grad");
}
