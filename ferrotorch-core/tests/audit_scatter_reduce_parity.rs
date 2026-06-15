//! Focused PyTorch-parity coverage for `scatter_reduce`.
//!
//! Live torch 2.11.0+cu130 oracle highlights:
//! - `src.size(d) >= index.size(d)` is legal and `src` is read by index
//!   coordinates, not as a flat prefix.
//! - `include_self=false` overwrites only touched output slots; untouched
//!   slots keep `self`.
//! - if `src.requires_grad=True` and `src.shape != index.shape`, backward
//!   errors because PyTorch's `grad.gather(dim, index)` VJP is index-shaped.

use ferrotorch_core::autograd::graph::backward_with_grad;
use ferrotorch_core::grad_fns::indexing::{ScatterReduce, scatter_reduce};
use ferrotorch_core::{Tensor, TensorStorage};

fn t(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f64> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .unwrap()
}

fn assert_close(got: &[f64], expected: &[f64]) {
    assert_eq!(got.len(), expected.len());
    for (idx, (&g, &e)) in got.iter().zip(expected).enumerate() {
        assert!(
            (g - e).abs() < 1e-12,
            "mismatch at {idx}: expected {e}, got {g}"
        );
    }
}

#[test]
fn scatter_reduce_larger_src_is_coordinate_addressed() {
    let input = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let src = t(&[10.0, 20.0, 99.0, 40.0, 50.0, 99.0], &[2, 3], false);
    let index = [0, 1, 1, 0];
    let index_shape = [2, 2];

    let out = scatter_reduce(
        &input,
        0,
        &index,
        &index_shape,
        &src,
        ScatterReduce::Sum,
        true,
    )
    .unwrap();
    assert_eq!(out.data().unwrap(), &[11.0, 52.0, 3.0, 44.0, 25.0, 6.0]);
}

#[test]
fn scatter_reduce_include_self_false_keeps_untouched_self_slots() {
    let input = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    let src = t(&[10.0, 20.0], &[1, 2], false);
    let index = [0, 0];
    let index_shape = [1, 2];

    for reduce in [
        ScatterReduce::Sum,
        ScatterReduce::Mean,
        ScatterReduce::Prod,
        ScatterReduce::Amax,
        ScatterReduce::Amin,
    ] {
        let out = scatter_reduce(&input, 0, &index, &index_shape, &src, reduce, false).unwrap();
        assert_eq!(out.data().unwrap(), &[10.0, 20.0, 3.0, 4.0]);
    }
}

#[test]
fn scatter_reduce_mean_public_surface_matches_torch_forward_backward() {
    // Live torch 2.11.0+cu130:
    //   x=[0,2,3,4], s=[6,6,7], idx=[0,0,2], seed=[6,8,10,12]
    //   include_self=True:
    //     out=[4,2,5,4], x.grad=[2,8,5,12], s.grad=[2,2,5]
    //   include_self=False:
    //     out=[6,2,7,4], x.grad=[0,8,0,12], s.grad=[3,3,10]
    // PyTorch implements mean as sum divided by per-destination counts,
    // then zeroes grad_self at index-touched slots for include_self=false
    // (`FunctionsManual.cpp:7249-7255`, `:7274-7275`).
    let input = t(&[0.0, 2.0, 3.0, 4.0], &[4], true);
    let src = t(&[6.0, 6.0, 7.0], &[3], true);
    let out = input
        .scatter_reduce_t(0, &[0, 0, 2], &[3], &src, "mean", true)
        .unwrap();
    assert_eq!(out.data().unwrap(), &[4.0, 2.0, 5.0, 4.0]);
    let seed = t(&[6.0, 8.0, 10.0, 12.0], &[4], false);
    backward_with_grad(&out, Some(&seed)).unwrap();
    assert_eq!(
        input.grad().unwrap().unwrap().data().unwrap(),
        &[2.0, 8.0, 5.0, 12.0]
    );
    assert_eq!(
        src.grad().unwrap().unwrap().data().unwrap(),
        &[2.0, 2.0, 5.0]
    );

    let input = t(&[0.0, 2.0, 3.0, 4.0], &[4], true);
    let src = t(&[6.0, 6.0, 7.0], &[3], true);
    let out = input
        .scatter_reduce_t(0, &[0, 0, 2], &[3], &src, "mean", false)
        .unwrap();
    assert_eq!(out.data().unwrap(), &[6.0, 2.0, 7.0, 4.0]);
    let seed = t(&[6.0, 8.0, 10.0, 12.0], &[4], false);
    backward_with_grad(&out, Some(&seed)).unwrap();
    assert_eq!(
        input.grad().unwrap().unwrap().data().unwrap(),
        &[0.0, 8.0, 0.0, 12.0]
    );
    assert_eq!(
        src.grad().unwrap().unwrap().data().unwrap(),
        &[3.0, 3.0, 10.0]
    );
}

#[test]
fn scatter_reduce_mean_2d_dim1_counts_match_torch() {
    // Live torch 2.11.0+cu130:
    //   x=[[1,2,3],[4,5,6]]
    //   s=[[3,5,7],[9,11,13]]
    //   idx=[[0,0,2],[1,1,1]], dim=1
    //   seed=[[6,8,10],[12,16,18]]
    //   include_self=True:
    //     out=[[3,2,5],[4,9.5,6]]
    //     x.grad=[[2,8,5],[12,4,18]]
    //     s.grad=[[2,2,5],[4,4,4]]
    //   include_self=False:
    //     out=[[4,2,7],[4,11,6]]
    //     x.grad=[[0,8,0],[12,0,18]]
    //     s.grad=[[3,3,10],[16/3,16/3,16/3]]
    let input = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    let src = t(&[3.0, 5.0, 7.0, 9.0, 11.0, 13.0], &[2, 3], true);
    let out = input
        .scatter_reduce_t(1, &[0, 0, 2, 1, 1, 1], &[2, 3], &src, "mean", true)
        .unwrap();
    assert_eq!(out.data().unwrap(), &[3.0, 2.0, 5.0, 4.0, 9.5, 6.0]);
    let seed = t(&[6.0, 8.0, 10.0, 12.0, 16.0, 18.0], &[2, 3], false);
    backward_with_grad(&out, Some(&seed)).unwrap();
    assert_eq!(
        input.grad().unwrap().unwrap().data().unwrap(),
        &[2.0, 8.0, 5.0, 12.0, 4.0, 18.0]
    );
    assert_eq!(
        src.grad().unwrap().unwrap().data().unwrap(),
        &[2.0, 2.0, 5.0, 4.0, 4.0, 4.0]
    );

    let input = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    let src = t(&[3.0, 5.0, 7.0, 9.0, 11.0, 13.0], &[2, 3], true);
    let out = input
        .scatter_reduce_t(1, &[0, 0, 2, 1, 1, 1], &[2, 3], &src, "mean", false)
        .unwrap();
    assert_eq!(out.data().unwrap(), &[4.0, 2.0, 7.0, 4.0, 11.0, 6.0]);
    let seed = t(&[6.0, 8.0, 10.0, 12.0, 16.0, 18.0], &[2, 3], false);
    backward_with_grad(&out, Some(&seed)).unwrap();
    assert_eq!(
        input.grad().unwrap().unwrap().data().unwrap(),
        &[0.0, 8.0, 0.0, 12.0, 0.0, 18.0]
    );
    assert_close(
        src.grad().unwrap().unwrap().data().unwrap(),
        &[3.0, 3.0, 10.0, 16.0 / 3.0, 16.0 / 3.0, 16.0 / 3.0],
    );
}

#[test]
fn scatter_reduce_larger_src_backward_rejects_incompatible_src_grad_shape() {
    let input = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    let src = t(&[10.0, 20.0, 99.0, 40.0, 50.0, 99.0], &[2, 3], true);
    let index = [0, 1, 1, 0];
    let index_shape = [2, 2];
    let out = scatter_reduce(
        &input,
        0,
        &index,
        &index_shape,
        &src,
        ScatterReduce::Sum,
        true,
    )
    .unwrap();
    let go = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let err = out
        .grad_fn()
        .expect("scatter_reduce grad_fn")
        .backward(&go)
        .expect_err("PyTorch rejects index-shaped grad_src for larger src");
    assert!(
        format!("{err:?}").contains("ScatterReduceBackward0")
            || format!("{err:?}").contains("scatter_reduce backward"),
        "expected source-gradient shape contract error, got {err:?}"
    );
}

#[test]
fn scatter_reduce_strict_shape_and_index_validation() {
    let input = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    let src = t(&[10.0, 20.0, 30.0, 40.0], &[2, 2], false);

    assert!(
        scatter_reduce(&input, 0, &[0], &[1, 2], &src, ScatterReduce::Sum, true).is_err(),
        "flat index length must match index_shape product"
    );
    assert!(
        scatter_reduce(
            &input,
            0,
            &[0, 0, 0],
            &[3, 1],
            &src,
            ScatterReduce::Sum,
            true
        )
        .is_err(),
        "non-dim index extent cannot exceed input"
    );
    assert!(
        scatter_reduce(&input, 0, &[2], &[1, 1], &src, ScatterReduce::Sum, true).is_err(),
        "index values must be in bounds along dim"
    );

    let short_src = t(&[10.0, 20.0], &[1, 2], false);
    assert!(
        scatter_reduce(
            &input,
            0,
            &[0, 1, 1, 0],
            &[2, 2],
            &short_src,
            ScatterReduce::Sum,
            true
        )
        .is_err(),
        "index shape cannot exceed src shape on any axis"
    );
}
