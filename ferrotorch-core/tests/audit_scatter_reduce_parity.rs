//! Focused PyTorch-parity coverage for `scatter_reduce`.
//!
//! Live torch 2.11.0+cu130 oracle highlights:
//! - `src.size(d) >= index.size(d)` is legal and `src` is read by index
//!   coordinates, not as a flat prefix.
//! - `include_self=false` overwrites only touched output slots; untouched
//!   slots keep `self`.
//! - if `src.requires_grad=True` and `src.shape != index.shape`, backward
//!   errors because PyTorch's `grad.gather(dim, index)` VJP is index-shaped.

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
        ScatterReduce::Prod,
        ScatterReduce::Amax,
        ScatterReduce::Amin,
    ] {
        let out = scatter_reduce(&input, 0, &index, &index_shape, &src, reduce, false).unwrap();
        assert_eq!(out.data().unwrap(), &[10.0, 20.0, 3.0, 4.0]);
    }
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
