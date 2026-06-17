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

#[test]
fn scalar_scatter_reduce_matches_torch_effective_one_dim_contract() {
    // Live torch 2.11.0+cu130, scalar self=5, index=[0,0], src=[2,3]:
    // include_self=true  -> sum=10, mean=10/3, prod=30, amax=5, amin=2
    // include_self=false -> sum=5,  mean=2.5,  prod=6,  amax=3, amin=2
    let input = t(&[5.0], &[], false);
    let src = t(&[2.0, 3.0], &[2], false);
    let index = [0, 0];
    let index_shape = [2];

    let cases = [
        (ScatterReduce::Sum, true, 10.0),
        (ScatterReduce::Mean, true, 10.0 / 3.0),
        (ScatterReduce::Prod, true, 30.0),
        (ScatterReduce::Amax, true, 5.0),
        (ScatterReduce::Amin, true, 2.0),
        (ScatterReduce::Sum, false, 5.0),
        (ScatterReduce::Mean, false, 2.5),
        (ScatterReduce::Prod, false, 6.0),
        (ScatterReduce::Amax, false, 3.0),
        (ScatterReduce::Amin, false, 2.0),
    ];
    for (reduce, include_self, expected) in cases {
        let out = scatter_reduce(&input, 0, &index, &index_shape, &src, reduce, include_self)
            .expect("valid scalar scatter_reduce");
        assert_close(out.data().unwrap(), &[expected]);
        assert_eq!(out.shape(), &[] as &[usize]);
    }
}

#[test]
fn scalar_scatter_reduce_backward_matches_torch_value_aware_vjps() {
    // Live torch 2.11.0+cu130, scalar self=5, index=[0,0], src=[2,3],
    // `out.backward()` seed = 1:
    //   sum  include_self T/F: x.grad=1/0,   src.grad=[1,1]
    //   mean include_self T/F: x.grad=1/3/0, src.grad=[1/3,1/3] / [1/2,1/2]
    //   prod include_self T/F: x.grad=6/0,   src.grad=[15,10] / [3,2]
    //   amax include_self T/F: x.grad=1/0,   src.grad=[0,0] / [0,1]
    //   amin include_self T/F: x.grad=0/0,   src.grad=[1,0] / [1,0]
    let cases = [
        (ScatterReduce::Sum, true, 1.0, [1.0, 1.0]),
        (ScatterReduce::Sum, false, 0.0, [1.0, 1.0]),
        (ScatterReduce::Mean, true, 1.0 / 3.0, [1.0 / 3.0, 1.0 / 3.0]),
        (ScatterReduce::Mean, false, 0.0, [0.5, 0.5]),
        (ScatterReduce::Prod, true, 6.0, [15.0, 10.0]),
        (ScatterReduce::Prod, false, 0.0, [3.0, 2.0]),
        (ScatterReduce::Amax, true, 1.0, [0.0, 0.0]),
        (ScatterReduce::Amax, false, 0.0, [0.0, 1.0]),
        (ScatterReduce::Amin, true, 0.0, [1.0, 0.0]),
        (ScatterReduce::Amin, false, 0.0, [1.0, 0.0]),
    ];

    for (reduce, include_self, expected_x_grad, expected_src_grad) in cases {
        let input = t(&[5.0], &[], true);
        let src = t(&[2.0, 3.0], &[2], true);
        let out = scatter_reduce(&input, 0, &[0, 0], &[2], &src, reduce, include_self)
            .expect("valid scalar scatter_reduce forward");
        let seed = t(&[1.0], &[], false);
        backward_with_grad(&out, Some(&seed)).expect("valid scalar scatter_reduce backward");

        assert_close(
            input.grad().unwrap().unwrap().data().unwrap(),
            &[expected_x_grad],
        );
        assert_close(
            src.grad().unwrap().unwrap().data().unwrap(),
            &expected_src_grad,
        );
    }
}

#[test]
fn scalar_scatter_reduce_backward_handles_scalar_source_like_torch() {
    // PyTorch accepts scalar `src` with index shape [1] and returns a scalar
    // source gradient, not an index-shaped [1] gradient.
    let cases = [
        (ScatterReduce::Sum, true, 1.0, 1.0),
        (ScatterReduce::Sum, false, 0.0, 1.0),
        (ScatterReduce::Mean, true, 0.5, 0.5),
        (ScatterReduce::Mean, false, 0.0, 1.0),
        (ScatterReduce::Prod, true, 2.0, 5.0),
        (ScatterReduce::Prod, false, 0.0, 1.0),
        (ScatterReduce::Amax, true, 1.0, 0.0),
        (ScatterReduce::Amax, false, 0.0, 1.0),
        (ScatterReduce::Amin, true, 0.0, 1.0),
        (ScatterReduce::Amin, false, 0.0, 1.0),
    ];

    for (reduce, include_self, expected_x_grad, expected_src_grad) in cases {
        let input = t(&[5.0], &[], true);
        let src = t(&[2.0], &[], true);
        let out = scatter_reduce(&input, 0, &[0], &[1], &src, reduce, include_self)
            .expect("valid scalar-source scatter_reduce forward");
        let seed = t(&[1.0], &[], false);
        backward_with_grad(&out, Some(&seed)).expect("valid scalar-source backward");

        assert_close(
            input.grad().unwrap().unwrap().data().unwrap(),
            &[expected_x_grad],
        );
        let src_grad = src.grad().unwrap().unwrap();
        assert_eq!(src_grad.shape(), &[] as &[usize]);
        assert_close(src_grad.data().unwrap(), &[expected_src_grad]);
    }
}

#[test]
fn scalar_scatter_reduce_backward_rejects_torch_incompatible_src_grad_shapes() {
    let input = t(&[5.0], &[], true);
    let vector_src = t(&[2.0], &[1], true);
    let out = scatter_reduce(&input, 0, &[0], &[], &vector_src, ScatterReduce::Sum, true)
        .expect("PyTorch accepts scalar index with length-one vector src in forward");
    let seed = t(&[1.0], &[], false);
    let err = backward_with_grad(&out, Some(&seed))
        .expect_err("PyTorch rejects the scalar-shaped source gradient at backward time");
    assert!(
        format!("{err:?}").contains("index-shaped"),
        "unexpected backward shape error: {err:?}"
    );

    let input = t(&[5.0], &[], true);
    let rank_mismatched_empty_src = t(&[], &[0], true);
    let out = scatter_reduce(
        &input,
        0,
        &[],
        &[0, 2],
        &rank_mismatched_empty_src,
        ScatterReduce::Sum,
        true,
    )
    .expect("PyTorch skips forward source shape checks for empty index tensors");
    let err = backward_with_grad(&out, Some(&seed))
        .expect_err("PyTorch still rejects empty index-shaped source gradients on mismatch");
    assert!(
        format!("{err:?}").contains("empty scalar-index gradient"),
        "unexpected empty-index backward error: {err:?}"
    );
}

#[test]
fn scalar_scatter_reduce_empty_index_is_noop_after_dim_validation() {
    let input = t(&[5.0], &[], false);
    let src = t(&[], &[0], false);
    let out = scatter_reduce(&input, -1, &[], &[0, 2], &src, ScatterReduce::Sum, false)
        .expect("PyTorch skips scatter shape checks for empty index tensors");
    assert_eq!(out.data().unwrap(), &[5.0]);
    assert_eq!(out.shape(), &[] as &[usize]);

    let dim_err = scatter_reduce(&input, 1, &[], &[0], &src, ScatterReduce::Sum, true)
        .expect_err("PyTorch still validates dim before the empty-index no-op");
    assert!(
        format!("{dim_err:?}").contains("axis") || format!("{dim_err:?}").contains("dim"),
        "unexpected dim error: {dim_err:?}"
    );
}

#[test]
fn scalar_scatter_reduce_rejects_bad_source_shape_without_panicking() {
    let input = t(&[5.0], &[], false);
    let empty_src = t(&[], &[0], false);
    let short_src = t(&[2.0], &[1], false);
    let matrix_src = t(&[2.0], &[1, 1], false);

    assert!(
        scatter_reduce(&input, 0, &[0], &[1], &empty_src, ScatterReduce::Sum, true).is_err(),
        "non-empty scalar scatter_reduce must reject empty source instead of underflowing"
    );
    assert!(
        scatter_reduce(
            &input,
            0,
            &[0, 0],
            &[2],
            &short_src,
            ScatterReduce::Sum,
            true
        )
        .is_err(),
        "index extent cannot exceed scalar-effective src extent"
    );
    assert!(
        scatter_reduce(&input, 0, &[0], &[1], &matrix_src, ScatterReduce::Sum, true).is_err(),
        "non-empty scalar scatter_reduce only accepts effective-rank-one src"
    );
}

#[test]
fn scalar_scatter_reduce_rejects_out_of_bounds_index_without_reading_source() {
    let input = t(&[5.0], &[], false);
    let src = t(&[2.0], &[1], false);
    let err = scatter_reduce(&input, 0, &[1], &[1], &src, ScatterReduce::Sum, true)
        .expect_err("scalar effective dimension has size 1");
    assert!(
        format!("{err:?}").contains("IndexOutOfBounds"),
        "unexpected index error: {err:?}"
    );
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use ferrotorch_core::Device;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for scalar scatter_reduce GPU parity")
        });
    }

    fn cuda(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap()
    }

    #[test]
    fn scalar_scatter_reduce_cuda_uses_resident_effective_1d_path() {
        ensure_cuda_backend();

        let input = cuda(&[5.0], &[]);
        let src = cuda(&[2.0, 3.0], &[2]);
        let out = scatter_reduce(&input, 0, &[0, 0], &[2], &src, ScatterReduce::Sum, true).unwrap();
        assert!(out.is_cuda());
        assert_eq!(out.shape(), &[] as &[usize]);
        assert_eq!(out.data_vec().unwrap(), &[10.0]);

        let scalar_src = cuda(&[7.0], &[]);
        let out = scatter_reduce(
            &input,
            0,
            &[0],
            &[1],
            &scalar_src,
            ScatterReduce::Sum,
            false,
        )
        .unwrap();
        assert!(out.is_cuda());
        assert_eq!(out.shape(), &[] as &[usize]);
        assert_eq!(out.data_vec().unwrap(), &[7.0]);
    }

    #[test]
    fn scalar_scatter_reduce_cuda_backward_stays_resident_and_value_aware() {
        ensure_cuda_backend();

        let input = cuda(&[5.0], &[]).requires_grad_(true);
        let src = cuda(&[2.0, 3.0], &[2]).requires_grad_(true);
        let out =
            scatter_reduce(&input, 0, &[0, 0], &[2], &src, ScatterReduce::Prod, true).unwrap();
        let seed = cuda(&[1.0], &[]);
        backward_with_grad(&out, Some(&seed)).unwrap();

        let input_grad = input.grad().unwrap().unwrap();
        assert!(input_grad.is_cuda());
        assert_eq!(input_grad.shape(), &[] as &[usize]);
        assert_eq!(input_grad.data_vec().unwrap(), &[6.0]);

        let src_grad = src.grad().unwrap().unwrap();
        assert!(src_grad.is_cuda());
        assert_eq!(src_grad.shape(), &[2]);
        assert_eq!(src_grad.data_vec().unwrap(), &[15.0, 10.0]);
    }
}
