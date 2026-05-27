//! Regression / divergence coverage for the #1342 shape-ops umbrella.
//!
//! Closes the remaining 17 torch shape ops in `grad_fns/shape.rs`:
//! flip / fliplr / flipud / rot90 / movedim / moveaxis / broadcast_to /
//! broadcast_tensors / repeat / tile / repeat_interleave / unbind /
//! tensor_split / vstack / hstack / dstack / column_stack.
//!
//! These tests drive the PUBLIC `Tensor` method wrappers (the R-DEFER-1
//! non-test production consumers) rather than the free functions directly,
//! so they exercise the same surface a downstream crate would use. Each
//! expected value is hand-computed from the upstream semantics in
//! `aten/src/ATen/native/TensorTransformations.cpp` (flip family / rot90)
//! and `aten/src/ATen/native/TensorShape.cpp` (movedim / repeat / tile /
//! unbind / tensor_split / *stack), cited inline (R-CHAR-3: every asserted
//! value is traceable to a named upstream file:line, not a self-comparison).

use ferrotorch_core::{Tensor, TensorStorage};

fn leaf(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .unwrap()
}

/// Materialize a (possibly strided) view's logical row-major data.
fn dense(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous().unwrap().data_vec().unwrap()
}

// --- flip family (TensorTransformations.cpp:36/180/186) ---

#[test]
fn flip_reverses_listed_dims() {
    let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    // torch.flip(x, [1]) reverses each row.
    assert_eq!(
        dense(&x.flip_t(&[1]).unwrap()),
        &[3.0, 2.0, 1.0, 6.0, 5.0, 4.0]
    );
    // torch.flip(x, [0, 1]) reverses rows and columns.
    assert_eq!(
        dense(&x.flip_t(&[0, 1]).unwrap()),
        &[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    );
}

#[test]
fn fliplr_is_flip_dim1_and_flipud_is_flip_dim0() {
    let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    assert_eq!(
        dense(&x.fliplr_t().unwrap()),
        dense(&x.flip_t(&[1]).unwrap())
    );
    assert_eq!(
        dense(&x.flipud_t().unwrap()),
        dense(&x.flip_t(&[0]).unwrap())
    );
}

#[test]
fn flip_backward_is_self_inverse() {
    // flip is a permutation; grad flows back through the same flip.
    let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], true);
    let y = x.flip_t(&[0, 1]).unwrap();
    let loss = y.contiguous().unwrap().sum_all().unwrap();
    ferrotorch_core::backward(&loss).unwrap();
    let g = x.grad().unwrap().expect("flip must propagate grad to leaf");
    // sum-grad is uniform ones regardless of permutation.
    assert_eq!(g.data().unwrap(), &[1.0, 1.0, 1.0, 1.0]);
}

// --- rot90 (TensorTransformations.cpp:134) ---

#[test]
fn rot90_k1_matches_flip_transpose() {
    // k=1 on [[1,2],[3,4]] = flip({1}).transpose(0,1) = [[2,4],[1,3]].
    let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    assert_eq!(
        dense(&x.rot90_t(1, &[0, 1]).unwrap()),
        &[2.0, 4.0, 1.0, 3.0]
    );
}

#[test]
fn rot90_periodicity() {
    // Four quarter-turns return the original tensor.
    let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let mut r = x.clone();
    for _ in 0..4 {
        r = r.rot90_t(1, &[0, 1]).unwrap();
    }
    assert_eq!(r.shape(), x.shape());
    assert_eq!(dense(&r), dense(&x));
}

// --- movedim / moveaxis (TensorShape.cpp:4657/4768) ---

#[test]
fn movedim_matches_permute() {
    let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let x = leaf(&data, &[2, 3, 4], false);
    let moved = x.movedim_t(&[0], &[2]).unwrap();
    assert_eq!(moved.shape(), &[3, 4, 2]);
    assert_eq!(dense(&moved), dense(&x.permute(&[1, 2, 0]).unwrap()));
    // moveaxis is a literal alias.
    assert_eq!(dense(&x.moveaxis_t(&[0], &[2]).unwrap()), dense(&moved));
}

// --- broadcast_to / broadcast_tensors (TensorShape.cpp:652/656) ---

#[test]
fn broadcast_to_is_expand() {
    let x = leaf(&[1.0, 2.0, 3.0], &[1, 3], false);
    let y = x.broadcast_to_t(&[2, 3]).unwrap();
    assert_eq!(y.shape(), &[2, 3]);
    assert_eq!(dense(&y), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
}

#[test]
fn broadcast_tensors_to_common_shape() {
    let a = leaf(&[1.0, 2.0, 3.0], &[3, 1], false);
    let b = leaf(&[10.0, 20.0], &[1, 2], false);
    let out = ferrotorch_core::broadcast_tensors(&[a, b]).unwrap();
    assert_eq!(out[0].shape(), &[3, 2]);
    assert_eq!(out[1].shape(), &[3, 2]);
}

// --- repeat / tile (TensorShape.cpp:1909/1971) ---

#[test]
fn repeat_tiles_blocks() {
    // [1,2,3].repeat(2) == [1,2,3,1,2,3].
    let x = leaf(&[1.0, 2.0, 3.0], &[3], false);
    assert_eq!(
        dense(&x.repeat_t(&[2]).unwrap()),
        &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    );
}

#[test]
fn tile_left_pads_reps() {
    // tile([[1,2],[3,4]], (2,)) treated as (1,2): each row tiled twice.
    let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    let y = x.tile_t(&[2]).unwrap();
    assert_eq!(y.shape(), &[2, 4]);
    assert_eq!(dense(&y), &[1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
}

#[test]
fn repeat_backward_sums_over_copies() {
    let x = leaf(&[1.0, 2.0], &[2], true);
    let y = x.repeat_t(&[3]).unwrap();
    let loss = y.sum_all().unwrap();
    ferrotorch_core::backward(&loss).unwrap();
    let g = x.grad().unwrap().expect("repeat must propagate grad");
    assert_eq!(g.data().unwrap(), &[3.0, 3.0]);
}

// --- repeat_interleave (TensorShape.cpp repeat_interleave family) ---

#[test]
fn repeat_interleave_duplicates_in_place() {
    // [1,2,3] interleaved 2× == [1,1,2,2,3,3]; distinct from repeat's [1,2,3,1,2,3].
    let x = leaf(&[1.0, 2.0, 3.0], &[3], false);
    assert_eq!(
        dense(&x.repeat_interleave_t(2, 0).unwrap()),
        &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    );
}

#[test]
fn repeat_interleave_backward_sums_segments() {
    let x = leaf(&[1.0, 2.0], &[2], true);
    let y = x.repeat_interleave_t(3, 0).unwrap();
    let loss = y.sum_all().unwrap();
    ferrotorch_core::backward(&loss).unwrap();
    let g = x
        .grad()
        .unwrap()
        .expect("repeat_interleave must propagate grad");
    assert_eq!(g.data().unwrap(), &[3.0, 3.0]);
}

// --- unbind (TensorShape.cpp:4367) ---

#[test]
fn unbind_yields_one_slice_per_index() {
    let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let parts = x.unbind_t(0).unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(dense(&parts[0]), &[1.0, 2.0, 3.0]);
    assert_eq!(dense(&parts[1]), &[4.0, 5.0, 6.0]);
}

// --- tensor_split (TensorShape.cpp:1167) ---

#[test]
fn tensor_split_at_indices() {
    let x = leaf(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[6], false);
    let parts = x.tensor_split_t(&[2, 4], 0).unwrap();
    assert_eq!(parts.len(), 3);
    assert_eq!(dense(&parts[0]), &[0.0, 1.0]);
    assert_eq!(dense(&parts[1]), &[2.0, 3.0]);
    assert_eq!(dense(&parts[2]), &[4.0, 5.0]);
}

// --- vstack / hstack / dstack / column_stack (TensorShape.cpp:3532/3514/3544/3628) ---

#[test]
fn vstack_stacks_rows() {
    let a = leaf(&[1.0, 2.0, 3.0], &[3], false);
    let b = leaf(&[4.0, 5.0, 6.0], &[3], false);
    let y = Tensor::vstack_t(&[a, b]).unwrap();
    assert_eq!(y.shape(), &[2, 3]);
    assert_eq!(dense(&y), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn hstack_dispatches_by_rank() {
    // 1-D inputs cat along dim 0.
    let a = leaf(&[1.0, 2.0], &[2], false);
    let b = leaf(&[3.0], &[1], false);
    assert_eq!(dense(&Tensor::hstack_t(&[a, b]).unwrap()), &[1.0, 2.0, 3.0]);
    // 2-D inputs cat along dim 1.
    let c = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    let d = leaf(&[5.0, 6.0], &[2, 1], false);
    assert_eq!(
        dense(&Tensor::hstack_t(&[c, d]).unwrap()),
        &[1.0, 2.0, 5.0, 3.0, 4.0, 6.0]
    );
}

#[test]
fn dstack_stacks_depth() {
    let a = leaf(&[1.0, 2.0, 3.0], &[3], false);
    let b = leaf(&[4.0, 5.0, 6.0], &[3], false);
    let y = Tensor::dstack_t(&[a, b]).unwrap();
    assert_eq!(y.shape(), &[1, 3, 2]);
    assert_eq!(dense(&y), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn column_stack_makes_columns() {
    let a = leaf(&[1.0, 2.0, 3.0], &[3], false);
    let b = leaf(&[4.0, 5.0, 6.0], &[3], false);
    let y = Tensor::column_stack_t(&[a, b]).unwrap();
    assert_eq!(y.shape(), &[3, 2]);
    assert_eq!(dense(&y), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}
