//! Critic re-audit of #1342 shape ops (commit 5a487ed1a), per #1542 umbrella.
//!
//! Most of the 21 shape ops are non-diff structural / inherit autograd from an
//! existing *Backward. Exactly TWO carry a hand-written backward node:
//!   - `FlipBackward`           (flip is its own adjoint)
//!   - `RepeatInterleaveBackward` (segment-sum adjoint)
//!
//! The build's own audit file (`divergence_shape_ops_audit.rs`) tests the flip
//! backward ONLY via `sum_all().backward()`, whose incoming gradient is a tensor
//! of all-ones. A uniform incoming gradient is INVARIANT under both a flip
//! permutation and a segment-sum, so that test cannot distinguish a correct
//! adjoint from an identity / mis-indexed one (the #1555 failure class). This
//! file seeds backward with a NON-UNIFORM gradient via `backward_with_grad`, so
//! the permutation (flip) and the reduction (repeat_interleave) are actually
//! exercised.
//!
//! R-CHAR-3: every expected gradient below is the live-torch result, computed by
//!   import torch
//!   x = torch.tensor(...); x.requires_grad_(True)
//!   y = <op>(x); y.backward(<non-uniform grad>); print(x.grad)
//! and quoted inline — NOT copied from the ferrotorch side.

use ferrotorch_core::{Tensor, TensorStorage, backward_with_grad};

fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

fn dense(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous().unwrap().data_vec().unwrap()
}

// ---------------------------------------------------------------------------
// FlipBackward — non-uniform incoming gradient (TensorTransformations.cpp:36)
// ---------------------------------------------------------------------------

#[test]
fn critic_flip_backward_nonuniform_grad_dim1() {
    // torch:
    //   x = torch.tensor([[1.,2.,3.],[4.,5.,6.]], requires_grad=True)
    //   y = x.flip([1])                                  # [[3,2,1],[6,5,4]]
    //   y.backward(torch.tensor([[10.,20.,30.],[40.,50.,60.]]))
    //   x.grad -> [[30,20,10],[60,50,40]]
    // The grad must be the incoming grad re-flipped along dim 1. A uniform-ones
    // seed (the existing audit) would yield all-ones either way; this seed pins
    // the permutation.
    let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let y = x.flip_t(&[1]).unwrap();
    let g = Tensor::from_storage(
        TensorStorage::cpu(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
        vec![2, 3],
        false,
    )
    .unwrap();
    backward_with_grad(&y, Some(&g)).unwrap();
    let grad = x.grad().unwrap().expect("flip must propagate grad to leaf");
    assert_eq!(dense(&grad), vec![30.0, 20.0, 10.0, 60.0, 50.0, 40.0]);
}

// ---------------------------------------------------------------------------
// RepeatInterleaveBackward — segment-sum adjoint with non-uniform grad
// (REQ-14; aten/src/ATen/native/TensorShape.cpp repeat_interleave family)
// ---------------------------------------------------------------------------

#[test]
fn critic_repeat_interleave_backward_segment_sum_1d() {
    // torch:
    //   x = torch.tensor([1.,2.,3.], requires_grad=True)
    //   y = torch.repeat_interleave(x, 2)              # [1,1,2,2,3,3]
    //   y.backward(torch.tensor([1.,2.,3.,4.,5.,6.]))
    //   x.grad -> [3., 7., 11.]   (g0+g1, g2+g3, g4+g5)
    // A uniform seed would give [2,2,2] regardless of indexing; this seed pins
    // the segment-sum boundaries.
    let x = leaf(&[1.0, 2.0, 3.0], &[3]);
    let y = x.repeat_interleave_t(2, 0).unwrap();
    assert_eq!(dense(&y), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    let g = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        vec![6],
        false,
    )
    .unwrap();
    backward_with_grad(&y, Some(&g)).unwrap();
    let grad = x
        .grad()
        .unwrap()
        .expect("repeat_interleave must propagate grad");
    assert_eq!(dense(&grad), vec![3.0, 7.0, 11.0]);
}

#[test]
fn critic_repeat_interleave_backward_2d_dim1() {
    // torch:
    //   x = torch.tensor([[1.,2.],[3.,4.]], requires_grad=True)
    //   y = torch.repeat_interleave(x, 2, dim=1)  # [[1,1,2,2],[3,3,4,4]]
    //   g = torch.tensor([[10.,1.,20.,2.],[30.,3.,40.,4.]])
    //   y.backward(g)
    //   x.grad -> [[11., 22.],[33., 44.]]   (per row: g0+g1, g2+g3)
    // Exercises the outer/inner stride bookkeeping of the adjoint on dim 1.
    let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let y = x.repeat_interleave_t(2, 1).unwrap();
    assert_eq!(dense(&y), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]);
    let g = Tensor::from_storage(
        TensorStorage::cpu(vec![10.0, 1.0, 20.0, 2.0, 30.0, 3.0, 40.0, 4.0]),
        vec![2, 4],
        false,
    )
    .unwrap();
    backward_with_grad(&y, Some(&g)).unwrap();
    let grad = x
        .grad()
        .unwrap()
        .expect("repeat_interleave 2d must propagate grad");
    assert_eq!(dense(&grad), vec![11.0, 22.0, 33.0, 44.0]);
}

// ---------------------------------------------------------------------------
// Structural-op forward spot-checks vs torch (TensorShape.cpp / Transformations)
// ---------------------------------------------------------------------------

#[test]
fn critic_movedim_shape() {
    // torch.movedim(torch.zeros(2,3,4), 0, 2).shape == [3,4,2]
    let x =
        Tensor::from_storage(TensorStorage::cpu(vec![0.0f32; 24]), vec![2, 3, 4], false).unwrap();
    let y = x.movedim_t(&[0], &[2]).unwrap();
    assert_eq!(y.shape(), &[3, 4, 2]);
}

#[test]
fn critic_rot90_2x2_value() {
    // torch.rot90(torch.tensor([[1,2],[3,4]])) -> [[2,4],[1,3]]
    let x = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0]),
        vec![2, 2],
        false,
    )
    .unwrap();
    let y = x.rot90_t(1, &[0, 1]).unwrap();
    assert_eq!(dense(&y), vec![2.0, 4.0, 1.0, 3.0]);
}

#[test]
fn critic_tile_1d_value() {
    // torch.tile(torch.tensor([1.,2.]), (2,)) -> [1,2,1,2]
    let x = Tensor::from_storage(TensorStorage::cpu(vec![1.0f32, 2.0]), vec![2], false).unwrap();
    let y = x.tile_t(&[2]).unwrap();
    assert_eq!(dense(&y), vec![1.0, 2.0, 1.0, 2.0]);
}
