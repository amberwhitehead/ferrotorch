//! Regression coverage for CORE-097: meta-device `torch.matmul` parity for
//! rank-1 operands combined with batched operands.
//!
//! PyTorch source anchor:
//! `/home/doll/pytorch/aten/src/ATen/native/LinearAlgebra.cpp::_matmul_impl`
//! computes the general case as:
//! - `n = dim_tensor1 > 1 ? tensor1[-2] : 1`
//! - `m1 = tensor1[-1]`
//! - `m2 = dim_tensor2 > 1 ? tensor2[-2] : tensor2[0]`
//! - `p = dim_tensor2 > 1 ? tensor2[-1] : 1`
//!
//! It then broadcasts only the leading batch dims and emits `n`/`p` only for
//! operands whose original rank is greater than one.
//!
//! Live torch 2.11.0+cu130 oracle:
//! ```text
//! (3,) @ (2,3,4) -> (2,4)
//! (2,3,4) @ (4,) -> (2,3)
//! (3,) @ (5,2,3,4) -> (5,2,4)
//! (5,2,3,4) @ (4,) -> (5,2,3)
//! (0,) @ (2,0,4) -> (2,4)
//! (2,3,0) @ (0,) -> (2,3)
//! (2,0,3,4) @ (1,4,5) -> (2,0,3,5)
//! ```

use ferrotorch_core::creation::zeros_meta;
use ferrotorch_core::ops::linalg::matmul;
use ferrotorch_core::tensor::Tensor;

fn meta(shape: &[usize]) -> Tensor<f32> {
    zeros_meta(shape).expect("zeros_meta")
}

fn assert_meta_matmul_shape(a_shape: &[usize], b_shape: &[usize], expected: &[usize]) {
    let a = meta(a_shape);
    let b = meta(b_shape);
    let out = matmul(&a, &b)
        .unwrap_or_else(|err| panic!("matmul({a_shape:?}, {b_shape:?}) errored: {err}"));
    assert!(out.is_meta(), "meta inputs must stay on the meta device");
    assert_eq!(
        out.shape(),
        expected,
        "torch.matmul shape oracle for {a_shape:?} @ {b_shape:?}"
    );
}

#[test]
fn vector_lhs_against_batched_rhs_matches_torch_meta_shape() {
    assert_meta_matmul_shape(&[3], &[2, 3, 4], &[2, 4]);
    assert_meta_matmul_shape(&[3], &[5, 2, 3, 4], &[5, 2, 4]);
}

#[test]
fn batched_lhs_against_vector_rhs_matches_torch_meta_shape() {
    assert_meta_matmul_shape(&[2, 3, 4], &[4], &[2, 3]);
    assert_meta_matmul_shape(&[5, 2, 3, 4], &[4], &[5, 2, 3]);
}

#[test]
fn vector_batched_meta_shape_handles_zero_sized_dims() {
    assert_meta_matmul_shape(&[0], &[2, 0, 4], &[2, 4]);
    assert_meta_matmul_shape(&[2, 3, 0], &[0], &[2, 3]);
    assert_meta_matmul_shape(&[2, 0, 3, 4], &[1, 4, 5], &[2, 0, 3, 5]);
}

#[test]
fn invalid_vector_batched_shapes_return_errors_instead_of_panicking() {
    for (a_shape, b_shape) in [
        (&[3][..], &[2, 4, 5][..]),
        (&[2, 3, 4], &[5]),
        (&[2, 0, 3, 4], &[7, 4, 5]),
    ] {
        let a = meta(a_shape);
        let b = meta(b_shape);
        assert!(
            matmul(&a, &b).is_err(),
            "invalid torch.matmul shape {a_shape:?} @ {b_shape:?} must error"
        );
    }
}
