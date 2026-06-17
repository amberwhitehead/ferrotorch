//! CORE-142 / crosslink #1836: raw linalg matmul arms must accept valid
//! non-contiguous CPU views, matching PyTorch's strided BLAS/matmul behavior.
//!
//! Local PyTorch source anchors:
//! - `/home/doll/pytorch/aten/src/ATen/native/LinearAlgebra.cpp:_matmul_impl`
//!   dispatches 1D/2D/3D cases through `dot`, `mv`, `mm`, and `bmm`.
//! - `/home/doll/pytorch/aten/src/ATen/native/Blas.cpp:172-203` `dot`
//!   passes each vector's stride into `dot_impl`.
//!
//! Live oracle on this machine (torch 2.11.0+cu130) confirmed the exact
//! strided-view values asserted below.

use ferrotorch_core::ops::linalg::{bmm, dot, matmul, mv, transpose};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn range(n: usize, shape: &[usize]) -> Tensor<f32> {
    let data: Vec<f32> = (0..n).map(|v| v as f32).collect();
    t(&data, shape)
}

fn assert_vec_eq(actual: &Tensor<f32>, expected_shape: &[usize], expected: &[f32], label: &str) {
    assert_eq!(actual.shape(), expected_shape, "{label}: shape");
    let data = actual.data().expect(label);
    assert_eq!(data, expected, "{label}: data");
}

#[test]
fn dot_accepts_strided_1d_views() {
    let base = range(10, &[10]);
    let a = base
        .as_strided(&[4], &[2], Some(1))
        .expect("a = torch.arange(10)[1:8:2]");
    let b = base
        .as_strided(&[4], &[2], Some(2))
        .expect("b = torch.arange(10)[2:9:2]");

    assert!(!a.is_contiguous());
    assert!(!b.is_contiguous());

    let out = dot(&a, &b).expect("dot strided views");
    assert!(out.is_scalar());
    assert_eq!(
        out.item().expect("dot scalar"),
        100.0,
        "torch.dot(torch.arange(10)[1:8:2], torch.arange(10)[2:9:2])"
    );
}

#[test]
fn mv_accepts_transposed_matrix_and_strided_vector_views() {
    let matrix = range(12, &[3, 4])
        .transpose(0, 1)
        .expect("torch.arange(12).reshape(3,4).t()");
    let vector = range(6, &[6])
        .as_strided(&[3], &[2], Some(0))
        .expect("torch.arange(6)[::2]");

    assert!(!matrix.is_contiguous());
    assert!(!vector.is_contiguous());

    let out = mv(&matrix, &vector).expect("mv strided views");
    assert_vec_eq(
        &out,
        &[4],
        &[40.0, 46.0, 52.0, 58.0],
        "torch.mv(transposed matrix, strided vector)",
    );
}

#[test]
fn matmul_vector_matrix_arm_accepts_strided_views() {
    let vector = range(8, &[8])
        .as_strided(&[4], &[2], Some(1))
        .expect("torch.arange(8)[1::2]");
    let matrix = range(20, &[5, 4])
        .transpose(0, 1)
        .expect("torch.arange(20).reshape(5,4).t()");

    assert!(!vector.is_contiguous());
    assert!(!matrix.is_contiguous());

    let out = matmul(&vector, &matrix).expect("1D @ 2D strided views");
    assert_vec_eq(
        &out,
        &[5],
        &[34.0, 98.0, 162.0, 226.0, 290.0],
        "torch.matmul(strided vector, transposed matrix)",
    );
}

#[test]
fn bmm_accepts_attention_style_transposed_batch_views() {
    let a = range(2 * 3 * 4, &[2, 3, 4])
        .transpose(1, 2)
        .expect("torch.arange(24).reshape(2,3,4).transpose(1,2)");
    let b = range(2 * 5 * 3, &[2, 5, 3])
        .transpose(1, 2)
        .expect("torch.arange(30).reshape(2,5,3).transpose(1,2)");

    assert!(!a.is_contiguous());
    assert!(!b.is_contiguous());

    let out = bmm(&a, &b).expect("bmm transposed batch views");
    assert_vec_eq(
        &out,
        &[2, 4, 5],
        &[
            20.0, 56.0, 92.0, 128.0, 164.0, 23.0, 68.0, 113.0, 158.0, 203.0, 26.0, 80.0, 134.0,
            188.0, 242.0, 29.0, 92.0, 155.0, 218.0, 281.0, 776.0, 920.0, 1064.0, 1208.0, 1352.0,
            824.0, 977.0, 1130.0, 1283.0, 1436.0, 872.0, 1034.0, 1196.0, 1358.0, 1520.0, 920.0,
            1091.0, 1262.0, 1433.0, 1604.0,
        ],
        "torch.bmm(transpose(1,2), transpose(1,2))",
    );
}

#[test]
fn transpose_materializes_sliced_noncontiguous_matrix_view() {
    let base = range(20, &[4, 5]);
    let sliced_columns = base
        .as_strided(&[4, 3], &[5, 2], Some(0))
        .expect("torch.arange(20).reshape(4,5)[:, ::2]");

    assert!(!sliced_columns.is_contiguous());

    let out = transpose(&sliced_columns).expect("transpose sliced view");
    assert_vec_eq(
        &out,
        &[3, 4],
        &[
            0.0, 5.0, 10.0, 15.0, 2.0, 7.0, 12.0, 17.0, 4.0, 9.0, 14.0, 19.0,
        ],
        "torch.arange(20).reshape(4,5)[:, ::2].t()",
    );
}
