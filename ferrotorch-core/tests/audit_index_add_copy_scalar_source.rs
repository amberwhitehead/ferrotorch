use ferrotorch_core::grad_fns::indexing::{index_add, index_copy};
use ferrotorch_core::int_tensor::IntTensor;
use ferrotorch_core::{Tensor, TensorStorage};

fn f32_tensor(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("tensor")
}

fn idx(data: &[i64], shape: &[usize]) -> IntTensor<i64> {
    IntTensor::from_vec(data.to_vec(), shape.to_vec()).expect("index")
}

#[test]
fn index_copy_rejects_scalar_source_for_non_scalar_destination_slice() {
    let input = f32_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let index = idx(&[1], &[1]);
    let source = f32_tensor(&[99.0], &[], false);

    let err = index_copy(&input, 1, &index, &source)
        .expect_err("torch rejects scalar source when the destination slice has non-empty shape");
    let msg = err.to_string();
    assert!(
        msg.contains("source tensor shape") || msg.contains("destination slice"),
        "unexpected error: {msg}"
    );
}

#[test]
fn index_copy_accepts_scalar_source_for_1d_destination_slice() {
    let input = f32_tensor(&[1.0, 2.0, 3.0, 4.0], &[4], false);
    let index = idx(&[1], &[1]);
    let source = f32_tensor(&[99.0], &[], false);

    let out = index_copy(&input, 0, &index, &source).expect("1-D scalar source is valid");
    assert_eq!(out.data().unwrap(), &[1.0, 99.0, 3.0, 4.0]);
}

#[test]
fn index_copy_scalar_source_empty_index_rejected_but_empty_1d_source_allowed() {
    let input = f32_tensor(&[5.0], &[], false);
    let empty_index = idx(&[], &[0]);
    let scalar_source = f32_tensor(&[9.0], &[], false);
    assert!(
        index_copy(&input, 0, &empty_index, &scalar_source).is_err(),
        "torch rejects scalar source unless the index has one element"
    );

    let empty_source = f32_tensor(&[], &[0], false);
    let out = index_copy(&input, 0, &empty_index, &empty_source)
        .expect("torch treats empty 1-D source + empty index as a no-op");
    assert_eq!(out.shape(), &[] as &[usize]);
    assert_eq!(out.data().unwrap(), &[5.0]);
}

#[test]
fn index_add_zero_dim_empty_index_rejected() {
    let input = f32_tensor(&[5.0], &[], false);
    let source = f32_tensor(&[2.0], &[], false);
    let empty_index = idx(&[], &[0]);

    assert!(
        index_add(&input, 0, &empty_index, &source, 1.0).is_err(),
        "torch rejects index_add on a 0-D input with an empty index"
    );
}
