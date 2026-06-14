use ferrotorch_core::{Tensor, TensorStorage, creation::zeros_meta};

#[test]
fn fallible_stride_view_rejects_out_of_bounds_nonempty_view() {
    let base = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0]),
        vec![4],
        false,
    )
    .expect("base tensor");

    let err = base
        .try_stride_view(vec![3], vec![1], 2)
        .expect_err("nonempty view must not extend past storage");
    assert!(
        format!("{err:?}").contains("beyond storage length"),
        "unexpected error: {err:?}"
    );
}

#[test]
fn public_stride_view_panics_before_invalid_metadata_can_escape() {
    let base = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0]),
        vec![4],
        false,
    )
    .expect("base tensor");

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = base.stride_view(vec![3], vec![1], 2);
    }));
    assert!(
        result.is_err(),
        "invalid public stride_view must fail immediately"
    );
}

#[test]
fn empty_stride_view_allows_offset_beyond_storage_like_pytorch() {
    let base = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0]),
        vec![4],
        false,
    )
    .expect("base tensor");

    let view = base
        .try_stride_view(vec![0], vec![1], 100)
        .expect("empty view with arbitrary signed offset");
    assert_eq!(view.shape(), &[0]);
    assert_eq!(view.storage_offset(), 100);
    assert_eq!(view.data_vec().expect("empty data vec"), Vec::<f32>::new());
}

#[test]
fn empty_stride_view_still_rejects_unrepresentable_dimension_metadata() {
    let base = Tensor::from_storage(TensorStorage::cpu(vec![1.0f32]), vec![1], false)
        .expect("base tensor");

    let err = base
        .try_stride_view(vec![0, usize::MAX], vec![1, 1], 0)
        .expect_err("dimension cannot fit signed metadata");
    assert!(
        format!("{err:?}").contains("exceeds i64::MAX"),
        "unexpected error: {err:?}"
    );
}

#[test]
fn tensor_construction_rejects_shape_numel_overflow() {
    let err = Tensor::from_storage(TensorStorage::cpu(vec![1.0f32]), vec![usize::MAX, 2], false)
        .expect_err("overflowed shape must be rejected");
    assert!(
        format!("{err:?}").contains("overflows usize"),
        "unexpected error: {err:?}"
    );
}

#[test]
fn meta_factory_rejects_shape_numel_overflow_before_wrapping() {
    let err = zeros_meta::<f32>(&[usize::MAX, 2]).expect_err("overflowed meta shape");
    assert!(
        format!("{err:?}").contains("overflows usize"),
        "unexpected error: {err:?}"
    );
}

#[test]
fn zero_numel_shape_still_rejects_unrepresentable_stride_metadata() {
    let err = Tensor::from_storage(
        TensorStorage::cpu(Vec::<f32>::new()),
        vec![0, usize::MAX],
        false,
    )
    .expect_err("dimension cannot fit signed stride metadata");
    assert!(
        format!("{err:?}").contains("exceeds isize::MAX"),
        "unexpected error: {err:?}"
    );
}
