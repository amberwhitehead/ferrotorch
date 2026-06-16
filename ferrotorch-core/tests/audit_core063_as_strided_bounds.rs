//! CORE-063 (#1757): `as_strided` metadata must be validated with checked
//! signed geometry, matching PyTorch's storage-size and numel overflow guards.
//!
//! Local PyTorch references:
//! - `/home/doll/pytorch/aten/src/ATen/native/TensorShape.cpp`
//!   `as_strided_tensorimpl` delegates to `setStrided`.
//! - `/home/doll/pytorch/aten/src/ATen/EmptyTensor.cpp`
//!   `computeStorageNbytes` checks strided storage-size overflow and returns
//!   zero for zero-sized shapes.
//! - `/home/doll/pytorch/c10/core/TensorImpl.h` `safe_numel` rejects
//!   logical element-count overflow.

use ferrotorch_core::{FerrotorchError, Tensor, TensorStorage};

fn scalar() -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(vec![1.0_f32]), vec![], false).expect("scalar")
}

fn expect_invalid_contains<T: std::fmt::Debug>(
    label: &str,
    result: Result<T, FerrotorchError>,
    needle: &str,
) {
    match result {
        Err(FerrotorchError::InvalidArgument { message }) => assert!(
            message.contains(needle),
            "{label}: expected error containing {needle:?}, got {message:?}"
        ),
        other => panic!("{label}: expected InvalidArgument, got {other:?}"),
    }
}

#[test]
#[cfg(target_pointer_width = "64")]
fn as_strided_rejects_shape_dimension_above_signed_i64_1757() {
    let x = scalar();
    expect_invalid_contains(
        "as_strided usize::MAX dim",
        x.as_strided(&[usize::MAX], &[0], Some(0)),
        "exceeds i64::MAX",
    );
}

#[test]
#[cfg(target_pointer_width = "64")]
fn as_strided_rejects_logical_numel_over_i64_even_when_storage_fits_1757() {
    let x = scalar();
    let too_many = [i64::MAX as usize, 2];
    expect_invalid_contains(
        "as_strided zero-stride numel overflow",
        x.as_strided(&too_many, &[0, 0], Some(0)),
        "element count overflows i64",
    );
    expect_invalid_contains(
        "as_strided_copy zero-stride numel overflow",
        x.as_strided_copy(&too_many, &[0, 0], Some(0)),
        "element count overflows i64",
    );
    expect_invalid_contains(
        "as_strided_scatter zero-stride numel overflow",
        x.as_strided_scatter(&x, &too_many, &[0, 0], Some(0)),
        "element count overflows i64",
    );
}

#[test]
#[cfg(target_pointer_width = "64")]
fn as_strided_allows_i64_max_zero_stride_view_when_numel_is_representable_1757() {
    let x = scalar();
    let v = x
        .as_strided(&[i64::MAX as usize], &[0], Some(0))
        .expect("i64::MAX zero-stride view");

    assert_eq!(v.shape(), &[i64::MAX as usize]);
    assert_eq!(v.strides(), &[0]);
    assert_eq!(v.storage_offset(), 0);
    assert_eq!(v.storage_len(), 1);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn as_strided_singleton_dim_allows_huge_stride_because_no_step_is_taken_1757() {
    let x = scalar();
    let v = x
        .as_strided(&[1], &[isize::MAX], Some(0))
        .expect("size-1 huge-stride view");

    assert_eq!(v.data_vec().expect("logical singleton"), vec![1.0]);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn as_strided_empty_view_allows_large_signed_offset_without_touching_storage_1757() {
    let x = scalar();
    let v = x
        .as_strided(
            &[0, i64::MAX as usize],
            &[isize::MAX, isize::MAX],
            Some(i64::MAX as usize),
        )
        .expect("empty as_strided view");

    assert_eq!(v.shape(), &[0, i64::MAX as usize]);
    assert_eq!(v.storage_offset(), i64::MAX as usize);
    assert_eq!(v.data_vec().expect("empty data"), Vec::<f32>::new());
}

#[test]
#[cfg(target_pointer_width = "64")]
fn as_strided_rejects_positive_extent_signed_overflow_1757() {
    let x = scalar();
    expect_invalid_contains(
        "as_strided offset plus positive stride overflow",
        x.as_strided(&[2], &[isize::MAX], Some(1)),
        "overflows signed offset range",
    );
}

#[test]
#[cfg(target_pointer_width = "64")]
fn as_strided_rejects_negative_stride_extent_overflow_1757() {
    let x = scalar();
    expect_invalid_contains(
        "as_strided negative stride extent overflow",
        x.as_strided(&[3], &[isize::MIN], Some(0)),
        "overflow signed offset range",
    );
}
