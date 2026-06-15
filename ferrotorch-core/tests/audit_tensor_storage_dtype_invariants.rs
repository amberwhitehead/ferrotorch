use ferrotorch_core::{
    BoolTensor, ComplexTensor, IntTensor, Tensor, TensorStorage,
    creation::zeros_meta,
    shape::{
        c_contiguous_strides, channels_last_3d_strides, channels_last_strides, checked_byte_count,
        checked_channels_last_3d_strides, checked_channels_last_strides, numel,
    },
};

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

#[test]
fn checked_storage_byte_count_rejects_itemsize_overflow() {
    let err = checked_byte_count((usize::MAX / 2) + 1, 2, "byte_probe")
        .expect_err("byte count must not wrap");
    assert!(
        format!("{err:?}").contains("storage size calculation overflowed"),
        "unexpected error: {err:?}"
    );
}

#[test]
fn public_numel_panics_on_overflow_instead_of_wrapping() {
    let result = std::panic::catch_unwind(|| {
        let _ = numel(&[usize::MAX, 2]);
    });
    assert!(result.is_err(), "overflowed numel must fail loudly");
}

#[test]
fn public_c_contiguous_strides_panics_on_unrepresentable_dimension() {
    let result = std::panic::catch_unwind(|| {
        let _ = c_contiguous_strides(&[0, usize::MAX]);
    });
    assert!(
        result.is_err(),
        "signed stride helper must not cast usize::MAX to a negative stride"
    );
}

#[test]
fn checked_channels_last_strides_rejects_signed_stride_overflow() {
    let err = checked_channels_last_strides(&[1, isize::MAX as usize, 1, 2], "channels_last_probe")
        .expect_err("channels-last N stride cannot fit isize");
    assert!(
        format!("{err:?}").contains("exceeds isize::MAX"),
        "unexpected error: {err:?}"
    );
}

#[test]
fn checked_channels_last_3d_strides_rejects_signed_stride_overflow() {
    let err = checked_channels_last_3d_strides(
        &[1, isize::MAX as usize, 1, 1, 2],
        "channels_last_3d_probe",
    )
    .expect_err("channels-last-3d N stride cannot fit isize");
    assert!(
        format!("{err:?}").contains("exceeds isize::MAX"),
        "unexpected error: {err:?}"
    );
}

#[test]
fn public_channels_last_helpers_panic_instead_of_wrapping() {
    let cl = std::panic::catch_unwind(|| {
        let _ = channels_last_strides(&[1, isize::MAX as usize, 1, 2]);
    });
    assert!(cl.is_err(), "channels-last helper must not wrap N stride");

    let cl3d = std::panic::catch_unwind(|| {
        let _ = channels_last_3d_strides(&[1, isize::MAX as usize, 1, 1, 2]);
    });
    assert!(
        cl3d.is_err(),
        "channels-last-3d helper must not wrap N stride"
    );
}

#[test]
fn wrapper_tensor_constructors_reject_shape_product_overflow() {
    let bool_err = BoolTensor::from_vec(Vec::new(), vec![usize::MAX, 2])
        .expect_err("BoolTensor shape product overflow");
    assert!(
        format!("{bool_err:?}").contains("overflows usize"),
        "unexpected error: {bool_err:?}"
    );

    let int_err = IntTensor::<i64>::from_vec(Vec::new(), vec![usize::MAX, 2])
        .expect_err("IntTensor shape product overflow");
    assert!(
        format!("{int_err:?}").contains("overflows usize"),
        "unexpected error: {int_err:?}"
    );

    let complex_err = ComplexTensor::<f32>::from_re_im(Vec::new(), Vec::new(), vec![usize::MAX, 2])
        .expect_err("ComplexTensor shape product overflow");
    assert!(
        format!("{complex_err:?}").contains("overflows usize"),
        "unexpected error: {complex_err:?}"
    );
}

#[test]
fn wrapper_infallible_zeros_helpers_panic_on_shape_product_overflow() {
    let bool_result = std::panic::catch_unwind(|| {
        let _ = BoolTensor::zeros(&[usize::MAX, 2]);
    });
    assert!(bool_result.is_err(), "BoolTensor::zeros must not wrap");

    let int_result = std::panic::catch_unwind(|| {
        let _ = IntTensor::<i64>::zeros(&[usize::MAX, 2]);
    });
    assert!(int_result.is_err(), "IntTensor::zeros must not wrap");

    let complex_result = std::panic::catch_unwind(|| {
        let _ = ComplexTensor::<f32>::zeros(&[usize::MAX, 2]);
    });
    assert!(
        complex_result.is_err(),
        "ComplexTensor::zeros must not wrap"
    );
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use ferrotorch_core::{DType, Device, FerrotorchError, gpu_dispatch::gpu_backend};
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for CORE-004 probes");
        });
    }

    fn cuda_base() -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0]),
            vec![4],
            false,
        )
        .expect("base tensor")
        .to(Device::Cuda(0))
        .expect("upload base tensor")
    }

    #[test]
    fn cuda_fallible_stride_view_rejects_oob_before_strided_copy_can_read() {
        ensure_cuda_backend();
        let base = cuda_base();

        let err = base
            .try_stride_view(vec![3], vec![1], 2)
            .expect_err("nonempty CUDA view must not extend past storage");

        assert_eq!(base.device(), Device::Cuda(0));
        assert!(
            matches!(err, FerrotorchError::InvalidArgument { .. }),
            "expected InvalidArgument, got {err:?}"
        );
        assert!(
            format!("{err:?}").contains("beyond storage length"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn cuda_empty_stride_view_allows_arbitrary_offset_without_reading_storage() {
        ensure_cuda_backend();
        let base = cuda_base();

        let view = base
            .try_stride_view(vec![0], vec![1], 100)
            .expect("empty CUDA view with arbitrary offset");

        assert_eq!(view.device(), Device::Cuda(0));
        assert_eq!(view.shape(), &[0]);
        assert_eq!(view.storage_offset(), 100);
        let host = view.to(Device::Cpu).expect("copy empty CUDA view to CPU");
        assert_eq!(host.data_vec().expect("empty host data"), Vec::<f32>::new());
    }

    #[test]
    fn cuda_alloc_zeros_rejects_byte_count_overflow_before_driver_call() {
        ensure_cuda_backend();
        let backend = gpu_backend().expect("registered CUDA backend");

        let err = backend
            .alloc_zeros(usize::MAX, DType::F16, 0)
            .expect_err("byte-count overflow must reject before CUDA allocation");

        assert!(
            format!("{err:?}").contains("storage size calculation overflowed"),
            "unexpected error: {err:?}"
        );
    }
}
