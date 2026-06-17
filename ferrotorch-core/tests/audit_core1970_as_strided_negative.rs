//! Regression for #1970: PyTorch rejects negative strides for the whole
//! public as_strided family before installing view metadata or launching
//! copy/scatter kernels.
//!
//! Live oracle, torch 2.11.0+cu130:
//! ```python
//! x = torch.arange(4., device=device)
//! src = torch.ones(4, device=device)
//! torch.as_strided(x, (4,), (-1,), 3)
//! torch.as_strided(x, (0,), (-1,), 3)
//! torch.as_strided_copy(x, (4,), (-1,), 3)
//! torch.as_strided_copy(x, (0,), (-1,), 3)
//! torch.as_strided_scatter(x, src, (4,), (-1,), 3)
//! torch.as_strided_scatter(x, src[:0], (0,), (-1,), 3)
//! # RuntimeError: as_strided: Negative strides are not supported at the moment, got strides: [-1]
//! ```

use ferrotorch_core::{
    FerrotorchError, Tensor, TensorStorage, as_strided, as_strided_copy, as_strided_scatter,
};

fn tensor(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("cpu tensor")
}

fn expect_negative_stride_error<T: std::fmt::Debug>(
    label: &str,
    result: Result<T, FerrotorchError>,
) {
    match result {
        Err(FerrotorchError::InvalidArgument { message }) => assert_eq!(
            message,
            "as_strided: Negative strides are not supported at the moment, got strides: [-1]",
            "{label}"
        ),
        other => panic!("{label}: expected InvalidArgument negative-stride error, got {other:?}"),
    }
}

fn assert_negative_stride_family_rejects(
    x: &Tensor<f32>,
    src: &Tensor<f32>,
    empty_src: &Tensor<f32>,
) {
    expect_negative_stride_error(
        "method as_strided negative stride",
        x.as_strided(&[4], &[-1], Some(3)),
    );
    expect_negative_stride_error(
        "free as_strided negative stride",
        as_strided(x, &[4], &[-1], Some(3)),
    );
    expect_negative_stride_error(
        "as_strided zero-sized negative stride",
        x.as_strided(&[0], &[-1], Some(3)),
    );
    expect_negative_stride_error(
        "as_strided_copy negative stride",
        x.as_strided_copy(&[4], &[-1], Some(3)),
    );
    expect_negative_stride_error(
        "free as_strided_copy negative stride",
        as_strided_copy(x, &[4], &[-1], Some(3)),
    );
    expect_negative_stride_error(
        "as_strided_copy zero-sized negative stride",
        x.as_strided_copy(&[0], &[-1], Some(3)),
    );
    expect_negative_stride_error(
        "as_strided_scatter negative stride",
        x.as_strided_scatter(src, &[4], &[-1], Some(3)),
    );
    expect_negative_stride_error(
        "free as_strided_scatter negative stride",
        as_strided_scatter(x, src, &[4], &[-1], Some(3)),
    );
    expect_negative_stride_error(
        "as_strided_scatter zero-sized negative stride",
        x.as_strided_scatter(empty_src, &[0], &[-1], Some(3)),
    );
}

#[test]
fn cpu_as_strided_family_rejects_negative_strides_like_torch() {
    let x = tensor(&[0.0, 1.0, 2.0, 3.0], &[4], false);
    let src = tensor(&[10.0, 11.0, 12.0, 13.0], &[4], false);
    let empty_src = tensor(&[], &[0], false);

    assert_negative_stride_family_rejects(&x, &src, &empty_src);
}

#[test]
fn failed_negative_stride_view_does_not_create_autograd_metadata() {
    let x = tensor(&[0.0, 1.0, 2.0, 3.0], &[4], true);
    expect_negative_stride_error(
        "tracked as_strided negative stride",
        x.as_strided(&[4], &[-1], Some(3)),
    );

    assert!(
        x.grad().expect("grad handle").is_none(),
        "failed as_strided must not attach or run a backward edge"
    );
}

#[cfg(feature = "gpu")]
mod cuda {
    use super::*;
    use std::sync::Once;

    use ferrotorch_core::device::Device;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for negative-stride probes");
        });
    }

    fn cuda_tensor(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
        tensor(data, shape, requires_grad)
            .to(Device::Cuda(0))
            .expect("upload")
            .requires_grad_(requires_grad)
    }

    #[test]
    fn cuda_as_strided_family_rejects_negative_strides_before_kernels() {
        ensure_cuda_backend();
        let x = cuda_tensor(&[0.0, 1.0, 2.0, 3.0], &[4], false);
        let src = cuda_tensor(&[10.0, 11.0, 12.0, 13.0], &[4], false);
        let empty_src = cuda_tensor(&[], &[0], false);

        assert_negative_stride_family_rejects(&x, &src, &empty_src);
        assert_eq!(x.device(), Device::Cuda(0));
        assert_eq!(src.device(), Device::Cuda(0));
    }
}
