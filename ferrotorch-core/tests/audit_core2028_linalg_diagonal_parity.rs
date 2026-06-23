//! Batched `torch.linalg.diagonal` parity audit.
//!
//! PyTorch reference inspected before implementation:
//! `aten/src/ATen/native/LinearAlgebra.cpp:2215` aliases
//! `torch.linalg.diagonal(A, offset, dim1=-2, dim2=-1)` to `A.diagonal(...)`;
//! `aten/src/ATen/native/TensorShape.cpp:4645` implements the backward as
//! zeros shaped like the input, then `grad_input.diagonal(...).copy_(grad)`.

use ferrotorch_core::grad_fns::linalg::{DiagonalBackward, diagonal_differentiable};
use ferrotorch_core::linalg;
use ferrotorch_core::{Tensor, TensorStorage};

fn cpu(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .unwrap()
}

fn short_type_name<T>() -> String {
    std::any::type_name::<T>()
        .rsplit("::")
        .next()
        .unwrap()
        .split('<')
        .next()
        .unwrap()
        .to_string()
}

fn assert_close(got: &[f32], want: &[f32], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= 1e-6,
            "{ctx}: element {i}: got {g}, want {w}"
        );
    }
}

#[test]
fn linalg_diagonal_batched_forward_uses_last_two_dims_like_pytorch() {
    let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let a = cpu(&data, &[2, 3, 4], false);

    let pos = linalg::diagonal(&a, 1).unwrap();
    assert_eq!(pos.shape(), &[2, 3]);
    assert_close(
        &pos.data_vec().unwrap(),
        &[1.0, 6.0, 11.0, 13.0, 18.0, 23.0],
        "batched diagonal offset +1",
    );

    let neg = linalg::diagonal(&a, -1).unwrap();
    assert_eq!(neg.shape(), &[2, 2]);
    assert_close(
        &neg.data_vec().unwrap(),
        &[4.0, 9.0, 16.0, 21.0],
        "batched diagonal offset -1",
    );
}

#[test]
fn linalg_diagonal_empty_offset_preserves_batch_shape() {
    let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let a = cpu(&data, &[2, 3, 4], false);

    let empty = linalg::diagonal(&a, 99).unwrap();

    assert_eq!(empty.shape(), &[2, 0]);
    assert!(empty.data_vec().unwrap().is_empty());
}

#[test]
fn diagonal_differentiable_batched_backward_scatters_to_input_shape() {
    let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let a = cpu(&data, &[2, 3, 4], true);
    let grad = cpu(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);

    let out = diagonal_differentiable(&a, -1).unwrap();
    let node = out
        .grad_fn()
        .expect("diagonal_differentiable must attach a grad_fn");
    assert_eq!(node.name(), short_type_name::<DiagonalBackward<f32>>());
    assert_eq!(out.shape(), &[2, 2]);
    assert_close(
        &out.data_vec().unwrap(),
        &[4.0, 9.0, 16.0, 21.0],
        "differentiable batched diagonal forward",
    );

    out.backward_with_gradient(&grad).unwrap();

    let mut want = vec![0.0_f32; 24];
    want[4] = 1.0;
    want[9] = 2.0;
    want[16] = 3.0;
    want[21] = 4.0;
    let got = a.grad().unwrap().expect("a.grad").data_vec().unwrap();
    assert_close(&got, &want, "batched diagonal backward scatter");
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::{assert_close, cpu};
    use ferrotorch_core::device::Device;
    use ferrotorch_core::linalg;
    use ferrotorch_core::{Tensor, TensorStorage};
    use std::sync::Once;

    static INIT: Once = Once::new();

    fn ensure_cuda() {
        INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("FERROTORCH_ENABLE_GPU=1 and CUDA backend required")
        });
    }

    #[test]
    fn cuda_linalg_diagonal_strided_batched_view_backward_stays_resident() {
        ensure_cuda();
        let base_data: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let base = Tensor::from_storage(TensorStorage::cpu(base_data), vec![2, 4, 3], false)
            .unwrap()
            .to(Device::Cuda(0))
            .expect("upload base")
            .requires_grad_(true);
        let a = base.transpose(1, 2).expect("cuda transpose view");

        let out = linalg::diagonal(&a, 1).expect("cuda batched diagonal");

        assert_eq!(out.device(), Device::Cuda(0));
        assert_eq!(out.shape(), &[2, 3]);
        assert_close(
            &out.data_vec().unwrap(),
            &[3.0, 7.0, 11.0, 15.0, 19.0, 23.0],
            "cuda strided batched diagonal forward",
        );

        out.backward_with_gradient(
            &cpu(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false)
                .to(Device::Cuda(0))
                .expect("upload grad"),
        )
        .unwrap();
        let grad = base.grad().unwrap().expect("base.grad");
        assert_eq!(grad.device(), Device::Cuda(0));

        let mut want = vec![0.0_f32; 24];
        want[3] = 1.0;
        want[7] = 2.0;
        want[11] = 3.0;
        want[15] = 4.0;
        want[19] = 5.0;
        want[23] = 6.0;
        assert_close(
            &grad.data_vec().unwrap(),
            &want,
            "cuda strided batched diagonal backward",
        );
    }
}
