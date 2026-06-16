//! CORE-146 follow-up: `torch.linalg.cross` broadcasts same-rank batch
//! dimensions and runs on CUDA with autograd. PyTorch source:
//! `/home/doll/pytorch/aten/src/ATen/native/Cross.cpp` expands both operands
//! to `infer_size(input.sizes(), other.sizes())`; derivatives.yaml routes
//! `da = linalg_cross(other, grad)` and `db = linalg_cross(grad, self)`.

use ferrotorch_core::linalg::cross;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use half::{bf16, f16};

fn tensor(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f64> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .unwrap()
}

fn tensor_f16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f16> {
    Tensor::from_storage(
        TensorStorage::cpu(data.iter().copied().map(f16::from_f32).collect()),
        shape.to_vec(),
        requires_grad,
    )
    .unwrap()
}

fn tensor_bf16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<bf16> {
    Tensor::from_storage(
        TensorStorage::cpu(data.iter().copied().map(bf16::from_f32).collect()),
        shape.to_vec(),
        requires_grad,
    )
    .unwrap()
}

fn assert_close(actual: &[f64], expected: &[f64], label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= 1e-12,
            "{label}[{i}] actual={a} expected={e} diff={}",
            (a - e).abs()
        );
    }
}

fn assert_close_f32(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{label}[{i}] actual={a} expected={e} diff={}",
            (a - e).abs()
        );
    }
}

#[test]
fn cross_broadcasts_same_rank_batch_dims_like_torch() {
    let a = tensor(&[1.0, 0.0, 0.0], &[1, 3], false);
    let b = tensor(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[2, 3], false);

    let c = cross(&a, &b, -1).expect("broadcasted linalg.cross");

    assert_eq!(c.shape(), &[2, 3]);
    assert_close(
        &c.data_vec().unwrap(),
        &[0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
        "cross broadcast forward",
    );
}

#[test]
fn cross_rejects_rank_broadcast_like_torch_linalg_cross() {
    let a = tensor(&[1.0, 0.0, 0.0], &[3], false);
    let b = tensor(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[2, 3], false);

    let err = cross(&a, &b, -1).expect_err("rank mismatch must be rejected");

    assert!(
        err.to_string().contains("same number of dimensions"),
        "unexpected error: {err:?}"
    );
}

#[test]
fn cross_broadcast_backward_sums_to_original_operand_shapes() {
    let a = tensor(&[1.0, 0.0, 0.0], &[1, 3], true);
    let b = tensor(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[2, 3], true);

    let loss = cross(&a, &b, -1)
        .expect("broadcasted linalg.cross")
        .sum_all()
        .expect("sum");
    loss.backward().expect("backward");

    let ga = a.grad().unwrap().expect("a.grad").data_vec().unwrap();
    let gb = b.grad().unwrap().expect("b.grad").data_vec().unwrap();

    assert_eq!(a.grad().unwrap().unwrap().shape(), &[1, 3]);
    assert_eq!(b.grad().unwrap().unwrap().shape(), &[2, 3]);
    assert_close(&ga, &[0.0, 1.0, -1.0], "broadcast grad a");
    assert_close(&gb, &[0.0, 1.0, -1.0, 0.0, 1.0, -1.0], "broadcast grad b");
}

#[test]
fn cross_reduced_precision_forward_keeps_dtype_contract() {
    let expected = [0.0, 0.0, 1.0, 0.0, -1.0, 0.0];

    let a16 = tensor_f16(&[1.0, 0.0, 0.0], &[1, 3], false);
    let b16 = tensor_f16(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[2, 3], false);
    let c16 = cross(&a16, &b16, -1).expect("f16 cross");
    assert_eq!(c16.shape(), &[2, 3]);
    assert_close_f32(
        &c16.data_vec()
            .unwrap()
            .into_iter()
            .map(|v| v.to_f32())
            .collect::<Vec<_>>(),
        &expected,
        0.0,
        "cpu f16 cross",
    );

    let ab = tensor_bf16(&[1.0, 0.0, 0.0], &[1, 3], false);
    let bb = tensor_bf16(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[2, 3], false);
    let cb = cross(&ab, &bb, -1).expect("bf16 cross");
    assert_eq!(cb.shape(), &[2, 3]);
    assert_close_f32(
        &cb.data_vec()
            .unwrap()
            .into_iter()
            .map(|v| v.to_f32())
            .collect::<Vec<_>>(),
        &expected,
        0.0,
        "cpu bf16 cross",
    );
}

#[test]
fn cross_reduced_precision_backward_sums_broadcast_axes() {
    let grad_a = [0.0, 1.0, -1.0];
    let grad_b = [0.0, 1.0, -1.0, 0.0, 1.0, -1.0];

    let a16 = tensor_f16(&[1.0, 0.0, 0.0], &[1, 3], true);
    let b16 = tensor_f16(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[2, 3], true);
    cross(&a16, &b16, -1)
        .expect("f16 cross")
        .sum_all()
        .expect("f16 sum")
        .backward()
        .expect("f16 backward");
    assert_close_f32(
        &a16.grad()
            .unwrap()
            .expect("f16 a.grad")
            .data_vec()
            .unwrap()
            .into_iter()
            .map(|v| v.to_f32())
            .collect::<Vec<_>>(),
        &grad_a,
        0.0,
        "cpu f16 grad a",
    );
    assert_close_f32(
        &b16.grad()
            .unwrap()
            .expect("f16 b.grad")
            .data_vec()
            .unwrap()
            .into_iter()
            .map(|v| v.to_f32())
            .collect::<Vec<_>>(),
        &grad_b,
        0.0,
        "cpu f16 grad b",
    );

    let ab = tensor_bf16(&[1.0, 0.0, 0.0], &[1, 3], true);
    let bb = tensor_bf16(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[2, 3], true);
    cross(&ab, &bb, -1)
        .expect("bf16 cross")
        .sum_all()
        .expect("bf16 sum")
        .backward()
        .expect("bf16 backward");
    assert_close_f32(
        &ab.grad()
            .unwrap()
            .expect("bf16 a.grad")
            .data_vec()
            .unwrap()
            .into_iter()
            .map(|v| v.to_f32())
            .collect::<Vec<_>>(),
        &grad_a,
        0.0,
        "cpu bf16 grad a",
    );
    assert_close_f32(
        &bb.grad()
            .unwrap()
            .expect("bf16 b.grad")
            .data_vec()
            .unwrap()
            .into_iter()
            .map(|v| v.to_f32())
            .collect::<Vec<_>>(),
        &grad_b,
        0.0,
        "cpu bf16 grad b",
    );
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::{assert_close, assert_close_f32, cross, tensor, tensor_bf16, tensor_f16};
    use ferrotorch_core::device::Device;
    use ferrotorch_core::storage::TensorStorage;
    use ferrotorch_core::tensor::Tensor;
    use half::{bf16, f16};
    use std::sync::Once;

    static INIT: Once = Once::new();

    fn ensure_cuda() {
        INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("FERROTORCH_ENABLE_GPU=1 and CUDA backend required")
        });
    }

    fn cuda_leaf(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        ensure_cuda();
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
            .unwrap()
            .to(Device::Cuda(0))
            .expect("upload")
            .requires_grad_(true)
    }

    fn cuda_leaf_f16(data: &[f32], shape: &[usize]) -> Tensor<f16> {
        ensure_cuda();
        tensor_f16(data, shape, false)
            .to(Device::Cuda(0))
            .expect("upload")
            .requires_grad_(true)
    }

    fn cuda_leaf_bf16(data: &[f32], shape: &[usize]) -> Tensor<bf16> {
        ensure_cuda();
        tensor_bf16(data, shape, false)
            .to(Device::Cuda(0))
            .expect("upload")
            .requires_grad_(true)
    }

    #[test]
    fn cuda_cross_f64_broadcast_forward_and_backward_stay_resident() {
        let a = cuda_leaf(&[1.0, 0.0, 0.0], &[1, 3]);
        let b = cuda_leaf(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[2, 3]);

        let c = cross(&a, &b, -1).expect("cuda broadcasted linalg.cross");
        assert_eq!(c.device(), Device::Cuda(0));
        assert_eq!(c.shape(), &[2, 3]);
        assert_close(
            &c.data_vec().unwrap(),
            &[0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
            "cuda cross broadcast forward",
        );

        c.sum_all().expect("sum").backward().expect("backward");
        let ga = a.grad().unwrap().expect("a.grad");
        let gb = b.grad().unwrap().expect("b.grad");
        assert_eq!(ga.device(), Device::Cuda(0));
        assert_eq!(gb.device(), Device::Cuda(0));
        assert_eq!(ga.shape(), &[1, 3]);
        assert_eq!(gb.shape(), &[2, 3]);
        assert_close(&ga.data_vec().unwrap(), &[0.0, 1.0, -1.0], "cuda grad a");
        assert_close(
            &gb.data_vec().unwrap(),
            &[0.0, 1.0, -1.0, 0.0, 1.0, -1.0],
            "cuda grad b",
        );
    }

    #[test]
    fn cuda_cross_f32_dim0_uses_kernel() {
        ensure_cuda();
        let a = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
            vec![3, 2],
            false,
        )
        .unwrap()
        .to(Device::Cuda(0))
        .expect("upload a");
        let b = Tensor::from_storage(
            TensorStorage::cpu(vec![6.0_f32, 5.0, 4.0, 3.0, 2.0, 1.0]),
            vec![3, 2],
            false,
        )
        .unwrap()
        .to(Device::Cuda(0))
        .expect("upload b");

        let c = cross(&a, &b, 0).expect("cuda f32 dim0 cross");
        assert_eq!(c.device(), Device::Cuda(0));
        assert_eq!(c.shape(), &[3, 2]);
        let got: Vec<f64> = c.data_vec().unwrap().into_iter().map(f64::from).collect();
        assert_close(
            &got,
            &[-14.0, -14.0, 28.0, 28.0, -14.0, -14.0],
            "cuda f32 dim0",
        );
    }

    #[test]
    fn cuda_cross_empty_broadcast_result_stays_on_device() {
        ensure_cuda();
        let a = tensor(&[], &[0, 3], false)
            .to(Device::Cuda(0))
            .expect("upload a");
        let b = tensor(&[1.0, 0.0, 0.0], &[1, 3], false)
            .to(Device::Cuda(0))
            .expect("upload b");

        let c = cross(&a, &b, -1).expect("empty cuda broadcast cross");
        assert_eq!(c.device(), Device::Cuda(0));
        assert_eq!(c.shape(), &[0, 3]);
        assert_eq!(c.data_vec().unwrap(), Vec::<f64>::new());
    }

    #[test]
    fn cuda_cross_f16_bf16_broadcast_backward_stays_resident() {
        let expected = [0.0, 0.0, 1.0, 0.0, -1.0, 0.0];
        let grad_a = [0.0, 1.0, -1.0];
        let grad_b = [0.0, 1.0, -1.0, 0.0, 1.0, -1.0];

        let a16 = cuda_leaf_f16(&[1.0, 0.0, 0.0], &[1, 3]);
        let b16 = cuda_leaf_f16(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[2, 3]);
        let c16 = cross(&a16, &b16, -1).expect("cuda f16 cross");
        assert_eq!(c16.device(), Device::Cuda(0));
        assert_close_f32(
            &c16.data_vec()
                .unwrap()
                .into_iter()
                .map(|v| v.to_f32())
                .collect::<Vec<_>>(),
            &expected,
            0.0,
            "cuda f16 forward",
        );
        c16.sum_all()
            .expect("sum f16")
            .backward()
            .expect("backward f16");
        let ga16 = a16.grad().unwrap().expect("f16 a.grad");
        let gb16 = b16.grad().unwrap().expect("f16 b.grad");
        assert_eq!(ga16.device(), Device::Cuda(0));
        assert_eq!(gb16.device(), Device::Cuda(0));
        assert_close_f32(
            &ga16
                .data_vec()
                .unwrap()
                .into_iter()
                .map(|v| v.to_f32())
                .collect::<Vec<_>>(),
            &grad_a,
            0.0,
            "cuda f16 grad a",
        );
        assert_close_f32(
            &gb16
                .data_vec()
                .unwrap()
                .into_iter()
                .map(|v| v.to_f32())
                .collect::<Vec<_>>(),
            &grad_b,
            0.0,
            "cuda f16 grad b",
        );

        let ab = cuda_leaf_bf16(&[1.0, 0.0, 0.0], &[1, 3]);
        let bb = cuda_leaf_bf16(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[2, 3]);
        let cb = cross(&ab, &bb, -1).expect("cuda bf16 cross");
        assert_eq!(cb.device(), Device::Cuda(0));
        assert_close_f32(
            &cb.data_vec()
                .unwrap()
                .into_iter()
                .map(|v| v.to_f32())
                .collect::<Vec<_>>(),
            &expected,
            0.0,
            "cuda bf16 forward",
        );
        cb.sum_all()
            .expect("sum bf16")
            .backward()
            .expect("backward bf16");
        let gab = ab.grad().unwrap().expect("bf16 a.grad");
        let gbb = bb.grad().unwrap().expect("bf16 b.grad");
        assert_eq!(gab.device(), Device::Cuda(0));
        assert_eq!(gbb.device(), Device::Cuda(0));
        assert_close_f32(
            &gab.data_vec()
                .unwrap()
                .into_iter()
                .map(|v| v.to_f32())
                .collect::<Vec<_>>(),
            &grad_a,
            0.0,
            "cuda bf16 grad a",
        );
        assert_close_f32(
            &gbb.data_vec()
                .unwrap()
                .into_iter()
                .map(|v| v.to_f32())
                .collect::<Vec<_>>(),
            &grad_b,
            0.0,
            "cuda bf16 grad b",
        );
    }
}
