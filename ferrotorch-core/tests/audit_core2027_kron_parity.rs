//! Kronecker product parity audit.
//!
//! PyTorch reference inspected before implementation:
//! `aten/src/ATen/native/LinearAlgebra.cpp:3476-3531` (`KronImpl`) pads the
//! shorter input rank with leading singleton dimensions, reshapes inputs to
//! alternating `[a_i, 1]` and `[1, b_i]` axes, multiplies those views, and
//! views the product as `[a_i * b_i]`.

use ferrotorch_core::grad_fns::linalg::{KronBackward, kron_differentiable};
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

fn numel(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    }
}

fn padded_shape(shape: &[usize], rank: usize) -> Vec<usize> {
    let mut out = vec![1; rank.saturating_sub(shape.len())];
    out.extend_from_slice(shape);
    out
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut out = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        out[i] = out[i + 1] * shape[i + 1];
    }
    out
}

fn unravel(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let s = strides(shape);
    let mut out = vec![0; shape.len()];
    for (i, (&dim, &stride)) in shape.iter().zip(s.iter()).enumerate() {
        if dim == 0 {
            return out;
        }
        out[i] = flat / stride;
        flat %= stride;
    }
    out
}

fn flat_index(coords: &[usize], shape: &[usize]) -> usize {
    coords
        .iter()
        .zip(strides(shape).iter())
        .map(|(&coord, &stride)| coord * stride)
        .sum()
}

fn kron_reference(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> (Vec<f32>, Vec<usize>) {
    let rank = a_shape.len().max(b_shape.len());
    let ap = padded_shape(a_shape, rank);
    let bp = padded_shape(b_shape, rank);
    let out_shape: Vec<usize> = ap.iter().zip(bp.iter()).map(|(&ad, &bd)| ad * bd).collect();
    let mut out = vec![0.0; numel(&out_shape)];

    for (flat, slot) in out.iter_mut().enumerate() {
        let coord = unravel(flat, &out_shape);
        let mut ac = vec![0; rank];
        let mut bc = vec![0; rank];
        for axis in 0..rank {
            ac[axis] = coord[axis] / bp[axis];
            bc[axis] = coord[axis] % bp[axis];
        }
        *slot = a[flat_index(&ac, &ap)] * b[flat_index(&bc, &bp)];
    }

    (out, out_shape)
}

fn kron_backward_reference(
    grad: &[f32],
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> (Vec<f32>, Vec<f32>) {
    let rank = a_shape.len().max(b_shape.len());
    let ap = padded_shape(a_shape, rank);
    let bp = padded_shape(b_shape, rank);
    let out_shape: Vec<usize> = ap.iter().zip(bp.iter()).map(|(&ad, &bd)| ad * bd).collect();
    let mut ga = vec![0.0; a.len()];
    let mut gb = vec![0.0; b.len()];

    for (flat, &g) in grad.iter().enumerate() {
        let coord = unravel(flat, &out_shape);
        let mut ac = vec![0; rank];
        let mut bc = vec![0; rank];
        for axis in 0..rank {
            ac[axis] = coord[axis] / bp[axis];
            bc[axis] = coord[axis] % bp[axis];
        }
        let ai = flat_index(&ac, &ap);
        let bi = flat_index(&bc, &bp);
        ga[ai] += g * b[bi];
        gb[bi] += g * a[ai];
    }

    (ga, gb)
}

fn assert_close(got: &[f32], want: &[f32], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= 1e-4,
            "{ctx}: element {i}: got {g}, want {w}"
        );
    }
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

#[test]
fn kron_public_mixed_rank_forward_backward_matches_reference() {
    let a_shape = [2usize, 1, 3];
    let b_shape = [4usize, 5];
    let a_data = [1.0, -2.0, 3.0, 4.0, -5.0, 6.0];
    let b_data: Vec<f32> = (1..=20).map(|v| v as f32 * 0.25 - 2.0).collect();
    let (want, out_shape) = kron_reference(&a_data, &a_shape, &b_data, &b_shape);
    let grad_data: Vec<f32> = (0..want.len()).map(|i| (i as f32 % 11.0) - 5.0).collect();
    let (want_ga, want_gb) =
        kron_backward_reference(&grad_data, &a_data, &a_shape, &b_data, &b_shape);

    let a = cpu(&a_data, &a_shape, true);
    let b = cpu(&b_data, &b_shape, true);
    let out = linalg::kron(&a, &b).unwrap();
    let node = out.grad_fn().expect("kron must attach a grad_fn");
    assert_eq!(node.name(), short_type_name::<KronBackward<f32>>());
    assert_eq!(out.shape(), out_shape.as_slice());
    assert_close(&out.data_vec().unwrap(), &want, "kron mixed-rank forward");

    out.backward_with_gradient(&cpu(&grad_data, &out_shape, false))
        .unwrap();
    assert_close(
        &a.grad().unwrap().unwrap().data_vec().unwrap(),
        &want_ga,
        "kron mixed-rank dA",
    );
    assert_close(
        &b.grad().unwrap().unwrap().data_vec().unwrap(),
        &want_gb,
        "kron mixed-rank dB",
    );
}

#[test]
fn kron_differentiable_supports_scalar_and_vector_inputs() {
    let scalar = cpu(&[3.0], &[], true);
    let vector = cpu(&[2.0, -4.0, 5.0], &[3], true);
    let out = kron_differentiable(&scalar, &vector).unwrap();
    assert_eq!(out.shape(), &[3]);
    assert_close(
        &out.data_vec().unwrap(),
        &[6.0, -12.0, 15.0],
        "scalar kron vector",
    );

    out.backward_with_gradient(&cpu(&[1.0, 2.0, 3.0], &[3], false))
        .unwrap();
    assert_close(
        &scalar.grad().unwrap().unwrap().data_vec().unwrap(),
        &[9.0],
        "scalar kron vector dscalar",
    );
    assert_close(
        &vector.grad().unwrap().unwrap().data_vec().unwrap(),
        &[3.0, 6.0, 9.0],
        "scalar kron vector dvector",
    );

    let lhs = cpu(&[2.0], &[], false);
    let rhs = cpu(&[-7.0], &[], false);
    let scalar_out = kron_differentiable(&lhs, &rhs).unwrap();
    assert_eq!(scalar_out.shape(), &[] as &[usize]);
    assert_close(
        &scalar_out.data_vec().unwrap(),
        &[-14.0],
        "scalar kron scalar",
    );
}

#[test]
fn kron_empty_dimension_matches_pytorch_shape() {
    let a = cpu(&[], &[0, 2], false);
    let b = cpu(&[1.0, 2.0, 3.0], &[3], false);
    let out = linalg::kron(&a, &b).unwrap();
    assert_eq!(out.shape(), &[0, 6]);
    assert!(out.data_vec().unwrap().is_empty());
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use ferrotorch_core::Device;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for kron audit");
        });
    }

    fn cuda(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
        cpu(data, shape, false)
            .to(Device::Cuda(0))
            .unwrap()
            .requires_grad_(requires_grad)
    }

    fn host(t: &Tensor<f32>) -> Vec<f32> {
        t.cpu().unwrap().data_vec().unwrap()
    }

    #[test]
    fn cuda_kron_mixed_rank_forward_backward_matches_reference_and_stays_resident() {
        ensure_cuda_backend();
        let a_shape = [2usize, 3];
        let b_shape = [4usize];
        let a_data = [1.0, -2.0, 3.0, 4.0, -5.0, 6.0];
        let b_data = [0.5, -1.0, 1.5, -2.0];
        let (want, out_shape) = kron_reference(&a_data, &a_shape, &b_data, &b_shape);
        let grad_data: Vec<f32> = (0..want.len()).map(|i| (i as f32 % 7.0) - 3.0).collect();
        let (want_ga, want_gb) =
            kron_backward_reference(&grad_data, &a_data, &a_shape, &b_data, &b_shape);

        let a = cuda(&a_data, &a_shape, true);
        let b = cuda(&b_data, &b_shape, true);
        let out = linalg::kron(&a, &b).unwrap();
        assert!(out.is_cuda(), "kron forward must stay CUDA-resident");
        assert_eq!(out.shape(), out_shape.as_slice());
        assert_close(&host(&out), &want, "cuda kron forward");

        out.backward_with_gradient(&cuda(&grad_data, &out_shape, false))
            .unwrap();
        let ga = a.grad().unwrap().unwrap();
        let gb = b.grad().unwrap().unwrap();
        assert!(ga.is_cuda(), "kron dA must stay CUDA-resident");
        assert!(gb.is_cuda(), "kron dB must stay CUDA-resident");
        assert_close(&host(&ga), &want_ga, "cuda kron dA");
        assert_close(&host(&gb), &want_gb, "cuda kron dB");
    }

    #[test]
    fn cuda_kron_rejects_cpu_cuda_mixed_devices_like_pytorch() {
        ensure_cuda_backend();
        let cpu_scalar = cpu(&[2.0], &[], false);
        let cuda_vector = cuda(&[1.0, 2.0], &[2], false);
        assert!(
            linalg::kron(&cpu_scalar, &cuda_vector).is_err(),
            "torch.kron rejects CPU/CUDA mixed tensor inputs"
        );
    }
}
