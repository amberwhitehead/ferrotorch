//! Linalg first-group backward / storage-offset audit.
//!
//! PyTorch source contracts checked before this test was authored:
//! - `tools/autograd/derivatives.yaml:606-608`: `dot` VJP is
//!   `grad * other.conj()` / `grad * self.conj()`.
//! - `tools/autograd/derivatives.yaml:1218-1243`: `mm` and `mv` VJPs use
//!   matrix products with the opposite operand's transpose/conjugate.
//! - `tools/autograd/derivatives.yaml:383-384`: `matmul` delegates to
//!   `matmul_backward`.
//! - `tools/autograd/derivatives.yaml:378-380`:
//!   `self = grad.bmm(mat2.transpose(1, 2).conj())`,
//!   `mat2 = self.transpose(1, 2).conj().bmm(grad)`.
//! - `aten/src/ATen/native/LinearAlgebra.cpp:290-338`: `bmm` requires both
//!   inputs to be 3-D, matching batch sizes and contraction dimension, and
//!   returns `[batch, rows, cols]`.
//!
//! The CUDA cases target a real implementation hazard: a row-narrowed CUDA
//! tensor is C-contiguous by strides but carries a nonzero storage_offset.
//! PyTorch kernels honor that view offset; ferrotorch raw CUDA launchers need a
//! packed on-device buffer before raw linalg kernels read.

use ferrotorch_core::grad_fns::linalg::{
    BmmBackward, DotBackward, MatmulBackward, MmBackward, MvBackward,
};
use ferrotorch_core::{GradFn, Tensor, TensorStorage};

fn cpu_f32(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .unwrap()
}

fn assert_close(got: &[f32], want: &[f32], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-4,
            "{ctx}: element {i}: got {g}, want {w}"
        );
    }
}

#[test]
fn dot_backward_constructor_matches_torch_vjp() {
    let a = cpu_f32(&[2.0, 3.0, 4.0], &[3], true);
    let b = cpu_f32(&[5.0, 7.0, 11.0], &[3], true);
    let grad = cpu_f32(&[2.0], &[], false);

    let node = DotBackward::<f32>::new(a, b);
    let grads = node.backward(&grad).unwrap();
    assert_eq!(grads.len(), 2);
    assert_close(
        &grads[0].as_ref().unwrap().data_vec().unwrap(),
        &[10.0, 14.0, 22.0],
        "dot dA",
    );
    assert_close(
        &grads[1].as_ref().unwrap().data_vec().unwrap(),
        &[4.0, 6.0, 8.0],
        "dot dB",
    );
}

#[test]
fn mv_backward_constructor_matches_torch_vjp() {
    let a = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    let x = cpu_f32(&[7.0, 11.0, 13.0], &[3], true);
    let grad = cpu_f32(&[2.0, 3.0], &[2], false);

    let node = MvBackward::<f32>::new(a, x);
    let grads = node.backward(&grad).unwrap();
    assert_eq!(grads.len(), 2);
    assert_close(
        &grads[0].as_ref().unwrap().data_vec().unwrap(),
        &[14.0, 22.0, 26.0, 21.0, 33.0, 39.0],
        "mv dA",
    );
    assert_close(
        &grads[1].as_ref().unwrap().data_vec().unwrap(),
        &[14.0, 19.0, 24.0],
        "mv dX",
    );
}

#[test]
fn mm_backward_constructor_matches_torch_vjp() {
    let a = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    let b = cpu_f32(&[7.0, 11.0, 13.0, 17.0, 19.0, 23.0], &[3, 2], true);
    let grad = cpu_f32(&[2.0, 3.0, 5.0, 7.0], &[2, 2], false);

    let node = MmBackward::<f32>::new(a, b);
    let grads = node.backward(&grad).unwrap();
    assert_eq!(grads.len(), 2);
    assert_close(
        &grads[0].as_ref().unwrap().data_vec().unwrap(),
        &[47.0, 77.0, 107.0, 112.0, 184.0, 256.0],
        "mm dA",
    );
    assert_close(
        &grads[1].as_ref().unwrap().data_vec().unwrap(),
        &[22.0, 31.0, 29.0, 41.0, 36.0, 51.0],
        "mm dB",
    );
}

#[test]
fn matmul_vm_backward_constructor_matches_torch_vjp() {
    let a = cpu_f32(&[2.0, 3.0], &[2], true);
    let b = cpu_f32(&[5.0, 7.0, 11.0, 13.0], &[2, 2], true);
    let grad = cpu_f32(&[17.0, 19.0], &[2], false);

    let node = MatmulBackward::<f32>::new(a, b);
    let grads = node.backward(&grad).unwrap();
    assert_eq!(grads.len(), 2);
    assert_close(
        &grads[0].as_ref().unwrap().data_vec().unwrap(),
        &[218.0, 434.0],
        "matmul vm dA",
    );
    assert_close(
        &grads[1].as_ref().unwrap().data_vec().unwrap(),
        &[34.0, 38.0, 51.0, 57.0],
        "matmul vm dB",
    );
}

#[test]
fn bmm_backward_constructor_matches_torch_vjp() {
    let a = cpu_f32(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[2, 2, 3],
        true,
    );
    let b = cpu_f32(
        &[2.0, 0.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        &[2, 3, 2],
        true,
    );
    let grad = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2], false);

    let node = BmmBackward::<f32>::new(a, b);
    let grads = node.backward(&grad).unwrap();
    assert_eq!(grads.len(), 2);
    assert_close(
        &grads[0].as_ref().unwrap().data_vec().unwrap(),
        &[
            2.0, 7.0, 14.0, 6.0, 15.0, 32.0, 72.0, 94.0, 116.0, 98.0, 128.0, 158.0,
        ],
        "dA",
    );
    assert_close(
        &grads[1].as_ref().unwrap().data_vec().unwrap(),
        &[
            13.0, 18.0, 17.0, 24.0, 21.0, 30.0, 105.0, 122.0, 117.0, 136.0, 129.0, 150.0,
        ],
        "dB",
    );
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use ferrotorch_core::Device;
    use ferrotorch_core::grad_fns::linalg::{bmm, bmm_differentiable};
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for bmm storage-offset audit");
        });
    }

    fn to_cuda(t: Tensor<f32>, requires_grad: bool) -> Tensor<f32> {
        t.to(Device::Cuda(0)).unwrap().requires_grad_(requires_grad)
    }

    fn host(t: &Tensor<f32>) -> Vec<f32> {
        t.cpu().unwrap().data_vec().unwrap()
    }

    fn full_a(requires_grad: bool) -> Tensor<f32> {
        let data: Vec<f32> = (1..=18).map(|v| v as f32).collect();
        to_cuda(cpu_f32(&data, &[3, 2, 3], false), requires_grad)
    }

    fn full_b(requires_grad: bool) -> Tensor<f32> {
        let data: Vec<f32> = (101..=118).map(|v| v as f32).collect();
        to_cuda(cpu_f32(&data, &[3, 3, 2], false), requires_grad)
    }

    #[test]
    fn cuda_dot_backward_narrowed_offset_constructor_matches_torch_and_stays_resident() {
        ensure_cuda_backend();
        let a_full = to_cuda(cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5], true), true);
        let b_full = to_cuda(cpu_f32(&[7.0, 11.0, 13.0, 17.0, 19.0], &[5], true), true);
        let a = a_full.narrow(0, 1, 3).unwrap();
        let b = b_full.narrow(0, 1, 3).unwrap();
        let grad = to_cuda(cpu_f32(&[2.0], &[], false), false);
        assert!(a.is_contiguous() && a.storage_offset() != 0);
        assert!(b.is_contiguous() && b.storage_offset() != 0);

        let node = DotBackward::<f32>::new(a, b);
        let grads = node.backward(&grad).unwrap();
        let ga = grads[0].as_ref().unwrap();
        let gb = grads[1].as_ref().unwrap();
        assert!(ga.is_cuda(), "dot dA must stay CUDA-resident");
        assert!(gb.is_cuda(), "dot dB must stay CUDA-resident");
        assert_close(&host(ga), &[22.0, 26.0, 34.0], "offset dot dA");
        assert_close(&host(gb), &[4.0, 6.0, 8.0], "offset dot dB");
    }

    #[test]
    fn cuda_mv_backward_narrowed_offset_constructor_matches_torch_and_stays_resident() {
        ensure_cuda_backend();
        let a_full = to_cuda(
            cpu_f32(
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                &[3, 3],
                true,
            ),
            true,
        );
        let x_full = to_cuda(cpu_f32(&[7.0, 11.0, 13.0, 17.0], &[4], true), true);
        let a = a_full.narrow(0, 1, 2).unwrap();
        let x = x_full.narrow(0, 1, 3).unwrap();
        let grad = to_cuda(cpu_f32(&[2.0, 3.0], &[2], false), false);
        assert!(a.is_contiguous() && a.storage_offset() != 0);
        assert!(x.is_contiguous() && x.storage_offset() != 0);

        let node = MvBackward::<f32>::new(a, x);
        let grads = node.backward(&grad).unwrap();
        let ga = grads[0].as_ref().unwrap();
        let gx = grads[1].as_ref().unwrap();
        assert!(ga.is_cuda(), "mv dA must stay CUDA-resident");
        assert!(gx.is_cuda(), "mv dX must stay CUDA-resident");
        assert_close(
            &host(ga),
            &[22.0, 26.0, 34.0, 33.0, 39.0, 51.0],
            "offset mv dA",
        );
        assert_close(&host(gx), &[29.0, 34.0, 39.0], "offset mv dX");
    }

    #[test]
    fn cuda_mm_backward_narrowed_offset_constructor_matches_torch_and_stays_resident() {
        ensure_cuda_backend();
        let a_full = to_cuda(
            cpu_f32(
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                &[3, 3],
                true,
            ),
            true,
        );
        let b_full = to_cuda(
            cpu_f32(
                &[101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
                &[4, 2],
                true,
            ),
            true,
        );
        let a = a_full.narrow(0, 1, 2).unwrap();
        let b = b_full.narrow(0, 1, 3).unwrap();
        let grad = to_cuda(cpu_f32(&[2.0, 3.0, 5.0, 7.0], &[2, 2], false), false);
        assert!(a.is_contiguous() && a.storage_offset() != 0);
        assert!(b.is_contiguous() && b.storage_offset() != 0);

        let node = MmBackward::<f32>::new(a, b);
        let grads = node.backward(&grad).unwrap();
        let ga = grads[0].as_ref().unwrap();
        let gb = grads[1].as_ref().unwrap();
        assert!(ga.is_cuda(), "mm dA must stay CUDA-resident");
        assert!(gb.is_cuda(), "mm dB must stay CUDA-resident");
        assert_close(
            &host(ga),
            &[518.0, 528.0, 538.0, 1243.0, 1267.0, 1291.0],
            "offset mm dA",
        );
        assert_close(
            &host(gb),
            &[43.0, 61.0, 50.0, 71.0, 57.0, 81.0],
            "offset mm dB",
        );
    }

    #[test]
    fn cuda_matmul_vm_backward_narrowed_offset_constructor_matches_torch_and_stays_resident() {
        ensure_cuda_backend();
        let a_full = to_cuda(cpu_f32(&[1.0, 2.0, 3.0], &[3], true), true);
        let b_full = to_cuda(
            cpu_f32(&[5.0, 7.0, 13.0, 17.0, 19.0, 23.0], &[3, 2], true),
            true,
        );
        let a = a_full.narrow(0, 1, 2).unwrap();
        let b = b_full.narrow(0, 1, 2).unwrap();
        let grad = to_cuda(cpu_f32(&[2.0, 3.0], &[2], false), false);
        assert!(a.is_contiguous() && a.storage_offset() != 0);
        assert!(b.is_contiguous() && b.storage_offset() != 0);

        let node = MatmulBackward::<f32>::new(a, b);
        let grads = node.backward(&grad).unwrap();
        let ga = grads[0].as_ref().unwrap();
        let gb = grads[1].as_ref().unwrap();
        assert!(ga.is_cuda(), "matmul vm dA must stay CUDA-resident");
        assert!(gb.is_cuda(), "matmul vm dB must stay CUDA-resident");
        assert_close(&host(ga), &[77.0, 107.0], "offset matmul vm dA");
        assert_close(&host(gb), &[4.0, 6.0, 6.0, 9.0], "offset matmul vm dB");
    }

    #[test]
    fn cuda_bmm_narrowed_offset_inputs_match_torch_and_stay_resident() {
        ensure_cuda_backend();
        let a = full_a(false).narrow(0, 1, 2).unwrap();
        let b = full_b(false).narrow(0, 1, 2).unwrap();
        assert!(a.is_contiguous() && a.storage_offset() != 0);
        assert!(b.is_contiguous() && b.storage_offset() != 0);

        let out = bmm(&a, &b).unwrap();
        assert!(out.is_cuda(), "bmm result must stay CUDA-resident");
        assert_close(
            &host(&out),
            &[
                2620.0, 2644.0, 3601.0, 3634.0, 4834.0, 4876.0, 5869.0, 5920.0,
            ],
            "narrowed bmm forward",
        );
    }

    #[test]
    fn cuda_bmm_backward_narrowed_offset_inputs_match_torch_and_stay_resident() {
        ensure_cuda_backend();
        let a_full = full_a(true);
        let b_full = full_b(true);
        let a = a_full.narrow(0, 1, 2).unwrap();
        let b = b_full.narrow(0, 1, 2).unwrap();
        assert!(a.is_contiguous() && a.storage_offset() != 0);
        assert!(b.is_contiguous() && b.storage_offset() != 0);

        let out = bmm_differentiable(&a, &b).unwrap();
        assert!(out.is_cuda(), "tracked bmm result must stay CUDA-resident");
        out.sum_all().unwrap().backward().unwrap();

        let ga = a_full.grad().unwrap().unwrap();
        let gb = b_full.grad().unwrap().unwrap();
        assert!(ga.is_cuda(), "A full gradient must stay CUDA-resident");
        assert!(gb.is_cuda(), "B full gradient must stay CUDA-resident");
        assert_close(
            &host(&ga),
            &[
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 215.0, 219.0, 223.0, 215.0, 219.0, 223.0, 227.0,
                231.0, 235.0, 227.0, 231.0, 235.0,
            ],
            "narrowed bmm dA through full leaf",
        );
        assert_close(
            &host(&gb),
            &[
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 17.0, 17.0, 19.0, 19.0, 21.0, 21.0, 29.0, 29.0, 31.0,
                31.0, 33.0, 33.0,
            ],
            "narrowed bmm dB through full leaf",
        );
    }
}
