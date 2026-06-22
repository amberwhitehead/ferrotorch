//! Fused-affine linalg conformance audit.
//!
//! PyTorch contracts inspected before this test was authored:
//! - `tools/autograd/derivatives.yaml:238`: `addbmm` VJP uses
//!   `grad.unsqueeze(0).expand(...).bmm(batch2.transpose(1, 2))` and
//!   `batch1.transpose(1, 2).bmm(expanded_grad)`.
//! - `tools/autograd/derivatives.yaml:256`: `addmm` VJP is
//!   `self=beta*grad`, `mat1=alpha*(grad @ mat2.T)`,
//!   `mat2=alpha*(mat1.T @ grad)`.
//! - `tools/autograd/derivatives.yaml:267`: `addmv` VJP is
//!   `self=beta*grad`, `mat=alpha*grad.ger(vec)`,
//!   `vec=alpha*(mat.T @ grad)`.
//! - `tools/autograd/derivatives.yaml:273`: `addr` VJP is
//!   `self=beta*grad`, `vec1=alpha*(grad @ vec2)`,
//!   `vec2=alpha*(grad.T @ vec1)`.
//! - `tools/autograd/derivatives.yaml:359`: `baddbmm` VJP is
//!   `self=beta*grad`, `batch1=alpha*grad.bmm(batch2.transpose(1, 2))`,
//!   `batch2=alpha*batch1.transpose(1, 2).bmm(grad)`.
//! - `aten/src/ATen/native/LinearAlgebra.cpp:1200-1239,1606-1620`:
//!   `self`/bias is broadcast to the exact output shape, and `beta == 0`
//!   skips reading self values without skipping shape validation.
//!
//! These tests specifically protect against the previous implementation style:
//! raw `.data()?` host loops in fused forwards/backwards, which rejected CUDA
//! tensors or forced a CPU round trip instead of composing resident linalg ops.

use ferrotorch_core::grad_fns::linalg::{
    AddbmmBackward, AddmmBackward, AddmvBackward, AddrBackward, BaddbmmBackward,
};
use ferrotorch_core::linalg as linalg_fwd;
use ferrotorch_core::{Tensor, TensorStorage};

fn cpu(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .unwrap()
}

#[cfg(feature = "gpu")]
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

fn assert_grad_node<N>(out: &Tensor<f32>) {
    let node = out.grad_fn().expect("tracked output must have grad_fn");
    assert_eq!(node.name(), short_type_name::<N>());
}

#[test]
fn fused_affine_forwards_attach_the_public_backward_nodes() {
    let bias_2d = cpu(&[0.5, -1.0, 1.5, -2.0], &[2, 2], true);
    let bias_1d = cpu(&[0.5, -1.0], &[2], true);
    let bias_3d = cpu(
        &[0.5, -1.0, 1.5, -2.0, 2.5, -3.0, 3.5, -4.0],
        &[2, 2, 2],
        true,
    );
    let mat1 = cpu(&[1.0, 2.0, 3.0, 4.0], &[2, 2], true);
    let mat2 = cpu(&[5.0, 6.0, 7.0, 8.0], &[2, 2], true);
    let vec1 = cpu(&[2.0, 3.0], &[2], true);
    let vec2 = cpu(&[5.0, 7.0], &[2], true);
    let batch1 = cpu(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2], true);
    let batch2 = cpu(&[2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 3.0, 4.0], &[2, 2, 2], true);

    assert_grad_node::<AddmmBackward<f32>>(
        &linalg_fwd::addmm(&bias_2d, &mat1, &mat2, 0.5, 1.5).unwrap(),
    );
    assert_grad_node::<AddmvBackward<f32>>(
        &linalg_fwd::addmv(&bias_1d, &mat1, &vec1, 0.5, 1.5).unwrap(),
    );
    assert_grad_node::<AddrBackward<f32>>(
        &linalg_fwd::addr(&bias_2d, &vec1, &vec2, 0.5, 1.5).unwrap(),
    );
    assert_grad_node::<BaddbmmBackward<f32>>(
        &linalg_fwd::baddbmm(&bias_3d, &batch1, &batch2, 0.5, 1.5).unwrap(),
    );
    assert_grad_node::<AddbmmBackward<f32>>(
        &linalg_fwd::addbmm(&bias_2d, &batch1, &batch2, 0.5, 1.5).unwrap(),
    );
}

#[test]
fn beta_zero_still_validates_bias_broadcast_shape() {
    let bad_2d = cpu(&[1.0, 2.0, 3.0], &[3], false);
    let mat1 = cpu(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    let mat2 = cpu(&[5.0, 6.0, 7.0, 8.0], &[2, 2], false);
    assert!(
        linalg_fwd::addmm(&bad_2d, &mat1, &mat2, 0.0, 1.0).is_err(),
        "torch validates addmm self broadcast shape even when beta == 0"
    );

    let bad_3d = cpu(&[1.0, 2.0, 3.0], &[3], false);
    let batch1 = cpu(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2], false);
    let batch2 = cpu(&[2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 3.0, 4.0], &[2, 2, 2], false);
    assert!(
        linalg_fwd::baddbmm(&bad_3d, &batch1, &batch2, 0.0, 1.0).is_err(),
        "torch validates baddbmm self broadcast shape even when beta == 0"
    );
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
                .expect("CUDA backend must initialize for fused-affine audit");
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
    fn cuda_addmm_forward_backward_matches_torch_vjp_and_stays_resident() {
        ensure_cuda_backend();
        let bias = cuda(&[1.0, -2.0], &[1, 2], true);
        let mat1 = cuda(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let mat2 = cuda(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2], true);
        let out = linalg_fwd::addmm(&bias, &mat1, &mat2, 0.5, 2.0).unwrap();
        assert!(out.is_cuda(), "addmm forward must stay CUDA-resident");
        assert_close(&host(&out), &[116.5, 127.0, 278.5, 307.0], "addmm out");

        out.backward_with_gradient(&cuda(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false))
            .unwrap();
        let gb = bias.grad().unwrap().unwrap();
        let gm1 = mat1.grad().unwrap().unwrap();
        let gm2 = mat2.grad().unwrap().unwrap();
        assert!(gb.is_cuda() && gm1.is_cuda() && gm2.is_cuda());
        assert_close(&host(&gb), &[2.0, 3.0], "addmm dbias");
        assert_close(
            &host(&gm1),
            &[46.0, 58.0, 70.0, 106.0, 134.0, 162.0],
            "addmm dmat1",
        );
        assert_close(
            &host(&gm2),
            &[26.0, 36.0, 34.0, 48.0, 42.0, 60.0],
            "addmm dmat2",
        );
    }

    #[test]
    fn cuda_addmv_forward_backward_matches_torch_vjp_and_stays_resident() {
        ensure_cuda_backend();
        let bias = cuda(&[1.0], &[1], true);
        let mat = cuda(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], true);
        let vec = cuda(&[7.0, 8.0], &[2], true);
        let out = linalg_fwd::addmv(&bias, &mat, &vec, 0.5, 1.5).unwrap();
        assert!(out.is_cuda(), "addmv forward must stay CUDA-resident");
        assert_close(&host(&out), &[35.0, 80.0, 125.0], "addmv out");

        out.backward_with_gradient(&cuda(&[2.0, 3.0, 4.0], &[3], false))
            .unwrap();
        let gb = bias.grad().unwrap().unwrap();
        let gm = mat.grad().unwrap().unwrap();
        let gv = vec.grad().unwrap().unwrap();
        assert!(gb.is_cuda() && gm.is_cuda() && gv.is_cuda());
        assert_close(&host(&gb), &[4.5], "addmv dbias");
        assert_close(
            &host(&gm),
            &[21.0, 24.0, 31.5, 36.0, 42.0, 48.0],
            "addmv dmat",
        );
        assert_close(&host(&gv), &[46.5, 60.0], "addmv dvec");
    }

    #[test]
    fn cuda_addr_forward_backward_matches_torch_vjp_and_stays_resident() {
        ensure_cuda_backend();
        let bias = cuda(&[1.0, -2.0], &[2, 1], true);
        let vec1 = cuda(&[2.0, 3.0], &[2], true);
        let vec2 = cuda(&[5.0, 7.0, 11.0], &[3], true);
        let out = linalg_fwd::addr(&bias, &vec1, &vec2, 0.25, 2.0).unwrap();
        assert!(out.is_cuda(), "addr forward must stay CUDA-resident");
        assert_close(
            &host(&out),
            &[20.25, 28.25, 44.25, 29.5, 41.5, 65.5],
            "addr out",
        );

        out.backward_with_gradient(&cuda(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false))
            .unwrap();
        let gb = bias.grad().unwrap().unwrap();
        let gv1 = vec1.grad().unwrap().unwrap();
        let gv2 = vec2.grad().unwrap().unwrap();
        assert!(gb.is_cuda() && gv1.is_cuda() && gv2.is_cuda());
        assert_close(&host(&gb), &[1.5, 3.75], "addr dbias");
        assert_close(&host(&gv1), &[104.0, 242.0], "addr dvec1");
        assert_close(&host(&gv2), &[28.0, 38.0, 48.0], "addr dvec2");
    }

    #[test]
    fn cuda_baddbmm_forward_backward_matches_torch_vjp_and_stays_resident() {
        ensure_cuda_backend();
        let bias = cuda(&[1.0, -2.0, 3.0, -4.0], &[1, 2, 2], true);
        let batch1 = cuda(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2], true);
        let batch2 = cuda(&[2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 3.0, 4.0], &[2, 2, 2], true);
        let out = linalg_fwd::baddbmm(&bias, &batch1, &batch2, 0.5, 1.0).unwrap();
        assert!(out.is_cuda(), "baddbmm forward must stay CUDA-resident");
        assert_close(
            &host(&out),
            &[4.5, 3.0, 11.5, 6.0, 18.5, 28.0, 25.5, 37.0],
            "baddbmm out",
        );

        out.backward_with_gradient(&cuda(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 2, 2],
            false,
        ))
        .unwrap();
        let gb = bias.grad().unwrap().unwrap();
        let gb1 = batch1.grad().unwrap().unwrap();
        let gb2 = batch2.grad().unwrap().unwrap();
        assert!(gb.is_cuda() && gb1.is_cuda() && gb2.is_cuda());
        assert_close(&host(&gb), &[3.0, 4.0, 5.0, 6.0], "baddbmm dbias");
        assert_close(
            &host(&gb1),
            &[2.0, 5.0, 6.0, 11.0, 6.0, 39.0, 8.0, 53.0],
            "baddbmm db1",
        );
        assert_close(
            &host(&gb2),
            &[10.0, 14.0, 14.0, 20.0, 74.0, 86.0, 86.0, 100.0],
            "baddbmm db2",
        );
    }

    #[test]
    fn cuda_addbmm_forward_backward_matches_torch_vjp_and_stays_resident() {
        ensure_cuda_backend();
        let bias = cuda(&[1.0, -2.0], &[1, 2], true);
        let batch1 = cuda(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2], true);
        let batch2 = cuda(&[2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 3.0, 4.0], &[2, 2, 2], true);
        let out = linalg_fwd::addbmm(&bias, &batch1, &batch2, 0.5, 1.0).unwrap();
        assert!(out.is_cuda(), "addbmm forward must stay CUDA-resident");
        assert_close(&host(&out), &[22.5, 32.0, 34.5, 46.0], "addbmm out");

        out.backward_with_gradient(&cuda(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false))
            .unwrap();
        let gb = bias.grad().unwrap().unwrap();
        let gb1 = batch1.grad().unwrap().unwrap();
        let gb2 = batch2.grad().unwrap().unwrap();
        assert!(gb.is_cuda() && gb1.is_cuda() && gb2.is_cuda());
        assert_close(&host(&gb), &[2.0, 3.0], "addbmm dbias");
        assert_close(
            &host(&gb1),
            &[2.0, 5.0, 6.0, 11.0, 2.0, 11.0, 4.0, 25.0],
            "addbmm db1",
        );
        assert_close(
            &host(&gb2),
            &[10.0, 14.0, 14.0, 20.0, 26.0, 38.0, 30.0, 44.0],
            "addbmm db2",
        );
    }
}
