//! Red-then-green regression tests for audit finding CORE-074 (crosslink
//! #1768): sparse operations on dense tensors silently sever autograd
//! (CLASS-S — `SparseTensor::spmm` and `sparse_matmul_24` built outputs
//! with `from_storage(..., false)` on every lane, and
//! `SemiStructuredSparseTensor::compress` extracted a gradient-tracking
//! tensor into plain vectors; a loss through these APIs could not train
//! the dense operand).
//!
//! Observed at HEAD (red run, 2026-06-12): `spmm` output had no grad_fn
//! (`backward_with_gradient` errored / left `d.grad()` empty);
//! `sparse_matmul_24` likewise; `compress` of a `requires_grad` tensor
//! silently succeeded detached.
//!
//! torch oracles (live session, torch 2.11.0+cu130):
//!
//! ```python
//! # (1) torch.sparse.mm DOES flow gradient to the dense operand:
//! >>> sp = torch.sparse_coo_tensor(torch.tensor([[0,0,1],[0,2,1]]),
//! ...                              torch.tensor([1.,2.,3.]), (2,3))
//! >>> d = torch.tensor([[1.,4.],[2.,5.],[3.,6.]], requires_grad=True)
//! >>> out = torch.sparse.mm(sp, d)         # [[7,16],[6,15]]
//! >>> (out * torch.tensor([[1.,2.],[3.,4.]])).sum().backward()
//! >>> d.grad                                # = sp^T @ w
//! tensor([[ 1.,  2.], [ 9., 12.], [ 2.,  4.]])
//!
//! # (2) matmul against the 2:4-masked weight, grad wrt the dense a:
//! >>> bm  # 2:4 mask of [[1,4,2,3],[-5,2,0,1],[.5,-.25,8,7],[3,6,-2,.125]]
//! [[0,4,0,3],[-5,2,0,0],[0,0,8,7],[3,6,0,0]]
//! >>> a = torch.tensor([[1.,2.,3.,4.],[0.5,-1.,2.,-0.25]], requires_grad=True)
//! >>> out = a @ bm                          # [[2,32,24,24],[4.25,-1.5,16,15.5]]
//! >>> out.backward(torch.tensor([[1.,2.,3.,4.],[5.,6.,7.,8.]]))
//! >>> a.grad                                # = g @ bm^T
//! tensor([[ 20.,  -1.,  52.,  15.], [ 48., -13., 112.,  51.]])
//!
//! # (3) semi-structured autograd errors upstream (CUTLASS, 2.11.0+cu130):
//! >>> torch.mm(to_sparse_semi_structured(wd), x_requires_grad)
//! NotImplementedError: `SparseSemiStructuredTensorCUTLASS` matmul:
//! operation is not supported
//! ```
//!
//! Post-fix contract: spmm and sparse_matmul_24 attach real backward
//! edges for the dense operand (path a); compress on a tracked input
//! returns a structured error (path b — trainable 2:4 weights tracked
//! in #1969), so no silent detach remains.

use ferrotorch_core::{
    FerrotorchError, SemiStructuredSparseTensor, SparseTensor, Tensor, TensorStorage,
    sparse_matmul_24,
};

fn mk_f32(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data), shape, requires_grad).unwrap()
}

/// spmm gradient reaches the dense LEAF with torch's values
/// (oracle 1: d.grad = sp^T @ w = [[1,2],[9,12],[2,4]]).
#[test]
fn core074_cpu_spmm_backward_flows_to_dense_leaf() {
    let sp = SparseTensor::new(
        vec![vec![0, 0], vec![0, 2], vec![1, 1]],
        vec![1.0f32, 2.0, 3.0],
        vec![2, 3],
    )
    .unwrap();
    let d = mk_f32(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![3, 2], true);

    let out = sp.spmm(&d).expect("spmm forward");
    // Forward values unchanged (torch: [[7,16],[6,15]]).
    let o = out.data_vec().expect("out data");
    assert!((o[0] - 7.0).abs() < 1e-6 && (o[1] - 16.0).abs() < 1e-6);
    assert!((o[2] - 6.0).abs() < 1e-6 && (o[3] - 15.0).abs() < 1e-6);

    let w = mk_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
    out.backward_with_gradient(&w)
        .expect("spmm output must carry a backward edge (pre-fix: detached)");

    let grad = d
        .grad()
        .expect("grad access")
        .expect("gradient must REACH the dense leaf");
    let g = grad.data_vec().expect("grad data");
    let expected = [1.0f32, 2.0, 9.0, 12.0, 2.0, 4.0];
    for (i, (got, exp)) in g.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "d.grad[{i}]: got {got}, torch oracle {exp}"
        );
    }
}

/// sparse_matmul_24 gradient reaches the dense operand `a`
/// (oracle 2: a.grad = g @ bm^T = [[20,-1,52,15],[48,-13,112,51]]).
#[test]
fn core074_cpu_matmul24_backward_flows_to_a() {
    let b_dense = mk_f32(
        vec![
            1.0, 4.0, 2.0, 3.0, //
            -5.0, 2.0, 0.0, 1.0, //
            0.5, -0.25, 8.0, 7.0, //
            3.0, 6.0, -2.0, 0.125,
        ],
        vec![4, 4],
        false,
    );
    let b = SemiStructuredSparseTensor::compress(&b_dense).unwrap();

    let a = mk_f32(
        vec![1.0, 2.0, 3.0, 4.0, 0.5, -1.0, 2.0, -0.25],
        vec![2, 4],
        true,
    );
    let out = sparse_matmul_24(&a, &b).expect("matmul24 forward");
    let o = out.data_vec().expect("out data");
    let out_expected = [2.0f32, 32.0, 24.0, 24.0, 4.25, -1.5, 16.0, 15.5];
    for (i, (got, exp)) in o.iter().zip(out_expected.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-5, "out[{i}]: got {got}, exp {exp}");
    }

    let g = mk_f32(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![2, 4],
        false,
    );
    out.backward_with_gradient(&g)
        .expect("matmul24 output must carry a backward edge (pre-fix: detached)");

    let grad = a
        .grad()
        .expect("grad access")
        .expect("gradient must REACH the `a` leaf");
    let ga = grad.data_vec().expect("grad data");
    let expected = [20.0f32, -1.0, 52.0, 15.0, 48.0, -13.0, 112.0, 51.0];
    for (i, (got, exp)) in ga.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "a.grad[{i}]: got {got}, torch oracle {exp}"
        );
    }
}

/// compress of a gradient-tracking tensor: structured error, never a
/// silent detach (oracle 3: upstream raises NotImplementedError as soon
/// as autograd is involved; trainable 2:4 weights tracked in #1969).
#[test]
fn core074_compress_rejects_requires_grad_input() {
    let w = mk_f32(vec![1.0, 4.0, 2.0, 3.0], vec![4], true);
    let r = SemiStructuredSparseTensor::compress(&w);
    assert!(
        matches!(r, Err(FerrotorchError::InvalidArgument { .. })),
        "compress of a requires_grad tensor must error (pre-fix: silent \
         detach into plain vectors), got {:?}",
        r.map(|s| s.values().to_vec())
    );

    // Without grad tracking the same input keeps compressing.
    let w2 = mk_f32(vec![1.0, 4.0, 2.0, 3.0], vec![4], false);
    SemiStructuredSparseTensor::compress(&w2).expect("untracked input still compresses");
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
                .expect("CUDA backend must initialize for the GPU lane");
        });
    }

    /// CUDA spmm backward: gradient reaches the CUDA dense leaf ON CUDA
    /// with the torch oracle values (R-ORACLE-3 device assertion).
    #[test]
    fn core074_gpu_spmm_backward_grad_on_cuda() {
        ensure_cuda_backend();
        let sp = SparseTensor::new(
            vec![vec![0, 0], vec![0, 2], vec![1, 1]],
            vec![1.0f32, 2.0, 3.0],
            vec![2, 3],
        )
        .unwrap();
        let d_cpu = mk_f32(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![3, 2], false);
        let d = d_cpu
            .to(Device::Cuda(0))
            .expect("d->cuda")
            .requires_grad_(true);

        let out = sp.spmm(&d).expect("cuda spmm forward");
        assert!(out.is_cuda(), "spmm output stays on CUDA");

        let w = mk_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false)
            .to(Device::Cuda(0))
            .expect("w->cuda");
        out.backward_with_gradient(&w)
            .expect("CUDA spmm output must carry a backward edge");

        let grad = d
            .grad()
            .expect("grad access")
            .expect("gradient must reach the CUDA dense leaf");
        assert!(grad.is_cuda(), "gradient must live on CUDA");
        let g = grad.cpu().unwrap().data_vec().unwrap();
        let expected = [1.0f32, 2.0, 9.0, 12.0, 2.0, 4.0];
        for (i, (got, exp)) in g.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "cuda d.grad[{i}]: got {got}, torch oracle {exp}"
            );
        }
    }

    /// CUDA sparse_matmul_24 backward: gradient reaches the CUDA `a`
    /// leaf on CUDA with the torch oracle values.
    #[test]
    fn core074_gpu_matmul24_backward_grad_on_cuda() {
        ensure_cuda_backend();
        let b_dense = mk_f32(
            vec![
                1.0, 4.0, 2.0, 3.0, //
                -5.0, 2.0, 0.0, 1.0, //
                0.5, -0.25, 8.0, 7.0, //
                3.0, 6.0, -2.0, 0.125,
            ],
            vec![4, 4],
            false,
        );
        let b = SemiStructuredSparseTensor::compress(&b_dense).unwrap();

        let a_cpu = mk_f32(
            vec![1.0, 2.0, 3.0, 4.0, 0.5, -1.0, 2.0, -0.25],
            vec![2, 4],
            false,
        );
        let a = a_cpu
            .to(Device::Cuda(0))
            .expect("a->cuda")
            .requires_grad_(true);

        let out = sparse_matmul_24(&a, &b).expect("cuda matmul24 forward");
        assert!(out.is_cuda(), "matmul24 output stays on CUDA");

        let g = mk_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 4],
            false,
        )
        .to(Device::Cuda(0))
        .expect("g->cuda");
        out.backward_with_gradient(&g)
            .expect("CUDA matmul24 output must carry a backward edge");

        let grad = a
            .grad()
            .expect("grad access")
            .expect("gradient must reach the CUDA `a` leaf");
        assert!(grad.is_cuda(), "gradient must live on CUDA");
        let ga = grad.cpu().unwrap().data_vec().unwrap();
        let expected = [20.0f32, -1.0, 52.0, 15.0, 48.0, -13.0, 112.0, 51.0];
        for (i, (got, exp)) in ga.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "cuda a.grad[{i}]: got {got}, torch oracle {exp}"
            );
        }
    }
}
