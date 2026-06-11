//! CORE-128 (#1822, CLASS-V High) regression battery: the CUDA backwards of
//! `gather` / `scatter` / `scatter_add` must produce valid f64 gradients —
//! the forwards explicitly support f64 (dim-aware `*_dim_f64` kernels), so
//! the VJPs must too.
//!
//! Pre-fix observed behavior (R-AHON-1 probe, pasted in #1822): the f64 CUDA
//! forward succeeds for all three ops, but the engine backward fails with
//! `Err(InvalidArgument { message: "expected F32 buffer, handle is tagged
//! float64" })` because the backwards unconditionally called
//! `scatter_add_1d_f32` / `masked_zero_f32` / `index_select_1d_f32` on the
//! f64 `grad_output` handle.
//!
//! All gradient expectations below are pasted from a LIVE
//! `torch==2.11.0+cu130` session on the same device class (RTX 3090),
//! `dtype=torch.float64, device='cuda'` (R-ORACLE-1(b)). f64 comparisons are
//! exact (`==`): every value is a small integer reachable without rounding
//! in either dtype, and the VJPs are pure data movement (no accumulation
//! beyond exact small-integer sums).

#![cfg(feature = "gpu")]

use ferrotorch_core::autograd::graph::{backward, backward_with_grad};
use ferrotorch_core::{Device, Tensor, TensorStorage, gather, scatter, scatter_add};
use std::sync::Once;

static GPU_INIT: Once = Once::new();
fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the CORE-128 GPU pins");
    });
}

fn cuda_f64(data: &[f64], shape: &[usize], rg: bool) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(rg)
}

/// Live torch (float64, cuda):
///   x = arange(6).reshape(2,3); x.gather(1, [[2,0],[1,1]]).sum().backward()
///   -> x.grad = [1,0,1, 0,2,0] on cuda:0.
#[test]
fn core128_cuda_f64_gather_backward_grad_flows_to_leaf() {
    ensure_cuda_backend();
    let x = cuda_f64(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3], true);
    let out = gather(&x, 1, &[2, 0, 1, 1], &[2, 2]).unwrap();
    assert!(out.is_cuda(), "forward must stay CUDA-resident");
    backward(&out.sum_all().unwrap()).expect("f64 CUDA gather backward must succeed");
    let g = x
        .grad()
        .unwrap()
        .expect("grad must reach the leaf (R-ORACLE-3)");
    assert!(g.is_cuda(), "gradient must stay CUDA-resident");
    assert_eq!(
        g.cpu().unwrap().data_vec().unwrap(),
        vec![1.0, 0.0, 1.0, 0.0, 2.0, 0.0],
    );
}

/// Live torch (float64, cuda): inp = arange(12).reshape(3,4),
/// src = (arange(8)*10).reshape(2,4), idx = [[1,0,2,1],[2,1,0,0]];
/// inp.scatter(0, idx, src).sum().backward() ->
///   grad_input = [1,0,0,0, 0,0,1,0, 0,1,0,1], grad_src = ones(2,4).
#[test]
fn core128_cuda_f64_scatter_backward_both_grads() {
    ensure_cuda_backend();
    let inp_d: Vec<f64> = (0..12).map(|v| v as f64).collect();
    let src_d: Vec<f64> = (0..8).map(|v| (v * 10) as f64).collect();
    let inp = cuda_f64(&inp_d, &[3, 4], true);
    let src = cuda_f64(&src_d, &[2, 4], true);
    let idx: [usize; 8] = [1, 0, 2, 1, 2, 1, 0, 0];
    let out = scatter(&inp, 0, &idx, &[2, 4], &src).unwrap();
    // Forward oracle (same session): [0,10,60,70, 0,50,6,30, 40,9,20,11].
    assert_eq!(
        out.cpu().unwrap().data_vec().unwrap(),
        vec![
            0.0, 10.0, 60.0, 70.0, 0.0, 50.0, 6.0, 30.0, 40.0, 9.0, 20.0, 11.0
        ],
    );
    backward(&out.sum_all().unwrap()).expect("f64 CUDA scatter backward must succeed");
    let gi = inp.grad().unwrap().expect("grad_input must exist");
    let gs = src.grad().unwrap().expect("grad_src must exist");
    assert!(
        gi.is_cuda() && gs.is_cuda(),
        "grads must stay CUDA-resident"
    );
    assert_eq!(
        gi.cpu().unwrap().data_vec().unwrap(),
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    );
    assert_eq!(gs.cpu().unwrap().data_vec().unwrap(), vec![1.0; 8]);
}

/// Live torch (float64, cuda): same inp/src/idx;
/// o = inp.scatter_add(0, idx, src); o.backward(arange(1,13).reshape(3,4)) ->
///   grad_input = [1..12] (identity), grad_src = [5,2,11,8, 9,6,3,4]
///   (grad gathered at the scattered positions).
#[test]
fn core128_cuda_f64_scatter_add_backward_nonuniform_grad() {
    ensure_cuda_backend();
    let inp_d: Vec<f64> = (0..12).map(|v| v as f64).collect();
    let src_d: Vec<f64> = (0..8).map(|v| (v * 10) as f64).collect();
    let inp = cuda_f64(&inp_d, &[3, 4], true);
    let src = cuda_f64(&src_d, &[2, 4], true);
    let idx: [usize; 8] = [1, 0, 2, 1, 2, 1, 0, 0];
    let out = scatter_add(&inp, 0, &idx, &[2, 4], &src).unwrap();
    // Forward oracle: [0,11,62,73, 4,55,6,37, 48,9,30,11].
    assert_eq!(
        out.cpu().unwrap().data_vec().unwrap(),
        vec![
            0.0, 11.0, 62.0, 73.0, 4.0, 55.0, 6.0, 37.0, 48.0, 9.0, 30.0, 11.0
        ],
    );
    let g_d: Vec<f64> = (1..13).map(|v| v as f64).collect();
    let g = cuda_f64(&g_d, &[3, 4], false);
    backward_with_grad(&out, Some(&g)).expect("f64 CUDA scatter_add backward must succeed");
    let gi = inp.grad().unwrap().expect("grad_input must exist");
    let gs = src.grad().unwrap().expect("grad_src must exist");
    assert!(
        gi.is_cuda() && gs.is_cuda(),
        "grads must stay CUDA-resident"
    );
    assert_eq!(gi.cpu().unwrap().data_vec().unwrap(), g_d);
    assert_eq!(
        gs.cpu().unwrap().data_vec().unwrap(),
        vec![5.0, 2.0, 11.0, 8.0, 9.0, 6.0, 3.0, 4.0],
    );
}
