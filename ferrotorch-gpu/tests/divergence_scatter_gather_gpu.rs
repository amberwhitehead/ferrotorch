//! Live-GPU verification of the dim-aware `gather` / `scatter` /
//! `scatter_value` / `scatter_add` family on CUDA (crosslink #1545 / sub
//! #1535). These exercise the real PTX kernels in
//! `ferrotorch-gpu/src/scatter_gather_kernels.rs` through the
//! `ferrotorch-core::ops::indexing` CUDA dispatch branches and verify byte-
//! exact parity with `torch.gather` / `torch.Tensor.scatter_` /
//! `scatter_(dim, index, value)` / `torch.Tensor.scatter_add_`.
//!
//! # R-CHAR-3 provenance (live torch 2.x, CUDA)
//!
//! Every expected value below is the output of the named torch python on the
//! identical fixture; these are symbolic constants traceable to the exact
//! torch call (not self-derived from ferrotorch).
//!
//! ```python
//! import torch
//! d = "cuda"
//!
//! # gather 2D dim=0
//! inp = torch.tensor([[1.,2.],[3.,4.],[5.,6.]], device=d)         # [3,2]
//! idx = torch.tensor([[2,0],[1,1]], device=d)                     # [2,2]
//! torch.gather(inp, 0, idx)        # tensor([[5.,2.],[3.,4.]])
//!
//! # gather 2D dim=1
//! inp = torch.tensor([[1.,2.,3.],[4.,5.,6.]], device=d)           # [2,3]
//! idx = torch.tensor([[0,2],[1,0]], device=d)                     # [2,2]
//! torch.gather(inp, 1, idx)        # tensor([[1.,3.],[5.,4.]])
//!
//! # gather 3D dim=2
//! inp = torch.arange(1.,13.,device=d).reshape(2,2,3)              # [2,2,3]
//! idx = torch.tensor([[[2,0],[1,1]],[[0,2],[2,0]]], device=d)     # [2,2,2]
//! torch.gather(inp, 2, idx)
//! #   tensor([[[ 3.,  1.],[ 5.,  5.]],[[ 7.,  9.],[12., 10.]]])
//!
//! # scatter dim=0
//! inp = torch.zeros(3,3, device=d)
//! idx = torch.tensor([[0,1,2],[2,0,1]], device=d)
//! src = torch.tensor([[1.,2.,3.],[4.,5.,6.]], device=d)
//! inp.scatter_(0, idx, src)
//! #   tensor([[1.,5.,0.],[0.,2.,6.],[4.,0.,3.]])
//!
//! # scatter dim=1
//! inp = torch.zeros(2,4, device=d)
//! idx = torch.tensor([[0,2],[3,1]], device=d)
//! src = torch.tensor([[5.,6.],[7.,8.]], device=d)
//! inp.scatter_(1, idx, src)
//! #   tensor([[5.,0.,6.,0.],[0.,8.,0.,7.]])
//!
//! # scatter_value dim=0
//! inp = torch.zeros(3,2, device=d)
//! idx = torch.tensor([[0,1],[2,0]], device=d)
//! inp.scatter_(0, idx, 9.0)
//! #   (0,0)->out[0][0]; (0,1)->out[1][1]; (1,0)->out[2][0]; (1,1)->out[0][1]
//! #   tensor([[9.,9.],[0.,9.],[9.,0.]])
//!
//! # scatter_value dim=1
//! inp = torch.zeros(2,3, device=d)
//! idx = torch.tensor([[0,2],[1,1]], device=d)
//! inp.scatter_(1, idx, 7.0)
//! #   tensor([[7.,0.,7.],[0.,7.,0.]])
//!
//! # scatter_add DUPLICATE indices dim=0 (atomic accumulation — the key case)
//! inp = torch.tensor([1.,2.,3.], device=d)                        # [3]
//! idx = torch.tensor([0,2,0,2,0], device=d)                       # 3 hits -> 0
//! src = torch.tensor([10.,20.,30.,40.,50.], device=d)
//! inp.scatter_add_(0, idx, src)
//! #   tensor([1+10+30+50, 2, 3+20+40]) == tensor([91., 2., 63.])
//!
//! # scatter_add DUPLICATE indices dim=1
//! inp = torch.zeros(2,3, device=d)
//! idx = torch.tensor([[0,0,0],[1,1,2]], device=d)
//! src = torch.tensor([[1.,2.,3.],[4.,5.,6.]], device=d)
//! inp.scatter_add_(1, idx, src)
//! #   tensor([[6.,0.,0.],[9.,0.,6.]])
//! ```
//!
//! The f64 variants use the identical fixtures with `dtype=torch.float64`;
//! torch's f64 outputs equal the f32 outputs exactly for these integer-valued
//! data (no rounding).

#![cfg(feature = "cuda")]

use ferrotorch_core::ops::indexing::scatter_value;
use ferrotorch_core::{Device, Tensor, TensorStorage, gather, scatter, scatter_add};
use ferrotorch_gpu::init_cuda_backend;
use half::{bf16, f16};

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn cpu_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu f32 tensor")
}

fn cpu_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu f64 tensor")
}

fn cpu_f16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f16> {
    Tensor::from_storage(
        TensorStorage::cpu(data.iter().copied().map(f16::from_f32).collect()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("cpu f16 tensor")
}

fn cpu_bf16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<bf16> {
    Tensor::from_storage(
        TensorStorage::cpu(data.iter().copied().map(bf16::from_f32).collect()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("cpu bf16 tensor")
}

fn host_f32(t: &Tensor<f32>) -> Vec<f32> {
    t.cpu().expect("cpu()").data().unwrap().to_vec()
}

fn host_f64(t: &Tensor<f64>) -> Vec<f64> {
    t.cpu().expect("cpu()").data().unwrap().to_vec()
}

fn host_f16(t: &Tensor<f16>) -> Vec<f32> {
    t.cpu()
        .expect("cpu()")
        .data()
        .unwrap()
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

fn host_bf16(t: &Tensor<bf16>) -> Vec<f32> {
    t.cpu()
        .expect("cpu()")
        .data()
        .unwrap()
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

fn cuda_f16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f16> {
    cpu_f16(data, shape, false)
        .to(Device::Cuda(0))
        .expect("to cuda f16")
        .requires_grad_(requires_grad)
}

fn cuda_bf16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<bf16> {
    cpu_bf16(data, shape, false)
        .to(Device::Cuda(0))
        .expect("to cuda bf16")
        .requires_grad_(requires_grad)
}

// ===========================================================================
// gather
// ===========================================================================

#[test]
fn gather_gpu_2d_dim0_matches_torch() {
    ensure_cuda();
    let cpu = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");
    let index = [2usize, 0, 1, 1];
    let out = gather(&gpu, 0, &index, &[2, 2]).expect("gpu gather dim0");
    assert!(out.is_cuda(), "gather result must stay GPU-resident");
    // torch.gather(inp, 0, idx) == [[5,2],[3,4]]
    assert_eq!(host_f32(&out), vec![5.0, 2.0, 3.0, 4.0]);
    // GPU == ferrotorch CPU reference.
    let cpu_out = gather(&cpu, 0, &index, &[2, 2]).expect("cpu gather dim0");
    assert_eq!(host_f32(&out), cpu_out.data().unwrap().to_vec());
}

#[test]
fn gather_gpu_2d_dim1_matches_torch() {
    ensure_cuda();
    let cpu = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");
    let index = [0usize, 2, 1, 0];
    let out = gather(&gpu, 1, &index, &[2, 2]).expect("gpu gather dim1");
    assert!(out.is_cuda());
    // torch.gather(inp, 1, idx) == [[1,3],[5,4]]
    assert_eq!(host_f32(&out), vec![1.0, 3.0, 5.0, 4.0]);
}

#[test]
fn gather_gpu_3d_dim2_matches_torch() {
    ensure_cuda();
    // inp = arange(1..13).reshape(2,2,3)
    let data: Vec<f32> = (1..=12).map(|v| v as f32).collect();
    let cpu = cpu_f32(&data, &[2, 2, 3]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");
    // idx = [[[2,0],[1,1]],[[0,2],[2,0]]]  shape [2,2,2]
    let index = [2usize, 0, 1, 1, 0, 2, 2, 0];
    let out = gather(&gpu, 2, &index, &[2, 2, 2]).expect("gpu gather dim2");
    assert!(out.is_cuda());
    // torch.gather(inp, 2, idx) == [[[3,1],[5,5]],[[7,9],[12,10]]]
    assert_eq!(
        host_f32(&out),
        vec![3.0, 1.0, 5.0, 5.0, 7.0, 9.0, 12.0, 10.0]
    );
    let cpu_out = gather(&cpu, 2, &index, &[2, 2, 2]).expect("cpu gather dim2");
    assert_eq!(host_f32(&out), cpu_out.data().unwrap().to_vec());
}

#[test]
fn gather_gpu_f64_dim1_matches_torch() {
    ensure_cuda();
    let cpu = cpu_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda f64");
    let index = [0usize, 2, 1, 0];
    let out = gather(&gpu, 1, &index, &[2, 2]).expect("gpu gather f64 dim1");
    assert!(out.is_cuda(), "f64 gather must stay GPU-resident");
    assert_eq!(host_f64(&out), vec![1.0, 3.0, 5.0, 4.0]);
}

// ===========================================================================
// scatter
// ===========================================================================

#[test]
fn scatter_gpu_dim0_matches_torch() {
    ensure_cuda();
    let cpu = cpu_f32(&[0.0; 9], &[3, 3]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");
    let src = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let src_gpu = src.to(Device::Cuda(0)).expect("to cuda src");
    let index = [0usize, 1, 2, 2, 0, 1];
    let out = scatter(&gpu, 0, &index, &[2, 3], &src_gpu).expect("gpu scatter dim0");
    assert!(out.is_cuda());
    // inp.scatter_(0, idx, src) == [[1,5,0],[0,2,6],[4,0,3]]
    assert_eq!(
        host_f32(&out),
        vec![1.0, 5.0, 0.0, 0.0, 2.0, 6.0, 4.0, 0.0, 3.0]
    );
    let cpu_out = scatter(&cpu, 0, &index, &[2, 3], &src).expect("cpu scatter dim0");
    assert_eq!(host_f32(&out), cpu_out.data().unwrap().to_vec());
}

#[test]
fn scatter_gpu_dim1_matches_torch() {
    ensure_cuda();
    let cpu = cpu_f32(&[0.0; 8], &[2, 4]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");
    let src = cpu_f32(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let src_gpu = src.to(Device::Cuda(0)).expect("to cuda src");
    let index = [0usize, 2, 3, 1];
    let out = scatter(&gpu, 1, &index, &[2, 2], &src_gpu).expect("gpu scatter dim1");
    assert!(out.is_cuda());
    // inp.scatter_(1, idx, src) == [[5,0,6,0],[0,8,0,7]]
    assert_eq!(host_f32(&out), vec![5.0, 0.0, 6.0, 0.0, 0.0, 8.0, 0.0, 7.0]);
}

#[test]
fn scatter_gpu_f64_dim0_matches_torch() {
    ensure_cuda();
    let cpu = cpu_f64(&[0.0; 9], &[3, 3]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda f64");
    let src = cpu_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let src_gpu = src.to(Device::Cuda(0)).expect("to cuda src f64");
    let index = [0usize, 1, 2, 2, 0, 1];
    let out = scatter(&gpu, 0, &index, &[2, 3], &src_gpu).expect("gpu scatter f64 dim0");
    assert!(out.is_cuda());
    assert_eq!(
        host_f64(&out),
        vec![1.0, 5.0, 0.0, 0.0, 2.0, 6.0, 4.0, 0.0, 3.0]
    );
}

// ===========================================================================
// scatter_value
// ===========================================================================

#[test]
fn scatter_value_gpu_dim0_matches_torch() {
    ensure_cuda();
    let cpu = cpu_f32(&[0.0; 6], &[3, 2]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");
    let index = [0usize, 1, 2, 0];
    let out = scatter_value(&gpu, 0, &index, &[2, 2], 9.0).expect("gpu scatter_value dim0");
    assert!(out.is_cuda());
    // inp.scatter_(0, idx, 9.0) == [[9,9],[0,9],[9,0]]
    assert_eq!(host_f32(&out), vec![9.0, 9.0, 0.0, 9.0, 9.0, 0.0]);
    let cpu_out = scatter_value(&cpu, 0, &index, &[2, 2], 9.0).expect("cpu scatter_value dim0");
    assert_eq!(host_f32(&out), cpu_out.data().unwrap().to_vec());
}

#[test]
fn scatter_value_gpu_dim1_matches_torch() {
    ensure_cuda();
    let cpu = cpu_f32(&[0.0; 6], &[2, 3]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");
    let index = [0usize, 2, 1, 1];
    let out = scatter_value(&gpu, 1, &index, &[2, 2], 7.0).expect("gpu scatter_value dim1");
    assert!(out.is_cuda());
    // inp.scatter_(1, idx, 7.0) == [[7,0,7],[0,7,0]]
    assert_eq!(host_f32(&out), vec![7.0, 0.0, 7.0, 0.0, 7.0, 0.0]);
}

#[test]
fn scatter_value_gpu_f64_dim1_matches_torch() {
    ensure_cuda();
    let cpu = cpu_f64(&[0.0; 6], &[2, 3]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda f64");
    let index = [0usize, 2, 1, 1];
    let out = scatter_value(&gpu, 1, &index, &[2, 2], 7.0).expect("gpu scatter_value f64 dim1");
    assert!(out.is_cuda());
    assert_eq!(host_f64(&out), vec![7.0, 0.0, 7.0, 0.0, 7.0, 0.0]);
}

// ===========================================================================
// scatter_add — DUPLICATE INDICES (the key atomic-accumulation case)
// ===========================================================================

#[test]
fn scatter_add_gpu_f32_duplicate_indices_dim0_matches_torch() {
    ensure_cuda();
    // inp = [1,2,3]; idx = [0,2,0,2,0] (three hits on slot 0, two on slot 2);
    // src = [10,20,30,40,50]. A last-write-wins / non-atomic kernel FAILS this.
    let cpu = cpu_f32(&[1.0, 2.0, 3.0], &[3]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");
    let src = cpu_f32(&[10.0, 20.0, 30.0, 40.0, 50.0], &[5]);
    let src_gpu = src.to(Device::Cuda(0)).expect("to cuda src");
    let index = [0usize, 2, 0, 2, 0];
    let out = scatter_add(&gpu, 0, &index, &[5], &src_gpu).expect("gpu scatter_add dup");
    assert!(out.is_cuda());
    // 1 + 10 + 30 + 50 = 91; 2; 3 + 20 + 40 = 63
    assert_eq!(host_f32(&out), vec![91.0, 2.0, 63.0]);
    let cpu_out = scatter_add(&cpu, 0, &index, &[5], &src).expect("cpu scatter_add dup");
    assert_eq!(host_f32(&out), cpu_out.data().unwrap().to_vec());
}

#[test]
fn scatter_add_gpu_f64_duplicate_indices_dim0_matches_torch() {
    ensure_cuda();
    let cpu = cpu_f64(&[1.0, 2.0, 3.0], &[3]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda f64");
    let src = cpu_f64(&[10.0, 20.0, 30.0, 40.0, 50.0], &[5]);
    let src_gpu = src.to(Device::Cuda(0)).expect("to cuda src f64");
    let index = [0usize, 2, 0, 2, 0];
    let out = scatter_add(&gpu, 0, &index, &[5], &src_gpu).expect("gpu scatter_add f64 dup");
    assert!(out.is_cuda(), "f64 scatter_add must stay GPU-resident");
    // f64 atomic add (atom.global.add.f64, sm_60+) accumulates all duplicates.
    assert_eq!(host_f64(&out), vec![91.0, 2.0, 63.0]);
}

#[test]
fn scatter_add_gpu_f32_duplicate_indices_dim1_matches_torch() {
    ensure_cuda();
    // 2D dim=1, duplicates within a row.
    let cpu = cpu_f32(&[0.0; 6], &[2, 3]);
    let gpu = cpu.to(Device::Cuda(0)).expect("to cuda");
    let src = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let src_gpu = src.to(Device::Cuda(0)).expect("to cuda src");
    // idx = [[0,0,0],[1,1,2]]
    let index = [0usize, 0, 0, 1, 1, 2];
    let out = scatter_add(&gpu, 1, &index, &[2, 3], &src_gpu).expect("gpu scatter_add dim1 dup");
    assert!(out.is_cuda());
    // row0: all three into col0 -> [1+2+3,0,0] = [6,0,0]
    // row1: 4,5 into col1, 6 into col2 -> [0,9,6]
    assert_eq!(host_f32(&out), vec![6.0, 0.0, 0.0, 0.0, 9.0, 6.0]);
}

// ===========================================================================
// f16 / bf16 CUDA parity: forward and direct VJP stay resident
// ===========================================================================

#[test]
fn gather_gpu_f16_forward_backward_matches_torch() {
    ensure_cuda();
    let x = cuda_f16(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    let index = [2usize, 0, 1, 1];
    let out = gather(&x, 1, &index, &[2, 2]).expect("f16 gather");
    assert!(out.is_cuda(), "f16 gather output must stay CUDA-resident");
    assert_eq!(host_f16(&out), vec![3.0, 1.0, 5.0, 5.0]);

    let grad_output = cuda_f16(&[1.0; 4], &[2, 2], false);
    let grads = out
        .grad_fn()
        .expect("tracked gather must carry grad_fn")
        .backward(&grad_output)
        .expect("f16 gather backward");
    let grad = grads[0].as_ref().expect("input grad");
    assert!(grad.is_cuda(), "f16 gather grad must stay CUDA-resident");
    assert_eq!(host_f16(grad), vec![1.0, 0.0, 1.0, 0.0, 2.0, 0.0]);
}

#[test]
fn gather_gpu_bf16_forward_backward_matches_torch() {
    ensure_cuda();
    let x = cuda_bf16(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    let index = [2usize, 0, 1, 1];
    let out = gather(&x, 1, &index, &[2, 2]).expect("bf16 gather");
    assert!(out.is_cuda(), "bf16 gather output must stay CUDA-resident");
    assert_eq!(host_bf16(&out), vec![3.0, 1.0, 5.0, 5.0]);

    let grad_output = cuda_bf16(&[1.0; 4], &[2, 2], false);
    let grads = out
        .grad_fn()
        .expect("tracked gather must carry grad_fn")
        .backward(&grad_output)
        .expect("bf16 gather backward");
    let grad = grads[0].as_ref().expect("input grad");
    assert!(grad.is_cuda(), "bf16 gather grad must stay CUDA-resident");
    assert_eq!(host_bf16(grad), vec![1.0, 0.0, 1.0, 0.0, 2.0, 0.0]);
}

#[test]
fn scatter_gpu_f16_forward_backward_matches_torch() {
    ensure_cuda();
    let input = cuda_f16(&[1.0; 6], &[2, 3], true);
    let src = cuda_f16(&[10.0, 20.0, 40.0, 50.0], &[2, 2], true);
    let index = [2usize, 0, 1, 2];
    let out = scatter(&input, 1, &index, &[2, 2], &src).expect("f16 scatter");
    assert!(out.is_cuda(), "f16 scatter output must stay CUDA-resident");
    assert_eq!(host_f16(&out), vec![20.0, 1.0, 10.0, 1.0, 40.0, 50.0]);

    let grad_output = cuda_f16(&[1.0; 6], &[2, 3], false);
    let grads = out
        .grad_fn()
        .expect("tracked scatter must carry grad_fn")
        .backward(&grad_output)
        .expect("f16 scatter backward");
    let grad_input = grads[0].as_ref().expect("input grad");
    let grad_src = grads[1].as_ref().expect("src grad");
    assert!(grad_input.is_cuda());
    assert!(grad_src.is_cuda());
    assert_eq!(host_f16(grad_input), vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
    assert_eq!(host_f16(grad_src), vec![1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn scatter_value_gpu_bf16_forward_backward_matches_torch() {
    ensure_cuda();
    let input = cuda_bf16(&[1.0; 6], &[2, 3], true);
    let index = [2usize, 0, 1, 1];
    let out = scatter_value(&input, 1, &index, &[2, 2], bf16::from_f32(7.0))
        .expect("bf16 scalar scatter");
    assert!(
        out.is_cuda(),
        "bf16 scatter_value output must stay CUDA-resident"
    );
    assert_eq!(host_bf16(&out), vec![7.0, 1.0, 7.0, 1.0, 7.0, 1.0]);

    let grad_output = cuda_bf16(&[1.0; 6], &[2, 3], false);
    let grads = out
        .grad_fn()
        .expect("tracked scatter_value must carry grad_fn")
        .backward(&grad_output)
        .expect("bf16 scatter_value backward");
    let grad_input = grads[0].as_ref().expect("input grad");
    assert!(grad_input.is_cuda());
    assert_eq!(host_bf16(grad_input), vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    assert_eq!(
        grads.len(),
        1,
        "torch scatter(dim, index, scalar) exposes only the input tensor to autograd"
    );
}

#[test]
fn scatter_add_gpu_f16_duplicate_forward_backward_matches_torch() {
    ensure_cuda();
    let input = cuda_f16(&[1.0; 6], &[2, 3], true);
    let src = cuda_f16(&[10.0, 20.0, 40.0, 50.0], &[2, 2], true);
    let index = [2usize, 0, 1, 1];
    let out = scatter_add(&input, 1, &index, &[2, 2], &src).expect("f16 scatter_add");
    assert!(
        out.is_cuda(),
        "f16 scatter_add output must stay CUDA-resident"
    );
    assert_eq!(host_f16(&out), vec![21.0, 1.0, 11.0, 1.0, 91.0, 1.0]);

    let grad_output = cuda_f16(&[1.0; 6], &[2, 3], false);
    let grads = out
        .grad_fn()
        .expect("tracked scatter_add must carry grad_fn")
        .backward(&grad_output)
        .expect("f16 scatter_add backward");
    let grad_input = grads[0].as_ref().expect("input grad");
    let grad_src = grads[1].as_ref().expect("src grad");
    assert!(grad_input.is_cuda());
    assert!(grad_src.is_cuda());
    assert_eq!(host_f16(grad_input), vec![1.0; 6]);
    assert_eq!(host_f16(grad_src), vec![1.0; 4]);
}

#[test]
fn scatter_add_gpu_bf16_duplicate_forward_backward_matches_torch() {
    ensure_cuda();
    let input = cuda_bf16(&[1.0; 6], &[2, 3], true);
    let src = cuda_bf16(&[10.0, 20.0, 40.0, 50.0], &[2, 2], true);
    let index = [2usize, 0, 1, 1];
    let out = scatter_add(&input, 1, &index, &[2, 2], &src).expect("bf16 scatter_add");
    assert!(
        out.is_cuda(),
        "bf16 scatter_add output must stay CUDA-resident"
    );
    assert_eq!(host_bf16(&out), vec![21.0, 1.0, 11.0, 1.0, 91.0, 1.0]);

    let grad_output = cuda_bf16(&[1.0; 6], &[2, 3], false);
    let grads = out
        .grad_fn()
        .expect("tracked scatter_add must carry grad_fn")
        .backward(&grad_output)
        .expect("bf16 scatter_add backward");
    let grad_input = grads[0].as_ref().expect("input grad");
    let grad_src = grads[1].as_ref().expect("src grad");
    assert!(grad_input.is_cuda());
    assert!(grad_src.is_cuda());
    assert_eq!(host_bf16(grad_input), vec![1.0; 6]);
    assert_eq!(host_bf16(grad_src), vec![1.0; 4]);
}
