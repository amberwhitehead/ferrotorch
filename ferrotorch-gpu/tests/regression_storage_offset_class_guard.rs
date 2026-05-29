//! REGRESSION GUARD for the #1658 storage-offset GPU-dispatch class fix
//! (commit b91c8192c), produced by the ACToR completeness re-audit.
//!
//! Two guards, both PASSING under the current implementation, locking the
//! behaviours the discriminator verified safe so a future edit cannot silently
//! regress them:
//!
//! PART A — narrowed-offset CUDA views for the arithmetic family that is
//! SAFE BY CONSTRUCTION (it predates #1658 and normalises via
//! `ensure_contig_for_gpu`, which routes a `storage_offset != 0` view through
//! the on-device `strided_copy` kernel — see arithmetic.rs:98-113). Confirms
//! add/mul(scale)/neg/abs on a row-narrowed CUDA view honour storage_offset.
//!
//! PART B — backward-through-`.contiguous()` for the GPU reductions the #1658
//! fix touched (sum/prod/amax shadow the input with `.contiguous()` BEFORE the
//! grad_fn capture). For an already-contiguous offset-0 input, `.contiguous()`
//! is a clone; this guard verifies the autograd graph still flows through the
//! cloned input so gradients are unchanged (the fix did not sever the graph).
//!
//! # R-CHAR-3 provenance (live torch 2.11.0+cu130, this env, RTX 3090)
//!
//! ```python
//! import torch
//! full = torch.arange(1,9,dtype=torch.float32).reshape(4,2); view = full[1:4]
//! (view + 1).flatten().tolist()   # ADD1 -> [4,5,6,7,8,9]
//! (view * 2).flatten().tolist()   # MUL2 -> [6,8,10,12,14,16]
//! (-view).flatten().tolist()      # NEG  -> [-3,-4,-5,-6,-7,-8]
//! view.abs().flatten().tolist()   # ABS  -> [3,4,5,6,7,8]
//! # backward grads (device-invariant autograd VJP):
//! x=torch.tensor([2.,3.,4.],requires_grad=True); x.prod().backward(); x.grad  # [12,8,6]
//! x=torch.tensor([1.,5.,3.],requires_grad=True); x.amax().backward(); x.grad  # [0,1,0]
//! x=torch.tensor([[1.,2.],[3.,4.]],requires_grad=True); x.sum().backward(); x.grad # ones
//! ```

#![cfg(feature = "cuda")]

use ferrotorch_core::grad_fns::reduction::{amax, prod, sum};
use ferrotorch_core::{Device, Tensor, TensorStorage};
use ferrotorch_gpu::init_cuda_backend;

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = init_cuda_backend();
    });
}

fn cpu_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu f32 tensor")
}

fn host_f32(t: &Tensor<f32>) -> Vec<f32> {
    t.cpu().expect("cpu()").data().unwrap().to_vec()
}

fn assert_close(got: &[f32], want: &[f32], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= 1e-4,
            "{ctx}: element {i}: got {g}, want {w}"
        );
    }
}

fn narrowed_cuda_view() -> Tensor<f32> {
    let full = cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2])
        .to(Device::Cuda(0))
        .expect("to cuda");
    let view = full.narrow(0, 1, 3).expect("narrow rows 1..4");
    assert!(view.is_contiguous() && view.storage_offset() != 0);
    view
}

// ── PART A: arithmetic family (safe via ensure_contig_for_gpu) ─────────────
const TORCH_ADD1: [f32; 6] = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
const TORCH_MUL2: [f32; 6] = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
const TORCH_NEG: [f32; 6] = [-3.0, -4.0, -5.0, -6.0, -7.0, -8.0];
const TORCH_ABS: [f32; 6] = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

#[test]
fn arithmetic_add_narrowed_offset_view_gpu_safe() {
    ensure_cuda();
    let v = narrowed_cuda_view();
    let ones = cpu_f32(&[1.0; 6], &[3, 2]).to(Device::Cuda(0)).unwrap();
    let r = v.add_t(&ones).expect("add_t");
    assert_close(&host_f32(&r), &TORCH_ADD1, "add_t narrowed");
}

#[test]
fn arithmetic_mul_neg_abs_narrowed_offset_view_gpu_safe() {
    ensure_cuda();
    let v = narrowed_cuda_view();
    let twos = cpu_f32(&[2.0; 6], &[3, 2]).to(Device::Cuda(0)).unwrap();
    let m = v.mul_t(&twos).expect("mul_t");
    assert_close(&host_f32(&m), &TORCH_MUL2, "mul_t narrowed");

    let n = narrowed_cuda_view().neg_t().expect("neg_t");
    assert_close(&host_f32(&n), &TORCH_NEG, "neg_t narrowed");

    let a = narrowed_cuda_view()
        .neg_t()
        .unwrap()
        .abs_t()
        .expect("abs_t");
    assert_close(&host_f32(&a), &TORCH_ABS, "abs_t(neg) narrowed");
}

// ── PART B: backward through `.contiguous()` (offset-0 contiguous input) ───
const TORCH_PROD_GRAD: [f32; 3] = [12.0, 8.0, 6.0];
const TORCH_AMAX_GRAD: [f32; 3] = [0.0, 1.0, 0.0];
const TORCH_SUM_GRAD: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

#[test]
fn prod_backward_through_contiguous_gpu_grads_intact() {
    ensure_cuda();
    let x = cpu_f32(&[2.0, 3.0, 4.0], &[3])
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(true);
    let y = prod(&x).expect("prod");
    y.backward().expect("backward");
    let g = x.grad().expect("grad result").expect("grad present");
    assert_close(
        &host_f32(&g),
        &TORCH_PROD_GRAD,
        "prod backward through contiguous",
    );
}

#[test]
fn amax_backward_through_contiguous_gpu_grads_intact() {
    ensure_cuda();
    let x = cpu_f32(&[1.0, 5.0, 3.0], &[3])
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(true);
    let y = amax(&x).expect("amax");
    y.backward().expect("backward");
    let g = x.grad().expect("grad result").expect("grad present");
    assert_close(
        &host_f32(&g),
        &TORCH_AMAX_GRAD,
        "amax backward through contiguous",
    );
}

#[test]
fn sum_backward_through_contiguous_gpu_grads_intact() {
    ensure_cuda();
    let x = cpu_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(true);
    let y = sum(&x).expect("sum");
    y.backward().expect("backward");
    let g = x.grad().expect("grad result").expect("grad present");
    assert_close(
        &host_f32(&g),
        &TORCH_SUM_GRAD,
        "sum backward through contiguous",
    );
}
