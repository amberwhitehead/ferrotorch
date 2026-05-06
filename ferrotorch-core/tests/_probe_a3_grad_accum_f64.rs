//! Permanent regression sentinel for #789: GPU f64 grad accumulation
//! was routed through a CPU host-detour path despite `add_f64` being
//! implemented on `CudaBackendImpl`.
//!
//! Pre-fix, `tensor.rs:566` gated the GPU-native accumulation branch
//! with `is_f32 && existing.is_cuda() && incoming.is_cuda()`, so f64
//! tensors fell through to the CPU host-detour:
//!   - download both grads via `data_vec()` (D2H)
//!   - sum on CPU
//!   - re-upload via `TensorStorage::on_device(buf, device)` (H2D)
//!
//! Routing-trace pre-fix: every multi-branch backward over an f64 GPU
//! leaf incurs (numel * sizeof(T)) bytes of D2H *plus* H2D per
//! accumulation event — a 1-D length-4 tensor with two branches
//! triggers two D2H + one H2D round-trips inside `accumulate_grad`,
//! none of which the f32 path does. The CPU detour does happen to
//! produce correct values (the H2D re-upload preserves the leaf's
//! device), so the visible symptom is silent slowness and the latent
//! symptom is that any f64-GPU-only kernel surface (e.g. a custom
//! backend without `cpu_to_gpu`) breaks accumulation entirely.
//!
//! Post-fix the GPU branch dispatches by element size: size_of::<T>()
//! == 4 → `backend.add_f32`, otherwise → `backend.add_f64`. This
//! mirrors the canonical pattern in
//! `autograd::graph::accumulate_non_leaf_grad`.
//!
//! What this probe pins (sentinel-style — values are pre-fix-correct
//! by accident of the H2D re-upload, so the assertions below are the
//! post-fix invariant):
//!   (a) accumulated leaf gradient values are correct
//!   (b) the grad tensor lives on the same Cuda device as the leaf
//!       (no host detour visible from the public surface)
//!   (c) the f32 control path keeps working — guards against the
//!       fix accidentally regressing f32 dispatch.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::Device;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::grad_fns::arithmetic::add;
use ferrotorch_core::grad_fns::reduction::sum as op_sum;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the GPU probe suite");
    });
}

/// f32 control: confirms the existing GPU-native path still works.
#[test]
fn grad_accumulate_twice_on_gpu_f32() {
    ensure_cuda_backend();
    let cpu = from_vec::<f32>(vec![1.0, 2.0, 3.0, 4.0], &[4]).expect("cpu tensor");
    let x = cpu
        .to(Device::Cuda(0))
        .expect("cpu->gpu")
        .requires_grad_(true);

    // y = x + x — backward accumulates x.grad twice (1.0 from each branch).
    let y = add(&x, &x).expect("add");
    let s = op_sum(&y).expect("sum");
    s.backward().expect("backward");

    let grad = x.grad().expect("grad").expect("Some(grad)");
    assert!(
        grad.is_cuda(),
        "f32 GPU grad must remain on GPU after accumulation"
    );
    let host = grad.cpu().expect("grad cpu").data_vec().expect("data_vec");
    // d(sum(x+x))/dx = 2 for every element.
    assert_eq!(host, vec![2.0_f32, 2.0, 2.0, 2.0]);
}

/// f64 regression case for #789. Pre-fix this *would* still produce
/// correct values (the CPU detour computes the right sum), but the
/// grad's TensorStorage was a re-uploaded CPU buffer rather than the
/// handle returned by `add_f64`. Post-fix the grad is the direct
/// `add_f64` handle. We assert: (a) value correctness, (b) grad
/// remains on the original GPU device.
#[test]
fn grad_accumulate_twice_on_gpu_f64() {
    ensure_cuda_backend();
    let cpu = from_vec::<f64>(vec![1.0, 2.0, 3.0, 4.0], &[4]).expect("cpu tensor");
    let x = cpu
        .to(Device::Cuda(0))
        .expect("cpu->gpu")
        .requires_grad_(true);

    // y = x + x — backward accumulates x.grad twice.
    let y = add(&x, &x).expect("add");
    let s = op_sum(&y).expect("sum");
    s.backward().expect("backward");

    let grad = x.grad().expect("grad").expect("Some(grad)");
    assert!(
        grad.is_cuda(),
        "f64 GPU grad must remain on GPU after accumulation (#789)"
    );
    assert_eq!(
        grad.device(),
        Device::Cuda(0),
        "grad device must match the leaf's device"
    );
    let host = grad.cpu().expect("grad cpu").data_vec().expect("data_vec");
    assert_eq!(host, vec![2.0_f64, 2.0, 2.0, 2.0]);
}

/// f64 with three accumulation rounds — exercises the iterated
/// accumulation path (each call to `accumulate_grad` after the first
/// hits the GPU branch we just fixed).
#[test]
fn grad_accumulate_three_branches_on_gpu_f64() {
    ensure_cuda_backend();
    let cpu = from_vec::<f64>(vec![10.0, 20.0], &[2]).expect("cpu tensor");
    let x = cpu
        .to(Device::Cuda(0))
        .expect("cpu->gpu")
        .requires_grad_(true);

    // y = ((x + x) + x) — three branches into x; backward accumulates
    // grad three times on the leaf.
    let y = add(&x, &x).expect("add1");
    let y = add(&y, &x).expect("add2");
    let s = op_sum(&y).expect("sum");
    s.backward().expect("backward");

    let grad = x.grad().expect("grad").expect("Some(grad)");
    assert!(grad.is_cuda(), "grad must remain on GPU");
    let host = grad.cpu().expect("grad cpu").data_vec().expect("data_vec");
    assert_eq!(host, vec![3.0_f64, 3.0]);
}
