//! Permanent regression sentinel for #803 + #791 — einsum cluster.
//!
//! ## #803 — einsum forward silently CPU-detours on GPU
//!
//! Pre-fix expectation: when `einsum` is called on CUDA tensors (whether
//! matmul-style `"ij,jk->ik"`, batched `"bij,bjk->bik"`, axis-sum `"ij->i"`,
//! reduce-all `"ij->"`, trace `"ii->"`, or transpose `"ij->ji"`) the
//! implementation calls `data_vec()` on the operands, computes everything
//! on CPU, and returns a CPU tensor regardless of the input device. This
//! is a §3 violation per `rust-gpu-discipline` (silent CPU fallback in a
//! non-autograd path). The probe asserts both:
//!   1. einsum on CUDA produces the correct numeric result, AND
//!   2. einsum on CUDA does NOT silently demote to CPU storage — the
//!      returned tensor is on the same device as the input(s).
//!
//! Before the fix, (1) succeeds but (2) fails because the implementation
//! always wraps `result` in `TensorStorage::cpu(...)`.
//!
//! ## #791 — EinsumBackwardSingle wrong equation for projection cases
//!
//! Pre-fix expectation: for forward `einsum_differentiable("ij->i", &[&a])`
//! the backward calls into `einsum(reverse_eq=`"i->ij"`, &[grad_output])`,
//! but `j` doesn't appear on the input side, so `build_dim_map` rejects
//! it with `InvalidArgument`. `loss.backward()` therefore panics
//! (or returns an Err) when the user is just doing a perfectly valid
//! axis-sum.
//!
//! Post-fix expectation: `loss.backward()` succeeds and `a.grad()` matches
//! `ones_like(a)` (analytic gradient of `sum(sum(a, dim=1))`).
//!
//! This file is a permanent test — it pins both bugs so that future
//! refactors of `einsum.rs` can't silently re-introduce either failure mode.

use ferrotorch_core::einsum::einsum_differentiable;
#[cfg(feature = "gpu")]
use ferrotorch_core::einsum::einsum;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
#[cfg(feature = "gpu")]
use ferrotorch_core::Device;

#[cfg(feature = "gpu")]
fn t_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn leaf_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch: {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "{label}: index {i}: {a} vs {e} (diff {})",
            (a - e).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// CPU baseline for #791 — projection axis-sum backward
// ---------------------------------------------------------------------------

#[test]
fn cpu_einsum_axis_sum_backward_projects_via_broadcast() {
    // Forward: r = einsum("ij->i", a) where a is 2x3. Then loss = sum(r)
    // is the full reduction. The analytic d(loss)/da = ones_like(a).
    let a = leaf_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let r = einsum_differentiable("ij->i", &[&a]).expect("forward");
    assert_eq!(r.shape(), &[2]);
    assert_close(r.data().unwrap(), &[6.0, 15.0], 1e-6, "fwd axis_sum");

    let loss = ferrotorch_core::grad_fns::reduction::sum(&r).expect("loss");
    loss.backward().expect("backward must not crash on axis_sum projection");
    let grad = a.grad().unwrap().expect("a should have grad");
    assert_eq!(grad.shape(), &[2, 3]);
    assert_close(
        grad.data().unwrap(),
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        1e-6,
        "axis_sum grad",
    );
}

#[test]
fn cpu_einsum_full_reduce_backward_projects_via_broadcast() {
    // Forward: r = einsum("ij->", a) — full reduction to scalar.
    // d(r)/da = ones_like(a).
    let a = leaf_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let r = einsum_differentiable("ij->", &[&a]).expect("forward");
    assert!(r.is_scalar());
    assert!((r.item().unwrap() - 10.0).abs() < 1e-6);

    let loss = ferrotorch_core::grad_fns::reduction::sum(&r).expect("loss");
    loss.backward().expect("backward must not crash on full reduction");
    let grad = a.grad().unwrap().expect("a should have grad");
    assert_close(
        grad.data().unwrap(),
        &[1.0, 1.0, 1.0, 1.0],
        1e-6,
        "full reduce grad",
    );
}

#[test]
fn cpu_einsum_three_d_projection_backward_works() {
    // Forward: r = einsum("ijk->ij", a) — sum over last axis.
    // d(sum(r))/da = ones_like(a).
    #[rustfmt::skip]
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let a = leaf_f32(&data, &[2, 3, 4]);
    let r = einsum_differentiable("ijk->ij", &[&a]).expect("forward");
    assert_eq!(r.shape(), &[2, 3]);

    let loss = ferrotorch_core::grad_fns::reduction::sum(&r).expect("loss");
    loss.backward().expect("backward must not crash");
    let grad = a.grad().unwrap().expect("grad");
    let ones = vec![1.0_f32; 24];
    assert_close(grad.data().unwrap(), &ones, 1e-6, "ijk->ij grad");
}

// ---------------------------------------------------------------------------
// CPU baseline for "trace" and "permutation" cases (#791 must not regress)
// ---------------------------------------------------------------------------

#[test]
fn cpu_einsum_transpose_backward_still_works() {
    // Permutation case (in_subs == out_subs as a multiset). Backward via
    // reverse-equation should still work post-fix.
    let a = leaf_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let r = einsum_differentiable("ij->ji", &[&a]).expect("forward");
    assert_eq!(r.shape(), &[3, 2]);

    let loss = ferrotorch_core::grad_fns::reduction::sum(&r).expect("loss");
    loss.backward().expect("backward");
    let grad = a.grad().unwrap().expect("grad");
    let ones = vec![1.0_f32; 6];
    assert_close(grad.data().unwrap(), &ones, 1e-6, "transpose grad");
}

#[test]
fn cpu_einsum_trace_backward_still_works() {
    // Trace: in has repeated index; the existing has_repeated branch
    // must continue to handle it.
    let a = leaf_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let r = einsum_differentiable("ii->", &[&a]).expect("forward");
    assert!(r.is_scalar());
    assert!((r.item().unwrap() - 5.0).abs() < 1e-6);

    let loss = ferrotorch_core::grad_fns::reduction::sum(&r).expect("loss");
    loss.backward().expect("backward");
    let grad = a.grad().unwrap().expect("grad");
    // Gradient of trace w.r.t. matrix is the identity matrix.
    assert_close(grad.data().unwrap(), &[1.0, 0.0, 0.0, 1.0], 1e-6, "trace grad");
}

// ---------------------------------------------------------------------------
// GPU paths for #803 — forward must stay on device
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for the GPU probe suite");
        });
    }

    fn upload(t: Tensor<f32>) -> Tensor<f32> {
        t.to(Device::Cuda(0)).expect("upload to cuda")
    }

    #[test]
    fn gpu_einsum_matmul_stays_on_device() {
        ensure_cuda_backend();
        // 2x2 @ 2x2 — the most common contraction.
        let a = upload(t_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let b = upload(t_f32(&[5.0, 6.0, 7.0, 8.0], &[2, 2]));
        assert!(a.is_cuda());
        assert!(b.is_cuda());

        let c = einsum("ij,jk->ik", &[&a, &b]).expect("einsum");
        // Bug #803: pre-fix the result is silently CPU-backed.
        assert!(
            c.is_cuda(),
            "#803: einsum forward must stay on device; got device {:?}",
            c.device()
        );
        // And produce the correct numeric values.
        let host = c.cpu().unwrap();
        assert_close(host.data().unwrap(), &[19.0, 22.0, 43.0, 50.0], 1e-5, "mm");
    }

    #[test]
    fn gpu_einsum_bmm_stays_on_device() {
        ensure_cuda_backend();
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 1.0];
        let b_data: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let a = upload(t_f32(&a_data, &[2, 2, 2]));
        let b = upload(t_f32(&b_data, &[2, 2, 2]));
        let c = einsum("bij,bjk->bik", &[&a, &b]).expect("einsum");
        assert!(c.is_cuda(), "#803: bmm einsum must stay on device");
        let host = c.cpu().unwrap();
        assert_close(
            host.data().unwrap(),
            &[19.0, 22.0, 43.0, 50.0, 9.0, 10.0, 11.0, 12.0],
            1e-5,
            "bmm",
        );
    }

    #[test]
    fn gpu_einsum_axis_sum_stays_on_device() {
        ensure_cuda_backend();
        let a = upload(t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let c = einsum("ij->i", &[&a]).expect("einsum");
        assert!(c.is_cuda(), "#803: axis_sum einsum must stay on device");
        let host = c.cpu().unwrap();
        assert_close(host.data().unwrap(), &[6.0, 15.0], 1e-5, "axis_sum");
    }

    #[test]
    fn gpu_einsum_full_reduce_stays_on_device() {
        ensure_cuda_backend();
        let a = upload(t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let c = einsum("ij->", &[&a]).expect("einsum");
        assert!(c.is_cuda(), "#803: full reduce einsum must stay on device");
        let host = c.cpu().unwrap();
        assert!((host.item().unwrap() - 21.0).abs() < 1e-5);
    }

    #[test]
    fn gpu_einsum_transpose_stays_on_device() {
        ensure_cuda_backend();
        let a = upload(t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let c = einsum("ij->ji", &[&a]).expect("einsum");
        assert!(c.is_cuda(), "#803: transpose einsum must stay on device");
        let host = c.cpu().unwrap();
        assert_close(
            host.data().unwrap(),
            &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
            1e-6,
            "transpose",
        );
    }

    #[test]
    fn gpu_einsum_trace_returns_err_not_silent_cpu_detour() {
        // #803 scope-narrowing decision: equations with repeated input
        // indices ("ii->" trace, "ii->i" diagonal) have no clean
        // decomposition into the existing GPU primitives. Per §3, we
        // return Err(NotImplementedOnCuda) instead of silently moving
        // the operand to CPU. The follow-up that adds an on-device
        // diagonal-extract kernel can flip this to a passing forward.
        ensure_cuda_backend();
        let a = upload(t_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let r = einsum("ii->", &[&a]);
        match r {
            Err(ferrotorch_core::FerrotorchError::NotImplementedOnCuda { op }) => {
                assert_eq!(
                    op, "einsum_repeated_index",
                    "expected einsum_repeated_index marker, got {op:?}"
                );
            }
            Ok(t) => panic!(
                "#803: trace einsum on GPU must return Err(NotImplementedOnCuda), \
                 not silently produce a {:?} tensor",
                t.device()
            ),
            Err(other) => panic!("unexpected error variant: {other:?}"),
        }
    }

    // ---------------------------------------------------------------------
    // GPU coverage for #791 — backward must not crash for projections
    // ---------------------------------------------------------------------

    #[test]
    fn gpu_einsum_axis_sum_backward_projects_via_broadcast() {
        ensure_cuda_backend();
        let a_cpu = leaf_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let a = a_cpu.to(Device::Cuda(0)).expect("upload");
        let r = einsum_differentiable("ij->i", &[&a]).expect("forward");

        let loss = ferrotorch_core::grad_fns::reduction::sum(&r).expect("loss");
        loss.backward().expect("#791 backward must not crash");
        let grad = a.grad().unwrap().expect("grad_a");
        let host = grad.cpu().unwrap();
        assert_close(
            host.data().unwrap(),
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            1e-6,
            "axis_sum grad gpu",
        );
    }
}
