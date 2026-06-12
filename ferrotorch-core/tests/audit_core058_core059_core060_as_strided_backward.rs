//! Regression tests for CORE-058 (#1752), CORE-059 (#1753), CORE-060 (#1754)
//! — the `as_strided` autograd family in `ferrotorch-core/src/stride_tricks.rs`.
//!
//! - CORE-058 (CLASS-V): `AsStridedBackward` used overwrite-scatter, so
//!   overlapping views (sliding windows, zero strides, negative strides)
//!   returned last-write-wins gradients instead of SUMMED gradients.
//! - CORE-059 (CLASS-S/U): the backward allocated its gradient base with
//!   CPU-only `creation::zeros` (CUDA grad_output -> DeviceMismatch) and
//!   treated the saved absolute storage_offset as an offset into a fresh
//!   buffer of `input.shape()` (valid forwards on offset/transposed/chained
//!   views failed or mis-scattered).
//! - CORE-060 (CLASS-S): `as_strided_copy` / `as_strided_scatter` returned
//!   `requires_grad = false` on every path (silent detach), although torch
//!   differentiates both.
//!
//! Every numeric expectation below is pinned from a live torch session
//! (`python3`, torch 2.11.0+cu130, RTX 3090) quoted next to each assertion
//! (R-ORACLE-1). Gradient assertions check gradient FLOW to the original
//! leaf, never `requires_grad` flags alone (R-ORACLE-3); CUDA tests assert
//! result AND gradient device.
//!
//! All values are small integers exactly representable in f32/f64, and the
//! backward is pure scatter-add bookkeeping (no transcendental drift), so
//! comparisons are exact (`==` on `data_vec`), matching the conformance
//! suite's bit-exact policy for data-movement ops.

use ferrotorch_core::autograd::backward;
use ferrotorch_core::{Tensor, TensorStorage};

#[cfg(feature = "gpu")]
use ferrotorch_core::Device;

fn leaf_f64(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f64> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .unwrap()
}

fn grad_of(t: &Tensor<f64>) -> Vec<f64> {
    t.grad()
        .unwrap()
        .expect("leaf should have a gradient")
        .data_vec()
        .unwrap()
}

// ===========================================================================
// CORE-058 (#1752) — overlap-aware gradient accumulation
// ===========================================================================

/// torch oracle:
/// ```python
/// x = torch.arange(1., 6., dtype=torch.float64, requires_grad=True)
/// x.as_strided([3,3],[1,1],0).sum().backward()
/// x.grad  # tensor([1., 2., 3., 2., 1.], dtype=torch.float64)
/// ```
#[test]
fn core058_overlapping_sliding_window_grads_sum() {
    let x = leaf_f64(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5], true);
    let v = x.as_strided(&[3, 3], &[1, 1], Some(0)).unwrap();
    let s = v.contiguous().unwrap().sum_all().unwrap();
    backward(&s).unwrap();
    assert_eq!(grad_of(&x), vec![1.0, 2.0, 3.0, 2.0, 1.0]);
}

/// torch oracle:
/// ```python
/// x = torch.tensor([7.,8.,9.], dtype=torch.float64, requires_grad=True)
/// x.as_strided([5],[0],1).sum().backward()
/// x.grad  # tensor([0., 5., 0.], dtype=torch.float64)
/// ```
#[test]
fn core058_zero_stride_broadcast_backward_sums_multiplicity() {
    let x = leaf_f64(&[7.0, 8.0, 9.0], &[3], true);
    let v = x.as_strided(&[5], &[0], Some(1)).unwrap();
    let s = v.contiguous().unwrap().sum_all().unwrap();
    backward(&s).unwrap();
    assert_eq!(grad_of(&x), vec![0.0, 5.0, 0.0]);
}

/// Negative strides: torch's `as_strided` rejects them ("Negative strides
/// are not supported at the moment"), so the oracle is the equivalent
/// composite — `y = x.flip(0)` (y\[k\] = x\[4-k\]) followed by a positive-
/// stride overlapping view. ferrotorch's `x.as_strided([2,3],[-1,-1], 4)`
/// reads storage\[4-(i+j)\], element-for-element identical to
/// `x.flip(0).as_strided([2,3],[1,1],0)`.
///
/// torch oracle:
/// ```python
/// x = torch.arange(1., 6., dtype=torch.float64, requires_grad=True)
/// x.flip(0).as_strided([2,3],[1,1],0).sum().backward()
/// x.grad  # tensor([0., 1., 2., 2., 1.], dtype=torch.float64)
/// ```
#[test]
fn core058_negative_stride_overlapping_backward() {
    let x = leaf_f64(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5], true);
    let v = x.as_strided(&[2, 3], &[-1, -1], Some(4)).unwrap();
    let s = v.contiguous().unwrap().sum_all().unwrap();
    backward(&s).unwrap();
    assert_eq!(grad_of(&x), vec![0.0, 1.0, 2.0, 2.0, 1.0]);
}

// ===========================================================================
// CORE-059 (#1753) — input view geometry (offset / strides) in backward
// ===========================================================================

/// A valid forward on a narrowed (nonzero-offset) input must not fail in
/// backward, and the gradient must land at the right storage positions.
///
/// torch oracle:
/// ```python
/// x = torch.arange(0., 10., dtype=torch.float64, requires_grad=True)
/// n = x[2:7]                      # storage_offset 2
/// n.as_strided([2,2],[2,1],3).sum().backward()   # absolute offset 3
/// x.grad  # tensor([0., 0., 0., 1., 1., 1., 1., 0., 0., 0.])
/// ```
#[test]
fn core059_narrowed_input_offset_view_backward() {
    let x = leaf_f64(
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[10],
        true,
    );
    let n = x.narrow(0, 2, 5).unwrap(); // storage_offset 2, shape [5]
    let v = n.as_strided(&[2, 2], &[2, 1], Some(3)).unwrap();
    let s = v.sum_all().unwrap(); // [2,2]/[2,1] is contiguous
    backward(&s).unwrap();
    assert_eq!(
        grad_of(&x),
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    );
}

/// The saved input's own stride pattern (here a transpose) must be honored
/// when gathering the gradient back out of the scatter base.
///
/// torch oracle:
/// ```python
/// x = torch.arange(0., 6., dtype=torch.float64).reshape(2,3).clone().requires_grad_(True)
/// t = x.t()                       # shape [3,2], strides [1,3]
/// t.as_strided([2,2],[1,3],1).sum().backward()
/// x.grad  # tensor([[0., 1., 1.], [0., 1., 1.]])
/// ```
#[test]
fn core059_transposed_input_backward() {
    let x = leaf_f64(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3], true);
    let t = x.t().unwrap(); // [3,2], strides [1,3]
    let v = t.as_strided(&[2, 2], &[1, 3], Some(1)).unwrap();
    let s = v.contiguous().unwrap().sum_all().unwrap();
    backward(&s).unwrap();
    assert_eq!(grad_of(&x), vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
}

/// Chained `as_strided` views: the second view's backward sees the FIRST
/// view (not the leaf) as its input and must respect that geometry.
///
/// torch oracle:
/// ```python
/// x = torch.arange(0., 6., dtype=torch.float64, requires_grad=True)
/// v1 = x.as_strided([2,3],[3,1],0)
/// v1.as_strided([2,2],[1,1],1).sum().backward()
/// x.grad  # tensor([0., 1., 2., 1., 0., 0.])
/// ```
#[test]
fn core059_chained_views_backward() {
    let x = leaf_f64(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[6], true);
    let v1 = x.as_strided(&[2, 3], &[3, 1], Some(0)).unwrap();
    let v2 = v1.as_strided(&[2, 2], &[1, 1], Some(1)).unwrap();
    let s = v2.contiguous().unwrap().sum_all().unwrap();
    backward(&s).unwrap();
    assert_eq!(grad_of(&x), vec![0.0, 1.0, 2.0, 1.0, 0.0, 0.0]);
}

/// Chain where the INTERMEDIATE view itself overlaps: exercises the
/// divide-by-multiplicity step of torch's as_strided_backward (the input
/// geometry of the second backward node is overlapping).
///
/// torch oracle:
/// ```python
/// x = torch.arange(1., 6., dtype=torch.float64, requires_grad=True)
/// v1 = x.as_strided([3,3],[1,1],0)   # overlapping intermediate
/// v1.as_strided([2],[4],0).sum().backward()
/// x.grad  # tensor([1., 0., 0., 0., 1.])
/// ```
#[test]
fn core059_chained_overlapping_intermediate_backward() {
    let x = leaf_f64(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5], true);
    let v1 = x.as_strided(&[3, 3], &[1, 1], Some(0)).unwrap();
    let v2 = v1.as_strided(&[2], &[4], Some(0)).unwrap();
    let s = v2.contiguous().unwrap().sum_all().unwrap();
    backward(&s).unwrap();
    assert_eq!(grad_of(&x), vec![1.0, 0.0, 0.0, 0.0, 1.0]);
}

// ===========================================================================
// CORE-060 (#1754) — as_strided_copy / as_strided_scatter differentiability
// ===========================================================================

/// torch oracle:
/// ```python
/// x = torch.arange(1., 6., dtype=torch.float64, requires_grad=True)
/// c = torch.as_strided_copy(x, [3,3],[1,1],0)
/// c.requires_grad, type(c.grad_fn).__name__  # (True, 'AsStridedBackward0_copy')
/// c.sum().backward(); x.grad  # tensor([1., 2., 3., 2., 1.])
/// ```
#[test]
fn core060_as_strided_copy_flows_gradients() {
    let x = leaf_f64(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5], true);
    let c = x.as_strided_copy(&[3, 3], &[1, 1], Some(0)).unwrap();
    assert!(
        c.requires_grad(),
        "as_strided_copy of a requires_grad input must track gradients (torch: True)"
    );
    let s = c.sum_all().unwrap();
    backward(&s).unwrap();
    assert_eq!(grad_of(&x), vec![1.0, 2.0, 3.0, 2.0, 1.0]);
}

/// A non-tracking input must stay non-tracking (no spurious graph).
#[test]
fn core060_as_strided_copy_no_grad_input_stays_detached() {
    let x = leaf_f64(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5], false);
    let c = x.as_strided_copy(&[3, 3], &[1, 1], Some(0)).unwrap();
    assert!(!c.requires_grad());
}

/// `as_strided_scatter` gradients w.r.t. BOTH base and src, weighted so
/// gather positions are distinguishable.
///
/// src grad — torch oracle (torch agrees with finite differences here):
/// ```python
/// base = torch.zeros(6, dtype=torch.float64, requires_grad=True)
/// src = torch.tensor([10.,20.,30.], dtype=torch.float64, requires_grad=True)
/// out = torch.as_strided_scatter(base, src, [3],[2],0)
/// (out * torch.arange(1.,7.)).sum().backward()
/// src.grad   # tensor([1., 3., 5.])
/// ```
///
/// base grad — DELIBERATE DIVERGENCE from torch 2.11.0 (#1754 result
/// comment + follow-up issue): out[1,3,5] = base[1,3,5] pass through and
/// out[0,2,4] = src, so d(loss)/d(base) = [0,2,0,4,0,6]. torch's analytic
/// formula (`as_strided_scatter_backward`, FunctionsManual.cpp:3366-3389 at
/// baseline 2ec0222669) returns the OPPOSITE masking, [1,0,3,0,5,0], and
/// fails torch's own gradcheck:
/// ```python
/// base.grad  # tensor([1., 0., 3., 0., 5., 0.])  <- torch 2.11.0 (wrong)
/// torch.autograd.gradcheck(f, (base, src))
/// # GradcheckError: Jacobian mismatch for output 0 with respect to input 0,
/// # numerical: [0., 2., 0., 4., 0., 6.]  analytical: [1., 0., 3., 0., 5., 0.]
/// ```
/// Per R-ORACLE-4 we pin exactly one contract: the finite-difference
/// (mathematically correct) Jacobian.
#[test]
fn core060_as_strided_scatter_flows_gradients_to_base_and_src() {
    let base = leaf_f64(&[0.0; 6], &[6], true);
    let src = leaf_f64(&[10.0, 20.0, 30.0], &[3], true);
    let out = base.as_strided_scatter(&src, &[3], &[2], Some(0)).unwrap();
    assert!(
        out.requires_grad(),
        "as_strided_scatter with tracking inputs must track gradients (torch: True)"
    );
    let w = leaf_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], false);
    let s = out.mul_t(&w).unwrap().sum_all().unwrap();
    backward(&s).unwrap();
    assert_eq!(
        grad_of(&base),
        vec![0.0, 2.0, 0.0, 4.0, 0.0, 6.0],
        "base grad = upstream grad with the scattered region zeroed (finite differences; \
         torch 2.11.0's divergent analytic value is [1,0,3,0,5,0], see #1754)"
    );
    assert_eq!(grad_of(&src), vec![1.0, 3.0, 5.0]);
}

/// Full-cover scatter: every output position comes from `src`, so the base
/// gradient is exactly zero (finite differences; torch 2.11.0's divergent
/// analytic value is the full upstream grad [1,2,3,4,5,6], see #1754).
///
/// ```python
/// base = torch.arange(1., 7., dtype=torch.float64).clone().requires_grad_(True)
/// src = torch.full((2,3), 5., dtype=torch.float64, requires_grad=True)
/// out = torch.as_strided_scatter(base, src, [2,3],[3,1],0)
/// (out * torch.arange(1.,7.)).sum().backward()
/// src.grad   # tensor([[1., 2., 3.], [4., 5., 6.]])  (torch == finite diff)
/// base.grad  # torch 2.11.0: [1,2,3,4,5,6]; finite differences: all zeros
/// ```
#[test]
fn core060_as_strided_scatter_full_cover_base_grad_is_zero() {
    let base = leaf_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], true);
    let src = leaf_f64(&[5.0; 6], &[2, 3], true);
    let out = base
        .as_strided_scatter(&src, &[2, 3], &[3, 1], Some(0))
        .unwrap();
    let w = leaf_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], false);
    let s = out.mul_t(&w).unwrap().sum_all().unwrap();
    backward(&s).unwrap();
    assert_eq!(grad_of(&base), vec![0.0; 6]);
    assert_eq!(grad_of(&src), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

/// Non-tracking inputs stay detached.
#[test]
fn core060_as_strided_scatter_no_grad_inputs_stay_detached() {
    let base = leaf_f64(&[0.0; 6], &[6], false);
    let src = leaf_f64(&[1.0, 2.0, 3.0], &[3], false);
    let out = base.as_strided_scatter(&src, &[3], &[2], Some(0)).unwrap();
    assert!(!out.requires_grad());
}

// ===========================================================================
// CUDA lanes — gated on the `gpu` feature (real kernels, no host bounce)
// ===========================================================================

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for the GPU regression suite");
        });
    }

    fn cuda_leaf_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        leaf_f64(data, shape, false)
            .to(Device::Cuda(0))
            .unwrap()
            .requires_grad_(true)
    }

    fn cuda_grad_of(t: &Tensor<f64>) -> Vec<f64> {
        let g = t.grad().unwrap().expect("CUDA leaf should have a gradient");
        assert!(g.is_cuda(), "gradient must stay on CUDA (R-ORACLE-3)");
        g.data_vec().unwrap()
    }

    /// torch oracle (CUDA):
    /// ```python
    /// x = torch.arange(1., 6., dtype=torch.float64, device="cuda", requires_grad=True)
    /// x.as_strided([3,3],[1,1],0).sum().backward()
    /// x.grad.device, x.grad  # (device(type='cuda', index=0), tensor([1., 2., 3., 2., 1.]))
    /// ```
    #[test]
    fn core059_cuda_overlapping_backward_sums_on_device() {
        ensure_cuda_backend();
        let x = cuda_leaf_f64(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
        let v = x.as_strided(&[3, 3], &[1, 1], Some(0)).unwrap();
        assert!(v.is_cuda());
        let s = v.contiguous().unwrap().sum_all().unwrap();
        backward(&s).unwrap();
        assert_eq!(cuda_grad_of(&x), vec![1.0, 2.0, 3.0, 2.0, 1.0]);
    }

    /// Non-overlapping CUDA view backward (the strided_scatter fast path).
    ///
    /// ```python
    /// x = torch.arange(1., 7., dtype=torch.float64, device="cuda", requires_grad=True)
    /// x.as_strided([2,3],[3,1],0).sum().backward()
    /// x.grad  # tensor([1., 1., 1., 1., 1., 1.], device='cuda:0')
    /// ```
    #[test]
    fn core059_cuda_non_overlapping_backward_on_device() {
        ensure_cuda_backend();
        let x = cuda_leaf_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]);
        let v = x.as_strided(&[2, 3], &[3, 1], Some(0)).unwrap();
        let s = v.contiguous().unwrap().sum_all().unwrap();
        backward(&s).unwrap();
        assert_eq!(cuda_grad_of(&x), vec![1.0; 6]);
    }

    /// Narrowed-offset CUDA backward (offset-delta handling on device).
    ///
    /// ```python
    /// x = torch.arange(0., 10., dtype=torch.float64, device="cuda", requires_grad=True)
    /// x[2:7].as_strided([2,2],[2,1],3).sum().backward()
    /// x.grad  # tensor([0., 0., 0., 1., 1., 1., 1., 0., 0., 0.], device='cuda:0')
    /// ```
    #[test]
    fn core059_cuda_narrowed_offset_view_backward() {
        ensure_cuda_backend();
        let x = cuda_leaf_f64(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[10]);
        let n = x.narrow(0, 2, 5).unwrap();
        let v = n.as_strided(&[2, 2], &[2, 1], Some(3)).unwrap();
        let s = v.sum_all().unwrap();
        backward(&s).unwrap();
        assert_eq!(
            cuda_grad_of(&x),
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        );
    }

    /// torch oracle (CUDA):
    /// ```python
    /// x = torch.arange(1., 6., dtype=torch.float64, device="cuda", requires_grad=True)
    /// c = torch.as_strided_copy(x, [3,3],[1,1],0)
    /// c.sum().backward()
    /// x.grad.device, x.grad  # cuda:0, tensor([1., 2., 3., 2., 1.])
    /// ```
    #[test]
    fn core060_cuda_as_strided_copy_flows_gradients() {
        ensure_cuda_backend();
        let x = cuda_leaf_f64(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
        let c = x.as_strided_copy(&[3, 3], &[1, 1], Some(0)).unwrap();
        assert!(c.is_cuda(), "as_strided_copy must stay on CUDA");
        assert!(c.requires_grad());
        let s = c.sum_all().unwrap();
        backward(&s).unwrap();
        assert_eq!(cuda_grad_of(&x), vec![1.0, 2.0, 3.0, 2.0, 1.0]);
    }

    /// CUDA as_strided_scatter grads; src grad matches torch, base grad pins
    /// the finite-difference contract (see CPU test + #1754 for the
    /// documented torch 2.11.0 divergence).
    ///
    /// ```python
    /// base = torch.zeros(6, dtype=torch.float64, device="cuda", requires_grad=True)
    /// src = torch.tensor([10.,20.,30.], dtype=torch.float64, device="cuda", requires_grad=True)
    /// out = torch.as_strided_scatter(base, src, [3],[2],0)
    /// (out * torch.arange(1.,7., device="cuda")).sum().backward()
    /// src.grad   # tensor([1., 3., 5.], device='cuda:0')   (== finite diff)
    /// base.grad  # torch 2.11.0: [1,0,3,0,5,0]; finite diff: [0,2,0,4,0,6]
    /// ```
    #[test]
    fn core060_cuda_as_strided_scatter_flows_gradients() {
        ensure_cuda_backend();
        let base = cuda_leaf_f64(&[0.0; 6], &[6]);
        let src = cuda_leaf_f64(&[10.0, 20.0, 30.0], &[3]);
        let out = base.as_strided_scatter(&src, &[3], &[2], Some(0)).unwrap();
        assert!(out.is_cuda(), "as_strided_scatter must stay on CUDA");
        assert!(out.requires_grad());
        let w = leaf_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], false)
            .to(Device::Cuda(0))
            .unwrap();
        let s = out.mul_t(&w).unwrap().sum_all().unwrap();
        backward(&s).unwrap();
        assert_eq!(cuda_grad_of(&base), vec![0.0, 2.0, 0.0, 4.0, 0.0, 6.0]);
        assert_eq!(cuda_grad_of(&src), vec![1.0, 3.0, 5.0]);
    }

    /// f32 CUDA sliding window — exercises the f32 kernel arms.
    ///
    /// ```python
    /// x = torch.arange(1., 6., dtype=torch.float32, device="cuda", requires_grad=True)
    /// x.as_strided([3,3],[1,1],0).sum().backward()
    /// x.grad  # tensor([1., 2., 3., 2., 1.], device='cuda:0')
    /// ```
    #[test]
    fn core059_cuda_overlapping_backward_f32() {
        ensure_cuda_backend();
        let x: Tensor<f32> = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0, 5.0]),
            vec![5],
            false,
        )
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(true);
        let v = x.as_strided(&[3, 3], &[1, 1], Some(0)).unwrap();
        let s = v.contiguous().unwrap().sum_all().unwrap();
        backward(&s).unwrap();
        let g = x.grad().unwrap().expect("grad");
        assert!(g.is_cuda());
        assert_eq!(g.data_vec().unwrap(), vec![1.0f32, 2.0, 3.0, 2.0, 1.0]);
    }
}
