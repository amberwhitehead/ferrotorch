//! CORE-043 (#1737, CLASS-U/S Medium) regression battery: the host-mask
//! `where_` entry point (`grad_fns/comparison.rs`, surfaced as
//! `Tensor::where_t` / `Tensor::where_bt_t`) must validate its operands with
//! structured errors instead of `debug_assert_eq!` (compiled out in release —
//! the default test lane here), and must reject cross-device operands.
//!
//! Pre-fix observed behavior (R-AHON-1 probe at HEAD, pasted in #1737):
//! - condition longer than the operands: extra mask entries silently ignored
//!   by nested `zip` truncation (release builds only — debug builds panic on
//!   the `debug_assert_eq!`).
//! - `y` longer than `x`: `y`'s tail silently dropped, result returned under
//!   `x`'s shape.
//! - equal-numel different shapes (`[2,3]` vs `[3,2]`): accepted, result
//!   stamped with `x`'s shape.
//! - `where_bt`: condition shape validated by numel only, so a `[6]` mask
//!   against `[2,3]` operands was accepted.
//! - mixed-device `x` CUDA + `y` CPU: silently accepted (`y` downloaded via
//!   `data_vec`, result uploaded to `x`'s device).
//!
//! torch contract (live torch 2.11.0+cu130, RTX 3090 — session quoted per
//! test, R-ORACLE-1(b)): each of these raises `RuntimeError`. This host-mask
//! surface implements the same-shape (non-broadcasting) subset of
//! `torch.where`; every red case below is one torch itself rejects, so a
//! structured `Err` is exact parity (broadcasting `torch.where` lives in
//! `grad_fns::indexing::where_cond_bcast`).

use ferrotorch_core::TensorStorage;
use ferrotorch_core::error::FerrotorchError;
use ferrotorch_core::tensor::Tensor;

fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

/// (a) Condition longer than the operands → structured ShapeMismatch.
///
/// Live torch 2.11.0+cu130:
/// ```text
/// >>> torch.where(torch.tensor([True]*6), torch.arange(4.), torch.arange(4.)+10)
/// RuntimeError: The size of tensor a (6) must match the size of tensor b (4)
/// at non-singleton dimension 0
/// ```
/// Pre-fix release behavior: Ok — the two extra mask entries are silently
/// ignored by zip truncation (`debug_assert_eq!` is compiled out).
#[test]
fn where_rejects_condition_longer_than_operands() {
    let x = t(&[0.0, 1.0, 2.0, 3.0], &[4]);
    let y = t(&[10.0, 11.0, 12.0, 13.0], &[4]);
    let cond = vec![true; 6];
    let r = x.where_t(&cond, &y);
    match r {
        Err(FerrotorchError::ShapeMismatch { .. }) => {}
        other => panic!(
            "where_ with cond len 6 vs numel 4 must be ShapeMismatch \
             (torch: RuntimeError); got {other:?}"
        ),
    }
}

/// (a') Condition shorter than the operands → structured ShapeMismatch.
///
/// Live torch (same session): a `[2]` bool mask against `[4]` operands raises
/// `RuntimeError: The size of tensor a (2) must match the size of tensor b
/// (4) at non-singleton dimension 0` (non-broadcastable: 2 vs 4).
#[test]
fn where_rejects_condition_shorter_than_operands() {
    let x = t(&[0.0, 1.0, 2.0, 3.0], &[4]);
    let y = t(&[10.0, 11.0, 12.0, 13.0], &[4]);
    let cond = vec![true, false];
    let r = x.where_t(&cond, &y);
    match r {
        Err(FerrotorchError::ShapeMismatch { .. }) => {}
        other => panic!(
            "where_ with cond len 2 vs numel 4 must be ShapeMismatch \
             (torch: RuntimeError); got {other:?}"
        ),
    }
}

/// (b) `y` numel differs from `x` → structured ShapeMismatch.
///
/// Live torch 2.11.0+cu130:
/// ```text
/// >>> torch.where(torch.tensor([True,False,True,False]), torch.arange(4.),
/// ...             torch.arange(6.))
/// RuntimeError: The size of tensor a (4) must match the size of tensor b (6)
/// at non-singleton dimension 0
/// ```
/// Pre-fix release behavior: Ok — `y`'s last two elements silently truncated.
#[test]
fn where_rejects_y_numel_mismatch() {
    let x = t(&[0.0, 1.0, 2.0, 3.0], &[4]);
    let y = t(&[10.0, 11.0, 12.0, 13.0, 14.0, 15.0], &[6]);
    let cond = vec![true, false, true, false];
    let r = x.where_t(&cond, &y);
    match r {
        Err(FerrotorchError::ShapeMismatch { .. }) => {}
        other => panic!(
            "where_ with x numel 4 vs y numel 6 must be ShapeMismatch \
             (torch: RuntimeError); got {other:?}"
        ),
    }
}

/// (c) Equal-numel, different shapes → structured ShapeMismatch.
///
/// Live torch 2.11.0+cu130:
/// ```text
/// >>> x = torch.arange(6.).reshape(2,3); y = torch.arange(6.).reshape(3,2)
/// >>> cond = torch.tensor([[True,False,True],[False,True,False]])
/// >>> torch.where(cond, x, y)
/// RuntimeError: The size of tensor a (3) must match the size of tensor b (2)
/// at non-singleton dimension 1
/// ```
/// Pre-fix behavior (debug AND release — numel matches, so the
/// debug_assert passes too): Ok, result stamped with `x`'s `[2,3]` shape.
#[test]
fn where_rejects_equal_numel_different_shapes() {
    let x = t(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
    let y = t(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[3, 2]);
    let cond = vec![true, false, true, false, true, false];
    let r = x.where_t(&cond, &y);
    match r {
        Err(FerrotorchError::ShapeMismatch { .. }) => {}
        other => panic!(
            "where_ with shapes [2,3] vs [3,2] (equal numel) must be \
             ShapeMismatch (torch: RuntimeError); got {other:?}"
        ),
    }
}

/// (d) `where_bt`: condition SHAPE must match, not just numel.
///
/// Live torch (same session): a `[6]` bool mask against `[2,3]` operands is
/// non-broadcastable (6 vs 3 at the trailing dim) →
/// `RuntimeError: The size of tensor a (6) must match the size of tensor b
/// (3) at non-singleton dimension 1`.
/// Pre-fix behavior: `where_bt` checked `cond.numel() == x.numel()` only, so
/// the `[6]` mask was accepted against `[2,3]` operands.
#[test]
fn where_bt_rejects_condition_shape_mismatch_with_equal_numel() {
    use ferrotorch_core::BoolTensor;
    let x = t(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
    let y = t(&[6.0, 7.0, 8.0, 9.0, 10.0, 11.0], &[2, 3]);
    let cond = BoolTensor::from_vec(vec![true, false, true, false, true, false], vec![6]).unwrap();
    let r = x.where_bt_t(&cond, &y);
    match r {
        Err(FerrotorchError::ShapeMismatch { .. }) => {}
        other => panic!(
            "where_bt with cond shape [6] vs x shape [2,3] must be \
             ShapeMismatch (torch: RuntimeError); got {other:?}"
        ),
    }
}

/// Happy path stays intact: same-shape CPU select with grad flow to BOTH
/// leaves (R-ORACLE-3: gradient values reaching the original leaves).
///
/// Live torch 2.11.0+cu130:
/// ```text
/// >>> x = torch.tensor([1.,2.,3.,4.], requires_grad=True)
/// >>> y = torch.tensor([10.,20.,30.,40.], requires_grad=True)
/// >>> out = torch.where(torch.tensor([True,False,True,False]), x, y)
/// >>> out
/// tensor([ 1., 20.,  3., 40.], grad_fn=<WhereBackward0>)
/// >>> out.sum().backward(); x.grad, y.grad
/// (tensor([1., 0., 1., 0.]), tensor([0., 1., 0., 1.]))
/// ```
#[test]
fn where_same_shape_cpu_still_selects_and_routes_grads() {
    use ferrotorch_core::autograd::graph::backward_with_grad;
    let x = t(&[1.0, 2.0, 3.0, 4.0], &[4]).requires_grad_(true);
    let y = t(&[10.0, 20.0, 30.0, 40.0], &[4]).requires_grad_(true);
    let cond = vec![true, false, true, false];
    let out = x.where_t(&cond, &y).unwrap();
    assert_eq!(out.data().unwrap(), &[1.0, 20.0, 3.0, 40.0]);

    let ones = t(&[1.0, 1.0, 1.0, 1.0], &[4]);
    backward_with_grad(&out, Some(&ones)).unwrap();
    let gx = x.grad().unwrap().unwrap();
    let gy = y.grad().unwrap().unwrap();
    assert_eq!(gx.data().unwrap(), &[1.0, 0.0, 1.0, 0.0]);
    assert_eq!(gy.data().unwrap(), &[0.0, 1.0, 0.0, 1.0]);
}

// ---------------------------------------------------------------------------
// CUDA cases (gpu feature + hardware)
// ---------------------------------------------------------------------------
#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use ferrotorch_core::Device;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();
    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for the CORE-043 GPU pins");
        });
    }

    /// (e) Mixed-device operands → structured DeviceMismatch.
    ///
    /// Live torch 2.11.0+cu130 (RTX 3090):
    /// ```text
    /// >>> torch.where(cond.cuda(), x.cuda(), y_cpu)
    /// RuntimeError: Expected all tensors to be on the same device, but found
    /// at least two devices, cuda:0 and cpu!
    /// ```
    /// Pre-fix behavior: Ok — `y` read via `data_vec`, result uploaded to
    /// `x`'s CUDA device, the cross-device mix silently accepted.
    #[test]
    fn where_rejects_mixed_device_operands() {
        ensure_cuda_backend();
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[4]).to(Device::Cuda(0)).unwrap();
        let y = t(&[10.0, 20.0, 30.0, 40.0], &[4]);
        let cond = vec![true, false, true, false];
        let r = x.where_t(&cond, &y);
        match r {
            Err(FerrotorchError::DeviceMismatch { expected, got }) => {
                assert_eq!(expected, Device::Cuda(0));
                assert_eq!(got, Device::Cpu);
            }
            other => panic!(
                "where_ with x on cuda:0 and y on cpu must be DeviceMismatch \
                 (torch: 'Expected all tensors to be on the same device, but \
                 found at least two devices, cuda:0 and cpu!'); got {other:?}"
            ),
        }
    }

    /// First-class BoolTensor conditions follow torch's tensor device rule:
    /// the condition must be on the same device as `x` and `y`. Only the
    /// host-mask `where_t(&[bool], ...)` convenience entry uploads a raw mask.
    ///
    /// Live torch 2.11.0+cu130:
    /// ```text
    /// >>> torch.where(torch.tensor([True,False,True,False]),
    /// ...             torch.tensor([1.,2.,3.,4.], device='cuda'),
    /// ...             torch.tensor([10.,20.,30.,40.], device='cuda'))
    /// RuntimeError: Expected all tensors to be on the same device, but found
    /// at least two devices, cuda:0 and cpu!
    /// ```
    #[test]
    fn where_bt_rejects_cpu_condition_with_cuda_operands() {
        use ferrotorch_core::BoolTensor;

        ensure_cuda_backend();
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[4]).to(Device::Cuda(0)).unwrap();
        let y = t(&[10.0, 20.0, 30.0, 40.0], &[4])
            .to(Device::Cuda(0))
            .unwrap();
        let cond =
            BoolTensor::from_vec(vec![true, false, true, false], vec![4]).expect("condition");
        let r = x.where_bt_t(&cond, &y);
        match r {
            Err(FerrotorchError::DeviceMismatch { expected, got }) => {
                assert_eq!(expected, Device::Cuda(0));
                assert_eq!(got, Device::Cpu);
            }
            other => panic!(
                "where_bt with CPU condition and CUDA x/y must be DeviceMismatch \
                 (torch: expected all tensors on same device); got {other:?}"
            ),
        }
    }

    /// (f) Valid same-device CUDA call: correct values, result on CUDA
    /// (R-ORACLE-3 device assert). The host `&[bool]` mask entry performs a
    /// DOCUMENTED host round trip (R-LOUD-2; see the `where_` doc-comment).
    ///
    /// Live torch 2.11.0+cu130 (RTX 3090):
    /// ```text
    /// >>> torch.where(torch.tensor([True,False,True,False], device='cuda'),
    /// ...             torch.tensor([1.,2.,3.,4.], device='cuda'),
    /// ...             torch.tensor([10.,20.,30.,40.], device='cuda'))
    /// tensor([ 1., 20.,  3., 40.], device='cuda:0')
    /// ```
    #[test]
    fn where_same_device_cuda_selects_and_stays_resident() {
        ensure_cuda_backend();
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[4]).to(Device::Cuda(0)).unwrap();
        let y = t(&[10.0, 20.0, 30.0, 40.0], &[4])
            .to(Device::Cuda(0))
            .unwrap();
        let cond = vec![true, false, true, false];
        let out = x.where_t(&cond, &y).unwrap();
        assert!(
            out.is_cuda(),
            "where_ on CUDA operands must return a CUDA result (got {:?})",
            out.device()
        );
        let host = out.to(Device::Cpu).unwrap();
        assert_eq!(host.data().unwrap(), &[1.0, 20.0, 3.0, 40.0]);
    }
}
