//! Red-then-green regression tests for audit finding CORE-012 (crosslink
//! #1706): `Tensor::to` / `Tensor::to_pinned` attach `ToDeviceBackward`
//! only when `requires_grad && !is_leaf`, so moving a requires-grad LEAF
//! across devices builds a fresh, independent leaf — the graph is severed
//! and backward on the transferred tensor can never reach the original
//! source leaf. PyTorch treats a differentiable `.to(other_device)` as a
//! copy WITH a backward edge, leaf or not.
//!
//! Every expectation below is quoted from a LIVE torch session
//! (torch 2.11.0+cu130, RTX 3090; R-ORACLE-1 path (b)):
//!
//! ```text
//! >>> x = torch.tensor([1., 2., 3.], requires_grad=True)
//! >>> y = x.to('cuda')
//! >>> y.is_leaf, type(y.grad_fn).__name__
//! (False, 'ToCopyBackward0')
//! >>> (y * y).sum().backward()
//! >>> x.grad, x.grad.device, y.grad
//! (tensor([2., 4., 6.]), device(type='cpu'), None)
//! >>> xc = torch.tensor([4., 5.], device='cuda', requires_grad=True)
//! >>> yc = xc.to('cpu'); (3 * yc).sum().backward()
//! >>> xc.grad            # tensor([3., 3.], device='cuda:0')
//! >>> xm = torch.tensor([1.], requires_grad=True); ym = xm.to('meta')
//! >>> ym.is_leaf, type(ym.grad_fn).__name__, ym.requires_grad
//! (False, 'ToCopyBackward0', True)
//! >>> ym.sum().backward()
//! NotImplementedError: Cannot copy out of meta tensor; no data!
//! ```
//!
//! Tolerance justification (R-ORACLE-5): NONE — all values are small
//! integers exactly representable in f32; exact equality throughout.

use ferrotorch_core::Device;
#[cfg(feature = "gpu")]
use ferrotorch_core::grad_fns::arithmetic::mul;
#[cfg(feature = "gpu")]
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

fn leaf_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

#[cfg(feature = "gpu")]
fn plain_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

// Default-lane red: transfer of a requires-grad CPU leaf to Meta must
// attach the transfer edge, exactly like any other device move.
// torch oracle: ym.is_leaf == False, grad_fn == ToCopyBackward0,
// requires_grad == True.
#[test]
fn core012_cpu_to_meta_leaf_keeps_graph() {
    let x = leaf_f32(&[1.0], &[1]);
    let y = x.to(Device::Meta).expect("to meta");
    assert!(
        y.requires_grad(),
        "requires_grad must propagate (torch: True)"
    );
    assert!(
        !y.is_leaf(),
        "x.to(Meta) of a requires-grad leaf must be a NON-leaf (torch: is_leaf=False) — CORE-012"
    );
    assert!(
        y.grad_fn().is_some(),
        "x.to(Meta) must carry a transfer grad_fn (torch: ToCopyBackward0) — CORE-012"
    );
}

// Default-lane red: backward through the meta transfer fails LOUDLY with
// the structured meta error, mirroring torch's
// `NotImplementedError: Cannot copy out of meta tensor; no data!`
// (the gradient of a meta tensor is meta; it cannot be copied out to the
// source device). R-ORACLE-4: exactly one contract — Err, never a
// silently absent gradient.
#[test]
fn core012_cpu_to_meta_backward_errors_loudly() {
    let x = leaf_f32(&[1.0], &[1]);
    let y = x.to(Device::Meta).expect("to meta");
    let err = y
        .backward()
        .expect_err("backward through a meta transfer must error (torch: NotImplementedError)");
    let msg = err.to_string();
    assert!(
        msg.contains("meta"),
        "error must name the meta device (torch: 'Cannot copy out of meta tensor'); got: {msg}"
    );
    assert!(
        x.grad().expect("grad access").is_none(),
        "no gradient may be fabricated for the source leaf"
    );
}

// Default-lane pin (torch no_grad oracle): under no_grad, `.to()` output
// does NOT track — requires_grad False, leaf, no grad_fn.
// torch: with no_grad(): x.to('cuda') -> (False, True, None); the
// device-agnostic Meta arm pins the same contract on the default lane.
#[test]
fn core012_cpu_to_meta_under_no_grad_does_not_track() {
    let x = leaf_f32(&[1.0], &[1]);
    let y = ferrotorch_core::autograd::no_grad::no_grad(|| x.to(Device::Meta)).expect("to meta");
    assert!(
        !y.requires_grad(),
        "under no_grad, .to() output must not require grad (torch: False)"
    );
    assert!(y.is_leaf());
    assert!(y.grad_fn().is_none());
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for the GPU lane of this suite");
        });
    }

    // The audit probe, forward direction. torch oracle:
    // y = x.to('cuda') -> is_leaf False, grad_fn ToCopyBackward0;
    // (y*y).sum().backward() -> x.grad == [2, 4, 6] on cpu.
    #[test]
    fn core012_gpu_cpu_to_cuda_leaf_grad_reaches_source() {
        ensure_cuda_backend();
        let x = leaf_f32(&[1., 2., 3.], &[3]);
        let y = x.to(Device::Cuda(0)).expect("upload");
        assert_eq!(y.device(), Device::Cuda(0));
        assert!(y.requires_grad());
        assert!(
            !y.is_leaf(),
            "x.to(cuda) of a requires-grad leaf must be a NON-leaf (torch: is_leaf=False)"
        );
        assert!(
            y.grad_fn().is_some(),
            "x.to(cuda) must carry a transfer grad_fn (torch: ToCopyBackward0)"
        );
        let loss = sum(&mul(&y, &y).expect("y*y")).expect("sum");
        loss.backward().expect("backward");
        // R-ORACLE-3: gradient FLOW to the ORIGINAL cpu leaf, on its device.
        let g = x
            .grad()
            .expect("grad access")
            .expect("the original CPU leaf must receive the gradient (CORE-012)");
        assert_eq!(
            g.device(),
            Device::Cpu,
            "leaf grad lives on the leaf's device"
        );
        assert_eq!(
            g.data().unwrap(),
            &[2.0f32, 4.0, 6.0],
            "torch oracle: x.grad == [2, 4, 6]"
        );
    }

    // Reverse direction. torch oracle: yc = xc.to('cpu');
    // (3*yc).sum().backward() -> xc.grad == [3, 3] on cuda:0.
    #[test]
    fn core012_gpu_cuda_to_cpu_leaf_grad_reaches_source() {
        ensure_cuda_backend();
        let xc = plain_f32(&[4., 5.], &[2])
            .to(Device::Cuda(0))
            .expect("upload")
            .requires_grad_(true); // true CUDA leaf
        let yc = xc.to(Device::Cpu).expect("download");
        assert_eq!(yc.device(), Device::Cpu);
        assert!(
            !yc.is_leaf() && yc.grad_fn().is_some(),
            "xc.to(cpu) of a requires-grad CUDA leaf must be a tracked non-leaf"
        );
        let w = plain_f32(&[3., 3.], &[2]);
        let loss = sum(&mul(&yc, &w).expect("3*yc")).expect("sum");
        loss.backward().expect("backward");
        let g = xc
            .grad()
            .expect("grad access")
            .expect("the original CUDA leaf must receive the gradient (CORE-012)");
        assert_eq!(
            g.device(),
            Device::Cuda(0),
            "leaf grad must be CUDA-resident, like torch's"
        );
        assert_eq!(
            g.cpu().expect("D2H").data().unwrap(),
            &[3.0f32, 3.0],
            "torch oracle: xc.grad == [3, 3]"
        );
    }

    // Pinned-memory path. torch oracle: pin_memory + to(cuda) is a
    // differentiable chain (PinMemoryBackward0 / ToCopyBackward0);
    // (yp*yp).sum().backward() -> xp.grad == 2*xp == [14, 16] on cpu.
    #[test]
    fn core012_gpu_to_pinned_leaf_grad_reaches_source() {
        ensure_cuda_backend();
        let xp = leaf_f32(&[7., 8.], &[2]);
        let yp = xp.to_pinned(Device::Cuda(0)).expect("pinned upload");
        assert_eq!(yp.device(), Device::Cuda(0));
        assert!(
            !yp.is_leaf() && yp.grad_fn().is_some(),
            "to_pinned of a requires-grad leaf must be a tracked non-leaf"
        );
        let loss = sum(&mul(&yp, &yp).expect("yp*yp")).expect("sum");
        loss.backward().expect("backward");
        let g = xp
            .grad()
            .expect("grad access")
            .expect("the original CPU leaf must receive the gradient (CORE-012)");
        assert_eq!(g.device(), Device::Cpu);
        assert_eq!(
            g.data().unwrap(),
            &[14.0f32, 16.0],
            "torch oracle: xp.grad == 2*xp == [14, 16]"
        );
    }
}
