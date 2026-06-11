//! Red-then-green regression tests for audit finding CORE-011 (crosslink
//! #1705): `Tensor::view_operation` — the AUTOGRAD-aware sibling of
//! `view_reshape` — materializes non-contiguous inputs via
//! `data_vec() + TensorStorage::cpu` and recurses, so reshape / flatten /
//! squeeze / unsqueeze on a non-contiguous CUDA tensor stays CUDA with
//! gradients disabled but SILENTLY lands on CPU with gradients enabled.
//! `view_reshape` was already fixed (#750) to materialize through the
//! device-aware `contiguous()` path; this suite pins the same contract on
//! `view_operation`.
//!
//! Every expectation below is quoted from a LIVE torch session
//! (torch 2.11.0+cu130, RTX 3090; R-ORACLE-1 path (b)):
//!
//! ```text
//! >>> x = torch.arange(6., device='cuda').reshape(2,3).requires_grad_(True)
//! >>> v = x.t()                            # non-contiguous CUDA view
//! >>> v.reshape(6).device, v.flatten().device, v.unsqueeze(0).device
//! (device(type='cuda', index=0),) * 3      # all stay cuda:0, non-leaf
//! >>> y = v.reshape(6); y                  # tensor([0., 3., 1., 4., 2., 5.], ...)
//! >>> (y * torch.arange(1., 7., device='cuda')).sum().backward()
//! >>> x.grad                               # tensor([[1., 3., 5.],
//! ...                                      #         [2., 4., 6.]], device='cuda:0')
//! ```
//!
//! Tolerance justification (R-ORACLE-5): NONE — all values are small
//! integers exactly representable in f32; products/sums stay below 2^24.
//! Exact equality throughout.

use ferrotorch_core::grad_fns::arithmetic::mul;
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::grad_fns::shape::{reshape, transpose_2d};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

#[cfg(feature = "gpu")]
use ferrotorch_core::Device;
#[cfg(feature = "gpu")]
use ferrotorch_core::grad_fns::shape::{flatten, movedim, squeeze, unsqueeze};

/// CPU leaf with `requires_grad = false`; callers opt in via
/// `.requires_grad_(true)` AFTER any device transfer so the tracked tensor
/// is a true leaf on its final device (robust to the CORE-012 fix, which
/// makes `.to()` of a tracking leaf a non-leaf, as in torch).
fn plain_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

// CPU no-drift pin: the materialization path must keep producing torch's
// forward and backward values. torch (same session as the module doc,
// device='cpu'): y = x.t().reshape(6) == [0, 3, 1, 4, 2, 5];
// (y * [1..6]).sum().backward() -> x.grad == [[1, 3, 5], [2, 4, 6]].
#[test]
fn core011_cpu_reshape_noncontiguous_values_and_grad() {
    let x = plain_f32(&[0., 1., 2., 3., 4., 5.], &[2, 3]).requires_grad_(true);
    let v = transpose_2d(&x).expect("transpose");
    assert!(!v.is_contiguous(), "precondition: v must be non-contiguous");
    let y = reshape(&v, &[6]).expect("reshape through view_operation");
    assert_eq!(y.data().unwrap(), &[0.0f32, 3.0, 1.0, 4.0, 2.0, 5.0]);
    let w = plain_f32(&[1., 2., 3., 4., 5., 6.], &[6]);
    let loss = sum(&mul(&y, &w).expect("mul")).expect("sum");
    loss.backward().expect("backward");
    let g = x.grad().expect("grad access").expect("leaf grad present");
    assert_eq!(g.data().unwrap(), &[1.0f32, 3.0, 5.0, 2.0, 4.0, 6.0]);
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

    /// CUDA leaf `[[0,1,2],[3,4,5]]` with `requires_grad = true`, plus its
    /// transposed (3, 2) non-contiguous tracking view.
    fn cuda_leaf_and_t() -> (Tensor<f32>, Tensor<f32>) {
        let x = plain_f32(&[0., 1., 2., 3., 4., 5.], &[2, 3])
            .to(Device::Cuda(0))
            .expect("upload x")
            .requires_grad_(true);
        let v = transpose_2d(&x).expect("transpose");
        assert!(!v.is_contiguous(), "precondition: v must be non-contiguous");
        assert_eq!(v.device(), Device::Cuda(0));
        assert!(v.requires_grad());
        (x, v)
    }

    // The audit probe: reshape of a non-contiguous CUDA view WITH grad
    // enabled. Pre-fix: result silently lands on CPU. torch oracle:
    // v.reshape(6).device == cuda:0, values [0, 3, 1, 4, 2, 5].
    #[test]
    fn core011_gpu_reshape_noncontiguous_stays_cuda() {
        ensure_cuda_backend();
        let (_x, v) = cuda_leaf_and_t();
        let y = reshape(&v, &[6]).expect("reshape");
        assert_eq!(
            y.device(),
            Device::Cuda(0),
            "reshape of a non-contiguous CUDA view with grad enabled must stay CUDA (CORE-011)"
        );
        assert_eq!(
            y.cpu().expect("D2H").data().unwrap(),
            &[0.0f32, 3.0, 1.0, 4.0, 2.0, 5.0],
            "torch oracle: v.reshape(6) == [0, 3, 1, 4, 2, 5]"
        );
    }

    // torch oracle: v.flatten().device == cuda:0, same values as reshape(6).
    #[test]
    fn core011_gpu_flatten_noncontiguous_stays_cuda() {
        ensure_cuda_backend();
        let (_x, v) = cuda_leaf_and_t();
        let y = flatten(&v).expect("flatten");
        assert_eq!(
            y.device(),
            Device::Cuda(0),
            "flatten of a non-contiguous CUDA view with grad enabled must stay CUDA (CORE-011)"
        );
        assert_eq!(
            y.cpu().expect("D2H").data().unwrap(),
            &[0.0f32, 3.0, 1.0, 4.0, 2.0, 5.0],
            "torch oracle: v.flatten() == [0, 3, 1, 4, 2, 5]"
        );
    }

    // torch oracle: v.unsqueeze(0).device == cuda:0, shape (1, 3, 2).
    #[test]
    fn core011_gpu_unsqueeze_noncontiguous_stays_cuda() {
        ensure_cuda_backend();
        let (_x, v) = cuda_leaf_and_t();
        let y = unsqueeze(&v, 0).expect("unsqueeze");
        assert_eq!(
            y.device(),
            Device::Cuda(0),
            "unsqueeze of a non-contiguous CUDA view with grad enabled must stay CUDA (CORE-011)"
        );
        assert_eq!(y.shape(), &[1, 3, 2]);
        assert_eq!(
            y.cpu().expect("D2H").data().unwrap(),
            &[0.0f32, 3.0, 1.0, 4.0, 2.0, 5.0]
        );
    }

    // squeeze needs a size-1 axis on a non-contiguous tensor:
    // x: CUDA leaf shape (2, 1, 3); v = movedim(x, [2], [0]) — shape
    // (3, 2, 1), strides [1, 3, 3], non-contiguous. torch oracle
    // (same session): v.squeeze(2).device == cuda:0, values
    // [[0, 3], [1, 4], [2, 5]].
    #[test]
    fn core011_gpu_squeeze_noncontiguous_stays_cuda() {
        ensure_cuda_backend();
        let x = plain_f32(&[0., 1., 2., 3., 4., 5.], &[2, 1, 3])
            .to(Device::Cuda(0))
            .expect("upload x")
            .requires_grad_(true);
        let v = movedim(&x, &[2], &[0]).expect("movedim");
        assert!(!v.is_contiguous(), "precondition: v must be non-contiguous");
        let y = squeeze(&v, 2).expect("squeeze");
        assert_eq!(
            y.device(),
            Device::Cuda(0),
            "squeeze of a non-contiguous CUDA view with grad enabled must stay CUDA (CORE-011)"
        );
        assert_eq!(y.shape(), &[3, 2]);
        assert_eq!(
            y.cpu().expect("D2H").data().unwrap(),
            &[0.0f32, 3.0, 1.0, 4.0, 2.0, 5.0]
        );
    }

    // R-ORACLE-3: gradient FLOW back to the CUDA leaf, on the leaf's
    // device. torch oracle: (v.reshape(6) * [1..6]).sum().backward() ->
    // x.grad device cuda:0, values [[1, 3, 5], [2, 4, 6]].
    #[test]
    fn core011_gpu_reshape_backward_reaches_cuda_leaf() {
        ensure_cuda_backend();
        let (x, v) = cuda_leaf_and_t();
        let y = reshape(&v, &[6]).expect("reshape");
        let w = plain_f32(&[1., 2., 3., 4., 5., 6.], &[6])
            .to(Device::Cuda(0))
            .expect("upload w");
        let loss = sum(&mul(&y, &w).expect("mul")).expect("sum");
        loss.backward().expect("backward");
        let g = x.grad().expect("grad access").expect("leaf grad present");
        assert_eq!(
            g.device(),
            Device::Cuda(0),
            "leaf gradient must be CUDA-resident, like torch's (CORE-011)"
        );
        assert_eq!(
            g.cpu().expect("D2H").data().unwrap(),
            &[1.0f32, 3.0, 5.0, 2.0, 4.0, 6.0],
            "torch oracle: x.grad == [[1, 3, 5], [2, 4, 6]]"
        );
    }
}
