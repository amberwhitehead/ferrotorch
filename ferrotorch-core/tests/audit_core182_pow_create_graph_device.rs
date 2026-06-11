//! Red-then-green regression tests for audit finding CORE-182 (crosslink
//! #1876): `PowBackward`'s higher-order (`create_graph`) branch builds its
//! exponent tensor with `TensorStorage::cpu` and `mul()`s it against the
//! CUDA-resident `a^(exp-1)`, so any `grad(..., create_graph=true)` through
//! a CUDA `pow` fails with an unrelated-looking `DeviceMismatch`. The
//! non-higher-order GPU branch directly below performs the missing
//! `.to(device)` hop; the higher-order branch must build on `a`'s device
//! the same way.
//!
//! Upstream reference: `pow_backward` at
//! `pytorch/torch/csrc/autograd/FunctionsManual.cpp:537-549` computes
//! `grad * (exp * self.pow(exp - 1)).conj()` with scalar-tensor kernels on
//! `self`'s device — there is no host-resident operand for CUDA inputs.
//!
//! Every numerical expectation below is quoted from a LIVE torch session
//! (torch 2.11.0+cu130, R-ORACLE-1 path (b)); identical values on CPU:
//!
//! ```text
//! >>> x = torch.tensor([2.0, 3.0], device='cuda', requires_grad=True)
//! >>> y = x.pow(3.0).sum()
//! >>> g, = torch.autograd.grad(y, x, create_graph=True)
//! >>> g            # tensor([12., 27.], device='cuda:0', grad_fn=...)
//! >>> g2, = torch.autograd.grad(g.sum(), x)
//! >>> g2           # tensor([12., 18.], device='cuda:0')  (= 6x)
//! ```
//!
//! Tolerance justification (R-ORACLE-5): NONE — inputs are small integers,
//! `3 * x^2` and `6 * x` are exact in f32 for |x| <= 3; exact equality.

use ferrotorch_core::autograd::grad;
use ferrotorch_core::grad_fns::arithmetic::pow;
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

#[cfg(feature = "gpu")]
use ferrotorch_core::Device;

fn leaf_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

// CPU no-drift pin: the higher-order branch must keep producing torch's
// first- AND second-order values through f(x) = sum(x^3):
//   f'(x) = 3x^2 -> [12., 27.] ; f''(x) = 6x -> [12., 18.]
#[test]
fn core182_cpu_pow_create_graph_first_and_second_order_f32() {
    let x = leaf_f32(&[2.0, 3.0], &[2]);
    let y = sum(&pow(&x, 3.0).expect("forward pow")).expect("sum");
    let grads = grad(&y, &[&x], true, true).expect("first-order grad create_graph=true");
    let g = grads[0].as_ref().expect("first-order grad present");
    assert_eq!(
        g.data().unwrap(),
        &[12.0f32, 27.0],
        "3x^2 vs torch [12, 27]"
    );
    // Second order: differentiate sum(g) again. R-ORACLE-3: this asserts
    // gradient FLOW back to the leaf x, not a requires_grad flag.
    let g_sum = sum(g).expect("sum of first-order grad");
    let grads2 = grad(&g_sum, &[&x], false, false).expect("second-order grad");
    let g2 = grads2[0].as_ref().expect("second-order grad present");
    assert_eq!(g2.data().unwrap(), &[12.0f32, 18.0], "6x vs torch [12, 18]");
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

    // The audit probe: grad(..., create_graph=true) through a CUDA pow.
    // Pre-fix: Err(DeviceMismatch) out of `mul`'s device guard (the
    // exponent tensor is CPU-resident). torch oracle: succeeds, returns
    // tensor([12., 27.], device='cuda:0').
    #[test]
    fn core182_gpu_pow_create_graph_first_order_f32() {
        ensure_cuda_backend();
        let dev = Device::Cuda(0);
        let x = leaf_f32(&[2.0, 3.0], &[2]).to(dev).expect("upload x");
        let y = sum(&pow(&x, 3.0).expect("forward pow cuda")).expect("sum");
        let grads = grad(&y, &[&x], true, true)
            .expect("grad(..., create_graph=true) through CUDA pow must succeed");
        let g = grads[0].as_ref().expect("first-order grad present");
        // R-ORACLE-3: the gradient must be CUDA-resident, like torch's.
        assert_eq!(g.device(), dev, "first-order grad must be CUDA-resident");
        assert_eq!(
            g.cpu().expect("D2H").data().unwrap(),
            &[12.0f32, 27.0],
            "3x^2 vs torch cuda [12, 27]"
        );
    }

    // Second-order value through the CUDA create_graph path — torch:
    //   >>> g2, = torch.autograd.grad(g.sum(), x)   # tensor([12., 18.], cuda:0)
    #[test]
    fn core182_gpu_pow_create_graph_second_order_f32() {
        ensure_cuda_backend();
        let dev = Device::Cuda(0);
        let x = leaf_f32(&[2.0, 3.0], &[2]).to(dev).expect("upload x");
        let y = sum(&pow(&x, 3.0).expect("forward pow cuda")).expect("sum");
        let grads = grad(&y, &[&x], true, true).expect("first-order grad create_graph=true");
        let g = grads[0].as_ref().expect("first-order grad present");
        let g_sum = sum(g).expect("sum of first-order grad");
        let grads2 = grad(&g_sum, &[&x], false, false).expect("second-order grad");
        let g2 = grads2[0].as_ref().expect("second-order grad present");
        assert_eq!(g2.device(), dev, "second-order grad must be CUDA-resident");
        assert_eq!(
            g2.cpu().expect("D2H").data().unwrap(),
            &[12.0f32, 18.0],
            "6x vs torch cuda [12, 18]"
        );
    }
}
