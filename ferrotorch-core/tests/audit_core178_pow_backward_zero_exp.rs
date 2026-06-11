//! Red-then-green regression tests for audit finding CORE-178 (crosslink
//! #1872): `PowBackward` computes `grad * exp * a^(exp-1)` unconditionally,
//! so for `exp == 0` every element where `a == 0` evaluates `0 * a^(-1) =
//! 0 * inf = NaN`. Upstream returns zeros BEFORE evaluating `a^(exp-1)`:
//!
//! `pytorch/torch/csrc/autograd/FunctionsManual.cpp:537-540`:
//! ```cpp
//! Tensor pow_backward(Tensor grad, const Tensor& self, const Scalar& exponent) {
//!   if (exponent.equal(0.0)) {
//!     return at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
//!   } else { ... }
//! ```
//!
//! Every numerical expectation below is quoted from a LIVE torch session
//! (torch 2.11.0+cu130, R-ORACLE-1 path (b)); the generating snippet is
//! pasted in a comment next to each expected block.
//!
//! Coverage: CPU direct branch (f32 + f64), weighted-cotangent (mul-routed)
//! branch, create_graph (higher-order) branch, CUDA branch with
//! gradient-device assertion (R-ORACLE-3).

use ferrotorch_core::autograd::grad;
use ferrotorch_core::grad_fns::arithmetic::{mul, pow};
use ferrotorch_core::grad_fns::reduction::sum;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

#[cfg(feature = "gpu")]
use ferrotorch_core::Device;

fn leaf_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}
fn leaf_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

// No tolerance needed anywhere in this file: torch's pow backward at
// exponent zero is EXACTLY zeros_like(self) (a constant, not a computed
// value), so every assertion is exact equality against 0.0.

// torch oracle (torch 2.11.0+cu130):
//   >>> x = torch.tensor([0.0, 1.0, -2.0, 0.0, 3.5], requires_grad=True)
//   >>> y = x.pow(0.0)
//   >>> y.sum().backward()
//   >>> y      # tensor([1., 1., 1., 1., 1.])
//   >>> x.grad # tensor([0., 0., 0., 0., 0.])
#[test]
fn core178_pow_zero_exp_backward_cpu_f32() {
    let x = leaf_f32(&[0.0, 1.0, -2.0, 0.0, 3.5], &[5]);
    let y = pow(&x, 0.0).expect("forward pow exp=0");
    // Forward is pinned elsewhere; sanity-check it stays all-ones.
    for (i, &v) in y.data().unwrap().iter().enumerate() {
        assert_eq!(v, 1.0, "forward x.pow(0) index {i}");
    }
    let loss = sum(&y).expect("sum");
    loss.backward().expect("backward");
    let g = x.grad().unwrap().expect("grad present");
    // Audit probe: pre-fix this is [NaN, 0, -0, NaN, 0] — NaN wherever
    // a == 0 (0 * a^(-1) = 0 * inf). torch: exact zeros.
    for (i, &v) in g.data().unwrap().iter().enumerate() {
        assert!(
            v == 0.0,
            "grad index {i}: got {v}, torch oracle 0.0 (zeros_like before a^(exp-1))"
        );
    }
}

// Same oracle in float64:
//   >>> x = torch.tensor([0.0, 1.0, -2.0, 0.0, 3.5], dtype=torch.float64, requires_grad=True)
//   >>> x.pow(0.0).sum().backward()
//   >>> x.grad # tensor([0., 0., 0., 0., 0.], dtype=torch.float64)
#[test]
fn core178_pow_zero_exp_backward_cpu_f64() {
    let x = leaf_f64(&[0.0, 1.0, -2.0, 0.0, 3.5], &[5]);
    let y = pow(&x, 0.0).expect("forward pow exp=0");
    let loss = sum(&y).expect("sum");
    loss.backward().expect("backward");
    let g = x.grad().unwrap().expect("grad present");
    for (i, &v) in g.data().unwrap().iter().enumerate() {
        assert!(v == 0.0, "grad index {i}: got {v}, torch oracle 0.0");
    }
}

// Non-uniform cotangent — the zeros must not be scaled by grad_output:
//   >>> x = torch.tensor([0.0, -0.0, 5.0], requires_grad=True)
//   >>> y = x.pow(0.0)
//   >>> y.backward(torch.tensor([2.0, 3.0, 4.0]))
//   >>> x.grad # tensor([0., 0., 0.])
#[test]
fn core178_pow_zero_exp_backward_weighted_cotangent_f32() {
    let x = leaf_f32(&[0.0, -0.0, 5.0], &[3]);
    let y = pow(&x, 0.0).expect("forward pow exp=0");
    let w =
        Tensor::from_storage(TensorStorage::cpu(vec![2.0f32, 3.0, 4.0]), vec![3], false).unwrap();
    let loss = sum(&mul(&y, &w).expect("weight")).expect("sum");
    loss.backward().expect("backward");
    let g = x.grad().unwrap().expect("grad present");
    for (i, &v) in g.data().unwrap().iter().enumerate() {
        assert!(v == 0.0, "grad index {i}: got {v}, torch oracle 0.0");
    }
}

// create_graph (higher-order) branch — torch:
//   >>> x = torch.tensor([0.0, 2.0], requires_grad=True)
//   >>> y = x.pow(0.0).sum()
//   >>> g, = torch.autograd.grad(y, x, create_graph=True)
//   >>> g  # tensor([0., 0.])  (constant; g.grad_fn is None upstream)
#[test]
fn core178_pow_zero_exp_create_graph_cpu_f32() {
    let x = leaf_f32(&[0.0, 2.0], &[2]);
    let y = sum(&pow(&x, 0.0).expect("forward pow exp=0")).expect("sum");
    let grads = grad(&y, &[&x], true, true).expect("grad create_graph=true");
    let g = grads[0].as_ref().expect("grad present");
    for (i, &v) in g.data().unwrap().iter().enumerate() {
        assert!(v == 0.0, "grad index {i}: got {v}, torch oracle 0.0");
    }
}

// CUDA branch (gpu feature lane) — torch:
//   >>> xc = torch.tensor([0.0, 1.0, -2.0], device='cuda', requires_grad=True)
//   >>> xc.pow(0.0).sum().backward()
//   >>> xc.grad         # tensor([0., 0., 0.], device='cuda:0')
//   >>> xc.grad.device  # device(type='cuda', index=0)
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

    #[test]
    fn core178_gpu_pow_zero_exp_backward_f32() {
        ensure_cuda_backend();
        let dev = Device::Cuda(0);
        let x = leaf_f32(&[0.0, 1.0, -2.0], &[3]).to(dev).expect("upload x");
        let y = pow(&x, 0.0).expect("forward pow exp=0 cuda");
        let loss = sum(&y).expect("sum");
        loss.backward().expect("backward cuda");
        let g = x.grad().unwrap().expect("grad present");
        // R-ORACLE-3: the zero gradient must be CUDA-resident, not a CPU
        // demotion.
        assert_eq!(g.device(), dev, "grad must be CUDA-resident");
        let g_cpu = g.cpu().expect("D2H readback");
        for (i, &v) in g_cpu.data().unwrap().iter().enumerate() {
            assert!(v == 0.0, "grad index {i}: got {v}, torch oracle 0.0");
        }
    }

    #[test]
    fn core178_gpu_pow_zero_exp_backward_f64() {
        ensure_cuda_backend();
        let dev = Device::Cuda(0);
        let x = leaf_f64(&[0.0, 4.0], &[2]).to(dev).expect("upload x");
        let y = pow(&x, 0.0).expect("forward pow exp=0 cuda f64");
        let loss = sum(&y).expect("sum");
        loss.backward().expect("backward cuda f64");
        let g = x.grad().unwrap().expect("grad present");
        assert_eq!(g.device(), dev, "grad must be CUDA-resident");
        let g_cpu = g.cpu().expect("D2H readback");
        for (i, &v) in g_cpu.data().unwrap().iter().enumerate() {
            assert!(v == 0.0, "grad index {i}: got {v}, torch oracle 0.0");
        }
    }
}
