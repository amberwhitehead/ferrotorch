//! Red-then-green regression tests for audit finding CORE-181 (crosslink
//! #1875): `RemainderBackward` computes the other-operand gradient with a
//! naive `floor(a / b)` chain, while torch defines it via
//! `div(..., rounding_mode="floor")` = `c10::div_floor_floating`:
//!
//! `pytorch/tools/autograd/derivatives.yaml:1455-1457`:
//! ```yaml
//! - name: remainder.Tensor(Tensor self, Tensor other) -> Tensor
//!   self: grad
//!   other: -grad * self.div(other, /*rounding_mode=*/"floor")
//! ```
//!
//! The exact fmod-based, fixup-corrected kernel
//! (`pytorch/c10/util/generic_math.h:34-58`) already exists in this file's
//! `floor_divide_inner`; the backward must route through it.
//!
//! Pin inputs are boundary cases where the naive chain and the exact
//! kernel DIVERGE, verified live (torch 2.11.0+cu130, R-ORACLE-1 path (b)):
//!
//!   a=-7.0,  b=0.7  (f32): naive floor(a/b) = -10.0, div_floor = -11.0
//!   a=-1e-30, b=1e30 (f32): naive floor(a/b) = -0.0,  div_floor = -1.0
//!     (the quotient -1e-60 underflows to -0; the fmod form recovers the
//!      true floor -1, exercising the signed-zero/fixup path)
//!   a=-7.0,  b=0.7  (f64): naive floor(a/b) = -10,   div_floor = -11.0
//!
//! Tolerance justification (R-ORACLE-5): NONE — every expected gradient is
//! a small integer (a floor-quotient, optionally times an integer
//! cotangent), exact in both f32 and f64; assertions are exact equality.

use ferrotorch_core::grad_fns::arithmetic::{mul, remainder};
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
fn t_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

/// `remainder(a, b).sum().backward()` with both leaves tracked; returns
/// (a.grad, b.grad) data.
fn rem_grads_f32(adata: &[f32], bdata: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let a = leaf_f32(adata, &[adata.len()]);
    let b = leaf_f32(bdata, &[bdata.len()]);
    let y = remainder(&a, &b).expect("forward remainder");
    sum(&y).expect("sum").backward().expect("backward");
    let ga = a.grad().unwrap().expect("a.grad present");
    let gb = b.grad().unwrap().expect("b.grad present");
    (ga.data().unwrap().to_vec(), gb.data().unwrap().to_vec())
}
fn rem_grads_f64(adata: &[f64], bdata: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let a = leaf_f64(adata, &[adata.len()]);
    let b = leaf_f64(bdata, &[bdata.len()]);
    let y = remainder(&a, &b).expect("forward remainder");
    sum(&y).expect("sum").backward().expect("backward");
    let ga = a.grad().unwrap().expect("a.grad present");
    let gb = b.grad().unwrap().expect("b.grad present");
    (ga.data().unwrap().to_vec(), gb.data().unwrap().to_vec())
}

// torch oracle (torch 2.11.0+cu130):
//   >>> a = torch.tensor([-7.0], requires_grad=True)
//   >>> b = torch.tensor([0.7], requires_grad=True)
//   >>> y = torch.remainder(a, b); y.sum().backward()
//   >>> y       # tensor([0.7000], grad_fn=...) = 0.6999998688697815
//   >>> a.grad  # tensor([1.])
//   >>> b.grad  # tensor([11.])
// Pre-fix ferrotorch: floor(-7/0.7) evaluates -10 in f32 arithmetic
// (the f32 quotient rounds to exactly -10.0), so db = 10.0, not 11.0.
#[test]
fn core181_rem_bgrad_floor_boundary_f32() {
    let (ga, gb) = rem_grads_f32(&[-7.0], &[0.7]);
    assert_eq!(ga, vec![1.0f32], "a.grad vs torch [1.0]");
    assert_eq!(
        gb,
        vec![11.0f32],
        "b.grad: torch oracle 11.0 (div_floor_floating -> -11), naive floor gives 10.0"
    );
}

// torch oracle:
//   >>> a = torch.tensor([-1e-30], requires_grad=True)
//   >>> b = torch.tensor([1e30], requires_grad=True)
//   >>> torch.remainder(a, b).sum().backward()
//   >>> a.grad  # tensor([1.])
//   >>> b.grad  # tensor([1.])
// Pre-fix ferrotorch: a/b = -1e-60 underflows to -0.0 in f32,
// floor(-0.0) = -0.0, so db = 0.0 where torch has 1.0.
#[test]
fn core181_rem_bgrad_underflow_signed_zero_f32() {
    let (ga, gb) = rem_grads_f32(&[-1e-30], &[1e30]);
    assert_eq!(ga, vec![1.0f32], "a.grad vs torch [1.0]");
    assert_eq!(
        gb,
        vec![1.0f32],
        "b.grad: torch oracle 1.0 (div_floor recovers floor = -1), naive underflows to -0"
    );
}

// f64 pin of the same boundary — torch oracle:
//   >>> a = torch.tensor([-7.0], dtype=torch.float64, requires_grad=True)
//   >>> b = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
//   >>> torch.remainder(a, b).sum().backward()
//   >>> b.grad  # tensor([11.], dtype=torch.float64)
//   (math.floor(-7.0/0.7) == -10 — the naive chain diverges in f64 too)
#[test]
fn core181_rem_bgrad_floor_boundary_f64() {
    let (ga, gb) = rem_grads_f64(&[-7.0], &[0.7]);
    assert_eq!(ga, vec![1.0f64], "a.grad vs torch [1.0]");
    assert_eq!(gb, vec![11.0f64], "b.grad vs torch [11.0]");
}

// torch oracle:
//   >>> a = torch.tensor([-1e-300], dtype=torch.float64, requires_grad=True)
//   >>> b = torch.tensor([1e300], dtype=torch.float64, requires_grad=True)
//   >>> torch.remainder(a, b).sum().backward()
//   >>> b.grad  # tensor([1.], dtype=torch.float64)
#[test]
fn core181_rem_bgrad_underflow_signed_zero_f64() {
    let (_, gb) = rem_grads_f64(&[-1e-300], &[1e300]);
    assert_eq!(gb, vec![1.0f64], "b.grad vs torch [1.0]");
}

// Sign-quadrant pins (naive and exact agree here — no-drift guards):
//   >>> for a, b in [(7.,.7), (5.,3.), (-5.,3.), (5.,-3.), (-5.,-3.)]:
//   ...   torch.remainder(a_leaf, b_leaf).sum().backward()
//   b.grad: (7,.7) -> -10. ; (5,3) -> -1. ; (-5,3) -> 2. ;
//           (5,-3) -> 2.  ; (-5,-3) -> -1.   (a.grad = 1. each)
#[test]
fn core181_rem_bgrad_sign_quadrants_f32() {
    for (av, bv, expected_gb) in [
        (7.0f32, 0.7f32, -10.0f32),
        (5.0, 3.0, -1.0),
        (-5.0, 3.0, 2.0),
        (5.0, -3.0, 2.0),
        (-5.0, -3.0, -1.0),
    ] {
        let (ga, gb) = rem_grads_f32(&[av], &[bv]);
        assert_eq!(ga, vec![1.0f32], "a.grad for ({av}, {bv})");
        assert_eq!(gb, vec![expected_gb], "b.grad for ({av}, {bv}) vs torch");
    }
}

// Non-uniform cotangent through the divergent boundary — torch oracle:
//   >>> a = torch.tensor([-7.0, 5.0], requires_grad=True)
//   >>> b = torch.tensor([0.7, 3.0], requires_grad=True)
//   >>> torch.remainder(a, b).backward(torch.tensor([2.0, 3.0]))
//   >>> a.grad  # tensor([2., 3.])
//   >>> b.grad  # tensor([22., -3.])   (= [2*11, 3*(-1)])
// Pre-fix ferrotorch: b.grad[0] = 20.0 (2 * naive 10).
#[test]
fn core181_rem_bgrad_weighted_cotangent_f32() {
    let a = leaf_f32(&[-7.0, 5.0], &[2]);
    let b = leaf_f32(&[0.7, 3.0], &[2]);
    let y = remainder(&a, &b).expect("forward remainder");
    let w = t_f32(&[2.0, 3.0], &[2]);
    sum(&mul(&y, &w).expect("weight"))
        .expect("sum")
        .backward()
        .expect("backward");
    let ga = a.grad().unwrap().expect("a.grad present");
    let gb = b.grad().unwrap().expect("b.grad present");
    assert_eq!(ga.data().unwrap(), &[2.0f32, 3.0], "a.grad vs torch [2, 3]");
    assert_eq!(
        gb.data().unwrap(),
        &[22.0f32, -3.0],
        "b.grad vs torch [22, -3]"
    );
}

// Broadcasting reduction across the divergent boundary — torch oracle:
//   >>> a = torch.tensor([[-7.0, 5.0], [7.0, -5.0]], requires_grad=True)
//   >>> b = torch.tensor([0.7, 3.0], requires_grad=True)
//   >>> torch.remainder(a, b).sum().backward()
//   >>> a.grad  # tensor([[1., 1.], [1., 1.]])
//   >>> b.grad  # tensor([1., 1.])   (= [11 + (-10), 2 + (-1)])
// Pre-fix ferrotorch: b.grad[0] = 0.0 (naive: 10 + (-10)).
#[test]
fn core181_rem_bgrad_broadcast_f32() {
    let a = leaf_f32(&[-7.0, 5.0, 7.0, -5.0], &[2, 2]);
    let b = leaf_f32(&[0.7, 3.0], &[2]);
    let y = remainder(&a, &b).expect("forward remainder broadcast");
    sum(&y).expect("sum").backward().expect("backward");
    let ga = a.grad().unwrap().expect("a.grad present");
    let gb = b.grad().unwrap().expect("b.grad present");
    assert_eq!(ga.shape(), &[2, 2]);
    assert_eq!(gb.shape(), &[2]);
    assert_eq!(
        ga.data().unwrap(),
        &[1.0f32, 1.0, 1.0, 1.0],
        "a.grad vs torch ones"
    );
    assert_eq!(gb.data().unwrap(), &[1.0f32, 1.0], "b.grad vs torch [1, 1]");
}

// CUDA lane (gpu feature) — torch oracle on cuda:0:
//   >>> a = torch.tensor([-7.0], device='cuda', requires_grad=True)
//   >>> b = torch.tensor([0.7], device='cuda', requires_grad=True)
//   >>> torch.remainder(a, b).sum().backward()
//   >>> a.grad, b.grad  # (tensor([1.], device='cuda:0'), tensor([11.], device='cuda:0'))
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
    fn core181_gpu_rem_bgrad_floor_boundary_f32() {
        ensure_cuda_backend();
        let dev = Device::Cuda(0);
        let a = leaf_f32(&[-7.0], &[1]).to(dev).expect("upload a");
        let b = leaf_f32(&[0.7], &[1]).to(dev).expect("upload b");
        let y = remainder(&a, &b).expect("forward remainder cuda");
        sum(&y).expect("sum").backward().expect("backward cuda");
        let ga = a.grad().unwrap().expect("a.grad present");
        let gb = b.grad().unwrap().expect("b.grad present");
        // R-ORACLE-3: gradients must be CUDA-resident.
        assert_eq!(ga.device(), dev, "a.grad must be CUDA-resident");
        assert_eq!(gb.device(), dev, "b.grad must be CUDA-resident");
        assert_eq!(
            ga.cpu().expect("D2H").data().unwrap(),
            &[1.0f32],
            "a.grad vs torch [1.0]"
        );
        assert_eq!(
            gb.cpu().expect("D2H").data().unwrap(),
            &[11.0f32],
            "b.grad vs torch [11.0]"
        );
    }
}
