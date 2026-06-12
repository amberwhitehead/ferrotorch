//! Regression tests for audit finding CORE-061 (#1755): every public einops
//! transform (`rearrange`, `rearrange_with`, `repeat`, `reduce`) silently
//! severed the autograd graph on every path (identity view, general permute,
//! repeat, and both reduce paths — the leading bare `view_reshape` detached
//! before the differentiable reductions ever saw the tensor).
//!
//! These tests assert gradient FLOW with torch-derived VALUES at the original
//! leaf (R-ORACLE-3) — never `requires_grad` flags alone — for every pattern
//! class: identity, reorder, split, merge, split+reorder+merge, repeat (new
//! axis / reordered / merged), and each `EinopsReduction` discriminator, on
//! contiguous AND non-contiguous inputs (the former fallback selector), CPU
//! and CUDA.
//!
//! ## Oracle (R-ORACLE-1b)
//!
//! Every gradient expectation traces to a live torch 2.11.0+cu130 +
//! einops 0.8.2 session:
//!
//! ```python
//! import torch
//! from einops import rearrange, repeat, reduce
//! x = torch.arange(n, dtype=torch.float32).reshape(shape).requires_grad_(True)
//! y = <einops op>(x)
//! w = (torch.arange(y.numel(), dtype=torch.float32) + 1.0).reshape(y.shape)
//! (y * w).sum().backward()
//! x.grad.flatten().tolist()
//! ```
//!
//! The weighted loss makes the upstream gradient DISTINCT per output element,
//! so a coordinate-mapping error in any backward shows up as wrong values,
//! not just missing flow. The per-case oracle output is quoted on each test.
//!
//! Tolerance: all grads here are exact small integers in f32 (bit-exact)
//! except `reduce mean` whose grads are multiples of 1/3 — those use 1e-6 abs
//! (f32 eps 1.19e-7 relative at magnitude <= 2.67, one multiply + one divide;
//! analytic bound << 1e-6).

use ferrotorch_core::einops::{EinopsReduction, rearrange, rearrange_with, reduce, repeat};
use ferrotorch_core::grad_fns::{arithmetic::mul, reduction::sum};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

/// CPU leaf with an integer ramp 0..n, requires_grad=true.
fn grad_leaf(shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), true).unwrap()
}

/// `loss = (y * w).sum()` with `w = 1..=numel` reshaped to `y.shape`, then
/// backward. Mirrors the oracle snippet in the module doc.
fn weighted_backward(y: &Tensor<f32>) {
    let n = y.numel();
    let w_data: Vec<f32> = (1..=n).map(|i| i as f32).collect();
    let w = Tensor::from_storage(TensorStorage::cpu(w_data), y.shape().to_vec(), false).unwrap();
    let w = if y.is_cpu() {
        w
    } else {
        w.to(y.device()).unwrap()
    };
    let prod = mul(y, &w).expect("y * w");
    let loss = sum(&prod).expect("sum to scalar");
    loss.backward().expect("backward");
}

/// Gradients here are exact in f32 (integer values) — see module doc.
#[allow(clippy::float_cmp)]
fn assert_grad(leaf: &Tensor<f32>, expected: &[f32], label: &str) {
    let g = leaf
        .grad()
        .unwrap()
        .unwrap_or_else(|| panic!("{label}: no gradient reached the leaf (CORE-061 detach)"));
    assert_eq!(
        g.device(),
        leaf.device(),
        "{label}: gradient device must match leaf device"
    );
    let g_cpu = if g.is_cpu() {
        g
    } else {
        g.cpu().expect("grad D2H")
    };
    assert_eq!(
        g_cpu.data().unwrap(),
        expected,
        "{label}: leaf gradient vs torch oracle"
    );
}

/// Tolerant variant for the mean case (non-representable 1/3 multiples).
fn assert_grad_close(leaf: &Tensor<f32>, expected: &[f32], tol: f32, label: &str) {
    let g = leaf
        .grad()
        .unwrap()
        .unwrap_or_else(|| panic!("{label}: no gradient reached the leaf (CORE-061 detach)"));
    assert_eq!(g.device(), leaf.device(), "{label}: gradient device");
    let g_cpu = if g.is_cpu() {
        g
    } else {
        g.cpu().expect("grad D2H")
    };
    let actual = g_cpu.data().unwrap();
    assert_eq!(actual.len(), expected.len(), "{label}: grad numel");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{label}: grad[{i}] = {a} vs oracle {e} (tol {tol})"
        );
    }
}

// ---------------------------------------------------------------------------
// rearrange — identity / reorder / split / merge / combined
// ---------------------------------------------------------------------------

/// Oracle grad: `[1, 2, 3, 4, 5, 6]` (identity passes w straight through).
#[test]
fn rearrange_identity_backward() {
    let x = grad_leaf(&[2, 3]);
    let y = rearrange(&x, "a b -> a b").unwrap();
    assert!(y.requires_grad(), "identity rearrange output must track");
    weighted_backward(&y);
    assert_grad(&x, &[1., 2., 3., 4., 5., 6.], "rearrange 'a b -> a b'");
}

/// Oracle grad: `[1, 3, 5, 2, 4, 6]`.
#[test]
fn rearrange_reorder_backward() {
    let x = grad_leaf(&[2, 3]);
    let y = rearrange(&x, "a b -> b a").unwrap();
    assert!(y.requires_grad(), "reorder rearrange output must track");
    weighted_backward(&y);
    assert_grad(&x, &[1., 3., 5., 2., 4., 6.], "rearrange 'a b -> b a'");
}

/// Oracle grad: `[1..12]` (split is a pure metadata change).
#[test]
fn rearrange_split_backward() {
    let x = grad_leaf(&[4, 3]);
    let y = rearrange_with(&x, "(a b) c -> a b c", &[("a", 2)]).unwrap();
    assert!(y.requires_grad(), "split rearrange output must track");
    weighted_backward(&y);
    assert_grad(
        &x,
        &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        "rearrange '(a b) c -> a b c'",
    );
}

/// Oracle grad: `[1..12]` (merge is a pure metadata change).
#[test]
fn rearrange_merge_backward() {
    let x = grad_leaf(&[2, 2, 3]);
    let y = rearrange(&x, "a b c -> a (b c)").unwrap();
    weighted_backward(&y);
    assert_grad(
        &x,
        &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        "rearrange 'a b c -> a (b c)'",
    );
}

/// Split + reorder + merge in one pattern.
/// Oracle grad: `[1, 5, 9, 3, 7, 11, 2, 6, 10, 4, 8, 12]`.
#[test]
fn rearrange_split_reorder_merge_backward() {
    let x = grad_leaf(&[4, 3]);
    let y = rearrange_with(&x, "(a b) c -> c (b a)", &[("a", 2)]).unwrap();
    weighted_backward(&y);
    assert_grad(
        &x,
        &[1., 5., 9., 3., 7., 11., 2., 6., 10., 4., 8., 12.],
        "rearrange '(a b) c -> c (b a)'",
    );
}

/// Non-contiguous input (transposed view) — the former legacy-fallback
/// selector. The composition x.transpose -> rearrange 'a b -> b a' is the
/// identity on x, so the oracle grad is w itself: `[1, 2, 3, 4, 5, 6]`.
#[test]
fn rearrange_noncontiguous_input_backward() {
    let x = grad_leaf(&[2, 3]);
    let xt = x.transpose(0, 1).unwrap(); // [3, 2], non-contiguous view
    assert!(!xt.is_contiguous(), "precondition: non-contiguous input");
    let y = rearrange(&xt, "a b -> b a").unwrap();
    weighted_backward(&y);
    assert_grad(
        &x,
        &[1., 2., 3., 4., 5., 6.],
        "rearrange non-contiguous 'a b -> b a'",
    );
}

// ---------------------------------------------------------------------------
// repeat — new axis / reorder+new / merged
// ---------------------------------------------------------------------------

/// Oracle grad: `[3, 7, 11, 15, 19, 23]` (each input cell sums its 2 copies).
#[test]
fn repeat_new_axis_backward() {
    let x = grad_leaf(&[2, 3]);
    let y = repeat(&x, "a b -> a b c", &[("c", 2)]).unwrap();
    assert!(y.requires_grad(), "repeat output must track");
    weighted_backward(&y);
    assert_grad(&x, &[3., 7., 11., 15., 19., 23.], "repeat 'a b -> a b c'");
}

/// Reorder + new axis (the CORE-062 pattern class, now with grads).
/// Oracle grad: `[3, 11, 19, 7, 15, 23]`.
#[test]
fn repeat_reorder_new_axis_backward() {
    let x = grad_leaf(&[2, 3]);
    let y = repeat(&x, "a b -> b a c", &[("c", 2)]).unwrap();
    weighted_backward(&y);
    assert_grad(&x, &[3., 11., 19., 7., 15., 23.], "repeat 'a b -> b a c'");
}

/// New axis merged with a reordered kept axis.
/// Oracle grad: `[9, 27, 45, 12, 30, 48]`.
#[test]
fn repeat_merged_new_axis_backward() {
    let x = grad_leaf(&[2, 3]);
    let y = repeat(&x, "a b -> (b c) a", &[("c", 3)]).unwrap();
    weighted_backward(&y);
    assert_grad(
        &x,
        &[9., 27., 45., 12., 30., 48.],
        "repeat 'a b -> (b c) a'",
    );
}

// ---------------------------------------------------------------------------
// reduce — Sum / Mean / Max / Min, fast-path and reordered-kept shapes
// ---------------------------------------------------------------------------

/// Axis-aligned fast-path shape.
/// Oracle grad: `[1.0 x12, 2.0 x12]`.
#[test]
fn reduce_sum_fast_path_backward() {
    let x = grad_leaf(&[2, 3, 4]);
    let y = reduce(&x, "a b c -> a", EinopsReduction::Sum).unwrap();
    assert!(y.requires_grad(), "reduce output must track");
    weighted_backward(&y);
    let mut expected = vec![1.0f32; 12];
    expected.extend(std::iter::repeat_n(2.0f32, 12));
    assert_grad(&x, &expected, "reduce sum 'a b c -> a'");
}

/// Reordered kept axes (the former always-detached fallback).
/// Oracle grad: `[1,3,5,7] x3 then [2,4,6,8] x3`.
#[test]
fn reduce_sum_reorder_backward() {
    let x = grad_leaf(&[2, 3, 4]);
    let y = reduce(&x, "a b c -> c a", EinopsReduction::Sum).unwrap();
    weighted_backward(&y);
    let expected = [
        1., 3., 5., 7., 1., 3., 5., 7., 1., 3., 5., 7., 2., 4., 6., 8., 2., 4., 6., 8., 2., 4., 6.,
        8.,
    ];
    assert_grad(&x, &expected, "reduce sum 'a b c -> c a'");
}

/// Oracle grad: multiples of 1/3 (see module doc for the tolerance bound):
/// `[1/3, 1, 5/3, 7/3] x3 then [2/3, 4/3, 2, 8/3] x3`.
#[test]
fn reduce_mean_reorder_backward() {
    let x = grad_leaf(&[2, 3, 4]);
    let y = reduce(&x, "a b c -> c a", EinopsReduction::Mean).unwrap();
    weighted_backward(&y);
    let third = 1.0f32 / 3.0;
    let row_a = [third, 3. * third, 5. * third, 7. * third];
    let row_b = [2. * third, 4. * third, 6. * third, 8. * third];
    let mut expected = Vec::new();
    for _ in 0..3 {
        expected.extend_from_slice(&row_a);
    }
    for _ in 0..3 {
        expected.extend_from_slice(&row_b);
    }
    assert_grad_close(&x, &expected, 1e-6, "reduce mean 'a b c -> c a'");
}

/// Tie-free max: gradient lands on the (unique) argmax cells only.
/// Oracle grad: `[0 x8, 1, 3, 5, 7, 0 x8, 2, 4, 6, 8]`.
#[test]
fn reduce_max_reorder_backward() {
    let x = grad_leaf(&[2, 3, 4]);
    let y = reduce(&x, "a b c -> c a", EinopsReduction::Max).unwrap();
    weighted_backward(&y);
    let mut expected = vec![0.0f32; 8];
    expected.extend_from_slice(&[1., 3., 5., 7.]);
    expected.extend(std::iter::repeat_n(0.0f32, 8));
    expected.extend_from_slice(&[2., 4., 6., 8.]);
    assert_grad(&x, &expected, "reduce max 'a b c -> c a'");
}

/// Tie-free min: gradient lands on the (unique) argmin cells only.
/// Oracle grad: `[1, 3, 5, 7, 0 x8, 2, 4, 6, 8, 0 x8]`.
#[test]
fn reduce_min_reorder_backward() {
    let x = grad_leaf(&[2, 3, 4]);
    let y = reduce(&x, "a b c -> c a", EinopsReduction::Min).unwrap();
    weighted_backward(&y);
    let mut expected = vec![1.0f32, 3., 5., 7.];
    expected.extend(std::iter::repeat_n(0.0f32, 8));
    expected.extend_from_slice(&[2., 4., 6., 8.]);
    expected.extend(std::iter::repeat_n(0.0f32, 8));
    assert_grad(&x, &expected, "reduce min 'a b c -> c a'");
}

/// R-LOUD-3: untracked inputs stay honestly untracked on every path.
#[test]
fn untracked_inputs_stay_untracked() {
    let n: usize = 24;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let x = Tensor::from_storage(TensorStorage::cpu(data), vec![2, 3, 4], false).unwrap();
    assert!(!rearrange(&x, "a b c -> c b a").unwrap().requires_grad());
    assert!(
        !repeat(&x, "a b c -> c a b n", &[("n", 2)])
            .unwrap()
            .requires_grad()
    );
    assert!(
        !reduce(&x, "a b c -> c a", EinopsReduction::Sum)
            .unwrap()
            .requires_grad()
    );
}

// ---------------------------------------------------------------------------
// GPU lanes — gradient flow with device-checked grads (R-ORACLE-3 / CORE-196)
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use ferrotorch_core::device::Device;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for the gpu einops autograd tests");
        });
    }

    /// CUDA leaf (uploaded, re-marked as leaf — the torch
    /// `x.to('cuda').detach().requires_grad_(True)` idiom).
    fn cuda_grad_leaf(shape: &[usize]) -> Tensor<f32> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false)
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap()
            .requires_grad_(true)
    }

    /// Oracle values identical to the CPU lanes (same quoted torch session).
    #[test]
    fn gpu_rearrange_reorder_backward() {
        ensure_cuda_backend();
        let x = cuda_grad_leaf(&[2, 3]);
        let y = rearrange(&x, "a b -> b a").unwrap();
        assert_eq!(y.device(), Device::Cuda(0), "forward stays on device");
        weighted_backward(&y);
        assert_grad(&x, &[1., 3., 5., 2., 4., 6.], "gpu rearrange 'a b -> b a'");
    }

    #[test]
    fn gpu_repeat_reorder_new_axis_backward() {
        ensure_cuda_backend();
        let x = cuda_grad_leaf(&[2, 3]);
        let y = repeat(&x, "a b -> b a c", &[("c", 2)]).unwrap();
        assert_eq!(y.device(), Device::Cuda(0), "forward stays on device");
        weighted_backward(&y);
        assert_grad(
            &x,
            &[3., 11., 19., 7., 15., 23.],
            "gpu repeat 'a b -> b a c'",
        );
    }

    #[test]
    fn gpu_reduce_sum_reorder_backward() {
        ensure_cuda_backend();
        let x = cuda_grad_leaf(&[2, 3, 4]);
        let y = reduce(&x, "a b c -> c a", EinopsReduction::Sum).unwrap();
        assert_eq!(y.device(), Device::Cuda(0), "forward stays on device");
        weighted_backward(&y);
        let expected = [
            1., 3., 5., 7., 1., 3., 5., 7., 1., 3., 5., 7., 2., 4., 6., 8., 2., 4., 6., 8., 2., 4.,
            6., 8.,
        ];
        assert_grad(&x, &expected, "gpu reduce sum 'a b c -> c a'");
    }

    /// CUDA Max/Min backward is a LOUD structured error, never a silent
    /// detach: the forward carries a real graph (requires_grad=true), and
    /// backward surfaces `CummaxBackward`'s `NotImplementedOnCuda` (tracked
    /// follow-up: #1962; tie-semantics follow-up: #1963). When #1962 lands
    /// this test should be updated to assert torch-oracle gradient values
    /// instead.
    #[test]
    fn gpu_reduce_max_backward_is_loud_error_not_silent_detach() {
        ensure_cuda_backend();
        let x = cuda_grad_leaf(&[2, 3, 4]);
        let y = reduce(&x, "a b c -> c a", EinopsReduction::Max).unwrap();
        assert_eq!(y.device(), Device::Cuda(0), "forward stays on device");
        assert!(
            y.requires_grad(),
            "CUDA reduce-max forward must carry the graph (pre-fix: silent detach)"
        );
        let n = y.numel();
        let w_data: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        let w = Tensor::from_storage(TensorStorage::cpu(w_data), y.shape().to_vec(), false)
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap();
        let prod = mul(&y, &w).expect("y * w");
        let loss = sum(&prod).expect("sum to scalar");
        let err = loss
            .backward()
            .expect_err("CUDA reduce-max backward must be a structured error, not silence");
        let msg = format!("{err}");
        assert!(
            msg.contains("CummaxBackward") || msg.contains("not implemented on CUDA"),
            "expected NotImplementedOnCuda(CummaxBackward), got: {msg}"
        );
    }
}
