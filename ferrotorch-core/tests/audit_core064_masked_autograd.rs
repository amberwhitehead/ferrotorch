//! Regression tests for audit finding CORE-064 (#1758): every value-producing
//! `MaskedTensor` API constructed a detached result — `filled` / `to_tensor`
//! copied into fresh `requires_grad = false` tensors, and `masked_sum` /
//! `masked_mean` / `masked_min` / `masked_max` returned fresh detached CPU or
//! GPU tensors on every path. A masked loss therefore silently stopped
//! training the underlying data tensor.
//!
//! These tests assert gradient FLOW with torch-derived VALUES at the original
//! leaf (R-ORACLE-3) — never `requires_grad` flags alone — for sum, mean,
//! min/max (unique extremum AND tie split), filled/to_tensor, all-valid,
//! all-masked, untracked (R-LOUD-3), f32 + f64, CPU and CUDA.
//!
//! ## Oracle (R-ORACLE-1b) — live torch 2.11.0+cu130, probed 2026-06-11
//!
//! ```python
//! import torch
//! t = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
//! m = torch.tensor([True, False, True, True])
//! torch.masked.sum(t, mask=m).backward()
//! t.grad                                   # tensor([1., 0., 1., 1.])
//! (torch.masked.sum(t, mask=m) * 3.0).backward()   # fresh leaf
//! t.grad                                   # tensor([3., 0., 3., 3.])
//! t2 = torch.tensor([10.0, 0.0, 30.0, 0.0, 50.0], requires_grad=True)
//! m2 = torch.tensor([True, False, True, False, True])
//! (torch.masked.mean(t2, mask=m2) * 6.0).backward()
//! t2.grad                                  # tensor([2., 0., 2., 0., 2.])
//! t3 = torch.tensor([5.0, 1.0, 9.0, 2.0], requires_grad=True)
//! m3 = torch.tensor([True, False, False, True])
//! torch.masked.amin(t3, mask=m3).backward()
//! t3.grad                                  # tensor([0., 0., 0., 1.])
//! torch.masked.amax(t4, mask=m3).backward()        # t4 same data
//! t4.grad                                  # tensor([1., 0., 0., 0.])
//! t5 = torch.tensor([5.0, 5.0, 1.0, 5.0], requires_grad=True)
//! m5 = torch.tensor([True, True, True, False])
//! (torch.masked.amax(t5, mask=m5) * 4.0).backward()
//! t5.grad                                  # tensor([2., 2., 0., 0.])  ← tie split
//! t6 = torch.tensor([-2.0, 7.0, -2.0, -2.0], requires_grad=True)
//! torch.masked.amin(t6, mask=m5).backward()
//! t6.grad                                  # tensor([0.5000, 0.0000, 0.5000, 0.0000])
//! t7 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
//! out7 = t7.masked_fill(torch.tensor([False, True, False]), 7.0)
//! (out7 * torch.tensor([10.0, 20.0, 30.0])).sum().backward()
//! t7.grad                                  # tensor([10.,  0., 30.])  ← filled/to_tensor
//! # all-masked (mask all False): every op backward gives zero grads
//! t8 = torch.tensor([1.0, 2.0], requires_grad=True)
//! m8 = torch.tensor([False, False])
//! torch.masked.sum(t8, mask=m8).backward();  t8.grad   # tensor([0., 0.])
//! torch.masked.mean(...).backward()  → grad tensor([0., 0.])  (value nan)
//! torch.masked.amax(...).backward()  → grad tensor([0., 0.])  (value -inf)
//! ```
//!
//! All-masked extremum FORWARD value stays under the #1924 NaN pin (ferrotorch
//! NaN sentinel vs torch's ±inf identity payload); the BACKWARD contract is
//! NOT divergent — torch routes zero gradient to the data leaf and so do we.
//!
//! Tolerance: every gradient above is an exact small f32/f64 value (integers,
//! halves) — bit-exact compare.

use ferrotorch_core::grad_fns::{arithmetic::mul, reduction::sum};
use ferrotorch_core::masked::{
    MaskedTensor, masked_count, masked_max, masked_mean, masked_min, masked_sum,
};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

/// CPU leaf with given data, requires_grad=true.
fn grad_leaf_f32(data: &[f32]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], true).unwrap()
}

fn grad_leaf_f64(data: &[f64]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], true).unwrap()
}

/// Gradients here are exact in f32 (integers / halves) — see module doc.
#[allow(clippy::float_cmp)]
fn assert_grad_f32(leaf: &Tensor<f32>, expected: &[f32], label: &str) {
    let g = leaf
        .grad()
        .unwrap()
        .unwrap_or_else(|| panic!("{label}: no gradient reached the leaf (CORE-064 detach)"));
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

#[allow(clippy::float_cmp)]
fn assert_grad_f64(leaf: &Tensor<f64>, expected: &[f64], label: &str) {
    let g = leaf
        .grad()
        .unwrap()
        .unwrap_or_else(|| panic!("{label}: no gradient reached the leaf (CORE-064 detach)"));
    assert_eq!(g.device(), leaf.device(), "{label}: gradient device");
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

/// 0-d scalar weight tensor on the same device as `y`, then `(y * w).backward()`.
fn scaled_backward_f32(y: &Tensor<f32>, w: f32) {
    let wt = Tensor::from_storage(TensorStorage::cpu(vec![w]), vec![], false).unwrap();
    let wt = if y.is_cpu() {
        wt
    } else {
        wt.to(y.device()).unwrap()
    };
    let loss = mul(y, &wt).expect("y * w");
    loss.backward().expect("backward");
}

fn scaled_backward_f64(y: &Tensor<f64>, w: f64) {
    let wt = Tensor::from_storage(TensorStorage::cpu(vec![w]), vec![], false).unwrap();
    let wt = if y.is_cpu() {
        wt
    } else {
        wt.to(y.device()).unwrap()
    };
    let loss = mul(y, &wt).expect("y * w");
    loss.backward().expect("backward");
}

// ---------------------------------------------------------------------------
// masked_sum
// ---------------------------------------------------------------------------

/// Oracle: `torch.masked.sum` grad = `[1., 0., 1., 1.]`.
#[test]
fn masked_sum_backward_reaches_leaf() {
    let x = grad_leaf_f32(&[1.0, 2.0, 3.0, 4.0]);
    let mt = MaskedTensor::new(x.clone(), vec![true, false, true, true]).unwrap();
    let y = masked_sum(&mt).unwrap();
    assert!(
        y.requires_grad(),
        "masked_sum output must carry the graph (pre-fix: silent detach)"
    );
    y.backward().expect("backward");
    assert_grad_f32(&x, &[1.0, 0.0, 1.0, 1.0], "masked_sum");
}

/// Oracle: `(torch.masked.sum(t, mask=m) * 3.0)` grad = `[3., 0., 3., 3.]`.
#[test]
fn masked_sum_weighted_backward() {
    let x = grad_leaf_f32(&[1.0, 2.0, 3.0, 4.0]);
    let mt = MaskedTensor::new(x.clone(), vec![true, false, true, true]).unwrap();
    let y = masked_sum(&mt).unwrap();
    scaled_backward_f32(&y, 3.0);
    assert_grad_f32(&x, &[3.0, 0.0, 3.0, 3.0], "masked_sum * 3");
}

/// All-valid mask: grad = 1 everywhere (degenerates to plain sum).
#[test]
fn masked_sum_all_valid_backward() {
    let x = grad_leaf_f32(&[1.0, 2.0, 3.0]);
    let mt = MaskedTensor::new(x.clone(), vec![true, true, true]).unwrap();
    masked_sum(&mt).unwrap().backward().expect("backward");
    assert_grad_f32(&x, &[1.0, 1.0, 1.0], "masked_sum all-valid");
}

// ---------------------------------------------------------------------------
// masked_mean
// ---------------------------------------------------------------------------

/// Oracle: `(torch.masked.mean(t2, mask=m2) * 6.0)` grad = `[2., 0., 2., 0., 2.]`
/// (3 valid entries → 6/3 = 2 at each valid position).
#[test]
fn masked_mean_weighted_backward() {
    let x = grad_leaf_f32(&[10.0, 0.0, 30.0, 0.0, 50.0]);
    let mt = MaskedTensor::new(x.clone(), vec![true, false, true, false, true]).unwrap();
    let y = masked_mean(&mt).unwrap();
    assert!(y.requires_grad(), "masked_mean output must carry the graph");
    scaled_backward_f32(&y, 6.0);
    assert_grad_f32(&x, &[2.0, 0.0, 2.0, 0.0, 2.0], "masked_mean * 6");
}

/// f64 lane, same oracle values.
#[test]
fn masked_mean_f64_weighted_backward() {
    let x = grad_leaf_f64(&[10.0, 0.0, 30.0, 0.0, 50.0]);
    let mt = MaskedTensor::new(x.clone(), vec![true, false, true, false, true]).unwrap();
    let y = masked_mean(&mt).unwrap();
    scaled_backward_f64(&y, 6.0);
    assert_grad_f64(&x, &[2.0, 0.0, 2.0, 0.0, 2.0], "masked_mean f64 * 6");
}

// ---------------------------------------------------------------------------
// masked_min / masked_max — unique extremum and tie split
// ---------------------------------------------------------------------------

/// Oracle: `torch.masked.amin` grad = `[0., 0., 0., 1.]` (unique min 2.0 at idx 3).
#[test]
fn masked_min_unique_backward() {
    let x = grad_leaf_f32(&[5.0, 1.0, 9.0, 2.0]);
    let mt = MaskedTensor::new(x.clone(), vec![true, false, false, true]).unwrap();
    let y = masked_min(&mt).unwrap();
    assert!(y.requires_grad(), "masked_min output must carry the graph");
    y.backward().expect("backward");
    assert_grad_f32(&x, &[0.0, 0.0, 0.0, 1.0], "masked_min unique");
}

/// Oracle: `torch.masked.amax` grad = `[1., 0., 0., 0.]` (unique max 5.0 at idx 0;
/// the larger 9.0 is masked out and must receive NO gradient).
#[test]
fn masked_max_unique_backward() {
    let x = grad_leaf_f32(&[5.0, 1.0, 9.0, 2.0]);
    let mt = MaskedTensor::new(x.clone(), vec![true, false, false, true]).unwrap();
    let y = masked_max(&mt).unwrap();
    y.backward().expect("backward");
    assert_grad_f32(&x, &[1.0, 0.0, 0.0, 0.0], "masked_max unique");
}

/// TIE CONTRACT (probed live, module doc): gradient splits EVENLY among the
/// VALID positions equal to the extremum. Oracle: `[2., 2., 0., 0.]` with
/// upstream grad 4 and two valid maxima (the third 5.0 is masked → 0).
#[test]
fn masked_max_tie_split_backward() {
    let x = grad_leaf_f32(&[5.0, 5.0, 1.0, 5.0]);
    let mt = MaskedTensor::new(x.clone(), vec![true, true, true, false]).unwrap();
    let y = masked_max(&mt).unwrap();
    scaled_backward_f32(&y, 4.0);
    assert_grad_f32(&x, &[2.0, 2.0, 0.0, 0.0], "masked_max tie * 4");
}

/// Oracle: `torch.masked.amin` tie grad = `[0.5, 0., 0.5, 0.]` (two valid
/// minima of three equal values; the third -2.0 is masked → 0).
#[test]
fn masked_min_tie_split_backward() {
    let x = grad_leaf_f32(&[-2.0, 7.0, -2.0, -2.0]);
    let mt = MaskedTensor::new(x.clone(), vec![true, true, true, false]).unwrap();
    let y = masked_min(&mt).unwrap();
    y.backward().expect("backward");
    assert_grad_f32(&x, &[0.5, 0.0, 0.5, 0.0], "masked_min tie");
}

/// f64 tie lane.
#[test]
fn masked_max_tie_f64_backward() {
    let x = grad_leaf_f64(&[5.0, 5.0, 1.0, 5.0]);
    let mt = MaskedTensor::new(x.clone(), vec![true, true, true, false]).unwrap();
    let y = masked_max(&mt).unwrap();
    scaled_backward_f64(&y, 4.0);
    assert_grad_f64(&x, &[2.0, 2.0, 0.0, 0.0], "masked_max f64 tie * 4");
}

// ---------------------------------------------------------------------------
// filled / to_tensor
// ---------------------------------------------------------------------------

/// Oracle: `t.masked_fill(~mask, fill)` weighted grad = `[10., 0., 30.]` —
/// gradient passes through at valid positions, zero where the constant fill
/// replaced the value.
#[test]
fn filled_weighted_backward() {
    let x = grad_leaf_f32(&[1.0, 2.0, 3.0]);
    let mt = MaskedTensor::new(x.clone(), vec![true, false, true]).unwrap();
    let y = mt.filled().unwrap();
    assert!(y.requires_grad(), "filled output must carry the graph");
    let w = Tensor::from_storage(
        TensorStorage::cpu(vec![10.0_f32, 20.0, 30.0]),
        vec![3],
        false,
    )
    .unwrap();
    let prod = mul(&y, &w).expect("y * w");
    sum(&prod).expect("sum").backward().expect("backward");
    assert_grad_f32(&x, &[10.0, 0.0, 30.0], "filled weighted");
}

/// `to_tensor` is the same op (alias); a non-zero fill_value is a CONSTANT and
/// must not change the gradient. Oracle: same `[10., 0., 30.]`.
#[test]
fn to_tensor_with_fill_value_backward() {
    let x = grad_leaf_f32(&[1.0, 2.0, 3.0]);
    let mt = MaskedTensor::new(x.clone(), vec![true, false, true])
        .unwrap()
        .with_fill_value(-99.0);
    let y = mt.to_tensor().unwrap();
    assert!(y.requires_grad(), "to_tensor output must carry the graph");
    let w = Tensor::from_storage(
        TensorStorage::cpu(vec![10.0_f32, 20.0, 30.0]),
        vec![3],
        false,
    )
    .unwrap();
    let prod = mul(&y, &w).expect("y * w");
    sum(&prod).expect("sum").backward().expect("backward");
    assert_grad_f32(&x, &[10.0, 0.0, 30.0], "to_tensor fill=-99 weighted");
}

// ---------------------------------------------------------------------------
// All-masked: zero gradients reach the leaf (torch-verified)
// ---------------------------------------------------------------------------

/// Oracle (quoted in module doc): all-masked sum/mean/amax backward all give
/// `tensor([0., 0.])`. The extremum FORWARD value remains the #1924-pinned
/// NaN sentinel; the backward contract (zero grads) matches torch exactly.
#[test]
fn all_masked_backward_routes_zero_grads() {
    type MaskedRed = fn(&MaskedTensor<f32>) -> ferrotorch_core::FerrotorchResult<Tensor<f32>>;
    for (name, op) in [
        ("masked_sum", masked_sum as MaskedRed),
        ("masked_mean", masked_mean as MaskedRed),
        ("masked_min", masked_min as MaskedRed),
        ("masked_max", masked_max as MaskedRed),
    ] {
        let x = grad_leaf_f32(&[1.0, 2.0]);
        let mt = MaskedTensor::new(x.clone(), vec![false, false]).unwrap();
        let y = op(&mt).unwrap();
        assert!(
            y.requires_grad(),
            "{name}: all-masked output must still carry the graph (torch does)"
        );
        y.backward().expect("backward");
        assert_grad_f32(&x, &[0.0, 0.0], &format!("{name} all-masked"));
    }
}

// ---------------------------------------------------------------------------
// R-LOUD-3: untracked inputs stay honestly untracked
// ---------------------------------------------------------------------------

#[test]
fn untracked_inputs_stay_untracked() {
    let x =
        Tensor::from_storage(TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0]), vec![3], false).unwrap();
    let mt = MaskedTensor::new(x, vec![true, false, true]).unwrap();
    assert!(!masked_sum(&mt).unwrap().requires_grad());
    assert!(!masked_mean(&mt).unwrap().requires_grad());
    assert!(!masked_min(&mt).unwrap().requires_grad());
    assert!(!masked_max(&mt).unwrap().requires_grad());
    assert!(!masked_count(&mt).unwrap().requires_grad());
    assert!(!mt.filled().unwrap().requires_grad());
    assert!(!mt.to_tensor().unwrap().requires_grad());
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
                .expect("CUDA backend must initialize for the gpu masked autograd tests");
        });
    }

    /// CUDA leaf (uploaded, re-marked as leaf — the torch
    /// `x.to('cuda').detach().requires_grad_(True)` idiom).
    fn cuda_grad_leaf_f32(data: &[f32]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap()
            .requires_grad_(true)
    }

    fn cuda_grad_leaf_f64(data: &[f64]) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
            .unwrap()
            .to(Device::Cuda(0))
            .unwrap()
            .requires_grad_(true)
    }

    /// Oracle values identical to the CPU lanes (same quoted torch session).
    #[test]
    fn gpu_masked_sum_backward() {
        ensure_cuda_backend();
        let x = cuda_grad_leaf_f32(&[1.0, 2.0, 3.0, 4.0]);
        let mt = MaskedTensor::new(x.clone(), vec![true, false, true, true]).unwrap();
        let y = masked_sum(&mt).unwrap();
        assert!(y.requires_grad(), "gpu masked_sum must carry the graph");
        y.backward().expect("backward");
        assert_grad_f32(&x, &[1.0, 0.0, 1.0, 1.0], "gpu masked_sum");
    }

    #[test]
    fn gpu_masked_mean_weighted_backward() {
        ensure_cuda_backend();
        let x = cuda_grad_leaf_f32(&[10.0, 0.0, 30.0, 0.0, 50.0]);
        let mt = MaskedTensor::new(x.clone(), vec![true, false, true, false, true]).unwrap();
        let y = masked_mean(&mt).unwrap();
        scaled_backward_f32(&y, 6.0);
        assert_grad_f32(&x, &[2.0, 0.0, 2.0, 0.0, 2.0], "gpu masked_mean * 6");
    }

    #[test]
    fn gpu_masked_mean_f64_weighted_backward() {
        ensure_cuda_backend();
        let x = cuda_grad_leaf_f64(&[10.0, 0.0, 30.0, 0.0, 50.0]);
        let mt = MaskedTensor::new(x.clone(), vec![true, false, true, false, true]).unwrap();
        let y = masked_mean(&mt).unwrap();
        scaled_backward_f64(&y, 6.0);
        assert_grad_f64(&x, &[2.0, 0.0, 2.0, 0.0, 2.0], "gpu masked_mean f64 * 6");
    }

    /// Tie split on CUDA: `[2., 2., 0., 0.]` (same oracle as CPU lane).
    #[test]
    fn gpu_masked_max_tie_backward() {
        ensure_cuda_backend();
        let x = cuda_grad_leaf_f32(&[5.0, 5.0, 1.0, 5.0]);
        let mt = MaskedTensor::new(x.clone(), vec![true, true, true, false]).unwrap();
        let y = masked_max(&mt).unwrap();
        scaled_backward_f32(&y, 4.0);
        assert_grad_f32(&x, &[2.0, 2.0, 0.0, 0.0], "gpu masked_max tie * 4");
    }

    #[test]
    fn gpu_masked_min_unique_backward() {
        ensure_cuda_backend();
        let x = cuda_grad_leaf_f32(&[5.0, 1.0, 9.0, 2.0]);
        let mt = MaskedTensor::new(x.clone(), vec![true, false, false, true]).unwrap();
        let y = masked_min(&mt).unwrap();
        y.backward().expect("backward");
        assert_grad_f32(&x, &[0.0, 0.0, 0.0, 1.0], "gpu masked_min unique");
    }

    #[test]
    fn gpu_filled_weighted_backward() {
        ensure_cuda_backend();
        let x = cuda_grad_leaf_f32(&[1.0, 2.0, 3.0]);
        let mt = MaskedTensor::new(x.clone(), vec![true, false, true]).unwrap();
        let y = mt.filled().unwrap();
        assert!(y.requires_grad(), "gpu filled must carry the graph");
        let w = Tensor::from_storage(
            TensorStorage::cpu(vec![10.0_f32, 20.0, 30.0]),
            vec![3],
            false,
        )
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
        let prod = mul(&y, &w).expect("y * w");
        sum(&prod).expect("sum").backward().expect("backward");
        assert_grad_f32(&x, &[10.0, 0.0, 30.0], "gpu filled weighted");
    }

    /// All-masked CUDA extrema: forward stays the #1924-pinned NaN; backward
    /// routes zero grads to the CUDA leaf (torch: `tensor([0., 0.])`).
    #[test]
    fn gpu_all_masked_extremum_backward_zero_grads() {
        ensure_cuda_backend();
        type MaskedRed = fn(&MaskedTensor<f32>) -> ferrotorch_core::FerrotorchResult<Tensor<f32>>;
        for (name, op) in [
            ("masked_min", masked_min as MaskedRed),
            ("masked_max", masked_max as MaskedRed),
        ] {
            let x = cuda_grad_leaf_f32(&[1.0, 2.0]);
            let mt = MaskedTensor::new(x.clone(), vec![false, false]).unwrap();
            let y = op(&mt).unwrap();
            assert!(y.requires_grad(), "{name}: all-masked must carry graph");
            y.backward().expect("backward");
            assert_grad_f32(&x, &[0.0, 0.0], &format!("gpu {name} all-masked"));
        }
    }
}
