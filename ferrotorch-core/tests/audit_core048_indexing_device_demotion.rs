//! CORE-048 (#1742, CLASS-S High) regression battery: the advanced-indexing
//! family (`scatter_reduce`, `index_add`, `index_copy`, `take`, `put`,
//! `masked_scatter`) must
//!
//!   1. preserve device residency — CUDA operands in, CUDA output out
//!      (resident kernel/composite path, R-LOUD-2);
//!   2. reject mixed-device operand sets with a structured
//!      `FerrotorchError::DeviceMismatch` (R-LOUD-1) — torch contract:
//!      "Expected all tensors to be on the same device, but got index is on
//!      cpu, different from other tensors on cuda:0" for every op in the
//!      family (live torch 2.11.0+cu130 probe, pasted in #1742);
//!   3. deliver gradients on the leaf tensors' devices (R-ORACLE-3).
//!
//! Pre-fix observed behavior (R-AHON-1 probe at HEAD, pasted in #1742):
//! every op read CUDA operands through `data_vec`, computed on host, and
//! returned `device=Cpu` outputs (silent demotion); binary forms accepted
//! CUDA+CPU operand mixes and silently combined the downloaded values;
//! all-CUDA index tensors errored `GpuTensorNotAccessible` (torch succeeds);
//! gradients of CUDA leaves landed on CPU.
//!
//! All numerical expectations are pasted from a LIVE `torch==2.11.0+cu130`
//! session on the same device class (RTX 3090) — snippets quoted per test
//! (R-ORACLE-1(b)). Comparisons are exact (`==`): every expected value is a
//! small integer or exact dyadic rational (0.5) reachable without rounding
//! in both f32 and f64, and the ops are pure data movement plus at most one
//! exact product/sum per element.

#![cfg(feature = "gpu")]

use ferrotorch_core::autograd::graph::backward_with_grad;
use ferrotorch_core::grad_fns::indexing::{
    ScatterReduce, index_add, index_copy, masked_scatter, put, scatter_reduce, take,
};
use ferrotorch_core::{
    BoolTensor, Device, FerrotorchError, Float, IntTensor, Tensor, TensorStorage,
};
use half::{bf16, f16};
use std::sync::Once;

static GPU_INIT: Once = Once::new();
fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the CORE-048 GPU pins");
    });
}

/// Build a CUDA-resident float tensor from f64 literals (exactly
/// representable in both dtypes for every value used in this suite).
fn t_cuda<T: Float>(data: &[f64], shape: &[usize], rg: bool) -> Tensor<T> {
    let cast: Vec<T> = data
        .iter()
        .map(|&v| <T as num_traits::NumCast>::from(v).unwrap())
        .collect();
    Tensor::from_storage(TensorStorage::cpu(cast), shape.to_vec(), false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap()
        .requires_grad_(rg)
}

/// Build a host (CPU) float tensor from f64 literals.
fn t_cpu<T: Float>(data: &[f64], shape: &[usize], rg: bool) -> Tensor<T> {
    let cast: Vec<T> = data
        .iter()
        .map(|&v| <T as num_traits::NumCast>::from(v).unwrap())
        .collect();
    Tensor::from_storage(TensorStorage::cpu(cast), shape.to_vec(), false)
        .unwrap()
        .requires_grad_(rg)
}

fn idx_cpu(vals: &[i64]) -> IntTensor<i64> {
    IntTensor::from_vec(vals.to_vec(), vec![vals.len()]).unwrap()
}

fn idx_cuda(vals: &[i64]) -> IntTensor<i64> {
    idx_cpu(vals).to(Device::Cuda(0)).unwrap()
}

/// Read any-device tensor back as exact f64 values for oracle comparison.
fn host_f64<T: Float>(t: &Tensor<T>) -> Vec<f64> {
    t.data_vec()
        .unwrap()
        .into_iter()
        .map(|v| <f64 as num_traits::NumCast>::from(v).unwrap())
        .collect()
}

fn assert_cuda_with<T: Float>(t: &Tensor<T>, what: &str, expected: &[f64]) {
    assert!(
        t.is_cuda(),
        "{what} must be CUDA-resident (CORE-048: silent CPU demotion), got {:?}",
        t.device()
    );
    assert_eq!(host_f64(t), expected, "{what} values vs live torch oracle");
}

fn grad_of<T: Float>(leaf: &Tensor<T>) -> Tensor<T> {
    leaf.grad()
        .unwrap()
        .expect("grad must reach the leaf (R-ORACLE-3)")
}

// ---------------------------------------------------------------------------
// scatter_reduce
// ---------------------------------------------------------------------------

/// Live torch (cuda, both dtypes):
///   x = tensor([1.,2.,3.,4.], requires_grad=True)
///   s = tensor([10.,20.], requires_grad=True)
///   out = x.scatter_reduce(0, tensor([0,2]), s, 'sum', include_self=True)
///   out.backward(tensor([1.,2.,3.,4.]))
///   -> out [11., 2., 23., 4.] on cuda:0; x.grad [1,2,3,4]; s.grad [1,3].
fn scatter_reduce_sum_cuda_resident<T: Float>() {
    ensure_cuda_backend();
    let x = t_cuda::<T>(&[1.0, 2.0, 3.0, 4.0], &[4], true);
    let s = t_cuda::<T>(&[10.0, 20.0], &[2], true);
    let out = scatter_reduce(&x, 0, &[0, 2], &[2], &s, ScatterReduce::Sum, true).unwrap();
    assert_cuda_with(&out, "scatter_reduce(sum) output", &[11.0, 2.0, 23.0, 4.0]);

    let seed = t_cuda::<T>(&[1.0, 2.0, 3.0, 4.0], &[4], false);
    backward_with_grad(&out, Some(&seed)).unwrap();
    let gx = grad_of(&x);
    let gs = grad_of(&s);
    assert_cuda_with(&gx, "scatter_reduce(sum) grad_input", &[1.0, 2.0, 3.0, 4.0]);
    assert_cuda_with(&gs, "scatter_reduce(sum) grad_src", &[1.0, 3.0]);
}

#[test]
fn scatter_reduce_sum_cuda_resident_f32() {
    scatter_reduce_sum_cuda_resident::<f32>();
}

#[test]
fn scatter_reduce_sum_cuda_resident_f64() {
    scatter_reduce_sum_cuda_resident::<f64>();
}

/// Live torch (cuda, both dtypes):
///   x = tensor([[1.,2.,3.],[4.,5.,6.]], requires_grad=True)
///   s = tensor([[10.,20.],[30.,40.]], requires_grad=True)
///   idx = tensor([[0,1],[1,0]])
///   out = x.scatter_reduce(0, idx, s, 'sum', include_self=False)
///   out.backward(tensor([[1.,2.,3.],[4.,5.,6.]]))
///   -> out [[10.,40.,3.],[30.,20.,6.]];
///      x.grad [[0.,0.,3.],[0.,0.,6.]];
///      s.grad [[1.,5.],[4.,2.]].
///
/// This pins the resident sum backward path: `grad_self.scatter(dim,index,0)`
/// and `grad.gather(dim,index)` both operate over an index shape smaller than
/// the input's non-dim axis, so a rectangular dim-only shortcut would be wrong.
fn scatter_reduce_sum_include_self_false_cuda_resident<T: Float>() {
    ensure_cuda_backend();
    let x = t_cuda::<T>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    let s = t_cuda::<T>(&[10.0, 20.0, 30.0, 40.0], &[2, 2], true);
    let index = [0, 1, 1, 0];
    let index_shape = [2, 2];
    let out = scatter_reduce(&x, 0, &index, &index_shape, &s, ScatterReduce::Sum, false).unwrap();
    assert_cuda_with(
        &out,
        "scatter_reduce(sum,!include_self) output",
        &[10.0, 40.0, 3.0, 30.0, 20.0, 6.0],
    );

    let seed = t_cuda::<T>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    backward_with_grad(&out, Some(&seed)).unwrap();
    assert_cuda_with(
        &grad_of(&x),
        "scatter_reduce(sum,!include_self) grad_input",
        &[0.0, 0.0, 3.0, 0.0, 0.0, 6.0],
    );
    assert_cuda_with(
        &grad_of(&s),
        "scatter_reduce(sum,!include_self) grad_src",
        &[1.0, 5.0, 4.0, 2.0],
    );
}

#[test]
fn scatter_reduce_sum_include_self_false_cuda_resident_f32() {
    scatter_reduce_sum_include_self_false_cuda_resident::<f32>();
}

#[test]
fn scatter_reduce_sum_include_self_false_cuda_resident_f64() {
    scatter_reduce_sum_include_self_false_cuda_resident::<f64>();
}

/// The value-aware reduce modes run their backward formulas on CUDA-resident
/// tensors. Live torch (cuda, both dtypes):
///   x = tensor([5.,2.,30.,4.], requires_grad=True); s = tensor([5.,20.], rg)
///   out = x.scatter_reduce(0, tensor([0,2]), s, 'amax', include_self=True)
///   out.backward(tensor([1.,2.,3.,4.]))
///   -> out [5., 2., 30., 4.]; x.grad [0.5, 2., 3., 4.]; s.grad [0.5, 0.]
///   (the tie at slot 0 splits grad 1.0 evenly between self and src).
///   amin include_self=false with self/src ties at the touched slot counts self
///   in PyTorch's denominator, then zeros grad_self at touched positions:
///   x=[1,2,3,4], s=[1,1], idx=[0,0], seed=[6,7,8,9]
///   -> out [1,2,3,4]; x.grad [0,7,8,9]; s.grad [2,2].
///   prod zero cases:
///   x=[2,3,4,5], s=[0,7], idx=[0,2], seed=[1,2,3,4]
///   -> x.grad [0,2,21,4], s.grad [2,12].
fn scatter_reduce_value_aware_cuda_resident<T: Float>() {
    ensure_cuda_backend();
    // amax with a self/src tie at slot 0.
    let x = t_cuda::<T>(&[5.0, 2.0, 30.0, 4.0], &[4], true);
    let s = t_cuda::<T>(&[5.0, 20.0], &[2], true);
    let out = scatter_reduce(&x, 0, &[0, 2], &[2], &s, ScatterReduce::Amax, true).unwrap();
    assert_cuda_with(&out, "scatter_reduce(amax) output", &[5.0, 2.0, 30.0, 4.0]);
    let seed = t_cuda::<T>(&[1.0, 2.0, 3.0, 4.0], &[4], false);
    backward_with_grad(&out, Some(&seed)).unwrap();
    assert_cuda_with(
        &grad_of(&x),
        "scatter_reduce(amax) grad_input",
        &[0.5, 2.0, 3.0, 4.0],
    );
    assert_cuda_with(&grad_of(&s), "scatter_reduce(amax) grad_src", &[0.5, 0.0]);

    // amin with include_self=false: self contributes to PyTorch's tie count,
    // then grad_self is zeroed at touched positions.
    let x = t_cuda::<T>(&[1.0, 2.0, 3.0, 4.0], &[4], true);
    let s = t_cuda::<T>(&[1.0, 1.0], &[2], true);
    let out = scatter_reduce(&x, 0, &[0, 0], &[2], &s, ScatterReduce::Amin, false).unwrap();
    assert_cuda_with(
        &out,
        "scatter_reduce(amin,!include_self) output",
        &[1.0, 2.0, 3.0, 4.0],
    );
    let seed = t_cuda::<T>(&[6.0, 7.0, 8.0, 9.0], &[4], false);
    backward_with_grad(&out, Some(&seed)).unwrap();
    assert_cuda_with(
        &grad_of(&x),
        "scatter_reduce(amin,!include_self) grad_input",
        &[0.0, 7.0, 8.0, 9.0],
    );
    assert_cuda_with(
        &grad_of(&s),
        "scatter_reduce(amin,!include_self) grad_src",
        &[2.0, 2.0],
    );

    // prod with ordinary nonzero factors.
    let x = t_cuda::<T>(&[2.0, 3.0, 4.0, 5.0], &[4], true);
    let s = t_cuda::<T>(&[6.0, 7.0], &[2], true);
    let out = scatter_reduce(&x, 0, &[0, 2], &[2], &s, ScatterReduce::Prod, true).unwrap();
    assert_cuda_with(&out, "scatter_reduce(prod) output", &[12.0, 3.0, 28.0, 5.0]);
    let seed = t_cuda::<T>(&[1.0, 2.0, 3.0, 4.0], &[4], false);
    backward_with_grad(&out, Some(&seed)).unwrap();
    assert_cuda_with(
        &grad_of(&x),
        "scatter_reduce(prod) grad_input",
        &[6.0, 2.0, 21.0, 4.0],
    );
    assert_cuda_with(&grad_of(&s), "scatter_reduce(prod) grad_src", &[2.0, 12.0]);

    // prod with one zero in a scatter bucket: PyTorch uses the exclusive
    // product branch for that zero source.
    let x = t_cuda::<T>(&[2.0, 3.0, 4.0, 5.0], &[4], true);
    let s = t_cuda::<T>(&[0.0, 7.0], &[2], true);
    let out = scatter_reduce(&x, 0, &[0, 2], &[2], &s, ScatterReduce::Prod, true).unwrap();
    assert_cuda_with(
        &out,
        "scatter_reduce(prod zero) output",
        &[0.0, 3.0, 28.0, 5.0],
    );
    let seed = t_cuda::<T>(&[1.0, 2.0, 3.0, 4.0], &[4], false);
    backward_with_grad(&out, Some(&seed)).unwrap();
    assert_cuda_with(
        &grad_of(&x),
        "scatter_reduce(prod zero) grad_input",
        &[0.0, 2.0, 21.0, 4.0],
    );
    assert_cuda_with(
        &grad_of(&s),
        "scatter_reduce(prod zero) grad_src",
        &[2.0, 12.0],
    );
}

#[test]
fn scatter_reduce_value_aware_cuda_resident_f32() {
    scatter_reduce_value_aware_cuda_resident::<f32>();
}

#[test]
fn scatter_reduce_value_aware_cuda_resident_f64() {
    scatter_reduce_value_aware_cuda_resident::<f64>();
}

/// Live torch (cuda, both dtypes):
///   x=[0,2,3,4], s=[6,6,7], idx=[0,0,2], seed=[6,8,10,12]
///   include_self=True:
///     out=[4,2,5,4], x.grad=[2,8,5,12], s.grad=[2,2,5]
///   include_self=False:
///     out=[6,2,7,4], x.grad=[0,8,0,12], s.grad=[3,3,10]
///
/// Mean is a resident composite: sum scatter_reduce divided by a resident
/// scatter_add count tensor. The exact integer-valued oracle still exercises
/// duplicate-index denominators 2 and 3.
fn scatter_reduce_mean_cuda_resident<T: Float>() {
    ensure_cuda_backend();
    let x = t_cuda::<T>(&[0.0, 2.0, 3.0, 4.0], &[4], true);
    let s = t_cuda::<T>(&[6.0, 6.0, 7.0], &[3], true);
    let out = x
        .scatter_reduce_t(0, &[0, 0, 2], &[3], &s, "mean", true)
        .unwrap();
    assert_cuda_with(&out, "scatter_reduce(mean) output", &[4.0, 2.0, 5.0, 4.0]);
    let seed = t_cuda::<T>(&[6.0, 8.0, 10.0, 12.0], &[4], false);
    backward_with_grad(&out, Some(&seed)).unwrap();
    assert_cuda_with(
        &grad_of(&x),
        "scatter_reduce(mean) grad_input",
        &[2.0, 8.0, 5.0, 12.0],
    );
    assert_cuda_with(
        &grad_of(&s),
        "scatter_reduce(mean) grad_src",
        &[2.0, 2.0, 5.0],
    );

    let x = t_cuda::<T>(&[0.0, 2.0, 3.0, 4.0], &[4], true);
    let s = t_cuda::<T>(&[6.0, 6.0, 7.0], &[3], true);
    let out = x
        .scatter_reduce_t(0, &[0, 0, 2], &[3], &s, "mean", false)
        .unwrap();
    assert_cuda_with(
        &out,
        "scatter_reduce(mean,!include_self) output",
        &[6.0, 2.0, 7.0, 4.0],
    );
    let seed = t_cuda::<T>(&[6.0, 8.0, 10.0, 12.0], &[4], false);
    backward_with_grad(&out, Some(&seed)).unwrap();
    assert_cuda_with(
        &grad_of(&x),
        "scatter_reduce(mean,!include_self) grad_input",
        &[0.0, 8.0, 0.0, 12.0],
    );
    assert_cuda_with(
        &grad_of(&s),
        "scatter_reduce(mean,!include_self) grad_src",
        &[3.0, 3.0, 10.0],
    );
}

#[test]
fn scatter_reduce_mean_cuda_resident_f32() {
    scatter_reduce_mean_cuda_resident::<f32>();
}

#[test]
fn scatter_reduce_mean_cuda_resident_f64() {
    scatter_reduce_mean_cuda_resident::<f64>();
}

/// torch: `cuda_x.scatter_reduce(0, cuda_idx, cpu_src, 'sum')` ->
/// RuntimeError "Expected all tensors to be on the same device, but got src
/// is on cpu, different from other tensors on cuda:0".
#[test]
fn scatter_reduce_mixed_device_rejected() {
    ensure_cuda_backend();
    let x_cuda = t_cuda::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], false);
    let x_cpu = t_cpu::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], false);
    let s_cuda = t_cuda::<f32>(&[10.0, 20.0], &[2], false);
    let s_cpu = t_cpu::<f32>(&[10.0, 20.0], &[2], false);

    let err = scatter_reduce(&x_cuda, 0, &[0, 2], &[2], &s_cpu, ScatterReduce::Sum, true)
        .expect_err("CUDA input + CPU src must be rejected (CORE-048)");
    assert!(
        matches!(err, FerrotorchError::DeviceMismatch { .. }),
        "expected DeviceMismatch, got {err:?}"
    );
    let err = scatter_reduce(&x_cpu, 0, &[0, 2], &[2], &s_cuda, ScatterReduce::Sum, true)
        .expect_err("CPU input + CUDA src must be rejected (CORE-048)");
    assert!(
        matches!(err, FerrotorchError::DeviceMismatch { .. }),
        "expected DeviceMismatch, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// index_add
// ---------------------------------------------------------------------------

/// Live torch (cuda, both dtypes):
///   x = tensor([[1.,2.,3.],[4.,5.,6.]], requires_grad=True)
///   s = tensor([[10.,20.],[30.,40.]], requires_grad=True)
///   out = x.index_add(1, tensor([2,0]), s, alpha=2.0)
///   out.backward(tensor([[1.,2.,3.],[4.,5.,6.]]))
///   -> out [[41., 2., 23.], [84., 5., 66.]] on cuda:0;
///      x.grad [[1,2,3],[4,5,6]]; s.grad [[6., 2.], [12., 8.]].
fn index_add_cuda_resident<T: Float>() {
    ensure_cuda_backend();
    let x = t_cuda::<T>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    let s = t_cuda::<T>(&[10.0, 20.0, 30.0, 40.0], &[2, 2], true);
    let idx = idx_cuda(&[2, 0]);
    let out = index_add(&x, 1, &idx, &s, 2.0).unwrap();
    assert_cuda_with(
        &out,
        "index_add output",
        &[41.0, 2.0, 23.0, 84.0, 5.0, 66.0],
    );

    let seed = t_cuda::<T>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    backward_with_grad(&out, Some(&seed)).unwrap();
    assert_cuda_with(
        &grad_of(&x),
        "index_add grad_input",
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    assert_cuda_with(
        &grad_of(&s),
        "index_add grad_source",
        &[6.0, 2.0, 12.0, 8.0],
    );
}

#[test]
fn index_add_cuda_resident_f32() {
    index_add_cuda_resident::<f32>();
}

#[test]
fn index_add_cuda_resident_f64() {
    index_add_cuda_resident::<f64>();
}

/// torch rejects every operand-device mix for index_add, including a CPU
/// index against CUDA tensors ("...but got index is on cpu...").
#[test]
fn index_add_mixed_device_rejected() {
    ensure_cuda_backend();
    let x_cuda = t_cuda::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], false);
    let x_cpu = t_cpu::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], false);
    let s_cuda = t_cuda::<f32>(&[10.0, 20.0], &[2], false);
    let s_cpu = t_cpu::<f32>(&[10.0, 20.0], &[2], false);
    let i_cuda = idx_cuda(&[0, 2]);
    let i_cpu = idx_cpu(&[0, 2]);

    for (name, r) in [
        (
            "CUDA input + CPU index",
            index_add(&x_cuda, 0, &i_cpu, &s_cuda, 1.0),
        ),
        (
            "CUDA input + CPU source",
            index_add(&x_cuda, 0, &i_cuda, &s_cpu, 1.0),
        ),
        (
            "CPU input + CUDA index",
            index_add(&x_cpu, 0, &i_cuda, &s_cpu, 1.0),
        ),
    ] {
        let err = r.expect_err(name);
        assert!(
            matches!(err, FerrotorchError::DeviceMismatch { .. }),
            "{name}: expected DeviceMismatch, got {err:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// index_copy
// ---------------------------------------------------------------------------

/// Live torch (cuda, both dtypes):
///   x = tensor([[1.,2.],[3.,4.],[5.,6.]], requires_grad=True)
///   s = tensor([[10.,20.],[30.,40.]], requires_grad=True)
///   out = x.index_copy(0, tensor([2,0]), s)
///   out.backward(tensor([[1.,2.],[3.,4.],[5.,6.]]))
///   -> out [[30., 40.], [3., 4.], [10., 20.]] on cuda:0;
///      x.grad [[0,0],[3,4],[0,0]]; s.grad [[5., 6.], [1., 2.]].
fn index_copy_cuda_resident<T: Float>() {
    ensure_cuda_backend();
    let x = t_cuda::<T>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], true);
    let s = t_cuda::<T>(&[10.0, 20.0, 30.0, 40.0], &[2, 2], true);
    let idx = idx_cuda(&[2, 0]);
    let out = index_copy(&x, 0, &idx, &s).unwrap();
    assert_cuda_with(
        &out,
        "index_copy output",
        &[30.0, 40.0, 3.0, 4.0, 10.0, 20.0],
    );

    let seed = t_cuda::<T>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], false);
    backward_with_grad(&out, Some(&seed)).unwrap();
    assert_cuda_with(
        &grad_of(&x),
        "index_copy grad_input",
        &[0.0, 0.0, 3.0, 4.0, 0.0, 0.0],
    );
    assert_cuda_with(
        &grad_of(&s),
        "index_copy grad_source",
        &[5.0, 6.0, 1.0, 2.0],
    );
}

#[test]
fn index_copy_cuda_resident_f32() {
    index_copy_cuda_resident::<f32>();
}

#[test]
fn index_copy_cuda_resident_f64() {
    index_copy_cuda_resident::<f64>();
}

#[test]
fn index_copy_mixed_device_rejected() {
    ensure_cuda_backend();
    let x_cuda = t_cuda::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], false);
    let x_cpu = t_cpu::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], false);
    let s_cuda = t_cuda::<f32>(&[10.0, 20.0], &[2], false);
    let s_cpu = t_cpu::<f32>(&[10.0, 20.0], &[2], false);
    let i_cuda = idx_cuda(&[0, 2]);
    let i_cpu = idx_cpu(&[0, 2]);

    for (name, r) in [
        (
            "CUDA input + CPU index",
            index_copy(&x_cuda, 0, &i_cpu, &s_cuda),
        ),
        (
            "CUDA input + CPU source",
            index_copy(&x_cuda, 0, &i_cuda, &s_cpu),
        ),
        (
            "CPU input + CUDA source",
            index_copy(&x_cpu, 0, &i_cpu, &s_cuda),
        ),
    ] {
        let err = r.expect_err(name);
        assert!(
            matches!(err, FerrotorchError::DeviceMismatch { .. }),
            "{name}: expected DeviceMismatch, got {err:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// take
// ---------------------------------------------------------------------------

/// Live torch (cuda, both dtypes):
///   x = tensor([[10.,20.,30.],[40.,50.,60.]], requires_grad=True)
///   out = torch.take(x, tensor([0, -1, 4, 0]))
///   out.backward(tensor([1.,2.,3.,4.]))
///   -> out [10., 60., 50., 10.] on cuda:0 (negative index wraps);
///      x.grad [[5., 0., 0.], [0., 3., 2.]] (duplicate flat-0 accumulates).
fn take_cuda_resident<T: Float>() {
    ensure_cuda_backend();
    let x = t_cuda::<T>(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3], true);
    let idx = idx_cuda(&[0, -1, 4, 0]);
    let out = take(&x, &idx).unwrap();
    assert_cuda_with(&out, "take output", &[10.0, 60.0, 50.0, 10.0]);

    let seed = t_cuda::<T>(&[1.0, 2.0, 3.0, 4.0], &[4], false);
    backward_with_grad(&out, Some(&seed)).unwrap();
    assert_cuda_with(
        &grad_of(&x),
        "take grad_input",
        &[5.0, 0.0, 0.0, 0.0, 3.0, 2.0],
    );
}

#[test]
fn take_cuda_resident_f32() {
    take_cuda_resident::<f32>();
}

#[test]
fn take_cuda_resident_f64() {
    take_cuda_resident::<f64>();
}

#[test]
fn take_cuda_resident_f16() {
    take_cuda_resident::<f16>();
}

#[test]
fn take_cuda_resident_bf16() {
    take_cuda_resident::<bf16>();
}

fn take_put_cuda_empty_resident<T: Float>() {
    ensure_cuda_backend();
    let x = t_cuda::<T>(&[1.0, 2.0, 3.0], &[3], true);
    let idx = idx_cuda(&[]);

    let taken = take(&x, &idx).unwrap();
    assert!(taken.is_cuda(), "empty take output must stay CUDA-resident");
    assert_eq!(taken.shape(), &[0]);
    assert_eq!(host_f64(&taken), &[] as &[f64]);

    let src = t_cuda::<T>(&[], &[0], true);
    let put_out = put(&x, &idx, &src, false).unwrap();
    assert_cuda_with(&put_out, "empty put output", &[1.0, 2.0, 3.0]);
    let seed = t_cuda::<T>(&[1.0, 2.0, 3.0], &[3], false);
    backward_with_grad(&put_out, Some(&seed)).unwrap();
    assert_cuda_with(&grad_of(&x), "empty put grad_input", &[1.0, 2.0, 3.0]);
    let gs = grad_of(&src);
    assert!(
        gs.is_cuda(),
        "empty put grad_source must stay CUDA-resident"
    );
    assert_eq!(gs.shape(), &[0]);
    assert_eq!(host_f64(&gs), &[] as &[f64]);
}

#[test]
fn take_put_cuda_empty_resident_f16() {
    take_put_cuda_empty_resident::<f16>();
}

#[test]
fn take_put_cuda_empty_resident_bf16() {
    take_put_cuda_empty_resident::<bf16>();
}

/// torch rejects both directions: `torch.take(cuda_x, cpu_idx)` AND
/// `torch.take(cpu_x, cuda_idx)` -> "Expected all tensors to be on the same
/// device" (live probe pasted in #1742).
#[test]
fn take_mixed_device_rejected() {
    ensure_cuda_backend();
    let x_cuda = t_cuda::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], false);
    let x_cpu = t_cpu::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], false);
    let i_cuda = idx_cuda(&[0, 2]);
    let i_cpu = idx_cpu(&[0, 2]);

    let err = take(&x_cuda, &i_cpu).expect_err("CUDA input + CPU index");
    assert!(
        matches!(err, FerrotorchError::DeviceMismatch { .. }),
        "expected DeviceMismatch, got {err:?}"
    );
    let err = take(&x_cpu, &i_cuda).expect_err("CPU input + CUDA index");
    assert!(
        matches!(err, FerrotorchError::DeviceMismatch { .. }),
        "expected DeviceMismatch, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// put
// ---------------------------------------------------------------------------

/// Live torch (cuda, both dtypes):
///   x = tensor([1.,2.,3.,4.], requires_grad=True)
///   s = tensor([10.,20.,30.], requires_grad=True)
///   out = x.put(tensor([1,3,1]), s, accumulate=True)
///   out.backward(tensor([1.,2.,3.,4.]))
///   -> out [1., 42., 3., 24.]; x.grad [1,2,3,4]; s.grad [2., 4., 2.]
///   x = tensor([1.,2.,3.,4.], rg); s = tensor([10.,20.], rg)
///   out = x.put(tensor([3,0]), s, accumulate=False)
///   out.backward(tensor([1.,2.,3.,4.]))
///   -> out [20., 2., 3., 10.]; x.grad [0,2,3,0]; s.grad [4., 1.].
fn put_cuda_resident<T: Float>() {
    ensure_cuda_backend();
    // accumulate=true with a duplicate flat index (atomic-add path).
    let x = t_cuda::<T>(&[1.0, 2.0, 3.0, 4.0], &[4], true);
    let s = t_cuda::<T>(&[10.0, 20.0, 30.0], &[3], true);
    let idx = idx_cuda(&[1, 3, 1]);
    let out = put(&x, &idx, &s, true).unwrap();
    assert_cuda_with(&out, "put(accumulate) output", &[1.0, 42.0, 3.0, 24.0]);
    let seed = t_cuda::<T>(&[1.0, 2.0, 3.0, 4.0], &[4], false);
    backward_with_grad(&out, Some(&seed)).unwrap();
    assert_cuda_with(
        &grad_of(&x),
        "put(accumulate) grad_input",
        &[1.0, 2.0, 3.0, 4.0],
    );
    assert_cuda_with(
        &grad_of(&s),
        "put(accumulate) grad_source",
        &[2.0, 4.0, 2.0],
    );

    // accumulate=false, unique indices (overwrite path).
    let x = t_cuda::<T>(&[1.0, 2.0, 3.0, 4.0], &[4], true);
    let s = t_cuda::<T>(&[10.0, 20.0], &[2], true);
    let idx = idx_cuda(&[3, 0]);
    let out = put(&x, &idx, &s, false).unwrap();
    assert_cuda_with(&out, "put(overwrite) output", &[20.0, 2.0, 3.0, 10.0]);
    let seed = t_cuda::<T>(&[1.0, 2.0, 3.0, 4.0], &[4], false);
    backward_with_grad(&out, Some(&seed)).unwrap();
    assert_cuda_with(
        &grad_of(&x),
        "put(overwrite) grad_input",
        &[0.0, 2.0, 3.0, 0.0],
    );
    assert_cuda_with(&grad_of(&s), "put(overwrite) grad_source", &[4.0, 1.0]);
}

#[test]
fn put_cuda_resident_f32() {
    put_cuda_resident::<f32>();
}

#[test]
fn put_cuda_resident_f64() {
    put_cuda_resident::<f64>();
}

#[test]
fn put_cuda_resident_f16() {
    put_cuda_resident::<f16>();
}

#[test]
fn put_cuda_resident_bf16() {
    put_cuda_resident::<bf16>();
}

fn put_cuda_accumulate_odd_len_duplicate_16bit<T: Float>() {
    ensure_cuda_backend();
    let x = t_cuda::<T>(&[1.0, 2.0, 3.0], &[3], true);
    let s = t_cuda::<T>(&[10.0, 20.0], &[2], true);
    let idx = idx_cuda(&[2, 2]);
    let out = put(&x, &idx, &s, true).unwrap();
    assert_cuda_with(
        &out,
        "put 16-bit odd-len duplicate output",
        &[1.0, 2.0, 33.0],
    );

    let seed = t_cuda::<T>(&[1.0, 2.0, 3.0], &[3], false);
    backward_with_grad(&out, Some(&seed)).unwrap();
    assert_cuda_with(
        &grad_of(&x),
        "put 16-bit odd-len duplicate grad_input",
        &[1.0, 2.0, 3.0],
    );
    assert_cuda_with(
        &grad_of(&s),
        "put 16-bit odd-len duplicate grad_source",
        &[3.0, 3.0],
    );
}

#[test]
fn put_cuda_accumulate_odd_len_duplicate_f16() {
    put_cuda_accumulate_odd_len_duplicate_16bit::<f16>();
}

#[test]
fn put_cuda_accumulate_odd_len_duplicate_bf16() {
    put_cuda_accumulate_odd_len_duplicate_16bit::<bf16>();
}

#[test]
fn put_source_index_numel_mismatch_rejected_like_torch() {
    let x = t_cpu::<f32>(&[1.0, 2.0, 3.0], &[3], false);
    let idx = idx_cpu(&[0, 2]);
    let src = t_cpu::<f32>(&[10.0, 20.0, 30.0], &[3], false);
    let err = put(&x, &idx, &src, false).expect_err("source/index mismatch");
    assert!(
        matches!(err, FerrotorchError::ShapeMismatch { .. }),
        "expected ShapeMismatch, got {err:?}"
    );
}

#[test]
fn put_mixed_device_rejected() {
    ensure_cuda_backend();
    let x_cuda = t_cuda::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], false);
    let x_cpu = t_cpu::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], false);
    let s_cuda = t_cuda::<f32>(&[10.0, 20.0], &[2], false);
    let s_cpu = t_cpu::<f32>(&[10.0, 20.0], &[2], false);
    let i_cuda = idx_cuda(&[0, 2]);
    let i_cpu = idx_cpu(&[0, 2]);

    for (name, r) in [
        (
            "CUDA input + CPU index",
            put(&x_cuda, &i_cpu, &s_cuda, false),
        ),
        (
            "CUDA input + CPU source",
            put(&x_cuda, &i_cuda, &s_cpu, true),
        ),
        (
            "CPU input + CUDA source",
            put(&x_cpu, &i_cpu, &s_cuda, false),
        ),
    ] {
        let err = r.expect_err(name);
        assert!(
            matches!(err, FerrotorchError::DeviceMismatch { .. }),
            "{name}: expected DeviceMismatch, got {err:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// masked_scatter
// ---------------------------------------------------------------------------

/// The all-CUDA forward already runs on-device (#1662); CORE-048 adds the
/// missing pieces: gradients on the leaf devices and mixed-device rejection.
///
/// Live torch (cuda, both dtypes):
///   x = tensor([1.,2.,3.,4.], requires_grad=True)
///   s = tensor([10.,20.], requires_grad=True)
///   out = x.masked_scatter(tensor([True,False,True,False]), s)
///   out.backward(tensor([1.,2.,3.,4.]))
///   -> out [10., 2., 20., 4.] on cuda:0; x.grad [0,2,0,4]; s.grad [1., 3.].
fn masked_scatter_cuda_resident<T: Float>() {
    ensure_cuda_backend();
    let x = t_cuda::<T>(&[1.0, 2.0, 3.0, 4.0], &[4], true);
    let s = t_cuda::<T>(&[10.0, 20.0], &[2], true);
    let mask = BoolTensor::from_vec(vec![true, false, true, false], vec![4])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    let out = masked_scatter(&x, &mask, &s).unwrap();
    assert_cuda_with(&out, "masked_scatter output", &[10.0, 2.0, 20.0, 4.0]);

    let seed = t_cuda::<T>(&[1.0, 2.0, 3.0, 4.0], &[4], false);
    backward_with_grad(&out, Some(&seed)).unwrap();
    assert_cuda_with(
        &grad_of(&x),
        "masked_scatter grad_input",
        &[0.0, 2.0, 0.0, 4.0],
    );
    assert_cuda_with(&grad_of(&s), "masked_scatter grad_source", &[1.0, 3.0]);
}

#[test]
fn masked_scatter_cuda_resident_f32() {
    masked_scatter_cuda_resident::<f32>();
}

#[test]
fn masked_scatter_cuda_resident_f64() {
    masked_scatter_cuda_resident::<f64>();
}

/// Live torch (cuda, f16/bf16):
///   x = tensor([[1,2,3],[4,5,6]], dtype=d, device='cuda', requires_grad=True)
///   s = tensor([10,20,30,40,50], dtype=d, device='cuda', requires_grad=True)
///   m = tensor([True,False,True], device='cuda')
///   out = x.masked_scatter(m, s)
///   out.backward(tensor([[1,2,3],[4,5,6]], dtype=d, device='cuda'))
///   -> out [[10,2,20],[30,5,40]];
///      x.grad [[0,2,0],[0,5,0]];
///      s.grad [1,3,4,6,0].
///
/// The tail zero is important: PyTorch's
/// `masked_scatter_backward_symint` pads `grad.masked_select(mask)` back to
/// `source.sizes()` before `view`. This also proves the half/bfloat CUDA
/// backward path is not limited to exact `source.numel()==mask.sum()` cases.
fn masked_scatter_cuda_broadcast_padded_source_16bit<T: Float>() {
    ensure_cuda_backend();
    let x = t_cuda::<T>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    let s = t_cuda::<T>(&[10.0, 20.0, 30.0, 40.0, 50.0], &[5], true);
    let mask = BoolTensor::from_vec(vec![true, false, true], vec![3])
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();

    let out = masked_scatter(&x, &mask, &s).unwrap();
    assert_cuda_with(
        &out,
        "masked_scatter broadcast output",
        &[10.0, 2.0, 20.0, 30.0, 5.0, 40.0],
    );

    let seed = t_cuda::<T>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    backward_with_grad(&out, Some(&seed)).unwrap();
    assert_cuda_with(
        &grad_of(&x),
        "masked_scatter broadcast grad_input",
        &[0.0, 2.0, 0.0, 0.0, 5.0, 0.0],
    );
    assert_cuda_with(
        &grad_of(&s),
        "masked_scatter broadcast padded grad_source",
        &[1.0, 3.0, 4.0, 6.0, 0.0],
    );
}

#[test]
fn masked_scatter_cuda_broadcast_padded_source_f16() {
    masked_scatter_cuda_broadcast_padded_source_16bit::<f16>();
}

#[test]
fn masked_scatter_cuda_broadcast_padded_source_bf16() {
    masked_scatter_cuda_broadcast_padded_source_16bit::<bf16>();
}

/// torch rejects all mixes incl. the host-accessible-mask fallback the audit
/// flagged: `cuda_x.masked_scatter(cpu_mask, cuda_src)` -> "Expected all
/// tensors to be on the same device, but got mask is on cpu...".
#[test]
fn masked_scatter_mixed_device_rejected() {
    ensure_cuda_backend();
    let x_cuda = t_cuda::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], false);
    let x_cpu = t_cpu::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], false);
    let s_cuda = t_cuda::<f32>(&[10.0, 20.0], &[2], false);
    let s_cpu = t_cpu::<f32>(&[10.0, 20.0], &[2], false);
    let m_cpu = BoolTensor::from_vec(vec![true, false, true, false], vec![4]).unwrap();
    let m_cuda = m_cpu.to(Device::Cuda(0)).unwrap();

    for (name, r) in [
        (
            "CUDA input + CPU mask (host-accessible-mask fallback)",
            masked_scatter(&x_cuda, &m_cpu, &s_cuda),
        ),
        (
            "CUDA input + CPU mask + CPU source",
            masked_scatter(&x_cuda, &m_cpu, &s_cpu),
        ),
        (
            "CUDA input + CUDA mask + CPU source",
            masked_scatter(&x_cuda, &m_cuda, &s_cpu),
        ),
        (
            "CPU input + CUDA mask",
            masked_scatter(&x_cpu, &m_cuda, &s_cpu),
        ),
    ] {
        let err = r.expect_err(name);
        assert!(
            matches!(err, FerrotorchError::DeviceMismatch { .. }),
            "{name}: expected DeviceMismatch, got {err:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// CPU control — the all-CPU path is untouched by the CORE-048 fix.
// ---------------------------------------------------------------------------

/// All-CPU operands keep working exactly as before (same oracle values as
/// the CUDA tests above — torch CPU and CUDA agree on every one of these).
#[test]
fn cpu_lane_control_unchanged() {
    let x = t_cpu::<f32>(&[1.0, 2.0, 3.0, 4.0], &[4], false);
    let s = t_cpu::<f32>(&[10.0, 20.0], &[2], false);
    let i = idx_cpu(&[0, 2]);

    let out = scatter_reduce(&x, 0, &[0, 2], &[2], &s, ScatterReduce::Sum, true).unwrap();
    assert!(!out.is_cuda());
    assert_eq!(host_f64(&out), &[11.0, 2.0, 23.0, 4.0]);

    let out = index_add(&x, 0, &i, &s, 1.0).unwrap();
    assert!(!out.is_cuda());
    assert_eq!(host_f64(&out), &[11.0, 2.0, 23.0, 4.0]);

    let out = index_copy(&x, 0, &i, &s).unwrap();
    assert!(!out.is_cuda());
    assert_eq!(host_f64(&out), &[10.0, 2.0, 20.0, 4.0]);

    let out = take(&x, &i).unwrap();
    assert!(!out.is_cuda());
    assert_eq!(host_f64(&out), &[1.0, 3.0]);

    let out = put(&x, &i, &s, false).unwrap();
    assert!(!out.is_cuda());
    assert_eq!(host_f64(&out), &[10.0, 2.0, 20.0, 4.0]);

    let mask = BoolTensor::from_vec(vec![true, false, true, false], vec![4]).unwrap();
    let out = masked_scatter(&x, &mask, &s).unwrap();
    assert!(!out.is_cuda());
    assert_eq!(host_f64(&out), &[10.0, 2.0, 20.0, 4.0]);
}
