//! Red-then-green regression tests for audit finding CORE-080 (crosslink
//! #1774): unsupported CUDA sparse-SGD dtypes silently move the parameter
//! to CPU (CLASS-S — the implementation comment claimed unsupported CUDA
//! dtypes would fail at `data_vec`, but `Tensor::data_vec`
//! (tensor.rs:758-769) explicitly downloads CUDA tensors; the generic CPU
//! lane then performed the update and reassigned `*param` with CPU
//! storage, so a SUCCESSFUL optimizer step demoted the device).
//!
//! Observed at HEAD (red run, 2026-06-12, `--features gpu`, RTX 3090):
//! `apply_sgd` on a bf16 CUDA param returned `Ok(())` with
//! `param.is_cuda() == false` afterwards — silent CPU demotion.
//!
//! Contract (goal-audit-fix.md R-LOUD-1 plus PyTorch parity): CUDA sparse
//! SGD for half-precision floating dtypes runs on device and never silently
//! migrates a parameter to CPU. PyTorch's `optim.SGD` sparse path uses
//! `param.add_(grad, alpha=-lr)` on CUDA for `float16` and `bfloat16`.

#![cfg(feature = "gpu")]

use ferrotorch_core::{Device, SparseGrad, Tensor, TensorStorage};
use half::{bf16, f16};
use std::sync::Once;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend().expect("CUDA backend must initialize for the GPU lane");
    });
}

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-3,
            "{label}: element {i} mismatch: actual={a} expected={e}"
        );
    }
}

/// bf16 CUDA sparse SGD must run on device, accumulate duplicate sparse rows,
/// and preserve CUDA residency (torch oracle: [[-2,-1.5,-1],[-1,-1,-1]]).
#[test]
fn core080_gpu_bf16_param_updates_on_device_without_cpu_demotion() {
    ensure_cuda_backend();

    let data: Vec<bf16> = (0..6).map(|i| bf16::from_f32(i as f32)).collect();
    let cpu_param =
        Tensor::<bf16>::from_storage(TensorStorage::cpu(data.clone()), vec![2, 3], false).unwrap();
    let mut param = cpu_param.to(Device::Cuda(0)).expect("bf16 param->cuda");
    assert!(param.is_cuda(), "precondition: param starts on CUDA");

    let grad = SparseGrad::<bf16>::new(
        vec![1, 0, 1],
        vec![
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
            bf16::from_f32(3.0),
            bf16::from_f32(4.0),
            bf16::from_f32(5.0),
            bf16::from_f32(6.0),
            bf16::from_f32(7.0),
            bf16::from_f32(8.0),
            bf16::from_f32(9.0),
        ],
        vec![3],
    )
    .expect("bf16 grad");

    grad.apply_sgd(&mut param, bf16::from_f32(0.5))
        .expect("bf16 CUDA sparse SGD must run on device");

    assert!(
        param.is_cuda(),
        "param must remain on CUDA after bf16 sparse SGD (pre-fix: silently demoted to CPU)"
    );
    let back = param.cpu().expect("gpu->cpu");
    let after: Vec<f32> = back
        .data()
        .expect("data")
        .iter()
        .map(|x| x.to_f32())
        .collect();
    assert_close(
        &after,
        &[-2.0, -1.5, -1.0, -1.0, -1.0, -1.0],
        "bf16 sparse SGD",
    );
}

/// f16 takes the same CUDA composite path as bf16 and must likewise stay on
/// device with PyTorch's sparse-update semantics.
#[test]
fn core080_gpu_f16_param_updates_on_device_without_cpu_demotion() {
    ensure_cuda_backend();

    let data: Vec<f16> = (0..6).map(|i| f16::from_f32(i as f32)).collect();
    let cpu_param =
        Tensor::<f16>::from_storage(TensorStorage::cpu(data), vec![2, 3], false).unwrap();
    let mut param = cpu_param.to(Device::Cuda(0)).expect("f16 param->cuda");
    assert!(param.is_cuda(), "precondition: param starts on CUDA");

    let grad = SparseGrad::<f16>::new(
        vec![1, 0, 1],
        vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
            f16::from_f32(5.0),
            f16::from_f32(6.0),
            f16::from_f32(7.0),
            f16::from_f32(8.0),
            f16::from_f32(9.0),
        ],
        vec![3],
    )
    .expect("f16 grad");

    grad.apply_sgd(&mut param, f16::from_f32(0.5))
        .expect("f16 CUDA sparse SGD must run on device");

    assert!(
        param.is_cuda(),
        "param must remain on CUDA after f16 sparse SGD (pre-fix: silently demoted to CPU)"
    );
    let back = param.cpu().expect("gpu->cpu");
    let after: Vec<f32> = back
        .data()
        .expect("data")
        .iter()
        .map(|x| x.to_f32())
        .collect();
    assert_close(
        &after,
        &[-2.0, -1.5, -1.0, -1.0, -1.0, -1.0],
        "f16 sparse SGD",
    );
}

/// The f16/bf16 segment kernels use an internal guard element for odd
/// half-word output lengths. Tensor storage must still expose the logical
/// length, or the in-place optimizer landing rejects the update.
#[test]
fn core080_gpu_bf16_odd_numel_update_uses_logical_storage_len() {
    ensure_cuda_backend();

    let data: Vec<bf16> = (0..3).map(|i| bf16::from_f32(i as f32)).collect();
    let cpu_param = Tensor::<bf16>::from_storage(TensorStorage::cpu(data), vec![3], false).unwrap();
    let mut param = cpu_param.to(Device::Cuda(0)).expect("bf16 param->cuda");

    let grad =
        SparseGrad::<bf16>::new(vec![2], vec![bf16::from_f32(2.0)], vec![]).expect("bf16 grad");
    grad.apply_sgd(&mut param, bf16::from_f32(0.5))
        .expect("odd-numel bf16 sparse SGD must land in-place");

    assert!(param.is_cuda(), "odd-numel bf16 update must stay CUDA");
    let back = param.cpu().expect("gpu->cpu");
    let after: Vec<f32> = back
        .data()
        .expect("data")
        .iter()
        .map(|x| x.to_f32())
        .collect();
    assert_close(&after, &[0.0, 1.0, 1.0], "odd bf16 sparse SGD");
}

#[test]
fn core080_gpu_f16_odd_numel_update_uses_logical_storage_len() {
    ensure_cuda_backend();

    let data: Vec<f16> = (0..3).map(|i| f16::from_f32(i as f32)).collect();
    let cpu_param = Tensor::<f16>::from_storage(TensorStorage::cpu(data), vec![3], false).unwrap();
    let mut param = cpu_param.to(Device::Cuda(0)).expect("f16 param->cuda");

    let grad = SparseGrad::<f16>::new(vec![2], vec![f16::from_f32(2.0)], vec![]).expect("f16 grad");
    grad.apply_sgd(&mut param, f16::from_f32(0.5))
        .expect("odd-numel f16 sparse SGD must land in-place");

    assert!(param.is_cuda(), "odd-numel f16 update must stay CUDA");
    let back = param.cpu().expect("gpu->cpu");
    let after: Vec<f32> = back
        .data()
        .expect("data")
        .iter()
        .map(|x| x.to_f32())
        .collect();
    assert_close(&after, &[0.0, 1.0, 1.0], "odd f16 sparse SGD");
}
