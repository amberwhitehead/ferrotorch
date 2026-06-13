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
//! Contract (goal-audit-fix.md R-LOUD-1; torch parity: the dispatcher
//! raises NotImplementedError for missing CUDA kernels, it never
//! silently migrates a parameter to CPU). This is CLASS-S **path (b)**:
//! the error boundary, NOT the half-precision feature — the on-device
//! f16/bf16 lane is tracked in #1966.

#![cfg(feature = "gpu")]

use ferrotorch_core::{Device, FerrotorchError, SparseGrad, Tensor, TensorStorage};
use half::bf16;
use std::sync::Once;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend().expect("CUDA backend must initialize for the GPU lane");
    });
}

/// A bf16 CUDA param must get a structured `NotImplementedOnCuda` error
/// — never a successful step that leaves the param on CPU.
#[test]
fn core080_gpu_bf16_param_errors_instead_of_cpu_demotion() {
    ensure_cuda_backend();

    let data: Vec<bf16> = (0..6).map(|i| bf16::from_f32(i as f32)).collect();
    let cpu_param =
        Tensor::<bf16>::from_storage(TensorStorage::cpu(data.clone()), vec![2, 3], false).unwrap();
    let mut param = cpu_param.to(Device::Cuda(0)).expect("bf16 param->cuda");
    assert!(param.is_cuda(), "precondition: param starts on CUDA");

    let grad = SparseGrad::<bf16>::new(
        vec![1],
        vec![
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
            bf16::from_f32(3.0),
        ],
        vec![3],
    )
    .expect("bf16 grad");

    let r = grad.apply_sgd(&mut param, bf16::from_f32(0.5));
    match r {
        Err(FerrotorchError::NotImplementedOnCuda { op }) => {
            assert!(
                op.contains("apply_sgd"),
                "error must name the op, got: {op}"
            );
        }
        other => panic!(
            "bf16 CUDA sparse SGD must return NotImplementedOnCuda (feature: #1966), got {other:?}"
        ),
    }

    // The param must be left on CUDA and untouched (the error fires
    // BEFORE any mutation or device change).
    assert!(
        param.is_cuda(),
        "param must remain on CUDA after the rejected step (pre-fix: silently demoted to CPU)"
    );
    let back = param.cpu().expect("gpu->cpu");
    let after = back.data().expect("data");
    for (i, (a, b)) in after.iter().zip(data.iter()).enumerate() {
        assert_eq!(a, b, "param elem {i} must be untouched");
    }
}
