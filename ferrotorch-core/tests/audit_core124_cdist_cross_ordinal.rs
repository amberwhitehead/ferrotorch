//! CORE-124 (#1818, CLASS-U High) regression battery: the `cdist` CUDA guard
//! (`ferrotorch-core/src/ops/tensor_ops.rs`) must enforce EXACT device
//! equality — ordinal included — before any backend access. Pre-fix it only
//! checked `is_cuda()` on both operands, so two tensors on DIFFERENT GPU
//! ordinals passed the guard and reached one backend kernel with a pointer
//! owned by another device (backend downcast errors at best, invalid
//! device-memory access at worst).
//!
//! Upstream contract (R-ORACLE-1(c) cite): PyTorch's cdist hard-checks the
//! operand devices against each other before dispatch —
//! `aten/src/ATen/native/Distance.cpp` `cdist_impl`:
//! `TORCH_CHECK(device1 == device2, "X1 and X2 must have the same device
//! type. X1: ", device1, " X2: ", device2)` plus the
//! `TORCH_CHECK(p == 2 || ...)`-adjacent same-GPU guards; cross-ordinal
//! pairs raise `RuntimeError`, they never reach a kernel.
//!
//! Single-GPU honesty note (this host has ONE RTX 3090): a REAL allocation
//! on `cuda:1` is not constructible here, so the cross-ordinal operand is a
//! `GpuBufferHandle::new(Box::new(()), ordinal=1, ...)`-tagged tensor. That
//! pins exactly the property under test — the guard must reject on DEVICE
//! METADATA before any backend/kernel access (a guard that only fails once
//! the kernel dereferences the foreign pointer is precisely the bug). The
//! `gpu` module additionally runs the real-CUDA(0)-operand variant.

use ferrotorch_core::gpu_dispatch::GpuBufferHandle;
use ferrotorch_core::{DType, Device, FerrotorchError, Tensor, TensorStorage};

fn cpu_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu tensor")
}

/// A tensor whose storage is TAGGED as residing on `Cuda(ordinal)` without a
/// real allocation (inner is `()`); `len`/dtype metadata are consistent with
/// the shape. Any code that touches the buffer (instead of refusing on
/// device metadata) fails loudly on the downcast — which is the point.
fn cuda_tagged_f32(ordinal: usize, shape: &[usize]) -> Tensor<f32> {
    let len: usize = shape.iter().product();
    let handle = GpuBufferHandle::new(Box::new(()), ordinal, len, DType::F32);
    Tensor::from_storage(TensorStorage::gpu(handle), shape.to_vec(), false)
        .expect("cuda-tagged tensor")
}

#[track_caller]
fn assert_device_mismatch<T: std::fmt::Debug>(
    r: Result<T, FerrotorchError>,
    expected: Device,
    got: Device,
    what: &str,
) {
    match r {
        Err(FerrotorchError::DeviceMismatch {
            expected: e,
            got: g,
        }) => {
            assert_eq!(e, expected, "{what}: mismatch `expected` field");
            assert_eq!(g, got, "{what}: mismatch `got` field");
        }
        Err(other) => panic!("{what}: expected DeviceMismatch, got Err({other:?})"),
        Ok(v) => panic!("{what}: expected DeviceMismatch, got Ok({v:?})"),
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Validation-path lane (runs on BOTH the cpu and gpu CI lanes — the guard
// must reject on metadata BEFORE looking up the backend, so no backend is
// needed). Pre-fix observed: `is_cuda() && is_cuda()` passes, then the
// backend lookup decides the outcome (`NotImplementedOnCuda` without a
// backend; a kernel fed a foreign-device pointer with one).
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn cdist_cross_ordinal_rejected_before_backend_access() {
    let x1 = cuda_tagged_f32(0, &[2, 3]);
    let x2 = cuda_tagged_f32(1, &[2, 3]);
    assert_eq!(x1.device(), Device::Cuda(0));
    assert_eq!(x2.device(), Device::Cuda(1));
    assert_device_mismatch(
        ferrotorch_core::ops::tensor_ops::cdist(&x1, &x2, 2.0),
        Device::Cuda(0),
        Device::Cuda(1),
        "cdist(cuda:0, cuda:1)",
    );
    // And the mirrored order.
    assert_device_mismatch(
        ferrotorch_core::ops::tensor_ops::cdist(&x2, &x1, 2.0),
        Device::Cuda(1),
        Device::Cuda(0),
        "cdist(cuda:1, cuda:0)",
    );
}

/// CPU × CUDA mixes are the same structured refusal (both directions).
#[test]
fn cdist_cpu_cuda_mix_is_device_mismatch() {
    let cpu = cpu_f32(&[0.0; 6], &[2, 3]);
    let gpu = cuda_tagged_f32(0, &[2, 3]);
    assert_device_mismatch(
        ferrotorch_core::ops::tensor_ops::cdist(&cpu, &gpu, 2.0),
        Device::Cpu,
        Device::Cuda(0),
        "cdist(cpu, cuda:0)",
    );
    assert_device_mismatch(
        ferrotorch_core::ops::tensor_ops::cdist(&gpu, &cpu, 2.0),
        Device::Cuda(0),
        Device::Cpu,
        "cdist(cuda:0, cpu)",
    );
}

// ─────────────────────────────────────────────────────────────────────────
// CUDA lane: one REAL Cuda(0) operand against the cuda:1-tagged operand,
// with the backend initialized — the guard must still refuse on metadata
// (pre-fix this is the configuration that reached the kernel).
// ─────────────────────────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for the CORE-124 GPU pins");
        });
    }

    #[test]
    fn cdist_real_cuda0_vs_tagged_cuda1_rejected() {
        ensure_cuda_backend();
        let x1 = cpu_f32(&[0.0, 0.5, 2.0, 1.0, 1.0, 1.0], &[2, 3])
            .to(Device::Cuda(0))
            .expect("upload x1");
        let x2 = cuda_tagged_f32(1, &[2, 3]);
        assert_device_mismatch(
            ferrotorch_core::ops::tensor_ops::cdist(&x1, &x2, 2.0),
            Device::Cuda(0),
            Device::Cuda(1),
            "cdist(real cuda:0, tagged cuda:1)",
        );
        assert_device_mismatch(
            ferrotorch_core::ops::tensor_ops::cdist(&x2, &x1, 2.0),
            Device::Cuda(1),
            Device::Cuda(0),
            "cdist(tagged cuda:1, real cuda:0)",
        );
    }

    /// Control: same-ordinal operands still compute (the guard rejects on
    /// INEQUALITY only). Oracle: `torch.cdist` p=2 on the CORE-122 vectors.
    #[test]
    fn cdist_same_ordinal_still_computes() {
        ensure_cuda_backend();
        let x1 = cpu_f32(&[0.0, 0.5, 2.0, 1.0, 1.0, 1.0], &[2, 3])
            .to(Device::Cuda(0))
            .expect("upload x1");
        let x2 = cpu_f32(&[0.0, 1.5, 0.5, 1.0, 1.0, 1.0], &[2, 3])
            .to(Device::Cuda(0))
            .expect("upload x2");
        let r = ferrotorch_core::ops::tensor_ops::cdist(&x1, &x2, 2.0).expect("same-device cdist");
        assert_eq!(r.device(), Device::Cuda(0), "output device");
        let host = r.cpu().expect("D2H");
        let d = host.data().expect("data");
        // LIVE torch 2.11.0 oracle: [[1.8027756, 1.5], [1.2247449, 0.0]];
        // 1e-5 abs (f32 eps ~1.2e-7, m=3 accumulation + sqrt).
        let expected = [1.802_775_6_f32, 1.5, 1.224_744_9, 0.0];
        for (i, (&g, &e)) in d.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() <= 1e-5, "cdist[{i}]: got {g}, expected {e}");
        }
    }
}
