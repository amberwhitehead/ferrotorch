//! Phase 2a sentinel (GPU dtype-parity epic, crosslink #1185): `IntTensor<I>`
//! is device-aware — integer data can live on CUDA via the Phase-0 DType-tagged
//! raw-byte transport.
//!
//! PyTorch faithfulness: an integer tensor is ordinary raw-byte storage plus a
//! `ScalarType` tag (`I32` / `I64`); no autograd (PyTorch's
//! `isDifferentiableType(int) == false`). Phase 2a adds NO integer compute
//! kernels — only storage, host↔device transfer, and handle access.
//!
//! What this probe asserts, for each of i32 / i64:
//!   1. A CPU `IntTensor` moved to CUDA reports `.is_cuda()` and a non-CPU
//!      `.device()`.
//!   2. The GPU handle's `.dtype()` is the authoritative tag (`I32` / `I64`),
//!      and `buffer_elem_size` derives from it (4 / 8) — this is the line that,
//!      in later phases, stops an i32 handle (4 bytes) being read as f32.
//!   3. A `.to(Cuda) -> .to(Cpu)` round-trip preserves the values bit-exact.
//!   4. Calling the CPU-data accessor `.data()` on a GPU-resident IntTensor
//!      returns `Err(GpuTensorNotAccessible)` — no silent host readback
//!      (rust-gpu-discipline §3).

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::device::Device;
use ferrotorch_core::error::FerrotorchError;
use ferrotorch_core::gpu_dispatch;
use ferrotorch_core::int_tensor::IntTensor;
use ferrotorch_core::DType;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialise for Phase 2a int-device probe");
    });
}

#[test]
fn probe_phase2a_int_device() {
    ensure_cuda_backend();
    let backend = gpu_dispatch::gpu_backend().expect("backend registered");

    let mut pass = 0usize;
    let mut fail = 0usize;

    macro_rules! check {
        ($cond:expr, $msg:expr) => {
            if $cond {
                pass += 1;
                println!("PASS: {}", $msg);
            } else {
                fail += 1;
                println!("FAIL: {}", $msg);
            }
        };
    }

    // -- i32 -----------------------------------------------------------------
    {
        let host: Vec<i32> = vec![1, -2, 3, i32::MIN, i32::MAX, 0];
        let t = IntTensor::<i32>::from_vec(host.clone(), vec![6]).expect("from_vec i32");
        check!(!t.is_cuda(), "i32 CPU tensor is not cuda before transfer");

        let g = t.to(Device::Cuda(0)).expect("to cuda i32");
        check!(g.is_cuda(), "i32 tensor is_cuda() after .to(Cuda)");
        check!(
            matches!(g.device(), Device::Cuda(0)),
            "i32 tensor device() == Cuda(0)"
        );

        let handle = g.gpu_handle().expect("gpu_handle i32");
        check!(
            handle.dtype() == DType::I32,
            format!("i32 handle tagged I32 (got {})", handle.dtype())
        );
        check!(
            backend.buffer_elem_size(handle) == 4,
            "i32 buffer_elem_size == 4 (derived from tag)"
        );

        // CPU-data accessor on a GPU tensor must error, not silently read VRAM.
        check!(
            matches!(g.data(), Err(FerrotorchError::GpuTensorNotAccessible)),
            "i32 GPU tensor .data() returns Err(GpuTensorNotAccessible)"
        );

        let back = g.to(Device::Cpu).expect("to cpu i32");
        check!(!back.is_cuda(), "i32 tensor not cuda after .to(Cpu)");
        let got = back.data().expect("data i32 after round-trip");
        check!(
            got == host.as_slice(),
            "i32 CPU->CUDA->CPU round-trip values bit-exact"
        );
    }

    // -- i64 -----------------------------------------------------------------
    {
        let host: Vec<i64> = vec![1, -2, 3, i64::MIN, i64::MAX, 0, 1_000_000_000_000];
        let t = IntTensor::<i64>::from_vec(host.clone(), vec![7]).expect("from_vec i64");
        check!(!t.is_cuda(), "i64 CPU tensor is not cuda before transfer");

        let g = t.to(Device::Cuda(0)).expect("to cuda i64");
        check!(g.is_cuda(), "i64 tensor is_cuda() after .to(Cuda)");

        let handle = g.gpu_handle().expect("gpu_handle i64");
        check!(
            handle.dtype() == DType::I64,
            format!("i64 handle tagged I64 (got {})", handle.dtype())
        );
        check!(
            backend.buffer_elem_size(handle) == 8,
            "i64 buffer_elem_size == 8 (derived from tag)"
        );

        check!(
            matches!(g.data(), Err(FerrotorchError::GpuTensorNotAccessible)),
            "i64 GPU tensor .data() returns Err(GpuTensorNotAccessible)"
        );

        let back = g.to(Device::Cpu).expect("to cpu i64");
        let got = back.data().expect("data i64 after round-trip");
        check!(
            got == host.as_slice(),
            "i64 CPU->CUDA->CPU round-trip values bit-exact"
        );
    }

    println!("PASS: {pass}, FAIL: {fail}");
    assert_eq!(fail, 0, "Phase 2a int-device probe had {fail} failures");
}
