//! Phase 0 sentinel (GPU dtype-parity epic, crosslink #1185): the
//! `GpuBufferHandle` dtype tag is authoritative.
//!
//! PyTorch faithfulness: `StorageImpl` holds raw bytes with no dtype; the
//! `ScalarType` tag lives above storage as runtime metadata. ferrotorch now
//! mirrors this — the handle carries a `DType` tag, set by the backend on
//! upload and read back via `GpuBufferHandle::dtype()`.
//!
//! What this probe asserts, for each of f32 / f64 / bf16:
//!   1. A CPU tensor moved to CUDA produces a GPU storage whose handle reports
//!      the correct `.dtype()` (the authoritative tag, not inferred from byte
//!      width). This is the line that, in later phases, stops f16 (also 2
//!      bytes) from colliding with bf16 and i32 (also 4 bytes) from colliding
//!      with f32.
//!   2. `buffer_elem_size(handle)` is derived from that tag (`dtype.size_of()`).
//!   3. A `.to(Cuda) -> .to(Cpu)` round-trip preserves the values exactly,
//!      confirming the tag did not change which concrete buffer was used.
//!
//! Behavior-preserving: Phase 0 adds no new working dtype. f32/f64/bf16 behave
//! exactly as before; this probe just pins the tag down.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::creation::from_slice;
use ferrotorch_core::device::Device;
use ferrotorch_core::gpu_dispatch;
use ferrotorch_core::DType;
use half::bf16;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialise for Phase 0 dtype-tag probe");
    });
}

#[test]
// reason: the round-trip fixtures (1.0, 2.0, 3.5, ...) are all exactly
// representable in their respective dtypes, so bit-exact equality (not an
// epsilon) is the correct assertion for "the bytes survived the transfer".
#[allow(clippy::float_cmp)]
fn probe_phase0_dtype_tag() {
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

    // -- f32 -----------------------------------------------------------------
    {
        let host = [1.0f32, 2.0, 3.0, 4.0];
        let t = from_slice(&host, &[4]).expect("from_slice f32");
        let g = t.to(Device::Cuda(0)).expect("to cuda f32");
        let handle = g.gpu_handle().expect("gpu_handle f32");
        check!(
            handle.dtype() == DType::F32,
            format!("f32 handle tagged F32 (got {})", handle.dtype())
        );
        check!(
            backend.buffer_elem_size(handle) == 4,
            "f32 buffer_elem_size == 4 (derived from tag)"
        );
        let back = g.to(Device::Cpu).expect("to cpu f32");
        let got = back.data().expect("data f32");
        check!(got == &host[..], "f32 round-trip values preserved");
    }

    // -- f64 -----------------------------------------------------------------
    {
        let host = [1.5f64, 2.5, 3.5];
        let t = from_slice(&host, &[3]).expect("from_slice f64");
        let g = t.to(Device::Cuda(0)).expect("to cuda f64");
        let handle = g.gpu_handle().expect("gpu_handle f64");
        check!(
            handle.dtype() == DType::F64,
            format!("f64 handle tagged F64 (got {})", handle.dtype())
        );
        check!(
            backend.buffer_elem_size(handle) == 8,
            "f64 buffer_elem_size == 8 (derived from tag)"
        );
        let back = g.to(Device::Cpu).expect("to cpu f64");
        let got = back.data().expect("data f64");
        check!(got == &host[..], "f64 round-trip values preserved");
    }

    // -- bf16 ----------------------------------------------------------------
    {
        let host = [
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
            bf16::from_f32(-3.0),
        ];
        let t = from_slice(&host, &[3]).expect("from_slice bf16");
        let g = t.to(Device::Cuda(0)).expect("to cuda bf16");
        let handle = g.gpu_handle().expect("gpu_handle bf16");
        check!(
            handle.dtype() == DType::BF16,
            format!("bf16 handle tagged BF16 (got {})", handle.dtype())
        );
        check!(
            backend.buffer_elem_size(handle) == 2,
            "bf16 buffer_elem_size == 2 (derived from tag)"
        );
        let back = g.to(Device::Cpu).expect("to cpu bf16");
        let got = back.data().expect("data bf16");
        check!(got == &host[..], "bf16 round-trip values preserved");
    }

    println!("PASS: {pass}, FAIL: {fail}");
    assert_eq!(fail, 0, "Phase 0 dtype-tag probe had {fail} failures");
}
