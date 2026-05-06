//! Integration tests for `gpu_fill_f64` (#780).
//!
//! Verifies the GPU-resident f64 fill kernel:
//! - Bit-exact value propagation for representative scalars (0.0, -1.0,
//!   3.141592653589793, 1e308 near-max, 1e-308 near-min subnormal boundary).
//! - Correct buffer length (`n` f64 elements, not `n` f32 elements).
//! - Empty-input early-return path.
//!
//! These cases catch the three failure modes called out in #780:
//! (1) byte-stride bug — using f32 stride with f64 values would corrupt
//!     odd-indexed elements; reading back `n=10` filled values catches this.
//! (2) f32 truncation — values like `3.141592653589793` round-trip
//!     bit-exactly only if the kernel pipeline stays in f64.
//! (3) under-sized output buffer — `alloc_zeros_f32` instead of
//!     `alloc_zeros_f64` would fault at the write boundary.

#![cfg(feature = "cuda")]

use ferrotorch_gpu::GpuDevice;
use ferrotorch_gpu::init_cuda_backend;

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

/// Read back a freshly-filled buffer and assert every element is bit-exact.
///
/// `bit-exact` matters: `fill` is the identity on its scalar — copying
/// the same f64 bit-pattern to every element. Any deviation (truncation,
/// stride bug, lane corruption) shows up as a non-equal element.
fn assert_filled_bit_exact(n: usize, scalar: f64) {
    let device = GpuDevice::new(0).expect("CUDA device 0");
    let buf = ferrotorch_gpu::kernels::gpu_fill_f64(n, scalar, &device).expect("gpu_fill_f64");
    assert_eq!(buf.len(), n, "buffer length mismatch for fill_f64({n})");

    let host = ferrotorch_gpu::transfer::gpu_to_cpu(&buf, &device).expect("gpu_to_cpu");
    assert_eq!(host.len(), n);
    for (i, &v) in host.iter().enumerate() {
        assert_eq!(
            v.to_bits(),
            scalar.to_bits(),
            "fill_f64 element[{i}] = {v:?} (bits={:#018x}); expected scalar = {scalar:?} \
             (bits={:#018x})",
            v.to_bits(),
            scalar.to_bits(),
        );
    }
}

#[test]
fn fill_f64_pi_n10_bit_exact() {
    // 3.141592653589793 — the textbook double-precision π. Bit-exact
    // round-trip proves the kernel pipeline is f64 end-to-end (any f32
    // truncation would lose ~7 trailing decimal digits).
    ensure_cuda();
    assert_filled_bit_exact(10, std::f64::consts::PI);
}

#[test]
fn fill_f64_zero() {
    // 0.0 — bit pattern 0x0000_0000_0000_0000. Trivial but catches the
    // "buffer is zeroed by alloc_zeros and the kernel didn't actually
    // run" false-positive (which would also report success for 0.0).
    // Combined with the non-zero cases below this is fine.
    ensure_cuda();
    assert_filled_bit_exact(64, 0.0);
}

#[test]
fn fill_f64_negative_one() {
    // -1.0 — bit pattern 0xBFF0_0000_0000_0000. Confirms the kernel
    // propagates the sign bit and exponent correctly through the
    // `.f64` `ld.param` / `st.global` path.
    ensure_cuda();
    assert_filled_bit_exact(64, -1.0);
}

#[test]
fn fill_f64_near_max() {
    // 1e308 — within an order of magnitude of `f64::MAX` (~1.8e308).
    // Catches any silent f32 truncation: 1e308 is unrepresentable in
    // f32 (overflows to +inf), so an f32 detour anywhere in the pipe
    // would surface as `+inf` in the readback rather than the original
    // value.
    ensure_cuda();
    assert_filled_bit_exact(32, 1e308_f64);
}

#[test]
fn fill_f64_near_min_subnormal() {
    // 1e-308 — near the f64 subnormal boundary (`f64::MIN_POSITIVE`
    // ≈ 2.225e-308). Catches f32 truncation in the other direction:
    // 1e-308 underflows to 0 in f32, so an f32 detour would surface
    // as 0 in the readback rather than the original subnormal.
    ensure_cuda();
    assert_filled_bit_exact(32, 1e-308_f64);
}

#[test]
fn fill_f64_n10_catches_stride_bug() {
    // Specific n=10, scalar=3.141592653589793: this is the canonical
    // case from the dispatch — reading back 10 filled f64 values
    // exposes the f32-stride byte-offset bug. With `shl.b64 ..., 2`
    // (4-byte stride) the kernel would write at half-spaced offsets,
    // so `host[1]` and `host[3]` etc. would read whatever the
    // alloc_zeros zeroing left there (i.e., 0.0), not π.
    ensure_cuda();
    let device = GpuDevice::new(0).expect("CUDA device 0");
    let scalar = std::f64::consts::PI;
    let buf = ferrotorch_gpu::kernels::gpu_fill_f64(10, scalar, &device).expect("gpu_fill_f64");
    let host = ferrotorch_gpu::transfer::gpu_to_cpu(&buf, &device).expect("gpu_to_cpu");
    assert_eq!(host.len(), 10);
    for (i, &v) in host.iter().enumerate() {
        // Use bit-exact comparison: stride bug would leave half the
        // elements as 0.0 (alloc_zeros baseline), which diverges from
        // π by every bit, so any non-bit-exact result fails this.
        assert_eq!(
            v.to_bits(),
            scalar.to_bits(),
            "stride-bug check: fill_f64(n=10, scalar=π)[{i}] = {v}, expected π. \
             A 4-byte stride leaves alternating elements as 0.0.",
        );
    }
}

#[test]
fn fill_f64_n0_returns_empty_buffer() {
    // n == 0: kernel must early-exit (skipping launch entirely). The
    // returned buffer must be zero-length and not error.
    ensure_cuda();
    let device = GpuDevice::new(0).expect("CUDA device 0");
    let buf = ferrotorch_gpu::kernels::gpu_fill_f64(0, std::f64::consts::PI, &device)
        .expect("gpu_fill_f64 n=0");
    assert_eq!(buf.len(), 0);
    let host = ferrotorch_gpu::transfer::gpu_to_cpu(&buf, &device).expect("gpu_to_cpu");
    assert!(host.is_empty());
}
