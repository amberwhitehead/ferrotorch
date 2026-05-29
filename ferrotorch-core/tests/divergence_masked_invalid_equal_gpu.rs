//! Discriminator divergence pin (#1659): the GPU `masked_invalid` /
//! `masked_equal` paths in `ferrotorch-core/src/masked.rs` mis-size the boolean
//! predicate mask when the input data buffer is a POOLED, over-allocated CUDA
//! buffer (the `ROUND_ELEMENTS = 256` allocation granularity showing through).
//!
//! # Mechanism (verified by reading the source, not assumed)
//!
//! `masked_invalid` (`masked.rs:485`) and `masked_equal` (`masked.rs:520`) both
//! materialise their CUDA input with `data.contiguous()` (`masked.rs:493` /
//! `masked.rs:536`) before computing the predicate on-device. When the input is
//! a NON-contiguous CUDA view (e.g. a transpose — offset 0, so this is
//! INDEPENDENT of `storage_offset`), `contiguous()` routes through the GPU
//! `strided_copy_f32` fast path (`ferrotorch-core/src/methods.rs:1599`), whose
//! output is allocated by `alloc_zeros_f32(n)`
//! (`ferrotorch-gpu/src/kernels.rs:15982`). That allocator ROUNDS the cudarc
//! `CudaSlice` up to a multiple of `ROUND_ELEMENTS = 256`
//! (`ferrotorch-gpu/src/pool.rs:114`, `alloc_zeros_f32` at
//! `transfer.rs:86`), so a 6-element tensor sits in a 256-element slice
//! (`CudaBuffer.len` stays 6 but `CudaBuffer::inner()` exposes the raw
//! 256-element `CudaSlice`).
//!
//! The predicate kernel launcher reads the RAW cudarc slice length, not the
//! logical numel:
//!
//!   `ferrotorch-gpu/src/masked_kernels.rs:902`  (isfinite — `launch_predicate`)
//!   `ferrotorch-gpu/src/masked_kernels.rs:946`  (ne_scalar — `launch_predicate_scalar`)
//!       let n = input.len();          // <- cudarc CudaSlice::len() == 256, NOT 6
//!       let mut out = stream.alloc_zeros::<u8>(n)?;   // 256-byte mask
//!
//! `wrap_slice_bool` (`backend_impl.rs:378-388`) then records the bool buffer's
//! logical `len` as `slice.len()` == 256, and the readback
//! `predicate_mask_gpu` -> `gpu_to_cpu` -> `transfer::gpu_to_cpu` truncates to
//! `buffer.len()` == 256 (`transfer.rs:71`). The decoder at
//!
//!   ===> BUGGY DECODE: `ferrotorch-core/src/masked.rs:555-561` <===
//!        `predicate_mask_gpu`: `Ok(bytes.iter().map(|&b| b != 0).collect())`
//!
//! therefore returns a `Vec<bool>` of length 256, which `MaskedTensor::new`
//! (`masked.rs:78`) rejects: `mask length 256 != data numel 6`.
//!
//! # Precise fix direction (for the fixer — do NOT implement here)
//!
//! Truncate the decoded mask to the tensor's logical numel before constructing
//! the `MaskedTensor`. The cleanest target is `predicate_mask_gpu`
//! (`masked.rs:555`): take the expected `numel` as a parameter and return
//! `bytes[..numel]` (the pooled tail bytes are garbage/zero and must be
//! dropped), OR have the launcher pass the LOGICAL `n` (`input` handle's
//! `.len()`, i.e. `CudaBuffer::len()` == 6) into `launch_predicate` /
//! `launch_predicate_scalar` instead of the raw cudarc `CudaSlice::len()` —
//! mirroring `masked_fill_dt` (`backend_impl.rs:8054`: `let n = input.len();`),
//! which already passes the logical handle len for exactly this reason (#1660).
//!
//! # Oracle (R-CHAR-3)
//!
//! Expected masks are computed by the CPU `masked_invalid` / `masked_equal`
//! reference paths (`masked.rs:503-511` `f.is_finite()` walk and
//! `masked.rs:544-546` `v != value` walk) on the equivalent CPU transposed
//! tensor — the documented torch-convention reference, NOT copied from the GPU
//! side. `isfinite` mirrors `aten/src/ATen/native/TensorCompare.cpp:484`
//! (`(self == self) * (self.abs() != inf)`); `masked_equal`'s valid mask is the
//! `numpy.ma.masked_equal` complement `v != value`.
//!
//! Tracking: #1659. These tests are LEFT UNMARKED (not `#[ignore]`) because the
//! divergence is a hard error (the public constructor returns `Err` on a
//! perfectly valid CUDA input), and the failing test IS the release block.

#![cfg(feature = "gpu")]

use ferrotorch_core::masked::{masked_equal, masked_invalid};
use ferrotorch_core::{Device, Tensor, TensorStorage};
use std::sync::Once;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the #1659 GPU pooled-mask pin");
    });
}

fn cpu_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu f32 tensor")
}

fn cpu_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu f64 tensor")
}

// ── #1659 (A) masked_invalid on a pooled (non-contiguous) CUDA buffer ────────

/// A non-contiguous CUDA view (transpose of a [2,3] -> [3,2], offset 0)
/// materialises via `strided_copy_f32` into a POOLED 256-element buffer. The
/// isfinite predicate launcher reads that raw 256-element slice length, so the
/// decoded mask is 256 bytes while `data.numel()` is 6 — `MaskedTensor::new`
/// rejects it ("mask length 256 != data numel 6"). Upstream / CPU reference:
/// a 6-element finite/NaN/inf mask. INDEPENDENT of storage_offset.
#[test]
fn divergence_1659_masked_invalid_gpu_pooled_buffer_numel_mismatch() {
    ensure_cuda_backend();

    // [2,3] row-major; transpose -> [3,2] non-contiguous, offset 0, numel 6.
    let base = [1.0_f32, f32::NAN, 3.0, f32::INFINITY, 5.0, 6.0];

    // CPU oracle: build the SAME logical transposed tensor on CPU and run the
    // CPU masked_invalid reference (masked.rs:503-511, IEEE is_finite walk).
    let cpu_t = cpu_f32(&base, &[2, 3])
        .transpose(0, 1)
        .expect("cpu transpose");
    assert!(!cpu_t.is_contiguous(), "transpose must be non-contiguous");
    assert_eq!(cpu_t.numel(), 6);
    let cpu_mt = masked_invalid(cpu_t).expect("cpu masked_invalid reference");
    let expected: Vec<bool> = cpu_mt.mask().to_vec();
    // Logical transposed order is [1, inf, NaN, 5, 3, 6]; isfinite ->
    // [T, F, F, T, T, T].
    assert_eq!(
        expected,
        vec![true, false, false, true, true, true],
        "CPU reference sanity: transposed isfinite mask"
    );

    // GPU: same transposed view, uploaded to CUDA.
    let gpu_t = cpu_f32(&base, &[2, 3])
        .to(Device::Cuda(0))
        .expect("upload [2,3] to cuda")
        .transpose(0, 1)
        .expect("gpu transpose");
    assert!(gpu_t.is_cuda(), "input must be CUDA-resident");
    assert!(!gpu_t.is_contiguous());
    assert_eq!(gpu_t.numel(), 6);

    let mt = masked_invalid(gpu_t)
        .expect("masked_invalid on a pooled CUDA buffer must NOT error (#1659)");
    assert_eq!(
        mt.mask(),
        expected.as_slice(),
        "GPU masked_invalid mask must equal the 6-element CPU reference, \
         not the 256-element pooled-buffer readback (#1659)"
    );
}

// ── #1659 (B) masked_equal on a pooled (non-contiguous) CUDA buffer ──────────

/// Same pooled-buffer trigger for `masked_equal`. `ne_scalar_mask` ->
/// `launch_predicate_scalar` (`masked_kernels.rs:946`) reads the raw 256-element
/// slice length, producing a 256-byte mask vs. the 6-element data. CPU/numpy
/// reference (masked.rs:544-546, `v != value`): for value=5.0 over the
/// transposed logical order [1, inf, NaN, 5, 3, 6], the valid mask is
/// [T, T, T, F, T, T].
#[test]
fn divergence_1659_masked_equal_gpu_pooled_buffer_numel_mismatch() {
    ensure_cuda_backend();

    let base = [1.0_f32, f32::NAN, 3.0, f32::INFINITY, 5.0, 6.0];
    let value = 5.0_f32;

    let cpu_t = cpu_f32(&base, &[2, 3])
        .transpose(0, 1)
        .expect("cpu transpose");
    assert!(!cpu_t.is_contiguous());
    let cpu_mt = masked_equal(cpu_t, value).expect("cpu masked_equal reference");
    let expected: Vec<bool> = cpu_mt.mask().to_vec();
    assert_eq!(
        expected,
        vec![true, true, true, false, true, true],
        "CPU reference sanity: NaN != 5.0 valid; only the 5.0 entry masked"
    );

    let gpu_t = cpu_f32(&base, &[2, 3])
        .to(Device::Cuda(0))
        .expect("upload [2,3] to cuda")
        .transpose(0, 1)
        .expect("gpu transpose");
    assert!(gpu_t.is_cuda());
    assert!(!gpu_t.is_contiguous());
    assert_eq!(gpu_t.numel(), 6);

    let mt = masked_equal(gpu_t, value)
        .expect("masked_equal on a pooled CUDA buffer must NOT error (#1659)");
    assert_eq!(
        mt.mask(),
        expected.as_slice(),
        "GPU masked_equal mask must equal the 6-element CPU reference, \
         not the 256-element pooled-buffer readback (#1659)"
    );
}

// ── #1659 (C) f64 masked_invalid — confirm the bug spans both widths ─────────

/// f64 path: `isfinite_mask_f64` shares the same `launch_predicate` site
/// (`masked_kernels.rs:902`) and the same pooled-buffer over-read, so the
/// divergence is width-independent. CPU reference: 6-element is_finite walk.
#[test]
fn divergence_1659_masked_invalid_f64_gpu_pooled_buffer_numel_mismatch() {
    ensure_cuda_backend();

    let base = [1.0_f64, f64::NAN, 3.0, f64::NEG_INFINITY, 5.0, 6.0];

    let cpu_t = cpu_f64(&base, &[2, 3])
        .transpose(0, 1)
        .expect("cpu transpose");
    let cpu_mt = masked_invalid(cpu_t).expect("cpu masked_invalid f64 reference");
    let expected: Vec<bool> = cpu_mt.mask().to_vec();
    assert_eq!(
        expected,
        vec![true, false, false, true, true, true],
        "CPU reference sanity: transposed f64 isfinite mask"
    );

    let gpu_t = cpu_f64(&base, &[2, 3])
        .to(Device::Cuda(0))
        .expect("upload f64 [2,3] to cuda")
        .transpose(0, 1)
        .expect("gpu transpose");
    assert!(gpu_t.is_cuda());
    assert_eq!(gpu_t.numel(), 6);

    let mt = masked_invalid(gpu_t)
        .expect("f64 masked_invalid on a pooled CUDA buffer must NOT error (#1659)");
    assert_eq!(
        mt.mask(),
        expected.as_slice(),
        "GPU f64 masked_invalid mask must equal the 6-element CPU reference (#1659)"
    );
}
