//! GPU forward kernel for `roll` (cyclic shift along a single axis), f32.
//!
//! Mirrors `torch.roll(input, shifts, dim)` for a single dimension:
//!
//! ```text
//! For every (o, k_new, i) in [0, outer) x [0, dim_size) x [0, inner):
//!     k_src = (k_new - shift_norm + dim_size) % dim_size
//!     out[(o*dim_size + k_new)*inner + i] = in[(o*dim_size + k_src)*inner + i]
//! ```
//!
//! `shift_norm` is the caller-normalised non-negative shift
//! (`0 <= shift_norm < dim_size`). The `(outer, dim_size, inner)` factorisation
//! matches the CPU `roll_cpu_inner` shared helper in
//! `ferrotorch_core::ops::tensor_ops`, so the two paths address memory the
//! same way and remain bit-exact.
//!
//! # Why a single-axis kernel
//!
//! The public `roll<T>` API in `ferrotorch-core` is single-axis
//! (`shifts: i64, dim: usize`). Multi-axis rolls are expressed as repeated
//! single-axis calls. The kernel matches that surface 1:1 so the dispatch
//! shim in `ops/tensor_ops::roll` remains a thin GPU fast-path branch.
//!
//! # Kernel layout
//!
//! - Grid: `((numel + 255) / 256, 1, 1)`. One thread per output element.
//! - Block: `(256, 1, 1)`.
//! - No shared memory.
//!
//! `RollBackward` is also a roll (with `-shifts` in place of `shifts`); the
//! grad path in `ferrotorch_core::grad_fns::shape::RollBackward` dispatches
//! back through this same kernel via the `GpuBackend::roll_f32` trait method.

#[cfg(feature = "cuda")]
use cudarc::driver::{LaunchConfig, PushKernelArg};

#[cfg(feature = "cuda")]
use crate::buffer::CudaBuffer;
#[cfg(feature = "cuda")]
use crate::device::GpuDevice;
#[cfg(feature = "cuda")]
use crate::error::{GpuError, GpuResult};
#[cfg(feature = "cuda")]
use crate::transfer::alloc_zeros_f32;

/// PTX source for the f32 roll forward kernel.
///
/// ABI: `(in_ptr, out_ptr, outer, dim_size, inner, shift_norm, total)`
/// where `total = outer * dim_size * inner` and
/// `0 <= shift_norm < dim_size`.
///
/// Each thread:
///   1. Computes its output index `out_idx` from the launch grid.
///   2. Decomposes `out_idx` into `(o, k_new, i)` via
///      `o = out_idx / (dim_size * inner)`,
///      `k_new = (out_idx / inner) % dim_size`,
///      `i = out_idx % inner`.
///   3. Computes `k_src = (k_new + dim_size - shift_norm) % dim_size`
///      (the forward roll's inverse address map).
///   4. Reads `in[(o*dim_size + k_src)*inner + i]` and writes to
///      `out[out_idx]`.
#[cfg(feature = "cuda")]
pub(crate) const ROLL_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry roll_f32_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 outer,
    .param .u32 dim_size,
    .param .u32 inner,
    .param .u32 shift_norm,
    .param .u32 total
) {
    .reg .u32 %tid_r, %bid_r, %bdim_r, %out_idx, %total_r;
    .reg .u32 %outer_r, %dim_r, %inner_r, %shift_r;
    .reg .u32 %dim_inner, %o, %tmp, %rem, %k_new, %i_idx;
    .reg .u32 %k_src, %src_idx, %sum;
    .reg .u64 %in_p, %out_p, %off, %addr;
    .reg .f32 %val;
    .reg .pred %p_oob;

    ld.param.u64 %in_p,    [in_ptr];
    ld.param.u64 %out_p,   [out_ptr];
    ld.param.u32 %outer_r, [outer];
    ld.param.u32 %dim_r,   [dim_size];
    ld.param.u32 %inner_r, [inner];
    ld.param.u32 %shift_r, [shift_norm];
    ld.param.u32 %total_r, [total];

    mov.u32 %tid_r,  %tid.x;
    mov.u32 %bid_r,  %ctaid.x;
    mov.u32 %bdim_r, %ntid.x;
    mad.lo.u32 %out_idx, %bid_r, %bdim_r, %tid_r;

    setp.ge.u32 %p_oob, %out_idx, %total_r;
    @%p_oob bra DONE;

    // dim_inner = dim_size * inner
    mul.lo.u32 %dim_inner, %dim_r, %inner_r;

    // o = out_idx / dim_inner
    div.u32 %o, %out_idx, %dim_inner;
    mul.lo.u32 %tmp, %o, %dim_inner;
    sub.u32 %rem, %out_idx, %tmp;          // rem = out_idx % dim_inner

    // k_new = rem / inner
    div.u32 %k_new, %rem, %inner_r;
    mul.lo.u32 %tmp, %k_new, %inner_r;
    sub.u32 %i_idx, %rem, %tmp;            // i_idx = rem % inner

    // k_src = (k_new + dim_size - shift_norm) % dim_size
    // dim_size - shift_norm is computed as an unsigned subtraction; the
    // caller guarantees 0 <= shift_norm < dim_size (or dim_size == 0, which
    // the caller short-circuits before launching this kernel).
    sub.u32 %tmp, %dim_r, %shift_r;        // tmp = dim_size - shift_norm
    add.u32 %sum, %k_new, %tmp;            // sum = k_new + dim_size - shift_norm
    // sum < 2*dim_size, so a single conditional subtract suffices, but using
    // `rem.u32` keeps the code defensive against any future caller passing
    // shift_norm == dim_size (treated as 0, no underflow either way).
    rem.u32 %k_src, %sum, %dim_r;

    // src_idx = (o * dim_size + k_src) * inner + i_idx
    mul.lo.u32 %src_idx, %o, %dim_r;
    add.u32 %src_idx, %src_idx, %k_src;
    mul.lo.u32 %src_idx, %src_idx, %inner_r;
    add.u32 %src_idx, %src_idx, %i_idx;

    // load in[src_idx]
    cvt.u64.u32 %off, %src_idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in_p, %off;
    ld.global.f32 %val, [%addr];

    // store out[out_idx]
    cvt.u64.u32 %off, %out_idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %out_p, %off;
    st.global.f32 [%addr], %val;

DONE:
    ret;
}
";

/// GPU forward roll on a contiguous f32 buffer factored as
/// `outer * dim_size * inner`.
///
/// Performs `out[(o*dim_size + k_new)*inner + i] =
///   in[(o*dim_size + ((k_new - shift_norm + dim_size) mod dim_size))*inner + i]`
/// for every `(o, k_new, i)` in the index space. Equivalent to
/// `torch.roll(input.reshape(outer, dim_size, inner), shifts, dim=1)
/// .reshape(input.shape)` with `shifts = shift_norm` (post-normalisation).
///
/// # Arguments
///
/// - `input` — `outer * dim_size * inner` f32 elements on `device`.
/// - `outer` — product of dims before the roll axis.
/// - `dim_size` — size of the roll axis. Must satisfy `dim_size > 0` (callers
///   handle the empty-axis case before invoking this).
/// - `inner` — product of dims after the roll axis.
/// - `shift_norm` — the already-normalised non-negative shift, with
///   `0 <= shift_norm < dim_size`. The CPU shim normalises before calling.
/// - `device` — owning GPU device.
///
/// # Errors
///
/// - [`GpuError::ShapeMismatch`] when `input.len() != outer * dim_size * inner`
///   or when any of `dim_size`, `shift_norm` violates its precondition.
/// - [`GpuError::DeviceMismatch`] when the buffer is on a different device.
/// - [`GpuError::PtxCompileFailed`] if the PTX module fails to compile.
/// - [`GpuError::Driver`] on launch failure.
#[cfg(feature = "cuda")]
pub fn gpu_roll_f32(
    input: &CudaBuffer<f32>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    shift_norm: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    if input.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: device.ordinal(),
            got: input.device_ordinal(),
        });
    }

    let expected_len = outer.saturating_mul(dim_size).saturating_mul(inner);
    if input.len() != expected_len {
        return Err(GpuError::ShapeMismatch {
            op: "roll_f32",
            expected: vec![outer, dim_size, inner],
            got: vec![input.len()],
        });
    }

    if dim_size == 0 {
        return Err(GpuError::ShapeMismatch {
            op: "roll_f32",
            expected: vec![outer, 1, inner],
            got: vec![outer, 0, inner],
        });
    }
    if shift_norm >= dim_size {
        return Err(GpuError::ShapeMismatch {
            op: "roll_f32",
            expected: vec![dim_size.saturating_sub(1)],
            got: vec![shift_norm],
        });
    }

    let total = expected_len;
    if total == 0 {
        // outer * inner == 0 — empty tensor on a non-empty roll axis.
        // No threads to launch; just hand back an empty buffer.
        return alloc_zeros_f32(0, device);
    }
    if total > u32::MAX as usize {
        return Err(GpuError::ShapeMismatch {
            op: "roll_f32",
            expected: vec![u32::MAX as usize],
            got: vec![total],
        });
    }

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        ROLL_F32_PTX,
        "roll_f32_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "roll_f32_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(total, device)?;

    let outer_u32 = outer as u32;
    let dim_size_u32 = dim_size as u32;
    let inner_u32 = inner as u32;
    let shift_norm_u32 = shift_norm as u32;
    let total_u32 = total as u32;

    let block_dim: u32 = 256;
    let grid_x = total_u32.div_ceil(block_dim);
    let cfg = LaunchConfig {
        grid_dim: (grid_x.max(1), 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY:
    // - `f` is the `CudaFunction` for `roll_f32_kernel` resolved by
    //   `module_cache::get_or_compile(ctx, ROLL_F32_PTX, ...)`; the launch
    //   ABI `(in, out, outer, dim_size, inner, shift_norm, total)` matches
    //   the PTX `.entry` signature above one-for-one.
    // - `input` lives on `device` (validated at the top of this fn) and has
    //   exactly `total = outer * dim_size * inner` f32 elements (validated
    //   against `expected_len`).
    // - `out` was just allocated to `total` f32 elements by `alloc_zeros_f32`;
    //   it cannot alias `input` because cudarc returns a fresh `CudaSlice`
    //   and Rust's borrow checker keeps the `&mut` borrow on `out` exclusive.
    // - Grid is sized so every thread either has `out_idx < total` or exits
    //   early via the `setp.ge.u32` predicate at the top of the PTX.
    // - For each in-bounds thread the kernel computes
    //   `(o, k_new, i_idx)` such that `0 <= o < outer`,
    //   `0 <= k_new < dim_size`, `0 <= i_idx < inner`, then
    //   `k_src = (k_new + dim_size - shift_norm) mod dim_size`
    //   which is in `[0, dim_size)` because `shift_norm < dim_size`.
    //   Therefore `src_idx = (o*dim_size + k_src)*inner + i_idx` is in
    //   `[0, total)`, placing every load inside `input` and every store
    //   inside `out`.
    // - `outer * dim_size * inner` is range-checked above against
    //   `u32::MAX`, so the kernel's u32 index arithmetic cannot overflow.
    // - cudarc copies the by-reference `u32` params into the launch
    //   parameter buffer; their lifetime is tied to this stack frame
    //   which outlives the synchronous `launch` call.
    // - Stream sync is the caller's responsibility (matches the rest of
    //   the kernel module, e.g. `gpu_cumsum`).
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&outer_u32)
            .arg(&dim_size_u32)
            .arg(&inner_u32)
            .arg(&shift_norm_u32)
            .arg(&total_u32)
            .launch(cfg)?;
    }

    Ok(out)
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::transfer::{cpu_to_gpu, gpu_to_cpu};

    /// CPU reference: factorised single-axis roll that mirrors the kernel's
    /// index map exactly.
    fn cpu_roll_ref(
        data: &[f32],
        outer: usize,
        dim_size: usize,
        inner: usize,
        shift_norm: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; data.len()];
        for o in 0..outer {
            for k_new in 0..dim_size {
                let k_src = (k_new + dim_size - shift_norm) % dim_size;
                for i in 0..inner {
                    let src = (o * dim_size + k_src) * inner + i;
                    let dst = (o * dim_size + k_new) * inner + i;
                    out[dst] = data[src];
                }
            }
        }
        out
    }

    #[test]
    fn roll_1d_positive_shift_matches_cpu() {
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        // 1-D length-8 tensor, shift by 3.
        let outer = 1;
        let dim_size = 8;
        let inner = 1;
        let shift = 3;
        let x: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let xg = cpu_to_gpu(&x, &device).unwrap();
        let yg = gpu_roll_f32(&xg, outer, dim_size, inner, shift, &device).unwrap();
        let got = gpu_to_cpu(&yg, &device).unwrap();
        let expected = cpu_roll_ref(&x, outer, dim_size, inner, shift);
        // shift=3 maps [0,1,2,3,4,5,6,7] -> [5,6,7,0,1,2,3,4]
        assert_eq!(expected, vec![5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(got, expected);
    }

    #[test]
    fn roll_2d_inner_axis_matches_cpu() {
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        // 2-D [3, 5] tensor, roll along the inner axis by 2.
        // Factorisation: outer=3, dim_size=5, inner=1.
        let outer = 3;
        let dim_size = 5;
        let inner = 1;
        let shift = 2;
        let x: Vec<f32> = (0..15).map(|i| i as f32).collect();
        let xg = cpu_to_gpu(&x, &device).unwrap();
        let yg = gpu_roll_f32(&xg, outer, dim_size, inner, shift, &device).unwrap();
        let got = gpu_to_cpu(&yg, &device).unwrap();
        let expected = cpu_roll_ref(&x, outer, dim_size, inner, shift);
        assert_eq!(got, expected);
    }

    #[test]
    fn roll_2d_outer_axis_matches_cpu() {
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        // 2-D [4, 3] tensor, roll along the OUTER axis by 1.
        // Factorisation: outer=1, dim_size=4, inner=3.
        let outer = 1;
        let dim_size = 4;
        let inner = 3;
        let shift = 1;
        let x: Vec<f32> = (0..12).map(|i| (i as f32) * 0.5 - 1.0).collect();
        let xg = cpu_to_gpu(&x, &device).unwrap();
        let yg = gpu_roll_f32(&xg, outer, dim_size, inner, shift, &device).unwrap();
        let got = gpu_to_cpu(&yg, &device).unwrap();
        let expected = cpu_roll_ref(&x, outer, dim_size, inner, shift);
        assert_eq!(got, expected);
    }

    #[test]
    fn roll_negative_shift_via_normalization_matches_cpu() {
        // The kernel sees the already-normalised non-negative shift; the
        // caller normalises a negative `shifts` via
        // `((shifts % n) + n) % n`. We replicate that here.
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let outer = 1;
        let dim_size = 6;
        let inner = 1;
        let raw_shift: i64 = -1;
        let dim_i64 = dim_size as i64;
        let shift_norm = (((raw_shift % dim_i64) + dim_i64) % dim_i64) as usize;
        assert_eq!(shift_norm, 5); // -1 ≡ 5 mod 6
        let x: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let xg = cpu_to_gpu(&x, &device).unwrap();
        let yg = gpu_roll_f32(&xg, outer, dim_size, inner, shift_norm, &device).unwrap();
        let got = gpu_to_cpu(&yg, &device).unwrap();
        // roll([0..6], -1) -> [1, 2, 3, 4, 5, 0]
        assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.0]);
    }

    #[test]
    fn roll_3d_middle_axis_then_inner_axis_matches_cpu_composed() {
        // Multi-axis roll = sequence of single-axis rolls.
        // Shape [2, 4, 3]. Roll axis=1 by 1, then roll axis=2 by 2.
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let shape = [2usize, 4, 3];
        let x: Vec<f32> = (0..(shape[0] * shape[1] * shape[2]))
            .map(|i| (i as f32).sin())
            .collect();
        // First roll: axis 1, outer=2, dim_size=4, inner=3, shift=1.
        let xg = cpu_to_gpu(&x, &device).unwrap();
        let yg1 = gpu_roll_f32(&xg, 2, 4, 3, 1, &device).unwrap();
        let y1 = gpu_to_cpu(&yg1, &device).unwrap();
        let expected1 = cpu_roll_ref(&x, 2, 4, 3, 1);
        assert_eq!(y1, expected1);

        // Second roll on top: axis 2, outer = 2*4 = 8, dim_size=3, inner=1, shift=2.
        let yg2 = gpu_roll_f32(&yg1, 8, 3, 1, 2, &device).unwrap();
        let y2 = gpu_to_cpu(&yg2, &device).unwrap();
        let expected2 = cpu_roll_ref(&expected1, 8, 3, 1, 2);
        assert_eq!(y2, expected2);
    }

    #[test]
    fn roll_zero_shift_is_identity() {
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let outer = 2;
        let dim_size = 5;
        let inner = 4;
        let n = outer * dim_size * inner;
        let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.25).collect();
        let xg = cpu_to_gpu(&x, &device).unwrap();
        let yg = gpu_roll_f32(&xg, outer, dim_size, inner, 0, &device).unwrap();
        let got = gpu_to_cpu(&yg, &device).unwrap();
        assert_eq!(got, x);
    }

    #[test]
    fn roll_rejects_shift_at_dim_size() {
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let x = vec![0.0f32; 8];
        let xg = cpu_to_gpu(&x, &device).unwrap();
        // shift_norm == dim_size is a precondition violation; the wrapper
        // should reject it rather than silently wrap.
        let err = gpu_roll_f32(&xg, 1, 8, 1, 8, &device);
        assert!(matches!(err, Err(GpuError::ShapeMismatch { .. })));
    }

    #[test]
    fn roll_rejects_wrong_length() {
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        // Claim shape outer=2, dim_size=3, inner=2 (12 elems) but pass 10.
        let x = vec![0.0f32; 10];
        let xg = cpu_to_gpu(&x, &device).unwrap();
        let err = gpu_roll_f32(&xg, 2, 3, 2, 1, &device);
        assert!(matches!(err, Err(GpuError::ShapeMismatch { .. })));
    }
}
