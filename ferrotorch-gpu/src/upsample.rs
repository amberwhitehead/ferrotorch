//! GPU nearest-neighbor 2x upsample (f32).
//!
//! Mirrors `torch.nn.functional.interpolate(..., mode="nearest",
//! scale_factor=2.0)` for input shape `[B, C, H, W]`:
//!
//! ```text
//! For each (b, c, oh, ow) in [B, C, 2H, 2W):
//!     out[b, c, oh, ow] = in[b, c, oh / 2, ow / 2]
//! ```
//!
//! # Why this is its own module
//!
//! The SD VAE up-blocks use NEAREST upsampling by factor 2 between
//! the resnet stacks. The CPU path goes via
//! `ferrotorch_nn::Upsample`; this module supplies the matching GPU
//! primitive so the GPU forward path is end-to-end on device.
//!
//! # Kernel layout
//!
//! - Grid: `((B*C*OH*OW + 255) / 256, 1, 1)`. One thread per output
//!   element; each thread reads one input element and writes one
//!   output element.
//! - Block: `(256, 1, 1)`.
//! - No shared memory.

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

/// PTX source for the nearest-2x-upsample forward kernel.
///
/// ABI: `(in_ptr, out_ptr, batch, channels, h_in, w_in, total)`
/// where `total = batch * channels * (2*h_in) * (2*w_in)`.
#[cfg(feature = "cuda")]
pub(crate) const NEAREST_UPSAMPLE2X_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry nearest_upsample2x_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 batch,
    .param .u32 channels,
    .param .u32 h_in,
    .param .u32 w_in,
    .param .u32 total
) {
    .reg .u32 %r_tid, %r_bid, %r_bdim, %idx, %out_idx, %total_r, %h_r, %w_r, %c_r;
    .reg .u32 %hi2, %wi2, %hw_out, %chw_out, %tmp;
    .reg .u32 %oh_t, %ow_t, %hi, %wi, %ci, %bi, %in_idx;
    .reg .u64 %in, %out, %off;
    .reg .f32 %val;
    .reg .pred %oob;

    ld.param.u64 %in,  [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %c_r, [channels];
    ld.param.u32 %h_r, [h_in];
    ld.param.u32 %w_r, [w_in];
    ld.param.u32 %total_r, [total];

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_bid, %ctaid.x;
    mov.u32 %r_bdim, %ntid.x;
    mad.lo.u32 %out_idx, %r_bid, %r_bdim, %r_tid;

    setp.ge.u32 %oob, %out_idx, %total_r;
    @%oob bra DONE;

    mov.u32 %idx, %out_idx;

    // 2H, 2W and per-spatial / per-channel strides over the *output*.
    shl.b32 %hi2, %h_r, 1;             // 2*H_in == H_out
    shl.b32 %wi2, %w_r, 1;             // 2*W_in == W_out
    mul.lo.u32 %hw_out, %hi2, %wi2;    // H_out * W_out
    mul.lo.u32 %chw_out, %c_r, %hw_out; // C * H_out * W_out

    // bi = idx / chw_out; idx %= chw_out
    div.u32 %bi, %idx, %chw_out;
    mul.lo.u32 %tmp, %bi, %chw_out;
    sub.u32 %idx, %idx, %tmp;

    // ci = idx / hw_out; idx %= hw_out
    div.u32 %ci, %idx, %hw_out;
    mul.lo.u32 %tmp, %ci, %hw_out;
    sub.u32 %idx, %idx, %tmp;

    // oh = idx / (2W); ow = idx % (2W)
    div.u32 %oh_t, %idx, %wi2;
    mul.lo.u32 %tmp, %oh_t, %wi2;
    sub.u32 %ow_t, %idx, %tmp;

    // hi = oh/2, wi = ow/2
    shr.u32 %hi, %oh_t, 1;
    shr.u32 %wi, %ow_t, 1;

    // in_idx = ((bi * C + ci) * H_in + hi) * W_in + wi
    mul.lo.u32 %in_idx, %bi, %c_r;
    add.u32 %in_idx, %in_idx, %ci;
    mul.lo.u32 %in_idx, %in_idx, %h_r;
    add.u32 %in_idx, %in_idx, %hi;
    mul.lo.u32 %in_idx, %in_idx, %w_r;
    add.u32 %in_idx, %in_idx, %wi;

    // Load in[in_idx]
    cvt.u64.u32 %off, %in_idx;
    shl.b64 %off, %off, 2;
    add.u64 %off, %in, %off;
    ld.global.f32 %val, [%off];

    // Store to out[out_idx]
    cvt.u64.u32 %off, %out_idx;
    shl.b64 %off, %off, 2;
    add.u64 %off, %out, %off;
    st.global.f32 [%off], %val;

DONE:
    ret;
}
";

/// GPU forward nearest-2x upsample on `[B, C, H, W]` f32 buffer.
///
/// Output shape is `[B, C, 2H, 2W]`. Each output pixel `(oh, ow)`
/// pulls the value from input pixel `(oh / 2, ow / 2)` (PyTorch
/// `mode="nearest"` semantics).
///
/// # Arguments
///
/// - `input` — `[B * C * H * W]` row-major f32.
/// - `batch`, `channels`, `h`, `w` — input dims.
/// - `device` — owning GPU device.
///
/// # Errors
///
/// - [`GpuError::ShapeMismatch`] when the buffer length disagrees
///   with declared dims.
/// - [`GpuError::DeviceMismatch`] when buffer is on a different
///   device.
/// - [`GpuError::PtxCompileFailed`] if the PTX module fails to compile.
/// - [`GpuError::Driver`] on launch failure.
#[cfg(feature = "cuda")]
pub fn gpu_nearest_upsample2x_f32(
    input: &CudaBuffer<f32>,
    batch: usize,
    channels: usize,
    h: usize,
    w: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    let in_len = batch * channels * h * w;
    if input.len() != in_len {
        return Err(GpuError::ShapeMismatch {
            op: "nearest_upsample2x",
            expected: vec![batch, channels, h, w],
            got: vec![input.len()],
        });
    }
    if input.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: device.ordinal(),
            got: input.device_ordinal(),
        });
    }

    let out_len = batch * channels * (h * 2) * (w * 2);
    if out_len == 0 {
        return alloc_zeros_f32(out_len, device);
    }

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        NEAREST_UPSAMPLE2X_PTX,
        "nearest_upsample2x_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "nearest_upsample2x_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(out_len, device)?;
    let batch_u32 = batch as u32;
    let channels_u32 = channels as u32;
    let h_u32 = h as u32;
    let w_u32 = w as u32;
    let total_u32 = out_len as u32;

    let block_dim: u32 = 256;
    let grid_x = total_u32.div_ceil(block_dim);
    let cfg = LaunchConfig {
        grid_dim: (grid_x.max(1), 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY:
    // - `f` is a valid `CudaFunction` for `nearest_upsample2x_kernel`
    //   returned by `module_cache::get_or_compile`; ABI
    //   `(in_ptr, out_ptr, batch, channels, h_in, w_in, total)`
    //   matches `NEAREST_UPSAMPLE2X_PTX`.
    // - `input.len() == batch * channels * h * w` and lives on
    //   `device` (validated above).
    // - `out` was just allocated to `batch * channels * 2h * 2w`
    //   and cannot alias `input` (Rust `&mut` borrow on `out`).
    // - Grid is sized so `grid_x * 256 >= total`; each thread
    //   first checks `idx < total` and exits early otherwise.
    // - The PTX kernel computes `(bi, ci, oh, ow)` from `idx` via
    //   `idx /= chw_out`, then `/= hw_out`, then `/= (2W)`; reads
    //   `in[((bi*C+ci)*H+hi)*W+wi]` with `hi=oh/2, wi=ow/2`. Both
    //   `(hi, wi)` are < `(H, W)` because `oh < 2H, ow < 2W`,
    //   placing every load inside `input` and every store inside
    //   `out`.
    // - `total_u32 = out_len as u32`: caller is responsible for
    //   keeping `out_len <= u32::MAX`. For SD VAE the largest is
    //   `1 * 512 * 512 * 512 = 134_217_728 < 2^32`.
    // - All `u32` params (`batch_u32` etc.) are passed by-reference;
    //   cudarc copies them into the launch parameter buffer.
    // - Stream sync is the caller's responsibility.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&batch_u32)
            .arg(&channels_u32)
            .arg(&h_u32)
            .arg(&w_u32)
            .arg(&total_u32)
            .launch(cfg)?;
    }

    Ok(out)
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::transfer::{cpu_to_gpu, gpu_to_cpu};

    fn cpu_nearest_upsample2x_ref(
        x: &[f32],
        b: usize,
        c: usize,
        h: usize,
        w: usize,
    ) -> Vec<f32> {
        let oh = 2 * h;
        let ow = 2 * w;
        let mut out = vec![0.0f32; b * c * oh * ow];
        for bi in 0..b {
            for ci in 0..c {
                for ohi in 0..oh {
                    for owi in 0..ow {
                        let hi = ohi / 2;
                        let wi = owi / 2;
                        let src = x[((bi * c + ci) * h + hi) * w + wi];
                        out[((bi * c + ci) * oh + ohi) * ow + owi] = src;
                    }
                }
            }
        }
        out
    }

    #[test]
    fn upsample_small_matches_cpu() {
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let b = 2;
        let c = 3;
        let h = 4;
        let w = 5;
        let n = b * c * h * w;
        let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.1 - 0.5).collect();
        let xg = cpu_to_gpu(&x, &device).unwrap();
        let yg = gpu_nearest_upsample2x_f32(&xg, b, c, h, w, &device).unwrap();
        let got = gpu_to_cpu(&yg, &device).unwrap();
        let expected = cpu_nearest_upsample2x_ref(&x, b, c, h, w);
        assert_eq!(got.len(), expected.len());
        for (a, e) in got.iter().zip(expected.iter()) {
            // Exact match -- nearest is a pure gather.
            assert_eq!(a, e);
        }
    }

    #[test]
    fn upsample_sd_vae_intermediate_shape() {
        // First SD VAE upsample: [1, 512, 64, 64] -> [1, 512, 128, 128].
        // We test a fraction of that to keep the test fast.
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let b = 1;
        let c = 32;
        let h = 16;
        let w = 16;
        let n = b * c * h * w;
        let x: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
        let xg = cpu_to_gpu(&x, &device).unwrap();
        let yg = gpu_nearest_upsample2x_f32(&xg, b, c, h, w, &device).unwrap();
        let got = gpu_to_cpu(&yg, &device).unwrap();
        let expected = cpu_nearest_upsample2x_ref(&x, b, c, h, w);
        assert_eq!(got.len(), b * c * 2 * h * 2 * w);
        for (a, e) in got.iter().zip(expected.iter()) {
            assert_eq!(a, e);
        }
    }

    #[test]
    fn upsample_validates_input_len() {
        let device = match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        };
        let x = vec![0.0f32; 10]; // claim shape [1, 2, 2, 2] = 8 (mismatch).
        let xg = cpu_to_gpu(&x, &device).unwrap();
        let res = gpu_nearest_upsample2x_f32(&xg, 1, 2, 2, 2, &device);
        assert!(matches!(res, Err(GpuError::ShapeMismatch { .. })));
    }
}
