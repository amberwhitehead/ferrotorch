//! GPU-accelerated 2-D convolution via im2col + cuBLAS GEMM.
//!
//! Conv2d is decomposed into two steps:
//!
//! 1. **im2col** -- rearrange each input receptive-field patch into a column
//!    of a matrix.  Runs entirely on the GPU via a custom PTX kernel.
//! 2. **GEMM** -- matrix-multiply the reshaped weight matrix with the column
//!    matrix using cuBLAS SGEMM on the GPU.
//!
//! The entire conv2d pipeline stays on-device.  Zero CPU roundtrips are
//! performed during the forward pass.
//!
//! # Layout
//!
//! All tensors use row-major (NCHW) layout, matching PyTorch's default:
//!
//! - **input**: `[B, C_in, H, W]`
//! - **weight**: `[C_out, C_in, kH, kW]`
//! - **bias**: `[C_out]`
//! - **output**: `[B, C_out, H_out, W_out]`
//!
//! # CPU fallback
//!
//! When the `cuda` feature is disabled, all functions return
//! `GpuError::NoCudaFeature`.

use crate::blas::gpu_matmul_f32;
use crate::buffer::CudaBuffer;
use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};

// ---------------------------------------------------------------------------
// CPU im2col (kept for testing / reference)
// ---------------------------------------------------------------------------

/// Extract image patches into columns on the CPU.
///
/// Given a 4-D input `[B, C, H, W]` (flattened row-major), produces a
/// flattened 3-D output `[B, C*kH*kW, H_out*W_out]` where each column
/// is one flattened receptive-field patch.
///
/// Returns `(columns, col_rows, col_cols)` where:
/// - `col_rows = C_in * kH * kW`
/// - `col_cols = H_out * W_out`
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn im2col_cpu(
    input: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> (Vec<f32>, usize, usize) {
    let h_out = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    let w_out = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    let col_rows = channels * kernel_h * kernel_w;
    let col_cols = h_out * w_out;

    let mut cols = vec![0.0f32; batch * col_rows * col_cols];

    // Cache-friendly im2col: iterate in (batch, c, kh, kw, oh, ow) order.
    // The inner loop writes to sequential col positions AND reads from
    // spatially-adjacent input positions (consecutive ow → consecutive iw).
    // For the non-padded interior region, we skip the per-element bounds
    // check entirely.
    let chw = channels * height * width;
    let hw = height * width;

    for b in 0..batch {
        let input_b = b * chw;
        let col_b = b * col_rows * col_cols;

        for c in 0..channels {
            let input_c = input_b + c * hw;

            for kh in 0..kernel_h {
                for kw in 0..kernel_w {
                    let row = c * kernel_h * kernel_w + kh * kernel_w + kw;
                    let row_off = col_b + row * col_cols;

                    // Compute the safe interior range where no padding check is needed.
                    let oh_start = if kh < pad_h {
                        (pad_h - kh).div_ceil(stride_h)
                    } else {
                        0
                    };
                    let oh_end =
                        ((height + pad_h).saturating_sub(kh)).min(h_out * stride_h) / stride_h;
                    let oh_end = oh_end.min(h_out);

                    let ow_start = if kw < pad_w {
                        (pad_w - kw).div_ceil(stride_w)
                    } else {
                        0
                    };
                    let ow_end =
                        ((width + pad_w).saturating_sub(kw)).min(w_out * stride_w) / stride_w;
                    let ow_end = ow_end.min(w_out);

                    // Padded rows before interior.
                    for oh in 0..oh_start {
                        let ih = oh * stride_h + kh;
                        for ow in 0..w_out {
                            let iw = ow * stride_w + kw;
                            let col = oh * w_out + ow;
                            cols[row_off + col] = if ih >= pad_h
                                && iw >= pad_w
                                && (ih - pad_h) < height
                                && (iw - pad_w) < width
                            {
                                input[input_c + (ih - pad_h) * width + (iw - pad_w)]
                            } else {
                                0.0
                            };
                        }
                    }

                    // Interior rows — no bounds check needed for the interior columns.
                    for oh in oh_start..oh_end {
                        let real_h = oh * stride_h + kh - pad_h;
                        let input_row = input_c + real_h * width;

                        // Padded columns before interior.
                        for ow in 0..ow_start {
                            let iw = ow * stride_w + kw;
                            let col = oh * w_out + ow;
                            cols[row_off + col] = if iw >= pad_w && (iw - pad_w) < width {
                                input[input_row + (iw - pad_w)]
                            } else {
                                0.0
                            };
                        }

                        // Fast interior: sequential reads from input, sequential writes to cols.
                        for ow in ow_start..ow_end {
                            let real_w = ow * stride_w + kw - pad_w;
                            cols[row_off + oh * w_out + ow] = input[input_row + real_w];
                        }

                        // Padded columns after interior.
                        for ow in ow_end..w_out {
                            let iw = ow * stride_w + kw;
                            let col = oh * w_out + ow;
                            cols[row_off + col] = if iw >= pad_w && (iw - pad_w) < width {
                                input[input_row + (iw - pad_w)]
                            } else {
                                0.0
                            };
                        }
                    }

                    // Padded rows after interior.
                    for oh in oh_end..h_out {
                        let ih = oh * stride_h + kh;
                        for ow in 0..w_out {
                            let iw = ow * stride_w + kw;
                            let col = oh * w_out + ow;
                            cols[row_off + col] = if ih >= pad_h
                                && iw >= pad_w
                                && (ih - pad_h) < height
                                && (iw - pad_w) < width
                            {
                                input[input_c + (ih - pad_h) * width + (iw - pad_w)]
                            } else {
                                0.0
                            };
                        }
                    }
                }
            }
        }
    }

    (cols, col_rows, col_cols)
}

// ---------------------------------------------------------------------------
// PTX kernel source strings
// ---------------------------------------------------------------------------

/// PTX kernel for im2col with dilation + group support.
///
/// Each thread computes ONE element of the column matrix
/// `[col_rows, col_cols]` for a single batch element and a single group.
/// The caller passes `channels` = channels-per-group (`C_in / groups`)
/// and `channel_offset` = `g * channels-per-group` so the kernel can read
/// from the correct contiguous channel slab in the original `[B, C_in, H, W]`
/// input. With `groups == 1` and `dilation == (1, 1)` this matches the
/// original pre-Pass-2A im2col.
///
/// Global thread index maps to `(row, col)`:
///
/// - `row = tid / col_cols`
/// - `col = tid % col_cols`
///
/// From `(row, col)` we derive the input coordinate:
///
/// - `c  = row / (kH * kW)`              (channel within the group)
/// - `kh = (row / kW) % kH`
/// - `kw = row % kW`
/// - `oh = col / W_out`
/// - `ow = col % W_out`
/// - `ih = oh * stride_h + kh * dil_h - pad_h`
/// - `iw = ow * stride_w + kw * dil_w - pad_w`
///
/// The input read uses channel `channel_offset + c` so the kernel sees the
/// per-group slice of the dense `[B, C_in, H, W]` input directly without
/// a copy.
///
/// If `(ih, iw)` is in-bounds, we read from the input; otherwise we write 0.
#[cfg(feature = "cuda")]
const IM2COL_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry im2col_kernel(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u32 batch_idx,
    .param .u32 c_in_total,
    .param .u32 channel_offset,
    .param .u32 channels,
    .param .u32 height,
    .param .u32 width,
    .param .u32 kernel_h,
    .param .u32 kernel_w,
    .param .u32 stride_h,
    .param .u32 stride_w,
    .param .u32 pad_h,
    .param .u32 pad_w,
    .param .u32 dil_h,
    .param .u32 dil_w,
    .param .u32 h_out,
    .param .u32 w_out,
    .param .u32 col_rows,
    .param .u32 col_cols
) {
    .reg .u32 %gtid, %bid, %bdim, %n_total;
    .reg .u32 %batch, %CTOT, %COFF, %C, %H, %W, %kH, %kW, %sH, %sW, %pH, %pW;
    .reg .u32 %dH, %dW, %hO, %wO;
    .reg .u32 %cR, %cC;
    .reg .u32 %row, %col, %c, %kh, %kw, %oh, %ow;
    .reg .u32 %c_global;
    .reg .u32 %kHkW, %HW, %CHW;
    .reg .u32 %ih_raw, %iw_raw, %ih, %iw;
    .reg .u32 %kh_d, %kw_d;
    .reg .u32 %t0;
    .reg .u64 %inp, %outp, %off64;
    .reg .f32 %val, %zero;
    .reg .pred %p_ge_n;
    .reg .pred %p_ih_ge0, %p_iw_ge0, %p_ih_lt_H, %p_iw_lt_W, %p_in_bounds;

    // Load parameters
    ld.param.u64 %inp,   [input_ptr];
    ld.param.u64 %outp,  [output_ptr];
    ld.param.u32 %batch, [batch_idx];
    ld.param.u32 %CTOT,  [c_in_total];
    ld.param.u32 %COFF,  [channel_offset];
    ld.param.u32 %C,     [channels];
    ld.param.u32 %H,     [height];
    ld.param.u32 %W,     [width];
    ld.param.u32 %kH,    [kernel_h];
    ld.param.u32 %kW,    [kernel_w];
    ld.param.u32 %sH,    [stride_h];
    ld.param.u32 %sW,    [stride_w];
    ld.param.u32 %pH,    [pad_h];
    ld.param.u32 %pW,    [pad_w];
    ld.param.u32 %dH,    [dil_h];
    ld.param.u32 %dW,    [dil_w];
    ld.param.u32 %hO,    [h_out];
    ld.param.u32 %wO,    [w_out];
    ld.param.u32 %cR,    [col_rows];
    ld.param.u32 %cC,    [col_cols];

    // Global thread ID
    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %gtid, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %gtid;

    // Total elements = col_rows * col_cols
    mul.lo.u32 %n_total, %cR, %cC;
    setp.ge.u32 %p_ge_n, %gtid, %n_total;
    @%p_ge_n bra DONE;

    // row = gtid / col_cols,  col = gtid % col_cols
    div.u32 %row, %gtid, %cC;
    rem.u32 %col, %gtid, %cC;

    // kHkW = kH * kW
    mul.lo.u32 %kHkW, %kH, %kW;

    // c  = row / (kH * kW)   (channel within the group)
    div.u32 %c, %row, %kHkW;

    // kh = (row / kW) % kH
    div.u32 %t0, %row, %kW;
    rem.u32 %kh, %t0, %kH;

    // kw = row % kW
    rem.u32 %kw, %row, %kW;

    // oh = col / W_out,  ow = col % W_out
    div.u32 %oh, %col, %wO;
    rem.u32 %ow, %col, %wO;

    // Apply dilation to the kernel offsets:
    // kh_d = kh * dil_h, kw_d = kw * dil_w
    mul.lo.u32 %kh_d, %kh, %dH;
    mul.lo.u32 %kw_d, %kw, %dW;

    // ih_raw = oh * stride_h + kh_d   (before subtracting pad)
    mad.lo.u32 %ih_raw, %oh, %sH, %kh_d;
    // iw_raw = ow * stride_w + kw_d
    mad.lo.u32 %iw_raw, %ow, %sW, %kw_d;

    // Check bounds: ih_raw >= pad_h  &&  iw_raw >= pad_w
    //               (ih_raw - pad_h) < H  &&  (iw_raw - pad_w) < W
    setp.ge.u32 %p_ih_ge0, %ih_raw, %pH;
    setp.ge.u32 %p_iw_ge0, %iw_raw, %pW;

    // ih = ih_raw - pad_h  (might underflow if ih_raw < pad_h, but we guard)
    sub.u32 %ih, %ih_raw, %pH;
    sub.u32 %iw, %iw_raw, %pW;

    setp.lt.u32 %p_ih_lt_H, %ih, %H;
    setp.lt.u32 %p_iw_lt_W, %iw, %W;

    // Combine: all four conditions must hold
    and.pred %p_in_bounds, %p_ih_ge0, %p_iw_ge0;
    and.pred %p_in_bounds, %p_in_bounds, %p_ih_lt_H;
    and.pred %p_in_bounds, %p_in_bounds, %p_iw_lt_W;

    mov.f32 %zero, 0f00000000;
    mov.f32 %val, %zero;

    @!%p_in_bounds bra WRITE_OUT;

    // c_global = channel_offset + c (channel into the dense [B, C_in, H, W] input)
    add.u32 %c_global, %COFF, %c;

    // Read input[batch * C_in_total*H*W + c_global * H*W + ih * W + iw]
    mul.lo.u32 %HW, %H, %W;
    mul.lo.u32 %CHW, %CTOT, %HW;
    // offset = batch*CHW + c_global*HW + ih*W + iw
    mad.lo.u32 %t0, %batch, %CHW, %iw;
    mad.lo.u32 %t0, %c_global, %HW, %t0;
    mad.lo.u32 %t0, %ih, %W, %t0;

    // Byte offset (f32 = 4 bytes)
    cvt.u64.u32 %off64, %t0;
    shl.b64 %off64, %off64, 2;
    add.u64 %inp, %inp, %off64;
    ld.global.f32 %val, [%inp];

WRITE_OUT:
    // Write to output[row * col_cols + col] = output[gtid]
    cvt.u64.u32 %off64, %gtid;
    shl.b64 %off64, %off64, 2;
    add.u64 %outp, %outp, %off64;
    st.global.f32 [%outp], %val;

DONE:
    ret;
}
";

/// PTX kernel for bias addition: `output[i] += bias[c]` where
/// `c = i / spatial_size` (integer division).
///
/// The output is `[C_out, spatial_size]` in row-major order.  Each thread
/// handles one element.
#[cfg(feature = "cuda")]
const BIAS_ADD_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry bias_add_kernel(
    .param .u64 output_ptr,
    .param .u64 bias_ptr,
    .param .u32 spatial_size,
    .param .u32 n
) {
    .reg .u32 %gid, %bid, %bdim, %n_reg, %sp, %c;
    .reg .u64 %out, %bias, %off;
    .reg .f32 %vo, %vb;
    .reg .pred %p;

    ld.param.u64 %out,   [output_ptr];
    ld.param.u64 %bias,  [bias_ptr];
    ld.param.u32 %sp,    [spatial_size];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %gid, %tid.x;
    mad.lo.u32 %gid, %bid, %bdim, %gid;

    setp.ge.u32 %p, %gid, %n_reg;
    @%p bra DONE;

    // c = gid / spatial_size
    div.u32 %c, %gid, %sp;

    // Load output[gid]
    cvt.u64.u32 %off, %gid;
    shl.b64 %off, %off, 2;
    add.u64 %out, %out, %off;
    ld.global.f32 %vo, [%out];

    // Load bias[c]
    cvt.u64.u32 %off, %c;
    shl.b64 %off, %off, 2;
    add.u64 %bias, %bias, %off;
    ld.global.f32 %vb, [%bias];

    // output[tid] += bias[c]
    add.f32 %vo, %vo, %vb;
    st.global.f32 [%out], %vo;

DONE:
    ret;
}
";

// ---------------------------------------------------------------------------
// Launch configuration helper
// ---------------------------------------------------------------------------

/// Standard 1-D launch config for `n` elements, 256 threads per block.
///
/// # Errors
///
/// Returns [`GpuError::ShapeMismatch`] if `n` exceeds `u32::MAX`.
#[cfg(feature = "cuda")]
fn launch_cfg(n: usize) -> GpuResult<cudarc::driver::LaunchConfig> {
    if n > u32::MAX as usize {
        return Err(GpuError::ShapeMismatch {
            op: "kernel_launch",
            expected: vec![u32::MAX as usize],
            got: vec![n],
        });
    }
    const BLOCK: u32 = 256;
    let grid = ((n as u32).saturating_add(BLOCK - 1)) / BLOCK;
    Ok(cudarc::driver::LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    })
}

// ---------------------------------------------------------------------------
// GPU conv2d -- fully on-device: GPU im2col + GPU GEMM + GPU bias add
// ---------------------------------------------------------------------------

/// Compute a 2-D convolution on the GPU using im2col + cuBLAS GEMM.
///
/// The **entire pipeline runs on-device** with zero CPU roundtrips:
///
/// 1. **im2col PTX kernel** -- `input (GPU) -> columns (GPU)`.
/// 2. **cuBLAS GEMM** -- `weight (GPU) @ columns (GPU) -> output (GPU)`.
/// 3. **bias_add PTX kernel** -- `output (GPU) += bias (GPU)`.
///
/// # Arguments
///
/// - `input` -- GPU buffer containing `[B, C_in, H, W]` flattened in
///   row-major order.
/// - `weight` -- GPU buffer containing
///   `[C_out, C_in / groups, kH, kW]` flattened in row-major order. The
///   second axis is the per-group input channel count, matching PyTorch's
///   grouped-convolution layout.
/// - `bias` -- optional GPU buffer containing `[C_out]` bias values.
/// - `input_shape` -- `[B, C_in, H, W]`.
/// - `weight_shape` -- `[C_out, C_in / groups, kH, kW]`.
/// - `stride` -- `(stride_h, stride_w)`.
/// - `padding` -- `(pad_h, pad_w)`.
/// - `dilation` -- `(dil_h, dil_w)`. `(1, 1)` for the dense case.
/// - `groups` -- channel-group count. `1` for the dense case. Must divide
///   both `C_in` and `C_out`.
/// - `device` -- the GPU device that owns all buffers.
///
/// # Returns
///
/// A tuple `(output_buffer, output_shape)` where `output_shape` is
/// `[B, C_out, H_out, W_out]`.
///
/// # Errors
///
/// - [`GpuError::ShapeMismatch`] if buffer lengths are inconsistent with
///   shapes, or if `groups` does not divide both `C_in` and `C_out`.
/// - [`GpuError::DeviceMismatch`] if buffers are on different devices.
/// - [`GpuError::Driver`] or [`GpuError::Blas`] on CUDA runtime errors.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
pub fn gpu_conv2d_f32(
    input: &CudaBuffer<f32>,
    weight: &CudaBuffer<f32>,
    bias: Option<&CudaBuffer<f32>>,
    input_shape: [usize; 4],
    weight_shape: [usize; 4],
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
    device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, [usize; 4])> {
    use cudarc::driver::PushKernelArg;

    let [batch, c_in, h, w] = input_shape;
    let [c_out, c_in_per_group_w, kh, kw] = weight_shape;

    // Validate groups.
    if groups == 0 {
        return Err(GpuError::ShapeMismatch {
            op: "conv2d",
            expected: vec![1],
            got: vec![0],
        });
    }
    if c_in % groups != 0 || c_out % groups != 0 {
        return Err(GpuError::ShapeMismatch {
            op: "conv2d",
            expected: vec![c_in, c_out],
            got: vec![groups, groups],
        });
    }
    let c_in_per_group = c_in / groups;
    let c_out_per_group = c_out / groups;

    // Weight's input-channel axis must equal c_in / groups (PyTorch layout).
    if c_in_per_group_w != c_in_per_group {
        return Err(GpuError::ShapeMismatch {
            op: "conv2d",
            expected: vec![c_in_per_group],
            got: vec![c_in_per_group_w],
        });
    }

    if dilation.0 == 0 || dilation.1 == 0 {
        return Err(GpuError::ShapeMismatch {
            op: "conv2d",
            expected: vec![1, 1],
            got: vec![dilation.0, dilation.1],
        });
    }

    // Validate buffer sizes.
    let expected_input_len = batch * c_in * h * w;
    if input.len() != expected_input_len {
        return Err(GpuError::ShapeMismatch {
            op: "conv2d",
            expected: input_shape.to_vec(),
            got: vec![input.len()],
        });
    }

    let expected_weight_len = c_out * c_in_per_group * kh * kw;
    if weight.len() != expected_weight_len {
        return Err(GpuError::ShapeMismatch {
            op: "conv2d",
            expected: weight_shape.to_vec(),
            got: vec![weight.len()],
        });
    }

    if let Some(b) = bias {
        if b.len() != c_out {
            return Err(GpuError::ShapeMismatch {
                op: "conv2d",
                expected: vec![c_out],
                got: vec![b.len()],
            });
        }
    }

    // Validate devices.
    if input.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: device.ordinal(),
            got: input.device_ordinal(),
        });
    }
    if weight.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: device.ordinal(),
            got: weight.device_ordinal(),
        });
    }

    // Effective kernel extent after dilation.
    let eff_kh = dilation.0 * (kh - 1) + 1;
    let eff_kw = dilation.1 * (kw - 1) + 1;

    // Compute output spatial dimensions.
    let h_out = (h + 2 * padding.0 - eff_kh) / stride.0 + 1;
    let w_out = (w + 2 * padding.1 - eff_kw) / stride.1 + 1;
    let output_shape = [batch, c_out, h_out, w_out];

    // Handle degenerate case.
    if batch == 0 || c_out == 0 || h_out == 0 || w_out == 0 {
        let out = crate::transfer::alloc_zeros_f32(0, device)?;
        return Ok((out, output_shape));
    }

    // Per-group im2col / GEMM dimensions.
    let group_col_rows = c_in_per_group * kh * kw;
    let col_cols = h_out * w_out;
    let group_col_elems = group_col_rows * col_cols;
    let group_out_elems = c_out_per_group * col_cols;
    let out_elems_per_batch = c_out * col_cols;
    let total_out_elems = batch * out_elems_per_batch;
    let weight_per_group_elems = c_out_per_group * c_in_per_group * kh * kw;

    // -----------------------------------------------------------------------
    // Allocate GPU buffers (all on-device, no CPU roundtrips)
    // -----------------------------------------------------------------------

    // Column buffer: reused for each (batch, group) pair —
    // shape [group_col_rows, col_cols].
    let mut col_buf = crate::transfer::alloc_zeros_f32(group_col_elems, device)?;

    // Per-group weight buffer (reused across batches/groups for groups > 1).
    // For groups == 1 we pass `weight` directly to GEMM and skip this copy
    // entirely so the dense path is unchanged.
    let mut weight_group_buf = if groups > 1 {
        Some(crate::transfer::alloc_zeros_f32(
            weight_per_group_elems,
            device,
        )?)
    } else {
        None
    };

    // Full output buffer: [B, C_out, H_out * W_out] contiguous.
    let mut output_buf = crate::transfer::alloc_zeros_f32(total_out_elems, device)?;

    // -----------------------------------------------------------------------
    // Load PTX modules (cached after first compilation)
    // -----------------------------------------------------------------------

    let ctx = device.context();
    let stream = device.stream();

    let ord = device.ordinal() as u32;
    let im2col_fn = crate::module_cache::get_or_compile(ctx, IM2COL_PTX, "im2col_kernel", ord)?;

    let bias_fn = if bias.is_some() {
        Some(crate::module_cache::get_or_compile(
            ctx,
            BIAS_ADD_PTX,
            "bias_add_kernel",
            ord,
        )?)
    } else {
        None
    };

    // -----------------------------------------------------------------------
    // Per-(batch, group) loop: im2col -> GEMM -> D2D copy -> bias add
    // -----------------------------------------------------------------------

    let im2col_cfg = launch_cfg(group_col_elems)?;

    let c_in_total_u32 = c_in as u32;
    let channels_u32 = c_in_per_group as u32;
    let height_u32 = h as u32;
    let width_u32 = w as u32;
    let kh_u32 = kh as u32;
    let kw_u32 = kw as u32;
    let sh_u32 = stride.0 as u32;
    let sw_u32 = stride.1 as u32;
    let ph_u32 = padding.0 as u32;
    let pw_u32 = padding.1 as u32;
    let dh_u32 = dilation.0 as u32;
    let dw_u32 = dilation.1 as u32;
    let ho_u32 = h_out as u32;
    let wo_u32 = w_out as u32;
    let cr_u32 = group_col_rows as u32;
    let cc_u32 = col_cols as u32;

    for b in 0..batch {
        let b_u32 = b as u32;

        for g in 0..groups {
            let channel_offset_u32 = (g * c_in_per_group) as u32;

            // --- im2col: input (GPU) -> col_buf (GPU) for this (b, g) ---
            //
            // SAFETY: The kernel reads from `input` within the batch-element
            // region for batch `b` and channels
            // `[g*c_in_per_group .. (g+1)*c_in_per_group]`, and writes to
            // `col_buf` which holds `group_col_rows * col_cols` elements.
            // Both buffers are device-resident with sufficient length. The
            // grid covers exactly `group_col_elems` threads.
            unsafe {
                stream
                    .launch_builder(&im2col_fn)
                    .arg(input.inner())
                    .arg(col_buf.inner_mut())
                    .arg(&b_u32)
                    .arg(&c_in_total_u32)
                    .arg(&channel_offset_u32)
                    .arg(&channels_u32)
                    .arg(&height_u32)
                    .arg(&width_u32)
                    .arg(&kh_u32)
                    .arg(&kw_u32)
                    .arg(&sh_u32)
                    .arg(&sw_u32)
                    .arg(&ph_u32)
                    .arg(&pw_u32)
                    .arg(&dh_u32)
                    .arg(&dw_u32)
                    .arg(&ho_u32)
                    .arg(&wo_u32)
                    .arg(&cr_u32)
                    .arg(&cc_u32)
                    .launch(im2col_cfg)?;
            }

            // --- Per-group GEMM: w_g @ col_buf ---
            // w_g:    [c_out_per_group, group_col_rows]
            // col_buf:[group_col_rows, col_cols]
            // result: [c_out_per_group, col_cols]
            //
            // For groups > 1 we copy the contiguous chunk of `weight`
            // belonging to group `g` into the reusable `weight_group_buf`
            // (D2D, all on-device). For groups == 1 we pass `weight`
            // directly so the dense path takes zero extra copies.
            let gemm_out = if let Some(wg_buf) = weight_group_buf.as_mut() {
                let w_start = g * weight_per_group_elems;
                let w_end = w_start + weight_per_group_elems;
                let w_src = weight.inner().slice(w_start..w_end);
                let mut w_dst = wg_buf.inner_mut().slice_mut(0..weight_per_group_elems);
                stream.memcpy_dtod(&w_src, &mut w_dst)?;
                gpu_matmul_f32(
                    wg_buf,
                    &col_buf,
                    c_out_per_group,
                    group_col_rows,
                    col_cols,
                    device,
                )?
            } else {
                gpu_matmul_f32(
                    weight,
                    &col_buf,
                    c_out_per_group,
                    group_col_rows,
                    col_cols,
                    device,
                )?
            };

            // --- D2D copy: gemm_out -> output_buf[b, group_slice] ---
            //
            // Output for batch b, group g sits at:
            //   output_buf[b * out_elems_per_batch + g * group_out_elems ..]
            // The destination is a contiguous chunk because the channel axis
            // is the slow-moving axis in [B, C_out, col_cols].
            let out_start = b * out_elems_per_batch + g * group_out_elems;
            let out_end = out_start + group_out_elems;
            let gemm_view = gemm_out.inner().slice(0..group_out_elems);
            let mut out_view = output_buf.inner_mut().slice_mut(out_start..out_end);
            stream.memcpy_dtod(&gemm_view, &mut out_view)?;
        }

        // --- Bias add (if present, in-place on output_buf for this batch) ---
        if let (Some(bias_buf), Some(bias_func)) = (bias, &bias_fn) {
            let n_bias = out_elems_per_batch as u32;
            let spatial = col_cols as u32;
            let bias_cfg = launch_cfg(out_elems_per_batch)?;

            let batch_start = b * out_elems_per_batch;
            let batch_end = batch_start + out_elems_per_batch;

            // We need to launch the bias kernel on the sub-region of
            // output_buf for this batch element.  We pass a mutable view.
            let mut out_view = output_buf.inner_mut().slice_mut(batch_start..batch_end);

            // SAFETY: The kernel reads `out_elems_per_batch` elements from
            // the output sub-region and `c_out` elements from `bias_buf`,
            // then writes back the summed values.  All buffers are
            // device-resident with sufficient length. The bias kernel uses
            // `c = i / spatial_size`, which yields the correct C_out index
            // for the contiguous `[C_out, col_cols]` batch slab regardless
            // of `groups` (the channel axis is dense across groups in the
            // output).
            unsafe {
                stream
                    .launch_builder(bias_func)
                    .arg(&mut out_view)
                    .arg(bias_buf.inner())
                    .arg(&spatial)
                    .arg(&n_bias)
                    .launch(bias_cfg)?;
            }
        }
    }

    Ok((output_buf, output_shape))
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn gpu_conv2d_f32(
    _input: &CudaBuffer<f32>,
    _weight: &CudaBuffer<f32>,
    _bias: Option<&CudaBuffer<f32>>,
    _input_shape: [usize; 4],
    _weight_shape: [usize; 4],
    _stride: (usize, usize),
    _padding: (usize, usize),
    _dilation: (usize, usize),
    _groups: usize,
    _device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, [usize; 4])> {
    Err(GpuError::NoCudaFeature)
}

// ---------------------------------------------------------------------------
// Convenience: CPU-only conv2d reference (for testing)
// ---------------------------------------------------------------------------

/// Pure CPU conv2d for reference/testing.
///
/// Same im2col + matmul approach, entirely on the CPU.
/// Used by tests to verify GPU results.
#[cfg(test)]
fn cpu_conv2d_reference(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    input_shape: [usize; 4],
    weight_shape: [usize; 4],
    stride: (usize, usize),
    padding: (usize, usize),
) -> (Vec<f32>, [usize; 4]) {
    let [batch, c_in, h, w] = input_shape;
    let [c_out, _c_in_w, kh, kw] = weight_shape;

    let h_out = (h + 2 * padding.0 - kh) / stride.0 + 1;
    let w_out = (w + 2 * padding.1 - kw) / stride.1 + 1;
    let output_shape = [batch, c_out, h_out, w_out];

    let col_rows = c_in * kh * kw;
    let col_cols = h_out * w_out;

    let (cols, _, _) = im2col_cpu(
        input, batch, c_in, h, w, kh, kw, stride.0, stride.1, padding.0, padding.1,
    );

    let mut output = Vec::with_capacity(batch * c_out * col_cols);

    for b in 0..batch {
        let cols_start = b * col_rows * col_cols;

        // weight_2d [C_out, col_rows] @ cols_b [col_rows, col_cols] = out_b [C_out, col_cols]
        for co in 0..c_out {
            for j in 0..col_cols {
                let mut sum = 0.0f32;
                for p in 0..col_rows {
                    sum += weight[co * col_rows + p] * cols[cols_start + p * col_cols + j];
                }
                if let Some(bias_data) = bias {
                    sum += bias_data[co];
                }
                output.push(sum);
            }
        }
    }

    (output, output_shape)
}

// ---------------------------------------------------------------------------
// Tests -- require a real CUDA GPU
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use crate::device::GpuDevice;
    use crate::transfer::{cpu_to_gpu, gpu_to_cpu};

    /// Helper: compare two f32 slices with tolerance.
    fn assert_close(got: &[f32], expected: &[f32], tol: f32, label: &str) {
        assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < tol,
                "{label}: element {i}: got {g}, expected {e}, diff {}",
                (g - e).abs(),
            );
        }
    }

    // -- Output shape correctness ---------------------------------------------

    #[test]
    fn conv2d_output_shape_no_padding() {
        // Input: [1, 1, 5, 5], Weight: [1, 1, 3, 3], stride=1, padding=0
        // H_out = (5 - 3) / 1 + 1 = 3
        // W_out = (5 - 3) / 1 + 1 = 3
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let input_data: Vec<f32> = (0..25).map(|i| i as f32).collect();
        let weight_data: Vec<f32> = vec![1.0; 9]; // 3x3 all-ones kernel

        let input = cpu_to_gpu(&input_data, &dev).expect("input to gpu");
        let weight = cpu_to_gpu(&weight_data, &dev).expect("weight to gpu");

        let (out, shape) = gpu_conv2d_f32(
            &input,
            &weight,
            None,
            [1, 1, 5, 5],
            [1, 1, 3, 3],
            (1, 1),
            (0, 0),
            (1, 1),
            1,
            &dev,
        )
        .expect("gpu_conv2d_f32");

        assert_eq!(shape, [1, 1, 3, 3]);
        assert_eq!(out.len(), 9);
    }

    #[test]
    fn conv2d_output_shape_with_padding() {
        // Input: [1, 1, 5, 5], Weight: [1, 1, 3, 3], stride=1, padding=1
        // H_out = (5 + 2 - 3) / 1 + 1 = 5
        // W_out = (5 + 2 - 3) / 1 + 1 = 5
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let input_data: Vec<f32> = (0..25).map(|i| i as f32).collect();
        let weight_data: Vec<f32> = vec![1.0; 9];

        let input = cpu_to_gpu(&input_data, &dev).expect("input to gpu");
        let weight = cpu_to_gpu(&weight_data, &dev).expect("weight to gpu");

        let (out, shape) = gpu_conv2d_f32(
            &input,
            &weight,
            None,
            [1, 1, 5, 5],
            [1, 1, 3, 3],
            (1, 1),
            (1, 1),
            (1, 1),
            1,
            &dev,
        )
        .expect("gpu_conv2d_f32");

        assert_eq!(shape, [1, 1, 5, 5]);
        assert_eq!(out.len(), 25);
    }

    #[test]
    fn conv2d_output_shape_stride2() {
        // Input: [1, 1, 6, 6], Weight: [1, 1, 3, 3], stride=2, padding=0
        // H_out = (6 - 3) / 2 + 1 = 2
        // W_out = (6 - 3) / 2 + 1 = 2
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let input_data: Vec<f32> = (0..36).map(|i| i as f32).collect();
        let weight_data: Vec<f32> = vec![1.0; 9];

        let input = cpu_to_gpu(&input_data, &dev).expect("input to gpu");
        let weight = cpu_to_gpu(&weight_data, &dev).expect("weight to gpu");

        let (out, shape) = gpu_conv2d_f32(
            &input,
            &weight,
            None,
            [1, 1, 6, 6],
            [1, 1, 3, 3],
            (2, 2),
            (0, 0),
            (1, 1),
            1,
            &dev,
        )
        .expect("gpu_conv2d_f32");

        assert_eq!(shape, [1, 1, 2, 2]);
        assert_eq!(out.len(), 4);
    }

    // -- Correctness vs CPU reference -----------------------------------------

    #[test]
    fn conv2d_correctness_vs_cpu() {
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        // Input: [2, 3, 8, 8], Weight: [4, 3, 3, 3], stride=1, padding=1
        let input_shape = [2, 3, 8, 8];
        let weight_shape = [4, 3, 3, 3];
        let stride = (1, 1);
        let padding = (1, 1);

        let input_len: usize = input_shape.iter().product();
        let weight_len: usize = weight_shape.iter().product();

        // Deterministic non-trivial data.
        let input_data: Vec<f32> = (0..input_len)
            .map(|i| ((i * 7 + 13) % 100) as f32 / 100.0)
            .collect();
        let weight_data: Vec<f32> = (0..weight_len)
            .map(|i| ((i * 11 + 3) % 100) as f32 / 100.0 - 0.5)
            .collect();
        let bias_data: Vec<f32> = vec![0.1, -0.2, 0.3, -0.1];

        // CPU reference.
        let (expected_output, expected_shape) = cpu_conv2d_reference(
            &input_data,
            &weight_data,
            Some(&bias_data),
            input_shape,
            weight_shape,
            stride,
            padding,
        );

        // GPU.
        let input_gpu = cpu_to_gpu(&input_data, &dev).expect("input to gpu");
        let weight_gpu = cpu_to_gpu(&weight_data, &dev).expect("weight to gpu");
        let bias_gpu = cpu_to_gpu(&bias_data, &dev).expect("bias to gpu");

        let (out_gpu, out_shape) = gpu_conv2d_f32(
            &input_gpu,
            &weight_gpu,
            Some(&bias_gpu),
            input_shape,
            weight_shape,
            stride,
            padding,
            (1, 1),
            1,
            &dev,
        )
        .expect("gpu_conv2d_f32");

        assert_eq!(out_shape, expected_shape);

        let out_host = gpu_to_cpu(&out_gpu, &dev).expect("gpu_to_cpu");
        assert_close(&out_host, &expected_output, 1e-3, "conv2d vs cpu");
    }

    #[test]
    fn conv2d_correctness_no_bias() {
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let input_shape = [1, 2, 4, 4];
        let weight_shape = [3, 2, 3, 3];
        let stride = (1, 1);
        let padding = (0, 0);

        let input_len: usize = input_shape.iter().product();
        let weight_len: usize = weight_shape.iter().product();

        let input_data: Vec<f32> = (0..input_len)
            .map(|i| ((i * 3 + 7) % 50) as f32 / 50.0)
            .collect();
        let weight_data: Vec<f32> = (0..weight_len)
            .map(|i| ((i * 5 + 1) % 40) as f32 / 40.0 - 0.5)
            .collect();

        let (expected_output, expected_shape) = cpu_conv2d_reference(
            &input_data,
            &weight_data,
            None,
            input_shape,
            weight_shape,
            stride,
            padding,
        );

        let input_gpu = cpu_to_gpu(&input_data, &dev).expect("input to gpu");
        let weight_gpu = cpu_to_gpu(&weight_data, &dev).expect("weight to gpu");

        let (out_gpu, out_shape) = gpu_conv2d_f32(
            &input_gpu,
            &weight_gpu,
            None,
            input_shape,
            weight_shape,
            stride,
            padding,
            (1, 1),
            1,
            &dev,
        )
        .expect("gpu_conv2d_f32");

        assert_eq!(out_shape, expected_shape);

        let out_host = gpu_to_cpu(&out_gpu, &dev).expect("gpu_to_cpu");
        assert_close(&out_host, &expected_output, 1e-3, "conv2d no bias");
    }

    // -- 1x1 kernel -----------------------------------------------------------

    #[test]
    fn conv2d_1x1_kernel() {
        // 1x1 convolution is just a per-pixel linear layer.
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let input_shape = [1, 3, 4, 4];
        let weight_shape = [2, 3, 1, 1];
        let stride = (1, 1);
        let padding = (0, 0);

        let input_len: usize = input_shape.iter().product();
        let weight_len: usize = weight_shape.iter().product();

        let input_data: Vec<f32> = (0..input_len)
            .map(|i| i as f32 / input_len as f32)
            .collect();
        let weight_data: Vec<f32> = (0..weight_len).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let bias_data: Vec<f32> = vec![0.5, -0.5];

        let (expected_output, expected_shape) = cpu_conv2d_reference(
            &input_data,
            &weight_data,
            Some(&bias_data),
            input_shape,
            weight_shape,
            stride,
            padding,
        );

        // 1x1 conv: output spatial dims = input spatial dims.
        assert_eq!(expected_shape, [1, 2, 4, 4]);

        let input_gpu = cpu_to_gpu(&input_data, &dev).expect("input to gpu");
        let weight_gpu = cpu_to_gpu(&weight_data, &dev).expect("weight to gpu");
        let bias_gpu = cpu_to_gpu(&bias_data, &dev).expect("bias to gpu");

        let (out_gpu, out_shape) = gpu_conv2d_f32(
            &input_gpu,
            &weight_gpu,
            Some(&bias_gpu),
            input_shape,
            weight_shape,
            stride,
            padding,
            (1, 1),
            1,
            &dev,
        )
        .expect("gpu_conv2d_f32");

        assert_eq!(out_shape, expected_shape);

        let out_host = gpu_to_cpu(&out_gpu, &dev).expect("gpu_to_cpu");
        assert_close(&out_host, &expected_output, 1e-4, "conv2d 1x1");
    }

    // -- Multi-batch ----------------------------------------------------------

    #[test]
    fn conv2d_multi_batch() {
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let input_shape = [4, 2, 6, 6];
        let weight_shape = [3, 2, 3, 3];
        let stride = (1, 1);
        let padding = (1, 1);

        let input_len: usize = input_shape.iter().product();
        let weight_len: usize = weight_shape.iter().product();

        let input_data: Vec<f32> = (0..input_len)
            .map(|i| ((i * 13 + 5) % 200) as f32 / 200.0 - 0.5)
            .collect();
        let weight_data: Vec<f32> = (0..weight_len)
            .map(|i| ((i * 17 + 11) % 100) as f32 / 100.0 - 0.5)
            .collect();

        let (expected_output, expected_shape) = cpu_conv2d_reference(
            &input_data,
            &weight_data,
            None,
            input_shape,
            weight_shape,
            stride,
            padding,
        );

        let input_gpu = cpu_to_gpu(&input_data, &dev).expect("input to gpu");
        let weight_gpu = cpu_to_gpu(&weight_data, &dev).expect("weight to gpu");

        let (out_gpu, out_shape) = gpu_conv2d_f32(
            &input_gpu,
            &weight_gpu,
            None,
            input_shape,
            weight_shape,
            stride,
            padding,
            (1, 1),
            1,
            &dev,
        )
        .expect("gpu_conv2d_f32");

        assert_eq!(out_shape, expected_shape);

        let out_host = gpu_to_cpu(&out_gpu, &dev).expect("gpu_to_cpu");
        assert_close(&out_host, &expected_output, 1e-3, "conv2d multi-batch");
    }

    // -- Shape validation errors ----------------------------------------------

    #[test]
    fn conv2d_channel_mismatch() {
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let input_data = vec![0.0f32; 3 * 4 * 4]; // N=1, C_in=3
        let weight_data = vec![0.0f32; 2 * 5 * 3 * 3]; // C_in_w=5 (mismatch!)

        let input = cpu_to_gpu(&input_data, &dev).expect("input to gpu");
        let weight = cpu_to_gpu(&weight_data, &dev).expect("weight to gpu");

        let err = gpu_conv2d_f32(
            &input,
            &weight,
            None,
            [1, 3, 4, 4],
            [2, 5, 3, 3],
            (1, 1),
            (0, 0),
            (1, 1),
            1,
            &dev,
        )
        .unwrap_err();

        match err {
            GpuError::ShapeMismatch { op: "conv2d", .. } => {}
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn conv2d_wrong_input_length() {
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        // Claim shape [1, 1, 4, 4] = 16 elements, but buffer has 10.
        let input_data = vec![0.0f32; 10];
        let weight_data = vec![0.0f32; 3 * 3]; // C_out=1, C_in=1

        let input = cpu_to_gpu(&input_data, &dev).expect("input to gpu");
        let weight = cpu_to_gpu(&weight_data, &dev).expect("weight to gpu");

        let err = gpu_conv2d_f32(
            &input,
            &weight,
            None,
            [1, 1, 4, 4],
            [1, 1, 3, 3],
            (1, 1),
            (0, 0),
            (1, 1),
            1,
            &dev,
        )
        .unwrap_err();

        match err {
            GpuError::ShapeMismatch { op: "conv2d", .. } => {}
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn conv2d_wrong_bias_length() {
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let input_data = vec![0.0f32; 5 * 5]; // N=1, C_in=1
        let weight_data = vec![0.0f32; 2 * 3 * 3]; // C_out=2, C_in=1
        let bias_data = vec![0.0f32; 5]; // should be 2, not 5

        let input = cpu_to_gpu(&input_data, &dev).expect("input to gpu");
        let weight = cpu_to_gpu(&weight_data, &dev).expect("weight to gpu");
        let bias = cpu_to_gpu(&bias_data, &dev).expect("bias to gpu");

        let err = gpu_conv2d_f32(
            &input,
            &weight,
            Some(&bias),
            [1, 1, 5, 5],
            [2, 1, 3, 3],
            (1, 1),
            (0, 0),
            (1, 1),
            1,
            &dev,
        )
        .unwrap_err();

        match err {
            GpuError::ShapeMismatch { op: "conv2d", .. } => {}
            other => panic!("unexpected error: {other}"),
        }
    }

    // -- Stride > 1 correctness -----------------------------------------------

    #[test]
    fn conv2d_stride2_correctness() {
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let input_shape = [1, 1, 6, 6];
        let weight_shape = [1, 1, 3, 3];
        let stride = (2, 2);
        let padding = (0, 0);

        let input_data: Vec<f32> = (0..36).map(|i| i as f32).collect();
        let weight_data: Vec<f32> = vec![1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0];

        let (expected_output, expected_shape) = cpu_conv2d_reference(
            &input_data,
            &weight_data,
            None,
            input_shape,
            weight_shape,
            stride,
            padding,
        );

        let input_gpu = cpu_to_gpu(&input_data, &dev).expect("input to gpu");
        let weight_gpu = cpu_to_gpu(&weight_data, &dev).expect("weight to gpu");

        let (out_gpu, out_shape) = gpu_conv2d_f32(
            &input_gpu,
            &weight_gpu,
            None,
            input_shape,
            weight_shape,
            stride,
            padding,
            (1, 1),
            1,
            &dev,
        )
        .expect("gpu_conv2d_f32");

        assert_eq!(out_shape, expected_shape);
        assert_eq!(out_shape, [1, 1, 2, 2]);

        let out_host = gpu_to_cpu(&out_gpu, &dev).expect("gpu_to_cpu");
        assert_close(&out_host, &expected_output, 1e-4, "conv2d stride 2");
    }

    // -- GPU-only pipeline: no gpu_to_cpu / cpu_to_gpu in the hot path --------

    #[test]
    fn conv2d_gpu_pipeline_structural() {
        // This test verifies structural correctness of the fully on-device
        // pipeline by running a larger conv2d that would be expensive if
        // any unintended CPU roundtrips were happening.  We compare against
        // the CPU reference for correctness.
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let input_shape = [8, 16, 32, 32];
        let weight_shape = [32, 16, 3, 3];
        let stride = (1, 1);
        let padding = (1, 1);

        let input_len: usize = input_shape.iter().product();
        let weight_len: usize = weight_shape.iter().product();

        let input_data: Vec<f32> = (0..input_len)
            .map(|i| ((i * 7 + 13) % 256) as f32 / 256.0 - 0.5)
            .collect();
        let weight_data: Vec<f32> = (0..weight_len)
            .map(|i| ((i * 11 + 3) % 128) as f32 / 128.0 - 0.5)
            .collect();
        let bias_data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.01).collect();

        let (expected_output, expected_shape) = cpu_conv2d_reference(
            &input_data,
            &weight_data,
            Some(&bias_data),
            input_shape,
            weight_shape,
            stride,
            padding,
        );

        let input_gpu = cpu_to_gpu(&input_data, &dev).expect("input to gpu");
        let weight_gpu = cpu_to_gpu(&weight_data, &dev).expect("weight to gpu");
        let bias_gpu = cpu_to_gpu(&bias_data, &dev).expect("bias to gpu");

        let (out_gpu, out_shape) = gpu_conv2d_f32(
            &input_gpu,
            &weight_gpu,
            Some(&bias_gpu),
            input_shape,
            weight_shape,
            stride,
            padding,
            (1, 1),
            1,
            &dev,
        )
        .expect("gpu_conv2d_f32");

        assert_eq!(out_shape, expected_shape);

        let out_host = gpu_to_cpu(&out_gpu, &dev).expect("gpu_to_cpu");
        assert_close(&out_host, &expected_output, 1e-2, "conv2d gpu pipeline");
    }

    // -- Pass 2A: groups + dilation natively on the GPU -----------------------

    /// Pure CPU reference for grouped + dilated conv2d. Done by direct
    /// O(N) loops rather than im2col so the test depends on the math, not
    /// a shared im2col implementation.
    #[allow(clippy::too_many_arguments)]
    fn cpu_conv2d_groups_dilation_reference(
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        input_shape: [usize; 4],
        weight_shape: [usize; 4],
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> (Vec<f32>, [usize; 4]) {
        let [batch, c_in, h, w] = input_shape;
        let [c_out, c_in_per_group, kh, kw] = weight_shape;
        assert_eq!(c_in % groups, 0);
        assert_eq!(c_out % groups, 0);
        assert_eq!(c_in_per_group, c_in / groups);

        let eff_kh = dilation.0 * (kh - 1) + 1;
        let eff_kw = dilation.1 * (kw - 1) + 1;
        let h_out = (h + 2 * padding.0 - eff_kh) / stride.0 + 1;
        let w_out = (w + 2 * padding.1 - eff_kw) / stride.1 + 1;
        let c_out_per_group = c_out / groups;

        let mut output = vec![0.0f32; batch * c_out * h_out * w_out];

        for b in 0..batch {
            for g in 0..groups {
                for co_g in 0..c_out_per_group {
                    let co = g * c_out_per_group + co_g;
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let mut sum = 0.0f32;
                            for ci_g in 0..c_in_per_group {
                                let ci = g * c_in_per_group + ci_g;
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        let ih = (oh * stride.0 + ki * dilation.0)
                                            as isize
                                            - padding.0 as isize;
                                        let iw = (ow * stride.1 + kj * dilation.1)
                                            as isize
                                            - padding.1 as isize;
                                        if ih >= 0
                                            && iw >= 0
                                            && (ih as usize) < h
                                            && (iw as usize) < w
                                        {
                                            let in_idx = b * c_in * h * w
                                                + ci * h * w
                                                + (ih as usize) * w
                                                + (iw as usize);
                                            // weight is [C_out, C_in/groups, kH, kW]
                                            let w_idx = co * c_in_per_group * kh * kw
                                                + ci_g * kh * kw
                                                + ki * kw
                                                + kj;
                                            sum += input[in_idx] * weight[w_idx];
                                        }
                                    }
                                }
                            }
                            if let Some(bd) = bias {
                                sum += bd[co];
                            }
                            let out_idx =
                                b * c_out * h_out * w_out + co * h_out * w_out + oh * w_out + ow;
                            output[out_idx] = sum;
                        }
                    }
                }
            }
        }

        (output, [batch, c_out, h_out, w_out])
    }

    /// Pass 2A north-star test: GPU conv2d with `groups=2` and
    /// `dilation=(2, 2)` matches the CPU reference within F32 elementwise
    /// tolerance. A stub that ignores `groups` (e.g. fold groups back to
    /// 1, or take the dense path) produces a different output sum and
    /// fails this test on the very first batch row — see the
    /// discrimination claim in the audit notes.
    #[test]
    fn conv2d_groups_dilation_native_gpu() {
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        // Realistic shape with both groups > 1 and dilation > 1.
        // c_in=4, groups=2 -> c_in/group=2; c_out=6, groups=2 -> c_out/group=3.
        let batch = 2usize;
        let c_in = 4usize;
        let c_out = 6usize;
        let groups = 2usize;
        let kh = 3usize;
        let kw = 3usize;
        let h = 9usize;
        let w = 9usize;
        let stride = (1, 1);
        let padding = (2, 2);
        let dilation = (2, 2);

        let input_shape = [batch, c_in, h, w];
        let weight_shape = [c_out, c_in / groups, kh, kw];

        let input_len: usize = input_shape.iter().product();
        let weight_len: usize = weight_shape.iter().product();

        // Deterministic non-trivial data.
        let input_data: Vec<f32> = (0..input_len)
            .map(|i| ((i * 7 + 13) % 97) as f32 / 97.0 - 0.5)
            .collect();
        let weight_data: Vec<f32> = (0..weight_len)
            .map(|i| ((i * 11 + 5) % 53) as f32 / 53.0 - 0.5)
            .collect();
        let bias_data: Vec<f32> = (0..c_out).map(|i| (i as f32 - 3.0) * 0.1).collect();

        // CPU reference (independent of im2col_cpu — see helper).
        let (expected, expected_shape) = cpu_conv2d_groups_dilation_reference(
            &input_data,
            &weight_data,
            Some(&bias_data),
            input_shape,
            weight_shape,
            stride,
            padding,
            dilation,
            groups,
        );

        // GPU.
        let input_gpu = cpu_to_gpu(&input_data, &dev).expect("input to gpu");
        let weight_gpu = cpu_to_gpu(&weight_data, &dev).expect("weight to gpu");
        let bias_gpu = cpu_to_gpu(&bias_data, &dev).expect("bias to gpu");

        let (out_gpu, out_shape) = gpu_conv2d_f32(
            &input_gpu,
            &weight_gpu,
            Some(&bias_gpu),
            input_shape,
            weight_shape,
            stride,
            padding,
            dilation,
            groups,
            &dev,
        )
        .expect("gpu_conv2d_f32 groups+dilation");

        assert_eq!(out_shape, expected_shape);

        let out_host = gpu_to_cpu(&out_gpu, &dev).expect("gpu_to_cpu");
        // F32 elementwise tolerance for fused matmul + im2col + bias add.
        assert_close(
            &out_host,
            &expected,
            1e-5,
            "conv2d groups=2 dilation=(2,2)",
        );
    }
}
