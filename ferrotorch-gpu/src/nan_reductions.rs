//! CUDA kernels for NaN-skipping reductions.

#![cfg(feature = "cuda")]

use std::sync::OnceLock;

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::buffer::CudaBuffer;
use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};
use crate::module_cache::get_or_compile;
use crate::transfer::{alloc_zeros_f32, alloc_zeros_f64};

const BLOCK_SIZE: u32 = 256;

fn launch_1d(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n as u32).div_ceil(BLOCK_SIZE), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn validate_axis(
    op: &'static str,
    input_len: usize,
    other_len: Option<usize>,
    outer: usize,
    axis: usize,
    inner: usize,
) -> GpuResult<(usize, usize)> {
    let total_in = outer
        .checked_mul(axis)
        .and_then(|v| v.checked_mul(inner))
        .ok_or(GpuError::ShapeMismatch {
            op,
            expected: vec![outer, axis, inner],
            got: vec![usize::MAX],
        })?;
    let total_out = outer.checked_mul(inner).ok_or(GpuError::ShapeMismatch {
        op,
        expected: vec![outer, inner],
        got: vec![usize::MAX],
    })?;
    if input_len != total_in || other_len.is_some_and(|len| len != total_out) {
        return Err(GpuError::ShapeMismatch {
            op,
            expected: vec![total_in, total_out],
            got: vec![input_len, other_len.unwrap_or(total_out)],
        });
    }
    Ok((total_in, total_out))
}

const NAN_REDUCE_AXIS_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry nan_reduce_axis_f32_kernel(
    .param .u64 input_ptr,
    .param .u64 out_ptr,
    .param .u32 outer_size,
    .param .u32 axis_size,
    .param .u32 inner_size,
    .param .u32 total_output,
    .param .u32 take_mean
) {
    .reg .u32 %r_tid, %bid, %bdim, %total, %outer, %axis, %inner, %flag;
    .reg .u32 %outer_idx, %inner_idx, %k, %base, %idx, %count;
    .reg .u64 %in, %out, %off, %addr;
    .reg .f32 %val, %sum, %count_f, %zero;
    .reg .pred %p, %lp, %is_nan, %take_mean_p;

    ld.param.u64 %in, [input_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %outer, [outer_size];
    ld.param.u32 %axis, [axis_size];
    ld.param.u32 %inner, [inner_size];
    ld.param.u32 %total, [total_output];
    ld.param.u32 %flag, [take_mean];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;
    setp.ge.u32 %p, %r_tid, %total;
    @%p bra DONE;

    div.u32 %outer_idx, %r_tid, %inner;
    rem.u32 %inner_idx, %r_tid, %inner;
    mul.lo.u32 %base, %outer_idx, %axis;
    mul.lo.u32 %base, %base, %inner;
    add.u32 %base, %base, %inner_idx;
    mov.f32 %zero, 0f00000000;
    mov.f32 %sum, 0f00000000;
    mov.u32 %count, 0;
    mov.u32 %k, 0;
LOOP:
    setp.ge.u32 %lp, %k, %axis;
    @%lp bra LOOP_DONE;
    mul.lo.u32 %idx, %k, %inner;
    add.u32 %idx, %base, %idx;
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in, %off;
    ld.global.f32 %val, [%addr];
    setp.nan.f32 %is_nan, %val, %val;
    @%is_nan bra SKIP;
    add.f32 %sum, %sum, %val;
    add.u32 %count, %count, 1;
SKIP:
    add.u32 %k, %k, 1;
    bra LOOP;
LOOP_DONE:
    setp.ne.u32 %take_mean_p, %flag, 0;
    @!%take_mean_p bra STORE;
    cvt.rn.f32.u32 %count_f, %count;
    div.rn.f32 %sum, %sum, %count_f;

STORE:
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %out, %off;
    st.global.f32 [%addr], %sum;
DONE:
    ret;
}
";

const NAN_REDUCE_AXIS_BACKWARD_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry nan_reduce_axis_backward_f32_kernel(
    .param .u64 input_ptr,
    .param .u64 grad_out_ptr,
    .param .u64 grad_in_ptr,
    .param .u32 outer_size,
    .param .u32 axis_size,
    .param .u32 inner_size,
    .param .u32 total_input,
    .param .u32 take_mean
) {
    .reg .u32 %r_tid, %bid, %bdim, %total, %outer, %axis, %inner, %flag;
    .reg .u32 %tmp, %outer_idx, %inner_idx, %d_idx, %k, %src_idx, %go_idx, %count;
    .reg .u64 %in, %go, %gi, %off, %addr;
    .reg .f32 %val, %grad, %go_val, %count_f, %zero;
    .reg .pred %p, %lp, %is_nan, %take_mean_p, %count_zero;

    ld.param.u64 %in, [input_ptr];
    ld.param.u64 %go, [grad_out_ptr];
    ld.param.u64 %gi, [grad_in_ptr];
    ld.param.u32 %outer, [outer_size];
    ld.param.u32 %axis, [axis_size];
    ld.param.u32 %inner, [inner_size];
    ld.param.u32 %total, [total_input];
    ld.param.u32 %flag, [take_mean];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;
    setp.ge.u32 %p, %r_tid, %total;
    @%p bra DONE;

    mov.f32 %zero, 0f00000000;
    rem.u32 %inner_idx, %r_tid, %inner;
    div.u32 %tmp, %r_tid, %inner;
    rem.u32 %d_idx, %tmp, %axis;
    div.u32 %outer_idx, %tmp, %axis;
    mul.lo.u32 %go_idx, %outer_idx, %inner;
    add.u32 %go_idx, %go_idx, %inner_idx;

    cvt.u64.u32 %off, %go_idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %go, %off;
    ld.global.f32 %go_val, [%addr];

    mov.u32 %count, 0;
    mov.u32 %k, 0;
COUNT_LOOP:
    setp.ge.u32 %lp, %k, %axis;
    @%lp bra COUNT_DONE;
    mul.lo.u32 %src_idx, %outer_idx, %axis;
    add.u32 %src_idx, %src_idx, %k;
    mul.lo.u32 %src_idx, %src_idx, %inner;
    add.u32 %src_idx, %src_idx, %inner_idx;
    cvt.u64.u32 %off, %src_idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in, %off;
    ld.global.f32 %val, [%addr];
    setp.nan.f32 %is_nan, %val, %val;
    @%is_nan bra COUNT_SKIP;
    add.u32 %count, %count, 1;
COUNT_SKIP:
    add.u32 %k, %k, 1;
    bra COUNT_LOOP;
COUNT_DONE:
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in, %off;
    ld.global.f32 %val, [%addr];

    setp.ne.u32 %take_mean_p, %flag, 0;
    @%take_mean_p bra MEAN_PATH;
    setp.nan.f32 %is_nan, %val, %val;
    selp.f32 %grad, %zero, %go_val, %is_nan;
    bra STORE;

MEAN_PATH:
    setp.eq.u32 %count_zero, %count, 0;
    @%count_zero bra MEAN_NAN;
    setp.nan.f32 %is_nan, %val, %val;
    @%is_nan bra MEAN_ZERO;
    cvt.rn.f32.u32 %count_f, %count;
    div.rn.f32 %grad, %go_val, %count_f;
    bra STORE;
MEAN_ZERO:
    mov.f32 %grad, 0f00000000;
    bra STORE;
MEAN_NAN:
    div.rn.f32 %grad, %zero, %zero;

STORE:
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %gi, %off;
    st.global.f32 [%addr], %grad;
DONE:
    ret;
}
";

/// NaN-skipping sum/mean along one axis for f32.
pub fn nan_reduce_axis_f32(
    input: &CudaBuffer<f32>,
    outer: usize,
    axis: usize,
    inner: usize,
    take_mean: bool,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    let (_, total_out) =
        validate_axis("nan_reduce_axis_f32", input.len(), None, outer, axis, inner)?;
    let mut out = alloc_zeros_f32(total_out, device)?;
    if total_out == 0 {
        return Ok(out);
    }
    let f = get_or_compile(
        device.context(),
        NAN_REDUCE_AXIS_F32_PTX,
        "nan_reduce_axis_f32_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "nan_reduce_axis_f32_kernel",
        source: e,
    })?;
    let cfg = launch_1d(total_out);
    let outer_u32 = outer as u32;
    let axis_u32 = axis as u32;
    let inner_u32 = inner as u32;
    let total_u32 = total_out as u32;
    let take_mean_u32 = u32::from(take_mean);
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&outer_u32)
            .arg(&axis_u32)
            .arg(&inner_u32)
            .arg(&total_u32)
            .arg(&take_mean_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Backward for f32 `nansum`/`nanmean` along one axis.
pub fn nan_reduce_axis_backward_f32(
    input: &CudaBuffer<f32>,
    grad_output: &CudaBuffer<f32>,
    outer: usize,
    axis: usize,
    inner: usize,
    take_mean: bool,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    let (total_in, total_out) = validate_axis(
        "nan_reduce_axis_backward_f32",
        input.len(),
        Some(grad_output.len()),
        outer,
        axis,
        inner,
    )?;
    let mut out = alloc_zeros_f32(total_in, device)?;
    if total_in == 0 {
        return Ok(out);
    }
    if grad_output.len() != total_out {
        return Err(GpuError::ShapeMismatch {
            op: "nan_reduce_axis_backward_f32",
            expected: vec![total_out],
            got: vec![grad_output.len()],
        });
    }
    let f = get_or_compile(
        device.context(),
        NAN_REDUCE_AXIS_BACKWARD_F32_PTX,
        "nan_reduce_axis_backward_f32_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "nan_reduce_axis_backward_f32_kernel",
        source: e,
    })?;
    let cfg = launch_1d(total_in);
    let outer_u32 = outer as u32;
    let axis_u32 = axis as u32;
    let inner_u32 = inner as u32;
    let total_u32 = total_in as u32;
    let take_mean_u32 = u32::from(take_mean);
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(input.inner())
            .arg(grad_output.inner())
            .arg(out.inner_mut())
            .arg(&outer_u32)
            .arg(&axis_u32)
            .arg(&inner_u32)
            .arg(&total_u32)
            .arg(&take_mean_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// NaN-skipping sum/mean along one axis for f64.
pub fn nan_reduce_axis_f64(
    input: &CudaBuffer<f64>,
    outer: usize,
    axis: usize,
    inner: usize,
    take_mean: bool,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    static CACHE: OnceLock<String> = OnceLock::new();
    let (_, total_out) =
        validate_axis("nan_reduce_axis_f64", input.len(), None, outer, axis, inner)?;
    let mut out = alloc_zeros_f64(total_out, device)?;
    if total_out == 0 {
        return Ok(out);
    }
    let ptx = crate::kernels::get_f64_ptx(
        &CACHE,
        NAN_REDUCE_AXIS_F32_PTX,
        "nan_reduce_axis_f32_kernel",
        "nan_reduce_axis_f64_kernel",
    );
    let f = get_or_compile(
        device.context(),
        ptx,
        "nan_reduce_axis_f64_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "nan_reduce_axis_f64_kernel",
        source: e,
    })?;
    let cfg = launch_1d(total_out);
    let outer_u32 = outer as u32;
    let axis_u32 = axis as u32;
    let inner_u32 = inner as u32;
    let total_u32 = total_out as u32;
    let take_mean_u32 = u32::from(take_mean);
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&outer_u32)
            .arg(&axis_u32)
            .arg(&inner_u32)
            .arg(&total_u32)
            .arg(&take_mean_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Backward for f64 `nansum`/`nanmean` along one axis.
pub fn nan_reduce_axis_backward_f64(
    input: &CudaBuffer<f64>,
    grad_output: &CudaBuffer<f64>,
    outer: usize,
    axis: usize,
    inner: usize,
    take_mean: bool,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    static CACHE: OnceLock<String> = OnceLock::new();
    let (total_in, _) = validate_axis(
        "nan_reduce_axis_backward_f64",
        input.len(),
        Some(grad_output.len()),
        outer,
        axis,
        inner,
    )?;
    let mut out = alloc_zeros_f64(total_in, device)?;
    if total_in == 0 {
        return Ok(out);
    }
    let ptx = crate::kernels::get_f64_ptx(
        &CACHE,
        NAN_REDUCE_AXIS_BACKWARD_F32_PTX,
        "nan_reduce_axis_backward_f32_kernel",
        "nan_reduce_axis_backward_f64_kernel",
    );
    let f = get_or_compile(
        device.context(),
        ptx,
        "nan_reduce_axis_backward_f64_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "nan_reduce_axis_backward_f64_kernel",
        source: e,
    })?;
    let cfg = launch_1d(total_in);
    let outer_u32 = outer as u32;
    let axis_u32 = axis as u32;
    let inner_u32 = inner as u32;
    let total_u32 = total_in as u32;
    let take_mean_u32 = u32::from(take_mean);
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(input.inner())
            .arg(grad_output.inner())
            .arg(out.inner_mut())
            .arg(&outer_u32)
            .arg(&axis_u32)
            .arg(&inner_u32)
            .arg(&total_u32)
            .arg(&take_mean_u32)
            .launch(cfg)?;
    }
    Ok(out)
}
