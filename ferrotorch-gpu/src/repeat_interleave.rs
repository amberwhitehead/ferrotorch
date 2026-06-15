//! CUDA kernels for scalar-repeat `repeat_interleave`.
//!
//! The public ferrotorch surface mirrors PyTorch's scalar `repeats` overload:
//! every slice along one logical axis is duplicated `repeats` times
//! consecutively. Forward is a pure gather; backward sums each repeated run
//! back into its source slice.

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

#[cfg(feature = "cuda")]
use crate::buffer::CudaBuffer;
#[cfg(feature = "cuda")]
use crate::device::GpuDevice;
#[cfg(feature = "cuda")]
use crate::error::{GpuError, GpuResult};
#[cfg(feature = "cuda")]
use crate::transfer::{alloc_zeros_bf16, alloc_zeros_f32, alloc_zeros_f64};

#[cfg(feature = "cuda")]
const BLOCK_SIZE: u32 = 256;

#[cfg(feature = "cuda")]
macro_rules! repeat_interleave_forward_ptx {
    ($name:literal, $shift:literal, $load:literal, $store:literal, $vreg:literal) => {
        concat!(
            ".version 7.0\n.target sm_52\n.address_size 64\n",
            ".visible .entry ",
            $name,
            "(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 dim_size,
    .param .u32 inner,
    .param .u32 repeats,
    .param .u32 total
) {
    .reg .u32 %idx, %bid, %bdim, %dim_r, %inner_r, %rep_r, %total_r;
    .reg .u32 %inner_idx, %tmp, %outer_dim_coord, %out_dim, %axis_out, %axis_in;
    .reg .u32 %outer, %src_idx;
    .reg .u64 %in_p, %out_p, %off, %addr;
    .reg ",
            $vreg,
            " %val;
    .reg .pred %p;

    ld.param.u64 %in_p, [in_ptr];
    ld.param.u64 %out_p, [out_ptr];
    ld.param.u32 %dim_r, [dim_size];
    ld.param.u32 %inner_r, [inner];
    ld.param.u32 %rep_r, [repeats];
    ld.param.u32 %total_r, [total];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %total_r;
    @%p bra DONE;

    rem.u32 %inner_idx, %idx, %inner_r;
    div.u32 %tmp, %idx, %inner_r;
    mul.lo.u32 %out_dim, %dim_r, %rep_r;
    rem.u32 %axis_out, %tmp, %out_dim;
    div.u32 %outer, %tmp, %out_dim;
    div.u32 %axis_in, %axis_out, %rep_r;

    mul.lo.u32 %src_idx, %outer, %dim_r;
    add.u32 %src_idx, %src_idx, %axis_in;
    mul.lo.u32 %src_idx, %src_idx, %inner_r;
    add.u32 %src_idx, %src_idx, %inner_idx;

    cvt.u64.u32 %off, %src_idx;
    shl.b64 %off, %off, ",
            $shift,
            ";
    add.u64 %addr, %in_p, %off;
    ",
            $load,
            " %val, [%addr];

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, ",
            $shift,
            ";
    add.u64 %addr, %out_p, %off;
    ",
            $store,
            " [%addr], %val;

DONE:
    ret;
}
"
        )
    };
}

#[cfg(feature = "cuda")]
macro_rules! repeat_interleave_backward_float_ptx {
    ($name:literal, $shift:literal, $load:literal, $store:literal, $vreg:literal, $add:literal, $zero:literal) => {
        concat!(
            ".version 7.0\n.target sm_52\n.address_size 64\n",
            ".visible .entry ",
            $name,
            "(
    .param .u64 grad_ptr,
    .param .u64 out_ptr,
    .param .u32 dim_size,
    .param .u32 inner,
    .param .u32 repeats,
    .param .u32 total
) {
    .reg .u32 %idx, %bid, %bdim, %dim_r, %inner_r, %rep_r, %total_r;
    .reg .u32 %inner_idx, %tmp, %axis_in, %outer, %out_dim, %base, %r, %go_idx;
    .reg .u64 %grad_p, %out_p, %off, %addr;
    .reg ",
            $vreg,
            " %acc, %val;
    .reg .pred %p;

    ld.param.u64 %grad_p, [grad_ptr];
    ld.param.u64 %out_p, [out_ptr];
    ld.param.u32 %dim_r, [dim_size];
    ld.param.u32 %inner_r, [inner];
    ld.param.u32 %rep_r, [repeats];
    ld.param.u32 %total_r, [total];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %total_r;
    @%p bra DONE;

    rem.u32 %inner_idx, %idx, %inner_r;
    div.u32 %tmp, %idx, %inner_r;
    rem.u32 %axis_in, %tmp, %dim_r;
    div.u32 %outer, %tmp, %dim_r;
    mul.lo.u32 %out_dim, %dim_r, %rep_r;
    mul.lo.u32 %base, %outer, %out_dim;
    mul.lo.u32 %tmp, %axis_in, %rep_r;
    add.u32 %base, %base, %tmp;
    mul.lo.u32 %base, %base, %inner_r;
    add.u32 %base, %base, %inner_idx;

    mov",
            $vreg,
            " %acc, ",
            $zero,
            ";
    mov.u32 %r, 0;
LOOP:
    setp.ge.u32 %p, %r, %rep_r;
    @%p bra STORE;
    mul.lo.u32 %tmp, %r, %inner_r;
    add.u32 %go_idx, %base, %tmp;
    cvt.u64.u32 %off, %go_idx;
    shl.b64 %off, %off, ",
            $shift,
            ";
    add.u64 %addr, %grad_p, %off;
    ",
            $load,
            " %val, [%addr];
    ",
            $add,
            " %acc, %acc, %val;
    add.u32 %r, %r, 1;
    bra LOOP;

STORE:
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, ",
            $shift,
            ";
    add.u64 %addr, %out_p, %off;
    ",
            $store,
            " [%addr], %acc;
DONE:
    ret;
}
"
        )
    };
}

#[cfg(feature = "cuda")]
macro_rules! repeat_interleave_backward_16_ptx {
    ($name:literal, $target:literal, $load:literal, $store:literal) => {
        concat!(
            ".version 7.0\n.target ",
            $target,
            "\n.address_size 64\n",
            ".visible .entry ",
            $name,
            "(
    .param .u64 grad_ptr,
    .param .u64 out_ptr,
    .param .u32 dim_size,
    .param .u32 inner,
    .param .u32 repeats,
    .param .u32 total
) {
    .reg .u32 %idx, %bid, %bdim, %dim_r, %inner_r, %rep_r, %total_r;
    .reg .u32 %inner_idx, %tmp, %axis_in, %outer, %out_dim, %base, %r, %go_idx;
    .reg .u64 %grad_p, %out_p, %off, %addr;
    .reg .b16 %raw, %out_h, %zero16;
    .reg .b32 %bits;
    .reg .f32 %acc, %val;
    .reg .pred %p;

    ld.param.u64 %grad_p, [grad_ptr];
    ld.param.u64 %out_p, [out_ptr];
    ld.param.u32 %dim_r, [dim_size];
    ld.param.u32 %inner_r, [inner];
    ld.param.u32 %rep_r, [repeats];
    ld.param.u32 %total_r, [total];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %total_r;
    @%p bra DONE;

    rem.u32 %inner_idx, %idx, %inner_r;
    div.u32 %tmp, %idx, %inner_r;
    rem.u32 %axis_in, %tmp, %dim_r;
    div.u32 %outer, %tmp, %dim_r;
    mul.lo.u32 %out_dim, %dim_r, %rep_r;
    mul.lo.u32 %base, %outer, %out_dim;
    mul.lo.u32 %tmp, %axis_in, %rep_r;
    add.u32 %base, %base, %tmp;
    mul.lo.u32 %base, %base, %inner_r;
    add.u32 %base, %base, %inner_idx;

    mov.f32 %acc, 0f00000000;
    mov.u32 %r, 0;
LOOP:
    setp.ge.u32 %p, %r, %rep_r;
    @%p bra STORE;
    mul.lo.u32 %tmp, %r, %inner_r;
    add.u32 %go_idx, %base, %tmp;
    cvt.u64.u32 %off, %go_idx;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %grad_p, %off;
    ld.global.b16 %raw, [%addr];
",
            $load,
            "
    add.rn.f32 %acc, %acc, %val;
    add.u32 %r, %r, 1;
    bra LOOP;

STORE:
",
            $store,
            "
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %out_p, %off;
    st.global.b16 [%addr], %out_h;
DONE:
    ret;
}
"
        )
    };
}

#[cfg(feature = "cuda")]
const RI_FWD_F32_PTX: &str = repeat_interleave_forward_ptx!(
    "repeat_interleave_f32_kernel",
    "2",
    "ld.global.b32",
    "st.global.b32",
    ".b32"
);
#[cfg(feature = "cuda")]
const RI_FWD_F64_PTX: &str = repeat_interleave_forward_ptx!(
    "repeat_interleave_f64_kernel",
    "3",
    "ld.global.b64",
    "st.global.b64",
    ".b64"
);
#[cfg(feature = "cuda")]
const RI_FWD_U16_PTX: &str = repeat_interleave_forward_ptx!(
    "repeat_interleave_u16_kernel",
    "1",
    "ld.global.b16",
    "st.global.b16",
    ".b16"
);

#[cfg(feature = "cuda")]
const RI_BWD_F32_PTX: &str = repeat_interleave_backward_float_ptx!(
    "repeat_interleave_backward_f32_kernel",
    "2",
    "ld.global.f32",
    "st.global.f32",
    ".f32",
    "add.rn.f32",
    "0f00000000"
);
#[cfg(feature = "cuda")]
const RI_BWD_F64_PTX: &str = repeat_interleave_backward_float_ptx!(
    "repeat_interleave_backward_f64_kernel",
    "3",
    "ld.global.f64",
    "st.global.f64",
    ".f64",
    "add.rn.f64",
    "0d0000000000000000"
);
#[cfg(feature = "cuda")]
const RI_BWD_F16_PTX: &str = repeat_interleave_backward_16_ptx!(
    "repeat_interleave_backward_f16_kernel",
    "sm_52",
    "    cvt.f32.f16 %val, %raw;",
    "    cvt.rn.f16.f32 %out_h, %acc;"
);
#[cfg(feature = "cuda")]
const RI_BWD_BF16_PTX: &str = repeat_interleave_backward_16_ptx!(
    "repeat_interleave_backward_bf16_kernel",
    "sm_80",
    "    mov.b16 %zero16, 0;
    mov.b32 %bits, {%zero16, %raw};
    mov.b32 %val, %bits;",
    "    cvt.rn.bf16.f32 %out_h, %acc;"
);

#[cfg(feature = "cuda")]
fn checked_mul(a: usize, b: usize, op: &'static str) -> GpuResult<usize> {
    a.checked_mul(b).ok_or_else(|| GpuError::ShapeMismatch {
        op,
        expected: vec![usize::MAX],
        got: vec![a, b],
    })
}

#[cfg(feature = "cuda")]
fn checked_total(
    outer: usize,
    dim_size: usize,
    inner: usize,
    repeats: usize,
    backward: bool,
    op: &'static str,
) -> GpuResult<usize> {
    for value in [outer, dim_size, inner, repeats] {
        if value > u32::MAX as usize {
            return Err(GpuError::ShapeMismatch {
                op,
                expected: vec![u32::MAX as usize],
                got: vec![value],
            });
        }
    }
    let base = checked_mul(checked_mul(outer, dim_size, op)?, inner, op)?;
    let total = if backward {
        base
    } else {
        checked_mul(base, repeats, op)?
    };
    if total > u32::MAX as usize {
        return Err(GpuError::ShapeMismatch {
            op,
            expected: vec![u32::MAX as usize],
            got: vec![total],
        });
    }
    Ok(total)
}

#[cfg(feature = "cuda")]
fn launch_config(total: usize) -> LaunchConfig {
    let grid_x = (total as u32).div_ceil(BLOCK_SIZE);
    LaunchConfig {
        grid_dim: (grid_x.max(1), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[cfg(feature = "cuda")]
fn launch_forward<T>(
    input: &T,
    input_len: usize,
    input_device: Option<usize>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    repeats: usize,
    device: &GpuDevice,
    ptx: &'static str,
    kernel: &'static str,
) -> GpuResult<(usize, cudarc::driver::CudaFunction)>
where
    T: ?Sized,
{
    if let Some(input_device) = input_device
        && input_device != device.ordinal()
    {
        return Err(GpuError::DeviceMismatch {
            expected: device.ordinal(),
            got: input_device,
        });
    }
    let expected_input = checked_mul(checked_mul(outer, dim_size, kernel)?, inner, kernel)?;
    if input_len != expected_input {
        return Err(GpuError::ShapeMismatch {
            op: kernel,
            expected: vec![expected_input],
            got: vec![input_len],
        });
    }
    let total = checked_total(outer, dim_size, inner, repeats, false, kernel)?;
    let f =
        crate::module_cache::get_or_compile(device.context(), ptx, kernel, device.ordinal() as u32)
            .map_err(|e| GpuError::PtxCompileFailed { kernel, source: e })?;
    let _ = input;
    Ok((total, f))
}

/// Forward repeat-interleave for contiguous f32 CUDA buffers.
#[cfg(feature = "cuda")]
pub fn gpu_repeat_interleave_f32(
    input: &CudaBuffer<f32>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    repeats: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    let (total, f) = launch_forward(
        input,
        input.len(),
        Some(input.device_ordinal()),
        outer,
        dim_size,
        inner,
        repeats,
        device,
        RI_FWD_F32_PTX,
        "repeat_interleave_f32_kernel",
    )?;
    let mut out = alloc_zeros_f32(total, device)?;
    if total == 0 {
        return Ok(out);
    }
    let dim_u32 = dim_size as u32;
    let inner_u32 = inner as u32;
    let repeats_u32 = repeats as u32;
    let total_u32 = total as u32;
    // SAFETY: params match PTX ABI; total and factorization are u32-checked,
    // and the kernel maps each output element to a valid input element.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&dim_u32)
            .arg(&inner_u32)
            .arg(&repeats_u32)
            .arg(&total_u32)
            .launch(launch_config(total))?;
    }
    Ok(out)
}

/// Forward repeat-interleave for contiguous f64 CUDA buffers.
#[cfg(feature = "cuda")]
pub fn gpu_repeat_interleave_f64(
    input: &CudaBuffer<f64>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    repeats: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    let (total, f) = launch_forward(
        input,
        input.len(),
        Some(input.device_ordinal()),
        outer,
        dim_size,
        inner,
        repeats,
        device,
        RI_FWD_F64_PTX,
        "repeat_interleave_f64_kernel",
    )?;
    let mut out = alloc_zeros_f64(total, device)?;
    if total == 0 {
        return Ok(out);
    }
    let dim_u32 = dim_size as u32;
    let inner_u32 = inner as u32;
    let repeats_u32 = repeats as u32;
    let total_u32 = total as u32;
    // SAFETY: same invariant as `gpu_repeat_interleave_f32`; element width is
    // the only difference.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&dim_u32)
            .arg(&inner_u32)
            .arg(&repeats_u32)
            .arg(&total_u32)
            .launch(launch_config(total))?;
    }
    Ok(out)
}

/// Forward repeat-interleave for contiguous f16/bf16 CUDA buffers.
#[cfg(feature = "cuda")]
pub fn gpu_repeat_interleave_u16(
    input: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    repeats: usize,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    let (total, f) = launch_forward(
        input,
        input.len(),
        None,
        outer,
        dim_size,
        inner,
        repeats,
        device,
        RI_FWD_U16_PTX,
        "repeat_interleave_u16_kernel",
    )?;
    let mut out = alloc_zeros_bf16(total, device)?;
    if total == 0 {
        return Ok(out);
    }
    let dim_u32 = dim_size as u32;
    let inner_u32 = inner as u32;
    let repeats_u32 = repeats as u32;
    let total_u32 = total as u32;
    // SAFETY: same invariant as `gpu_repeat_interleave_f32`; values are raw
    // u16 payloads and the backend wrapper preserves the dtype tag.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(input)
            .arg(&mut out)
            .arg(&dim_u32)
            .arg(&inner_u32)
            .arg(&repeats_u32)
            .arg(&total_u32)
            .launch(launch_config(total))?;
    }
    Ok(out)
}

#[cfg(feature = "cuda")]
fn checked_backward_len(
    grad_len: usize,
    outer: usize,
    dim_size: usize,
    inner: usize,
    repeats: usize,
    op: &'static str,
) -> GpuResult<usize> {
    let out_total = checked_total(outer, dim_size, inner, repeats, false, op)?;
    if grad_len != out_total {
        return Err(GpuError::ShapeMismatch {
            op,
            expected: vec![out_total],
            got: vec![grad_len],
        });
    }
    checked_total(outer, dim_size, inner, repeats, true, op)
}

/// Backward repeat-interleave segment-sum for f32 CUDA buffers.
#[cfg(feature = "cuda")]
pub fn gpu_repeat_interleave_backward_f32(
    grad: &CudaBuffer<f32>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    repeats: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    if grad.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: device.ordinal(),
            got: grad.device_ordinal(),
        });
    }
    let total = checked_backward_len(
        grad.len(),
        outer,
        dim_size,
        inner,
        repeats,
        "repeat_interleave_backward_f32",
    )?;
    let f = crate::module_cache::get_or_compile(
        device.context(),
        RI_BWD_F32_PTX,
        "repeat_interleave_backward_f32_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "repeat_interleave_backward_f32_kernel",
        source: e,
    })?;
    let mut out = alloc_zeros_f32(total, device)?;
    if total == 0 {
        return Ok(out);
    }
    let dim_u32 = dim_size as u32;
    let inner_u32 = inner as u32;
    let repeats_u32 = repeats as u32;
    let total_u32 = total as u32;
    // SAFETY: grad length equals the forward output length, output length
    // equals the original input length, and each thread sums only the run of
    // repeated gradient elements mapped to its source element.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(grad.inner())
            .arg(out.inner_mut())
            .arg(&dim_u32)
            .arg(&inner_u32)
            .arg(&repeats_u32)
            .arg(&total_u32)
            .launch(launch_config(total))?;
    }
    Ok(out)
}

/// Backward repeat-interleave segment-sum for f64 CUDA buffers.
#[cfg(feature = "cuda")]
pub fn gpu_repeat_interleave_backward_f64(
    grad: &CudaBuffer<f64>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    repeats: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    if grad.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: device.ordinal(),
            got: grad.device_ordinal(),
        });
    }
    let total = checked_backward_len(
        grad.len(),
        outer,
        dim_size,
        inner,
        repeats,
        "repeat_interleave_backward_f64",
    )?;
    let f = crate::module_cache::get_or_compile(
        device.context(),
        RI_BWD_F64_PTX,
        "repeat_interleave_backward_f64_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "repeat_interleave_backward_f64_kernel",
        source: e,
    })?;
    let mut out = alloc_zeros_f64(total, device)?;
    if total == 0 {
        return Ok(out);
    }
    let dim_u32 = dim_size as u32;
    let inner_u32 = inner as u32;
    let repeats_u32 = repeats as u32;
    let total_u32 = total as u32;
    // SAFETY: same invariant as `gpu_repeat_interleave_backward_f32`; element
    // width and accumulator type differ only.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(grad.inner())
            .arg(out.inner_mut())
            .arg(&dim_u32)
            .arg(&inner_u32)
            .arg(&repeats_u32)
            .arg(&total_u32)
            .launch(launch_config(total))?;
    }
    Ok(out)
}

#[cfg(feature = "cuda")]
fn launch_backward_u16(
    grad: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    repeats: usize,
    device: &GpuDevice,
    ptx: &'static str,
    kernel: &'static str,
) -> GpuResult<CudaSlice<u16>> {
    let total = checked_backward_len(grad.len(), outer, dim_size, inner, repeats, kernel)?;
    let f =
        crate::module_cache::get_or_compile(device.context(), ptx, kernel, device.ordinal() as u32)
            .map_err(|e| GpuError::PtxCompileFailed { kernel, source: e })?;
    let mut out = alloc_zeros_bf16(total, device)?;
    if total == 0 {
        return Ok(out);
    }
    let dim_u32 = dim_size as u32;
    let inner_u32 = inner as u32;
    let repeats_u32 = repeats as u32;
    let total_u32 = total as u32;
    // SAFETY: same segment-sum invariant as f32/f64; values are decoded to
    // f32 accumulators and rounded back to the requested 16-bit dtype.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(grad)
            .arg(&mut out)
            .arg(&dim_u32)
            .arg(&inner_u32)
            .arg(&repeats_u32)
            .arg(&total_u32)
            .launch(launch_config(total))?;
    }
    Ok(out)
}

/// Backward repeat-interleave segment-sum for f16 CUDA buffers.
#[cfg(feature = "cuda")]
pub fn gpu_repeat_interleave_backward_f16(
    grad: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    repeats: usize,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch_backward_u16(
        grad,
        outer,
        dim_size,
        inner,
        repeats,
        device,
        RI_BWD_F16_PTX,
        "repeat_interleave_backward_f16_kernel",
    )
}

/// Backward repeat-interleave segment-sum for bf16 CUDA buffers.
#[cfg(feature = "cuda")]
pub fn gpu_repeat_interleave_backward_bf16(
    grad: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    repeats: usize,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch_backward_u16(
        grad,
        outer,
        dim_size,
        inner,
        repeats,
        device,
        RI_BWD_BF16_PTX,
        "repeat_interleave_backward_bf16_kernel",
    )
}
