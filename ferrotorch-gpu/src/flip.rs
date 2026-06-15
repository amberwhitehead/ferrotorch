//! CUDA kernels for `torch.flip`-style axis reversal.
//!
//! `flip` is a pure relocating gather: no arithmetic is performed on tensor
//! values, so f32, f64, f16, and bf16 all share the same index transform with
//! only the element width changing. The f16/bf16 entry point copies `u16` bit
//! patterns and relies on the caller's dtype tag to distinguish the two types,
//! matching PyTorch's `ScalarType`-over-storage design.

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
macro_rules! flip_ptx {
    ($name:literal, $shift:literal, $load:literal, $store:literal, $vreg:literal) => {
        concat!(
            ".version 7.0\n.target sm_52\n.address_size 64\n",
            ".visible .entry ",
            $name,
            "(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u64 meta_ptr,
    .param .u32 total,
    .param .u32 rank
) {
    .reg .u32 %idx, %bid, %bdim, %total_r, %rank_r, %d, %meta_idx;
    .reg .u32 %rem, %src_idx, %coord, %src_coord, %shape_d, %stride_d;
    .reg .u32 %flag, %tmp;
    .reg .u64 %in_p, %out_p, %meta_p, %off, %addr;
    .reg ",
            $vreg,
            " %val;
    .reg .pred %p;

    ld.param.u64 %in_p,    [in_ptr];
    ld.param.u64 %out_p,   [out_ptr];
    ld.param.u64 %meta_p,  [meta_ptr];
    ld.param.u32 %total_r, [total];
    ld.param.u32 %rank_r,  [rank];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %total_r;
    @%p bra DONE;

    mov.u32 %rem, %idx;
    mov.u32 %src_idx, 0;
    mov.u32 %d, 0;

LOOP:
    setp.ge.u32 %p, %d, %rank_r;
    @%p bra COPY;

    cvt.u64.u32 %off, %d;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %meta_p, %off;
    ld.global.u32 %shape_d, [%addr];

    add.u32 %meta_idx, %rank_r, %d;
    cvt.u64.u32 %off, %meta_idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %meta_p, %off;
    ld.global.u32 %stride_d, [%addr];

    div.u32 %coord, %rem, %stride_d;
    mul.lo.u32 %tmp, %coord, %stride_d;
    sub.u32 %rem, %rem, %tmp;

    add.u32 %meta_idx, %rank_r, %rank_r;
    add.u32 %meta_idx, %meta_idx, %d;
    cvt.u64.u32 %off, %meta_idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %meta_p, %off;
    ld.global.u32 %flag, [%addr];

    mov.u32 %src_coord, %coord;
    setp.eq.u32 %p, %flag, 0;
    @%p bra ACCUM;
    sub.u32 %tmp, %shape_d, 1;
    sub.u32 %src_coord, %tmp, %coord;

ACCUM:
    mad.lo.u32 %src_idx, %src_coord, %stride_d, %src_idx;
    add.u32 %d, %d, 1;
    bra LOOP;

COPY:
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
const FLIP_F32_PTX: &str = flip_ptx!(
    "flip_f32_kernel",
    "2",
    "ld.global.b32",
    "st.global.b32",
    ".b32"
);
#[cfg(feature = "cuda")]
const FLIP_F64_PTX: &str = flip_ptx!(
    "flip_f64_kernel",
    "3",
    "ld.global.b64",
    "st.global.b64",
    ".b64"
);
#[cfg(feature = "cuda")]
const FLIP_U16_PTX: &str = flip_ptx!(
    "flip_u16_kernel",
    "1",
    "ld.global.b16",
    "st.global.b16",
    ".b16"
);

#[cfg(feature = "cuda")]
fn checked_numel(shape: &[usize], op: &'static str) -> GpuResult<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| GpuError::ShapeMismatch {
            op,
            expected: vec![usize::MAX],
            got: shape.to_vec(),
        })
    })
}

#[cfg(feature = "cuda")]
fn checked_metadata(shape: &[usize], dims: &[usize], op: &'static str) -> GpuResult<Vec<u32>> {
    let mut strides = vec![1usize; shape.len()];
    let mut acc = 1usize;
    for (idx, &dim) in shape.iter().enumerate().rev() {
        strides[idx] = acc;
        acc = acc
            .checked_mul(dim)
            .ok_or_else(|| GpuError::ShapeMismatch {
                op,
                expected: vec![usize::MAX],
                got: shape.to_vec(),
            })?;
    }

    let mut flags = vec![0u32; shape.len()];
    for &dim in dims {
        if dim >= shape.len() {
            return Err(GpuError::ShapeMismatch {
                op,
                expected: vec![shape.len()],
                got: vec![dim],
            });
        }
        flags[dim] = 1;
    }

    let mut meta = Vec::with_capacity(shape.len() * 3);
    for &dim in shape {
        meta.push(u32::try_from(dim).map_err(|_| GpuError::ShapeMismatch {
            op,
            expected: vec![u32::MAX as usize],
            got: vec![dim],
        })?);
    }
    for stride in strides {
        meta.push(u32::try_from(stride).map_err(|_| GpuError::ShapeMismatch {
            op,
            expected: vec![u32::MAX as usize],
            got: vec![stride],
        })?);
    }
    meta.extend(flags);
    Ok(meta)
}

#[cfg(feature = "cuda")]
fn validate_len(
    input_len: usize,
    input_device: Option<usize>,
    shape: &[usize],
    device: &GpuDevice,
    op: &'static str,
) -> GpuResult<usize> {
    if let Some(input_device) = input_device
        && input_device != device.ordinal()
    {
        return Err(GpuError::DeviceMismatch {
            expected: device.ordinal(),
            got: input_device,
        });
    }

    let total = checked_numel(shape, op)?;
    if input_len != total {
        return Err(GpuError::ShapeMismatch {
            op,
            expected: shape.to_vec(),
            got: vec![input_len],
        });
    }
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

/// Flip a contiguous f32 CUDA buffer along normalized dimensions.
#[cfg(feature = "cuda")]
pub fn gpu_flip_f32(
    input: &CudaBuffer<f32>,
    shape: &[usize],
    dims: &[usize],
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    let total = validate_len(
        input.len(),
        Some(input.device_ordinal()),
        shape,
        device,
        "flip_f32",
    )?;
    if total == 0 {
        return alloc_zeros_f32(0, device);
    }

    let f = crate::module_cache::get_or_compile(
        device.context(),
        FLIP_F32_PTX,
        "flip_f32_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "flip_f32_kernel",
        source: e,
    })?;
    let meta = checked_metadata(shape, dims, "flip_f32")?;
    let meta_dev = device.stream().clone_htod(&meta)?;
    let mut out = alloc_zeros_f32(total, device)?;
    let total_u32 = total as u32;
    let rank_u32 = shape.len() as u32;

    // SAFETY:
    // - The launch ABI matches `flip_f32_kernel` exactly:
    //   `(input, output, metadata, total, rank)`.
    // - `input` is validated to be on `device` and exactly `shape.numel()`
    //   elements. `out` is freshly allocated to the same element count.
    // - `metadata` contains `rank` u32 shapes, `rank` C-order strides, and
    //   `rank` flip flags. All shape/stride/total values are checked to fit
    //   the kernel's u32 arithmetic before launch.
    // - For every in-bounds output flat index, the coordinate decomposition
    //   and optional coordinate reversal produce a source element in
    //   `[0, total)`, so every load and store is in-bounds.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&meta_dev)
            .arg(&total_u32)
            .arg(&rank_u32)
            .launch(launch_config(total))?;
    }

    Ok(out)
}

/// Flip a contiguous f64 CUDA buffer along normalized dimensions.
#[cfg(feature = "cuda")]
pub fn gpu_flip_f64(
    input: &CudaBuffer<f64>,
    shape: &[usize],
    dims: &[usize],
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    let total = validate_len(
        input.len(),
        Some(input.device_ordinal()),
        shape,
        device,
        "flip_f64",
    )?;
    if total == 0 {
        return alloc_zeros_f64(0, device);
    }

    let f = crate::module_cache::get_or_compile(
        device.context(),
        FLIP_F64_PTX,
        "flip_f64_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "flip_f64_kernel",
        source: e,
    })?;
    let meta = checked_metadata(shape, dims, "flip_f64")?;
    let meta_dev = device.stream().clone_htod(&meta)?;
    let mut out = alloc_zeros_f64(total, device)?;
    let total_u32 = total as u32;
    let rank_u32 = shape.len() as u32;

    // SAFETY: same address-map invariant as `gpu_flip_f32`; this path only
    // changes element width and copies raw 64-bit payloads.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&meta_dev)
            .arg(&total_u32)
            .arg(&rank_u32)
            .launch(launch_config(total))?;
    }

    Ok(out)
}

/// Flip a contiguous f16/bf16 CUDA buffer along normalized dimensions.
#[cfg(feature = "cuda")]
pub fn gpu_flip_u16(
    input: &CudaSlice<u16>,
    shape: &[usize],
    dims: &[usize],
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    let total = validate_len(input.len(), None, shape, device, "flip_u16")?;
    if total == 0 {
        return alloc_zeros_bf16(0, device);
    }

    let f = crate::module_cache::get_or_compile(
        device.context(),
        FLIP_U16_PTX,
        "flip_u16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "flip_u16_kernel",
        source: e,
    })?;
    let meta = checked_metadata(shape, dims, "flip_u16")?;
    let meta_dev = device.stream().clone_htod(&meta)?;
    let mut out = alloc_zeros_bf16(total, device)?;
    let total_u32 = total as u32;
    let rank_u32 = shape.len() as u32;

    // SAFETY: same address-map invariant as `gpu_flip_f32`; f16/bf16 payloads
    // are copied as raw u16 bit patterns and re-tagged by the backend wrapper.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(input)
            .arg(&mut out)
            .arg(&meta_dev)
            .arg(&total_u32)
            .arg(&rank_u32)
            .launch(launch_config(total))?;
    }

    Ok(out)
}
