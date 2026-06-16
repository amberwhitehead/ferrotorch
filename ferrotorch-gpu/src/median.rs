//! CUDA `median(dim)` / `nanmedian(dim)` kernels.
//!
//! PyTorch lowers the CUDA indexed median path through
//! `/home/doll/pytorch/aten/src/ATen/native/cuda/Sorting.cpp` into
//! `launch_median_kernel` in `Sorting.cu`: validate the reduced dimension,
//! keep values and int64 indices on CUDA, count NaNs per slice, select the
//! lower median rank, then find an index of the selected value. PyTorch marks
//! duplicate median index choice as nondeterministic; this module chooses the
//! lowest matching index so autograd routing is stable while preserving the
//! documented value semantics.

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits};

use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};
use crate::module_cache::get_or_compile;

const BLOCK_SIZE: u32 = 256;

macro_rules! median_ptx {
    ($entry:literal, $ty:literal, $shift:literal) => {
        concat!(
            ".version 7.0\n",
            ".target sm_52\n",
            ".address_size 64\n\n",
            ".visible .entry ",
            $entry,
            "(\n",
            "    .param .u64 in_ptr,\n",
            "    .param .u64 vals_ptr,\n",
            "    .param .u64 idx_ptr,\n",
            "    .param .u32 outer,\n",
            "    .param .u32 dim,\n",
            "    .param .u32 inner,\n",
            "    .param .u32 total,\n",
            "    .param .u32 ignore_nan\n",
            ") {\n",
            "    .reg .u32 %tid_r, %bid_r, %bdim_r, %s, %no, %nd, %ni, %tot, %ign;\n",
            "    .reg .u32 %outer_idx, %inner_idx, %d, %scan, %base, %elem, %tmpu;\n",
            "    .reg .u32 %num_nan, %first_nan, %effective, %target, %rank, %cand_idx;\n",
            "    .reg .u64 %in_p, %vp, %ip, %off, %addr, %idx_off;\n",
            "    .reg .",
            $ty,
            " %v, %cand;\n",
            "    .reg .s64 %idx64;\n",
            "    .reg .pred %p_oob, %p_loop, %p_isnan, %p_has_nan, %p_ignore, %p_not_ignore;\n",
            "    .reg .pred %p_store_nan, %p_all_nan, %p_cand_loop, %p_scan_loop, %p_skip;\n",
            "    .reg .pred %p_less, %p_eq, %p_rank;\n\n",
            "    ld.param.u64 %in_p, [in_ptr];\n",
            "    ld.param.u64 %vp,   [vals_ptr];\n",
            "    ld.param.u64 %ip,   [idx_ptr];\n",
            "    ld.param.u32 %no,   [outer];\n",
            "    ld.param.u32 %nd,   [dim];\n",
            "    ld.param.u32 %ni,   [inner];\n",
            "    ld.param.u32 %tot,  [total];\n",
            "    ld.param.u32 %ign,  [ignore_nan];\n\n",
            "    mov.u32 %tid_r,  %tid.x;\n",
            "    mov.u32 %bid_r,  %ctaid.x;\n",
            "    mov.u32 %bdim_r, %ntid.x;\n",
            "    mad.lo.u32 %s, %bid_r, %bdim_r, %tid_r;\n",
            "    setp.ge.u32 %p_oob, %s, %tot;\n",
            "    @%p_oob bra DONE;\n\n",
            "    div.u32 %outer_idx, %s, %ni;\n",
            "    rem.u32 %inner_idx, %s, %ni;\n",
            "    mul.lo.u32 %base, %outer_idx, %nd;\n",
            "    mul.lo.u32 %base, %base, %ni;\n",
            "    add.u32 %base, %base, %inner_idx;\n",
            "    setp.ne.u32 %p_ignore, %ign, 0;\n",
            "    not.pred %p_not_ignore, %p_ignore;\n\n",
            "    mov.u32 %num_nan, 0;\n",
            "    mov.u32 %first_nan, 0;\n",
            "    mov.pred %p_has_nan, 0;\n",
            "    mov.u32 %d, 0;\n",
            "COUNT_LOOP:\n",
            "    setp.ge.u32 %p_loop, %d, %nd;\n",
            "    @%p_loop bra COUNT_DONE;\n",
            "    mul.lo.u32 %elem, %d, %ni;\n",
            "    add.u32 %elem, %base, %elem;\n",
            "    cvt.u64.u32 %off, %elem;\n",
            "    shl.b64 %off, %off, ",
            $shift,
            ";\n",
            "    add.u64 %addr, %in_p, %off;\n",
            "    ld.global.",
            $ty,
            " %v, [%addr];\n",
            "    testp.notanumber.",
            $ty,
            " %p_isnan, %v;\n",
            "    @!%p_isnan bra COUNT_NEXT;\n",
            "    @!%p_has_nan mov.u32 %first_nan, %d;\n",
            "    @!%p_has_nan mov.pred %p_has_nan, 1;\n",
            "    add.u32 %num_nan, %num_nan, 1;\n",
            "COUNT_NEXT:\n",
            "    add.u32 %d, %d, 1;\n",
            "    bra COUNT_LOOP;\n\n",
            "COUNT_DONE:\n",
            "    and.pred %p_store_nan, %p_not_ignore, %p_has_nan;\n",
            "    setp.eq.u32 %p_all_nan, %num_nan, %nd;\n",
            "    and.pred %p_all_nan, %p_all_nan, %p_ignore;\n",
            "    or.pred %p_store_nan, %p_store_nan, %p_all_nan;\n",
            "    @%p_store_nan bra STORE_FIRST_NAN;\n\n",
            "    mov.u32 %effective, %nd;\n",
            "    @%p_ignore sub.u32 %effective, %nd, %num_nan;\n",
            "    sub.u32 %target, %effective, 1;\n",
            "    shr.u32 %target, %target, 1;\n",
            "    mov.u32 %cand_idx, 0;\n",
            "CAND_LOOP:\n",
            "    setp.ge.u32 %p_cand_loop, %cand_idx, %nd;\n",
            "    @%p_cand_loop bra DONE;\n",
            "    mul.lo.u32 %elem, %cand_idx, %ni;\n",
            "    add.u32 %elem, %base, %elem;\n",
            "    cvt.u64.u32 %off, %elem;\n",
            "    shl.b64 %off, %off, ",
            $shift,
            ";\n",
            "    add.u64 %addr, %in_p, %off;\n",
            "    ld.global.",
            $ty,
            " %cand, [%addr];\n",
            "    testp.notanumber.",
            $ty,
            " %p_isnan, %cand;\n",
            "    @%p_isnan bra CAND_NEXT;\n",
            "    mov.u32 %rank, 0;\n",
            "    mov.u32 %tmpu, 0;\n",
            "    mov.u32 %scan, 0;\n",
            "SCAN_LOOP:\n",
            "    setp.ge.u32 %p_scan_loop, %scan, %nd;\n",
            "    @%p_scan_loop bra SCAN_DONE;\n",
            "    mul.lo.u32 %elem, %scan, %ni;\n",
            "    add.u32 %elem, %base, %elem;\n",
            "    cvt.u64.u32 %off, %elem;\n",
            "    shl.b64 %off, %off, ",
            $shift,
            ";\n",
            "    add.u64 %addr, %in_p, %off;\n",
            "    ld.global.",
            $ty,
            " %v, [%addr];\n",
            "    testp.notanumber.",
            $ty,
            " %p_skip, %v;\n",
            "    @%p_skip bra SCAN_NEXT;\n",
            "    setp.lt.",
            $ty,
            " %p_less, %v, %cand;\n",
            "    @%p_less add.u32 %rank, %rank, 1;\n",
            "    setp.eq.",
            $ty,
            " %p_eq, %v, %cand;\n",
            "    @%p_eq add.u32 %tmpu, %tmpu, 1;\n",
            "SCAN_NEXT:\n",
            "    add.u32 %scan, %scan, 1;\n",
            "    bra SCAN_LOOP;\n\n",
            "SCAN_DONE:\n",
            "    setp.gt.u32 %p_rank, %rank, %target;\n",
            "    @%p_rank bra CAND_NEXT;\n",
            "    add.u32 %tmpu, %rank, %tmpu;\n",
            "    setp.ge.u32 %p_rank, %target, %tmpu;\n",
            "    @%p_rank bra CAND_NEXT;\n",
            "    cvt.u64.u32 %off, %s;\n",
            "    shl.b64 %addr, %off, ",
            $shift,
            ";\n",
            "    add.u64 %addr, %vp, %addr;\n",
            "    st.global.",
            $ty,
            " [%addr], %cand;\n",
            "    shl.b64 %idx_off, %off, 3;\n",
            "    add.u64 %addr, %ip, %idx_off;\n",
            "    cvt.s64.u32 %idx64, %cand_idx;\n",
            "    st.global.s64 [%addr], %idx64;\n",
            "    bra DONE;\n\n",
            "CAND_NEXT:\n",
            "    add.u32 %cand_idx, %cand_idx, 1;\n",
            "    bra CAND_LOOP;\n\n",
            "STORE_FIRST_NAN:\n",
            "    mul.lo.u32 %elem, %first_nan, %ni;\n",
            "    add.u32 %elem, %base, %elem;\n",
            "    cvt.u64.u32 %off, %elem;\n",
            "    shl.b64 %off, %off, ",
            $shift,
            ";\n",
            "    add.u64 %addr, %in_p, %off;\n",
            "    ld.global.",
            $ty,
            " %v, [%addr];\n",
            "    cvt.u64.u32 %off, %s;\n",
            "    shl.b64 %addr, %off, ",
            $shift,
            ";\n",
            "    add.u64 %addr, %vp, %addr;\n",
            "    st.global.",
            $ty,
            " [%addr], %v;\n",
            "    shl.b64 %idx_off, %off, 3;\n",
            "    add.u64 %addr, %ip, %idx_off;\n",
            "    cvt.s64.u32 %idx64, %first_nan;\n",
            "    st.global.s64 [%addr], %idx64;\n",
            "DONE:\n",
            "    ret;\n",
            "}\n"
        )
    };
}

macro_rules! median_16_ptx {
    ($entry:literal, $version:literal, $target:literal, $decode_v:literal, $decode_cand:literal) => {
        concat!(
            ".version ",
            $version,
            "\n.target ",
            $target,
            "\n.address_size 64\n\n",
            ".visible .entry ",
            $entry,
            "(\n",
            "    .param .u64 in_ptr,\n",
            "    .param .u64 vals_ptr,\n",
            "    .param .u64 idx_ptr,\n",
            "    .param .u32 outer,\n",
            "    .param .u32 dim,\n",
            "    .param .u32 inner,\n",
            "    .param .u32 total,\n",
            "    .param .u32 ignore_nan\n",
            ") {\n",
            "    .reg .u32 %tid_r, %bid_r, %bdim_r, %s, %no, %nd, %ni, %tot, %ign;\n",
            "    .reg .u32 %outer_idx, %inner_idx, %d, %scan, %base, %elem, %tmpu;\n",
            "    .reg .u32 %num_nan, %first_nan, %effective, %target, %rank, %cand_idx;\n",
            "    .reg .u32 %v_bits, %cand_bits;\n",
            "    .reg .u64 %in_p, %vp, %ip, %off, %addr, %idx_off;\n",
            "    .reg .b16 %v_raw, %cand_raw, %zero16;\n",
            "    .reg .f32 %v, %cand;\n",
            "    .reg .s64 %idx64;\n",
            "    .reg .pred %p_oob, %p_loop, %p_isnan, %p_has_nan, %p_ignore, %p_not_ignore;\n",
            "    .reg .pred %p_store_nan, %p_all_nan, %p_cand_loop, %p_scan_loop, %p_skip;\n",
            "    .reg .pred %p_less, %p_eq, %p_rank;\n\n",
            "    ld.param.u64 %in_p, [in_ptr];\n",
            "    ld.param.u64 %vp,   [vals_ptr];\n",
            "    ld.param.u64 %ip,   [idx_ptr];\n",
            "    ld.param.u32 %no,   [outer];\n",
            "    ld.param.u32 %nd,   [dim];\n",
            "    ld.param.u32 %ni,   [inner];\n",
            "    ld.param.u32 %tot,  [total];\n",
            "    ld.param.u32 %ign,  [ignore_nan];\n\n",
            "    mov.u32 %tid_r,  %tid.x;\n",
            "    mov.u32 %bid_r,  %ctaid.x;\n",
            "    mov.u32 %bdim_r, %ntid.x;\n",
            "    mad.lo.u32 %s, %bid_r, %bdim_r, %tid_r;\n",
            "    setp.ge.u32 %p_oob, %s, %tot;\n",
            "    @%p_oob bra DONE;\n",
            "    mov.b16 %zero16, 0;\n\n",
            "    div.u32 %outer_idx, %s, %ni;\n",
            "    rem.u32 %inner_idx, %s, %ni;\n",
            "    mul.lo.u32 %base, %outer_idx, %nd;\n",
            "    mul.lo.u32 %base, %base, %ni;\n",
            "    add.u32 %base, %base, %inner_idx;\n",
            "    setp.ne.u32 %p_ignore, %ign, 0;\n",
            "    not.pred %p_not_ignore, %p_ignore;\n\n",
            "    mov.u32 %num_nan, 0;\n",
            "    mov.u32 %first_nan, 0;\n",
            "    mov.pred %p_has_nan, 0;\n",
            "    mov.u32 %d, 0;\n",
            "COUNT_LOOP:\n",
            "    setp.ge.u32 %p_loop, %d, %nd;\n",
            "    @%p_loop bra COUNT_DONE;\n",
            "    mul.lo.u32 %elem, %d, %ni;\n",
            "    add.u32 %elem, %base, %elem;\n",
            "    cvt.u64.u32 %off, %elem;\n",
            "    shl.b64 %off, %off, 1;\n",
            "    add.u64 %addr, %in_p, %off;\n",
            "    ld.global.b16 %v_raw, [%addr];\n",
            $decode_v,
            "\n",
            "    testp.notanumber.f32 %p_isnan, %v;\n",
            "    @!%p_isnan bra COUNT_NEXT;\n",
            "    @!%p_has_nan mov.u32 %first_nan, %d;\n",
            "    @!%p_has_nan mov.pred %p_has_nan, 1;\n",
            "    add.u32 %num_nan, %num_nan, 1;\n",
            "COUNT_NEXT:\n",
            "    add.u32 %d, %d, 1;\n",
            "    bra COUNT_LOOP;\n\n",
            "COUNT_DONE:\n",
            "    and.pred %p_store_nan, %p_not_ignore, %p_has_nan;\n",
            "    setp.eq.u32 %p_all_nan, %num_nan, %nd;\n",
            "    and.pred %p_all_nan, %p_all_nan, %p_ignore;\n",
            "    or.pred %p_store_nan, %p_store_nan, %p_all_nan;\n",
            "    @%p_store_nan bra STORE_FIRST_NAN;\n\n",
            "    mov.u32 %effective, %nd;\n",
            "    @%p_ignore sub.u32 %effective, %nd, %num_nan;\n",
            "    sub.u32 %target, %effective, 1;\n",
            "    shr.u32 %target, %target, 1;\n",
            "    mov.u32 %cand_idx, 0;\n",
            "CAND_LOOP:\n",
            "    setp.ge.u32 %p_cand_loop, %cand_idx, %nd;\n",
            "    @%p_cand_loop bra DONE;\n",
            "    mul.lo.u32 %elem, %cand_idx, %ni;\n",
            "    add.u32 %elem, %base, %elem;\n",
            "    cvt.u64.u32 %off, %elem;\n",
            "    shl.b64 %off, %off, 1;\n",
            "    add.u64 %addr, %in_p, %off;\n",
            "    ld.global.b16 %cand_raw, [%addr];\n",
            $decode_cand,
            "\n",
            "    testp.notanumber.f32 %p_isnan, %cand;\n",
            "    @%p_isnan bra CAND_NEXT;\n",
            "    mov.u32 %rank, 0;\n",
            "    mov.u32 %tmpu, 0;\n",
            "    mov.u32 %scan, 0;\n",
            "SCAN_LOOP:\n",
            "    setp.ge.u32 %p_scan_loop, %scan, %nd;\n",
            "    @%p_scan_loop bra SCAN_DONE;\n",
            "    mul.lo.u32 %elem, %scan, %ni;\n",
            "    add.u32 %elem, %base, %elem;\n",
            "    cvt.u64.u32 %off, %elem;\n",
            "    shl.b64 %off, %off, 1;\n",
            "    add.u64 %addr, %in_p, %off;\n",
            "    ld.global.b16 %v_raw, [%addr];\n",
            $decode_v,
            "\n",
            "    testp.notanumber.f32 %p_skip, %v;\n",
            "    @%p_skip bra SCAN_NEXT;\n",
            "    setp.lt.f32 %p_less, %v, %cand;\n",
            "    @%p_less add.u32 %rank, %rank, 1;\n",
            "    setp.eq.f32 %p_eq, %v, %cand;\n",
            "    @%p_eq add.u32 %tmpu, %tmpu, 1;\n",
            "SCAN_NEXT:\n",
            "    add.u32 %scan, %scan, 1;\n",
            "    bra SCAN_LOOP;\n\n",
            "SCAN_DONE:\n",
            "    setp.gt.u32 %p_rank, %rank, %target;\n",
            "    @%p_rank bra CAND_NEXT;\n",
            "    add.u32 %tmpu, %rank, %tmpu;\n",
            "    setp.ge.u32 %p_rank, %target, %tmpu;\n",
            "    @%p_rank bra CAND_NEXT;\n",
            "    cvt.u64.u32 %off, %s;\n",
            "    shl.b64 %addr, %off, 1;\n",
            "    add.u64 %addr, %vp, %addr;\n",
            "    st.global.b16 [%addr], %cand_raw;\n",
            "    shl.b64 %idx_off, %off, 3;\n",
            "    add.u64 %addr, %ip, %idx_off;\n",
            "    cvt.s64.u32 %idx64, %cand_idx;\n",
            "    st.global.s64 [%addr], %idx64;\n",
            "    bra DONE;\n\n",
            "CAND_NEXT:\n",
            "    add.u32 %cand_idx, %cand_idx, 1;\n",
            "    bra CAND_LOOP;\n\n",
            "STORE_FIRST_NAN:\n",
            "    mul.lo.u32 %elem, %first_nan, %ni;\n",
            "    add.u32 %elem, %base, %elem;\n",
            "    cvt.u64.u32 %off, %elem;\n",
            "    shl.b64 %off, %off, 1;\n",
            "    add.u64 %addr, %in_p, %off;\n",
            "    ld.global.b16 %v_raw, [%addr];\n",
            "    cvt.u64.u32 %off, %s;\n",
            "    shl.b64 %addr, %off, 1;\n",
            "    add.u64 %addr, %vp, %addr;\n",
            "    st.global.b16 [%addr], %v_raw;\n",
            "    shl.b64 %idx_off, %off, 3;\n",
            "    add.u64 %addr, %ip, %idx_off;\n",
            "    cvt.s64.u32 %idx64, %first_nan;\n",
            "    st.global.s64 [%addr], %idx64;\n",
            "DONE:\n",
            "    ret;\n",
            "}\n"
        )
    };
}

const MEDIAN_F32_PTX: &str = median_ptx!("median_f32_kernel", "f32", "2");
const MEDIAN_F64_PTX: &str = median_ptx!("median_f64_kernel", "f64", "3");
const MEDIAN_F16_PTX: &str = median_16_ptx!(
    "median_f16_kernel",
    "7.0",
    "sm_60",
    "    cvt.f32.f16 %v, %v_raw;",
    "    cvt.f32.f16 %cand, %cand_raw;"
);
const MEDIAN_BF16_PTX: &str = median_16_ptx!(
    "median_bf16_kernel",
    "7.8",
    "sm_80",
    "    mov.b32 %v_bits, {%zero16, %v_raw}; mov.b32 %v, %v_bits;",
    "    mov.b32 %cand_bits, {%zero16, %cand_raw}; mov.b32 %cand, %cand_bits;"
);

fn launch_config(lanes: usize) -> LaunchConfig {
    let grid = ((lanes as u32).saturating_add(BLOCK_SIZE - 1)) / BLOCK_SIZE;
    LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_median<V>(
    input: &CudaSlice<V>,
    outer: usize,
    dim: usize,
    inner: usize,
    ignore_nan: bool,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
) -> GpuResult<(CudaSlice<V>, CudaSlice<i64>)>
where
    V: DeviceRepr + ValidAsZeroBits,
{
    if dim == 0 {
        return Err(GpuError::InvalidState {
            message: "median_dim: reduced dimension must be non-empty".into(),
        });
    }
    let lanes = outer
        .checked_mul(inner)
        .ok_or_else(|| GpuError::InvalidState {
            message: format!("median_dim: lane extent {outer} * {inner} overflows usize"),
        })?;
    let expected_input = outer
        .checked_mul(dim)
        .and_then(|n| n.checked_mul(inner))
        .ok_or_else(|| GpuError::InvalidState {
            message: format!("median_dim: input extent {outer} * {dim} * {inner} overflows usize"),
        })?;
    if input.len() < expected_input {
        return Err(GpuError::LengthMismatch {
            a: input.len(),
            b: expected_input,
        });
    }
    if outer > u32::MAX as usize
        || dim > u32::MAX as usize
        || inner > u32::MAX as usize
        || lanes > u32::MAX as usize
        || expected_input > u32::MAX as usize
    {
        return Err(GpuError::LengthMismatch {
            a: outer.max(dim).max(inner).max(lanes).max(expected_input),
            b: u32::MAX as usize,
        });
    }

    let stream = device.stream();
    if lanes == 0 {
        return Ok((stream.alloc_zeros::<V>(0)?, stream.alloc_zeros::<i64>(0)?));
    }

    let f = get_or_compile(device.context(), ptx, kernel_name, device.ordinal() as u32).map_err(
        |e| GpuError::PtxCompileFailed {
            kernel: kernel_name,
            source: e,
        },
    )?;
    let mut out_vals = stream.alloc_zeros::<V>(lanes)?;
    let mut out_idx = stream.alloc_zeros::<i64>(lanes)?;
    let outer_u = outer as u32;
    let dim_u = dim as u32;
    let inner_u = inner as u32;
    let total_u = lanes as u32;
    let ignore_u = u32::from(ignore_nan);
    let cfg = launch_config(lanes);

    // SAFETY:
    // - `f` is the PTX entry `kernel_name`; the pushed parameters match the
    //   ABI `(in_ptr, vals_ptr, idx_ptr, outer, dim, inner, total, ignore_nan)`.
    // - `input` has at least `outer * dim * inner` elements (checked above).
    //   Each lane reads only `base + d * inner` for `d in [0, dim)`.
    // - `out_vals` and `out_idx` are fresh `outer * inner` buffers. One lane
    //   writes exactly one value and one index at offset `s`.
    // - All products and scalar launch parameters are range-checked for the
    //   kernel's u32 index arithmetic.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(&mut out_vals)
            .arg(&mut out_idx)
            .arg(&outer_u)
            .arg(&dim_u)
            .arg(&inner_u)
            .arg(&total_u)
            .arg(&ignore_u)
            .launch(cfg)?;
    }
    Ok((out_vals, out_idx))
}

/// Compute `median(input, dim)` or `nanmedian(input, dim)` for f32 CUDA data.
#[allow(clippy::too_many_arguments)]
pub fn gpu_median_dim_f32(
    input: &CudaSlice<f32>,
    outer: usize,
    dim: usize,
    inner: usize,
    ignore_nan: bool,
    device: &GpuDevice,
) -> GpuResult<(CudaSlice<f32>, CudaSlice<i64>)> {
    launch_median(
        input,
        outer,
        dim,
        inner,
        ignore_nan,
        device,
        MEDIAN_F32_PTX,
        "median_f32_kernel",
    )
}

/// Compute `median(input, dim)` or `nanmedian(input, dim)` for f64 CUDA data.
#[allow(clippy::too_many_arguments)]
pub fn gpu_median_dim_f64(
    input: &CudaSlice<f64>,
    outer: usize,
    dim: usize,
    inner: usize,
    ignore_nan: bool,
    device: &GpuDevice,
) -> GpuResult<(CudaSlice<f64>, CudaSlice<i64>)> {
    launch_median(
        input,
        outer,
        dim,
        inner,
        ignore_nan,
        device,
        MEDIAN_F64_PTX,
        "median_f64_kernel",
    )
}

/// Compute `median(input, dim)` or `nanmedian(input, dim)` for f16 CUDA data.
#[allow(clippy::too_many_arguments)]
pub fn gpu_median_dim_f16(
    input: &CudaSlice<u16>,
    outer: usize,
    dim: usize,
    inner: usize,
    ignore_nan: bool,
    device: &GpuDevice,
) -> GpuResult<(CudaSlice<u16>, CudaSlice<i64>)> {
    launch_median(
        input,
        outer,
        dim,
        inner,
        ignore_nan,
        device,
        MEDIAN_F16_PTX,
        "median_f16_kernel",
    )
}

/// Compute `median(input, dim)` or `nanmedian(input, dim)` for bf16 CUDA data.
#[allow(clippy::too_many_arguments)]
pub fn gpu_median_dim_bf16(
    input: &CudaSlice<u16>,
    outer: usize,
    dim: usize,
    inner: usize,
    ignore_nan: bool,
    device: &GpuDevice,
) -> GpuResult<(CudaSlice<u16>, CudaSlice<i64>)> {
    launch_median(
        input,
        outer,
        dim,
        inner,
        ignore_nan,
        device,
        MEDIAN_BF16_PTX,
        "median_bf16_kernel",
    )
}
