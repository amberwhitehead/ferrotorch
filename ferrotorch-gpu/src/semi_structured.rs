//! CUDA kernels for PyTorch-style semi-structured sparse packing.
//!
//! The public `torch.sparse.to_sparse_semi_structured(..., _FORCE_CUTLASS=True)`
//! conversion stores retained values separately from CUTLASS metadata. For
//! f16/bf16 and f32 the metadata dtype is `int16`; the layout is the CUTLASS
//! `ColumnMajorInterleaved<2>` reorder used by PyTorch's
//! `torch/sparse/_semi_structured_conversions.py`.
//!
//! These kernels intentionally do not read source tensors back to host. They
//! mirror PyTorch's converter behavior for malformed sparsity patterns: the
//! boolean minimization selects a hardware-valid pair per group rather than
//! raising, so dense inputs with more or fewer nonzeros still produce a packed
//! representation.

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::buffer::CudaBuffer;
use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};
use crate::module_cache::get_or_compile;
use crate::transfer::{alloc_zeros, alloc_zeros_bf16, alloc_zeros_f32};

const BLOCK_SIZE: u32 = 256;

fn launch_1d(n: usize) -> GpuResult<LaunchConfig> {
    if n > u32::MAX as usize {
        return Err(GpuError::LengthMismatch {
            a: n,
            b: u32::MAX as usize,
        });
    }
    let n = n as u32;
    let grid = n.saturating_add(BLOCK_SIZE - 1) / BLOCK_SIZE;
    Ok(LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn validate_cutlass_shape(
    op: &'static str,
    input_len: usize,
    rows: usize,
    cols: usize,
    row_multiple: usize,
    col_multiple: usize,
) -> GpuResult<()> {
    if rows == 0
        || cols == 0
        || !rows.is_multiple_of(row_multiple)
        || !cols.is_multiple_of(col_multiple)
    {
        return Err(GpuError::ShapeMismatch {
            op,
            expected: vec![row_multiple, col_multiple],
            got: vec![rows, cols],
        });
    }
    let dense_len = rows
        .checked_mul(cols)
        .ok_or(GpuError::LengthMismatch { a: rows, b: cols })?;
    if input_len < dense_len {
        return Err(GpuError::LengthMismatch {
            a: input_len,
            b: dense_len,
        });
    }
    Ok(())
}

const CUTLASS_PACK_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry semi_structured_pack_cutlass_f32_kernel(
    .param .u64 dense_ptr,
    .param .u64 values_ptr,
    .param .u64 meta_ptr,
    .param .u32 rows,
    .param .u32 cols,
    .param .u32 meta_cols
) {
    .reg .u64 %dense, %values, %meta, %addr, %off64;
    .reg .u32 %tid_r, %bid, %bdim, %total, %rows, %cols, %meta_cols;
    .reg .u32 %row, %mc, %q, %pair, %base, %out_idx, %half_cols;
    .reg .u32 %m0, %m1, %m2, %m3, %nm0, %nm1, %expr0, %expr1, %expr2;
    .reg .u32 %bit0, %bit1, %bit2, %bit3, %idx0, %idx1, %nib, %shift, %nib_shifted, %meta_val, %sel;
    .reg .u32 %dst_row, %dst_col, %tmp, %tmp2, %cols_maj, %cols_min, %meta_off;
    .reg .f32 %v0, %v1, %outv;
    .reg .b16 %mh;
    .reg .pred %p, %p0, %p1, %choose0, %even_row, %odd_row, %odd_col, %even_col, %topright, %bottomleft, %ptmp;

    ld.param.u64 %dense, [dense_ptr];
    ld.param.u64 %values, [values_ptr];
    ld.param.u64 %meta, [meta_ptr];
    ld.param.u32 %rows, [rows];
    ld.param.u32 %cols, [cols];
    ld.param.u32 %meta_cols, [meta_cols];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %tid_r, %tid.x;
    mad.lo.u32 %tid_r, %bid, %bdim, %tid_r;
    mul.lo.u32 %total, %rows, %meta_cols;
    setp.ge.u32 %p, %tid_r, %total;
    @%p bra DONE;

    div.u32 %row, %tid_r, %meta_cols;
    rem.u32 %mc, %tid_r, %meta_cols;
    shr.u32 %half_cols, %cols, 1;
    mov.u32 %meta_val, 0;
    mov.u32 %q, 0;

LOOP:
    setp.ge.u32 %p, %q, 4;
    @%p bra STORE_META;

    mul.lo.u32 %pair, %mc, 4;
    add.u32 %pair, %pair, %q;
    mul.lo.u32 %base, %row, %cols;
    mad.lo.u32 %base, %pair, 2, %base;

    cvt.u64.u32 %off64, %base;
    shl.b64 %off64, %off64, 2;
    add.u64 %addr, %dense, %off64;
    ld.global.f32 %v0, [%addr];
    add.u64 %addr, %addr, 4;
    ld.global.f32 %v1, [%addr];

    setp.neu.f32 %p0, %v0, 0f00000000;
    setp.neu.f32 %p1, %v1, 0f00000000;
    selp.u32 %m0, 1, 0, %p0;
    mov.u32 %m1, %m0;
    selp.u32 %m2, 1, 0, %p1;
    mov.u32 %m3, %m2;

    and.b32 %expr0, %m0, %m1;
    xor.b32 %nm0, %m0, 1;
    and.b32 %expr1, %nm0, %m1;
    xor.b32 %nm1, %m1, 1;
    and.b32 %expr2, %nm0, %nm1;
    mov.u32 %bit0, %expr1;
    mov.u32 %bit1, %expr2;
    or.b32 %bit2, %expr0, %expr2;
    or.b32 %bit2, %bit2, %m3;
    or.b32 %bit3, %expr1, %nm1;
    shl.b32 %tmp, %bit1, 1;
    or.b32 %idx0, %bit0, %tmp;
    shl.b32 %tmp, %bit3, 1;
    or.b32 %idx1, %bit2, %tmp;

    shr.u32 %sel, %idx0, 1;
    setp.eq.u32 %choose0, %sel, 0;
    selp.f32 %outv, %v0, %v1, %choose0;
    mad.lo.u32 %out_idx, %row, %half_cols, %pair;
    cvt.u64.u32 %off64, %out_idx;
    shl.b64 %off64, %off64, 2;
    add.u64 %addr, %values, %off64;
    st.global.f32 [%addr], %outv;

    shl.b32 %tmp, %idx1, 2;
    or.b32 %nib, %idx0, %tmp;
    shl.b32 %shift, %q, 2;
    shl.b32 %nib_shifted, %nib, %shift;
    or.b32 %meta_val, %meta_val, %nib_shifted;

    add.u32 %q, %q, 1;
    bra LOOP;

STORE_META:
    div.u32 %tmp, %row, 32;
    mul.lo.u32 %dst_row, %tmp, 32;
    rem.u32 %tmp, %row, 8;
    mad.lo.u32 %dst_row, %tmp, 4, %dst_row;
    rem.u32 %tmp, %row, 32;
    div.u32 %tmp, %tmp, 8;
    add.u32 %dst_row, %dst_row, %tmp;
    mov.u32 %dst_col, %mc;

    rem.u32 %tmp, %dst_row, 2;
    setp.eq.u32 %even_row, %tmp, 0;
    setp.ne.u32 %odd_row, %tmp, 0;
    rem.u32 %tmp2, %dst_col, 2;
    setp.ne.u32 %odd_col, %tmp2, 0;
    setp.eq.u32 %even_col, %tmp2, 0;
    and.pred %topright, %even_row, %odd_col;
    and.pred %bottomleft, %odd_row, %even_col;
    @%topright add.u32 %dst_row, %dst_row, 1;
    @%topright sub.u32 %dst_col, %dst_col, 1;
    @%bottomleft sub.u32 %dst_row, %dst_row, 1;
    @%bottomleft add.u32 %dst_col, %dst_col, 1;

    div.u32 %cols_maj, %dst_col, 2;
    rem.u32 %cols_min, %dst_col, 2;
    mul.lo.u32 %meta_off, %cols_maj, %rows;
    shl.b32 %meta_off, %meta_off, 1;
    mad.lo.u32 %meta_off, %dst_row, 2, %meta_off;
    add.u32 %meta_off, %meta_off, %cols_min;
    cvt.u64.u32 %off64, %meta_off;
    shl.b64 %off64, %off64, 1;
    add.u64 %addr, %meta, %off64;
    cvt.u16.u32 %mh, %meta_val;
    st.global.b16 [%addr], %mh;

DONE:
    ret;
}
";

const CUTLASS_PACK_U16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry semi_structured_pack_cutlass_u16_kernel(
    .param .u64 dense_ptr,
    .param .u64 values_ptr,
    .param .u64 meta_ptr,
    .param .u32 rows,
    .param .u32 cols,
    .param .u32 meta_cols
) {
    .reg .u64 %dense, %values, %meta, %addr, %off64;
    .reg .u32 %tid_r, %bid, %bdim, %total, %rows, %cols, %meta_cols;
    .reg .u32 %row, %mc, %q, %group, %base, %value_base, %half_cols;
    .reg .u32 %bits0, %bits1, %bits2, %bits3, %mag0, %mag1, %mag2, %mag3;
    .reg .u32 %m0, %m1, %m2, %m3, %nm0, %nm1, %expr0, %expr1, %expr2;
    .reg .u32 %bit0, %bit1, %bit2, %bit3, %idx0, %idx1, %nib, %shift, %nib_shifted, %meta_val;
    .reg .u32 %dst_row, %dst_col, %tmp, %tmp2, %cols_maj, %cols_min, %meta_off;
    .reg .b16 %v0, %v1, %v2, %v3, %s0, %s1, %mh;
    .reg .pred %p, %p0, %p1, %p2, %p3, %pidx, %even_row, %odd_row, %odd_col, %even_col, %topright, %bottomleft;

    ld.param.u64 %dense, [dense_ptr];
    ld.param.u64 %values, [values_ptr];
    ld.param.u64 %meta, [meta_ptr];
    ld.param.u32 %rows, [rows];
    ld.param.u32 %cols, [cols];
    ld.param.u32 %meta_cols, [meta_cols];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %tid_r, %tid.x;
    mad.lo.u32 %tid_r, %bid, %bdim, %tid_r;
    mul.lo.u32 %total, %rows, %meta_cols;
    setp.ge.u32 %p, %tid_r, %total;
    @%p bra DONE;

    div.u32 %row, %tid_r, %meta_cols;
    rem.u32 %mc, %tid_r, %meta_cols;
    shr.u32 %half_cols, %cols, 1;
    mov.u32 %meta_val, 0;
    mov.u32 %q, 0;

LOOP:
    setp.ge.u32 %p, %q, 4;
    @%p bra STORE_META;

    mul.lo.u32 %group, %mc, 4;
    add.u32 %group, %group, %q;
    mul.lo.u32 %base, %row, %cols;
    mad.lo.u32 %base, %group, 4, %base;

    cvt.u64.u32 %off64, %base;
    shl.b64 %off64, %off64, 1;
    add.u64 %addr, %dense, %off64;
    ld.global.b16 %v0, [%addr];
    add.u64 %addr, %addr, 2;
    ld.global.b16 %v1, [%addr];
    add.u64 %addr, %addr, 2;
    ld.global.b16 %v2, [%addr];
    add.u64 %addr, %addr, 2;
    ld.global.b16 %v3, [%addr];

    cvt.u32.u16 %bits0, %v0;
    cvt.u32.u16 %bits1, %v1;
    cvt.u32.u16 %bits2, %v2;
    cvt.u32.u16 %bits3, %v3;
    and.b32 %mag0, %bits0, 0x7fff;
    and.b32 %mag1, %bits1, 0x7fff;
    and.b32 %mag2, %bits2, 0x7fff;
    and.b32 %mag3, %bits3, 0x7fff;
    setp.ne.u32 %p0, %mag0, 0;
    setp.ne.u32 %p1, %mag1, 0;
    setp.ne.u32 %p2, %mag2, 0;
    setp.ne.u32 %p3, %mag3, 0;
    selp.u32 %m0, 1, 0, %p0;
    selp.u32 %m1, 1, 0, %p1;
    selp.u32 %m2, 1, 0, %p2;
    selp.u32 %m3, 1, 0, %p3;

    and.b32 %expr0, %m0, %m1;
    xor.b32 %nm0, %m0, 1;
    and.b32 %expr1, %nm0, %m1;
    xor.b32 %nm1, %m1, 1;
    and.b32 %expr2, %nm0, %nm1;
    mov.u32 %bit0, %expr1;
    mov.u32 %bit1, %expr2;
    or.b32 %bit2, %expr0, %expr2;
    or.b32 %bit2, %bit2, %m3;
    or.b32 %bit3, %expr1, %nm1;
    shl.b32 %tmp, %bit1, 1;
    or.b32 %idx0, %bit0, %tmp;
    shl.b32 %tmp, %bit3, 1;
    or.b32 %idx1, %bit2, %tmp;

    mov.b16 %s0, %v0;
    setp.eq.u32 %pidx, %idx0, 1;
    @%pidx mov.b16 %s0, %v1;
    setp.eq.u32 %pidx, %idx0, 2;
    @%pidx mov.b16 %s0, %v2;
    setp.eq.u32 %pidx, %idx0, 3;
    @%pidx mov.b16 %s0, %v3;

    mov.b16 %s1, %v0;
    setp.eq.u32 %pidx, %idx1, 1;
    @%pidx mov.b16 %s1, %v1;
    setp.eq.u32 %pidx, %idx1, 2;
    @%pidx mov.b16 %s1, %v2;
    setp.eq.u32 %pidx, %idx1, 3;
    @%pidx mov.b16 %s1, %v3;

    mad.lo.u32 %value_base, %row, %half_cols, %group;
    shl.b32 %value_base, %value_base, 1;
    cvt.u64.u32 %off64, %value_base;
    shl.b64 %off64, %off64, 1;
    add.u64 %addr, %values, %off64;
    st.global.b16 [%addr], %s0;
    add.u64 %addr, %addr, 2;
    st.global.b16 [%addr], %s1;

    shl.b32 %tmp, %idx1, 2;
    or.b32 %nib, %idx0, %tmp;
    shl.b32 %shift, %q, 2;
    shl.b32 %nib_shifted, %nib, %shift;
    or.b32 %meta_val, %meta_val, %nib_shifted;

    add.u32 %q, %q, 1;
    bra LOOP;

STORE_META:
    div.u32 %tmp, %row, 32;
    mul.lo.u32 %dst_row, %tmp, 32;
    rem.u32 %tmp, %row, 8;
    mad.lo.u32 %dst_row, %tmp, 4, %dst_row;
    rem.u32 %tmp, %row, 32;
    div.u32 %tmp, %tmp, 8;
    add.u32 %dst_row, %dst_row, %tmp;
    mov.u32 %dst_col, %mc;

    rem.u32 %tmp, %dst_row, 2;
    setp.eq.u32 %even_row, %tmp, 0;
    setp.ne.u32 %odd_row, %tmp, 0;
    rem.u32 %tmp2, %dst_col, 2;
    setp.ne.u32 %odd_col, %tmp2, 0;
    setp.eq.u32 %even_col, %tmp2, 0;
    and.pred %topright, %even_row, %odd_col;
    and.pred %bottomleft, %odd_row, %even_col;
    @%topright add.u32 %dst_row, %dst_row, 1;
    @%topright sub.u32 %dst_col, %dst_col, 1;
    @%bottomleft sub.u32 %dst_row, %dst_row, 1;
    @%bottomleft add.u32 %dst_col, %dst_col, 1;

    div.u32 %cols_maj, %dst_col, 2;
    rem.u32 %cols_min, %dst_col, 2;
    mul.lo.u32 %meta_off, %cols_maj, %rows;
    shl.b32 %meta_off, %meta_off, 1;
    mad.lo.u32 %meta_off, %dst_row, 2, %meta_off;
    add.u32 %meta_off, %meta_off, %cols_min;
    cvt.u64.u32 %off64, %meta_off;
    shl.b64 %off64, %off64, 1;
    add.u64 %addr, %meta, %off64;
    cvt.u16.u32 %mh, %meta_val;
    st.global.b16 [%addr], %mh;

DONE:
    ret;
}
";

const COPY_U16_TAIL_TO_I16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry copy_u16_tail_to_i16_kernel(
    .param .u64 src_ptr,
    .param .u64 dst_ptr,
    .param .u32 offset,
    .param .u32 len
) {
    .reg .u64 %src, %dst, %addr, %off64;
    .reg .u32 %tid_r, %bid, %bdim, %offset, %len, %src_idx;
    .reg .b16 %v;
    .reg .pred %p;

    ld.param.u64 %src, [src_ptr];
    ld.param.u64 %dst, [dst_ptr];
    ld.param.u32 %offset, [offset];
    ld.param.u32 %len, [len];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %tid_r, %tid.x;
    mad.lo.u32 %tid_r, %bid, %bdim, %tid_r;
    setp.ge.u32 %p, %tid_r, %len;
    @%p bra DONE;

    add.u32 %src_idx, %offset, %tid_r;
    cvt.u64.u32 %off64, %src_idx;
    shl.b64 %off64, %off64, 1;
    add.u64 %addr, %src, %off64;
    ld.global.b16 %v, [%addr];

    cvt.u64.u32 %off64, %tid_r;
    shl.b64 %off64, %off64, 1;
    add.u64 %addr, %dst, %off64;
    st.global.b16 [%addr], %v;

DONE:
    ret;
}
";

fn launch_pack_f32(
    dense: &CudaSlice<f32>,
    values: &mut CudaSlice<f32>,
    meta: &mut CudaSlice<i16>,
    rows: usize,
    cols: usize,
    device: &GpuDevice,
) -> GpuResult<()> {
    let meta_cols = cols / 8;
    let total = rows
        .checked_mul(meta_cols)
        .ok_or(GpuError::LengthMismatch {
            a: rows,
            b: meta_cols,
        })?;
    if total == 0 {
        return Ok(());
    }
    let f = get_or_compile(
        device.context(),
        CUTLASS_PACK_F32_PTX,
        "semi_structured_pack_cutlass_f32_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "semi_structured_pack_cutlass_f32_kernel",
        source: e,
    })?;
    let cfg = launch_1d(total)?;
    let rows_u = rows as u32;
    let cols_u = cols as u32;
    let meta_cols_u = meta_cols as u32;
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(dense)
            .arg(values)
            .arg(meta)
            .arg(&rows_u)
            .arg(&cols_u)
            .arg(&meta_cols_u)
            .launch(cfg)?;
    }
    Ok(())
}

fn launch_pack_u16(
    dense: &CudaSlice<u16>,
    values: &mut CudaSlice<u16>,
    meta: &mut CudaSlice<i16>,
    rows: usize,
    cols: usize,
    device: &GpuDevice,
) -> GpuResult<()> {
    let meta_cols = cols / 16;
    let total = rows
        .checked_mul(meta_cols)
        .ok_or(GpuError::LengthMismatch {
            a: rows,
            b: meta_cols,
        })?;
    if total == 0 {
        return Ok(());
    }
    let f = get_or_compile(
        device.context(),
        CUTLASS_PACK_U16_PTX,
        "semi_structured_pack_cutlass_u16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "semi_structured_pack_cutlass_u16_kernel",
        source: e,
    })?;
    let cfg = launch_1d(total)?;
    let rows_u = rows as u32;
    let cols_u = cols as u32;
    let meta_cols_u = meta_cols as u32;
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(dense)
            .arg(values)
            .arg(meta)
            .arg(&rows_u)
            .arg(&cols_u)
            .arg(&meta_cols_u)
            .launch(cfg)?;
    }
    Ok(())
}

/// Pack an f32 CUDA matrix into CUTLASS semi-structured values and int16 metadata.
pub fn pack_cutlass_f32(
    dense: &CudaBuffer<f32>,
    rows: usize,
    cols: usize,
    device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, CudaBuffer<i16>)> {
    validate_cutlass_shape("pack_cutlass_f32", dense.len(), rows, cols, 32, 32)?;
    let value_len = rows.checked_mul(cols / 2).ok_or(GpuError::LengthMismatch {
        a: rows,
        b: cols / 2,
    })?;
    let meta_len = rows.checked_mul(cols / 8).ok_or(GpuError::LengthMismatch {
        a: rows,
        b: cols / 8,
    })?;
    let mut values = alloc_zeros_f32(value_len, device)?;
    let mut meta: CudaBuffer<i16> = alloc_zeros(meta_len, device)?;
    launch_pack_f32(
        dense.inner(),
        values.inner_mut(),
        meta.inner_mut(),
        rows,
        cols,
        device,
    )?;
    Ok((values, meta))
}

/// Pack an f16 or bf16 CUDA matrix into CUTLASS semi-structured values and metadata.
pub fn pack_cutlass_u16(
    dense: &CudaSlice<u16>,
    rows: usize,
    cols: usize,
    device: &GpuDevice,
) -> GpuResult<(CudaSlice<u16>, CudaBuffer<i16>)> {
    validate_cutlass_shape("pack_cutlass_u16", dense.len(), rows, cols, 32, 64)?;
    let value_len = rows.checked_mul(cols / 2).ok_or(GpuError::LengthMismatch {
        a: rows,
        b: cols / 2,
    })?;
    let meta_len = rows
        .checked_mul(cols / 16)
        .ok_or(GpuError::LengthMismatch {
            a: rows,
            b: cols / 16,
        })?;
    let mut values = alloc_zeros_bf16(value_len, device)?;
    let mut meta: CudaBuffer<i16> = alloc_zeros(meta_len, device)?;
    launch_pack_u16(dense, &mut values, meta.inner_mut(), rows, cols, device)?;
    Ok((values, meta))
}

/// Copy a 16-bit metadata tail out of a u16-backed packed tensor into an int16 buffer.
///
/// cuSPARSELt stores f16/bf16 values and int16 metadata in one opaque 16-bit
/// allocation. Ferrotorch keeps the public packed tensor intact and also
/// exposes metadata as an `IntTensor<i16>` by copying the tail on-device.
pub fn copy_u16_tail_to_i16(
    packed: &CudaSlice<u16>,
    offset: usize,
    len: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<i16>> {
    let end = offset
        .checked_add(len)
        .ok_or(GpuError::LengthMismatch { a: offset, b: len })?;
    if end > packed.len() {
        return Err(GpuError::LengthMismatch {
            a: packed.len(),
            b: end,
        });
    }
    if offset > u32::MAX as usize || len > u32::MAX as usize {
        return Err(GpuError::LengthMismatch {
            a: offset.max(len),
            b: u32::MAX as usize,
        });
    }

    let mut out: CudaBuffer<i16> = alloc_zeros(len, device)?;
    if len == 0 {
        return Ok(out);
    }

    let f = get_or_compile(
        device.context(),
        COPY_U16_TAIL_TO_I16_PTX,
        "copy_u16_tail_to_i16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "copy_u16_tail_to_i16_kernel",
        source: e,
    })?;
    let cfg = launch_1d(len)?;
    let offset_u = offset as u32;
    let len_u = len as u32;
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(packed)
            .arg(out.inner_mut())
            .arg(&offset_u)
            .arg(&len_u)
            .launch(cfg)?;
    }
    Ok(out)
}
