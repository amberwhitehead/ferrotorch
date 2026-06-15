//! Triangular-mask GPU compute kernels — `torch.triu` / `torch.tril`
//! (crosslink #1545 / sub #1535).
//!
//! Hand-written PTX owned by Rust (no CUDA C++, no nvrtc), loaded via
//! [`crate::module_cache::get_or_compile`] exactly like [`crate::reduce_arg`].
//!
//! # Semantics (PyTorch parity)
//!
//! For a `[batch.., rows, cols]` C-contiguous buffer the upper/lower triangular
//! mask is applied to the LAST TWO DIMS of EVERY trailing `[rows, cols]`
//! matrix, batching over all leading dims (`batch` = product of the leading
//! dims). Element `(row, col)` within a matrix is **preserved** when the
//! predicate holds and **zeroed** otherwise:
//!
//! - **triu** keeps `(row, col)` when `col - row >= k`.
//! - **tril** keeps `(row, col)` when `col - row <= k`.
//!
//! This is exactly PyTorch's CUDA predicate at
//! `aten/src/ATen/native/cuda/TriangularOps.cu:100`
//! (`mask = upper ? (col + i - row >= k) : (col + i - row <= k)`) and the
//! per-trailing-matrix batching at `TriangularOps.cu:120`
//! (`N_padded = multiply_integers(sizes[..last]) * last_dim_padded`) plus the
//! ferrotorch CPU path in `ferrotorch_core::ops::tensor_ops`
//! (`triu`: `c >= r + diagonal`, `tril`: `c <= r + diagonal`). Because the op
//! only copies-or-zeros (no arithmetic), the GPU result is **bit-for-bit
//! identical** to both the CPU ferrotorch path and `torch.{triu,tril}` —
//! there is no float-tolerance question.
//!
//! `k` (the diagonal offset) is signed. We pass it as `s32` and evaluate
//! `diff = col - row` in signed 32-bit arithmetic, so negative diagonals
//! work. PyTorch itself uses `int32_t` index math when the tensor fits
//! (`TriangularOps.cu:125`), which is always true for a single-device 2-D
//! matrix.
//!
//! # Launch scheme
//!
//! One thread per element. `total = batch * rows * cols` threads; thread `t`
//! computes `col = t % cols`, `row = (t / cols) % rows` (so the row index
//! resets per trailing `[rows, cols]` matrix — `batch == 1` reduces to the
//! plain 2-D case since `row % rows == row` for `row < rows`), then writes
//! one element. The mask is a pure function of the within-matrix `(row, col)`,
//! so it repeats identically across the `batch` leading matrices.
//!
//! ## REQ status (per `.design/ferrotorch-gpu/triangular.md`)
//!
//! Full evidence rows live in the design doc; this synopsis is one line per
//! REQ.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (f32 triu/tril) | SHIPPED | `pub fn gpu_triu_f32` / `gpu_tril_f32` in `triangular.rs`; consumer `CudaBackendImpl::triu_f32`/`tril_f32` in `backend_impl.rs` |
//! | REQ-2 (f64 triu/tril) | SHIPPED | `pub fn gpu_triu_f64` / `gpu_tril_f64` in `triangular.rs`; consumer `CudaBackendImpl::triu_f64`/`tril_f64` in `backend_impl.rs` |
//! | REQ-3 (signed diagonal) | SHIPPED | `setp.ge.s32` / `setp.le.s32 %diff, %k` in the PTX templates; verified by `triu_f32_negative_diag` / `tril_f32_positive_diag` unit tests |
//! | REQ-4 (dispatch wiring) | SHIPPED | `fn triu_f32`/`tril_f32`/`triu_f64`/`tril_f64` in `backend_impl.rs`; consumer the `input.is_cuda()` branch of `triu`/`tril` in `ferrotorch-core/src/ops/tensor_ops.rs` |

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits};

use crate::buffer::CudaBuffer;
use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};
use crate::module_cache::get_or_compile;
use crate::transfer::{alloc_zeros_bf16, alloc_zeros_f32, alloc_zeros_f64};

const BLOCK_SIZE: u32 = 256;

fn launch_1d(n: usize) -> LaunchConfig {
    let grid = ((n as u32).saturating_add(BLOCK_SIZE - 1)) / BLOCK_SIZE;
    LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    }
}

// `op` selector pushed to the kernels: 0 = triu (keep diff >= k), 1 = tril
// (keep diff <= k), where diff = col - row.
const OP_TRIU: u32 = 0;
const OP_TRIL: u32 = 1;

// ===========================================================================
// f32
//
// Params: (in_ptr, out_ptr, batch, rows, cols, k, op)
//   in  : f32[batch * rows * cols]   (C-contiguous, [batch.., rows, cols])
//   out : f32[batch * rows * cols]
// Thread t in [0, batch*rows*cols): col = t % cols; row = (t / cols) % rows.
//   (row resets per trailing [rows, cols] matrix — batches over leading dims,
//    matching aten/src/ATen/native/cuda/TriangularOps.cu:120)
//   diff = (s32)col - (s32)row
//   keep = (op == 0) ? (diff >= k) : (diff <= k)
//   out[t] = keep ? in[t] : 0.0f
// ===========================================================================
const TRIANGULAR_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry triangular_f32_kernel(
    .param .u64 in_ptr, .param .u64 out_ptr,
    .param .u32 batch, .param .u32 rows, .param .u32 cols, .param .s32 k, .param .u32 op
) {
    .reg .u32 %gtid, %bid, %bdim, %batch, %rows, %cols, %op_r, %row_u, %col_u, %total, %tmp;
    .reg .s32 %row_s, %col_s, %diff, %k_r;
    .reg .u64 %in, %out, %off, %addr;
    .reg .f32 %v, %zero;
    .reg .pred %p, %is_triu, %not_triu, %keep, %ge, %le, %a, %b;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %batch, [batch];
    ld.param.u32 %rows, [rows];
    ld.param.u32 %cols, [cols];
    ld.param.s32 %k_r, [k];
    ld.param.u32 %op_r, [op];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %gtid, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %gtid;

    mul.lo.u32 %total, %rows, %cols;
    mul.lo.u32 %total, %total, %batch;
    setp.ge.u32 %p, %gtid, %total;
    @%p bra DONE;

    rem.u32 %col_u, %gtid, %cols;
    div.u32 %tmp, %gtid, %cols;
    rem.u32 %row_u, %tmp, %rows;
    cvt.s32.u32 %row_s, %row_u;
    cvt.s32.u32 %col_s, %col_u;
    sub.s32 %diff, %col_s, %row_s;

    setp.eq.u32 %is_triu, %op_r, 0;
    not.pred %not_triu, %is_triu;
    setp.ge.s32 %ge, %diff, %k_r;
    setp.le.s32 %le, %diff, %k_r;
    // keep = (is_triu && ge) || (!is_triu && le)
    and.pred %a, %ge, %is_triu;
    and.pred %b, %le, %not_triu;
    or.pred %keep, %a, %b;

    cvt.u64.u32 %off, %gtid;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in, %off;
    ld.global.f32 %v, [%addr];
    mov.f32 %zero, 0f00000000;
    selp.f32 %v, %v, %zero, %keep;

    add.u64 %addr, %out, %off;
    st.global.f32 [%addr], %v;
DONE:
    ret;
}
";

// ===========================================================================
// f64 — same structure, 8-byte value stride.
// ===========================================================================
const TRIANGULAR_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry triangular_f64_kernel(
    .param .u64 in_ptr, .param .u64 out_ptr,
    .param .u32 batch, .param .u32 rows, .param .u32 cols, .param .s32 k, .param .u32 op
) {
    .reg .u32 %gtid, %bid, %bdim, %batch, %rows, %cols, %op_r, %row_u, %col_u, %total, %tmp;
    .reg .s32 %row_s, %col_s, %diff, %k_r;
    .reg .u64 %in, %out, %off, %addr;
    .reg .f64 %v, %zero;
    .reg .pred %p, %is_triu, %not_triu, %keep, %ge, %le, %a, %b;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %batch, [batch];
    ld.param.u32 %rows, [rows];
    ld.param.u32 %cols, [cols];
    ld.param.s32 %k_r, [k];
    ld.param.u32 %op_r, [op];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %gtid, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %gtid;

    mul.lo.u32 %total, %rows, %cols;
    mul.lo.u32 %total, %total, %batch;
    setp.ge.u32 %p, %gtid, %total;
    @%p bra DONE;

    rem.u32 %col_u, %gtid, %cols;
    div.u32 %tmp, %gtid, %cols;
    rem.u32 %row_u, %tmp, %rows;
    cvt.s32.u32 %row_s, %row_u;
    cvt.s32.u32 %col_s, %col_u;
    sub.s32 %diff, %col_s, %row_s;

    setp.eq.u32 %is_triu, %op_r, 0;
    not.pred %not_triu, %is_triu;
    setp.ge.s32 %ge, %diff, %k_r;
    setp.le.s32 %le, %diff, %k_r;
    and.pred %a, %ge, %is_triu;
    and.pred %b, %le, %not_triu;
    or.pred %keep, %a, %b;

    cvt.u64.u32 %off, %gtid;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %in, %off;
    ld.global.f64 %v, [%addr];
    mov.f64 %zero, 0d0000000000000000;
    selp.f64 %v, %v, %zero, %keep;

    add.u64 %addr, %out, %off;
    st.global.f64 [%addr], %v;
DONE:
    ret;
}
";

// ===========================================================================
// u16 — raw f16/bf16 payload copy/zero. The dtype tag is preserved by
// CudaBackendImpl, so one raw-bit kernel covers both 16-bit float formats.
// ===========================================================================
const TRIANGULAR_U16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry triangular_u16_kernel(
    .param .u64 in_ptr, .param .u64 out_ptr,
    .param .u32 batch, .param .u32 rows, .param .u32 cols, .param .s32 k, .param .u32 op
) {
    .reg .u32 %gtid, %bid, %bdim, %batch, %rows, %cols, %op_r, %row_u, %col_u, %total, %tmp;
    .reg .s32 %row_s, %col_s, %diff, %k_r;
    .reg .u64 %in, %out, %off, %addr;
    .reg .b16 %v, %zero;
    .reg .pred %p, %is_triu, %not_triu, %keep, %ge, %le, %a, %b;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %batch, [batch];
    ld.param.u32 %rows, [rows];
    ld.param.u32 %cols, [cols];
    ld.param.s32 %k_r, [k];
    ld.param.u32 %op_r, [op];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %gtid, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %gtid;

    mul.lo.u32 %total, %rows, %cols;
    mul.lo.u32 %total, %total, %batch;
    setp.ge.u32 %p, %gtid, %total;
    @%p bra DONE;

    rem.u32 %col_u, %gtid, %cols;
    div.u32 %tmp, %gtid, %cols;
    rem.u32 %row_u, %tmp, %rows;
    cvt.s32.u32 %row_s, %row_u;
    cvt.s32.u32 %col_s, %col_u;
    sub.s32 %diff, %col_s, %row_s;

    setp.eq.u32 %is_triu, %op_r, 0;
    not.pred %not_triu, %is_triu;
    setp.ge.s32 %ge, %diff, %k_r;
    setp.le.s32 %le, %diff, %k_r;
    and.pred %a, %ge, %is_triu;
    and.pred %b, %le, %not_triu;
    or.pred %keep, %a, %b;

    cvt.u64.u32 %off, %gtid;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %in, %off;
    ld.global.b16 %v, [%addr];
    mov.b16 %zero, 0;
    selp.b16 %v, %v, %zero, %keep;

    add.u64 %addr, %out, %off;
    st.global.b16 [%addr], %v;
DONE:
    ret;
}
";

/// Launch a triangular-mask kernel over a value buffer of element type `V`.
///
/// `in_slice` holds at least `batch * rows * cols` `V`-elements (contiguous,
/// `[batch.., rows, cols]` C-order; `batch` is the product of all leading
/// dims, `1` for a plain 2-D matrix). The mask is applied to every trailing
/// `[rows, cols]` matrix (mirrors `TriangularOps.cu:120`). `op` is [`OP_TRIU`]
/// or [`OP_TRIL`]; `k` is the signed diagonal offset. Writes into `out_slice`
/// (also `batch * rows * cols`).
#[allow(clippy::too_many_arguments)]
fn launch_triangular<V: DeviceRepr + ValidAsZeroBits>(
    in_slice: &CudaSlice<V>,
    out_slice: &mut CudaSlice<V>,
    batch: usize,
    rows: usize,
    cols: usize,
    k: i64,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
    op: u32,
    elem_bytes: usize,
) -> GpuResult<()> {
    let total = rows
        .checked_mul(cols)
        .and_then(|mn| mn.checked_mul(batch))
        .ok_or(GpuError::LengthMismatch { a: rows, b: cols })?;
    if total == 0 {
        return Ok(());
    }
    // The input slice may be POOL-OVERSIZED (its `CudaSlice::len()` is the
    // rounded allocation, not the logical numel). We only require it holds AT
    // LEAST `total` elements; the kernel reads strictly within `[0, total)`.
    if in_slice.len() < total {
        return Err(GpuError::LengthMismatch {
            a: in_slice.len(),
            b: total,
        });
    }
    let stream = device.stream();
    let ctx = device.context();
    let f = get_or_compile(ctx, ptx, kernel_name, device.ordinal() as u32).map_err(|e| {
        GpuError::PtxCompileFailed {
            kernel: kernel_name,
            source: e,
        }
    })?;
    let cfg = launch_1d(total);
    // `rows`/`cols` are 2-D matrix dims that always fit in u32 on a single
    // device (PyTorch uses int32 index math here too — TriangularOps.cu:125).
    // `k` is the signed diagonal; clamp into i32 range — out-of-range diagonals
    // are degenerate (the whole matrix is kept or zeroed) and i32::MIN/MAX
    // preserve that.
    let batch_u = batch as u32;
    let rows_u = rows as u32;
    let cols_u = cols as u32;
    let k_i32 = k.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    let _ = elem_bytes; // documents the per-dtype value stride encoded in the PTX
    // SAFETY:
    // - `f` is the PTX entry `kernel_name`; its 7-arg signature
    //   (in_ptr, out_ptr, batch, rows, cols, k, op) matches the args pushed
    //   below in order.
    // - `in_slice` holds at least `batch*rows*cols` `V`-elements (checked above).
    // - `out_slice` is the caller's fresh `total`-element buffer, the only
    //   `&mut`, non-aliased with `in_slice` (distinct allocations).
    // - Each thread reads `in[t]` and writes `out[t]` for `t in [0,total)`,
    //   bound-checked by `setp.ge.u32 %p, %gtid, %total; @%p bra DONE`.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(in_slice)
            .arg(out_slice)
            .arg(&batch_u)
            .arg(&rows_u)
            .arg(&cols_u)
            .arg(&k_i32)
            .arg(&op)
            .launch(cfg)?;
    }
    Ok(())
}

/// `triu` over an f32 `[batch.., rows, cols]` buffer (`batch` = product of the
/// leading dims, `1` for plain 2-D). Masks each trailing `[rows, cols]` matrix.
/// Returns a fresh resident buffer of `batch * rows * cols` elements.
pub fn gpu_triu_f32(
    input: &CudaBuffer<f32>,
    batch: usize,
    rows: usize,
    cols: usize,
    k: i64,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    let mut out = alloc_zeros_f32(batch * rows * cols, device)?;
    launch_triangular(
        input.inner(),
        out.inner_mut(),
        batch,
        rows,
        cols,
        k,
        device,
        TRIANGULAR_F32_PTX,
        "triangular_f32_kernel",
        OP_TRIU,
        4,
    )?;
    Ok(out)
}

/// `tril` over an f32 `[batch.., rows, cols]` buffer. Masks each trailing
/// `[rows, cols]` matrix. Returns a fresh resident buffer.
pub fn gpu_tril_f32(
    input: &CudaBuffer<f32>,
    batch: usize,
    rows: usize,
    cols: usize,
    k: i64,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    let mut out = alloc_zeros_f32(batch * rows * cols, device)?;
    launch_triangular(
        input.inner(),
        out.inner_mut(),
        batch,
        rows,
        cols,
        k,
        device,
        TRIANGULAR_F32_PTX,
        "triangular_f32_kernel",
        OP_TRIL,
        4,
    )?;
    Ok(out)
}

/// `triu` over an f64 `[batch.., rows, cols]` buffer. Masks each trailing
/// `[rows, cols]` matrix. Returns a fresh resident buffer.
pub fn gpu_triu_f64(
    input: &CudaBuffer<f64>,
    batch: usize,
    rows: usize,
    cols: usize,
    k: i64,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    let mut out = alloc_zeros_f64(batch * rows * cols, device)?;
    launch_triangular(
        input.inner(),
        out.inner_mut(),
        batch,
        rows,
        cols,
        k,
        device,
        TRIANGULAR_F64_PTX,
        "triangular_f64_kernel",
        OP_TRIU,
        8,
    )?;
    Ok(out)
}

/// `tril` over an f64 `[batch.., rows, cols]` buffer. Masks each trailing
/// `[rows, cols]` matrix. Returns a fresh resident buffer.
pub fn gpu_tril_f64(
    input: &CudaBuffer<f64>,
    batch: usize,
    rows: usize,
    cols: usize,
    k: i64,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    let mut out = alloc_zeros_f64(batch * rows * cols, device)?;
    launch_triangular(
        input.inner(),
        out.inner_mut(),
        batch,
        rows,
        cols,
        k,
        device,
        TRIANGULAR_F64_PTX,
        "triangular_f64_kernel",
        OP_TRIL,
        8,
    )?;
    Ok(out)
}

/// `triu` over a raw f16/bf16 `[batch.., rows, cols]` payload buffer. Masks each
/// trailing `[rows, cols]` matrix and returns a fresh resident `u16` buffer.
pub fn gpu_triu_u16(
    input: &CudaSlice<u16>,
    batch: usize,
    rows: usize,
    cols: usize,
    k: i64,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    let mut out = alloc_zeros_bf16(batch * rows * cols, device)?;
    launch_triangular(
        input,
        &mut out,
        batch,
        rows,
        cols,
        k,
        device,
        TRIANGULAR_U16_PTX,
        "triangular_u16_kernel",
        OP_TRIU,
        2,
    )?;
    Ok(out)
}

/// `tril` over a raw f16/bf16 `[batch.., rows, cols]` payload buffer.
pub fn gpu_tril_u16(
    input: &CudaSlice<u16>,
    batch: usize,
    rows: usize,
    cols: usize,
    k: i64,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    let mut out = alloc_zeros_bf16(batch * rows * cols, device)?;
    launch_triangular(
        input,
        &mut out,
        batch,
        rows,
        cols,
        k,
        device,
        TRIANGULAR_U16_PTX,
        "triangular_u16_kernel",
        OP_TRIL,
        2,
    )?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transfer::{cpu_to_gpu, gpu_to_cpu};

    fn dev() -> GpuDevice {
        GpuDevice::new(0).expect("cuda device")
    }

    /// CPU reference matching `ferrotorch_core::ops::tensor_ops::{triu,tril}`:
    /// triu keeps `c - r >= k`, tril keeps `c - r <= k`, else 0.0.
    fn cpu_ref(data: &[f32], rows: usize, cols: usize, k: i64, triu: bool) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let diff = c as i64 - r as i64;
                let keep = if triu { diff >= k } else { diff <= k };
                if keep {
                    out[r * cols + c] = data[r * cols + c];
                }
            }
        }
        out
    }

    #[test]
    fn triu_f32_main_diag() {
        let d = dev();
        // 3x3 matrix 1..9, triu k=0 keeps upper triangle incl. diagonal.
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let h = cpu_to_gpu(&data, &d).unwrap();
        let out = gpu_triu_f32(&h, 1, 3, 3, 0, &d).unwrap();
        let got = gpu_to_cpu(&out, &d).unwrap();
        let want = cpu_ref(&data, 3, 3, 0, true);
        assert_eq!(&got[..9], &want[..]);
        // explicit: lower-left zeroed
        assert_eq!(&got[..9], &[1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]);
    }

    #[test]
    fn tril_f32_main_diag() {
        let d = dev();
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let h = cpu_to_gpu(&data, &d).unwrap();
        let out = gpu_tril_f32(&h, 1, 3, 3, 0, &d).unwrap();
        let got = gpu_to_cpu(&out, &d).unwrap();
        let want = cpu_ref(&data, 3, 3, 0, false);
        assert_eq!(&got[..9], &want[..]);
        assert_eq!(&got[..9], &[1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn triu_f32_negative_diag() {
        let d = dev();
        // k=-1 keeps the sub-diagonal too.
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let h = cpu_to_gpu(&data, &d).unwrap();
        let out = gpu_triu_f32(&h, 1, 3, 3, -1, &d).unwrap();
        let got = gpu_to_cpu(&out, &d).unwrap();
        let want = cpu_ref(&data, 3, 3, -1, true);
        assert_eq!(&got[..9], &want[..]);
        assert_eq!(&got[..9], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 8.0, 9.0]);
    }

    #[test]
    fn tril_f32_positive_diag() {
        let d = dev();
        // k=1 keeps the super-diagonal too.
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let h = cpu_to_gpu(&data, &d).unwrap();
        let out = gpu_tril_f32(&h, 1, 3, 3, 1, &d).unwrap();
        let got = gpu_to_cpu(&out, &d).unwrap();
        let want = cpu_ref(&data, 3, 3, 1, false);
        assert_eq!(&got[..9], &want[..]);
        assert_eq!(&got[..9], &[1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn triu_tril_f32_nonsquare() {
        let d = dev();
        // 2x4 rectangular.
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let h = cpu_to_gpu(&data, &d).unwrap();
        let up = gpu_triu_f32(&h, 1, 2, 4, 0, &d).unwrap();
        let lo = gpu_tril_f32(&h, 1, 2, 4, 0, &d).unwrap();
        assert_eq!(
            &gpu_to_cpu(&up, &d).unwrap()[..8],
            &cpu_ref(&data, 2, 4, 0, true)[..]
        );
        assert_eq!(
            &gpu_to_cpu(&lo, &d).unwrap()[..8],
            &cpu_ref(&data, 2, 4, 0, false)[..]
        );
    }

    #[test]
    fn triu_f64_main_diag() {
        let d = dev();
        let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let h = cpu_to_gpu(&data, &d).unwrap();
        let out = gpu_triu_f64(&h, 1, 3, 3, 0, &d).unwrap();
        let got = gpu_to_cpu(&out, &d).unwrap();
        assert_eq!(&got[..9], &[1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]);
    }

    /// Batched: two `[3,3]` matrices (batch=2). The triu mask must apply
    /// per-matrix, so the row index resets across the batch boundary. Mirrors
    /// `aten/src/ATen/native/cuda/TriangularOps.cu:120`.
    #[test]
    fn triu_f32_batched_two_3x3() {
        let d = dev();
        // arange(18) reshaped [2,3,3].
        let data: Vec<f32> = (0..18).map(|i| i as f32).collect();
        let h = cpu_to_gpu(&data, &d).unwrap();
        let out = gpu_triu_f32(&h, 2, 3, 3, 0, &d).unwrap();
        let got = gpu_to_cpu(&out, &d).unwrap();
        // torch.triu(arange(18).reshape(2,3,3), 0).flatten()
        assert_eq!(
            &got[..18],
            &[
                0.0f32, 1.0, 2.0, 0.0, 4.0, 5.0, 0.0, 0.0, 8.0, // matrix 0
                9.0, 10.0, 11.0, 0.0, 13.0, 14.0, 0.0, 0.0, 17.0, // matrix 1
            ]
        );
    }
}
