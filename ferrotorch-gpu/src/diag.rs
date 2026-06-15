//! Diagonal GPU compute kernels — `torch.diag` / `torch.diagflat`
//! (crosslink #1545 / sub #1535).
//!
//! Hand-written PTX owned by Rust (no CUDA C++, no nvrtc), loaded via
//! [`crate::module_cache::get_or_compile`] exactly like [`crate::triangular`].
//!
//! # Semantics (PyTorch parity)
//!
//! `torch.diag` is `diag_embed` for a 1-D input and `diagonal_copy` for a 2-D
//! input (`aten/src/ATen/native/TensorShape.cpp:4610`
//! `if (ndim == 1) return at::diag_embed(self, offset); else return
//! at::diagonal_copy(self, offset);`). Both are pure gather/scatter — each
//! output (or input) element is copied to/from one slot with no arithmetic, so
//! the GPU result is **bit-for-bit identical** to the ferrotorch CPU `diag`
//! and to `torch.diag` — no float-tolerance question.
//!
//! ## diag_embed (1-D `n` -> 2-D `[size, size]`, `size = n + |k|`)
//!
//! Output is zero everywhere except the `k`-th diagonal. For input element `i`
//! (`i in [0, n)`) the destination cell is `(r, c) = (i, i + k)` when `k >= 0`
//! and `(r, c) = (i + |k|, i)` when `k < 0` — matching the ferrotorch CPU path
//! (`(i, i + offset)` / `(i + offset, i)`, `offset = |k|`) and PyTorch's
//! `diag_embed` offset convention. One thread per input element scatters one
//! value into a pre-zeroed buffer.
//!
//! ## diag_extract (2-D `[rows, cols]` -> 1-D `diag_len`)
//!
//! `start = (0, k)` when `k >= 0`, `(|k|, 0)` when `k < 0`;
//! `diag_len = min(rows - start_r, cols - start_c)`. Output element `i` reads
//! `in[(start_r + i) * cols + (start_c + i)]`. One thread per output element.
//!
//! ## REQ status (per `.design/ferrotorch-gpu/diag.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (f32 diag_embed) | SHIPPED | `pub fn gpu_diag_embed_f32 in diag.rs`; consumer `CudaBackendImpl::diag_embed_f32 in backend_impl.rs` dispatched from `ops::tensor_ops::diag` |
//! | REQ-2 (f32 diag_extract) | SHIPPED | `pub fn gpu_diag_extract_f32 in diag.rs`; consumer `CudaBackendImpl::diag_extract_f32 in backend_impl.rs` dispatched from `ops::tensor_ops::diag` |
//! | REQ-3 (f64 variants) | SHIPPED | `pub fn gpu_diag_embed_f64` / `gpu_diag_extract_f64 in diag.rs`; consumer `CudaBackendImpl::diag_embed_f64`/`diag_extract_f64 in backend_impl.rs` |
//! | REQ-4 (signed offset) | SHIPPED | `setp.lt.s32 %k` branch selecting `(i, i+k)` vs `(i+|k|, i)` in the embed PTX, `start` shift in extract; verified by `diag_embed_f32_negative_offset` / `diag_extract_f32_positive_offset` unit tests |

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

// ===========================================================================
// diag_embed (1-D -> 2-D scatter)
//
// Params: (in_ptr, out_ptr, n, size, k)
//   in  : V[n]                 (1-D source)
//   out : V[size * size]       (pre-zeroed by the caller; size = n + |k|)
// Thread i in [0, n): r/c chosen by sign of k; out[r*size + c] = in[i].
//   k >= 0 : (r, c) = (i,        i + k)
//   k <  0 : (r, c) = (i + (-k), i)
// (matches ferrotorch CPU diag 1-D path + torch diag_embed offset convention)
// ===========================================================================
const DIAG_EMBED_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry diag_embed_f32_kernel(
    .param .u64 in_ptr, .param .u64 out_ptr,
    .param .u32 n, .param .u32 size, .param .s32 k
) {
    .reg .u32 %gtid, %bid, %bdim, %tdx, %n, %size, %r, %c, %lin;
    .reg .s32 %k_r, %i_s, %absk;
    .reg .u64 %in, %out, %off, %addr;
    .reg .f32 %v;
    .reg .pred %p, %kneg;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n, [n];
    ld.param.u32 %size, [size];
    ld.param.s32 %k_r, [k];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %tdx, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %tdx;
    setp.ge.u32 %p, %gtid, %n;
    @%p bra DONE;

    cvt.s32.u32 %i_s, %gtid;
    setp.lt.s32 %kneg, %k_r, 0;
    @%kneg bra KNEG;
    // k >= 0: r = i, c = i + k
    mov.u32 %r, %gtid;
    add.s32 %i_s, %i_s, %k_r;
    cvt.u32.s32 %c, %i_s;
    bra COMPUTE;
KNEG:
    // k < 0: r = i + (-k), c = i
    sub.s32 %absk, 0, %k_r;
    add.s32 %i_s, %i_s, %absk;
    cvt.u32.s32 %r, %i_s;
    mov.u32 %c, %gtid;
COMPUTE:
    mad.lo.u32 %lin, %r, %size, %c;

    // load in[gtid]
    cvt.u64.u32 %off, %gtid;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in, %off;
    ld.global.f32 %v, [%addr];
    // store out[lin]
    cvt.u64.u32 %off, %lin;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %out, %off;
    st.global.f32 [%addr], %v;
DONE:
    ret;
}
";

const DIAG_EMBED_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry diag_embed_f64_kernel(
    .param .u64 in_ptr, .param .u64 out_ptr,
    .param .u32 n, .param .u32 size, .param .s32 k
) {
    .reg .u32 %gtid, %bid, %bdim, %tdx, %n, %size, %r, %c, %lin;
    .reg .s32 %k_r, %i_s, %absk;
    .reg .u64 %in, %out, %off, %addr;
    .reg .f64 %v;
    .reg .pred %p, %kneg;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n, [n];
    ld.param.u32 %size, [size];
    ld.param.s32 %k_r, [k];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %tdx, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %tdx;
    setp.ge.u32 %p, %gtid, %n;
    @%p bra DONE;

    cvt.s32.u32 %i_s, %gtid;
    setp.lt.s32 %kneg, %k_r, 0;
    @%kneg bra KNEG;
    mov.u32 %r, %gtid;
    add.s32 %i_s, %i_s, %k_r;
    cvt.u32.s32 %c, %i_s;
    bra COMPUTE;
KNEG:
    sub.s32 %absk, 0, %k_r;
    add.s32 %i_s, %i_s, %absk;
    cvt.u32.s32 %r, %i_s;
    mov.u32 %c, %gtid;
COMPUTE:
    mad.lo.u32 %lin, %r, %size, %c;

    cvt.u64.u32 %off, %gtid;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %in, %off;
    ld.global.f64 %v, [%addr];
    cvt.u64.u32 %off, %lin;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %out, %off;
    st.global.f64 [%addr], %v;
DONE:
    ret;
}
";

const DIAG_EMBED_U16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry diag_embed_u16_kernel(
    .param .u64 in_ptr, .param .u64 out_ptr,
    .param .u32 n, .param .u32 size, .param .s32 k
) {
    .reg .u32 %gtid, %bid, %bdim, %tdx, %n, %size, %r, %c, %lin;
    .reg .s32 %k_r, %i_s, %absk;
    .reg .u64 %in, %out, %off, %addr;
    .reg .b16 %v;
    .reg .pred %p, %kneg;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n, [n];
    ld.param.u32 %size, [size];
    ld.param.s32 %k_r, [k];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %tdx, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %tdx;
    setp.ge.u32 %p, %gtid, %n;
    @%p bra DONE;

    cvt.s32.u32 %i_s, %gtid;
    setp.lt.s32 %kneg, %k_r, 0;
    @%kneg bra KNEG;
    mov.u32 %r, %gtid;
    add.s32 %i_s, %i_s, %k_r;
    cvt.u32.s32 %c, %i_s;
    bra COMPUTE;
KNEG:
    sub.s32 %absk, 0, %k_r;
    add.s32 %i_s, %i_s, %absk;
    cvt.u32.s32 %r, %i_s;
    mov.u32 %c, %gtid;
COMPUTE:
    mad.lo.u32 %lin, %r, %size, %c;

    cvt.u64.u32 %off, %gtid;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %in, %off;
    ld.global.b16 %v, [%addr];
    cvt.u64.u32 %off, %lin;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %out, %off;
    st.global.b16 [%addr], %v;
DONE:
    ret;
}
";

// ===========================================================================
// diag_extract (2-D -> 1-D gather)
//
// Params: (in_ptr, out_ptr, diag_len, cols, start_r, start_c)
//   in  : V[rows * cols]
//   out : V[diag_len]
// Thread i in [0, diag_len): out[i] = in[(start_r + i) * cols + (start_c + i)].
// ===========================================================================
const DIAG_EXTRACT_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry diag_extract_f32_kernel(
    .param .u64 in_ptr, .param .u64 out_ptr,
    .param .u32 diag_len, .param .u32 cols, .param .u32 start_r, .param .u32 start_c
) {
    .reg .u32 %gtid, %bid, %bdim, %tdx, %dl, %cols, %sr, %sc, %row, %col, %lin;
    .reg .u64 %in, %out, %off, %addr;
    .reg .f32 %v;
    .reg .pred %p;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %dl, [diag_len];
    ld.param.u32 %cols, [cols];
    ld.param.u32 %sr, [start_r];
    ld.param.u32 %sc, [start_c];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %tdx, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %tdx;
    setp.ge.u32 %p, %gtid, %dl;
    @%p bra DONE;

    add.u32 %row, %sr, %gtid;
    add.u32 %col, %sc, %gtid;
    mad.lo.u32 %lin, %row, %cols, %col;

    cvt.u64.u32 %off, %lin;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in, %off;
    ld.global.f32 %v, [%addr];
    cvt.u64.u32 %off, %gtid;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %out, %off;
    st.global.f32 [%addr], %v;
DONE:
    ret;
}
";

const DIAG_EXTRACT_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry diag_extract_f64_kernel(
    .param .u64 in_ptr, .param .u64 out_ptr,
    .param .u32 diag_len, .param .u32 cols, .param .u32 start_r, .param .u32 start_c
) {
    .reg .u32 %gtid, %bid, %bdim, %tdx, %dl, %cols, %sr, %sc, %row, %col, %lin;
    .reg .u64 %in, %out, %off, %addr;
    .reg .f64 %v;
    .reg .pred %p;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %dl, [diag_len];
    ld.param.u32 %cols, [cols];
    ld.param.u32 %sr, [start_r];
    ld.param.u32 %sc, [start_c];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %tdx, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %tdx;
    setp.ge.u32 %p, %gtid, %dl;
    @%p bra DONE;

    add.u32 %row, %sr, %gtid;
    add.u32 %col, %sc, %gtid;
    mad.lo.u32 %lin, %row, %cols, %col;

    cvt.u64.u32 %off, %lin;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %in, %off;
    ld.global.f64 %v, [%addr];
    cvt.u64.u32 %off, %gtid;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %out, %off;
    st.global.f64 [%addr], %v;
DONE:
    ret;
}
";

const DIAG_EXTRACT_U16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry diag_extract_u16_kernel(
    .param .u64 in_ptr, .param .u64 out_ptr,
    .param .u32 diag_len, .param .u32 cols, .param .u32 start_r, .param .u32 start_c
) {
    .reg .u32 %gtid, %bid, %bdim, %tdx, %dl, %cols, %sr, %sc, %row, %col, %lin;
    .reg .u64 %in, %out, %off, %addr;
    .reg .b16 %v;
    .reg .pred %p;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %dl, [diag_len];
    ld.param.u32 %cols, [cols];
    ld.param.u32 %sr, [start_r];
    ld.param.u32 %sc, [start_c];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %tdx, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %tdx;
    setp.ge.u32 %p, %gtid, %dl;
    @%p bra DONE;

    add.u32 %row, %sr, %gtid;
    add.u32 %col, %sc, %gtid;
    mad.lo.u32 %lin, %row, %cols, %col;

    cvt.u64.u32 %off, %lin;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %in, %off;
    ld.global.b16 %v, [%addr];
    cvt.u64.u32 %off, %gtid;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %out, %off;
    st.global.b16 [%addr], %v;
DONE:
    ret;
}
";

// diag_scatter is the VJP of diag_extract / diagonal: scatter a 1-D gradient
// onto a rectangular zero matrix.
const DIAG_SCATTER_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry diag_scatter_f32_kernel(
    .param .u64 in_ptr, .param .u64 out_ptr,
    .param .u32 diag_len, .param .u32 cols, .param .u32 start_r, .param .u32 start_c
) {
    .reg .u32 %gtid, %bid, %bdim, %tdx, %dl, %cols, %sr, %sc, %row, %col, %lin;
    .reg .u64 %in, %out, %off, %addr;
    .reg .f32 %v;
    .reg .pred %p;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %dl, [diag_len];
    ld.param.u32 %cols, [cols];
    ld.param.u32 %sr, [start_r];
    ld.param.u32 %sc, [start_c];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %tdx, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %tdx;
    setp.ge.u32 %p, %gtid, %dl;
    @%p bra DONE;

    add.u32 %row, %sr, %gtid;
    add.u32 %col, %sc, %gtid;
    mad.lo.u32 %lin, %row, %cols, %col;

    cvt.u64.u32 %off, %gtid;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in, %off;
    ld.global.f32 %v, [%addr];
    cvt.u64.u32 %off, %lin;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %out, %off;
    st.global.f32 [%addr], %v;
DONE:
    ret;
}
";

const DIAG_SCATTER_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry diag_scatter_f64_kernel(
    .param .u64 in_ptr, .param .u64 out_ptr,
    .param .u32 diag_len, .param .u32 cols, .param .u32 start_r, .param .u32 start_c
) {
    .reg .u32 %gtid, %bid, %bdim, %tdx, %dl, %cols, %sr, %sc, %row, %col, %lin;
    .reg .u64 %in, %out, %off, %addr;
    .reg .f64 %v;
    .reg .pred %p;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %dl, [diag_len];
    ld.param.u32 %cols, [cols];
    ld.param.u32 %sr, [start_r];
    ld.param.u32 %sc, [start_c];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %tdx, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %tdx;
    setp.ge.u32 %p, %gtid, %dl;
    @%p bra DONE;

    add.u32 %row, %sr, %gtid;
    add.u32 %col, %sc, %gtid;
    mad.lo.u32 %lin, %row, %cols, %col;

    cvt.u64.u32 %off, %gtid;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %in, %off;
    ld.global.f64 %v, [%addr];
    cvt.u64.u32 %off, %lin;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %out, %off;
    st.global.f64 [%addr], %v;
DONE:
    ret;
}
";

const DIAG_SCATTER_U16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry diag_scatter_u16_kernel(
    .param .u64 in_ptr, .param .u64 out_ptr,
    .param .u32 diag_len, .param .u32 cols, .param .u32 start_r, .param .u32 start_c
) {
    .reg .u32 %gtid, %bid, %bdim, %tdx, %dl, %cols, %sr, %sc, %row, %col, %lin;
    .reg .u64 %in, %out, %off, %addr;
    .reg .b16 %v;
    .reg .pred %p;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %dl, [diag_len];
    ld.param.u32 %cols, [cols];
    ld.param.u32 %sr, [start_r];
    ld.param.u32 %sc, [start_c];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %tdx, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %tdx;
    setp.ge.u32 %p, %gtid, %dl;
    @%p bra DONE;

    add.u32 %row, %sr, %gtid;
    add.u32 %col, %sc, %gtid;
    mad.lo.u32 %lin, %row, %cols, %col;

    cvt.u64.u32 %off, %gtid;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %in, %off;
    ld.global.b16 %v, [%addr];
    cvt.u64.u32 %off, %lin;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %out, %off;
    st.global.b16 [%addr], %v;
DONE:
    ret;
}
";

fn diag_start_len(rows: usize, cols: usize, k: i64) -> (usize, usize, usize) {
    let (start_r, start_c) = if k >= 0 {
        (0usize, k as usize)
    } else {
        (k.unsigned_abs() as usize, 0usize)
    };
    let diag_len = rows
        .saturating_sub(start_r)
        .min(cols.saturating_sub(start_c));
    (start_r, start_c, diag_len)
}

fn checked_diag_embed_size(n: usize, k: i64) -> GpuResult<usize> {
    let offset = usize::try_from(k.unsigned_abs()).map_err(|_| GpuError::ShapeMismatch {
        op: "diag_embed",
        expected: vec![usize::MAX],
        got: vec![n],
    })?;
    let size = n
        .checked_add(offset)
        .ok_or_else(|| GpuError::ShapeMismatch {
            op: "diag_embed",
            expected: vec![usize::MAX],
            got: vec![n, offset],
        })?;
    size.checked_mul(size)
        .ok_or_else(|| GpuError::ShapeMismatch {
            op: "diag_embed",
            expected: vec![usize::MAX],
            got: vec![size, size],
        })?;
    Ok(size)
}

/// Scatter a 1-D `n`-element buffer onto the `k`-th diagonal of a fresh
/// zero-initialised `[size, size]` buffer (`size = n + |k|`). One thread per
/// input element. Returns the resident output buffer.
fn launch_diag_embed<V: DeviceRepr + ValidAsZeroBits>(
    in_slice: &CudaSlice<V>,
    out_slice: &mut CudaSlice<V>,
    n: usize,
    size: usize,
    k: i64,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
) -> GpuResult<()> {
    if n == 0 {
        return Ok(());
    }
    if in_slice.len() < n {
        return Err(GpuError::LengthMismatch {
            a: in_slice.len(),
            b: n,
        });
    }
    let total = size
        .checked_mul(size)
        .ok_or(GpuError::LengthMismatch { a: size, b: size })?;
    if out_slice.len() < total {
        return Err(GpuError::LengthMismatch {
            a: out_slice.len(),
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
    let cfg = launch_1d(n);
    let n_u = n as u32;
    let size_u = size as u32;
    let k_i32 = k.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    // SAFETY:
    // - `f` is the PTX entry `kernel_name`; its 5-arg signature
    //   (in_ptr, out_ptr, n, size, k) matches the args pushed below in order.
    // - `in_slice` holds >= `n` `V`-elements (checked); `out_slice` holds
    //   >= `size*size` (checked) and is a fresh zeroed buffer (only `&mut`,
    //   distinct allocation from `in_slice`).
    // - Each thread `i in [0, n)` (bound-checked by `setp.ge.u32 %p, %gtid, %n`)
    //   reads `in[i]` and writes one in-bounds cell `out[r*size + c]` with
    //   `r, c < size` (since `i < n` and `size = n + |k|`).
    unsafe {
        stream
            .launch_builder(&f)
            .arg(in_slice)
            .arg(out_slice)
            .arg(&n_u)
            .arg(&size_u)
            .arg(&k_i32)
            .launch(cfg)?;
    }
    Ok(())
}

/// Gather the `k`-th diagonal of a `[rows, cols]` buffer into a fresh
/// `diag_len`-element buffer. One thread per output element. Returns the
/// resident output.
#[allow(clippy::too_many_arguments)]
fn launch_diag_extract<V: DeviceRepr + ValidAsZeroBits>(
    in_slice: &CudaSlice<V>,
    out_slice: &mut CudaSlice<V>,
    rows: usize,
    cols: usize,
    diag_len: usize,
    start_r: usize,
    start_c: usize,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
) -> GpuResult<()> {
    if diag_len == 0 {
        return Ok(());
    }
    let in_total = rows
        .checked_mul(cols)
        .ok_or(GpuError::LengthMismatch { a: rows, b: cols })?;
    if in_slice.len() < in_total {
        return Err(GpuError::LengthMismatch {
            a: in_slice.len(),
            b: in_total,
        });
    }
    if out_slice.len() < diag_len {
        return Err(GpuError::LengthMismatch {
            a: out_slice.len(),
            b: diag_len,
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
    let cfg = launch_1d(diag_len);
    let dl_u = diag_len as u32;
    let cols_u = cols as u32;
    let sr_u = start_r as u32;
    let sc_u = start_c as u32;
    // SAFETY:
    // - `f` is the PTX entry `kernel_name`; its 6-arg signature
    //   (in_ptr, out_ptr, diag_len, cols, start_r, start_c) matches the args
    //   pushed below in order.
    // - `in_slice` holds >= `rows*cols` (checked); `out_slice` >= `diag_len`
    //   (checked), distinct allocation, only `&mut`.
    // - Each thread `i in [0, diag_len)` (bound-checked) reads
    //   `in[(start_r+i)*cols + (start_c+i)]`, which is in bounds because
    //   `start_r + diag_len <= rows` and `start_c + diag_len <= cols` (the
    //   caller derives `diag_len = min(rows-start_r, cols-start_c)`).
    unsafe {
        stream
            .launch_builder(&f)
            .arg(in_slice)
            .arg(out_slice)
            .arg(&dl_u)
            .arg(&cols_u)
            .arg(&sr_u)
            .arg(&sc_u)
            .launch(cfg)?;
    }
    Ok(())
}

/// Scatter a 1-D `diag_len`-element gradient into a fresh zeroed rectangular
/// `[rows, cols]` output. One thread per input element.
#[allow(clippy::too_many_arguments)]
fn launch_diag_scatter<V: DeviceRepr + ValidAsZeroBits>(
    in_slice: &CudaSlice<V>,
    out_slice: &mut CudaSlice<V>,
    rows: usize,
    cols: usize,
    diag_len: usize,
    start_r: usize,
    start_c: usize,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
) -> GpuResult<()> {
    if diag_len == 0 {
        return Ok(());
    }
    if in_slice.len() < diag_len {
        return Err(GpuError::LengthMismatch {
            a: in_slice.len(),
            b: diag_len,
        });
    }
    let out_total = rows
        .checked_mul(cols)
        .ok_or(GpuError::LengthMismatch { a: rows, b: cols })?;
    if out_slice.len() < out_total {
        return Err(GpuError::LengthMismatch {
            a: out_slice.len(),
            b: out_total,
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
    let cfg = launch_1d(diag_len);
    let dl_u = diag_len as u32;
    let cols_u = cols as u32;
    let sr_u = start_r as u32;
    let sc_u = start_c as u32;
    // SAFETY:
    // - `f` is the PTX entry `kernel_name`; its 6-arg signature
    //   (in_ptr, out_ptr, diag_len, cols, start_r, start_c) matches the pushed
    //   args.
    // - `in_slice` holds >= `diag_len`; `out_slice` holds >= `rows*cols` and is
    //   a fresh zeroed output.
    // - Each thread writes one in-bounds diagonal slot because `diag_len` was
    //   derived as `min(rows-start_r, cols-start_c)`.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(in_slice)
            .arg(out_slice)
            .arg(&dl_u)
            .arg(&cols_u)
            .arg(&sr_u)
            .arg(&sc_u)
            .launch(cfg)?;
    }
    Ok(())
}

/// `diag` for a 1-D f32 input: scatter `n` elements onto the `k`-th diagonal
/// of a `[size, size]` matrix (`size = n + |k|`). Returns the resident output.
pub fn gpu_diag_embed_f32(
    input: &CudaBuffer<f32>,
    n: usize,
    k: i64,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    let size = checked_diag_embed_size(n, k)?;
    let mut out = alloc_zeros_f32(size * size, device)?;
    launch_diag_embed(
        input.inner(),
        out.inner_mut(),
        n,
        size,
        k,
        device,
        DIAG_EMBED_F32_PTX,
        "diag_embed_f32_kernel",
    )?;
    Ok(out)
}

/// `diag` for a 1-D f64 input. See [`gpu_diag_embed_f32`].
pub fn gpu_diag_embed_f64(
    input: &CudaBuffer<f64>,
    n: usize,
    k: i64,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    let size = checked_diag_embed_size(n, k)?;
    let mut out = alloc_zeros_f64(size * size, device)?;
    launch_diag_embed(
        input.inner(),
        out.inner_mut(),
        n,
        size,
        k,
        device,
        DIAG_EMBED_F64_PTX,
        "diag_embed_f64_kernel",
    )?;
    Ok(out)
}

/// `diag` for a 1-D raw f16/bf16 payload buffer.
pub fn gpu_diag_embed_u16(
    input: &CudaSlice<u16>,
    n: usize,
    k: i64,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    let size = checked_diag_embed_size(n, k)?;
    let mut out = alloc_zeros_bf16(size * size, device)?;
    launch_diag_embed(
        input,
        &mut out,
        n,
        size,
        k,
        device,
        DIAG_EMBED_U16_PTX,
        "diag_embed_u16_kernel",
    )?;
    Ok(out)
}

/// `diag` for a 2-D f32 input: gather the `k`-th diagonal of `[rows, cols]`
/// into a `diag_len`-element vector. `start`/`diag_len` follow the ferrotorch
/// CPU `diag` 2-D path.
pub fn gpu_diag_extract_f32(
    input: &CudaBuffer<f32>,
    rows: usize,
    cols: usize,
    k: i64,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    let (start_r, start_c, diag_len) = diag_start_len(rows, cols, k);
    let mut out = alloc_zeros_f32(diag_len.max(1), device)?;
    launch_diag_extract(
        input.inner(),
        out.inner_mut(),
        rows,
        cols,
        diag_len,
        start_r,
        start_c,
        device,
        DIAG_EXTRACT_F32_PTX,
        "diag_extract_f32_kernel",
    )?;
    Ok(out)
}

/// `diag` for a 2-D f64 input. See [`gpu_diag_extract_f32`].
pub fn gpu_diag_extract_f64(
    input: &CudaBuffer<f64>,
    rows: usize,
    cols: usize,
    k: i64,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    let (start_r, start_c, diag_len) = diag_start_len(rows, cols, k);
    let mut out = alloc_zeros_f64(diag_len.max(1), device)?;
    launch_diag_extract(
        input.inner(),
        out.inner_mut(),
        rows,
        cols,
        diag_len,
        start_r,
        start_c,
        device,
        DIAG_EXTRACT_F64_PTX,
        "diag_extract_f64_kernel",
    )?;
    Ok(out)
}

/// `diag` for a 2-D raw f16/bf16 payload buffer.
pub fn gpu_diag_extract_u16(
    input: &CudaSlice<u16>,
    rows: usize,
    cols: usize,
    k: i64,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    let (start_r, start_c, diag_len) = diag_start_len(rows, cols, k);
    let mut out = alloc_zeros_bf16(diag_len.max(1), device)?;
    launch_diag_extract(
        input,
        &mut out,
        rows,
        cols,
        diag_len,
        start_r,
        start_c,
        device,
        DIAG_EXTRACT_U16_PTX,
        "diag_extract_u16_kernel",
    )?;
    Ok(out)
}

/// Scatter a 1-D f32 gradient onto the `k`-th diagonal of a zero `[rows, cols]`.
pub fn gpu_diag_scatter_f32(
    input: &CudaBuffer<f32>,
    rows: usize,
    cols: usize,
    k: i64,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    let (start_r, start_c, diag_len) = diag_start_len(rows, cols, k);
    let mut out = alloc_zeros_f32(rows * cols, device)?;
    launch_diag_scatter(
        input.inner(),
        out.inner_mut(),
        rows,
        cols,
        diag_len,
        start_r,
        start_c,
        device,
        DIAG_SCATTER_F32_PTX,
        "diag_scatter_f32_kernel",
    )?;
    Ok(out)
}

/// Scatter a 1-D f64 gradient onto the `k`-th diagonal of a zero `[rows, cols]`.
pub fn gpu_diag_scatter_f64(
    input: &CudaBuffer<f64>,
    rows: usize,
    cols: usize,
    k: i64,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    let (start_r, start_c, diag_len) = diag_start_len(rows, cols, k);
    let mut out = alloc_zeros_f64(rows * cols, device)?;
    launch_diag_scatter(
        input.inner(),
        out.inner_mut(),
        rows,
        cols,
        diag_len,
        start_r,
        start_c,
        device,
        DIAG_SCATTER_F64_PTX,
        "diag_scatter_f64_kernel",
    )?;
    Ok(out)
}

/// Scatter a 1-D raw f16/bf16 gradient onto the `k`-th diagonal of a zero
/// `[rows, cols]` payload buffer.
pub fn gpu_diag_scatter_u16(
    input: &CudaSlice<u16>,
    rows: usize,
    cols: usize,
    k: i64,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    let (start_r, start_c, diag_len) = diag_start_len(rows, cols, k);
    let mut out = alloc_zeros_bf16(rows * cols, device)?;
    launch_diag_scatter(
        input,
        &mut out,
        rows,
        cols,
        diag_len,
        start_r,
        start_c,
        device,
        DIAG_SCATTER_U16_PTX,
        "diag_scatter_u16_kernel",
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

    /// CPU reference matching `ferrotorch_core::ops::tensor_ops::diag` 1-D path.
    fn cpu_embed(data: &[f32], k: i64) -> Vec<f32> {
        let n = data.len();
        let offset = k.unsigned_abs() as usize;
        let size = n + offset;
        let mut out = vec![0.0f32; size * size];
        for (i, &val) in data.iter().enumerate() {
            let (r, c) = if k >= 0 {
                (i, i + offset)
            } else {
                (i + offset, i)
            };
            out[r * size + c] = val;
        }
        out
    }

    #[test]
    fn diag_embed_f32_main() {
        let d = dev();
        // torch.diag(tensor([1,2,3])) main diagonal
        let data = vec![1.0f32, 2.0, 3.0];
        let h = cpu_to_gpu(&data, &d).unwrap();
        let out = gpu_diag_embed_f32(&h, 3, 0, &d).unwrap();
        let got = gpu_to_cpu(&out, &d).unwrap();
        assert_eq!(&got[..9], &cpu_embed(&data, 0)[..]);
        assert_eq!(&got[..9], &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn diag_embed_f32_positive_offset() {
        let d = dev();
        // torch.diag(tensor([1,2]), 1) -> 3x3 with 1,2 on the super-diagonal.
        let data = vec![1.0f32, 2.0];
        let h = cpu_to_gpu(&data, &d).unwrap();
        let out = gpu_diag_embed_f32(&h, 2, 1, &d).unwrap();
        let got = gpu_to_cpu(&out, &d).unwrap();
        assert_eq!(&got[..9], &cpu_embed(&data, 1)[..]);
        assert_eq!(&got[..9], &[0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn diag_embed_f32_negative_offset() {
        let d = dev();
        // torch.diag(tensor([1,2]), -1) -> sub-diagonal.
        let data = vec![1.0f32, 2.0];
        let h = cpu_to_gpu(&data, &d).unwrap();
        let out = gpu_diag_embed_f32(&h, 2, -1, &d).unwrap();
        let got = gpu_to_cpu(&out, &d).unwrap();
        assert_eq!(&got[..9], &cpu_embed(&data, -1)[..]);
        assert_eq!(&got[..9], &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn diag_extract_f32_main() {
        let d = dev();
        // torch.diag(arange(1,10).reshape(3,3)) -> [1,5,9]
        let data: Vec<f32> = (1..=9).map(|i| i as f32).collect();
        let h = cpu_to_gpu(&data, &d).unwrap();
        let out = gpu_diag_extract_f32(&h, 3, 3, 0, &d).unwrap();
        let got = gpu_to_cpu(&out, &d).unwrap();
        assert_eq!(&got[..3], &[1.0, 5.0, 9.0]);
    }

    #[test]
    fn diag_extract_f32_positive_offset() {
        let d = dev();
        // torch.diag(arange(1,10).reshape(3,3), 1) -> [2,6]
        let data: Vec<f32> = (1..=9).map(|i| i as f32).collect();
        let h = cpu_to_gpu(&data, &d).unwrap();
        let out = gpu_diag_extract_f32(&h, 3, 3, 1, &d).unwrap();
        let got = gpu_to_cpu(&out, &d).unwrap();
        assert_eq!(&got[..2], &[2.0, 6.0]);
    }

    #[test]
    fn diag_extract_f32_negative_offset() {
        let d = dev();
        // torch.diag(arange(1,10).reshape(3,3), -1) -> [4,8]
        let data: Vec<f32> = (1..=9).map(|i| i as f32).collect();
        let h = cpu_to_gpu(&data, &d).unwrap();
        let out = gpu_diag_extract_f32(&h, 3, 3, -1, &d).unwrap();
        let got = gpu_to_cpu(&out, &d).unwrap();
        assert_eq!(&got[..2], &[4.0, 8.0]);
    }

    #[test]
    fn diag_embed_f64_main() {
        let d = dev();
        let data = vec![1.0f64, 2.0, 3.0];
        let h = cpu_to_gpu(&data, &d).unwrap();
        let out = gpu_diag_embed_f64(&h, 3, 0, &d).unwrap();
        let got = gpu_to_cpu(&out, &d).unwrap();
        assert_eq!(&got[..9], &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn diag_extract_f64_main() {
        let d = dev();
        let data: Vec<f64> = (1..=9).map(|i| i as f64).collect();
        let h = cpu_to_gpu(&data, &d).unwrap();
        let out = gpu_diag_extract_f64(&h, 3, 3, 0, &d).unwrap();
        let got = gpu_to_cpu(&out, &d).unwrap();
        assert_eq!(&got[..3], &[1.0, 5.0, 9.0]);
    }
}
