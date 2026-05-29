//! Dim-aware `gather` / `scatter` / `scatter_value` / `scatter_add` GPU
//! kernels — `torch.gather` / `torch.Tensor.scatter_` /
//! `scatter_(value)` / `scatter_add_` (crosslink #1545 / sub #1535).
//!
//! Hand-written PTX owned by Rust (no CUDA C++, no nvrtc), loaded via
//! [`crate::module_cache::get_or_compile`] exactly like
//! [`crate::triangular`] and [`crate::gather_int`].
//!
//! # Semantics (PyTorch parity)
//!
//! These are the *dim-parameterised, full-rank-index* family. For an N-D
//! C-contiguous tensor and an axis `dim`, every shape decomposes into
//! `[outer, axis, inner]` where `outer = product(shape[0..dim])` and
//! `inner = product(shape[dim+1..])`. Within that decomposition, the flat
//! position of element `(o, a, k)` is `o*axis*inner + a*inner + k`.
//!
//! - **gather**: `input` is `[outer, in_dim, inner]`; `index` and `output`
//!   are both `[outer, out_dim, inner]`. Thread `t` (one per output element,
//!   `t in [0, outer*out_dim*inner)`) reads the parallel index `idx[t]`, then
//!   `out[t] = input[o*in_dim*inner + idx[t]*inner + k]` where
//!   `o = t / (out_dim*inner)`, `k = t % inner`. This is exactly
//!   `output[i,j,k] = input[index[i,j,k], j, k]` for `dim=0` (and the
//!   analogous swap for higher dims) — `aten::gather` /
//!   `gather_out_cuda` in `aten/src/ATen/native/cuda/ScatterGatherKernel.cu`.
//!
//! - **scatter**: `output` starts as a CLONE of `input` (`[outer, out_dim,
//!   inner]`); `index` and `src` are `[outer, idx_dim, inner]`. Thread `t`
//!   (one per index element) reads `idx[t]` and writes
//!   `out[o*out_dim*inner + idx[t]*inner + k] = src[t]`. Mirrors
//!   `aten::scatter` (the `tensor_assign` reduce op,
//!   `ScatterGatherKernel.cu:527`).
//!
//! - **scatter_value**: identical to scatter but every written cell takes a
//!   single broadcast scalar `value` instead of `src[t]`. Mirrors the
//!   `scatter.value` overload (`Tensor::scatter_(dim, index, Scalar)`).
//!
//! - **scatter_add**: identical addressing to scatter, but the write is an
//!   ATOMIC add (`out[dst] += src[t]`). Duplicate index values into the same
//!   `dst` accumulate correctly — that is the whole reason for the atomic.
//!   Mirrors `aten::scatter_add` whose CUDA reduce op is `fastAtomicAdd`
//!   (`ScatterGatherKernel.cu:41-44`).
//!
//! - **scatter_add_segments**: the segmented ROW scatter-add used by GNN
//!   message passing (`ferrotorch-core::ops::scatter::scatter_add_segments`).
//!   `src` is `[E, D]`; `index` is a per-ROW `i64` segment id (length `E`,
//!   uploaded from the host `&[i64]`); output is the zero-initialised
//!   `[dim_size, D]` with `out[index[e], :] += src[e, :]` accumulated over all
//!   rows. One thread per `(e, d)` element, `atom.global.add.f{32,64}` so
//!   duplicate segment ids sum. This is the same primitive
//!   `torch.zeros(dim_size, D).index_add_(0, index, src)` computes (and
//!   `torch_scatter.scatter_add(src, index, dim=0, dim_size=N)`).
//!
//! # Index dtype
//!
//! The index is supplied as a GPU-resident `i64` buffer (PyTorch's index
//! tensors are `int64`). The kernels read it with `ld.global.s64`. The core
//! dispatch (`ferrotorch-core/src/ops/indexing.rs`) uploads the host
//! `&[usize]` index as `i64` before calling these launchers.
//!
//! # Out-of-range indices
//!
//! Matching PyTorch CUDA, an out-of-range index value along `dim` is
//! undefined behaviour on the device (no host round-trip to validate — that
//! would defeat the no-CPU contract). The core CPU validator
//! (`validate_gather_shapes`) already rejects OOB indices on the host before
//! the upload, so the resident path only ever sees in-bounds indices in
//! practice; the device kernel itself does not re-check (mirrors upstream).
//!
//! # `.target sm_60`
//!
//! `scatter_add` uses `atom.global.add.f64`, which requires `sm_60+`. The
//! gather/scatter/scatter_value kernels are pure index movement and would
//! compile at `sm_52`, but we hold the whole file at `sm_60` for a single
//! consistent target (the live RTX 3090 is `sm_86`, so this is satisfied).
//!
//! ## REQ status (per `.design/ferrotorch-gpu/scatter_gather.md`)
//!
//! Full evidence rows (impl + non-test production consumer + upstream
//! cites) live in the design doc; this synopsis is a one-line summary per
//! REQ.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (f32 gather/scatter/scatter_value/scatter_add) | SHIPPED | `gpu_gather_dim_f32` / `gpu_scatter_dim_f32` / `gpu_scatter_value_dim_f32` / `gpu_scatter_add_dim_f32` in `scatter_gather_kernels.rs`; consumer `CudaBackendImpl::gather_dim_f32` etc. in `backend_impl.rs` |
//! | REQ-2 (f64 family) | SHIPPED | `gpu_*_dim_f64` in `scatter_gather_kernels.rs`; consumer `CudaBackendImpl::*_dim_f64` in `backend_impl.rs` |
//! | REQ-3 (atomic scatter_add) | SHIPPED | `atom.global.add.f32` / `atom.global.add.f64` in the scatter_add PTX; verified by the duplicate-index unit test `scatter_add_dim_f32_duplicate_indices` |
//! | REQ-4 (dispatch wiring) | SHIPPED | the four `CudaBackendImpl::*_dim_{f32,f64}` overrides; consumer the `is_cuda()` branch of `gather`/`scatter`/`scatter_value`/`scatter_add` in `ferrotorch-core/src/ops/indexing.rs` |
//! | REQ-5 (segmented row scatter-add) | SHIPPED | `gpu_scatter_add_segments_f32`/`_f64` (`atom.global.add.f{32,64}` over a per-row i64 segment index, zero-init output); consumer `CudaBackendImpl::scatter_add_segments_f32`/`_f64` in `backend_impl.rs`, themselves consumed by the `is_cuda()` branch of `ferrotorch_core::ops::scatter::scatter_add_segments` |

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits};

use crate::buffer::CudaBuffer;
use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};
use crate::module_cache::get_or_compile;
use crate::transfer::{alloc_zeros_f32, alloc_zeros_f64};

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
// gather PTX. Params: (in_ptr, idx_ptr, out_ptr, outer, in_dim, out_dim,
//                      inner, total)
//   in  : V[outer * in_dim  * inner]   (C-contiguous)
//   idx : i64[outer * out_dim * inner] (parallel to output)
//   out : V[outer * out_dim * inner]
// Thread t in [0, total = outer*out_dim*inner):
//   o = t / (out_dim*inner); rem = t % (out_dim*inner); k = rem % inner
//   sel = idx[t]
//   src = (o*in_dim + sel)*inner + k
//   out[t] = in[src]
// ($wsh = value byte-width shift "2"=f32 / "3"=f64; $ldv/$stv/$vreg per width)
// ===========================================================================
macro_rules! gather_dim_ptx {
    ($kname:literal, $wsh:literal, $ldv:literal, $stv:literal, $vreg:literal) => {
        concat!(
            ".version 7.0\n.target sm_60\n.address_size 64\n",
            ".visible .entry ",
            $kname,
            "(
    .param .u64 in_ptr, .param .u64 idx_ptr, .param .u64 out_ptr,
    .param .u32 outer, .param .u32 in_dim, .param .u32 out_dim,
    .param .u32 inner, .param .u32 total
) {
    .reg .u32 %gtid, %bid, %bdim, %tot, %indim, %outdim, %inn;
    .reg .u32 %o, %rem, %k, %slab, %sel, %srcelem;
    .reg .u64 %in, %idx, %out, %off, %addr;
    .reg .s64 %selv;
    .reg ",
            $vreg,
            " %v;
    .reg .pred %p;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %idx, [idx_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %tot, [total];
    ld.param.u32 %indim, [in_dim];
    ld.param.u32 %outdim, [out_dim];
    ld.param.u32 %inn, [inner];

    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %gtid, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %gtid;
    setp.ge.u32 %p, %gtid, %tot; @%p bra DONE;

    mul.lo.u32 %slab, %outdim, %inn;
    div.u32 %o, %gtid, %slab;
    rem.u32 %rem, %gtid, %slab;
    rem.u32 %k, %rem, %inn;

    cvt.u64.u32 %off, %gtid; shl.b64 %off, %off, 3; add.u64 %addr, %idx, %off;
    ld.global.s64 %selv, [%addr];
    cvt.u32.s64 %sel, %selv;

    mul.lo.u32 %srcelem, %o, %indim;
    add.u32 %srcelem, %srcelem, %sel;
    mul.lo.u32 %srcelem, %srcelem, %inn;
    add.u32 %srcelem, %srcelem, %k;

    cvt.u64.u32 %off, %srcelem; shl.b64 %off, %off, ",
            $wsh,
            "; add.u64 %addr, %in, %off;
    ",
            $ldv,
            " %v, [%addr];

    cvt.u64.u32 %off, %gtid; shl.b64 %off, %off, ",
            $wsh,
            "; add.u64 %addr, %out, %off;
    ",
            $stv,
            " [%addr], %v;
DONE:
    ret;
}
"
        )
    };
}

// ===========================================================================
// scatter PTX. Params: (out_ptr, idx_ptr, src_ptr, outer, out_dim, idx_dim,
//                       inner, total)
//   out : V[outer * out_dim * inner]   (PRE-CLONED from input by the launcher)
//   idx : i64[outer * idx_dim * inner]
//   src : V[outer * idx_dim * inner]
// Thread t in [0, total = outer*idx_dim*inner):
//   o = t / (idx_dim*inner); rem = t % (idx_dim*inner); k = rem % inner
//   sel = idx[t]
//   dst = (o*out_dim + sel)*inner + k
//   out[dst] = src[t]
// ===========================================================================
macro_rules! scatter_dim_ptx {
    ($kname:literal, $wsh:literal, $ldv:literal, $stv:literal, $vreg:literal) => {
        concat!(
            ".version 7.0\n.target sm_60\n.address_size 64\n",
            ".visible .entry ",
            $kname,
            "(
    .param .u64 out_ptr, .param .u64 idx_ptr, .param .u64 src_ptr,
    .param .u32 outer, .param .u32 out_dim, .param .u32 idx_dim,
    .param .u32 inner, .param .u32 total
) {
    .reg .u32 %gtid, %bid, %bdim, %tot, %outdim, %idxdim, %inn;
    .reg .u32 %o, %rem, %k, %slab, %sel, %dstelem;
    .reg .u64 %out, %idx, %src, %off, %addr;
    .reg .s64 %selv;
    .reg ",
            $vreg,
            " %v;
    .reg .pred %p;

    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %idx, [idx_ptr];
    ld.param.u64 %src, [src_ptr];
    ld.param.u32 %tot, [total];
    ld.param.u32 %outdim, [out_dim];
    ld.param.u32 %idxdim, [idx_dim];
    ld.param.u32 %inn, [inner];

    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %gtid, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %gtid;
    setp.ge.u32 %p, %gtid, %tot; @%p bra DONE;

    mul.lo.u32 %slab, %idxdim, %inn;
    div.u32 %o, %gtid, %slab;
    rem.u32 %rem, %gtid, %slab;
    rem.u32 %k, %rem, %inn;

    cvt.u64.u32 %off, %gtid; shl.b64 %off, %off, 3; add.u64 %addr, %idx, %off;
    ld.global.s64 %selv, [%addr];
    cvt.u32.s64 %sel, %selv;

    cvt.u64.u32 %off, %gtid; shl.b64 %off, %off, ",
            $wsh,
            "; add.u64 %addr, %src, %off;
    ",
            $ldv,
            " %v, [%addr];

    mul.lo.u32 %dstelem, %o, %outdim;
    add.u32 %dstelem, %dstelem, %sel;
    mul.lo.u32 %dstelem, %dstelem, %inn;
    add.u32 %dstelem, %dstelem, %k;

    cvt.u64.u32 %off, %dstelem; shl.b64 %off, %off, ",
            $wsh,
            "; add.u64 %addr, %out, %off;
    ",
            $stv,
            " [%addr], %v;
DONE:
    ret;
}
"
        )
    };
}

// ===========================================================================
// scatter_value PTX. Params: (out_ptr, idx_ptr, value, outer, out_dim,
//                             idx_dim, inner, total)
//   value is the broadcast scalar (passed as the value-typed param).
//   Same addressing as scatter, but every cell takes `value`.
// ===========================================================================
macro_rules! scatter_value_dim_ptx {
    ($kname:literal, $wsh:literal, $stv:literal, $vreg:literal, $valparam:literal,
     $ldval:literal) => {
        concat!(
            ".version 7.0\n.target sm_60\n.address_size 64\n",
            ".visible .entry ",
            $kname,
            "(
    .param .u64 out_ptr, .param .u64 idx_ptr, .param ",
            $valparam,
            " value,
    .param .u32 outer, .param .u32 out_dim, .param .u32 idx_dim,
    .param .u32 inner, .param .u32 total
) {
    .reg .u32 %gtid, %bid, %bdim, %tot, %outdim, %idxdim, %inn;
    .reg .u32 %o, %rem, %k, %slab, %sel, %dstelem;
    .reg .u64 %out, %idx, %off, %addr;
    .reg .s64 %selv;
    .reg ",
            $vreg,
            " %v;
    .reg .pred %p;

    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %idx, [idx_ptr];
    ",
            $ldval,
            " %v, [value];
    ld.param.u32 %tot, [total];
    ld.param.u32 %outdim, [out_dim];
    ld.param.u32 %idxdim, [idx_dim];
    ld.param.u32 %inn, [inner];

    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %gtid, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %gtid;
    setp.ge.u32 %p, %gtid, %tot; @%p bra DONE;

    mul.lo.u32 %slab, %idxdim, %inn;
    div.u32 %o, %gtid, %slab;
    rem.u32 %rem, %gtid, %slab;
    rem.u32 %k, %rem, %inn;

    cvt.u64.u32 %off, %gtid; shl.b64 %off, %off, 3; add.u64 %addr, %idx, %off;
    ld.global.s64 %selv, [%addr];
    cvt.u32.s64 %sel, %selv;

    mul.lo.u32 %dstelem, %o, %outdim;
    add.u32 %dstelem, %dstelem, %sel;
    mul.lo.u32 %dstelem, %dstelem, %inn;
    add.u32 %dstelem, %dstelem, %k;

    cvt.u64.u32 %off, %dstelem; shl.b64 %off, %off, ",
            $wsh,
            "; add.u64 %addr, %out, %off;
    ",
            $stv,
            " [%addr], %v;
DONE:
    ret;
}
"
        )
    };
}

// ===========================================================================
// scatter_add PTX. Same addressing as scatter, but ATOMIC add into out[dst].
//   $atom = "atom.global.add.f32" / "atom.global.add.f64"
// ===========================================================================
macro_rules! scatter_add_dim_ptx {
    ($kname:literal, $wsh:literal, $ldv:literal, $atom:literal, $vreg:literal) => {
        concat!(
            ".version 7.0\n.target sm_60\n.address_size 64\n",
            ".visible .entry ",
            $kname,
            "(
    .param .u64 out_ptr, .param .u64 idx_ptr, .param .u64 src_ptr,
    .param .u32 outer, .param .u32 out_dim, .param .u32 idx_dim,
    .param .u32 inner, .param .u32 total
) {
    .reg .u32 %gtid, %bid, %bdim, %tot, %outdim, %idxdim, %inn;
    .reg .u32 %o, %rem, %k, %slab, %sel, %dstelem;
    .reg .u64 %out, %idx, %src, %off, %addr;
    .reg .s64 %selv;
    .reg ",
            $vreg,
            " %v, %dummy;
    .reg .pred %p;

    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %idx, [idx_ptr];
    ld.param.u64 %src, [src_ptr];
    ld.param.u32 %tot, [total];
    ld.param.u32 %outdim, [out_dim];
    ld.param.u32 %idxdim, [idx_dim];
    ld.param.u32 %inn, [inner];

    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %gtid, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %gtid;
    setp.ge.u32 %p, %gtid, %tot; @%p bra DONE;

    mul.lo.u32 %slab, %idxdim, %inn;
    div.u32 %o, %gtid, %slab;
    rem.u32 %rem, %gtid, %slab;
    rem.u32 %k, %rem, %inn;

    cvt.u64.u32 %off, %gtid; shl.b64 %off, %off, 3; add.u64 %addr, %idx, %off;
    ld.global.s64 %selv, [%addr];
    cvt.u32.s64 %sel, %selv;

    cvt.u64.u32 %off, %gtid; shl.b64 %off, %off, ",
            $wsh,
            "; add.u64 %addr, %src, %off;
    ",
            $ldv,
            " %v, [%addr];

    mul.lo.u32 %dstelem, %o, %outdim;
    add.u32 %dstelem, %dstelem, %sel;
    mul.lo.u32 %dstelem, %dstelem, %inn;
    add.u32 %dstelem, %dstelem, %k;

    cvt.u64.u32 %off, %dstelem; shl.b64 %off, %off, ",
            $wsh,
            "; add.u64 %addr, %out, %off;
    ",
            $atom,
            " %dummy, [%addr], %v;
DONE:
    ret;
}
"
        )
    };
}

// ── PTX constants ───────────────────────────────────────────────────────────
const GATHER_DIM_F32_PTX: &str = gather_dim_ptx!(
    "gather_dim_f32_kernel",
    "2",
    "ld.global.f32",
    "st.global.f32",
    ".f32"
);
const GATHER_DIM_F64_PTX: &str = gather_dim_ptx!(
    "gather_dim_f64_kernel",
    "3",
    "ld.global.f64",
    "st.global.f64",
    ".f64"
);
const SCATTER_DIM_F32_PTX: &str = scatter_dim_ptx!(
    "scatter_dim_f32_kernel",
    "2",
    "ld.global.f32",
    "st.global.f32",
    ".f32"
);
const SCATTER_DIM_F64_PTX: &str = scatter_dim_ptx!(
    "scatter_dim_f64_kernel",
    "3",
    "ld.global.f64",
    "st.global.f64",
    ".f64"
);
const SCATTER_VALUE_DIM_F32_PTX: &str = scatter_value_dim_ptx!(
    "scatter_value_dim_f32_kernel",
    "2",
    "st.global.f32",
    ".f32",
    ".f32",
    "ld.param.f32"
);
const SCATTER_VALUE_DIM_F64_PTX: &str = scatter_value_dim_ptx!(
    "scatter_value_dim_f64_kernel",
    "3",
    "st.global.f64",
    ".f64",
    ".f64",
    "ld.param.f64"
);
const SCATTER_ADD_DIM_F32_PTX: &str = scatter_add_dim_ptx!(
    "scatter_add_dim_f32_kernel",
    "2",
    "ld.global.f32",
    "atom.global.add.f32",
    ".f32"
);
const SCATTER_ADD_DIM_F64_PTX: &str = scatter_add_dim_ptx!(
    "scatter_add_dim_f64_kernel",
    "3",
    "ld.global.f64",
    "atom.global.add.f64",
    ".f64"
);

// ===========================================================================
// scatter_add_segments PTX. The segmented row-scatter-add used by GNN message
// passing (`ferrotorch-core/src/ops/scatter.rs::scatter_add_segments`).
//
// Params: (out_ptr, idx_ptr, src_ptr, e, d, total)
//   out : V[dim_size * d]   (PRE-ZEROED by the launcher; dim_size implicit)
//   idx : i64[e]            (one segment/output-row id per src row)
//   src : V[e * d]          (E rows, D features, C-contiguous)
// Thread t in [0, total = e*d):
//   row = t / d; col = t % d
//   seg = idx[row]
//   dst = seg*d + col
//   out[dst] += src[t]      (ATOMIC — duplicate seg ids accumulate)
//
// Distinct from scatter_add_dim_ptx: the index is per-ROW (length E), not a
// full-rank per-element index, and addressing is the flat `seg*d + col` row
// scatter rather than the `[outer, axis, inner]` decomposition. The atomic add
// is the same `atom.global.add.f{32,64}` (`sm_60+`) — duplicate segment ids
// into the same output row are the whole reason for the atomic.
// ($wsh = value byte-width shift "2"=f32 / "3"=f64; $ldv/$atom/$vreg per width)
// ===========================================================================
macro_rules! scatter_add_segments_ptx {
    ($kname:literal, $wsh:literal, $ldv:literal, $atom:literal, $vreg:literal) => {
        concat!(
            ".version 7.0\n.target sm_60\n.address_size 64\n",
            ".visible .entry ",
            $kname,
            "(
    .param .u64 out_ptr, .param .u64 idx_ptr, .param .u64 src_ptr,
    .param .u32 e, .param .u32 d, .param .u32 total
) {
    .reg .u32 %gtid, %bid, %bdim, %tot, %dd, %row, %col, %seg, %dstelem;
    .reg .u64 %out, %idx, %src, %off, %addr;
    .reg .s64 %segv;
    .reg ",
            $vreg,
            " %v, %dummy;
    .reg .pred %p;

    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %idx, [idx_ptr];
    ld.param.u64 %src, [src_ptr];
    ld.param.u32 %tot, [total];
    ld.param.u32 %dd, [d];

    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %gtid, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %gtid;
    setp.ge.u32 %p, %gtid, %tot; @%p bra DONE;

    div.u32 %row, %gtid, %dd;
    rem.u32 %col, %gtid, %dd;

    cvt.u64.u32 %off, %row; shl.b64 %off, %off, 3; add.u64 %addr, %idx, %off;
    ld.global.s64 %segv, [%addr];
    cvt.u32.s64 %seg, %segv;

    cvt.u64.u32 %off, %gtid; shl.b64 %off, %off, ",
            $wsh,
            "; add.u64 %addr, %src, %off;
    ",
            $ldv,
            " %v, [%addr];

    mul.lo.u32 %dstelem, %seg, %dd;
    add.u32 %dstelem, %dstelem, %col;

    cvt.u64.u32 %off, %dstelem; shl.b64 %off, %off, ",
            $wsh,
            "; add.u64 %addr, %out, %off;
    ",
            $atom,
            " %dummy, [%addr], %v;
DONE:
    ret;
}
"
        )
    };
}

const SCATTER_ADD_SEGMENTS_F32_PTX: &str = scatter_add_segments_ptx!(
    "scatter_add_segments_f32_kernel",
    "2",
    "ld.global.f32",
    "atom.global.add.f32",
    ".f32"
);
const SCATTER_ADD_SEGMENTS_F64_PTX: &str = scatter_add_segments_ptx!(
    "scatter_add_segments_f64_kernel",
    "3",
    "ld.global.f64",
    "atom.global.add.f64",
    ".f64"
);

// ===========================================================================
// gather launchers
// ===========================================================================

/// Compile + launch the gather kernel for value type `V`. `input` holds at
/// least `outer*in_dim*inner` elements; `idx` holds `outer*out_dim*inner`
/// `i64` indices parallel to the output. Returns a fresh resident buffer of
/// `outer*out_dim*inner` `V`-elements.
#[allow(clippy::too_many_arguments)]
fn launch_gather<V: DeviceRepr + ValidAsZeroBits>(
    input: &CudaSlice<V>,
    idx: &CudaSlice<i64>,
    out: &mut CudaSlice<V>,
    outer: usize,
    in_dim: usize,
    out_dim: usize,
    inner: usize,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
) -> GpuResult<()> {
    let total = outer
        .checked_mul(out_dim)
        .and_then(|x| x.checked_mul(inner))
        .ok_or(GpuError::LengthMismatch {
            a: outer,
            b: out_dim,
        })?;
    if total == 0 {
        return Ok(());
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
    let (outer_u, indim_u, outdim_u, inner_u, total_u) = (
        outer as u32,
        in_dim as u32,
        out_dim as u32,
        inner as u32,
        total as u32,
    );
    // SAFETY:
    // - `f` is the PTX entry `kernel_name`; its 8-arg signature
    //   (in, idx, out, outer, in_dim, out_dim, inner, total) matches the args
    //   pushed below in order.
    // - `input` (V) and `idx` (i64) are immutable inputs; `out` is the fresh
    //   `total`-element V buffer, the only `&mut`, non-aliased.
    // - Each thread writes one `out[t]` for `t in [0,total)` (bound-checked by
    //   `setp.ge.u32 %p, %gtid, %tot; @%p bra DONE`). The source element is
    //   computed from `idx[t]`; an out-of-range index is documented UB matching
    //   PyTorch CUDA (module note), and the core CPU validator rejects OOB
    //   indices before this launch in practice.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(idx)
            .arg(out)
            .arg(&outer_u)
            .arg(&indim_u)
            .arg(&outdim_u)
            .arg(&inner_u)
            .arg(&total_u)
            .launch(cfg)?;
    }
    Ok(())
}

/// f32 dim-aware `gather`. `input` is `[outer, in_dim, inner]`; `idx` is the
/// resident `i64` index `[outer, out_dim, inner]` parallel to the output.
/// Returns a fresh `[outer, out_dim, inner]` buffer.
#[allow(clippy::too_many_arguments)]
pub fn gpu_gather_dim_f32(
    input: &CudaBuffer<f32>,
    idx: &CudaSlice<i64>,
    outer: usize,
    in_dim: usize,
    out_dim: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    let mut out = alloc_zeros_f32(outer * out_dim * inner, device)?;
    launch_gather(
        input.inner(),
        idx,
        out.inner_mut(),
        outer,
        in_dim,
        out_dim,
        inner,
        device,
        GATHER_DIM_F32_PTX,
        "gather_dim_f32_kernel",
    )?;
    Ok(out)
}

/// f64 dim-aware `gather`. Companion of [`gpu_gather_dim_f32`].
#[allow(clippy::too_many_arguments)]
pub fn gpu_gather_dim_f64(
    input: &CudaBuffer<f64>,
    idx: &CudaSlice<i64>,
    outer: usize,
    in_dim: usize,
    out_dim: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    let mut out = alloc_zeros_f64(outer * out_dim * inner, device)?;
    launch_gather(
        input.inner(),
        idx,
        out.inner_mut(),
        outer,
        in_dim,
        out_dim,
        inner,
        device,
        GATHER_DIM_F64_PTX,
        "gather_dim_f64_kernel",
    )?;
    Ok(out)
}

// ===========================================================================
// scatter / scatter_add launchers (out is pre-cloned from input)
// ===========================================================================

/// Compile + launch a scatter-family kernel (`scatter` or `scatter_add`) that
/// mutates `out` in place. `out` must already be a clone of `input`
/// (`[outer, out_dim, inner]`). `idx`/`src` are `[outer, idx_dim, inner]`.
#[allow(clippy::too_many_arguments)]
fn launch_scatter<V: DeviceRepr + ValidAsZeroBits>(
    out: &mut CudaSlice<V>,
    idx: &CudaSlice<i64>,
    src: &CudaSlice<V>,
    outer: usize,
    out_dim: usize,
    idx_dim: usize,
    inner: usize,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
) -> GpuResult<()> {
    let total = outer
        .checked_mul(idx_dim)
        .and_then(|x| x.checked_mul(inner))
        .ok_or(GpuError::LengthMismatch {
            a: outer,
            b: idx_dim,
        })?;
    if total == 0 {
        return Ok(());
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
    let (outer_u, outdim_u, idxdim_u, inner_u, total_u) = (
        outer as u32,
        out_dim as u32,
        idx_dim as u32,
        inner as u32,
        total as u32,
    );
    // SAFETY:
    // - `f` is the PTX entry `kernel_name`; its 8-arg signature
    //   (out, idx, src, outer, out_dim, idx_dim, inner, total) matches the
    //   pushed args in order.
    // - `out` is the caller's pre-cloned `outer*out_dim*inner` buffer (the only
    //   `&mut`); `idx` (i64) and `src` (V) are immutable inputs, distinct
    //   allocations from `out`.
    // - Each thread `t in [0,total)` (bound-checked) reads `idx[t]`/`src[t]` and
    //   writes/atomically-adds one `out[dst]` where `dst` is computed from
    //   `idx[t]`. The core CPU validator rejects OOB index values before this
    //   launch, so `dst < outer*out_dim*inner`. For `scatter_add` the write is
    //   `atom.global.add`, so concurrent threads targeting the same `dst`
    //   accumulate without a data race.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(out)
            .arg(idx)
            .arg(src)
            .arg(&outer_u)
            .arg(&outdim_u)
            .arg(&idxdim_u)
            .arg(&inner_u)
            .arg(&total_u)
            .launch(cfg)?;
    }
    Ok(())
}

/// f32 dim-aware `scatter`. `input` is `[outer, out_dim, inner]`; the result
/// is a clone of `input` with `out[..idx[t].., ..] = src[t]` applied.
#[allow(clippy::too_many_arguments)]
pub fn gpu_scatter_dim_f32(
    input: &CudaBuffer<f32>,
    idx: &CudaSlice<i64>,
    src: &CudaBuffer<f32>,
    outer: usize,
    out_dim: usize,
    idx_dim: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    let mut out = clone_f32(input, outer * out_dim * inner, device)?;
    launch_scatter(
        out.inner_mut(),
        idx,
        src.inner(),
        outer,
        out_dim,
        idx_dim,
        inner,
        device,
        SCATTER_DIM_F32_PTX,
        "scatter_dim_f32_kernel",
    )?;
    Ok(out)
}

/// f64 dim-aware `scatter`. Companion of [`gpu_scatter_dim_f32`].
#[allow(clippy::too_many_arguments)]
pub fn gpu_scatter_dim_f64(
    input: &CudaBuffer<f64>,
    idx: &CudaSlice<i64>,
    src: &CudaBuffer<f64>,
    outer: usize,
    out_dim: usize,
    idx_dim: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    let mut out = clone_f64(input, outer * out_dim * inner, device)?;
    launch_scatter(
        out.inner_mut(),
        idx,
        src.inner(),
        outer,
        out_dim,
        idx_dim,
        inner,
        device,
        SCATTER_DIM_F64_PTX,
        "scatter_dim_f64_kernel",
    )?;
    Ok(out)
}

/// f32 dim-aware `scatter_add`. Like [`gpu_scatter_dim_f32`] but accumulates
/// (`out[dst] += src[t]`) via `atom.global.add.f32`, so duplicate index values
/// targeting the same `dst` sum correctly.
#[allow(clippy::too_many_arguments)]
pub fn gpu_scatter_add_dim_f32(
    input: &CudaBuffer<f32>,
    idx: &CudaSlice<i64>,
    src: &CudaBuffer<f32>,
    outer: usize,
    out_dim: usize,
    idx_dim: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    let mut out = clone_f32(input, outer * out_dim * inner, device)?;
    launch_scatter(
        out.inner_mut(),
        idx,
        src.inner(),
        outer,
        out_dim,
        idx_dim,
        inner,
        device,
        SCATTER_ADD_DIM_F32_PTX,
        "scatter_add_dim_f32_kernel",
    )?;
    Ok(out)
}

/// f64 dim-aware `scatter_add`. Companion of [`gpu_scatter_add_dim_f32`]; uses
/// `atom.global.add.f64` (`sm_60+`).
#[allow(clippy::too_many_arguments)]
pub fn gpu_scatter_add_dim_f64(
    input: &CudaBuffer<f64>,
    idx: &CudaSlice<i64>,
    src: &CudaBuffer<f64>,
    outer: usize,
    out_dim: usize,
    idx_dim: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    let mut out = clone_f64(input, outer * out_dim * inner, device)?;
    launch_scatter(
        out.inner_mut(),
        idx,
        src.inner(),
        outer,
        out_dim,
        idx_dim,
        inner,
        device,
        SCATTER_ADD_DIM_F64_PTX,
        "scatter_add_dim_f64_kernel",
    )?;
    Ok(out)
}

// ===========================================================================
// scatter_add_segments launchers (out is zero-initialised; segmented row add)
// ===========================================================================

/// Compile + launch the segmented row-scatter-add kernel. `out` is the
/// pre-zeroed `dim_size*d` output buffer (the only `&mut`); `idx` is the
/// resident `i64` segment id per src row (`e` ids); `src` is `[e, d]`. Thread
/// `t in [0, e*d)` atomically adds `src[t]` into `out[idx[t/d]*d + t%d]`.
fn launch_scatter_add_segments<V: DeviceRepr + ValidAsZeroBits>(
    out: &mut CudaSlice<V>,
    idx: &CudaSlice<i64>,
    src: &CudaSlice<V>,
    e: usize,
    d: usize,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
) -> GpuResult<()> {
    let total = e
        .checked_mul(d)
        .ok_or(GpuError::LengthMismatch { a: e, b: d })?;
    if total == 0 {
        return Ok(());
    }
    let stream = device.stream();
    let ctx = device.context();
    let f = get_or_compile(ctx, ptx, kernel_name, device.ordinal() as u32).map_err(|err| {
        GpuError::PtxCompileFailed {
            kernel: kernel_name,
            source: err,
        }
    })?;
    let cfg = launch_1d(total);
    let (e_u, d_u, total_u) = (e as u32, d as u32, total as u32);
    // SAFETY:
    // - `f` is the PTX entry `kernel_name`; its 6-arg signature
    //   (out, idx, src, e, d, total) matches the args pushed below in order.
    // - `out` is the caller's pre-zeroed `dim_size*d` buffer (the only `&mut`,
    //   non-aliased); `idx` (i64, length `e`) and `src` (V, length `e*d`) are
    //   immutable inputs in distinct allocations.
    // - Each thread `t in [0,total=e*d)` is bound-checked
    //   (`setp.ge.u32 %p, %gtid, %tot; @%p bra DONE`). It reads `idx[t/d]` and
    //   atomically adds `src[t]` into `out[seg*d + t%d]`. The core CPU validator
    //   (`scatter_add_segments`) rejects negative / `>= dim_size` segment ids
    //   before this launch, so `seg*d + col < dim_size*d` always. Concurrent
    //   threads whose segment id collides accumulate via `atom.global.add`
    //   without a data race.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(out)
            .arg(idx)
            .arg(src)
            .arg(&e_u)
            .arg(&d_u)
            .arg(&total_u)
            .launch(cfg)?;
    }
    Ok(())
}

/// f32 segmented row-scatter-add. `src` is `[e, d]`; `idx` is the resident
/// `i64` segment id per src row (length `e`). Returns a fresh zero-initialised
/// `[dim_size, d]` buffer with `out[idx[row], :] += src[row, :]` accumulated
/// over all rows (atomic — duplicate segment ids sum). Output rows with no
/// contributing row stay 0.
pub fn gpu_scatter_add_segments_f32(
    src: &CudaBuffer<f32>,
    idx: &CudaSlice<i64>,
    e: usize,
    d: usize,
    dim_size: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    let mut out = alloc_zeros_f32(dim_size * d, device)?;
    launch_scatter_add_segments(
        out.inner_mut(),
        idx,
        src.inner(),
        e,
        d,
        device,
        SCATTER_ADD_SEGMENTS_F32_PTX,
        "scatter_add_segments_f32_kernel",
    )?;
    Ok(out)
}

/// f64 segmented row-scatter-add. Companion of
/// [`gpu_scatter_add_segments_f32`]; uses `atom.global.add.f64` (`sm_60+`).
pub fn gpu_scatter_add_segments_f64(
    src: &CudaBuffer<f64>,
    idx: &CudaSlice<i64>,
    e: usize,
    d: usize,
    dim_size: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    let mut out = alloc_zeros_f64(dim_size * d, device)?;
    launch_scatter_add_segments(
        out.inner_mut(),
        idx,
        src.inner(),
        e,
        d,
        device,
        SCATTER_ADD_SEGMENTS_F64_PTX,
        "scatter_add_segments_f64_kernel",
    )?;
    Ok(out)
}

// ===========================================================================
// scatter_value launchers (out is pre-cloned from input; scalar broadcast)
// ===========================================================================

/// f32 dim-aware `scatter_value`. Clones `input` (`[outer, out_dim, inner]`)
/// and writes the broadcast scalar `value` at every position named by `idx`.
#[allow(clippy::too_many_arguments)]
pub fn gpu_scatter_value_dim_f32(
    input: &CudaBuffer<f32>,
    idx: &CudaSlice<i64>,
    value: f32,
    outer: usize,
    out_dim: usize,
    idx_dim: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    let mut out = clone_f32(input, outer * out_dim * inner, device)?;
    launch_scatter_value_f32(
        out.inner_mut(),
        idx,
        value,
        outer,
        out_dim,
        idx_dim,
        inner,
        device,
    )?;
    Ok(out)
}

/// f64 dim-aware `scatter_value`. Companion of [`gpu_scatter_value_dim_f32`].
#[allow(clippy::too_many_arguments)]
pub fn gpu_scatter_value_dim_f64(
    input: &CudaBuffer<f64>,
    idx: &CudaSlice<i64>,
    value: f64,
    outer: usize,
    out_dim: usize,
    idx_dim: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    let mut out = clone_f64(input, outer * out_dim * inner, device)?;
    launch_scatter_value_f64(
        out.inner_mut(),
        idx,
        value,
        outer,
        out_dim,
        idx_dim,
        inner,
        device,
    )?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn launch_scatter_value_f32(
    out: &mut CudaSlice<f32>,
    idx: &CudaSlice<i64>,
    value: f32,
    outer: usize,
    out_dim: usize,
    idx_dim: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<()> {
    let total = outer
        .checked_mul(idx_dim)
        .and_then(|x| x.checked_mul(inner))
        .ok_or(GpuError::LengthMismatch {
            a: outer,
            b: idx_dim,
        })?;
    if total == 0 {
        return Ok(());
    }
    let stream = device.stream();
    let ctx = device.context();
    let f = get_or_compile(
        ctx,
        SCATTER_VALUE_DIM_F32_PTX,
        "scatter_value_dim_f32_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "scatter_value_dim_f32_kernel",
        source: e,
    })?;
    let cfg = launch_1d(total);
    let (outer_u, outdim_u, idxdim_u, inner_u, total_u) = (
        outer as u32,
        out_dim as u32,
        idx_dim as u32,
        inner as u32,
        total as u32,
    );
    // SAFETY: see `launch_scatter` — same addressing and bounds. The 8-arg
    // signature here is (out, idx, value, outer, out_dim, idx_dim, inner,
    // total); `out` is the only `&mut`, pre-cloned and non-aliased; `value` is
    // a by-value f32 scalar param. Each thread writes one in-bounds `out[dst]`.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(out)
            .arg(idx)
            .arg(&value)
            .arg(&outer_u)
            .arg(&outdim_u)
            .arg(&idxdim_u)
            .arg(&inner_u)
            .arg(&total_u)
            .launch(cfg)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_scatter_value_f64(
    out: &mut CudaSlice<f64>,
    idx: &CudaSlice<i64>,
    value: f64,
    outer: usize,
    out_dim: usize,
    idx_dim: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<()> {
    let total = outer
        .checked_mul(idx_dim)
        .and_then(|x| x.checked_mul(inner))
        .ok_or(GpuError::LengthMismatch {
            a: outer,
            b: idx_dim,
        })?;
    if total == 0 {
        return Ok(());
    }
    let stream = device.stream();
    let ctx = device.context();
    let f = get_or_compile(
        ctx,
        SCATTER_VALUE_DIM_F64_PTX,
        "scatter_value_dim_f64_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "scatter_value_dim_f64_kernel",
        source: e,
    })?;
    let cfg = launch_1d(total);
    let (outer_u, outdim_u, idxdim_u, inner_u, total_u) = (
        outer as u32,
        out_dim as u32,
        idx_dim as u32,
        inner as u32,
        total as u32,
    );
    // SAFETY: see `launch_scatter_value_f32` — identical contract, f64 value.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(out)
            .arg(idx)
            .arg(&value)
            .arg(&outer_u)
            .arg(&outdim_u)
            .arg(&idxdim_u)
            .arg(&inner_u)
            .arg(&total_u)
            .launch(cfg)?;
    }
    Ok(())
}

// ===========================================================================
// helpers: device-to-device clone into a fresh `len`-element buffer
// ===========================================================================

/// Clone the first `len` elements of `input` into a fresh f32 buffer.
fn clone_f32(
    input: &CudaBuffer<f32>,
    len: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    let mut out = alloc_zeros_f32(len, device)?;
    if len > 0 {
        let stream = device.stream();
        stream.memcpy_dtod(&input.inner().slice(0..len), out.inner_mut())?;
    }
    Ok(out)
}

/// Clone the first `len` elements of `input` into a fresh f64 buffer.
fn clone_f64(
    input: &CudaBuffer<f64>,
    len: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    let mut out = alloc_zeros_f64(len, device)?;
    if len > 0 {
        let stream = device.stream();
        stream.memcpy_dtod(&input.inner().slice(0..len), out.inner_mut())?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transfer::{cpu_to_gpu, gpu_to_cpu};

    fn dev() -> GpuDevice {
        GpuDevice::new(0).expect("cuda device")
    }

    fn htod_i64(d: &GpuDevice, v: &[i64]) -> CudaSlice<i64> {
        d.stream().clone_htod(&v.to_vec()).expect("htod i64")
    }

    // gather: input [2,3] dim=1, index [2,2] -> outer=2 in_dim=3 out_dim=2 inner=1.
    // output[i][j] = input[i][index[i][j]]; idx=[[0,2],[1,0]] -> [[1,3],[5,4]].
    #[test]
    fn gather_dim_f32_dim1() {
        let d = dev();
        let inp = cpu_to_gpu(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &d).unwrap();
        let idx = htod_i64(&d, &[0i64, 2, 1, 0]);
        let out = gpu_gather_dim_f32(&inp, &idx, 2, 3, 2, 1, &d).unwrap();
        assert_eq!(gpu_to_cpu(&out, &d).unwrap()[..4], [1.0f32, 3.0, 5.0, 4.0]);
    }

    // scatter: input zeros [2,3] dim=1, src [[5],[6]], index [[2],[0]]
    // -> outer=2 out_dim=3 idx_dim=1 inner=1 -> [[0,0,5],[6,0,0]].
    #[test]
    fn scatter_dim_f32_dim1() {
        let d = dev();
        let inp = cpu_to_gpu(&[0.0f32; 6], &d).unwrap();
        let src = cpu_to_gpu(&[5.0f32, 6.0], &d).unwrap();
        let idx = htod_i64(&d, &[2i64, 0]);
        let out = gpu_scatter_dim_f32(&inp, &idx, &src, 2, 3, 1, 1, &d).unwrap();
        assert_eq!(
            gpu_to_cpu(&out, &d).unwrap()[..6],
            [0.0f32, 0.0, 5.0, 6.0, 0.0, 0.0]
        );
    }

    // scatter_value: input zeros [5] dim=0, value=9, index=[1,3,0]
    // -> [9,9,0,9,0].
    #[test]
    fn scatter_value_dim_f32_1d() {
        let d = dev();
        let inp = cpu_to_gpu(&[0.0f32; 5], &d).unwrap();
        let idx = htod_i64(&d, &[1i64, 3, 0]);
        let out = gpu_scatter_value_dim_f32(&inp, &idx, 9.0, 1, 5, 3, 1, &d).unwrap();
        assert_eq!(
            gpu_to_cpu(&out, &d).unwrap()[..5],
            [9.0f32, 9.0, 0.0, 9.0, 0.0]
        );
    }

    // scatter_add with DUPLICATE indices (atomic accumulation is the key case):
    // input=[1,2,3] dim=0, src=[10,20,30] at index=[0,2,0]
    // -> [1+10+30, 2, 3+20] = [41,2,23].
    #[test]
    fn scatter_add_dim_f32_duplicate_indices() {
        let d = dev();
        let inp = cpu_to_gpu(&[1.0f32, 2.0, 3.0], &d).unwrap();
        let src = cpu_to_gpu(&[10.0f32, 20.0, 30.0], &d).unwrap();
        let idx = htod_i64(&d, &[0i64, 2, 0]);
        let out = gpu_scatter_add_dim_f32(&inp, &idx, &src, 1, 3, 3, 1, &d).unwrap();
        assert_eq!(gpu_to_cpu(&out, &d).unwrap()[..3], [41.0f32, 2.0, 23.0]);
    }

    #[test]
    fn scatter_add_dim_f64_duplicate_indices() {
        let d = dev();
        let inp = cpu_to_gpu(&[1.0f64, 2.0, 3.0], &d).unwrap();
        let src = cpu_to_gpu(&[10.0f64, 20.0, 30.0], &d).unwrap();
        let idx = htod_i64(&d, &[0i64, 2, 0]);
        let out = gpu_scatter_add_dim_f64(&inp, &idx, &src, 1, 3, 3, 1, &d).unwrap();
        assert_eq!(gpu_to_cpu(&out, &d).unwrap()[..3], [41.0f64, 2.0, 23.0]);
    }

    #[test]
    fn gather_dim_f64_dim0() {
        let d = dev();
        // input [3,2] dim=0, index [[2,0],[1,1]] -> [[5,2],[3,4]].
        let inp = cpu_to_gpu(&[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], &d).unwrap();
        let idx = htod_i64(&d, &[2i64, 0, 1, 1]);
        let out = gpu_gather_dim_f64(&inp, &idx, 1, 3, 2, 2, &d).unwrap();
        assert_eq!(gpu_to_cpu(&out, &d).unwrap()[..4], [5.0f64, 2.0, 3.0, 4.0]);
    }

    // scatter_add_segments doc example: src=[[1,2],[3,4],[5,6]], index=[0,1,0],
    // dim_size=2 -> out[0]=src[0]+src[2]=[6,8], out[1]=src[1]=[3,4].
    #[test]
    fn scatter_add_segments_f32_basic() {
        let d = dev();
        let src = cpu_to_gpu(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &d).unwrap();
        let idx = htod_i64(&d, &[0i64, 1, 0]);
        let out = gpu_scatter_add_segments_f32(&src, &idx, 3, 2, 2, &d).unwrap();
        assert_eq!(gpu_to_cpu(&out, &d).unwrap()[..4], [6.0f32, 8.0, 3.0, 4.0]);
    }

    // Duplicate-segment atomic case: 4 rows of D=2, all into segment 0,
    // dim_size=2 -> out[0] = column sums, out[1] stays exactly 0.
    #[test]
    fn scatter_add_segments_f32_duplicate_and_empty_row() {
        let d = dev();
        let src = cpu_to_gpu(&[1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0], &d).unwrap();
        let idx = htod_i64(&d, &[0i64, 0, 0, 0]);
        let out = gpu_scatter_add_segments_f32(&src, &idx, 4, 2, 2, &d).unwrap();
        let got = gpu_to_cpu(&out, &d).unwrap();
        assert_eq!(got[..4], [10.0f32, 100.0, 0.0, 0.0]);
    }

    #[test]
    fn scatter_add_segments_f64_basic() {
        let d = dev();
        let src = cpu_to_gpu(&[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], &d).unwrap();
        let idx = htod_i64(&d, &[0i64, 1, 0]);
        let out = gpu_scatter_add_segments_f64(&src, &idx, 3, 2, 2, &d).unwrap();
        assert_eq!(gpu_to_cpu(&out, &d).unwrap()[..4], [6.0f64, 8.0, 3.0, 4.0]);
    }
}
