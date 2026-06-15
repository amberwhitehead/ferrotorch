//! `index_select` / `gather` GPU kernels driven by a GPU-resident **integer**
//! index buffer (`i32` / `i64`) — crosslink #1185 Phase 2c.
//!
//! The pre-existing `index_select_1d` / `index_select_dim` kernels in
//! [`crate::kernels`] take an **f32-encoded** index slice (the legacy
//! `&[u32]`-uploaded-as-f32 contract). Phase 2c's new capability is supplying
//! the indices as a real GPU-resident `IntTensor<i32|i64>` (the Llama token-id
//! path), so these kernels read the index with `ld.global.s32` / `ld.global.s64`
//! directly — no f32 detour, no host round-trip.
//!
//! # Element movement is dtype-generic (raw bytes)
//!
//! `index_select` / `gather` only *copy* elements; they never do arithmetic on
//! the value. So the value kernels are parameterised by **byte width** (2 / 4 /
//! 8), exactly like PyTorch's `gather` which dispatches on `itemsize` for the
//! plain copy. A 4-byte copy kernel serves both f32 and i32; an 8-byte copy
//! serves both f64 and i64; a 2-byte copy serves f16 and bf16. Combined with
//! the two index widths and the two ops, that is 3×2×2 = 12 hand-written PTX
//! entries.
//!
//! # Layouts (C-order, contiguous)
//!
//! `index_select(dim)`: input `[outer, in_dim, inner]`, index `[out_dim]`
//! (1-D), output `[outer, out_dim, inner]`. Thread `t in [0, outer*out_dim*
//! inner)` decomposes to `(o, i, k)` and writes
//! `out[t] = input[o*in_dim*inner + index[i]*inner + k]`.
//!
//! `gather(dim)`: the fast path uses input `[outer, in_dim, inner]`, with index
//! AND output both `[outer, out_dim, inner]`. The general `gather_nd_*` entries
//! use C-order `input_shape`/`input_strides` and `index_shape` metadata so
//! `index`/output shape is authoritative even when non-gather dimensions are
//! smaller than the input. Thread `t` decodes the output coordinate from
//! `index_shape`, replaces only `coord[dim]` with `index[t]`, then copies
//! `input[src_flat]` to `out[t]`.
//!
//! # Out-of-range indices
//!
//! PyTorch CUDA checks index bounds on the device: `index_select` uses a
//! device assert in `aten/src/ATen/native/cuda/Indexing.cu`, and `gather`
//! checks `0 <= idx_dim < index_size` in
//! `aten/src/ATen/native/cuda/ScatterGatherKernel.cu`. The value-copy kernels
//! below still compute unchecked addresses for speed, but this module also
//! exposes `check_indices_*_in_bounds` validator kernels. Core calls that
//! resident scan before the unchecked copy kernel and reads back only a tiny
//! status payload, never the full index buffer.
//!
//! ## REQ status (per `.design/ferrotorch-gpu/gather_int.md`)
//!
//! Full evidence rows (impl + non-test production consumer + upstream
//! cites) live in the design doc; this synopsis is a one-line summary per
//! REQ.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (20 isel/gather entries) | SHIPPED | 20 `pub fn isel_* / gather_*` entries in `gather_int.rs` (via `select_entry!` macro invocations); consumer `CudaBackendImpl::gather_or_select in backend_impl.rs` dispatches all 20 cells through the `run!` macro |
//! | REQ-2 (12 PTX macro-expansions) | SHIPPED | `index_select_ptx!` and `gather_ptx!` macros in `gather_int.rs` expand 12 PTX entries (6 select × {W2,W4,W8} × {I32,I64} + 6 gather × ditto), resolved by `isel_ptx / gathr_ptx` in the same file |
//! | REQ-3 (C-order layout contract) | SHIPPED | layout contract documented in `gather_int.rs` module `//!` block and reflected in the PTX address math; verified by unit tests' expected-output construction |
//! | REQ-4 (device-side bounds gate) | SHIPPED | `check_indices_i32_in_bounds` / `check_indices_i64_in_bounds` scan index tensors on CUDA and return a scalar status before unchecked copy kernels launch, matching upstream device-side bound checks in `Indexing.cu` / `ScatterGatherKernel.cu` without full index D2H |
//! | REQ-5 (consumer wiring) | SHIPPED | `CudaBackendImpl::gather_or_select in backend_impl.rs` is the production consumer; ferrotorch-core's `Tensor::index_select / Tensor::gather` dispatch through it via the `GpuBackend::gather_or_select` trait method when source is CUDA-resident |
//! | REQ-6 (resident index_select backward expansion) | SHIPPED | `expand_index_select_i64` expands a saved 1-D CUDA i64 index to the per-output scatter index buffer consumed by `IndexSelectDimBackward`, with unit coverage in `expand_index_select_i64_repeats_index_across_outer_and_inner` |

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits};

use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};
use crate::module_cache::get_or_compile;

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
// Macro-built PTX. Parameters:
//   $kname  : entry name (str)
//   $wsh    : value byte-width shift ("1"=2B, "2"=4B, "3"=8B)
//   $ish    : index byte-width shift ("2"=i32, "3"=i64)
//   $ldi    : index load op ("ld.global.s32" / "ld.global.s64")
//   $icvt   : cvt source suffix for sel widening ("s32" / "s64")
//   $ldv    : value load op ("ld.global.u16" / ".u32" / ".u64")
//   $stv    : value store op (matching width)
//   $vreg   : value register type (".u16" / ".u32" / ".u64")
//   $ireg   : index register type (".s32" / ".s64")
//
// index_select: sel = idx[i]  (i = which output row along the dim)
// ===========================================================================
macro_rules! index_select_ptx {
    ($kname:literal, $wsh:literal, $ish:literal, $ldi:literal, $icvt:literal,
     $ldv:literal, $stv:literal, $vreg:literal, $ireg:literal) => {
        concat!(
            ".version 7.0\n.target sm_52\n.address_size 64\n",
            ".visible .entry ",
            $kname,
            "(
    .param .u64 in_ptr, .param .u64 idx_ptr, .param .u64 out_ptr,
    .param .u32 outer, .param .u32 in_dim, .param .u32 out_dim,
    .param .u32 inner, .param .u32 total
) {
    .reg .u32 %gtid, %bid, %bdim, %tot, %indim, %outdim, %inn;
    .reg .u32 %o, %rem, %i, %k, %slab, %sel, %srcelem;
    .reg .u64 %in, %idx, %out, %off, %addr;
    .reg ",
            $ireg,
            " %selv;
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
    div.u32 %i, %rem, %inn;
    rem.u32 %k, %rem, %inn;

    cvt.u64.u32 %off, %i; shl.b64 %off, %off, ",
            $ish,
            "; add.u64 %addr, %idx, %off;
    ",
            $ldi,
            " %selv, [%addr];
    cvt.u32.",
            $icvt,
            " %sel, %selv;

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

// gather: sel = idx[t]  (index buffer is parallel to output)
macro_rules! gather_ptx {
    ($kname:literal, $wsh:literal, $ish:literal, $ldi:literal, $icvt:literal,
     $ldv:literal, $stv:literal, $vreg:literal, $ireg:literal) => {
        concat!(
            ".version 7.0\n.target sm_52\n.address_size 64\n",
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
    .reg ",
            $ireg,
            " %selv;
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

    cvt.u64.u32 %off, %gtid; shl.b64 %off, %off, ",
            $ish,
            "; add.u64 %addr, %idx, %off;
    ",
            $ldi,
            " %selv, [%addr];
    cvt.u32.",
            $icvt,
            " %sel, %selv;

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

// gather_nd: rank-aware gather for valid PyTorch layouts where index/output
// shape may be smaller than input on non-gather axes.
macro_rules! gather_nd_ptx {
    ($kname:literal, $wsh:literal, $ish:literal, $ldi:literal, $icvt:literal,
     $ldv:literal, $stv:literal, $vreg:literal, $ireg:literal) => {
        concat!(
            ".version 7.0\n.target sm_52\n.address_size 64\n",
            ".visible .entry ",
            $kname,
            "(
    .param .u64 in_ptr, .param .u64 idx_ptr,
    .param .u64 input_strides_ptr, .param .u64 index_shape_ptr,
    .param .u64 out_ptr,
    .param .u32 rank, .param .u32 dim, .param .u32 total
) {
    .reg .u32 %gtid, %bid, %bdim, %tot, %rank, %dim, %axis, %rem;
    .reg .u32 %size, %coord, %sel, %stride, %srcelem;
    .reg .u64 %in, %idx, %istr, %ishape, %out, %off, %addr;
    .reg ",
            $ireg,
            " %selv;
    .reg ",
            $vreg,
            " %v;
    .reg .pred %p;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %idx, [idx_ptr];
    ld.param.u64 %istr, [input_strides_ptr];
    ld.param.u64 %ishape, [index_shape_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %rank, [rank];
    ld.param.u32 %dim, [dim];
    ld.param.u32 %tot, [total];

    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %gtid, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %gtid;
    setp.ge.u32 %p, %gtid, %tot; @%p bra DONE;

    mov.u32 %axis, %rank;
    mov.u32 %rem, %gtid;
    mov.u32 %srcelem, 0;
LOOP:
    setp.eq.u32 %p, %axis, 0; @%p bra LOOP_DONE;
    sub.u32 %axis, %axis, 1;

    cvt.u64.u32 %off, %axis; shl.b64 %off, %off, 2; add.u64 %addr, %ishape, %off;
    ld.global.u32 %size, [%addr];
    rem.u32 %coord, %rem, %size;
    div.u32 %rem, %rem, %size;

    setp.ne.u32 %p, %axis, %dim; @%p bra USE_OUTPUT_COORD;
    cvt.u64.u32 %off, %gtid; shl.b64 %off, %off, ",
            $ish,
            "; add.u64 %addr, %idx, %off;
    ",
            $ldi,
            " %selv, [%addr];
    cvt.u32.",
            $icvt,
            " %sel, %selv;
    mov.u32 %coord, %sel;
USE_OUTPUT_COORD:
    cvt.u64.u32 %off, %axis; shl.b64 %off, %off, 2; add.u64 %addr, %istr, %off;
    ld.global.u32 %stride, [%addr];
    mad.lo.u32 %srcelem, %coord, %stride, %srcelem;
    bra LOOP;

LOOP_DONE:
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

// Device-resident bounds validator for i32/i64 index tensors. One thread
// checks one index value; the first invalid thread atomically claims `flag`
// and stores its flat position plus offending value into `bad`.
macro_rules! check_bounds_ptx {
    ($kname:literal, $ish:literal, $load:literal) => {
        concat!(
            ".version 7.0\n.target sm_52\n.address_size 64\n",
            ".visible .entry ",
            $kname,
            "(
    .param .u64 idx_ptr,
    .param .u64 flag_ptr,
    .param .u64 bad_ptr,
    .param .u32 total,
    .param .s64 dim_size
) {
    .reg .u32 %gtid, %bid, %bdim, %tot, %old;
    .reg .u64 %idx, %flag, %bad, %off, %addr, %bad_value_addr;
    .reg .s32 %val32;
    .reg .s64 %val, %dim, %pos;
    .reg .pred %p, %neg, %too_hi, %invalid, %already_claimed;

    ld.param.u64 %idx, [idx_ptr];
    ld.param.u64 %flag, [flag_ptr];
    ld.param.u64 %bad, [bad_ptr];
    ld.param.u32 %tot, [total];
    ld.param.s64 %dim, [dim_size];

    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %gtid, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %gtid;
    setp.ge.u32 %p, %gtid, %tot; @%p bra DONE;

    cvt.u64.u32 %off, %gtid; shl.b64 %off, %off, ",
            $ish,
            "; add.u64 %addr, %idx, %off;
    ",
            $load,
            "
    setp.lt.s64 %neg, %val, 0;
    setp.ge.s64 %too_hi, %val, %dim;
    or.pred %invalid, %neg, %too_hi;
    @!%invalid bra DONE;

    atom.global.cas.b32 %old, [%flag], 0, 1;
    setp.ne.u32 %already_claimed, %old, 0;
    @%already_claimed bra DONE;

    cvt.s64.u32 %pos, %gtid;
    st.global.s64 [%bad], %pos;
    add.u64 %bad_value_addr, %bad, 8;
    st.global.s64 [%bad_value_addr], %val;
DONE:
    ret;
}
"
        )
    };
}

// Expand a 1-D i64 index_select index into the per-output i64 index buffer
// consumed by dim-aware scatter-add backward kernels.
macro_rules! expand_index_select_i64_ptx {
    () => {
        concat!(
            ".version 7.0\n.target sm_52\n.address_size 64\n",
            ".visible .entry expand_index_select_i64_kernel(
    .param .u64 idx_ptr,
    .param .u64 out_ptr,
    .param .u32 out_dim,
    .param .u32 inner,
    .param .u32 total
) {
    .reg .u32 %gtid, %bid, %bdim, %tot, %outdim, %inn;
    .reg .u32 %rem, %i, %slab;
    .reg .u64 %idx, %out, %off, %addr;
    .reg .s64 %val;
    .reg .pred %p;

    ld.param.u64 %idx, [idx_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %outdim, [out_dim];
    ld.param.u32 %inn, [inner];
    ld.param.u32 %tot, [total];

    mov.u32 %bid, %ctaid.x; mov.u32 %bdim, %ntid.x; mov.u32 %gtid, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %gtid;
    setp.ge.u32 %p, %gtid, %tot; @%p bra DONE;

    mul.lo.u32 %slab, %outdim, %inn;
    rem.u32 %rem, %gtid, %slab;
    div.u32 %i, %rem, %inn;

    cvt.u64.u32 %off, %i; shl.b64 %off, %off, 3; add.u64 %addr, %idx, %off;
    ld.global.s64 %val, [%addr];

    cvt.u64.u32 %off, %gtid; shl.b64 %off, %off, 3; add.u64 %addr, %out, %off;
    st.global.s64 [%addr], %val;
DONE:
    ret;
}
"
        )
    };
}

// ── index_select PTX constants (value width × index width) ──────────────────
const ISEL_W2_I32_PTX: &str = index_select_ptx!(
    "isel_w2_i32_kernel",
    "1",
    "2",
    "ld.global.s32",
    "s32",
    "ld.global.u16",
    "st.global.u16",
    ".u16",
    ".s32"
);
const ISEL_W2_I64_PTX: &str = index_select_ptx!(
    "isel_w2_i64_kernel",
    "1",
    "3",
    "ld.global.s64",
    "s64",
    "ld.global.u16",
    "st.global.u16",
    ".u16",
    ".s64"
);
const ISEL_W4_I32_PTX: &str = index_select_ptx!(
    "isel_w4_i32_kernel",
    "2",
    "2",
    "ld.global.s32",
    "s32",
    "ld.global.u32",
    "st.global.u32",
    ".u32",
    ".s32"
);
const ISEL_W4_I64_PTX: &str = index_select_ptx!(
    "isel_w4_i64_kernel",
    "2",
    "3",
    "ld.global.s64",
    "s64",
    "ld.global.u32",
    "st.global.u32",
    ".u32",
    ".s64"
);
const ISEL_W8_I32_PTX: &str = index_select_ptx!(
    "isel_w8_i32_kernel",
    "3",
    "2",
    "ld.global.s32",
    "s32",
    "ld.global.u64",
    "st.global.u64",
    ".u64",
    ".s32"
);
const ISEL_W8_I64_PTX: &str = index_select_ptx!(
    "isel_w8_i64_kernel",
    "3",
    "3",
    "ld.global.s64",
    "s64",
    "ld.global.u64",
    "st.global.u64",
    ".u64",
    ".s64"
);

// ── gather PTX constants ────────────────────────────────────────────────────
const GATHER_W2_I32_PTX: &str = gather_ptx!(
    "gather_w2_i32_kernel",
    "1",
    "2",
    "ld.global.s32",
    "s32",
    "ld.global.u16",
    "st.global.u16",
    ".u16",
    ".s32"
);
const GATHER_W2_I64_PTX: &str = gather_ptx!(
    "gather_w2_i64_kernel",
    "1",
    "3",
    "ld.global.s64",
    "s64",
    "ld.global.u16",
    "st.global.u16",
    ".u16",
    ".s64"
);
const GATHER_W4_I32_PTX: &str = gather_ptx!(
    "gather_w4_i32_kernel",
    "2",
    "2",
    "ld.global.s32",
    "s32",
    "ld.global.u32",
    "st.global.u32",
    ".u32",
    ".s32"
);
const GATHER_W4_I64_PTX: &str = gather_ptx!(
    "gather_w4_i64_kernel",
    "2",
    "3",
    "ld.global.s64",
    "s64",
    "ld.global.u32",
    "st.global.u32",
    ".u32",
    ".s64"
);
const GATHER_W8_I32_PTX: &str = gather_ptx!(
    "gather_w8_i32_kernel",
    "3",
    "2",
    "ld.global.s32",
    "s32",
    "ld.global.u64",
    "st.global.u64",
    ".u64",
    ".s32"
);
const GATHER_W8_I64_PTX: &str = gather_ptx!(
    "gather_w8_i64_kernel",
    "3",
    "3",
    "ld.global.s64",
    "s64",
    "ld.global.u64",
    "st.global.u64",
    ".u64",
    ".s64"
);

// ── gather_nd PTX constants ─────────────────────────────────────────────────
const GATHER_ND_W2_I32_PTX: &str = gather_nd_ptx!(
    "gather_nd_w2_i32_kernel",
    "1",
    "2",
    "ld.global.s32",
    "s32",
    "ld.global.u16",
    "st.global.u16",
    ".u16",
    ".s32"
);
const GATHER_ND_W2_I64_PTX: &str = gather_nd_ptx!(
    "gather_nd_w2_i64_kernel",
    "1",
    "3",
    "ld.global.s64",
    "s64",
    "ld.global.u16",
    "st.global.u16",
    ".u16",
    ".s64"
);
const GATHER_ND_W4_I32_PTX: &str = gather_nd_ptx!(
    "gather_nd_w4_i32_kernel",
    "2",
    "2",
    "ld.global.s32",
    "s32",
    "ld.global.u32",
    "st.global.u32",
    ".u32",
    ".s32"
);
const GATHER_ND_W4_I64_PTX: &str = gather_nd_ptx!(
    "gather_nd_w4_i64_kernel",
    "2",
    "3",
    "ld.global.s64",
    "s64",
    "ld.global.u32",
    "st.global.u32",
    ".u32",
    ".s64"
);
const GATHER_ND_W8_I32_PTX: &str = gather_nd_ptx!(
    "gather_nd_w8_i32_kernel",
    "3",
    "2",
    "ld.global.s32",
    "s32",
    "ld.global.u64",
    "st.global.u64",
    ".u64",
    ".s32"
);
const GATHER_ND_W8_I64_PTX: &str = gather_nd_ptx!(
    "gather_nd_w8_i64_kernel",
    "3",
    "3",
    "ld.global.s64",
    "s64",
    "ld.global.u64",
    "st.global.u64",
    ".u64",
    ".s64"
);

const CHECK_BOUNDS_I32_PTX: &str = check_bounds_ptx!(
    "check_bounds_i32_kernel",
    "2",
    "ld.global.s32 %val32, [%addr];
    cvt.s64.s32 %val, %val32;"
);
const CHECK_BOUNDS_I64_PTX: &str = check_bounds_ptx!(
    "check_bounds_i64_kernel",
    "3",
    "ld.global.s64 %val, [%addr];"
);
const EXPAND_INDEX_SELECT_I64_PTX: &str = expand_index_select_i64_ptx!();

/// Byte width of a value element, used to pick the copy kernel.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ValWidth {
    /// 2-byte element (f16 / bf16).
    W2,
    /// 4-byte element (f32 / i32).
    W4,
    /// 8-byte element (f64 / i64).
    W8,
}

/// Index element width.
#[derive(Clone, Copy, PartialEq, Eq)]
enum IdxWidth {
    /// 32-bit signed index.
    I32,
    /// 64-bit signed index.
    I64,
}

#[allow(clippy::too_many_arguments)]
fn launch_select<V: DeviceRepr + ValidAsZeroBits, I: DeviceRepr + ValidAsZeroBits>(
    input: &CudaSlice<V>,
    idx: &CudaSlice<I>,
    outer: usize,
    in_dim: usize,
    out_dim: usize,
    inner: usize,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
) -> GpuResult<CudaSlice<V>> {
    let total = outer
        .checked_mul(out_dim)
        .and_then(|x| x.checked_mul(inner))
        .ok_or(GpuError::LengthMismatch {
            a: outer,
            b: out_dim,
        })?;
    let stream = device.stream();
    if total == 0 {
        return Ok(stream.alloc_zeros::<V>(0)?);
    }
    let ctx = device.context();
    let f = get_or_compile(ctx, ptx, kernel_name, device.ordinal() as u32).map_err(|e| {
        GpuError::PtxCompileFailed {
            kernel: kernel_name,
            source: e,
        }
    })?;
    let mut out = stream.alloc_zeros::<V>(total)?;
    let cfg = launch_1d(total);
    let outer_u = checked_u32("outer", outer)?;
    let indim_u = checked_u32("in_dim", in_dim)?;
    let outdim_u = checked_u32("out_dim", out_dim)?;
    let inner_u = checked_u32("inner", inner)?;
    let total_u = checked_u32("total", total)?;
    // SAFETY:
    // - `f` is the PTX entry `kernel_name`; its 8-arg signature
    //   (in, idx, out, outer, in_dim, out_dim, inner, total) matches the args
    //   pushed below in order.
    // - `input` (V-elements) and `idx` (I-elements) are immutable inputs; `out`
    //   is the fresh `total`-element V buffer, the only `&mut`, non-aliased.
    // - Each thread writes one `out[t]` for `t in [0,total)` (bound-checked).
    //   The source element is computed from `idx[..]`; callers must run the
    //   resident `check_indices_*_in_bounds` gate before launching so invalid
    //   indices cannot reach this unchecked address math.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(idx)
            .arg(&mut out)
            .arg(&outer_u)
            .arg(&indim_u)
            .arg(&outdim_u)
            .arg(&inner_u)
            .arg(&total_u)
            .launch(cfg)?;
    }
    Ok(out)
}

fn checked_u32(name: &'static str, value: usize) -> GpuResult<u32> {
    u32::try_from(value).map_err(|_| GpuError::InvalidState {
        message: format!("{name}={value} exceeds gather kernel u32 indexing limit"),
    })
}

fn checked_i64(name: &'static str, value: usize) -> GpuResult<i64> {
    i64::try_from(value).map_err(|_| GpuError::InvalidState {
        message: format!("{name}={value} exceeds signed 64-bit indexing limit"),
    })
}

fn launch_check_bounds<I: DeviceRepr + ValidAsZeroBits>(
    idx: &CudaSlice<I>,
    n: usize,
    dim: usize,
    dim_size: usize,
    op: &'static str,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
) -> GpuResult<()> {
    if n > idx.len() {
        return Err(GpuError::LengthMismatch { a: n, b: idx.len() });
    }
    if n == 0 {
        return Ok(());
    }
    let total_u = checked_u32("index_len", n)?;
    let dim_size_i = checked_i64("dim_size", dim_size)?;
    let stream = device.stream();
    let ctx = device.context();
    let f = get_or_compile(ctx, ptx, kernel_name, device.ordinal() as u32).map_err(|e| {
        GpuError::PtxCompileFailed {
            kernel: kernel_name,
            source: e,
        }
    })?;
    let flag_init = [0_u32];
    let bad_init = [-1_i64, 0_i64];
    let mut flag = stream.clone_htod(&flag_init)?;
    let mut bad = stream.clone_htod(&bad_init)?;
    let cfg = launch_1d(n);
    // SAFETY:
    // - `f` is one of the `check_bounds_*` PTX entries; the pushed arguments
    //   match `(idx, flag, bad, total, dim_size)` exactly.
    // - `idx` is read-only. `flag` is a one-element u32 status buffer claimed
    //   with `atom.global.cas.b32`; `bad` is a two-element i64 buffer written
    //   only by the winner. Stream ordering makes the later D2H status read
    //   observe these writes after the kernel completes.
    // - Each active thread reads `idx[t]` for `t < n`; `n <= idx.len()` was
    //   checked above, and there are no writes into the index tensor.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(idx)
            .arg(&mut flag)
            .arg(&mut bad)
            .arg(&total_u)
            .arg(&dim_size_i)
            .launch(cfg)?;
    }
    let flag_host = stream.clone_dtoh(&flag)?;
    if flag_host[0] == 0 {
        return Ok(());
    }
    let bad_host = stream.clone_dtoh(&bad)?;
    Err(GpuError::InvalidState {
        message: format!(
            "{op}: index {} is out of bounds for dimension {dim} with size {dim_size} \
             (flat index position {}, valid range 0..{dim_size})",
            bad_host[1], bad_host[0]
        ),
    })
}

/// Expand a 1-D i64 `index_select(dim)` index into the per-output i64 index
/// buffer used by dim-aware scatter-add backward kernels.
///
/// For output layout `[outer, out_dim, inner]`, element
/// `t = ((o * out_dim + i) * inner + k)` receives `out[t] = idx[i]`. The result
/// stays CUDA-resident and is intended for autograd backward wiring; it does not
/// validate index bounds because the forward path already ran the resident
/// bounds gate before saving the index.
pub fn expand_index_select_i64(
    idx: &CudaSlice<i64>,
    outer: usize,
    out_dim: usize,
    inner: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i64>> {
    if idx.len() != out_dim {
        return Err(GpuError::LengthMismatch {
            a: out_dim,
            b: idx.len(),
        });
    }
    let total = outer
        .checked_mul(out_dim)
        .and_then(|x| x.checked_mul(inner))
        .ok_or(GpuError::LengthMismatch {
            a: outer,
            b: out_dim,
        })?;
    let stream = d.stream();
    if total == 0 {
        return Ok(stream.alloc_zeros::<i64>(0)?);
    }
    let ctx = d.context();
    let f = get_or_compile(
        ctx,
        EXPAND_INDEX_SELECT_I64_PTX,
        "expand_index_select_i64_kernel",
        d.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "expand_index_select_i64_kernel",
        source: e,
    })?;
    let mut out = stream.alloc_zeros::<i64>(total)?;
    let cfg = launch_1d(total);
    let out_dim_u = checked_u32("out_dim", out_dim)?;
    let inner_u = checked_u32("inner", inner)?;
    let total_u = checked_u32("total", total)?;
    // SAFETY:
    // - `f` is `expand_index_select_i64_kernel`; its 5-arg signature
    //   `(idx, out, out_dim, inner, total)` matches the pushed args.
    // - `idx` is read-only and has exactly `out_dim` elements. `out` is freshly
    //   allocated with `total = outer*out_dim*inner` elements and is the only
    //   mutable buffer.
    // - Each active thread writes one `out[t]`; `total > 0` implies
    //   `out_dim > 0` and `inner > 0`, so the PTX div/rem operands are nonzero.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(idx)
            .arg(&mut out)
            .arg(&out_dim_u)
            .arg(&inner_u)
            .arg(&total_u)
            .launch(cfg)?;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn launch_gather_nd<V: DeviceRepr + ValidAsZeroBits, I: DeviceRepr + ValidAsZeroBits>(
    input: &CudaSlice<V>,
    idx: &CudaSlice<I>,
    input_strides: &[u32],
    index_shape: &[u32],
    dim: usize,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
) -> GpuResult<CudaSlice<V>> {
    let rank = index_shape.len();
    if rank == 0 || input_strides.len() != rank || dim >= rank {
        return Err(GpuError::InvalidState {
            message: format!(
                "gather_nd: invalid metadata rank={rank} strides={} dim={dim}",
                input_strides.len()
            ),
        });
    }
    let total = index_shape.iter().try_fold(1usize, |acc, &d| {
        acc.checked_mul(d as usize).ok_or(GpuError::LengthMismatch {
            a: acc,
            b: d as usize,
        })
    })?;
    let stream = device.stream();
    if total == 0 {
        return Ok(stream.alloc_zeros::<V>(0)?);
    }
    let ctx = device.context();
    let f = get_or_compile(ctx, ptx, kernel_name, device.ordinal() as u32).map_err(|e| {
        GpuError::PtxCompileFailed {
            kernel: kernel_name,
            source: e,
        }
    })?;
    let input_strides_vec = input_strides.to_vec();
    let index_shape_vec = index_shape.to_vec();
    let input_strides_dev = stream.clone_htod(&input_strides_vec)?;
    let index_shape_dev = stream.clone_htod(&index_shape_vec)?;
    let mut out = stream.alloc_zeros::<V>(total)?;
    let cfg = launch_1d(total);
    let rank_u = checked_u32("rank", rank)?;
    let dim_u = checked_u32("dim", dim)?;
    let total_u = checked_u32("total", total)?;
    // SAFETY:
    // - `f` is one of the `gather_nd_*` PTX entries; its signature
    //   (in, idx, input_strides, index_shape, out, rank, dim, total) matches
    //   the arguments pushed below.
    // - `input_strides_dev` and `index_shape_dev` are exact `rank`-element
    //   metadata buffers uploaded for this launch; `input` and `idx` are read
    //   only; `out` is freshly allocated and exclusively mutable.
    // - Each active thread writes exactly one `out[t]`, bounded by `total`.
    //   Source bounds are guaranteed by the core-side shape and index checks,
    //   matching PyTorch's CUDA contract for checked indices.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(idx)
            .arg(&input_strides_dev)
            .arg(&index_shape_dev)
            .arg(&mut out)
            .arg(&rank_u)
            .arg(&dim_u)
            .arg(&total_u)
            .launch(cfg)?;
    }
    Ok(out)
}

fn isel_ptx(vw: ValWidth, iw: IdxWidth) -> (&'static str, &'static str) {
    match (vw, iw) {
        (ValWidth::W2, IdxWidth::I32) => (ISEL_W2_I32_PTX, "isel_w2_i32_kernel"),
        (ValWidth::W2, IdxWidth::I64) => (ISEL_W2_I64_PTX, "isel_w2_i64_kernel"),
        (ValWidth::W4, IdxWidth::I32) => (ISEL_W4_I32_PTX, "isel_w4_i32_kernel"),
        (ValWidth::W4, IdxWidth::I64) => (ISEL_W4_I64_PTX, "isel_w4_i64_kernel"),
        (ValWidth::W8, IdxWidth::I32) => (ISEL_W8_I32_PTX, "isel_w8_i32_kernel"),
        (ValWidth::W8, IdxWidth::I64) => (ISEL_W8_I64_PTX, "isel_w8_i64_kernel"),
    }
}

fn gathr_ptx(vw: ValWidth, iw: IdxWidth) -> (&'static str, &'static str) {
    match (vw, iw) {
        (ValWidth::W2, IdxWidth::I32) => (GATHER_W2_I32_PTX, "gather_w2_i32_kernel"),
        (ValWidth::W2, IdxWidth::I64) => (GATHER_W2_I64_PTX, "gather_w2_i64_kernel"),
        (ValWidth::W4, IdxWidth::I32) => (GATHER_W4_I32_PTX, "gather_w4_i32_kernel"),
        (ValWidth::W4, IdxWidth::I64) => (GATHER_W4_I64_PTX, "gather_w4_i64_kernel"),
        (ValWidth::W8, IdxWidth::I32) => (GATHER_W8_I32_PTX, "gather_w8_i32_kernel"),
        (ValWidth::W8, IdxWidth::I64) => (GATHER_W8_I64_PTX, "gather_w8_i64_kernel"),
    }
}

fn gather_nd_ptx_for(vw: ValWidth, iw: IdxWidth) -> (&'static str, &'static str) {
    match (vw, iw) {
        (ValWidth::W2, IdxWidth::I32) => (GATHER_ND_W2_I32_PTX, "gather_nd_w2_i32_kernel"),
        (ValWidth::W2, IdxWidth::I64) => (GATHER_ND_W2_I64_PTX, "gather_nd_w2_i64_kernel"),
        (ValWidth::W4, IdxWidth::I32) => (GATHER_ND_W4_I32_PTX, "gather_nd_w4_i32_kernel"),
        (ValWidth::W4, IdxWidth::I64) => (GATHER_ND_W4_I64_PTX, "gather_nd_w4_i64_kernel"),
        (ValWidth::W8, IdxWidth::I32) => (GATHER_ND_W8_I32_PTX, "gather_nd_w8_i32_kernel"),
        (ValWidth::W8, IdxWidth::I64) => (GATHER_ND_W8_I64_PTX, "gather_nd_w8_i64_kernel"),
    }
}

macro_rules! select_entry {
    ($name:ident, $vty:ty, $vw:expr, $idxty:ty, $iw:expr, $sel:ident) => {
        #[doc = concat!("`", stringify!($sel), "` on a ", stringify!($vty), " value buffer with a ", stringify!($idxty), " index buffer.")]
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            input: &CudaSlice<$vty>,
            idx: &CudaSlice<$idxty>,
            outer: usize,
            in_dim: usize,
            out_dim: usize,
            inner: usize,
            d: &GpuDevice,
        ) -> GpuResult<CudaSlice<$vty>> {
            let (ptx, name) = $sel($vw, $iw);
            launch_select(input, idx, outer, in_dim, out_dim, inner, d, ptx, name)
        }
    };
}

macro_rules! gather_nd_entry {
    ($name:ident, $vty:ty, $vw:expr, $idxty:ty, $iw:expr) => {
        #[doc = concat!("Rank-aware `gather` on a ", stringify!($vty), " value buffer with a ", stringify!($idxty), " index buffer.")]
        pub fn $name(
            input: &CudaSlice<$vty>,
            idx: &CudaSlice<$idxty>,
            input_strides: &[u32],
            index_shape: &[u32],
            dim: usize,
            d: &GpuDevice,
        ) -> GpuResult<CudaSlice<$vty>> {
            let (ptx, name) = gather_nd_ptx_for($vw, $iw);
            launch_gather_nd(input, idx, input_strides, index_shape, dim, d, ptx, name)
        }
    };
}

/// Validate an `i32` index buffer against `0 <= idx < dim_size` on CUDA.
pub fn check_indices_i32_in_bounds(
    idx: &CudaSlice<i32>,
    n: usize,
    dim: usize,
    dim_size: usize,
    op: &'static str,
    d: &GpuDevice,
) -> GpuResult<()> {
    launch_check_bounds(
        idx,
        n,
        dim,
        dim_size,
        op,
        d,
        CHECK_BOUNDS_I32_PTX,
        "check_bounds_i32_kernel",
    )
}

/// Validate an `i64` index buffer against `0 <= idx < dim_size` on CUDA.
pub fn check_indices_i64_in_bounds(
    idx: &CudaSlice<i64>,
    n: usize,
    dim: usize,
    dim_size: usize,
    op: &'static str,
    d: &GpuDevice,
) -> GpuResult<()> {
    launch_check_bounds(
        idx,
        n,
        dim,
        dim_size,
        op,
        d,
        CHECK_BOUNDS_I64_PTX,
        "check_bounds_i64_kernel",
    )
}

// index_select: value f32/f64/i32/i64/u16(f16,bf16) × index i32/i64
select_entry!(
    isel_f32_i32,
    f32,
    ValWidth::W4,
    i32,
    IdxWidth::I32,
    isel_ptx
);
select_entry!(
    isel_f32_i64,
    f32,
    ValWidth::W4,
    i64,
    IdxWidth::I64,
    isel_ptx
);
select_entry!(
    isel_f64_i32,
    f64,
    ValWidth::W8,
    i32,
    IdxWidth::I32,
    isel_ptx
);
select_entry!(
    isel_f64_i64,
    f64,
    ValWidth::W8,
    i64,
    IdxWidth::I64,
    isel_ptx
);
select_entry!(
    isel_i32_i32,
    i32,
    ValWidth::W4,
    i32,
    IdxWidth::I32,
    isel_ptx
);
select_entry!(
    isel_i32_i64,
    i32,
    ValWidth::W4,
    i64,
    IdxWidth::I64,
    isel_ptx
);
select_entry!(
    isel_i64_i32,
    i64,
    ValWidth::W8,
    i32,
    IdxWidth::I32,
    isel_ptx
);
select_entry!(
    isel_i64_i64,
    i64,
    ValWidth::W8,
    i64,
    IdxWidth::I64,
    isel_ptx
);
select_entry!(
    isel_u16_i32,
    u16,
    ValWidth::W2,
    i32,
    IdxWidth::I32,
    isel_ptx
);
select_entry!(
    isel_u16_i64,
    u16,
    ValWidth::W2,
    i64,
    IdxWidth::I64,
    isel_ptx
);

// gather: same matrix
select_entry!(
    gather_f32_i32,
    f32,
    ValWidth::W4,
    i32,
    IdxWidth::I32,
    gathr_ptx
);
select_entry!(
    gather_f32_i64,
    f32,
    ValWidth::W4,
    i64,
    IdxWidth::I64,
    gathr_ptx
);
select_entry!(
    gather_f64_i32,
    f64,
    ValWidth::W8,
    i32,
    IdxWidth::I32,
    gathr_ptx
);
select_entry!(
    gather_f64_i64,
    f64,
    ValWidth::W8,
    i64,
    IdxWidth::I64,
    gathr_ptx
);
select_entry!(
    gather_i32_i32,
    i32,
    ValWidth::W4,
    i32,
    IdxWidth::I32,
    gathr_ptx
);
select_entry!(
    gather_i32_i64,
    i32,
    ValWidth::W4,
    i64,
    IdxWidth::I64,
    gathr_ptx
);
select_entry!(
    gather_i64_i32,
    i64,
    ValWidth::W8,
    i32,
    IdxWidth::I32,
    gathr_ptx
);
select_entry!(
    gather_i64_i64,
    i64,
    ValWidth::W8,
    i64,
    IdxWidth::I64,
    gathr_ptx
);
select_entry!(
    gather_u16_i32,
    u16,
    ValWidth::W2,
    i32,
    IdxWidth::I32,
    gathr_ptx
);
select_entry!(
    gather_u16_i64,
    u16,
    ValWidth::W2,
    i64,
    IdxWidth::I64,
    gathr_ptx
);

// gather_nd: same value/index matrix, but with full-rank C-order metadata.
gather_nd_entry!(gather_nd_f32_i32, f32, ValWidth::W4, i32, IdxWidth::I32);
gather_nd_entry!(gather_nd_f32_i64, f32, ValWidth::W4, i64, IdxWidth::I64);
gather_nd_entry!(gather_nd_f64_i32, f64, ValWidth::W8, i32, IdxWidth::I32);
gather_nd_entry!(gather_nd_f64_i64, f64, ValWidth::W8, i64, IdxWidth::I64);
gather_nd_entry!(gather_nd_i32_i32, i32, ValWidth::W4, i32, IdxWidth::I32);
gather_nd_entry!(gather_nd_i32_i64, i32, ValWidth::W4, i64, IdxWidth::I64);
gather_nd_entry!(gather_nd_i64_i32, i64, ValWidth::W8, i32, IdxWidth::I32);
gather_nd_entry!(gather_nd_i64_i64, i64, ValWidth::W8, i64, IdxWidth::I64);
gather_nd_entry!(gather_nd_u16_i32, u16, ValWidth::W2, i32, IdxWidth::I32);
gather_nd_entry!(gather_nd_u16_i64, u16, ValWidth::W2, i64, IdxWidth::I64);

#[cfg(test)]
mod tests {
    use super::*;

    fn dev() -> GpuDevice {
        GpuDevice::new(0).expect("cuda device")
    }

    #[test]
    fn index_select_dim0_f32_i64() {
        let d = dev();
        // input [4,2] rows; select rows [2,0,2] -> output [3,2]
        let inp = d
            .stream()
            .clone_htod(&vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
            .unwrap();
        let idx = d.stream().clone_htod(&vec![2i64, 0, 2]).unwrap();
        // outer=1 in_dim=4 out_dim=3 inner=2
        let out = isel_f32_i64(&inp, &idx, 1, 4, 3, 2, &d).unwrap();
        assert_eq!(
            d.stream().clone_dtoh(&out).unwrap(),
            vec![4.0f32, 5.0, 0.0, 1.0, 4.0, 5.0]
        );
    }

    #[test]
    fn index_select_dim1_f32_i32() {
        let d = dev();
        // input [2,3], select along dim=1 indices [2,0] -> outer=2 in_dim=3 out_dim=2 inner=1
        let inp = d
            .stream()
            .clone_htod(&vec![10.0f32, 11.0, 12.0, 20.0, 21.0, 22.0])
            .unwrap();
        let idx = d.stream().clone_htod(&vec![2i32, 0]).unwrap();
        let out = isel_f32_i32(&inp, &idx, 2, 3, 2, 1, &d).unwrap();
        assert_eq!(
            d.stream().clone_dtoh(&out).unwrap(),
            vec![12.0f32, 10.0, 22.0, 20.0]
        );
    }

    #[test]
    fn gather_dim1_i32_values() {
        let d = dev();
        // gather along dim=1: input [2,3], index [2,2] -> outer=2 in_dim=3 out_dim=2 inner=1
        let inp = d.stream().clone_htod(&vec![5i32, 6, 7, 8, 9, 10]).unwrap();
        let idx = d.stream().clone_htod(&vec![0i64, 2, 2, 1]).unwrap();
        let out = gather_i32_i64(&inp, &idx, 2, 3, 2, 1, &d).unwrap();
        // row0: in[0,0]=5 in[0,2]=7 ; row1: in[1,2]=10 in[1,1]=9
        assert_eq!(d.stream().clone_dtoh(&out).unwrap(), vec![5i32, 7, 10, 9]);
    }

    #[test]
    fn gather_nd_dim1_smaller_batch_f32_i64() {
        let d = dev();
        let inp = d
            .stream()
            .clone_htod(&vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let idx = d.stream().clone_htod(&vec![1i64, 0]).unwrap();
        let out = gather_nd_f32_i64(&inp, &idx, &[3, 1], &[1, 2], 1, &d).unwrap();
        assert_eq!(d.stream().clone_dtoh(&out).unwrap(), vec![2.0f32, 1.0]);
    }

    #[test]
    fn gather_nd_dim0_smaller_column_f32_i64() {
        let d = dev();
        let inp = d
            .stream()
            .clone_htod(&vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let idx = d.stream().clone_htod(&vec![1i64, 0]).unwrap();
        let out = gather_nd_f32_i64(&inp, &idx, &[3, 1], &[2, 1], 0, &d).unwrap();
        assert_eq!(d.stream().clone_dtoh(&out).unwrap(), vec![4.0f32, 1.0]);
    }

    #[test]
    fn gather_nd_dim1_smaller_batch_i64_values() {
        let d = dev();
        let inp = d.stream().clone_htod(&vec![1i64, 2, 3, 4, 5, 6]).unwrap();
        let idx = d.stream().clone_htod(&vec![1i32, 0]).unwrap();
        let out = gather_nd_i64_i32(&inp, &idx, &[3, 1], &[1, 2], 1, &d).unwrap();
        assert_eq!(d.stream().clone_dtoh(&out).unwrap(), vec![2i64, 1]);
    }

    #[test]
    fn expand_index_select_i64_repeats_index_across_outer_and_inner() {
        let d = dev();
        let idx = d.stream().clone_htod(&vec![2i64, 0]).unwrap();
        let out = expand_index_select_i64(&idx, 2, 2, 3, &d).unwrap();
        assert_eq!(
            d.stream().clone_dtoh(&out).unwrap(),
            vec![2i64, 2, 2, 0, 0, 0, 2, 2, 2, 0, 0, 0]
        );
    }

    #[test]
    fn index_select_f64_and_i64_values() {
        let d = dev();
        let inp = d.stream().clone_htod(&vec![1.5f64, 2.5, 3.5, 4.5]).unwrap();
        let idx = d.stream().clone_htod(&vec![1i32, 0]).unwrap();
        let out = isel_f64_i32(&inp, &idx, 1, 2, 2, 2, &d).unwrap();
        assert_eq!(
            d.stream().clone_dtoh(&out).unwrap(),
            vec![3.5f64, 4.5, 1.5, 2.5]
        );
    }
}
