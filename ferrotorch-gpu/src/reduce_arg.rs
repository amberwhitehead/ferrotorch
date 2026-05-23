//! Argmax / argmin GPU compute kernels — crosslink #1185 Phase 2c.
//!
//! Hand-written PTX owned by Rust (no CUDA C++, no nvrtc), loaded via
//! [`crate::module_cache::get_or_compile`] exactly like [`crate::int_kernels`].
//!
//! # Semantics (PyTorch parity)
//!
//! - Output dtype is **always i64** (PyTorch returns int64 indices).
//! - Tie-break is the **FIRST** occurrence (lowest index), matching
//!   `torch.argmax` / `torch.argmin`. We achieve this with a STRICT comparison
//!   (`>` for argmax, `<` for argmin): the accumulator is updated only when a
//!   later element is strictly better, so equal values never displace the
//!   earlier index.
//! - Value comparison happens in the value dtype's own registers — f32/f64 use
//!   IEEE float compares (`setp.gt.f32/f64`); i32/i64 use signed-integer
//!   compares (`setp.gt.s32/s64`); bf16/f16 decode each 16-bit element to f32
//!   (`cvt.f32.f16` after a bf16→f32 hi-half splat / an f16 widening cvt) and
//!   compare in f32. NaN handling matches a strict `>`/`<`: a NaN never wins,
//!   so the first non-NaN-or-index-0 element is kept (PyTorch's argmax on CUDA
//!   propagates NaN; this is a documented minor divergence — see the module
//!   note on NaN below).
//!
//! # Launch scheme
//!
//! Logical layout: input = `[outer, dim_size, inner]` (flat C-order). One
//! thread per `(outer_idx, inner_idx)` pair (i.e. per output slice). Each
//! thread serially scans the `dim_size` elements at stride `inner`, tracking
//! the best value and its index, then writes one i64 index. `total =
//! outer * inner` threads.
//!
//! - **Global** argmax/argmin (flatten): the caller sets `outer = 1`,
//!   `dim_size = numel`, `inner = 1` → a single thread folds the whole buffer.
//! - **Along-dim**: `outer = product(shape[..dim])`, `dim_size = shape[dim]`,
//!   `inner = product(shape[dim+1..])`.
//!
//! This serial-per-slice scheme guarantees the result equals a left-to-right
//! scan, so the first-occurrence tie-break is exact and matches the CPU
//! reference bit-for-bit.
//!
//! # NaN
//!
//! With a strict `>` compare, a NaN value is never `> acc`, so it is skipped.
//! If element 0 is NaN it seeds `acc` and is only displaced by a real `>`
//! compare (which against NaN is false), so a slice of all-NaN reports index 0.
//! PyTorch's CUDA argmax returns the index of a NaN if present. This is a known
//! minor divergence; ferrotorch's float reductions elsewhere take the same
//! pragmatic strict-compare stance. Documented, not silently hidden.

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

// `op` selector pushed to the kernels: 0 = argmax, 1 = argmin.
const ARG_MAX: u32 = 0;
const ARG_MIN: u32 = 1;

// ===========================================================================
// f32
//
// Params: (in_ptr, out_ptr, outer, dim_size, inner, total, op)
//   in  : f32[outer * dim_size * inner]
//   out : i64[outer * inner]   (= total)
// Thread t in [0, total): outer_idx = t / inner; inner_idx = t % inner.
//   base = outer_idx * dim_size * inner + inner_idx
//   scan j in [0, dim_size): v = in[base + j*inner]
//     argmax: keep first j with strictly-greatest v
//     argmin: keep first j with strictly-least v
//   out[t] = best_j  (as s64)
// ===========================================================================
const ARGREDUCE_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry argreduce_f32_kernel(
    .param .u64 in_ptr, .param .u64 out_ptr,
    .param .u32 outer, .param .u32 dim_size, .param .u32 inner,
    .param .u32 total, .param .u32 op
) {
    .reg .u32 %gtid, %bid, %bdim, %tot, %dim, %inn, %op_r;
    .reg .u32 %oidx, %iidx, %base, %j, %elem, %best_j;
    .reg .u64 %in, %out, %off, %addr;
    .reg .f32 %v, %acc;
    .reg .s64 %best_s64;
    .reg .pred %p, %is_max, %not_max, %better, %lt, %gt;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %tot, [total];
    ld.param.u32 %dim, [dim_size];
    ld.param.u32 %inn, [inner];
    ld.param.u32 %op_r, [op];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %gtid, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %gtid;
    setp.ge.u32 %p, %gtid, %tot;
    @%p bra DONE;

    setp.eq.u32 %is_max, %op_r, 0;
    not.pred %not_max, %is_max;

    div.u32 %oidx, %gtid, %inn;
    rem.u32 %iidx, %gtid, %inn;
    // base = (oidx * dim) * inn + iidx
    mul.lo.u32 %base, %oidx, %dim;
    mul.lo.u32 %base, %base, %inn;
    add.u32 %base, %base, %iidx;

    // seed with element j=0
    cvt.u64.u32 %off, %base;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in, %off;
    ld.global.f32 %acc, [%addr];
    mov.u32 %best_j, 0;

    mov.u32 %j, 1;
LOOP:
    setp.ge.u32 %p, %j, %dim;
    @%p bra STORE;

    // elem = base + j*inn
    mul.lo.u32 %elem, %j, %inn;
    add.u32 %elem, %elem, %base;
    cvt.u64.u32 %off, %elem;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in, %off;
    ld.global.f32 %v, [%addr];

    // strict compare: argmax keeps first-greatest, argmin first-least
    setp.gt.f32 %gt, %v, %acc;
    setp.lt.f32 %lt, %v, %acc;
    and.pred %gt, %gt, %is_max;
    and.pred %lt, %lt, %not_max;
    or.pred %better, %gt, %lt;
    @%better mov.f32 %acc, %v;
    @%better mov.u32 %best_j, %j;

    add.u32 %j, %j, 1;
    bra LOOP;
STORE:
    cvt.s64.u32 %best_s64, %best_j;
    cvt.u64.u32 %off, %gtid;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %out, %off;
    st.global.s64 [%addr], %best_s64;
DONE:
    ret;
}
";

// ===========================================================================
// f64 — same structure, 8-byte value stride, .f64 compares.
// ===========================================================================
const ARGREDUCE_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry argreduce_f64_kernel(
    .param .u64 in_ptr, .param .u64 out_ptr,
    .param .u32 outer, .param .u32 dim_size, .param .u32 inner,
    .param .u32 total, .param .u32 op
) {
    .reg .u32 %gtid, %bid, %bdim, %tot, %dim, %inn, %op_r;
    .reg .u32 %oidx, %iidx, %base, %j, %elem, %best_j;
    .reg .u64 %in, %out, %off, %addr;
    .reg .f64 %v, %acc;
    .reg .s64 %best_s64;
    .reg .pred %p, %is_max, %not_max, %better, %lt, %gt;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %tot, [total];
    ld.param.u32 %dim, [dim_size];
    ld.param.u32 %inn, [inner];
    ld.param.u32 %op_r, [op];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %gtid, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %gtid;
    setp.ge.u32 %p, %gtid, %tot;
    @%p bra DONE;

    setp.eq.u32 %is_max, %op_r, 0;
    not.pred %not_max, %is_max;

    div.u32 %oidx, %gtid, %inn;
    rem.u32 %iidx, %gtid, %inn;
    mul.lo.u32 %base, %oidx, %dim;
    mul.lo.u32 %base, %base, %inn;
    add.u32 %base, %base, %iidx;

    cvt.u64.u32 %off, %base;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %in, %off;
    ld.global.f64 %acc, [%addr];
    mov.u32 %best_j, 0;

    mov.u32 %j, 1;
LOOP:
    setp.ge.u32 %p, %j, %dim;
    @%p bra STORE;

    mul.lo.u32 %elem, %j, %inn;
    add.u32 %elem, %elem, %base;
    cvt.u64.u32 %off, %elem;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %in, %off;
    ld.global.f64 %v, [%addr];

    setp.gt.f64 %gt, %v, %acc;
    setp.lt.f64 %lt, %v, %acc;
    and.pred %gt, %gt, %is_max;
    and.pred %lt, %lt, %not_max;
    or.pred %better, %gt, %lt;
    @%better mov.f64 %acc, %v;
    @%better mov.u32 %best_j, %j;

    add.u32 %j, %j, 1;
    bra LOOP;
STORE:
    cvt.s64.u32 %best_s64, %best_j;
    cvt.u64.u32 %off, %gtid;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %out, %off;
    st.global.s64 [%addr], %best_s64;
DONE:
    ret;
}
";

// ===========================================================================
// i32 — signed-integer compares (.s32), 4-byte value stride.
// ===========================================================================
const ARGREDUCE_I32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry argreduce_i32_kernel(
    .param .u64 in_ptr, .param .u64 out_ptr,
    .param .u32 outer, .param .u32 dim_size, .param .u32 inner,
    .param .u32 total, .param .u32 op
) {
    .reg .u32 %gtid, %bid, %bdim, %tot, %dim, %inn, %op_r;
    .reg .u32 %oidx, %iidx, %base, %j, %elem, %best_j;
    .reg .u64 %in, %out, %off, %addr;
    .reg .s32 %v, %acc;
    .reg .s64 %best_s64;
    .reg .pred %p, %is_max, %not_max, %better, %lt, %gt;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %tot, [total];
    ld.param.u32 %dim, [dim_size];
    ld.param.u32 %inn, [inner];
    ld.param.u32 %op_r, [op];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %gtid, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %gtid;
    setp.ge.u32 %p, %gtid, %tot;
    @%p bra DONE;

    setp.eq.u32 %is_max, %op_r, 0;
    not.pred %not_max, %is_max;

    div.u32 %oidx, %gtid, %inn;
    rem.u32 %iidx, %gtid, %inn;
    mul.lo.u32 %base, %oidx, %dim;
    mul.lo.u32 %base, %base, %inn;
    add.u32 %base, %base, %iidx;

    cvt.u64.u32 %off, %base;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in, %off;
    ld.global.s32 %acc, [%addr];
    mov.u32 %best_j, 0;

    mov.u32 %j, 1;
LOOP:
    setp.ge.u32 %p, %j, %dim;
    @%p bra STORE;

    mul.lo.u32 %elem, %j, %inn;
    add.u32 %elem, %elem, %base;
    cvt.u64.u32 %off, %elem;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in, %off;
    ld.global.s32 %v, [%addr];

    setp.gt.s32 %gt, %v, %acc;
    setp.lt.s32 %lt, %v, %acc;
    and.pred %gt, %gt, %is_max;
    and.pred %lt, %lt, %not_max;
    or.pred %better, %gt, %lt;
    @%better mov.s32 %acc, %v;
    @%better mov.u32 %best_j, %j;

    add.u32 %j, %j, 1;
    bra LOOP;
STORE:
    cvt.s64.u32 %best_s64, %best_j;
    cvt.u64.u32 %off, %gtid;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %out, %off;
    st.global.s64 [%addr], %best_s64;
DONE:
    ret;
}
";

// ===========================================================================
// i64 — signed-integer compares (.s64), 8-byte value stride.
// ===========================================================================
const ARGREDUCE_I64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry argreduce_i64_kernel(
    .param .u64 in_ptr, .param .u64 out_ptr,
    .param .u32 outer, .param .u32 dim_size, .param .u32 inner,
    .param .u32 total, .param .u32 op
) {
    .reg .u32 %gtid, %bid, %bdim, %tot, %dim, %inn, %op_r;
    .reg .u32 %oidx, %iidx, %base, %j, %elem, %best_j;
    .reg .u64 %in, %out, %off, %addr;
    .reg .s64 %v, %acc, %best_s64;
    .reg .pred %p, %is_max, %not_max, %better, %lt, %gt;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %tot, [total];
    ld.param.u32 %dim, [dim_size];
    ld.param.u32 %inn, [inner];
    ld.param.u32 %op_r, [op];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %gtid, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %gtid;
    setp.ge.u32 %p, %gtid, %tot;
    @%p bra DONE;

    setp.eq.u32 %is_max, %op_r, 0;
    not.pred %not_max, %is_max;

    div.u32 %oidx, %gtid, %inn;
    rem.u32 %iidx, %gtid, %inn;
    mul.lo.u32 %base, %oidx, %dim;
    mul.lo.u32 %base, %base, %inn;
    add.u32 %base, %base, %iidx;

    cvt.u64.u32 %off, %base;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %in, %off;
    ld.global.s64 %acc, [%addr];
    mov.u32 %best_j, 0;

    mov.u32 %j, 1;
LOOP:
    setp.ge.u32 %p, %j, %dim;
    @%p bra STORE;

    mul.lo.u32 %elem, %j, %inn;
    add.u32 %elem, %elem, %base;
    cvt.u64.u32 %off, %elem;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %in, %off;
    ld.global.s64 %v, [%addr];

    setp.gt.s64 %gt, %v, %acc;
    setp.lt.s64 %lt, %v, %acc;
    and.pred %gt, %gt, %is_max;
    and.pred %lt, %lt, %not_max;
    or.pred %better, %gt, %lt;
    @%better mov.s64 %acc, %v;
    @%better mov.u32 %best_j, %j;

    add.u32 %j, %j, 1;
    bra LOOP;
STORE:
    cvt.s64.u32 %best_s64, %best_j;
    cvt.u64.u32 %off, %gtid;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %out, %off;
    st.global.s64 [%addr], %best_s64;
DONE:
    ret;
}
";

// ===========================================================================
// f16 / bf16 — 2-byte storage decoded to f32 for comparison.
//
// f16: widen each 16-bit lane with `cvt.f32.f16`.
// bf16: a bf16 is the high 16 bits of an f32; splat into the high half of a
//   32-bit register (`shl.b32 by 16`) and reinterpret as f32 (`mov.b32`).
// Both then compare in f32 exactly like ARGREDUCE_F32.
// ===========================================================================
const ARGREDUCE_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry argreduce_f16_kernel(
    .param .u64 in_ptr, .param .u64 out_ptr,
    .param .u32 outer, .param .u32 dim_size, .param .u32 inner,
    .param .u32 total, .param .u32 op
) {
    .reg .u32 %gtid, %bid, %bdim, %tot, %dim, %inn, %op_r;
    .reg .u32 %oidx, %iidx, %base, %j, %elem, %best_j;
    .reg .u64 %in, %out, %off, %addr;
    .reg .b16 %h;
    .reg .f32 %v, %acc;
    .reg .s64 %best_s64;
    .reg .pred %p, %is_max, %not_max, %better, %lt, %gt;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %tot, [total];
    ld.param.u32 %dim, [dim_size];
    ld.param.u32 %inn, [inner];
    ld.param.u32 %op_r, [op];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %gtid, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %gtid;
    setp.ge.u32 %p, %gtid, %tot;
    @%p bra DONE;

    setp.eq.u32 %is_max, %op_r, 0;
    not.pred %not_max, %is_max;

    div.u32 %oidx, %gtid, %inn;
    rem.u32 %iidx, %gtid, %inn;
    mul.lo.u32 %base, %oidx, %dim;
    mul.lo.u32 %base, %base, %inn;
    add.u32 %base, %base, %iidx;

    cvt.u64.u32 %off, %base;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %in, %off;
    ld.global.b16 %h, [%addr];
    cvt.f32.f16 %acc, %h;
    mov.u32 %best_j, 0;

    mov.u32 %j, 1;
LOOP:
    setp.ge.u32 %p, %j, %dim;
    @%p bra STORE;

    mul.lo.u32 %elem, %j, %inn;
    add.u32 %elem, %elem, %base;
    cvt.u64.u32 %off, %elem;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %in, %off;
    ld.global.b16 %h, [%addr];
    cvt.f32.f16 %v, %h;

    setp.gt.f32 %gt, %v, %acc;
    setp.lt.f32 %lt, %v, %acc;
    and.pred %gt, %gt, %is_max;
    and.pred %lt, %lt, %not_max;
    or.pred %better, %gt, %lt;
    @%better mov.f32 %acc, %v;
    @%better mov.u32 %best_j, %j;

    add.u32 %j, %j, 1;
    bra LOOP;
STORE:
    cvt.s64.u32 %best_s64, %best_j;
    cvt.u64.u32 %off, %gtid;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %out, %off;
    st.global.s64 [%addr], %best_s64;
DONE:
    ret;
}
";

const ARGREDUCE_BF16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry argreduce_bf16_kernel(
    .param .u64 in_ptr, .param .u64 out_ptr,
    .param .u32 outer, .param .u32 dim_size, .param .u32 inner,
    .param .u32 total, .param .u32 op
) {
    .reg .u32 %gtid, %bid, %bdim, %tot, %dim, %inn, %op_r;
    .reg .u32 %oidx, %iidx, %base, %j, %elem, %best_j, %bits;
    .reg .u16 %h;
    .reg .u64 %in, %out, %off, %addr;
    .reg .f32 %v, %acc;
    .reg .s64 %best_s64;
    .reg .pred %p, %is_max, %not_max, %better, %lt, %gt;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %tot, [total];
    ld.param.u32 %dim, [dim_size];
    ld.param.u32 %inn, [inner];
    ld.param.u32 %op_r, [op];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %gtid, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %gtid;
    setp.ge.u32 %p, %gtid, %tot;
    @%p bra DONE;

    setp.eq.u32 %is_max, %op_r, 0;
    not.pred %not_max, %is_max;

    div.u32 %oidx, %gtid, %inn;
    rem.u32 %iidx, %gtid, %inn;
    mul.lo.u32 %base, %oidx, %dim;
    mul.lo.u32 %base, %base, %inn;
    add.u32 %base, %base, %iidx;

    cvt.u64.u32 %off, %base;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %in, %off;
    ld.global.u16 %h, [%addr];
    cvt.u32.u16 %bits, %h;
    shl.b32 %bits, %bits, 16;
    mov.b32 %acc, %bits;
    mov.u32 %best_j, 0;

    mov.u32 %j, 1;
LOOP:
    setp.ge.u32 %p, %j, %dim;
    @%p bra STORE;

    mul.lo.u32 %elem, %j, %inn;
    add.u32 %elem, %elem, %base;
    cvt.u64.u32 %off, %elem;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %in, %off;
    ld.global.u16 %h, [%addr];
    cvt.u32.u16 %bits, %h;
    shl.b32 %bits, %bits, 16;
    mov.b32 %v, %bits;

    setp.gt.f32 %gt, %v, %acc;
    setp.lt.f32 %lt, %v, %acc;
    and.pred %gt, %gt, %is_max;
    and.pred %lt, %lt, %not_max;
    or.pred %better, %gt, %lt;
    @%better mov.f32 %acc, %v;
    @%better mov.u32 %best_j, %j;

    add.u32 %j, %j, 1;
    bra LOOP;
STORE:
    cvt.s64.u32 %best_s64, %best_j;
    cvt.u64.u32 %off, %gtid;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %out, %off;
    st.global.s64 [%addr], %best_s64;
DONE:
    ret;
}
";

/// Launch an argmax/argmin kernel over a value buffer of native element type
/// `V`, returning a fresh resident `CudaSlice<i64>` of `outer * inner` indices.
///
/// `in_slice` holds `outer * dim_size * inner` `V`-elements (contiguous,
/// `[outer, dim_size, inner]` C-order). `op` is [`ARG_MAX`] or [`ARG_MIN`].
#[allow(clippy::too_many_arguments)]
fn launch_argreduce<V: DeviceRepr + ValidAsZeroBits>(
    in_slice: &CudaSlice<V>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
    op: u32,
) -> GpuResult<CudaSlice<i64>> {
    let total = outer.checked_mul(inner).ok_or(GpuError::LengthMismatch {
        a: outer,
        b: inner,
    })?;
    let expect = outer
        .checked_mul(dim_size)
        .and_then(|x| x.checked_mul(inner))
        .ok_or(GpuError::LengthMismatch {
            a: outer,
            b: dim_size,
        })?;
    // The input slice may be POOL-OVERSIZED (its `CudaSlice::len()` is the
    // rounded allocation, not the logical numel) when it comes from a pooled
    // float op like `relu`. We only require it holds AT LEAST `expect`
    // elements; the kernel reads strictly within `[0, expect)`.
    if in_slice.len() < expect {
        return Err(GpuError::LengthMismatch {
            a: in_slice.len(),
            b: expect,
        });
    }
    let stream = device.stream();
    if total == 0 || dim_size == 0 {
        // Empty reduction dim is guarded by the high-level wrapper (PyTorch
        // raises); an empty output is a valid 0-length i64 buffer.
        return Ok(stream.alloc_zeros::<i64>(total)?);
    }
    let ctx = device.context();
    let f = get_or_compile(ctx, ptx, kernel_name, device.ordinal() as u32).map_err(|e| {
        GpuError::PtxCompileFailed {
            kernel: kernel_name,
            source: e,
        }
    })?;
    let mut out = stream.alloc_zeros::<i64>(total)?;
    let cfg = launch_1d(total);
    let (outer_u, dim_u, inner_u, total_u) = (
        outer as u32,
        dim_size as u32,
        inner as u32,
        total as u32,
    );
    // SAFETY:
    // - `f` is the PTX entry `kernel_name`; its 7-arg signature
    //   (in_ptr, out_ptr, outer, dim_size, inner, total, op) matches the
    //   args pushed below in order.
    // - `in_slice` holds exactly `outer*dim_size*inner` `V`-elements (checked).
    // - `out` is a fresh `total`-element i64 buffer, the only `&mut`, non-aliased.
    // - Each thread reads `in[base + j*inner]` for `j in [0,dim_size)` (in range)
    //   and writes one `out[tid]` for `tid in [0,total)` (bound-checked by
    //   `setp.ge.u32 %p, %gtid, %tot`).
    unsafe {
        stream
            .launch_builder(&f)
            .arg(in_slice)
            .arg(&mut out)
            .arg(&outer_u)
            .arg(&dim_u)
            .arg(&inner_u)
            .arg(&total_u)
            .arg(&op)
            .launch(cfg)?;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Public entry points — one per value dtype × {argmax, argmin}.
// All return a resident `CudaSlice<i64>` of indices (PyTorch int64 parity).
// ---------------------------------------------------------------------------

macro_rules! arg_entry {
    ($name:ident, $ty:ty, $ptx:ident, $kname:literal, $op:expr) => {
        #[doc = concat!("`", stringify!($name), "` over a ", stringify!($ty), " value buffer.")]
        pub fn $name(
            input: &CudaSlice<$ty>,
            outer: usize,
            dim_size: usize,
            inner: usize,
            d: &GpuDevice,
        ) -> GpuResult<CudaSlice<i64>> {
            launch_argreduce(input, outer, dim_size, inner, d, $ptx, $kname, $op)
        }
    };
}

arg_entry!(gpu_argmax_f32, f32, ARGREDUCE_F32_PTX, "argreduce_f32_kernel", ARG_MAX);
arg_entry!(gpu_argmin_f32, f32, ARGREDUCE_F32_PTX, "argreduce_f32_kernel", ARG_MIN);
arg_entry!(gpu_argmax_f64, f64, ARGREDUCE_F64_PTX, "argreduce_f64_kernel", ARG_MAX);
arg_entry!(gpu_argmin_f64, f64, ARGREDUCE_F64_PTX, "argreduce_f64_kernel", ARG_MIN);
arg_entry!(gpu_argmax_i32, i32, ARGREDUCE_I32_PTX, "argreduce_i32_kernel", ARG_MAX);
arg_entry!(gpu_argmin_i32, i32, ARGREDUCE_I32_PTX, "argreduce_i32_kernel", ARG_MIN);
arg_entry!(gpu_argmax_i64, i64, ARGREDUCE_I64_PTX, "argreduce_i64_kernel", ARG_MAX);
arg_entry!(gpu_argmin_i64, i64, ARGREDUCE_I64_PTX, "argreduce_i64_kernel", ARG_MIN);

/// `argmax` over an f16 (bit-pattern `u16`) value buffer.
pub fn gpu_argmax_f16(
    input: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i64>> {
    launch_argreduce(input, outer, dim_size, inner, d, ARGREDUCE_F16_PTX, "argreduce_f16_kernel", ARG_MAX)
}
/// `argmin` over an f16 (bit-pattern `u16`) value buffer.
pub fn gpu_argmin_f16(
    input: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i64>> {
    launch_argreduce(input, outer, dim_size, inner, d, ARGREDUCE_F16_PTX, "argreduce_f16_kernel", ARG_MIN)
}
/// `argmax` over a bf16 (bit-pattern `u16`) value buffer.
pub fn gpu_argmax_bf16(
    input: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i64>> {
    launch_argreduce(input, outer, dim_size, inner, d, ARGREDUCE_BF16_PTX, "argreduce_bf16_kernel", ARG_MAX)
}
/// `argmin` over a bf16 (bit-pattern `u16`) value buffer.
pub fn gpu_argmin_bf16(
    input: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i64>> {
    launch_argreduce(input, outer, dim_size, inner, d, ARGREDUCE_BF16_PTX, "argreduce_bf16_kernel", ARG_MIN)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev() -> GpuDevice {
        GpuDevice::new(0).expect("cuda device")
    }

    #[test]
    fn argmax_argmin_f32_global() {
        let d = dev();
        let h = d.stream().clone_htod(&vec![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]).unwrap();
        let mx = gpu_argmax_f32(&h, 1, 7, 1, &d).unwrap();
        let mn = gpu_argmin_f32(&h, 1, 7, 1, &d).unwrap();
        assert_eq!(d.stream().clone_dtoh(&mx).unwrap(), vec![5i64]);
        assert_eq!(d.stream().clone_dtoh(&mn).unwrap(), vec![1i64]);
    }

    #[test]
    fn argmax_f32_tie_first_index() {
        let d = dev();
        // two maxima at 0 and 3 -> first index 0
        let h = d.stream().clone_htod(&vec![5.0f32, 1.0, 2.0, 5.0]).unwrap();
        let mx = gpu_argmax_f32(&h, 1, 4, 1, &d).unwrap();
        assert_eq!(d.stream().clone_dtoh(&mx).unwrap(), vec![0i64]);
    }

    #[test]
    fn argmax_f32_along_dim() {
        let d = dev();
        // shape [2,3], argmax along dim=1 -> outer=2 dim=3 inner=1
        let h = d.stream().clone_htod(&vec![1.0f32, 9.0, 2.0, 7.0, 3.0, 4.0]).unwrap();
        let mx = gpu_argmax_f32(&h, 2, 3, 1, &d).unwrap();
        assert_eq!(d.stream().clone_dtoh(&mx).unwrap(), vec![1i64, 0i64]);
    }

    #[test]
    fn argmax_along_dim0_inner() {
        let d = dev();
        // shape [2,3], argmax along dim=0 -> outer=1 dim=2 inner=3
        let h = d.stream().clone_htod(&vec![1.0f32, 9.0, 2.0, 7.0, 3.0, 4.0]).unwrap();
        let mx = gpu_argmax_f32(&h, 1, 2, 3, &d).unwrap();
        // col0: 1 vs 7 ->1 ; col1: 9 vs 3 ->0; col2: 2 vs 4 ->1
        assert_eq!(d.stream().clone_dtoh(&mx).unwrap(), vec![1i64, 0i64, 1i64]);
    }

    #[test]
    fn argmax_i32_and_i64() {
        let d = dev();
        let hi = d.stream().clone_htod(&vec![-3i32, 7, 7, 2]).unwrap();
        let mx = gpu_argmax_i32(&hi, 1, 4, 1, &d).unwrap();
        assert_eq!(d.stream().clone_dtoh(&mx).unwrap(), vec![1i64]); // first 7
        let hl = d.stream().clone_htod(&vec![10i64, -5, 100, 100]).unwrap();
        let mn = gpu_argmin_i64(&hl, 1, 4, 1, &d).unwrap();
        assert_eq!(d.stream().clone_dtoh(&mn).unwrap(), vec![1i64]);
    }

    #[test]
    fn argmax_f16_bf16() {
        let d = dev();
        let f16bits: Vec<u16> = [1.0f32, 5.0, 2.0].iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
        let h16 = d.stream().clone_htod(&f16bits).unwrap();
        let mx = gpu_argmax_f16(&h16, 1, 3, 1, &d).unwrap();
        assert_eq!(d.stream().clone_dtoh(&mx).unwrap(), vec![1i64]);
        let bf16bits: Vec<u16> = [1.0f32, 2.0, 8.0].iter().map(|&v| half::bf16::from_f32(v).to_bits()).collect();
        let hb = d.stream().clone_htod(&bf16bits).unwrap();
        let mx2 = gpu_argmax_bf16(&hb, 1, 3, 1, &d).unwrap();
        assert_eq!(d.stream().clone_dtoh(&mx2).unwrap(), vec![2i64]);
    }
}
