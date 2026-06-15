//! Boolean / comparison GPU compute kernels — crosslink #1185 Phase 3b.
//!
//! Hand-written PTX owned by Rust (no CUDA C++, no nvrtc, no external toolchain
//! at load time), loaded via [`crate::module_cache::get_or_compile`] exactly
//! like [`crate::int_kernels`] / [`crate::f16`] / [`crate::bf16`]. Boolean
//! buffers are stored as native `CudaSlice<u8>` (cudarc `DeviceRepr` for `u8`;
//! a `bool` is one byte holding 0 or 1). The [`crate::backend_impl`] handle is
//! tagged `DType::Bool` so a u8 bool buffer is never read as an i8/u8 integer.
//!
//! # Operations
//!
//! - **Comparison** (value dtype in, u8 0/1 out): for each value dtype
//!   `f32 / f64 / bf16 / f16 / i32 / i64`, the six operators
//!   `eq / ne / lt / le / gt / ge`. bf16/f16 inputs (u16 bit patterns) are
//!   decoded to f32 first, then compared (matching the bf16/f16 elementwise
//!   kernels). The output buffer is `CudaSlice<u8>` (1 byte per element).
//!   Broadcasted i32/i64 comparisons use the rank-general broadcast comparison
//!   launcher below, matching PyTorch's TensorIterator CUDA residency.
//! - **Logical** (u8 in, u8 out): `and / or / xor` (binary), `not` (unary).
//!   Inputs are treated as "nonzero == true"; outputs are canonical 0/1.
//! - **signbit** (value dtype in, u8 0/1 out): reads the raw sign bit for
//!   `f32 / f64 / bf16`; for f16, mirrors PyTorch CUDA by returning false for
//!   all NaN payloads while preserving signed zero and finite negative values.
//! - **Reductions** to a 1-element u8 buffer: `any` (OR-reduce), `all`
//!   (AND-reduce), global. One launched thread folds all `n` elements serially
//!   (matching `int_kernels`' reduction harness), so the result equals a
//!   left-fold over the buffer.
//!
//! # PyTorch parity (rust-gpu-discipline §3)
//!
//! A comparison's result dtype is `bool` regardless of the value dtype — the
//! output is always a `DType::Bool` (u8) buffer. NaN comparisons follow IEEE:
//! `eq/lt/le/gt/ge` involving NaN are false, `ne` involving NaN is true. PTX
//! `setp.{eq,lt,le,gt,ge}.f32` are unordered-false / `setp.ne.f32` is
//! unordered-true, which is exactly the IEEE / PyTorch behaviour.
//!
//! ## REQ status (per `.design/ferrotorch-gpu/bool_kernels.md`)
//!
//! Full evidence rows (impl + non-test production consumer + upstream
//! cites) live in the design doc; this synopsis is a one-line summary per
//! REQ.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (comparison ops) | SHIPPED | `pub fn gpu_cmp_f32 / gpu_cmp_f64 / gpu_cmp_i32 / gpu_cmp_i64 / gpu_cmp_bf16 / gpu_cmp_f16 in bool_kernels.rs` thin-wrap `launch_cmp` / `launch_cmp_half` templated per (dtype, op); consumer bool-result arms of `eq/ne/lt/le/gt/ge` op handlers in `backend_impl.rs` |
//! | REQ-2 (logical binary) | SHIPPED | `pub fn gpu_and_bool / gpu_or_bool / gpu_xor_bool in bool_kernels.rs` thin-wrap `launch_logic_bin`; consumer `CudaBackendImpl::and_bool / or_bool / xor_bool in backend_impl.rs` |
//! | REQ-3 (`gpu_not_bool`) | SHIPPED | `pub fn gpu_not_bool in bool_kernels.rs`; consumer `CudaBackendImpl::not_bool in backend_impl.rs` |
//! | REQ-4 (`gpu_any_bool`/`gpu_all_bool`) | SHIPPED | `pub fn gpu_any_bool / gpu_all_bool in bool_kernels.rs`; consumer `CudaBackendImpl::any_bool / all_bool in backend_impl.rs` |
//! | REQ-5 (NaN comparison semantics) | SHIPPED | PTX `setp.{eq,lt,le,gt,ge}.f32` (unordered-false) and `setp.ne.f32` (unordered-true) inside the comparison kernels in `bool_kernels.rs`; consumer bool-comparison ops in `backend_impl.rs` rely on this for IEEE-NaN parity |
//! | REQ-6 (half-precision compare) | SHIPPED | `fn cmp_half_ptx in bool_kernels.rs` decodes bf16 via `mov.b32 %ua, {%zero16, %ha}` and f16 via `cvt.f32.f16 %fa, %ha` then `setp.{op}.f32`; consumer `pub fn gpu_cmp_bf16 / gpu_cmp_f16` invoke `launch_cmp_half` from bool-comparison arms of `backend_impl.rs` |
//! | REQ-7 (SAFETY annotations) | SHIPPED | every `unsafe { stream.launch_builder(&f)... }` in `bool_kernels.rs` (`launch_cmp`, `launch_not`, `launch_signbit`, `launch_reduce_bool`) carries a multi-line `SAFETY:` comment; consumer SAFETY contract inherited via each public wrapper |
//! | REQ-8 (empty-input short-circuit) | SHIPPED | `launch_cmp` and `launch_not` short-circuit `n == 0` via `if n == 0 { return Ok(stream.alloc_zeros::<u8>(0)?); }`; `launch_reduce_bool` short-circuits with empty-identity clone_htod; consumer backend dispatch path (`torch.any(empty)`) |
//! | REQ-9 (on-device bool broadcast, #1663) | SHIPPED | `pub fn gpu_broadcast_bool in bool_kernels.rs` (u8 strided gather over `BOOL_BROADCAST_PTX`, 8-dim unrolled); consumer `CudaBackendImpl::broadcast_bool in backend_impl.rs`, itself consumed by `grad_fns::indexing::broadcast_bool_tensor`'s CUDA branch (the path `masked_scatter` / `masked_fill_bcast` / `masked_select_bcast` / `where_cond_bcast` flow through). Mirrors `expand_outplace` at `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2406`. |
//! | REQ-10 (`signbit`) | SHIPPED | `pub fn gpu_signbit_f32 / gpu_signbit_f64 / gpu_signbit_f16 / gpu_signbit_bf16 in bool_kernels.rs`; consumer `CudaBackendImpl::signbit_mask` and `grad_fns::transcendental::signbit` CUDA branch. |
//! | REQ-11 (integer broadcast comparison) | SHIPPED | `pub fn gpu_cmp_broadcast_i32 / gpu_cmp_broadcast_i64 in bool_kernels.rs` use rank-general shape/stride metadata buffers; consumer `CudaBackendImpl::compare_broadcast`, so `BoolTensor::compare_int` keeps broadcasted integer operands and bool outputs CUDA-resident. |

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits};

use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};
use crate::module_cache::{get_or_compile, get_or_compile_owned};

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
// PTX builders.
//
// A comparison kernel reads `a[i]`, `b[i]` (each `IN_BYTES` wide), computes a
// predicate, and writes `out[i]` as one u8 (0 or 1). `in_off = i << IN_SHIFT`,
// `out_off = i` (1 byte per bool). All bound-checked against `n`.
//
// Rather than hand-write 36 near-identical comparison kernels, we generate the
// PTX as owned `String`s at module-load time (once, cached by `get_or_compile`
// keyed on the kernel name). The body differs only in: the load type, the
// `setp` form, and the input element shift.
// ===========================================================================

/// PTX prologue computing `%idx` (global thread id) and bound-checking it
/// against the `n` param, branching to `DONE` when out of range. Shared by the
/// float and integer comparison kernels.
fn cmp_ptx(
    kernel_name: &str,
    in_shift: u32,  // log2(IN_BYTES): f32/i32→2, f64/i64→3
    load_ty: &str,  // "f32" | "f64" | "s32" | "s64"
    reg_decl: &str, // register decls for the value regs
    setp: &str,     // e.g. "setp.lt.f32 %c, %va, %vb;"
) -> String {
    format!(
        "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry {kernel_name}(
    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 out_ptr, .param .u32 n
) {{
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %b, %out, %ioff, %ooff;
    {reg_decl}
    .reg .u16 %res;
    .reg .pred %p, %c;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %nr, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr;
    @%p bra DONE;

    cvt.u64.u32 %ioff, %idx;
    shl.b64 %ioff, %ioff, {in_shift};
    add.u64 %a, %a, %ioff;
    add.u64 %b, %b, %ioff;
    // output is 1 byte per element: out_off = idx
    cvt.u64.u32 %ooff, %idx;
    add.u64 %out, %out, %ooff;

    ld.global.{load_ty} %va, [%a];
    ld.global.{load_ty} %vb, [%b];
    {setp}
    selp.u16 %res, 1, 0, %c;
    st.global.u8 [%out], %res;
DONE:
    ret;
}}
"
    )
}

/// PTX for rank-general broadcast comparison over i32/i64 buffers.
///
/// Each thread maps its flat output index through `out_shape` and per-input
/// broadcast strides to load one element from `a` and `b`, then writes one bool
/// byte. Strides are element strides, not byte strides; `in_shift` selects the
/// byte shift for the value dtype.
fn cmp_broadcast_ptx(
    kernel_name: &str,
    in_shift: u32,  // log2(IN_BYTES): i32→2, i64→3
    load_ty: &str,  // "s32" | "s64"
    reg_decl: &str, // register decls for the value regs
    setp: &str,     // e.g. "setp.lt.s32 %c, %va, %vb;"
) -> String {
    format!(
        "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry {kernel_name}(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u64 a_strides_ptr,
    .param .u64 b_strides_ptr,
    .param .u64 out_shape_ptr,
    .param .u32 n,
    .param .u32 ndim
) {{
    .reg .u32 %idx, %bid, %bdim, %nr, %ndim_r;
    .reg .u32 %remaining, %a_idx, %b_idx, %d;
    .reg .u32 %shape_d, %a_str_d, %b_str_d, %coord;
    .reg .u64 %a, %b, %out, %a_str, %b_str, %oshape;
    .reg .u64 %off_a, %off_b, %off_out, %d64, %tmp;
    {reg_decl}
    .reg .u16 %res;
    .reg .pred %p, %loop_p, %c;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %a_str, [a_strides_ptr];
    ld.param.u64 %b_str, [b_strides_ptr];
    ld.param.u64 %oshape, [out_shape_ptr];
    ld.param.u32 %nr, [n];
    ld.param.u32 %ndim_r, [ndim];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr;
    @%p bra DONE;

    mov.u32 %remaining, %idx;
    mov.u32 %a_idx, 0;
    mov.u32 %b_idx, 0;
    mov.u32 %d, %ndim_r;

LOOP:
    setp.eq.u32 %loop_p, %d, 0;
    @%loop_p bra END_LOOP;
    sub.u32 %d, %d, 1;

    cvt.u64.u32 %d64, %d;
    shl.b64 %d64, %d64, 2;

    add.u64 %tmp, %oshape, %d64;
    ld.global.u32 %shape_d, [%tmp];
    add.u64 %tmp, %a_str, %d64;
    ld.global.u32 %a_str_d, [%tmp];
    add.u64 %tmp, %b_str, %d64;
    ld.global.u32 %b_str_d, [%tmp];

    rem.u32 %coord, %remaining, %shape_d;
    div.u32 %remaining, %remaining, %shape_d;
    mad.lo.u32 %a_idx, %coord, %a_str_d, %a_idx;
    mad.lo.u32 %b_idx, %coord, %b_str_d, %b_idx;

    bra LOOP;
END_LOOP:

    cvt.u64.u32 %off_a, %a_idx;
    shl.b64 %off_a, %off_a, {in_shift};
    add.u64 %off_a, %a, %off_a;
    ld.global.{load_ty} %va, [%off_a];

    cvt.u64.u32 %off_b, %b_idx;
    shl.b64 %off_b, %off_b, {in_shift};
    add.u64 %off_b, %b, %off_b;
    ld.global.{load_ty} %vb, [%off_b];

    {setp}
    selp.u16 %res, 1, 0, %c;

    cvt.u64.u32 %off_out, %idx;
    add.u64 %off_out, %out, %off_out;
    st.global.u8 [%off_out], %res;
DONE:
    ret;
}}
"
    )
}

/// PTX for a comparison whose value type is bf16 or f16 (a u16 bit pattern).
/// The two halves are loaded as `.b16` and decoded to f32 (`decode`), then
/// compared in f32. `target` is the required `.target` line (sm_53 for the
/// f16 `cvt.f32.f16`, sm_80 for the bf16 splat path — both supported by the
/// host RTX 3090, sm_86).
fn cmp_half_ptx(
    kernel_name: &str,
    target: &str, // e.g. "sm_53" | "sm_80"
    decode: &str, // PTX decoding %ha (b16) → %fa (f32) and %hb → %fb
    setp: &str,   // e.g. "setp.lt.f32 %c, %fa, %fb;"
) -> String {
    format!(
        "\
.version 7.0
.target {target}
.address_size 64

.visible .entry {kernel_name}(
    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 out_ptr, .param .u32 n
) {{
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %b, %out, %ioff, %ooff;
    .reg .b16 %ha, %hb, %zero16;
    .reg .b32 %ua, %ub;
    .reg .u16 %res;
    .reg .f32 %fa, %fb;
    .reg .pred %p, %c;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %nr, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr;
    @%p bra DONE;

    cvt.u64.u32 %ioff, %idx;
    shl.b64 %ioff, %ioff, 1;
    add.u64 %a, %a, %ioff;
    add.u64 %b, %b, %ioff;
    cvt.u64.u32 %ooff, %idx;
    add.u64 %out, %out, %ooff;

    mov.b16 %zero16, 0;
    ld.global.b16 %ha, [%a];
    ld.global.b16 %hb, [%b];
    {decode}
    {setp}
    selp.u16 %res, 1, 0, %c;
    st.global.u8 [%out], %res;
DONE:
    ret;
}}
"
    )
}

// bf16 decode: a bf16 is the high 16 bits of an f32. Compose a b32 from
// {low=0, high=bf16_bits} and reinterpret as f32 (mirrors crate::bf16's
// `mov.b32 %u, {%zero16, %b16}; mov.b32 %f, %u` pattern).
const BF16_DECODE: &str = "\
    mov.b32 %ua, {%zero16, %ha}; mov.b32 %fa, %ua;
    mov.b32 %ub, {%zero16, %hb}; mov.b32 %fb, %ub;";
// f16 decode: hardware convert IEEE half → f32 (cvt.f32.f16, sm_53+).
const F16_DECODE: &str = "\
    cvt.f32.f16 %fa, %ha;
    cvt.f32.f16 %fb, %hb;";

/// PTX for a logical binary op (`and`/`or`/`xor`) over u8 bool buffers.
/// `nonzero(a) OP nonzero(b)` → canonical 0/1.
fn logic_bin_ptx(kernel_name: &str, op: &str /* "and"|"or"|"xor" */) -> String {
    format!(
        "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry {kernel_name}(
    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 out_ptr, .param .u32 n
) {{
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %b, %out, %off;
    .reg .u16 %va, %vb, %res;
    .reg .pred %pa, %pb, %pr, %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %nr, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr;
    @%p bra DONE;

    cvt.u64.u32 %off, %idx;
    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.u8 %va, [%a];
    ld.global.u8 %vb, [%b];
    setp.ne.u16 %pa, %va, 0;
    setp.ne.u16 %pb, %vb, 0;
    {op}.pred %pr, %pa, %pb;
    selp.u16 %res, 1, 0, %pr;
    st.global.u8 [%out], %res;
DONE:
    ret;
}}
"
    )
}

const NOT_BOOL_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry not_bool_kernel(
    .param .u64 a_ptr, .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %out, %off;
    .reg .u16 %va, %res;
    .reg .pred %pa, %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %nr, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr;
    @%p bra DONE;

    cvt.u64.u32 %off, %idx;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.u8 %va, [%a];
    // res = (va == 0) ? 1 : 0
    setp.eq.u16 %pa, %va, 0;
    selp.u16 %res, 1, 0, %pa;
    st.global.u8 [%out], %res;
DONE:
    ret;
}
";

const SIGNBIT_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry signbit_f32_kernel(
    .param .u64 a_ptr, .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr, %bits, %sign;
    .reg .u64 %a, %out, %ioff, %ooff;
    .reg .u16 %res;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %nr, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr;
    @%p bra DONE;

    cvt.u64.u32 %ioff, %idx;
    shl.b64 %ioff, %ioff, 2;
    add.u64 %a, %a, %ioff;
    cvt.u64.u32 %ooff, %idx;
    add.u64 %out, %out, %ooff;

    ld.global.u32 %bits, [%a];
    shr.u32 %sign, %bits, 31;
    cvt.u16.u32 %res, %sign;
    st.global.u8 [%out], %res;
DONE:
    ret;
}
";

const SIGNBIT_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry signbit_f64_kernel(
    .param .u64 a_ptr, .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %out, %ioff, %ooff, %bits, %sign;
    .reg .u16 %res;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %nr, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr;
    @%p bra DONE;

    cvt.u64.u32 %ioff, %idx;
    shl.b64 %ioff, %ioff, 3;
    add.u64 %a, %a, %ioff;
    cvt.u64.u32 %ooff, %idx;
    add.u64 %out, %out, %ooff;

    ld.global.u64 %bits, [%a];
    shr.u64 %sign, %bits, 63;
    cvt.u16.u64 %res, %sign;
    st.global.u8 [%out], %res;
DONE:
    ret;
}
";

const SIGNBIT_F16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry signbit_f16_kernel(
    .param .u64 a_ptr, .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr, %bits, %exp, %mant, %sign;
    .reg .u64 %a, %out, %ioff, %ooff;
    .reg .u16 %h, %res;
    .reg .pred %p, %exp_all, %mant_nz, %is_nan;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %nr, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr;
    @%p bra DONE;

    cvt.u64.u32 %ioff, %idx;
    shl.b64 %ioff, %ioff, 1;
    add.u64 %a, %a, %ioff;
    cvt.u64.u32 %ooff, %idx;
    add.u64 %out, %out, %ooff;

    ld.global.u16 %h, [%a];
    cvt.u32.u16 %bits, %h;
    and.b32 %exp, %bits, 0x7C00;
    and.b32 %mant, %bits, 0x03FF;
    setp.eq.u32 %exp_all, %exp, 0x7C00;
    setp.ne.u32 %mant_nz, %mant, 0;
    and.pred %is_nan, %exp_all, %mant_nz;
    shr.u32 %sign, %bits, 15;
    cvt.u16.u32 %res, %sign;
    selp.u16 %res, 0, %res, %is_nan;
    st.global.u8 [%out], %res;
DONE:
    ret;
}
";

const SIGNBIT_BF16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry signbit_bf16_kernel(
    .param .u64 a_ptr, .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr, %bits, %sign;
    .reg .u64 %a, %out, %ioff, %ooff;
    .reg .u16 %h, %res;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %nr, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr;
    @%p bra DONE;

    cvt.u64.u32 %ioff, %idx;
    shl.b64 %ioff, %ioff, 1;
    add.u64 %a, %a, %ioff;
    cvt.u64.u32 %ooff, %idx;
    add.u64 %out, %out, %ooff;

    ld.global.u16 %h, [%a];
    cvt.u32.u16 %bits, %h;
    shr.u32 %sign, %bits, 15;
    cvt.u16.u32 %res, %sign;
    st.global.u8 [%out], %res;
DONE:
    ret;
}
";

// Reduction: one launched thread (thread 0) folds all n bytes. `op`: 0 = any
// (OR-reduce, identity 0), 1 = all (AND-reduce, identity 1). Each input byte is
// normalised to 0/1 (nonzero → 1). Output is one u8 (0/1). The host guards the
// n == 0 case (any of empty = false, all of empty = true) before launching.
const REDUCE_BOOL_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry reduce_bool_kernel(
    .param .u64 a_ptr, .param .u64 out_ptr, .param .u32 n, .param .u32 op
) {
    .reg .u32 %idx, %bid, %bdim, %nr, %op_r, %i;
    .reg .u64 %a, %out, %off, %cur;
    .reg .u16 %acc, %v, %vn;
    .reg .pred %only0, %p, %is_any, %pacc, %pv;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %nr, [n];
    ld.param.u32 %op_r, [op];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ne.u32 %only0, %idx, 0;
    @%only0 bra DONE;

    setp.eq.u32 %is_any, %op_r, 0;

    // Initialise accumulator from a[0] normalised to 0/1 (n >= 1 guaranteed).
    ld.global.u8 %v, [%a];
    setp.ne.u16 %pv, %v, 0;
    selp.u16 %acc, 1, 0, %pv;
    mov.u32 %i, 1;
LOOP:
    setp.ge.u32 %p, %i, %nr;
    @%p bra STORE;

    cvt.u64.u32 %off, %i;
    add.u64 %cur, %a, %off;
    ld.global.u8 %v, [%cur];
    setp.ne.u16 %pv, %v, 0;
    selp.u16 %vn, 1, 0, %pv;

    setp.ne.u16 %pacc, %acc, 0;
    setp.ne.u16 %pv, %vn, 0;
    // any: acc = acc OR v ; all: acc = acc AND v
    @%is_any or.pred %pacc, %pacc, %pv;
    @!%is_any and.pred %pacc, %pacc, %pv;
    selp.u16 %acc, 1, 0, %pacc;

    add.u32 %i, %i, 1;
    bra LOOP;
STORE:
    st.global.u8 [%out], %acc;
DONE:
    ret;
}
";

const REDUCE_ANY: u32 = 0;
const REDUCE_ALL: u32 = 1;

// Float-value reductions for `torch.any`, `torch.all`, and
// `torch.count_nonzero`. Logical input layout is `[outer, dim_size, inner]`;
// one thread folds one `(outer, inner)` slice. NaN is nonzero because PTX
// `setp.ne.f{32,64}` is unordered-true, matching `NaN != 0` in PyTorch.
const REDUCE_FLOAT_F32_BOOL_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry reduce_float_f32_bool_kernel(
    .param .u64 in_ptr, .param .u64 out_ptr,
    .param .u32 outer, .param .u32 dim_size, .param .u32 inner,
    .param .u32 total, .param .u32 op
) {
    .reg .u32 %gtid, %bid, %bdim, %tot, %dim, %inn, %op_r;
    .reg .u32 %oidx, %iidx, %base, %j, %elem;
    .reg .u64 %in, %out, %off, %addr;
    .reg .f32 %v;
    .reg .u16 %acc;
    .reg .pred %p, %is_any, %nz, %nan, %accp;

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

    setp.eq.u32 %is_any, %op_r, 0;
    selp.u16 %acc, 0, 1, %is_any;

    div.u32 %oidx, %gtid, %inn;
    rem.u32 %iidx, %gtid, %inn;
    mul.lo.u32 %base, %oidx, %dim;
    mul.lo.u32 %base, %base, %inn;
    add.u32 %base, %base, %iidx;

    mov.u32 %j, 0;
LOOP:
    setp.ge.u32 %p, %j, %dim;
    @%p bra STORE;

    mul.lo.u32 %elem, %j, %inn;
    add.u32 %elem, %elem, %base;
    cvt.u64.u32 %off, %elem;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in, %off;
    ld.global.f32 %v, [%addr];
    setp.ne.f32 %nz, %v, 0f00000000;
    setp.nan.f32 %nan, %v, %v;
    or.pred %nz, %nz, %nan;

    setp.ne.u16 %accp, %acc, 0;
    @%is_any or.pred %accp, %accp, %nz;
    @!%is_any and.pred %accp, %accp, %nz;
    selp.u16 %acc, 1, 0, %accp;

    add.u32 %j, %j, 1;
    bra LOOP;
STORE:
    cvt.u64.u32 %off, %gtid;
    add.u64 %addr, %out, %off;
    st.global.u8 [%addr], %acc;
DONE:
    ret;
}
";

const REDUCE_FLOAT_F64_BOOL_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry reduce_float_f64_bool_kernel(
    .param .u64 in_ptr, .param .u64 out_ptr,
    .param .u32 outer, .param .u32 dim_size, .param .u32 inner,
    .param .u32 total, .param .u32 op
) {
    .reg .u32 %gtid, %bid, %bdim, %tot, %dim, %inn, %op_r;
    .reg .u32 %oidx, %iidx, %base, %j, %elem;
    .reg .u64 %in, %out, %off, %addr;
    .reg .f64 %v;
    .reg .u16 %acc;
    .reg .pred %p, %is_any, %nz, %nan, %accp;

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

    setp.eq.u32 %is_any, %op_r, 0;
    selp.u16 %acc, 0, 1, %is_any;

    div.u32 %oidx, %gtid, %inn;
    rem.u32 %iidx, %gtid, %inn;
    mul.lo.u32 %base, %oidx, %dim;
    mul.lo.u32 %base, %base, %inn;
    add.u32 %base, %base, %iidx;

    mov.u32 %j, 0;
LOOP:
    setp.ge.u32 %p, %j, %dim;
    @%p bra STORE;

    mul.lo.u32 %elem, %j, %inn;
    add.u32 %elem, %elem, %base;
    cvt.u64.u32 %off, %elem;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %in, %off;
    ld.global.f64 %v, [%addr];
    setp.ne.f64 %nz, %v, 0d0000000000000000;
    setp.nan.f64 %nan, %v, %v;
    or.pred %nz, %nz, %nan;

    setp.ne.u16 %accp, %acc, 0;
    @%is_any or.pred %accp, %accp, %nz;
    @!%is_any and.pred %accp, %accp, %nz;
    selp.u16 %acc, 1, 0, %accp;

    add.u32 %j, %j, 1;
    bra LOOP;
STORE:
    cvt.u64.u32 %off, %gtid;
    add.u64 %addr, %out, %off;
    st.global.u8 [%addr], %acc;
DONE:
    ret;
}
";

const REDUCE_FLOAT_F16_BOOL_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry reduce_float_f16_bool_kernel(
    .param .u64 in_ptr, .param .u64 out_ptr,
    .param .u32 outer, .param .u32 dim_size, .param .u32 inner,
    .param .u32 total, .param .u32 op
) {
    .reg .u32 %gtid, %bid, %bdim, %tot, %dim, %inn, %op_r;
    .reg .u32 %oidx, %iidx, %base, %j, %elem;
    .reg .u64 %in, %out, %off, %addr;
    .reg .b16 %h;
    .reg .f32 %v;
    .reg .u16 %acc;
    .reg .pred %p, %is_any, %nz, %nan, %accp;

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

    setp.eq.u32 %is_any, %op_r, 0;
    selp.u16 %acc, 0, 1, %is_any;

    div.u32 %oidx, %gtid, %inn;
    rem.u32 %iidx, %gtid, %inn;
    mul.lo.u32 %base, %oidx, %dim;
    mul.lo.u32 %base, %base, %inn;
    add.u32 %base, %base, %iidx;

    mov.u32 %j, 0;
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
    setp.ne.f32 %nz, %v, 0f00000000;
    setp.nan.f32 %nan, %v, %v;
    or.pred %nz, %nz, %nan;

    setp.ne.u16 %accp, %acc, 0;
    @%is_any or.pred %accp, %accp, %nz;
    @!%is_any and.pred %accp, %accp, %nz;
    selp.u16 %acc, 1, 0, %accp;

    add.u32 %j, %j, 1;
    bra LOOP;
STORE:
    cvt.u64.u32 %off, %gtid;
    add.u64 %addr, %out, %off;
    st.global.u8 [%addr], %acc;
DONE:
    ret;
}
";

const REDUCE_FLOAT_BF16_BOOL_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry reduce_float_bf16_bool_kernel(
    .param .u64 in_ptr, .param .u64 out_ptr,
    .param .u32 outer, .param .u32 dim_size, .param .u32 inner,
    .param .u32 total, .param .u32 op
) {
    .reg .u32 %gtid, %bid, %bdim, %tot, %dim, %inn, %op_r;
    .reg .u32 %oidx, %iidx, %base, %j, %elem, %bits;
    .reg .u16 %h;
    .reg .u64 %in, %out, %off, %addr;
    .reg .f32 %v;
    .reg .u16 %acc;
    .reg .pred %p, %is_any, %nz, %nan, %accp;

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

    setp.eq.u32 %is_any, %op_r, 0;
    selp.u16 %acc, 0, 1, %is_any;

    div.u32 %oidx, %gtid, %inn;
    rem.u32 %iidx, %gtid, %inn;
    mul.lo.u32 %base, %oidx, %dim;
    mul.lo.u32 %base, %base, %inn;
    add.u32 %base, %base, %iidx;

    mov.u32 %j, 0;
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
    setp.ne.f32 %nz, %v, 0f00000000;
    setp.nan.f32 %nan, %v, %v;
    or.pred %nz, %nz, %nan;

    setp.ne.u16 %accp, %acc, 0;
    @%is_any or.pred %accp, %accp, %nz;
    @!%is_any and.pred %accp, %accp, %nz;
    selp.u16 %acc, 1, 0, %accp;

    add.u32 %j, %j, 1;
    bra LOOP;
STORE:
    cvt.u64.u32 %off, %gtid;
    add.u64 %addr, %out, %off;
    st.global.u8 [%addr], %acc;
DONE:
    ret;
}
";

const REDUCE_FLOAT_F32_COUNT_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry reduce_float_f32_count_kernel(
    .param .u64 in_ptr, .param .u64 out_ptr,
    .param .u32 outer, .param .u32 dim_size, .param .u32 inner,
    .param .u32 total
) {
    .reg .u32 %gtid, %bid, %bdim, %tot, %dim, %inn;
    .reg .u32 %oidx, %iidx, %base, %j, %elem;
    .reg .u64 %in, %out, %off, %addr;
    .reg .f32 %v;
    .reg .s64 %count;
    .reg .pred %p, %nz, %nan;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %tot, [total];
    ld.param.u32 %dim, [dim_size];
    ld.param.u32 %inn, [inner];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %gtid, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %gtid;
    setp.ge.u32 %p, %gtid, %tot;
    @%p bra DONE;

    div.u32 %oidx, %gtid, %inn;
    rem.u32 %iidx, %gtid, %inn;
    mul.lo.u32 %base, %oidx, %dim;
    mul.lo.u32 %base, %base, %inn;
    add.u32 %base, %base, %iidx;

    mov.s64 %count, 0;
    mov.u32 %j, 0;
LOOP:
    setp.ge.u32 %p, %j, %dim;
    @%p bra STORE;

    mul.lo.u32 %elem, %j, %inn;
    add.u32 %elem, %elem, %base;
    cvt.u64.u32 %off, %elem;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in, %off;
    ld.global.f32 %v, [%addr];
    setp.ne.f32 %nz, %v, 0f00000000;
    setp.nan.f32 %nan, %v, %v;
    or.pred %nz, %nz, %nan;
    @%nz add.s64 %count, %count, 1;

    add.u32 %j, %j, 1;
    bra LOOP;
STORE:
    cvt.u64.u32 %off, %gtid;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %out, %off;
    st.global.s64 [%addr], %count;
DONE:
    ret;
}
";

const REDUCE_FLOAT_F64_COUNT_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry reduce_float_f64_count_kernel(
    .param .u64 in_ptr, .param .u64 out_ptr,
    .param .u32 outer, .param .u32 dim_size, .param .u32 inner,
    .param .u32 total
) {
    .reg .u32 %gtid, %bid, %bdim, %tot, %dim, %inn;
    .reg .u32 %oidx, %iidx, %base, %j, %elem;
    .reg .u64 %in, %out, %off, %addr;
    .reg .f64 %v;
    .reg .s64 %count;
    .reg .pred %p, %nz, %nan;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %tot, [total];
    ld.param.u32 %dim, [dim_size];
    ld.param.u32 %inn, [inner];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %gtid, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %gtid;
    setp.ge.u32 %p, %gtid, %tot;
    @%p bra DONE;

    div.u32 %oidx, %gtid, %inn;
    rem.u32 %iidx, %gtid, %inn;
    mul.lo.u32 %base, %oidx, %dim;
    mul.lo.u32 %base, %base, %inn;
    add.u32 %base, %base, %iidx;

    mov.s64 %count, 0;
    mov.u32 %j, 0;
LOOP:
    setp.ge.u32 %p, %j, %dim;
    @%p bra STORE;

    mul.lo.u32 %elem, %j, %inn;
    add.u32 %elem, %elem, %base;
    cvt.u64.u32 %off, %elem;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %in, %off;
    ld.global.f64 %v, [%addr];
    setp.ne.f64 %nz, %v, 0d0000000000000000;
    setp.nan.f64 %nan, %v, %v;
    or.pred %nz, %nz, %nan;
    @%nz add.s64 %count, %count, 1;

    add.u32 %j, %j, 1;
    bra LOOP;
STORE:
    cvt.u64.u32 %off, %gtid;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %out, %off;
    st.global.s64 [%addr], %count;
DONE:
    ret;
}
";

const REDUCE_FLOAT_F16_COUNT_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry reduce_float_f16_count_kernel(
    .param .u64 in_ptr, .param .u64 out_ptr,
    .param .u32 outer, .param .u32 dim_size, .param .u32 inner,
    .param .u32 total
) {
    .reg .u32 %gtid, %bid, %bdim, %tot, %dim, %inn;
    .reg .u32 %oidx, %iidx, %base, %j, %elem;
    .reg .u64 %in, %out, %off, %addr;
    .reg .b16 %h;
    .reg .f32 %v;
    .reg .s64 %count;
    .reg .pred %p, %nz, %nan;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %tot, [total];
    ld.param.u32 %dim, [dim_size];
    ld.param.u32 %inn, [inner];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %gtid, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %gtid;
    setp.ge.u32 %p, %gtid, %tot;
    @%p bra DONE;

    div.u32 %oidx, %gtid, %inn;
    rem.u32 %iidx, %gtid, %inn;
    mul.lo.u32 %base, %oidx, %dim;
    mul.lo.u32 %base, %base, %inn;
    add.u32 %base, %base, %iidx;

    mov.s64 %count, 0;
    mov.u32 %j, 0;
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
    setp.ne.f32 %nz, %v, 0f00000000;
    setp.nan.f32 %nan, %v, %v;
    or.pred %nz, %nz, %nan;
    @%nz add.s64 %count, %count, 1;

    add.u32 %j, %j, 1;
    bra LOOP;
STORE:
    cvt.u64.u32 %off, %gtid;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %out, %off;
    st.global.s64 [%addr], %count;
DONE:
    ret;
}
";

const REDUCE_FLOAT_BF16_COUNT_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry reduce_float_bf16_count_kernel(
    .param .u64 in_ptr, .param .u64 out_ptr,
    .param .u32 outer, .param .u32 dim_size, .param .u32 inner,
    .param .u32 total
) {
    .reg .u32 %gtid, %bid, %bdim, %tot, %dim, %inn;
    .reg .u32 %oidx, %iidx, %base, %j, %elem, %bits;
    .reg .u16 %h;
    .reg .u64 %in, %out, %off, %addr;
    .reg .f32 %v;
    .reg .s64 %count;
    .reg .pred %p, %nz, %nan;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %tot, [total];
    ld.param.u32 %dim, [dim_size];
    ld.param.u32 %inn, [inner];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %gtid, %tid.x;
    mad.lo.u32 %gtid, %bid, %bdim, %gtid;
    setp.ge.u32 %p, %gtid, %tot;
    @%p bra DONE;

    div.u32 %oidx, %gtid, %inn;
    rem.u32 %iidx, %gtid, %inn;
    mul.lo.u32 %base, %oidx, %dim;
    mul.lo.u32 %base, %base, %inn;
    add.u32 %base, %base, %iidx;

    mov.s64 %count, 0;
    mov.u32 %j, 0;
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
    setp.ne.f32 %nz, %v, 0f00000000;
    setp.nan.f32 %nan, %v, %v;
    or.pred %nz, %nz, %nan;
    @%nz add.s64 %count, %count, 1;

    add.u32 %j, %j, 1;
    bra LOOP;
STORE:
    cvt.u64.u32 %off, %gtid;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %out, %off;
    st.global.s64 [%addr], %count;
DONE:
    ret;
}
";

// ===========================================================================
// Launch harness
// ===========================================================================

/// Launch a comparison `(a_ptr, b_ptr, out_ptr, n)` kernel over a value buffer
/// of native element type `T` (`f32`/`f64`/`i32`/`i64`), producing a fresh
/// `CudaSlice<u8>` of `n` 0/1 bytes resident on `device`.
///
/// `n` is the LOGICAL element count of the operands (`CudaBuffer::len()`), not
/// the raw `CudaSlice::len()`. The raw slices may be OVER-ALLOCATED past `n`:
/// a `.contiguous()`-materialised view is backed by a pooled buffer whose raw
/// `CudaSlice::len()` is rounded up to a multiple of `ROUND_ELEMENTS` (#1660),
/// whereas a `clone_htod` operand is backed by an exact-length slice. We must
/// therefore validate and launch on the logical `n`, treating each raw slice as
/// a backing store that need only be `>= n`; comparing raw lens would spuriously
/// reject `256 vs 6`. The caller (dispatch site) supplies `n` from the logical
/// buffer len and is responsible for the operand-shape equality check.
fn launch_cmp<T: DeviceRepr + ValidAsZeroBits>(
    a: &CudaSlice<T>,
    b: &CudaSlice<T>,
    n: usize,
    device: &GpuDevice,
    ptx: String,
    kernel_name: String,
    err_label: &'static str,
) -> GpuResult<CudaSlice<u8>> {
    if a.len() < n || b.len() < n {
        return Err(GpuError::LengthMismatch {
            a: a.len().min(b.len()),
            b: n,
        });
    }
    let stream = device.stream();
    if n == 0 {
        return Ok(stream.alloc_zeros::<u8>(0)?);
    }
    let ctx = device.context();
    // The comparison PTX is built at runtime (one variant per value-dtype ×
    // operator), so use the owned-string cache, which hashes the PTX, compiles
    // once, and reuses thereafter. `kernel_name` is the entry-point name inside
    // that PTX. `err_label` is a `'static` family name for error reporting only.
    let f = get_or_compile_owned(ctx, ptx, kernel_name, device.ordinal() as u32).map_err(|e| {
        GpuError::PtxCompileFailed {
            kernel: err_label,
            source: e,
        }
    })?;
    let mut out = stream.alloc_zeros::<u8>(n)?;
    let cfg = launch_1d(n);
    let n_u32 = n as u32;
    // SAFETY:
    // - `f` is the PTX entry `kernel_name` just compiled; its signature is
    //   (a_ptr: u64, b_ptr: u64, out_ptr: u64, n: u32), matching the four args
    //   pushed below in order.
    // - `a`, `b` are immutable input buffers backing AT LEAST `n` `T`-elements
    //   each (`a.len() >= n` / `b.len() >= n` enforced above; either may be a
    //   pooled, over-allocated `.contiguous()` materialisation, #1660); the
    //   kernel reads only `[0, n)`.
    // - `out` was alloc'd `n` `u8`-elements from `stream` and is the only `&mut`
    //   here, non-aliased with the immutable inputs (distinct allocations).
    // - The kernel reads `a[i]`/`b[i]` and writes `out[i]` only within `[0, n)`
    //   per the `setp.ge.u32 %p, %idx, %nr` bound check.
    // - `n_u32` is non-truncating: `launch_1d` already cast `n as u32` to size
    //   the grid covering `[0, n)`.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a)
            .arg(b)
            .arg(&mut out)
            .arg(&n_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

fn checked_product_u32(shape: &[usize], op: &'static str) -> GpuResult<usize> {
    let n = if shape.is_empty() {
        1
    } else {
        shape.iter().try_fold(1usize, |acc, &d| {
            acc.checked_mul(d).ok_or(GpuError::ShapeMismatch {
                op,
                expected: vec![usize::MAX],
                got: shape.to_vec(),
            })
        })?
    };
    if n > u32::MAX as usize {
        return Err(GpuError::ShapeMismatch {
            op,
            expected: vec![u32::MAX as usize],
            got: vec![n],
        });
    }
    Ok(n)
}

fn checked_u32_vec(values: &[usize], op: &'static str) -> GpuResult<Vec<u32>> {
    values
        .iter()
        .map(|&v| {
            u32::try_from(v).map_err(|_| GpuError::ShapeMismatch {
                op,
                expected: vec![u32::MAX as usize],
                got: vec![v],
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn launch_cmp_broadcast<T: DeviceRepr + ValidAsZeroBits>(
    a: &CudaSlice<T>,
    b: &CudaSlice<T>,
    a_numel: usize,
    b_numel: usize,
    out_shape: &[usize],
    a_src_strides: &[usize],
    b_src_strides: &[usize],
    device: &GpuDevice,
    ptx: String,
    kernel_name: String,
    err_label: &'static str,
) -> GpuResult<CudaSlice<u8>> {
    if out_shape.len() != a_src_strides.len() || out_shape.len() != b_src_strides.len() {
        return Err(GpuError::ShapeMismatch {
            op: err_label,
            expected: vec![out_shape.len()],
            got: vec![a_src_strides.len(), b_src_strides.len()],
        });
    }
    if a.len() != a_numel {
        return Err(GpuError::LengthMismatch {
            a: a.len(),
            b: a_numel,
        });
    }
    if b.len() != b_numel {
        return Err(GpuError::LengthMismatch {
            a: b.len(),
            b: b_numel,
        });
    }
    let n = checked_product_u32(out_shape, err_label)?;
    let ndim_u32 = u32::try_from(out_shape.len()).map_err(|_| GpuError::ShapeMismatch {
        op: err_label,
        expected: vec![u32::MAX as usize],
        got: vec![out_shape.len()],
    })?;
    let stream = device.stream();
    if n == 0 {
        return Ok(stream.alloc_zeros::<u8>(0)?);
    }
    let out_shape_u32 = checked_u32_vec(out_shape, err_label)?;
    let a_stride_u32 = checked_u32_vec(a_src_strides, err_label)?;
    let b_stride_u32 = checked_u32_vec(b_src_strides, err_label)?;

    let ctx = device.context();
    let f = get_or_compile_owned(ctx, ptx, kernel_name, device.ordinal() as u32).map_err(|e| {
        GpuError::PtxCompileFailed {
            kernel: err_label,
            source: e,
        }
    })?;
    let a_str_buf = stream.clone_htod(&a_stride_u32)?;
    let b_str_buf = stream.clone_htod(&b_stride_u32)?;
    let shape_buf = stream.clone_htod(&out_shape_u32)?;
    let mut out = stream.alloc_zeros::<u8>(n)?;
    let cfg = launch_1d(n);
    let n_u32 = n as u32;
    // SAFETY:
    // - `f` is the broadcast-comparison PTX entry compiled from `ptx`; its ABI
    //   is `(a, b, out, a_strides, b_strides, out_shape, n, ndim)`, matching
    //   the eight arguments below.
    // - `a_str_buf`, `b_str_buf`, and `shape_buf` are fresh u32 device buffers
    //   with length `ndim`. The kernel reads exactly `ndim` entries from each.
    // - `a` and `b` exactly back their logical shape products
    //   (`a_numel`/`b_numel`). The stride arrays are produced from valid
    //   PyTorch broadcast rules by the caller, so each collapsed source offset
    //   stays within `[0, a_numel)` or `[0, b_numel)`.
    // - `out` is freshly allocated for `n` bool bytes and is the only mutable
    //   buffer passed to the launch.
    // - The PTX checks `tid >= n` before any memory access.
    // - `n_u32` and every metadata entry are checked above before narrowing.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a)
            .arg(b)
            .arg(&mut out)
            .arg(&a_str_buf)
            .arg(&b_str_buf)
            .arg(&shape_buf)
            .arg(&n_u32)
            .arg(&ndim_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Launch a comparison over a half-precision (u16 bit-pattern) value buffer.
/// Identical contract to [`launch_cmp`], specialised to `CudaSlice<u16>`.
fn launch_cmp_half(
    a: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    n: usize,
    device: &GpuDevice,
    ptx: String,
    kernel_name: String,
) -> GpuResult<CudaSlice<u8>> {
    launch_cmp::<u16>(a, b, n, device, ptx, kernel_name, "cmp_half")
}

/// Launch a logical binary `(a_ptr, b_ptr, out_ptr, n)` kernel over two u8 bool
/// buffers, producing a fresh `CudaSlice<u8>` of `n` 0/1 bytes.
fn launch_logic_bin(
    a: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    device: &GpuDevice,
    ptx: String,
    kernel_name: &'static str,
) -> GpuResult<CudaSlice<u8>> {
    // Logical binary ops consume compare-result bool buffers, which are always
    // exact-length (no `.contiguous()` pooled over-allocation reaches here), so
    // logical len == raw len. Preserve the strict operand-equality guard here
    // and pass that shared length as the logical `n`.
    if a.len() != b.len() {
        return Err(GpuError::LengthMismatch {
            a: a.len(),
            b: b.len(),
        });
    }
    launch_cmp::<u8>(
        a,
        b,
        a.len(),
        device,
        ptx,
        kernel_name.to_string(),
        kernel_name,
    )
}

/// Launch the unary NOT `(a_ptr, out_ptr, n)` kernel over a u8 bool buffer.
fn launch_not(a: &CudaSlice<u8>, device: &GpuDevice) -> GpuResult<CudaSlice<u8>> {
    let n = a.len();
    let stream = device.stream();
    if n == 0 {
        return Ok(stream.alloc_zeros::<u8>(0)?);
    }
    let ctx = device.context();
    let f = get_or_compile(
        ctx,
        NOT_BOOL_PTX,
        "not_bool_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "not_bool_kernel",
        source: e,
    })?;
    let mut out = stream.alloc_zeros::<u8>(n)?;
    let cfg = launch_1d(n);
    let n_u32 = n as u32;
    // SAFETY:
    // - `f` resolves to the unary PTX entry `(a_ptr, out_ptr, n)`.
    // - `a` is the caller's input of `n` u8 elements; the bound check limits
    //   reads to `[0, n)`.
    // - `out` is freshly alloc'd `n` u8 elements, exclusively borrowed here.
    // - `n_u32` is non-truncating (see `launch_cmp`).
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a)
            .arg(&mut out)
            .arg(&n_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Launch a raw sign-bit predicate over a value buffer, producing u8 0/1.
fn launch_signbit<T: DeviceRepr + ValidAsZeroBits>(
    a: &CudaSlice<T>,
    n: usize,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
) -> GpuResult<CudaSlice<u8>> {
    if a.len() < n {
        return Err(GpuError::LengthMismatch { a: a.len(), b: n });
    }
    let stream = device.stream();
    if n == 0 {
        return Ok(stream.alloc_zeros::<u8>(0)?);
    }
    let ctx = device.context();
    let f = get_or_compile(ctx, ptx, kernel_name, device.ordinal() as u32).map_err(|e| {
        GpuError::PtxCompileFailed {
            kernel: kernel_name,
            source: e,
        }
    })?;
    let mut out = stream.alloc_zeros::<u8>(n)?;
    let cfg = launch_1d(n);
    let n_u32 = n as u32;
    // SAFETY:
    // - `f` resolves to `(a_ptr, out_ptr, n)` and the pushed args match.
    // - `a` backs at least the logical `n` elements; pooled over-allocation is
    //   permitted, but the kernel reads only `[0, n)`.
    // - `out` is freshly allocated for exactly `n` bool bytes and is the only
    //   mutable buffer passed to the kernel.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a)
            .arg(&mut out)
            .arg(&n_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Launch the serial bool reduction `(a_ptr, out_ptr, n, op)`, returning a
/// 1-element u8 buffer. `op` selects any (OR) / all (AND).
fn launch_reduce_bool(
    a: &CudaSlice<u8>,
    device: &GpuDevice,
    op: u32,
    empty_identity: u8,
) -> GpuResult<CudaSlice<u8>> {
    let n = a.len();
    let stream = device.stream();
    if n == 0 {
        // any of empty = false (0), all of empty = true (1).
        let host = [empty_identity];
        return Ok(stream.clone_htod(&host)?);
    }
    let ctx = device.context();
    let f = get_or_compile(
        ctx,
        REDUCE_BOOL_PTX,
        "reduce_bool_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "reduce_bool_kernel",
        source: e,
    })?;
    let mut out = stream.alloc_zeros::<u8>(1)?;
    // Single block, single active thread (thread 0 folds serially).
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_u32 = n as u32;
    // SAFETY:
    // - `f` is the reduce PTX entry `(a_ptr, out_ptr, n, op)`; four args below
    //   match in order.
    // - `a` is the caller's input of `n` u8 elements; the kernel reads `a[0..n)`
    //   from thread 0 only (`setp.ne.u32 %only0, %idx, 0` gates the rest off)
    //   and writes the single `out[0]`.
    // - `out` is a freshly alloc'd 1-element buffer, exclusively borrowed.
    // - `op` is one of {0, 1} (REDUCE_ANY / REDUCE_ALL).
    // - `n_u32` is non-truncating for any host-allocatable contiguous buffer.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a)
            .arg(&mut out)
            .arg(&n_u32)
            .arg(&op)
            .launch(cfg)?;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn launch_reduce_float_bool<T: DeviceRepr + ValidAsZeroBits>(
    input: &CudaSlice<T>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
    op: u32,
) -> GpuResult<CudaSlice<u8>> {
    let total = outer
        .checked_mul(inner)
        .ok_or(GpuError::LengthMismatch { a: outer, b: inner })?;
    let expect = outer
        .checked_mul(dim_size)
        .and_then(|x| x.checked_mul(inner))
        .ok_or(GpuError::LengthMismatch {
            a: outer,
            b: dim_size,
        })?;
    if input.len() < expect {
        return Err(GpuError::LengthMismatch {
            a: input.len(),
            b: expect,
        });
    }
    let stream = device.stream();
    if total == 0 {
        return Ok(stream.alloc_zeros::<u8>(0)?);
    }
    let ctx = device.context();
    let f = get_or_compile(ctx, ptx, kernel_name, device.ordinal() as u32).map_err(|e| {
        GpuError::PtxCompileFailed {
            kernel: kernel_name,
            source: e,
        }
    })?;
    let mut out = stream.alloc_zeros::<u8>(total)?;
    let cfg = launch_1d(total);
    let (outer_u, dim_u, inner_u, total_u) =
        (outer as u32, dim_size as u32, inner as u32, total as u32);
    // SAFETY: `f` is the PTX entry `(in, out, outer, dim_size, inner, total,
    // op)`. `input` backs at least `outer*dim_size*inner` elements; `out` is
    // freshly allocated for `outer*inner` bool bytes. Each thread writes one
    // output slot and scans only the corresponding in-bounds slice.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
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

#[allow(clippy::too_many_arguments)]
fn launch_reduce_float_count<T: DeviceRepr + ValidAsZeroBits>(
    input: &CudaSlice<T>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
) -> GpuResult<CudaSlice<i64>> {
    let total = outer
        .checked_mul(inner)
        .ok_or(GpuError::LengthMismatch { a: outer, b: inner })?;
    let expect = outer
        .checked_mul(dim_size)
        .and_then(|x| x.checked_mul(inner))
        .ok_or(GpuError::LengthMismatch {
            a: outer,
            b: dim_size,
        })?;
    if input.len() < expect {
        return Err(GpuError::LengthMismatch {
            a: input.len(),
            b: expect,
        });
    }
    let stream = device.stream();
    if total == 0 {
        return Ok(stream.alloc_zeros::<i64>(0)?);
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
    let (outer_u, dim_u, inner_u, total_u) =
        (outer as u32, dim_size as u32, inner as u32, total as u32);
    // SAFETY: `f` is the PTX entry `(in, out, outer, dim_size, inner, total)`.
    // Bounds and exclusive output ownership match `launch_reduce_float_bool`.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(&mut out)
            .arg(&outer_u)
            .arg(&dim_u)
            .arg(&inner_u)
            .arg(&total_u)
            .launch(cfg)?;
    }
    Ok(out)
}

// ===========================================================================
// Public entry points
// ===========================================================================

/// The `setp` PTX form for a comparison `op` over the given PTX type
/// (`"f32"`/`"f64"`/`"s32"`/`"s64"`). Result predicate is `%c`.
fn setp_for(op: &str, ty: &str) -> String {
    format!("setp.{op}.{ty} %c, %va, %vb;")
}

/// f32 comparison: `out = (a OP b)` as a u8 0/1 buffer. `op` ∈
/// {eq,ne,lt,le,gt,ge}.
pub fn gpu_cmp_f32(
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    n: usize,
    op: &str,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    let name = format!("cmp_{op}_f32_kernel");
    let ptx = cmp_ptx(&name, 2, "f32", ".reg .f32 %va, %vb;", &setp_for(op, "f32"));
    launch_cmp::<f32>(a, b, n, d, ptx, name, "cmp_f32")
}

/// f64 comparison → u8 0/1 buffer.
pub fn gpu_cmp_f64(
    a: &CudaSlice<f64>,
    b: &CudaSlice<f64>,
    n: usize,
    op: &str,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    let name = format!("cmp_{op}_f64_kernel");
    let ptx = cmp_ptx(&name, 3, "f64", ".reg .f64 %va, %vb;", &setp_for(op, "f64"));
    launch_cmp::<f64>(a, b, n, d, ptx, name, "cmp_f64")
}

/// i32 comparison → u8 0/1 buffer (signed compare).
pub fn gpu_cmp_i32(
    a: &CudaSlice<i32>,
    b: &CudaSlice<i32>,
    n: usize,
    op: &str,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    let name = format!("cmp_{op}_i32_kernel");
    let ptx = cmp_ptx(&name, 2, "s32", ".reg .s32 %va, %vb;", &setp_for(op, "s32"));
    launch_cmp::<i32>(a, b, n, d, ptx, name, "cmp_i32")
}

/// i64 comparison → u8 0/1 buffer (signed compare).
pub fn gpu_cmp_i64(
    a: &CudaSlice<i64>,
    b: &CudaSlice<i64>,
    n: usize,
    op: &str,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    let name = format!("cmp_{op}_i64_kernel");
    let ptx = cmp_ptx(&name, 3, "s64", ".reg .s64 %va, %vb;", &setp_for(op, "s64"));
    launch_cmp::<i64>(a, b, n, d, ptx, name, "cmp_i64")
}

/// Broadcasted i32 comparison → u8 0/1 buffer.
#[allow(clippy::too_many_arguments)]
pub fn gpu_cmp_broadcast_i32(
    a: &CudaSlice<i32>,
    b: &CudaSlice<i32>,
    a_numel: usize,
    b_numel: usize,
    out_shape: &[usize],
    a_src_strides: &[usize],
    b_src_strides: &[usize],
    op: &str,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    let name = format!("cmp_broadcast_{op}_i32_kernel");
    let ptx = cmp_broadcast_ptx(&name, 2, "s32", ".reg .s32 %va, %vb;", &setp_for(op, "s32"));
    launch_cmp_broadcast::<i32>(
        a,
        b,
        a_numel,
        b_numel,
        out_shape,
        a_src_strides,
        b_src_strides,
        d,
        ptx,
        name,
        "cmp_broadcast_i32",
    )
}

/// Broadcasted i64 comparison → u8 0/1 buffer.
#[allow(clippy::too_many_arguments)]
pub fn gpu_cmp_broadcast_i64(
    a: &CudaSlice<i64>,
    b: &CudaSlice<i64>,
    a_numel: usize,
    b_numel: usize,
    out_shape: &[usize],
    a_src_strides: &[usize],
    b_src_strides: &[usize],
    op: &str,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    let name = format!("cmp_broadcast_{op}_i64_kernel");
    let ptx = cmp_broadcast_ptx(&name, 3, "s64", ".reg .s64 %va, %vb;", &setp_for(op, "s64"));
    launch_cmp_broadcast::<i64>(
        a,
        b,
        a_numel,
        b_numel,
        out_shape,
        a_src_strides,
        b_src_strides,
        d,
        ptx,
        name,
        "cmp_broadcast_i64",
    )
}

/// bf16 comparison (u16 bit patterns decoded to f32) → u8 0/1 buffer.
pub fn gpu_cmp_bf16(
    a: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    n: usize,
    op: &str,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    let name = format!("cmp_{op}_bf16_kernel");
    let setp = format!("setp.{op}.f32 %c, %fa, %fb;");
    let ptx = cmp_half_ptx(&name, "sm_52", BF16_DECODE, &setp);
    launch_cmp_half(a, b, n, d, ptx, name)
}

/// f16 comparison (u16 bit patterns decoded to f32) → u8 0/1 buffer.
pub fn gpu_cmp_f16(
    a: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    n: usize,
    op: &str,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    let name = format!("cmp_{op}_f16_kernel");
    let setp = format!("setp.{op}.f32 %c, %fa, %fb;");
    let ptx = cmp_half_ptx(&name, "sm_53", F16_DECODE, &setp);
    launch_cmp_half(a, b, n, d, ptx, name)
}

/// Logical AND of two u8 bool buffers → u8 0/1 buffer.
pub fn gpu_and_bool(
    a: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    let ptx = logic_bin_ptx("and_bool_kernel", "and");
    launch_logic_bin(a, b, d, ptx, "and_bool_kernel")
}

/// Logical OR of two u8 bool buffers → u8 0/1 buffer.
pub fn gpu_or_bool(
    a: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    let ptx = logic_bin_ptx("or_bool_kernel", "or");
    launch_logic_bin(a, b, d, ptx, "or_bool_kernel")
}

/// Logical XOR of two u8 bool buffers → u8 0/1 buffer.
pub fn gpu_xor_bool(
    a: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    let ptx = logic_bin_ptx("xor_bool_kernel", "xor");
    launch_logic_bin(a, b, d, ptx, "xor_bool_kernel")
}

/// Logical NOT of a u8 bool buffer → u8 0/1 buffer.
pub fn gpu_not_bool(a: &CudaSlice<u8>, d: &GpuDevice) -> GpuResult<CudaSlice<u8>> {
    launch_not(a, d)
}

/// f32 `signbit`: raw bit 31 → u8 0/1, preserving signed zero and signed NaN.
pub fn gpu_signbit_f32(a: &CudaSlice<f32>, n: usize, d: &GpuDevice) -> GpuResult<CudaSlice<u8>> {
    launch_signbit(a, n, d, SIGNBIT_F32_PTX, "signbit_f32_kernel")
}

/// f64 `signbit`: raw bit 63 → u8 0/1.
pub fn gpu_signbit_f64(a: &CudaSlice<f64>, n: usize, d: &GpuDevice) -> GpuResult<CudaSlice<u8>> {
    launch_signbit(a, n, d, SIGNBIT_F64_PTX, "signbit_f64_kernel")
}

/// f16 `signbit`: bit 15 for non-NaN values; NaN payloads return 0 per CUDA
/// PyTorch parity.
pub fn gpu_signbit_f16(a: &CudaSlice<u16>, n: usize, d: &GpuDevice) -> GpuResult<CudaSlice<u8>> {
    launch_signbit(a, n, d, SIGNBIT_F16_PTX, "signbit_f16_kernel")
}

/// bf16 `signbit`: raw u16 bit 15 → u8 0/1.
pub fn gpu_signbit_bf16(a: &CudaSlice<u16>, n: usize, d: &GpuDevice) -> GpuResult<CudaSlice<u8>> {
    launch_signbit(a, n, d, SIGNBIT_BF16_PTX, "signbit_bf16_kernel")
}

/// Global OR-reduction (`torch.any`) → 1-element u8 buffer (0/1).
pub fn gpu_any_bool(a: &CudaSlice<u8>, d: &GpuDevice) -> GpuResult<CudaSlice<u8>> {
    launch_reduce_bool(a, d, REDUCE_ANY, 0)
}

/// Global AND-reduction (`torch.all`) → 1-element u8 buffer (0/1).
pub fn gpu_all_bool(a: &CudaSlice<u8>, d: &GpuDevice) -> GpuResult<CudaSlice<u8>> {
    launch_reduce_bool(a, d, REDUCE_ALL, 1)
}

/// Float-value `any` over logical `[outer, dim_size, inner]` → bool bytes.
pub fn gpu_any_f32(
    input: &CudaSlice<f32>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    launch_reduce_float_bool(
        input,
        outer,
        dim_size,
        inner,
        d,
        REDUCE_FLOAT_F32_BOOL_PTX,
        "reduce_float_f32_bool_kernel",
        REDUCE_ANY,
    )
}

/// Float-value `all` over f32 logical `[outer, dim_size, inner]` → bool bytes.
pub fn gpu_all_f32(
    input: &CudaSlice<f32>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    launch_reduce_float_bool(
        input,
        outer,
        dim_size,
        inner,
        d,
        REDUCE_FLOAT_F32_BOOL_PTX,
        "reduce_float_f32_bool_kernel",
        REDUCE_ALL,
    )
}

/// Count nonzero f32 values over logical `[outer, dim_size, inner]` → i64.
pub fn gpu_count_nonzero_f32(
    input: &CudaSlice<f32>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i64>> {
    launch_reduce_float_count(
        input,
        outer,
        dim_size,
        inner,
        d,
        REDUCE_FLOAT_F32_COUNT_PTX,
        "reduce_float_f32_count_kernel",
    )
}

/// Float-value `any` over f64 logical `[outer, dim_size, inner]` → bool bytes.
pub fn gpu_any_f64(
    input: &CudaSlice<f64>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    launch_reduce_float_bool(
        input,
        outer,
        dim_size,
        inner,
        d,
        REDUCE_FLOAT_F64_BOOL_PTX,
        "reduce_float_f64_bool_kernel",
        REDUCE_ANY,
    )
}

/// Float-value `all` over f64 logical `[outer, dim_size, inner]` → bool bytes.
pub fn gpu_all_f64(
    input: &CudaSlice<f64>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    launch_reduce_float_bool(
        input,
        outer,
        dim_size,
        inner,
        d,
        REDUCE_FLOAT_F64_BOOL_PTX,
        "reduce_float_f64_bool_kernel",
        REDUCE_ALL,
    )
}

/// Count nonzero f64 values over logical `[outer, dim_size, inner]` → i64.
pub fn gpu_count_nonzero_f64(
    input: &CudaSlice<f64>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i64>> {
    launch_reduce_float_count(
        input,
        outer,
        dim_size,
        inner,
        d,
        REDUCE_FLOAT_F64_COUNT_PTX,
        "reduce_float_f64_count_kernel",
    )
}

/// Float-value `any` over f16 bit-pattern storage → bool bytes.
pub fn gpu_any_f16(
    input: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    launch_reduce_float_bool(
        input,
        outer,
        dim_size,
        inner,
        d,
        REDUCE_FLOAT_F16_BOOL_PTX,
        "reduce_float_f16_bool_kernel",
        REDUCE_ANY,
    )
}

/// Float-value `all` over f16 bit-pattern storage → bool bytes.
pub fn gpu_all_f16(
    input: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    launch_reduce_float_bool(
        input,
        outer,
        dim_size,
        inner,
        d,
        REDUCE_FLOAT_F16_BOOL_PTX,
        "reduce_float_f16_bool_kernel",
        REDUCE_ALL,
    )
}

/// Count nonzero f16 bit-pattern values over logical slices → i64.
pub fn gpu_count_nonzero_f16(
    input: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i64>> {
    launch_reduce_float_count(
        input,
        outer,
        dim_size,
        inner,
        d,
        REDUCE_FLOAT_F16_COUNT_PTX,
        "reduce_float_f16_count_kernel",
    )
}

/// Float-value `any` over bf16 bit-pattern storage → bool bytes.
pub fn gpu_any_bf16(
    input: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    launch_reduce_float_bool(
        input,
        outer,
        dim_size,
        inner,
        d,
        REDUCE_FLOAT_BF16_BOOL_PTX,
        "reduce_float_bf16_bool_kernel",
        REDUCE_ANY,
    )
}

/// Float-value `all` over bf16 bit-pattern storage → bool bytes.
pub fn gpu_all_bf16(
    input: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    launch_reduce_float_bool(
        input,
        outer,
        dim_size,
        inner,
        d,
        REDUCE_FLOAT_BF16_BOOL_PTX,
        "reduce_float_bf16_bool_kernel",
        REDUCE_ALL,
    )
}

/// Count nonzero bf16 bit-pattern values over logical slices → i64.
pub fn gpu_count_nonzero_bf16(
    input: &CudaSlice<u16>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i64>> {
    launch_reduce_float_count(
        input,
        outer,
        dim_size,
        inner,
        d,
        REDUCE_FLOAT_BF16_COUNT_PTX,
        "reduce_float_bf16_count_kernel",
    )
}

// ===========================================================================
// On-device bool broadcast (#1663)
//
// `broadcast_bool` expands a `DType::Bool` (u8 0/1) buffer from its own shape
// to a larger `out_shape` using NumPy / torch broadcasting rules (align
// trailing dims; a size-1 or absent input dim replicates). This is the resident
// analog of the CPU `grad_fns::indexing::broadcast_bool_tensor`, mirroring the
// `expand_outplace(mask, self)` step PyTorch performs for masked ops at
// `aten/src/ATen/native/TensorAdvancedIndexing.cpp:2406`. Broadcasting reduces
// to a STRIDED GATHER over u8 elements: output flat index `i` -> input flat
// index via per-output-dim element strides, where a broadcast (size-1 / absent)
// input dim contributes stride 0. This is structurally identical to the f32
// `crate::kernels::STRIDED_COPY_PTX`, specialised to a 1-byte element (no
// `shl.b64 .., 2`; the offset is the index itself). 8 dims unrolled, each stride
// passed as an individual u32 param (20 params, within the ~4KB param limit).
// ===========================================================================

/// Maximum rank supported by [`gpu_broadcast_bool`]. Matches the unrolled PTX.
pub const BOOL_BROADCAST_MAX_DIMS: usize = 8;

/// PTX for `bool_broadcast_kernel`: a u8 strided gather with up to 8 dims.
///
/// Thread `i` computes:
///   flat = i
///   src = 0
///   for d in 0..8:
///       coord = flat / out_stride[d]
///       flat  = flat % out_stride[d]
///       src  += coord * src_stride[d]
///   out[i] = in[src]
///
/// For tensors with fewer than 8 dims, unused positions are padded with
/// `out_stride[d] = n + 1` (so `flat / out_stride[d] == 0`) and
/// `src_stride[d] = 0` (so the contribution is zero). u8 elements: the byte
/// offset equals the element index (no shift).
const BOOL_BROADCAST_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry bool_broadcast_kernel(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u32 n,
    .param .u32 os0, .param .u32 os1, .param .u32 os2, .param .u32 os3,
    .param .u32 os4, .param .u32 os5, .param .u32 os6, .param .u32 os7,
    .param .u32 ss0, .param .u32 ss1, .param .u32 ss2, .param .u32 ss3,
    .param .u32 ss4, .param .u32 ss5, .param .u32 ss6, .param .u32 ss7
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u32 %flat, %src_idx, %coord, %tmp, %os, %ss;
    .reg .u64 %in, %out, %off;
    .reg .u16 %val;
    .reg .pred %p;

    ld.param.u64 %in, [input_ptr];
    ld.param.u64 %out, [output_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    mov.u32 %flat, %r_tid;
    mov.u32 %src_idx, 0;

    // Dim 0
    ld.param.u32 %os, [os0];
    ld.param.u32 %ss, [ss0];
    div.u32 %coord, %flat, %os;
    mul.lo.u32 %tmp, %coord, %os;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ss;
    add.u32 %src_idx, %src_idx, %tmp;

    // Dim 1
    ld.param.u32 %os, [os1];
    ld.param.u32 %ss, [ss1];
    div.u32 %coord, %flat, %os;
    mul.lo.u32 %tmp, %coord, %os;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ss;
    add.u32 %src_idx, %src_idx, %tmp;

    // Dim 2
    ld.param.u32 %os, [os2];
    ld.param.u32 %ss, [ss2];
    div.u32 %coord, %flat, %os;
    mul.lo.u32 %tmp, %coord, %os;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ss;
    add.u32 %src_idx, %src_idx, %tmp;

    // Dim 3
    ld.param.u32 %os, [os3];
    ld.param.u32 %ss, [ss3];
    div.u32 %coord, %flat, %os;
    mul.lo.u32 %tmp, %coord, %os;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ss;
    add.u32 %src_idx, %src_idx, %tmp;

    // Dim 4
    ld.param.u32 %os, [os4];
    ld.param.u32 %ss, [ss4];
    div.u32 %coord, %flat, %os;
    mul.lo.u32 %tmp, %coord, %os;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ss;
    add.u32 %src_idx, %src_idx, %tmp;

    // Dim 5
    ld.param.u32 %os, [os5];
    ld.param.u32 %ss, [ss5];
    div.u32 %coord, %flat, %os;
    mul.lo.u32 %tmp, %coord, %os;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ss;
    add.u32 %src_idx, %src_idx, %tmp;

    // Dim 6
    ld.param.u32 %os, [os6];
    ld.param.u32 %ss, [ss6];
    div.u32 %coord, %flat, %os;
    mul.lo.u32 %tmp, %coord, %os;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ss;
    add.u32 %src_idx, %src_idx, %tmp;

    // Dim 7
    ld.param.u32 %os, [os7];
    ld.param.u32 %ss, [ss7];
    div.u32 %coord, %flat, %os;
    mul.lo.u32 %tmp, %coord, %os;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ss;
    add.u32 %src_idx, %src_idx, %tmp;

    // Load in[src_idx] (1 byte per element: byte offset == element index).
    cvt.u64.u32 %off, %src_idx;
    add.u64 %off, %in, %off;
    ld.global.u8 %val, [%off];

    // Store out[r_tid].
    cvt.u64.u32 %off, %r_tid;
    add.u64 %off, %out, %off;
    st.global.u8 [%off], %val;

DONE:
    ret;
}
";

/// Pad-and-validate the (`out_shape`, broadcast `src_strides`) pair for the
/// [`gpu_broadcast_bool`] kernel.
///
/// Returns a fixed-size `[MAX_DIMS]` pair where `out_stride[d]` is the
/// contiguous output element stride (unused trailing dims filled with `n + 1`
/// so `flat / out_stride[d] == 0`) and `src_stride[d]` is the broadcast input
/// element stride (unused dims filled with 0). `out_shape` and `src_strides`
/// must have equal length, at most [`BOOL_BROADCAST_MAX_DIMS`]; `n` is
/// `product(out_shape)`.
fn pad_bool_broadcast_params(
    out_shape: &[usize],
    src_strides: &[usize],
    n: usize,
) -> GpuResult<(
    [u32; BOOL_BROADCAST_MAX_DIMS],
    [u32; BOOL_BROADCAST_MAX_DIMS],
)> {
    if out_shape.len() != src_strides.len() {
        return Err(GpuError::ShapeMismatch {
            op: "bool_broadcast_pad",
            expected: vec![out_shape.len()],
            got: vec![src_strides.len()],
        });
    }
    if out_shape.len() > BOOL_BROADCAST_MAX_DIMS {
        return Err(GpuError::ShapeMismatch {
            op: "bool_broadcast_pad",
            expected: vec![BOOL_BROADCAST_MAX_DIMS],
            got: vec![out_shape.len()],
        });
    }
    let rank = out_shape.len();
    // Contiguous output strides: stride[rank-1] = 1, stride[d] = stride[d+1] * shape[d+1].
    let mut out_stride = [0u32; BOOL_BROADCAST_MAX_DIMS];
    if rank > 0 {
        let mut acc: usize = 1;
        for d in (0..rank).rev() {
            if acc > u32::MAX as usize {
                return Err(GpuError::ShapeMismatch {
                    op: "bool_broadcast_stride_overflow",
                    expected: vec![u32::MAX as usize],
                    got: vec![acc],
                });
            }
            out_stride[d] = acc as u32;
            acc = acc.saturating_mul(out_shape[d]);
        }
    }
    // Pad unused dims with `n + 1` so `flat / out_stride[d] == 0`.
    let pad_val = (n as u32).saturating_add(1).max(1);
    out_stride[rank..BOOL_BROADCAST_MAX_DIMS].fill(pad_val);

    let mut src_stride_out = [0u32; BOOL_BROADCAST_MAX_DIMS];
    for d in 0..rank {
        let s = src_strides[d];
        if s > u32::MAX as usize {
            return Err(GpuError::ShapeMismatch {
                op: "bool_broadcast_src_stride_overflow",
                expected: vec![u32::MAX as usize],
                got: vec![s],
            });
        }
        src_stride_out[d] = s as u32;
    }
    Ok((out_stride, src_stride_out))
}

/// Broadcast a u8 bool buffer to `out_shape` entirely on device (#1663).
///
/// `src_strides` are the per-output-dim broadcast input element strides, aligned
/// with `out_shape`: the contiguous input stride where the input dim equals the
/// output dim, or `0` where the input dim is size-1 or absent (the standard
/// NumPy/torch broadcast pattern). The caller (the [`crate::backend_impl`]
/// `broadcast_bool` dispatch) computes these from the (in_shape, out_shape)
/// pair, mirroring the CPU `broadcast_in_flat` index map. Returns a fresh
/// `CudaSlice<u8>` of `product(out_shape)` 0/1 bytes resident on `device` — no
/// host round trip.
#[allow(clippy::too_many_lines, reason = "8-dim unrolled launch arg list")]
pub fn gpu_broadcast_bool(
    input: &CudaSlice<u8>,
    out_shape: &[usize],
    src_strides: &[usize],
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u8>> {
    let n: usize = if out_shape.is_empty() {
        1
    } else {
        out_shape.iter().product()
    };
    let (out_stride, src_stride) = pad_bool_broadcast_params(out_shape, src_strides, n)?;
    let stream = device.stream();
    if n == 0 {
        return Ok(stream.alloc_zeros::<u8>(0)?);
    }
    let ctx = device.context();
    let f = get_or_compile(
        ctx,
        BOOL_BROADCAST_PTX,
        "bool_broadcast_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "bool_broadcast_kernel",
        source: e,
    })?;
    let mut out = stream.alloc_zeros::<u8>(n)?;
    let cfg = launch_1d(n);
    let n_u32 = n as u32;
    // SAFETY:
    // - `f` is the `bool_broadcast_kernel` PTX entry just compiled; its ABI is
    //   `(in_ptr: u64, out_ptr: u64, n: u32, out_stride[0..8]: u32,
    //   src_stride[0..8]: u32)` — 19 args matching the order pushed below.
    // - `input` is an immutable u8 buffer backing at least one element; the
    //   broadcast src index for every output thread stays within `input.len()`
    //   because each `src_stride[d]` is the contiguous input stride for a
    //   non-broadcast dim (0 for broadcast/absent dims), so the maximum src
    //   index equals the input's last contiguous element — validated by the
    //   caller's (in_shape, out_shape) broadcast-compatibility check.
    // - `out` was freshly `alloc_zeros::<u8>(n)` (the only `&mut`), non-aliased
    //   with `input` (distinct allocations).
    // - Each thread `i in [0, n)` decodes a multi-dim index via `out_stride`
    //   then sums `coord_d * src_stride[d]`; the `tid >= n` PTX bound check
    //   short-circuits OOB threads. Unused trailing dims are no-ops
    //   (`out_stride[d] = n+1` -> coord 0, `src_stride[d] = 0`).
    // - `n_u32 = n as u32` sized the grid in `launch_1d`; all 18 stride refs and
    //   the `n_u32` ref live to the trailing `?`.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(&mut out)
            .arg(&n_u32)
            .arg(&out_stride[0])
            .arg(&out_stride[1])
            .arg(&out_stride[2])
            .arg(&out_stride[3])
            .arg(&out_stride[4])
            .arg(&out_stride[5])
            .arg(&out_stride[6])
            .arg(&out_stride[7])
            .arg(&src_stride[0])
            .arg(&src_stride[1])
            .arg(&src_stride[2])
            .arg(&src_stride[3])
            .arg(&src_stride[4])
            .arg(&src_stride[5])
            .arg(&src_stride[6])
            .arg(&src_stride[7])
            .launch(cfg)?;
    }
    Ok(out)
}
