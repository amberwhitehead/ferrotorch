//! Integer (`i32` / `i64`) GPU compute kernels — crosslink #1185 Phase 2b.
//!
//! Hand-written PTX owned by Rust (no CUDA C++, no nvrtc, no external
//! toolchain at load time), loaded via `module_cache::get_or_compile` exactly
//! like [`crate::f16`] / [`crate::bf16`]. Unlike the half-precision modules,
//! integer buffers are **native** `CudaSlice<i32>` / `CudaSlice<i64>`
//! (cudarc `DeviceRepr` — no `u16` bit-pattern transmute): the on-device
//! element type matches the logical element type, and the
//! [`crate::backend_impl::CudaBackendImpl`] handle is tagged `DType::I32` /
//! `DType::I64` so an i32 buffer (4 bytes) is never read as f32.
//!
//! # Operations
//!
//! - Elementwise binary: `add`, `sub`, `mul`, `floor_divide`, `remainder`,
//!   `bitand`, `bitor`, `bitxor`, `shl`, `shr`.
//! - Elementwise unary: `neg`, `bitnot`.
//! - Reductions to a 1-element buffer: `sum`, `prod`, `min`, `max`.
//!
//! # PyTorch-parity semantics (the CPU reference in `ferrotorch-core` matches
//! the same rules — see `int_tensor.rs`)
//!
//! - `floor_divide` floors toward −∞ (NOT C truncation toward zero). PTX
//!   `div.s{32,64}` truncates toward zero; we then subtract 1 from the quotient
//!   when the remainder is nonzero AND the operand signs differ
//!   (`(rem < 0) != (b < 0)`). Matches `torch.floor_divide`.
//! - `remainder` takes the sign of the **divisor** (Python / `torch.remainder`),
//!   not the C `%` sign of the dividend. We compute the truncating remainder
//!   `r = rem.s(a, b)` (sign of dividend) and add `b` when `r != 0` AND the
//!   signs of `r` and `b` differ. This is exactly `a - floor_divide(a,b)*b`.
//! - `shr` on signed integers is an ARITHMETIC (sign-extending) shift
//!   (`shr.s{32,64}`), matching PyTorch `__rshift__` on signed dtypes.
//! - `sum` / `prod` accumulate in the SAME integer width (wrapping on
//!   overflow — PyTorch does NOT upcast integer `sum` by default).
//! - Integer division / remainder by zero is NOT trapped: PTX `div.s` / `rem.s`
//!   by zero returns an implementation-defined value (PyTorch on CUDA likewise
//!   does not trap). No host round-trip is taken to special-case it.
//!
//! Reductions use a single-thread serial accumulator (one launched thread folds
//! all `n` elements with an integer accumulator — no f32 detour). This keeps
//! the result exactly equal to a left-fold over the buffer, matching the CPU
//! reference bit-for-bit.
//!
//! ## REQ status (per `.design/ferrotorch-gpu/int_kernels.md`)
//!
//! Full evidence rows (impl + non-test production consumer + upstream
//! cites) live in the design doc; this synopsis is a one-line summary per
//! REQ.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (add/sub/mul) | SHIPPED | `pub fn gpu_add_i32 / gpu_add_i64 / gpu_sub_i32 / gpu_sub_i64 / gpu_mul_i32 / gpu_mul_i64 in int_kernels.rs` wrap `launch_binary` with PTX `add.s* / sub.s* / mul.lo.s*`; consumer six int arms of arithmetic dispatchers in `backend_impl.rs` |
//! | REQ-2 (floor_div/remainder) | SHIPPED | `pub fn gpu_floor_div_i32 / gpu_floor_div_i64 / gpu_remainder_i32 / gpu_remainder_i64 in int_kernels.rs` (PTX trunc-then-floor-correct / rem-then-sign-adjust); consumer floor_div/remainder arms in `backend_impl.rs` |
//! | REQ-3 (bitand/bitor/bitxor) | SHIPPED | `pub fn gpu_bitand_i32 / gpu_bitand_i64 / gpu_bitor_i32 / gpu_bitor_i64 / gpu_bitxor_i32 / gpu_bitxor_i64 in int_kernels.rs`; consumer six bitwise arms in `backend_impl.rs` |
//! | REQ-4 (shl/shr) | SHIPPED | `pub fn gpu_shl_i32 / gpu_shl_i64` (PTX `shl.b{32,64}`) + `pub fn gpu_shr_i32 / gpu_shr_i64` (PTX `shr.s{32,64}`, arithmetic) in `int_kernels.rs`; consumer shl/shr arms in `backend_impl.rs` |
//! | REQ-5 (neg/bitnot) | SHIPPED | `pub fn gpu_neg_i32 / gpu_neg_i64` (PTX `sub.s* 0, %va`) + `pub fn gpu_bitnot_i32 / gpu_bitnot_i64` (PTX `not.b{32,64}`) in `int_kernels.rs`; consumer neg/bitnot arms in `backend_impl.rs` |
//! | REQ-6 (sum/prod/min/max) | SHIPPED | `pub fn gpu_sum_i32 / gpu_sum_i64 / gpu_prod_i32 / gpu_prod_i64 / gpu_min_i32 / gpu_min_i64 / gpu_max_i32 / gpu_max_i64 in int_kernels.rs` wrap `launch_reduce` with `REDUCE_SUM/PROD/MIN/MAX` op codes; consumer eight integer reduction arms in `backend_impl.rs` |
//! | REQ-7 (div-by-zero no-trap) | SHIPPED | module `//!` doc-comment in `int_kernels.rs` states no zero-check; the `FLOORDIV_*_PTX / REMAINDER_*_PTX` kernels include no zero-check branch; consumer backend integer arm relies on the documented no-trap contract |
//! | REQ-8 (SAFETY annotations) | SHIPPED | three `unsafe { stream.launch_builder(&f)...launch(cfg)? }` blocks in `fn launch_binary / launch_unary / launch_reduce in int_kernels.rs` each carry a multi-line `SAFETY:` comment; consumer SAFETY contract inherited via every `pub fn gpu_*_i{32,64}` wrapper |
//! | REQ-9 (empty-input short-circuit) | SHIPPED | `fn launch_binary / launch_unary in int_kernels.rs` open with `if n == 0 { return Ok(stream.alloc_zeros::<T>(0)?); }`; `fn launch_reduce` short-circuits with empty-identity clone_htod (`0/1/T::MAX/T::MIN` for sum/prod/min/max); consumer backend empty-int handling |

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
// Elementwise binary kernels — i32
//
// Each thread loads a[i], b[i] (one 32-bit signed int each), computes one
// result, stores out[i]. `off = i << 2` (4 bytes per i32). Bound-checked
// against `n`.
// ===========================================================================

const ADD_I32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry add_i32_kernel(
    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %b, %out, %off;
    .reg .s32 %va, %vb, %vr;
    .reg .pred %p;

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
    shl.b64 %off, %off, 2;
    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.s32 %va, [%a];
    ld.global.s32 %vb, [%b];
    add.s32 %vr, %va, %vb;
    st.global.s32 [%out], %vr;
DONE:
    ret;
}
";

const SUB_I32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry sub_i32_kernel(
    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %b, %out, %off;
    .reg .s32 %va, %vb, %vr;
    .reg .pred %p;

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
    shl.b64 %off, %off, 2;
    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.s32 %va, [%a];
    ld.global.s32 %vb, [%b];
    sub.s32 %vr, %va, %vb;
    st.global.s32 [%out], %vr;
DONE:
    ret;
}
";

const MUL_I32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry mul_i32_kernel(
    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %b, %out, %off;
    .reg .s32 %va, %vb, %vr;
    .reg .pred %p;

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
    shl.b64 %off, %off, 2;
    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.s32 %va, [%a];
    ld.global.s32 %vb, [%b];
    mul.lo.s32 %vr, %va, %vb;
    st.global.s32 [%out], %vr;
DONE:
    ret;
}
";

// floor_divide (i32): truncated quotient via div.s32, then floor-correct.
// q = a / b (toward zero); r = a - q*b; if (r != 0 && (r<0) != (b<0)) q -= 1.
const FLOORDIV_I32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry floordiv_i32_kernel(
    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %b, %out, %off;
    .reg .s32 %va, %vb, %q, %r, %qm1, %zero;
    .reg .pred %p, %rnz, %rneg, %bneg, %diff;

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
    shl.b64 %off, %off, 2;
    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.s32 %va, [%a];
    ld.global.s32 %vb, [%b];
    mov.s32 %zero, 0;

    div.s32 %q, %va, %vb;
    rem.s32 %r, %va, %vb;

    // diff = (r < 0) XOR (b < 0); correction when r != 0 && diff.
    setp.ne.s32 %rnz, %r, %zero;
    setp.lt.s32 %rneg, %r, %zero;
    setp.lt.s32 %bneg, %vb, %zero;
    xor.pred %diff, %rneg, %bneg;
    and.pred %diff, %diff, %rnz;

    sub.s32 %qm1, %q, 1;
    selp.s32 %q, %qm1, %q, %diff;

    st.global.s32 [%out], %q;
DONE:
    ret;
}
";

// remainder (i32): Python/torch sign-of-divisor remainder.
// r = a % b (trunc, sign of dividend); if (r != 0 && (r<0) != (b<0)) r += b.
const REMAINDER_I32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry remainder_i32_kernel(
    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %b, %out, %off;
    .reg .s32 %va, %vb, %r, %rpb, %zero;
    .reg .pred %p, %rnz, %rneg, %bneg, %diff;

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
    shl.b64 %off, %off, 2;
    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.s32 %va, [%a];
    ld.global.s32 %vb, [%b];
    mov.s32 %zero, 0;

    rem.s32 %r, %va, %vb;

    setp.ne.s32 %rnz, %r, %zero;
    setp.lt.s32 %rneg, %r, %zero;
    setp.lt.s32 %bneg, %vb, %zero;
    xor.pred %diff, %rneg, %bneg;
    and.pred %diff, %diff, %rnz;

    add.s32 %rpb, %r, %vb;
    selp.s32 %r, %rpb, %r, %diff;

    st.global.s32 [%out], %r;
DONE:
    ret;
}
";

const BITAND_I32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry bitand_i32_kernel(
    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %b, %out, %off;
    .reg .b32 %va, %vb, %vr;
    .reg .pred %p;

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
    shl.b64 %off, %off, 2;
    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.b32 %va, [%a];
    ld.global.b32 %vb, [%b];
    and.b32 %vr, %va, %vb;
    st.global.b32 [%out], %vr;
DONE:
    ret;
}
";

const BITOR_I32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry bitor_i32_kernel(
    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %b, %out, %off;
    .reg .b32 %va, %vb, %vr;
    .reg .pred %p;

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
    shl.b64 %off, %off, 2;
    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.b32 %va, [%a];
    ld.global.b32 %vb, [%b];
    or.b32 %vr, %va, %vb;
    st.global.b32 [%out], %vr;
DONE:
    ret;
}
";

const BITXOR_I32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry bitxor_i32_kernel(
    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %b, %out, %off;
    .reg .b32 %va, %vb, %vr;
    .reg .pred %p;

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
    shl.b64 %off, %off, 2;
    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.b32 %va, [%a];
    ld.global.b32 %vb, [%b];
    xor.b32 %vr, %va, %vb;
    st.global.b32 [%out], %vr;
DONE:
    ret;
}
";

// shl (i32): logical left shift by b[i] (b is an i32 buffer; shift amount is
// the low bits). PTX shl.b32 shifts a b32 value; shift count taken from %vb.
const SHL_I32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry shl_i32_kernel(
    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr, %sh;
    .reg .u64 %a, %b, %out, %off;
    .reg .b32 %va, %vr;
    .reg .pred %p;

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
    shl.b64 %off, %off, 2;
    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.b32 %va, [%a];
    ld.global.u32 %sh, [%b];
    shl.b32 %vr, %va, %sh;
    st.global.b32 [%out], %vr;
DONE:
    ret;
}
";

// shr (i32): ARITHMETIC (sign-extending) right shift — shr.s32. Matches
// PyTorch __rshift__ on signed dtypes.
const SHR_I32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry shr_i32_kernel(
    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr, %sh;
    .reg .u64 %a, %b, %out, %off;
    .reg .s32 %va, %vr;
    .reg .pred %p;

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
    shl.b64 %off, %off, 2;
    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.s32 %va, [%a];
    ld.global.u32 %sh, [%b];
    shr.s32 %vr, %va, %sh;
    st.global.s32 [%out], %vr;
DONE:
    ret;
}
";

// ===========================================================================
// Elementwise unary kernels — i32 (neg, bitnot)
// ===========================================================================

const NEG_I32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry neg_i32_kernel(
    .param .u64 a_ptr, .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %out, %off;
    .reg .s32 %va, %vr, %zero;
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

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.s32 %va, [%a];
    mov.s32 %zero, 0;
    sub.s32 %vr, %zero, %va;
    st.global.s32 [%out], %vr;
DONE:
    ret;
}
";

const BITNOT_I32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry bitnot_i32_kernel(
    .param .u64 a_ptr, .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %out, %off;
    .reg .b32 %va, %vr;
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

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.b32 %va, [%a];
    not.b32 %vr, %va;
    st.global.b32 [%out], %vr;
DONE:
    ret;
}
";

// ===========================================================================
// Reduction kernels — i32. One launched thread serially folds all n elements
// with an integer accumulator (wrapping add/mul; min/max). The result is a
// 1-element buffer. Left-fold semantics match the CPU reference exactly.
// `op`: 0=sum, 1=prod, 2=min, 3=max.
// ===========================================================================

const REDUCE_I32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry reduce_i32_kernel(
    .param .u64 a_ptr, .param .u64 out_ptr, .param .u32 n, .param .u32 op
) {
    .reg .u32 %idx, %bid, %bdim, %nr, %op_r, %i;
    .reg .u64 %a, %out, %off, %cur;
    .reg .s32 %acc, %v, %prod, %mn, %mx;
    .reg .pred %p, %only0, %is_sum, %is_prod, %is_min, %lt, %gt;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %nr, [n];
    ld.param.u32 %op_r, [op];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    // Only thread 0 performs the reduction.
    setp.ne.u32 %only0, %idx, 0;
    @%only0 bra DONE;

    setp.eq.u32 %is_sum, %op_r, 0;
    setp.eq.u32 %is_prod, %op_r, 1;
    setp.eq.u32 %is_min, %op_r, 2;

    // Initialise accumulator from a[0] (n >= 1 guaranteed by the host).
    ld.global.s32 %acc, [%a];
    mov.u32 %i, 1;
LOOP:
    setp.ge.u32 %p, %i, %nr;
    @%p bra STORE;

    cvt.u64.u32 %off, %i;
    shl.b64 %off, %off, 2;
    add.u64 %cur, %a, %off;
    ld.global.s32 %v, [%cur];

    // sum
    @%is_sum add.s32 %acc, %acc, %v;
    // prod
    mul.lo.s32 %prod, %acc, %v;
    @%is_prod mov.s32 %acc, %prod;
    // min
    setp.lt.s32 %lt, %v, %acc;
    @%is_min selp.s32 %acc, %v, %acc, %lt;
    // max (the remaining op): only update when op==3.
    setp.gt.s32 %gt, %v, %acc;
    @%is_sum bra SKIPMAX;
    @%is_prod bra SKIPMAX;
    @%is_min bra SKIPMAX;
    selp.s32 %acc, %v, %acc, %gt;
SKIPMAX:

    add.u32 %i, %i, 1;
    bra LOOP;
STORE:
    st.global.s32 [%out], %acc;
DONE:
    ret;
}
";

// ===========================================================================
// i64 kernels. Identical structure to i32 except: element offset shift is 3
// (8 bytes per i64), loads/stores are .s64/.b64/.u64, arithmetic is the .s64
// / .b64 form, and mul is mul.lo.s64. shift count for shl/shr is still a u32.
// ===========================================================================

const ADD_I64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry add_i64_kernel(
    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %b, %out, %off;
    .reg .s64 %va, %vb, %vr;
    .reg .pred %p;

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
    shl.b64 %off, %off, 3;
    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.s64 %va, [%a];
    ld.global.s64 %vb, [%b];
    add.s64 %vr, %va, %vb;
    st.global.s64 [%out], %vr;
DONE:
    ret;
}
";

const SUB_I64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry sub_i64_kernel(
    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %b, %out, %off;
    .reg .s64 %va, %vb, %vr;
    .reg .pred %p;

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
    shl.b64 %off, %off, 3;
    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.s64 %va, [%a];
    ld.global.s64 %vb, [%b];
    sub.s64 %vr, %va, %vb;
    st.global.s64 [%out], %vr;
DONE:
    ret;
}
";

const MUL_I64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry mul_i64_kernel(
    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %b, %out, %off;
    .reg .s64 %va, %vb, %vr;
    .reg .pred %p;

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
    shl.b64 %off, %off, 3;
    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.s64 %va, [%a];
    ld.global.s64 %vb, [%b];
    mul.lo.s64 %vr, %va, %vb;
    st.global.s64 [%out], %vr;
DONE:
    ret;
}
";

const FLOORDIV_I64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry floordiv_i64_kernel(
    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %b, %out, %off;
    .reg .s64 %va, %vb, %q, %r, %qm1, %zero;
    .reg .pred %p, %rnz, %rneg, %bneg, %diff;

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
    shl.b64 %off, %off, 3;
    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.s64 %va, [%a];
    ld.global.s64 %vb, [%b];
    mov.s64 %zero, 0;

    div.s64 %q, %va, %vb;
    rem.s64 %r, %va, %vb;

    setp.ne.s64 %rnz, %r, %zero;
    setp.lt.s64 %rneg, %r, %zero;
    setp.lt.s64 %bneg, %vb, %zero;
    xor.pred %diff, %rneg, %bneg;
    and.pred %diff, %diff, %rnz;

    sub.s64 %qm1, %q, 1;
    selp.s64 %q, %qm1, %q, %diff;

    st.global.s64 [%out], %q;
DONE:
    ret;
}
";

const REMAINDER_I64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry remainder_i64_kernel(
    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %b, %out, %off;
    .reg .s64 %va, %vb, %r, %rpb, %zero;
    .reg .pred %p, %rnz, %rneg, %bneg, %diff;

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
    shl.b64 %off, %off, 3;
    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.s64 %va, [%a];
    ld.global.s64 %vb, [%b];
    mov.s64 %zero, 0;

    rem.s64 %r, %va, %vb;

    setp.ne.s64 %rnz, %r, %zero;
    setp.lt.s64 %rneg, %r, %zero;
    setp.lt.s64 %bneg, %vb, %zero;
    xor.pred %diff, %rneg, %bneg;
    and.pred %diff, %diff, %rnz;

    add.s64 %rpb, %r, %vb;
    selp.s64 %r, %rpb, %r, %diff;

    st.global.s64 [%out], %r;
DONE:
    ret;
}
";

const BITAND_I64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry bitand_i64_kernel(
    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %b, %out, %off;
    .reg .b64 %va, %vb, %vr;
    .reg .pred %p;

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
    shl.b64 %off, %off, 3;
    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.b64 %va, [%a];
    ld.global.b64 %vb, [%b];
    and.b64 %vr, %va, %vb;
    st.global.b64 [%out], %vr;
DONE:
    ret;
}
";

const BITOR_I64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry bitor_i64_kernel(
    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %b, %out, %off;
    .reg .b64 %va, %vb, %vr;
    .reg .pred %p;

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
    shl.b64 %off, %off, 3;
    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.b64 %va, [%a];
    ld.global.b64 %vb, [%b];
    or.b64 %vr, %va, %vb;
    st.global.b64 [%out], %vr;
DONE:
    ret;
}
";

const BITXOR_I64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry bitxor_i64_kernel(
    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %b, %out, %off;
    .reg .b64 %va, %vb, %vr;
    .reg .pred %p;

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
    shl.b64 %off, %off, 3;
    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.b64 %va, [%a];
    ld.global.b64 %vb, [%b];
    xor.b64 %vr, %va, %vb;
    st.global.b64 [%out], %vr;
DONE:
    ret;
}
";

// shl (i64): shift count is the low bits of the i64 in b[i]; we read the low
// 32 bits as the count (PyTorch shift amounts are small).
const SHL_I64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry shl_i64_kernel(
    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr, %sh;
    .reg .u64 %a, %b, %out, %off;
    .reg .b64 %va, %vr;
    .reg .s64 %vb;
    .reg .pred %p;

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
    shl.b64 %off, %off, 3;
    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.b64 %va, [%a];
    ld.global.s64 %vb, [%b];
    cvt.u32.u64 %sh, %vb;
    shl.b64 %vr, %va, %sh;
    st.global.b64 [%out], %vr;
DONE:
    ret;
}
";

// shr (i64): ARITHMETIC right shift — shr.s64.
const SHR_I64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry shr_i64_kernel(
    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr, %sh;
    .reg .u64 %a, %b, %out, %off;
    .reg .s64 %va, %vb, %vr;
    .reg .pred %p;

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
    shl.b64 %off, %off, 3;
    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.s64 %va, [%a];
    ld.global.s64 %vb, [%b];
    cvt.u32.u64 %sh, %vb;
    shr.s64 %vr, %va, %sh;
    st.global.s64 [%out], %vr;
DONE:
    ret;
}
";

const NEG_I64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry neg_i64_kernel(
    .param .u64 a_ptr, .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %out, %off;
    .reg .s64 %va, %vr, %zero;
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

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 3;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.s64 %va, [%a];
    mov.s64 %zero, 0;
    sub.s64 %vr, %zero, %va;
    st.global.s64 [%out], %vr;
DONE:
    ret;
}
";

const BITNOT_I64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry bitnot_i64_kernel(
    .param .u64 a_ptr, .param .u64 out_ptr, .param .u32 n
) {
    .reg .u32 %idx, %bid, %bdim, %nr;
    .reg .u64 %a, %out, %off;
    .reg .b64 %va, %vr;
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

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 3;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.b64 %va, [%a];
    not.b64 %vr, %va;
    st.global.b64 [%out], %vr;
DONE:
    ret;
}
";

const REDUCE_I64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry reduce_i64_kernel(
    .param .u64 a_ptr, .param .u64 out_ptr, .param .u32 n, .param .u32 op
) {
    .reg .u32 %idx, %bid, %bdim, %nr, %op_r, %i;
    .reg .u64 %a, %out, %off, %cur;
    .reg .s64 %acc, %v, %prod;
    .reg .pred %p, %only0, %is_sum, %is_prod, %is_min, %lt, %gt;

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

    setp.eq.u32 %is_sum, %op_r, 0;
    setp.eq.u32 %is_prod, %op_r, 1;
    setp.eq.u32 %is_min, %op_r, 2;

    ld.global.s64 %acc, [%a];
    mov.u32 %i, 1;
LOOP:
    setp.ge.u32 %p, %i, %nr;
    @%p bra STORE;

    cvt.u64.u32 %off, %i;
    shl.b64 %off, %off, 3;
    add.u64 %cur, %a, %off;
    ld.global.s64 %v, [%cur];

    @%is_sum add.s64 %acc, %acc, %v;
    mul.lo.s64 %prod, %acc, %v;
    @%is_prod mov.s64 %acc, %prod;
    setp.lt.s64 %lt, %v, %acc;
    @%is_min selp.s64 %acc, %v, %acc, %lt;
    setp.gt.s64 %gt, %v, %acc;
    @%is_sum bra SKIPMAX;
    @%is_prod bra SKIPMAX;
    @%is_min bra SKIPMAX;
    selp.s64 %acc, %v, %acc, %gt;
SKIPMAX:

    add.u32 %i, %i, 1;
    bra LOOP;
STORE:
    st.global.s64 [%out], %acc;
DONE:
    ret;
}
";

// ===========================================================================
// Launch harness (generic over the native integer element type `T`)
// ===========================================================================

const REDUCE_SUM: u32 = 0;
const REDUCE_PROD: u32 = 1;
const REDUCE_MIN: u32 = 2;
const REDUCE_MAX: u32 = 3;

/// Launch a binary `(a_ptr, b_ptr, out_ptr, n)` integer kernel. `T` is the
/// native on-device element type (`i32` / `i64`); `out` is a fresh allocation
/// of `n` `T`-elements that stays resident on `device`.
fn launch_binary<T: DeviceRepr + ValidAsZeroBits>(
    a: &CudaSlice<T>,
    b: &CudaSlice<T>,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
) -> GpuResult<CudaSlice<T>> {
    if a.len() != b.len() {
        return Err(GpuError::LengthMismatch {
            a: a.len(),
            b: b.len(),
        });
    }
    let n = a.len();
    let stream = device.stream();
    if n == 0 {
        return Ok(stream.alloc_zeros::<T>(0)?);
    }
    let ctx = device.context();
    let f = get_or_compile(ctx, ptx, kernel_name, device.ordinal() as u32).map_err(|e| {
        GpuError::PtxCompileFailed {
            kernel: kernel_name,
            source: e,
        }
    })?;
    let mut out = stream.alloc_zeros::<T>(n)?;
    let cfg = launch_1d(n);
    let n_u32 = n as u32;
    // SAFETY:
    // - `f` is the PTX entry `kernel_name` just compiled from `ptx`; its
    //   signature is (a_ptr: u64, b_ptr: u64, out_ptr: u64, n: u32), matching
    //   the four args pushed below in order.
    // - `a`, `b` are immutable input buffers each of `n` `T`-elements
    //   (length equality enforced above); `n` is bound to `a.len()`.
    // - `out` was alloc'd `n` `T`-elements from `stream` and is the only
    //   `&mut` here, non-aliased with the immutable inputs.
    // - The kernel reads `a[i]`/`b[i]` and writes `out[i]` only within
    //   `[0, n)` per the `setp.ge.u32 %p, %idx, %nr` bound check.
    // - `n_u32` is non-truncating: `launch_1d` already cast `n as u32` to size
    //   the grid that covers `[0, n)`.
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

/// Launch a unary `(a_ptr, out_ptr, n)` integer kernel.
fn launch_unary<T: DeviceRepr + ValidAsZeroBits>(
    a: &CudaSlice<T>,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
) -> GpuResult<CudaSlice<T>> {
    let n = a.len();
    let stream = device.stream();
    if n == 0 {
        return Ok(stream.alloc_zeros::<T>(0)?);
    }
    let ctx = device.context();
    let f = get_or_compile(ctx, ptx, kernel_name, device.ordinal() as u32).map_err(|e| {
        GpuError::PtxCompileFailed {
            kernel: kernel_name,
            source: e,
        }
    })?;
    let mut out = stream.alloc_zeros::<T>(n)?;
    let cfg = launch_1d(n);
    let n_u32 = n as u32;
    // SAFETY:
    // - `f` resolves to the unary PTX entry `(a_ptr, out_ptr, n)`.
    // - `a` is the caller's input of `n` `T`-elements; the bound check limits
    //   reads to `[0, n)`.
    // - `out` is freshly alloc'd `n` elements, exclusively borrowed here.
    // - `n_u32` is non-truncating (see `launch_binary`).
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

/// Launch the serial reduction kernel `(a_ptr, out_ptr, n, op)`, returning a
/// 1-element resident buffer. `op` selects sum/prod/min/max.
fn launch_reduce<T: DeviceRepr + ValidAsZeroBits>(
    a: &CudaSlice<T>,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
    op: u32,
    empty_identity: T,
) -> GpuResult<CudaSlice<T>> {
    let n = a.len();
    let stream = device.stream();
    if n == 0 {
        // PyTorch: sum/prod of empty = identity (0 / 1); min/max of empty
        // raises on CPU but here we surface the identity in the 1-element
        // buffer (the IntTensor wrapper guards empties before calling).
        let host = [empty_identity];
        return Ok(stream.clone_htod(&host)?);
    }
    let ctx = device.context();
    let f = get_or_compile(ctx, ptx, kernel_name, device.ordinal() as u32).map_err(|e| {
        GpuError::PtxCompileFailed {
            kernel: kernel_name,
            source: e,
        }
    })?;
    let mut out = stream.alloc_zeros::<T>(1)?;
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
    // - `a` is the caller's input of `n` `T`-elements; the kernel reads
    //   `a[0..n)` from thread 0 only (`setp.ne.u32 %only0, %idx, 0` gates the
    //   rest off) and writes the single `out[0]`.
    // - `out` is a freshly alloc'd 1-element buffer, exclusively borrowed.
    // - `op` is one of {0,1,2,3} (the REDUCE_* constants).
    // - `n_u32` is non-truncating for any `n` the host can allocate as a
    //   contiguous integer buffer.
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

// ---------------------------------------------------------------------------
// Public i32 entry points
// ---------------------------------------------------------------------------

/// Elementwise `out = a + b` (i32, on-device, wrapping on overflow).
pub fn gpu_add_i32(
    a: &CudaSlice<i32>,
    b: &CudaSlice<i32>,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i32>> {
    launch_binary(a, b, d, ADD_I32_PTX, "add_i32_kernel")
}
/// Elementwise `out = a - b` (i32).
pub fn gpu_sub_i32(
    a: &CudaSlice<i32>,
    b: &CudaSlice<i32>,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i32>> {
    launch_binary(a, b, d, SUB_I32_PTX, "sub_i32_kernel")
}
/// Elementwise `out = a * b` (i32, wrapping).
pub fn gpu_mul_i32(
    a: &CudaSlice<i32>,
    b: &CudaSlice<i32>,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i32>> {
    launch_binary(a, b, d, MUL_I32_PTX, "mul_i32_kernel")
}
/// Elementwise floor division `out = floor_divide(a, b)` (i32, floors to −∞).
pub fn gpu_floor_div_i32(
    a: &CudaSlice<i32>,
    b: &CudaSlice<i32>,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i32>> {
    launch_binary(a, b, d, FLOORDIV_I32_PTX, "floordiv_i32_kernel")
}
/// Elementwise `out = remainder(a, b)` (i32, sign of divisor).
pub fn gpu_remainder_i32(
    a: &CudaSlice<i32>,
    b: &CudaSlice<i32>,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i32>> {
    launch_binary(a, b, d, REMAINDER_I32_PTX, "remainder_i32_kernel")
}
/// Elementwise bitwise AND (i32).
pub fn gpu_bitand_i32(
    a: &CudaSlice<i32>,
    b: &CudaSlice<i32>,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i32>> {
    launch_binary(a, b, d, BITAND_I32_PTX, "bitand_i32_kernel")
}
/// Elementwise bitwise OR (i32).
pub fn gpu_bitor_i32(
    a: &CudaSlice<i32>,
    b: &CudaSlice<i32>,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i32>> {
    launch_binary(a, b, d, BITOR_I32_PTX, "bitor_i32_kernel")
}
/// Elementwise bitwise XOR (i32).
pub fn gpu_bitxor_i32(
    a: &CudaSlice<i32>,
    b: &CudaSlice<i32>,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i32>> {
    launch_binary(a, b, d, BITXOR_I32_PTX, "bitxor_i32_kernel")
}
/// Elementwise left shift `out = a << b` (i32).
pub fn gpu_shl_i32(
    a: &CudaSlice<i32>,
    b: &CudaSlice<i32>,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i32>> {
    launch_binary(a, b, d, SHL_I32_PTX, "shl_i32_kernel")
}
/// Elementwise arithmetic right shift `out = a >> b` (i32, sign-extending).
pub fn gpu_shr_i32(
    a: &CudaSlice<i32>,
    b: &CudaSlice<i32>,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i32>> {
    launch_binary(a, b, d, SHR_I32_PTX, "shr_i32_kernel")
}
/// Elementwise negate `out = -a` (i32).
pub fn gpu_neg_i32(a: &CudaSlice<i32>, d: &GpuDevice) -> GpuResult<CudaSlice<i32>> {
    launch_unary(a, d, NEG_I32_PTX, "neg_i32_kernel")
}
/// Elementwise bitwise NOT `out = !a` (i32).
pub fn gpu_bitnot_i32(a: &CudaSlice<i32>, d: &GpuDevice) -> GpuResult<CudaSlice<i32>> {
    launch_unary(a, d, BITNOT_I32_PTX, "bitnot_i32_kernel")
}
/// Sum-reduce to a 1-element buffer (i32, wrapping accumulator).
pub fn gpu_sum_i32(a: &CudaSlice<i32>, d: &GpuDevice) -> GpuResult<CudaSlice<i32>> {
    launch_reduce(a, d, REDUCE_I32_PTX, "reduce_i32_kernel", REDUCE_SUM, 0)
}
/// Product-reduce to a 1-element buffer (i32, wrapping accumulator).
pub fn gpu_prod_i32(a: &CudaSlice<i32>, d: &GpuDevice) -> GpuResult<CudaSlice<i32>> {
    launch_reduce(a, d, REDUCE_I32_PTX, "reduce_i32_kernel", REDUCE_PROD, 1)
}
/// Min-reduce to a 1-element buffer (i32).
pub fn gpu_min_i32(a: &CudaSlice<i32>, d: &GpuDevice) -> GpuResult<CudaSlice<i32>> {
    launch_reduce(
        a,
        d,
        REDUCE_I32_PTX,
        "reduce_i32_kernel",
        REDUCE_MIN,
        i32::MAX,
    )
}
/// Max-reduce to a 1-element buffer (i32).
pub fn gpu_max_i32(a: &CudaSlice<i32>, d: &GpuDevice) -> GpuResult<CudaSlice<i32>> {
    launch_reduce(
        a,
        d,
        REDUCE_I32_PTX,
        "reduce_i32_kernel",
        REDUCE_MAX,
        i32::MIN,
    )
}

// ---------------------------------------------------------------------------
// Public i64 entry points
// ---------------------------------------------------------------------------

/// Elementwise `out = a + b` (i64, wrapping).
pub fn gpu_add_i64(
    a: &CudaSlice<i64>,
    b: &CudaSlice<i64>,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i64>> {
    launch_binary(a, b, d, ADD_I64_PTX, "add_i64_kernel")
}
/// Elementwise `out = a - b` (i64).
pub fn gpu_sub_i64(
    a: &CudaSlice<i64>,
    b: &CudaSlice<i64>,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i64>> {
    launch_binary(a, b, d, SUB_I64_PTX, "sub_i64_kernel")
}
/// Elementwise `out = a * b` (i64, wrapping).
pub fn gpu_mul_i64(
    a: &CudaSlice<i64>,
    b: &CudaSlice<i64>,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i64>> {
    launch_binary(a, b, d, MUL_I64_PTX, "mul_i64_kernel")
}
/// Elementwise floor division (i64, floors to −∞).
pub fn gpu_floor_div_i64(
    a: &CudaSlice<i64>,
    b: &CudaSlice<i64>,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i64>> {
    launch_binary(a, b, d, FLOORDIV_I64_PTX, "floordiv_i64_kernel")
}
/// Elementwise remainder (i64, sign of divisor).
pub fn gpu_remainder_i64(
    a: &CudaSlice<i64>,
    b: &CudaSlice<i64>,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i64>> {
    launch_binary(a, b, d, REMAINDER_I64_PTX, "remainder_i64_kernel")
}
/// Elementwise bitwise AND (i64).
pub fn gpu_bitand_i64(
    a: &CudaSlice<i64>,
    b: &CudaSlice<i64>,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i64>> {
    launch_binary(a, b, d, BITAND_I64_PTX, "bitand_i64_kernel")
}
/// Elementwise bitwise OR (i64).
pub fn gpu_bitor_i64(
    a: &CudaSlice<i64>,
    b: &CudaSlice<i64>,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i64>> {
    launch_binary(a, b, d, BITOR_I64_PTX, "bitor_i64_kernel")
}
/// Elementwise bitwise XOR (i64).
pub fn gpu_bitxor_i64(
    a: &CudaSlice<i64>,
    b: &CudaSlice<i64>,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i64>> {
    launch_binary(a, b, d, BITXOR_I64_PTX, "bitxor_i64_kernel")
}
/// Elementwise left shift (i64).
pub fn gpu_shl_i64(
    a: &CudaSlice<i64>,
    b: &CudaSlice<i64>,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i64>> {
    launch_binary(a, b, d, SHL_I64_PTX, "shl_i64_kernel")
}
/// Elementwise arithmetic right shift (i64, sign-extending).
pub fn gpu_shr_i64(
    a: &CudaSlice<i64>,
    b: &CudaSlice<i64>,
    d: &GpuDevice,
) -> GpuResult<CudaSlice<i64>> {
    launch_binary(a, b, d, SHR_I64_PTX, "shr_i64_kernel")
}
/// Elementwise negate (i64).
pub fn gpu_neg_i64(a: &CudaSlice<i64>, d: &GpuDevice) -> GpuResult<CudaSlice<i64>> {
    launch_unary(a, d, NEG_I64_PTX, "neg_i64_kernel")
}
/// Elementwise bitwise NOT (i64).
pub fn gpu_bitnot_i64(a: &CudaSlice<i64>, d: &GpuDevice) -> GpuResult<CudaSlice<i64>> {
    launch_unary(a, d, BITNOT_I64_PTX, "bitnot_i64_kernel")
}
/// Sum-reduce (i64, wrapping accumulator).
pub fn gpu_sum_i64(a: &CudaSlice<i64>, d: &GpuDevice) -> GpuResult<CudaSlice<i64>> {
    launch_reduce(a, d, REDUCE_I64_PTX, "reduce_i64_kernel", REDUCE_SUM, 0)
}
/// Product-reduce (i64, wrapping accumulator).
pub fn gpu_prod_i64(a: &CudaSlice<i64>, d: &GpuDevice) -> GpuResult<CudaSlice<i64>> {
    launch_reduce(a, d, REDUCE_I64_PTX, "reduce_i64_kernel", REDUCE_PROD, 1)
}
/// Min-reduce (i64).
pub fn gpu_min_i64(a: &CudaSlice<i64>, d: &GpuDevice) -> GpuResult<CudaSlice<i64>> {
    launch_reduce(
        a,
        d,
        REDUCE_I64_PTX,
        "reduce_i64_kernel",
        REDUCE_MIN,
        i64::MAX,
    )
}
/// Max-reduce (i64).
pub fn gpu_max_i64(a: &CudaSlice<i64>, d: &GpuDevice) -> GpuResult<CudaSlice<i64>> {
    launch_reduce(
        a,
        d,
        REDUCE_I64_PTX,
        "reduce_i64_kernel",
        REDUCE_MAX,
        i64::MIN,
    )
}
