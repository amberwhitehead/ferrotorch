//! Native IEEE float16 (`half::f16`) GPU kernels.
//!
//! Hand-written PTX owned by Rust: no CUDA C++ source, no nvrtc runtime
//! compiler, no external toolchain at load time. Each kernel is a
//! `&'static str` containing PTX 7.0 targeting sm_52+ (sm_53 for the
//! `gelu` kernel, which uses an `abs.f32` on the half path), loaded via
//! `cudarc::driver::CudaContext::load_module` through the existing
//! `module_cache::get_or_compile` path.
//!
//! # The f16 pattern vs. bf16
//!
//! f16 and bf16 are BOTH 2 bytes and BOTH stored on-device as
//! `CudaSlice<u16>`. They are disambiguated *only* by the
//! `GpuBufferHandle` `DType` tag (`DType::F16` vs `DType::BF16`) — see
//! `backend_impl::unwrap_buffer_f16`. The numerical difference is the
//! bit layout: f16 is a true IEEE-754 binary16 (1 sign / 5 exponent /
//! 10 mantissa), bf16 is the top 16 bits of an f32 (1 / 8 / 7).
//!
//! Because f16 is a genuine IEEE format, PTX provides **native**
//! conversion instructions, which the bf16 module has to hand-roll:
//!
//! - **f16 → f32**: `cvt.f32.f16 %f, %h` where `%h` is a `.b16` and `%f`
//!   is `.f32`. Lossless. (bf16 had to do `mov.b32 %u, {0, %h}`.)
//! - **f32 → f16, round-to-nearest-even**: `cvt.rn.f16.f32 %h, %f`.
//!   (bf16 had to add the rounding bias `0x7FFF + bit[16]` then shift.)
//!
//! All arithmetic happens in `.f32` registers per thread; storage is
//! always `u16` (`.b16`) in global memory. No whole-tensor f32
//! intermediate materialisation. Reductions accumulate in f32 (PyTorch
//! parity for half-precision reductions).
//!
//! ## REQ status (per `.design/ferrotorch-gpu/f16.md`)
//!
//! Full evidence rows (impl + non-test production consumer + upstream
//! cites) live in the design doc; this synopsis is a one-line summary per
//! REQ.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (elementwise binary) | SHIPPED | `pub fn gpu_mul_f16 / gpu_add_f16 / gpu_sub_f16 / gpu_div_f16 in f16.rs`; consumer `CudaBackendImpl::add_f16 / sub_f16 / mul_f16 / div_f16 in backend_impl.rs` |
//! | REQ-2 (unary activations) | SHIPPED | `pub fn gpu_silu_f16 / gpu_relu_f16 / gpu_gelu_f16 / gpu_exp_f16 / gpu_log_f16 / gpu_tanh_f16 / gpu_sigmoid_f16 / gpu_sqrt_f16 / gpu_neg_f16 in f16.rs`; consumer the f16 arm of activation dispatchers in `backend_impl.rs` |
//! | REQ-3 (`gpu_scale_f16` / `gpu_fill_f16`) | SHIPPED | `pub fn gpu_scale_f16 / gpu_fill_f16 in f16.rs`; consumers `CudaBackendImpl::scale_f16 / fill_f16 in backend_impl.rs` |
//! | REQ-4 (broadcast binary) | SHIPPED | `pub fn gpu_broadcast_add_f16 / gpu_broadcast_sub_f16 / gpu_broadcast_mul_f16 / gpu_broadcast_div_f16 in f16.rs`; consumer the f16 broadcast arms in `backend_impl.rs` |
//! | REQ-5 (reductions) | SHIPPED | `pub fn gpu_sum_f16 / gpu_mean_f16 / gpu_prod_f16 / gpu_sum_axis_f16 / gpu_mean_axis_f16 / gpu_prod_axis_f16 in f16.rs`; consumer f16 reduction arms in `backend_impl.rs` |
//! | REQ-6 (norm/softmax) | SHIPPED | `pub fn gpu_layernorm_f16 / gpu_rmsnorm_f16 / gpu_softmax_f16 in f16.rs`; consumer f16 softmax/layernorm/rmsnorm dispatchers in `backend_impl.rs` |
//! | REQ-7 (SAFETY annotations) | SHIPPED | every `unsafe { ... launch(cfg)? }` in `f16.rs` carries a multi-line `SAFETY:` comment naming compile site, entry signature, buffer alloc, bound check, `n as u32` non-truncation; consumer SAFETY contract inherited via each public wrapper called from `backend_impl.rs` |
//! | REQ-8 (empty-input short-circuit) | SHIPPED | every `pub fn gpu_*_f16 in f16.rs` opens with `if n == 0 { return Ok(stream.alloc_zeros::<u16>(0)?); }`; consumer backend dispatch path relies on the no-launch short circuit |

#![cfg(feature = "cuda")]

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};
use crate::module_cache::get_or_compile;

const BLOCK_SIZE: u32 = 256;

// ===========================================================================
// Elementwise binary kernels (add, sub, mul, div)
// ===========================================================================

const MUL_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry mul_f16_kernel(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %b, %out, %off;
    .reg .b16 %a_b16, %b_b16, %out_h;
    .reg .f32 %va, %vb, %vr;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.b16 %a_b16, [%a];
    ld.global.b16 %b_b16, [%b];
    cvt.f32.f16 %va, %a_b16;
    cvt.f32.f16 %vb, %b_b16;

    mul.f32 %vr, %va, %vb;

    cvt.rn.f16.f32 %out_h, %vr;
    st.global.b16 [%out], %out_h;

DONE:
    ret;
}
";

const ADD_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry add_f16_kernel(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %b, %out, %off;
    .reg .b16 %a_b16, %b_b16, %out_h;
    .reg .f32 %va, %vb, %vr;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.b16 %a_b16, [%a];
    ld.global.b16 %b_b16, [%b];
    cvt.f32.f16 %va, %a_b16;
    cvt.f32.f16 %vb, %b_b16;

    add.f32 %vr, %va, %vb;

    cvt.rn.f16.f32 %out_h, %vr;
    st.global.b16 [%out], %out_h;

DONE:
    ret;
}
";

const SUB_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry sub_f16_kernel(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %b, %out, %off;
    .reg .b16 %a_b16, %b_b16, %out_h;
    .reg .f32 %va, %vb, %vr;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.b16 %a_b16, [%a];
    ld.global.b16 %b_b16, [%b];
    cvt.f32.f16 %va, %a_b16;
    cvt.f32.f16 %vb, %b_b16;

    sub.f32 %vr, %va, %vb;

    cvt.rn.f16.f32 %out_h, %vr;
    st.global.b16 [%out], %out_h;

DONE:
    ret;
}
";

const DIV_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry div_f16_kernel(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %b, %out, %off;
    .reg .b16 %a_b16, %b_b16, %out_h;
    .reg .f32 %va, %vb, %vr, %nan;
    .reg .pred %p, %nan_a, %nan_b, %store_nan;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.b16 %a_b16, [%a];
    ld.global.b16 %b_b16, [%b];
    cvt.f32.f16 %va, %a_b16;
    cvt.f32.f16 %vb, %b_b16;

    setp.nan.f32 %nan_a, %va, %va;
    setp.nan.f32 %nan_b, %vb, %vb;
    or.pred %store_nan, %nan_a, %nan_b;
    @%store_nan bra STORE_NAN;

    div.rn.f32 %vr, %va, %vb;

    cvt.rn.f16.f32 %out_h, %vr;
    st.global.b16 [%out], %out_h;
    bra DONE;

STORE_NAN:
    mov.f32 %nan, 0f7FC00000;
    cvt.rn.f16.f32 %out_h, %nan;
    st.global.b16 [%out], %out_h;

DONE:
    ret;
}
";

const CROSS_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry cross_f16_kernel(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u32 stride_axis,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %stride, %tmp, %coord, %base, %idx0, %idx1, %idx2, %two_stride;
    .reg .u64 %a, %b, %out, %off, %addr;
    .reg .b16 %a0_h, %a1_h, %a2_h, %b0_h, %b1_h, %b2_h, %out_h;
    .reg .f32 %a0, %a1, %a2, %b0, %b1, %b2, %m0, %m1, %res;
    .reg .pred %p, %is0, %is1;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %stride, [stride_axis];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    div.u32 %tmp, %r_tid, %stride;
    rem.u32 %coord, %tmp, 3;
    mul.lo.u32 %tmp, %coord, %stride;
    sub.u32 %base, %r_tid, %tmp;
    add.u32 %idx0, %base, 0;
    add.u32 %idx1, %base, %stride;
    add.u32 %two_stride, %stride, %stride;
    add.u32 %idx2, %base, %two_stride;

    cvt.u64.u32 %off, %idx0;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %a, %off;
    ld.global.b16 %a0_h, [%addr];
    add.u64 %addr, %b, %off;
    ld.global.b16 %b0_h, [%addr];
    cvt.f32.f16 %a0, %a0_h;
    cvt.f32.f16 %b0, %b0_h;

    cvt.u64.u32 %off, %idx1;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %a, %off;
    ld.global.b16 %a1_h, [%addr];
    add.u64 %addr, %b, %off;
    ld.global.b16 %b1_h, [%addr];
    cvt.f32.f16 %a1, %a1_h;
    cvt.f32.f16 %b1, %b1_h;

    cvt.u64.u32 %off, %idx2;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %a, %off;
    ld.global.b16 %a2_h, [%addr];
    add.u64 %addr, %b, %off;
    ld.global.b16 %b2_h, [%addr];
    cvt.f32.f16 %a2, %a2_h;
    cvt.f32.f16 %b2, %b2_h;

    setp.eq.u32 %is0, %coord, 0;
    @%is0 bra COORD0;
    setp.eq.u32 %is1, %coord, 1;
    @%is1 bra COORD1;

COORD2:
    mul.f32 %m0, %a0, %b1;
    mul.f32 %m1, %a1, %b0;
    sub.f32 %res, %m0, %m1;
    bra STORE;

COORD0:
    mul.f32 %m0, %a1, %b2;
    mul.f32 %m1, %a2, %b1;
    sub.f32 %res, %m0, %m1;
    bra STORE;

COORD1:
    mul.f32 %m0, %a2, %b0;
    mul.f32 %m1, %a0, %b2;
    sub.f32 %res, %m0, %m1;

STORE:
    cvt.rn.f16.f32 %out_h, %res;
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %out, %off;
    st.global.b16 [%addr], %out_h;

DONE:
    ret;
}
";

const ABS_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry abs_f16_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %bits;
    .reg .u64 %a, %out, %off;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.u16 %bits, [%a];
    and.b32 %bits, %bits, 0x7fff;
    st.global.u16 [%out], %bits;

DONE:
    ret;
}
";

const ABS_BACKWARD_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry abs_backward_f16_kernel(
    .param .u64 grad_ptr,
    .param .u64 input_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u32 %grad_bits, %input_bits, %input_mag, %input_sign;
    .reg .u64 %grad, %input, %out, %off;
    .reg .pred %p, %is_zero, %is_nan;

    ld.param.u64 %grad, [grad_ptr];
    ld.param.u64 %input, [input_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %grad, %grad, %off;
    add.u64 %input, %input, %off;
    add.u64 %out, %out, %off;

    ld.global.u16 %grad_bits, [%grad];
    ld.global.u16 %input_bits, [%input];
    and.b32 %input_mag, %input_bits, 0x7fff;
    setp.eq.u32 %is_zero, %input_mag, 0;
    @%is_zero bra STORE_ZERO;
    setp.gt.u32 %is_nan, %input_mag, 0x7c00;
    @%is_nan bra STORE_ZERO;

    and.b32 %input_sign, %input_bits, 0x8000;
    xor.b32 %grad_bits, %grad_bits, %input_sign;
    st.global.u16 [%out], %grad_bits;
    bra DONE;

STORE_ZERO:
    st.global.u16 [%out], 0;

DONE:
    ret;
}
";

// ===========================================================================
// Elementwise unary activation kernels (silu, relu, gelu)
// ===========================================================================

const SILU_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry silu_f16_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .b16 %a_b16, %out_h;
    .reg .f32 %va, %neg_a, %log2e, %x, %e, %one, %denom, %sig, %vr;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.b16 %a_b16, [%a];
    cvt.f32.f16 %va, %a_b16;

    neg.f32 %neg_a, %va;
    mov.f32 %log2e, 0f3FB8AA3B;
    mul.f32 %x, %neg_a, %log2e;
    ex2.approx.f32 %e, %x;
    mov.f32 %one, 0f3F800000;
    add.f32 %denom, %one, %e;
    div.approx.f32 %sig, %one, %denom;
    mul.f32 %vr, %va, %sig;

    cvt.rn.f16.f32 %out_h, %vr;
    st.global.b16 [%out], %out_h;

DONE:
    ret;
}
";

const RELU_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry relu_f16_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .b16 %a_b16, %out_h;
    .reg .f32 %va, %zero, %vr;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.b16 %a_b16, [%a];
    cvt.f32.f16 %va, %a_b16;

    mov.f32 %zero, 0f00000000;
    max.f32 %vr, %va, %zero;

    cvt.rn.f16.f32 %out_h, %vr;
    st.global.b16 [%out], %out_h;

DONE:
    ret;
}
";

const CLAMP_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry clamp_f16_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n,
    .param .f32 min_val,
    .param .f32 max_val
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %in, %out, %off;
    .reg .b16 %x_h, %out_h;
    .reg .f32 %x, %mn, %mx, %result;
    .reg .pred %p;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];
    ld.param.f32 %mn, [min_val];
    ld.param.f32 %mx, [max_val];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %in, %in, %off;
    add.u64 %out, %out, %off;

    ld.global.b16 %x_h, [%in];
    cvt.f32.f16 %x, %x_h;
    max.f32 %result, %x, %mn;
    min.f32 %result, %result, %mx;
    cvt.rn.f16.f32 %out_h, %result;
    st.global.b16 [%out], %out_h;

DONE:
    ret;
}
";

const CLAMP_BACKWARD_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry clamp_backward_f16_kernel(
    .param .u64 grad_ptr,
    .param .u64 input_ptr,
    .param .u64 out_ptr,
    .param .f32 min_val,
    .param .f32 max_val,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %g, %x, %out, %off;
    .reg .b16 %g_h, %x_h, %out_h;
    .reg .f32 %vg, %vx, %vmin, %vmax, %vr;
    .reg .pred %p, %plo, %phi, %pin;

    ld.param.u64 %g, [grad_ptr];
    ld.param.u64 %x, [input_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.f32 %vmin, [min_val];
    ld.param.f32 %vmax, [max_val];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %g, %g, %off;
    add.u64 %x, %x, %off;
    add.u64 %out, %out, %off;

    ld.global.b16 %g_h, [%g];
    ld.global.b16 %x_h, [%x];
    cvt.f32.f16 %vg, %g_h;
    cvt.f32.f16 %vx, %x_h;

    setp.ge.f32 %plo, %vx, %vmin;
    setp.le.f32 %phi, %vx, %vmax;
    and.pred %pin, %plo, %phi;

    mov.f32 %vr, 0f00000000;
    @%pin mov.f32 %vr, %vg;
    cvt.rn.f16.f32 %out_h, %vr;
    st.global.b16 [%out], %out_h;

DONE:
    ret;
}
";

// GELU: 0.5 * x * (1 + erf(x / sqrt(2))). Hastings erf approximation in f32.
const GELU_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry gelu_f16_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %in, %out, %off;
    .reg .b16 %x_b16, %out_h;
    .reg .f32 %x, %inv_sqrt2, %arg, %erf_v, %one, %half_c, %sum, %y;
    .reg .f32 %ax, %t, %p_const, %one2, %neg_xx, %log2e, %exp_v, %poly, %c_a1, %c_a2, %c_a3, %c_a4, %c_a5;
    .reg .pred %p, %signp;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %in, %in, %off;
    add.u64 %out, %out, %off;

    ld.global.b16 %x_b16, [%in];
    cvt.f32.f16 %x, %x_b16;

    // arg = x / sqrt(2); 1/sqrt(2) = 0.70710678118f -> 0x3F3504F3
    mov.f32 %inv_sqrt2, 0f3F3504F3;
    mul.f32 %arg, %x, %inv_sqrt2;

    // ax = |arg|
    abs.f32 %ax, %arg;

    // t = 1 / (1 + 0.3275911 * ax); p = 0.3275911 -> 0x3EA7BA05
    mov.f32 %p_const, 0f3EA7BA05;
    mov.f32 %one2, 0f3F800000;
    fma.rn.f32 %t, %p_const, %ax, %one2;
    rcp.approx.f32 %t, %t;

    // exp_v = exp(-ax*ax) computed as 2^((-ax*ax) * log2(e)).
    mul.f32 %neg_xx, %ax, %ax;
    neg.f32 %neg_xx, %neg_xx;
    mov.f32 %log2e, 0f3FB8AA3B;
    mul.f32 %neg_xx, %neg_xx, %log2e;
    ex2.approx.f32 %exp_v, %neg_xx;

    // Horner-eval poly = ((((a5*t + a4)*t + a3)*t + a2)*t + a1) * t
    mov.f32 %c_a5, 0f3F87DC22;
    mov.f32 %c_a4, 0fBFBA00E3;
    mov.f32 %c_a3, 0f3FB5F0E3;
    mov.f32 %c_a2, 0fBE91A98E;
    mov.f32 %c_a1, 0f3E827906;
    mul.f32 %poly, %c_a5, %t;
    add.f32 %poly, %poly, %c_a4;
    mul.f32 %poly, %poly, %t;
    add.f32 %poly, %poly, %c_a3;
    mul.f32 %poly, %poly, %t;
    add.f32 %poly, %poly, %c_a2;
    mul.f32 %poly, %poly, %t;
    add.f32 %poly, %poly, %c_a1;
    mul.f32 %poly, %poly, %t;

    // erf(|arg|) = 1 - poly * exp_v
    mul.f32 %poly, %poly, %exp_v;
    mov.f32 %one, 0f3F800000;
    sub.f32 %erf_v, %one, %poly;

    // Restore sign.
    setp.lt.f32 %signp, %arg, 0f00000000;
    @%signp neg.f32 %erf_v, %erf_v;

    // y = 0.5 * x * (1 + erf_v)
    add.f32 %sum, %one, %erf_v;
    mov.f32 %half_c, 0f3F000000;
    mul.f32 %y, %half_c, %x;
    mul.f32 %y, %y, %sum;

    cvt.rn.f16.f32 %out_h, %y;
    st.global.b16 [%out], %out_h;

DONE:
    ret;
}
";

fn launch_1d(n: usize) -> LaunchConfig {
    let grid = ((n as u32).saturating_add(BLOCK_SIZE - 1)) / BLOCK_SIZE;
    LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn launch_binary(
    a: &cudarc::driver::CudaSlice<u16>,
    b: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    if a.len() != b.len() {
        return Err(GpuError::LengthMismatch {
            a: a.len(),
            b: b.len(),
        });
    }
    let n = a.len();
    if n == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(0)?);
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile(ctx, ptx, kernel_name, device.ordinal() as u32).map_err(|e| {
        GpuError::PtxCompileFailed {
            kernel: kernel_name,
            source: e,
        }
    })?;

    let mut out = stream.alloc_zeros::<u16>(n)?;
    let cfg = launch_1d(n);
    let n_u32 = n as u32;
    // SAFETY:
    // - `f` is a valid PTX function compiled just above via
    //   `module_cache::get_or_compile` from `ptx`; the corresponding entry
    //   point (`add_f16_kernel` / `sub_f16_kernel` / `mul_f16_kernel` /
    //   `div_f16_kernel`) has signature (a_ptr: u64, b_ptr: u64,
    //   out_ptr: u64, n: u32) which matches the four args pushed below.
    // - `a` and `b` are non-aliased input buffers each of length `n` u16
    //   elements; equality of `a.len()`/`b.len()` is enforced by the
    //   `LengthMismatch` guard, and `n` is bound to `a.len()`.
    // - `out` was alloc'd with exactly `n` u16 elements from `stream`;
    //   it is exclusively owned by this scope and bound to `stream` until
    //   the launch completes.
    // - The kernel reads `a[i]`/`b[i]` and writes `out[i]` only within
    //   `[0, n)` per the PTX bound-check `setp.ge.u32 %p, %r_tid, %n_reg`.
    // - `n` fits in u32 because `launch_1d` already cast `n as u32` to
    //   compute the grid, so the `n_u32` cast is non-truncating for any
    //   `n` the grid covered.
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

// Helper that launches a unary u16-in / u16-out PTX with a `(a_ptr, out_ptr, n)`
// signature. silu/relu/gelu/exp/log/tanh/sigmoid all share this shape.
fn launch_unary(
    a: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    let n = a.len();
    if n == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(0)?);
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile(ctx, ptx, kernel_name, device.ordinal() as u32).map_err(|e| {
        GpuError::PtxCompileFailed {
            kernel: kernel_name,
            source: e,
        }
    })?;
    let mut out = stream.alloc_zeros::<u16>(n)?;
    let cfg = launch_1d(n);
    let n_u32 = n as u32;
    // SAFETY:
    // - `f` resolves to a PTX entry with the unary `(a_ptr, out_ptr, n)`
    //   signature shared by the f16 unary kernels in this module.
    // - `a` is the caller's f16 input buffer of length `n`; the PTX
    //   bound-check ensures only `[0, n)` is read.
    // - `out` is freshly alloc'd above with `n` elements and exclusively
    //   bound here.
    // - `n_u32` is non-truncating because `launch_1d(n)` already cast
    //   `n as u32` to compute the grid.
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

fn launch_clamp(
    a: &cudarc::driver::CudaSlice<u16>,
    min_val: f32,
    max_val: f32,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    let n = a.len();
    if n == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(0)?);
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile(
        ctx,
        CLAMP_F16_PTX,
        "clamp_f16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "clamp_f16_kernel",
        source: e,
    })?;
    let mut out = stream.alloc_zeros::<u16>(n)?;
    let cfg = launch_1d(n);
    let n_u32 = n as u32;
    // SAFETY:
    // - `f` is the `clamp_f16_kernel` PTX entry with ABI
    //   `(in_ptr, out_ptr, n, min_val, max_val)`, matching the args below.
    // - `a` is an f16 u16 buffer of length `n`; `out` is freshly allocated
    //   with `n` u16 elements and cannot alias `a`.
    // - The kernel guards `i < n` before reading or writing; `n_u32` is the
    //   same cast used by `launch_1d`.
    // - Scalar args are stack f32 values whose lifetimes cover the launch call.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a)
            .arg(&mut out)
            .arg(&n_u32)
            .arg(&min_val)
            .arg(&max_val)
            .launch(cfg)?;
    }
    Ok(out)
}

fn launch_clamp_backward(
    grad: &cudarc::driver::CudaSlice<u16>,
    input: &cudarc::driver::CudaSlice<u16>,
    min_val: f32,
    max_val: f32,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    if grad.len() != input.len() {
        return Err(GpuError::LengthMismatch {
            a: grad.len(),
            b: input.len(),
        });
    }
    let n = input.len();
    if n == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(0)?);
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile(
        ctx,
        CLAMP_BACKWARD_F16_PTX,
        "clamp_backward_f16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "clamp_backward_f16_kernel",
        source: e,
    })?;
    let mut out = stream.alloc_zeros::<u16>(n)?;
    let cfg = launch_1d(n);
    let n_u32 = n as u32;
    // SAFETY:
    // - `f` is the `clamp_backward_f16_kernel` entry with ABI
    //   `(grad_ptr, input_ptr, out_ptr, min_val, max_val, n)`.
    // - `grad.len() == input.len() == n` is checked above; `out` is freshly
    //   allocated for `n` u16 elements and cannot alias either input.
    // - The PTX bounds-check guards every read/write with `i < n`.
    // - Scalar arg lifetimes cover the launch call, and `n_u32` matches
    //   `launch_1d`'s cast.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(grad)
            .arg(input)
            .arg(&mut out)
            .arg(&min_val)
            .arg(&max_val)
            .arg(&n_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Elementwise `out = a * b` on f16 (u16-stored) GPU buffers.
pub fn gpu_mul_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    b: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    launch_binary(a, b, device, MUL_F16_PTX, "mul_f16_kernel")
}

/// Elementwise `out = a + b` on f16 (u16-stored) GPU buffers.
pub fn gpu_add_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    b: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    launch_binary(a, b, device, ADD_F16_PTX, "add_f16_kernel")
}

/// Elementwise `out = a - b` on f16 (u16-stored) GPU buffers.
pub fn gpu_sub_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    b: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    launch_binary(a, b, device, SUB_F16_PTX, "sub_f16_kernel")
}

/// Same-shape `torch.linalg.cross` on f16 (u16-stored) GPU buffers.
///
/// Operands must already be broadcast/materialized to the same contiguous
/// output shape. `stride_axis` is the C-contiguous element stride for the
/// size-3 cross dimension.
pub fn gpu_cross_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    b: &cudarc::driver::CudaSlice<u16>,
    stride_axis: usize,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    if a.len() != b.len() {
        return Err(GpuError::LengthMismatch {
            a: a.len(),
            b: b.len(),
        });
    }
    let n = a.len();
    if n == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(0)?);
    }
    if stride_axis == 0 || stride_axis > u32::MAX as usize || n > u32::MAX as usize {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_cross_f16",
            expected: vec![1, u32::MAX as usize],
            got: vec![stride_axis, n],
        });
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile(
        ctx,
        CROSS_F16_PTX,
        "cross_f16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "cross_f16_kernel",
        source: e,
    })?;

    let mut out = stream.alloc_zeros::<u16>(n)?;
    let cfg = launch_1d(n);
    let stride_u32 = stride_axis as u32;
    let n_u32 = n as u32;
    // SAFETY:
    // - `f` is compiled from `CROSS_F16_PTX`; the entry signature is
    //   `(a_ptr, b_ptr, out_ptr, stride_axis, n)`.
    // - `a.len() == b.len() == n`, `out` is freshly allocated for `n`
    //   u16 f16 bit-patterns, and `n > 0` here.
    // - `stride_axis` is positive and both `stride_axis`/`n` fit in the
    //   PTX `.u32` parameters by the guard above.
    // - Caller validates `stride_axis` as the contiguous stride of a
    //   length-3 axis, so `base + 2 * stride_axis` remains in-bounds for
    //   every logical vector touched by a thread. The kernel also skips
    //   threads with `tid >= n`.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a)
            .arg(b)
            .arg(&mut out)
            .arg(&stride_u32)
            .arg(&n_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Elementwise `out = a / b` on f16 (u16-stored) GPU buffers.
pub fn gpu_div_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    b: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    launch_binary(a, b, device, DIV_F16_PTX, "div_f16_kernel")
}

/// Elementwise `out = abs(a)` on f16 (u16-stored) GPU buffers.
pub fn gpu_abs_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    launch_unary(a, device, ABS_F16_PTX, "abs_f16_kernel")
}

/// Backward for f16 `abs`: `out = grad * sign(input)`.
pub fn gpu_abs_backward_f16(
    grad: &cudarc::driver::CudaSlice<u16>,
    input: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    launch_binary(
        grad,
        input,
        device,
        ABS_BACKWARD_F16_PTX,
        "abs_backward_f16_kernel",
    )
}

/// Elementwise `out = silu(a) = a * sigmoid(a)` on f16 GPU buffers.
pub fn gpu_silu_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    launch_unary(a, device, SILU_F16_PTX, "silu_f16_kernel")
}

/// Elementwise `out = max(0, a)` on f16 GPU buffers.
pub fn gpu_relu_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    launch_unary(a, device, RELU_F16_PTX, "relu_f16_kernel")
}

/// Elementwise scalar clamp on f16 GPU buffers. Computed in f32 opmath and
/// rounded back to f16 with round-to-nearest-even.
pub fn gpu_clamp_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    min_val: f32,
    max_val: f32,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    launch_clamp(a, min_val, max_val, device)
}

/// Backward for scalar f16 clamp. Grad/input are f16 buffers; comparisons use
/// f32 opmath and output is rounded back to f16.
pub fn gpu_clamp_backward_f16(
    grad: &cudarc::driver::CudaSlice<u16>,
    input: &cudarc::driver::CudaSlice<u16>,
    min_val: f32,
    max_val: f32,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    launch_clamp_backward(grad, input, min_val, max_val, device)
}

/// Apply GELU activation `gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`
/// elementwise to an f16 GPU buffer. Computed in f32 (Hastings erf
/// approximation) then rounded back to f16 with round-to-nearest-even.
pub fn gpu_gelu_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    launch_unary(a, device, GELU_F16_PTX, "gelu_f16_kernel")
}

// ===========================================================================
// Elementwise unary transcendentals (exp, log, tanh, sigmoid)
// ===========================================================================

// exp via ex2.approx.f32 on x * log2(e); inputs/outputs round-trip through f16.
const EXP_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry exp_f16_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .b16 %a_b16, %out_h;
    .reg .f32 %va, %x, %log2e, %vr;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.b16 %a_b16, [%a];
    cvt.f32.f16 %va, %a_b16;

    // exp(x) = 2^(x * log2(e)); log2(e) = 0x3FB8AA3B as f32 bits.
    mov.f32 %log2e, 0f3FB8AA3B;
    mul.f32 %x, %va, %log2e;
    ex2.approx.f32 %vr, %x;

    cvt.rn.f16.f32 %out_h, %vr;
    st.global.b16 [%out], %out_h;

DONE:
    ret;
}
";

// log: ln(x) = lg2(x) * ln(2). Uses `lg2.approx.f32` + mul with ln(2) immediate.
const LOG_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry log_f16_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .b16 %a_b16, %out_h;
    .reg .f32 %va, %vr;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.b16 %a_b16, [%a];
    cvt.f32.f16 %va, %a_b16;

    // ln(x) = lg2(x) * ln(2); ln(2) = 0x3F317218.
    lg2.approx.f32 %vr, %va;
    mul.f32 %vr, %vr, 0f3F317218;

    cvt.rn.f16.f32 %out_h, %vr;
    st.global.b16 [%out], %out_h;

DONE:
    ret;
}
";

const SIN_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry sin_f16_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .b16 %a_b16, %out_h;
    .reg .f32 %va, %vr, %abs;
    .reg .pred %p, %is_nan, %is_inf, %special;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.b16 %a_b16, [%a];
    cvt.f32.f16 %va, %a_b16;
    setp.nan.f32 %is_nan, %va, %va;
    abs.f32 %abs, %va;
    setp.eq.f32 %is_inf, %abs, 0f7F800000;
    or.pred %special, %is_nan, %is_inf;
    @%special bra SIN_NAN;
    sin.approx.f32 %vr, %va;
    bra SIN_STORE;

SIN_NAN:
    mov.f32 %vr, 0f7FC00000;

SIN_STORE:
    cvt.rn.f16.f32 %out_h, %vr;
    st.global.b16 [%out], %out_h;

DONE:
    ret;
}
";

const COS_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry cos_f16_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .b16 %a_b16, %out_h;
    .reg .f32 %va, %vr, %abs;
    .reg .pred %p, %is_nan, %is_inf, %special;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.b16 %a_b16, [%a];
    cvt.f32.f16 %va, %a_b16;
    setp.nan.f32 %is_nan, %va, %va;
    abs.f32 %abs, %va;
    setp.eq.f32 %is_inf, %abs, 0f7F800000;
    or.pred %special, %is_nan, %is_inf;
    @%special bra COS_NAN;
    cos.approx.f32 %vr, %va;
    bra COS_STORE;

COS_NAN:
    mov.f32 %vr, 0f7FC00000;

COS_STORE:
    cvt.rn.f16.f32 %out_h, %vr;
    st.global.b16 [%out], %out_h;

DONE:
    ret;
}
";

// tanh(x) = (e^(2x) - 1) / (e^(2x) + 1). Uses ex2.approx.f32 internally.
const TANH_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry tanh_f16_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .b16 %a_b16, %out_h;
    .reg .f32 %va, %two_x, %log2e, %arg, %e, %num, %den, %vr, %one;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.b16 %a_b16, [%a];
    cvt.f32.f16 %va, %a_b16;

    // e^(2x) via ex2.approx.f32((2x) * log2(e)).
    add.f32 %two_x, %va, %va;
    mov.f32 %log2e, 0f3FB8AA3B;
    mul.f32 %arg, %two_x, %log2e;
    ex2.approx.f32 %e, %arg;

    mov.f32 %one, 0f3F800000;
    sub.f32 %num, %e, %one;
    add.f32 %den, %e, %one;
    div.approx.f32 %vr, %num, %den;

    cvt.rn.f16.f32 %out_h, %vr;
    st.global.b16 [%out], %out_h;

DONE:
    ret;
}
";

// sqrt(x) via sqrt.rn.f32. Inputs/outputs round-trip through f16.
const SQRT_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry sqrt_f16_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .b16 %a_b16, %out_h;
    .reg .f32 %va, %vr, %zero, %nan;
    .reg .pred %p, %is_nan, %is_neg;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.b16 %a_b16, [%a];
    cvt.f32.f16 %va, %a_b16;

    setp.nan.f32 %is_nan, %va, %va;
    @%is_nan bra STORE_NAN;
    mov.f32 %zero, 0f00000000;
    setp.lt.f32 %is_neg, %va, %zero;
    @%is_neg bra STORE_NAN;

    sqrt.rn.f32 %vr, %va;

    cvt.rn.f16.f32 %out_h, %vr;
    st.global.b16 [%out], %out_h;
    bra DONE;

STORE_NAN:
    mov.f32 %nan, 0f7FC00000;
    cvt.rn.f16.f32 %out_h, %nan;
    st.global.b16 [%out], %out_h;

DONE:
    ret;
}
";

// sigmoid(x) = 1 / (1 + e^(-x)). Same ex2.approx.f32 path as silu's internal sigmoid.
const SIGMOID_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry sigmoid_f16_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .b16 %a_b16, %out_h;
    .reg .f32 %va, %neg_a, %log2e, %arg, %e, %one, %denom, %vr;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.b16 %a_b16, [%a];
    cvt.f32.f16 %va, %a_b16;

    neg.f32 %neg_a, %va;
    mov.f32 %log2e, 0f3FB8AA3B;
    mul.f32 %arg, %neg_a, %log2e;
    ex2.approx.f32 %e, %arg;
    mov.f32 %one, 0f3F800000;
    add.f32 %denom, %one, %e;
    div.approx.f32 %vr, %one, %denom;

    cvt.rn.f16.f32 %out_h, %vr;
    st.global.b16 [%out], %out_h;

DONE:
    ret;
}
";

/// Elementwise `out = exp(a)` on f16 (u16-stored) GPU buffers.
pub fn gpu_exp_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    launch_unary(a, device, EXP_F16_PTX, "exp_f16_kernel")
}

/// Elementwise `out = ln(a)` on f16 (u16-stored) GPU buffers.
pub fn gpu_log_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    launch_unary(a, device, LOG_F16_PTX, "log_f16_kernel")
}

/// Elementwise `out = sin(a)` on f16 GPU buffers.
pub fn gpu_sin_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    launch_unary(a, device, SIN_F16_PTX, "sin_f16_kernel")
}

/// Elementwise `out = cos(a)` on f16 GPU buffers.
pub fn gpu_cos_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    launch_unary(a, device, COS_F16_PTX, "cos_f16_kernel")
}

/// Elementwise `out = tanh(a)` on f16 (u16-stored) GPU buffers.
pub fn gpu_tanh_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    launch_unary(a, device, TANH_F16_PTX, "tanh_f16_kernel")
}

/// Elementwise `out = sigmoid(a) = 1 / (1 + exp(-a))` on f16 GPU buffers.
pub fn gpu_sigmoid_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    launch_unary(a, device, SIGMOID_F16_PTX, "sigmoid_f16_kernel")
}

/// Elementwise `out = sqrt(a)` on f16 (u16-stored) GPU buffers.
pub fn gpu_sqrt_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    launch_unary(a, device, SQRT_F16_PTX, "sqrt_f16_kernel")
}

// ===========================================================================
// Scale (out = a * scalar) and neg
// ===========================================================================

const FILL_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry fill_f16_kernel(
    .param .u64 out_ptr,
    .param .f32 scalar,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %out, %off;
    .reg .b16 %out_h;
    .reg .f32 %scalar_r;
    .reg .pred %p;

    ld.param.u64 %out, [out_ptr];
    ld.param.f32 %scalar_r, [scalar];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %out, %out, %off;

    cvt.rn.f16.f32 %out_h, %scalar_r;
    st.global.b16 [%out], %out_h;

DONE:
    ret;
}
";

const SCALE_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry scale_f16_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .f32 scale,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .b16 %a_b16, %out_h;
    .reg .f32 %va, %scale_r, %vr;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.f32 %scale_r, [scale];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.b16 %a_b16, [%a];
    cvt.f32.f16 %va, %a_b16;

    mul.f32 %vr, %va, %scale_r;

    cvt.rn.f16.f32 %out_h, %vr;
    st.global.b16 [%out], %out_h;

DONE:
    ret;
}
";

/// Allocate an f16 buffer and fill it on the GPU with `scalar`, rounded once
/// from f32 to IEEE f16 using PTX round-to-nearest-even.
pub fn gpu_fill_f16(
    n: usize,
    scalar: f32,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    if n == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(0)?);
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile(
        ctx,
        FILL_F16_PTX,
        "fill_f16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "fill_f16_kernel",
        source: e,
    })?;

    let mut out = stream.alloc_zeros::<u16>(n)?;
    let cfg = launch_1d(n);
    let n_u32 = n as u32;
    // SAFETY:
    // - `f` is the `fill_f16_kernel` PTX entry compiled above; signature
    //   (out_ptr: u64, scalar: f32, n: u32) matches the three args below.
    // - `out` was alloc'd with exactly `n` u16 elements from `stream`; it is
    //   exclusively owned by this scope and mutably borrowed only for launch.
    // - `scalar` is a stack f32 copied into the kernel arg list; the kernel
    //   converts it to f16 using PTX RNE before storing the u16 bit pattern.
    // - The kernel writes `out[i]` only within `[0, n)` per the PTX
    //   bound-check.
    // - `n` fits in u32 because `launch_1d` uses the same cast to compute the
    //   covered grid.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(&mut out)
            .arg(&scalar)
            .arg(&n_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Multiply every element of an f16 buffer by an f32 scalar on the GPU,
/// returning a fresh allocation. Each thread converts its element to f32,
/// multiplies by `scale`, and rounds the product back to f16 (RNE).
pub fn gpu_scale_f16(
    input: &cudarc::driver::CudaSlice<u16>,
    scale: f32,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    let n = input.len();
    if n == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(0)?);
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile(
        ctx,
        SCALE_F16_PTX,
        "scale_f16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "scale_f16_kernel",
        source: e,
    })?;

    let mut out = stream.alloc_zeros::<u16>(n)?;
    let cfg = launch_1d(n);
    let n_u32 = n as u32;
    // SAFETY:
    // - `f` is the `scale_f16_kernel` PTX entry compiled above; signature
    //   (a_ptr: u64, out_ptr: u64, scale: f32, n: u32) matches the four
    //   args below in order.
    // - `input` is the immutable f16 buffer of length `n`; the empty-shape
    //   early-return guarantees `n > 0` here.
    // - `out` was alloc'd `n` u16 elements from `stream`; we hold the only
    //   `&mut out`, non-aliased with the immutable `input`.
    // - `scale` is a stack `f32` borrowed only for the `arg(&scale)` copy.
    // - The kernel reads `input[i]`/writes `out[i]` only within `[0, n)`
    //   per the PTX bound-check.
    // - `n` fits in u32 (`launch_1d` already cast it for the grid).
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(&mut out)
            .arg(&scale)
            .arg(&n_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Elementwise `out = -a` on f16 (u16-stored) GPU buffers.
///
/// Implemented as `scale_f16(a, -1.0)`: multiplying by `-1.0` is exact in
/// f32 (no rounding error from the scale), so the semantics match a
/// hand-rolled negate kernel.
pub fn gpu_neg_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    gpu_scale_f16(a, -1.0_f32, device)
}

// ===========================================================================
// Reductions (sum, mean) on f16 with f32 accumulator (PyTorch parity)
// ===========================================================================

// Block-stride axis reduction. Given an f16 buffer interpreted as
// [outer, axis, inner] (numpy-style), sum reduce along `axis` and produce
// an f16 buffer of shape [outer, inner]. f32 accumulator inside. One thread
// per output element; serial-sum over the axis (axis_size usually small).
const SUM_AXIS_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry sum_axis_f16_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 outer,
    .param .u32 axis_size,
    .param .u32 inner,
    .param .u32 do_mean
) {
    .reg .u32 %r_tid, %bid, %bdim, %outer_r, %axis_r, %inner_r;
    .reg .u32 %total_out, %oi, %ii, %k, %a_idx, %do_mean_r;
    .reg .u64 %a, %out, %off;
    .reg .b16 %a_b16, %out_h;
    .reg .f32 %acc, %va, %scale;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %outer_r, [outer];
    ld.param.u32 %axis_r, [axis_size];
    ld.param.u32 %inner_r, [inner];
    ld.param.u32 %do_mean_r, [do_mean];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    mul.lo.u32 %total_out, %outer_r, %inner_r;
    setp.ge.u32 %p, %r_tid, %total_out;
    @%p bra DONE;

    div.u32 %oi, %r_tid, %inner_r;
    rem.u32 %ii, %r_tid, %inner_r;

    mov.f32 %acc, 0f00000000;
    mov.u32 %k, 0;
LOOP:
    setp.ge.u32 %p, %k, %axis_r;
    @%p bra LOOP_END;

    // a_idx = oi * axis_size * inner + k * inner + ii.
    mul.lo.u32 %a_idx, %oi, %axis_r;
    add.u32 %a_idx, %a_idx, %k;
    mul.lo.u32 %a_idx, %a_idx, %inner_r;
    add.u32 %a_idx, %a_idx, %ii;

    cvt.u64.u32 %off, %a_idx;
    shl.b64 %off, %off, 1;
    add.u64 %off, %a, %off;
    ld.global.b16 %a_b16, [%off];
    cvt.f32.f16 %va, %a_b16;
    add.f32 %acc, %acc, %va;

    add.u32 %k, %k, 1;
    bra LOOP;
LOOP_END:

    // If do_mean, divide by axis_size.
    setp.eq.u32 %p, %do_mean_r, 0;
    @%p bra STORE;
    cvt.rn.f32.u32 %scale, %axis_r;
    div.approx.f32 %acc, %acc, %scale;

STORE:
    cvt.rn.f16.f32 %out_h, %acc;
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %off, %out, %off;
    st.global.b16 [%off], %out_h;

DONE:
    ret;
}
";

const PROD_AXIS_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry prod_axis_f16_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 outer,
    .param .u32 axis_size,
    .param .u32 inner
) {
    .reg .u32 %r_tid, %bid, %bdim, %outer_r, %axis_r, %inner_r;
    .reg .u32 %total_out, %oi, %ii, %k, %a_idx;
    .reg .u64 %a, %out, %off;
    .reg .b16 %a_b16, %out_h;
    .reg .f32 %acc, %va;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %outer_r, [outer];
    ld.param.u32 %axis_r, [axis_size];
    ld.param.u32 %inner_r, [inner];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    mul.lo.u32 %total_out, %outer_r, %inner_r;
    setp.ge.u32 %p, %r_tid, %total_out;
    @%p bra DONE;

    div.u32 %oi, %r_tid, %inner_r;
    rem.u32 %ii, %r_tid, %inner_r;

    mov.f32 %acc, 0f3F800000;
    mov.u32 %k, 0;
LOOP:
    setp.ge.u32 %p, %k, %axis_r;
    @%p bra LOOP_END;

    mul.lo.u32 %a_idx, %oi, %axis_r;
    add.u32 %a_idx, %a_idx, %k;
    mul.lo.u32 %a_idx, %a_idx, %inner_r;
    add.u32 %a_idx, %a_idx, %ii;

    cvt.u64.u32 %off, %a_idx;
    shl.b64 %off, %off, 1;
    add.u64 %off, %a, %off;
    ld.global.b16 %a_b16, [%off];
    cvt.f32.f16 %va, %a_b16;
    mul.f32 %acc, %acc, %va;

    add.u32 %k, %k, 1;
    bra LOOP;
LOOP_END:

    cvt.rn.f16.f32 %out_h, %acc;
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %off, %out, %off;
    st.global.b16 [%off], %out_h;

DONE:
    ret;
}
";

const PROD_AXIS_BACKWARD_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry prod_axis_backward_f16_kernel(
    .param .u64 input_ptr,
    .param .u64 grad_output_ptr,
    .param .u64 grad_input_ptr,
    .param .u32 outer,
    .param .u32 axis_size,
    .param .u32 inner
) {
    .reg .u32 %r_tid, %bid, %bdim, %outer_r, %axis_r, %inner_r;
    .reg .u32 %axis_inner, %total_in, %oi, %tmp, %di, %ii, %k, %idx, %go_idx;
    .reg .u64 %input, %go, %gi, %off;
    .reg .b16 %x_b16, %go_b16, %out_h;
    .reg .f32 %acc, %vx, %vgo, %vr;
    .reg .pred %p, %p_skip;

    ld.param.u64 %input, [input_ptr];
    ld.param.u64 %go, [grad_output_ptr];
    ld.param.u64 %gi, [grad_input_ptr];
    ld.param.u32 %outer_r, [outer];
    ld.param.u32 %axis_r, [axis_size];
    ld.param.u32 %inner_r, [inner];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    mul.lo.u32 %axis_inner, %axis_r, %inner_r;
    mul.lo.u32 %total_in, %outer_r, %axis_inner;
    setp.ge.u32 %p, %r_tid, %total_in;
    @%p bra DONE;

    div.u32 %oi, %r_tid, %axis_inner;
    rem.u32 %tmp, %r_tid, %axis_inner;
    div.u32 %di, %tmp, %inner_r;
    rem.u32 %ii, %tmp, %inner_r;

    mul.lo.u32 %go_idx, %oi, %inner_r;
    add.u32 %go_idx, %go_idx, %ii;
    cvt.u64.u32 %off, %go_idx;
    shl.b64 %off, %off, 1;
    add.u64 %off, %go, %off;
    ld.global.b16 %go_b16, [%off];
    cvt.f32.f16 %vgo, %go_b16;

    mov.f32 %acc, 0f3F800000;
    mov.u32 %k, 0;
LOOP:
    setp.ge.u32 %p, %k, %axis_r;
    @%p bra LOOP_END;
    setp.eq.u32 %p_skip, %k, %di;
    @%p_skip bra SKIP_MUL;

    mul.lo.u32 %idx, %oi, %axis_r;
    add.u32 %idx, %idx, %k;
    mul.lo.u32 %idx, %idx, %inner_r;
    add.u32 %idx, %idx, %ii;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 1;
    add.u64 %off, %input, %off;
    ld.global.b16 %x_b16, [%off];
    cvt.f32.f16 %vx, %x_b16;
    mul.f32 %acc, %acc, %vx;

SKIP_MUL:
    add.u32 %k, %k, 1;
    bra LOOP;
LOOP_END:

    mul.f32 %vr, %acc, %vgo;
    cvt.rn.f16.f32 %out_h, %vr;
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %off, %gi, %off;
    st.global.b16 [%off], %out_h;

DONE:
    ret;
}
";

fn validate_prod_axis_dims_f16(
    op: &'static str,
    input_len: usize,
    grad_output_len: Option<usize>,
    outer: usize,
    axis_size: usize,
    inner: usize,
) -> GpuResult<(usize, usize)> {
    let total_input = outer
        .checked_mul(axis_size)
        .and_then(|v| v.checked_mul(inner))
        .ok_or(GpuError::ShapeMismatch {
            op,
            expected: vec![outer, axis_size, inner],
            got: vec![usize::MAX],
        })?;
    let total_output = outer.checked_mul(inner).ok_or(GpuError::ShapeMismatch {
        op,
        expected: vec![outer, inner],
        got: vec![usize::MAX],
    })?;
    if input_len != total_input {
        return Err(GpuError::ShapeMismatch {
            op,
            expected: vec![total_input],
            got: vec![input_len],
        });
    }
    if let Some(len) = grad_output_len
        && len != total_output
    {
        return Err(GpuError::ShapeMismatch {
            op,
            expected: vec![total_output],
            got: vec![len],
        });
    }
    Ok((total_input, total_output))
}

/// Axis-reduce sum: f16 [outer, axis, inner] -> f16 [outer, inner].
/// f32 accumulator (PyTorch parity).
pub fn gpu_sum_axis_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    outer: usize,
    axis_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    let total = outer.checked_mul(inner).ok_or(GpuError::ShapeMismatch {
        op: "sum_axis_f16",
        expected: vec![outer, inner],
        got: vec![usize::MAX],
    })?;
    let stream = device.stream();
    if total == 0 {
        return Ok(stream.alloc_zeros::<u16>(0)?);
    }
    let ctx = device.context();
    let f = get_or_compile(
        ctx,
        SUM_AXIS_F16_PTX,
        "sum_axis_f16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "sum_axis_f16_kernel",
        source: e,
    })?;
    let mut out = stream.alloc_zeros::<u16>(total)?;
    let cfg = launch_1d(total);
    let outer_u32 = outer as u32;
    let axis_u32 = axis_size as u32;
    let inner_u32 = inner as u32;
    let do_mean: u32 = 0;
    // SAFETY:
    // - `f` is the `sum_axis_f16_kernel` PTX entry whose signature is
    //   (a_ptr, out_ptr, outer, axis_size, inner, do_mean). The six args
    //   below match in order.
    // - `a` is the caller's f16 input. The kernel reads
    //   `a[oi * axis * inner + k * inner + ii]` for `(oi, ii)` derived from
    //   the global thread id `[0, outer*inner)` and `k ∈ [0, axis_size)`;
    //   the bound check short-circuits threads beyond `total`.
    // - `out` is freshly alloc'd `total` u16 elements.
    // - `outer_u32 * inner_u32` was overflow-checked above.
    // - `do_mean = 0` selects the sum path (mean branch guarded off).
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a)
            .arg(&mut out)
            .arg(&outer_u32)
            .arg(&axis_u32)
            .arg(&inner_u32)
            .arg(&do_mean)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Axis-reduce mean: f16 [outer, axis, inner] -> f16 [outer, inner].
/// f32 accumulator, divides by axis_size in f32 before rounding back.
pub fn gpu_mean_axis_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    outer: usize,
    axis_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    let total = outer.checked_mul(inner).ok_or(GpuError::ShapeMismatch {
        op: "mean_axis_f16",
        expected: vec![outer, inner],
        got: vec![usize::MAX],
    })?;
    let stream = device.stream();
    if total == 0 {
        return Ok(stream.alloc_zeros::<u16>(0)?);
    }
    if axis_size == 0 {
        // PyTorch: mean over zero-length axis = NaN. f16 quiet NaN = 0x7E00.
        let mut out = stream.alloc_zeros::<u16>(total)?;
        let nan_bits = vec![0x7E00_u16; total];
        stream.memcpy_htod(&nan_bits, &mut out)?;
        return Ok(out);
    }
    let ctx = device.context();
    let f = get_or_compile(
        ctx,
        SUM_AXIS_F16_PTX,
        "sum_axis_f16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "sum_axis_f16_kernel",
        source: e,
    })?;
    let mut out = stream.alloc_zeros::<u16>(total)?;
    let cfg = launch_1d(total);
    let outer_u32 = outer as u32;
    let axis_u32 = axis_size as u32;
    let inner_u32 = inner as u32;
    let do_mean: u32 = 1;
    // SAFETY: identical to gpu_sum_axis_f16 above except `do_mean = 1`
    // routes the kernel through its `div.approx.f32 %acc, %acc, %scale`
    // arm before the round-and-store epilogue.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a)
            .arg(&mut out)
            .arg(&outer_u32)
            .arg(&axis_u32)
            .arg(&inner_u32)
            .arg(&do_mean)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Axis-reduce product: f16 [outer, axis, inner] -> f16 [outer, inner].
/// Accumulates in f32 and rounds once to f16 at the output.
pub fn gpu_prod_axis_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    outer: usize,
    axis_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    let (_, total) =
        validate_prod_axis_dims_f16("prod_axis_f16", a.len(), None, outer, axis_size, inner)?;
    let stream = device.stream();
    if total == 0 {
        return Ok(stream.alloc_zeros::<u16>(0)?);
    }
    let ctx = device.context();
    let f = get_or_compile(
        ctx,
        PROD_AXIS_F16_PTX,
        "prod_axis_f16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "prod_axis_f16_kernel",
        source: e,
    })?;
    let mut out = stream.alloc_zeros::<u16>(total)?;
    let cfg = launch_1d(total);
    let outer_u32 = outer as u32;
    let axis_u32 = axis_size as u32;
    let inner_u32 = inner as u32;
    // SAFETY:
    // - `f` is the `prod_axis_f16_kernel` entry with args
    //   (a_ptr, out_ptr, outer, axis_size, inner).
    // - `validate_prod_axis_dims_f16` proves `a.len() == outer*axis*inner`;
    //   the kernel launches one thread per `outer*inner` output and reads
    //   only k in `[0, axis_size)`.
    // - `out` is a fresh `total`-element u16 allocation.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a)
            .arg(&mut out)
            .arg(&outer_u32)
            .arg(&axis_u32)
            .arg(&inner_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Backward for f16 axis product.
///
/// Computes `grad_output[o, i] * product(input[o, d != current, i])`.
/// This direct prefix/suffix-equivalent product preserves PyTorch's zero
/// semantics without dividing by the forward product.
pub fn gpu_prod_axis_backward_f16(
    input: &cudarc::driver::CudaSlice<u16>,
    grad_output: &cudarc::driver::CudaSlice<u16>,
    outer: usize,
    axis_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    let (total_input, _) = validate_prod_axis_dims_f16(
        "prod_axis_backward_f16",
        input.len(),
        Some(grad_output.len()),
        outer,
        axis_size,
        inner,
    )?;
    let stream = device.stream();
    if total_input == 0 {
        return Ok(stream.alloc_zeros::<u16>(0)?);
    }
    let ctx = device.context();
    let f = get_or_compile(
        ctx,
        PROD_AXIS_BACKWARD_F16_PTX,
        "prod_axis_backward_f16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "prod_axis_backward_f16_kernel",
        source: e,
    })?;
    let mut out = stream.alloc_zeros::<u16>(total_input)?;
    let cfg = launch_1d(total_input);
    let outer_u32 = outer as u32;
    let axis_u32 = axis_size as u32;
    let inner_u32 = inner as u32;
    // SAFETY:
    // - `f` is the `prod_axis_backward_f16_kernel` entry with args
    //   (input_ptr, grad_output_ptr, grad_input_ptr, outer, axis, inner).
    // - `validate_prod_axis_dims_f16` proves the input and grad_output
    //   lengths match `[outer, axis, inner]` and `[outer, inner]`.
    // - `out` is a fresh `total_input` allocation; each thread writes its
    //   own input-position gradient exactly once.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(grad_output)
            .arg(&mut out)
            .arg(&outer_u32)
            .arg(&axis_u32)
            .arg(&inner_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Sum-reduce an f16 buffer to a scalar (1-element) f16 buffer.
/// f32 accumulator (PyTorch parity). Implemented as a single-element
/// axis reduction with `outer=1, axis_size=n, inner=1`.
pub fn gpu_sum_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    let n = a.len();
    if n == 0 {
        // PyTorch: sum of empty tensor is 0.
        return Ok(device.stream().alloc_zeros::<u16>(1)?);
    }
    gpu_sum_axis_f16(a, 1, n, 1, device)
}

/// Mean of an f16 buffer to a scalar (1-element) f16 buffer.
/// Computes the f32-accumulated sum then divides on-device by `n`.
pub fn gpu_mean_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    let n = a.len();
    if n == 0 {
        // PyTorch raises on mean of empty; mirror via NaN scalar
        // (f16 quiet NaN = 0x7E00).
        let stream = device.stream();
        let mut out = stream.alloc_zeros::<u16>(1)?;
        let nan_bits = vec![0x7E00_u16];
        stream.memcpy_htod(&nan_bits, &mut out)?;
        return Ok(out);
    }
    let sum = gpu_sum_f16(a, device)?;
    let inv_n = 1.0f32 / (n as f32);
    gpu_scale_f16(&sum, inv_n, device)
}

/// Product of an f16 buffer to a scalar (1-element) f16 buffer.
/// Empty input returns the multiplicative identity, matching PyTorch.
pub fn gpu_prod_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    gpu_prod_axis_f16(a, 1, a.len(), 1, device)
}

/// Backward for global f16 product. Output length equals input length.
pub fn gpu_prod_backward_f16(
    input: &cudarc::driver::CudaSlice<u16>,
    grad_output: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    gpu_prod_axis_backward_f16(input, grad_output, 1, input.len(), 1, device)
}

const STD_VAR_AXIS_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry std_var_axis_f16_kernel(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u32 outer_size,
    .param .u32 axis_size,
    .param .u32 inner_size,
    .param .u32 total_output,
    .param .f32 correction,
    .param .u32 take_sqrt
) {
    .reg .u32 %r_tid, %bid, %bdim, %total, %outer_sz, %axis_sz, %inner_sz;
    .reg .u32 %outer_idx, %inner_idx, %k, %base, %idx, %flag;
    .reg .u64 %in, %out, %off, %addr;
    .reg .b16 %h;
    .reg .f32 %val, %sum, %mean, %ss, %dv, %denom, %axis_f, %corr, %outv, %zero;
    .reg .pred %p, %lp, %empty, %do_sqrt;

    ld.param.u64 %in, [input_ptr];
    ld.param.u64 %out, [output_ptr];
    ld.param.u32 %outer_sz, [outer_size];
    ld.param.u32 %axis_sz, [axis_size];
    ld.param.u32 %inner_sz, [inner_size];
    ld.param.u32 %total, [total_output];
    ld.param.f32 %corr, [correction];
    ld.param.u32 %flag, [take_sqrt];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %total;
    @%p bra DONE;

    mov.f32 %zero, 0f00000000;
    div.u32 %outer_idx, %r_tid, %inner_sz;
    rem.u32 %inner_idx, %r_tid, %inner_sz;
    mul.lo.u32 %base, %outer_idx, %axis_sz;
    mul.lo.u32 %base, %base, %inner_sz;
    add.u32 %base, %base, %inner_idx;

    setp.eq.u32 %empty, %axis_sz, 0;
    @%empty bra EMPTY_SLICE;

    mov.u32 %k, 0;
    mov.f32 %sum, 0f00000000;
SUM_LOOP:
    setp.ge.u32 %lp, %k, %axis_sz;
    @%lp bra SUM_DONE;
    mul.lo.u32 %idx, %k, %inner_sz;
    add.u32 %idx, %base, %idx;
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %in, %off;
    ld.global.b16 %h, [%addr];
    cvt.f32.f16 %val, %h;
    add.f32 %sum, %sum, %val;
    add.u32 %k, %k, 1;
    bra SUM_LOOP;
SUM_DONE:
    cvt.rn.f32.u32 %axis_f, %axis_sz;
    div.rn.f32 %mean, %sum, %axis_f;

    mov.u32 %k, 0;
    mov.f32 %ss, 0f00000000;
SS_LOOP:
    setp.ge.u32 %lp, %k, %axis_sz;
    @%lp bra SS_DONE;
    mul.lo.u32 %idx, %k, %inner_sz;
    add.u32 %idx, %base, %idx;
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %in, %off;
    ld.global.b16 %h, [%addr];
    cvt.f32.f16 %val, %h;
    sub.f32 %dv, %val, %mean;
    fma.rn.f32 %ss, %dv, %dv, %ss;
    add.u32 %k, %k, 1;
    bra SS_LOOP;
SS_DONE:
    sub.f32 %denom, %axis_f, %corr;
    max.f32 %denom, %denom, %zero;
    div.rn.f32 %outv, %ss, %denom;
    setp.ne.u32 %do_sqrt, %flag, 0;
    @%do_sqrt sqrt.rn.f32 %outv, %outv;
    bra STORE;

EMPTY_SLICE:
    div.rn.f32 %outv, %zero, %zero;

STORE:
    cvt.rn.f16.f32 %h, %outv;
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %out, %off;
    st.global.b16 [%addr], %h;

DONE:
    ret;
}
";

const STD_VAR_AXIS_BACKWARD_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry std_var_axis_backward_f16_kernel(
    .param .u64 input_ptr,
    .param .u64 grad_out_ptr,
    .param .u64 result_ptr,
    .param .u64 grad_in_ptr,
    .param .u32 outer_size,
    .param .u32 axis_size,
    .param .u32 inner_size,
    .param .u32 total_input,
    .param .f32 correction,
    .param .u32 take_sqrt
) {
    .reg .u32 %r_tid, %bid, %bdim, %total, %outer_sz, %axis_sz, %inner_sz;
    .reg .u32 %tmp, %outer_idx, %d_idx, %inner_idx, %k, %src_idx, %go_idx, %flag;
    .reg .u64 %in, %go, %res, %gi, %off, %addr;
    .reg .b16 %h;
    .reg .f32 %val, %sum, %mean, %dv, %axis_f, %corr, %denom, %grad, %go_val;
    .reg .f32 %result, %zero, %two, %scale, %denom_result;
    .reg .pred %p, %lp, %is_std, %is_zero_result;

    ld.param.u64 %in, [input_ptr];
    ld.param.u64 %go, [grad_out_ptr];
    ld.param.u64 %res, [result_ptr];
    ld.param.u64 %gi, [grad_in_ptr];
    ld.param.u32 %outer_sz, [outer_size];
    ld.param.u32 %axis_sz, [axis_size];
    ld.param.u32 %inner_sz, [inner_size];
    ld.param.u32 %total, [total_input];
    ld.param.f32 %corr, [correction];
    ld.param.u32 %flag, [take_sqrt];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %total;
    @%p bra DONE;

    mov.f32 %zero, 0f00000000;
    mov.f32 %two, 0f40000000;
    rem.u32 %inner_idx, %r_tid, %inner_sz;
    div.u32 %tmp, %r_tid, %inner_sz;
    rem.u32 %d_idx, %tmp, %axis_sz;
    div.u32 %outer_idx, %tmp, %axis_sz;

    mul.lo.u32 %go_idx, %outer_idx, %inner_sz;
    add.u32 %go_idx, %go_idx, %inner_idx;
    cvt.u64.u32 %off, %go_idx;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %go, %off;
    ld.global.b16 %h, [%addr];
    cvt.f32.f16 %go_val, %h;
    add.u64 %addr, %res, %off;
    ld.global.b16 %h, [%addr];
    cvt.f32.f16 %result, %h;

    mov.u32 %k, 0;
    mov.f32 %sum, 0f00000000;
SUM_LOOP:
    setp.ge.u32 %lp, %k, %axis_sz;
    @%lp bra SUM_DONE;
    mul.lo.u32 %src_idx, %outer_idx, %axis_sz;
    add.u32 %src_idx, %src_idx, %k;
    mul.lo.u32 %src_idx, %src_idx, %inner_sz;
    add.u32 %src_idx, %src_idx, %inner_idx;
    cvt.u64.u32 %off, %src_idx;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %in, %off;
    ld.global.b16 %h, [%addr];
    cvt.f32.f16 %val, %h;
    add.f32 %sum, %sum, %val;
    add.u32 %k, %k, 1;
    bra SUM_LOOP;
SUM_DONE:
    cvt.rn.f32.u32 %axis_f, %axis_sz;
    div.rn.f32 %mean, %sum, %axis_f;
    sub.f32 %denom, %axis_f, %corr;
    max.f32 %denom, %denom, %zero;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %in, %off;
    ld.global.b16 %h, [%addr];
    cvt.f32.f16 %val, %h;
    sub.f32 %dv, %val, %mean;

    setp.ne.u32 %is_std, %flag, 0;
    @%is_std bra STD_PATH;

    div.rn.f32 %scale, %two, %denom;
    mul.f32 %grad, %go_val, %scale;
    mul.f32 %grad, %grad, %dv;
    bra STORE;

STD_PATH:
    setp.eq.f32 %is_zero_result, %result, %zero;
    @%is_zero_result bra ZERO_STD;
    mul.f32 %denom_result, %denom, %result;
    div.rn.f32 %scale, %go_val, %denom_result;
    mul.f32 %grad, %scale, %dv;
    bra STORE;

ZERO_STD:
    mov.f32 %grad, 0f00000000;

STORE:
    cvt.rn.f16.f32 %h, %grad;
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %addr, %gi, %off;
    st.global.b16 [%addr], %h;

DONE:
    ret;
}
";

/// Compute variance or standard deviation along one logical axis for f16.
///
/// The input is `[outer, axis_size, inner]`, output is `[outer, inner]`.
/// Arithmetic is f32 and the result is rounded back to f16 storage.
pub fn gpu_std_var_axis_f16(
    input: &cudarc::driver::CudaSlice<u16>,
    outer: usize,
    axis_size: usize,
    inner: usize,
    correction: f64,
    take_sqrt: bool,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    let total = outer.checked_mul(inner).ok_or(GpuError::ShapeMismatch {
        op: "std_var_axis_f16",
        expected: vec![outer, inner],
        got: vec![usize::MAX],
    })?;
    let expected_len =
        crate::shape_math::checked_mul3(outer, axis_size, inner, "std_var_axis_f16")?;
    if input.len() != expected_len {
        return Err(GpuError::ShapeMismatch {
            op: "std_var_axis_f16",
            expected: vec![outer, axis_size, inner],
            got: vec![input.len()],
        });
    }
    let stream = device.stream();
    if total == 0 {
        return Ok(stream.alloc_zeros::<u16>(0)?);
    }
    let ctx = device.context();
    let f = get_or_compile(
        ctx,
        STD_VAR_AXIS_F16_PTX,
        "std_var_axis_f16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "std_var_axis_f16_kernel",
        source: e,
    })?;
    let mut out = stream.alloc_zeros::<u16>(total)?;
    let cfg = launch_1d(total);
    let outer_u32 = outer as u32;
    let axis_u32 = axis_size as u32;
    let inner_u32 = inner as u32;
    let total_u32 = total as u32;
    let corr_f32 = correction as f32;
    let take_sqrt_u32 = u32::from(take_sqrt);
    // SAFETY: one thread computes one `[outer, inner]` output slice. The
    // input shape product is checked above and all address arithmetic is
    // bound-guarded by `total`.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(&mut out)
            .arg(&outer_u32)
            .arg(&axis_u32)
            .arg(&inner_u32)
            .arg(&total_u32)
            .arg(&corr_f32)
            .arg(&take_sqrt_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Backward pass for f16 axis variance or standard deviation.
pub fn gpu_std_var_axis_backward_f16(
    input: &cudarc::driver::CudaSlice<u16>,
    grad_output: &cudarc::driver::CudaSlice<u16>,
    result: &cudarc::driver::CudaSlice<u16>,
    outer: usize,
    axis_size: usize,
    inner: usize,
    correction: f64,
    take_sqrt: bool,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    let total_input = outer
        .checked_mul(axis_size)
        .and_then(|v| v.checked_mul(inner))
        .ok_or(GpuError::ShapeMismatch {
            op: "std_var_axis_backward_f16",
            expected: vec![outer, axis_size, inner],
            got: vec![usize::MAX],
        })?;
    let total_output = outer.checked_mul(inner).ok_or(GpuError::ShapeMismatch {
        op: "std_var_axis_backward_f16",
        expected: vec![outer, inner],
        got: vec![usize::MAX],
    })?;
    if input.len() != total_input
        || grad_output.len() != total_output
        || result.len() != total_output
    {
        return Err(GpuError::ShapeMismatch {
            op: "std_var_axis_backward_f16",
            expected: vec![total_input, total_output, total_output],
            got: vec![input.len(), grad_output.len(), result.len()],
        });
    }
    let stream = device.stream();
    if total_input == 0 {
        return Ok(stream.alloc_zeros::<u16>(0)?);
    }
    let ctx = device.context();
    let f = get_or_compile(
        ctx,
        STD_VAR_AXIS_BACKWARD_F16_PTX,
        "std_var_axis_backward_f16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "std_var_axis_backward_f16_kernel",
        source: e,
    })?;
    let mut out = stream.alloc_zeros::<u16>(total_input)?;
    let cfg = launch_1d(total_input);
    let outer_u32 = outer as u32;
    let axis_u32 = axis_size as u32;
    let inner_u32 = inner as u32;
    let total_u32 = total_input as u32;
    let corr_f32 = correction as f32;
    let take_sqrt_u32 = u32::from(take_sqrt);
    // SAFETY: one thread writes one input gradient, reading only its matching
    // input slice plus one `[outer, inner]` grad/result value. Buffer lengths
    // are checked against the logical products above.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(grad_output)
            .arg(result)
            .arg(&mut out)
            .arg(&outer_u32)
            .arg(&axis_u32)
            .arg(&inner_u32)
            .arg(&total_u32)
            .arg(&corr_f32)
            .arg(&take_sqrt_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

// ===========================================================================
// Softmax — row-wise, f32 accumulator, two-pass tree reduction
// ===========================================================================

// One block per row. Pass 1: thread-local max, then shared-memory tree-max.
// Pass 2: thread-local sum of exp(v - row_max), then shared-memory tree-sum.
// Pass 3: write exp((v-row_max)) * inv_sum rounded to f16.
const SOFTMAX_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.shared .align 4 .f32 softmax_f16_sdata[256];

.visible .entry softmax_f16_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 rows,
    .param .u32 cols
) {
    .reg .u32 %r_tid, %bid, %bdim, %rows_reg, %cols_reg, %j, %half, %otid;
    .reg .u64 %in, %out, %row_off, %off, %sbase, %saddr;
    .reg .b16 %x_b16, %out_h;
    .reg .f32 %x_f, %tmax, %other, %row_max, %sum, %inv_sum, %e, %scale, %log2e, %y_f;
    .reg .pred %p, %lp, %rp, %gp;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %rows_reg, [rows];
    ld.param.u32 %cols_reg, [cols];

    mov.u64 %sbase, softmax_f16_sdata;
    mov.f32 %log2e, 0f3FB8AA3B;

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;

    setp.ge.u32 %p, %bid, %rows_reg;
    @%p bra DONE;

    cvt.u64.u32 %row_off, %bid;
    cvt.u64.u32 %off, %cols_reg;
    mul.lo.u64 %row_off, %row_off, %off;
    shl.b64 %row_off, %row_off, 1;

    // Pass 1: thread-local max
    mov.f32 %tmax, 0fFF800000;   // -Inf
    mov.u32 %j, %r_tid;
MX:
    setp.ge.u32 %lp, %j, %cols_reg;
    @%lp bra MXD;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %in, %off;
    add.u64 %off, %off, %row_off;
    ld.global.b16 %x_b16, [%off];
    cvt.f32.f16 %x_f, %x_b16;
    setp.gt.f32 %gp, %x_f, %tmax;
    @%gp mov.f32 %tmax, %x_f;
    add.u32 %j, %j, %bdim;
    bra MX;
MXD:
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %tmax;
    bar.sync 0;

    mov.u32 %half, %bdim;
MR:
    shr.u32 %half, %half, 1;
    setp.eq.u32 %rp, %half, 0;
    @%rp bra MRD;
    setp.ge.u32 %rp, %r_tid, %half;
    @%rp bra MRS;
    add.u32 %otid, %r_tid, %half;
    cvt.u64.u32 %off, %otid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other, [%saddr];
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %tmax, [%saddr];
    setp.gt.f32 %gp, %other, %tmax;
    @%gp mov.f32 %tmax, %other;
    st.shared.f32 [%saddr], %tmax;
MRS:
    bar.sync 0;
    bra MR;
MRD:
    ld.shared.f32 %row_max, [%sbase];
    bar.sync 0;

    // Pass 2: thread-local sum of exp(v - row_max)
    mov.f32 %sum, 0f00000000;
    mov.u32 %j, %r_tid;
SE:
    setp.ge.u32 %lp, %j, %cols_reg;
    @%lp bra SED;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %in, %off;
    add.u64 %off, %off, %row_off;
    ld.global.b16 %x_b16, [%off];
    cvt.f32.f16 %x_f, %x_b16;
    sub.f32 %x_f, %x_f, %row_max;
    mul.f32 %scale, %x_f, %log2e;
    ex2.approx.f32 %e, %scale;
    add.f32 %sum, %sum, %e;
    add.u32 %j, %j, %bdim;
    bra SE;
SED:
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %sum;
    bar.sync 0;

    mov.u32 %half, %bdim;
SER:
    shr.u32 %half, %half, 1;
    setp.eq.u32 %rp, %half, 0;
    @%rp bra SERD;
    setp.ge.u32 %rp, %r_tid, %half;
    @%rp bra SERS;
    add.u32 %otid, %r_tid, %half;
    cvt.u64.u32 %off, %otid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other, [%saddr];
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %sum, [%saddr];
    add.f32 %sum, %sum, %other;
    st.shared.f32 [%saddr], %sum;
SERS:
    bar.sync 0;
    bra SER;
SERD:
    ld.shared.f32 %sum, [%sbase];
    rcp.approx.f32 %inv_sum, %sum;
    bar.sync 0;

    // Pass 3: write
    mov.u32 %j, %r_tid;
WR:
    setp.ge.u32 %lp, %j, %cols_reg;
    @%lp bra WRD;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %in, %off;
    add.u64 %off, %off, %row_off;
    ld.global.b16 %x_b16, [%off];
    cvt.f32.f16 %x_f, %x_b16;
    sub.f32 %x_f, %x_f, %row_max;
    mul.f32 %scale, %x_f, %log2e;
    ex2.approx.f32 %e, %scale;
    mul.f32 %y_f, %e, %inv_sum;

    cvt.rn.f16.f32 %out_h, %y_f;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %out, %off;
    add.u64 %off, %off, %row_off;
    st.global.b16 [%off], %out_h;
    add.u32 %j, %j, %bdim;
    bra WR;
WRD:

DONE:
    ret;
}
";

/// Apply row-wise softmax to an f16 `[rows, cols]` row-major tensor on the
/// GPU. One CUDA block per row, computing numerically-stable
/// `exp(x - max) / sum(exp(x - max))` in f32 and rounding back to f16 (RNE).
pub fn gpu_softmax_f16(
    input: &cudarc::driver::CudaSlice<u16>,
    rows: usize,
    cols: usize,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    if rows == 0 || cols == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(rows * cols)?);
    }
    if input.len() < rows * cols {
        return Err(GpuError::ShapeMismatch {
            op: "softmax_f16",
            expected: vec![rows, cols],
            got: vec![input.len()],
        });
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile(
        ctx,
        SOFTMAX_F16_PTX,
        "softmax_f16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "softmax_f16_kernel",
        source: e,
    })?;

    let mut out = stream.alloc_zeros::<u16>(rows * cols)?;
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    // SAFETY:
    // - `f` is the `softmax_f16_kernel` PTX entry; signature (in_ptr: u64,
    //   out_ptr: u64, rows: u32, cols: u32) matches the four args below.
    // - `input` length checked `>= rows * cols`; empty-shape early-return
    //   ensures `rows > 0 && cols > 0` at launch.
    // - `out` was alloc'd exactly `rows * cols` u16 elements; we hold the
    //   only `&mut out`, non-aliased with immutable `input`.
    // - The kernel declares 256 f32 words of `.shared`
    //   (`softmax_f16_sdata[256]`); `cfg.block_dim.x == BLOCK_SIZE == 256`
    //   so `r_tid < 256` and shared accesses at `r_tid * 4` stay within the
    //   1024-byte shared region.
    // - One block per row (grid_dim.x = rows); row and column loops guarded
    //   by `setp.ge.u32 ..., %rows_reg` / `..., %cols_reg`, confining
    //   reads/writes to `[0, rows * cols)`.
    // - `rows`/`cols` fit in u32 (their product sized `out`).
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(&mut out)
            .arg(&rows_u32)
            .arg(&cols_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

// ===========================================================================
// LayerNorm and RMSNorm — one block per row, f32 reductions
// ===========================================================================

const LAYERNORM_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.shared .align 4 .f32 layernorm_f16_sdata[256];

.visible .entry layernorm_f16_kernel(
    .param .u64 in_ptr,
    .param .u64 gamma_ptr,
    .param .u64 beta_ptr,
    .param .u64 out_ptr,
    .param .u32 rows,
    .param .u32 cols,
    .param .f32 eps
) {
    .reg .u32 %r_tid, %bid, %bdim, %rows_reg, %cols_reg, %j, %half, %otid;
    .reg .u64 %in, %gam, %bet, %out, %row_off, %off, %sbase, %saddr;
    .reg .b16 %x_b16, %g_b16, %b_b16, %out_h;
    .reg .f32 %x_f, %g_f, %b_f, %sum, %mean, %diff, %var, %eps_r, %inv_std, %normed, %r_f, %other, %n_f;
    .reg .pred %p, %lp, %rp;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %gam, [gamma_ptr];
    ld.param.u64 %bet, [beta_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %rows_reg, [rows];
    ld.param.u32 %cols_reg, [cols];
    ld.param.f32 %eps_r, [eps];

    mov.u64 %sbase, layernorm_f16_sdata;

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;

    setp.ge.u32 %p, %bid, %rows_reg;
    @%p bra DONE;

    cvt.u64.u32 %row_off, %bid;
    cvt.u64.u32 %off, %cols_reg;
    mul.lo.u64 %row_off, %row_off, %off;
    shl.b64 %row_off, %row_off, 1;
    cvt.rn.f32.u32 %n_f, %cols_reg;

    // Phase 1: sum(x) -> mean
    mov.f32 %sum, 0f00000000;
    mov.u32 %j, %r_tid;
SM:
    setp.ge.u32 %lp, %j, %cols_reg;
    @%lp bra SMD;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %in, %off;
    add.u64 %off, %off, %row_off;
    ld.global.b16 %x_b16, [%off];
    cvt.f32.f16 %x_f, %x_b16;
    add.f32 %sum, %sum, %x_f;
    add.u32 %j, %j, %bdim;
    bra SM;
SMD:
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %sum;
    bar.sync 0;

    mov.u32 %half, %bdim;
MR:
    shr.u32 %half, %half, 1;
    setp.eq.u32 %rp, %half, 0;
    @%rp bra MRD;
    setp.ge.u32 %rp, %r_tid, %half;
    @%rp bra MRS;
    add.u32 %otid, %r_tid, %half;
    cvt.u64.u32 %off, %otid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other, [%saddr];
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %sum, [%saddr];
    add.f32 %sum, %sum, %other;
    st.shared.f32 [%saddr], %sum;
MRS:
    bar.sync 0;
    bra MR;
MRD:
    ld.shared.f32 %sum, [%sbase];
    div.approx.f32 %mean, %sum, %n_f;
    bar.sync 0;

    // Phase 2: sum((x - mean)^2) -> var
    mov.f32 %var, 0f00000000;
    mov.u32 %j, %r_tid;
SV:
    setp.ge.u32 %lp, %j, %cols_reg;
    @%lp bra SVD;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %in, %off;
    add.u64 %off, %off, %row_off;
    ld.global.b16 %x_b16, [%off];
    cvt.f32.f16 %x_f, %x_b16;
    sub.f32 %diff, %x_f, %mean;
    fma.rn.f32 %var, %diff, %diff, %var;
    add.u32 %j, %j, %bdim;
    bra SV;
SVD:
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %var;
    bar.sync 0;

    mov.u32 %half, %bdim;
VR:
    shr.u32 %half, %half, 1;
    setp.eq.u32 %rp, %half, 0;
    @%rp bra VRD;
    setp.ge.u32 %rp, %r_tid, %half;
    @%rp bra VRS;
    add.u32 %otid, %r_tid, %half;
    cvt.u64.u32 %off, %otid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other, [%saddr];
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %var, [%saddr];
    add.f32 %var, %var, %other;
    st.shared.f32 [%saddr], %var;
VRS:
    bar.sync 0;
    bra VR;
VRD:
    ld.shared.f32 %var, [%sbase];
    div.approx.f32 %var, %var, %n_f;
    add.f32 %var, %var, %eps_r;
    sqrt.approx.f32 %inv_std, %var;
    rcp.approx.f32 %inv_std, %inv_std;
    bar.sync 0;

    // Phase 3: out = ((x - mean) * inv_std) * gamma + beta, rounded to f16
    mov.u32 %j, %r_tid;
NM:
    setp.ge.u32 %lp, %j, %cols_reg;
    @%lp bra NMD;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %in, %off;
    add.u64 %off, %off, %row_off;
    ld.global.b16 %x_b16, [%off];
    cvt.f32.f16 %x_f, %x_b16;
    sub.f32 %normed, %x_f, %mean;
    mul.f32 %normed, %normed, %inv_std;

    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %gam, %off;
    ld.global.b16 %g_b16, [%off];
    cvt.f32.f16 %g_f, %g_b16;

    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %bet, %off;
    ld.global.b16 %b_b16, [%off];
    cvt.f32.f16 %b_f, %b_b16;

    fma.rn.f32 %r_f, %g_f, %normed, %b_f;

    cvt.rn.f16.f32 %out_h, %r_f;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %out, %off;
    add.u64 %off, %off, %row_off;
    st.global.b16 [%off], %out_h;
    add.u32 %j, %j, %bdim;
    bra NM;
NMD:

DONE:
    ret;
}
";

const RMSNORM_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.shared .align 4 .f32 rmsnorm_f16_sdata[256];

.visible .entry rmsnorm_f16_kernel(
    .param .u64 in_ptr,
    .param .u64 w_ptr,
    .param .u64 out_ptr,
    .param .u32 rows,
    .param .u32 cols,
    .param .f32 eps
) {
    .reg .u32 %r_tid, %bid, %bdim, %rows_reg, %cols_reg, %j, %half, %otid;
    .reg .u64 %in, %w, %out, %row_off, %off, %sbase, %saddr;
    .reg .b16 %x_b16, %w_b16, %out_h;
    .reg .f32 %x_f, %w_f, %sq_sum, %eps_r, %inv_rms, %mean_sq, %r_f, %other, %n_f;
    .reg .pred %p, %lp, %rp;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %w, [w_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %rows_reg, [rows];
    ld.param.u32 %cols_reg, [cols];
    ld.param.f32 %eps_r, [eps];

    mov.u64 %sbase, rmsnorm_f16_sdata;

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;

    setp.ge.u32 %p, %bid, %rows_reg;
    @%p bra DONE;

    cvt.u64.u32 %row_off, %bid;
    cvt.u64.u32 %off, %cols_reg;
    mul.lo.u64 %row_off, %row_off, %off;
    shl.b64 %row_off, %row_off, 1;
    cvt.rn.f32.u32 %n_f, %cols_reg;

    // Phase 1: sum(x^2) in f32
    mov.f32 %sq_sum, 0f00000000;
    mov.u32 %j, %r_tid;
SS:
    setp.ge.u32 %lp, %j, %cols_reg;
    @%lp bra SSD;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %in, %off;
    add.u64 %off, %off, %row_off;
    ld.global.b16 %x_b16, [%off];
    cvt.f32.f16 %x_f, %x_b16;
    fma.rn.f32 %sq_sum, %x_f, %x_f, %sq_sum;
    add.u32 %j, %j, %bdim;
    bra SS;
SSD:
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %sq_sum;
    bar.sync 0;

    mov.u32 %half, %bdim;
SR:
    shr.u32 %half, %half, 1;
    setp.eq.u32 %rp, %half, 0;
    @%rp bra SRD;
    setp.ge.u32 %rp, %r_tid, %half;
    @%rp bra SRS;
    add.u32 %otid, %r_tid, %half;
    cvt.u64.u32 %off, %otid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other, [%saddr];
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %sq_sum, [%saddr];
    add.f32 %sq_sum, %sq_sum, %other;
    st.shared.f32 [%saddr], %sq_sum;
SRS:
    bar.sync 0;
    bra SR;
SRD:
    ld.shared.f32 %sq_sum, [%sbase];
    div.approx.f32 %mean_sq, %sq_sum, %n_f;
    add.f32 %mean_sq, %mean_sq, %eps_r;
    sqrt.approx.f32 %inv_rms, %mean_sq;
    rcp.approx.f32 %inv_rms, %inv_rms;
    bar.sync 0;

    // Phase 2: out = x * inv_rms * weight, rounded to f16
    mov.u32 %j, %r_tid;
NM:
    setp.ge.u32 %lp, %j, %cols_reg;
    @%lp bra NMD;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %in, %off;
    add.u64 %off, %off, %row_off;
    ld.global.b16 %x_b16, [%off];
    cvt.f32.f16 %x_f, %x_b16;

    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %w, %off;
    ld.global.b16 %w_b16, [%off];
    cvt.f32.f16 %w_f, %w_b16;

    mul.f32 %r_f, %x_f, %inv_rms;
    mul.f32 %r_f, %r_f, %w_f;

    cvt.rn.f16.f32 %out_h, %r_f;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %out, %off;
    add.u64 %off, %off, %row_off;
    st.global.b16 [%off], %out_h;
    add.u32 %j, %j, %bdim;
    bra NM;
NMD:

DONE:
    ret;
}
";

/// Apply LayerNorm to an f16 `[rows, cols]` row-major tensor on the GPU.
/// `out[r, c] = ((x - mean) / sqrt(var + eps)) * gamma[c] + beta[c]` with
/// row mean/variance reduced in f32, result rounded back to f16 (RNE).
pub fn gpu_layernorm_f16(
    input: &cudarc::driver::CudaSlice<u16>,
    gamma: &cudarc::driver::CudaSlice<u16>,
    beta: &cudarc::driver::CudaSlice<u16>,
    rows: usize,
    cols: usize,
    eps: f32,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    if rows == 0 || cols == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(rows * cols)?);
    }
    if input.len() < rows * cols {
        return Err(GpuError::ShapeMismatch {
            op: "layernorm_f16",
            expected: vec![rows, cols],
            got: vec![input.len()],
        });
    }
    if gamma.len() < cols || beta.len() < cols {
        return Err(GpuError::ShapeMismatch {
            op: "layernorm_f16",
            expected: vec![cols],
            got: vec![gamma.len().min(beta.len())],
        });
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile(
        ctx,
        LAYERNORM_F16_PTX,
        "layernorm_f16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "layernorm_f16_kernel",
        source: e,
    })?;

    let mut out = stream.alloc_zeros::<u16>(rows * cols)?;
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    // SAFETY:
    // - `f` is the `layernorm_f16_kernel` PTX entry; signature (in_ptr,
    //   gamma_ptr, beta_ptr, out_ptr, rows, cols, eps) matches the seven
    //   args below in order.
    // - `input` checked `>= rows*cols`; `gamma`/`beta` checked `>= cols`;
    //   empty-shape early-return ensures `rows > 0 && cols > 0` at launch.
    // - `out` alloc'd exactly `rows*cols` u16 elements; only `&mut out`.
    // - `eps` is a stack f32 borrowed only for the `arg(&eps)` copy.
    // - One block per row; row index and column loops bound-guarded,
    //   confining reads/writes to `[0, rows*cols)` of in/out and `[0, cols)`
    //   of gamma/beta. 256 f32 shared words match the 256-thread block.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(gamma)
            .arg(beta)
            .arg(&mut out)
            .arg(&rows_u32)
            .arg(&cols_u32)
            .arg(&eps)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Apply RMSNorm to an f16 `[rows, cols]` row-major tensor on the GPU.
/// `out[r, c] = (x / sqrt(mean(x^2) + eps)) * weight[c]`, reductions in f32,
/// result rounded back to f16 (RNE).
pub fn gpu_rmsnorm_f16(
    input: &cudarc::driver::CudaSlice<u16>,
    weight: &cudarc::driver::CudaSlice<u16>,
    rows: usize,
    cols: usize,
    eps: f32,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    if rows == 0 || cols == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(rows * cols)?);
    }
    if input.len() < rows * cols {
        return Err(GpuError::ShapeMismatch {
            op: "rmsnorm_f16",
            expected: vec![rows, cols],
            got: vec![input.len()],
        });
    }
    if weight.len() < cols {
        return Err(GpuError::ShapeMismatch {
            op: "rmsnorm_f16",
            expected: vec![cols],
            got: vec![weight.len()],
        });
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile(
        ctx,
        RMSNORM_F16_PTX,
        "rmsnorm_f16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "rmsnorm_f16_kernel",
        source: e,
    })?;

    let mut out = stream.alloc_zeros::<u16>(rows * cols)?;
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    // SAFETY:
    // - `f` is the `rmsnorm_f16_kernel` PTX entry; signature (in_ptr,
    //   w_ptr, out_ptr, rows, cols, eps) matches the six args below.
    // - `input` checked `>= rows*cols`; `weight` checked `>= cols`;
    //   empty-shape early-return ensures `rows > 0 && cols > 0`.
    // - `out` alloc'd exactly `rows*cols` u16 elements; only `&mut out`.
    // - `eps` is a stack f32 borrowed only for the `arg(&eps)` copy.
    // - One block per row; bound-guarded row index + column loops confine
    //   reads/writes to `[0, rows*cols)` of in/out and `[0, cols)` of
    //   weight. 256 f32 shared words match the 256-thread block.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(weight)
            .arg(&mut out)
            .arg(&rows_u32)
            .arg(&cols_u32)
            .arg(&eps)
            .launch(cfg)?;
    }
    Ok(out)
}

// ===========================================================================
// Broadcast binary ops (add/sub/mul/div on N-D broadcast shapes)
// ===========================================================================
//
// For each output element, decompose the flat output index into N-D
// coordinates, dot with broadcast strides for each operand, load the f16
// elements, compute in f32, and round back to f16.

const BROADCAST_ADD_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry broadcast_add_f16_kernel(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u64 a_strides_ptr,
    .param .u64 b_strides_ptr,
    .param .u64 out_shape_ptr,
    .param .u32 n,
    .param .u32 ndim
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %ndim_reg;
    .reg .u32 %remaining, %a_idx, %b_idx, %d;
    .reg .u32 %shape_d, %a_str_d, %b_str_d, %coord;
    .reg .u64 %a, %b, %out, %a_str, %b_str, %oshape;
    .reg .u64 %off_a, %off_b, %off_out, %d64, %tmp;
    .reg .b16 %a_b16, %b_b16, %out_h;
    .reg .f32 %va, %vb, %vr;
    .reg .pred %p, %loop_p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %a_str, [a_strides_ptr];
    ld.param.u64 %b_str, [b_strides_ptr];
    ld.param.u64 %oshape, [out_shape_ptr];
    ld.param.u32 %n_reg, [n];
    ld.param.u32 %ndim_reg, [ndim];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    mov.u32 %remaining, %r_tid;
    mov.u32 %a_idx, 0;
    mov.u32 %b_idx, 0;
    mov.u32 %d, %ndim_reg;

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
    shl.b64 %off_a, %off_a, 1;
    add.u64 %off_a, %a, %off_a;
    ld.global.b16 %a_b16, [%off_a];

    cvt.u64.u32 %off_b, %b_idx;
    shl.b64 %off_b, %off_b, 1;
    add.u64 %off_b, %b, %off_b;
    ld.global.b16 %b_b16, [%off_b];

    cvt.f32.f16 %va, %a_b16;
    cvt.f32.f16 %vb, %b_b16;
    add.f32 %vr, %va, %vb;

    cvt.rn.f16.f32 %out_h, %vr;
    cvt.u64.u32 %off_out, %r_tid;
    shl.b64 %off_out, %off_out, 1;
    add.u64 %off_out, %out, %off_out;
    st.global.b16 [%off_out], %out_h;

DONE:
    ret;
}
";

const BROADCAST_SUB_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry broadcast_sub_f16_kernel(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u64 a_strides_ptr,
    .param .u64 b_strides_ptr,
    .param .u64 out_shape_ptr,
    .param .u32 n,
    .param .u32 ndim
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %ndim_reg;
    .reg .u32 %remaining, %a_idx, %b_idx, %d;
    .reg .u32 %shape_d, %a_str_d, %b_str_d, %coord;
    .reg .u64 %a, %b, %out, %a_str, %b_str, %oshape;
    .reg .u64 %off_a, %off_b, %off_out, %d64, %tmp;
    .reg .b16 %a_b16, %b_b16, %out_h;
    .reg .f32 %va, %vb, %vr;
    .reg .pred %p, %loop_p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %a_str, [a_strides_ptr];
    ld.param.u64 %b_str, [b_strides_ptr];
    ld.param.u64 %oshape, [out_shape_ptr];
    ld.param.u32 %n_reg, [n];
    ld.param.u32 %ndim_reg, [ndim];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    mov.u32 %remaining, %r_tid;
    mov.u32 %a_idx, 0;
    mov.u32 %b_idx, 0;
    mov.u32 %d, %ndim_reg;

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
    shl.b64 %off_a, %off_a, 1;
    add.u64 %off_a, %a, %off_a;
    ld.global.b16 %a_b16, [%off_a];

    cvt.u64.u32 %off_b, %b_idx;
    shl.b64 %off_b, %off_b, 1;
    add.u64 %off_b, %b, %off_b;
    ld.global.b16 %b_b16, [%off_b];

    cvt.f32.f16 %va, %a_b16;
    cvt.f32.f16 %vb, %b_b16;
    sub.f32 %vr, %va, %vb;

    cvt.rn.f16.f32 %out_h, %vr;
    cvt.u64.u32 %off_out, %r_tid;
    shl.b64 %off_out, %off_out, 1;
    add.u64 %off_out, %out, %off_out;
    st.global.b16 [%off_out], %out_h;

DONE:
    ret;
}
";

const BROADCAST_MUL_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry broadcast_mul_f16_kernel(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u64 a_strides_ptr,
    .param .u64 b_strides_ptr,
    .param .u64 out_shape_ptr,
    .param .u32 n,
    .param .u32 ndim
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %ndim_reg;
    .reg .u32 %remaining, %a_idx, %b_idx, %d;
    .reg .u32 %shape_d, %a_str_d, %b_str_d, %coord;
    .reg .u64 %a, %b, %out, %a_str, %b_str, %oshape;
    .reg .u64 %off_a, %off_b, %off_out, %d64, %tmp;
    .reg .b16 %a_b16, %b_b16, %out_h;
    .reg .f32 %va, %vb, %vr;
    .reg .pred %p, %loop_p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %a_str, [a_strides_ptr];
    ld.param.u64 %b_str, [b_strides_ptr];
    ld.param.u64 %oshape, [out_shape_ptr];
    ld.param.u32 %n_reg, [n];
    ld.param.u32 %ndim_reg, [ndim];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    mov.u32 %remaining, %r_tid;
    mov.u32 %a_idx, 0;
    mov.u32 %b_idx, 0;
    mov.u32 %d, %ndim_reg;

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
    shl.b64 %off_a, %off_a, 1;
    add.u64 %off_a, %a, %off_a;
    ld.global.b16 %a_b16, [%off_a];

    cvt.u64.u32 %off_b, %b_idx;
    shl.b64 %off_b, %off_b, 1;
    add.u64 %off_b, %b, %off_b;
    ld.global.b16 %b_b16, [%off_b];

    cvt.f32.f16 %va, %a_b16;
    cvt.f32.f16 %vb, %b_b16;
    mul.f32 %vr, %va, %vb;

    cvt.rn.f16.f32 %out_h, %vr;
    cvt.u64.u32 %off_out, %r_tid;
    shl.b64 %off_out, %off_out, 1;
    add.u64 %off_out, %out, %off_out;
    st.global.b16 [%off_out], %out_h;

DONE:
    ret;
}
";

const BROADCAST_DIV_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry broadcast_div_f16_kernel(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u64 a_strides_ptr,
    .param .u64 b_strides_ptr,
    .param .u64 out_shape_ptr,
    .param .u32 n,
    .param .u32 ndim
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %ndim_reg;
    .reg .u32 %remaining, %a_idx, %b_idx, %d;
    .reg .u32 %shape_d, %a_str_d, %b_str_d, %coord;
    .reg .u64 %a, %b, %out, %a_str, %b_str, %oshape;
    .reg .u64 %off_a, %off_b, %off_out, %d64, %tmp;
    .reg .b16 %a_b16, %b_b16, %out_h;
    .reg .f32 %va, %vb, %vr, %nan;
    .reg .pred %p, %loop_p, %nan_a, %nan_b, %store_nan;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %a_str, [a_strides_ptr];
    ld.param.u64 %b_str, [b_strides_ptr];
    ld.param.u64 %oshape, [out_shape_ptr];
    ld.param.u32 %n_reg, [n];
    ld.param.u32 %ndim_reg, [ndim];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    mov.u32 %remaining, %r_tid;
    mov.u32 %a_idx, 0;
    mov.u32 %b_idx, 0;
    mov.u32 %d, %ndim_reg;

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
    shl.b64 %off_a, %off_a, 1;
    add.u64 %off_a, %a, %off_a;
    ld.global.b16 %a_b16, [%off_a];

    cvt.u64.u32 %off_b, %b_idx;
    shl.b64 %off_b, %off_b, 1;
    add.u64 %off_b, %b, %off_b;
    ld.global.b16 %b_b16, [%off_b];

    cvt.f32.f16 %va, %a_b16;
    cvt.f32.f16 %vb, %b_b16;
    setp.nan.f32 %nan_a, %va, %va;
    setp.nan.f32 %nan_b, %vb, %vb;
    or.pred %store_nan, %nan_a, %nan_b;
    @%store_nan bra STORE_NAN;
    div.rn.f32 %vr, %va, %vb;

    cvt.rn.f16.f32 %out_h, %vr;
    cvt.u64.u32 %off_out, %r_tid;
    shl.b64 %off_out, %off_out, 1;
    add.u64 %off_out, %out, %off_out;
    st.global.b16 [%off_out], %out_h;
    bra DONE;

STORE_NAN:
    mov.f32 %nan, 0f7FC00000;
    cvt.rn.f16.f32 %out_h, %nan;
    cvt.u64.u32 %off_out, %r_tid;
    shl.b64 %off_out, %off_out, 1;
    add.u64 %off_out, %out, %off_out;
    st.global.b16 [%off_out], %out_h;

DONE:
    ret;
}
";

const BROADCAST_ADDCMUL_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry broadcast_addcmul_f16_kernel(
    .param .u64 input_ptr,
    .param .u64 tensor1_ptr,
    .param .u64 tensor2_ptr,
    .param .u64 out_ptr,
    .param .u64 input_strides_ptr,
    .param .u64 tensor1_strides_ptr,
    .param .u64 tensor2_strides_ptr,
    .param .u64 out_shape_ptr,
    .param .f32 value,
    .param .u32 n,
    .param .u32 ndim
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %ndim_reg;
    .reg .u32 %remaining, %input_idx, %t1_idx, %t2_idx, %d;
    .reg .u32 %shape_d, %input_str_d, %t1_str_d, %t2_str_d, %coord;
    .reg .u64 %input, %t1, %t2, %out, %input_str, %t1_str, %t2_str, %oshape;
    .reg .u64 %off, %off_a, %off_b, %off_out, %d64, %tmp;
    .reg .b16 %input_h, %t1_h, %t2_h, %out_h;
    .reg .f32 %vi, %v1, %v2, %vr, %value, %prod;
    .reg .pred %p, %loop_p, %value_is_one;

    ld.param.u64 %input, [input_ptr];
    ld.param.u64 %t1, [tensor1_ptr];
    ld.param.u64 %t2, [tensor2_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %input_str, [input_strides_ptr];
    ld.param.u64 %t1_str, [tensor1_strides_ptr];
    ld.param.u64 %t2_str, [tensor2_strides_ptr];
    ld.param.u64 %oshape, [out_shape_ptr];
    ld.param.f32 %value, [value];
    ld.param.u32 %n_reg, [n];
    ld.param.u32 %ndim_reg, [ndim];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;
    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    mov.u32 %remaining, %r_tid;
    mov.u32 %input_idx, 0;
    mov.u32 %t1_idx, 0;
    mov.u32 %t2_idx, 0;
    mov.u32 %d, %ndim_reg;
LOOP:
    setp.eq.u32 %loop_p, %d, 0;
    @%loop_p bra END_LOOP;
    sub.u32 %d, %d, 1;
    cvt.u64.u32 %d64, %d;
    shl.b64 %d64, %d64, 2;
    add.u64 %tmp, %oshape, %d64;
    ld.global.u32 %shape_d, [%tmp];
    add.u64 %tmp, %input_str, %d64;
    ld.global.u32 %input_str_d, [%tmp];
    add.u64 %tmp, %t1_str, %d64;
    ld.global.u32 %t1_str_d, [%tmp];
    add.u64 %tmp, %t2_str, %d64;
    ld.global.u32 %t2_str_d, [%tmp];
    rem.u32 %coord, %remaining, %shape_d;
    div.u32 %remaining, %remaining, %shape_d;
    mad.lo.u32 %input_idx, %coord, %input_str_d, %input_idx;
    mad.lo.u32 %t1_idx, %coord, %t1_str_d, %t1_idx;
    mad.lo.u32 %t2_idx, %coord, %t2_str_d, %t2_idx;
    bra LOOP;
END_LOOP:
    cvt.u64.u32 %off, %input_idx;
    shl.b64 %off, %off, 1;
    add.u64 %off, %input, %off;
    ld.global.b16 %input_h, [%off];
    cvt.u64.u32 %off_a, %t1_idx;
    shl.b64 %off_a, %off_a, 1;
    add.u64 %off_a, %t1, %off_a;
    ld.global.b16 %t1_h, [%off_a];
    cvt.u64.u32 %off_b, %t2_idx;
    shl.b64 %off_b, %off_b, 1;
    add.u64 %off_b, %t2, %off_b;
    ld.global.b16 %t2_h, [%off_b];

    cvt.f32.f16 %vi, %input_h;
    cvt.f32.f16 %v1, %t1_h;
    cvt.f32.f16 %v2, %t2_h;
    setp.eq.f32 %value_is_one, %value, 0f3F800000;
    @%value_is_one fma.rn.f32 %vr, %v1, %v2, %vi;
    @%value_is_one bra STORE;
    mul.f32 %prod, %v1, %v2;
    fma.rn.f32 %vr, %value, %prod, %vi;
STORE:
    cvt.rn.f16.f32 %out_h, %vr;
    cvt.u64.u32 %off_out, %r_tid;
    shl.b64 %off_out, %off_out, 1;
    add.u64 %off_out, %out, %off_out;
    st.global.b16 [%off_out], %out_h;
DONE:
    ret;
}
";

const BROADCAST_ADDCDIV_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry broadcast_addcdiv_f16_kernel(
    .param .u64 input_ptr,
    .param .u64 tensor1_ptr,
    .param .u64 tensor2_ptr,
    .param .u64 out_ptr,
    .param .u64 input_strides_ptr,
    .param .u64 tensor1_strides_ptr,
    .param .u64 tensor2_strides_ptr,
    .param .u64 out_shape_ptr,
    .param .f32 value,
    .param .u32 n,
    .param .u32 ndim
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %ndim_reg;
    .reg .u32 %remaining, %input_idx, %t1_idx, %t2_idx, %d;
    .reg .u32 %shape_d, %input_str_d, %t1_str_d, %t2_str_d, %coord;
    .reg .u64 %input, %t1, %t2, %out, %input_str, %t1_str, %t2_str, %oshape;
    .reg .u64 %off, %off_a, %off_b, %off_out, %d64, %tmp;
    .reg .b16 %input_h, %t1_h, %t2_h, %out_h;
    .reg .f32 %vi, %v1, %v2, %vr, %value, %quot;
    .reg .pred %p, %loop_p;

    ld.param.u64 %input, [input_ptr];
    ld.param.u64 %t1, [tensor1_ptr];
    ld.param.u64 %t2, [tensor2_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %input_str, [input_strides_ptr];
    ld.param.u64 %t1_str, [tensor1_strides_ptr];
    ld.param.u64 %t2_str, [tensor2_strides_ptr];
    ld.param.u64 %oshape, [out_shape_ptr];
    ld.param.f32 %value, [value];
    ld.param.u32 %n_reg, [n];
    ld.param.u32 %ndim_reg, [ndim];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;
    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    mov.u32 %remaining, %r_tid;
    mov.u32 %input_idx, 0;
    mov.u32 %t1_idx, 0;
    mov.u32 %t2_idx, 0;
    mov.u32 %d, %ndim_reg;
LOOP:
    setp.eq.u32 %loop_p, %d, 0;
    @%loop_p bra END_LOOP;
    sub.u32 %d, %d, 1;
    cvt.u64.u32 %d64, %d;
    shl.b64 %d64, %d64, 2;
    add.u64 %tmp, %oshape, %d64;
    ld.global.u32 %shape_d, [%tmp];
    add.u64 %tmp, %input_str, %d64;
    ld.global.u32 %input_str_d, [%tmp];
    add.u64 %tmp, %t1_str, %d64;
    ld.global.u32 %t1_str_d, [%tmp];
    add.u64 %tmp, %t2_str, %d64;
    ld.global.u32 %t2_str_d, [%tmp];
    rem.u32 %coord, %remaining, %shape_d;
    div.u32 %remaining, %remaining, %shape_d;
    mad.lo.u32 %input_idx, %coord, %input_str_d, %input_idx;
    mad.lo.u32 %t1_idx, %coord, %t1_str_d, %t1_idx;
    mad.lo.u32 %t2_idx, %coord, %t2_str_d, %t2_idx;
    bra LOOP;
END_LOOP:
    cvt.u64.u32 %off, %input_idx;
    shl.b64 %off, %off, 1;
    add.u64 %off, %input, %off;
    ld.global.b16 %input_h, [%off];
    cvt.u64.u32 %off_a, %t1_idx;
    shl.b64 %off_a, %off_a, 1;
    add.u64 %off_a, %t1, %off_a;
    ld.global.b16 %t1_h, [%off_a];
    cvt.u64.u32 %off_b, %t2_idx;
    shl.b64 %off_b, %off_b, 1;
    add.u64 %off_b, %t2, %off_b;
    ld.global.b16 %t2_h, [%off_b];

    cvt.f32.f16 %vi, %input_h;
    cvt.f32.f16 %v1, %t1_h;
    cvt.f32.f16 %v2, %t2_h;
    div.rn.f32 %quot, %v1, %v2;
    fma.rn.f32 %vr, %value, %quot, %vi;
    cvt.rn.f16.f32 %out_h, %vr;
    cvt.u64.u32 %off_out, %r_tid;
    shl.b64 %off_out, %off_out, 1;
    add.u64 %off_out, %out, %off_out;
    st.global.b16 [%off_out], %out_h;
DONE:
    ret;
}
";

fn broadcast_extreme_f16_ptx(kernel_name: &str, instruction: &str) -> String {
    let op_block = format!(
        "\
    setp.nan.f32 %nan_a, %va, %va;
    setp.nan.f32 %nan_b, %vb, %vb;
    or.pred %store_nan, %nan_a, %nan_b;
    @%store_nan bra STORE_NAN;
    {instruction} %vr, %va, %vb;
    bra STORE_RESULT;

STORE_NAN:
    mov.f32 %vr, 0f7FC00000;
STORE_RESULT:"
    );
    BROADCAST_ADD_F16_PTX
        .replace("broadcast_add_f16_kernel", kernel_name)
        .replace(
            ".reg .pred %p, %loop_p;",
            ".reg .pred %p, %loop_p, %nan_a, %nan_b, %store_nan;",
        )
        .replace("    add.f32 %vr, %va, %vb;", &op_block)
}

const F16_FMOD_BLOCK: &str = "\
    setp.eq.f32 %zero_b, %vb, 0f00000000;
    setp.nan.f32 %nan_a, %va, %va;
    setp.nan.f32 %nan_b, %vb, %vb;
    abs.f32 %abs_a, %va;
    abs.f32 %abs_b, %vb;
    setp.eq.f32 %a_inf, %abs_a, 0f7F800000;
    or.pred %store_nan, %zero_b, %nan_a;
    or.pred %store_nan, %store_nan, %nan_b;
    or.pred %store_nan, %store_nan, %a_inf;
    @%store_nan bra STORE_NAN;

    setp.eq.f32 %b_inf, %abs_b, 0f7F800000;
    @%b_inf mov.f32 %vr, %va;
    @%b_inf bra F16_MOD_DONE;

    div.rn.f32 %q, %va, %vb;
    cvt.rzi.f32.f32 %trunc_q, %q;
    neg.f32 %ftmp, %trunc_q;
    fma.rn.f32 %vr, %ftmp, %vb, %va;
F16_MOD_DONE:";

const F16_REMAINDER_BLOCK: &str = "\
    setp.eq.f32 %zero_b, %vb, 0f00000000;
    setp.nan.f32 %nan_a, %va, %va;
    setp.nan.f32 %nan_b, %vb, %vb;
    abs.f32 %abs_a, %va;
    abs.f32 %abs_b, %vb;
    setp.eq.f32 %a_inf, %abs_a, 0f7F800000;
    or.pred %store_nan, %zero_b, %nan_a;
    or.pred %store_nan, %store_nan, %nan_b;
    or.pred %store_nan, %store_nan, %a_inf;
    @%store_nan bra STORE_NAN;

    setp.eq.f32 %b_inf, %abs_b, 0f7F800000;
    @%b_inf mov.f32 %m, %va;
    @%b_inf bra F16_REMAINDER_SIGN;

    div.rn.f32 %q, %va, %vb;
    cvt.rzi.f32.f32 %trunc_q, %q;
    neg.f32 %ftmp, %trunc_q;
    fma.rn.f32 %m, %ftmp, %vb, %va;

F16_REMAINDER_SIGN:
    setp.ne.f32 %m_nonzero, %m, 0f00000000;
    setp.lt.f32 %b_neg, %vb, 0f00000000;
    setp.lt.f32 %m_neg, %m, 0f00000000;
    not.pred %b_nonneg, %b_neg;
    not.pred %m_nonneg, %m_neg;
    and.pred %sign_a, %b_neg, %m_nonneg;
    and.pred %sign_b, %b_nonneg, %m_neg;
    or.pred %adjust, %sign_a, %sign_b;
    and.pred %adjust, %adjust, %m_nonzero;
    @%adjust add.f32 %m, %m, %vb;
    mov.f32 %vr, %m;";

const F16_DIV_TRUNC_BLOCK: &str = "\
    div.rn.f32 %q, %va, %vb;
    cvt.rzi.f32.f32 %vr, %q;";

const F16_DIV_FLOOR_BLOCK: &str = "\
    div.rn.f32 %q, %va, %vb;
    setp.eq.f32 %zero_b, %vb, 0f00000000;
    setp.eq.f32 %a_zero, %va, 0f00000000;
    and.pred %zero_zero, %zero_b, %a_zero;
    @%zero_zero bra STORE_NAN;
    @%zero_b mov.f32 %vr, %q;
    @%zero_b bra F16_DIV_ROUNDING_DONE;

    setp.nan.f32 %nan_a, %va, %va;
    setp.nan.f32 %nan_b, %vb, %vb;
    abs.f32 %abs_a, %va;
    abs.f32 %abs_b, %vb;
    setp.eq.f32 %a_inf, %abs_a, 0f7F800000;
    or.pred %store_nan, %nan_a, %nan_b;
    or.pred %store_nan, %store_nan, %a_inf;
    @%store_nan bra STORE_NAN;

    setp.eq.f32 %b_inf, %abs_b, 0f7F800000;
    @%b_inf bra F16_DIV_B_INF;

    cvt.rzi.f32.f32 %trunc_q, %q;
    neg.f32 %ftmp, %trunc_q;
    fma.rn.f32 %m, %ftmp, %vb, %va;
    sub.f32 %ftmp, %va, %m;
    div.rn.f32 %divv, %ftmp, %vb;

    setp.ne.f32 %m_nonzero, %m, 0f00000000;
    setp.lt.f32 %b_neg, %vb, 0f00000000;
    setp.lt.f32 %m_neg, %m, 0f00000000;
    not.pred %b_nonneg, %b_neg;
    not.pred %m_nonneg, %m_neg;
    and.pred %sign_a, %b_neg, %m_nonneg;
    and.pred %sign_b, %b_nonneg, %m_neg;
    or.pred %adjust, %sign_a, %sign_b;
    and.pred %adjust, %adjust, %m_nonzero;
    @%adjust sub.f32 %divv, %divv, 0f3F800000;

    setp.eq.f32 %div_zero, %divv, 0f00000000;
    @%div_zero mul.f32 %vr, %q, 0f00000000;
    @%div_zero bra F16_DIV_ROUNDING_DONE;

    cvt.rmi.f32.f32 %floorv, %divv;
    sub.f32 %ftmp, %divv, %floorv;
    setp.gt.f32 %gt_half, %ftmp, 0f3F000000;
    @%gt_half add.f32 %floorv, %floorv, 0f3F800000;
    mov.f32 %vr, %floorv;
    bra F16_DIV_ROUNDING_DONE;

F16_DIV_B_INF:
    setp.ne.f32 %a_nonzero, %va, 0f00000000;
    setp.lt.f32 %b_neg, %vb, 0f00000000;
    setp.lt.f32 %m_neg, %va, 0f00000000;
    not.pred %b_nonneg, %b_neg;
    not.pred %m_nonneg, %m_neg;
    and.pred %sign_a, %b_neg, %m_nonneg;
    and.pred %sign_b, %b_nonneg, %m_neg;
    or.pred %adjust, %sign_a, %sign_b;
    and.pred %adjust, %adjust, %a_nonzero;
    mul.f32 %vr, %q, 0f00000000;
    @%adjust mov.f32 %vr, 0fBF800000;
F16_DIV_ROUNDING_DONE:";

fn broadcast_div_like_f16_ptx(kernel_name: &str, op_block: &str) -> String {
    BROADCAST_DIV_F16_PTX
        .replace("broadcast_div_f16_kernel", kernel_name)
        .replace(
            ".reg .f32 %va, %vb, %vr, %nan;",
            ".reg .f32 %va, %vb, %vr, %nan, %q, %trunc_q, %m, %ftmp, %divv, %floorv, %abs_a, %abs_b;",
        )
        .replace(
            ".reg .pred %p, %loop_p, %nan_a, %nan_b, %store_nan;",
            ".reg .pred %p, %loop_p, %nan_a, %nan_b, %store_nan, %zero_b, %a_inf, %b_inf, \
             %m_nonzero, %b_neg, %m_neg, %b_nonneg, %m_nonneg, %sign_a, %sign_b, \
             %adjust, %div_zero, %gt_half, %a_nonzero, %a_zero, %zero_zero;",
        )
        .replace(
            "    setp.nan.f32 %nan_a, %va, %va;\n    setp.nan.f32 %nan_b, %vb, %vb;\n    or.pred %store_nan, %nan_a, %nan_b;\n    @%store_nan bra STORE_NAN;\n    div.rn.f32 %vr, %va, %vb;",
            op_block,
        )
}

// Computes row-major contiguous strides for `shape`, with broadcast (0)
// stride for dims of size 1. Mirrors the helper in `bf16.rs`.
fn broadcast_strides_f16(shape: &[usize], out_shape: &[usize]) -> Vec<u32> {
    let offset = out_shape.len() - shape.len();
    let mut strides = vec![0_u32; out_shape.len()];
    if !shape.is_empty() {
        let mut row_major = vec![1_usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            row_major[i] = row_major[i + 1] * shape[i + 1];
        }
        for (i, st) in strides.iter_mut().enumerate() {
            if i < offset {
                *st = 0;
            } else {
                let si = i - offset;
                if shape[si] == 1 {
                    *st = 0;
                } else {
                    *st = row_major[si] as u32;
                }
            }
        }
    }
    strides
}

#[allow(clippy::too_many_arguments)]
fn launch_broadcast_binary_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    b: &cudarc::driver::CudaSlice<u16>,
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    let out_numel: usize = crate::shape_math::numel(out_shape);
    let stream = device.stream();
    if out_numel == 0 {
        return Ok(stream.alloc_zeros::<u16>(0)?);
    }
    let a_str = broadcast_strides_f16(a_shape, out_shape);
    let b_str = broadcast_strides_f16(b_shape, out_shape);
    let shape_u32: Vec<u32> = out_shape.iter().map(|&d| d as u32).collect();

    let ctx = device.context();
    let f = get_or_compile(ctx, ptx, kernel_name, device.ordinal() as u32).map_err(|e| {
        GpuError::PtxCompileFailed {
            kernel: kernel_name,
            source: e,
        }
    })?;
    let a_str_buf = crate::transfer::cpu_to_gpu(&a_str, device)?;
    let b_str_buf = crate::transfer::cpu_to_gpu(&b_str, device)?;
    let shape_buf = crate::transfer::cpu_to_gpu(&shape_u32, device)?;
    let mut out = stream.alloc_zeros::<u16>(out_numel)?;
    let cfg = launch_1d(out_numel);
    let n_u32 = out_numel as u32;
    let ndim_u32 = out_shape.len() as u32;

    // SAFETY:
    // - `f` resolves to one of the four `broadcast_*_f16_kernel` PTX entry
    //   points with signature (a_ptr, b_ptr, out_ptr, a_str_ptr, b_str_ptr,
    //   shape_ptr, n, ndim). The eight args below match in order.
    // - `a`, `b` are caller-supplied f16 input buffers; the kernel only
    //   reads them at indices computed by collapsing `out_shape` against
    //   `a_str`/`b_str`. Broadcast (size-1) dims have stride 0, so the
    //   resulting linear offset stays inside the caller's buffers.
    // - `a_str_buf`, `b_str_buf`, `shape_buf` were freshly uploaded above
    //   from `&[u32]` slices of length `out_shape.len()` (== `ndim_u32`),
    //   and are kept alive across the launch by the `_kp*` bindings.
    // - `out` is freshly alloc'd `out_numel` u16 elements, uniquely borrowed.
    // - `n_u32` is non-truncating (`launch_1d` already cast it).
    // - `ndim_u32` is bounded by upstream rank validation (≤8).
    unsafe {
        let _kp = &a_str_buf;
        let _kp2 = &b_str_buf;
        let _kp3 = &shape_buf;
        stream
            .launch_builder(&f)
            .arg(a)
            .arg(b)
            .arg(&mut out)
            .arg(a_str_buf.inner())
            .arg(b_str_buf.inner())
            .arg(shape_buf.inner())
            .arg(&n_u32)
            .arg(&ndim_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn launch_broadcast_ternary_f16(
    input: &cudarc::driver::CudaSlice<u16>,
    tensor1: &cudarc::driver::CudaSlice<u16>,
    tensor2: &cudarc::driver::CudaSlice<u16>,
    input_shape: &[usize],
    tensor1_shape: &[usize],
    tensor2_shape: &[usize],
    out_shape: &[usize],
    value: f32,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    let out_numel: usize = crate::shape_math::numel(out_shape);
    let stream = device.stream();
    if out_numel == 0 {
        return Ok(stream.alloc_zeros::<u16>(0)?);
    }
    let input_str = broadcast_strides_f16(input_shape, out_shape);
    let tensor1_str = broadcast_strides_f16(tensor1_shape, out_shape);
    let tensor2_str = broadcast_strides_f16(tensor2_shape, out_shape);
    let shape_u32: Vec<u32> = out_shape.iter().map(|&d| d as u32).collect();

    let ctx = device.context();
    let f = get_or_compile(ctx, ptx, kernel_name, device.ordinal() as u32).map_err(|e| {
        GpuError::PtxCompileFailed {
            kernel: kernel_name,
            source: e,
        }
    })?;
    let input_str_buf = crate::transfer::cpu_to_gpu(&input_str, device)?;
    let tensor1_str_buf = crate::transfer::cpu_to_gpu(&tensor1_str, device)?;
    let tensor2_str_buf = crate::transfer::cpu_to_gpu(&tensor2_str, device)?;
    let shape_buf = crate::transfer::cpu_to_gpu(&shape_u32, device)?;
    let mut out = stream.alloc_zeros::<u16>(out_numel)?;
    let cfg = launch_1d(out_numel);
    let n_u32 = out_numel as u32;
    let ndim_u32 = out_shape.len() as u32;

    // SAFETY:
    // - `f` resolves to a `broadcast_addc{mul,div}_f16_kernel` entry with
    //   signature `(input, tensor1, tensor2, out, input_strides,
    //   tensor1_strides, tensor2_strides, out_shape, value, n, ndim)`.
    // - The three stride buffers and shape buffer were freshly uploaded from
    //   slices of length `out_shape.len()` and are kept alive across launch.
    // - The kernel reads f16 u16 storage only at row-major broadcast offsets
    //   computed from those stride buffers, and writes exactly one f16 u16
    //   element per output lane guarded by `tid < n`.
    unsafe {
        let _kp = &input_str_buf;
        let _kp2 = &tensor1_str_buf;
        let _kp3 = &tensor2_str_buf;
        let _kp4 = &shape_buf;
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(tensor1)
            .arg(tensor2)
            .arg(&mut out)
            .arg(input_str_buf.inner())
            .arg(tensor1_str_buf.inner())
            .arg(tensor2_str_buf.inner())
            .arg(shape_buf.inner())
            .arg(&value)
            .arg(&n_u32)
            .arg(&ndim_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Broadcast add `out[i] = a[bcast_a(i)] + b[bcast_b(i)]` on f16 buffers.
pub fn gpu_broadcast_add_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    b: &cudarc::driver::CudaSlice<u16>,
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    launch_broadcast_binary_f16(
        a,
        b,
        a_shape,
        b_shape,
        out_shape,
        device,
        BROADCAST_ADD_F16_PTX,
        "broadcast_add_f16_kernel",
    )
}

/// Broadcast sub `out[i] = a[bcast_a(i)] - b[bcast_b(i)]` on f16 buffers.
pub fn gpu_broadcast_sub_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    b: &cudarc::driver::CudaSlice<u16>,
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    launch_broadcast_binary_f16(
        a,
        b,
        a_shape,
        b_shape,
        out_shape,
        device,
        BROADCAST_SUB_F16_PTX,
        "broadcast_sub_f16_kernel",
    )
}

/// Broadcast mul `out[i] = a[bcast_a(i)] * b[bcast_b(i)]` on f16 buffers.
pub fn gpu_broadcast_mul_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    b: &cudarc::driver::CudaSlice<u16>,
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    launch_broadcast_binary_f16(
        a,
        b,
        a_shape,
        b_shape,
        out_shape,
        device,
        BROADCAST_MUL_F16_PTX,
        "broadcast_mul_f16_kernel",
    )
}

/// Broadcast div `out[i] = a[bcast_a(i)] / b[bcast_b(i)]` on f16 buffers.
pub fn gpu_broadcast_div_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    b: &cudarc::driver::CudaSlice<u16>,
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    launch_broadcast_binary_f16(
        a,
        b,
        a_shape,
        b_shape,
        out_shape,
        device,
        BROADCAST_DIV_F16_PTX,
        "broadcast_div_f16_kernel",
    )
}

/// Broadcast addcmul on f16 buffers with f32 opmath.
#[allow(clippy::too_many_arguments)]
pub fn gpu_broadcast_addcmul_f16(
    input: &cudarc::driver::CudaSlice<u16>,
    tensor1: &cudarc::driver::CudaSlice<u16>,
    tensor2: &cudarc::driver::CudaSlice<u16>,
    input_shape: &[usize],
    tensor1_shape: &[usize],
    tensor2_shape: &[usize],
    out_shape: &[usize],
    value: f32,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    launch_broadcast_ternary_f16(
        input,
        tensor1,
        tensor2,
        input_shape,
        tensor1_shape,
        tensor2_shape,
        out_shape,
        value,
        device,
        BROADCAST_ADDCMUL_F16_PTX,
        "broadcast_addcmul_f16_kernel",
    )
}

/// Broadcast addcdiv on f16 buffers with f32 opmath.
#[allow(clippy::too_many_arguments)]
pub fn gpu_broadcast_addcdiv_f16(
    input: &cudarc::driver::CudaSlice<u16>,
    tensor1: &cudarc::driver::CudaSlice<u16>,
    tensor2: &cudarc::driver::CudaSlice<u16>,
    input_shape: &[usize],
    tensor1_shape: &[usize],
    tensor2_shape: &[usize],
    out_shape: &[usize],
    value: f32,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    launch_broadcast_ternary_f16(
        input,
        tensor1,
        tensor2,
        input_shape,
        tensor1_shape,
        tensor2_shape,
        out_shape,
        value,
        device,
        BROADCAST_ADDCDIV_F16_PTX,
        "broadcast_addcdiv_f16_kernel",
    )
}

/// Broadcast rounded div on f16 buffers. `rounding_mode` is `trunc` or `floor`.
pub fn gpu_broadcast_div_rounding_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    b: &cudarc::driver::CudaSlice<u16>,
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
    rounding_mode: &str,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    let (ptx, kernel_name) = match rounding_mode {
        "trunc" => {
            static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
            (
                CACHE
                    .get_or_init(|| {
                        broadcast_div_like_f16_ptx(
                            "broadcast_div_trunc_f16_kernel",
                            F16_DIV_TRUNC_BLOCK,
                        )
                    })
                    .as_str(),
                "broadcast_div_trunc_f16_kernel",
            )
        }
        "floor" => {
            static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
            (
                CACHE
                    .get_or_init(|| {
                        broadcast_div_like_f16_ptx(
                            "broadcast_div_floor_f16_kernel",
                            F16_DIV_FLOOR_BLOCK,
                        )
                    })
                    .as_str(),
                "broadcast_div_floor_f16_kernel",
            )
        }
        other => {
            return Err(GpuError::InvalidState {
                message: format!("unsupported f16 div rounding_mode {other:?}"),
            });
        }
    };

    launch_broadcast_binary_f16(a, b, a_shape, b_shape, out_shape, device, ptx, kernel_name)
}

/// Broadcast fmod `out[i] = fmod(a, b)` on f16 buffers.
pub fn gpu_broadcast_fmod_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    b: &cudarc::driver::CudaSlice<u16>,
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let ptx: &'static str = CACHE
        .get_or_init(|| broadcast_div_like_f16_ptx("broadcast_fmod_f16_kernel", F16_FMOD_BLOCK))
        .as_str();
    launch_broadcast_binary_f16(
        a,
        b,
        a_shape,
        b_shape,
        out_shape,
        device,
        ptx,
        "broadcast_fmod_f16_kernel",
    )
}

/// Broadcast PyTorch remainder on f16 buffers.
pub fn gpu_broadcast_remainder_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    b: &cudarc::driver::CudaSlice<u16>,
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let ptx: &'static str = CACHE
        .get_or_init(|| {
            broadcast_div_like_f16_ptx("broadcast_remainder_f16_kernel", F16_REMAINDER_BLOCK)
        })
        .as_str();
    launch_broadcast_binary_f16(
        a,
        b,
        a_shape,
        b_shape,
        out_shape,
        device,
        ptx,
        "broadcast_remainder_f16_kernel",
    )
}

/// Broadcast max `out[i] = maximum(a[bcast_a(i)], b[bcast_b(i)])` on f16 buffers.
pub fn gpu_broadcast_maximum_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    b: &cudarc::driver::CudaSlice<u16>,
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let ptx: &'static str = CACHE
        .get_or_init(|| broadcast_extreme_f16_ptx("broadcast_maximum_f16_kernel", "max.f32"))
        .as_str();
    launch_broadcast_binary_f16(
        a,
        b,
        a_shape,
        b_shape,
        out_shape,
        device,
        ptx,
        "broadcast_maximum_f16_kernel",
    )
}

/// Broadcast min `out[i] = minimum(a[bcast_a(i)], b[bcast_b(i)])` on f16 buffers.
pub fn gpu_broadcast_minimum_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    b: &cudarc::driver::CudaSlice<u16>,
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let ptx: &'static str = CACHE
        .get_or_init(|| broadcast_extreme_f16_ptx("broadcast_minimum_f16_kernel", "min.f32"))
        .as_str();
    launch_broadcast_binary_f16(
        a,
        b,
        a_shape,
        b_shape,
        out_shape,
        device,
        ptx,
        "broadcast_minimum_f16_kernel",
    )
}
