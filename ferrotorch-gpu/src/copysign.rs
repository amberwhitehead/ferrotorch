//! CUDA kernels for `torch.copysign` parity.
//!
//! `copysign` is a bit-level primitive, not an arithmetic multiply by a sign
//! mask. The kernels in this module preserve signed zero and NaN sign/payload
//! behavior where PyTorch does, and intentionally encode PyTorch's reduced
//! precision CUDA quirks:
//!
//! - f32/f64: copy the raw sign bit and preserve the magnitude NaN payload.
//! - bf16: copy the sign bit for finite magnitudes; NaN magnitudes produce
//!   canonical positive `0x7fff`.
//! - f16: ignore NaN sign operands for finite magnitudes; NaN magnitudes produce
//!   canonical positive `0x7fff`.
//!
//! Backward is magnitude-only:
//! `grad_magnitude = where(magnitude == 0, 0, grad * result / magnitude)`.
//! It runs as a resident CUDA broadcast kernel so broadcasted inputs do not need
//! host expansion and signed zeros are not normalized through arithmetic.

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::buffer::CudaBuffer;
use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};
use crate::module_cache::get_or_compile;
use crate::transfer::{alloc_zeros_f32, alloc_zeros_f64, cpu_to_gpu};

const BLOCK_SIZE: u32 = 256;

const COPYSIGN_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry copysign_f32_kernel(
    .param .u64 mag_ptr,
    .param .u64 sign_ptr,
    .param .u64 out_ptr,
    .param .u64 mag_strides_ptr,
    .param .u64 sign_strides_ptr,
    .param .u64 out_shape_ptr,
    .param .s64 mag_offset,
    .param .s64 sign_offset,
    .param .u32 n,
    .param .u32 ndim
) {
    .reg .u32 %idx, %bid, %bdim, %nr, %ndimr;
    .reg .u32 %rem, %d, %shape_d, %coord;
    .reg .u32 %mbits, %sbits, %mag_abs, %sign_bit, %out_bits;
    .reg .u64 %mag, %sign, %out, %mstr, %sstr, %oshape;
    .reg .u64 %off_m, %off_s, %off_o, %d64, %tmp;
    .reg .s64 %midx, %sidx, %mstride, %sstride, %coord64, %prod;
    .reg .pred %p, %done_loop;

    ld.param.u64 %mag, [mag_ptr];
    ld.param.u64 %sign, [sign_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %mstr, [mag_strides_ptr];
    ld.param.u64 %sstr, [sign_strides_ptr];
    ld.param.u64 %oshape, [out_shape_ptr];
    ld.param.s64 %midx, [mag_offset];
    ld.param.s64 %sidx, [sign_offset];
    ld.param.u32 %nr, [n];
    ld.param.u32 %ndimr, [ndim];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr;
    @%p bra DONE;

    mov.u32 %rem, %idx;
    mov.u32 %d, %ndimr;
LOOP:
    setp.eq.u32 %done_loop, %d, 0;
    @%done_loop bra END_LOOP;
    sub.u32 %d, %d, 1;
    cvt.u64.u32 %d64, %d;
    shl.b64 %d64, %d64, 2;
    add.u64 %tmp, %oshape, %d64;
    ld.global.u32 %shape_d, [%tmp];
    cvt.u64.u32 %d64, %d;
    shl.b64 %d64, %d64, 3;
    add.u64 %tmp, %mstr, %d64;
    ld.global.s64 %mstride, [%tmp];
    add.u64 %tmp, %sstr, %d64;
    ld.global.s64 %sstride, [%tmp];
    rem.u32 %coord, %rem, %shape_d;
    div.u32 %rem, %rem, %shape_d;
    cvt.s64.u32 %coord64, %coord;
    mul.lo.s64 %prod, %coord64, %mstride;
    add.s64 %midx, %midx, %prod;
    mul.lo.s64 %prod, %coord64, %sstride;
    add.s64 %sidx, %sidx, %prod;
    bra LOOP;
END_LOOP:

    cvt.u64.s64 %off_m, %midx;
    shl.b64 %off_m, %off_m, 2;
    add.u64 %off_m, %mag, %off_m;
    ld.global.u32 %mbits, [%off_m];

    cvt.u64.s64 %off_s, %sidx;
    shl.b64 %off_s, %off_s, 2;
    add.u64 %off_s, %sign, %off_s;
    ld.global.u32 %sbits, [%off_s];

    and.b32 %mag_abs, %mbits, 0x7fffffff;
    and.b32 %sign_bit, %sbits, 0x80000000;
    or.b32 %out_bits, %mag_abs, %sign_bit;

    cvt.u64.u32 %off_o, %idx;
    shl.b64 %off_o, %off_o, 2;
    add.u64 %off_o, %out, %off_o;
    st.global.u32 [%off_o], %out_bits;
DONE:
    ret;
}
";

const COPYSIGN_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry copysign_f64_kernel(
    .param .u64 mag_ptr,
    .param .u64 sign_ptr,
    .param .u64 out_ptr,
    .param .u64 mag_strides_ptr,
    .param .u64 sign_strides_ptr,
    .param .u64 out_shape_ptr,
    .param .s64 mag_offset,
    .param .s64 sign_offset,
    .param .u32 n,
    .param .u32 ndim
) {
    .reg .u32 %idx, %bid, %bdim, %nr, %ndimr;
    .reg .u32 %rem, %d, %shape_d, %coord;
    .reg .u64 %mbits, %sbits, %mag_abs, %sign_bit, %out_bits, %abs_mask, %sgn_mask;
    .reg .u64 %mag, %sign, %out, %mstr, %sstr, %oshape;
    .reg .u64 %off_m, %off_s, %off_o, %d64, %tmp;
    .reg .s64 %midx, %sidx, %mstride, %sstride, %coord64, %prod;
    .reg .pred %p, %done_loop;

    ld.param.u64 %mag, [mag_ptr];
    ld.param.u64 %sign, [sign_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %mstr, [mag_strides_ptr];
    ld.param.u64 %sstr, [sign_strides_ptr];
    ld.param.u64 %oshape, [out_shape_ptr];
    ld.param.s64 %midx, [mag_offset];
    ld.param.s64 %sidx, [sign_offset];
    ld.param.u32 %nr, [n];
    ld.param.u32 %ndimr, [ndim];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr;
    @%p bra DONE;

    mov.u32 %rem, %idx;
    mov.u32 %d, %ndimr;
LOOP:
    setp.eq.u32 %done_loop, %d, 0;
    @%done_loop bra END_LOOP;
    sub.u32 %d, %d, 1;
    cvt.u64.u32 %d64, %d;
    shl.b64 %d64, %d64, 2;
    add.u64 %tmp, %oshape, %d64;
    ld.global.u32 %shape_d, [%tmp];
    cvt.u64.u32 %d64, %d;
    shl.b64 %d64, %d64, 3;
    add.u64 %tmp, %mstr, %d64;
    ld.global.s64 %mstride, [%tmp];
    add.u64 %tmp, %sstr, %d64;
    ld.global.s64 %sstride, [%tmp];
    rem.u32 %coord, %rem, %shape_d;
    div.u32 %rem, %rem, %shape_d;
    cvt.s64.u32 %coord64, %coord;
    mul.lo.s64 %prod, %coord64, %mstride;
    add.s64 %midx, %midx, %prod;
    mul.lo.s64 %prod, %coord64, %sstride;
    add.s64 %sidx, %sidx, %prod;
    bra LOOP;
END_LOOP:

    cvt.u64.s64 %off_m, %midx;
    shl.b64 %off_m, %off_m, 3;
    add.u64 %off_m, %mag, %off_m;
    ld.global.u64 %mbits, [%off_m];

    cvt.u64.s64 %off_s, %sidx;
    shl.b64 %off_s, %off_s, 3;
    add.u64 %off_s, %sign, %off_s;
    ld.global.u64 %sbits, [%off_s];

    mov.u64 %abs_mask, 0x7fffffffffffffff;
    mov.u64 %sgn_mask, 0x8000000000000000;
    and.b64 %mag_abs, %mbits, %abs_mask;
    and.b64 %sign_bit, %sbits, %sgn_mask;
    or.b64 %out_bits, %mag_abs, %sign_bit;

    cvt.u64.u32 %off_o, %idx;
    shl.b64 %off_o, %off_o, 3;
    add.u64 %off_o, %out, %off_o;
    st.global.u64 [%off_o], %out_bits;
DONE:
    ret;
}
";

const COPYSIGN_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry copysign_f16_kernel(
    .param .u64 mag_ptr,
    .param .u64 sign_ptr,
    .param .u64 out_ptr,
    .param .u64 mag_strides_ptr,
    .param .u64 sign_strides_ptr,
    .param .u64 out_shape_ptr,
    .param .s64 mag_offset,
    .param .s64 sign_offset,
    .param .u32 n,
    .param .u32 ndim
) {
    .reg .u32 %idx, %bid, %bdim, %nr, %ndimr;
    .reg .u32 %rem, %d, %shape_d, %coord;
    .reg .u32 %mbits, %sbits, %mexp, %mmant, %sexp, %smant, %mag_abs, %sign_bit, %out_bits;
    .reg .u64 %mag, %sign, %out, %mstr, %sstr, %oshape;
    .reg .u64 %off_m, %off_s, %off_o, %d64, %tmp;
    .reg .s64 %midx, %sidx, %mstride, %sstride, %coord64, %prod;
    .reg .u16 %mh, %sh, %oh;
    .reg .pred %p, %done_loop, %mexp_all, %mmant_nz, %m_nan, %sexp_all, %smant_nz, %s_nan;

    ld.param.u64 %mag, [mag_ptr];
    ld.param.u64 %sign, [sign_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %mstr, [mag_strides_ptr];
    ld.param.u64 %sstr, [sign_strides_ptr];
    ld.param.u64 %oshape, [out_shape_ptr];
    ld.param.s64 %midx, [mag_offset];
    ld.param.s64 %sidx, [sign_offset];
    ld.param.u32 %nr, [n];
    ld.param.u32 %ndimr, [ndim];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr;
    @%p bra DONE;

    mov.u32 %rem, %idx;
    mov.u32 %d, %ndimr;
LOOP:
    setp.eq.u32 %done_loop, %d, 0;
    @%done_loop bra END_LOOP;
    sub.u32 %d, %d, 1;
    cvt.u64.u32 %d64, %d;
    shl.b64 %d64, %d64, 2;
    add.u64 %tmp, %oshape, %d64;
    ld.global.u32 %shape_d, [%tmp];
    cvt.u64.u32 %d64, %d;
    shl.b64 %d64, %d64, 3;
    add.u64 %tmp, %mstr, %d64;
    ld.global.s64 %mstride, [%tmp];
    add.u64 %tmp, %sstr, %d64;
    ld.global.s64 %sstride, [%tmp];
    rem.u32 %coord, %rem, %shape_d;
    div.u32 %rem, %rem, %shape_d;
    cvt.s64.u32 %coord64, %coord;
    mul.lo.s64 %prod, %coord64, %mstride;
    add.s64 %midx, %midx, %prod;
    mul.lo.s64 %prod, %coord64, %sstride;
    add.s64 %sidx, %sidx, %prod;
    bra LOOP;
END_LOOP:

    cvt.u64.s64 %off_m, %midx;
    shl.b64 %off_m, %off_m, 1;
    add.u64 %off_m, %mag, %off_m;
    ld.global.u16 %mh, [%off_m];
    cvt.u32.u16 %mbits, %mh;

    cvt.u64.s64 %off_s, %sidx;
    shl.b64 %off_s, %off_s, 1;
    add.u64 %off_s, %sign, %off_s;
    ld.global.u16 %sh, [%off_s];
    cvt.u32.u16 %sbits, %sh;

    and.b32 %mexp, %mbits, 0x7c00;
    and.b32 %mmant, %mbits, 0x03ff;
    setp.eq.u32 %mexp_all, %mexp, 0x7c00;
    setp.ne.u32 %mmant_nz, %mmant, 0;
    and.pred %m_nan, %mexp_all, %mmant_nz;
    @%m_nan bra STORE_NAN;

    and.b32 %sexp, %sbits, 0x7c00;
    and.b32 %smant, %sbits, 0x03ff;
    setp.eq.u32 %sexp_all, %sexp, 0x7c00;
    setp.ne.u32 %smant_nz, %smant, 0;
    and.pred %s_nan, %sexp_all, %smant_nz;

    and.b32 %mag_abs, %mbits, 0x7fff;
    and.b32 %sign_bit, %sbits, 0x8000;
    @%s_nan mov.u32 %sign_bit, 0;
    or.b32 %out_bits, %mag_abs, %sign_bit;
    cvt.u16.u32 %oh, %out_bits;
    bra STORE;

STORE_NAN:
    mov.u16 %oh, 0x7fff;
STORE:
    cvt.u64.u32 %off_o, %idx;
    shl.b64 %off_o, %off_o, 1;
    add.u64 %off_o, %out, %off_o;
    st.global.u16 [%off_o], %oh;
DONE:
    ret;
}
";

const COPYSIGN_BF16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry copysign_bf16_kernel(
    .param .u64 mag_ptr,
    .param .u64 sign_ptr,
    .param .u64 out_ptr,
    .param .u64 mag_strides_ptr,
    .param .u64 sign_strides_ptr,
    .param .u64 out_shape_ptr,
    .param .s64 mag_offset,
    .param .s64 sign_offset,
    .param .u32 n,
    .param .u32 ndim
) {
    .reg .u32 %idx, %bid, %bdim, %nr, %ndimr;
    .reg .u32 %rem, %d, %shape_d, %coord;
    .reg .u32 %mbits, %sbits, %mexp, %mmant, %mag_abs, %sign_bit, %out_bits;
    .reg .u64 %mag, %sign, %out, %mstr, %sstr, %oshape;
    .reg .u64 %off_m, %off_s, %off_o, %d64, %tmp;
    .reg .s64 %midx, %sidx, %mstride, %sstride, %coord64, %prod;
    .reg .u16 %mh, %sh, %oh;
    .reg .pred %p, %done_loop, %mexp_all, %mmant_nz, %m_nan;

    ld.param.u64 %mag, [mag_ptr];
    ld.param.u64 %sign, [sign_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %mstr, [mag_strides_ptr];
    ld.param.u64 %sstr, [sign_strides_ptr];
    ld.param.u64 %oshape, [out_shape_ptr];
    ld.param.s64 %midx, [mag_offset];
    ld.param.s64 %sidx, [sign_offset];
    ld.param.u32 %nr, [n];
    ld.param.u32 %ndimr, [ndim];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr;
    @%p bra DONE;

    mov.u32 %rem, %idx;
    mov.u32 %d, %ndimr;
LOOP:
    setp.eq.u32 %done_loop, %d, 0;
    @%done_loop bra END_LOOP;
    sub.u32 %d, %d, 1;
    cvt.u64.u32 %d64, %d;
    shl.b64 %d64, %d64, 2;
    add.u64 %tmp, %oshape, %d64;
    ld.global.u32 %shape_d, [%tmp];
    cvt.u64.u32 %d64, %d;
    shl.b64 %d64, %d64, 3;
    add.u64 %tmp, %mstr, %d64;
    ld.global.s64 %mstride, [%tmp];
    add.u64 %tmp, %sstr, %d64;
    ld.global.s64 %sstride, [%tmp];
    rem.u32 %coord, %rem, %shape_d;
    div.u32 %rem, %rem, %shape_d;
    cvt.s64.u32 %coord64, %coord;
    mul.lo.s64 %prod, %coord64, %mstride;
    add.s64 %midx, %midx, %prod;
    mul.lo.s64 %prod, %coord64, %sstride;
    add.s64 %sidx, %sidx, %prod;
    bra LOOP;
END_LOOP:

    cvt.u64.s64 %off_m, %midx;
    shl.b64 %off_m, %off_m, 1;
    add.u64 %off_m, %mag, %off_m;
    ld.global.u16 %mh, [%off_m];
    cvt.u32.u16 %mbits, %mh;

    cvt.u64.s64 %off_s, %sidx;
    shl.b64 %off_s, %off_s, 1;
    add.u64 %off_s, %sign, %off_s;
    ld.global.u16 %sh, [%off_s];
    cvt.u32.u16 %sbits, %sh;

    and.b32 %mexp, %mbits, 0x7f80;
    and.b32 %mmant, %mbits, 0x007f;
    setp.eq.u32 %mexp_all, %mexp, 0x7f80;
    setp.ne.u32 %mmant_nz, %mmant, 0;
    and.pred %m_nan, %mexp_all, %mmant_nz;
    @%m_nan bra STORE_NAN;

    and.b32 %mag_abs, %mbits, 0x7fff;
    and.b32 %sign_bit, %sbits, 0x8000;
    or.b32 %out_bits, %mag_abs, %sign_bit;
    cvt.u16.u32 %oh, %out_bits;
    bra STORE;

STORE_NAN:
    mov.u16 %oh, 0x7fff;
STORE:
    cvt.u64.u32 %off_o, %idx;
    shl.b64 %off_o, %off_o, 1;
    add.u64 %off_o, %out, %off_o;
    st.global.u16 [%off_o], %oh;
DONE:
    ret;
}
";

const COPYSIGN_BACKWARD_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry copysign_backward_f32_kernel(
    .param .u64 grad_ptr,
    .param .u64 mag_ptr,
    .param .u64 result_ptr,
    .param .u64 out_ptr,
    .param .u64 grad_strides_ptr,
    .param .u64 mag_strides_ptr,
    .param .u64 result_strides_ptr,
    .param .u64 out_shape_ptr,
    .param .s64 grad_offset,
    .param .s64 mag_offset,
    .param .s64 result_offset,
    .param .u32 n,
    .param .u32 ndim
) {
    .reg .u32 %idx, %bid, %bdim, %nr, %ndimr;
    .reg .u32 %rem, %d, %shape_d, %coord;
    .reg .u64 %grad, %mag, %result, %out, %gstr, %mstr, %rstr, %oshape;
    .reg .u64 %off_g, %off_m, %off_r, %off_o, %d64, %tmp;
    .reg .s64 %gidx, %midx, %ridx, %gstride, %mstride, %rstride, %coord64, %prod;
    .reg .f32 %g, %m, %r, %ratio, %v;
    .reg .pred %p, %done_loop, %is_zero;

    ld.param.u64 %grad, [grad_ptr];
    ld.param.u64 %mag, [mag_ptr];
    ld.param.u64 %result, [result_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %gstr, [grad_strides_ptr];
    ld.param.u64 %mstr, [mag_strides_ptr];
    ld.param.u64 %rstr, [result_strides_ptr];
    ld.param.u64 %oshape, [out_shape_ptr];
    ld.param.s64 %gidx, [grad_offset];
    ld.param.s64 %midx, [mag_offset];
    ld.param.s64 %ridx, [result_offset];
    ld.param.u32 %nr, [n];
    ld.param.u32 %ndimr, [ndim];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr;
    @%p bra DONE;

    mov.u32 %rem, %idx;
    mov.u32 %d, %ndimr;
LOOP:
    setp.eq.u32 %done_loop, %d, 0;
    @%done_loop bra END_LOOP;
    sub.u32 %d, %d, 1;
    cvt.u64.u32 %d64, %d;
    shl.b64 %d64, %d64, 2;
    add.u64 %tmp, %oshape, %d64;
    ld.global.u32 %shape_d, [%tmp];
    cvt.u64.u32 %d64, %d;
    shl.b64 %d64, %d64, 3;
    add.u64 %tmp, %gstr, %d64;
    ld.global.s64 %gstride, [%tmp];
    add.u64 %tmp, %mstr, %d64;
    ld.global.s64 %mstride, [%tmp];
    add.u64 %tmp, %rstr, %d64;
    ld.global.s64 %rstride, [%tmp];
    rem.u32 %coord, %rem, %shape_d;
    div.u32 %rem, %rem, %shape_d;
    cvt.s64.u32 %coord64, %coord;
    mul.lo.s64 %prod, %coord64, %gstride;
    add.s64 %gidx, %gidx, %prod;
    mul.lo.s64 %prod, %coord64, %mstride;
    add.s64 %midx, %midx, %prod;
    mul.lo.s64 %prod, %coord64, %rstride;
    add.s64 %ridx, %ridx, %prod;
    bra LOOP;
END_LOOP:

    cvt.u64.s64 %off_g, %gidx;
    shl.b64 %off_g, %off_g, 2;
    add.u64 %off_g, %grad, %off_g;
    ld.global.f32 %g, [%off_g];
    cvt.u64.s64 %off_r, %ridx;
    shl.b64 %off_r, %off_r, 2;
    add.u64 %off_r, %result, %off_r;
    ld.global.f32 %r, [%off_r];
    cvt.u64.s64 %off_m, %midx;
    shl.b64 %off_m, %off_m, 2;
    add.u64 %off_m, %mag, %off_m;
    ld.global.f32 %m, [%off_m];

    setp.eq.f32 %is_zero, %m, 0f00000000;
    @%is_zero bra STORE_ZERO;
    div.rn.f32 %ratio, %r, %m;
    mul.rn.f32 %v, %g, %ratio;
    bra STORE;
STORE_ZERO:
    mov.f32 %v, 0f00000000;
STORE:
    cvt.u64.u32 %off_o, %idx;
    shl.b64 %off_o, %off_o, 2;
    add.u64 %off_o, %out, %off_o;
    st.global.f32 [%off_o], %v;
DONE:
    ret;
}
";

const COPYSIGN_BACKWARD_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry copysign_backward_f64_kernel(
    .param .u64 grad_ptr,
    .param .u64 mag_ptr,
    .param .u64 result_ptr,
    .param .u64 out_ptr,
    .param .u64 grad_strides_ptr,
    .param .u64 mag_strides_ptr,
    .param .u64 result_strides_ptr,
    .param .u64 out_shape_ptr,
    .param .s64 grad_offset,
    .param .s64 mag_offset,
    .param .s64 result_offset,
    .param .u32 n,
    .param .u32 ndim
) {
    .reg .u32 %idx, %bid, %bdim, %nr, %ndimr;
    .reg .u32 %rem, %d, %shape_d, %coord;
    .reg .u64 %grad, %mag, %result, %out, %gstr, %mstr, %rstr, %oshape;
    .reg .u64 %off_g, %off_m, %off_r, %off_o, %d64, %tmp;
    .reg .s64 %gidx, %midx, %ridx, %gstride, %mstride, %rstride, %coord64, %prod;
    .reg .f64 %g, %m, %r, %ratio, %v;
    .reg .pred %p, %done_loop, %is_zero;

    ld.param.u64 %grad, [grad_ptr];
    ld.param.u64 %mag, [mag_ptr];
    ld.param.u64 %result, [result_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %gstr, [grad_strides_ptr];
    ld.param.u64 %mstr, [mag_strides_ptr];
    ld.param.u64 %rstr, [result_strides_ptr];
    ld.param.u64 %oshape, [out_shape_ptr];
    ld.param.s64 %gidx, [grad_offset];
    ld.param.s64 %midx, [mag_offset];
    ld.param.s64 %ridx, [result_offset];
    ld.param.u32 %nr, [n];
    ld.param.u32 %ndimr, [ndim];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr;
    @%p bra DONE;

    mov.u32 %rem, %idx;
    mov.u32 %d, %ndimr;
LOOP:
    setp.eq.u32 %done_loop, %d, 0;
    @%done_loop bra END_LOOP;
    sub.u32 %d, %d, 1;
    cvt.u64.u32 %d64, %d;
    shl.b64 %d64, %d64, 2;
    add.u64 %tmp, %oshape, %d64;
    ld.global.u32 %shape_d, [%tmp];
    cvt.u64.u32 %d64, %d;
    shl.b64 %d64, %d64, 3;
    add.u64 %tmp, %gstr, %d64;
    ld.global.s64 %gstride, [%tmp];
    add.u64 %tmp, %mstr, %d64;
    ld.global.s64 %mstride, [%tmp];
    add.u64 %tmp, %rstr, %d64;
    ld.global.s64 %rstride, [%tmp];
    rem.u32 %coord, %rem, %shape_d;
    div.u32 %rem, %rem, %shape_d;
    cvt.s64.u32 %coord64, %coord;
    mul.lo.s64 %prod, %coord64, %gstride;
    add.s64 %gidx, %gidx, %prod;
    mul.lo.s64 %prod, %coord64, %mstride;
    add.s64 %midx, %midx, %prod;
    mul.lo.s64 %prod, %coord64, %rstride;
    add.s64 %ridx, %ridx, %prod;
    bra LOOP;
END_LOOP:

    cvt.u64.s64 %off_g, %gidx;
    shl.b64 %off_g, %off_g, 3;
    add.u64 %off_g, %grad, %off_g;
    ld.global.f64 %g, [%off_g];
    cvt.u64.s64 %off_r, %ridx;
    shl.b64 %off_r, %off_r, 3;
    add.u64 %off_r, %result, %off_r;
    ld.global.f64 %r, [%off_r];
    cvt.u64.s64 %off_m, %midx;
    shl.b64 %off_m, %off_m, 3;
    add.u64 %off_m, %mag, %off_m;
    ld.global.f64 %m, [%off_m];

    setp.eq.f64 %is_zero, %m, 0d0000000000000000;
    @%is_zero bra STORE_ZERO;
    div.rn.f64 %ratio, %r, %m;
    mul.rn.f64 %v, %g, %ratio;
    bra STORE;
STORE_ZERO:
    mov.f64 %v, 0d0000000000000000;
STORE:
    cvt.u64.u32 %off_o, %idx;
    shl.b64 %off_o, %off_o, 3;
    add.u64 %off_o, %out, %off_o;
    st.global.f64 [%off_o], %v;
DONE:
    ret;
}
";

const COPYSIGN_BACKWARD_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry copysign_backward_f16_kernel(
    .param .u64 grad_ptr,
    .param .u64 mag_ptr,
    .param .u64 result_ptr,
    .param .u64 out_ptr,
    .param .u64 grad_strides_ptr,
    .param .u64 mag_strides_ptr,
    .param .u64 result_strides_ptr,
    .param .u64 out_shape_ptr,
    .param .s64 grad_offset,
    .param .s64 mag_offset,
    .param .s64 result_offset,
    .param .u32 n,
    .param .u32 ndim
) {
    .reg .u32 %idx, %bid, %bdim, %nr, %ndimr;
    .reg .u32 %rem, %d, %shape_d, %coord;
    .reg .u64 %grad, %mag, %result, %out, %gstr, %mstr, %rstr, %oshape;
    .reg .u64 %off_g, %off_m, %off_r, %off_o, %d64, %tmp;
    .reg .s64 %gidx, %midx, %ridx, %gstride, %mstride, %rstride, %coord64, %prod;
    .reg .b16 %gh, %mh, %rh, %oh;
    .reg .f32 %g, %m, %r, %ratio, %v;
    .reg .pred %p, %done_loop, %is_zero, %is_nan;

    ld.param.u64 %grad, [grad_ptr];
    ld.param.u64 %mag, [mag_ptr];
    ld.param.u64 %result, [result_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %gstr, [grad_strides_ptr];
    ld.param.u64 %mstr, [mag_strides_ptr];
    ld.param.u64 %rstr, [result_strides_ptr];
    ld.param.u64 %oshape, [out_shape_ptr];
    ld.param.s64 %gidx, [grad_offset];
    ld.param.s64 %midx, [mag_offset];
    ld.param.s64 %ridx, [result_offset];
    ld.param.u32 %nr, [n];
    ld.param.u32 %ndimr, [ndim];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr;
    @%p bra DONE;

    mov.u32 %rem, %idx;
    mov.u32 %d, %ndimr;
LOOP:
    setp.eq.u32 %done_loop, %d, 0;
    @%done_loop bra END_LOOP;
    sub.u32 %d, %d, 1;
    cvt.u64.u32 %d64, %d;
    shl.b64 %d64, %d64, 2;
    add.u64 %tmp, %oshape, %d64;
    ld.global.u32 %shape_d, [%tmp];
    cvt.u64.u32 %d64, %d;
    shl.b64 %d64, %d64, 3;
    add.u64 %tmp, %gstr, %d64;
    ld.global.s64 %gstride, [%tmp];
    add.u64 %tmp, %mstr, %d64;
    ld.global.s64 %mstride, [%tmp];
    add.u64 %tmp, %rstr, %d64;
    ld.global.s64 %rstride, [%tmp];
    rem.u32 %coord, %rem, %shape_d;
    div.u32 %rem, %rem, %shape_d;
    cvt.s64.u32 %coord64, %coord;
    mul.lo.s64 %prod, %coord64, %gstride;
    add.s64 %gidx, %gidx, %prod;
    mul.lo.s64 %prod, %coord64, %mstride;
    add.s64 %midx, %midx, %prod;
    mul.lo.s64 %prod, %coord64, %rstride;
    add.s64 %ridx, %ridx, %prod;
    bra LOOP;
END_LOOP:

    cvt.u64.s64 %off_g, %gidx;
    shl.b64 %off_g, %off_g, 1;
    add.u64 %off_g, %grad, %off_g;
    ld.global.b16 %gh, [%off_g];
    cvt.u64.s64 %off_r, %ridx;
    shl.b64 %off_r, %off_r, 1;
    add.u64 %off_r, %result, %off_r;
    ld.global.b16 %rh, [%off_r];
    cvt.u64.s64 %off_m, %midx;
    shl.b64 %off_m, %off_m, 1;
    add.u64 %off_m, %mag, %off_m;
    ld.global.b16 %mh, [%off_m];

    cvt.f32.f16 %g, %gh;
    cvt.f32.f16 %r, %rh;
    cvt.f32.f16 %m, %mh;
    setp.eq.f32 %is_zero, %m, 0f00000000;
    @%is_zero bra STORE_ZERO;
    div.rn.f32 %ratio, %r, %m;
    mul.rn.f32 %v, %g, %ratio;
    setp.nan.f32 %is_nan, %v, %v;
    @%is_nan bra STORE_NAN;
    cvt.rn.f16.f32 %oh, %v;
    bra STORE;
STORE_ZERO:
    mov.b16 %oh, 0;
    bra STORE;
STORE_NAN:
    mov.b16 %oh, 0x7fff;
STORE:
    cvt.u64.u32 %off_o, %idx;
    shl.b64 %off_o, %off_o, 1;
    add.u64 %off_o, %out, %off_o;
    st.global.b16 [%off_o], %oh;
DONE:
    ret;
}
";

const COPYSIGN_BACKWARD_BF16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry copysign_backward_bf16_kernel(
    .param .u64 grad_ptr,
    .param .u64 mag_ptr,
    .param .u64 result_ptr,
    .param .u64 out_ptr,
    .param .u64 grad_strides_ptr,
    .param .u64 mag_strides_ptr,
    .param .u64 result_strides_ptr,
    .param .u64 out_shape_ptr,
    .param .s64 grad_offset,
    .param .s64 mag_offset,
    .param .s64 result_offset,
    .param .u32 n,
    .param .u32 ndim
) {
    .reg .u32 %idx, %bid, %bdim, %nr, %ndimr;
    .reg .u32 %rem, %d, %shape_d, %coord;
    .reg .u32 %gu, %ru, %mu, %bits, %round, %lsb;
    .reg .u64 %grad, %mag, %result, %out, %gstr, %mstr, %rstr, %oshape;
    .reg .u64 %off_g, %off_m, %off_r, %off_o, %d64, %tmp;
    .reg .s64 %gidx, %midx, %ridx, %gstride, %mstride, %rstride, %coord64, %prod;
    .reg .b16 %gh, %mh, %rh, %zero16;
    .reg .u16 %oh;
    .reg .f32 %g, %m, %r, %ratio, %v;
    .reg .pred %p, %done_loop, %is_zero, %is_nan;

    ld.param.u64 %grad, [grad_ptr];
    ld.param.u64 %mag, [mag_ptr];
    ld.param.u64 %result, [result_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %gstr, [grad_strides_ptr];
    ld.param.u64 %mstr, [mag_strides_ptr];
    ld.param.u64 %rstr, [result_strides_ptr];
    ld.param.u64 %oshape, [out_shape_ptr];
    ld.param.s64 %gidx, [grad_offset];
    ld.param.s64 %midx, [mag_offset];
    ld.param.s64 %ridx, [result_offset];
    ld.param.u32 %nr, [n];
    ld.param.u32 %ndimr, [ndim];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %idx, %tid.x;
    mad.lo.u32 %idx, %bid, %bdim, %idx;
    setp.ge.u32 %p, %idx, %nr;
    @%p bra DONE;

    mov.u32 %rem, %idx;
    mov.u32 %d, %ndimr;
LOOP:
    setp.eq.u32 %done_loop, %d, 0;
    @%done_loop bra END_LOOP;
    sub.u32 %d, %d, 1;
    cvt.u64.u32 %d64, %d;
    shl.b64 %d64, %d64, 2;
    add.u64 %tmp, %oshape, %d64;
    ld.global.u32 %shape_d, [%tmp];
    cvt.u64.u32 %d64, %d;
    shl.b64 %d64, %d64, 3;
    add.u64 %tmp, %gstr, %d64;
    ld.global.s64 %gstride, [%tmp];
    add.u64 %tmp, %mstr, %d64;
    ld.global.s64 %mstride, [%tmp];
    add.u64 %tmp, %rstr, %d64;
    ld.global.s64 %rstride, [%tmp];
    rem.u32 %coord, %rem, %shape_d;
    div.u32 %rem, %rem, %shape_d;
    cvt.s64.u32 %coord64, %coord;
    mul.lo.s64 %prod, %coord64, %gstride;
    add.s64 %gidx, %gidx, %prod;
    mul.lo.s64 %prod, %coord64, %mstride;
    add.s64 %midx, %midx, %prod;
    mul.lo.s64 %prod, %coord64, %rstride;
    add.s64 %ridx, %ridx, %prod;
    bra LOOP;
END_LOOP:

    cvt.u64.s64 %off_g, %gidx;
    shl.b64 %off_g, %off_g, 1;
    add.u64 %off_g, %grad, %off_g;
    ld.global.b16 %gh, [%off_g];
    cvt.u64.s64 %off_r, %ridx;
    shl.b64 %off_r, %off_r, 1;
    add.u64 %off_r, %result, %off_r;
    ld.global.b16 %rh, [%off_r];
    cvt.u64.s64 %off_m, %midx;
    shl.b64 %off_m, %off_m, 1;
    add.u64 %off_m, %mag, %off_m;
    ld.global.b16 %mh, [%off_m];

    mov.b16 %zero16, 0;
    mov.b32 %gu, {%zero16, %gh};
    mov.b32 %ru, {%zero16, %rh};
    mov.b32 %mu, {%zero16, %mh};
    mov.b32 %g, %gu;
    mov.b32 %r, %ru;
    mov.b32 %m, %mu;
    setp.eq.f32 %is_zero, %m, 0f00000000;
    @%is_zero bra STORE_ZERO;
    div.rn.f32 %ratio, %r, %m;
    mul.rn.f32 %v, %g, %ratio;
    setp.nan.f32 %is_nan, %v, %v;
    @%is_nan bra STORE_NAN;
    mov.b32 %bits, %v;
    shr.u32 %lsb, %bits, 16;
    and.b32 %lsb, %lsb, 1;
    add.u32 %round, %bits, 0x7fff;
    add.u32 %round, %round, %lsb;
    shr.u32 %bits, %round, 16;
    cvt.u16.u32 %oh, %bits;
    bra STORE;
STORE_ZERO:
    mov.u16 %oh, 0;
    bra STORE;
STORE_NAN:
    mov.u16 %oh, 0x7fff;
STORE:
    cvt.u64.u32 %off_o, %idx;
    shl.b64 %off_o, %off_o, 1;
    add.u64 %off_o, %out, %off_o;
    st.global.u16 [%off_o], %oh;
DONE:
    ret;
}
";

fn launch_cfg(n: usize, op: &'static str) -> GpuResult<LaunchConfig> {
    let n_u32 = u32::try_from(n).map_err(|_| GpuError::InvalidState {
        message: format!("{op} launch has {n} threads, exceeds u32::MAX"),
    })?;
    let grid = n_u32.saturating_add(BLOCK_SIZE - 1) / BLOCK_SIZE;
    Ok(LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn checked_numel(shape: &[usize], op: &'static str) -> GpuResult<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| GpuError::InvalidState {
            message: format!("{op}: shape {shape:?} element count overflows usize"),
        })
    })
}

fn checked_u32(value: usize, op: &'static str) -> GpuResult<u32> {
    u32::try_from(value).map_err(|_| GpuError::InvalidState {
        message: format!("{op}: dimension/index {value} exceeds u32::MAX"),
    })
}

fn checked_shape_u32(shape: &[usize], op: &'static str) -> GpuResult<Vec<u32>> {
    shape.iter().map(|&dim| checked_u32(dim, op)).collect()
}

fn checked_i64(value: usize, op: &'static str, what: &'static str) -> GpuResult<i64> {
    i64::try_from(value).map_err(|_| GpuError::InvalidState {
        message: format!("{op}: {what} {value} exceeds i64::MAX"),
    })
}

fn validate_view_bounds(
    storage_len: usize,
    shape: &[usize],
    strides: &[isize],
    offset: usize,
    op: &'static str,
) -> GpuResult<()> {
    if shape.len() != strides.len() {
        return Err(GpuError::ShapeMismatch {
            op,
            expected: vec![shape.len()],
            got: vec![strides.len()],
        });
    }
    let numel = checked_numel(shape, op)?;
    if numel == 0 {
        return Ok(());
    }

    let mut min_index = i128::try_from(offset).map_err(|_| GpuError::InvalidState {
        message: format!("{op}: storage offset {offset} exceeds i128::MAX"),
    })?;
    let mut max_index = min_index;
    for (&dim, &stride) in shape.iter().zip(strides) {
        if dim == 0 {
            return Ok(());
        }
        let span = i128::try_from(dim - 1)
            .map_err(|_| GpuError::InvalidState {
                message: format!("{op}: dimension {dim} exceeds i128::MAX"),
            })?
            .checked_mul(stride as i128)
            .ok_or_else(|| GpuError::InvalidState {
                message: format!("{op}: view span overflows for shape {shape:?}"),
            })?;
        if span >= 0 {
            max_index = max_index
                .checked_add(span)
                .ok_or_else(|| GpuError::InvalidState {
                    message: format!("{op}: view max offset overflows for shape {shape:?}"),
                })?;
        } else {
            min_index = min_index
                .checked_add(span)
                .ok_or_else(|| GpuError::InvalidState {
                    message: format!("{op}: view min offset overflows for shape {shape:?}"),
                })?;
        }
    }

    if min_index < 0 || max_index >= storage_len as i128 {
        return Err(GpuError::InvalidState {
            message: format!(
                "{op}: view shape {shape:?}, strides {strides:?}, offset {offset} address [{min_index}, {max_index}] outside storage length {storage_len}",
            ),
        });
    }
    checked_i64(offset, op, "storage offset")?;
    for &stride in strides {
        i64::try_from(stride).map_err(|_| GpuError::InvalidState {
            message: format!("{op}: stride {stride} is outside i64 range"),
        })?;
    }
    Ok(())
}

fn broadcast_source_strides(
    shape: &[usize],
    source_strides: &[isize],
    out_shape: &[usize],
    op: &'static str,
) -> GpuResult<Vec<i64>> {
    if shape.len() != source_strides.len() {
        return Err(GpuError::ShapeMismatch {
            op,
            expected: vec![shape.len()],
            got: vec![source_strides.len()],
        });
    }
    if shape.len() > out_shape.len() {
        return Err(GpuError::ShapeMismatch {
            op,
            expected: out_shape.to_vec(),
            got: shape.to_vec(),
        });
    }

    let mut strides = vec![0_i64; out_shape.len()];
    if shape.is_empty() {
        return Ok(strides);
    }

    let offset = out_shape.len() - shape.len();
    for (out_dim, stride_slot) in strides.iter_mut().enumerate() {
        if out_dim < offset {
            continue;
        }
        let in_dim = out_dim - offset;
        match (shape[in_dim], out_shape[out_dim]) {
            (1, _) => {}
            (actual, expected) if actual == expected => {
                *stride_slot =
                    i64::try_from(source_strides[in_dim]).map_err(|_| GpuError::InvalidState {
                        message: format!(
                            "{op}: source stride {} is outside i64 range",
                            source_strides[in_dim]
                        ),
                    })?;
            }
            _ => {
                return Err(GpuError::ShapeMismatch {
                    op,
                    expected: out_shape.to_vec(),
                    got: shape.to_vec(),
                });
            }
        }
    }
    Ok(strides)
}

fn validate_forward_lengths(
    magnitude_len: usize,
    sign_len: usize,
    magnitude_shape: &[usize],
    magnitude_strides: &[isize],
    magnitude_offset: usize,
    sign_shape: &[usize],
    sign_strides: &[isize],
    sign_offset: usize,
    op: &'static str,
) -> GpuResult<()> {
    validate_view_bounds(
        magnitude_len,
        magnitude_shape,
        magnitude_strides,
        magnitude_offset,
        op,
    )?;
    validate_view_bounds(sign_len, sign_shape, sign_strides, sign_offset, op)?;
    Ok(())
}

fn validate_backward_lengths(
    grad_len: usize,
    magnitude_len: usize,
    result_len: usize,
    grad_strides: &[isize],
    grad_offset: usize,
    magnitude_shape: &[usize],
    magnitude_strides: &[isize],
    magnitude_offset: usize,
    result_strides: &[isize],
    result_offset: usize,
    out_shape: &[usize],
    op: &'static str,
) -> GpuResult<()> {
    validate_view_bounds(grad_len, out_shape, grad_strides, grad_offset, op)?;
    validate_view_bounds(
        magnitude_len,
        magnitude_shape,
        magnitude_strides,
        magnitude_offset,
        op,
    )?;
    validate_view_bounds(result_len, out_shape, result_strides, result_offset, op)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_forward_f32(
    magnitude: &CudaBuffer<f32>,
    sign: &CudaBuffer<f32>,
    magnitude_shape: &[usize],
    magnitude_strides: &[isize],
    magnitude_offset: usize,
    sign_shape: &[usize],
    sign_strides: &[isize],
    sign_offset: usize,
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    let op = "copysign_f32";
    validate_forward_lengths(
        magnitude.len(),
        sign.len(),
        magnitude_shape,
        magnitude_strides,
        magnitude_offset,
        sign_shape,
        sign_strides,
        sign_offset,
        op,
    )?;
    let out_numel = checked_numel(out_shape, op)?;
    if out_numel == 0 {
        return alloc_zeros_f32(0, device);
    }

    let mag_strides = broadcast_source_strides(magnitude_shape, magnitude_strides, out_shape, op)?;
    let sign_strides = broadcast_source_strides(sign_shape, sign_strides, out_shape, op)?;
    let shape_u32 = checked_shape_u32(out_shape, op)?;
    let mag_strides_buf = cpu_to_gpu(&mag_strides, device)?;
    let sign_strides_buf = cpu_to_gpu(&sign_strides, device)?;
    let shape_buf = cpu_to_gpu(&shape_u32, device)?;
    let mag_offset_i64 = checked_i64(magnitude_offset, op, "magnitude offset")?;
    let sign_offset_i64 = checked_i64(sign_offset, op, "sign offset")?;
    let mut out = alloc_zeros_f32(out_numel, device)?;
    let f = get_or_compile(
        device.context(),
        COPYSIGN_F32_PTX,
        "copysign_f32_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "copysign_f32_kernel",
        source: e,
    })?;
    let cfg = launch_cfg(out_numel, op)?;
    let n_u32 = checked_u32(out_numel, op)?;
    let ndim_u32 = checked_u32(out_shape.len(), op)?;

    // SAFETY:
    // - The function is compiled from `COPYSIGN_F32_PTX` entry
    //   `copysign_f32_kernel`, whose ABI exactly matches these eight args.
    // - `magnitude`, `sign`, and `out` lengths were checked against the shape
    //   metadata; computed broadcast offsets stay inside their buffers.
    // - Metadata buffers are uploaded u32 arrays with length `ndim_u32` and
    //   are kept alive through launch by the local bindings.
    // - `out` is freshly allocated and uniquely borrowed for the launch.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(magnitude.inner())
            .arg(sign.inner())
            .arg(out.inner_mut())
            .arg(mag_strides_buf.inner())
            .arg(sign_strides_buf.inner())
            .arg(shape_buf.inner())
            .arg(&mag_offset_i64)
            .arg(&sign_offset_i64)
            .arg(&n_u32)
            .arg(&ndim_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn launch_forward_f64(
    magnitude: &CudaBuffer<f64>,
    sign: &CudaBuffer<f64>,
    magnitude_shape: &[usize],
    magnitude_strides: &[isize],
    magnitude_offset: usize,
    sign_shape: &[usize],
    sign_strides: &[isize],
    sign_offset: usize,
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    let op = "copysign_f64";
    validate_forward_lengths(
        magnitude.len(),
        sign.len(),
        magnitude_shape,
        magnitude_strides,
        magnitude_offset,
        sign_shape,
        sign_strides,
        sign_offset,
        op,
    )?;
    let out_numel = checked_numel(out_shape, op)?;
    if out_numel == 0 {
        return alloc_zeros_f64(0, device);
    }

    let mag_strides = broadcast_source_strides(magnitude_shape, magnitude_strides, out_shape, op)?;
    let sign_strides = broadcast_source_strides(sign_shape, sign_strides, out_shape, op)?;
    let shape_u32 = checked_shape_u32(out_shape, op)?;
    let mag_strides_buf = cpu_to_gpu(&mag_strides, device)?;
    let sign_strides_buf = cpu_to_gpu(&sign_strides, device)?;
    let shape_buf = cpu_to_gpu(&shape_u32, device)?;
    let mag_offset_i64 = checked_i64(magnitude_offset, op, "magnitude offset")?;
    let sign_offset_i64 = checked_i64(sign_offset, op, "sign offset")?;
    let mut out = alloc_zeros_f64(out_numel, device)?;
    let f = get_or_compile(
        device.context(),
        COPYSIGN_F64_PTX,
        "copysign_f64_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "copysign_f64_kernel",
        source: e,
    })?;
    let cfg = launch_cfg(out_numel, op)?;
    let n_u32 = checked_u32(out_numel, op)?;
    let ndim_u32 = checked_u32(out_shape.len(), op)?;

    // SAFETY: same argument and bounds contract as `launch_forward_f32`, with
    // 8-byte f64 element offsets in the PTX.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(magnitude.inner())
            .arg(sign.inner())
            .arg(out.inner_mut())
            .arg(mag_strides_buf.inner())
            .arg(sign_strides_buf.inner())
            .arg(shape_buf.inner())
            .arg(&mag_offset_i64)
            .arg(&sign_offset_i64)
            .arg(&n_u32)
            .arg(&ndim_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn launch_forward_u16(
    magnitude: &CudaSlice<u16>,
    sign: &CudaSlice<u16>,
    magnitude_shape: &[usize],
    magnitude_strides: &[isize],
    magnitude_offset: usize,
    sign_shape: &[usize],
    sign_strides: &[isize],
    sign_offset: usize,
    out_shape: &[usize],
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
    op: &'static str,
) -> GpuResult<CudaSlice<u16>> {
    validate_forward_lengths(
        magnitude.len(),
        sign.len(),
        magnitude_shape,
        magnitude_strides,
        magnitude_offset,
        sign_shape,
        sign_strides,
        sign_offset,
        op,
    )?;
    let out_numel = checked_numel(out_shape, op)?;
    let stream = device.stream();
    if out_numel == 0 {
        return Ok(stream.alloc_zeros::<u16>(0)?);
    }

    let mag_strides = broadcast_source_strides(magnitude_shape, magnitude_strides, out_shape, op)?;
    let sign_strides = broadcast_source_strides(sign_shape, sign_strides, out_shape, op)?;
    let shape_u32 = checked_shape_u32(out_shape, op)?;
    let mag_strides_buf = cpu_to_gpu(&mag_strides, device)?;
    let sign_strides_buf = cpu_to_gpu(&sign_strides, device)?;
    let shape_buf = cpu_to_gpu(&shape_u32, device)?;
    let mag_offset_i64 = checked_i64(magnitude_offset, op, "magnitude offset")?;
    let sign_offset_i64 = checked_i64(sign_offset, op, "sign offset")?;
    let mut out = stream.alloc_zeros::<u16>(out_numel)?;
    let f = get_or_compile(device.context(), ptx, kernel_name, device.ordinal() as u32).map_err(
        |e| GpuError::PtxCompileFailed {
            kernel: kernel_name,
            source: e,
        },
    )?;
    let cfg = launch_cfg(out_numel, op)?;
    let n_u32 = checked_u32(out_numel, op)?;
    let ndim_u32 = checked_u32(out_shape.len(), op)?;

    // SAFETY:
    // - `kernel_name` resolves to the u16 copysign forward ABI used below.
    // - Input lengths match their shape products, and broadcast strides are
    //   either valid contiguous strides or zero for broadcast dimensions.
    // - The output is a fresh `CudaSlice<u16>` with exactly `out_numel` slots.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(magnitude)
            .arg(sign)
            .arg(&mut out)
            .arg(mag_strides_buf.inner())
            .arg(sign_strides_buf.inner())
            .arg(shape_buf.inner())
            .arg(&mag_offset_i64)
            .arg(&sign_offset_i64)
            .arg(&n_u32)
            .arg(&ndim_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn launch_backward_f32(
    grad_output: &CudaBuffer<f32>,
    magnitude: &CudaBuffer<f32>,
    result: &CudaBuffer<f32>,
    grad_strides: &[isize],
    grad_offset: usize,
    magnitude_shape: &[usize],
    magnitude_strides: &[isize],
    magnitude_offset: usize,
    result_strides: &[isize],
    result_offset: usize,
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    let op = "copysign_backward_f32";
    validate_backward_lengths(
        grad_output.len(),
        magnitude.len(),
        result.len(),
        grad_strides,
        grad_offset,
        magnitude_shape,
        magnitude_strides,
        magnitude_offset,
        result_strides,
        result_offset,
        out_shape,
        op,
    )?;
    let out_numel = checked_numel(out_shape, op)?;
    if out_numel == 0 {
        return alloc_zeros_f32(0, device);
    }

    let grad_strides = broadcast_source_strides(out_shape, grad_strides, out_shape, op)?;
    let mag_strides = broadcast_source_strides(magnitude_shape, magnitude_strides, out_shape, op)?;
    let result_strides = broadcast_source_strides(out_shape, result_strides, out_shape, op)?;
    let shape_u32 = checked_shape_u32(out_shape, op)?;
    let grad_strides_buf = cpu_to_gpu(&grad_strides, device)?;
    let mag_strides_buf = cpu_to_gpu(&mag_strides, device)?;
    let result_strides_buf = cpu_to_gpu(&result_strides, device)?;
    let shape_buf = cpu_to_gpu(&shape_u32, device)?;
    let grad_offset_i64 = checked_i64(grad_offset, op, "grad offset")?;
    let mag_offset_i64 = checked_i64(magnitude_offset, op, "magnitude offset")?;
    let result_offset_i64 = checked_i64(result_offset, op, "result offset")?;
    let mut out = alloc_zeros_f32(out_numel, device)?;
    let f = get_or_compile(
        device.context(),
        COPYSIGN_BACKWARD_F32_PTX,
        "copysign_backward_f32_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "copysign_backward_f32_kernel",
        source: e,
    })?;
    let cfg = launch_cfg(out_numel, op)?;
    let n_u32 = checked_u32(out_numel, op)?;
    let ndim_u32 = checked_u32(out_shape.len(), op)?;

    // SAFETY:
    // - PTX ABI is `(grad, magnitude, result, out, mag_strides, out_shape, n,
    //   ndim)`, matching the pushed arguments.
    // - `grad_output` and `result` have `out_numel` elements; `magnitude` has
    //   `product(magnitude_shape)` elements and is addressed through checked
    //   broadcast strides.
    // - `out` is fresh and uniquely borrowed.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(grad_output.inner())
            .arg(magnitude.inner())
            .arg(result.inner())
            .arg(out.inner_mut())
            .arg(grad_strides_buf.inner())
            .arg(mag_strides_buf.inner())
            .arg(result_strides_buf.inner())
            .arg(shape_buf.inner())
            .arg(&grad_offset_i64)
            .arg(&mag_offset_i64)
            .arg(&result_offset_i64)
            .arg(&n_u32)
            .arg(&ndim_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn launch_backward_f64(
    grad_output: &CudaBuffer<f64>,
    magnitude: &CudaBuffer<f64>,
    result: &CudaBuffer<f64>,
    grad_strides: &[isize],
    grad_offset: usize,
    magnitude_shape: &[usize],
    magnitude_strides: &[isize],
    magnitude_offset: usize,
    result_strides: &[isize],
    result_offset: usize,
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    let op = "copysign_backward_f64";
    validate_backward_lengths(
        grad_output.len(),
        magnitude.len(),
        result.len(),
        grad_strides,
        grad_offset,
        magnitude_shape,
        magnitude_strides,
        magnitude_offset,
        result_strides,
        result_offset,
        out_shape,
        op,
    )?;
    let out_numel = checked_numel(out_shape, op)?;
    if out_numel == 0 {
        return alloc_zeros_f64(0, device);
    }

    let grad_strides = broadcast_source_strides(out_shape, grad_strides, out_shape, op)?;
    let mag_strides = broadcast_source_strides(magnitude_shape, magnitude_strides, out_shape, op)?;
    let result_strides = broadcast_source_strides(out_shape, result_strides, out_shape, op)?;
    let shape_u32 = checked_shape_u32(out_shape, op)?;
    let grad_strides_buf = cpu_to_gpu(&grad_strides, device)?;
    let mag_strides_buf = cpu_to_gpu(&mag_strides, device)?;
    let result_strides_buf = cpu_to_gpu(&result_strides, device)?;
    let shape_buf = cpu_to_gpu(&shape_u32, device)?;
    let grad_offset_i64 = checked_i64(grad_offset, op, "grad offset")?;
    let mag_offset_i64 = checked_i64(magnitude_offset, op, "magnitude offset")?;
    let result_offset_i64 = checked_i64(result_offset, op, "result offset")?;
    let mut out = alloc_zeros_f64(out_numel, device)?;
    let f = get_or_compile(
        device.context(),
        COPYSIGN_BACKWARD_F64_PTX,
        "copysign_backward_f64_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "copysign_backward_f64_kernel",
        source: e,
    })?;
    let cfg = launch_cfg(out_numel, op)?;
    let n_u32 = checked_u32(out_numel, op)?;
    let ndim_u32 = checked_u32(out_shape.len(), op)?;

    // SAFETY: same argument and bounds contract as `launch_backward_f32`, with
    // 8-byte f64 element offsets in the PTX.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(grad_output.inner())
            .arg(magnitude.inner())
            .arg(result.inner())
            .arg(out.inner_mut())
            .arg(grad_strides_buf.inner())
            .arg(mag_strides_buf.inner())
            .arg(result_strides_buf.inner())
            .arg(shape_buf.inner())
            .arg(&grad_offset_i64)
            .arg(&mag_offset_i64)
            .arg(&result_offset_i64)
            .arg(&n_u32)
            .arg(&ndim_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn launch_backward_u16(
    grad_output: &CudaSlice<u16>,
    magnitude: &CudaSlice<u16>,
    result: &CudaSlice<u16>,
    grad_strides: &[isize],
    grad_offset: usize,
    magnitude_shape: &[usize],
    magnitude_strides: &[isize],
    magnitude_offset: usize,
    result_strides: &[isize],
    result_offset: usize,
    out_shape: &[usize],
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
    op: &'static str,
) -> GpuResult<CudaSlice<u16>> {
    validate_backward_lengths(
        grad_output.len(),
        magnitude.len(),
        result.len(),
        grad_strides,
        grad_offset,
        magnitude_shape,
        magnitude_strides,
        magnitude_offset,
        result_strides,
        result_offset,
        out_shape,
        op,
    )?;
    let out_numel = checked_numel(out_shape, op)?;
    let stream = device.stream();
    if out_numel == 0 {
        return Ok(stream.alloc_zeros::<u16>(0)?);
    }

    let grad_strides = broadcast_source_strides(out_shape, grad_strides, out_shape, op)?;
    let mag_strides = broadcast_source_strides(magnitude_shape, magnitude_strides, out_shape, op)?;
    let result_strides = broadcast_source_strides(out_shape, result_strides, out_shape, op)?;
    let shape_u32 = checked_shape_u32(out_shape, op)?;
    let grad_strides_buf = cpu_to_gpu(&grad_strides, device)?;
    let mag_strides_buf = cpu_to_gpu(&mag_strides, device)?;
    let result_strides_buf = cpu_to_gpu(&result_strides, device)?;
    let shape_buf = cpu_to_gpu(&shape_u32, device)?;
    let grad_offset_i64 = checked_i64(grad_offset, op, "grad offset")?;
    let mag_offset_i64 = checked_i64(magnitude_offset, op, "magnitude offset")?;
    let result_offset_i64 = checked_i64(result_offset, op, "result offset")?;
    let mut out = stream.alloc_zeros::<u16>(out_numel)?;
    let f = get_or_compile(device.context(), ptx, kernel_name, device.ordinal() as u32).map_err(
        |e| GpuError::PtxCompileFailed {
            kernel: kernel_name,
            source: e,
        },
    )?;
    let cfg = launch_cfg(out_numel, op)?;
    let n_u32 = checked_u32(out_numel, op)?;
    let ndim_u32 = checked_u32(out_shape.len(), op)?;

    // SAFETY: same resident broadcast VJP contract as the f32/f64 launchers,
    // with u16 storage for reduced precision values.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(grad_output)
            .arg(magnitude)
            .arg(result)
            .arg(&mut out)
            .arg(grad_strides_buf.inner())
            .arg(mag_strides_buf.inner())
            .arg(result_strides_buf.inner())
            .arg(shape_buf.inner())
            .arg(&grad_offset_i64)
            .arg(&mag_offset_i64)
            .arg(&result_offset_i64)
            .arg(&n_u32)
            .arg(&ndim_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Broadcast `copysign` for f32 CUDA buffers.
///
/// The output has the magnitude of `magnitude` and the raw sign bit of `sign`.
pub fn gpu_copysign_f32(
    magnitude: &CudaBuffer<f32>,
    sign: &CudaBuffer<f32>,
    magnitude_shape: &[usize],
    magnitude_strides: &[isize],
    magnitude_offset: usize,
    sign_shape: &[usize],
    sign_strides: &[isize],
    sign_offset: usize,
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    if magnitude.device_ordinal() != sign.device_ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: magnitude.device_ordinal(),
            got: sign.device_ordinal(),
        });
    }
    launch_forward_f32(
        magnitude,
        sign,
        magnitude_shape,
        magnitude_strides,
        magnitude_offset,
        sign_shape,
        sign_strides,
        sign_offset,
        out_shape,
        device,
    )
}

/// Broadcast `copysign` for f64 CUDA buffers.
///
/// The output has the magnitude of `magnitude` and the raw sign bit of `sign`.
pub fn gpu_copysign_f64(
    magnitude: &CudaBuffer<f64>,
    sign: &CudaBuffer<f64>,
    magnitude_shape: &[usize],
    magnitude_strides: &[isize],
    magnitude_offset: usize,
    sign_shape: &[usize],
    sign_strides: &[isize],
    sign_offset: usize,
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    if magnitude.device_ordinal() != sign.device_ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: magnitude.device_ordinal(),
            got: sign.device_ordinal(),
        });
    }
    launch_forward_f64(
        magnitude,
        sign,
        magnitude_shape,
        magnitude_strides,
        magnitude_offset,
        sign_shape,
        sign_strides,
        sign_offset,
        out_shape,
        device,
    )
}

/// Broadcast `copysign` for f16 CUDA buffers.
///
/// Matches PyTorch CUDA: NaN sign operands do not make finite magnitudes
/// negative, while NaN magnitudes canonicalize to positive `0x7fff`.
pub fn gpu_copysign_f16(
    magnitude: &CudaSlice<u16>,
    sign: &CudaSlice<u16>,
    magnitude_shape: &[usize],
    magnitude_strides: &[isize],
    magnitude_offset: usize,
    sign_shape: &[usize],
    sign_strides: &[isize],
    sign_offset: usize,
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch_forward_u16(
        magnitude,
        sign,
        magnitude_shape,
        magnitude_strides,
        magnitude_offset,
        sign_shape,
        sign_strides,
        sign_offset,
        out_shape,
        device,
        COPYSIGN_F16_PTX,
        "copysign_f16_kernel",
        "copysign_f16",
    )
}

/// Broadcast `copysign` for bf16 CUDA buffers.
///
/// Matches PyTorch CUDA: finite magnitudes receive the raw sign bit from
/// `sign`, while NaN magnitudes canonicalize to positive `0x7fff`.
pub fn gpu_copysign_bf16(
    magnitude: &CudaSlice<u16>,
    sign: &CudaSlice<u16>,
    magnitude_shape: &[usize],
    magnitude_strides: &[isize],
    magnitude_offset: usize,
    sign_shape: &[usize],
    sign_strides: &[isize],
    sign_offset: usize,
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch_forward_u16(
        magnitude,
        sign,
        magnitude_shape,
        magnitude_strides,
        magnitude_offset,
        sign_shape,
        sign_strides,
        sign_offset,
        out_shape,
        device,
        COPYSIGN_BF16_PTX,
        "copysign_bf16_kernel",
        "copysign_bf16",
    )
}

/// Resident CUDA VJP for f32 `copysign`.
pub fn gpu_copysign_backward_f32(
    grad_output: &CudaBuffer<f32>,
    magnitude: &CudaBuffer<f32>,
    result: &CudaBuffer<f32>,
    grad_strides: &[isize],
    grad_offset: usize,
    magnitude_shape: &[usize],
    magnitude_strides: &[isize],
    magnitude_offset: usize,
    result_strides: &[isize],
    result_offset: usize,
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    launch_backward_f32(
        grad_output,
        magnitude,
        result,
        grad_strides,
        grad_offset,
        magnitude_shape,
        magnitude_strides,
        magnitude_offset,
        result_strides,
        result_offset,
        out_shape,
        device,
    )
}

/// Resident CUDA VJP for f64 `copysign`.
pub fn gpu_copysign_backward_f64(
    grad_output: &CudaBuffer<f64>,
    magnitude: &CudaBuffer<f64>,
    result: &CudaBuffer<f64>,
    grad_strides: &[isize],
    grad_offset: usize,
    magnitude_shape: &[usize],
    magnitude_strides: &[isize],
    magnitude_offset: usize,
    result_strides: &[isize],
    result_offset: usize,
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    launch_backward_f64(
        grad_output,
        magnitude,
        result,
        grad_strides,
        grad_offset,
        magnitude_shape,
        magnitude_strides,
        magnitude_offset,
        result_strides,
        result_offset,
        out_shape,
        device,
    )
}

/// Resident CUDA VJP for f16 `copysign`.
pub fn gpu_copysign_backward_f16(
    grad_output: &CudaSlice<u16>,
    magnitude: &CudaSlice<u16>,
    result: &CudaSlice<u16>,
    grad_strides: &[isize],
    grad_offset: usize,
    magnitude_shape: &[usize],
    magnitude_strides: &[isize],
    magnitude_offset: usize,
    result_strides: &[isize],
    result_offset: usize,
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch_backward_u16(
        grad_output,
        magnitude,
        result,
        grad_strides,
        grad_offset,
        magnitude_shape,
        magnitude_strides,
        magnitude_offset,
        result_strides,
        result_offset,
        out_shape,
        device,
        COPYSIGN_BACKWARD_F16_PTX,
        "copysign_backward_f16_kernel",
        "copysign_backward_f16",
    )
}

/// Resident CUDA VJP for bf16 `copysign`.
pub fn gpu_copysign_backward_bf16(
    grad_output: &CudaSlice<u16>,
    magnitude: &CudaSlice<u16>,
    result: &CudaSlice<u16>,
    grad_strides: &[isize],
    grad_offset: usize,
    magnitude_shape: &[usize],
    magnitude_strides: &[isize],
    magnitude_offset: usize,
    result_strides: &[isize],
    result_offset: usize,
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch_backward_u16(
        grad_output,
        magnitude,
        result,
        grad_strides,
        grad_offset,
        magnitude_shape,
        magnitude_strides,
        magnitude_offset,
        result_strides,
        result_offset,
        out_shape,
        device,
        COPYSIGN_BACKWARD_BF16_PTX,
        "copysign_backward_bf16_kernel",
        "copysign_backward_bf16",
    )
}
