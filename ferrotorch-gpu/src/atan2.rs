//! CUDA kernels for `torch.atan2` parity.
//!
//! PyTorch routes CUDA `atan2` through TensorIterator and the CUDA `atan2`
//! device math for floating dtypes, including f16 and bf16. This module keeps
//! the ferrotorch implementation resident on CUDA: broadcasted views are
//! addressed through shape/stride/offset metadata, forward handles IEEE
//! quadrant/signed-zero/infinity/NaN cases explicitly, and backward computes
//! the joint VJP on device before reducing broadcasted gradients in core.

#![cfg(feature = "cuda")]
#![allow(clippy::too_many_arguments)]

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::buffer::CudaBuffer;
use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};
use crate::module_cache::get_or_compile_owned;
use crate::transfer::{alloc_zeros_f32, alloc_zeros_f64, cpu_to_gpu};

const BLOCK_SIZE: u32 = 256;

const ATAN2_F32_BODY: &str = r"
    setp.nan.f32 %y_nan, %yv, %yv;
    @%y_nan bra STORE_NAN;
    setp.nan.f32 %x_nan, %xv, %xv;
    @%x_nan bra STORE_NAN;

    mov.b32 %ybits, %yv;
    mov.b32 %xbits, %xv;
    and.b32 %sign_y, %ybits, 0x80000000;
    and.b32 %sign_x, %xbits, 0x80000000;
    setp.ne.u32 %y_neg, %sign_y, 0;
    setp.ne.u32 %x_neg, %sign_x, 0;
    abs.f32 %ay, %yv;
    abs.f32 %ax, %xv;
    setp.eq.f32 %y_zero, %ay, 0f00000000;
    @%y_zero bra HANDLE_Y_ZERO;
    setp.eq.f32 %x_zero, %ax, 0f00000000;
    @%x_zero bra HANDLE_X_ZERO;
    setp.eq.f32 %y_inf, %ay, 0f7F800000;
    @%y_inf bra HANDLE_Y_INF;
    setp.eq.f32 %x_inf, %ax, 0f7F800000;
    @%x_inf bra HANDLE_X_INF;
    bra HANDLE_FINITE;

HANDLE_Y_ZERO:
    mov.f32 %angle, 0f00000000;
    @%x_neg mov.f32 %angle, 0f40490FDB;
    @%y_neg neg.f32 %angle, %angle;
    bra STORE_VALUE;

HANDLE_X_ZERO:
    mov.f32 %angle, 0f3FC90FDB;
    @%y_neg neg.f32 %angle, %angle;
    bra STORE_VALUE;

HANDLE_Y_INF:
    setp.eq.f32 %x_inf, %ax, 0f7F800000;
    @%x_inf bra HANDLE_BOTH_INF;
    mov.f32 %angle, 0f3FC90FDB;
    @%y_neg neg.f32 %angle, %angle;
    bra STORE_VALUE;

HANDLE_X_INF:
    mov.f32 %angle, 0f00000000;
    @%x_neg mov.f32 %angle, 0f40490FDB;
    @%y_neg neg.f32 %angle, %angle;
    bra STORE_VALUE;

HANDLE_BOTH_INF:
    mov.f32 %angle, 0f3F490FDB;
    @%x_neg mov.f32 %angle, 0f4016CBE4;
    @%y_neg neg.f32 %angle, %angle;
    bra STORE_VALUE;

HANDLE_FINITE:
    div.rn.f32 %r, %ay, %ax;
    mov.f32 %one, 0f3F800000;
    setp.gt.f32 %use_inv, %r, %one;
    div.rn.f32 %z, %one, %r;
    @!%use_inv mov.f32 %z, %r;

    setp.gt.f32 %use_shift, %z, 0f3ED413CD;
    @!%use_shift bra ATAN_POLY;
    sub.f32 %tmpf, %z, %one;
    add.f32 %tmpf2, %z, %one;
    div.rn.f32 %z, %tmpf, %tmpf2;

ATAN_POLY:
    mul.f32 %z2, %z, %z;
    mov.f32 %poly, 0fBD888889;
    fma.rn.f32 %poly, %poly, %z2, 0f3D9D89D9;
    fma.rn.f32 %poly, %poly, %z2, 0fBDBA2E8C;
    fma.rn.f32 %poly, %poly, %z2, 0f3DE38E39;
    fma.rn.f32 %poly, %poly, %z2, 0fBE124925;
    fma.rn.f32 %poly, %poly, %z2, 0f3E4CCCCD;
    fma.rn.f32 %poly, %poly, %z2, 0fBEAAAAAB;
    fma.rn.f32 %poly, %poly, %z2, %one;
    mul.f32 %angle, %poly, %z;
    @%use_shift add.f32 %angle, %angle, 0f3F490FDB;
    @%use_inv sub.f32 %angle, 0f3FC90FDB, %angle;
    @%x_neg sub.f32 %angle, 0f40490FDB, %angle;
    @%y_neg neg.f32 %angle, %angle;
";

const ATAN2_F64_BODY: &str = r"
    mov.b64 %ybits64, %yv;
    mov.b64 %xbits64, %xv;
    setp.nan.f64 %y_nan, %yv, %yv;
    @%y_nan bra STORE_Y_NAN;
    setp.nan.f64 %x_nan, %xv, %xv;
    @%x_nan bra STORE_X_NAN;

    and.b64 %sign_y64, %ybits64, 0x8000000000000000;
    and.b64 %sign_x64, %xbits64, 0x8000000000000000;
    setp.ne.u64 %y_neg, %sign_y64, 0;
    setp.ne.u64 %x_neg, %sign_x64, 0;
    abs.f64 %ay, %yv;
    abs.f64 %ax, %xv;
    setp.eq.f64 %y_zero, %ay, 0d0000000000000000;
    @%y_zero bra HANDLE_Y_ZERO;
    setp.eq.f64 %x_zero, %ax, 0d0000000000000000;
    @%x_zero bra HANDLE_X_ZERO;
    setp.eq.f64 %y_inf, %ay, 0d7FF0000000000000;
    @%y_inf bra HANDLE_Y_INF;
    setp.eq.f64 %x_inf, %ax, 0d7FF0000000000000;
    @%x_inf bra HANDLE_X_INF;
    bra HANDLE_FINITE;

HANDLE_Y_ZERO:
    mov.f64 %angle, 0d0000000000000000;
    @%x_neg mov.f64 %angle, 0d400921FB54442D18;
    @%y_neg neg.f64 %angle, %angle;
    bra STORE_VALUE;

HANDLE_X_ZERO:
    mov.f64 %angle, 0d3FF921FB54442D18;
    @%y_neg neg.f64 %angle, %angle;
    bra STORE_VALUE;

HANDLE_Y_INF:
    setp.eq.f64 %x_inf, %ax, 0d7FF0000000000000;
    @%x_inf bra HANDLE_BOTH_INF;
    mov.f64 %angle, 0d3FF921FB54442D18;
    @%y_neg neg.f64 %angle, %angle;
    bra STORE_VALUE;

HANDLE_X_INF:
    mov.f64 %angle, 0d0000000000000000;
    @%x_neg mov.f64 %angle, 0d400921FB54442D18;
    @%y_neg neg.f64 %angle, %angle;
    bra STORE_VALUE;

HANDLE_BOTH_INF:
    mov.f64 %angle, 0d3FE921FB54442D18;
    @%x_neg mov.f64 %angle, 0d4002D97C7F3321D2;
    @%y_neg neg.f64 %angle, %angle;
    bra STORE_VALUE;

HANDLE_FINITE:
    div.rn.f64 %r, %ay, %ax;
    mov.f64 %one, 0d3FF0000000000000;
    setp.gt.f64 %use_inv, %r, %one;
    div.rn.f64 %z, %one, %r;
    @!%use_inv mov.f64 %z, %r;

    setp.gt.f64 %use_shift, %z, 0d3FDA827999FCEF34;
    @!%use_shift bra ATAN_POLY;
    sub.f64 %tmpf, %z, %one;
    add.f64 %tmpf2, %z, %one;
    div.rn.f64 %z, %tmpf, %tmpf2;

ATAN_POLY:
    mul.f64 %z2, %z, %z;
    mov.f64 %poly, 0dBFA0842108421084;
    fma.rn.f64 %poly, %poly, %z2, 0d3FA1A7B9611A7B96;
    fma.rn.f64 %poly, %poly, %z2, 0dBFA2F684BDA12F68;
    fma.rn.f64 %poly, %poly, %z2, 0d3FA47AE147AE147B;
    fma.rn.f64 %poly, %poly, %z2, 0dBFA642C8590B2164;
    fma.rn.f64 %poly, %poly, %z2, 0d3FA8618618618618;
    fma.rn.f64 %poly, %poly, %z2, 0dBFAAF286BCA1AF28;
    fma.rn.f64 %poly, %poly, %z2, 0d3FAE1E1E1E1E1E1E;
    fma.rn.f64 %poly, %poly, %z2, 0dBFB1111111111111;
    fma.rn.f64 %poly, %poly, %z2, 0d3FB3B13B13B13B14;
    fma.rn.f64 %poly, %poly, %z2, 0dBFB745D1745D1746;
    fma.rn.f64 %poly, %poly, %z2, 0d3FBC71C71C71C71C;
    fma.rn.f64 %poly, %poly, %z2, 0dBFC2492492492492;
    fma.rn.f64 %poly, %poly, %z2, 0d3FC999999999999A;
    fma.rn.f64 %poly, %poly, %z2, 0dBFD5555555555555;
    fma.rn.f64 %poly, %poly, %z2, %one;
    mul.f64 %angle, %poly, %z;
    @%use_shift add.f64 %angle, %angle, 0d3FE921FB54442D18;
    @%use_inv sub.f64 %angle, 0d3FF921FB54442D18, %angle;
    @%x_neg sub.f64 %angle, 0d400921FB54442D18, %angle;
    @%y_neg neg.f64 %angle, %angle;
";

const F32_FORWARD_TEMPLATE: &str = r".version 7.0
.target sm_52
.address_size 64

.visible .entry @KERNEL@(
    .param .u64 y_ptr,
    .param .u64 x_ptr,
    .param .u64 out_ptr,
    .param .u64 y_strides_ptr,
    .param .u64 x_strides_ptr,
    .param .u64 out_shape_ptr,
    .param .s64 y_offset,
    .param .s64 x_offset,
    .param .u32 n,
    .param .u32 ndim
) {
    .reg .u32 %idx, %bid, %bdim, %nr, %ndimr;
    .reg .u32 %rem, %d, %shape_d, %coord;
    .reg .u32 %ybits, %xbits, %sign_y, %sign_x, %out_bits;
    .reg .u64 %y, %x, %out, %ystr, %xstr, %oshape;
    .reg .u64 %y_addr, %x_addr, %out_addr, %d64, %tmp;
    .reg .s64 %yidx, %xidx, %ystride, %xstride, %coord64, %prod;
    .reg .f32 %yv, %xv, %ay, %ax, %r, %z, %z2, %poly, %one, %tmpf, %tmpf2, %angle;
    .reg .pred %p, %done_loop, %y_nan, %x_nan, %y_neg, %x_neg, %y_zero, %x_zero, %y_inf, %x_inf, %use_inv, %use_shift;

    ld.param.u64 %y, [y_ptr];
    ld.param.u64 %x, [x_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %ystr, [y_strides_ptr];
    ld.param.u64 %xstr, [x_strides_ptr];
    ld.param.u64 %oshape, [out_shape_ptr];
    ld.param.s64 %yidx, [y_offset];
    ld.param.s64 %xidx, [x_offset];
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
    add.u64 %tmp, %ystr, %d64;
    ld.global.s64 %ystride, [%tmp];
    add.u64 %tmp, %xstr, %d64;
    ld.global.s64 %xstride, [%tmp];
    rem.u32 %coord, %rem, %shape_d;
    div.u32 %rem, %rem, %shape_d;
    cvt.s64.u32 %coord64, %coord;
    mul.lo.s64 %prod, %coord64, %ystride;
    add.s64 %yidx, %yidx, %prod;
    mul.lo.s64 %prod, %coord64, %xstride;
    add.s64 %xidx, %xidx, %prod;
    bra LOOP;
END_LOOP:

    cvt.u64.s64 %y_addr, %yidx;
    shl.b64 %y_addr, %y_addr, 2;
    add.u64 %y_addr, %y, %y_addr;
    ld.global.f32 %yv, [%y_addr];
    cvt.u64.s64 %x_addr, %xidx;
    shl.b64 %x_addr, %x_addr, 2;
    add.u64 %x_addr, %x, %x_addr;
    ld.global.f32 %xv, [%x_addr];
    cvt.u64.u32 %out_addr, %idx;
    shl.b64 %out_addr, %out_addr, 2;
    add.u64 %out_addr, %out, %out_addr;
@BODY@
STORE_VALUE:
    st.global.f32 [%out_addr], %angle;
    bra DONE;
STORE_NAN:
    mov.u32 %out_bits, 0x7fffffff;
    st.global.u32 [%out_addr], %out_bits;
DONE:
    ret;
}
";

const F64_FORWARD_TEMPLATE: &str = r".version 7.0
.target sm_52
.address_size 64

.visible .entry @KERNEL@(
    .param .u64 y_ptr,
    .param .u64 x_ptr,
    .param .u64 out_ptr,
    .param .u64 y_strides_ptr,
    .param .u64 x_strides_ptr,
    .param .u64 out_shape_ptr,
    .param .s64 y_offset,
    .param .s64 x_offset,
    .param .u32 n,
    .param .u32 ndim
) {
    .reg .u32 %idx, %bid, %bdim, %nr, %ndimr;
    .reg .u32 %rem, %d, %shape_d, %coord;
    .reg .u64 %ybits64, %xbits64, %sign_y64, %sign_x64, %out_bits64;
    .reg .u64 %y, %x, %out, %ystr, %xstr, %oshape;
    .reg .u64 %y_addr, %x_addr, %out_addr, %d64, %tmp;
    .reg .s64 %yidx, %xidx, %ystride, %xstride, %coord64, %prod;
    .reg .f64 %yv, %xv, %ay, %ax, %r, %z, %z2, %poly, %one, %tmpf, %tmpf2, %angle;
    .reg .pred %p, %done_loop, %y_nan, %x_nan, %y_neg, %x_neg, %y_zero, %x_zero, %y_inf, %x_inf, %use_inv, %use_shift;

    ld.param.u64 %y, [y_ptr];
    ld.param.u64 %x, [x_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %ystr, [y_strides_ptr];
    ld.param.u64 %xstr, [x_strides_ptr];
    ld.param.u64 %oshape, [out_shape_ptr];
    ld.param.s64 %yidx, [y_offset];
    ld.param.s64 %xidx, [x_offset];
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
    add.u64 %tmp, %ystr, %d64;
    ld.global.s64 %ystride, [%tmp];
    add.u64 %tmp, %xstr, %d64;
    ld.global.s64 %xstride, [%tmp];
    rem.u32 %coord, %rem, %shape_d;
    div.u32 %rem, %rem, %shape_d;
    cvt.s64.u32 %coord64, %coord;
    mul.lo.s64 %prod, %coord64, %ystride;
    add.s64 %yidx, %yidx, %prod;
    mul.lo.s64 %prod, %coord64, %xstride;
    add.s64 %xidx, %xidx, %prod;
    bra LOOP;
END_LOOP:

    cvt.u64.s64 %y_addr, %yidx;
    shl.b64 %y_addr, %y_addr, 3;
    add.u64 %y_addr, %y, %y_addr;
    ld.global.f64 %yv, [%y_addr];
    cvt.u64.s64 %x_addr, %xidx;
    shl.b64 %x_addr, %x_addr, 3;
    add.u64 %x_addr, %x, %x_addr;
    ld.global.f64 %xv, [%x_addr];
    cvt.u64.u32 %out_addr, %idx;
    shl.b64 %out_addr, %out_addr, 3;
    add.u64 %out_addr, %out, %out_addr;
@BODY@
STORE_VALUE:
    st.global.f64 [%out_addr], %angle;
    bra DONE;
STORE_Y_NAN:
    mov.b64 %out_bits64, %ybits64;
    st.global.u64 [%out_addr], %out_bits64;
    bra DONE;
STORE_X_NAN:
    mov.b64 %out_bits64, %xbits64;
    st.global.u64 [%out_addr], %out_bits64;
DONE:
    ret;
}
";

const U16_FORWARD_TEMPLATE: &str = r".version 7.0
.target sm_53
.address_size 64

.visible .entry @KERNEL@(
    .param .u64 y_ptr,
    .param .u64 x_ptr,
    .param .u64 out_ptr,
    .param .u64 y_strides_ptr,
    .param .u64 x_strides_ptr,
    .param .u64 out_shape_ptr,
    .param .s64 y_offset,
    .param .s64 x_offset,
    .param .u32 n,
    .param .u32 ndim
) {
    .reg .u32 %idx, %bid, %bdim, %nr, %ndimr;
    .reg .u32 %rem, %d, %shape_d, %coord;
    .reg .u32 %ybits, %xbits, %sign_y, %sign_x, %out_bits, %round, %lsb, %y_u32, %x_u32;
    .reg .u64 %y, %x, %out, %ystr, %xstr, %oshape;
    .reg .u64 %y_addr, %x_addr, %out_addr, %d64, %tmp;
    .reg .s64 %yidx, %xidx, %ystride, %xstride, %coord64, %prod;
    .reg .b16 %y_h, %x_h, %out_h, %zero16;
    .reg .u16 %out_u16;
    .reg .f32 %yv, %xv, %ay, %ax, %r, %z, %z2, %poly, %one, %tmpf, %tmpf2, %angle;
    .reg .pred %p, %done_loop, %y_nan, %x_nan, %y_neg, %x_neg, %y_zero, %x_zero, %y_inf, %x_inf, %use_inv, %use_shift;

    ld.param.u64 %y, [y_ptr];
    ld.param.u64 %x, [x_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %ystr, [y_strides_ptr];
    ld.param.u64 %xstr, [x_strides_ptr];
    ld.param.u64 %oshape, [out_shape_ptr];
    ld.param.s64 %yidx, [y_offset];
    ld.param.s64 %xidx, [x_offset];
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
    add.u64 %tmp, %ystr, %d64;
    ld.global.s64 %ystride, [%tmp];
    add.u64 %tmp, %xstr, %d64;
    ld.global.s64 %xstride, [%tmp];
    rem.u32 %coord, %rem, %shape_d;
    div.u32 %rem, %rem, %shape_d;
    cvt.s64.u32 %coord64, %coord;
    mul.lo.s64 %prod, %coord64, %ystride;
    add.s64 %yidx, %yidx, %prod;
    mul.lo.s64 %prod, %coord64, %xstride;
    add.s64 %xidx, %xidx, %prod;
    bra LOOP;
END_LOOP:

    cvt.u64.s64 %y_addr, %yidx;
    shl.b64 %y_addr, %y_addr, 1;
    add.u64 %y_addr, %y, %y_addr;
    cvt.u64.s64 %x_addr, %xidx;
    shl.b64 %x_addr, %x_addr, 1;
    add.u64 %x_addr, %x, %x_addr;
    cvt.u64.u32 %out_addr, %idx;
    shl.b64 %out_addr, %out_addr, 1;
    add.u64 %out_addr, %out, %out_addr;
@LOAD_YX@
@BODY@
STORE_VALUE:
@STORE_VALUE@
    bra DONE;
STORE_NAN:
    mov.u16 %out_u16, 0x7fff;
    st.global.u16 [%out_addr], %out_u16;
DONE:
    ret;
}
";

const BACKWARD_F32_TEMPLATE: &str = r".version 7.0
.target sm_52
.address_size 64

.visible .entry @KERNEL@(
    .param .u64 grad_ptr,
    .param .u64 y_ptr,
    .param .u64 x_ptr,
    .param .u64 out_ptr,
    .param .u64 grad_strides_ptr,
    .param .u64 y_strides_ptr,
    .param .u64 x_strides_ptr,
    .param .u64 out_shape_ptr,
    .param .s64 grad_offset,
    .param .s64 y_offset,
    .param .s64 x_offset,
    .param .u32 n,
    .param .u32 ndim,
    .param .u32 wrt_y
) {
    .reg .u32 %idx, %bid, %bdim, %nr, %ndimr, %wrt;
    .reg .u32 %rem, %d, %shape_d, %coord;
    .reg .u64 %grad, %y, %x, %out, %gstr, %ystr, %xstr, %oshape;
    .reg .u64 %g_addr, %y_addr, %x_addr, %out_addr, %d64, %tmp;
    .reg .s64 %gidx, %yidx, %xidx, %gstride, %ystride, %xstride, %coord64, %prod;
    .reg .f32 %g, %yv, %xv, %den, %num, %val, %tmpf;
    .reg .pred %p, %done_loop, %do_y, %den_zero;

    ld.param.u64 %grad, [grad_ptr];
    ld.param.u64 %y, [y_ptr];
    ld.param.u64 %x, [x_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %gstr, [grad_strides_ptr];
    ld.param.u64 %ystr, [y_strides_ptr];
    ld.param.u64 %xstr, [x_strides_ptr];
    ld.param.u64 %oshape, [out_shape_ptr];
    ld.param.s64 %gidx, [grad_offset];
    ld.param.s64 %yidx, [y_offset];
    ld.param.s64 %xidx, [x_offset];
    ld.param.u32 %nr, [n];
    ld.param.u32 %ndimr, [ndim];
    ld.param.u32 %wrt, [wrt_y];
    setp.ne.u32 %do_y, %wrt, 0;

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
    add.u64 %tmp, %ystr, %d64;
    ld.global.s64 %ystride, [%tmp];
    add.u64 %tmp, %xstr, %d64;
    ld.global.s64 %xstride, [%tmp];
    rem.u32 %coord, %rem, %shape_d;
    div.u32 %rem, %rem, %shape_d;
    cvt.s64.u32 %coord64, %coord;
    mul.lo.s64 %prod, %coord64, %gstride;
    add.s64 %gidx, %gidx, %prod;
    mul.lo.s64 %prod, %coord64, %ystride;
    add.s64 %yidx, %yidx, %prod;
    mul.lo.s64 %prod, %coord64, %xstride;
    add.s64 %xidx, %xidx, %prod;
    bra LOOP;
END_LOOP:

    cvt.u64.s64 %g_addr, %gidx;
    shl.b64 %g_addr, %g_addr, 2;
    add.u64 %g_addr, %grad, %g_addr;
    ld.global.f32 %g, [%g_addr];
    cvt.u64.s64 %y_addr, %yidx;
    shl.b64 %y_addr, %y_addr, 2;
    add.u64 %y_addr, %y, %y_addr;
    ld.global.f32 %yv, [%y_addr];
    cvt.u64.s64 %x_addr, %xidx;
    shl.b64 %x_addr, %x_addr, 2;
    add.u64 %x_addr, %x, %x_addr;
    ld.global.f32 %xv, [%x_addr];
    cvt.u64.u32 %out_addr, %idx;
    shl.b64 %out_addr, %out_addr, 2;
    add.u64 %out_addr, %out, %out_addr;

    mul.rn.f32 %den, %yv, %yv;
    fma.rn.f32 %den, %xv, %xv, %den;
    setp.eq.f32 %den_zero, %den, 0f00000000;
    @%den_zero bra STORE_ZERO;
    @%do_y mov.f32 %num, %xv;
    @!%do_y neg.f32 %num, %yv;
    mul.rn.f32 %val, %g, %num;
    div.rn.f32 %val, %val, %den;
    st.global.f32 [%out_addr], %val;
    bra DONE;
STORE_ZERO:
    mov.f32 %val, 0f00000000;
    st.global.f32 [%out_addr], %val;
DONE:
    ret;
}
";

const BACKWARD_F64_TEMPLATE: &str = r".version 7.0
.target sm_52
.address_size 64

.visible .entry @KERNEL@(
    .param .u64 grad_ptr,
    .param .u64 y_ptr,
    .param .u64 x_ptr,
    .param .u64 out_ptr,
    .param .u64 grad_strides_ptr,
    .param .u64 y_strides_ptr,
    .param .u64 x_strides_ptr,
    .param .u64 out_shape_ptr,
    .param .s64 grad_offset,
    .param .s64 y_offset,
    .param .s64 x_offset,
    .param .u32 n,
    .param .u32 ndim,
    .param .u32 wrt_y
) {
    .reg .u32 %idx, %bid, %bdim, %nr, %ndimr, %wrt;
    .reg .u32 %rem, %d, %shape_d, %coord;
    .reg .u64 %grad, %y, %x, %out, %gstr, %ystr, %xstr, %oshape;
    .reg .u64 %g_addr, %y_addr, %x_addr, %out_addr, %d64, %tmp;
    .reg .s64 %gidx, %yidx, %xidx, %gstride, %ystride, %xstride, %coord64, %prod;
    .reg .f64 %g, %yv, %xv, %den, %num, %val, %tmpf;
    .reg .pred %p, %done_loop, %do_y, %den_zero;

    ld.param.u64 %grad, [grad_ptr];
    ld.param.u64 %y, [y_ptr];
    ld.param.u64 %x, [x_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %gstr, [grad_strides_ptr];
    ld.param.u64 %ystr, [y_strides_ptr];
    ld.param.u64 %xstr, [x_strides_ptr];
    ld.param.u64 %oshape, [out_shape_ptr];
    ld.param.s64 %gidx, [grad_offset];
    ld.param.s64 %yidx, [y_offset];
    ld.param.s64 %xidx, [x_offset];
    ld.param.u32 %nr, [n];
    ld.param.u32 %ndimr, [ndim];
    ld.param.u32 %wrt, [wrt_y];
    setp.ne.u32 %do_y, %wrt, 0;

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
    add.u64 %tmp, %ystr, %d64;
    ld.global.s64 %ystride, [%tmp];
    add.u64 %tmp, %xstr, %d64;
    ld.global.s64 %xstride, [%tmp];
    rem.u32 %coord, %rem, %shape_d;
    div.u32 %rem, %rem, %shape_d;
    cvt.s64.u32 %coord64, %coord;
    mul.lo.s64 %prod, %coord64, %gstride;
    add.s64 %gidx, %gidx, %prod;
    mul.lo.s64 %prod, %coord64, %ystride;
    add.s64 %yidx, %yidx, %prod;
    mul.lo.s64 %prod, %coord64, %xstride;
    add.s64 %xidx, %xidx, %prod;
    bra LOOP;
END_LOOP:

    cvt.u64.s64 %g_addr, %gidx;
    shl.b64 %g_addr, %g_addr, 3;
    add.u64 %g_addr, %grad, %g_addr;
    ld.global.f64 %g, [%g_addr];
    cvt.u64.s64 %y_addr, %yidx;
    shl.b64 %y_addr, %y_addr, 3;
    add.u64 %y_addr, %y, %y_addr;
    ld.global.f64 %yv, [%y_addr];
    cvt.u64.s64 %x_addr, %xidx;
    shl.b64 %x_addr, %x_addr, 3;
    add.u64 %x_addr, %x, %x_addr;
    ld.global.f64 %xv, [%x_addr];
    cvt.u64.u32 %out_addr, %idx;
    shl.b64 %out_addr, %out_addr, 3;
    add.u64 %out_addr, %out, %out_addr;

    mul.rn.f64 %den, %yv, %yv;
    fma.rn.f64 %den, %xv, %xv, %den;
    setp.eq.f64 %den_zero, %den, 0d0000000000000000;
    @%den_zero bra STORE_ZERO;
    @%do_y mov.f64 %num, %xv;
    @!%do_y neg.f64 %num, %yv;
    mul.rn.f64 %val, %g, %num;
    div.rn.f64 %val, %val, %den;
    st.global.f64 [%out_addr], %val;
    bra DONE;
STORE_ZERO:
    mov.f64 %val, 0d0000000000000000;
    st.global.f64 [%out_addr], %val;
DONE:
    ret;
}
";

const BACKWARD_U16_TEMPLATE: &str = r".version 7.0
.target sm_53
.address_size 64

.visible .entry @KERNEL@(
    .param .u64 grad_ptr,
    .param .u64 y_ptr,
    .param .u64 x_ptr,
    .param .u64 out_ptr,
    .param .u64 grad_strides_ptr,
    .param .u64 y_strides_ptr,
    .param .u64 x_strides_ptr,
    .param .u64 out_shape_ptr,
    .param .s64 grad_offset,
    .param .s64 y_offset,
    .param .s64 x_offset,
    .param .u32 n,
    .param .u32 ndim,
    .param .u32 wrt_y
) {
    .reg .u32 %idx, %bid, %bdim, %nr, %ndimr, %wrt;
    .reg .u32 %rem, %d, %shape_d, %coord;
    .reg .u32 %bits, %round, %lsb, %g_u32, %y_u32, %x_u32;
    .reg .u64 %grad, %y, %x, %out, %gstr, %ystr, %xstr, %oshape;
    .reg .u64 %g_addr, %y_addr, %x_addr, %out_addr, %d64, %tmp;
    .reg .s64 %gidx, %yidx, %xidx, %gstride, %ystride, %xstride, %coord64, %prod;
    .reg .b16 %g_h, %y_h, %x_h, %out_h, %zero16;
    .reg .u16 %out_u16;
    .reg .f32 %g, %yv, %xv, %den, %num, %val, %tmpf;
    .reg .pred %p, %done_loop, %do_y, %den_zero, %is_nan;

    ld.param.u64 %grad, [grad_ptr];
    ld.param.u64 %y, [y_ptr];
    ld.param.u64 %x, [x_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %gstr, [grad_strides_ptr];
    ld.param.u64 %ystr, [y_strides_ptr];
    ld.param.u64 %xstr, [x_strides_ptr];
    ld.param.u64 %oshape, [out_shape_ptr];
    ld.param.s64 %gidx, [grad_offset];
    ld.param.s64 %yidx, [y_offset];
    ld.param.s64 %xidx, [x_offset];
    ld.param.u32 %nr, [n];
    ld.param.u32 %ndimr, [ndim];
    ld.param.u32 %wrt, [wrt_y];
    setp.ne.u32 %do_y, %wrt, 0;

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
    add.u64 %tmp, %ystr, %d64;
    ld.global.s64 %ystride, [%tmp];
    add.u64 %tmp, %xstr, %d64;
    ld.global.s64 %xstride, [%tmp];
    rem.u32 %coord, %rem, %shape_d;
    div.u32 %rem, %rem, %shape_d;
    cvt.s64.u32 %coord64, %coord;
    mul.lo.s64 %prod, %coord64, %gstride;
    add.s64 %gidx, %gidx, %prod;
    mul.lo.s64 %prod, %coord64, %ystride;
    add.s64 %yidx, %yidx, %prod;
    mul.lo.s64 %prod, %coord64, %xstride;
    add.s64 %xidx, %xidx, %prod;
    bra LOOP;
END_LOOP:

    cvt.u64.s64 %g_addr, %gidx;
    shl.b64 %g_addr, %g_addr, 1;
    add.u64 %g_addr, %grad, %g_addr;
    cvt.u64.s64 %y_addr, %yidx;
    shl.b64 %y_addr, %y_addr, 1;
    add.u64 %y_addr, %y, %y_addr;
    cvt.u64.s64 %x_addr, %xidx;
    shl.b64 %x_addr, %x_addr, 1;
    add.u64 %x_addr, %x, %x_addr;
    cvt.u64.u32 %out_addr, %idx;
    shl.b64 %out_addr, %out_addr, 1;
    add.u64 %out_addr, %out, %out_addr;
@LOAD_GYX@
    mul.rn.f32 %den, %yv, %yv;
    fma.rn.f32 %den, %xv, %xv, %den;
    setp.eq.f32 %den_zero, %den, 0f00000000;
    @%den_zero bra STORE_ZERO;
    @%do_y mov.f32 %num, %xv;
    @!%do_y neg.f32 %num, %yv;
    mul.rn.f32 %val, %g, %num;
    div.rn.f32 %val, %val, %den;
    setp.nan.f32 %is_nan, %val, %val;
    @%is_nan bra STORE_NAN;
@STORE_VALUE@
    bra DONE;
STORE_ZERO:
    mov.u16 %out_u16, 0;
    st.global.u16 [%out_addr], %out_u16;
    bra DONE;
STORE_NAN:
    mov.u16 %out_u16, 0x7fff;
    st.global.u16 [%out_addr], %out_u16;
DONE:
    ret;
}
";

const LOAD_F16_YX: &str = r"
    ld.global.b16 %y_h, [%y_addr];
    ld.global.b16 %x_h, [%x_addr];
    cvt.f32.f16 %yv, %y_h;
    cvt.f32.f16 %xv, %x_h;
";

const LOAD_BF16_YX: &str = r"
    ld.global.b16 %y_h, [%y_addr];
    ld.global.b16 %x_h, [%x_addr];
    mov.b16 %zero16, 0;
    mov.b32 %y_u32, {%zero16, %y_h};
    mov.b32 %x_u32, {%zero16, %x_h};
    mov.b32 %yv, %y_u32;
    mov.b32 %xv, %x_u32;
";

const STORE_F16_VALUE: &str = r"
    cvt.rn.f16.f32 %out_h, %angle;
    st.global.b16 [%out_addr], %out_h;
";

const STORE_BF16_VALUE: &str = r"
    mov.b32 %out_bits, %angle;
    shr.u32 %lsb, %out_bits, 16;
    and.b32 %lsb, %lsb, 1;
    add.u32 %round, %out_bits, 0x7fff;
    add.u32 %round, %round, %lsb;
    shr.u32 %out_bits, %round, 16;
    cvt.u16.u32 %out_u16, %out_bits;
    st.global.u16 [%out_addr], %out_u16;
";

const LOAD_F16_GYX: &str = r"
    ld.global.b16 %g_h, [%g_addr];
    ld.global.b16 %y_h, [%y_addr];
    ld.global.b16 %x_h, [%x_addr];
    cvt.f32.f16 %g, %g_h;
    cvt.f32.f16 %yv, %y_h;
    cvt.f32.f16 %xv, %x_h;
";

const LOAD_BF16_GYX: &str = r"
    ld.global.b16 %g_h, [%g_addr];
    ld.global.b16 %y_h, [%y_addr];
    ld.global.b16 %x_h, [%x_addr];
    mov.b16 %zero16, 0;
    mov.b32 %g_u32, {%zero16, %g_h};
    mov.b32 %y_u32, {%zero16, %y_h};
    mov.b32 %x_u32, {%zero16, %x_h};
    mov.b32 %g, %g_u32;
    mov.b32 %yv, %y_u32;
    mov.b32 %xv, %x_u32;
";

const STORE_F16_BACKWARD: &str = r"
    cvt.rn.f16.f32 %out_h, %val;
    st.global.b16 [%out_addr], %out_h;
";

const STORE_BF16_BACKWARD: &str = r"
    mov.b32 %bits, %val;
    shr.u32 %lsb, %bits, 16;
    and.b32 %lsb, %lsb, 1;
    add.u32 %round, %bits, 0x7fff;
    add.u32 %round, %round, %lsb;
    shr.u32 %bits, %round, 16;
    cvt.u16.u32 %out_u16, %bits;
    st.global.u16 [%out_addr], %out_u16;
";

fn substitute(mut template: String, replacements: &[(&str, &str)]) -> String {
    for (needle, replacement) in replacements {
        template = template.replace(needle, replacement);
    }
    template
}

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
    y_len: usize,
    x_len: usize,
    y_shape: &[usize],
    y_strides: &[isize],
    y_offset: usize,
    x_shape: &[usize],
    x_strides: &[isize],
    x_offset: usize,
    op: &'static str,
) -> GpuResult<()> {
    validate_view_bounds(y_len, y_shape, y_strides, y_offset, op)?;
    validate_view_bounds(x_len, x_shape, x_strides, x_offset, op)?;
    Ok(())
}

fn validate_backward_lengths(
    grad_len: usize,
    y_len: usize,
    x_len: usize,
    grad_strides: &[isize],
    grad_offset: usize,
    y_shape: &[usize],
    y_strides: &[isize],
    y_offset: usize,
    x_shape: &[usize],
    x_strides: &[isize],
    x_offset: usize,
    out_shape: &[usize],
    op: &'static str,
) -> GpuResult<()> {
    validate_view_bounds(grad_len, out_shape, grad_strides, grad_offset, op)?;
    validate_view_bounds(y_len, y_shape, y_strides, y_offset, op)?;
    validate_view_bounds(x_len, x_shape, x_strides, x_offset, op)?;
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn forward_f32_ptx(kernel_name: &'static str) -> String {
    substitute(
        F32_FORWARD_TEMPLATE.to_string(),
        &[("@KERNEL@", kernel_name), ("@BODY@", ATAN2_F32_BODY)],
    )
}

#[allow(clippy::too_many_lines)]
fn forward_f64_ptx(kernel_name: &'static str) -> String {
    substitute(
        F64_FORWARD_TEMPLATE.to_string(),
        &[("@KERNEL@", kernel_name), ("@BODY@", ATAN2_F64_BODY)],
    )
}

fn forward_u16_ptx(
    kernel_name: &'static str,
    load_yx: &'static str,
    store_value: &'static str,
) -> String {
    substitute(
        U16_FORWARD_TEMPLATE.to_string(),
        &[
            ("@KERNEL@", kernel_name),
            ("@LOAD_YX@", load_yx),
            ("@BODY@", ATAN2_F32_BODY),
            ("@STORE_VALUE@", store_value),
        ],
    )
}

fn backward_f32_ptx(kernel_name: &'static str) -> String {
    substitute(
        BACKWARD_F32_TEMPLATE.to_string(),
        &[("@KERNEL@", kernel_name)],
    )
}

fn backward_f64_ptx(kernel_name: &'static str) -> String {
    substitute(
        BACKWARD_F64_TEMPLATE.to_string(),
        &[("@KERNEL@", kernel_name)],
    )
}

fn backward_u16_ptx(
    kernel_name: &'static str,
    load_gyx: &'static str,
    store_value: &'static str,
) -> String {
    substitute(
        BACKWARD_U16_TEMPLATE.to_string(),
        &[
            ("@KERNEL@", kernel_name),
            ("@LOAD_GYX@", load_gyx),
            ("@STORE_VALUE@", store_value),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_forward_f32(
    y: &CudaBuffer<f32>,
    x: &CudaBuffer<f32>,
    y_shape: &[usize],
    y_strides: &[isize],
    y_offset: usize,
    x_shape: &[usize],
    x_strides: &[isize],
    x_offset: usize,
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    let op = "atan2_f32";
    validate_forward_lengths(
        y.len(),
        x.len(),
        y_shape,
        y_strides,
        y_offset,
        x_shape,
        x_strides,
        x_offset,
        op,
    )?;
    let out_numel = checked_numel(out_shape, op)?;
    if out_numel == 0 {
        return alloc_zeros_f32(0, device);
    }

    let y_strides = broadcast_source_strides(y_shape, y_strides, out_shape, op)?;
    let x_strides = broadcast_source_strides(x_shape, x_strides, out_shape, op)?;
    let shape_u32 = checked_shape_u32(out_shape, op)?;
    let y_strides_buf = cpu_to_gpu(&y_strides, device)?;
    let x_strides_buf = cpu_to_gpu(&x_strides, device)?;
    let shape_buf = cpu_to_gpu(&shape_u32, device)?;
    let y_offset_i64 = checked_i64(y_offset, op, "y offset")?;
    let x_offset_i64 = checked_i64(x_offset, op, "x offset")?;
    let mut out = alloc_zeros_f32(out_numel, device)?;
    let kernel_name = "atan2_f32_kernel";
    let f = get_or_compile_owned(
        device.context(),
        forward_f32_ptx(kernel_name),
        kernel_name.to_string(),
        device.ordinal() as u32,
    )
    .map_err(|source| GpuError::PtxCompileFailed {
        kernel: kernel_name,
        source,
    })?;
    let cfg = launch_cfg(out_numel, op)?;
    let n_u32 = checked_u32(out_numel, op)?;
    let ndim_u32 = checked_u32(out_shape.len(), op)?;

    // SAFETY: the generated PTX ABI matches the pushed arguments; all view
    // bounds and broadcast strides were checked against the backing buffers.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(y.inner())
            .arg(x.inner())
            .arg(out.inner_mut())
            .arg(y_strides_buf.inner())
            .arg(x_strides_buf.inner())
            .arg(shape_buf.inner())
            .arg(&y_offset_i64)
            .arg(&x_offset_i64)
            .arg(&n_u32)
            .arg(&ndim_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn launch_forward_f64(
    y: &CudaBuffer<f64>,
    x: &CudaBuffer<f64>,
    y_shape: &[usize],
    y_strides: &[isize],
    y_offset: usize,
    x_shape: &[usize],
    x_strides: &[isize],
    x_offset: usize,
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    let op = "atan2_f64";
    validate_forward_lengths(
        y.len(),
        x.len(),
        y_shape,
        y_strides,
        y_offset,
        x_shape,
        x_strides,
        x_offset,
        op,
    )?;
    let out_numel = checked_numel(out_shape, op)?;
    if out_numel == 0 {
        return alloc_zeros_f64(0, device);
    }

    let y_strides = broadcast_source_strides(y_shape, y_strides, out_shape, op)?;
    let x_strides = broadcast_source_strides(x_shape, x_strides, out_shape, op)?;
    let shape_u32 = checked_shape_u32(out_shape, op)?;
    let y_strides_buf = cpu_to_gpu(&y_strides, device)?;
    let x_strides_buf = cpu_to_gpu(&x_strides, device)?;
    let shape_buf = cpu_to_gpu(&shape_u32, device)?;
    let y_offset_i64 = checked_i64(y_offset, op, "y offset")?;
    let x_offset_i64 = checked_i64(x_offset, op, "x offset")?;
    let mut out = alloc_zeros_f64(out_numel, device)?;
    let kernel_name = "atan2_f64_kernel";
    let f = get_or_compile_owned(
        device.context(),
        forward_f64_ptx(kernel_name),
        kernel_name.to_string(),
        device.ordinal() as u32,
    )
    .map_err(|source| GpuError::PtxCompileFailed {
        kernel: kernel_name,
        source,
    })?;
    let cfg = launch_cfg(out_numel, op)?;
    let n_u32 = checked_u32(out_numel, op)?;
    let ndim_u32 = checked_u32(out_shape.len(), op)?;

    // SAFETY: same ABI and bounds contract as `launch_forward_f32`, with f64
    // element addressing in the generated PTX.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(y.inner())
            .arg(x.inner())
            .arg(out.inner_mut())
            .arg(y_strides_buf.inner())
            .arg(x_strides_buf.inner())
            .arg(shape_buf.inner())
            .arg(&y_offset_i64)
            .arg(&x_offset_i64)
            .arg(&n_u32)
            .arg(&ndim_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn launch_forward_u16(
    y: &CudaSlice<u16>,
    x: &CudaSlice<u16>,
    y_shape: &[usize],
    y_strides: &[isize],
    y_offset: usize,
    x_shape: &[usize],
    x_strides: &[isize],
    x_offset: usize,
    out_shape: &[usize],
    device: &GpuDevice,
    kernel_name: &'static str,
    ptx: String,
    op: &'static str,
) -> GpuResult<CudaSlice<u16>> {
    validate_forward_lengths(
        y.len(),
        x.len(),
        y_shape,
        y_strides,
        y_offset,
        x_shape,
        x_strides,
        x_offset,
        op,
    )?;
    let out_numel = checked_numel(out_shape, op)?;
    let stream = device.stream();
    if out_numel == 0 {
        return Ok(stream.alloc_zeros::<u16>(0)?);
    }

    let y_strides = broadcast_source_strides(y_shape, y_strides, out_shape, op)?;
    let x_strides = broadcast_source_strides(x_shape, x_strides, out_shape, op)?;
    let shape_u32 = checked_shape_u32(out_shape, op)?;
    let y_strides_buf = cpu_to_gpu(&y_strides, device)?;
    let x_strides_buf = cpu_to_gpu(&x_strides, device)?;
    let shape_buf = cpu_to_gpu(&shape_u32, device)?;
    let y_offset_i64 = checked_i64(y_offset, op, "y offset")?;
    let x_offset_i64 = checked_i64(x_offset, op, "x offset")?;
    let mut out = stream.alloc_zeros::<u16>(out_numel)?;
    let f = get_or_compile_owned(
        device.context(),
        ptx,
        kernel_name.to_string(),
        device.ordinal() as u32,
    )
    .map_err(|source| GpuError::PtxCompileFailed {
        kernel: kernel_name,
        source,
    })?;
    let cfg = launch_cfg(out_numel, op)?;
    let n_u32 = checked_u32(out_numel, op)?;
    let ndim_u32 = checked_u32(out_shape.len(), op)?;

    // SAFETY: the generated u16 PTX ABI matches these arguments and uses f32
    // opmath before storing f16/bf16-rounded results.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(y)
            .arg(x)
            .arg(&mut out)
            .arg(y_strides_buf.inner())
            .arg(x_strides_buf.inner())
            .arg(shape_buf.inner())
            .arg(&y_offset_i64)
            .arg(&x_offset_i64)
            .arg(&n_u32)
            .arg(&ndim_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn launch_backward_f32(
    grad_output: &CudaBuffer<f32>,
    y: &CudaBuffer<f32>,
    x: &CudaBuffer<f32>,
    grad_strides: &[isize],
    grad_offset: usize,
    y_shape: &[usize],
    y_strides: &[isize],
    y_offset: usize,
    x_shape: &[usize],
    x_strides: &[isize],
    x_offset: usize,
    out_shape: &[usize],
    wrt_y: bool,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    let op = "atan2_backward_f32";
    validate_backward_lengths(
        grad_output.len(),
        y.len(),
        x.len(),
        grad_strides,
        grad_offset,
        y_shape,
        y_strides,
        y_offset,
        x_shape,
        x_strides,
        x_offset,
        out_shape,
        op,
    )?;
    let out_numel = checked_numel(out_shape, op)?;
    if out_numel == 0 {
        return alloc_zeros_f32(0, device);
    }

    let grad_strides = broadcast_source_strides(out_shape, grad_strides, out_shape, op)?;
    let y_strides = broadcast_source_strides(y_shape, y_strides, out_shape, op)?;
    let x_strides = broadcast_source_strides(x_shape, x_strides, out_shape, op)?;
    let shape_u32 = checked_shape_u32(out_shape, op)?;
    let grad_strides_buf = cpu_to_gpu(&grad_strides, device)?;
    let y_strides_buf = cpu_to_gpu(&y_strides, device)?;
    let x_strides_buf = cpu_to_gpu(&x_strides, device)?;
    let shape_buf = cpu_to_gpu(&shape_u32, device)?;
    let grad_offset_i64 = checked_i64(grad_offset, op, "grad offset")?;
    let y_offset_i64 = checked_i64(y_offset, op, "y offset")?;
    let x_offset_i64 = checked_i64(x_offset, op, "x offset")?;
    let mut out = alloc_zeros_f32(out_numel, device)?;
    let kernel_name = "atan2_backward_f32_kernel";
    let f = get_or_compile_owned(
        device.context(),
        backward_f32_ptx(kernel_name),
        kernel_name.to_string(),
        device.ordinal() as u32,
    )
    .map_err(|source| GpuError::PtxCompileFailed {
        kernel: kernel_name,
        source,
    })?;
    let cfg = launch_cfg(out_numel, op)?;
    let n_u32 = checked_u32(out_numel, op)?;
    let ndim_u32 = checked_u32(out_shape.len(), op)?;
    let wrt_y_u32 = u32::from(wrt_y);

    // SAFETY: the backward PTX ABI matches these arguments; every source view
    // is bounds-checked and the output is a fresh contiguous buffer.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(grad_output.inner())
            .arg(y.inner())
            .arg(x.inner())
            .arg(out.inner_mut())
            .arg(grad_strides_buf.inner())
            .arg(y_strides_buf.inner())
            .arg(x_strides_buf.inner())
            .arg(shape_buf.inner())
            .arg(&grad_offset_i64)
            .arg(&y_offset_i64)
            .arg(&x_offset_i64)
            .arg(&n_u32)
            .arg(&ndim_u32)
            .arg(&wrt_y_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn launch_backward_f64(
    grad_output: &CudaBuffer<f64>,
    y: &CudaBuffer<f64>,
    x: &CudaBuffer<f64>,
    grad_strides: &[isize],
    grad_offset: usize,
    y_shape: &[usize],
    y_strides: &[isize],
    y_offset: usize,
    x_shape: &[usize],
    x_strides: &[isize],
    x_offset: usize,
    out_shape: &[usize],
    wrt_y: bool,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    let op = "atan2_backward_f64";
    validate_backward_lengths(
        grad_output.len(),
        y.len(),
        x.len(),
        grad_strides,
        grad_offset,
        y_shape,
        y_strides,
        y_offset,
        x_shape,
        x_strides,
        x_offset,
        out_shape,
        op,
    )?;
    let out_numel = checked_numel(out_shape, op)?;
    if out_numel == 0 {
        return alloc_zeros_f64(0, device);
    }

    let grad_strides = broadcast_source_strides(out_shape, grad_strides, out_shape, op)?;
    let y_strides = broadcast_source_strides(y_shape, y_strides, out_shape, op)?;
    let x_strides = broadcast_source_strides(x_shape, x_strides, out_shape, op)?;
    let shape_u32 = checked_shape_u32(out_shape, op)?;
    let grad_strides_buf = cpu_to_gpu(&grad_strides, device)?;
    let y_strides_buf = cpu_to_gpu(&y_strides, device)?;
    let x_strides_buf = cpu_to_gpu(&x_strides, device)?;
    let shape_buf = cpu_to_gpu(&shape_u32, device)?;
    let grad_offset_i64 = checked_i64(grad_offset, op, "grad offset")?;
    let y_offset_i64 = checked_i64(y_offset, op, "y offset")?;
    let x_offset_i64 = checked_i64(x_offset, op, "x offset")?;
    let mut out = alloc_zeros_f64(out_numel, device)?;
    let kernel_name = "atan2_backward_f64_kernel";
    let f = get_or_compile_owned(
        device.context(),
        backward_f64_ptx(kernel_name),
        kernel_name.to_string(),
        device.ordinal() as u32,
    )
    .map_err(|source| GpuError::PtxCompileFailed {
        kernel: kernel_name,
        source,
    })?;
    let cfg = launch_cfg(out_numel, op)?;
    let n_u32 = checked_u32(out_numel, op)?;
    let ndim_u32 = checked_u32(out_shape.len(), op)?;
    let wrt_y_u32 = u32::from(wrt_y);

    // SAFETY: same contract as `launch_backward_f32`, with f64 element
    // addressing in the generated PTX.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(grad_output.inner())
            .arg(y.inner())
            .arg(x.inner())
            .arg(out.inner_mut())
            .arg(grad_strides_buf.inner())
            .arg(y_strides_buf.inner())
            .arg(x_strides_buf.inner())
            .arg(shape_buf.inner())
            .arg(&grad_offset_i64)
            .arg(&y_offset_i64)
            .arg(&x_offset_i64)
            .arg(&n_u32)
            .arg(&ndim_u32)
            .arg(&wrt_y_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn launch_backward_u16(
    grad_output: &CudaSlice<u16>,
    y: &CudaSlice<u16>,
    x: &CudaSlice<u16>,
    grad_strides: &[isize],
    grad_offset: usize,
    y_shape: &[usize],
    y_strides: &[isize],
    y_offset: usize,
    x_shape: &[usize],
    x_strides: &[isize],
    x_offset: usize,
    out_shape: &[usize],
    wrt_y: bool,
    device: &GpuDevice,
    kernel_name: &'static str,
    ptx: String,
    op: &'static str,
) -> GpuResult<CudaSlice<u16>> {
    validate_backward_lengths(
        grad_output.len(),
        y.len(),
        x.len(),
        grad_strides,
        grad_offset,
        y_shape,
        y_strides,
        y_offset,
        x_shape,
        x_strides,
        x_offset,
        out_shape,
        op,
    )?;
    let out_numel = checked_numel(out_shape, op)?;
    let stream = device.stream();
    if out_numel == 0 {
        return Ok(stream.alloc_zeros::<u16>(0)?);
    }

    let grad_strides = broadcast_source_strides(out_shape, grad_strides, out_shape, op)?;
    let y_strides = broadcast_source_strides(y_shape, y_strides, out_shape, op)?;
    let x_strides = broadcast_source_strides(x_shape, x_strides, out_shape, op)?;
    let shape_u32 = checked_shape_u32(out_shape, op)?;
    let grad_strides_buf = cpu_to_gpu(&grad_strides, device)?;
    let y_strides_buf = cpu_to_gpu(&y_strides, device)?;
    let x_strides_buf = cpu_to_gpu(&x_strides, device)?;
    let shape_buf = cpu_to_gpu(&shape_u32, device)?;
    let grad_offset_i64 = checked_i64(grad_offset, op, "grad offset")?;
    let y_offset_i64 = checked_i64(y_offset, op, "y offset")?;
    let x_offset_i64 = checked_i64(x_offset, op, "x offset")?;
    let mut out = stream.alloc_zeros::<u16>(out_numel)?;
    let f = get_or_compile_owned(
        device.context(),
        ptx,
        kernel_name.to_string(),
        device.ordinal() as u32,
    )
    .map_err(|source| GpuError::PtxCompileFailed {
        kernel: kernel_name,
        source,
    })?;
    let cfg = launch_cfg(out_numel, op)?;
    let n_u32 = checked_u32(out_numel, op)?;
    let ndim_u32 = checked_u32(out_shape.len(), op)?;
    let wrt_y_u32 = u32::from(wrt_y);

    // SAFETY: the generated u16 backward PTX ABI matches these arguments and
    // writes one element in a fresh output for each logical broadcast position.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(grad_output)
            .arg(y)
            .arg(x)
            .arg(&mut out)
            .arg(grad_strides_buf.inner())
            .arg(y_strides_buf.inner())
            .arg(x_strides_buf.inner())
            .arg(shape_buf.inner())
            .arg(&grad_offset_i64)
            .arg(&y_offset_i64)
            .arg(&x_offset_i64)
            .arg(&n_u32)
            .arg(&ndim_u32)
            .arg(&wrt_y_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Broadcast `atan2(y, x)` for f32 CUDA buffers.
pub fn gpu_atan2_f32(
    y: &CudaBuffer<f32>,
    x: &CudaBuffer<f32>,
    y_shape: &[usize],
    y_strides: &[isize],
    y_offset: usize,
    x_shape: &[usize],
    x_strides: &[isize],
    x_offset: usize,
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    if y.device_ordinal() != x.device_ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: y.device_ordinal(),
            got: x.device_ordinal(),
        });
    }
    launch_forward_f32(
        y, x, y_shape, y_strides, y_offset, x_shape, x_strides, x_offset, out_shape, device,
    )
}

/// Broadcast `atan2(y, x)` for f64 CUDA buffers.
pub fn gpu_atan2_f64(
    y: &CudaBuffer<f64>,
    x: &CudaBuffer<f64>,
    y_shape: &[usize],
    y_strides: &[isize],
    y_offset: usize,
    x_shape: &[usize],
    x_strides: &[isize],
    x_offset: usize,
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    if y.device_ordinal() != x.device_ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: y.device_ordinal(),
            got: x.device_ordinal(),
        });
    }
    launch_forward_f64(
        y, x, y_shape, y_strides, y_offset, x_shape, x_strides, x_offset, out_shape, device,
    )
}

/// Broadcast `atan2(y, x)` for f16 CUDA buffers using f32 opmath.
pub fn gpu_atan2_f16(
    y: &CudaSlice<u16>,
    x: &CudaSlice<u16>,
    y_shape: &[usize],
    y_strides: &[isize],
    y_offset: usize,
    x_shape: &[usize],
    x_strides: &[isize],
    x_offset: usize,
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch_forward_u16(
        y,
        x,
        y_shape,
        y_strides,
        y_offset,
        x_shape,
        x_strides,
        x_offset,
        out_shape,
        device,
        "atan2_f16_kernel",
        forward_u16_ptx("atan2_f16_kernel", LOAD_F16_YX, STORE_F16_VALUE),
        "atan2_f16",
    )
}

/// Broadcast `atan2(y, x)` for bf16 CUDA buffers using f32 opmath.
pub fn gpu_atan2_bf16(
    y: &CudaSlice<u16>,
    x: &CudaSlice<u16>,
    y_shape: &[usize],
    y_strides: &[isize],
    y_offset: usize,
    x_shape: &[usize],
    x_strides: &[isize],
    x_offset: usize,
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch_forward_u16(
        y,
        x,
        y_shape,
        y_strides,
        y_offset,
        x_shape,
        x_strides,
        x_offset,
        out_shape,
        device,
        "atan2_bf16_kernel",
        forward_u16_ptx("atan2_bf16_kernel", LOAD_BF16_YX, STORE_BF16_VALUE),
        "atan2_bf16",
    )
}

/// Resident CUDA VJP for f32 `atan2`; `wrt_y` selects y-gradient vs x-gradient.
pub fn gpu_atan2_backward_f32(
    grad_output: &CudaBuffer<f32>,
    y: &CudaBuffer<f32>,
    x: &CudaBuffer<f32>,
    grad_strides: &[isize],
    grad_offset: usize,
    y_shape: &[usize],
    y_strides: &[isize],
    y_offset: usize,
    x_shape: &[usize],
    x_strides: &[isize],
    x_offset: usize,
    out_shape: &[usize],
    wrt_y: bool,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    launch_backward_f32(
        grad_output,
        y,
        x,
        grad_strides,
        grad_offset,
        y_shape,
        y_strides,
        y_offset,
        x_shape,
        x_strides,
        x_offset,
        out_shape,
        wrt_y,
        device,
    )
}

/// Resident CUDA VJP for f64 `atan2`; `wrt_y` selects y-gradient vs x-gradient.
pub fn gpu_atan2_backward_f64(
    grad_output: &CudaBuffer<f64>,
    y: &CudaBuffer<f64>,
    x: &CudaBuffer<f64>,
    grad_strides: &[isize],
    grad_offset: usize,
    y_shape: &[usize],
    y_strides: &[isize],
    y_offset: usize,
    x_shape: &[usize],
    x_strides: &[isize],
    x_offset: usize,
    out_shape: &[usize],
    wrt_y: bool,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    launch_backward_f64(
        grad_output,
        y,
        x,
        grad_strides,
        grad_offset,
        y_shape,
        y_strides,
        y_offset,
        x_shape,
        x_strides,
        x_offset,
        out_shape,
        wrt_y,
        device,
    )
}

/// Resident CUDA VJP for f16 `atan2`; `wrt_y` selects y-gradient vs x-gradient.
pub fn gpu_atan2_backward_f16(
    grad_output: &CudaSlice<u16>,
    y: &CudaSlice<u16>,
    x: &CudaSlice<u16>,
    grad_strides: &[isize],
    grad_offset: usize,
    y_shape: &[usize],
    y_strides: &[isize],
    y_offset: usize,
    x_shape: &[usize],
    x_strides: &[isize],
    x_offset: usize,
    out_shape: &[usize],
    wrt_y: bool,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch_backward_u16(
        grad_output,
        y,
        x,
        grad_strides,
        grad_offset,
        y_shape,
        y_strides,
        y_offset,
        x_shape,
        x_strides,
        x_offset,
        out_shape,
        wrt_y,
        device,
        "atan2_backward_f16_kernel",
        backward_u16_ptx(
            "atan2_backward_f16_kernel",
            LOAD_F16_GYX,
            STORE_F16_BACKWARD,
        ),
        "atan2_backward_f16",
    )
}

/// Resident CUDA VJP for bf16 `atan2`; `wrt_y` selects y-gradient vs x-gradient.
pub fn gpu_atan2_backward_bf16(
    grad_output: &CudaSlice<u16>,
    y: &CudaSlice<u16>,
    x: &CudaSlice<u16>,
    grad_strides: &[isize],
    grad_offset: usize,
    y_shape: &[usize],
    y_strides: &[isize],
    y_offset: usize,
    x_shape: &[usize],
    x_strides: &[isize],
    x_offset: usize,
    out_shape: &[usize],
    wrt_y: bool,
    device: &GpuDevice,
) -> GpuResult<CudaSlice<u16>> {
    launch_backward_u16(
        grad_output,
        y,
        x,
        grad_strides,
        grad_offset,
        y_shape,
        y_strides,
        y_offset,
        x_shape,
        x_strides,
        x_offset,
        out_shape,
        wrt_y,
        device,
        "atan2_backward_bf16_kernel",
        backward_u16_ptx(
            "atan2_backward_bf16_kernel",
            LOAD_BF16_GYX,
            STORE_BF16_BACKWARD,
        ),
        "atan2_backward_bf16",
    )
}
