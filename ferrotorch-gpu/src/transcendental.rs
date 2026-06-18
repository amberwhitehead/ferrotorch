//! CUDA-resident unary transcendental and rounding kernels.
//!
//! These kernels back `ferrotorch-core`'s autograd transcendental family for
//! operations that cannot be expressed accurately with the older primitive
//! PTX slots alone. They are hand-written/generated PTX loaded through the
//! crate module cache; there is no CUDA C, NVRTC, or host round trip.

use ferrotorch_core::gpu_dispatch::GpuUnaryOp;

use crate::buffer::CudaBuffer;
use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};

#[cfg(feature = "cuda")]
use crate::transfer::{alloc_zeros_f32, alloc_zeros_f64};

#[cfg(feature = "cuda")]
fn validate_device<T>(a: &CudaBuffer<T>, device: &GpuDevice, op: &'static str) -> GpuResult<()> {
    if a.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: device.ordinal(),
            got: a.device_ordinal(),
        });
    }
    if a.len() > u32::MAX as usize {
        return Err(GpuError::InvalidState {
            message: format!("{op}: input length {} exceeds PTX u32 ABI", a.len()),
        });
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn f32_kernel_name(op: GpuUnaryOp) -> &'static str {
    match op {
        GpuUnaryOp::Atan => "unary_special_atan_f32_kernel",
        GpuUnaryOp::Ceil => "unary_special_ceil_f32_kernel",
        GpuUnaryOp::Floor => "unary_special_floor_f32_kernel",
        GpuUnaryOp::Round => "unary_special_round_f32_kernel",
        GpuUnaryOp::Trunc => "unary_special_trunc_f32_kernel",
        GpuUnaryOp::Frac => "unary_special_frac_f32_kernel",
        GpuUnaryOp::Sign => "unary_special_sign_f32_kernel",
    }
}

#[cfg(feature = "cuda")]
fn f64_kernel_name(op: GpuUnaryOp) -> &'static str {
    match op {
        GpuUnaryOp::Atan => "unary_special_atan_f64_kernel",
        GpuUnaryOp::Ceil => "unary_special_ceil_f64_kernel",
        GpuUnaryOp::Floor => "unary_special_floor_f64_kernel",
        GpuUnaryOp::Round => "unary_special_round_f64_kernel",
        GpuUnaryOp::Trunc => "unary_special_trunc_f64_kernel",
        GpuUnaryOp::Frac => "unary_special_frac_f64_kernel",
        GpuUnaryOp::Sign => "unary_special_sign_f64_kernel",
    }
}

#[cfg(feature = "cuda")]
fn bf16_kernel_name(op: GpuUnaryOp) -> &'static str {
    match op {
        GpuUnaryOp::Atan => "unary_special_atan_bf16_kernel",
        GpuUnaryOp::Ceil => "unary_special_ceil_bf16_kernel",
        GpuUnaryOp::Floor => "unary_special_floor_bf16_kernel",
        GpuUnaryOp::Round => "unary_special_round_bf16_kernel",
        GpuUnaryOp::Trunc => "unary_special_trunc_bf16_kernel",
        GpuUnaryOp::Frac => "unary_special_frac_bf16_kernel",
        GpuUnaryOp::Sign => "unary_special_sign_bf16_kernel",
    }
}

#[cfg(feature = "cuda")]
fn f16_kernel_name(op: GpuUnaryOp) -> &'static str {
    match op {
        GpuUnaryOp::Atan => "unary_special_atan_f16_kernel",
        GpuUnaryOp::Ceil => "unary_special_ceil_f16_kernel",
        GpuUnaryOp::Floor => "unary_special_floor_f16_kernel",
        GpuUnaryOp::Round => "unary_special_round_f16_kernel",
        GpuUnaryOp::Trunc => "unary_special_trunc_f16_kernel",
        GpuUnaryOp::Frac => "unary_special_frac_f16_kernel",
        GpuUnaryOp::Sign => "unary_special_sign_f16_kernel",
    }
}

#[cfg(feature = "cuda")]
const ATAN_F32_BODY: &str = r"
    setp.nan.f32 %is_nan, %x, %x;
    @%is_nan bra STORE_NAN;
    mov.b32 %bits, %x;
    and.b32 %signbits, %bits, 0x80000000;
    setp.ne.u32 %is_neg, %signbits, 0;
    abs.f32 %abs, %x;
    setp.eq.f32 %is_inf, %abs, 0f7F800000;
    @%is_inf bra ATAN_INF;

    mov.f32 %one, 0f3F800000;
    setp.gt.f32 %use_inv, %abs, %one;
    div.rn.f32 %z, %one, %abs;
    @!%use_inv mov.f32 %z, %abs;

    setp.gt.f32 %use_shift, %z, 0f3ED413CD;
    @!%use_shift bra ATAN_POLY;
    sub.f32 %tmp, %z, %one;
    add.f32 %tmp2, %z, %one;
    div.rn.f32 %z, %tmp, %tmp2;

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
    mul.f32 %y, %poly, %z;
    @%use_shift add.f32 %y, %y, 0f3F490FDB;
    @%use_inv sub.f32 %y, 0f3FC90FDB, %y;
    @%is_neg neg.f32 %y, %y;
    bra STORE;

ATAN_INF:
    mov.f32 %y, 0f3FC90FDB;
    @%is_neg neg.f32 %y, %y;
    bra STORE;

STORE_NAN:
    mov.f32 %y, 0f7FC00000;
";

#[cfg(feature = "cuda")]
const ATAN_F64_BODY: &str = r"
    setp.nan.f64 %is_nan, %x, %x;
    @%is_nan bra STORE_NAN;
    mov.b64 %bits64, %x;
    and.b64 %signbits64, %bits64, 0x8000000000000000;
    setp.ne.u64 %is_neg64, %signbits64, 0;
    abs.f64 %abs, %x;
    setp.eq.f64 %is_inf, %abs, 0d7FF0000000000000;
    @%is_inf bra ATAN_INF;

    mov.f64 %one, 0d3FF0000000000000;
    setp.gt.f64 %use_inv, %abs, %one;
    div.rn.f64 %z, %one, %abs;
    @!%use_inv mov.f64 %z, %abs;

    setp.gt.f64 %use_shift, %z, 0d3FDA827999FCEF34;
    @!%use_shift bra ATAN_POLY;
    sub.f64 %tmp, %z, %one;
    add.f64 %tmp2, %z, %one;
    div.rn.f64 %z, %tmp, %tmp2;

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
    mul.f64 %y, %poly, %z;
    @%use_shift add.f64 %y, %y, 0d3FE921FB54442D18;
    @%use_inv sub.f64 %y, 0d3FF921FB54442D18, %y;
    @%is_neg64 neg.f64 %y, %y;
    bra STORE;

ATAN_INF:
    mov.f64 %y, 0d3FF921FB54442D18;
    @%is_neg64 neg.f64 %y, %y;
    bra STORE;

STORE_NAN:
    mov.f64 %y, 0d7FF8000000000000;
";

#[cfg(feature = "cuda")]
fn f32_body(op: GpuUnaryOp) -> &'static str {
    match op {
        GpuUnaryOp::Atan => ATAN_F32_BODY,
        GpuUnaryOp::Ceil => {
            "    setp.nan.f32 %is_nan, %x, %x;\n    @%is_nan bra STORE_NAN;\n    neg.f32 %tmp, %x;\n    cvt.rmi.f32.f32 %y, %tmp;\n    neg.f32 %y, %y;\n    bra STORE;\nSTORE_NAN:\n    mov.f32 %y, 0f7FC00000;\n"
        }
        GpuUnaryOp::Floor => {
            "    setp.nan.f32 %is_nan, %x, %x;\n    @%is_nan bra STORE_NAN;\n    cvt.rmi.f32.f32 %y, %x;\n    bra STORE;\nSTORE_NAN:\n    mov.f32 %y, 0f7FC00000;\n"
        }
        GpuUnaryOp::Round => {
            "    setp.nan.f32 %is_nan, %x, %x;\n    @%is_nan bra STORE_NAN;\n    cvt.rni.f32.f32 %y, %x;\n    bra STORE;\nSTORE_NAN:\n    mov.f32 %y, 0f7FC00000;\n"
        }
        GpuUnaryOp::Trunc => {
            "    setp.nan.f32 %is_nan, %x, %x;\n    @%is_nan bra STORE_NAN;\n    cvt.rzi.f32.f32 %y, %x;\n    bra STORE;\nSTORE_NAN:\n    mov.f32 %y, 0f7FC00000;\n"
        }
        GpuUnaryOp::Frac => {
            "    setp.nan.f32 %is_nan, %x, %x;\n    @%is_nan bra STORE_NAN;\n    cvt.rzi.f32.f32 %tmp, %x;\n    sub.f32 %y, %x, %tmp;\n    bra STORE;\nSTORE_NAN:\n    mov.f32 %y, 0f7FC00000;\n"
        }
        GpuUnaryOp::Sign => {
            "    mov.f32 %y, 0f00000000;\n    setp.gt.f32 %is_pos, %x, 0f00000000;\n    @%is_pos mov.f32 %y, 0f3F800000;\n    setp.lt.f32 %is_neg, %x, 0f00000000;\n    @%is_neg mov.f32 %y, 0fBF800000;\n"
        }
    }
}

#[cfg(feature = "cuda")]
fn f64_body(op: GpuUnaryOp) -> &'static str {
    match op {
        GpuUnaryOp::Atan => ATAN_F64_BODY,
        GpuUnaryOp::Ceil => {
            "    setp.nan.f64 %is_nan, %x, %x;\n    @%is_nan bra STORE_NAN;\n    neg.f64 %tmp, %x;\n    cvt.rmi.f64.f64 %y, %tmp;\n    neg.f64 %y, %y;\n    bra STORE;\nSTORE_NAN:\n    mov.f64 %y, 0d7FF8000000000000;\n"
        }
        GpuUnaryOp::Floor => {
            "    setp.nan.f64 %is_nan, %x, %x;\n    @%is_nan bra STORE_NAN;\n    cvt.rmi.f64.f64 %y, %x;\n    bra STORE;\nSTORE_NAN:\n    mov.f64 %y, 0d7FF8000000000000;\n"
        }
        GpuUnaryOp::Round => {
            "    setp.nan.f64 %is_nan, %x, %x;\n    @%is_nan bra STORE_NAN;\n    cvt.rni.f64.f64 %y, %x;\n    bra STORE;\nSTORE_NAN:\n    mov.f64 %y, 0d7FF8000000000000;\n"
        }
        GpuUnaryOp::Trunc => {
            "    setp.nan.f64 %is_nan, %x, %x;\n    @%is_nan bra STORE_NAN;\n    cvt.rzi.f64.f64 %y, %x;\n    bra STORE;\nSTORE_NAN:\n    mov.f64 %y, 0d7FF8000000000000;\n"
        }
        GpuUnaryOp::Frac => {
            "    setp.nan.f64 %is_nan, %x, %x;\n    @%is_nan bra STORE_NAN;\n    cvt.rzi.f64.f64 %tmp, %x;\n    sub.f64 %y, %x, %tmp;\n    bra STORE;\nSTORE_NAN:\n    mov.f64 %y, 0d7FF8000000000000;\n"
        }
        GpuUnaryOp::Sign => {
            "    mov.f64 %y, 0d0000000000000000;\n    setp.gt.f64 %is_pos, %x, 0d0000000000000000;\n    @%is_pos mov.f64 %y, 0d3FF0000000000000;\n    setp.lt.f64 %is_neg, %x, 0d0000000000000000;\n    @%is_neg mov.f64 %y, 0dBFF0000000000000;\n"
        }
    }
}

#[cfg(feature = "cuda")]
fn f32_ptx(kernel_name: &str, body: &str) -> String {
    format!(
        r".version 7.0
.target sm_52
.address_size 64

.visible .entry {kernel_name}(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {{
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %bits, %signbits;
    .reg .u64 %a, %out, %off;
    .reg .f32 %x, %y, %abs, %tmp, %tmp2, %z, %z2, %poly, %one;
    .reg .pred %p, %is_nan, %is_inf, %is_neg, %is_pos, %use_inv, %use_shift;

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
    shl.b64 %off, %off, 2;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;
    ld.global.f32 %x, [%a];
{body}
STORE:
    st.global.f32 [%out], %y;

DONE:
    ret;
}}
"
    )
}

#[cfg(feature = "cuda")]
fn f64_ptx(kernel_name: &str, body: &str) -> String {
    format!(
        r".version 7.0
.target sm_52
.address_size 64

.visible .entry {kernel_name}(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {{
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off, %bits64, %signbits64;
    .reg .f64 %x, %y, %abs, %tmp, %tmp2, %z, %z2, %poly, %one;
    .reg .pred %p, %is_nan, %is_inf, %is_neg, %is_neg64, %is_pos, %use_inv, %use_shift;

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
    shl.b64 %off, %off, 3;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;
    ld.global.f64 %x, [%a];
{body}
STORE:
    st.global.f64 [%out], %y;

DONE:
    ret;
}}
"
    )
}

#[cfg(feature = "cuda")]
fn bf16_ptx(kernel_name: &str, body: &str) -> String {
    format!(
        r".version 7.0
.target sm_52
.address_size 64

.visible .entry {kernel_name}(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {{
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %bits, %signbits, %x_u32, %round, %lsb;
    .reg .u64 %a, %out, %off;
    .reg .b16 %x_b16, %zero16;
    .reg .f32 %x, %y, %abs, %tmp, %tmp2, %z, %z2, %poly, %one;
    .reg .pred %p, %is_nan, %is_inf, %is_neg, %is_pos, %use_inv, %use_shift;

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
    ld.global.b16 %x_b16, [%a];
    mov.b16 %zero16, 0;
    mov.b32 %x_u32, {{%zero16, %x_b16}};
    mov.b32 %x, %x_u32;
{body}
STORE:
    mov.b32 %bits, %y;
    shr.u32 %lsb, %bits, 16;
    and.b32 %lsb, %lsb, 1;
    add.u32 %round, %bits, 0x7FFF;
    add.u32 %round, %round, %lsb;
    shr.u32 %bits, %round, 16;
    st.global.u16 [%out], %bits;

DONE:
    ret;
}}
"
    )
}

#[cfg(feature = "cuda")]
fn f16_ptx(kernel_name: &str, body: &str) -> String {
    format!(
        r".version 7.0
.target sm_53
.address_size 64

.visible .entry {kernel_name}(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {{
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %bits, %signbits;
    .reg .u64 %a, %out, %off;
    .reg .b16 %x_h, %out_h;
    .reg .f32 %x, %y, %abs, %tmp, %tmp2, %z, %z2, %poly, %one;
    .reg .pred %p, %is_nan, %is_inf, %is_neg, %is_pos, %use_inv, %use_shift;

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
    ld.global.b16 %x_h, [%a];
    cvt.f32.f16 %x, %x_h;
{body}
STORE:
    cvt.rn.f16.f32 %out_h, %y;
    st.global.b16 [%out], %out_h;

DONE:
    ret;
}}
"
    )
}

#[cfg(feature = "cuda")]
fn validate_u16_len(n: usize, op: &'static str) -> GpuResult<()> {
    if n > u32::MAX as usize {
        return Err(GpuError::InvalidState {
            message: format!("{op}: input length {n} exceeds PTX u32 ABI"),
        });
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn launch_u16_special(
    a: &cudarc::driver::CudaSlice<u16>,
    op: GpuUnaryOp,
    device: &GpuDevice,
    kernel_name: &'static str,
    ptx: String,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    let n = a.len();
    validate_u16_len(n, op.suffix())?;
    let f = crate::module_cache::get_or_compile_owned(
        device.context(),
        ptx,
        kernel_name.to_string(),
        device.ordinal() as u32,
    )
    .map_err(|source| GpuError::PtxCompileFailed {
        kernel: kernel_name,
        source,
    })?;

    let mut out = device.stream().alloc_zeros::<u16>(n)?;
    if n == 0 {
        return Ok(out);
    }
    let n_u32 = n as u32;
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(a)
            .arg(&mut out)
            .arg(&n_u32)
            .launch(LaunchConfig::for_num_elems(n_u32))?;
    }
    Ok(out)
}

#[cfg(feature = "cuda")]
/// Launch a CUDA-resident f32 unary special kernel.
pub fn gpu_unary_special_f32(
    a: &CudaBuffer<f32>,
    op: GpuUnaryOp,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    validate_device(a, device, op.suffix())?;
    let kernel_name = f32_kernel_name(op);
    let ptx = f32_ptx(kernel_name, f32_body(op));
    let f = crate::module_cache::get_or_compile_owned(
        device.context(),
        ptx,
        kernel_name.to_string(),
        device.ordinal() as u32,
    )
    .map_err(|source| GpuError::PtxCompileFailed {
        kernel: kernel_name,
        source,
    })?;

    let n = a.len();
    let mut out = alloc_zeros_f32(n, device)?;
    if n == 0 {
        return Ok(out);
    }
    let n_u32 = n as u32;
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(a.inner())
            .arg(out.inner_mut())
            .arg(&n_u32)
            .launch(LaunchConfig::for_num_elems(n_u32))?;
    }
    Ok(out)
}

#[cfg(feature = "cuda")]
/// Launch a CUDA-resident bf16 unary special kernel with f32 opmath and bf16
/// round-to-nearest-even storage, matching PyTorch CUDA reduced-precision
/// unary kernels.
pub fn gpu_unary_special_bf16(
    a: &cudarc::driver::CudaSlice<u16>,
    op: GpuUnaryOp,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    let kernel_name = bf16_kernel_name(op);
    launch_u16_special(
        a,
        op,
        device,
        kernel_name,
        bf16_ptx(kernel_name, f32_body(op)),
    )
}

#[cfg(feature = "cuda")]
/// Launch a CUDA-resident f16 unary special kernel with f32 opmath and f16
/// round-to-nearest-even storage, matching PyTorch CUDA reduced-precision
/// unary kernels.
pub fn gpu_unary_special_f16(
    a: &cudarc::driver::CudaSlice<u16>,
    op: GpuUnaryOp,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    let kernel_name = f16_kernel_name(op);
    launch_u16_special(
        a,
        op,
        device,
        kernel_name,
        f16_ptx(kernel_name, f32_body(op)),
    )
}

#[cfg(feature = "cuda")]
/// Launch a CUDA-resident f64 unary special kernel.
pub fn gpu_unary_special_f64(
    a: &CudaBuffer<f64>,
    op: GpuUnaryOp,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    validate_device(a, device, op.suffix())?;
    let kernel_name = f64_kernel_name(op);
    let ptx = f64_ptx(kernel_name, f64_body(op));
    let f = crate::module_cache::get_or_compile_owned(
        device.context(),
        ptx,
        kernel_name.to_string(),
        device.ordinal() as u32,
    )
    .map_err(|source| GpuError::PtxCompileFailed {
        kernel: kernel_name,
        source,
    })?;

    let n = a.len();
    let mut out = alloc_zeros_f64(n, device)?;
    if n == 0 {
        return Ok(out);
    }
    let n_u32 = n as u32;
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(a.inner())
            .arg(out.inner_mut())
            .arg(&n_u32)
            .launch(LaunchConfig::for_num_elems(n_u32))?;
    }
    Ok(out)
}

/// Stub for builds without CUDA support.
#[cfg(not(feature = "cuda"))]
pub fn gpu_unary_special_f32(
    _a: &CudaBuffer<f32>,
    op: GpuUnaryOp,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::Unsupported {
        op: op.suffix(),
        dtype: "f32",
    })
}

/// Stub for builds without CUDA support.
#[cfg(not(feature = "cuda"))]
pub fn gpu_unary_special_f64(
    _a: &CudaBuffer<f64>,
    op: GpuUnaryOp,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::Unsupported {
        op: op.suffix(),
        dtype: "f64",
    })
}
