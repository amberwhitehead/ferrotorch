//! CUDA-resident frequency factories for `torch.fft.fftfreq` and
//! `torch.fft.rfftfreq` parity.
//!
//! These kernels implement the same algorithm as PyTorch's
//! `aten/src/ATen/native/SpectralOps.cpp`: materialize integer frequency bins
//! and multiply by `1.0 / (n * d)`. The core crate routes CUDA factory calls
//! here so GPU outputs are allocated and filled on the requested device rather
//! than generated on CPU and uploaded afterward.

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

#[cfg(feature = "cuda")]
use ferrotorch_core::dtype::DType;

#[cfg(feature = "cuda")]
use crate::buffer::CudaBuffer;
#[cfg(feature = "cuda")]
use crate::device::GpuDevice;
#[cfg(feature = "cuda")]
use crate::error::{GpuError, GpuResult};
#[cfg(feature = "cuda")]
use crate::transfer::{alloc_zeros_f32, alloc_zeros_f64};

#[cfg(feature = "cuda")]
const FREQ_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry frequency_f32_kernel(
    .param .u64 out_ptr,
    .param .u32 n,
    .param .u32 len,
    .param .u32 negative_start,
    .param .f32 scale
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %len_reg, %neg_start, %nan_bits;
    .reg .u64 %out, %off;
    .reg .f32 %scale_reg, %bin, %n_f, %val;
    .reg .pred %done, %right, %nanp;

    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];
    ld.param.u32 %len_reg, [len];
    ld.param.u32 %neg_start, [negative_start];
    ld.param.f32 %scale_reg, [scale];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %done, %r_tid, %len_reg;
    @%done bra DONE;

    cvt.rn.f32.u32 %bin, %r_tid;
    setp.ge.u32 %right, %r_tid, %neg_start;
    @!%right bra BIN_READY;
    cvt.rn.f32.u32 %n_f, %n_reg;
    sub.rn.f32 %bin, %bin, %n_f;

BIN_READY:
    mul.rn.f32 %val, %bin, %scale_reg;
    // PyTorch CUDA canonicalizes the zero-bin NaN for f32/reduced dtypes.
    setp.nan.f32 %nanp, %val, %val;
    @%nanp mov.u32 %nan_bits, 0x7fffffff;
    @%nanp mov.b32 %val, %nan_bits;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %out, %out, %off;
    st.global.f32 [%out], %val;

DONE:
    ret;
}
";

#[cfg(feature = "cuda")]
const FREQ_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry frequency_f64_kernel(
    .param .u64 out_ptr,
    .param .u32 n,
    .param .u32 len,
    .param .u32 negative_start,
    .param .f64 scale
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %len_reg, %neg_start;
    .reg .u64 %out, %off;
    .reg .f64 %scale_reg, %bin, %n_f, %val;
    .reg .pred %done, %right;

    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];
    ld.param.u32 %len_reg, [len];
    ld.param.u32 %neg_start, [negative_start];
    ld.param.f64 %scale_reg, [scale];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %done, %r_tid, %len_reg;
    @%done bra DONE;

    cvt.rn.f64.u32 %bin, %r_tid;
    setp.ge.u32 %right, %r_tid, %neg_start;
    @!%right bra BIN_READY;
    cvt.rn.f64.u32 %n_f, %n_reg;
    sub.rn.f64 %bin, %bin, %n_f;

BIN_READY:
    mul.rn.f64 %val, %bin, %scale_reg;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 3;
    add.u64 %out, %out, %off;
    st.global.f64 [%out], %val;

DONE:
    ret;
}
";

#[cfg(feature = "cuda")]
const FREQ_F16_PTX: &str = "\
.version 7.0
.target sm_53
.address_size 64

.visible .entry frequency_f16_kernel(
    .param .u64 out_ptr,
    .param .u32 n,
    .param .u32 len,
    .param .u32 negative_start,
    .param .f32 scale
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %len_reg, %neg_start;
    .reg .u64 %out, %off;
    .reg .f32 %scale_reg, %bin, %n_f, %val;
    .reg .b16 %half;
    .reg .pred %done, %right;

    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];
    ld.param.u32 %len_reg, [len];
    ld.param.u32 %neg_start, [negative_start];
    ld.param.f32 %scale_reg, [scale];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %done, %r_tid, %len_reg;
    @%done bra DONE;

    cvt.rn.f32.u32 %bin, %r_tid;
    setp.ge.u32 %right, %r_tid, %neg_start;
    @!%right bra BIN_READY;
    cvt.rn.f32.u32 %n_f, %n_reg;
    sub.rn.f32 %bin, %bin, %n_f;

BIN_READY:
    mul.rn.f32 %val, %bin, %scale_reg;
    cvt.rn.f16.f32 %half, %val;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %out, %out, %off;
    st.global.b16 [%out], %half;

DONE:
    ret;
}
";

#[cfg(feature = "cuda")]
const FREQ_BF16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry frequency_bf16_kernel(
    .param .u64 out_ptr,
    .param .u32 n,
    .param .u32 len,
    .param .u32 negative_start,
    .param .f32 scale
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %len_reg, %neg_start;
    .reg .u64 %out, %off;
    .reg .f32 %scale_reg, %bin, %n_f, %val;
    .reg .u32 %bits, %round, %lsb, %result, %nan_bits;
    .reg .pred %done, %right, %nanp;

    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];
    ld.param.u32 %len_reg, [len];
    ld.param.u32 %neg_start, [negative_start];
    ld.param.f32 %scale_reg, [scale];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %done, %r_tid, %len_reg;
    @%done bra DONE;

    cvt.rn.f32.u32 %bin, %r_tid;
    setp.ge.u32 %right, %r_tid, %neg_start;
    @!%right bra BIN_READY;
    cvt.rn.f32.u32 %n_f, %n_reg;
    sub.rn.f32 %bin, %bin, %n_f;

BIN_READY:
    mul.rn.f32 %val, %bin, %scale_reg;
    setp.nan.f32 %nanp, %val, %val;
    @%nanp mov.u32 %nan_bits, 0x7fffffff;
    @%nanp mov.b32 %val, %nan_bits;
    mov.b32 %bits, %val;
    shr.u32 %lsb, %bits, 16;
    and.b32 %lsb, %lsb, 1;
    add.u32 %round, %bits, 0x7FFF;
    add.u32 %round, %round, %lsb;
    shr.u32 %result, %round, 16;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %out, %out, %off;
    st.global.u16 [%out], %result;

DONE:
    ret;
}
";

#[cfg(feature = "cuda")]
fn launch_cfg(len: usize, op: &'static str) -> GpuResult<LaunchConfig> {
    if len > u32::MAX as usize {
        return Err(GpuError::ShapeMismatch {
            op,
            expected: vec![u32::MAX as usize],
            got: vec![len],
        });
    }
    let block_dim = 256_u32;
    let len_u32 = len as u32;
    Ok(LaunchConfig {
        grid_dim: (len_u32.div_ceil(block_dim).max(1), 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    })
}

#[cfg(feature = "cuda")]
fn dtype_label(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => "F16",
        DType::BF16 => "BF16",
        DType::F32 => "F32",
        DType::F64 => "F64",
        _ => "unsupported",
    }
}

#[cfg(feature = "cuda")]
fn frequency_layout(n: usize, rfft: bool, op: &'static str) -> GpuResult<(usize, usize)> {
    if n > u32::MAX as usize {
        return Err(GpuError::ShapeMismatch {
            op,
            expected: vec![u32::MAX as usize],
            got: vec![n],
        });
    }
    let len = if rfft { n / 2 + 1 } else { n };
    let negative_start = if rfft { len } else { n.div_ceil(2) };
    Ok((len, negative_start))
}

#[cfg(feature = "cuda")]
fn launch_frequency_f32(
    n: usize,
    d: f64,
    rfft: bool,
    device: &GpuDevice,
    op: &'static str,
) -> GpuResult<CudaBuffer<f32>> {
    let (len, negative_start) = frequency_layout(n, rfft, op)?;
    let mut out = alloc_zeros_f32(len, device)?;
    if len == 0 {
        return Ok(out);
    }

    let f = crate::module_cache::get_or_compile(
        device.context(),
        FREQ_F32_PTX,
        "frequency_f32_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "frequency_f32_kernel",
        source: e,
    })?;

    let cfg = launch_cfg(len, op)?;
    let n_u32 = n as u32;
    let len_u32 = len as u32;
    let negative_start_u32 = negative_start as u32;
    let scale = (1.0 / ((n as f64) * d)) as f32;

    // SAFETY: The PTX ABI is `(out, n, len, negative_start, scale)`.
    // `out` is freshly allocated to `len` f32 elements; every launched thread
    // either exits at `tid >= len` or writes exactly `out[tid]`. `n`, `len`,
    // and `negative_start` are all checked to fit the kernel's u32 index
    // space before the launch. `negative_start == len` for rfftfreq, so the
    // subtraction branch is disabled for the non-negative frequency factory.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(out.inner_mut())
            .arg(&n_u32)
            .arg(&len_u32)
            .arg(&negative_start_u32)
            .arg(&scale)
            .launch(cfg)?;
    }
    Ok(out)
}

#[cfg(feature = "cuda")]
fn launch_frequency_f64(
    n: usize,
    d: f64,
    rfft: bool,
    device: &GpuDevice,
    op: &'static str,
) -> GpuResult<CudaBuffer<f64>> {
    let (len, negative_start) = frequency_layout(n, rfft, op)?;
    let mut out = alloc_zeros_f64(len, device)?;
    if len == 0 {
        return Ok(out);
    }

    let f = crate::module_cache::get_or_compile(
        device.context(),
        FREQ_F64_PTX,
        "frequency_f64_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "frequency_f64_kernel",
        source: e,
    })?;

    let cfg = launch_cfg(len, op)?;
    let n_u32 = n as u32;
    let len_u32 = len as u32;
    let negative_start_u32 = negative_start as u32;
    let scale = 1.0 / ((n as f64) * d);

    // SAFETY: Same proof as `launch_frequency_f32`, with an 8-byte f64 output
    // stride encoded in `FREQ_F64_PTX`.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(out.inner_mut())
            .arg(&n_u32)
            .arg(&len_u32)
            .arg(&negative_start_u32)
            .arg(&scale)
            .launch(cfg)?;
    }
    Ok(out)
}

#[cfg(feature = "cuda")]
fn launch_frequency_u16(
    n: usize,
    d: f64,
    rfft: bool,
    dtype: DType,
    device: &GpuDevice,
    op: &'static str,
) -> GpuResult<CudaSlice<u16>> {
    let (len, negative_start) = frequency_layout(n, rfft, op)?;
    let mut out = device.stream().alloc_zeros::<u16>(len)?;
    if len == 0 {
        return Ok(out);
    }

    let (ptx, kernel) = match dtype {
        DType::F16 => (FREQ_F16_PTX, "frequency_f16_kernel"),
        DType::BF16 => (FREQ_BF16_PTX, "frequency_bf16_kernel"),
        _ => {
            return Err(GpuError::Unsupported {
                op,
                dtype: dtype_label(dtype),
            });
        }
    };
    let f =
        crate::module_cache::get_or_compile(device.context(), ptx, kernel, device.ordinal() as u32)
            .map_err(|e| GpuError::PtxCompileFailed { kernel, source: e })?;

    let cfg = launch_cfg(len, op)?;
    let n_u32 = n as u32;
    let len_u32 = len as u32;
    let negative_start_u32 = negative_start as u32;
    let scale = (1.0 / ((n as f64) * d)) as f32;

    // SAFETY: Same index proof as `launch_frequency_f32`. The destination is
    // a fresh `[len]` u16 slice storing one f16/bf16 bit pattern per logical
    // element; the caller wraps it with the authoritative dtype tag.
    unsafe {
        device
            .stream()
            .launch_builder(&f)
            .arg(&mut out)
            .arg(&n_u32)
            .arg(&len_u32)
            .arg(&negative_start_u32)
            .arg(&scale)
            .launch(cfg)?;
    }
    Ok(out)
}

/// CUDA-resident `torch.fft.fftfreq` values for f32.
#[cfg(feature = "cuda")]
pub fn gpu_fftfreq_f32(n: usize, d: f64, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    launch_frequency_f32(n, d, false, device, "fftfreq")
}

/// CUDA-resident `torch.fft.rfftfreq` values for f32.
#[cfg(feature = "cuda")]
pub fn gpu_rfftfreq_f32(n: usize, d: f64, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    launch_frequency_f32(n, d, true, device, "rfftfreq")
}

/// CUDA-resident `torch.fft.fftfreq` values for f64.
#[cfg(feature = "cuda")]
pub fn gpu_fftfreq_f64(n: usize, d: f64, device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    launch_frequency_f64(n, d, false, device, "fftfreq")
}

/// CUDA-resident `torch.fft.rfftfreq` values for f64.
#[cfg(feature = "cuda")]
pub fn gpu_rfftfreq_f64(n: usize, d: f64, device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    launch_frequency_f64(n, d, true, device, "rfftfreq")
}

/// CUDA-resident `torch.fft.fftfreq` values for f16.
#[cfg(feature = "cuda")]
pub fn gpu_fftfreq_f16(n: usize, d: f64, device: &GpuDevice) -> GpuResult<CudaSlice<u16>> {
    launch_frequency_u16(n, d, false, DType::F16, device, "fftfreq")
}

/// CUDA-resident `torch.fft.rfftfreq` values for f16.
#[cfg(feature = "cuda")]
pub fn gpu_rfftfreq_f16(n: usize, d: f64, device: &GpuDevice) -> GpuResult<CudaSlice<u16>> {
    launch_frequency_u16(n, d, true, DType::F16, device, "rfftfreq")
}

/// CUDA-resident `torch.fft.fftfreq` values for bf16.
#[cfg(feature = "cuda")]
pub fn gpu_fftfreq_bf16(n: usize, d: f64, device: &GpuDevice) -> GpuResult<CudaSlice<u16>> {
    launch_frequency_u16(n, d, false, DType::BF16, device, "fftfreq")
}

/// CUDA-resident `torch.fft.rfftfreq` values for bf16.
#[cfg(feature = "cuda")]
pub fn gpu_rfftfreq_bf16(n: usize, d: f64, device: &GpuDevice) -> GpuResult<CudaSlice<u16>> {
    launch_frequency_u16(n, d, true, DType::BF16, device, "rfftfreq")
}
