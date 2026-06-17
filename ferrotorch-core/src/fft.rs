//! FFT operations for tensors.
//!
//! Complex values are represented as an extra trailing dimension of size 2,
//! where `[..., 0]` is the real part and `[..., 1]` is the imaginary part.
//! This matches PyTorch's convention for `torch.fft.*` operations.
//!
//! The CPU transform path accepts **f32 and f64** only, matching
//! `torch.fft.*`'s dtype contract: PyTorch's `promote_type_fft`
//! (`aten/src/ATen/native/SpectralOps.cpp:82-90`) does
//! `TORCH_CHECK(type == kFloat || type == kDouble, "Unsupported dtype ", type)`
//! on non-CUDA devices, so `torch.fft.fft(x.half())` / `.bfloat16()` on CPU
//! raise `RuntimeError: Unsupported dtype` (verified live against torch 2.11).
//! `half` FFT is supported *only* on CUDA, where it runs as a native
//! `complex_half` transform (`torch/fft/__init__.py:49`), NOT by upcasting to
//! f32. ferrotorch's CPU transforms therefore reject `f16`/`bf16` via
//! [`reject_half_cpu_fft`] rather than silently upcasting (#1545 / #1536); the
//! non-transform helpers (`fftshift`/`ifftshift`) stay dtype-permissive because
//! `torch.fft.fftshift` accepts `half`/`bfloat16` (a pure roll, verified live).
//! For the accepted f32/f64 dtypes the CPU path runs the transform in double
//! precision through [`ferray_fft`] (which carries numpy's direction-dependent
//! `norm` scaling and arbitrary-axis transforms) and casts the result back to
//! the input dtype. Every transform accepts `norm` ([`FftNorm`],
//! `backward`/`forward`/`ortho` — matching `torch.fft.*`'s `norm` kwarg) and
//! `dim` / `s` via the `*_norm` sibling of each public fn (#1294). The
//! historical `fft(input, n)` / `fft2(input)` / `fftn(input, s, axes)`
//! signatures remain as thin wrappers (default `dim=last`, `norm=Backward`)
//! so existing consumers (`complex_tensor.rs`, the differentiable wrappers)
//! compile unchanged.
//!
//! # GPU note
//!
//! cuFFT fast paths exist for f32/f64 on the last-axis complex and
//! real/Hermitian 1-D transforms (#579 / #605 / #634 / #636). They honor
//! PyTorch's `norm` modes with on-device post-scaling and perform last-axis
//! resize staging on device. Non-last-axis `dim` and multi-axis `s` still use
//! the ferray_fft CPU path. bf16/f16 GPU lowering is tracked under #1545. GPU
//! tensors never silently bounce through host memory on the cuFFT-capable
//! paths.
//!
//! ## REQ status (per `.design/ferrotorch-core/fft.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `fft` / `fft_norm`; consumer: `ComplexTensor::fft` (`complex_tensor.rs`) |
//! | REQ-2 | SHIPPED | `ifft` / `ifft_norm`; consumer: `ComplexTensor::ifft` |
//! | REQ-3 | SHIPPED | `rfft`/`irfft` (+ `*_norm`); consumer: `grad_fns::fft::rfft_differentiable` |
//! | REQ-4 | SHIPPED | `fft2`/`ifft2` (+ `*_norm`); consumer: `ComplexTensor::fft2`/`ifft2` |
//! | REQ-5 | SHIPPED | `fftn`/`ifftn`/`rfftn`/`irfftn` (+ `*_norm`); consumer: re-export in `lib.rs` + `grad_fns::fft::*_differentiable` |
//! | REQ-6 | SHIPPED | `hfft`/`ihfft` (+ `*_norm`); consumer: re-export in `lib.rs` |
//! | REQ-7 | SHIPPED | `fftfreq`/`rfftfreq`; consumer: re-export in `lib.rs` |
//! | REQ-8 | SHIPPED | `fftshift`/`ifftshift`; consumer: re-export in `lib.rs` |
//! | REQ-9 | SHIPPED | cuFFT dispatch in `fft_norm`/`ifft_norm` (default norm/dim); consumer: `ComplexTensor::fft`. CPU `f16`/`bf16` are rejected by `reject_half_cpu_fft` to match torch's `Unsupported dtype` error (`SpectralOps.cpp:88-90`, #1545/#1536); native CUDA complex-half lowering remains tracked under #1545 |

use ferray_core::Array as FerrayArray;
use ferray_core::IxDyn as FerrayIxDyn;
pub use ferray_fft::FftNorm;
use rustfft::num_complex::Complex;

use crate::dtype::{DType, Element, Float};
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// norm / dim conventions (#1294)
// ---------------------------------------------------------------------------
//
// `torch.fft.*` takes `norm` ∈ {None|"backward","forward","ortho"} and
// `dim`/`s` for the transform axes. The norm string maps 1:1 onto
// [`FftNorm`] (re-exported from `ferray_fft`): ferray_fft's `FftNorm` carries
// numpy's direction-dependent normalization semantics
// (`FftNorm::scale_factor`), which reproduce torch's `norm_from_string` +
// `fft_norm_mode` scaling byte-for-byte:
//   - "backward" → `FftNorm::Backward`: no scale on forward, `1/n` on inverse
//     (torch `norm_from_string("backward", forward) = none|by_n`, upstream
//     `aten/src/ATen/native/SpectralOps.cpp:116-119`).
//   - "forward"  → `FftNorm::Forward`:  `1/n` on forward, no scale on inverse
//     (upstream `:121-123`).
//   - "ortho"    → `FftNorm::Ortho`:    `1/sqrt(n)` both directions
//     (upstream `:125-127`).
//
// The `dim` axis refers to the *real signal layout* (torch's input shape).
// ferrotorch carries complex tensors as an interleaved `[..., 2]` real
// tensor; the `tensor_to_complex_array` bridge strips the trailing `2`, so
// the ferray array's axes match the real signal layout 1:1 and a torch `dim`
// (resolved against the real ndim) is passed straight through to ferray_fft.

/// Resolve the torch FFT `norm` string into a [`FftNorm`]. `None`/`"backward"`
/// is the default. Returns an `InvalidArgument` error for unknown modes,
/// mirroring upstream `norm_from_string`'s `TORCH_CHECK(false, "Invalid
/// normalization mode")` (`SpectralOps.cpp:129`).
pub fn fft_norm_from_str(norm: Option<&str>, op: &'static str) -> FerrotorchResult<FftNorm> {
    match norm {
        None | Some("backward") => Ok(FftNorm::Backward),
        Some("forward") => Ok(FftNorm::Forward),
        Some("ortho") => Ok(FftNorm::Ortho),
        Some(other) => Err(FerrotorchError::InvalidArgument {
            message: format!("{op}: invalid normalization mode: \"{other}\""),
        }),
    }
}

/// True when `T` is f32 (4-byte float), used to pick the f32 vs f64 GPU path.
#[inline]
fn is_f32<T: Float>() -> bool {
    std::mem::size_of::<T>() == 4
}

/// True when `T` is f64 (8-byte float).
#[inline]
fn is_f64<T: Float>() -> bool {
    std::mem::size_of::<T>() == 8
}

/// Reject half-precision (`f16` / `bf16`) inputs on the CPU transform path,
/// mirroring PyTorch's `promote_type_fft`
/// (`aten/src/ATen/native/SpectralOps.cpp:82-91`): on non-CUDA/XPU/meta
/// devices the FFT dtype check is
/// `TORCH_CHECK(type == kFloat || type == kDouble, "Unsupported dtype ", type)`
/// (`SpectralOps.cpp:90`), so `torch.fft.*` of a `half`/`bfloat16` tensor on
/// CPU raises `RuntimeError: Unsupported dtype Half|BFloat16` (verified live
/// against torch 2.11). Half FFT is supported *only* on CUDA, where it runs
/// as a native `complex_half` transform (`torch/fft/__init__.py:49` —
/// "Supports torch.half and torch.chalf on CUDA"), NOT by upcasting to f32.
///
/// `bf16` is rejected on every device upstream (`kBFloat16` is absent from the
/// accepted set at `SpectralOps.cpp:88` even when `maybe_support_half`).
///
/// This guard is applied to the spectral *transforms* only. `fftshift` /
/// `ifftshift` are pure axis rolls (not transforms) and `torch.fft.fftshift`
/// accepts `half`/`bfloat16` returning the same dtype (verified live), so they
/// deliberately do not call this guard.
#[inline]
fn reject_half_cpu_fft<T: Float>(op: &'static str) -> FerrotorchResult<()> {
    match <T as Element>::dtype() {
        DType::F16 | DType::BF16 => Err(FerrotorchError::InvalidArgument {
            message: format!(
                "{op}: Unsupported dtype {:?} — torch.fft.* does not support \
                 half/bfloat16 on CPU (half is CUDA-only as a native complex-half \
                 transform; see SpectralOps.cpp:88-90)",
                <T as Element>::dtype(),
            ),
        }),
        _ => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// 1-D complex-to-complex FFT along the last dimension (default `norm`).
///
/// The input tensor must have a trailing dimension of size 2 representing
/// complex numbers `[re, im]`. If `n` is provided, the signal is truncated
/// or zero-padded along the second-to-last dimension before transforming.
///
/// Returns a tensor with shape `[..., n, 2]` (or `[..., input_len, 2]` if
/// `n` is `None`). Thin wrapper over [`fft_norm`] with `dim=-1` (last signal
/// axis) and `norm=Backward`, preserving the historical `(input, n)`
/// signature used by `complex_tensor.rs` and the differentiable wrappers.
pub fn fft<T: Float>(input: &Tensor<T>, n: Option<usize>) -> FerrotorchResult<Tensor<T>> {
    fft_norm(input, n, None, FftNorm::Backward)
}

/// 1-D complex-to-complex FFT with explicit `dim` and `norm` (#1294).
///
/// `dim` is the transform axis in the *real signal* layout (the input minus
/// its trailing complex pair); `None` defaults to the last signal axis
/// (`torch.fft.fft`'s `dim=-1`). `norm` selects the normalization mode
/// (`torch.fft.fft`'s `norm` kwarg). Matches `torch.fft.fft(input, n, dim,
/// norm)` (`torch/fft/__init__.py:36`).
pub fn fft_norm<T: Float>(
    input: &Tensor<T>,
    n: Option<usize>,
    dim: Option<isize>,
    norm: FftNorm,
) -> FerrotorchResult<Tensor<T>> {
    reject_half_cpu_fft::<T>("fft")?;
    let shape = input.shape();

    // Input must end with a dim of 2 (complex representation).
    if shape.is_empty() || *shape.last().unwrap() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "fft: input must have trailing dimension 2 (complex), got shape {shape:?}"
            ),
        });
    }

    let ndim = shape.len();
    // Signal length is the second-to-last dim.
    if ndim < 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: "fft: input must have at least 2 dimensions ([..., n, 2])".into(),
        });
    }

    // GPU fast path: last-axis C2C via cuFFT. Resize and normalization are
    // staged on device so CUDA inputs do not fall through to the CPU bridge.
    if input.is_cuda() && (is_f32::<T>() || is_f64::<T>()) && is_last_signal_axis(dim, ndim - 1) {
        let input_n = shape[ndim - 2];
        let fft_n = n.unwrap_or(input_n);
        if fft_n == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "fft: n must be > 0".into(),
            });
        }
        let batch_shape = &shape[..ndim - 2];
        let batch_size: usize = crate::shape::numel(batch_shape).max(1);
        // GPU C2C dispatch via cuFFT (#579), with on-device pad/truncate
        // when `fft_n != input_n` (#605). Fully on-device — no host bounce.
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let input_contiguous = crate::autograd::no_grad::no_grad(|| input.contiguous())?;
        let buf = input_contiguous.gpu_handle()?;

        // Optional pad/truncate to fft_n.
        let (transformed_handle, owned);
        let buf_for_fft: &crate::gpu_dispatch::GpuBufferHandle = if fft_n == input_n {
            buf
        } else if is_f32::<T>() {
            owned = backend.pad_truncate_complex_f32(buf, batch_size, input_n, fft_n)?;
            transformed_handle = &owned;
            transformed_handle
        } else {
            owned = backend.pad_truncate_complex_f64(buf, batch_size, input_n, fft_n)?;
            transformed_handle = &owned;
            transformed_handle
        };

        let h = if is_f32::<T>() {
            backend.fft_c2c_f32(buf_for_fft, batch_size, fft_n, false)?
        } else {
            backend.fft_c2c_f64(buf_for_fft, batch_size, fft_n, false)?
        };
        let h = scale_cuda_fft_output::<T>(backend, h, cuda_forward_norm_scale(norm, fft_n))?;
        let mut out_shape = batch_shape.to_vec();
        out_shape.push(fft_n);
        out_shape.push(2);
        return Tensor::from_storage(TensorStorage::gpu(h), out_shape, false);
    }

    // CPU path: thread `n` / `dim` / `norm` through ferray_fft, which carries
    // numpy's direction-dependent norm scaling and arbitrary-axis transforms.
    let arr = tensor_to_complex_array(input, "fft")?;
    let result =
        ferray_fft::fft(&arr, n, dim, norm).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("fft: {e}"),
        })?;
    complex_array_to_tensor(&result)
}

/// True when `dim` (an optional torch FFT axis, in the *real signal* layout
/// of `signal_ndim` dimensions) names the last signal axis (`None`, `-1`, or
/// `signal_ndim - 1`).
#[inline]
fn is_last_signal_axis(dim: Option<isize>, signal_ndim: usize) -> bool {
    match dim {
        None => true,
        Some(d) => {
            let resolved = if d < 0 { signal_ndim as isize + d } else { d };
            resolved == signal_ndim as isize - 1
        }
    }
}

#[inline]
fn cuda_forward_norm_scale(norm: FftNorm, n: usize) -> f64 {
    match norm {
        FftNorm::Backward => 1.0,
        FftNorm::Forward => 1.0 / n as f64,
        FftNorm::Ortho => 1.0 / (n as f64).sqrt(),
    }
}

#[inline]
fn cuda_inverse_norm_scale(norm: FftNorm, n: usize) -> f64 {
    match norm {
        // The cuFFT inverse wrappers already apply PyTorch's backward-mode
        // inverse scale (1 / n). Forward/ortho are recovered by multiplying
        // the resident result, matching SpectralOps.cpp norm_from_string.
        FftNorm::Backward => 1.0,
        FftNorm::Forward => n as f64,
        FftNorm::Ortho => (n as f64).sqrt(),
    }
}

fn scale_cuda_fft_output<T: Float>(
    backend: &dyn crate::gpu_dispatch::GpuBackend,
    handle: crate::gpu_dispatch::GpuBufferHandle,
    scale: f64,
) -> FerrotorchResult<crate::gpu_dispatch::GpuBufferHandle> {
    if scale.to_bits() == 1.0f64.to_bits() {
        return Ok(handle);
    }
    if is_f32::<T>() {
        backend.scale_f32(&handle, scale as f32)
    } else if is_f64::<T>() {
        backend.scale_f64(&handle, scale)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "CUDA FFT scaling requires f32 or f64".into(),
        })
    }
}

fn resize_last_axis_real_cuda<T: Float>(
    input: &Tensor<T>,
    target_n: usize,
) -> FerrotorchResult<Tensor<T>> {
    let ndim = input.shape().len();
    let input_n = input.shape()[ndim - 1];
    if input_n == target_n {
        return input.contiguous();
    }
    if input_n > target_n {
        return input.narrow(ndim - 1, 0, target_n)?.contiguous();
    }

    let mut pad_shape = input.shape().to_vec();
    pad_shape[ndim - 1] = target_n - input_n;
    let zeros = crate::creation::full_on_device(
        &pad_shape,
        <T as num_traits::Zero>::zero(),
        input.device(),
        "rfft resize padding",
    )?;
    crate::grad_fns::shape::cat(&[input.clone(), zeros], (ndim - 1) as isize)?.contiguous()
}

/// Output-length validation + empty-frequency-axis short circuit for the
/// complex-to-real entry points ([`irfft_norm`] / [`hfft_norm`]) — CORE-155 /
/// #1849.
///
/// Mirrors upstream `fft_c2r` (`aten/src/ATen/native/SpectralOps.cpp:207-208`):
///
/// ```cpp
/// const auto n = n_opt.value_or(2*(input.sym_sizes()[dim] - 1));
/// TORCH_CHECK(n >= 1, "Invalid number of data points (", n, ") specified");
/// ```
///
/// The default output length is computed **lazily** and in **signed**
/// arithmetic, so a zero-length frequency axis with `n=None` reports torch's
/// `Invalid number of data points (-2) specified` instead of underflowing
/// `usize`, and an explicit `n` never evaluates the underflowing expression.
/// With a valid explicit `n`, torch zero-pads the empty spectrum
/// (`resize_fft_input`, `SpectralOps.cpp:209-211`) and the all-zero Hermitian
/// spectrum inverts to all zeros (verified live, torch 2.11:
/// `torch.fft.irfft(torch.zeros(0, dtype=torch.complex64), n=8)` is eight
/// zeros) — the wrapper returns those zeros directly, because delegating
/// would re-trigger the same eager `2 * (len - 1)` underflow duplicated
/// inside ferray-fft (`ferray-fft-0.4.1/src/real.rs:139`,
/// `src/hermitian.rs:72`).
///
/// Returns `Ok(Some(result))` when the empty frequency axis fully determines
/// the result, `Ok(None)` to proceed with the normal transform paths. Inputs
/// whose shape is not a well-formed complex layout (`ndim < 2` or no trailing
/// pair axis) and out-of-range `dim`s fall through unchanged so the entry
/// points / ferray bridge keep reporting their existing structured errors.
fn c2r_guard_empty_axis<T: Float>(
    input: &Tensor<T>,
    n: Option<usize>,
    dim: Option<isize>,
    op: &'static str,
) -> FerrotorchResult<Option<Tensor<T>>> {
    let shape = input.shape();
    let ndim = shape.len();
    if ndim < 2 || shape.last() != Some(&2) {
        return Ok(None);
    }
    let signal_ndim = ndim - 1;
    // Resolve `dim` against the signal layout, mirroring `maybe_wrap_dim`
    // (`SpectralOps.cpp:206`).
    let axis = match dim {
        None => signal_ndim - 1,
        Some(d) if d >= 0 && (d as usize) < signal_ndim => d as usize,
        Some(d) if d < 0 && d >= -(signal_ndim as isize) => (d + signal_ndim as isize) as usize,
        Some(_) => return Ok(None),
    };
    let m = shape[axis];
    let n_eff: i128 = match n {
        Some(v) => v as i128,
        None => 2 * (m as i128) - 2,
    };
    if n_eff < 1 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("{op}: Invalid number of data points ({n_eff}) specified"),
        });
    }
    if m == 0 {
        let mut out_shape: Vec<usize> = shape[..signal_ndim].to_vec();
        out_shape[axis] = n_eff as usize;
        let zeros = crate::creation::zeros::<T>(&out_shape)?;
        if input.is_cuda() {
            // The result is fully determined (all zeros) but must live on
            // the input's device; this is a fresh H2D upload of the freshly
            // created tensor, not a device round trip of input data.
            return zeros.to(input.device()).map(Some);
        }
        return Ok(Some(zeros));
    }
    Ok(None)
}

/// 1-D inverse FFT along the last dimension (default `norm`).
///
/// Input has shape `[..., n, 2]` (complex). Returns complex output of the
/// same shape (or `[..., n_out, 2]` if `n` is specified). Thin wrapper over
/// [`ifft_norm`] with `dim=-1`, `norm=Backward`.
pub fn ifft<T: Float>(input: &Tensor<T>, n: Option<usize>) -> FerrotorchResult<Tensor<T>> {
    ifft_norm(input, n, None, FftNorm::Backward)
}

/// 1-D inverse FFT with explicit `dim` and `norm` (#1294).
///
/// Matches `torch.fft.ifft(input, n, dim, norm)`
/// (`torch/fft/__init__.py:91`). See [`fft_norm`] for the `dim`/`norm`
/// conventions.
pub fn ifft_norm<T: Float>(
    input: &Tensor<T>,
    n: Option<usize>,
    dim: Option<isize>,
    norm: FftNorm,
) -> FerrotorchResult<Tensor<T>> {
    reject_half_cpu_fft::<T>("ifft")?;
    let shape = input.shape();

    if shape.is_empty() || *shape.last().unwrap() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "ifft: input must have trailing dimension 2 (complex), got shape {shape:?}"
            ),
        });
    }

    let ndim = shape.len();
    if ndim < 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: "ifft: input must have at least 2 dimensions ([..., n, 2])".into(),
        });
    }

    if input.is_cuda() && (is_f32::<T>() || is_f64::<T>()) && is_last_signal_axis(dim, ndim - 1) {
        let input_n = shape[ndim - 2];
        let fft_n = n.unwrap_or(input_n);
        if fft_n == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "ifft: n must be > 0".into(),
            });
        }
        let batch_shape = &shape[..ndim - 2];
        let batch_size: usize = crate::shape::numel(batch_shape).max(1);
        // GPU C2C dispatch via cuFFT, with on-device pad/truncate when
        // `fft_n != input_n` (#605).
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let input_contiguous = crate::autograd::no_grad::no_grad(|| input.contiguous())?;
        let buf = input_contiguous.gpu_handle()?;

        let (transformed_handle, owned);
        let buf_for_fft: &crate::gpu_dispatch::GpuBufferHandle = if fft_n == input_n {
            buf
        } else if is_f32::<T>() {
            owned = backend.pad_truncate_complex_f32(buf, batch_size, input_n, fft_n)?;
            transformed_handle = &owned;
            transformed_handle
        } else {
            owned = backend.pad_truncate_complex_f64(buf, batch_size, input_n, fft_n)?;
            transformed_handle = &owned;
            transformed_handle
        };

        let h = if is_f32::<T>() {
            backend.fft_c2c_f32(buf_for_fft, batch_size, fft_n, true)?
        } else {
            backend.fft_c2c_f64(buf_for_fft, batch_size, fft_n, true)?
        };
        let h = scale_cuda_fft_output::<T>(backend, h, cuda_inverse_norm_scale(norm, fft_n))?;
        let mut out_shape = batch_shape.to_vec();
        out_shape.push(fft_n);
        out_shape.push(2);
        return Tensor::from_storage(TensorStorage::gpu(h), out_shape, false);
    }

    let arr = tensor_to_complex_array(input, "ifft")?;
    let result =
        ferray_fft::ifft(&arr, n, dim, norm).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("ifft: {e}"),
        })?;
    complex_array_to_tensor(&result)
}

/// 1-D real-to-complex FFT along the last dimension (default `norm`).
///
/// Input is a real-valued tensor of shape `[..., n]`. Output has shape
/// `[..., n/2+1, 2]` representing the non-redundant complex coefficients.
/// Thin wrapper over [`rfft_norm`] with `dim=-1`, `norm=Backward`.
pub fn rfft<T: Float>(input: &Tensor<T>, n: Option<usize>) -> FerrotorchResult<Tensor<T>> {
    rfft_norm(input, n, None, FftNorm::Backward)
}

/// 1-D real-to-complex FFT with explicit `dim` and `norm` (#1294).
///
/// Matches `torch.fft.rfft(input, n, dim, norm)`
/// (`torch/fft/__init__.py:rfft`). `dim` indexes the real input's axes.
pub fn rfft_norm<T: Float>(
    input: &Tensor<T>,
    n: Option<usize>,
    dim: Option<isize>,
    norm: FftNorm,
) -> FerrotorchResult<Tensor<T>> {
    reject_half_cpu_fft::<T>("rfft")?;
    let shape = input.shape();
    if shape.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "rfft: input must have at least 1 dimension".into(),
        });
    }

    let ndim = shape.len();
    let input_n = shape[ndim - 1];

    // GPU fast path: last-axis R2C via cuFFT. PyTorch resizes `n` by slicing
    // or zero-padding before `_fft_r2c`; do that on-device rather than
    // falling through to the CPU ferray bridge.
    if input.is_cuda() && is_last_signal_axis(dim, ndim) {
        let fft_n = n.unwrap_or(input_n);
        if fft_n == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "rfft: n must be > 0".into(),
            });
        }
        let batch_shape = &shape[..ndim - 1];
        let batch_size: usize = crate::shape::numel(batch_shape).max(1);
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let resized =
            crate::autograd::no_grad::no_grad(|| resize_last_axis_real_cuda(input, fft_n))?;
        let buf = resized.gpu_handle()?;
        let h = if is_f32::<T>() {
            backend.rfft_r2c_f32(buf, batch_size, fft_n)?
        } else if is_f64::<T>() {
            backend.rfft_r2c_f64(buf, batch_size, fft_n)?
        } else {
            return Err(FerrotorchError::InvalidArgument {
                message: "rfft requires f32 or f64".into(),
            });
        };
        let h = scale_cuda_fft_output::<T>(backend, h, cuda_forward_norm_scale(norm, fft_n))?;
        let half_n = fft_n / 2 + 1;
        let mut out_shape = batch_shape.to_vec();
        out_shape.push(half_n);
        out_shape.push(2);
        return Tensor::from_storage(TensorStorage::gpu(h), out_shape, false);
    }

    let arr = tensor_to_real_array(input, "rfft")?;
    let result =
        ferray_fft::rfft(&arr, n, dim, norm).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("rfft: {e}"),
        })?;
    complex_array_to_tensor(&result)
}

/// 1-D complex-to-real inverse FFT (default `norm`).
///
/// Input has shape `[..., n/2+1, 2]` (Hermitian spectrum). Output is
/// real-valued with shape `[..., n]`. If `n` is `None`, uses `2*(m-1)`
/// where `m` is the input's second-to-last dimension. Thin wrapper over
/// [`irfft_norm`] with `dim=-1`, `norm=Backward`.
pub fn irfft<T: Float>(input: &Tensor<T>, n: Option<usize>) -> FerrotorchResult<Tensor<T>> {
    irfft_norm(input, n, None, FftNorm::Backward)
}

/// 1-D complex-to-real inverse FFT with explicit `dim` and `norm` (#1294).
///
/// Matches `torch.fft.irfft(input, n, dim, norm)`. `dim` indexes the real
/// *output* axes (equivalently, the input's freq axis); ferray_fft resolves
/// it against the complex input layout (trailing `2` stripped by the bridge).
pub fn irfft_norm<T: Float>(
    input: &Tensor<T>,
    n: Option<usize>,
    dim: Option<isize>,
    norm: FftNorm,
) -> FerrotorchResult<Tensor<T>> {
    reject_half_cpu_fft::<T>("irfft")?;
    let shape = input.shape();

    if shape.is_empty() || *shape.last().unwrap() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "irfft: input must have trailing dimension 2 (complex), got shape {shape:?}"
            ),
        });
    }

    let ndim = shape.len();
    if ndim < 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: "irfft: input must have at least 2 dimensions ([..., n/2+1, 2])".into(),
        });
    }

    // CORE-155 / #1849: validate the output length (torch's "Invalid number
    // of data points") and short-circuit the empty-frequency-axis case
    // BEFORE the default length is ever computed (the eager
    // `n.unwrap_or(2 * (half_n - 1))` underflowed on `[0, 2]` inputs).
    if let Some(short_circuit) = c2r_guard_empty_axis(input, n, dim, "irfft")? {
        return Ok(short_circuit);
    }

    let half_n = shape[ndim - 2];

    // GPU fast path: last-axis C2R via cuFFT. PyTorch resizes the Hermitian
    // input to `n / 2 + 1` when `n` is explicit; do the same with the
    // existing on-device complex pad/truncate kernels.
    if input.is_cuda() && is_last_signal_axis(dim, ndim - 1) {
        // Past `c2r_guard_empty_axis` the last-axis case has `half_n >= 1`
        // and the requested length is `>= 1`, so the lazy default cannot
        // underflow and `output_n >= 1` (CORE-155 / #1849).
        let output_n = match n {
            Some(v) => v,
            None => 2 * (half_n - 1),
        };
        let target_half_n = output_n / 2 + 1;
        let batch_shape = &shape[..ndim - 2];
        let batch_size: usize = crate::shape::numel(batch_shape).max(1);
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let input_contiguous = crate::autograd::no_grad::no_grad(|| input.contiguous())?;
        let buf = input_contiguous.gpu_handle()?;
        let (transformed_handle, owned);
        let buf_for_fft: &crate::gpu_dispatch::GpuBufferHandle = if half_n == target_half_n {
            buf
        } else if is_f32::<T>() {
            owned = backend.pad_truncate_complex_f32(buf, batch_size, half_n, target_half_n)?;
            transformed_handle = &owned;
            transformed_handle
        } else if is_f64::<T>() {
            owned = backend.pad_truncate_complex_f64(buf, batch_size, half_n, target_half_n)?;
            transformed_handle = &owned;
            transformed_handle
        } else {
            return Err(FerrotorchError::InvalidArgument {
                message: "irfft requires f32 or f64".into(),
            });
        };
        let h = if is_f32::<T>() {
            backend.irfft_c2r_f32(buf_for_fft, batch_size, output_n)?
        } else if is_f64::<T>() {
            backend.irfft_c2r_f64(buf_for_fft, batch_size, output_n)?
        } else {
            return Err(FerrotorchError::InvalidArgument {
                message: "irfft requires f32 or f64".into(),
            });
        };
        let h = scale_cuda_fft_output::<T>(backend, h, cuda_inverse_norm_scale(norm, output_n))?;
        let mut out_shape = batch_shape.to_vec();
        out_shape.push(output_n);
        return Tensor::from_storage(TensorStorage::gpu(h), out_shape, false);
    }

    // CPU path: ferray_fft 0.3.8 performs the Hermitian projection + the
    // canonical half-size slice/zero-pad internally (matches PyTorch's
    // `aten::_fft_c2r` — see #807/#808), threading `n` / `dim` / `norm`.
    let arr = tensor_to_complex_array(input, "irfft")?;
    let result =
        ferray_fft::irfft(&arr, n, dim, norm).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("irfft: {e}"),
        })?;
    real_array_to_tensor(&result)
}

/// 2-D FFT (complex-to-complex) along the last two spatial dimensions
/// (default `s`/`dim`/`norm`). Thin wrapper over [`fft2_norm`].
///
/// Input has shape `[..., rows, cols, 2]` (complex). Output has the same shape.
pub fn fft2<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    fft2_norm(input, None, None, FftNorm::Backward)
}

/// 2-D FFT with explicit `s` / `dim` / `norm` (#1294).
///
/// Matches `torch.fft.fft2(input, s, dim, norm)`
/// (`torch/fft/__init__.py:132`). `dim` defaults to the last two signal axes;
/// `s` resizes each transform axis. ferray_fft's `fft2` accepts an arbitrary
/// `axes` list, so `dim` lists of any length (op_db emits `dim=[-3,-2,-1]`
/// for `fft2`, which torch treats as an N-D transform) are honoured directly.
pub fn fft2_norm<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    dim: Option<&[isize]>,
    norm: FftNorm,
) -> FerrotorchResult<Tensor<T>> {
    reject_half_cpu_fft::<T>("fft2")?;
    let shape = input.shape();

    if shape.is_empty() || *shape.last().unwrap() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "fft2: input must have trailing dimension 2 (complex), got shape {shape:?}"
            ),
        });
    }

    let ndim = shape.len();
    if ndim < 3 {
        return Err(FerrotorchError::InvalidArgument {
            message: "fft2: input must have at least 3 dimensions ([..., rows, cols, 2])".into(),
        });
    }

    let rows = shape[ndim - 3];
    let cols = shape[ndim - 2];
    let batch_dims: usize = crate::shape::numel(&shape[..ndim - 3]).max(1);

    // GPU fast path via cufftPlan2d (#634): unbatched (or batch=1) f32/f64,
    // default last-two axes / backward-norm / no resize only.
    if input.is_cuda()
        && batch_dims == 1
        && (is_f32::<T>() || is_f64::<T>())
        && norm == FftNorm::Backward
        && dim.is_none()
        && s.is_none()
    {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let h = if is_f32::<T>() {
            backend.fft2_c2c_f32(input.gpu_handle()?, rows, cols, false)?
        } else {
            backend.fft2_c2c_f64(input.gpu_handle()?, rows, cols, false)?
        };
        return Tensor::from_storage(TensorStorage::gpu(h), shape.to_vec(), false);
    }

    let arr = tensor_to_complex_array(input, "fft2")?;
    let result =
        ferray_fft::fft2(&arr, s, dim, norm).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("fft2: {e}"),
        })?;
    complex_array_to_tensor(&result)
}

/// 2-D inverse FFT (complex-to-complex) along the last two spatial dimensions
/// (default `s`/`dim`/`norm`). Thin wrapper over [`ifft2_norm`].
///
/// Input has shape `[..., rows, cols, 2]` (complex). Output has the same shape.
pub fn ifft2<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    ifft2_norm(input, None, None, FftNorm::Backward)
}

/// 2-D inverse FFT with explicit `s` / `dim` / `norm` (#1294).
///
/// Matches `torch.fft.ifft2(input, s, dim, norm)`
/// (`torch/fft/__init__.py:193`).
pub fn ifft2_norm<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    dim: Option<&[isize]>,
    norm: FftNorm,
) -> FerrotorchResult<Tensor<T>> {
    reject_half_cpu_fft::<T>("ifft2")?;
    let shape = input.shape();

    if shape.is_empty() || *shape.last().unwrap() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "ifft2: input must have trailing dimension 2 (complex), got shape {shape:?}"
            ),
        });
    }

    let ndim = shape.len();
    if ndim < 3 {
        return Err(FerrotorchError::InvalidArgument {
            message: "ifft2: input must have at least 3 dimensions ([..., rows, cols, 2])".into(),
        });
    }

    let rows = shape[ndim - 3];
    let cols = shape[ndim - 2];
    let batch_dims: usize = crate::shape::numel(&shape[..ndim - 3]).max(1);

    if input.is_cuda()
        && batch_dims == 1
        && (is_f32::<T>() || is_f64::<T>())
        && norm == FftNorm::Backward
        && dim.is_none()
        && s.is_none()
    {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let h = if is_f32::<T>() {
            backend.fft2_c2c_f32(input.gpu_handle()?, rows, cols, true)?
        } else {
            backend.fft2_c2c_f64(input.gpu_handle()?, rows, cols, true)?
        };
        return Tensor::from_storage(TensorStorage::gpu(h), shape.to_vec(), false);
    }

    let arr = tensor_to_complex_array(input, "ifft2")?;
    let result =
        ferray_fft::ifft2(&arr, s, dim, norm).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("ifft2: {e}"),
        })?;
    complex_array_to_tensor(&result)
}

// ---------------------------------------------------------------------------
// ferray-fft round-trip helpers
// ---------------------------------------------------------------------------
//
// The following helpers move data between ferrotorch's complex-as-trailing-
// dim-2 convention and ferray-fft's `Array<Complex<f64>, IxDyn>` native
// representation. Computation always runs in f64 to support every
// `T: Float` (including bf16, which ferray-fft itself does not implement).

/// Build an `Array<Complex<f64>, IxDyn>` from a tensor whose last dimension
/// is 2 (re, im). Returns the array shape **without** the trailing 2.
fn tensor_to_complex_array<T: Float>(
    input: &Tensor<T>,
    op: &'static str,
) -> FerrotorchResult<FerrayArray<Complex<f64>, FerrayIxDyn>> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op });
    }

    let shape = input.shape();
    if shape.is_empty() || *shape.last().unwrap() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "{op}: input must have trailing dimension 2 (complex), got shape {shape:?}"
            ),
        });
    }

    let data = input.data_vec()?;
    let total_complex = data.len() / 2;
    let mut complex_data = Vec::with_capacity(total_complex);
    for i in 0..total_complex {
        let re = data[i * 2].to_f64().unwrap();
        let im = data[i * 2 + 1].to_f64().unwrap();
        complex_data.push(Complex::new(re, im));
    }

    let inner_shape: Vec<usize> = shape[..shape.len() - 1].to_vec();
    FerrayArray::from_vec(FerrayIxDyn::new(&inner_shape), complex_data).map_err(|e| {
        FerrotorchError::InvalidArgument {
            message: format!("{op}: failed to build ferray array: {e}"),
        }
    })
}

/// Build a real `Array<f64, IxDyn>` from a real-valued tensor.
fn tensor_to_real_array<T: Float>(
    input: &Tensor<T>,
    op: &'static str,
) -> FerrotorchResult<FerrayArray<f64, FerrayIxDyn>> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op });
    }
    let data = input.data_vec()?;
    let real_data: Vec<f64> = data.iter().map(|v| v.to_f64().unwrap()).collect();
    FerrayArray::from_vec(FerrayIxDyn::new(input.shape()), real_data).map_err(|e| {
        FerrotorchError::InvalidArgument {
            message: format!("{op}: failed to build ferray array: {e}"),
        }
    })
}

/// Direct `f64 -> T` float conversion with `as`-cast semantics for the FFT
/// array->tensor bridges (CORE-157 / #1851): a finite f64 spectrum bin that
/// overflows the narrower target saturates to ±inf, matching torch, which
/// computes f32 FFTs natively in f32 and returns `inf` bins on overflow
/// (verified live, torch 2.11:
/// `torch.fft.rfft(torch.tensor([3e38, 3e38])) == [inf+0j, 0+0j]`). NaN and
/// ±inf sources pass through unchanged.
///
/// The general-purpose fallible [`crate::numeric_cast::cast`] (whose #815
/// saturation guard *rejects* finite overflow) stays in use everywhere else:
/// its contract is argument validation, while this conversion is the
/// value-domain result encoding of a transform whose torch contract is
/// saturate-to-inf.
#[inline]
fn f64_to_float_saturating<T: Float>(v: f64) -> T {
    match num_traits::NumCast::from(v) {
        Some(x) => x,
        // Unreachable for float targets: num_traits' float->float conversion
        // is `Some(v as T)` (`num-traits-0.2.19/src/cast.rs:269-278`) and the
        // `half` impls saturate to ±inf. Stay total without panicking,
        // preserving the sign.
        None => {
            if v.is_sign_negative() {
                T::neg_infinity()
            } else {
                T::infinity()
            }
        }
    }
}

/// Convert an `Array<Complex<f64>, IxDyn>` back to a `Tensor<T>` with the
/// trailing 2-dim representing complex pairs. Finite overflow of the target
/// dtype saturates to ±inf per [`f64_to_float_saturating`] (CORE-157 /
/// #1851).
fn complex_array_to_tensor<T: Float>(
    arr: &FerrayArray<Complex<f64>, FerrayIxDyn>,
) -> FerrotorchResult<Tensor<T>> {
    let shape = arr.shape().to_vec();
    let total: usize = crate::shape::numel(&shape);
    let mut out_data: Vec<T> = Vec::with_capacity(total * 2);
    for c in arr.iter() {
        out_data.push(f64_to_float_saturating(c.re));
        out_data.push(f64_to_float_saturating(c.im));
    }
    let mut out_shape = shape;
    out_shape.push(2);
    Tensor::from_storage(TensorStorage::cpu(out_data), out_shape, false)
}

/// Convert an `Array<f64, IxDyn>` back to a real `Tensor<T>`. Finite overflow
/// of the target dtype saturates to ±inf per [`f64_to_float_saturating`]
/// (CORE-157 / #1851).
fn real_array_to_tensor<T: Float>(
    arr: &FerrayArray<f64, FerrayIxDyn>,
) -> FerrotorchResult<Tensor<T>> {
    let shape = arr.shape().to_vec();
    let out_data: Vec<T> = arr.iter().map(|&v| f64_to_float_saturating(v)).collect();
    Tensor::from_storage(TensorStorage::cpu(out_data), shape, false)
}

// ---------------------------------------------------------------------------
// N-D complex FFT (fftn, ifftn)
// ---------------------------------------------------------------------------

/// N-dimensional complex-to-complex FFT.
///
/// Input has shape `[..., 2]` representing complex values (last dim = re/im).
/// Transforms over the inner dimensions specified by `axes`, or all inner
/// dimensions if `axes` is `None`. The trailing complex dim is always
/// excluded from the transform set.
///
/// `s` optionally specifies the output length along each transform axis
/// (truncate or zero-pad).
///
/// # GPU note
///
/// The 3-D case (shape `[d, h, w, 2]`, `axes=None`, `s=None`) dispatches to
/// `cufftPlan3d` on CUDA f32/f64 tensors (#636). Other ranks remain CPU-only.
pub fn fftn<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
) -> FerrotorchResult<Tensor<T>> {
    fftn_norm(input, s, axes, FftNorm::Backward)
}

/// N-dimensional complex-to-complex FFT with explicit `norm` (#1294).
///
/// Matches `torch.fft.fftn(input, s, dim, norm)`
/// (`torch/fft/__init__.py:246`). Only the default `norm=Backward` case
/// dispatches to cuFFT; other norm modes take the ferray_fft CPU path.
pub fn fftn_norm<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: FftNorm,
) -> FerrotorchResult<Tensor<T>> {
    reject_half_cpu_fft::<T>("fftn")?;
    // GPU fast paths (#636, #966):
    // - axes=None, s=None: dispatch by rank (rank-2 -> cufftPlanMany rank=2,
    //   rank-3 -> cufftPlan3d).
    // - axes=Some(...), s=None: axes-aware dispatch via cufftPlanMany (#966).
    //   The axes list is normalized to non-negative indices and passed to
    //   gpu_fftn_axes_c2c_f32/f64. s!=None still falls through to CPU
    //   (pad/truncate requires a separate pre-pass not yet implemented on GPU).
    if input.is_cuda()
        && (is_f32::<T>() || is_f64::<T>())
        && s.is_none()
        && norm == FftNorm::Backward
    {
        let shape = input.shape();
        let ndim = shape.len();
        // Last dim must be 2 (interleaved complex).
        if ndim >= 2 && shape[ndim - 1] == 2 {
            let spatial_ndim = ndim - 1; // dims excluding the trailing complex dim
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;

            if axes.is_none() {
                // Default: transform all spatial dims.
                if spatial_ndim == 2 {
                    let h = shape[0];
                    let w = shape[1];
                    let h_out = if is_f32::<T>() {
                        backend.fftn2d_c2c_f32(input.gpu_handle()?, h, w, false)?
                    } else {
                        backend.fftn2d_c2c_f64(input.gpu_handle()?, h, w, false)?
                    };
                    return Tensor::from_storage(TensorStorage::gpu(h_out), shape.to_vec(), false);
                }
                if spatial_ndim == 3 {
                    let d = shape[0];
                    let h = shape[1];
                    let w = shape[2];
                    let h_out = if is_f32::<T>() {
                        backend.fftn3d_c2c_f32(input.gpu_handle()?, d, h, w, false)?
                    } else {
                        backend.fftn3d_c2c_f64(input.gpu_handle()?, d, h, w, false)?
                    };
                    return Tensor::from_storage(TensorStorage::gpu(h_out), shape.to_vec(), false);
                }
            } else if let Some(ax) = axes {
                // Axes-override path (#966): normalize isize axes to usize.
                let norm_axes: Vec<usize> = ax
                    .iter()
                    .map(|&a| {
                        if a < 0 {
                            (spatial_ndim as isize + a) as usize
                        } else {
                            a as usize
                        }
                    })
                    .collect();
                // cufftPlanMany with inembed=NULL, istride=1 is only correct
                // when the transform axes are the innermost (last) spatial
                // dimensions in contiguous order. For other axes layouts
                // cuFFT would need stride/embed parameters encoding the full
                // tensor layout, which requires a pre-permute step not yet
                // implemented. Fall through to CPU for non-innermost axes.
                // A set of axes is "innermost" iff it equals the last
                // norm_axes.len() spatial dimensions: {spatial_ndim - r,
                // ..., spatial_ndim - 1} in any order.
                let r = norm_axes.len();
                let innermost_set: std::collections::HashSet<usize> =
                    (spatial_ndim - r..spatial_ndim).collect();
                let axes_set: std::collections::HashSet<usize> =
                    norm_axes.iter().copied().collect();
                if norm_axes.iter().all(|&a| a < spatial_ndim) && axes_set == innermost_set {
                    // Sort axes ascending so cufftPlanMany rank matches shape order.
                    let mut sorted_axes = norm_axes.clone();
                    sorted_axes.sort_unstable();
                    let spatial_shape = &shape[..spatial_ndim];
                    let h_out = if is_f32::<T>() {
                        backend.fftn_axes_c2c_f32(
                            input.gpu_handle()?,
                            spatial_shape,
                            &sorted_axes,
                            false,
                        )?
                    } else {
                        backend.fftn_axes_c2c_f64(
                            input.gpu_handle()?,
                            spatial_shape,
                            &sorted_axes,
                            false,
                        )?
                    };
                    return Tensor::from_storage(TensorStorage::gpu(h_out), shape.to_vec(), false);
                }
            }
        }
    }
    let arr = tensor_to_complex_array(input, "fftn")?;
    let result =
        ferray_fft::fftn(&arr, s, axes, norm).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("fftn: {e}"),
        })?;
    complex_array_to_tensor(&result)
}

/// N-dimensional inverse complex FFT.
///
/// See [`fftn`] for parameter semantics. Normalization divides by the
/// product of the transform-axis lengths (matches `torch.fft.ifftn`).
///
/// # GPU note
///
/// The 3-D case (shape `[d, h, w, 2]`, `axes=None`, `s=None`) dispatches to
/// `cufftPlan3d` on CUDA f32/f64 tensors (#636). Other ranks remain CPU-only.
pub fn ifftn<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
) -> FerrotorchResult<Tensor<T>> {
    ifftn_norm(input, s, axes, FftNorm::Backward)
}

/// N-dimensional inverse complex FFT with explicit `norm` (#1294).
///
/// Matches `torch.fft.ifftn(input, s, dim, norm)`. Only the default
/// `norm=Backward` case dispatches to cuFFT.
pub fn ifftn_norm<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: FftNorm,
) -> FerrotorchResult<Tensor<T>> {
    reject_half_cpu_fft::<T>("ifftn")?;
    // GPU fast paths (#636, #966): mirrors fftn dispatch logic exactly,
    // with inverse=true for all cuFFT calls.
    if input.is_cuda()
        && (is_f32::<T>() || is_f64::<T>())
        && s.is_none()
        && norm == FftNorm::Backward
    {
        let shape = input.shape();
        let ndim = shape.len();
        if ndim >= 2 && shape[ndim - 1] == 2 {
            let spatial_ndim = ndim - 1;
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;

            if axes.is_none() {
                if spatial_ndim == 2 {
                    let h = shape[0];
                    let w = shape[1];
                    let h_out = if is_f32::<T>() {
                        backend.fftn2d_c2c_f32(input.gpu_handle()?, h, w, true)?
                    } else {
                        backend.fftn2d_c2c_f64(input.gpu_handle()?, h, w, true)?
                    };
                    return Tensor::from_storage(TensorStorage::gpu(h_out), shape.to_vec(), false);
                }
                if spatial_ndim == 3 {
                    let d = shape[0];
                    let h = shape[1];
                    let w = shape[2];
                    let h_out = if is_f32::<T>() {
                        backend.fftn3d_c2c_f32(input.gpu_handle()?, d, h, w, true)?
                    } else {
                        backend.fftn3d_c2c_f64(input.gpu_handle()?, d, h, w, true)?
                    };
                    return Tensor::from_storage(TensorStorage::gpu(h_out), shape.to_vec(), false);
                }
            } else if let Some(ax) = axes {
                let norm_axes: Vec<usize> = ax
                    .iter()
                    .map(|&a| {
                        if a < 0 {
                            (spatial_ndim as isize + a) as usize
                        } else {
                            a as usize
                        }
                    })
                    .collect();
                // Same innermost-axes restriction as fftn: GPU path only when
                // axes form the last r spatial dimensions (cufftPlanMany
                // inembed=NULL, istride=1 contract).
                let r = norm_axes.len();
                let innermost_set: std::collections::HashSet<usize> =
                    (spatial_ndim - r..spatial_ndim).collect();
                let axes_set: std::collections::HashSet<usize> =
                    norm_axes.iter().copied().collect();
                if norm_axes.iter().all(|&a| a < spatial_ndim) && axes_set == innermost_set {
                    let mut sorted_axes = norm_axes.clone();
                    sorted_axes.sort_unstable();
                    let spatial_shape = &shape[..spatial_ndim];
                    let h_out = if is_f32::<T>() {
                        backend.fftn_axes_c2c_f32(
                            input.gpu_handle()?,
                            spatial_shape,
                            &sorted_axes,
                            true,
                        )?
                    } else {
                        backend.fftn_axes_c2c_f64(
                            input.gpu_handle()?,
                            spatial_shape,
                            &sorted_axes,
                            true,
                        )?
                    };
                    return Tensor::from_storage(TensorStorage::gpu(h_out), shape.to_vec(), false);
                }
            }
        }
    }
    let arr = tensor_to_complex_array(input, "ifftn")?;
    let result =
        ferray_fft::ifftn(&arr, s, axes, norm).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("ifftn: {e}"),
        })?;
    complex_array_to_tensor(&result)
}

// ---------------------------------------------------------------------------
// N-D real FFT (rfftn, irfftn)
// ---------------------------------------------------------------------------

/// N-dimensional real-to-complex FFT.
///
/// Input is real-valued with shape `[..., n]`. The last transform axis
/// produces `n/2 + 1` complex coefficients (Hermitian symmetry); other
/// transform axes return full length. Output shape is the input shape
/// with the last transform axis replaced by `n/2 + 1` and a trailing 2
/// appended for complex.
pub fn rfftn<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
) -> FerrotorchResult<Tensor<T>> {
    rfftn_norm(input, s, axes, FftNorm::Backward)
}

/// N-dimensional real-to-complex FFT with explicit `norm` (#1294).
///
/// Matches `torch.fft.rfftn(input, s, dim, norm)`.
pub fn rfftn_norm<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: FftNorm,
) -> FerrotorchResult<Tensor<T>> {
    reject_half_cpu_fft::<T>("rfftn")?;
    let arr = tensor_to_real_array(input, "rfftn")?;
    let result =
        ferray_fft::rfftn(&arr, s, axes, norm).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("rfftn: {e}"),
        })?;
    complex_array_to_tensor(&result)
}

/// N-dimensional complex-to-real inverse FFT.
///
/// Inverse of [`rfftn`]. Input has shape `[..., n/2 + 1, 2]` along the
/// last transform axis; output is real with that axis restored to
/// `n` (or whatever `s` specifies).
pub fn irfftn<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
) -> FerrotorchResult<Tensor<T>> {
    irfftn_norm(input, s, axes, FftNorm::Backward)
}

/// N-dimensional complex-to-real inverse FFT with explicit `norm` (#1294).
///
/// Matches `torch.fft.irfftn(input, s, dim, norm)`.
pub fn irfftn_norm<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: FftNorm,
) -> FerrotorchResult<Tensor<T>> {
    reject_half_cpu_fft::<T>("irfftn")?;
    let arr = tensor_to_complex_array(input, "irfftn")?;
    // #808: ferray-fft 0.3.8 now performs the Hermitian projection
    // internally inside its c2r path (matches PyTorch's `aten::_fft_c2r`
    // and scipy/pocketfft semantics). The downstream pre-projection
    // mitigation is no longer needed.
    let result =
        ferray_fft::irfftn(&arr, s, axes, norm).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("irfftn: {e}"),
        })?;
    real_array_to_tensor(&result)
}

/// 2-D real-to-complex FFT (`torch.fft.rfft2`).
///
/// Input is real-valued with shape `[..., rows, cols]`. The 2-D transform
/// runs over the last two axes; only the last axis is Hermitian-truncated
/// (to `cols/2 + 1`), the rows axis goes full length. Output shape is the
/// input shape with `cols` replaced by `cols/2 + 1` and a trailing `2`
/// appended for the complex pair. The 2-D specialization of [`rfftn`] over
/// the trailing two axes (matches `aten::fft_rfft2_symint`'s `return
/// fft_rfftn_symint(...)` delegation).
pub fn rfft2<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
) -> FerrotorchResult<Tensor<T>> {
    rfft2_norm(input, s, axes, FftNorm::Backward)
}

/// 2-D real-to-complex FFT with explicit `norm` (#1294).
///
/// Matches `torch.fft.rfft2(input, s, dim, norm)`.
pub fn rfft2_norm<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: FftNorm,
) -> FerrotorchResult<Tensor<T>> {
    reject_half_cpu_fft::<T>("rfft2")?;
    let arr = tensor_to_real_array(input, "rfft2")?;
    let result =
        ferray_fft::rfft2(&arr, s, axes, norm).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("rfft2: {e}"),
        })?;
    complex_array_to_tensor(&result)
}

/// 2-D complex-to-real inverse FFT (`torch.fft.irfft2`).
///
/// Inverse of [`rfft2`]. Input has shape `[..., rows, cols/2 + 1, 2]`; output
/// is real with the last transform axis restored to `cols` (or whatever `s`
/// specifies). The 2-D specialization of [`irfftn`] over the trailing two
/// axes.
pub fn irfft2<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
) -> FerrotorchResult<Tensor<T>> {
    irfft2_norm(input, s, axes, FftNorm::Backward)
}

/// 2-D complex-to-real inverse FFT with explicit `norm` (#1294).
///
/// Matches `torch.fft.irfft2(input, s, dim, norm)`.
pub fn irfft2_norm<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: FftNorm,
) -> FerrotorchResult<Tensor<T>> {
    reject_half_cpu_fft::<T>("irfft2")?;
    let arr = tensor_to_complex_array(input, "irfft2")?;
    let result =
        ferray_fft::irfft2(&arr, s, axes, norm).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("irfft2: {e}"),
        })?;
    real_array_to_tensor(&result)
}

// ---------------------------------------------------------------------------
// Hermitian FFT (hfft, ihfft)
// ---------------------------------------------------------------------------

/// 1-D FFT of a Hermitian-symmetric complex spectrum, returning real output.
///
/// Input has shape `[..., n/2 + 1, 2]`; output has shape `[..., n]` (real).
/// If `n` is `None`, uses `2 * (input_len - 1)`.
///
/// The Hermitian condition `X[k] = conj(X[-k])` is implicit in the input.
///
/// # GPU note
///
/// CUDA f32/f64 tensors dispatch to `gpu_hfft_*` via cuFFT C2R + conj PTX
/// kernel (#636). Parity: `hfft(x, n) == irfft(conj(x), n)`.
pub fn hfft<T: Float>(input: &Tensor<T>, n: Option<usize>) -> FerrotorchResult<Tensor<T>> {
    hfft_norm(input, n, None, FftNorm::Backward)
}

/// 1-D Hermitian FFT with explicit `dim` and `norm` (#1294).
///
/// Matches `torch.fft.hfft(input, n, dim, norm)`. Only the default last-axis
/// / backward-norm case dispatches to cuFFT.
pub fn hfft_norm<T: Float>(
    input: &Tensor<T>,
    n: Option<usize>,
    dim: Option<isize>,
    norm: FftNorm,
) -> FerrotorchResult<Tensor<T>> {
    reject_half_cpu_fft::<T>("hfft")?;
    // CORE-155 / #1849: validate the output length and short-circuit the
    // empty-frequency-axis case BEFORE the CUDA gate or the ferray bridge
    // can evaluate the underflowing `2 * (half_n - 1)` default.
    if let Some(short_circuit) = c2r_guard_empty_axis(input, n, dim, "hfft")? {
        return Ok(short_circuit);
    }
    // GPU fast path (#636): hfft = conj + irfft, fully on-device. Restricted
    // to default last-axis / backward-norm (cuFFT can't honour dim/norm here).
    if input.is_cuda()
        && (is_f32::<T>() || is_f64::<T>())
        && norm == FftNorm::Backward
        && is_last_signal_axis(dim, input.ndim().saturating_sub(1))
    {
        let shape = input.shape();
        // Input must be [..., half_n, 2].
        if shape.len() >= 2 && *shape.last().unwrap() == 2 {
            let ndim = shape.len();
            let half_in = shape[ndim - 2];
            // Past `c2r_guard_empty_axis`: `half_in >= 1`, so the lazy
            // default cannot underflow (CORE-155 / #1849).
            let n_out = match n {
                Some(v) => v,
                None => 2 * (half_in - 1),
            };
            // GPU path only when half_in == n_out/2+1 (no pad/truncate needed).
            if half_in == n_out / 2 + 1 {
                let batch_shape = &shape[..ndim - 2];
                let batch_size: usize = crate::shape::numel(batch_shape).max(1);
                let backend =
                    crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
                let h_out = if is_f32::<T>() {
                    backend.hfft_f32(input.gpu_handle()?, batch_size, half_in, n_out)?
                } else {
                    backend.hfft_f64(input.gpu_handle()?, batch_size, half_in, n_out)?
                };
                let mut out_shape = batch_shape.to_vec();
                out_shape.push(n_out);
                return Tensor::from_storage(TensorStorage::gpu(h_out), out_shape, false);
            }
        }
    }
    let arr = tensor_to_complex_array(input, "hfft")?;
    // #808: ferray-fft 0.3.8 performs the Hermitian projection internally
    // (its `hfft` delegates to `irfft`, which now projects the c2r axis
    // bins before invoking realfft). The downstream pre-projection
    // mitigation is no longer needed.
    let result =
        ferray_fft::hfft(&arr, n, dim, norm).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("hfft: {e}"),
        })?;
    real_array_to_tensor(&result)
}

/// 1-D inverse FFT of a real signal, returning a Hermitian-symmetric spectrum.
///
/// Input has shape `[..., n]` (real); output has shape `[..., n/2 + 1, 2]`
/// (complex pairs).
///
/// # GPU note
///
/// CUDA f32/f64 tensors dispatch to `gpu_ihfft_*` via cuFFT R2C + conj PTX
/// kernel (#636). Parity: `ihfft(x) == conj(rfft(x)) / n`.
pub fn ihfft<T: Float>(input: &Tensor<T>, n: Option<usize>) -> FerrotorchResult<Tensor<T>> {
    ihfft_norm(input, n, None, FftNorm::Backward)
}

/// 1-D inverse Hermitian FFT with explicit `dim` and `norm` (#1294).
///
/// Matches `torch.fft.ihfft(input, n, dim, norm)`. Only the default last-axis
/// / backward-norm / no-resize case dispatches to cuFFT.
pub fn ihfft_norm<T: Float>(
    input: &Tensor<T>,
    n: Option<usize>,
    dim: Option<isize>,
    norm: FftNorm,
) -> FerrotorchResult<Tensor<T>> {
    reject_half_cpu_fft::<T>("ihfft")?;
    // GPU fast path (#636): ihfft = rfft + scale(1/n) + conj, fully on-device.
    if input.is_cuda()
        && (is_f32::<T>() || is_f64::<T>())
        && norm == FftNorm::Backward
        && is_last_signal_axis(dim, input.ndim())
    {
        let shape = input.shape();
        if !shape.is_empty() {
            let ndim = shape.len();
            let input_n = shape[ndim - 1];
            let fft_n = n.unwrap_or(input_n);
            // GPU path only when fft_n == input_n (no pad/truncate).
            if fft_n == input_n {
                let batch_shape = &shape[..ndim - 1];
                let batch_size: usize = crate::shape::numel(batch_shape).max(1);
                let backend =
                    crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
                let h_out = if is_f32::<T>() {
                    backend.ihfft_f32(input.gpu_handle()?, batch_size, fft_n)?
                } else {
                    backend.ihfft_f64(input.gpu_handle()?, batch_size, fft_n)?
                };
                let half_n = fft_n / 2 + 1;
                let mut out_shape = batch_shape.to_vec();
                out_shape.push(half_n);
                out_shape.push(2);
                return Tensor::from_storage(TensorStorage::gpu(h_out), out_shape, false);
            }
        }
    }
    let arr = tensor_to_real_array(input, "ihfft")?;
    let result =
        ferray_fft::ihfft(&arr, n, dim, norm).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("ihfft: {e}"),
        })?;
    complex_array_to_tensor(&result)
}

/// 2-D FFT of a Hermitian-symmetric spectrum, returning real output
/// (`torch.fft.hfft2`).
///
/// Input has shape `[..., rows, cols/2 + 1, 2]` (Hermitian complex); output is
/// real with shape `[..., rows, n]` where `n` is the last entry of `s` (or
/// `2 * (cols/2+1 - 1)` by default). The 2-D specialization of [`hfftn`] over
/// the trailing two axes (matches `aten::fft_hfft2_symint`).
pub fn hfft2<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
) -> FerrotorchResult<Tensor<T>> {
    hfft2_norm(input, s, axes, FftNorm::Backward)
}

/// 2-D Hermitian FFT with explicit `norm` (#1294).
///
/// Matches `torch.fft.hfft2(input, s, dim, norm)`.
pub fn hfft2_norm<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: FftNorm,
) -> FerrotorchResult<Tensor<T>> {
    reject_half_cpu_fft::<T>("hfft2")?;
    let arr = tensor_to_complex_array(input, "hfft2")?;
    let result =
        ferray_fft::hfft2(&arr, s, axes, norm).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("hfft2: {e}"),
        })?;
    real_array_to_tensor(&result)
}

/// 2-D inverse FFT of a real signal, returning a Hermitian-symmetric spectrum
/// (`torch.fft.ihfft2`).
///
/// Inverse of [`hfft2`]. Input is real with shape `[..., rows, cols]`; output
/// is Hermitian complex `[..., rows, cols/2 + 1, 2]`. The 2-D specialization of
/// [`ihfftn`] over the trailing two axes.
pub fn ihfft2<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
) -> FerrotorchResult<Tensor<T>> {
    ihfft2_norm(input, s, axes, FftNorm::Backward)
}

/// 2-D inverse Hermitian FFT with explicit `norm` (#1294).
///
/// Matches `torch.fft.ihfft2(input, s, dim, norm)`.
pub fn ihfft2_norm<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: FftNorm,
) -> FerrotorchResult<Tensor<T>> {
    reject_half_cpu_fft::<T>("ihfft2")?;
    let arr = tensor_to_real_array(input, "ihfft2")?;
    let result =
        ferray_fft::ihfft2(&arr, s, axes, norm).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("ihfft2: {e}"),
        })?;
    complex_array_to_tensor(&result)
}

/// N-D FFT of a Hermitian-symmetric spectrum, returning real output
/// (`torch.fft.hfftn`). Generalizes [`hfft`] / [`hfft2`] to arbitrary axes.
///
/// Input has shape `[..., 2]` (Hermitian complex); only the last transform
/// axis is the half-spectrum (`n_last/2 + 1`). Output is real with that axis
/// restored to `n_last` (or whatever the last entry of `s` specifies). The
/// N-D analog of `aten::fft_hfftn_symint`.
pub fn hfftn<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
) -> FerrotorchResult<Tensor<T>> {
    hfftn_norm(input, s, axes, FftNorm::Backward)
}

/// N-D Hermitian FFT with explicit `norm` (#1294).
///
/// Matches `torch.fft.hfftn(input, s, dim, norm)`.
pub fn hfftn_norm<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: FftNorm,
) -> FerrotorchResult<Tensor<T>> {
    reject_half_cpu_fft::<T>("hfftn")?;
    let arr = tensor_to_complex_array(input, "hfftn")?;
    let result =
        ferray_fft::hfftn(&arr, s, axes, norm).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("hfftn: {e}"),
        })?;
    real_array_to_tensor(&result)
}

/// N-D inverse FFT of a real signal, returning a Hermitian-symmetric spectrum
/// (`torch.fft.ihfftn`). Generalizes [`ihfft`] / [`ihfft2`] to arbitrary axes.
///
/// Input is real with shape `[..., n]`; the last transform axis becomes
/// `n_last/2 + 1` complex coefficients (trailing 2 appended). The N-D analog
/// of `aten::fft_ihfftn_symint`.
pub fn ihfftn<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
) -> FerrotorchResult<Tensor<T>> {
    ihfftn_norm(input, s, axes, FftNorm::Backward)
}

/// N-D inverse Hermitian FFT with explicit `norm` (#1294).
///
/// Matches `torch.fft.ihfftn(input, s, dim, norm)`.
pub fn ihfftn_norm<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: FftNorm,
) -> FerrotorchResult<Tensor<T>> {
    reject_half_cpu_fft::<T>("ihfftn")?;
    let arr = tensor_to_real_array(input, "ihfftn")?;
    let result =
        ferray_fft::ihfftn(&arr, s, axes, norm).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("ihfftn: {e}"),
        })?;
    complex_array_to_tensor(&result)
}

// ---------------------------------------------------------------------------
// Frequency helpers (fftfreq, rfftfreq)
// ---------------------------------------------------------------------------

/// Discrete Fourier Transform sample frequencies.
///
/// Returns a length-`n` `Tensor<f64>` on CPU containing the frequency bin
/// centers in cycles per unit of the sample spacing `d`. Matches
/// `torch.fft.fftfreq` and `numpy.fft.fftfreq`.
pub fn fftfreq(n: usize, d: f64) -> FerrotorchResult<Tensor<f64>> {
    let arr = ferray_fft::fftfreq(n, d).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("fftfreq: {e}"),
    })?;
    let data: Vec<f64> = arr.iter().copied().collect();
    Tensor::from_storage(TensorStorage::cpu(data), vec![n], false)
}

/// Sample frequencies for `rfft` (non-negative half).
///
/// Returns a length-`n/2 + 1` `Tensor<f64>` on CPU. Matches
/// `torch.fft.rfftfreq`.
pub fn rfftfreq(n: usize, d: f64) -> FerrotorchResult<Tensor<f64>> {
    let arr = ferray_fft::rfftfreq(n, d).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("rfftfreq: {e}"),
    })?;
    let len = arr.shape()[0];
    let data: Vec<f64> = arr.iter().copied().collect();
    Tensor::from_storage(TensorStorage::cpu(data), vec![len], false)
}

// ---------------------------------------------------------------------------
// Shift helpers (fftshift, ifftshift)
// ---------------------------------------------------------------------------

/// True when a shift input is complex-encoded under the module's interleaved
/// convention: `ndim >= 2` with a trailing pair axis of size 2 (CORE-156 /
/// #1850).
///
/// Upstream `fft_fftshift` / `fft_ifftshift`
/// (`aten/src/ATen/native/SpectralOps.cpp:767-789`) roll the tensor's *dims*
/// — for a complex tensor the re/im pair is dtype payload, not a dim, and is
/// never rolled. In the interleaved `[..., 2]` representation the pair axis
/// is therefore metadata: axes resolve against the signal layout (trailing 2
/// stripped) and `axes=None` shifts every *signal* axis only.
///
/// Inherent encoding ambiguity, documented contract: a genuinely REAL tensor
/// whose last dim happens to be 2 is indistinguishable from a length-`[...]`
/// complex tensor and is treated as complex (its pair axis is not shifted).
/// Callers with such real data must shift explicitly per axis on a layout
/// that does not end in 2.
#[inline]
fn is_complex_encoded_shift_input(shape: &[usize]) -> bool {
    shape.len() >= 2 && shape.last() == Some(&2)
}

/// Shift the zero-frequency component to the center along the given axes.
///
/// Matches `torch.fft.fftshift` (`aten/src/ATen/native/SpectralOps.cpp:767`,
/// shift `size[dim] / 2` per axis via `roll`). For complex-encoded inputs
/// (`[..., 2]` interleaved pairs, `ndim >= 2`) `axes` resolves against the
/// signal layout with the trailing pair axis stripped — like every transform
/// entry point in this module — and `axes=None` shifts every *signal* axis;
/// the re/im pair axis is metadata and is never shifted (CORE-156 / #1850;
/// see [`is_complex_encoded_shift_input`] for the encoding-ambiguity
/// contract). For other inputs (e.g. `fftshift(fftfreq(n))`, shape `[n]`)
/// `axes=None` shifts every axis, matching torch on real tensors.
pub fn fftshift<T: Float>(
    input: &Tensor<T>,
    axes: Option<&[isize]>,
) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "fftshift" });
    }
    if is_complex_encoded_shift_input(input.shape()) {
        let arr = tensor_to_complex_array(input, "fftshift")?;
        let shifted =
            ferray_fft::fftshift(&arr, axes).map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("fftshift: {e}"),
            })?;
        return complex_array_to_tensor(&shifted);
    }
    let arr = tensor_to_real_array(input, "fftshift")?;
    let shifted =
        ferray_fft::fftshift(&arr, axes).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("fftshift: {e}"),
        })?;
    real_array_to_tensor(&shifted)
}

/// Inverse of [`fftshift`].
///
/// Differs from `fftshift` only on odd-length axes. Matches
/// `torch.fft.ifftshift` (`aten/src/ATen/native/SpectralOps.cpp:779`, shift
/// `(size[dim] + 1) / 2` per axis). Complex-encoded inputs follow the same
/// signal-layout axis resolution as [`fftshift`] (CORE-156 / #1850).
pub fn ifftshift<T: Float>(
    input: &Tensor<T>,
    axes: Option<&[isize]>,
) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "ifftshift" });
    }
    if is_complex_encoded_shift_input(input.shape()) {
        let arr = tensor_to_complex_array(input, "ifftshift")?;
        let shifted =
            ferray_fft::ifftshift(&arr, axes).map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("ifftshift: {e}"),
            })?;
        return complex_array_to_tensor(&shifted);
    }
    let arr = tensor_to_real_array(input, "ifftshift")?;
    let shifted =
        ferray_fft::ifftshift(&arr, axes).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("ifftshift: {e}"),
        })?;
    real_array_to_tensor(&shifted)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::TensorStorage;

    /// Create a tensor from data and shape.
    fn t(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    fn assert_close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(
            a.len(),
            b.len(),
            "length mismatch: {} vs {}",
            a.len(),
            b.len()
        );
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "index {i}: {x} vs {y} (diff {})",
                (x - y).abs()
            );
        }
    }

    /// Build a complex tensor of shape [n, 2] from a slice of (re, im) pairs.
    fn complex_tensor(pairs: &[(f64, f64)]) -> Tensor<f64> {
        let mut data = Vec::with_capacity(pairs.len() * 2);
        for &(re, im) in pairs {
            data.push(re);
            data.push(im);
        }
        t(&data, &[pairs.len(), 2])
    }

    // -----------------------------------------------------------------------
    // fft of zeros is zeros
    // -----------------------------------------------------------------------

    #[test]
    fn fft_of_zeros() {
        let input = complex_tensor(&[(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]);
        let result = fft(&input, None).unwrap();
        assert_eq!(result.shape(), &[4, 2]);
        let d = result.data().unwrap();
        for &v in d {
            assert!(v.abs() < 1e-12, "expected 0, got {v}");
        }
    }

    // -----------------------------------------------------------------------
    // fft of ones: DC component = n, rest = 0
    // -----------------------------------------------------------------------

    #[test]
    fn fft_of_ones() {
        let n = 8;
        let pairs: Vec<(f64, f64)> = vec![(1.0, 0.0); n];
        let input = complex_tensor(&pairs);
        let result = fft(&input, None).unwrap();
        assert_eq!(result.shape(), &[n, 2]);
        let d = result.data().unwrap();

        // DC component (index 0): re = n, im = 0.
        assert!(
            (d[0] - n as f64).abs() < 1e-10,
            "DC re = {}, expected {n}",
            d[0]
        );
        assert!(d[1].abs() < 1e-10, "DC im = {}", d[1]);

        // All other bins should be 0.
        for i in 1..n {
            assert!(d[i * 2].abs() < 1e-10, "bin {i} re = {}", d[i * 2]);
            assert!(d[i * 2 + 1].abs() < 1e-10, "bin {i} im = {}", d[i * 2 + 1]);
        }
    }

    // -----------------------------------------------------------------------
    // fft of a pure cosine: peaks at k and n-k
    // -----------------------------------------------------------------------

    #[test]
    fn fft_pure_cosine() {
        let n = 16;
        let k = 3; // frequency bin
        let pi = std::f64::consts::PI;

        // x[i] = cos(2*pi*k*i/n)
        let pairs: Vec<(f64, f64)> = (0..n)
            .map(|i| ((2.0 * pi * k as f64 * i as f64 / n as f64).cos(), 0.0))
            .collect();
        let input = complex_tensor(&pairs);
        let result = fft(&input, None).unwrap();
        let d = result.data().unwrap();

        // Magnitudes: bin k and bin n-k should have magnitude n/2.
        // All others should be ~0.
        for i in 0..n {
            let mag = (d[i * 2] * d[i * 2] + d[i * 2 + 1] * d[i * 2 + 1]).sqrt();
            if i == k || i == n - k {
                assert!(
                    (mag - n as f64 / 2.0).abs() < 1e-8,
                    "bin {i}: magnitude {mag}, expected {}",
                    n as f64 / 2.0
                );
            } else {
                assert!(mag < 1e-8, "bin {i}: magnitude {mag}, expected ~0");
            }
        }
    }

    // -----------------------------------------------------------------------
    // fft -> ifft round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn fft_ifft_roundtrip() {
        let pairs = vec![
            (1.0, 2.0),
            (-1.0, 0.5),
            (3.0, -1.0),
            (0.0, 0.0),
            (-2.5, 1.5),
            (0.7, -0.3),
        ];
        let input = complex_tensor(&pairs);
        let spectrum = fft(&input, None).unwrap();
        let recovered = ifft(&spectrum, None).unwrap();
        let d = recovered.data().unwrap();

        for (i, &(re, im)) in pairs.iter().enumerate() {
            assert!(
                (d[i * 2] - re).abs() < 1e-10,
                "re at {i}: {} vs {re}",
                d[i * 2]
            );
            assert!(
                (d[i * 2 + 1] - im).abs() < 1e-10,
                "im at {i}: {} vs {im}",
                d[i * 2 + 1]
            );
        }
    }

    // -----------------------------------------------------------------------
    // rfft + irfft round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn rfft_irfft_roundtrip() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let n = original.len();
        let input = t(&original, &[n]);

        let spectrum = rfft(&input, None).unwrap();
        // n=8 -> n/2+1 = 5 complex values -> shape [5, 2].
        assert_eq!(spectrum.shape(), &[5, 2]);

        let recovered = irfft(&spectrum, Some(n)).unwrap();
        assert_eq!(recovered.shape(), &[n]);
        let d = recovered.data().unwrap();
        assert_close(d, &original, 1e-10);
    }

    // -----------------------------------------------------------------------
    // rfft output shape
    // -----------------------------------------------------------------------

    #[test]
    fn rfft_output_shape() {
        // Even length.
        let input = t(&[0.0; 8], &[8]);
        let result = rfft(&input, None).unwrap();
        assert_eq!(result.shape(), &[5, 2]); // 8/2+1 = 5

        // Odd length.
        let input = t(&[0.0; 7], &[7]);
        let result = rfft(&input, None).unwrap();
        assert_eq!(result.shape(), &[4, 2]); // 7/2+1 = 4
    }

    // -----------------------------------------------------------------------
    // rfft + irfft round-trip with odd length
    // -----------------------------------------------------------------------

    #[test]
    fn rfft_irfft_roundtrip_odd() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let n = original.len();
        let input = t(&original, &[n]);

        let spectrum = rfft(&input, None).unwrap();
        assert_eq!(spectrum.shape(), &[3, 2]); // 5/2+1 = 3

        let recovered = irfft(&spectrum, Some(n)).unwrap();
        assert_eq!(recovered.shape(), &[n]);
        assert_close(recovered.data().unwrap(), &original, 1e-10);
    }

    // -----------------------------------------------------------------------
    // fft with n parameter (padding/truncation)
    // -----------------------------------------------------------------------

    #[test]
    fn fft_with_padding() {
        // Pad [1+0j, 1+0j] to length 4 -> FFT of [1, 1, 0, 0].
        let input = complex_tensor(&[(1.0, 0.0), (1.0, 0.0)]);
        let result = fft(&input, Some(4)).unwrap();
        assert_eq!(result.shape(), &[4, 2]);
        let d = result.data().unwrap();
        // DC = 2.0.
        assert!((d[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn fft_with_truncation() {
        // Truncate [1, 2, 3, 4] to length 2 -> FFT of [1, 2].
        let input = complex_tensor(&[(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]);
        let result = fft(&input, Some(2)).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        let d = result.data().unwrap();
        // FFT of [1, 2] = [3, -1].
        assert!((d[0] - 3.0).abs() < 1e-10);
        assert!(d[1].abs() < 1e-10);
        assert!((d[2] - (-1.0)).abs() < 1e-10);
        assert!(d[3].abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // fft2 round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn fft2_ifft2_roundtrip() {
        // 2x3 complex matrix.
        let pairs = vec![
            (1.0, 0.0),
            (2.0, 0.0),
            (3.0, 0.0),
            (4.0, 0.0),
            (5.0, 0.0),
            (6.0, 0.0),
        ];
        let mut data = Vec::new();
        for &(re, im) in &pairs {
            data.push(re);
            data.push(im);
        }
        let input = t(&data, &[2, 3, 2]);
        let spectrum = fft2(&input).unwrap();
        assert_eq!(spectrum.shape(), &[2, 3, 2]);

        let recovered = ifft2(&spectrum).unwrap();
        assert_eq!(recovered.shape(), &[2, 3, 2]);
        let d = recovered.data().unwrap();
        for (i, &(re, im)) in pairs.iter().enumerate() {
            assert!(
                (d[i * 2] - re).abs() < 1e-9,
                "re at {i}: {} vs {re}",
                d[i * 2]
            );
            assert!(
                (d[i * 2 + 1] - im).abs() < 1e-9,
                "im at {i}: {} vs {im}",
                d[i * 2 + 1]
            );
        }
    }

    // -----------------------------------------------------------------------
    // Batched FFT
    // -----------------------------------------------------------------------

    #[test]
    fn fft_batched() {
        // Batch of 2 signals, each length 4.
        // Signal 0: [1, 0, 0, 0] (impulse) -> all ones.
        // Signal 1: [1, 1, 1, 1] (constant) -> [4, 0, 0, 0].
        let data = vec![
            // batch 0: impulse
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // batch 1: constant
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        ];
        let input = t(&data, &[2, 4, 2]);
        let result = fft(&input, None).unwrap();
        assert_eq!(result.shape(), &[2, 4, 2]);
        let d = result.data().unwrap();

        // Batch 0: all bins should be (1, 0).
        for i in 0..4 {
            assert!((d[i * 2] - 1.0).abs() < 1e-10, "batch0 bin {i} re");
            assert!(d[i * 2 + 1].abs() < 1e-10, "batch0 bin {i} im");
        }

        // Batch 1: DC = (4, 0), rest = (0, 0).
        let off = 4 * 2;
        assert!((d[off] - 4.0).abs() < 1e-10, "batch1 DC re");
        assert!(d[off + 1].abs() < 1e-10, "batch1 DC im");
        for i in 1..4 {
            assert!(d[off + i * 2].abs() < 1e-10, "batch1 bin {i} re");
            assert!(d[off + i * 2 + 1].abs() < 1e-10, "batch1 bin {i} im");
        }
    }

    // -----------------------------------------------------------------------
    // f32 support
    // -----------------------------------------------------------------------

    #[test]
    fn fft_f32() {
        let data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let input = Tensor::from_storage(TensorStorage::cpu(data), vec![4, 2], false).unwrap();
        let result = fft(&input, None).unwrap();
        assert_eq!(result.shape(), &[4, 2]);
        let d = result.data().unwrap();
        for i in 0..4 {
            assert!((d[i * 2] - 1.0).abs() < 1e-5, "bin {i} re = {}", d[i * 2]);
            assert!(d[i * 2 + 1].abs() < 1e-5, "bin {i} im = {}", d[i * 2 + 1]);
        }
    }

    // -----------------------------------------------------------------------
    // Half-precision dtype rejection (#1545 / #1536)
    //
    // Oracle (R-CHAR-3): live torch 2.11 + `SpectralOps.cpp:88-90`. On CPU,
    // `promote_type_fft` does `TORCH_CHECK(type == kFloat || type == kDouble,
    // "Unsupported dtype ", type)`, so BOTH of these raise on CPU:
    //   >>> torch.fft.fft(torch.tensor([1.,2.,3.,4.], dtype=torch.float16))
    //   RuntimeError: Unsupported dtype Half
    //   >>> torch.fft.fft(torch.tensor([1.,2.,3.,4.], dtype=torch.bfloat16))
    //   RuntimeError: Unsupported dtype BFloat16
    //   >>> torch.fft.rfft(...float16/bfloat16...)  ->  same RuntimeError
    // `half` FFT is CUDA-only (`torch/fft/__init__.py:49`), as a native
    // complex-half transform — NOT a CPU upcast to f32. ferrotorch's CPU
    // transforms therefore return Err for f16/bf16 instead of silently
    // upcasting. These tests pin that the divergence (prior: silent Ok) is
    // closed.
    // -----------------------------------------------------------------------

    #[test]
    fn fft_f16_cpu_rejects_matching_torch_unsupported_dtype() {
        use half::f16;
        // [2, 2] interleaved complex f16 (one complex value per row).
        let data: Vec<f16> = vec![
            f16::from_f32(1.0),
            f16::from_f32(0.0),
            f16::from_f32(2.0),
            f16::from_f32(0.0),
        ];
        let input = Tensor::from_storage(TensorStorage::cpu(data), vec![2, 2], false).unwrap();
        let r = fft(&input, None);
        assert!(
            r.is_err(),
            "torch.fft.fft(half) raises RuntimeError: Unsupported dtype Half on CPU; \
             ferrotorch must Err too, not silently upcast"
        );
        let msg = format!("{}", r.unwrap_err());
        assert!(
            msg.contains("Unsupported dtype"),
            "expected 'Unsupported dtype' (mirrors SpectralOps.cpp:90), got: {msg}"
        );
    }

    #[test]
    fn fft_bf16_cpu_rejects_matching_torch_unsupported_dtype() {
        use half::bf16;
        let data: Vec<bf16> = vec![
            bf16::from_f32(1.0),
            bf16::from_f32(0.0),
            bf16::from_f32(2.0),
            bf16::from_f32(0.0),
        ];
        let input = Tensor::from_storage(TensorStorage::cpu(data), vec![2, 2], false).unwrap();
        let r = fft(&input, None);
        assert!(
            r.is_err(),
            "torch.fft.fft(bfloat16) raises RuntimeError: Unsupported dtype BFloat16 on CPU"
        );
        assert!(format!("{}", r.unwrap_err()).contains("Unsupported dtype"));
    }

    #[test]
    fn rfft_f16_and_bf16_cpu_reject() {
        use half::{bf16, f16};
        let f16_real: Vec<f16> = vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];
        let f16_in = Tensor::from_storage(TensorStorage::cpu(f16_real), vec![4], false).unwrap();
        assert!(
            rfft(&f16_in, None).is_err(),
            "torch.fft.rfft(half) raises Unsupported dtype Half on CPU"
        );

        let bf16_real: Vec<bf16> = vec![
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
            bf16::from_f32(3.0),
            bf16::from_f32(4.0),
        ];
        let bf16_in = Tensor::from_storage(TensorStorage::cpu(bf16_real), vec![4], false).unwrap();
        assert!(
            rfft(&bf16_in, None).is_err(),
            "torch.fft.rfft(bfloat16) raises Unsupported dtype BFloat16 on CPU"
        );
    }

    #[test]
    fn nd_and_hermitian_transforms_reject_half() {
        use half::f16;
        // fftn / fft2 over a [2, 2, 2] interleaved-complex f16 grid.
        let cdata: Vec<f16> = (0..8).map(|i| f16::from_f32(i as f32)).collect();
        let c_in = Tensor::from_storage(TensorStorage::cpu(cdata), vec![2, 2, 2], false).unwrap();
        assert!(fft2(&c_in).is_err(), "fft2 must reject f16");
        assert!(fftn(&c_in, None, None).is_err(), "fftn must reject f16");
        assert!(ifftn(&c_in, None, None).is_err(), "ifftn must reject f16");

        // Real-input transforms: rfftn / ihfftn / ihfft over a [4] real f16.
        let rdata: Vec<f16> = (1..=4).map(|i| f16::from_f32(i as f32)).collect();
        let r_in = Tensor::from_storage(TensorStorage::cpu(rdata), vec![4], false).unwrap();
        assert!(ihfft(&r_in, None).is_err(), "ihfft must reject f16");
        assert!(rfftn(&r_in, None, None).is_err(), "rfftn must reject f16");
    }

    #[test]
    fn fftshift_stays_dtype_permissive_for_half() {
        // Oracle (R-CHAR-3): torch.fft.fftshift accepts half/bfloat16 and
        // returns the same dtype (a pure axis roll, NOT a transform):
        //   >>> torch.fft.fftshift(torch.arange(8).to(torch.float16))
        //   tensor([4,5,6,7,0,1,2,3], dtype=torch.float16)   # OK, no raise
        // So the half-rejection guard must NOT apply to fftshift/ifftshift.
        use half::f16;
        let data: Vec<f16> = (0..8).map(|i| f16::from_f32(i as f32)).collect();
        let input = Tensor::from_storage(TensorStorage::cpu(data), vec![8], false).unwrap();
        let shifted = fftshift(&input, None).expect("fftshift(half) must succeed like torch");
        let got: Vec<f32> = shifted.data().unwrap().iter().map(|v| v.to_f32()).collect();
        assert_eq!(got, vec![4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0]);
    }

    // -----------------------------------------------------------------------
    // fftn / ifftn round-trip — agrees with 1-D fft for 1 axis
    // -----------------------------------------------------------------------

    #[test]
    fn fftn_matches_fft_1d() {
        let pairs = vec![(1.0, 2.0), (3.0, -1.0), (-2.0, 0.5), (0.0, 1.0)];
        let input = complex_tensor(&pairs);
        let by_fft = fft(&input, None).unwrap();
        let by_fftn = fftn(&input, None, None).unwrap();
        assert_close(by_fft.data().unwrap(), by_fftn.data().unwrap(), 1e-9);
    }

    #[test]
    fn fftn_ifftn_roundtrip_2d() {
        // 3x4 complex grid (12 complex values, 24 floats).
        let mut data = Vec::with_capacity(24);
        for i in 0..12 {
            data.push(i as f64);
            data.push((i as f64) * 0.5);
        }
        let input = t(&data, &[3, 4, 2]);
        let spectrum = fftn(&input, None, None).unwrap();
        assert_eq!(spectrum.shape(), &[3, 4, 2]);
        let recovered = ifftn(&spectrum, None, None).unwrap();
        assert_eq!(recovered.shape(), &[3, 4, 2]);
        assert_close(recovered.data().unwrap(), input.data().unwrap(), 1e-9);
    }

    #[test]
    fn fftn_ifftn_roundtrip_3d() {
        // 2x2x3 complex grid.
        let mut data = Vec::with_capacity(2 * 2 * 3 * 2);
        for i in 0..(2 * 2 * 3) {
            data.push(i as f64 + 1.0);
            data.push((i as f64) * 0.3);
        }
        let input = t(&data, &[2, 2, 3, 2]);
        let spectrum = fftn(&input, None, None).unwrap();
        assert_eq!(spectrum.shape(), &[2, 2, 3, 2]);
        let recovered = ifftn(&spectrum, None, None).unwrap();
        assert_close(recovered.data().unwrap(), input.data().unwrap(), 1e-9);
    }

    // -----------------------------------------------------------------------
    // rfftn / irfftn round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn rfftn_irfftn_roundtrip_2d() {
        let original: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let input = t(&original, &[3, 4]);
        let spectrum = rfftn(&input, None, None).unwrap();
        // Last transform axis 4 -> 4/2 + 1 = 3 complex values.
        assert_eq!(spectrum.shape(), &[3, 3, 2]);
        let recovered = irfftn(&spectrum, Some(&[3, 4]), None).unwrap();
        assert_eq!(recovered.shape(), &[3, 4]);
        assert_close(recovered.data().unwrap(), &original, 1e-9);
    }

    // -----------------------------------------------------------------------
    // hfft / ihfft round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn hfft_ihfft_roundtrip() {
        let original = vec![1.0, 2.5, -1.5, 0.5, 3.0, -2.0, 0.0, 1.0];
        let n = original.len();
        let input = t(&original, &[n]);
        // ihfft(real n) -> complex n/2+1 -> hfft -> real n.
        let spectrum = ihfft(&input, None).unwrap();
        assert_eq!(spectrum.shape(), &[n / 2 + 1, 2]);
        let recovered = hfft(&spectrum, Some(n)).unwrap();
        assert_eq!(recovered.shape(), &[n]);
        assert_close(recovered.data().unwrap(), &original, 1e-9);
    }

    // -----------------------------------------------------------------------
    // fftfreq / rfftfreq numerical correctness
    // -----------------------------------------------------------------------

    #[test]
    fn fftfreq_known_values() {
        // numpy: fftfreq(8, 1.0) = [0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125]
        let f = fftfreq(8, 1.0).unwrap();
        let expected = [0.0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125];
        assert_close(f.data().unwrap(), &expected, 1e-12);
    }

    #[test]
    fn rfftfreq_known_values() {
        // numpy: rfftfreq(8, 1.0) = [0, 0.125, 0.25, 0.375, 0.5]
        let f = rfftfreq(8, 1.0).unwrap();
        let expected = [0.0, 0.125, 0.25, 0.375, 0.5];
        assert_close(f.data().unwrap(), &expected, 1e-12);
    }

    #[test]
    fn fftfreq_with_sample_spacing() {
        // d = 0.1: bin 1 = 1/(8*0.1) = 1.25
        let f = fftfreq(8, 0.1).unwrap();
        let d = f.data().unwrap();
        assert!((d[1] - 1.25).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // fftshift / ifftshift
    // -----------------------------------------------------------------------

    #[test]
    fn fftshift_basic_even() {
        // Even length: [0,1,2,3,4,5,6,7] -> [4,5,6,7,0,1,2,3]
        let input = t(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[8]);
        let shifted = fftshift(&input, None).unwrap();
        let d = shifted.data().unwrap();
        assert_close(d, &[4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0], 1e-12);
    }

    #[test]
    fn fftshift_ifftshift_even_inverse() {
        // For even-length axes, ifftshift undoes fftshift exactly.
        let input = t(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[8]);
        let shifted = fftshift(&input, None).unwrap();
        let unshifted = ifftshift(&shifted, None).unwrap();
        assert_close(unshifted.data().unwrap(), input.data().unwrap(), 1e-12);
    }

    #[test]
    fn fftshift_ifftshift_odd_inverse() {
        // Odd-length: fftshift and ifftshift differ but compose to identity.
        let input = t(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5]);
        let shifted = fftshift(&input, None).unwrap();
        let unshifted = ifftshift(&shifted, None).unwrap();
        assert_close(unshifted.data().unwrap(), input.data().unwrap(), 1e-12);
    }

    #[test]
    fn fftshift_axes_arg() {
        // 2x4: shift only the last axis -> [[2,3,0,1],[6,7,4,5]]
        let input = t(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[2, 4]);
        let shifted = fftshift(&input, Some(&[-1])).unwrap();
        assert_close(
            shifted.data().unwrap(),
            &[2.0, 3.0, 0.0, 1.0, 6.0, 7.0, 4.0, 5.0],
            1e-12,
        );
    }

    // -----------------------------------------------------------------------
    // GPU discipline: GPU tensors return DeviceUnavailable, not silent CPU bounce.
    // We can't construct a real CUDA tensor in this CPU-only test context, but
    // we verify the existing `is_cuda` rejection path is intact for the new
    // wrappers by checking that the helpers carry the same gate. This is
    // exercised in integration tests on machines with CUDA.
    // -----------------------------------------------------------------------

    #[test]
    fn fftn_agrees_with_fft2_for_2d() {
        // 2D complex grid; fftn over last 2 axes should match fft2.
        let mut data = Vec::with_capacity(2 * 3 * 2);
        for i in 0..6 {
            data.push((i as f64) - 3.0);
            data.push((i as f64) * 0.7);
        }
        let input = t(&data, &[2, 3, 2]);
        let by_fft2 = fft2(&input).unwrap();
        let by_fftn = fftn(&input, None, None).unwrap();
        assert_close(by_fft2.data().unwrap(), by_fftn.data().unwrap(), 1e-9);
    }

    // -----------------------------------------------------------------------
    // 2-D / N-D real + Hermitian forward ops (#1299)
    // -----------------------------------------------------------------------

    #[test]
    fn rfft2_output_shape_and_irfft2_roundtrip() {
        // Real 3x4 input → rfft2 over last 2 axes. Last axis 4 → 4/2+1 = 3.
        let original: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let input = t(&original, &[3, 4]);
        let spectrum = rfft2(&input, None, None).unwrap();
        // Rows (3) full length, cols (4) → 3 complex, trailing 2.
        assert_eq!(spectrum.shape(), &[3, 3, 2]);
        let recovered = irfft2(&spectrum, Some(&[3, 4]), None).unwrap();
        assert_eq!(recovered.shape(), &[3, 4]);
        assert_close(recovered.data().unwrap(), &original, 1e-9);
    }

    #[test]
    fn rfft2_matches_rfftn_over_last_two_axes() {
        // rfft2 is the 2-D specialization of rfftn over the trailing 2 axes
        // (aten::fft_rfft2_symint delegates to fft_rfftn_symint).
        let original: Vec<f64> = (1..=24).map(|x| x as f64 * 0.5).collect();
        let input = t(&original, &[2, 3, 4]);
        let by_rfft2 = rfft2(&input, None, None).unwrap();
        let by_rfftn = rfftn(&input, None, Some(&[-2, -1])).unwrap();
        assert_eq!(by_rfft2.shape(), by_rfftn.shape());
        assert_close(by_rfft2.data().unwrap(), by_rfftn.data().unwrap(), 1e-9);
    }

    #[test]
    fn ihfft2_hfft2_roundtrip() {
        // Real 4x4 → ihfft2 → Hermitian complex → hfft2 → real 4x4.
        let original: Vec<f64> = (1..=16).map(|x| x as f64).collect();
        let input = t(&original, &[4, 4]);
        let spectrum = ihfft2(&input, None, None).unwrap();
        // Last axis 4 → 4/2+1 = 3 complex coefficients, trailing 2.
        assert_eq!(spectrum.shape(), &[4, 3, 2]);
        let recovered = hfft2(&spectrum, Some(&[4, 4]), None).unwrap();
        assert_eq!(recovered.shape(), &[4, 4]);
        assert_close(recovered.data().unwrap(), &original, 1e-9);
    }

    #[test]
    fn ihfftn_hfftn_roundtrip_3d() {
        // Real 2x2x4 → ihfftn (all axes) → Hermitian complex → hfftn → real.
        let original: Vec<f64> = (1..=16).map(|x| x as f64 * 0.25).collect();
        let input = t(&original, &[2, 2, 4]);
        let spectrum = ihfftn(&input, None, None).unwrap();
        // Last transform axis 4 → 4/2+1 = 3; trailing 2 appended.
        assert_eq!(spectrum.shape(), &[2, 2, 3, 2]);
        let recovered = hfftn(&spectrum, Some(&[2, 2, 4]), None).unwrap();
        assert_eq!(recovered.shape(), &[2, 2, 4]);
        assert_close(recovered.data().unwrap(), &original, 1e-9);
    }

    // -----------------------------------------------------------------------
    // norm / dim threading (#1294)
    // -----------------------------------------------------------------------

    #[test]
    fn fft_norm_from_str_maps_modes() {
        // Maps the torch norm strings 1:1 onto FftNorm (SpectralOps.cpp:116-130).
        assert_eq!(fft_norm_from_str(None, "fft").unwrap(), FftNorm::Backward);
        assert_eq!(
            fft_norm_from_str(Some("backward"), "fft").unwrap(),
            FftNorm::Backward
        );
        assert_eq!(
            fft_norm_from_str(Some("forward"), "fft").unwrap(),
            FftNorm::Forward
        );
        assert_eq!(
            fft_norm_from_str(Some("ortho"), "fft").unwrap(),
            FftNorm::Ortho
        );
        assert!(fft_norm_from_str(Some("bogus"), "fft").is_err());
    }

    #[test]
    fn fft_ortho_norm_scales_dc_by_sqrt_n() {
        // FFT of n ones: DC bin = sum = n (backward), n/sqrt(n)=sqrt(n) (ortho),
        // n/n = 1 (forward). These are the closed-form DFT values, traceable to
        // numpy's norm semantics (torch/fft/__init__.py:57-69).
        let n = 8usize;
        let pairs: Vec<(f64, f64)> = vec![(1.0, 0.0); n];
        let input = complex_tensor(&pairs);

        let backward = fft_norm(&input, None, None, FftNorm::Backward).unwrap();
        assert!((backward.data().unwrap()[0] - n as f64).abs() < 1e-10);

        let ortho = fft_norm(&input, None, None, FftNorm::Ortho).unwrap();
        // DC = n / sqrt(n) = sqrt(n).
        assert!((ortho.data().unwrap()[0] - (n as f64).sqrt()).abs() < 1e-10);

        let forward = fft_norm(&input, None, None, FftNorm::Forward).unwrap();
        // DC = n / n = 1.
        assert!((forward.data().unwrap()[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn fft_ortho_is_unitary_roundtrip() {
        // ortho fft followed by ortho ifft recovers the input (unitary pair).
        let pairs = vec![(1.0, 2.0), (-1.0, 0.5), (3.0, -1.0), (0.0, 0.0)];
        let input = complex_tensor(&pairs);
        let spectrum = fft_norm(&input, None, None, FftNorm::Ortho).unwrap();
        let recovered = ifft_norm(&spectrum, None, None, FftNorm::Ortho).unwrap();
        let d = recovered.data().unwrap();
        for (i, &(re, im)) in pairs.iter().enumerate() {
            assert!((d[i * 2] - re).abs() < 1e-10, "re {i}");
            assert!((d[i * 2 + 1] - im).abs() < 1e-10, "im {i}");
        }
    }

    #[test]
    fn fft_dim_transforms_named_axis() {
        // A [2, 4, 2] complex tensor (2 rows, 4 cols, complex pair). Transform
        // along dim=-2 (the rows axis) vs the default dim=-1 (cols). Compare
        // against the equivalent default-axis transform of the transposed data.
        // Build rows = two distinct constant signals so the DC bins differ.
        // Row 0 = all (1,0); row 1 = all (2,0). shape [2, 4, 2].
        let mut data = Vec::new();
        for r in 0..2 {
            for _c in 0..4 {
                data.push((r + 1) as f64);
                data.push(0.0);
            }
        }
        let input = t(&data, &[2, 4, 2]);
        // dim=-2 transforms along the rows (length-2 axis). For a length-2 DFT
        // of [1, 2] along each column, bin0 = 3, bin1 = -1 (real).
        let out = fft_norm(&input, None, Some(-2), FftNorm::Backward).unwrap();
        assert_eq!(out.shape(), &[2, 4, 2]);
        let d = out.data().unwrap();
        // For each column c: out[0, c] = 3 + 0i, out[1, c] = -1 + 0i.
        for c in 0..4 {
            let bin0 = d[c * 2]; // row 0, col c, re
            let bin1 = d[(4 + c) * 2]; // row 1, col c, re
            assert!((bin0 - 3.0).abs() < 1e-10, "col {c} bin0 = {bin0}");
            assert!((bin1 - (-1.0)).abs() < 1e-10, "col {c} bin1 = {bin1}");
        }
    }

    #[test]
    fn rfft_dim_transforms_named_axis() {
        // Real [4, 3] input, rfft along dim=-2 (length-4 axis). Output last
        // transform axis becomes 4/2+1 = 3 along axis 0: shape [3, 3, 2].
        let original: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let input = t(&original, &[4, 3]);
        let out = rfft_norm(&input, None, Some(-2), FftNorm::Backward).unwrap();
        assert_eq!(out.shape(), &[3, 3, 2]);
        // Round-trip via irfft along the same axis recovers the signal.
        let back = irfft_norm(&out, Some(4), Some(-2), FftNorm::Backward).unwrap();
        assert_eq!(back.shape(), &[4, 3]);
        assert_close(back.data().unwrap(), &original, 1e-9);
    }

    #[test]
    fn fftn_s_resizes_named_axes() {
        // fftn over a [3, 4] complex grid with s=[2, 8], dim=[0, 1]: axis 0
        // truncated to 2, axis 1 zero-padded to 8. Output [2, 8, 2].
        let mut data = Vec::with_capacity(3 * 4 * 2);
        for i in 0..12 {
            data.push(i as f64);
            data.push(0.0);
        }
        let input = t(&data, &[3, 4, 2]);
        let out = fftn_norm(&input, Some(&[2, 8]), Some(&[0, 1]), FftNorm::Backward).unwrap();
        assert_eq!(out.shape(), &[2, 8, 2]);
    }

    #[test]
    fn hfft2_matches_hfftn_over_last_two_axes() {
        // hfft2 is the 2-D specialization of hfftn over the trailing 2 axes.
        // Build a Hermitian-shaped complex input [2, 3, 3, 2] (last axis = 3
        // half-spectrum bins) by running ihfftn on a real signal first.
        let original: Vec<f64> = (1..=24).map(|x| x as f64 * 0.3).collect();
        let real_in = t(&original, &[2, 3, 4]);
        let spectrum = ihfftn(&real_in, Some(&[3, 4]), Some(&[-2, -1])).unwrap();
        let by_hfft2 = hfft2(&spectrum, Some(&[3, 4]), None).unwrap();
        let by_hfftn = hfftn(&spectrum, Some(&[3, 4]), Some(&[-2, -1])).unwrap();
        assert_eq!(by_hfft2.shape(), by_hfftn.shape());
        assert_close(by_hfft2.data().unwrap(), by_hfftn.data().unwrap(), 1e-9);
    }
}
