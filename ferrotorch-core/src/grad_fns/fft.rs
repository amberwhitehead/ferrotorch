//! Backward functions for FFT operations.
//!
//! The key mathematical identities:
//! - `d/dx FFT(x) = FFT(grad)` (FFT is linear, so its own Jacobian)
//! - `d/dx IFFT(x) = IFFT(grad)` (same reasoning)
//!
//! More precisely, for the backward pass of `y = fft(x)`:
//!   `grad_input = ifft(grad_output) * n`  (because our ifft divides by n)
//!
//! For `y = ifft(x)`:
//!   `grad_input = fft(grad_output) / n`
//!
//! ## REQ status (per `.design/ferrotorch-core/grad_fns/fft.md`)
//!
//! As of #1294 every op honours `norm`/`dim`/`s` and reaches `0 skipped,
//! 0 failed` at `--seeds 8` (full scope). The wrappers thread the norm/dim
//! via their `*_differentiable_norm` siblings; the backward VJP uses the
//! `adjoint_norm` identity (same fft_norm_mode int, flipped direction —
//! `tools/autograd/derivatives.yaml:2960-2961`).
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (`fft.fft`) | SHIPPED | `fft_differentiable` / `fft_differentiable_norm` + `FftBackward`; consumer: default wrapper delegates to the norm path; `[fft.fft] 64/64 passed (0 skipped, 0 failed)`. |
//! | REQ-2 (`fft.ifft`) | SHIPPED | `ifft_differentiable` / `ifft_differentiable_norm` + `IfftBackward`; `[fft.ifft] 64/64 passed (0 skipped, 0 failed)`. |
//! | REQ-3 (`fft.rfft`) | SHIPPED | `rfft_differentiable` / `rfft_differentiable_norm` + `RfftBackward`/`RfftnBackward`; `[fft.rfft] 64/64 passed (0 skipped, 0 failed)`. |
//! | REQ-4 (`fft.irfft`) | SHIPPED | `irfft_differentiable` / `irfft_differentiable_norm` + `IrfftBackward`/`IrfftnBackward`; `[fft.irfft] 64/64 passed (0 skipped, 0 failed)`. |
//! | REQ-5 (`fft.fftn`) | SHIPPED | `fftn_differentiable` / `fftn_differentiable_norm` + `FftnBackward` (closes #1296); `[fft.fftn] 72/72 passed (0 skipped, 0 failed)`. |
//! | REQ-6 (`fft.ifftn`) | SHIPPED | `ifftn_differentiable` / `ifftn_differentiable_norm` + `IfftnBackward`; `[fft.ifftn] 72/72 passed (0 skipped, 0 failed)`. |
//! | REQ-7 (`fft.rfftn`) | SHIPPED | `rfftn_differentiable` / `rfftn_differentiable_norm` + `RfftnBackward`; `[fft.rfftn] 72/72 passed (0 skipped, 0 failed)`. |
//! | REQ-8 (`fft.irfftn`) | SHIPPED | `irfftn_differentiable` / `irfftn_differentiable_norm` + `IrfftnBackward`; `[fft.irfftn] 72/72 passed (0 skipped, 0 failed)`. |
//! | REQ-9 (`fft.hfft`) | SHIPPED | `hfft_differentiable` + `HfftBackward`; parity arm calls forward `hfft_norm`; `[fft.hfft] 64/64 passed (0 skipped, 0 failed)`. |
//! | REQ-10 (`fft.ihfft`) | SHIPPED | `ihfft_differentiable` + `IhfftBackward`; parity arm calls forward `ihfft_norm`; `[fft.ihfft] 64/64 passed (0 skipped, 0 failed)`. |
//! | REQ-11 (`fft.fft2`) | SHIPPED | `fft2_differentiable` / `fft2_differentiable_norm` + `Fft2Backward`/`FftnBackward` (closes #1300); `[fft.fft2] 56/56 passed (0 skipped, 0 failed)`. |
//! | REQ-12 (`fft.ifft2`) | SHIPPED | `ifft2_differentiable` / `ifft2_differentiable_norm` + `Ifft2Backward`/`IfftnBackward` (closes #1300); `[fft.ifft2] 56/56 passed (0 skipped, 0 failed)`. |
//! | REQ-13 (`fft.rfft2`) | SHIPPED | forward `fft::rfft2`/`rfft2_norm` (closes #1299-forward); autograd wrapper follow-up; `[fft.rfft2] 56/56 passed (0 skipped, 0 failed)`. |
//! | REQ-14 (`fft.irfft2`) | SHIPPED | forward `fft::irfft2`/`irfft2_norm` (closes #1299-forward); autograd wrapper follow-up; `[fft.irfft2] 56/56 passed (0 skipped, 0 failed)`. |
//! | REQ-15 (`fft.hfft2`) | SHIPPED | forward `fft::hfft2`/`hfft2_norm` (closes #1299-forward); autograd wrapper follow-up; `[fft.hfft2] 56/56 passed (0 skipped, 0 failed)`. |
//! | REQ-16 (`fft.ihfft2`) | SHIPPED | forward `fft::ihfft2`/`ihfft2_norm` (closes #1299-forward); autograd wrapper follow-up; `[fft.ihfft2] 56/56 passed (0 skipped, 0 failed)`. |
//! | REQ-17 (`fft.hfftn`) | SHIPPED | forward `fft::hfftn`/`hfftn_norm` (closes #1299-forward); autograd wrapper follow-up; `[fft.hfftn] 72/72 passed (0 skipped, 0 failed)`. |
//! | REQ-18 (`fft.ihfftn`) | SHIPPED | forward `fft::ihfftn`/`ihfftn_norm` (closes #1299-forward); autograd wrapper follow-up; `[fft.ihfftn] 72/72 passed (0 skipped, 0 failed)`.

use std::sync::Arc;

use crate::autograd::no_grad::is_grad_enabled;
use crate::dtype::Float;
use crate::error::FerrotorchResult;
use crate::fft;
use crate::fft::FftNorm;
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

/// Adjoint normalization mode for an FFT VJP (#1294).
///
/// PyTorch's autograd uses the SAME `fft_norm_mode` int for the backward pass
/// but flips the transform direction (`_fft_c2c(grad, dim, normalization,
/// !forward)`, `tools/autograd/derivatives.yaml:2960-2961`). The mode int is
/// direction-independent (`aten/src/ATen/native/SpectralOpsUtils.h:15-19`).
/// Expressed via ferray_fft's direction-dependent [`FftNorm`], the adjoint of
/// a forward transform with norm `X` is the inverse transform with norm
/// `adjoint_norm(X)`:
///
/// - `Backward` (fwd-scale 1) ↔ `Forward` (inv-scale 1)
/// - `Forward` (fwd-scale 1/n) ↔ `Backward` (inv-scale 1/n)
/// - `Ortho` (scale 1/√n both) → `Ortho`
///
/// This holds symmetrically for c2c forward/inverse and (combined with the
/// Hermitian-doubling correction) for the r2c / c2r VJPs.
#[inline]
fn adjoint_norm(norm: FftNorm) -> FftNorm {
    match norm {
        FftNorm::Backward => FftNorm::Forward,
        FftNorm::Forward => FftNorm::Backward,
        FftNorm::Ortho => FftNorm::Ortho,
    }
}

#[inline]
fn gpu_f32<T: Float>() -> bool {
    std::mem::size_of::<T>() == 4
}

#[inline]
fn gpu_f64<T: Float>() -> bool {
    std::mem::size_of::<T>() == 8
}

fn resize_last_axis_real_tensor<T: Float>(
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
        "rfft backward resize padding",
    )?;
    crate::grad_fns::shape::cat(&[input.clone(), zeros], (ndim - 1) as isize)?.contiguous()
}

fn resize_last_axis_complex_tensor<T: Float>(
    input: &Tensor<T>,
    target_n: usize,
) -> FerrotorchResult<Tensor<T>> {
    let shape = input.shape();
    if shape.len() < 2 || shape.last() != Some(&2) {
        return Err(crate::error::FerrotorchError::InvalidArgument {
            message: format!(
                "complex resize: input must have trailing complex pair, got {shape:?}"
            ),
        });
    }
    let ndim = shape.len();
    let input_n = shape[ndim - 2];
    if input_n == target_n {
        return input.contiguous();
    }

    if input.is_cuda() {
        let batch_shape = &shape[..ndim - 2];
        let batch_size = crate::shape::numel(batch_shape).max(1);
        let backend = crate::gpu_dispatch::gpu_backend()
            .ok_or(crate::error::FerrotorchError::DeviceUnavailable)?;
        let contiguous = input.contiguous()?;
        let buf = contiguous.gpu_handle()?;
        let handle = if gpu_f32::<T>() {
            backend.pad_truncate_complex_f32(buf, batch_size, input_n, target_n)?
        } else if gpu_f64::<T>() {
            backend.pad_truncate_complex_f64(buf, batch_size, input_n, target_n)?
        } else {
            return Err(crate::error::FerrotorchError::NotImplementedOnCuda {
                op: "complex_resize",
            });
        };
        let mut out_shape = batch_shape.to_vec();
        out_shape.push(target_n);
        out_shape.push(2);
        return Tensor::from_storage(TensorStorage::gpu(handle), out_shape, false);
    }

    if input_n > target_n {
        return input.narrow(ndim - 2, 0, target_n)?.contiguous();
    }
    let mut pad_shape = shape.to_vec();
    pad_shape[ndim - 2] = target_n - input_n;
    let zeros = crate::creation::full_on_device(
        &pad_shape,
        <T as num_traits::Zero>::zero(),
        input.device(),
        "irfft backward spectrum padding",
    )?;
    crate::grad_fns::shape::cat(&[input.clone(), zeros], (ndim - 2) as isize)?.contiguous()
}

fn scale_cuda_tensor<T: Float>(input: &Tensor<T>, scale: T) -> FerrotorchResult<Tensor<T>> {
    let contiguous = input.contiguous()?;
    let backend = crate::gpu_dispatch::gpu_backend()
        .ok_or(crate::error::FerrotorchError::DeviceUnavailable)?;
    let handle = if gpu_f32::<T>() {
        let scale = <T as num_traits::ToPrimitive>::to_f32(&scale).ok_or_else(|| {
            crate::error::FerrotorchError::InvalidArgument {
                message: "CUDA FFT scale is not representable as f32".into(),
            }
        })?;
        backend.scale_f32(contiguous.gpu_handle()?, scale)?
    } else if gpu_f64::<T>() {
        let scale = <T as num_traits::ToPrimitive>::to_f64(&scale).ok_or_else(|| {
            crate::error::FerrotorchError::InvalidArgument {
                message: "CUDA FFT scale is not representable as f64".into(),
            }
        })?;
        backend.scale_f64(contiguous.gpu_handle()?, scale)?
    } else {
        return Err(crate::error::FerrotorchError::NotImplementedOnCuda { op: "fft_scale" });
    };
    Tensor::from_storage(
        TensorStorage::gpu(handle),
        contiguous.shape().to_vec(),
        false,
    )
}

// ---------------------------------------------------------------------------
// FftBackward
// ---------------------------------------------------------------------------

/// Backward for `y = fft(x, n)`.
///
/// VJP: `grad_x = ifft(grad_y) * n` (un-normalized inverse).
/// Since our `ifft` already divides by n, the grad is just `ifft(grad_y) * n`,
/// but actually the correct VJP for a normalized FFT pair where
/// `fft` has no normalization and `ifft` divides by n is:
/// `grad_x = conj(fft(conj(grad_y))) / n = ifft(grad_y) * n / n = ifft(grad_y)` ... wait.
///
/// Let's be precise. Our conventions:
/// - `fft`: no normalization (forward sum without 1/n).
/// - `ifft`: divides by n.
///
/// For `y = FFT(x)` (unnormalized), the Jacobian is the DFT matrix W.
/// The VJP is `grad_x = W^H @ grad_y = n * IFFT(grad_y)`.
///
/// But our `ifft` already computes `(1/n) * W^H @ input`, so
/// `grad_x = n * ifft(grad_y)`.
#[derive(Debug)]
pub struct FftBackward<T: Float> {
    input: Tensor<T>,
    n: Option<usize>,
    /// Transform axis in the real-signal layout (`None` = last).
    dim: Option<isize>,
    /// Forward normalization mode (`torch.fft.fft`'s `norm`).
    norm: FftNorm,
}

impl<T: Float> FftBackward<T> {
    pub fn new(input: Tensor<T>, n: Option<usize>) -> Self {
        Self::new_norm(input, n, None, FftNorm::Backward)
    }

    pub fn new_norm(input: Tensor<T>, n: Option<usize>, dim: Option<isize>, norm: FftNorm) -> Self {
        Self {
            input,
            n,
            dim,
            norm,
        }
    }
}

impl<T: Float> GradFn<T> for FftBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let grad_input = if self.input.requires_grad() {
            // Adjoint of forward `fft(norm)` is `ifft(adjoint_norm(norm))`
            // (same mode int, flipped direction — derivatives.yaml:2960-2961).
            // The `n` resize is re-applied so grad shape matches the input.
            let inv = fft::ifft_norm(grad_output, self.n, self.dim, adjoint_norm(self.norm))?;
            Some(inv)
        } else {
            None
        };
        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "FftBackward"
    }
}

// ---------------------------------------------------------------------------
// IfftBackward
// ---------------------------------------------------------------------------

/// Backward for `y = ifft(x, n)`.
///
/// Our `ifft(x)` = (1/n) * W^H @ x, so the VJP is:
/// `grad_x = (1/n) * W @ grad_y = (1/n) * fft(grad_y)`.
///
/// Since our `fft` is unnormalized: `grad_x = fft(grad_y) / n`.
#[derive(Debug)]
pub struct IfftBackward<T: Float> {
    input: Tensor<T>,
    n: Option<usize>,
    dim: Option<isize>,
    norm: FftNorm,
}

impl<T: Float> IfftBackward<T> {
    pub fn new(input: Tensor<T>, n: Option<usize>) -> Self {
        Self::new_norm(input, n, None, FftNorm::Backward)
    }

    pub fn new_norm(input: Tensor<T>, n: Option<usize>, dim: Option<isize>, norm: FftNorm) -> Self {
        Self {
            input,
            n,
            dim,
            norm,
        }
    }
}

impl<T: Float> GradFn<T> for IfftBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let grad_input = if self.input.requires_grad() {
            // Adjoint of forward `ifft(norm)` is `fft(adjoint_norm(norm))`.
            let fwd = fft::fft_norm(grad_output, self.n, self.dim, adjoint_norm(self.norm))?;
            Some(fwd)
        } else {
            None
        };
        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "IfftBackward"
    }
}

// ---------------------------------------------------------------------------
// RfftBackward
// ---------------------------------------------------------------------------

/// Backward for `y = rfft(x, n)` (real → Hermitian-truncated complex).
///
/// VJP derivation (matches PyTorch `FftR2CBackward` for `norm="backward"`):
///
/// The forward `Y = rfft(x)` with output length `K = N/2 + 1` is the linear
/// map `Y[k] = sum_{n} x[n] exp(-2π i k n / N)` for `k = 0..K-1`. For a
/// real-valued upstream cotangent `grad_Y`, the cotangent on `x` is
///
/// ```text
///   grad_x[n] = real( sum_{k=0..K-1} grad_Y[k] * exp(+2π i k n / N) )
/// ```
///
/// (i.e., the **partial** unnormalized inverse over the half-spectrum, NOT
/// the Hermitian-extended full inverse). Implementing this as `irfft(grad_Y,
/// N)` would Hermitian-extend the spectrum and **double** the interior
/// freqs, then divide by `N` — both wrong by a factor of `N` and by the
/// boundary correction.
///
/// PyTorch's reference path is equivalent to: zero-pad `grad_Y` along the
/// freq axis from `K` to `N`, run an unnormalized inverse complex FFT, take
/// the real part. We compute this by calling our normalized
/// `fft::ifft(zero_padded, N)` (which divides by `N`) and multiplying by `N`
/// to undo the normalization.
#[derive(Debug)]
pub struct RfftBackward<T: Float> {
    input: Tensor<T>,
    _n: Option<usize>,
    /// The actual FFT length used in the forward pass.
    fft_n: usize,
}

impl<T: Float> RfftBackward<T> {
    pub fn new(input: Tensor<T>, n: Option<usize>, fft_n: usize) -> Self {
        Self {
            input,
            _n: n,
            fft_n,
        }
    }
}

impl<T: Float> GradFn<T> for RfftBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let grad_input = if self.input.requires_grad() {
            // Zero-pad grad_Y from [..., K, 2] to [..., N, 2] along the freq axis.
            let go_shape = grad_output.shape();
            if go_shape.len() < 2 || go_shape[go_shape.len() - 1] != 2 {
                return Err(crate::error::FerrotorchError::InvalidArgument {
                    message: format!(
                        "RfftBackward: grad_output must have trailing complex pair, got {go_shape:?}"
                    ),
                });
            }
            let k = go_shape[go_shape.len() - 2];
            let n = self.fft_n;
            let batch_shape = &go_shape[..go_shape.len() - 2];
            let batch_size: usize = crate::shape::numel(batch_shape).max(1);

            if grad_output.device().is_cuda() {
                let original_n = *self.input.shape().last().unwrap();
                let grad = crate::autograd::no_grad::no_grad(|| {
                    let padded = resize_last_axis_complex_tensor(grad_output, n)?;
                    // `FftNorm::Forward` cancels the cuFFT inverse wrapper's
                    // 1/n scale, matching PyTorch's unnormalized C2C adjoint.
                    let inv = fft::ifft_norm(&padded, Some(n), None, FftNorm::Forward)?;
                    let real_singleton = inv.narrow(inv.shape().len() - 1, 0, 1)?.contiguous()?;
                    let mut resized_shape = batch_shape.to_vec();
                    resized_shape.push(n);
                    let grad_resized = real_singleton.view_reshape(resized_shape)?;
                    resize_last_axis_real_tensor(&grad_resized, original_n)
                })?;
                return Ok(vec![Some(grad)]);
            }

            let go_data = grad_output.data_vec()?;

            let mut padded = vec![T::from(0.0).unwrap(); batch_size * n * 2];
            for b in 0..batch_size {
                let src_off = b * k * 2;
                let dst_off = b * n * 2;
                let copy_pairs = k.min(n);
                for kk in 0..copy_pairs {
                    padded[dst_off + kk * 2] = go_data[src_off + kk * 2];
                    padded[dst_off + kk * 2 + 1] = go_data[src_off + kk * 2 + 1];
                }
            }
            let mut padded_shape = batch_shape.to_vec();
            padded_shape.push(n);
            padded_shape.push(2);
            let padded_t = Tensor::from_storage(TensorStorage::cpu(padded), padded_shape, false)?;

            // ifft is normalized (divides by N); multiply by N to unnormalize.
            let inv = fft::ifft(&padded_t, Some(n))?;
            let inv_data = inv.data_vec()?;
            let scale = T::from(n).unwrap();
            // Take real part: drop the trailing 2 axis.
            let mut grad_x_data = Vec::with_capacity(batch_size * n);
            for b in 0..batch_size {
                for nn in 0..n {
                    grad_x_data.push(inv_data[b * n * 2 + nn * 2] * scale);
                }
            }
            let mut out_shape = batch_shape.to_vec();
            out_shape.push(n);
            let t = Tensor::from_storage(TensorStorage::cpu(grad_x_data), out_shape, false)?;
            let original_n = *self.input.shape().last().unwrap();
            Some(crate::autograd::no_grad::no_grad(|| {
                resize_last_axis_real_tensor(&t, original_n)
            })?)
        } else {
            None
        };
        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "RfftBackward"
    }
}

// ---------------------------------------------------------------------------
// IrfftBackward
// ---------------------------------------------------------------------------

/// Backward for `y = irfft(x, n)` (Hermitian-truncated complex → real).
///
/// VJP derivation (matches PyTorch `FftC2RBackward` for `norm="backward"`):
///
/// Forward `y = irfft(x, N)` reconstructs the real signal by Hermitian-
/// extending `x` (shape `[..., K, 2]`, `K = N/2 + 1`) to length `N`, running
/// the unnormalized inverse FFT, and dividing by `N`. As a real-linear map
/// over `x`,
///
/// ```text
///   y[n] = (1/N) * (
///       x_re[0]
///     + (-1)^n * x_re[N/2]                                  (only when N even)
///     + 2 * sum_{k=1..N/2-1} (x_re[k] cos(2π k n/N) - x_im[k] sin(2π k n/N))
///   )
/// ```
///
/// Differentiating w.r.t. real `grad_y[n]` and assembling the cotangent on
/// `x` (complex of shape `[..., K, 2]`) gives, with `F = rfft(grad_y, N)`:
///
/// - boundary: `grad_x[0]    = F[0]    / N`
/// - boundary: `grad_x[N/2]  = F[N/2]  / N`     (when `N` even)
/// - interior: `grad_x[k]    = 2 * F[k] / N`    (for `k = 1..K-2`)
///
/// The factor of 2 is the Hermitian-doubling correction: each interior `k`
/// in the half-spectrum corresponds to **two** entries in the full DFT
/// (`k` and `N-k`), so the chain rule contributes twice. PyTorch's
/// `_fft_c2r_backward` handles this exactly the same way.
///
/// Net change vs. the previous (buggy) `rfft(grad_y, N)` call:
///   - divide all entries by `N` (the missing normalization),
///   - multiply interior entries by 2 (the Hermitian-doubling correction).
#[derive(Debug)]
pub struct IrfftBackward<T: Float> {
    input: Tensor<T>,
    _n: Option<usize>,
    /// The output length used in the forward pass.
    output_n: usize,
}

impl<T: Float> IrfftBackward<T> {
    pub fn new(input: Tensor<T>, n: Option<usize>, output_n: usize) -> Self {
        Self {
            input,
            _n: n,
            output_n,
        }
    }
}

impl<T: Float> GradFn<T> for IrfftBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let grad_input = if self.input.requires_grad() {
            let n = self.output_n;
            let k = n / 2 + 1;

            if grad_output.device().is_cuda() {
                let input_shape = self.input.shape();
                let original_half_n = input_shape[input_shape.len() - 2];
                let grad = crate::autograd::no_grad::no_grad(|| {
                    // rfft(grad_y, N) returns shape [..., K, 2]. Scale by
                    // 1/N, then double only Hermitian-interior frequencies.
                    let f = fft::rfft(grad_output, Some(n))?;
                    let inv_n = T::from(1.0).unwrap() / T::from(n).unwrap();
                    let scaled = scale_cuda_tensor(&f, inv_n)?;
                    let axis = scaled.shape().len() - 2;
                    let k = scaled.shape()[axis];

                    let mut pieces = Vec::with_capacity(3);
                    pieces.push(scaled.narrow(axis, 0, 1)?.contiguous()?);

                    let interior_len = if n.is_multiple_of(2) {
                        k.saturating_sub(2)
                    } else {
                        k.saturating_sub(1)
                    };
                    if interior_len > 0 {
                        let interior = scaled.narrow(axis, 1, interior_len)?.contiguous()?;
                        pieces.push(scale_cuda_tensor(&interior, T::from(2.0).unwrap())?);
                    }

                    if n.is_multiple_of(2) && k > 1 {
                        pieces.push(scaled.narrow(axis, k - 1, 1)?.contiguous()?);
                    }

                    let corrected = if pieces.len() == 1 {
                        pieces.remove(0)
                    } else {
                        crate::grad_fns::shape::cat(&pieces, axis as isize)?.contiguous()?
                    };
                    resize_last_axis_complex_tensor(&corrected, original_half_n)
                })?;
                return Ok(vec![Some(grad)]);
            }

            // rfft(grad_y, N) returns shape [..., K, 2] — same shape as x.
            let f = fft::rfft(grad_output, Some(n))?;
            let f_shape = f.shape().to_vec();
            let f_data = f.data_vec()?;
            let total_pairs = f_data.len() / 2;
            let batch_size = total_pairs / k;

            let inv_n = T::from(1.0).unwrap() / T::from(n).unwrap();
            let two = T::from(2.0).unwrap();
            let mut out = Vec::with_capacity(f_data.len());
            for b in 0..batch_size {
                for kk in 0..k {
                    let interior = kk > 0 && (kk < k - 1 || n % 2 == 1);
                    // For odd N there's no Nyquist sample; every k>0 is interior.
                    let factor = if interior { two * inv_n } else { inv_n };
                    out.push(f_data[(b * k + kk) * 2] * factor);
                    out.push(f_data[(b * k + kk) * 2 + 1] * factor);
                }
            }
            let t = Tensor::from_storage(TensorStorage::cpu(out), f_shape, false)?;
            let input_shape = self.input.shape();
            let original_half_n = input_shape[input_shape.len() - 2];
            Some(crate::autograd::no_grad::no_grad(|| {
                resize_last_axis_complex_tensor(&t, original_half_n)
            })?)
        } else {
            None
        };
        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "IrfftBackward"
    }
}

// ---------------------------------------------------------------------------
// Differentiable forward wrappers
// ---------------------------------------------------------------------------

/// Resolve a torch `dim` (real-signal layout, `signal_ndim` dims) into a
/// non-negative axis. Used to translate the 1-D `dim` into the `axes=[dim]`
/// list the N-D rfft/irfft backward path consumes.
#[inline]
fn resolve_signal_axis(dim: Option<isize>, signal_ndim: usize) -> usize {
    match dim {
        None => signal_ndim.saturating_sub(1),
        Some(d) if d < 0 => (signal_ndim as isize + d).max(0) as usize,
        Some(d) => d as usize,
    }
}

fn tensor_from_fft_operation<T: Float, G: GradFn<T> + 'static>(
    result: Tensor<T>,
    grad_fn: Arc<G>,
) -> FerrotorchResult<Tensor<T>> {
    let (storage, shape) = result.into_storage_and_shape()?;
    let grad_fn: Arc<dyn GradFn<T>> = grad_fn;
    Tensor::from_operation(storage, shape, grad_fn)
}

/// Differentiable 1-D FFT (default `dim`/`norm`). Attaches `FftBackward`.
pub fn fft_differentiable<T: Float>(
    input: &Tensor<T>,
    n: Option<usize>,
) -> FerrotorchResult<Tensor<T>> {
    fft_differentiable_norm(input, n, None, FftNorm::Backward)
}

/// Differentiable 1-D FFT with explicit `dim` / `norm` (#1294). Attaches a
/// `FftBackward` that threads the adjoint norm/dim. Matches `torch.fft.fft`.
pub fn fft_differentiable_norm<T: Float>(
    input: &Tensor<T>,
    n: Option<usize>,
    dim: Option<isize>,
    norm: FftNorm,
) -> FerrotorchResult<Tensor<T>> {
    let result = fft::fft_norm(input, n, dim, norm)?;

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(FftBackward::new_norm(input.clone(), n, dim, norm));
        tensor_from_fft_operation(result, grad_fn)
    } else {
        Ok(result)
    }
}

/// Differentiable 1-D inverse FFT (default `dim`/`norm`). Attaches `IfftBackward`.
pub fn ifft_differentiable<T: Float>(
    input: &Tensor<T>,
    n: Option<usize>,
) -> FerrotorchResult<Tensor<T>> {
    ifft_differentiable_norm(input, n, None, FftNorm::Backward)
}

/// Differentiable 1-D inverse FFT with explicit `dim` / `norm` (#1294).
pub fn ifft_differentiable_norm<T: Float>(
    input: &Tensor<T>,
    n: Option<usize>,
    dim: Option<isize>,
    norm: FftNorm,
) -> FerrotorchResult<Tensor<T>> {
    let result = fft::ifft_norm(input, n, dim, norm)?;

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(IfftBackward::new_norm(input.clone(), n, dim, norm));
        tensor_from_fft_operation(result, grad_fn)
    } else {
        Ok(result)
    }
}

/// Differentiable 1-D real FFT (default `dim`/`norm`). Attaches `RfftBackward`.
pub fn rfft_differentiable<T: Float>(
    input: &Tensor<T>,
    n: Option<usize>,
) -> FerrotorchResult<Tensor<T>> {
    let input_n = *input.shape().last().unwrap();
    let fft_n = n.unwrap_or(input_n);
    let result = fft::rfft(input, n)?;

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(RfftBackward::new(input.clone(), n, fft_n));
        tensor_from_fft_operation(result, grad_fn)
    } else {
        Ok(result)
    }
}

/// Differentiable 1-D real FFT with explicit `dim` / `norm` (#1294).
///
/// `rfft` along `dim` is the 1-axis specialization of `rfftn` over
/// `axes=[dim]`; the backward routes through the (norm- and axis-general)
/// [`RfftnBackward`] to reuse its proven stride-walk for the zero-pad +
/// adjoint inverse. The default `dim=-1`/`norm=Backward` path is identical to
/// [`rfft_differentiable`] (which keeps the cheaper [`RfftBackward`]).
pub fn rfft_differentiable_norm<T: Float>(
    input: &Tensor<T>,
    n: Option<usize>,
    dim: Option<isize>,
    norm: FftNorm,
) -> FerrotorchResult<Tensor<T>> {
    let in_shape = input.shape();
    let axis = resolve_signal_axis(dim, in_shape.len());
    // Fast path: default last-axis backward-norm → cheap 1-D backward.
    if norm == FftNorm::Backward && axis == in_shape.len() - 1 {
        return rfft_differentiable(input, n);
    }
    let result = fft::rfft_norm(input, n, dim, norm)?;

    if is_grad_enabled() && input.requires_grad() {
        let last_axis_n = n.unwrap_or(in_shape[axis]);
        let s_back = vec![last_axis_n];
        let axes_back = vec![axis as isize];
        let out_shape = result.shape().to_vec();
        // last_axis_logical in the rfft output (trailing 2 excluded) is `axis`.
        let grad_fn = Arc::new(RfftnBackward::new_norm(
            input.clone(),
            Some(s_back),
            Some(axes_back),
            out_shape,
            last_axis_n,
            axis,
            last_axis_n,
            norm,
        ));
        tensor_from_fft_operation(result, grad_fn)
    } else {
        Ok(result)
    }
}

/// Differentiable 1-D inverse real FFT (default `dim`/`norm`). Attaches `IrfftBackward`.
pub fn irfft_differentiable<T: Float>(
    input: &Tensor<T>,
    n: Option<usize>,
) -> FerrotorchResult<Tensor<T>> {
    let shape = input.shape();
    let half_n = shape[shape.len() - 2];
    let output_n = n.unwrap_or(2 * (half_n - 1));
    let result = fft::irfft(input, n)?;

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(IrfftBackward::new(input.clone(), n, output_n));
        tensor_from_fft_operation(result, grad_fn)
    } else {
        Ok(result)
    }
}

/// Differentiable 1-D inverse real FFT with explicit `dim` / `norm` (#1294).
///
/// `irfft` along `dim` is the 1-axis specialization of `irfftn` over
/// `axes=[dim]`; the backward routes through [`IrfftnBackward`]. Default
/// `dim=-1`/`norm=Backward` keeps the cheaper [`IrfftBackward`].
pub fn irfft_differentiable_norm<T: Float>(
    input: &Tensor<T>,
    n: Option<usize>,
    dim: Option<isize>,
    norm: FftNorm,
) -> FerrotorchResult<Tensor<T>> {
    let in_shape = input.shape();
    // The complex input's freq axis lives in the real-output layout; resolve
    // `dim` against the real-output ndim (= input ndim - 1, trailing 2 dropped).
    let real_ndim = in_shape.len().saturating_sub(1);
    let axis = resolve_signal_axis(dim, real_ndim);
    let half_n = in_shape[axis];
    let output_n = n.unwrap_or(2 * (half_n.saturating_sub(1)));
    if norm == FftNorm::Backward && axis == real_ndim.saturating_sub(1) {
        return irfft_differentiable(input, n);
    }
    let result = fft::irfft_norm(input, n, dim, norm)?;

    if is_grad_enabled() && input.requires_grad() {
        let s_back = vec![output_n];
        let axes_back = vec![axis as isize];
        let grad_fn = Arc::new(IrfftnBackward::new_norm(
            input.clone(),
            Some(s_back),
            Some(axes_back),
            output_n,
            axis,
            output_n,
            norm,
        ));
        tensor_from_fft_operation(result, grad_fn)
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// FftnBackward / IfftnBackward — N-D complex FFT backward.
// ---------------------------------------------------------------------------
//
// Math (FftNorm::Backward convention, matches torch.fft):
//   y = fftn(x, s, axes)   → grad_x = prod(s) * ifftn(grad_y, s, axes)
//   y = ifftn(x, s, axes)  → grad_x = fftn(grad_y, s, axes) / prod(s)
//
// The shape of the transform output along each transform axis is the value
// in `s` (or the input length if `s` is `None`). We persist `s` and `axes`
// from the forward to keep the backward shape-stable.

#[derive(Debug)]
pub struct FftnBackward<T: Float> {
    input: Tensor<T>,
    s: Option<Vec<usize>>,
    axes: Option<Vec<isize>>,
    /// Product of the transform-axis lengths in the forward output (retained
    /// for the legacy `new` ctor; the threaded-norm backward no longer needs
    /// an explicit scale — ferray's `adjoint_norm` inverse carries it).
    #[allow(dead_code, reason = "retained for the grandfathered `new` ctor API")]
    norm_n: usize,
    norm: FftNorm,
}

impl<T: Float> FftnBackward<T> {
    pub fn new(
        input: Tensor<T>,
        s: Option<Vec<usize>>,
        axes: Option<Vec<isize>>,
        norm_n: usize,
    ) -> Self {
        Self {
            input,
            s,
            axes,
            norm_n,
            norm: FftNorm::Backward,
        }
    }

    pub fn new_norm(
        input: Tensor<T>,
        s: Option<Vec<usize>>,
        axes: Option<Vec<isize>>,
        norm_n: usize,
        norm: FftNorm,
    ) -> Self {
        Self {
            input,
            s,
            axes,
            norm_n,
            norm,
        }
    }
}

impl<T: Float> GradFn<T> for FftnBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let grad_input = if self.input.requires_grad() {
            // Adjoint of `fftn(norm)` is `ifftn(adjoint_norm(norm))`.
            let inv = fft::ifftn_norm(
                grad_output,
                self.s.as_deref(),
                self.axes.as_deref(),
                adjoint_norm(self.norm),
            )?;
            Some(inv)
        } else {
            None
        };
        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "FftnBackward"
    }
}

#[derive(Debug)]
pub struct IfftnBackward<T: Float> {
    input: Tensor<T>,
    s: Option<Vec<usize>>,
    axes: Option<Vec<isize>>,
    #[allow(dead_code, reason = "retained for the grandfathered `new` ctor API")]
    norm_n: usize,
    norm: FftNorm,
}

impl<T: Float> IfftnBackward<T> {
    pub fn new(
        input: Tensor<T>,
        s: Option<Vec<usize>>,
        axes: Option<Vec<isize>>,
        norm_n: usize,
    ) -> Self {
        Self {
            input,
            s,
            axes,
            norm_n,
            norm: FftNorm::Backward,
        }
    }

    pub fn new_norm(
        input: Tensor<T>,
        s: Option<Vec<usize>>,
        axes: Option<Vec<isize>>,
        norm_n: usize,
        norm: FftNorm,
    ) -> Self {
        Self {
            input,
            s,
            axes,
            norm_n,
            norm,
        }
    }
}

impl<T: Float> GradFn<T> for IfftnBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let grad_input = if self.input.requires_grad() {
            // Adjoint of `ifftn(norm)` is `fftn(adjoint_norm(norm))`.
            let fwd = fft::fftn_norm(
                grad_output,
                self.s.as_deref(),
                self.axes.as_deref(),
                adjoint_norm(self.norm),
            )?;
            Some(fwd)
        } else {
            None
        };
        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "IfftnBackward"
    }
}

// ---------------------------------------------------------------------------
// RfftnBackward / IrfftnBackward — N-D real FFT backward.
// ---------------------------------------------------------------------------
//
// VJPs:
//   y = rfftn(x, s, axes) (real → Hermitian complex)
//     → grad_x = irfftn(grad_y, s=original_real_shape, axes)
//   y = irfftn(x, s, axes) (Hermitian complex → real)
//     → grad_x = rfftn(grad_y, s=original_real_shape, axes)

/// Backward for `y = rfftn(x, s, axes)` (real → Hermitian-truncated complex,
/// N-D). Generalizes `RfftBackward` to multiple transform axes.
///
/// Only the **last** transform axis is Hermitian-truncated; the others go
/// full length. As in the 1-D case, the correct VJP is the
/// **partial** unnormalized inverse over the half-spectrum:
///
/// ```text
///   grad_x = real( ifftn_unnormalized(zero_pad_last_freq_axis(grad_Y), s, axes) )
/// ```
///
/// We use `fft::ifftn` (which divides by `prod(s)`) and multiply by
/// `prod(s)` to undo the normalization. The previous implementation called
/// `fft::irfftn(grad_Y, s, axes)` which Hermitian-extends and divides — both
/// errors of #809.
#[derive(Debug)]
pub struct RfftnBackward<T: Float> {
    input: Tensor<T>,
    /// Output sizes along the transform axes (passed to irfftn for backward).
    s: Option<Vec<usize>>,
    axes: Option<Vec<isize>>,
    /// `rfftn` output shape (used to invert the half-spectrum truncation).
    out_shape: Vec<usize>,
    /// Length of the last transform axis in the original real signal
    /// (so we know how far to zero-pad the freq axis).
    last_axis_n: usize,
    /// Logical index of the last transform axis in the rfftn output (the
    /// truncated freq axis). Trailing complex pair is excluded.
    last_axis_logical: usize,
    /// Product of transform-axis lengths (retained for the grandfathered
    /// `new` ctor; the threaded-norm backward uses `adjoint_norm` instead).
    #[allow(dead_code, reason = "retained for the grandfathered `new` ctor API")]
    norm_n: usize,
    /// Forward normalization mode.
    norm: FftNorm,
}

impl<T: Float> RfftnBackward<T> {
    pub fn new(
        input: Tensor<T>,
        s: Option<Vec<usize>>,
        axes: Option<Vec<isize>>,
        out_shape: Vec<usize>,
        last_axis_n: usize,
        last_axis_logical: usize,
        norm_n: usize,
    ) -> Self {
        Self::new_norm(
            input,
            s,
            axes,
            out_shape,
            last_axis_n,
            last_axis_logical,
            norm_n,
            FftNorm::Backward,
        )
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors the forward's persisted state (s/axes/out_shape/\
                  last_axis_n/last_axis_logical/norm_n/norm); a struct-literal \
                  ctor wrapper would not reduce the captured fields"
    )]
    pub fn new_norm(
        input: Tensor<T>,
        s: Option<Vec<usize>>,
        axes: Option<Vec<isize>>,
        out_shape: Vec<usize>,
        last_axis_n: usize,
        last_axis_logical: usize,
        norm_n: usize,
        norm: FftNorm,
    ) -> Self {
        Self {
            input,
            s,
            axes,
            out_shape,
            last_axis_n,
            last_axis_logical,
            norm_n,
            norm,
        }
    }
}

impl<T: Float> GradFn<T> for RfftnBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let grad_input = if self.input.requires_grad() {
            let device = grad_output.device();
            // 1. Zero-pad along the last transform axis from K = n_last/2 + 1
            //    to n_last. The trailing axis (size 2) is the complex pair —
            //    untouched.
            let go_shape = grad_output.shape();
            if go_shape != self.out_shape.as_slice() {
                return Err(crate::error::FerrotorchError::InvalidArgument {
                    message: format!(
                        "RfftnBackward: grad_output shape {go_shape:?} does not match \
                         forward output {:?}",
                        self.out_shape
                    ),
                });
            }
            let go_data = grad_output.data_vec()?;
            let mut padded_shape = self.out_shape.clone();
            // Replace the half-spectrum dim with the full last_axis_n.
            // out_shape layout: [..., last_axis_K, 2]. last_axis_logical is the
            // index of the K dim within this layout (relative to the trailing 2).
            padded_shape[self.last_axis_logical] = self.last_axis_n;
            let padded_total: usize = crate::shape::numel(&padded_shape);
            let mut padded = vec![T::from(0.0).unwrap(); padded_total];
            // Compute strides for both shapes (row-major).
            let go_strides = row_major_strides(go_shape);
            let pad_strides = row_major_strides(&padded_shape);
            // K = original last_axis dim in go_shape (half-spectrum).
            let k = go_shape[self.last_axis_logical];
            // Iterate every element of grad_output and copy into padded.
            for flat in 0..go_data.len() / 2 {
                // Compute multi-index for the [..., K, 2]-stripped layout
                // (i.e., excluding the trailing 2). flat indexes complex pairs
                // here; the trailing 2 is handled in the inner loop.
                let mut idx = vec![0usize; go_shape.len() - 1];
                let mut rem = flat;
                let logical_strides = row_major_strides(&go_shape[..go_shape.len() - 1]);
                for d in 0..idx.len() {
                    idx[d] = rem / logical_strides[d];
                    rem %= logical_strides[d];
                }
                // Source offset (real start of pair).
                let mut src = 0usize;
                for d in 0..idx.len() {
                    src += idx[d] * go_strides[d];
                }
                // Destination offset (uses padded_shape strides).
                let mut dst = 0usize;
                for d in 0..idx.len() {
                    dst += idx[d] * pad_strides[d];
                }
                padded[dst] = go_data[src];
                padded[dst + 1] = go_data[src + 1];
                let _ = k; // silence unused-variable warnings on some paths
            }
            let padded_t = Tensor::from_storage(TensorStorage::cpu(padded), padded_shape, false)?;

            // 2. Adjoint inverse FFT with `adjoint_norm(norm)`: ferray's
            //    direction-dependent scaling carries the un-normalization
            //    (e.g. Backward→Forward yields the un-normalized inverse,
            //    matching the old `prod(s) * ifftn` for norm="backward").
            let inv = fft::ifftn_norm(
                &padded_t,
                self.s.as_deref(),
                self.axes.as_deref(),
                adjoint_norm(self.norm),
            )?;
            // 3. Take real part (drop trailing 2).
            let inv_data = inv.data_vec()?;
            let inv_shape = inv.shape().to_vec();
            let real_n_pairs = inv_data.len() / 2;
            let mut grad_x_data = Vec::with_capacity(real_n_pairs);
            for i in 0..real_n_pairs {
                grad_x_data.push(inv_data[i * 2]);
            }
            // Drop trailing 2 from shape.
            let mut out_shape = inv_shape;
            let _ = out_shape.pop();
            let t = Tensor::from_storage(TensorStorage::cpu(grad_x_data), out_shape, false)?;
            Some(if device.is_cuda() { t.to(device)? } else { t })
        } else {
            None
        };
        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "RfftnBackward"
    }
}

/// Row-major strides for a shape (in elements, not bytes).
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

/// Backward for `y = irfftn(x, s, axes)` (Hermitian-truncated complex → real,
/// N-D). Generalizes `IrfftBackward` to multiple transform axes.
///
/// The forward Hermitian-extends along the LAST transform axis only (the
/// other transform axes go full length). So the VJP is:
///
/// ```text
///   grad_x = rfftn(grad_y, s, axes) / prod(s),
///   then multiply interior frequencies along the last freq axis by 2.
/// ```
///
/// `grad_x_re/im[k]` for `k` along the last freq axis:
/// - boundary (`k = 0` and, when `n_last` is even, `k = n_last/2`):
///   divide by `prod(s)` only;
/// - interior (`k = 1..K-2` for even `n_last`, or `k = 1..K-1` for odd):
///   multiply by `2 / prod(s)`.
///
/// Same Hermitian-doubling correction as 1-D `IrfftBackward`, applied along
/// the truncated axis.
#[derive(Debug)]
pub struct IrfftnBackward<T: Float> {
    input: Tensor<T>,
    s: Option<Vec<usize>>,
    axes: Option<Vec<isize>>,
    /// Length of the last transform axis in the real output (so we can
    /// detect Nyquist parity).
    last_axis_n: usize,
    /// Logical index of the last transform axis in the original Hermitian
    /// input (i.e., in the rfftn output layout, half-spectrum dim).
    last_axis_logical: usize,
    /// `prod(s)` (retained for the grandfathered `new` ctor; the threaded-norm
    /// backward folds normalization into `adjoint_norm(norm)` instead).
    #[allow(dead_code, reason = "retained for the grandfathered `new` ctor API")]
    norm_n: usize,
    /// Forward normalization mode.
    norm: FftNorm,
}

impl<T: Float> IrfftnBackward<T> {
    pub fn new(
        input: Tensor<T>,
        s: Option<Vec<usize>>,
        axes: Option<Vec<isize>>,
        last_axis_n: usize,
        last_axis_logical: usize,
        norm_n: usize,
    ) -> Self {
        Self::new_norm(
            input,
            s,
            axes,
            last_axis_n,
            last_axis_logical,
            norm_n,
            FftNorm::Backward,
        )
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors the forward's persisted state; a struct-literal \
                  wrapper would not reduce the captured fields"
    )]
    pub fn new_norm(
        input: Tensor<T>,
        s: Option<Vec<usize>>,
        axes: Option<Vec<isize>>,
        last_axis_n: usize,
        last_axis_logical: usize,
        norm_n: usize,
        norm: FftNorm,
    ) -> Self {
        Self {
            input,
            s,
            axes,
            last_axis_n,
            last_axis_logical,
            norm_n,
            norm,
        }
    }
}

impl<T: Float> GradFn<T> for IrfftnBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let grad_input = if self.input.requires_grad() {
            let device = grad_output.device();
            // 1. rfftn with `adjoint_norm(norm)`: ferray's forward-direction
            //    scaling carries the normalization. For norm="backward",
            //    adjoint=Forward gives forward-scale 1/prod(s), reproducing the
            //    old `rfftn(Backward) * (1/prod(s))`; "ortho"→Ortho gives
            //    1/sqrt(prod(s)), etc.
            let f = fft::rfftn_norm(
                grad_output,
                self.s.as_deref(),
                self.axes.as_deref(),
                adjoint_norm(self.norm),
            )?;
            let f_shape = f.shape().to_vec();
            let f_data = f.data_vec()?;

            // 2. Compute strides for the output of rfftn (which is the same
            //    shape as the input to irfftn, i.e., what the forward took).
            //    The trailing axis is the complex pair. The half-spectrum is
            //    at index `last_axis_logical` in the layout that excludes the
            //    trailing 2. The Hermitian-doubling correction (factor 2 on
            //    interior freqs) is applied on top of the already-normalized
            //    `f`.
            let two = T::from(2.0).unwrap();
            // K (half-spectrum length on the truncated axis).
            let k = f_shape[self.last_axis_logical];
            let n_last = self.last_axis_n;

            // Iterate every complex pair and apply the right factor.
            let strides_logical = row_major_strides(&f_shape[..f_shape.len() - 1]);
            let logical_total: usize = crate::shape::numel(&f_shape[..f_shape.len() - 1]);
            let mut out = vec![T::from(0.0).unwrap(); f_data.len()];
            for flat in 0..logical_total {
                let mut rem = flat;
                let mut idx = vec![0usize; strides_logical.len()];
                for d in 0..idx.len() {
                    idx[d] = rem / strides_logical[d];
                    rem %= strides_logical[d];
                }
                let kk = idx[self.last_axis_logical];
                // Boundary: kk == 0 always; kk == K-1 only when n_last is even.
                // `f` is already normalized; only the Hermitian-doubling
                // factor (1 boundary / 2 interior) remains.
                let is_boundary = kk == 0 || (n_last.is_multiple_of(2) && kk == k - 1);
                let factor = if is_boundary {
                    T::from(1.0).unwrap()
                } else {
                    two
                };
                let pair_offset = flat * 2;
                out[pair_offset] = f_data[pair_offset] * factor;
                out[pair_offset + 1] = f_data[pair_offset + 1] * factor;
            }
            let t = Tensor::from_storage(TensorStorage::cpu(out), f_shape, false)?;
            Some(if device.is_cuda() { t.to(device)? } else { t })
        } else {
            None
        };
        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "IrfftnBackward"
    }
}

// ---------------------------------------------------------------------------
// HfftBackward / IhfftBackward — Hermitian FFT backward.
// ---------------------------------------------------------------------------
//
// hfft maps Hermitian-symmetric complex `[..., n/2+1, 2]` → real `[..., n]`.
// ihfft is the inverse: real `[..., n]` → Hermitian complex `[..., n/2+1, 2]`.
//
// VJPs (matching torch.fft.hfft / ihfft):
//   y = hfft(x, n)  → grad_x = ihfft(grad_y, n=input_n)
//   y = ihfft(x, n) → grad_x = hfft(grad_y, n=input_n)

/// Backward for `y = hfft(x, n)` (Hermitian complex `[..., K, 2]` → real
/// `[..., n]`).
///
/// Forward (FftNorm::Backward): `hfft(x, N) = irfft_unnormalized(conj(x), N)`,
/// i.e., `y[n] = real(sum_{k=0..N-1} conj(x_full[k]) exp(+2π i k n / N))` with
/// no `1/N` scaling. Expanding using the Hermitian symmetry of `x`,
///
/// ```text
///   y[n] = x_re[0]
///        + (-1)^n * x_re[N/2]                                   (only for even N)
///        + 2 * sum_{k=1..N/2-1} (x_re[k] cos(2π k n/N) + x_im[k] sin(2π k n/N))
/// ```
///
/// VJP from real `grad_y` to Hermitian complex `grad_x` (shape `[..., K, 2]`),
/// with `F = rfft(grad_y, N)` (unnormalized, so `re(F[k]) = sum_n grad_y[n]
/// cos(2π k n/N)`, `im(F[k]) = -sum_n grad_y[n] sin(2π k n/N)`):
///
/// - boundary: `grad_x_re[0]   = re(F[0])`,    `grad_x_im[0]   = 0`
/// - boundary: `grad_x_re[N/2] = re(F[N/2])`,  `grad_x_im[N/2] = 0`   (even N)
/// - interior: `grad_x_re[k]   = 2 * re(F[k])`
/// - interior: `grad_x_im[k]   = -2 * im(F[k])`  (sign: `+sin → -im(F)`)
///
/// Concretely: `grad_x = conj(rfft(grad_y, N))` for boundary entries,
/// `grad_x = 2 * conj(rfft(grad_y, N))` for interior.
///
/// The previous implementation called `fft::ihfft(grad_y, n_forward)`, which
/// is `conj(rfft(grad_y))/N`, missing both the `* N` (FftNorm::Backward
/// hfft is unnormalized forward) and the interior `* 2` correction.
#[derive(Debug)]
pub struct HfftBackward<T: Float> {
    input: Tensor<T>,
    /// Length of the original Hermitian spectrum (input's second-to-last dim).
    input_n: usize,
    /// Length of the real signal produced by hfft (so we know Nyquist parity).
    output_n: usize,
}

impl<T: Float> HfftBackward<T> {
    pub fn new(input: Tensor<T>, input_n: usize, output_n: usize) -> Self {
        Self {
            input,
            input_n,
            output_n,
        }
    }
}

impl<T: Float> GradFn<T> for HfftBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let grad_input = if self.input.requires_grad() {
            let device = grad_output.device();
            let n = self.output_n;
            let k = self.input_n; // K = N/2 + 1 for even N (or (N+1)/2 for odd).
            // F = unnormalized rfft of grad_y.
            let f = fft::rfft(grad_output, Some(n))?;
            let f_data = f.data_vec()?;
            let f_shape = f.shape().to_vec();
            let total_pairs = f_data.len() / 2;
            let batch_size = total_pairs / k;
            let two = T::from(2.0).unwrap();
            let mut out = Vec::with_capacity(f_data.len());
            for b in 0..batch_size {
                for kk in 0..k {
                    // Boundary: kk == 0; kk == K-1 only when n is even.
                    let is_boundary = kk == 0 || (n.is_multiple_of(2) && kk == k - 1);
                    let factor = if is_boundary {
                        T::from(1.0).unwrap()
                    } else {
                        two
                    };
                    let re = f_data[(b * k + kk) * 2];
                    let im = f_data[(b * k + kk) * 2 + 1];
                    // grad_x = factor * conj(F[k]). conj negates the imag part.
                    out.push(re * factor);
                    out.push(-im * factor);
                }
            }
            let t = Tensor::from_storage(TensorStorage::cpu(out), f_shape, false)?;
            Some(if device.is_cuda() { t.to(device)? } else { t })
        } else {
            None
        };
        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "HfftBackward"
    }
}

/// Backward for `y = ihfft(x, n)` (real `[..., N]` → Hermitian complex
/// `[..., K, 2]`).
///
/// Forward (FftNorm::Backward): `ihfft(x, N) = conj(rfft(x, N)) / N`. As a
/// real-linear map,
///
/// ```text
///   y_re[k] = (1/N) * sum_n x[n] cos(2π k n/N)
///   y_im[k] = (1/N) * sum_n x[n] sin(2π k n/N)
/// ```
///
/// The Hermitian half-spectrum has K = N/2 + 1 entries. The cotangent on
/// real `x` is the partial unnormalized inverse over that half:
///
/// ```text
///   grad_x[n] = (1/N) * sum_{k=0..K-1} (
///                  grad_y_re[k] cos(2π k n/N) + grad_y_im[k] sin(2π k n/N)
///              )
///            = (1/N) * real( sum_{k=0..K-1} conj(grad_y[k]) exp(+2π i k n/N) )
/// ```
///
/// (Because `conj(grad_y[k]) = grad_y_re[k] - i grad_y_im[k]` and
/// `exp(+i θ) = cos θ + i sin θ`, the real part picks up the desired
/// sign-correct combination.)
///
/// Implementation: zero-pad `conj(grad_y)` along the freq axis from `K` to
/// `N`, run our normalized `fft::ifft` (which already supplies the `1/N`),
/// take the real part. No further scaling needed.
///
/// The previous implementation called `fft::hfft(grad_y, input_n)`, which
/// is the unnormalized inverse with conj — wrong by an `N` factor and by
/// the absent boundary/interior treatment.
#[derive(Debug)]
pub struct IhfftBackward<T: Float> {
    input: Tensor<T>,
    /// Length of the original real signal (input's last dim).
    input_n: usize,
}

impl<T: Float> IhfftBackward<T> {
    pub fn new(input: Tensor<T>, input_n: usize) -> Self {
        Self { input, input_n }
    }
}

impl<T: Float> GradFn<T> for IhfftBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let grad_input = if self.input.requires_grad() {
            let device = grad_output.device();
            let n = self.input_n;
            let go_shape = grad_output.shape();
            if go_shape.len() < 2 || go_shape[go_shape.len() - 1] != 2 {
                return Err(crate::error::FerrotorchError::InvalidArgument {
                    message: format!(
                        "IhfftBackward: grad_output must have trailing complex pair, got {go_shape:?}"
                    ),
                });
            }
            let k = go_shape[go_shape.len() - 2];
            let batch_shape = &go_shape[..go_shape.len() - 2];
            let batch_size: usize = crate::shape::numel(batch_shape).max(1);
            let go_data = grad_output.data_vec()?;

            // Zero-pad conj(grad_y) from [..., K, 2] to [..., N, 2].
            let mut padded = vec![T::from(0.0).unwrap(); batch_size * n * 2];
            for b in 0..batch_size {
                let src_off = b * k * 2;
                let dst_off = b * n * 2;
                let copy_pairs = k.min(n);
                for kk in 0..copy_pairs {
                    let re = go_data[src_off + kk * 2];
                    let im = go_data[src_off + kk * 2 + 1];
                    padded[dst_off + kk * 2] = re;
                    padded[dst_off + kk * 2 + 1] = -im; // conj
                }
            }
            let mut padded_shape = batch_shape.to_vec();
            padded_shape.push(n);
            padded_shape.push(2);
            let padded_t = Tensor::from_storage(TensorStorage::cpu(padded), padded_shape, false)?;

            // Normalized ifft: divides by N — exactly the 1/N we want.
            let inv = fft::ifft(&padded_t, Some(n))?;
            let inv_data = inv.data_vec()?;
            // Take real part: drop trailing 2.
            let mut grad_x_data = Vec::with_capacity(batch_size * n);
            for b in 0..batch_size {
                for nn in 0..n {
                    grad_x_data.push(inv_data[b * n * 2 + nn * 2]);
                }
            }
            let mut out_shape = batch_shape.to_vec();
            out_shape.push(n);
            let t = Tensor::from_storage(TensorStorage::cpu(grad_x_data), out_shape, false)?;
            Some(if device.is_cuda() { t.to(device)? } else { t })
        } else {
            None
        };
        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "IhfftBackward"
    }
}

// ---------------------------------------------------------------------------
// Differentiable forward wrappers — N-D + Hermitian (#580)
// ---------------------------------------------------------------------------

/// Compute the product of transform-axis lengths used for normalization.
/// Mirrors how the forward pass would interpret `s` / `axes`:
///   - If `s` is given, multiply its entries.
///   - Else if `axes` is given, multiply the input's lengths along those axes.
///   - Else multiply the inner dims (excluding the trailing complex pair).
fn fftn_norm_n<T: Float>(input: &Tensor<T>, s: Option<&[usize]>, axes: Option<&[isize]>) -> usize {
    if let Some(s_slice) = s {
        return crate::shape::numel(s_slice).max(1);
    }
    let shape = input.shape();
    let ndim = shape.len();
    if let Some(axes_slice) = axes {
        let mut dims = Vec::with_capacity(axes_slice.len());
        for &a in axes_slice {
            // Resolve negative axes against `ndim - 1` (excluding trailing
            // complex pair).
            let logical_ndim = ndim.saturating_sub(1);
            let resolved = if a < 0 {
                (logical_ndim as isize + a) as usize
            } else {
                a as usize
            };
            dims.push(shape[resolved]);
        }
        return crate::shape::numel(&dims).max(1);
    }
    // Default: all inner dims (skip the trailing 2).
    if ndim < 2 {
        1
    } else {
        crate::shape::numel(&shape[..ndim - 1]).max(1)
    }
}

/// Same as [`fftn_norm_n`] but for real inputs: there is no trailing complex
/// pair, so all dims except the leading batch are candidates.
fn rfftn_norm_n<T: Float>(input: &Tensor<T>, s: Option<&[usize]>, axes: Option<&[isize]>) -> usize {
    if let Some(s_slice) = s {
        return crate::shape::numel(s_slice).max(1);
    }
    let shape = input.shape();
    let ndim = shape.len();
    if let Some(axes_slice) = axes {
        let mut dims = Vec::with_capacity(axes_slice.len());
        for &a in axes_slice {
            let resolved = if a < 0 {
                (ndim as isize + a) as usize
            } else {
                a as usize
            };
            dims.push(shape[resolved]);
        }
        return crate::shape::numel(&dims).max(1);
    }
    crate::shape::numel(shape).max(1)
}

/// Differentiable N-D FFT (default `norm`). Attaches `FftnBackward`.
pub fn fftn_differentiable<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
) -> FerrotorchResult<Tensor<T>> {
    fftn_differentiable_norm(input, s, axes, FftNorm::Backward)
}

/// Differentiable N-D FFT with explicit `norm` (#1294). Matches
/// `torch.fft.fftn`; `axes` is torch's `dim`.
pub fn fftn_differentiable_norm<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: FftNorm,
) -> FerrotorchResult<Tensor<T>> {
    let result = fft::fftn_norm(input, s, axes, norm)?;

    if is_grad_enabled() && input.requires_grad() {
        let norm_n = fftn_norm_n(input, s, axes);
        let grad_fn = Arc::new(FftnBackward::new_norm(
            input.clone(),
            s.map(|v| v.to_vec()),
            axes.map(|v| v.to_vec()),
            norm_n,
            norm,
        ));
        tensor_from_fft_operation(result, grad_fn)
    } else {
        Ok(result)
    }
}

/// Differentiable N-D inverse FFT (default `norm`). Attaches `IfftnBackward`.
pub fn ifftn_differentiable<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
) -> FerrotorchResult<Tensor<T>> {
    ifftn_differentiable_norm(input, s, axes, FftNorm::Backward)
}

/// Differentiable N-D inverse FFT with explicit `norm` (#1294).
pub fn ifftn_differentiable_norm<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: FftNorm,
) -> FerrotorchResult<Tensor<T>> {
    let result = fft::ifftn_norm(input, s, axes, norm)?;

    if is_grad_enabled() && input.requires_grad() {
        let norm_n = fftn_norm_n(input, s, axes);
        let grad_fn = Arc::new(IfftnBackward::new_norm(
            input.clone(),
            s.map(|v| v.to_vec()),
            axes.map(|v| v.to_vec()),
            norm_n,
            norm,
        ));
        tensor_from_fft_operation(result, grad_fn)
    } else {
        Ok(result)
    }
}

/// Differentiable N-D real FFT (default `norm`). Attaches `RfftnBackward`.
pub fn rfftn_differentiable<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
) -> FerrotorchResult<Tensor<T>> {
    rfftn_differentiable_norm(input, s, axes, FftNorm::Backward)
}

/// Differentiable N-D real FFT with explicit `norm` (#1294). Matches
/// `torch.fft.rfftn`.
pub fn rfftn_differentiable_norm<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: FftNorm,
) -> FerrotorchResult<Tensor<T>> {
    let _ = rfftn_norm_n::<T>; // keep helper available for symmetry; not needed in fwd
    let result = fft::rfftn_norm(input, s, axes, norm)?;

    if is_grad_enabled() && input.requires_grad() {
        // Backward needs the original real-input shape along the transform
        // axes. We pass the input's shape segment so irfftn can reconstruct.
        let s_back: Vec<usize> = match (s, axes) {
            (Some(s_slice), _) => s_slice.to_vec(),
            (None, Some(axes_slice)) => {
                let shape = input.shape();
                axes_slice
                    .iter()
                    .map(|&a| {
                        let resolved = if a < 0 {
                            (shape.len() as isize + a) as usize
                        } else {
                            a as usize
                        };
                        shape[resolved]
                    })
                    .collect()
            }
            (None, None) => input.shape().to_vec(),
        };
        // Resolve the last transform axis (logical, in real-input space).
        let in_shape = input.shape();
        let last_axis_logical = match axes {
            Some(axes_slice) => {
                let a = *axes_slice.last().unwrap();
                if a < 0 {
                    (in_shape.len() as isize + a) as usize
                } else {
                    a as usize
                }
            }
            None => in_shape.len() - 1,
        };
        let last_axis_n = s_back
            .last()
            .copied()
            .unwrap_or(in_shape[last_axis_logical]);
        let norm_n: usize = crate::shape::numel(&s_back).max(1);
        let out_shape = result.shape().to_vec();
        let grad_fn = Arc::new(RfftnBackward::new_norm(
            input.clone(),
            Some(s_back),
            axes.map(|v| v.to_vec()),
            out_shape,
            last_axis_n,
            last_axis_logical,
            norm_n,
            norm,
        ));
        tensor_from_fft_operation(result, grad_fn)
    } else {
        Ok(result)
    }
}

/// Differentiable N-D inverse real FFT (default `norm`). Attaches `IrfftnBackward`.
pub fn irfftn_differentiable<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
) -> FerrotorchResult<Tensor<T>> {
    irfftn_differentiable_norm(input, s, axes, FftNorm::Backward)
}

/// Differentiable N-D inverse real FFT with explicit `norm` (#1294). Matches
/// `torch.fft.irfftn`.
pub fn irfftn_differentiable_norm<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: FftNorm,
) -> FerrotorchResult<Tensor<T>> {
    let result = fft::irfftn_norm(input, s, axes, norm)?;

    if is_grad_enabled() && input.requires_grad() {
        // The forward output length along each transform axis becomes the
        // original real shape; back-pass uses the same `s` to reconstruct.
        let s_back: Vec<usize> = match s {
            Some(s_slice) => s_slice.to_vec(),
            None => result.shape().to_vec(),
        };
        // Resolve the last transform axis. Layout: input is the
        // Hermitian-truncated complex tensor with trailing 2; for the real
        // output (`result`), the last freq axis is the last entry of `axes`
        // (or the last axis if `axes` is `None`).
        let res_shape = result.shape();
        let last_axis_logical_real = match axes {
            Some(axes_slice) => {
                let a = *axes_slice.last().unwrap();
                if a < 0 {
                    (res_shape.len() as isize + a) as usize
                } else {
                    a as usize
                }
            }
            None => res_shape.len() - 1,
        };
        // For the input (`x` to irfftn), the half-spectrum axis is at the
        // same logical index since irfftn's input layout is `[..., 2]` and
        // the freq axis maps 1:1 with the real output's last transform axis.
        let last_axis_logical = last_axis_logical_real;
        let last_axis_n = *s_back.last().unwrap_or(&res_shape[last_axis_logical_real]);
        let norm_n: usize = crate::shape::numel(&s_back).max(1);
        let grad_fn = Arc::new(IrfftnBackward::new_norm(
            input.clone(),
            Some(s_back),
            axes.map(|v| v.to_vec()),
            last_axis_n,
            last_axis_logical,
            norm_n,
            norm,
        ));
        tensor_from_fft_operation(result, grad_fn)
    } else {
        Ok(result)
    }
}

/// Differentiable Hermitian FFT (complex spectrum → real signal). Attaches
/// `HfftBackward` when grad is needed.
pub fn hfft_differentiable<T: Float>(
    input: &Tensor<T>,
    n: Option<usize>,
) -> FerrotorchResult<Tensor<T>> {
    let shape = input.shape();
    let input_n = shape[shape.len() - 2];
    let result = fft::hfft(input, n)?;
    // hfft output's last dim is the real-signal length N. Persist it so the
    // backward can detect Nyquist parity (boundary vs. interior k).
    let output_n = *result.shape().last().unwrap();

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(HfftBackward::new(input.clone(), input_n, output_n));
        tensor_from_fft_operation(result, grad_fn)
    } else {
        Ok(result)
    }
}

/// Differentiable inverse Hermitian FFT (real signal → Hermitian spectrum).
/// Attaches `IhfftBackward` when grad is needed.
pub fn ihfft_differentiable<T: Float>(
    input: &Tensor<T>,
    n: Option<usize>,
) -> FerrotorchResult<Tensor<T>> {
    let shape = input.shape();
    // FFT length N: defaults to the input's last dim (real signal length).
    // When `n` is supplied, ihfft truncates/pads the input before the
    // transform — the backward reconstructs grad over that same `N`.
    let input_n = n.unwrap_or(*shape.last().unwrap());
    let result = fft::ihfft(input, n)?;

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(IhfftBackward::new(input.clone(), input_n));
        tensor_from_fft_operation(result, grad_fn)
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Fft2Backward / Ifft2Backward — 2-D complex FFT backward (#1300).
// ---------------------------------------------------------------------------
//
// `torch.fft.fft2` / `ifft2` are the 2-D specializations of `fftn` / `ifftn`
// over the trailing two spatial axes (`aten::fft_fft2_symint` literally does
// `return fft_fftn_symint(self, s, dim, norm)` at SpectralOps.cpp:644-652).
// The VJPs are therefore the FftnBackward / IfftnBackward identities with the
// transform set fixed to the last two axes:
//   y = fft2(x)   → grad_x = (rows*cols) * ifft2(grad_y)
//   y = ifft2(x)  → grad_x = fft2(grad_y) / (rows*cols)
// `norm_n = rows * cols` is the product of the two transform-axis lengths,
// captured from the forward input shape (`shape[ndim-3] * shape[ndim-2]`,
// since the trailing axis of size 2 is the complex pair).

/// Backward for `y = fft2(x)` (un-normalized 2-D forward DFT over the last two
/// spatial axes). VJP: `grad_x = norm_n * ifft2(grad_y)` (our `ifft2` divides
/// by `rows*cols`; multiplying by `norm_n` undoes that to yield the
/// un-normalized inverse).
#[derive(Debug)]
pub struct Fft2Backward<T: Float> {
    input: Tensor<T>,
    /// Product of the two transform-axis lengths (rows * cols).
    norm_n: usize,
}

impl<T: Float> Fft2Backward<T> {
    pub fn new(input: Tensor<T>, norm_n: usize) -> Self {
        Self { input, norm_n }
    }
}

impl<T: Float> GradFn<T> for Fft2Backward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let grad_input = if self.input.requires_grad() {
            let device = grad_output.device();
            let inv = fft::ifft2(grad_output)?;
            let scale = T::from(self.norm_n).unwrap();
            if device.is_cuda() {
                Some(crate::autograd::no_grad::no_grad(|| {
                    scale_cuda_tensor(&inv, scale)
                })?)
            } else {
                let inv_data = inv.data_vec()?;
                let scaled: Vec<T> = inv_data.iter().map(|&v| v * scale).collect();
                let t =
                    Tensor::from_storage(TensorStorage::cpu(scaled), inv.shape().to_vec(), false)?;
                Some(t)
            }
        } else {
            None
        };
        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "Fft2Backward"
    }
}

/// Backward for `y = ifft2(x)` (1/(rows*cols)-normalized 2-D inverse). VJP:
/// `grad_x = fft2(grad_y) / norm_n`.
#[derive(Debug)]
pub struct Ifft2Backward<T: Float> {
    input: Tensor<T>,
    norm_n: usize,
}

impl<T: Float> Ifft2Backward<T> {
    pub fn new(input: Tensor<T>, norm_n: usize) -> Self {
        Self { input, norm_n }
    }
}

impl<T: Float> GradFn<T> for Ifft2Backward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let grad_input = if self.input.requires_grad() {
            let device = grad_output.device();
            let fwd = fft::fft2(grad_output)?;
            let scale = T::from(1.0).unwrap() / T::from(self.norm_n).unwrap();
            if device.is_cuda() {
                Some(crate::autograd::no_grad::no_grad(|| {
                    scale_cuda_tensor(&fwd, scale)
                })?)
            } else {
                let fwd_data = fwd.data_vec()?;
                let scaled: Vec<T> = fwd_data.iter().map(|&v| v * scale).collect();
                let t =
                    Tensor::from_storage(TensorStorage::cpu(scaled), fwd.shape().to_vec(), false)?;
                Some(t)
            }
        } else {
            None
        };
        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "Ifft2Backward"
    }
}

/// Product of the two transform-axis lengths for `fft2` / `ifft2`. The forward
/// kernels are fixed to the trailing two spatial axes; for input shape
/// `[..., rows, cols, 2]` this is `rows * cols`.
fn fft2_norm_n<T: Float>(input: &Tensor<T>) -> usize {
    let shape = input.shape();
    let ndim = shape.len();
    if ndim < 3 {
        return 1;
    }
    (shape[ndim - 3] * shape[ndim - 2]).max(1)
}

/// Differentiable 2-D FFT (default `s`/`dim`/`norm`). Attaches `Fft2Backward`.
pub fn fft2_differentiable<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let result = fft::fft2(input)?;

    if is_grad_enabled() && input.requires_grad() {
        let norm_n = fft2_norm_n(input);
        let grad_fn = Arc::new(Fft2Backward::new(input.clone(), norm_n));
        tensor_from_fft_operation(result, grad_fn)
    } else {
        Ok(result)
    }
}

/// Differentiable 2-D FFT with explicit `s` / `dim` / `norm` (#1294).
///
/// `torch.fft.fft2` delegates to `fft_fftn_symint`
/// (`aten/src/ATen/native/SpectralOps.cpp:644-652`); the default last-two-axes
/// / backward-norm case keeps the cheaper [`Fft2Backward`] (GPU-capable),
/// while any explicit `s`/`dim`/`norm` routes through
/// [`fftn_differentiable_norm`] (op_db emits `dim=[-3,-2,-1]` for `fft2`,
/// which torch treats as an N-D transform).
pub fn fft2_differentiable_norm<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    dim: Option<&[isize]>,
    norm: FftNorm,
) -> FerrotorchResult<Tensor<T>> {
    if s.is_none() && dim.is_none() && norm == FftNorm::Backward {
        return fft2_differentiable(input);
    }
    // Default axes for fft2 are the last two; fftn over those axes is identical.
    let default_axes: [isize; 2] = [-2, -1];
    let axes = dim.unwrap_or(&default_axes);
    fftn_differentiable_norm(input, s, Some(axes), norm)
}

/// Differentiable 2-D inverse FFT (default `s`/`dim`/`norm`). Attaches `Ifft2Backward`.
pub fn ifft2_differentiable<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let result = fft::ifft2(input)?;

    if is_grad_enabled() && input.requires_grad() {
        let norm_n = fft2_norm_n(input);
        let grad_fn = Arc::new(Ifft2Backward::new(input.clone(), norm_n));
        tensor_from_fft_operation(result, grad_fn)
    } else {
        Ok(result)
    }
}

/// Differentiable 2-D inverse FFT with explicit `s` / `dim` / `norm` (#1294).
pub fn ifft2_differentiable_norm<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    dim: Option<&[isize]>,
    norm: FftNorm,
) -> FerrotorchResult<Tensor<T>> {
    if s.is_none() && dim.is_none() && norm == FftNorm::Backward {
        return ifft2_differentiable(input);
    }
    let default_axes: [isize; 2] = [-2, -1];
    let axes = dim.unwrap_or(&default_axes);
    ifftn_differentiable_norm(input, s, Some(axes), norm)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::TensorStorage;

    fn leaf(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
    }

    fn no_grad_leaf(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "length mismatch: {} vs {}",
            actual.len(),
            expected.len()
        );
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "index {i}: {a} vs {e} (diff {})",
                (a - e).abs()
            );
        }
    }

    #[test]
    fn fft_differentiable_attaches_grad_fn() {
        // Complex input [4, 2] with requires_grad.
        let input = leaf(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[4, 2]);
        let result = fft_differentiable(&input, None).unwrap();
        assert!(result.grad_fn().is_some());
        assert_eq!(result.grad_fn().unwrap().name(), "FftBackward");
    }

    #[test]
    fn fft_differentiable_no_grad_when_not_needed() {
        let input = no_grad_leaf(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[4, 2]);
        let result = fft_differentiable(&input, None).unwrap();
        assert!(result.grad_fn().is_none());
    }

    #[test]
    fn fft_backward_identity_check() {
        // For FFT of an impulse [1,0,0,0] -> [1,1,1,1] (all real).
        // grad_output = ones_like(output) = [[1,0],[1,0],[1,0],[1,0]].
        // grad_input = n * ifft(grad_output).
        // ifft([1,1,1,1]) = [1,0,0,0] (impulse).
        // So grad_input = 4 * [1,0,0,0] = [4,0,0,0].
        let input = leaf(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[4, 2]);
        let result = fft_differentiable(&input, None).unwrap();

        let grad_out = no_grad_leaf(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], &[4, 2]);
        let grads = result.grad_fn().unwrap().backward(&grad_out).unwrap();
        assert!(grads[0].is_some());

        let g = grads[0].as_ref().unwrap();
        assert_eq!(g.shape(), &[4, 2]);
        let gd = g.data().unwrap();
        // Should be [4, 0, 0, 0, 0, 0, 0, 0].
        assert_close(gd, &[4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1e-10);
    }

    #[test]
    fn ifft_backward_identity_check() {
        // ifft([1,1,1,1]) = [1,0,0,0].
        // grad_output = [[1,0],[0,0],[0,0],[0,0]].
        // grad_input = fft(grad_output) / n.
        // fft([1,0,0,0]) = [1,1,1,1].
        // grad_input = [1,1,1,1] / 4 = [0.25, 0.25, 0.25, 0.25].
        let input = leaf(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], &[4, 2]);
        let result = ifft_differentiable(&input, None).unwrap();

        let grad_out = no_grad_leaf(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[4, 2]);
        let grads = result.grad_fn().unwrap().backward(&grad_out).unwrap();
        assert!(grads[0].is_some());

        let g = grads[0].as_ref().unwrap();
        let gd = g.data().unwrap();
        // Each complex value should be (0.25, 0.0).
        assert_close(gd, &[0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0], 1e-10);
    }

    #[test]
    fn rfft_differentiable_attaches_grad_fn() {
        let input = leaf(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let result = rfft_differentiable(&input, None).unwrap();
        assert!(result.grad_fn().is_some());
        assert_eq!(result.grad_fn().unwrap().name(), "RfftBackward");
    }

    #[test]
    fn irfft_differentiable_attaches_grad_fn() {
        // Input: [3, 2] complex -> irfft -> [4] real.
        let input = leaf(&[10.0, 0.0, -2.0, 2.0, -2.0, 0.0], &[3, 2]);
        let result = irfft_differentiable(&input, Some(4)).unwrap();
        assert!(result.grad_fn().is_some());
        assert_eq!(result.grad_fn().unwrap().name(), "IrfftBackward");
    }

    #[test]
    fn no_grad_context_disables_tracking() {
        let input = leaf(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[4, 2]);
        let result =
            crate::autograd::no_grad::no_grad(|| fft_differentiable(&input, None).unwrap());
        assert!(result.grad_fn().is_none());
    }

    // -----------------------------------------------------------------------
    // N-D FFT differentiable wrappers (#580)
    // -----------------------------------------------------------------------

    #[test]
    fn fftn_differentiable_attaches_grad_fn() {
        // 2x2 complex input: [[1+0i, 0+0i], [0+0i, 0+0i]] → flat [2, 2, 2].
        let input = leaf(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[2, 2, 2]);
        let result = fftn_differentiable(&input, None, None).unwrap();
        assert!(result.grad_fn().is_some());
        assert_eq!(result.grad_fn().unwrap().name(), "FftnBackward");
    }

    #[test]
    fn ifftn_differentiable_attaches_grad_fn() {
        let input = leaf(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], &[2, 2, 2]);
        let result = ifftn_differentiable(&input, None, None).unwrap();
        assert!(result.grad_fn().is_some());
        assert_eq!(result.grad_fn().unwrap().name(), "IfftnBackward");
    }

    #[test]
    fn fftn_no_grad_when_not_needed() {
        let input = no_grad_leaf(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[2, 2, 2]);
        let result = fftn_differentiable(&input, None, None).unwrap();
        assert!(result.grad_fn().is_none());
    }

    #[test]
    fn fftn_backward_returns_real_grad_for_impulse() {
        // 2x2 impulse: real [[1,0],[0,0]] (encoded complex as
        // [[1+0i, 0+0i], [0+0i, 0+0i]]). fftn → all-ones 2x2 complex
        // (DFT-2D of a corner impulse). grad_y = ones → grad_x =
        // prod_s * ifftn(ones) = 4 * impulse / 4 → impulse_complex.
        let input = leaf(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[2, 2, 2]);
        let result = fftn_differentiable(&input, None, None).unwrap();
        // grad_y = ones (4 complex pairs).
        let grad_out = no_grad_leaf(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], &[2, 2, 2]);
        let grads = result.grad_fn().unwrap().backward(&grad_out).unwrap();
        let g = grads[0].as_ref().unwrap();
        // Expected: 4 * ifftn(ones) over a 2x2 grid → 4 * impulse / 4 = impulse.
        assert_close(
            g.data().unwrap(),
            &[4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1e-9,
        );
    }

    #[test]
    fn rfftn_differentiable_attaches_grad_fn() {
        // Real 2x2 input → rfftn → [2, 2, 2] complex (n/2+1 along last).
        let input = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let result = rfftn_differentiable(&input, None, None).unwrap();
        assert!(result.grad_fn().is_some());
        assert_eq!(result.grad_fn().unwrap().name(), "RfftnBackward");
    }

    #[test]
    fn irfftn_differentiable_attaches_grad_fn() {
        // Hermitian-shaped complex input [2, 2, 2]: 2 batch × 2 freq × complex.
        let input = leaf(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], &[2, 2, 2]);
        let result = irfftn_differentiable(&input, Some(&[2, 2]), None).unwrap();
        assert!(result.grad_fn().is_some());
        assert_eq!(result.grad_fn().unwrap().name(), "IrfftnBackward");
    }

    #[test]
    fn hfft_differentiable_attaches_grad_fn() {
        // Hermitian spectrum [3, 2] → real [4]. n=4 means input_n=3 (n/2+1).
        let input = leaf(&[10.0, 0.0, -2.0, 2.0, -2.0, 0.0], &[3, 2]);
        let result = hfft_differentiable(&input, Some(4)).unwrap();
        assert!(result.grad_fn().is_some());
        assert_eq!(result.grad_fn().unwrap().name(), "HfftBackward");
    }

    #[test]
    fn ihfft_differentiable_attaches_grad_fn() {
        let input = leaf(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let result = ihfft_differentiable(&input, None).unwrap();
        assert!(result.grad_fn().is_some());
        assert_eq!(result.grad_fn().unwrap().name(), "IhfftBackward");
    }

    #[test]
    fn fftn_norm_n_default_inner_dims() {
        // shape [2, 3, 4, 2] (last dim is complex pair) → norm_n = 2*3*4 = 24.
        let input = no_grad_leaf(&vec![0.0; 2 * 3 * 4 * 2], &[2, 3, 4, 2]);
        let n = fftn_norm_n(&input, None, None);
        assert_eq!(n, 2 * 3 * 4);
    }

    #[test]
    fn fftn_norm_n_with_explicit_s() {
        let input = no_grad_leaf(&[0.0; 8 * 2], &[2, 2, 2, 2]);
        let n = fftn_norm_n(&input, Some(&[3, 5]), None);
        assert_eq!(n, 15);
    }

    #[test]
    fn fftn_norm_n_with_axes() {
        // Axes = [1, 2] → norm_n = shape[1] * shape[2] = 3 * 4 = 12.
        let input = no_grad_leaf(&vec![0.0; 2 * 3 * 4 * 2], &[2, 3, 4, 2]);
        let n = fftn_norm_n(&input, None, Some(&[1, 2]));
        assert_eq!(n, 12);
    }

    // -----------------------------------------------------------------------
    // 2-D FFT differentiable wrappers (#1300)
    // -----------------------------------------------------------------------

    #[test]
    fn fft2_differentiable_attaches_grad_fn() {
        // 2x3 complex grid → [2, 3, 2].
        let input = leaf(&[0.0; 2 * 3 * 2], &[2, 3, 2]);
        let result = fft2_differentiable(&input).unwrap();
        assert!(result.grad_fn().is_some());
        assert_eq!(result.grad_fn().unwrap().name(), "Fft2Backward");
    }

    #[test]
    fn ifft2_differentiable_attaches_grad_fn() {
        let input = leaf(&[0.0; 2 * 3 * 2], &[2, 3, 2]);
        let result = ifft2_differentiable(&input).unwrap();
        assert!(result.grad_fn().is_some());
        assert_eq!(result.grad_fn().unwrap().name(), "Ifft2Backward");
    }

    #[test]
    fn fft2_no_grad_when_not_needed() {
        let input = no_grad_leaf(&[0.0; 2 * 3 * 2], &[2, 3, 2]);
        let result = fft2_differentiable(&input).unwrap();
        assert!(result.grad_fn().is_none());
    }

    #[test]
    fn fft2_norm_n_is_rows_times_cols() {
        // [4, 5, 2] → rows=4, cols=5 → norm_n=20.
        let input = no_grad_leaf(&vec![0.0; 4 * 5 * 2], &[4, 5, 2]);
        assert_eq!(fft2_norm_n(&input), 20);
        // Batched [2, 3, 4, 2] → rows=3, cols=4 → norm_n=12.
        let batched = no_grad_leaf(&vec![0.0; 2 * 3 * 4 * 2], &[2, 3, 4, 2]);
        assert_eq!(fft2_norm_n(&batched), 12);
    }

    #[test]
    fn fft2_backward_returns_grad_for_corner_impulse() {
        // 2x2 corner impulse encoded complex: [[1+0i,0],[0,0]] → flat [2,2,2].
        // fft2 of a corner impulse is the all-ones 2x2 complex grid.
        // grad_y = ones → grad_x = norm_n * ifft2(ones) = 4 * (impulse/4)
        //        = impulse. So grad_x = [4,0, 0,0, 0,0, 0,0]?  No: ifft2(ones)
        // over a 2x2 grid = corner impulse (value 1 at [0,0]); times norm_n=4
        // → [4,0] at the corner, zeros elsewhere.
        let input = leaf(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[2, 2, 2]);
        let result = fft2_differentiable(&input).unwrap();
        let grad_out = no_grad_leaf(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], &[2, 2, 2]);
        let grads = result.grad_fn().unwrap().backward(&grad_out).unwrap();
        let g = grads[0].as_ref().unwrap();
        assert_close(
            g.data().unwrap(),
            &[4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1e-9,
        );
    }

    // -----------------------------------------------------------------------
    // norm / dim threading in autograd (#1294)
    // -----------------------------------------------------------------------

    #[test]
    fn adjoint_norm_swaps_backward_forward_fixes_ortho() {
        // The adjoint (same fft_norm_mode int, flipped direction) maps
        // Backward<->Forward and Ortho->Ortho (derivatives.yaml:2960-2961 +
        // SpectralOpsUtils.h:15-19).
        assert_eq!(adjoint_norm(FftNorm::Backward), FftNorm::Forward);
        assert_eq!(adjoint_norm(FftNorm::Forward), FftNorm::Backward);
        assert_eq!(adjoint_norm(FftNorm::Ortho), FftNorm::Ortho);
    }

    #[test]
    fn fft_ortho_backward_is_ortho_inverse() {
        // For the unitary (ortho) FFT, the VJP equals the ortho inverse FFT of
        // grad_y (adjoint of a unitary map is its inverse). Check numerically:
        // grad_x = ifft(grad_y, ortho). Use an impulse grad_y so the expected
        // grad_x is the ortho IFFT of the impulse = constant 1/sqrt(n) bins.
        let input = leaf(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[4, 2]);
        let result = fft_differentiable_norm(&input, None, None, FftNorm::Ortho).unwrap();
        assert!(result.grad_fn().is_some());
        let grad_out = no_grad_leaf(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[4, 2]);
        let grads = result.grad_fn().unwrap().backward(&grad_out).unwrap();
        let g = grads[0].as_ref().unwrap();
        // ortho ifft of an impulse [1,0,0,0] = constant 1/sqrt(4) = 0.5 in each
        // real bin (imag 0).
        let gd = g.data().unwrap();
        for k in 0..4 {
            assert!((gd[k * 2] - 0.5).abs() < 1e-9, "bin {k} re = {}", gd[k * 2]);
            assert!(gd[k * 2 + 1].abs() < 1e-9, "bin {k} im");
        }
    }

    #[test]
    fn fft_differentiable_norm_attaches_grad_fn_for_dim() {
        // Non-default dim still attaches a FftBackward node.
        let input = leaf(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[2, 2, 2]);
        let result = fft_differentiable_norm(&input, None, Some(-2), FftNorm::Backward).unwrap();
        assert!(result.grad_fn().is_some());
        assert_eq!(result.grad_fn().unwrap().name(), "FftBackward");
    }

    #[test]
    fn rfft_differentiable_norm_grad_matches_default_on_last_axis() {
        // For the default last-axis / backward-norm case, the _norm wrapper
        // must produce the identical grad as the legacy rfft_differentiable
        // (it delegates to the cheaper 1-D RfftBackward).
        let input = leaf(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let r_default = rfft_differentiable(&input, None).unwrap();
        let input2 = leaf(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let r_norm = rfft_differentiable_norm(&input2, None, None, FftNorm::Backward).unwrap();
        assert_close(r_norm.data().unwrap(), r_default.data().unwrap(), 1e-12);
        // grad_y = ones over the half-spectrum.
        let half = r_default.shape()[0];
        let go: Vec<f64> = vec![1.0; half * 2];
        let grad_out = no_grad_leaf(&go, r_default.shape());
        let g1 = r_default.grad_fn().unwrap().backward(&grad_out).unwrap();
        let g2 = r_norm.grad_fn().unwrap().backward(&grad_out).unwrap();
        assert_close(
            g2[0].as_ref().unwrap().data().unwrap(),
            g1[0].as_ref().unwrap().data().unwrap(),
            1e-12,
        );
    }

    #[test]
    fn rfftn_differentiable_norm_ortho_roundtrip_grad_shape() {
        // ortho rfftn attaches RfftnBackward with the threaded norm; the grad
        // recovers the input shape.
        let input = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let result = rfftn_differentiable_norm(&input, None, None, FftNorm::Ortho).unwrap();
        assert_eq!(result.grad_fn().unwrap().name(), "RfftnBackward");
        let go: Vec<f64> = vec![0.5; crate::shape::numel(result.shape())];
        let grad_out = no_grad_leaf(&go, result.shape());
        let grads = result.grad_fn().unwrap().backward(&grad_out).unwrap();
        assert_eq!(grads[0].as_ref().unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn fft2_ifft2_differentiable_roundtrip_values() {
        // Verify the differentiable forwards round-trip to identity. A
        // [2, 3, 2] complex tensor holds 2*3 = 6 complex pairs (12 floats);
        // the trailing dim of size 2 is the (re, im) pair.
        let mut complex = Vec::with_capacity(12);
        for i in 0..6 {
            complex.push(i as f64);
            complex.push(i as f64 * 0.5);
        }
        let input = leaf(&complex, &[2, 3, 2]);
        let spectrum = fft2_differentiable(&input).unwrap();
        let recovered = ifft2_differentiable(&spectrum).unwrap();
        assert_close(recovered.data().unwrap(), &complex, 1e-9);
    }
}
