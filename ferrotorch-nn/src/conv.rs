//! Convolution layers: 1-D, 2-D, 3-D and their transposed variants.
//!
//! Implements `Conv1d<T>`, `Conv2d<T>`, `Conv3d<T>`, `ConvTranspose1d<T>`,
//! `ConvTranspose2d<T>`, and `ConvTranspose3d<T>`.
//! Forward passes use the im2col + matmul approach; backward follows the
//! same structure in reverse.
//!
//! ## REQ status (per `.design/ferrotorch-nn/conv.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | impl: `pub struct Conv2d<T: Float>` here, mirroring `aten/src/ATen/native/Convolution.cpp:520-600`; non-test consumer: `ferrotorch-vision/src/models/resnet.rs` constructs `Conv2d::new(...)` for every residual block conv. |
//! | REQ-2 | SHIPPED | impl: the `Conv2d::new` / `Conv2d::new_full` constructors here with `groups` / `dilation` validation; non-test consumer: `ferrotorch-vision/src/models/vit.rs` and `convnext.rs` construct grouped or dilated `Conv2d` via `new_full`. |
//! | REQ-3 | SHIPPED | impl: `<Conv2d as Module>::forward` body here (im2col + matmul) mirroring `aten::convolution`; non-test consumer: every vision model forward invokes `Conv2d::forward` through its `Module` impl. |
//! | REQ-4 | SHIPPED | impl: `is_f32 && input.is_cuda()` dispatch to `backend.conv2d_f32` inside `<Conv2d as Module>::forward`; non-test consumer: `ferrotorch-gpu/src/backend_impl.rs` exposes `Backend::conv2d_f32`; vision-model training runs on CUDA trigger this dispatch end-to-end. |
//! | REQ-5 | SHIPPED | impl: `Conv2dBackward<T>: GradFn<T>` impl block here; non-test consumer: every gradient step on a vision model's `loss.backward()` traverses these `Conv2dBackward` nodes through `ferrotorch_core::autograd::engine`. |
//! | REQ-6 | SHIPPED | impl: `pub struct Conv1d` / `Conv3d` / `ConvTranspose{1,2,3}d` here. Conv1d/Conv3d carry `groups`/`dilation` (forward layers) via `Conv1d::new_full` / `Conv3d::new_full` + the per-group + dilated `<Conv1d as Module>::forward` / `<Conv3d as Module>::forward` and `Conv1dBackward` / `Conv3dBackward` (closes #1600 conv1d, #1601 conv3d; weight `[out, in/groups, *k]` per `torch/nn/modules/conv.py:171`, channel partition per `aten/src/ATen/native/Convolution.cpp:1723-1729`, `eff_kernel = dilation*(k-1)+1` per `aten/src/ATen/native/ConvUtils.h:255`). non-test consumer: `Conv1d::new` / `Conv3d::new` delegate to `new_full` in production and are called by `ferrotorch-nn/src/lazy_conv.rs` `LazyConv1d::materialize` / `LazyConv3d::materialize`; `ferrotorch-vision/src/models/inception.rs` uses `Conv2d` + `ConvTranspose2d`. |
//! | REQ-7 | SHIPPED | impl: `impl<T: Float> Module<T> for Conv2d<T>` block (and analogues for the other 5) here; non-test consumer: `ferrotorch_optim` walks `Module::parameters_mut()` across every conv in a training loop. |
//! | REQ-8 | SHIPPED | impl: the `Conv2d::set_weight` and `Conv2d::from_parts` methods here; non-test consumer: `ferrotorch-nn/src/functional.rs` (the stateless `nn::functional::conv2d` entry point) uses `Conv2d::from_parts` to drive the existing forward path with user-supplied parameters. |
//! | REQ-9 | SHIPPED | impl: `kaiming_uniform(&mut weight, NonLinearity::ReLU)` + `uniform_init(&mut b, -bound, bound)` (bound = 1/sqrt(fan_in)) inside every `Conv*d::new[_full]` here, mirroring `torch/nn/modules/conv.py:182-201`; non-test consumer: `Conv2d::new` is the path used by every vision-model constructor. (closes #1450 — bias U(-bound,bound). Kaiming gain divergence (`a=sqrt(5)` upstream vs `ReLU` here) remains as separate followup.) |
//! | REQ-10 | SHIPPED | impl: `Conv1d` / `Conv2d` / `Conv3d` each carry a `padding_mode: crate::padding::PaddingMode` field + `with_padding_mode(...)` builder here; when the mode is non-`Zeros`, the layer's `forward` calls `crate::padding::functional_pad_1d`/`_2d`/`_3d` (with `_reversed_padding_repeated_twice` amounts) and then runs the zero-padding im2col on the already-padded tensor, mirroring `torch/nn/modules/conv.py` `_ConvNd._conv_forward` (Conv1d `conv.py:367-378`, Conv3d `conv.py:716-732`). The 1-D/3-D pre-pads are autograd-aware (`Pad1dBackward` / `Pad3dBackward` in `padding.rs`), so input gradients flow through the boundary (the #1550 fix class). `ConvTranspose{1,2,3}d::with_padding_mode` rejects any non-`Zeros` mode via `fn reject_non_zeros_transpose` with the upstream `ValueError('Only "zeros" padding mode is supported for ...')` (`conv.py:755-758`). Closes #1443. Non-test consumer: `pub use conv::{Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d}` in `lib.rs` re-export; the `<Conv1d as Module>::forward` / `<Conv3d as Module>::forward` bodies consume `functional_pad_1d` / `functional_pad_3d` in production. |
//! | REQ-11 | NOT-STARTED | blocker #1441 (umbrella) — parity-sweep runner arms for all 6 conv ops are absent; sweep reports `0/N passed, N skipped` for each. The forward paths themselves are end-to-end verified by 60+ lib tests; only the runner-arm wiring is missing. |

use std::sync::Arc;

use ferrotorch_core::autograd::autocast_ops::autocast_guard;
use ferrotorch_core::autograd::no_grad::is_grad_enabled;
use ferrotorch_core::ops::linalg::{mm, transpose};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::{GradFn, Tensor};
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float};

use crate::init::{NonLinearity, kaiming_uniform, uniform as uniform_init};
use crate::module::Module;
use crate::parameter::Parameter;

// ---------------------------------------------------------------------------
// ConvTranspose padding_mode validation
// ---------------------------------------------------------------------------

/// Reject any non-`Zeros` `padding_mode` for a transposed convolution.
///
/// Upstream `_ConvTransposeNd.__init__` (`torch/nn/modules/conv.py:755-758`)
/// runs `if padding_mode != "zeros": raise ValueError(f'Only "zeros" padding
/// mode is supported for {self.__class__.__name__}')`. Only `"zeros"` is a
/// valid `padding_mode` for ConvTranspose layers; matching this exception
/// behaviour (rather than silently accepting the mode) is the R-DEV-2 contract.
/// Closes #1443.
fn reject_non_zeros_transpose(
    mode: crate::padding::PaddingMode,
    class_name: &str,
) -> FerrotorchResult<()> {
    if mode != crate::padding::PaddingMode::Zeros {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("Only \"zeros\" padding mode is supported for {class_name}"),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// im2col / col2im helpers
// ---------------------------------------------------------------------------

/// Extract image patches into columns (no dilation — calls [`im2col_dilated`]
/// with `(1, 1)` for the dilation rate).
///
/// Given a 4-D input `[B, C, H, W]`, produces a 3-D output
/// `[B, C * kH * kW, H_out * W_out]` where each column is one
/// flattened receptive-field patch.
// Internal kernel: argument set mirrors the 2-D convolution descriptor
// (B, C, H, W, kH, kW, padH, padW, strideH, strideW); a config
// struct would force allocation on every call in convolution hot paths.
#[allow(clippy::too_many_arguments)]
fn im2col<T: Float>(
    input: &[T],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> (Vec<T>, usize, usize) {
    im2col_dilated(
        input, batch, channels, height, width, kernel_h, kernel_w, stride_h, stride_w, pad_h,
        pad_w, 1, 1,
    )
}

/// Extract image patches into columns, supporting dilation `(dil_h, dil_w)`.
///
/// Given a 4-D input `[B, C, H, W]`, produces a 3-D output
/// `[B, C * kH * kW, H_out * W_out]` where each column is one flattened
/// receptive-field patch with kernel taps spaced by `dil_h`/`dil_w` along the
/// spatial axes.
///
/// Output spatial sizes follow the standard formula:
///
/// ```text
/// H_out = (H + 2*pad_h - dil_h*(kH - 1) - 1) / stride_h + 1
/// W_out = (W + 2*pad_w - dil_w*(kW - 1) - 1) / stride_w + 1
/// ```
// Internal kernel: argument set mirrors the 2-D convolution descriptor
// (B, C, H, W, kH, kW, strideH, strideW, padH, padW, dilH, dilW); a config
// struct would force allocation on every call in convolution hot paths.
#[allow(clippy::too_many_arguments)]
fn im2col_dilated<T: Float>(
    input: &[T],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    dil_h: usize,
    dil_w: usize,
) -> (Vec<T>, usize, usize) {
    let eff_kh = dil_h * (kernel_h - 1) + 1;
    let eff_kw = dil_w * (kernel_w - 1) + 1;
    let h_out = (height + 2 * pad_h - eff_kh) / stride_h + 1;
    let w_out = (width + 2 * pad_w - eff_kw) / stride_w + 1;
    let col_rows = channels * kernel_h * kernel_w;
    let col_cols = h_out * w_out;

    let zero = <T as num_traits::Zero>::zero();
    let mut cols = vec![zero; batch * col_rows * col_cols];

    for b in 0..batch {
        for c in 0..channels {
            for kh in 0..kernel_h {
                for kw in 0..kernel_w {
                    let row = c * kernel_h * kernel_w + kh * kernel_w + kw;
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            // The padded-coordinate of this kernel tap.
                            let ih = oh * stride_h + kh * dil_h;
                            let iw = ow * stride_w + kw * dil_w;
                            let col = oh * w_out + ow;

                            // Account for padding: the "virtual" input coordinate
                            // must be shifted back by the padding amount.
                            let val = if ih >= pad_h
                                && iw >= pad_w
                                && (ih - pad_h) < height
                                && (iw - pad_w) < width
                            {
                                let real_h = ih - pad_h;
                                let real_w = iw - pad_w;
                                input[b * channels * height * width
                                    + c * height * width
                                    + real_h * width
                                    + real_w]
                            } else {
                                zero
                            };

                            cols[b * col_rows * col_cols + row * col_cols + col] = val;
                        }
                    }
                }
            }
        }
    }

    (cols, col_rows, col_cols)
}

/// Scatter columns back into an image tensor (adjoint of [`im2col`]).
///
/// Given columns of shape `[B, C * kH * kW, H_out * W_out]`, accumulates
/// values back into a `[B, C, H, W]` tensor (with padding stripped).
///
/// `#[cfg(test)]`-gated: production backward paths (`Conv1dBackward`,
/// `Conv2dBackward`) call [`col2im_dilated`] directly with the layer's
/// dilation, so the only remaining caller of this non-dilated shim is the
/// im2col/col2im roundtrip unit test.
// Internal kernel: argument set is the adjoint of `im2col` (same descriptor
// inputs); refactoring to a config struct would diverge the two helpers'
// signatures unhelpfully.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn col2im<T: Float>(
    cols: &[T],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    h_out: usize,
    w_out: usize,
) -> Vec<T> {
    col2im_dilated(
        cols, batch, channels, height, width, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
        1, 1, h_out, w_out,
    )
}

/// Scatter columns back into an image tensor with dilation support
/// (adjoint of [`im2col_dilated`]).
///
/// Given columns of shape `[B, C * kH * kW, H_out * W_out]`, accumulates
/// values back into a `[B, C, H, W]` tensor (with padding stripped),
/// honouring `dil_h`/`dil_w` so kernel taps are spaced apart in the input.
// Internal kernel: argument set is the adjoint of `im2col_dilated` (same
// descriptor inputs); refactoring to a config struct would diverge the two
// helpers' signatures unhelpfully.
#[allow(clippy::too_many_arguments)]
fn col2im_dilated<T: Float>(
    cols: &[T],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    dil_h: usize,
    dil_w: usize,
    h_out: usize,
    w_out: usize,
) -> Vec<T> {
    let zero = <T as num_traits::Zero>::zero();
    let mut output = vec![zero; batch * channels * height * width];

    let col_rows = channels * kernel_h * kernel_w;
    let col_cols = h_out * w_out;

    for b in 0..batch {
        for c in 0..channels {
            for kh in 0..kernel_h {
                for kw in 0..kernel_w {
                    let row = c * kernel_h * kernel_w + kh * kernel_w + kw;
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let ih = oh * stride_h + kh * dil_h;
                            let iw = ow * stride_w + kw * dil_w;
                            let col = oh * w_out + ow;

                            if ih >= pad_h
                                && iw >= pad_w
                                && (ih - pad_h) < height
                                && (iw - pad_w) < width
                            {
                                let real_h = ih - pad_h;
                                let real_w = iw - pad_w;
                                output[b * channels * height * width
                                    + c * height * width
                                    + real_h * width
                                    + real_w] +=
                                    cols[b * col_rows * col_cols + row * col_cols + col];
                            }
                        }
                    }
                }
            }
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Conv2d
// ---------------------------------------------------------------------------

/// A 2-D convolution layer.
///
/// Applies a spatial convolution over an input `[B, C_in, H, W]` using
/// the im2col + matmul algorithm. Equivalent to `torch.nn.Conv2d`,
/// including the `groups` and `dilation` arguments (see
/// [`Conv2d::new_full`]).
///
/// # Shape
///
/// - Input: `[B, in_channels, H, W]`
/// - Output: `[B, out_channels, H_out, W_out]`
///
/// where `H_out = (H + 2 * padding.0 - dilation.0 * (kernel_size.0 - 1) - 1)
/// / stride.0 + 1`.
///
/// # GPU coverage
///
/// The CUDA fast path supplied by `ferrotorch-gpu` currently only covers
/// `groups == 1 && dilation == (1, 1)`. When the layer is constructed with
/// `groups > 1` or `dilation != (1, 1)`, [`Module::forward`] explicitly
/// skips the GPU dispatch at the gate (see the `if groups == 1 && dilation
/// == (1, 1)` guard in the forward) and runs the entire convolution on the
/// CPU. Wiring `groups`/`dilation` through the GPU backend signature is
/// tracked separately as a backend extension issue.
#[derive(Debug)]
pub struct Conv2d<T: Float> {
    /// Learnable kernel weights `[out_channels, in_channels / groups, kH, kW]`.
    weight: Parameter<T>,
    /// Optional learnable bias `[out_channels]`.
    bias: Option<Parameter<T>>,
    /// Number of input channels.
    in_channels: usize,
    /// Number of output channels (filters).
    out_channels: usize,
    /// Kernel spatial size `(kH, kW)`.
    kernel_size: (usize, usize),
    /// Stride `(sH, sW)`.
    stride: (usize, usize),
    /// Zero-padding `(pH, pW)` applied to both sides.
    padding: (usize, usize),
    /// Dilation `(dilH, dilW)`. `(1, 1)` is the dense default.
    dilation: (usize, usize),
    /// Number of blocked input/output channel groups. `1` is dense, `in_channels`
    /// is depthwise. Must divide both `in_channels` and `out_channels`.
    groups: usize,
    /// Boundary handling for the spatial padding. `Zeros` (default) routes
    /// through the existing im2col fast path; non-`Zeros` modes pre-pad
    /// the input via `crate::padding::functional_pad_2d` and then run the
    /// dense im2col over the already-padded tensor (matching the upstream
    /// `_ConvNd._conv_forward` shape: `F.pad(input, ..., mode=...)` first,
    /// then a `padding=0` convolution). Closes #1443.
    padding_mode: crate::padding::PaddingMode,
    /// Whether the module is in training mode.
    training: bool,
}

impl<T: Float> Conv2d<T> {
    /// Create a new `Conv2d` layer (dense, dilation `(1, 1)`, `groups = 1`).
    ///
    /// Weight is initialized with Kaiming uniform (ReLU gain).
    /// Bias, if enabled, is initialized U(-bound, bound) with
    /// `bound = 1/sqrt(fan_in)` per `torch/nn/modules/conv.py:198-201`.
    ///
    /// This is a thin shim over [`Conv2d::new_full`] preserved for
    /// backwards compatibility with existing callers (see Phase 5 of #1002).
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        bias: bool,
    ) -> FerrotorchResult<Self> {
        Self::new_full(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            (1, 1),
            1,
            bias,
        )
    }

    /// Create a new `Conv2d` layer with the full PyTorch-shaped argument set,
    /// including `dilation` and `groups`.
    ///
    /// `groups` must divide BOTH `in_channels` and `out_channels` (PyTorch
    /// `torch.nn.Conv2d` raises `ValueError` otherwise). `dilation` must be
    /// strictly positive in both dimensions. Weight is initialised with
    /// Kaiming uniform (ReLU gain); bias (if enabled) with U(-bound, bound)
    /// where `bound = 1/sqrt(fan_in)` per `torch/nn/modules/conv.py:198-201`.
    ///
    /// # GPU coverage caveat
    ///
    /// `Conv2d::forward`'s CUDA fast path is only taken when `groups == 1 &&
    /// dilation == (1, 1)`. With grouped or dilated configurations the
    /// dispatch gate explicitly falls through to the CPU implementation;
    /// kernel-side `groups`/`dilation` plumbing in the `ferrotorch-gpu`
    /// backend is a separate workitem.
    #[allow(clippy::too_many_arguments)]
    pub fn new_full(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
        bias: bool,
    ) -> FerrotorchResult<Self> {
        if in_channels == 0 || out_channels == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "in_channels and out_channels must be > 0".into(),
            });
        }
        if kernel_size.0 == 0 || kernel_size.1 == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "kernel_size must be > 0 in both dimensions".into(),
            });
        }
        if stride.0 == 0 || stride.1 == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "stride must be > 0 in both dimensions".into(),
            });
        }
        if dilation.0 == 0 || dilation.1 == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Conv2d::new_full: dilation must be > 0 in both dimensions, got {dilation:?}"
                ),
            });
        }
        if groups == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "Conv2d::new_full: groups must be > 0".into(),
            });
        }
        if in_channels % groups != 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Conv2d::new_full: groups={groups} must divide in_channels={in_channels}"
                ),
            });
        }
        if out_channels % groups != 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Conv2d::new_full: groups={groups} must divide out_channels={out_channels}"
                ),
            });
        }

        let (kh, kw) = kernel_size;
        // PyTorch weight layout is [C_out, C_in / groups, kH, kW].
        let mut weight = Parameter::zeros(&[out_channels, in_channels / groups, kh, kw])?;
        kaiming_uniform(&mut weight, NonLinearity::ReLU)?;

        let bias_param = if bias {
            let mut b = Parameter::zeros(&[out_channels])?;
            // `torch/nn/modules/conv.py:198-201`: `fan_in, _ = init._calculate_fan_in_and_fan_out(weight);
            //   bound = 1 / sqrt(fan_in); init.uniform_(self.bias, -bound, bound)`. For Conv2d
            //   `fan_in = (in_channels/groups) * kH * kW`.
            let fan_in = (in_channels / groups) * kh * kw;
            let bound = if fan_in > 0 {
                1.0 / (fan_in as f64).sqrt()
            } else {
                0.0
            };
            uniform_init(&mut b, -bound, bound)?;
            Some(b)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias: bias_param,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            padding_mode: crate::padding::PaddingMode::Zeros,
            training: true,
        })
    }

    /// Configure the boundary handling for the spatial padding.
    ///
    /// `Zeros` (default) uses the existing im2col zero-pad path.
    /// `Reflect`, `Replicate`, and `Circular` pre-pad the input via
    /// `crate::padding::functional_pad_2d(input, ...)` and then convolve
    /// with `padding = 0`, matching `torch.nn.Conv2d(..., padding_mode=...)`
    /// (`_ConvNd._conv_forward`'s `F.pad` shape). Closes #1443.
    pub fn with_padding_mode(mut self, mode: crate::padding::PaddingMode) -> Self {
        self.padding_mode = mode;
        self
    }

    /// Replace the kernel weights with a caller-supplied [`Parameter`].
    ///
    /// The new weight must have shape `[out_channels, in_channels / groups,
    /// kH, kW]` (i.e. the same shape as the existing parameter). Used by
    /// tests and tooling that need deterministic weights without going
    /// through [`Conv2d::from_parts`].
    pub fn set_weight(&mut self, weight: Parameter<T>) -> FerrotorchResult<()> {
        let expected = [
            self.out_channels,
            self.in_channels / self.groups,
            self.kernel_size.0,
            self.kernel_size.1,
        ];
        let got = weight.tensor().shape();
        if got != expected {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!("Conv2d::set_weight: expected shape {expected:?}, got {got:?}"),
            });
        }
        self.weight = weight;
        Ok(())
    }

    /// Number of channel groups (`1` is dense, `in_channels` is depthwise).
    pub fn groups(&self) -> usize {
        self.groups
    }

    /// Dilation `(dilH, dilW)` (`(1, 1)` is the dense default).
    pub fn dilation(&self) -> (usize, usize) {
        self.dilation
    }

    /// The number of learnable scalar parameters.
    ///
    /// For grouped convolutions the weight tensor has shape
    /// `[out_channels, in_channels / groups, kH, kW]` so the count is
    /// scaled by `1 / groups`.
    pub fn num_parameters(&self) -> usize {
        let w = self.out_channels
            * (self.in_channels / self.groups)
            * self.kernel_size.0
            * self.kernel_size.1;
        let b = if self.bias.is_some() {
            self.out_channels
        } else {
            0
        };
        w + b
    }

    /// Build a `Conv2d` from caller-supplied weight and optional bias tensors.
    ///
    /// `weight` must have shape `[out_channels, in_channels, kH, kW]`. If
    /// `bias` is provided, it must be 1-D of length `out_channels`. The
    /// stride and padding are passed through unchanged; the resulting layer
    /// is dense (`groups = 1`, `dilation = (1, 1)`) so this constructor is
    /// API-compatible with the pre-Phase-5 surface. This is the constructor
    /// used by `nn::functional::conv2d` so callers can drive the existing
    /// im2col + matmul forward path with their own parameters (e.g. for
    /// stateless functional dispatch or weight sharing across modules).
    pub fn from_parts(
        weight: Tensor<T>,
        bias: Option<Tensor<T>>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> FerrotorchResult<Self> {
        if weight.ndim() != 4 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Conv2d::from_parts: weight must be 4-D [out, in, kH, kW], got {:?}",
                    weight.shape()
                ),
            });
        }
        let out_channels = weight.shape()[0];
        let in_channels = weight.shape()[1];
        let kernel_size = (weight.shape()[2], weight.shape()[3]);
        if let Some(b) = &bias {
            if b.ndim() != 1 || b.shape()[0] != out_channels {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "Conv2d::from_parts: bias shape {:?} != [{}]",
                        b.shape(),
                        out_channels
                    ),
                });
            }
        }
        Ok(Self {
            weight: Parameter::new(weight),
            bias: bias.map(Parameter::new),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation: (1, 1),
            groups: 1,
            padding_mode: crate::padding::PaddingMode::Zeros,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for Conv2d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Record autocast decision for conv2d.
        let _autocast_cat = autocast_guard("conv2d");

        // Non-zero padding modes: pre-pad the input with the requested
        // boundary mode and then convolve with padding = 0. Mirrors
        // `torch/nn/modules/conv.py` `_ConvNd._conv_forward`:
        //   if self.padding_mode != 'zeros':
        //       input = F.pad(input,
        //                     self._reversed_padding_repeated_twice,
        //                     mode=self.padding_mode)
        //       conv2d(..., padding=0, ...)
        // Closes #1443.
        if self.padding_mode != crate::padding::PaddingMode::Zeros
            && (self.padding.0 != 0 || self.padding.1 != 0)
        {
            let padded = crate::padding::functional_pad_2d(
                input,
                self.padding.1,
                self.padding.1,
                self.padding.0,
                self.padding.0,
                self.padding_mode,
                <T as num_traits::Zero>::zero(),
            )?;
            // Recurse on a zero-padding variant. Build a shallow clone with
            // padding = (0, 0) and padding_mode = Zeros so the existing
            // im2col-on-zero-pad path runs without re-padding.
            let zero_padded_layer = Conv2d {
                weight: Parameter::new(self.weight.tensor().clone()),
                bias: self
                    .bias
                    .as_ref()
                    .map(|b| Parameter::new(b.tensor().clone())),
                in_channels: self.in_channels,
                out_channels: self.out_channels,
                kernel_size: self.kernel_size,
                stride: self.stride,
                padding: (0, 0),
                dilation: self.dilation,
                groups: self.groups,
                padding_mode: crate::padding::PaddingMode::Zeros,
                training: self.training,
            };
            return zero_padded_layer.forward(&padded);
        }

        // Validate input shape: [B, C_in, H, W].
        if input.ndim() != 4 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Conv2d expects 4-D input [B, C, H, W], got {:?}",
                    input.shape()
                ),
            });
        }

        let batch = input.shape()[0];
        let c_in = input.shape()[1];
        let h = input.shape()[2];
        let w = input.shape()[3];

        if c_in != self.in_channels {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Conv2d: expected {} input channels, got {}",
                    self.in_channels, c_in
                ),
            });
        }

        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        let (dh, dw) = self.dilation;
        let groups = self.groups;

        // Effective kernel extent after dilation.
        let eff_kh = dh * (kh - 1) + 1;
        let eff_kw = dw * (kw - 1) + 1;

        // Check that the (effective) kernel fits.
        let h_padded = h + 2 * ph;
        let w_padded = w + 2 * pw;
        if h_padded < eff_kh || w_padded < eff_kw {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Conv2d: padded input ({h_padded}, {w_padded}) is smaller than effective kernel ({eff_kh}, {eff_kw})"
                ),
            });
        }

        let h_out = (h_padded - eff_kh) / sh + 1;
        let w_out = (w_padded - eff_kw) / sw + 1;

        // Save the input device so we can restore it on the output.
        let input_device = input.device();

        // ---- GPU fast path: fully on-device conv2d ----
        //
        // Pass 2A (#1003): the CUDA backend's `conv2d_f32` accepts groups
        // and dilation natively. Every f32 CUDA input dispatches to the
        // GPU regardless of `groups` / `dilation`; the kernel does the
        // per-group im2col + GEMM on-device. The pre-Pass-2A
        // `gpu_eligible = groups == 1 && dilation == (1, 1)` gate is
        // gone — keeping it would be a stub-shaped CPU detour that
        // failure mode #15 explicitly forbids.
        //
        // Note: the weight layout passed to the backend is
        // `[C_out, C_in / groups, kH, kW]` — the PyTorch grouped-conv
        // convention. `Conv2d::new_full` already constructs `self.weight`
        // in that shape (see `Conv2d::new_full` for the `in_per_group =
        // in_channels / groups` slice).
        let is_f32 = std::mem::size_of::<T>() == 4;
        if is_f32 && input.is_cuda() {
            if let Some(backend) = ferrotorch_core::gpu_dispatch::gpu_backend() {
                let bias_handle = self
                    .bias
                    .as_ref()
                    .and_then(|b| b.tensor().gpu_handle().ok());
                let weight_shape = self.weight.tensor().shape();
                let weight_dim4: [usize; 4] = [
                    weight_shape[0],
                    weight_shape[1],
                    weight_shape[2],
                    weight_shape[3],
                ];
                let (out_handle, out_shape) = backend.conv2d_f32(
                    input.gpu_handle()?,
                    self.weight.tensor().gpu_handle()?,
                    bias_handle,
                    [batch, c_in, h, w],
                    weight_dim4,
                    self.stride,
                    self.padding,
                    self.dilation,
                    groups,
                )?;

                let result = Tensor::from_storage(
                    TensorStorage::gpu(out_handle),
                    out_shape.to_vec(),
                    false,
                )?;

                // For backward, fall through to CPU path if gradients needed
                // (GPU backward not yet implemented — stores input for recomputation)
                if is_grad_enabled()
                    && (input.requires_grad()
                        || self.weight.requires_grad()
                        || self.bias.as_ref().is_some_and(|b| b.requires_grad()))
                {
                    // Download cols for backward (CPU backward path).
                    let input_data = input.data_vec()?;
                    let (cols, col_rows, col_cols) =
                        im2col(&input_data, batch, c_in, h, w, kh, kw, sh, sw, ph, pw);
                    let grad_fn = Arc::new(Conv2dBackward {
                        input: input.clone(),
                        weight: self.weight.tensor().clone(),
                        bias: self.bias.as_ref().map(|b| b.tensor().clone()),
                        in_channels: self.in_channels,
                        out_channels: self.out_channels,
                        kernel_size: self.kernel_size,
                        stride: self.stride,
                        padding: self.padding,
                        dilation: self.dilation,
                        groups: self.groups,
                        cols,
                        col_rows,
                        col_cols,
                        h_out,
                        w_out,
                    });
                    return Tensor::from_operation(
                        result.into_storage_and_shape()?.0,
                        out_shape.to_vec(),
                        grad_fn,
                    );
                }

                return Ok(result);
            }
        }

        // ---- CPU path (handles dense, dilated, grouped, and grouped+dilated) ----
        let input_data = input.data_vec()?;
        let weight_data = self.weight.data_vec()?;

        let zero = <T as num_traits::Zero>::zero();
        let mut output = vec![zero; batch * self.out_channels * h_out * w_out];

        // Per-group dimensions.
        let in_per_group = self.in_channels / groups;
        let out_per_group = self.out_channels / groups;
        let weight_per_group_numel = out_per_group * in_per_group * kh * kw;
        let group_col_rows = in_per_group * kh * kw;
        let col_cols = h_out * w_out;

        // Saved im2col columns for autograd (full, ungrouped layout — channel
        // axis kept dense at C_in so the backward can accumulate grad_input
        // back into a `[B, C_in, H, W]` tensor uniformly across groups).
        let saved_cols_rows = self.in_channels * kh * kw;
        let mut saved_cols: Vec<T> = if is_grad_enabled()
            && (input.requires_grad()
                || self.weight.requires_grad()
                || self.bias.as_ref().is_some_and(|b| b.requires_grad()))
        {
            vec![zero; batch * saved_cols_rows * col_cols]
        } else {
            Vec::new()
        };
        let save_cols = !saved_cols.is_empty();

        for g in 0..groups {
            // Slice the input channels belonging to this group: [B, in_per_group, H, W].
            let mut group_input = vec![zero; batch * in_per_group * h * w];
            for b in 0..batch {
                for c in 0..in_per_group {
                    let src_c = g * in_per_group + c;
                    let src_start = b * self.in_channels * h * w + src_c * h * w;
                    let dst_start = b * in_per_group * h * w + c * h * w;
                    group_input[dst_start..dst_start + h * w]
                        .copy_from_slice(&input_data[src_start..src_start + h * w]);
                }
            }

            let (g_cols, g_col_rows, g_col_cols) = im2col_dilated(
                &group_input,
                batch,
                in_per_group,
                h,
                w,
                kh,
                kw,
                sh,
                sw,
                ph,
                pw,
                dh,
                dw,
            );
            debug_assert_eq!(g_col_rows, group_col_rows);
            debug_assert_eq!(g_col_cols, col_cols);

            // Save into the dense [C_in*kH*kW, col_cols] layout if backward will need it.
            if save_cols {
                for b in 0..batch {
                    for c in 0..in_per_group {
                        let dst_c = g * in_per_group + c;
                        for kk in 0..(kh * kw) {
                            let src_row = c * kh * kw + kk;
                            let dst_row = dst_c * kh * kw + kk;
                            let src_off = b * group_col_rows * col_cols + src_row * col_cols;
                            let dst_off = b * saved_cols_rows * col_cols + dst_row * col_cols;
                            saved_cols[dst_off..dst_off + col_cols]
                                .copy_from_slice(&g_cols[src_off..src_off + col_cols]);
                        }
                    }
                }
            }

            // Group's slice of the weight: [out_per_group, in_per_group, kh, kw]
            // flattened to [out_per_group, group_col_rows].
            let w_group_start = g * weight_per_group_numel;
            let w_group_end = w_group_start + weight_per_group_numel;
            let weight_group_2d = Tensor::from_storage(
                TensorStorage::cpu(weight_data[w_group_start..w_group_end].to_vec()),
                vec![out_per_group, group_col_rows],
                false,
            )?;

            for b in 0..batch {
                let col_start = b * group_col_rows * col_cols;
                let col_end = col_start + group_col_rows * col_cols;
                let cols_b = Tensor::from_storage(
                    TensorStorage::cpu(g_cols[col_start..col_end].to_vec()),
                    vec![group_col_rows, col_cols],
                    false,
                )?;

                let out_b = mm(&weight_group_2d, &cols_b)?;
                let out_data = out_b.data()?;
                // Place this group's output channels into [b, g*out_per_group..(g+1)*out_per_group, :, :].
                for oc in 0..out_per_group {
                    let dst_c = g * out_per_group + oc;
                    let dst_start = b * self.out_channels * h_out * w_out + dst_c * h_out * w_out;
                    let src_start = oc * h_out * w_out;
                    output[dst_start..dst_start + h_out * w_out]
                        .copy_from_slice(&out_data[src_start..src_start + h_out * w_out]);
                }
            }
        }

        // Add bias if present: broadcast [C_out] over [B, C_out, H_out, W_out].
        if let Some(ref bias) = self.bias {
            let bias_data = bias.data_vec()?;
            for b in 0..batch {
                for c in 0..self.out_channels {
                    let bval = bias_data[c];
                    for hw in 0..(h_out * w_out) {
                        output[b * self.out_channels * h_out * w_out + c * h_out * w_out + hw] +=
                            bval;
                    }
                }
            }
        }

        let result = Tensor::from_storage(
            TensorStorage::cpu(output),
            vec![batch, self.out_channels, h_out, w_out],
            false,
        )?;

        // Attach backward if gradients are enabled and any input/param requires grad.
        if save_cols {
            let grad_fn = Arc::new(Conv2dBackward {
                input: input.clone(),
                weight: self.weight.tensor().clone(),
                bias: self.bias.as_ref().map(|b| b.tensor().clone()),
                in_channels: self.in_channels,
                out_channels: self.out_channels,
                kernel_size: self.kernel_size,
                stride: self.stride,
                padding: self.padding,
                dilation: self.dilation,
                groups: self.groups,
                cols: saved_cols,
                col_rows: saved_cols_rows,
                col_cols,
                h_out,
                w_out,
            });
            Tensor::from_operation(
                TensorStorage::cpu(result.data()?.to_vec()),
                result.shape().to_vec(),
                grad_fn,
            )?
            .to(input_device) // restore device
        } else {
            result.to(input_device)
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = vec![("weight".to_string(), &self.weight)];
        if let Some(ref b) = self.bias {
            params.push(("bias".to_string(), b));
        }
        params
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// Conv2dBackward
// ---------------------------------------------------------------------------

/// Backward function for `Conv2d` forward pass.
///
/// Saved tensors:
/// - `input`: the original 4-D input
/// - `weight`: the 4-D kernel `[C_out, C_in / groups, kH, kW]`
/// - `bias`: optional 1-D bias
/// - `cols`: the im2col columns from the forward pass with **dense channel
///   layout** `[B, C_in * kH * kW, H_out * W_out]`. The forward saves into
///   this shape regardless of `groups` so the backward can reuse a uniform
///   indexing scheme; for `groups > 1` the per-group slice is taken at
///   gradient-computation time.
/// - `dilation`, `groups`: extra descriptors needed to reconstruct the
///   per-group + dilated math without re-reading them off the layer.
#[derive(Debug)]
struct Conv2dBackward<T: Float> {
    input: Tensor<T>,
    weight: Tensor<T>,
    bias: Option<Tensor<T>>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
    cols: Vec<T>,
    col_rows: usize,
    col_cols: usize,
    h_out: usize,
    w_out: usize,
}

impl<T: Float> GradFn<T> for Conv2dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // grad_output shape: [B, C_out, H_out, W_out].
        //
        // The backward computation is host-side (im2col / col2im / mm),
        // so the produced tensors land on CPU. They must be lifted back
        // onto the saved input/weight devices before being returned;
        // otherwise downstream gradient ops (e.g. relu_backward, the
        // optimizer step) see CPU tensors mixed with CUDA parameters
        // and either fall into the `NotImplementedOnCuda` branch or
        // fail device assertions in the optimizer. Surfaced by
        // `gpu_cnn_training_smoke` in
        // `ferrotorch/tests/gpu_training.rs` (#749 Section B).
        //
        // Note: this is a transitional fix that keeps the chain
        // device-consistent while the actual GPU im2col/col2im backward
        // kernels are written. A full Conv2dBackward GPU implementation
        // is tracked separately (see Section B report).
        let input_device = self.input.device();
        let weight_device = self.weight.device();
        let bias_device = self.bias.as_ref().map(|b| b.device());
        let go_data = grad_output.data_vec()?;
        let batch = self.input.shape()[0];
        let h = self.input.shape()[2];
        let w = self.input.shape()[3];
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        let (dh, dw) = self.dilation;
        let groups = self.groups;
        let in_per_group = self.in_channels / groups;
        let out_per_group = self.out_channels / groups;
        let group_col_rows = in_per_group * kh * kw;
        let zero = <T as num_traits::Zero>::zero();

        // --- grad_weight ---
        // Per group `g`:
        //   grad_output_b_g : [out_per_group, H_out * W_out]
        //   cols_b_g        : [in_per_group * kH * kW, H_out * W_out]
        //   gw_g           += grad_output_b_g @ cols_b_g^T
        // Stack groups along the C_out axis to recover [C_out, C_in/G, kH, kW].
        let grad_weight = if self.weight.requires_grad() {
            let weight_numel = self.out_channels * in_per_group * kh * kw;
            let mut gw_accum = vec![zero; weight_numel];
            let weight_per_group_numel = out_per_group * group_col_rows;

            for g in 0..groups {
                for b in 0..batch {
                    // Slice grad_output for this group: [out_per_group, h_out * w_out].
                    let mut go_g = vec![zero; out_per_group * self.h_out * self.w_out];
                    for oc in 0..out_per_group {
                        let src_c = g * out_per_group + oc;
                        let src_start = b * self.out_channels * self.h_out * self.w_out
                            + src_c * self.h_out * self.w_out;
                        let dst_start = oc * self.h_out * self.w_out;
                        go_g[dst_start..dst_start + self.h_out * self.w_out].copy_from_slice(
                            &go_data[src_start..src_start + self.h_out * self.w_out],
                        );
                    }
                    let go_b_g = Tensor::from_storage(
                        TensorStorage::cpu(go_g),
                        vec![out_per_group, self.h_out * self.w_out],
                        false,
                    )?;

                    // Slice cols for this group: [in_per_group * kH * kW, col_cols].
                    let mut cols_g = vec![zero; group_col_rows * self.col_cols];
                    for c in 0..in_per_group {
                        let src_c = g * in_per_group + c;
                        for kk in 0..(kh * kw) {
                            let src_row = src_c * kh * kw + kk;
                            let dst_row = c * kh * kw + kk;
                            let src_off =
                                b * self.col_rows * self.col_cols + src_row * self.col_cols;
                            let dst_off = dst_row * self.col_cols;
                            cols_g[dst_off..dst_off + self.col_cols]
                                .copy_from_slice(&self.cols[src_off..src_off + self.col_cols]);
                        }
                    }
                    let cols_b_g = Tensor::from_storage(
                        TensorStorage::cpu(cols_g),
                        vec![group_col_rows, self.col_cols],
                        false,
                    )?;

                    let cols_bt = transpose(&cols_b_g)?;
                    let gw_b = mm(&go_b_g, &cols_bt)?;
                    let gw_data = gw_b.data()?;

                    let dst_off = g * weight_per_group_numel;
                    for i in 0..weight_per_group_numel {
                        gw_accum[dst_off + i] += gw_data[i];
                    }
                }
            }

            Some(
                Tensor::from_storage(
                    TensorStorage::cpu(gw_accum),
                    vec![self.out_channels, in_per_group, kh, kw],
                    false,
                )?
                .to(weight_device)?,
            )
        } else {
            None
        };

        // --- grad_bias ---
        // Sum grad_output over batch, height, width: sum over [B, *, H_out, W_out]
        // Result shape: [C_out]. Bias is per-output-channel, identical for any
        // groups setting (shape `[C_out]`), so this is unchanged from the dense path.
        let grad_bias = match &self.bias {
            Some(b) if b.requires_grad() => {
                let mut gb = vec![zero; self.out_channels];
                for batch_idx in 0..batch {
                    for c in 0..self.out_channels {
                        for hw in 0..(self.h_out * self.w_out) {
                            gb[c] +=
                                go_data[batch_idx * self.out_channels * self.h_out * self.w_out
                                    + c * self.h_out * self.w_out
                                    + hw];
                        }
                    }
                }
                let target_dev = bias_device.unwrap_or(input_device);
                Some(
                    Tensor::from_storage(TensorStorage::cpu(gb), vec![self.out_channels], false)?
                        .to(target_dev)?,
                )
            }
            _ => None,
        };

        // --- grad_input ---
        // Per group `g`:
        //   weight_g_2d_T @ grad_output_b_g -> [in_per_group * kH * kW, H_out * W_out]
        //   then col2im_dilated -> [in_per_group, H, W] -> place into the right
        //   in-channel slice of [B, C_in, H, W].
        let grad_input = if self.input.requires_grad() {
            let weight_data = self.weight.data_vec()?;
            let mut grad_input_data = vec![zero; batch * self.in_channels * h * w];
            let weight_per_group_numel = out_per_group * group_col_rows;

            for g in 0..groups {
                let w_off = g * weight_per_group_numel;
                let weight_g_2d = Tensor::from_storage(
                    TensorStorage::cpu(weight_data[w_off..w_off + weight_per_group_numel].to_vec()),
                    vec![out_per_group, group_col_rows],
                    false,
                )?;
                let weight_g_t = transpose(&weight_g_2d)?;

                let mut grad_cols_g = vec![zero; batch * group_col_rows * self.col_cols];
                for b in 0..batch {
                    // Slice grad_output for this group/batch.
                    let mut go_g = vec![zero; out_per_group * self.h_out * self.w_out];
                    for oc in 0..out_per_group {
                        let src_c = g * out_per_group + oc;
                        let src_start = b * self.out_channels * self.h_out * self.w_out
                            + src_c * self.h_out * self.w_out;
                        let dst_start = oc * self.h_out * self.w_out;
                        go_g[dst_start..dst_start + self.h_out * self.w_out].copy_from_slice(
                            &go_data[src_start..src_start + self.h_out * self.w_out],
                        );
                    }
                    let go_b_g = Tensor::from_storage(
                        TensorStorage::cpu(go_g),
                        vec![out_per_group, self.h_out * self.w_out],
                        false,
                    )?;

                    let gc_b = mm(&weight_g_t, &go_b_g)?;
                    let gc_data = gc_b.data()?;
                    let gc_start = b * group_col_rows * self.col_cols;
                    grad_cols_g[gc_start..gc_start + group_col_rows * self.col_cols]
                        .copy_from_slice(gc_data);
                }

                // col2im_dilated scatters group's columns back to [B, in_per_group, H, W].
                let gi_g = col2im_dilated(
                    &grad_cols_g,
                    batch,
                    in_per_group,
                    h,
                    w,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    dh,
                    dw,
                    self.h_out,
                    self.w_out,
                );

                // Place into the corresponding slice of the dense [B, C_in, H, W] tensor.
                for b in 0..batch {
                    for c in 0..in_per_group {
                        let dst_c = g * in_per_group + c;
                        let dst_start = b * self.in_channels * h * w + dst_c * h * w;
                        let src_start = b * in_per_group * h * w + c * h * w;
                        grad_input_data[dst_start..dst_start + h * w]
                            .copy_from_slice(&gi_g[src_start..src_start + h * w]);
                    }
                }
            }

            Some(
                Tensor::from_storage(
                    TensorStorage::cpu(grad_input_data),
                    self.input.shape().to_vec(),
                    false,
                )?
                .to(input_device)?,
            )
        } else {
            None
        };

        // Return exactly as many gradients as inputs() returns.
        let mut grads = vec![grad_input, grad_weight];
        if self.bias.is_some() {
            grads.push(grad_bias);
        }
        Ok(grads)
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        let mut v = vec![&self.input, &self.weight];
        if let Some(ref b) = self.bias {
            v.push(b);
        }
        v
    }

    fn name(&self) -> &'static str {
        "Conv2dBackward"
    }
}

// ---------------------------------------------------------------------------
// Conv1d
// ---------------------------------------------------------------------------

/// A 1-D convolution layer for sequence data.
///
/// Applies a temporal convolution over an input `[B, C_in, L]` using
/// the im2col + matmul algorithm (delegates to the 2-D helpers with H=1).
/// Equivalent to `torch.nn.Conv1d`.
///
/// # Shape
///
/// - Input: `[B, in_channels, L]`
/// - Output: `[B, out_channels, L_out]`
///
/// where `L_out = (L + 2 * padding - kernel_size) / stride + 1`.
#[derive(Debug)]
pub struct Conv1d<T: Float> {
    /// Learnable kernel weights `[out_channels, in_channels / groups, kernel_size]`.
    weight: Parameter<T>,
    /// Optional learnable bias `[out_channels]`.
    bias: Option<Parameter<T>>,
    /// Number of input channels.
    in_channels: usize,
    /// Number of output channels (filters).
    out_channels: usize,
    /// Kernel length.
    kernel_size: usize,
    /// Stride.
    stride: usize,
    /// Zero-padding applied to both sides.
    padding: usize,
    /// Dilation. `1` is the dense default. Spaces kernel taps `dilation`
    /// apart along the temporal axis (`eff_kernel = dilation * (k - 1) + 1`),
    /// mirroring `torch.nn.Conv1d(..., dilation=1)` (`conv.py:337`).
    dilation: usize,
    /// Number of blocked input/output channel groups. `1` is dense,
    /// `in_channels` is depthwise. Must divide both `in_channels` and
    /// `out_channels`, mirroring `torch.nn.Conv1d(..., groups=1)`
    /// (`conv.py:338`, validation `conv.py:107-110`).
    groups: usize,
    /// Boundary handling for the spatial padding. `Zeros` (default) routes
    /// through the existing im2col zero-pad path; non-`Zeros` modes pre-pad
    /// the input via `crate::padding::functional_pad_1d` and then run the
    /// dense im2col over the already-padded tensor (matching the upstream
    /// `_ConvNd._conv_forward` for Conv1d: `F.pad(input, ..., mode=...)` first,
    /// then a `padding=0` convolution). See `torch/nn/modules/conv.py:367-378`.
    /// Closes #1443.
    padding_mode: crate::padding::PaddingMode,
    /// Whether the module is in training mode.
    training: bool,
}

impl<T: Float> Conv1d<T> {
    /// Create a new `Conv1d` layer (dense, dilation `1`, `groups = 1`).
    ///
    /// Weight is initialized with Kaiming uniform (ReLU gain).
    /// Bias, if enabled, is initialized U(-bound, bound) with
    /// `bound = 1/sqrt(fan_in)` per `torch/nn/modules/conv.py:198-201`.
    ///
    /// This is a thin shim over [`Conv1d::new_full`] preserved for callers
    /// that only need the dense configuration (e.g. `LazyConv1d::materialize`).
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
    ) -> FerrotorchResult<Self> {
        Self::new_full(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            1,
            1,
            bias,
        )
    }

    /// Create a new `Conv1d` layer with the full PyTorch-shaped argument set,
    /// including `dilation` and `groups`.
    ///
    /// `groups` must divide BOTH `in_channels` and `out_channels` (PyTorch
    /// `torch.nn.Conv1d` raises `ValueError` otherwise, `conv.py:107-110`).
    /// `dilation` must be strictly positive. Weight is initialised with
    /// Kaiming uniform (ReLU gain); bias (if enabled) with U(-bound, bound)
    /// where `bound = 1/sqrt(fan_in)`, `fan_in = (in_channels/groups) *
    /// kernel_size` per `torch/nn/modules/conv.py:198-201`.
    ///
    /// Weight layout is `[out_channels, in_channels / groups, kernel_size]`,
    /// the PyTorch grouped-conv convention (`conv.py:171`). Argument order
    /// `(.., dilation, groups, bias)` mirrors `Conv1d.__init__`
    /// (`conv.py:330-339`, R-DEV-2).
    #[allow(clippy::too_many_arguments)]
    pub fn new_full(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
    ) -> FerrotorchResult<Self> {
        if in_channels == 0 || out_channels == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "in_channels and out_channels must be > 0".into(),
            });
        }
        if kernel_size == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "kernel_size must be > 0".into(),
            });
        }
        if stride == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "stride must be > 0".into(),
            });
        }
        if dilation == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("Conv1d::new_full: dilation must be > 0, got {dilation}"),
            });
        }
        if groups == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "Conv1d::new_full: groups must be > 0".into(),
            });
        }
        // `torch/nn/modules/conv.py:107-110`: `in_channels % groups != 0`
        // and `out_channels % groups != 0` each raise ValueError.
        if in_channels % groups != 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Conv1d::new_full: groups={groups} must divide in_channels={in_channels}"
                ),
            });
        }
        if out_channels % groups != 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Conv1d::new_full: groups={groups} must divide out_channels={out_channels}"
                ),
            });
        }

        // PyTorch weight layout is [C_out, C_in / groups, k] (`conv.py:171`).
        let mut weight = Parameter::zeros(&[out_channels, in_channels / groups, kernel_size])?;
        kaiming_uniform(&mut weight, NonLinearity::ReLU)?;

        let bias_param = if bias {
            let mut b = Parameter::zeros(&[out_channels])?;
            // `torch/nn/modules/conv.py:198-201`: bias U(-bound, bound) with
            //   `bound = 1 / sqrt(fan_in)`, `fan_in = (in_channels/groups) * kernel_size`.
            let fan_in = (in_channels / groups) * kernel_size;
            let bound = if fan_in > 0 {
                1.0 / (fan_in as f64).sqrt()
            } else {
                0.0
            };
            uniform_init(&mut b, -bound, bound)?;
            Some(b)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias: bias_param,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            padding_mode: crate::padding::PaddingMode::Zeros,
            training: true,
        })
    }

    /// Number of channel groups (`1` is dense, `in_channels` is depthwise).
    pub fn groups(&self) -> usize {
        self.groups
    }

    /// Dilation (`1` is the dense default).
    pub fn dilation(&self) -> usize {
        self.dilation
    }

    /// Configure the boundary handling for the spatial padding.
    ///
    /// `Zeros` (default) uses the existing im2col zero-pad path.
    /// `Reflect`, `Replicate`, and `Circular` pre-pad the input via
    /// `crate::padding::functional_pad_1d(input, ...)` and then convolve
    /// with `padding = 0`, matching `torch.nn.Conv1d(..., padding_mode=...)`
    /// (`_ConvNd._conv_forward`'s `F.pad` shape, `conv.py:367-378`). The pad
    /// is autograd-aware (`Pad1dBackward`), so input gradients flow through
    /// the boundary. Closes #1443.
    pub fn with_padding_mode(mut self, mode: crate::padding::PaddingMode) -> Self {
        self.padding_mode = mode;
        self
    }

    /// The number of learnable scalar parameters.
    pub fn num_parameters(&self) -> usize {
        let w = self.out_channels * self.in_channels * self.kernel_size;
        let b = if self.bias.is_some() {
            self.out_channels
        } else {
            0
        };
        w + b
    }

    /// Build a `Conv1d` from caller-supplied weight and optional bias tensors.
    ///
    /// `weight` must have shape `[out_channels, in_channels, kernel_size]`.
    /// The resulting layer is dense (`groups = 1`, `dilation = 1`) so the
    /// constructor remains API-compatible with `nn::functional::conv1d`,
    /// which infers `in_channels = weight.shape()[1]` and cannot recover
    /// `groups` from the weight shape alone.
    pub fn from_parts(
        weight: Tensor<T>,
        bias: Option<Tensor<T>>,
        stride: usize,
        padding: usize,
    ) -> FerrotorchResult<Self> {
        if weight.ndim() != 3 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Conv1d::from_parts: weight must be 3-D [out, in, k], got {:?}",
                    weight.shape()
                ),
            });
        }
        let out_channels = weight.shape()[0];
        let in_channels = weight.shape()[1];
        let kernel_size = weight.shape()[2];
        if let Some(b) = &bias {
            if b.ndim() != 1 || b.shape()[0] != out_channels {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "Conv1d::from_parts: bias shape {:?} != [{}]",
                        b.shape(),
                        out_channels
                    ),
                });
            }
        }
        Ok(Self {
            weight: Parameter::new(weight),
            bias: bias.map(Parameter::new),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation: 1,
            groups: 1,
            padding_mode: crate::padding::PaddingMode::Zeros,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for Conv1d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Record autocast decision for conv1d.
        let _autocast_cat = autocast_guard("conv1d");

        // Non-zero padding modes: pre-pad the input with the requested
        // boundary mode and then convolve with padding = 0. Mirrors
        // `torch/nn/modules/conv.py` `Conv1d._conv_forward` (`conv.py:367-378`):
        //   if self.padding_mode != 'zeros':
        //       F.conv1d(F.pad(input, self._reversed_padding_repeated_twice,
        //                      mode=self.padding_mode), ..., padding=_single(0), ...)
        // For an int `padding=p`, `_reversed_padding_repeated_twice` is `[p, p]`
        // (`conv.py:157` `_reverse_repeat_tuple(self.padding, 2)`), i.e. a
        // symmetric `(pad_left, pad_right) = (p, p)`. The pre-pad is
        // autograd-aware (`Pad1dBackward`) so input gradients flow through the
        // boundary. Closes #1443.
        if self.padding_mode != crate::padding::PaddingMode::Zeros && self.padding != 0 {
            let padded = crate::padding::functional_pad_1d(
                input,
                self.padding,
                self.padding,
                self.padding_mode,
                <T as num_traits::Zero>::zero(),
            )?;
            // Recurse on a zero-padding variant: build a shallow clone with
            // padding = 0 and padding_mode = Zeros so the existing
            // im2col-on-zero-pad path runs without re-padding.
            let zero_padded_layer = Conv1d {
                weight: Parameter::new(self.weight.tensor().clone()),
                bias: self
                    .bias
                    .as_ref()
                    .map(|b| Parameter::new(b.tensor().clone())),
                in_channels: self.in_channels,
                out_channels: self.out_channels,
                kernel_size: self.kernel_size,
                stride: self.stride,
                padding: 0,
                dilation: self.dilation,
                groups: self.groups,
                padding_mode: crate::padding::PaddingMode::Zeros,
                training: self.training,
            };
            return zero_padded_layer.forward(&padded);
        }

        // Validate input shape: [B, C_in, L].
        if input.ndim() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Conv1d expects 3-D input [B, C, L], got {:?}",
                    input.shape()
                ),
            });
        }

        let batch = input.shape()[0];
        let c_in = input.shape()[1];
        let length = input.shape()[2];

        if c_in != self.in_channels {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Conv1d: expected {} input channels, got {}",
                    self.in_channels, c_in
                ),
            });
        }

        let k = self.kernel_size;
        let s = self.stride;
        let p = self.padding;
        let dil = self.dilation;
        let groups = self.groups;

        // Effective kernel extent after dilation, mirroring
        // `ConvUtils.h:255` `kernel = dilation * (weight_size - 1) + 1`.
        let eff_k = dil * (k - 1) + 1;
        let l_padded = length + 2 * p;
        if l_padded < eff_k {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Conv1d: padded input length ({l_padded}) is smaller than effective kernel ({eff_k})"
                ),
            });
        }

        let l_out = (l_padded - eff_k) / s + 1;

        // Save the input device so we can restore it on the output.
        let input_device = input.device();

        // Reshape input [B, C_in, L] -> [B, C_in, 1, L] and use the 2-D dilated
        // im2col with kernel (1, k), stride (1, s), padding (0, p), dilation
        // (1, dil) so the temporal dilation maps to the W axis. The CPU path
        // partitions channels by `groups` exactly like Conv2d: each group's
        // input slice [B, in_per_group, L] is convolved with its weight slice
        // and the outputs are stacked along the C_out axis (mirroring the
        // per-group subtensor/cat loop at `Convolution.cpp:1723-1729`).
        let input_data = input.data_vec()?;
        let weight_data = self.weight.data_vec()?;

        let zero = <T as num_traits::Zero>::zero();
        let mut output = vec![zero; batch * self.out_channels * l_out];

        // Per-group dimensions.
        let in_per_group = self.in_channels / groups;
        let out_per_group = self.out_channels / groups;
        let weight_per_group_numel = out_per_group * in_per_group * k;
        let group_col_rows = in_per_group * k;
        let col_cols = l_out;

        // Saved im2col columns for autograd (dense channel layout `[B,
        // C_in * k, L_out]` so the backward can accumulate grad_input back
        // into a `[B, C_in, L]` tensor uniformly across groups, exactly like
        // Conv2dBackward).
        let saved_cols_rows = self.in_channels * k;
        let mut saved_cols: Vec<T> = if is_grad_enabled()
            && (input.requires_grad()
                || self.weight.requires_grad()
                || self.bias.as_ref().is_some_and(|b| b.requires_grad()))
        {
            vec![zero; batch * saved_cols_rows * col_cols]
        } else {
            Vec::new()
        };
        let save_cols = !saved_cols.is_empty();

        for g in 0..groups {
            // Slice the input channels belonging to this group: [B, in_per_group, L].
            let mut group_input = vec![zero; batch * in_per_group * length];
            for b in 0..batch {
                for c in 0..in_per_group {
                    let src_c = g * in_per_group + c;
                    let src_start = b * self.in_channels * length + src_c * length;
                    let dst_start = b * in_per_group * length + c * length;
                    group_input[dst_start..dst_start + length]
                        .copy_from_slice(&input_data[src_start..src_start + length]);
                }
            }

            let (g_cols, g_col_rows, g_col_cols) = im2col_dilated(
                &group_input,
                batch,
                in_per_group,
                1,
                length,
                1,
                k,
                1,
                s,
                0,
                p,
                1,
                dil,
            );
            debug_assert_eq!(g_col_rows, group_col_rows);
            debug_assert_eq!(g_col_cols, col_cols);

            // Save into the dense [C_in * k, col_cols] layout if backward needs it.
            if save_cols {
                for b in 0..batch {
                    for c in 0..in_per_group {
                        let dst_c = g * in_per_group + c;
                        for kk in 0..k {
                            let src_row = c * k + kk;
                            let dst_row = dst_c * k + kk;
                            let src_off = b * group_col_rows * col_cols + src_row * col_cols;
                            let dst_off = b * saved_cols_rows * col_cols + dst_row * col_cols;
                            saved_cols[dst_off..dst_off + col_cols]
                                .copy_from_slice(&g_cols[src_off..src_off + col_cols]);
                        }
                    }
                }
            }

            // Group's slice of the weight: [out_per_group, in_per_group, k]
            // flattened to [out_per_group, group_col_rows].
            let w_group_start = g * weight_per_group_numel;
            let w_group_end = w_group_start + weight_per_group_numel;
            let weight_group_2d = Tensor::from_storage(
                TensorStorage::cpu(weight_data[w_group_start..w_group_end].to_vec()),
                vec![out_per_group, group_col_rows],
                false,
            )?;

            for b in 0..batch {
                let col_start = b * group_col_rows * col_cols;
                let col_end = col_start + group_col_rows * col_cols;
                let cols_b = Tensor::from_storage(
                    TensorStorage::cpu(g_cols[col_start..col_end].to_vec()),
                    vec![group_col_rows, col_cols],
                    false,
                )?;

                let out_b = mm(&weight_group_2d, &cols_b)?;
                let out_data = out_b.data()?;
                // Place this group's output channels into [b, g*out_per_group.., :].
                for oc in 0..out_per_group {
                    let dst_c = g * out_per_group + oc;
                    let dst_start = b * self.out_channels * l_out + dst_c * l_out;
                    let src_start = oc * l_out;
                    output[dst_start..dst_start + l_out]
                        .copy_from_slice(&out_data[src_start..src_start + l_out]);
                }
            }
        }

        // Add bias if present: broadcast [C_out] over [B, C_out, L_out].
        if let Some(ref bias) = self.bias {
            let bias_data = bias.data_vec()?;
            for b in 0..batch {
                for c in 0..self.out_channels {
                    let bval = bias_data[c];
                    for l in 0..l_out {
                        output[b * self.out_channels * l_out + c * l_out + l] += bval;
                    }
                }
            }
        }

        let result = Tensor::from_storage(
            TensorStorage::cpu(output),
            vec![batch, self.out_channels, l_out],
            false,
        )?;

        // Attach backward if gradients are enabled.
        if save_cols {
            let grad_fn = Arc::new(Conv1dBackward {
                input: input.clone(),
                weight: self.weight.tensor().clone(),
                bias: self.bias.as_ref().map(|b| b.tensor().clone()),
                in_channels: self.in_channels,
                out_channels: self.out_channels,
                kernel_size: self.kernel_size,
                stride: self.stride,
                padding: self.padding,
                dilation: self.dilation,
                groups: self.groups,
                cols: saved_cols,
                col_rows: saved_cols_rows,
                col_cols,
                l_out,
            });
            Tensor::from_operation(
                TensorStorage::cpu(result.data()?.to_vec()),
                result.shape().to_vec(),
                grad_fn,
            )?
            .to(input_device) // restore device
        } else {
            result.to(input_device)
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = vec![("weight".to_string(), &self.weight)];
        if let Some(ref b) = self.bias {
            params.push(("bias".to_string(), b));
        }
        params
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// Conv1dBackward
// ---------------------------------------------------------------------------

/// Backward function for `Conv1d` forward pass.
///
/// Saved `cols` use the **dense channel layout** `[B, C_in * k, L_out]`
/// (the forward saves into this shape regardless of `groups`), mirroring
/// `Conv2dBackward`'s grouped scheme so the per-group slice is taken at
/// gradient-computation time and grad_input accumulates uniformly across
/// groups. `dilation`/`groups` reconstruct the per-group + dilated math.
#[derive(Debug)]
struct Conv1dBackward<T: Float> {
    input: Tensor<T>,
    weight: Tensor<T>,
    bias: Option<Tensor<T>>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    cols: Vec<T>,
    col_rows: usize,
    col_cols: usize,
    l_out: usize,
}

impl<T: Float> GradFn<T> for Conv1dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // grad_output shape: [B, C_out, L_out]
        let input_device = self.input.device();
        let weight_device = self.weight.device();
        let bias_device = self.bias.as_ref().map(|b| b.device());
        let go_data = grad_output.data_vec()?;
        let batch = self.input.shape()[0];
        let length = self.input.shape()[2];
        let k = self.kernel_size;
        let s = self.stride;
        let p = self.padding;
        let dil = self.dilation;
        let groups = self.groups;
        let in_per_group = self.in_channels / groups;
        let out_per_group = self.out_channels / groups;
        let group_col_rows = in_per_group * k;
        let zero = <T as num_traits::Zero>::zero();

        // --- grad_weight ---
        // Per group `g`: gw_g += grad_output_b_g @ cols_b_g^T, stacked along
        // the C_out axis to recover [C_out, C_in/G, k]. Mirrors Conv2dBackward.
        let grad_weight = if self.weight.requires_grad() {
            let weight_numel = self.out_channels * in_per_group * k;
            let mut gw_accum = vec![zero; weight_numel];
            let weight_per_group_numel = out_per_group * group_col_rows;

            for g in 0..groups {
                for b in 0..batch {
                    // Slice grad_output for this group: [out_per_group, l_out].
                    let mut go_g = vec![zero; out_per_group * self.l_out];
                    for oc in 0..out_per_group {
                        let src_c = g * out_per_group + oc;
                        let src_start = b * self.out_channels * self.l_out + src_c * self.l_out;
                        let dst_start = oc * self.l_out;
                        go_g[dst_start..dst_start + self.l_out]
                            .copy_from_slice(&go_data[src_start..src_start + self.l_out]);
                    }
                    let go_b_g = Tensor::from_storage(
                        TensorStorage::cpu(go_g),
                        vec![out_per_group, self.l_out],
                        false,
                    )?;

                    // Slice cols for this group: [in_per_group * k, col_cols].
                    let mut cols_g = vec![zero; group_col_rows * self.col_cols];
                    for c in 0..in_per_group {
                        let src_c = g * in_per_group + c;
                        for kk in 0..k {
                            let src_row = src_c * k + kk;
                            let dst_row = c * k + kk;
                            let src_off =
                                b * self.col_rows * self.col_cols + src_row * self.col_cols;
                            let dst_off = dst_row * self.col_cols;
                            cols_g[dst_off..dst_off + self.col_cols]
                                .copy_from_slice(&self.cols[src_off..src_off + self.col_cols]);
                        }
                    }
                    let cols_b_g = Tensor::from_storage(
                        TensorStorage::cpu(cols_g),
                        vec![group_col_rows, self.col_cols],
                        false,
                    )?;

                    let cols_bt = transpose(&cols_b_g)?;
                    let gw_b = mm(&go_b_g, &cols_bt)?;
                    let gw_data = gw_b.data()?;

                    let dst_off = g * weight_per_group_numel;
                    for i in 0..weight_per_group_numel {
                        gw_accum[dst_off + i] += gw_data[i];
                    }
                }
            }

            Some(
                Tensor::from_storage(
                    TensorStorage::cpu(gw_accum),
                    vec![self.out_channels, in_per_group, k],
                    false,
                )?
                .to(weight_device)?,
            )
        } else {
            None
        };

        // --- grad_bias ---
        // Sum grad_output over batch + length. Bias is per-output-channel
        // ([C_out]), identical for any groups setting.
        let grad_bias = match &self.bias {
            Some(b) if b.requires_grad() => {
                let mut gb = vec![zero; self.out_channels];
                for batch_idx in 0..batch {
                    for c in 0..self.out_channels {
                        for l in 0..self.l_out {
                            gb[c] += go_data
                                [batch_idx * self.out_channels * self.l_out + c * self.l_out + l];
                        }
                    }
                }
                let target_dev = bias_device.unwrap_or(input_device);
                Some(
                    Tensor::from_storage(TensorStorage::cpu(gb), vec![self.out_channels], false)?
                        .to(target_dev)?,
                )
            }
            _ => None,
        };

        // --- grad_input ---
        // Per group `g`: weight_g_2d^T @ grad_output_b_g -> [in_per_group * k,
        // l_out], then col2im_dilated -> [in_per_group, 1, L] placed into the
        // right in-channel slice of [B, C_in, L]. Mirrors Conv2dBackward.
        let grad_input = if self.input.requires_grad() {
            let weight_data = self.weight.data_vec()?;
            let mut grad_input_data = vec![zero; batch * self.in_channels * length];
            let weight_per_group_numel = out_per_group * group_col_rows;

            for g in 0..groups {
                let w_off = g * weight_per_group_numel;
                let weight_g_2d = Tensor::from_storage(
                    TensorStorage::cpu(weight_data[w_off..w_off + weight_per_group_numel].to_vec()),
                    vec![out_per_group, group_col_rows],
                    false,
                )?;
                let weight_g_t = transpose(&weight_g_2d)?;

                let mut grad_cols_g = vec![zero; batch * group_col_rows * self.col_cols];
                for b in 0..batch {
                    let mut go_g = vec![zero; out_per_group * self.l_out];
                    for oc in 0..out_per_group {
                        let src_c = g * out_per_group + oc;
                        let src_start = b * self.out_channels * self.l_out + src_c * self.l_out;
                        let dst_start = oc * self.l_out;
                        go_g[dst_start..dst_start + self.l_out]
                            .copy_from_slice(&go_data[src_start..src_start + self.l_out]);
                    }
                    let go_b_g = Tensor::from_storage(
                        TensorStorage::cpu(go_g),
                        vec![out_per_group, self.l_out],
                        false,
                    )?;

                    let gc_b = mm(&weight_g_t, &go_b_g)?;
                    let gc_data = gc_b.data()?;
                    let gc_start = b * group_col_rows * self.col_cols;
                    grad_cols_g[gc_start..gc_start + group_col_rows * self.col_cols]
                        .copy_from_slice(gc_data);
                }

                // col2im_dilated scatters group's columns back to
                // [B, in_per_group, 1, L]; the W axis carries the dilation.
                let gi_g = col2im_dilated(
                    &grad_cols_g,
                    batch,
                    in_per_group,
                    1,
                    length,
                    1,
                    k,
                    1,
                    s,
                    0,
                    p,
                    1,
                    dil,
                    1,
                    self.l_out,
                );

                for b in 0..batch {
                    for c in 0..in_per_group {
                        let dst_c = g * in_per_group + c;
                        let dst_start = b * self.in_channels * length + dst_c * length;
                        let src_start = b * in_per_group * length + c * length;
                        grad_input_data[dst_start..dst_start + length]
                            .copy_from_slice(&gi_g[src_start..src_start + length]);
                    }
                }
            }

            Some(
                Tensor::from_storage(
                    TensorStorage::cpu(grad_input_data),
                    self.input.shape().to_vec(),
                    false,
                )?
                .to(input_device)?,
            )
        } else {
            None
        };

        let mut grads = vec![grad_input, grad_weight];
        if self.bias.is_some() {
            grads.push(grad_bias);
        }
        Ok(grads)
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        let mut v = vec![&self.input, &self.weight];
        if let Some(ref b) = self.bias {
            v.push(b);
        }
        v
    }

    fn name(&self) -> &'static str {
        "Conv1dBackward"
    }
}

// ---------------------------------------------------------------------------
// ConvTranspose2d
// ---------------------------------------------------------------------------

/// A 2-D transposed convolution (deconvolution) layer.
///
/// Applies a transposed spatial convolution over an input `[B, C_in, H, W]`.
/// Used for upsampling in generative models and decoder networks.
/// Equivalent to `torch.nn.ConvTranspose2d`.
///
/// # Implementation
///
/// The forward pass inserts `(stride - 1)` zeros between each input element
/// (fractionally-strided convolution), then applies a standard convolution
/// with the kernel flipped along both spatial axes.
///
/// # Shape
///
/// - Input: `[B, in_channels, H, W]`
/// - Output: `[B, out_channels, H_out, W_out]`
///
/// where `H_out = (H - 1) * stride.0 - 2 * padding.0 + kernel_size.0 + output_padding.0`.
#[derive(Debug)]
pub struct ConvTranspose2d<T: Float> {
    /// Learnable kernel weights `[in_channels, out_channels, kH, kW]`.
    ///
    /// Note: the channel ordering is transposed compared to `Conv2d`.
    weight: Parameter<T>,
    /// Optional learnable bias `[out_channels]`.
    bias: Option<Parameter<T>>,
    /// Number of input channels.
    in_channels: usize,
    /// Number of output channels.
    out_channels: usize,
    /// Kernel spatial size `(kH, kW)`.
    kernel_size: (usize, usize),
    /// Stride `(sH, sW)`.
    stride: (usize, usize),
    /// Zero-padding `(pH, pW)` removed from both sides of the output.
    padding: (usize, usize),
    /// Additional size added to one side of the output `(opH, opW)`.
    output_padding: (usize, usize),
    /// Whether the module is in training mode.
    training: bool,
}

impl<T: Float> ConvTranspose2d<T> {
    /// Create a new `ConvTranspose2d` layer.
    ///
    /// Weight is initialized with Kaiming uniform (ReLU gain).
    /// Bias, if enabled, is initialized U(-bound, bound) with
    /// `bound = 1/sqrt(fan_in)` per `torch/nn/modules/conv.py:198-201`.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        bias: bool,
    ) -> FerrotorchResult<Self> {
        if in_channels == 0 || out_channels == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "in_channels and out_channels must be > 0".into(),
            });
        }
        if kernel_size.0 == 0 || kernel_size.1 == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "kernel_size must be > 0 in both dimensions".into(),
            });
        }
        if stride.0 == 0 || stride.1 == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "stride must be > 0 in both dimensions".into(),
            });
        }
        if output_padding.0 >= stride.0 || output_padding.1 >= stride.1 {
            return Err(FerrotorchError::InvalidArgument {
                message: "output_padding must be strictly less than stride".into(),
            });
        }

        // Weight shape: [in_channels, out_channels, kH, kW] (transposed layout per
        // `torch/nn/modules/conv.py:161-167`).
        let (kh, kw) = kernel_size;
        let mut weight = Parameter::zeros(&[in_channels, out_channels, kh, kw])?;
        kaiming_uniform(&mut weight, NonLinearity::ReLU)?;

        let bias_param = if bias {
            let mut b = Parameter::zeros(&[out_channels])?;
            // `torch/nn/modules/conv.py:198-201`: bias U(-bound, bound) with
            //   `bound = 1 / sqrt(fan_in)`. For ConvTranspose2d weight shape
            //   `[in_channels, out_channels/groups, kH, kW]`, `_calculate_fan_in_and_fan_out`
            //   yields `fan_in = (out_channels/groups) * kH * kW`. groups=1 here.
            let fan_in = out_channels * kh * kw;
            let bound = if fan_in > 0 {
                1.0 / (fan_in as f64).sqrt()
            } else {
                0.0
            };
            uniform_init(&mut b, -bound, bound)?;
            Some(b)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias: bias_param,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            training: true,
        })
    }

    /// Configure the boundary handling for the spatial padding.
    ///
    /// Only [`crate::padding::PaddingMode::Zeros`] is accepted: upstream
    /// `_ConvTransposeNd.__init__` raises
    /// `ValueError('Only "zeros" padding mode is supported for ConvTranspose2d')`
    /// for any non-`zeros` mode (`torch/nn/modules/conv.py:755-758`). This
    /// matches that behaviour by returning an error rather than silently
    /// accepting the unsupported mode (R-DEV-2). The returned layer is
    /// unchanged (the only valid mode is `Zeros`, the constructed default).
    /// Closes #1443.
    pub fn with_padding_mode(self, mode: crate::padding::PaddingMode) -> FerrotorchResult<Self> {
        reject_non_zeros_transpose(mode, "ConvTranspose2d")?;
        Ok(self)
    }

    /// The number of learnable scalar parameters.
    pub fn num_parameters(&self) -> usize {
        let w = self.in_channels * self.out_channels * self.kernel_size.0 * self.kernel_size.1;
        let b = if self.bias.is_some() {
            self.out_channels
        } else {
            0
        };
        w + b
    }

    /// Build a `ConvTranspose2d` from caller-supplied weight and optional bias.
    ///
    /// `weight` must have shape `[in_channels, out_channels, kH, kW]` (note the
    /// transposed channel ordering vs `Conv2d`). Used by
    /// `nn::functional::conv_transpose2d`.
    pub fn from_parts(
        weight: Tensor<T>,
        bias: Option<Tensor<T>>,
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
    ) -> FerrotorchResult<Self> {
        if weight.ndim() != 4 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "ConvTranspose2d::from_parts: weight must be 4-D [in, out, kH, kW], got {:?}",
                    weight.shape()
                ),
            });
        }
        let in_channels = weight.shape()[0];
        let out_channels = weight.shape()[1];
        let kernel_size = (weight.shape()[2], weight.shape()[3]);
        if output_padding.0 >= stride.0 || output_padding.1 >= stride.1 {
            return Err(FerrotorchError::InvalidArgument {
                message: "output_padding must be strictly less than stride".into(),
            });
        }
        if let Some(b) = &bias {
            if b.ndim() != 1 || b.shape()[0] != out_channels {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "ConvTranspose2d::from_parts: bias shape {:?} != [{}]",
                        b.shape(),
                        out_channels
                    ),
                });
            }
        }
        Ok(Self {
            weight: Parameter::new(weight),
            bias: bias.map(Parameter::new),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            training: true,
        })
    }
}

/// Insert `(stride - 1)` zeros between each element along both spatial axes.
///
/// Given input `[B, C, H, W]`, produces `[B, C, H_up, W_up]` where
/// `H_up = (H - 1) * stride_h + 1` and `W_up = (W - 1) * stride_w + 1`.
fn stride_insert_zeros<T: Float>(
    input: &[T],
    batch: usize,
    channels: usize,
    h: usize,
    w: usize,
    stride_h: usize,
    stride_w: usize,
) -> (Vec<T>, usize, usize) {
    let h_up = (h - 1) * stride_h + 1;
    let w_up = (w - 1) * stride_w + 1;
    let zero = <T as num_traits::Zero>::zero();
    let mut out = vec![zero; batch * channels * h_up * w_up];

    for b in 0..batch {
        for c in 0..channels {
            for ih in 0..h {
                for iw in 0..w {
                    let oh = ih * stride_h;
                    let ow = iw * stride_w;
                    out[b * channels * h_up * w_up + c * h_up * w_up + oh * w_up + ow] =
                        input[b * channels * h * w + c * h * w + ih * w + iw];
                }
            }
        }
    }

    (out, h_up, w_up)
}

/// Flip a kernel along both spatial axes: `kernel[c_in, c_out, kh, kw]` ->
/// `kernel[c_out, c_in, kH-1-kh, kW-1-kw]` (also transposes channel dims).
fn flip_kernel<T: Float>(kernel: &[T], c_in: usize, c_out: usize, kh: usize, kw: usize) -> Vec<T> {
    let zero = <T as num_traits::Zero>::zero();
    let mut flipped = vec![zero; c_out * c_in * kh * kw];

    for ci in 0..c_in {
        for co in 0..c_out {
            for h in 0..kh {
                for w in 0..kw {
                    // Source: [c_in, c_out, h, w]
                    let src = ci * c_out * kh * kw + co * kh * kw + h * kw + w;
                    // Dest: [c_out, c_in, kH-1-h, kW-1-w]
                    let dst = co * c_in * kh * kw + ci * kh * kw + (kh - 1 - h) * kw + (kw - 1 - w);
                    flipped[dst] = kernel[src];
                }
            }
        }
    }

    flipped
}

impl<T: Float> Module<T> for ConvTranspose2d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Record autocast decision for conv_transpose2d.
        let _autocast_cat = autocast_guard("conv_transpose2d");

        // Validate input shape: [B, C_in, H, W].
        if input.ndim() != 4 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ConvTranspose2d expects 4-D input [B, C, H, W], got {:?}",
                    input.shape()
                ),
            });
        }

        let batch = input.shape()[0];
        let c_in = input.shape()[1];
        let h = input.shape()[2];
        let w = input.shape()[3];

        if c_in != self.in_channels {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "ConvTranspose2d: expected {} input channels, got {}",
                    self.in_channels, c_in
                ),
            });
        }

        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        let (oph, opw) = self.output_padding;

        // Save the input device so we can restore it on the output.
        let input_device = input.device();

        // Step 1: Insert zeros between input elements (stride insertion).
        let input_data = input.data_vec()?;
        let (upsampled, h_up_core, w_up_core) =
            stride_insert_zeros(&input_data, batch, c_in, h, w, sh, sw);

        // `output_padding` extends ONE side (bottom rows / right cols) of the
        // output. Those cells are NOT zero-padding: per upstream `col2im`
        // (`aten/src/ATen/native/im2col.h:104-146`) the transposed convolution
        // scatters into an output of size `(in-1)*stride - 2*pad +
        // dilation*(k-1) + output_padding + 1`
        // (`NaiveConvolutionTranspose2d.cpp:109-112`), so the trailing
        // `output_padding` rows/cols DO receive kernel-tap contributions.
        // Append `oph` rows / `opw` cols of zeros to the (stride-inserted)
        // upsampled signal so the internal stride-1 convolution emits all
        // `h_out * w_out` cells directly, instead of leaving the boundary at 0
        // (the #1560 divergence). Closes #1560.
        let h_up = h_up_core + oph;
        let w_up = w_up_core + opw;
        let upsampled = if oph > 0 || opw > 0 {
            let zero = <T as num_traits::Zero>::zero();
            let mut ext = vec![zero; batch * c_in * h_up * w_up];
            for b in 0..batch {
                for c in 0..c_in {
                    for ih in 0..h_up_core {
                        let src = ((b * c_in + c) * h_up_core + ih) * w_up_core;
                        let dst = ((b * c_in + c) * h_up + ih) * w_up;
                        ext[dst..dst + w_up_core].copy_from_slice(&upsampled[src..src + w_up_core]);
                    }
                }
            }
            ext
        } else {
            upsampled
        };

        // Step 2: Flip the kernel and transpose channel dimensions.
        // Weight: [in_channels, out_channels, kH, kW]
        // Flipped: [out_channels, in_channels, kH, kW] with spatial flip.
        let weight_data = self.weight.data_vec()?;
        let flipped = flip_kernel(&weight_data, self.in_channels, self.out_channels, kh, kw);

        // Step 3: Apply a regular convolution on the upsampled input using the
        // flipped kernel. The "padding" for this internal convolution is
        // `kernel_size - 1 - padding` to achieve the correct output size.
        let internal_pad_h = kh - 1 - ph;
        let internal_pad_w = kw - 1 - pw;

        // im2col on the upsampled input with stride=1.
        let (cols, col_rows, col_cols) = im2col(
            &upsampled,
            batch,
            c_in,
            h_up,
            w_up,
            kh,
            kw,
            1,
            1,
            internal_pad_h,
            internal_pad_w,
        );

        // The internal stride-1 convolution over the `output_padding`-extended
        // upsampled signal now emits exactly `h_out * w_out` cells.
        let h_out = (h_up + 2 * internal_pad_h - kh) + 1;
        let w_out = (w_up + 2 * internal_pad_w - kw) + 1;
        debug_assert_eq!(h_out, (h_up_core + 2 * internal_pad_h - kh) + 1 + oph);
        debug_assert_eq!(w_out, (w_up_core + 2 * internal_pad_w - kw) + 1 + opw);

        // Reshape flipped kernel to 2-D: [C_out, C_in * kH * kW]
        let flipped_2d = Tensor::from_storage(
            TensorStorage::cpu(flipped),
            vec![self.out_channels, col_rows],
            false,
        )?;

        // Per-batch matmul.
        let zero = <T as num_traits::Zero>::zero();
        let mut output = vec![zero; batch * self.out_channels * h_out * w_out];

        for b in 0..batch {
            let col_start = b * col_rows * col_cols;
            let col_end = col_start + col_rows * col_cols;
            let cols_b = Tensor::from_storage(
                TensorStorage::cpu(cols[col_start..col_end].to_vec()),
                vec![col_rows, col_cols],
                false,
            )?;

            let out_b = mm(&flipped_2d, &cols_b)?;
            let out_data = out_b.data()?;

            // Copy the full convolution result. The internal conv now emits all
            // `h_out * w_out` cells (including the `output_padding` boundary),
            // so no cell is left at 0 (the #1560 fix). `col_cols == h_out *
            // w_out` here.
            let out_start = b * self.out_channels * h_out * w_out;
            for c in 0..self.out_channels {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        output[out_start + c * h_out * w_out + oh * w_out + ow] =
                            out_data[c * h_out * w_out + oh * w_out + ow];
                    }
                }
            }
        }

        // Add bias if present.
        if let Some(ref bias) = self.bias {
            let bias_data = bias.data_vec()?;
            for b in 0..batch {
                for c in 0..self.out_channels {
                    let bval = bias_data[c];
                    for hw in 0..(h_out * w_out) {
                        output[b * self.out_channels * h_out * w_out + c * h_out * w_out + hw] +=
                            bval;
                    }
                }
            }
        }

        let result = Tensor::from_storage(
            TensorStorage::cpu(output),
            vec![batch, self.out_channels, h_out, w_out],
            false,
        )?;

        // Attach backward if gradients are enabled.
        if is_grad_enabled()
            && (input.requires_grad()
                || self.weight.requires_grad()
                || self.bias.as_ref().is_some_and(|b| b.requires_grad()))
        {
            let grad_fn = Arc::new(ConvTranspose2dBackward {
                input: input.clone(),
                weight: self.weight.tensor().clone(),
                bias: self.bias.as_ref().map(|b| b.tensor().clone()),
                in_channels: self.in_channels,
                out_channels: self.out_channels,
                kernel_size: self.kernel_size,
                stride: self.stride,
                padding: self.padding,
                _output_padding: self.output_padding,
                h_out,
                w_out,
            });
            Tensor::from_operation(
                TensorStorage::cpu(result.data()?.to_vec()),
                result.shape().to_vec(),
                grad_fn,
            )?
            .to(input_device) // restore device
        } else {
            result.to(input_device)
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = vec![("weight".to_string(), &self.weight)];
        if let Some(ref b) = self.bias {
            params.push(("bias".to_string(), b));
        }
        params
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// ConvTranspose2dBackward
// ---------------------------------------------------------------------------

/// Backward function for `ConvTranspose2d` forward pass.
///
/// The backward of a transposed convolution is a regular convolution.
#[derive(Debug)]
struct ConvTranspose2dBackward<T: Float> {
    input: Tensor<T>,
    weight: Tensor<T>,
    bias: Option<Tensor<T>>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    _output_padding: (usize, usize),
    h_out: usize,
    w_out: usize,
}

impl<T: Float> GradFn<T> for ConvTranspose2dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // grad_output shape: [B, C_out, H_out, W_out]
        let go_data = grad_output.data_vec()?;
        let batch = self.input.shape()[0];
        let h_in = self.input.shape()[2];
        let w_in = self.input.shape()[3];
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        // --- grad_input ---
        // The backward of ConvTranspose2d w.r.t. input is a regular Conv2d
        // of grad_output with the *original* (non-flipped) weight.
        // Weight is [in_channels, out_channels, kH, kW], we need it as
        // [in_channels, out_channels, kH, kW] reshaped to [in_channels, out_channels * kH * kW]
        // but actually we need a regular conv: grad_output [B, C_out, H_out, W_out]
        // convolved with weight^T [in_channels, out_channels, kH, kW] -> transposed to
        // [in_channels as filters over C_out channels].
        //
        // Reshape weight [C_in, C_out, kH, kW] -> [C_in, C_out * kH * kW] for matmul.
        let grad_input = if self.input.requires_grad() {
            let weight_data = self.weight.data_vec()?;
            let col_rows = self.out_channels * kh * kw;

            // Reshape weight to [C_in, C_out * kH * kW]
            let weight_2d = Tensor::from_storage(
                TensorStorage::cpu(weight_data),
                vec![self.in_channels, col_rows],
                false,
            )?;

            // im2col on grad_output with the conv parameters
            let (go_cols, _go_col_rows, go_col_cols) = im2col(
                &go_data,
                batch,
                self.out_channels,
                self.h_out,
                self.w_out,
                kh,
                kw,
                sh,
                sw,
                ph,
                pw,
            );

            let zero = <T as num_traits::Zero>::zero();
            let mut gi = vec![zero; batch * self.in_channels * h_in * w_in];

            for b in 0..batch {
                let col_start = b * col_rows * go_col_cols;
                let col_end = col_start + col_rows * go_col_cols;
                let go_cols_b = Tensor::from_storage(
                    TensorStorage::cpu(go_cols[col_start..col_end].to_vec()),
                    vec![col_rows, go_col_cols],
                    false,
                )?;

                let gi_b = mm(&weight_2d, &go_cols_b)?;
                let gi_data = gi_b.data()?;

                let out_start = b * self.in_channels * h_in * w_in;
                let copy_len = self.in_channels * h_in * w_in;
                gi[out_start..out_start + copy_len].copy_from_slice(&gi_data[..copy_len]);
            }

            Some(Tensor::from_storage(
                TensorStorage::cpu(gi),
                self.input.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };

        // --- grad_weight ---
        // grad_weight[c_in, c_out, kh, kw] = sum_b input_b (x) grad_output_b
        // where (x) is the cross-correlation with stride.
        let grad_weight = if self.weight.requires_grad() {
            let zero = <T as num_traits::Zero>::zero();
            let weight_numel = self.in_channels * self.out_channels * kh * kw;
            let mut gw = vec![zero; weight_numel];
            let input_data = self.input.data_vec()?;

            for b in 0..batch {
                for ci in 0..self.in_channels {
                    for co in 0..self.out_channels {
                        for dh in 0..kh {
                            for dw in 0..kw {
                                let mut acc = zero;
                                for ih in 0..h_in {
                                    for iw in 0..w_in {
                                        let oh = ih * sh + dh;
                                        let ow = iw * sw + dw;
                                        // Account for padding removal
                                        if oh >= ph
                                            && ow >= pw
                                            && (oh - ph) < self.h_out
                                            && (ow - pw) < self.w_out
                                        {
                                            let go_idx =
                                                b * self.out_channels * self.h_out * self.w_out
                                                    + co * self.h_out * self.w_out
                                                    + (oh - ph) * self.w_out
                                                    + (ow - pw);
                                            let in_idx = b * self.in_channels * h_in * w_in
                                                + ci * h_in * w_in
                                                + ih * w_in
                                                + iw;
                                            acc += input_data[in_idx] * go_data[go_idx];
                                        }
                                    }
                                }
                                gw[ci * self.out_channels * kh * kw
                                    + co * kh * kw
                                    + dh * kw
                                    + dw] += acc;
                            }
                        }
                    }
                }
            }

            Some(Tensor::from_storage(
                TensorStorage::cpu(gw),
                vec![self.in_channels, self.out_channels, kh, kw],
                false,
            )?)
        } else {
            None
        };

        // --- grad_bias ---
        let grad_bias = match &self.bias {
            Some(b) if b.requires_grad() => {
                let zero = <T as num_traits::Zero>::zero();
                let mut gb = vec![zero; self.out_channels];
                for batch_idx in 0..batch {
                    for c in 0..self.out_channels {
                        for hw in 0..(self.h_out * self.w_out) {
                            gb[c] +=
                                go_data[batch_idx * self.out_channels * self.h_out * self.w_out
                                    + c * self.h_out * self.w_out
                                    + hw];
                        }
                    }
                }
                Some(Tensor::from_storage(
                    TensorStorage::cpu(gb),
                    vec![self.out_channels],
                    false,
                )?)
            }
            _ => None,
        };

        let mut grads = vec![grad_input, grad_weight];
        if self.bias.is_some() {
            grads.push(grad_bias);
        }
        Ok(grads)
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        let mut v = vec![&self.input, &self.weight];
        if let Some(ref b) = self.bias {
            v.push(b);
        }
        v
    }

    fn name(&self) -> &'static str {
        "ConvTranspose2dBackward"
    }
}

// ---------------------------------------------------------------------------
// im2col_3d / col2im_3d helpers
// ---------------------------------------------------------------------------

/// Extract volume patches into columns for 3-D convolution.
///
/// Given a 5-D input `[B, C, D, H, W]`, produces a 3-D output
/// `[B, C * kD * kH * kW, D_out * H_out * W_out]` where each column is one
/// flattened receptive-field patch.
// Internal kernel: argument set mirrors the 3-D convolution descriptor
// (B, C, D, H, W, kD, kH, kW, ...); the 3-D extension of `im2col` carries
// proportionally more arguments than the 2-D version.
#[allow(clippy::too_many_arguments)]
fn im2col_3d<T: Float>(
    input: &[T],
    batch: usize,
    channels: usize,
    depth: usize,
    height: usize,
    width: usize,
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_d: usize,
    stride_h: usize,
    stride_w: usize,
    pad_d: usize,
    pad_h: usize,
    pad_w: usize,
) -> (Vec<T>, usize, usize) {
    im2col_3d_dilated(
        input, batch, channels, depth, height, width, kernel_d, kernel_h, kernel_w, stride_d,
        stride_h, stride_w, pad_d, pad_h, pad_w, 1, 1, 1,
    )
}

/// Extract volumetric patches into columns, supporting dilation
/// `(dil_d, dil_h, dil_w)`.
///
/// Given a 5-D input `[B, C, D, H, W]`, produces
/// `[B, C * kD * kH * kW, D_out * H_out * W_out]` where each column is one
/// flattened receptive-field patch with kernel taps spaced by the dilation
/// factors. Output spatial sizes follow `out = (in + 2*pad - dil*(k - 1) -
/// 1)/stride + 1`, mirroring `ConvUtils.h:255-256`.
// Internal kernel: argument set mirrors the 3-D convolution descriptor; a
// config struct would force allocation on every call in convolution hot paths.
#[allow(clippy::too_many_arguments)]
fn im2col_3d_dilated<T: Float>(
    input: &[T],
    batch: usize,
    channels: usize,
    depth: usize,
    height: usize,
    width: usize,
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_d: usize,
    stride_h: usize,
    stride_w: usize,
    pad_d: usize,
    pad_h: usize,
    pad_w: usize,
    dil_d: usize,
    dil_h: usize,
    dil_w: usize,
) -> (Vec<T>, usize, usize) {
    let eff_kd = dil_d * (kernel_d - 1) + 1;
    let eff_kh = dil_h * (kernel_h - 1) + 1;
    let eff_kw = dil_w * (kernel_w - 1) + 1;
    let d_out = (depth + 2 * pad_d - eff_kd) / stride_d + 1;
    let h_out = (height + 2 * pad_h - eff_kh) / stride_h + 1;
    let w_out = (width + 2 * pad_w - eff_kw) / stride_w + 1;
    let col_rows = channels * kernel_d * kernel_h * kernel_w;
    let col_cols = d_out * h_out * w_out;

    let zero = <T as num_traits::Zero>::zero();
    let mut cols = vec![zero; batch * col_rows * col_cols];

    for b in 0..batch {
        for c in 0..channels {
            for kd in 0..kernel_d {
                for kh in 0..kernel_h {
                    for kw in 0..kernel_w {
                        let row = c * kernel_d * kernel_h * kernel_w
                            + kd * kernel_h * kernel_w
                            + kh * kernel_w
                            + kw;
                        for od in 0..d_out {
                            for oh in 0..h_out {
                                for ow in 0..w_out {
                                    let id = od * stride_d + kd * dil_d;
                                    let ih = oh * stride_h + kh * dil_h;
                                    let iw = ow * stride_w + kw * dil_w;
                                    let col = od * h_out * w_out + oh * w_out + ow;

                                    let val = if id >= pad_d
                                        && ih >= pad_h
                                        && iw >= pad_w
                                        && (id - pad_d) < depth
                                        && (ih - pad_h) < height
                                        && (iw - pad_w) < width
                                    {
                                        let real_d = id - pad_d;
                                        let real_h = ih - pad_h;
                                        let real_w = iw - pad_w;
                                        input[b * channels * depth * height * width
                                            + c * depth * height * width
                                            + real_d * height * width
                                            + real_h * width
                                            + real_w]
                                    } else {
                                        zero
                                    };

                                    cols[b * col_rows * col_cols + row * col_cols + col] = val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    (cols, col_rows, col_cols)
}

/// Scatter columns back into a volume tensor with dilation support
/// (adjoint of `im2col_3d_dilated`). The non-dilated 3-D scatter is simply
/// this with `(dil_d, dil_h, dil_w) = (1, 1, 1)`; production callers
/// (`Conv3dBackward`) always pass the layer's dilation directly, so no
/// separate non-dilated shim is kept.
// Internal kernel: adjoint of `im2col_3d_dilated`; same descriptor signature.
#[allow(clippy::too_many_arguments)]
fn col2im_3d_dilated<T: Float>(
    cols: &[T],
    batch: usize,
    channels: usize,
    depth: usize,
    height: usize,
    width: usize,
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_d: usize,
    stride_h: usize,
    stride_w: usize,
    pad_d: usize,
    pad_h: usize,
    pad_w: usize,
    dil_d: usize,
    dil_h: usize,
    dil_w: usize,
    d_out: usize,
    h_out: usize,
    w_out: usize,
) -> Vec<T> {
    let zero = <T as num_traits::Zero>::zero();
    let mut output = vec![zero; batch * channels * depth * height * width];

    let col_rows = channels * kernel_d * kernel_h * kernel_w;
    let col_cols = d_out * h_out * w_out;

    for b in 0..batch {
        for c in 0..channels {
            for kd in 0..kernel_d {
                for kh in 0..kernel_h {
                    for kw in 0..kernel_w {
                        let row = c * kernel_d * kernel_h * kernel_w
                            + kd * kernel_h * kernel_w
                            + kh * kernel_w
                            + kw;
                        for od in 0..d_out {
                            for oh in 0..h_out {
                                for ow in 0..w_out {
                                    let id = od * stride_d + kd * dil_d;
                                    let ih = oh * stride_h + kh * dil_h;
                                    let iw = ow * stride_w + kw * dil_w;
                                    let col = od * h_out * w_out + oh * w_out + ow;

                                    if id >= pad_d
                                        && ih >= pad_h
                                        && iw >= pad_w
                                        && (id - pad_d) < depth
                                        && (ih - pad_h) < height
                                        && (iw - pad_w) < width
                                    {
                                        let real_d = id - pad_d;
                                        let real_h = ih - pad_h;
                                        let real_w = iw - pad_w;
                                        output[b * channels * depth * height * width
                                            + c * depth * height * width
                                            + real_d * height * width
                                            + real_h * width
                                            + real_w] +=
                                            cols[b * col_rows * col_cols + row * col_cols + col];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Conv3d
// ---------------------------------------------------------------------------

/// A 3-D convolution layer for volumetric data.
///
/// Applies a spatial convolution over an input `[B, C_in, D, H, W]` using
/// the im2col + matmul algorithm. Equivalent to `torch.nn.Conv3d`.
///
/// # Shape
///
/// - Input: `[B, in_channels, D, H, W]`
/// - Output: `[B, out_channels, D_out, H_out, W_out]`
///
/// where `D_out = (D + 2 * padding.0 - kernel_size.0) / stride.0 + 1` (and
/// analogously for H and W).
#[derive(Debug)]
pub struct Conv3d<T: Float> {
    /// Learnable kernel weights `[out_channels, in_channels / groups, kD, kH, kW]`.
    weight: Parameter<T>,
    /// Optional learnable bias `[out_channels]`.
    bias: Option<Parameter<T>>,
    /// Number of input channels.
    in_channels: usize,
    /// Number of output channels (filters).
    out_channels: usize,
    /// Kernel spatial size `(kD, kH, kW)`.
    kernel_size: (usize, usize, usize),
    /// Stride `(sD, sH, sW)`.
    stride: (usize, usize, usize),
    /// Zero-padding `(pD, pH, pW)` applied to both sides.
    padding: (usize, usize, usize),
    /// Dilation `(dilD, dilH, dilW)`. `(1, 1, 1)` is the dense default.
    /// Spaces kernel taps apart along each spatial axis (`eff_kernel =
    /// dilation * (k - 1) + 1`), mirroring `torch.nn.Conv3d(..., dilation=1)`
    /// (`conv.py:689`).
    dilation: (usize, usize, usize),
    /// Number of blocked input/output channel groups. `1` is dense,
    /// `in_channels` is depthwise. Must divide both `in_channels` and
    /// `out_channels`, mirroring `torch.nn.Conv3d(..., groups=1)`
    /// (`conv.py:690`, validation `conv.py:107-110`).
    groups: usize,
    /// Boundary handling for the spatial padding. `Zeros` (default) routes
    /// through the existing im2col zero-pad path; non-`Zeros` modes pre-pad
    /// the input via `crate::padding::functional_pad_3d` and then run the
    /// dense im2col over the already-padded tensor (matching the upstream
    /// `Conv3d._conv_forward`: `F.pad(input, ..., mode=...)` first, then a
    /// `padding=0` convolution). See `torch/nn/modules/conv.py:716-732`.
    /// Closes #1443.
    padding_mode: crate::padding::PaddingMode,
    /// Whether the module is in training mode.
    training: bool,
}

impl<T: Float> Conv3d<T> {
    /// Create a new `Conv3d` layer (dense, dilation `(1, 1, 1)`, `groups = 1`).
    ///
    /// Weight is initialized with Kaiming uniform (ReLU gain).
    /// Bias, if enabled, is initialized U(-bound, bound) with
    /// `bound = 1/sqrt(fan_in)` per `torch/nn/modules/conv.py:198-201`.
    ///
    /// This is a thin shim over [`Conv3d::new_full`] preserved for callers
    /// that only need the dense configuration (e.g. `LazyConv3d::materialize`).
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        bias: bool,
    ) -> FerrotorchResult<Self> {
        Self::new_full(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            (1, 1, 1),
            1,
            bias,
        )
    }

    /// Create a new `Conv3d` layer with the full PyTorch-shaped argument set,
    /// including `dilation` and `groups`.
    ///
    /// `groups` must divide BOTH `in_channels` and `out_channels` (PyTorch
    /// `torch.nn.Conv3d` raises `ValueError` otherwise, `conv.py:107-110`).
    /// `dilation` must be strictly positive in all dimensions. Weight is
    /// initialised with Kaiming uniform (ReLU gain); bias (if enabled) with
    /// U(-bound, bound) where `bound = 1/sqrt(fan_in)`, `fan_in =
    /// (in_channels/groups) * kD * kH * kW` per
    /// `torch/nn/modules/conv.py:198-201`.
    ///
    /// Weight layout is `[out_channels, in_channels / groups, kD, kH, kW]`,
    /// the PyTorch grouped-conv convention (`conv.py:171`). Argument order
    /// `(.., dilation, groups, bias)` mirrors `Conv3d.__init__`
    /// (`conv.py:682-691`, R-DEV-2).
    #[allow(clippy::too_many_arguments)]
    pub fn new_full(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        dilation: (usize, usize, usize),
        groups: usize,
        bias: bool,
    ) -> FerrotorchResult<Self> {
        if in_channels == 0 || out_channels == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "in_channels and out_channels must be > 0".into(),
            });
        }
        if kernel_size.0 == 0 || kernel_size.1 == 0 || kernel_size.2 == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "kernel_size must be > 0 in all dimensions".into(),
            });
        }
        if stride.0 == 0 || stride.1 == 0 || stride.2 == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "stride must be > 0 in all dimensions".into(),
            });
        }
        if dilation.0 == 0 || dilation.1 == 0 || dilation.2 == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Conv3d::new_full: dilation must be > 0 in all dimensions, got {dilation:?}"
                ),
            });
        }
        if groups == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "Conv3d::new_full: groups must be > 0".into(),
            });
        }
        // `torch/nn/modules/conv.py:107-110`: `in_channels % groups != 0`
        // and `out_channels % groups != 0` each raise ValueError.
        if in_channels % groups != 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Conv3d::new_full: groups={groups} must divide in_channels={in_channels}"
                ),
            });
        }
        if out_channels % groups != 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Conv3d::new_full: groups={groups} must divide out_channels={out_channels}"
                ),
            });
        }

        let (kd, kh, kw) = kernel_size;
        // PyTorch weight layout is [C_out, C_in / groups, kD, kH, kW] (`conv.py:171`).
        let mut weight = Parameter::zeros(&[out_channels, in_channels / groups, kd, kh, kw])?;
        kaiming_uniform(&mut weight, NonLinearity::ReLU)?;

        let bias_param = if bias {
            let mut b = Parameter::zeros(&[out_channels])?;
            // `torch/nn/modules/conv.py:198-201`: bias U(-bound, bound) with
            //   `bound = 1 / sqrt(fan_in)`, `fan_in = (in_channels/groups) * kD * kH * kW`.
            let fan_in = (in_channels / groups) * kd * kh * kw;
            let bound = if fan_in > 0 {
                1.0 / (fan_in as f64).sqrt()
            } else {
                0.0
            };
            uniform_init(&mut b, -bound, bound)?;
            Some(b)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias: bias_param,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            padding_mode: crate::padding::PaddingMode::Zeros,
            training: true,
        })
    }

    /// Number of channel groups (`1` is dense, `in_channels` is depthwise).
    pub fn groups(&self) -> usize {
        self.groups
    }

    /// Dilation `(dilD, dilH, dilW)` (`(1, 1, 1)` is the dense default).
    pub fn dilation(&self) -> (usize, usize, usize) {
        self.dilation
    }

    /// Configure the boundary handling for the spatial padding.
    ///
    /// `Zeros` (default) uses the existing im2col zero-pad path.
    /// `Reflect`, `Replicate`, and `Circular` pre-pad the input via
    /// `crate::padding::functional_pad_3d(input, ...)` and then convolve
    /// with `padding = 0`, matching `torch.nn.Conv3d(..., padding_mode=...)`
    /// (`Conv3d._conv_forward`'s `F.pad` shape, `conv.py:716-732`). The pad
    /// is autograd-aware (`Pad3dBackward`), so input gradients flow through
    /// the boundary. Closes #1443.
    pub fn with_padding_mode(mut self, mode: crate::padding::PaddingMode) -> Self {
        self.padding_mode = mode;
        self
    }

    /// The number of learnable scalar parameters.
    pub fn num_parameters(&self) -> usize {
        let w = self.out_channels
            * self.in_channels
            * self.kernel_size.0
            * self.kernel_size.1
            * self.kernel_size.2;
        let b = if self.bias.is_some() {
            self.out_channels
        } else {
            0
        };
        w + b
    }

    /// Build a `Conv3d` from caller-supplied weight and optional bias tensors.
    ///
    /// `weight` must have shape `[out_channels, in_channels, kD, kH, kW]`.
    /// The resulting layer is dense (`groups = 1`, `dilation = (1, 1, 1)`) so
    /// the constructor remains API-compatible with `nn::functional::conv3d`,
    /// which infers `in_channels = weight.shape()[1]` and cannot recover
    /// `groups` from the weight shape alone.
    pub fn from_parts(
        weight: Tensor<T>,
        bias: Option<Tensor<T>>,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
    ) -> FerrotorchResult<Self> {
        if weight.ndim() != 5 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Conv3d::from_parts: weight must be 5-D [out, in, kD, kH, kW], got {:?}",
                    weight.shape()
                ),
            });
        }
        let out_channels = weight.shape()[0];
        let in_channels = weight.shape()[1];
        let kernel_size = (weight.shape()[2], weight.shape()[3], weight.shape()[4]);
        if let Some(b) = &bias {
            if b.ndim() != 1 || b.shape()[0] != out_channels {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "Conv3d::from_parts: bias shape {:?} != [{}]",
                        b.shape(),
                        out_channels
                    ),
                });
            }
        }
        Ok(Self {
            weight: Parameter::new(weight),
            bias: bias.map(Parameter::new),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation: (1, 1, 1),
            groups: 1,
            padding_mode: crate::padding::PaddingMode::Zeros,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for Conv3d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Record autocast decision for conv3d.
        let _autocast_cat = autocast_guard("conv3d");

        // Non-zero padding modes: pre-pad the input with the requested
        // boundary mode and then convolve with padding = 0. Mirrors
        // `torch/nn/modules/conv.py` `Conv3d._conv_forward` (`conv.py:716-732`):
        //   if self.padding_mode != 'zeros':
        //       F.conv3d(F.pad(input, self._reversed_padding_repeated_twice,
        //                      mode=self.padding_mode), ..., padding=_triple(0), ...)
        // For padding `(pd, ph, pw)`, `_reversed_padding_repeated_twice` is
        // `[pw, pw, ph, ph, pd, pd]` (`conv.py:157` reverses the per-dim order
        // because `F.pad` takes paddings in reverse-dim order). The 6-tuple maps
        // to `functional_pad_3d(left=pw, right=pw, top=ph, bottom=ph,
        // front=pd, back=pd)`. The pre-pad is autograd-aware (`Pad3dBackward`)
        // so input gradients flow through the boundary. Closes #1443.
        if self.padding_mode != crate::padding::PaddingMode::Zeros
            && (self.padding.0 != 0 || self.padding.1 != 0 || self.padding.2 != 0)
        {
            let (pd, ph, pw) = self.padding;
            let padded = crate::padding::functional_pad_3d(
                input,
                pw,
                pw,
                ph,
                ph,
                pd,
                pd,
                self.padding_mode,
                <T as num_traits::Zero>::zero(),
            )?;
            // Recurse on a zero-padding variant: build a shallow clone with
            // padding = (0,0,0) and padding_mode = Zeros so the existing
            // im2col-on-zero-pad path runs without re-padding.
            let zero_padded_layer = Conv3d {
                weight: Parameter::new(self.weight.tensor().clone()),
                bias: self
                    .bias
                    .as_ref()
                    .map(|b| Parameter::new(b.tensor().clone())),
                in_channels: self.in_channels,
                out_channels: self.out_channels,
                kernel_size: self.kernel_size,
                stride: self.stride,
                padding: (0, 0, 0),
                dilation: self.dilation,
                groups: self.groups,
                padding_mode: crate::padding::PaddingMode::Zeros,
                training: self.training,
            };
            return zero_padded_layer.forward(&padded);
        }

        // Validate input shape: [B, C_in, D, H, W].
        if input.ndim() != 5 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Conv3d expects 5-D input [B, C, D, H, W], got {:?}",
                    input.shape()
                ),
            });
        }

        let batch = input.shape()[0];
        let c_in = input.shape()[1];
        let d = input.shape()[2];
        let h = input.shape()[3];
        let w = input.shape()[4];

        if c_in != self.in_channels {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Conv3d: expected {} input channels, got {}",
                    self.in_channels, c_in
                ),
            });
        }

        let (kd, kh, kw) = self.kernel_size;
        let (sd, sh, sw) = self.stride;
        let (pd, ph, pw) = self.padding;
        let (dd, dh, dw) = self.dilation;
        let groups = self.groups;

        // Effective kernel extent after dilation (`ConvUtils.h:255`).
        let eff_kd = dd * (kd - 1) + 1;
        let eff_kh = dh * (kh - 1) + 1;
        let eff_kw = dw * (kw - 1) + 1;

        let d_padded = d + 2 * pd;
        let h_padded = h + 2 * ph;
        let w_padded = w + 2 * pw;
        if d_padded < eff_kd || h_padded < eff_kh || w_padded < eff_kw {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Conv3d: padded input ({d_padded}, {h_padded}, {w_padded}) is smaller than effective kernel ({eff_kd}, {eff_kh}, {eff_kw})"
                ),
            });
        }

        let d_out = (d_padded - eff_kd) / sd + 1;
        let h_out = (h_padded - eff_kh) / sh + 1;
        let w_out = (w_padded - eff_kw) / sw + 1;

        // Save the input device so we can restore it on the output.
        let input_device = input.device();

        // ---- CPU path (dense, dilated, grouped, and grouped+dilated) ----
        // Partitions channels by `groups` exactly like Conv2d: each group's
        // input slice [B, in_per_group, D, H, W] is convolved with its weight
        // slice via the dilated 3-D im2col + GEMM and the outputs are stacked
        // along the C_out axis (mirroring `Convolution.cpp:1723-1729`).
        let input_data = input.data_vec()?;
        let weight_data = self.weight.data_vec()?;

        let zero = <T as num_traits::Zero>::zero();
        let spatial_in = d * h * w;
        let spatial_out = d_out * h_out * w_out;
        let mut output = vec![zero; batch * self.out_channels * spatial_out];

        // Per-group dimensions.
        let in_per_group = self.in_channels / groups;
        let out_per_group = self.out_channels / groups;
        let group_col_rows = in_per_group * kd * kh * kw;
        let weight_per_group_numel = out_per_group * group_col_rows;
        let col_cols = spatial_out;

        // Saved im2col columns for autograd (dense channel layout
        // `[B, C_in * kD * kH * kW, D_out*H_out*W_out]`), so the backward
        // accumulates grad_input uniformly across groups (like Conv2dBackward).
        let saved_cols_rows = self.in_channels * kd * kh * kw;
        let mut saved_cols: Vec<T> = if is_grad_enabled()
            && (input.requires_grad()
                || self.weight.requires_grad()
                || self.bias.as_ref().is_some_and(|b| b.requires_grad()))
        {
            vec![zero; batch * saved_cols_rows * col_cols]
        } else {
            Vec::new()
        };
        let save_cols = !saved_cols.is_empty();
        let kvol = kd * kh * kw;

        for g in 0..groups {
            // Slice the input channels for this group: [B, in_per_group, D, H, W].
            let mut group_input = vec![zero; batch * in_per_group * spatial_in];
            for b in 0..batch {
                for c in 0..in_per_group {
                    let src_c = g * in_per_group + c;
                    let src_start = b * self.in_channels * spatial_in + src_c * spatial_in;
                    let dst_start = b * in_per_group * spatial_in + c * spatial_in;
                    group_input[dst_start..dst_start + spatial_in]
                        .copy_from_slice(&input_data[src_start..src_start + spatial_in]);
                }
            }

            let (g_cols, g_col_rows, g_col_cols) = im2col_3d_dilated(
                &group_input,
                batch,
                in_per_group,
                d,
                h,
                w,
                kd,
                kh,
                kw,
                sd,
                sh,
                sw,
                pd,
                ph,
                pw,
                dd,
                dh,
                dw,
            );
            debug_assert_eq!(g_col_rows, group_col_rows);
            debug_assert_eq!(g_col_cols, col_cols);

            // Save into the dense [C_in * kvol, col_cols] layout if needed.
            if save_cols {
                for b in 0..batch {
                    for c in 0..in_per_group {
                        let dst_c = g * in_per_group + c;
                        for kk in 0..kvol {
                            let src_row = c * kvol + kk;
                            let dst_row = dst_c * kvol + kk;
                            let src_off = b * group_col_rows * col_cols + src_row * col_cols;
                            let dst_off = b * saved_cols_rows * col_cols + dst_row * col_cols;
                            saved_cols[dst_off..dst_off + col_cols]
                                .copy_from_slice(&g_cols[src_off..src_off + col_cols]);
                        }
                    }
                }
            }

            // Group's slice of the weight: [out_per_group, in_per_group, kD, kH, kW]
            // flattened to [out_per_group, group_col_rows].
            let w_group_start = g * weight_per_group_numel;
            let w_group_end = w_group_start + weight_per_group_numel;
            let weight_group_2d = Tensor::from_storage(
                TensorStorage::cpu(weight_data[w_group_start..w_group_end].to_vec()),
                vec![out_per_group, group_col_rows],
                false,
            )?;

            for b in 0..batch {
                let col_start = b * group_col_rows * col_cols;
                let col_end = col_start + group_col_rows * col_cols;
                let cols_b = Tensor::from_storage(
                    TensorStorage::cpu(g_cols[col_start..col_end].to_vec()),
                    vec![group_col_rows, col_cols],
                    false,
                )?;

                let out_b = mm(&weight_group_2d, &cols_b)?;
                let out_data = out_b.data()?;
                for oc in 0..out_per_group {
                    let dst_c = g * out_per_group + oc;
                    let dst_start = b * self.out_channels * spatial_out + dst_c * spatial_out;
                    let src_start = oc * spatial_out;
                    output[dst_start..dst_start + spatial_out]
                        .copy_from_slice(&out_data[src_start..src_start + spatial_out]);
                }
            }
        }

        // Add bias if present: broadcast [C_out] over [B, C_out, D_out, H_out, W_out].
        if let Some(ref bias) = self.bias {
            let bias_data = bias.data_vec()?;
            for b in 0..batch {
                for c in 0..self.out_channels {
                    let bval = bias_data[c];
                    for s in 0..spatial_out {
                        output[b * self.out_channels * spatial_out + c * spatial_out + s] += bval;
                    }
                }
            }
        }

        let result = Tensor::from_storage(
            TensorStorage::cpu(output),
            vec![batch, self.out_channels, d_out, h_out, w_out],
            false,
        )?;

        // Attach backward if gradients are enabled and any input/param requires grad.
        if save_cols {
            let grad_fn = Arc::new(Conv3dBackward {
                input: input.clone(),
                weight: self.weight.tensor().clone(),
                bias: self.bias.as_ref().map(|b| b.tensor().clone()),
                in_channels: self.in_channels,
                out_channels: self.out_channels,
                kernel_size: self.kernel_size,
                stride: self.stride,
                padding: self.padding,
                dilation: self.dilation,
                groups: self.groups,
                cols: saved_cols,
                col_rows: saved_cols_rows,
                col_cols,
                d_out,
                h_out,
                w_out,
            });
            Tensor::from_operation(
                TensorStorage::cpu(result.data()?.to_vec()),
                result.shape().to_vec(),
                grad_fn,
            )?
            .to(input_device) // restore device
        } else {
            result.to(input_device)
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = vec![("weight".to_string(), &self.weight)];
        if let Some(ref b) = self.bias {
            params.push(("bias".to_string(), b));
        }
        params
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// Conv3dBackward
// ---------------------------------------------------------------------------

/// Backward function for `Conv3d` forward pass.
///
/// Saved tensors:
/// - `input`: the original 5-D input
/// - `weight`: the 5-D kernel `[C_out, C_in / groups, kD, kH, kW]`
/// - `bias`: optional 1-D bias
/// - `cols`: the dilated im2col_3d columns with **dense channel layout**
///   `[B, C_in * kD * kH * kW, D_out*H_out*W_out]` (saved regardless of
///   `groups`, so the backward takes the per-group slice at gradient time,
///   mirroring `Conv2dBackward`).
/// - `dilation`, `groups`: descriptors to reconstruct the per-group +
///   dilated math.
#[derive(Debug)]
struct Conv3dBackward<T: Float> {
    input: Tensor<T>,
    weight: Tensor<T>,
    bias: Option<Tensor<T>>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    dilation: (usize, usize, usize),
    groups: usize,
    cols: Vec<T>,
    col_rows: usize,
    col_cols: usize,
    d_out: usize,
    h_out: usize,
    w_out: usize,
}

impl<T: Float> GradFn<T> for Conv3dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // grad_output shape: [B, C_out, D_out, H_out, W_out]
        let input_device = self.input.device();
        let weight_device = self.weight.device();
        let bias_device = self.bias.as_ref().map(|b| b.device());
        let go_data = grad_output.data_vec()?;
        let batch = self.input.shape()[0];
        let d = self.input.shape()[2];
        let h = self.input.shape()[3];
        let w = self.input.shape()[4];
        let (kd, kh, kw) = self.kernel_size;
        let (sd, sh, sw) = self.stride;
        let (pd, ph, pw) = self.padding;
        let (dd, dh, dw) = self.dilation;
        let groups = self.groups;
        let in_per_group = self.in_channels / groups;
        let out_per_group = self.out_channels / groups;
        let kvol = kd * kh * kw;
        let group_col_rows = in_per_group * kvol;
        let spatial_in = d * h * w;
        let spatial_out = self.d_out * self.h_out * self.w_out;
        let zero = <T as num_traits::Zero>::zero();

        // --- grad_weight ---
        // Per group `g`: gw_g += grad_output_b_g @ cols_b_g^T, stacked along
        // the C_out axis to recover [C_out, C_in/G, kD, kH, kW]. Mirrors
        // Conv2dBackward.
        let grad_weight = if self.weight.requires_grad() {
            let weight_numel = self.out_channels * group_col_rows;
            let mut gw_accum = vec![zero; weight_numel];
            let weight_per_group_numel = out_per_group * group_col_rows;

            for g in 0..groups {
                for b in 0..batch {
                    // Slice grad_output for this group: [out_per_group, spatial_out].
                    let mut go_g = vec![zero; out_per_group * spatial_out];
                    for oc in 0..out_per_group {
                        let src_c = g * out_per_group + oc;
                        let src_start = b * self.out_channels * spatial_out + src_c * spatial_out;
                        let dst_start = oc * spatial_out;
                        go_g[dst_start..dst_start + spatial_out]
                            .copy_from_slice(&go_data[src_start..src_start + spatial_out]);
                    }
                    let go_b_g = Tensor::from_storage(
                        TensorStorage::cpu(go_g),
                        vec![out_per_group, spatial_out],
                        false,
                    )?;

                    // Slice cols for this group: [in_per_group * kvol, col_cols].
                    let mut cols_g = vec![zero; group_col_rows * self.col_cols];
                    for c in 0..in_per_group {
                        let src_c = g * in_per_group + c;
                        for kk in 0..kvol {
                            let src_row = src_c * kvol + kk;
                            let dst_row = c * kvol + kk;
                            let src_off =
                                b * self.col_rows * self.col_cols + src_row * self.col_cols;
                            let dst_off = dst_row * self.col_cols;
                            cols_g[dst_off..dst_off + self.col_cols]
                                .copy_from_slice(&self.cols[src_off..src_off + self.col_cols]);
                        }
                    }
                    let cols_b_g = Tensor::from_storage(
                        TensorStorage::cpu(cols_g),
                        vec![group_col_rows, self.col_cols],
                        false,
                    )?;

                    let cols_bt = transpose(&cols_b_g)?;
                    let gw_b = mm(&go_b_g, &cols_bt)?;
                    let gw_data = gw_b.data()?;

                    let dst_off = g * weight_per_group_numel;
                    for i in 0..weight_per_group_numel {
                        gw_accum[dst_off + i] += gw_data[i];
                    }
                }
            }

            Some(
                Tensor::from_storage(
                    TensorStorage::cpu(gw_accum),
                    vec![self.out_channels, in_per_group, kd, kh, kw],
                    false,
                )?
                .to(weight_device)?,
            )
        } else {
            None
        };

        // --- grad_bias ---
        // Sum grad_output over batch + spatial. Bias is per-output-channel,
        // identical for any groups setting.
        let grad_bias = match &self.bias {
            Some(b) if b.requires_grad() => {
                let mut gb = vec![zero; self.out_channels];
                for batch_idx in 0..batch {
                    for c in 0..self.out_channels {
                        for s in 0..spatial_out {
                            gb[c] += go_data
                                [batch_idx * self.out_channels * spatial_out + c * spatial_out + s];
                        }
                    }
                }
                let target_dev = bias_device.unwrap_or(input_device);
                Some(
                    Tensor::from_storage(TensorStorage::cpu(gb), vec![self.out_channels], false)?
                        .to(target_dev)?,
                )
            }
            _ => None,
        };

        // --- grad_input ---
        // Per group `g`: weight_g_2d^T @ grad_output_b_g -> [in_per_group *
        // kvol, spatial_out], then col2im_3d_dilated -> [in_per_group, D, H, W]
        // placed into the right in-channel slice of [B, C_in, D, H, W].
        // Mirrors Conv2dBackward.
        let grad_input = if self.input.requires_grad() {
            let weight_data = self.weight.data_vec()?;
            let mut grad_input_data = vec![zero; batch * self.in_channels * spatial_in];
            let weight_per_group_numel = out_per_group * group_col_rows;

            for g in 0..groups {
                let w_off = g * weight_per_group_numel;
                let weight_g_2d = Tensor::from_storage(
                    TensorStorage::cpu(weight_data[w_off..w_off + weight_per_group_numel].to_vec()),
                    vec![out_per_group, group_col_rows],
                    false,
                )?;
                let weight_g_t = transpose(&weight_g_2d)?;

                let mut grad_cols_g = vec![zero; batch * group_col_rows * self.col_cols];
                for b in 0..batch {
                    let mut go_g = vec![zero; out_per_group * spatial_out];
                    for oc in 0..out_per_group {
                        let src_c = g * out_per_group + oc;
                        let src_start = b * self.out_channels * spatial_out + src_c * spatial_out;
                        let dst_start = oc * spatial_out;
                        go_g[dst_start..dst_start + spatial_out]
                            .copy_from_slice(&go_data[src_start..src_start + spatial_out]);
                    }
                    let go_b_g = Tensor::from_storage(
                        TensorStorage::cpu(go_g),
                        vec![out_per_group, spatial_out],
                        false,
                    )?;

                    let gc_b = mm(&weight_g_t, &go_b_g)?;
                    let gc_data = gc_b.data()?;
                    let gc_start = b * group_col_rows * self.col_cols;
                    grad_cols_g[gc_start..gc_start + group_col_rows * self.col_cols]
                        .copy_from_slice(gc_data);
                }

                // col2im_3d_dilated scatters group's columns back to
                // [B, in_per_group, D, H, W] honouring the dilation factors.
                let gi_g = col2im_3d_dilated(
                    &grad_cols_g,
                    batch,
                    in_per_group,
                    d,
                    h,
                    w,
                    kd,
                    kh,
                    kw,
                    sd,
                    sh,
                    sw,
                    pd,
                    ph,
                    pw,
                    dd,
                    dh,
                    dw,
                    self.d_out,
                    self.h_out,
                    self.w_out,
                );

                for b in 0..batch {
                    for c in 0..in_per_group {
                        let dst_c = g * in_per_group + c;
                        let dst_start = b * self.in_channels * spatial_in + dst_c * spatial_in;
                        let src_start = b * in_per_group * spatial_in + c * spatial_in;
                        grad_input_data[dst_start..dst_start + spatial_in]
                            .copy_from_slice(&gi_g[src_start..src_start + spatial_in]);
                    }
                }
            }

            Some(
                Tensor::from_storage(
                    TensorStorage::cpu(grad_input_data),
                    self.input.shape().to_vec(),
                    false,
                )?
                .to(input_device)?,
            )
        } else {
            None
        };

        // Return exactly as many gradients as inputs() returns.
        let mut grads = vec![grad_input, grad_weight];
        if self.bias.is_some() {
            grads.push(grad_bias);
        }
        Ok(grads)
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        let mut v = vec![&self.input, &self.weight];
        if let Some(ref b) = self.bias {
            v.push(b);
        }
        v
    }

    fn name(&self) -> &'static str {
        "Conv3dBackward"
    }
}

// ---------------------------------------------------------------------------
// ConvTranspose1d
// ---------------------------------------------------------------------------

/// A 1-D transposed convolution (deconvolution) layer.
///
/// Applies a transposed temporal convolution over an input `[B, C_in, L]`.
/// Used for upsampling in generative models and decoder networks.
/// Equivalent to `torch.nn.ConvTranspose1d`.
///
/// # Implementation
///
/// Delegates to the 2-D transposed convolution by adding a dummy spatial
/// dimension (H=1), then squeezes the output back to 3-D.
///
/// # Shape
///
/// - Input: `[B, in_channels, L]`
/// - Output: `[B, out_channels, L_out]`
///
/// where `L_out = (L - 1) * stride - 2 * padding + kernel_size + output_padding`.
#[derive(Debug)]
pub struct ConvTranspose1d<T: Float> {
    /// Learnable kernel weights `[in_channels, out_channels, kernel_size]`.
    ///
    /// Note: the channel ordering is transposed compared to `Conv1d`.
    weight: Parameter<T>,
    /// Optional learnable bias `[out_channels]`.
    bias: Option<Parameter<T>>,
    /// Number of input channels.
    in_channels: usize,
    /// Number of output channels.
    out_channels: usize,
    /// Kernel length.
    kernel_size: usize,
    /// Stride.
    stride: usize,
    /// Zero-padding removed from both sides of the output.
    padding: usize,
    /// Additional size added to one side of the output.
    output_padding: usize,
    /// Whether the module is in training mode.
    training: bool,
}

impl<T: Float> ConvTranspose1d<T> {
    /// Create a new `ConvTranspose1d` layer.
    ///
    /// Weight is initialized with Kaiming uniform (ReLU gain).
    /// Bias, if enabled, is initialized U(-bound, bound) with
    /// `bound = 1/sqrt(fan_in)` per `torch/nn/modules/conv.py:198-201`.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        bias: bool,
    ) -> FerrotorchResult<Self> {
        if in_channels == 0 || out_channels == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "in_channels and out_channels must be > 0".into(),
            });
        }
        if kernel_size == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "kernel_size must be > 0".into(),
            });
        }
        if stride == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "stride must be > 0".into(),
            });
        }
        if output_padding >= stride {
            return Err(FerrotorchError::InvalidArgument {
                message: "output_padding must be strictly less than stride".into(),
            });
        }

        // Weight shape: [in_channels, out_channels, kernel_size] (transposed layout).
        let mut weight = Parameter::zeros(&[in_channels, out_channels, kernel_size])?;
        kaiming_uniform(&mut weight, NonLinearity::ReLU)?;

        let bias_param = if bias {
            let mut b = Parameter::zeros(&[out_channels])?;
            // `torch/nn/modules/conv.py:198-201`: bias U(-bound, bound) with
            //   `bound = 1 / sqrt(fan_in)`. ConvTranspose1d: weight shape
            //   `[in_channels, out_channels/groups, K]`, fan_in = out_channels * K (groups=1).
            let fan_in = out_channels * kernel_size;
            let bound = if fan_in > 0 {
                1.0 / (fan_in as f64).sqrt()
            } else {
                0.0
            };
            uniform_init(&mut b, -bound, bound)?;
            Some(b)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias: bias_param,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            training: true,
        })
    }

    /// Configure the boundary handling for the spatial padding.
    ///
    /// Only [`crate::padding::PaddingMode::Zeros`] is accepted: upstream
    /// `_ConvTransposeNd.__init__` raises
    /// `ValueError('Only "zeros" padding mode is supported for ConvTranspose1d')`
    /// for any non-`zeros` mode (`torch/nn/modules/conv.py:755-758`). This
    /// matches that behaviour by returning an error rather than silently
    /// accepting the unsupported mode (R-DEV-2). The returned layer is
    /// unchanged (the only valid mode is `Zeros`, the constructed default).
    /// Closes #1443.
    pub fn with_padding_mode(self, mode: crate::padding::PaddingMode) -> FerrotorchResult<Self> {
        reject_non_zeros_transpose(mode, "ConvTranspose1d")?;
        Ok(self)
    }

    /// The number of learnable scalar parameters.
    pub fn num_parameters(&self) -> usize {
        let w = self.in_channels * self.out_channels * self.kernel_size;
        let b = if self.bias.is_some() {
            self.out_channels
        } else {
            0
        };
        w + b
    }

    /// Build a `ConvTranspose1d` from caller-supplied weight and optional bias.
    ///
    /// `weight` must have shape `[in_channels, out_channels, kernel_size]`
    /// (transposed channel ordering vs `Conv1d`). Used by
    /// `nn::functional::conv_transpose1d`.
    pub fn from_parts(
        weight: Tensor<T>,
        bias: Option<Tensor<T>>,
        stride: usize,
        padding: usize,
        output_padding: usize,
    ) -> FerrotorchResult<Self> {
        if weight.ndim() != 3 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "ConvTranspose1d::from_parts: weight must be 3-D [in, out, k], got {:?}",
                    weight.shape()
                ),
            });
        }
        let in_channels = weight.shape()[0];
        let out_channels = weight.shape()[1];
        let kernel_size = weight.shape()[2];
        if output_padding >= stride {
            return Err(FerrotorchError::InvalidArgument {
                message: "output_padding must be strictly less than stride".into(),
            });
        }
        if let Some(b) = &bias {
            if b.ndim() != 1 || b.shape()[0] != out_channels {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "ConvTranspose1d::from_parts: bias shape {:?} != [{}]",
                        b.shape(),
                        out_channels
                    ),
                });
            }
        }
        Ok(Self {
            weight: Parameter::new(weight),
            bias: bias.map(Parameter::new),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for ConvTranspose1d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Record autocast decision for conv_transpose1d.
        let _autocast_cat = autocast_guard("conv_transpose1d");

        // Validate input shape: [B, C_in, L].
        if input.ndim() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ConvTranspose1d expects 3-D input [B, C, L], got {:?}",
                    input.shape()
                ),
            });
        }

        let batch = input.shape()[0];
        let c_in = input.shape()[1];
        let length = input.shape()[2];

        if c_in != self.in_channels {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "ConvTranspose1d: expected {} input channels, got {}",
                    self.in_channels, c_in
                ),
            });
        }

        let k = self.kernel_size;
        let s = self.stride;
        let p = self.padding;
        let op = self.output_padding;

        // Save the input device so we can restore it on the output.
        let input_device = input.device();

        // Step 1: Insert zeros between input elements (stride insertion).
        // Treat [B, C, L] as [B, C, 1, L] for the 2-D helper.
        let input_data = input.data_vec()?;
        let (upsampled, _h_up, w_up_core) =
            stride_insert_zeros(&input_data, batch, c_in, 1, length, 1, s);

        // `output_padding` extends ONE side of the output by `op` cells. Those
        // cells are NOT zero-padding: per upstream `col2im`
        // (`aten/src/ATen/native/im2col.h:104-146`), the transposed convolution
        // scatters into an output of size `(L_in-1)*stride - 2*pad +
        // dilation*(k-1) + output_padding + 1`
        // (`NaiveConvolutionTranspose2d.cpp:109-112`), so the trailing
        // `output_padding` positions DO receive kernel-tap contributions. We
        // realise that by appending `op` trailing zeros to the (stride-inserted)
        // upsampled signal so the internal stride-1 convolution emits all
        // `l_out` columns directly, rather than copying `w_out_base` columns and
        // leaving the boundary at 0 (the #1560 divergence). Closes #1560.
        let w_up = w_up_core + op;
        let upsampled = if op > 0 {
            let zero = <T as num_traits::Zero>::zero();
            let mut ext = vec![zero; batch * c_in * w_up];
            for b in 0..batch {
                for c in 0..c_in {
                    let src = (b * c_in + c) * w_up_core;
                    let dst = (b * c_in + c) * w_up;
                    ext[dst..dst + w_up_core].copy_from_slice(&upsampled[src..src + w_up_core]);
                }
            }
            ext
        } else {
            upsampled
        };

        // Step 2: Flip the kernel and transpose channel dimensions.
        // Weight: [in_channels, out_channels, k] -> treat as [in_channels, out_channels, 1, k]
        let weight_data = self.weight.data_vec()?;
        let flipped = flip_kernel(&weight_data, self.in_channels, self.out_channels, 1, k);

        // Step 3: Apply a regular convolution on the upsampled input using the
        // flipped kernel with internal padding.
        let internal_pad_w = k - 1 - p;

        // im2col on the upsampled input [B, C, 1, w_up] with kernel (1, k), stride (1, 1).
        let (cols, col_rows, col_cols) = im2col(
            &upsampled,
            batch,
            c_in,
            1,
            w_up,
            1,
            k,
            1,
            1,
            0,
            internal_pad_w,
        );

        // The internal stride-1 convolution over the `op`-extended upsampled
        // signal now emits exactly `l_out` columns.
        let l_out = (w_up + 2 * internal_pad_w - k) + 1;
        debug_assert_eq!(l_out, (w_up_core + 2 * internal_pad_w - k) + 1 + op);

        // Reshape flipped kernel to 2-D: [C_out, C_in * 1 * k]
        let flipped_2d = Tensor::from_storage(
            TensorStorage::cpu(flipped),
            vec![self.out_channels, col_rows],
            false,
        )?;

        // Per-batch matmul.
        let zero = <T as num_traits::Zero>::zero();
        let mut output = vec![zero; batch * self.out_channels * l_out];

        for b in 0..batch {
            let col_start = b * col_rows * col_cols;
            let col_end = col_start + col_rows * col_cols;
            let cols_b = Tensor::from_storage(
                TensorStorage::cpu(cols[col_start..col_end].to_vec()),
                vec![col_rows, col_cols],
                false,
            )?;

            let out_b = mm(&flipped_2d, &cols_b)?;
            let out_data = out_b.data()?;

            // Copy the full convolution result. The internal conv now emits all
            // `l_out` columns (including the `output_padding` boundary), so no
            // cell is left at 0 (the #1560 fix). `col_cols == l_out` here.
            let out_start = b * self.out_channels * l_out;
            for c in 0..self.out_channels {
                for ow in 0..l_out {
                    output[out_start + c * l_out + ow] = out_data[c * l_out + ow];
                }
            }
        }

        // Add bias if present.
        if let Some(ref bias) = self.bias {
            let bias_data = bias.data_vec()?;
            for b in 0..batch {
                for c in 0..self.out_channels {
                    let bval = bias_data[c];
                    for l in 0..l_out {
                        output[b * self.out_channels * l_out + c * l_out + l] += bval;
                    }
                }
            }
        }

        let result = Tensor::from_storage(
            TensorStorage::cpu(output),
            vec![batch, self.out_channels, l_out],
            false,
        )?;

        // Attach backward if gradients are enabled.
        if is_grad_enabled()
            && (input.requires_grad()
                || self.weight.requires_grad()
                || self.bias.as_ref().is_some_and(|b| b.requires_grad()))
        {
            let grad_fn = Arc::new(ConvTranspose1dBackward {
                input: input.clone(),
                weight: self.weight.tensor().clone(),
                bias: self.bias.as_ref().map(|b| b.tensor().clone()),
                in_channels: self.in_channels,
                out_channels: self.out_channels,
                kernel_size: self.kernel_size,
                stride: self.stride,
                padding: self.padding,
                _output_padding: self.output_padding,
                l_out,
            });
            Tensor::from_operation(
                TensorStorage::cpu(result.data()?.to_vec()),
                result.shape().to_vec(),
                grad_fn,
            )?
            .to(input_device) // restore device
        } else {
            result.to(input_device)
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = vec![("weight".to_string(), &self.weight)];
        if let Some(ref b) = self.bias {
            params.push(("bias".to_string(), b));
        }
        params
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// ConvTranspose1dBackward
// ---------------------------------------------------------------------------

/// Backward function for `ConvTranspose1d` forward pass.
///
/// The backward of a transposed convolution is a regular convolution.
#[derive(Debug)]
struct ConvTranspose1dBackward<T: Float> {
    input: Tensor<T>,
    weight: Tensor<T>,
    bias: Option<Tensor<T>>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    _output_padding: usize,
    l_out: usize,
}

impl<T: Float> GradFn<T> for ConvTranspose1dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // grad_output shape: [B, C_out, L_out]
        let go_data = grad_output.data_vec()?;
        let batch = self.input.shape()[0];
        let l_in = self.input.shape()[2];
        let k = self.kernel_size;
        let s = self.stride;
        let p = self.padding;

        // --- grad_input ---
        // The backward of ConvTranspose1d w.r.t. input is a regular Conv1d
        // of grad_output with the original (non-flipped) weight.
        // Weight is [C_in, C_out, k], treat as [C_in, C_out, 1, k].
        let grad_input = if self.input.requires_grad() {
            let weight_data = self.weight.data_vec()?;
            let col_rows = self.out_channels * k;

            // Reshape weight to [C_in, C_out * k]
            let weight_2d = Tensor::from_storage(
                TensorStorage::cpu(weight_data),
                vec![self.in_channels, col_rows],
                false,
            )?;

            // im2col on grad_output [B, C_out, L_out] treated as [B, C_out, 1, L_out]
            let (go_cols, _go_col_rows, go_col_cols) = im2col(
                &go_data,
                batch,
                self.out_channels,
                1,
                self.l_out,
                1,
                k,
                1,
                s,
                0,
                p,
            );

            let zero = <T as num_traits::Zero>::zero();
            let mut gi = vec![zero; batch * self.in_channels * l_in];

            for b in 0..batch {
                let col_start = b * col_rows * go_col_cols;
                let col_end = col_start + col_rows * go_col_cols;
                let go_cols_b = Tensor::from_storage(
                    TensorStorage::cpu(go_cols[col_start..col_end].to_vec()),
                    vec![col_rows, go_col_cols],
                    false,
                )?;

                let gi_b = mm(&weight_2d, &go_cols_b)?;
                let gi_data = gi_b.data()?;

                let out_start = b * self.in_channels * l_in;
                let copy_len = self.in_channels * l_in;
                gi[out_start..out_start + copy_len].copy_from_slice(&gi_data[..copy_len]);
            }

            Some(Tensor::from_storage(
                TensorStorage::cpu(gi),
                self.input.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };

        // --- grad_weight ---
        // grad_weight[c_in, c_out, kw] = sum_b input_b cross-correlated with grad_output_b
        let grad_weight = if self.weight.requires_grad() {
            let zero = <T as num_traits::Zero>::zero();
            let weight_numel = self.in_channels * self.out_channels * k;
            let mut gw = vec![zero; weight_numel];
            let input_data = self.input.data_vec()?;

            for b in 0..batch {
                for ci in 0..self.in_channels {
                    for co in 0..self.out_channels {
                        for dw in 0..k {
                            let mut acc = zero;
                            for il in 0..l_in {
                                let ow = il * s + dw;
                                if ow >= p && (ow - p) < self.l_out {
                                    let go_idx = b * self.out_channels * self.l_out
                                        + co * self.l_out
                                        + (ow - p);
                                    let in_idx = b * self.in_channels * l_in + ci * l_in + il;
                                    acc += input_data[in_idx] * go_data[go_idx];
                                }
                            }
                            gw[ci * self.out_channels * k + co * k + dw] += acc;
                        }
                    }
                }
            }

            Some(Tensor::from_storage(
                TensorStorage::cpu(gw),
                vec![self.in_channels, self.out_channels, k],
                false,
            )?)
        } else {
            None
        };

        // --- grad_bias ---
        let grad_bias = match &self.bias {
            Some(b) if b.requires_grad() => {
                let zero = <T as num_traits::Zero>::zero();
                let mut gb = vec![zero; self.out_channels];
                for batch_idx in 0..batch {
                    for c in 0..self.out_channels {
                        for l in 0..self.l_out {
                            gb[c] += go_data
                                [batch_idx * self.out_channels * self.l_out + c * self.l_out + l];
                        }
                    }
                }
                Some(Tensor::from_storage(
                    TensorStorage::cpu(gb),
                    vec![self.out_channels],
                    false,
                )?)
            }
            _ => None,
        };

        let mut grads = vec![grad_input, grad_weight];
        if self.bias.is_some() {
            grads.push(grad_bias);
        }
        Ok(grads)
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        let mut v = vec![&self.input, &self.weight];
        if let Some(ref b) = self.bias {
            v.push(b);
        }
        v
    }

    fn name(&self) -> &'static str {
        "ConvTranspose1dBackward"
    }
}

// ---------------------------------------------------------------------------
// ConvTranspose3d
// ---------------------------------------------------------------------------

/// A 3-D transposed convolution (deconvolution) layer.
///
/// Applies a transposed volumetric convolution over an input `[B, C_in, D, H, W]`.
/// Used for upsampling in generative models and 3-D decoder networks.
/// Equivalent to `torch.nn.ConvTranspose3d`.
///
/// # Implementation
///
/// The forward pass inserts `(stride - 1)` zeros between each input element
/// along all three spatial axes (fractionally-strided convolution), then applies
/// a standard 3-D convolution with the kernel flipped along all spatial axes.
///
/// # Shape
///
/// - Input: `[B, in_channels, D, H, W]`
/// - Output: `[B, out_channels, D_out, H_out, W_out]`
///
/// where `D_out = (D - 1) * stride.0 - 2 * padding.0 + kernel_size.0 + output_padding.0`
/// (and analogously for H and W).
#[derive(Debug)]
pub struct ConvTranspose3d<T: Float> {
    /// Learnable kernel weights `[in_channels, out_channels, kD, kH, kW]`.
    ///
    /// Note: the channel ordering is transposed compared to `Conv3d`.
    weight: Parameter<T>,
    /// Optional learnable bias `[out_channels]`.
    bias: Option<Parameter<T>>,
    /// Number of input channels.
    in_channels: usize,
    /// Number of output channels.
    out_channels: usize,
    /// Kernel spatial size `(kD, kH, kW)`.
    kernel_size: (usize, usize, usize),
    /// Stride `(sD, sH, sW)`.
    stride: (usize, usize, usize),
    /// Zero-padding `(pD, pH, pW)` removed from both sides of the output.
    padding: (usize, usize, usize),
    /// Additional size added to one side of the output `(opD, opH, opW)`.
    output_padding: (usize, usize, usize),
    /// Whether the module is in training mode.
    training: bool,
}

impl<T: Float> ConvTranspose3d<T> {
    /// Create a new `ConvTranspose3d` layer.
    ///
    /// Weight is initialized with Kaiming uniform (ReLU gain).
    /// Bias, if enabled, is initialized U(-bound, bound) with
    /// `bound = 1/sqrt(fan_in)` per `torch/nn/modules/conv.py:198-201`.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        output_padding: (usize, usize, usize),
        bias: bool,
    ) -> FerrotorchResult<Self> {
        if in_channels == 0 || out_channels == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "in_channels and out_channels must be > 0".into(),
            });
        }
        if kernel_size.0 == 0 || kernel_size.1 == 0 || kernel_size.2 == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "kernel_size must be > 0 in all dimensions".into(),
            });
        }
        if stride.0 == 0 || stride.1 == 0 || stride.2 == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "stride must be > 0 in all dimensions".into(),
            });
        }
        if output_padding.0 >= stride.0
            || output_padding.1 >= stride.1
            || output_padding.2 >= stride.2
        {
            return Err(FerrotorchError::InvalidArgument {
                message: "output_padding must be strictly less than stride in all dimensions"
                    .into(),
            });
        }

        // Weight shape: [in_channels, out_channels, kD, kH, kW] (transposed layout).
        let (kd, kh, kw) = kernel_size;
        let mut weight = Parameter::zeros(&[in_channels, out_channels, kd, kh, kw])?;
        kaiming_uniform(&mut weight, NonLinearity::ReLU)?;

        let bias_param = if bias {
            let mut b = Parameter::zeros(&[out_channels])?;
            // `torch/nn/modules/conv.py:198-201`: bias U(-bound, bound) with
            //   `bound = 1 / sqrt(fan_in)`. ConvTranspose3d: fan_in = out_channels * kD*kH*kW.
            let fan_in = out_channels * kd * kh * kw;
            let bound = if fan_in > 0 {
                1.0 / (fan_in as f64).sqrt()
            } else {
                0.0
            };
            uniform_init(&mut b, -bound, bound)?;
            Some(b)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias: bias_param,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            training: true,
        })
    }

    /// Configure the boundary handling for the spatial padding.
    ///
    /// Only [`crate::padding::PaddingMode::Zeros`] is accepted: upstream
    /// `_ConvTransposeNd.__init__` raises
    /// `ValueError('Only "zeros" padding mode is supported for ConvTranspose3d')`
    /// for any non-`zeros` mode (`torch/nn/modules/conv.py:755-758`). This
    /// matches that behaviour by returning an error rather than silently
    /// accepting the unsupported mode (R-DEV-2). The returned layer is
    /// unchanged (the only valid mode is `Zeros`, the constructed default).
    /// Closes #1443.
    pub fn with_padding_mode(self, mode: crate::padding::PaddingMode) -> FerrotorchResult<Self> {
        reject_non_zeros_transpose(mode, "ConvTranspose3d")?;
        Ok(self)
    }

    /// The number of learnable scalar parameters.
    pub fn num_parameters(&self) -> usize {
        let w = self.in_channels
            * self.out_channels
            * self.kernel_size.0
            * self.kernel_size.1
            * self.kernel_size.2;
        let b = if self.bias.is_some() {
            self.out_channels
        } else {
            0
        };
        w + b
    }

    /// Build a `ConvTranspose3d` from caller-supplied weight and optional bias.
    ///
    /// `weight` must have shape `[in_channels, out_channels, kD, kH, kW]`
    /// (transposed channel ordering vs `Conv3d`). Used by
    /// `nn::functional::conv_transpose3d`.
    pub fn from_parts(
        weight: Tensor<T>,
        bias: Option<Tensor<T>>,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        output_padding: (usize, usize, usize),
    ) -> FerrotorchResult<Self> {
        if weight.ndim() != 5 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "ConvTranspose3d::from_parts: weight must be 5-D [in, out, kD, kH, kW], got {:?}",
                    weight.shape()
                ),
            });
        }
        let in_channels = weight.shape()[0];
        let out_channels = weight.shape()[1];
        let kernel_size = (weight.shape()[2], weight.shape()[3], weight.shape()[4]);
        if output_padding.0 >= stride.0
            || output_padding.1 >= stride.1
            || output_padding.2 >= stride.2
        {
            return Err(FerrotorchError::InvalidArgument {
                message: "output_padding must be strictly less than stride in all dimensions"
                    .into(),
            });
        }
        if let Some(b) = &bias {
            if b.ndim() != 1 || b.shape()[0] != out_channels {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "ConvTranspose3d::from_parts: bias shape {:?} != [{}]",
                        b.shape(),
                        out_channels
                    ),
                });
            }
        }
        Ok(Self {
            weight: Parameter::new(weight),
            bias: bias.map(Parameter::new),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            training: true,
        })
    }
}

/// Insert `(stride - 1)` zeros between each element along three spatial axes.
///
/// Given input `[B, C, D, H, W]`, produces `[B, C, D_up, H_up, W_up]` where
/// `D_up = (D - 1) * stride_d + 1` (and analogously for H, W).
// Internal kernel for ConvTranspose3d backward: arguments are the 3-D
// shape descriptor + per-axis stride; refactoring to a config struct would
// add allocation in a hot path.
#[allow(clippy::too_many_arguments)]
fn stride_insert_zeros_3d<T: Float>(
    input: &[T],
    batch: usize,
    channels: usize,
    d: usize,
    h: usize,
    w: usize,
    stride_d: usize,
    stride_h: usize,
    stride_w: usize,
) -> (Vec<T>, usize, usize, usize) {
    let d_up = (d - 1) * stride_d + 1;
    let h_up = (h - 1) * stride_h + 1;
    let w_up = (w - 1) * stride_w + 1;
    let zero = <T as num_traits::Zero>::zero();
    let mut out = vec![zero; batch * channels * d_up * h_up * w_up];

    for b in 0..batch {
        for c in 0..channels {
            for id in 0..d {
                for ih in 0..h {
                    for iw in 0..w {
                        let od = id * stride_d;
                        let oh = ih * stride_h;
                        let ow = iw * stride_w;
                        out[b * channels * d_up * h_up * w_up
                            + c * d_up * h_up * w_up
                            + od * h_up * w_up
                            + oh * w_up
                            + ow] = input
                            [b * channels * d * h * w + c * d * h * w + id * h * w + ih * w + iw];
                    }
                }
            }
        }
    }

    (out, d_up, h_up, w_up)
}

/// Flip a 3-D kernel along all spatial axes and transpose channel dimensions:
/// `kernel[c_in, c_out, kD, kH, kW]` ->
/// `kernel[c_out, c_in, kD-1-kd, kH-1-kh, kW-1-kw]`.
fn flip_kernel_3d<T: Float>(
    kernel: &[T],
    c_in: usize,
    c_out: usize,
    kd: usize,
    kh: usize,
    kw: usize,
) -> Vec<T> {
    let zero = <T as num_traits::Zero>::zero();
    let mut flipped = vec![zero; c_out * c_in * kd * kh * kw];

    for ci in 0..c_in {
        for co in 0..c_out {
            for dd in 0..kd {
                for dh in 0..kh {
                    for dw in 0..kw {
                        // Source: [c_in, c_out, dd, dh, dw]
                        let src = ci * c_out * kd * kh * kw
                            + co * kd * kh * kw
                            + dd * kh * kw
                            + dh * kw
                            + dw;
                        // Dest: [c_out, c_in, kD-1-dd, kH-1-dh, kW-1-dw]
                        let dst = co * c_in * kd * kh * kw
                            + ci * kd * kh * kw
                            + (kd - 1 - dd) * kh * kw
                            + (kh - 1 - dh) * kw
                            + (kw - 1 - dw);
                        flipped[dst] = kernel[src];
                    }
                }
            }
        }
    }

    flipped
}

impl<T: Float> Module<T> for ConvTranspose3d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Record autocast decision for conv_transpose3d.
        let _autocast_cat = autocast_guard("conv_transpose3d");

        // Validate input shape: [B, C_in, D, H, W].
        if input.ndim() != 5 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ConvTranspose3d expects 5-D input [B, C, D, H, W], got {:?}",
                    input.shape()
                ),
            });
        }

        let batch = input.shape()[0];
        let c_in = input.shape()[1];
        let d = input.shape()[2];
        let h = input.shape()[3];
        let w = input.shape()[4];

        if c_in != self.in_channels {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "ConvTranspose3d: expected {} input channels, got {}",
                    self.in_channels, c_in
                ),
            });
        }

        let (kd, kh, kw) = self.kernel_size;
        let (sd, sh, sw) = self.stride;
        let (pd, ph, pw) = self.padding;
        let (opd, oph, opw) = self.output_padding;

        // Save the input device so we can restore it on the output.
        let input_device = input.device();

        // Step 1: Insert zeros between input elements (stride insertion).
        let input_data = input.data_vec()?;
        let (upsampled, d_up_core, h_up_core, w_up_core) =
            stride_insert_zeros_3d(&input_data, batch, c_in, d, h, w, sd, sh, sw);

        // `output_padding` extends ONE side (trailing depth/rows/cols) of the
        // output. Those cells are NOT zero-padding: per upstream `col2im`
        // (`aten/src/ATen/native/im2col.h:104-146`) the transposed convolution
        // scatters into an output of size `(in-1)*stride - 2*pad +
        // dilation*(k-1) + output_padding + 1`
        // (`NaiveConvolutionTranspose3d.cpp` output-size formula), so the
        // trailing `output_padding` planes/rows/cols DO receive kernel-tap
        // contributions. Append `opd`/`oph`/`opw` zero planes/rows/cols to the
        // (stride-inserted) upsampled signal so the internal stride-1
        // convolution emits all output cells directly, rather than leaving the
        // boundary at 0 (the #1560 divergence). Closes #1560.
        let d_up = d_up_core + opd;
        let h_up = h_up_core + oph;
        let w_up = w_up_core + opw;
        let upsampled = if opd > 0 || oph > 0 || opw > 0 {
            let zero = <T as num_traits::Zero>::zero();
            let mut ext = vec![zero; batch * c_in * d_up * h_up * w_up];
            for b in 0..batch {
                for c in 0..c_in {
                    for id in 0..d_up_core {
                        for ih in 0..h_up_core {
                            let src =
                                (((b * c_in + c) * d_up_core + id) * h_up_core + ih) * w_up_core;
                            let dst = (((b * c_in + c) * d_up + id) * h_up + ih) * w_up;
                            ext[dst..dst + w_up_core]
                                .copy_from_slice(&upsampled[src..src + w_up_core]);
                        }
                    }
                }
            }
            ext
        } else {
            upsampled
        };

        // Step 2: Flip the kernel and transpose channel dimensions.
        let weight_data = self.weight.data_vec()?;
        let flipped = flip_kernel_3d(
            &weight_data,
            self.in_channels,
            self.out_channels,
            kd,
            kh,
            kw,
        );

        // Step 3: Apply a regular 3-D convolution on the upsampled input using the
        // flipped kernel. The "padding" for this internal convolution is
        // `kernel_size - 1 - padding` to achieve the correct output size.
        let internal_pad_d = kd - 1 - pd;
        let internal_pad_h = kh - 1 - ph;
        let internal_pad_w = kw - 1 - pw;

        // im2col_3d on the upsampled input with stride=1.
        let (cols, col_rows, col_cols) = im2col_3d(
            &upsampled,
            batch,
            c_in,
            d_up,
            h_up,
            w_up,
            kd,
            kh,
            kw,
            1,
            1,
            1,
            internal_pad_d,
            internal_pad_h,
            internal_pad_w,
        );

        // The internal stride-1 convolution over the `output_padding`-extended
        // upsampled signal now emits all output cells.
        let d_out = (d_up + 2 * internal_pad_d - kd) + 1;
        let h_out = (h_up + 2 * internal_pad_h - kh) + 1;
        let w_out = (w_up + 2 * internal_pad_w - kw) + 1;
        debug_assert_eq!(d_out, (d_up_core + 2 * internal_pad_d - kd) + 1 + opd);
        debug_assert_eq!(h_out, (h_up_core + 2 * internal_pad_h - kh) + 1 + oph);
        debug_assert_eq!(w_out, (w_up_core + 2 * internal_pad_w - kw) + 1 + opw);

        // Reshape flipped kernel to 2-D: [C_out, C_in * kD * kH * kW]
        let flipped_2d = Tensor::from_storage(
            TensorStorage::cpu(flipped),
            vec![self.out_channels, col_rows],
            false,
        )?;

        // Per-batch matmul.
        let zero = <T as num_traits::Zero>::zero();
        let spatial_out = d_out * h_out * w_out;
        let mut output = vec![zero; batch * self.out_channels * spatial_out];

        for b in 0..batch {
            let col_start = b * col_rows * col_cols;
            let col_end = col_start + col_rows * col_cols;
            let cols_b = Tensor::from_storage(
                TensorStorage::cpu(cols[col_start..col_end].to_vec()),
                vec![col_rows, col_cols],
                false,
            )?;

            let out_b = mm(&flipped_2d, &cols_b)?;
            let out_data = out_b.data()?;

            // Copy the full convolution result. The internal conv now emits all
            // output cells (including the `output_padding` boundary), so no cell
            // is left at 0 (the #1560 fix). `col_cols == spatial_out` here.
            let out_start = b * self.out_channels * spatial_out;
            for c in 0..self.out_channels {
                for od in 0..d_out {
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            output[out_start
                                + c * spatial_out
                                + od * h_out * w_out
                                + oh * w_out
                                + ow] =
                                out_data[c * spatial_out + od * h_out * w_out + oh * w_out + ow];
                        }
                    }
                }
            }
        }

        // Add bias if present.
        if let Some(ref bias) = self.bias {
            let bias_data = bias.data_vec()?;
            for b in 0..batch {
                for c in 0..self.out_channels {
                    let bval = bias_data[c];
                    for s in 0..spatial_out {
                        output[b * self.out_channels * spatial_out + c * spatial_out + s] += bval;
                    }
                }
            }
        }

        let result = Tensor::from_storage(
            TensorStorage::cpu(output),
            vec![batch, self.out_channels, d_out, h_out, w_out],
            false,
        )?;

        // Attach backward if gradients are enabled.
        if is_grad_enabled()
            && (input.requires_grad()
                || self.weight.requires_grad()
                || self.bias.as_ref().is_some_and(|b| b.requires_grad()))
        {
            let grad_fn = Arc::new(ConvTranspose3dBackward {
                input: input.clone(),
                weight: self.weight.tensor().clone(),
                bias: self.bias.as_ref().map(|b| b.tensor().clone()),
                in_channels: self.in_channels,
                out_channels: self.out_channels,
                kernel_size: self.kernel_size,
                stride: self.stride,
                padding: self.padding,
                _output_padding: self.output_padding,
                d_out,
                h_out,
                w_out,
            });
            Tensor::from_operation(
                TensorStorage::cpu(result.data()?.to_vec()),
                result.shape().to_vec(),
                grad_fn,
            )?
            .to(input_device) // restore device
        } else {
            result.to(input_device)
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = vec![("weight".to_string(), &self.weight)];
        if let Some(ref b) = self.bias {
            params.push(("bias".to_string(), b));
        }
        params
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// ConvTranspose3dBackward
// ---------------------------------------------------------------------------

/// Backward function for `ConvTranspose3d` forward pass.
///
/// The backward of a transposed 3-D convolution is a regular 3-D convolution.
#[derive(Debug)]
struct ConvTranspose3dBackward<T: Float> {
    input: Tensor<T>,
    weight: Tensor<T>,
    bias: Option<Tensor<T>>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    _output_padding: (usize, usize, usize),
    d_out: usize,
    h_out: usize,
    w_out: usize,
}

impl<T: Float> GradFn<T> for ConvTranspose3dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // grad_output shape: [B, C_out, D_out, H_out, W_out]
        let go_data = grad_output.data_vec()?;
        let batch = self.input.shape()[0];
        let d_in = self.input.shape()[2];
        let h_in = self.input.shape()[3];
        let w_in = self.input.shape()[4];
        let (kd, kh, kw) = self.kernel_size;
        let (sd, sh, sw) = self.stride;
        let (pd, ph, pw) = self.padding;
        let spatial_out = self.d_out * self.h_out * self.w_out;

        // --- grad_input ---
        // The backward of ConvTranspose3d w.r.t. input is a regular Conv3d
        // of grad_output with the original (non-flipped) weight.
        let grad_input = if self.input.requires_grad() {
            let weight_data = self.weight.data_vec()?;
            let col_rows = self.out_channels * kd * kh * kw;

            // Reshape weight to [C_in, C_out * kD * kH * kW]
            let weight_2d = Tensor::from_storage(
                TensorStorage::cpu(weight_data),
                vec![self.in_channels, col_rows],
                false,
            )?;

            // im2col_3d on grad_output with the conv parameters
            let (go_cols, _go_col_rows, go_col_cols) = im2col_3d(
                &go_data,
                batch,
                self.out_channels,
                self.d_out,
                self.h_out,
                self.w_out,
                kd,
                kh,
                kw,
                sd,
                sh,
                sw,
                pd,
                ph,
                pw,
            );

            let zero = <T as num_traits::Zero>::zero();
            let spatial_in = d_in * h_in * w_in;
            let mut gi = vec![zero; batch * self.in_channels * spatial_in];

            for b in 0..batch {
                let col_start = b * col_rows * go_col_cols;
                let col_end = col_start + col_rows * go_col_cols;
                let go_cols_b = Tensor::from_storage(
                    TensorStorage::cpu(go_cols[col_start..col_end].to_vec()),
                    vec![col_rows, go_col_cols],
                    false,
                )?;

                let gi_b = mm(&weight_2d, &go_cols_b)?;
                let gi_data = gi_b.data()?;

                let out_start = b * self.in_channels * spatial_in;
                let copy_len = self.in_channels * spatial_in;
                gi[out_start..out_start + copy_len].copy_from_slice(&gi_data[..copy_len]);
            }

            Some(Tensor::from_storage(
                TensorStorage::cpu(gi),
                self.input.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };

        // --- grad_weight ---
        // grad_weight[c_in, c_out, kd, kh, kw] = sum_b input_b (x) grad_output_b
        let grad_weight = if self.weight.requires_grad() {
            let zero = <T as num_traits::Zero>::zero();
            let weight_numel = self.in_channels * self.out_channels * kd * kh * kw;
            let mut gw = vec![zero; weight_numel];
            let input_data = self.input.data_vec()?;
            let spatial_in = d_in * h_in * w_in;

            for b in 0..batch {
                for ci in 0..self.in_channels {
                    for co in 0..self.out_channels {
                        for dd in 0..kd {
                            for dh in 0..kh {
                                for dw in 0..kw {
                                    let mut acc = zero;
                                    for id in 0..d_in {
                                        for ih in 0..h_in {
                                            for iw in 0..w_in {
                                                let od = id * sd + dd;
                                                let oh = ih * sh + dh;
                                                let ow = iw * sw + dw;
                                                if od >= pd
                                                    && oh >= ph
                                                    && ow >= pw
                                                    && (od - pd) < self.d_out
                                                    && (oh - ph) < self.h_out
                                                    && (ow - pw) < self.w_out
                                                {
                                                    let go_idx =
                                                        b * self.out_channels * spatial_out
                                                            + co * spatial_out
                                                            + (od - pd) * self.h_out * self.w_out
                                                            + (oh - ph) * self.w_out
                                                            + (ow - pw);
                                                    let in_idx = b * self.in_channels * spatial_in
                                                        + ci * spatial_in
                                                        + id * h_in * w_in
                                                        + ih * w_in
                                                        + iw;
                                                    acc += input_data[in_idx] * go_data[go_idx];
                                                }
                                            }
                                        }
                                    }
                                    gw[ci * self.out_channels * kd * kh * kw
                                        + co * kd * kh * kw
                                        + dd * kh * kw
                                        + dh * kw
                                        + dw] += acc;
                                }
                            }
                        }
                    }
                }
            }

            Some(Tensor::from_storage(
                TensorStorage::cpu(gw),
                vec![self.in_channels, self.out_channels, kd, kh, kw],
                false,
            )?)
        } else {
            None
        };

        // --- grad_bias ---
        let grad_bias = match &self.bias {
            Some(b) if b.requires_grad() => {
                let zero = <T as num_traits::Zero>::zero();
                let mut gb = vec![zero; self.out_channels];
                for batch_idx in 0..batch {
                    for c in 0..self.out_channels {
                        for s in 0..spatial_out {
                            gb[c] += go_data
                                [batch_idx * self.out_channels * spatial_out + c * spatial_out + s];
                        }
                    }
                }
                Some(Tensor::from_storage(
                    TensorStorage::cpu(gb),
                    vec![self.out_channels],
                    false,
                )?)
            }
            _ => None,
        };

        let mut grads = vec![grad_input, grad_weight];
        if self.bias.is_some() {
            grads.push(grad_bias);
        }
        Ok(grads)
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        let mut v = vec![&self.input, &self.weight];
        if let Some(ref b) = self.bias {
            v.push(b);
        }
        v
    }

    fn name(&self) -> &'static str {
        "ConvTranspose3dBackward"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::Module;

    // -----------------------------------------------------------------------
    // Bias init bounds — REQ-9 / closes #1450
    // -----------------------------------------------------------------------

    /// Verifies Conv2d bias is initialized within `U(-bound, bound)` where
    /// `bound = 1/sqrt((in_channels/groups) * kH * kW)` per
    /// `torch/nn/modules/conv.py:198-201`. Pre-fix the bias was zeros_init.
    #[test]
    fn test_conv2d_bias_init_bounded_uniform() {
        let in_c = 16usize;
        let out_c = 32usize;
        let kh = 3usize;
        let kw = 3usize;
        let groups = 1usize;
        let layer =
            Conv2d::<f32>::new_full(in_c, out_c, (kh, kw), (1, 1), (0, 0), (1, 1), groups, true)
                .unwrap();
        let bias = layer.bias.as_ref().expect("bias requested");
        let bias_data = bias.tensor().data_vec().unwrap();
        let fan_in = (in_c / groups) * kh * kw;
        let bound = 1.0_f32 / (fan_in as f32).sqrt();
        let mut nonzero = 0usize;
        for &b in &bias_data {
            assert!(
                b.abs() <= bound + 1e-6,
                "Conv2d bias element {b} exceeds bound {bound}"
            );
            if b != 0.0 {
                nonzero += 1;
            }
        }
        assert!(
            nonzero > out_c / 2,
            "expected most Conv2d bias entries to be nonzero; \
             would FAIL pre-fix when bias was zeros_init"
        );
    }

    /// Helper: create a tensor from flat data and shape.
    fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    /// Helper: create a leaf tensor that requires grad.
    fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
    }

    /// Assert two slices are element-wise close.
    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
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
                "index {i}: actual={a} expected={e} (diff {})",
                (a - e).abs()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Output shape
    // -----------------------------------------------------------------------

    #[test]
    fn test_output_shape_no_padding() {
        // Input: [1, 1, 5, 5], kernel 3x3, stride 1, padding 0
        // H_out = (5 - 3) / 1 + 1 = 3, W_out = 3
        let conv = Conv2d::<f32>::new(1, 1, (3, 3), (1, 1), (0, 0), false).unwrap();
        let input = t(&[0.0; 25], &[1, 1, 5, 5]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 3, 3]);
    }

    #[test]
    fn test_output_shape_with_padding() {
        // Input: [2, 3, 8, 8], kernel 3x3, stride 1, padding 1
        // H_out = (8 + 2 - 3) / 1 + 1 = 8
        let conv = Conv2d::<f32>::new(3, 16, (3, 3), (1, 1), (1, 1), true).unwrap();
        let input = t(&vec![0.0; 2 * 3 * 8 * 8], &[2, 3, 8, 8]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 16, 8, 8]);
    }

    #[test]
    fn test_output_shape_with_stride() {
        // Input: [1, 1, 6, 6], kernel 3x3, stride 2, padding 0
        // H_out = (6 - 3) / 2 + 1 = 2
        let conv = Conv2d::<f32>::new(1, 4, (3, 3), (2, 2), (0, 0), false).unwrap();
        let input = t(&[0.0; 36], &[1, 1, 6, 6]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 4, 2, 2]);
    }

    // -----------------------------------------------------------------------
    // 1x1 convolution == linear (per-pixel)
    // -----------------------------------------------------------------------

    #[test]
    fn test_1x1_conv_equals_linear() {
        // A 1x1 conv with 2 input channels and 3 output channels is equivalent
        // to a linear layer applied independently at each spatial position.
        //
        // weight shape: [3, 2, 1, 1] -- interpreted as a [3, 2] matrix
        // input shape: [1, 2, 2, 2]  -- 2 channels, 2x2 spatial
        //
        // For each pixel (h, w): output[:, h, w] = weight.squeeze() @ input[:, h, w]

        let weight_data: Vec<f32> = vec![
            1.0, 2.0, // out_channel 0: [1, 2]
            3.0, 4.0, // out_channel 1: [3, 4]
            5.0, 6.0, // out_channel 2: [5, 6]
        ];
        // Input: channel 0 = [[1, 2], [3, 4]], channel 1 = [[5, 6], [7, 8]]
        let input_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, // channel 0
            5.0, 6.0, 7.0, 8.0, // channel 1
        ];

        // Manually construct Conv2d with known weights.
        let weight_param = Parameter::from_slice(&weight_data, &[3, 2, 1, 1]).unwrap();
        let conv = Conv2d {
            weight: weight_param,
            bias: None,
            in_channels: 2,
            out_channels: 3,
            kernel_size: (1, 1),
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            groups: 1,
            padding_mode: crate::padding::PaddingMode::Zeros,
            training: false,
        };

        let input = t(&input_data, &[1, 2, 2, 2]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 3, 2, 2]);

        let out = output.data().unwrap();

        // Pixel (0,0): in = [1, 5], out = [1*1+2*5, 3*1+4*5, 5*1+6*5] = [11, 23, 35]
        // Pixel (0,1): in = [2, 6], out = [1*2+2*6, 3*2+4*6, 5*2+6*6] = [14, 30, 46]
        // Pixel (1,0): in = [3, 7], out = [1*3+2*7, 3*3+4*7, 5*3+6*7] = [17, 37, 57]
        // Pixel (1,1): in = [4, 8], out = [1*4+2*8, 3*4+4*8, 5*4+6*8] = [20, 44, 68]

        // Output layout: [C_out, H, W] = [3, 2, 2]
        // Channel 0: [11, 14, 17, 20]
        // Channel 1: [23, 30, 37, 44]
        // Channel 2: [35, 46, 57, 68]
        let expected = [
            11.0, 14.0, 17.0, 20.0, // out channel 0
            23.0, 30.0, 37.0, 44.0, // out channel 1
            35.0, 46.0, 57.0, 68.0, // out channel 2
        ];
        assert_close(out, &expected, 1e-5);
    }

    // -----------------------------------------------------------------------
    // Bias
    // -----------------------------------------------------------------------

    #[test]
    fn test_bias_addition() {
        // 1x1 conv with bias.
        let weight_data = vec![1.0f32]; // [1, 1, 1, 1]
        let bias_data = vec![10.0f32]; // [1]

        let conv = Conv2d {
            weight: Parameter::from_slice(&weight_data, &[1, 1, 1, 1]).unwrap(),
            bias: Some(Parameter::from_slice(&bias_data, &[1]).unwrap()),
            in_channels: 1,
            out_channels: 1,
            kernel_size: (1, 1),
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            groups: 1,
            padding_mode: crate::padding::PaddingMode::Zeros,
            training: false,
        };

        let input = t(&[2.0, 3.0, 4.0, 5.0], &[1, 1, 2, 2]);
        let output = conv.forward(&input).unwrap();
        // output = input * 1.0 + 10.0
        assert_close(output.data().unwrap(), &[12.0, 13.0, 14.0, 15.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // Backward shape
    // -----------------------------------------------------------------------

    #[test]
    fn test_backward_produces_correct_shapes() {
        // We manually invoke the backward function and check shapes.
        let weight_data = vec![1.0f32; 2 * 3 * 3]; // [2, 1, 3, 3]
        let input_data = vec![1.0f32; 5 * 5]; // [1, 1, 5, 5]
        let bias_data = vec![0.0f32; 2];

        let weight_param = Parameter::from_slice(&weight_data, &[2, 1, 3, 3]).unwrap();
        let bias_param = Parameter::from_slice(&bias_data, &[2]).unwrap();

        let conv = Conv2d {
            weight: weight_param,
            bias: Some(bias_param),
            in_channels: 1,
            out_channels: 2,
            kernel_size: (3, 3),
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            groups: 1,
            padding_mode: crate::padding::PaddingMode::Zeros,
            training: false,
        };

        // Forward to get the grad_fn.
        let input = leaf(&input_data, &[1, 1, 5, 5]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 2, 3, 3]);

        // Make sure grad_fn is attached.
        assert!(output.grad_fn().is_some());
        assert_eq!(output.grad_fn().unwrap().name(), "Conv2dBackward");

        // Construct a grad_output of the right shape.
        let grad_output = t(&[1.0; 2 * 3 * 3], &[1, 2, 3, 3]);
        let grads = output.grad_fn().unwrap().backward(&grad_output).unwrap();

        // grad_input shape should be [1, 1, 5, 5]
        assert!(grads[0].is_some());
        assert_eq!(grads[0].as_ref().unwrap().shape(), &[1, 1, 5, 5]);

        // grad_weight shape should be [2, 1, 3, 3]
        assert!(grads[1].is_some());
        assert_eq!(grads[1].as_ref().unwrap().shape(), &[2, 1, 3, 3]);

        // grad_bias shape should be [2]
        assert!(grads[2].is_some());
        assert_eq!(grads[2].as_ref().unwrap().shape(), &[2]);
    }

    // -----------------------------------------------------------------------
    // Parameter count
    // -----------------------------------------------------------------------

    #[test]
    fn test_parameter_count_with_bias() {
        let conv = Conv2d::<f32>::new(3, 16, (3, 3), (1, 1), (0, 0), true).unwrap();
        // weight: 16 * 3 * 3 * 3 = 432
        // bias:   16
        // total:  448
        assert_eq!(conv.num_parameters(), 448);
        assert_eq!(conv.parameters().len(), 2);
    }

    #[test]
    fn test_parameter_count_without_bias() {
        let conv = Conv2d::<f32>::new(3, 16, (3, 3), (1, 1), (0, 0), false).unwrap();
        assert_eq!(conv.num_parameters(), 432);
        assert_eq!(conv.parameters().len(), 1);
    }

    // -----------------------------------------------------------------------
    // Module trait
    // -----------------------------------------------------------------------

    #[test]
    fn test_named_parameters() {
        let conv = Conv2d::<f32>::new(1, 1, (3, 3), (1, 1), (0, 0), true).unwrap();
        let named = conv.named_parameters();
        assert_eq!(named.len(), 2);
        assert_eq!(named[0].0, "weight");
        assert_eq!(named[1].0, "bias");
    }

    #[test]
    fn test_train_eval() {
        let mut conv = Conv2d::<f32>::new(1, 1, (3, 3), (1, 1), (0, 0), false).unwrap();
        assert!(conv.is_training());
        conv.eval();
        assert!(!conv.is_training());
        conv.train();
        assert!(conv.is_training());
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_invalid_input_ndim() {
        let conv = Conv2d::<f32>::new(1, 1, (3, 3), (1, 1), (0, 0), false).unwrap();
        let input = t(&[1.0, 2.0, 3.0], &[3]);
        assert!(conv.forward(&input).is_err());
    }

    #[test]
    fn test_channel_mismatch() {
        let conv = Conv2d::<f32>::new(3, 1, (3, 3), (1, 1), (0, 0), false).unwrap();
        let input = t(&[0.0; 5 * 5], &[1, 1, 5, 5]);
        assert!(conv.forward(&input).is_err());
    }

    #[test]
    fn test_zero_channels_rejected() {
        assert!(Conv2d::<f32>::new(0, 16, (3, 3), (1, 1), (0, 0), false).is_err());
        assert!(Conv2d::<f32>::new(3, 0, (3, 3), (1, 1), (0, 0), false).is_err());
    }

    #[test]
    fn test_zero_kernel_rejected() {
        assert!(Conv2d::<f32>::new(1, 1, (0, 3), (1, 1), (0, 0), false).is_err());
    }

    #[test]
    fn test_zero_stride_rejected() {
        assert!(Conv2d::<f32>::new(1, 1, (3, 3), (0, 1), (0, 0), false).is_err());
    }

    // -----------------------------------------------------------------------
    // im2col / col2im roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_im2col_basic() {
        // 1 batch, 1 channel, 3x3 input, 2x2 kernel, stride 1, no padding
        // H_out = 2, W_out = 2
        // Columns: each column is a flattened 2x2 patch
        #[rustfmt::skip]
        let input: Vec<f32> = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let (cols, rows, n_cols) = im2col(&input, 1, 1, 3, 3, 2, 2, 1, 1, 0, 0);
        assert_eq!(rows, 4); // 1 * 2 * 2
        assert_eq!(n_cols, 4); // 2 * 2

        // Patch (0,0): [1, 2, 4, 5]
        // Patch (0,1): [2, 3, 5, 6]
        // Patch (1,0): [4, 5, 7, 8]
        // Patch (1,1): [5, 6, 8, 9]
        //
        // cols layout: [row][col] where row = c*kH*kW+kh*kW+kw, col = oh*W_out+ow
        // Row 0 (kh=0,kw=0): [1, 2, 4, 5]
        // Row 1 (kh=0,kw=1): [2, 3, 5, 6]
        // Row 2 (kh=1,kw=0): [4, 5, 7, 8]
        // Row 3 (kh=1,kw=1): [5, 6, 8, 9]
        assert_close(
            &cols,
            &[
                1.0, 2.0, 4.0, 5.0, // row 0
                2.0, 3.0, 5.0, 6.0, // row 1
                4.0, 5.0, 7.0, 8.0, // row 2
                5.0, 6.0, 8.0, 9.0, // row 3
            ],
            1e-7,
        );
    }

    #[test]
    fn test_col2im_roundtrip_no_overlap() {
        // With stride == kernel_size and no padding, im2col + col2im is lossless.
        // 1 batch, 1 channel, 4x4, kernel 2x2, stride 2, no padding
        // H_out = 2, W_out = 2
        #[rustfmt::skip]
        let input: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];

        let (cols, _rows, _n_cols) = im2col(&input, 1, 1, 4, 4, 2, 2, 2, 2, 0, 0);
        let recovered = col2im(&cols, 1, 1, 4, 4, 2, 2, 2, 2, 0, 0, 2, 2);

        assert_close(&recovered, &input, 1e-7);
    }

    // -----------------------------------------------------------------------
    // Forward correctness with a simple 3x3 kernel
    // -----------------------------------------------------------------------

    #[test]
    fn test_3x3_conv_forward() {
        // 1 batch, 1 channel, 3x3 input, 3x3 kernel, stride 1, no padding
        // Output: 1x1 (single value = sum of element-wise product)
        #[rustfmt::skip]
        let input_data: Vec<f32> = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        #[rustfmt::skip]
        let weight_data: Vec<f32> = vec![
            1.0, 0.0, -1.0,
            1.0, 0.0, -1.0,
            1.0, 0.0, -1.0,
        ];

        let conv = Conv2d {
            weight: Parameter::from_slice(&weight_data, &[1, 1, 3, 3]).unwrap(),
            bias: None,
            in_channels: 1,
            out_channels: 1,
            kernel_size: (3, 3),
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            groups: 1,
            padding_mode: crate::padding::PaddingMode::Zeros,
            training: false,
        };

        let input = t(&input_data, &[1, 1, 3, 3]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 1, 1]);

        // Expected: 1*1 + 0*2 + (-1)*3 + 1*4 + 0*5 + (-1)*6 + 1*7 + 0*8 + (-1)*9
        //         = 1 - 3 + 4 - 6 + 7 - 9 = -6
        assert_close(output.data().unwrap(), &[-6.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // Padding correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_padding_preserves_spatial_size() {
        // Input: [1, 1, 3, 3], kernel 3x3, stride 1, padding 1
        // H_out = (3 + 2 - 3) / 1 + 1 = 3 (same size!)
        let weight_data = vec![0.0f32; 9];
        let mut weight_data_center = weight_data;
        weight_data_center[4] = 1.0; // Center of 3x3 = identity

        let conv = Conv2d {
            weight: Parameter::from_slice(&weight_data_center, &[1, 1, 3, 3]).unwrap(),
            bias: None,
            in_channels: 1,
            out_channels: 1,
            kernel_size: (3, 3),
            stride: (1, 1),
            padding: (1, 1),
            dilation: (1, 1),
            groups: 1,
            padding_mode: crate::padding::PaddingMode::Zeros,
            training: false,
        };

        let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let input = t(&input_data, &[1, 1, 3, 3]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 3, 3]);

        // With center-only kernel + padding, output should equal input.
        assert_close(output.data().unwrap(), &input_data, 1e-5);
    }

    // ===================================================================
    // Conv1d tests
    // ===================================================================

    // -----------------------------------------------------------------------
    // Conv1d: output shape
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv1d_output_shape_no_padding() {
        // Input: [1, 1, 10], kernel 3, stride 1, padding 0
        // L_out = (10 - 3) / 1 + 1 = 8
        let conv = Conv1d::<f32>::new(1, 4, 3, 1, 0, false).unwrap();
        let input = t(&[0.0; 10], &[1, 1, 10]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_conv1d_output_shape_with_padding() {
        // Input: [2, 3, 16], kernel 3, stride 1, padding 1
        // L_out = (16 + 2 - 3) / 1 + 1 = 16
        let conv = Conv1d::<f32>::new(3, 8, 3, 1, 1, true).unwrap();
        let input = t(&vec![0.0; 2 * 3 * 16], &[2, 3, 16]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 8, 16]);
    }

    #[test]
    fn test_conv1d_output_shape_with_stride() {
        // Input: [1, 1, 10], kernel 3, stride 2, padding 0
        // L_out = (10 - 3) / 2 + 1 = 4
        let conv = Conv1d::<f32>::new(1, 2, 3, 2, 0, false).unwrap();
        let input = t(&[0.0; 10], &[1, 1, 10]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 2, 4]);
    }

    // -----------------------------------------------------------------------
    // Conv1d: 1x1 kernel correctness (acts as per-position linear)
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv1d_1x1_kernel_correctness() {
        // A kernel_size=1 Conv1d is equivalent to a linear layer applied at
        // each position independently.
        //
        // weight: [2, 1, 1] = [[3.0], [5.0]]
        // input:  [1, 1, 4] = [1, 2, 3, 4]
        // output: [1, 2, 4]
        //   out_ch 0: [3, 6, 9, 12]
        //   out_ch 1: [5, 10, 15, 20]
        let weight_data = vec![3.0f32, 5.0];
        let conv = Conv1d {
            weight: Parameter::from_slice(&weight_data, &[2, 1, 1]).unwrap(),
            bias: None,
            in_channels: 1,
            out_channels: 2,
            kernel_size: 1,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            padding_mode: crate::padding::PaddingMode::Zeros,
            training: false,
        };

        let input = t(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 2, 4]);
        assert_close(
            output.data().unwrap(),
            &[3.0, 6.0, 9.0, 12.0, 5.0, 10.0, 15.0, 20.0],
            1e-5,
        );
    }

    // -----------------------------------------------------------------------
    // Conv1d: forward correctness with a 3-wide kernel
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv1d_3_kernel_forward() {
        // Input: [1, 1, 5] = [1, 2, 3, 4, 5]
        // Kernel: [1, 1, 3] = [1, 0, -1]
        // Stride 1, padding 0 => L_out = 3
        // Expected: [1*1+0*2+(-1)*3, 1*2+0*3+(-1)*4, 1*3+0*4+(-1)*5] = [-2, -2, -2]
        let conv = Conv1d {
            weight: Parameter::from_slice(&[1.0f32, 0.0, -1.0], &[1, 1, 3]).unwrap(),
            bias: None,
            in_channels: 1,
            out_channels: 1,
            kernel_size: 3,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            padding_mode: crate::padding::PaddingMode::Zeros,
            training: false,
        };

        let input = t(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 3]);
        assert_close(output.data().unwrap(), &[-2.0, -2.0, -2.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // Conv1d: bias
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv1d_bias() {
        let conv = Conv1d {
            weight: Parameter::from_slice(&[1.0f32], &[1, 1, 1]).unwrap(),
            bias: Some(Parameter::from_slice(&[10.0f32], &[1]).unwrap()),
            in_channels: 1,
            out_channels: 1,
            kernel_size: 1,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            padding_mode: crate::padding::PaddingMode::Zeros,
            training: false,
        };

        let input = t(&[2.0, 3.0, 4.0], &[1, 1, 3]);
        let output = conv.forward(&input).unwrap();
        assert_close(output.data().unwrap(), &[12.0, 13.0, 14.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // Conv1d: edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv1d_invalid_ndim() {
        let conv = Conv1d::<f32>::new(1, 1, 3, 1, 0, false).unwrap();
        let input = t(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        assert!(conv.forward(&input).is_err());
    }

    #[test]
    fn test_conv1d_channel_mismatch() {
        let conv = Conv1d::<f32>::new(3, 1, 3, 1, 0, false).unwrap();
        let input = t(&[0.0; 10], &[1, 1, 10]);
        assert!(conv.forward(&input).is_err());
    }

    #[test]
    fn test_conv1d_zero_channels_rejected() {
        assert!(Conv1d::<f32>::new(0, 4, 3, 1, 0, false).is_err());
        assert!(Conv1d::<f32>::new(1, 0, 3, 1, 0, false).is_err());
    }

    #[test]
    fn test_conv1d_zero_kernel_rejected() {
        assert!(Conv1d::<f32>::new(1, 1, 0, 1, 0, false).is_err());
    }

    #[test]
    fn test_conv1d_zero_stride_rejected() {
        assert!(Conv1d::<f32>::new(1, 1, 3, 0, 0, false).is_err());
    }

    #[test]
    fn test_conv1d_parameter_count() {
        let conv = Conv1d::<f32>::new(3, 8, 5, 1, 0, true).unwrap();
        // weight: 8 * 3 * 5 = 120, bias: 8, total: 128
        assert_eq!(conv.num_parameters(), 128);
        assert_eq!(conv.parameters().len(), 2);
    }

    // ===================================================================
    // ConvTranspose2d tests
    // ===================================================================

    // -----------------------------------------------------------------------
    // ConvTranspose2d: output shape
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose2d_output_shape_basic() {
        // Input: [1, 1, 3, 3], kernel 3x3, stride 1, padding 0, output_padding 0
        // H_out = (3 - 1) * 1 - 0 + 3 + 0 = 5
        let conv =
            ConvTranspose2d::<f32>::new(1, 1, (3, 3), (1, 1), (0, 0), (0, 0), false).unwrap();
        let input = t(&[0.0; 9], &[1, 1, 3, 3]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 5, 5]);
    }

    #[test]
    fn test_conv_transpose2d_output_shape_stride2() {
        // Input: [1, 1, 2, 2], kernel 3x3, stride 2, padding 0, output_padding 0
        // H_out = (2 - 1) * 2 - 0 + 3 + 0 = 5
        let conv =
            ConvTranspose2d::<f32>::new(1, 1, (3, 3), (2, 2), (0, 0), (0, 0), false).unwrap();
        let input = t(&[0.0; 4], &[1, 1, 2, 2]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 5, 5]);
    }

    #[test]
    fn test_conv_transpose2d_output_shape_with_padding() {
        // Input: [1, 1, 3, 3], kernel 3x3, stride 2, padding 1, output_padding 0
        // H_out = (3 - 1) * 2 - 2 + 3 + 0 = 5
        let conv =
            ConvTranspose2d::<f32>::new(1, 1, (3, 3), (2, 2), (1, 1), (0, 0), false).unwrap();
        let input = t(&[0.0; 9], &[1, 1, 3, 3]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 5, 5]);
    }

    #[test]
    fn test_conv_transpose2d_output_shape_with_output_padding() {
        // Input: [1, 1, 3, 3], kernel 3x3, stride 2, padding 1, output_padding 1
        // H_out = (3 - 1) * 2 - 2 + 3 + 1 = 6
        let conv =
            ConvTranspose2d::<f32>::new(1, 1, (3, 3), (2, 2), (1, 1), (1, 1), false).unwrap();
        let input = t(&[0.0; 9], &[1, 1, 3, 3]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 6, 6]);
    }

    // -----------------------------------------------------------------------
    // ConvTranspose2d: stride=2 doubles spatial dims (upsampling)
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose2d_stride2_upsamples() {
        // With stride=2, kernel=2x2, padding=0, output_padding=0:
        // H_out = (H - 1) * 2 + 2 = 2 * H
        // So a 4x4 input becomes 8x8 — doubling spatial dims.
        let conv =
            ConvTranspose2d::<f32>::new(1, 1, (2, 2), (2, 2), (0, 0), (0, 0), false).unwrap();
        let input = t(&[0.0; 4 * 4], &[1, 1, 4, 4]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 8, 8]);
    }

    #[test]
    fn test_conv_transpose2d_stride2_upsamples_multichannel() {
        // [2, 8, 4, 4] -> [2, 16, 8, 8] with stride=2, kernel=2x2
        let conv =
            ConvTranspose2d::<f32>::new(8, 16, (2, 2), (2, 2), (0, 0), (0, 0), true).unwrap();
        let input = t(&vec![0.0; 2 * 8 * 4 * 4], &[2, 8, 4, 4]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 16, 8, 8]);
    }

    // -----------------------------------------------------------------------
    // ConvTranspose2d: 1x1 kernel correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose2d_1x1_kernel() {
        // With a 1x1 kernel, stride 1, no padding, the transposed conv is
        // equivalent to a regular 1x1 conv (just a per-pixel linear transform),
        // but with channels transposed:
        // weight shape: [in_channels=1, out_channels=2, 1, 1]
        // input: [1, 1, 2, 2]
        // Each output channel c gets: input * weight[0, c, 0, 0]
        let weight_data = vec![3.0f32, 7.0]; // [1, 2, 1, 1]
        let conv = ConvTranspose2d {
            weight: Parameter::from_slice(&weight_data, &[1, 2, 1, 1]).unwrap(),
            bias: None,
            in_channels: 1,
            out_channels: 2,
            kernel_size: (1, 1),
            stride: (1, 1),
            padding: (0, 0),
            output_padding: (0, 0),
            training: false,
        };

        let input = t(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 2, 2, 2]);

        // out_ch 0: input * 3 = [3, 6, 9, 12]
        // out_ch 1: input * 7 = [7, 14, 21, 28]
        assert_close(
            output.data().unwrap(),
            &[3.0, 6.0, 9.0, 12.0, 7.0, 14.0, 21.0, 28.0],
            1e-5,
        );
    }

    // -----------------------------------------------------------------------
    // ConvTranspose2d: correctness with stride insertion
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose2d_stride2_correctness() {
        // Input: [1, 1, 2, 2] = [[1, 2], [3, 4]]
        // Kernel: [1, 1, 2, 2] = [[1, 1], [1, 1]]  (all ones)
        // Stride=2, padding=0, output_padding=0
        // H_out = (2-1)*2 + 2 = 4, W_out = 4
        //
        // Stride insertion produces 3x3:
        //   [[1, 0, 2],
        //    [0, 0, 0],
        //    [3, 0, 4]]
        //
        // Flipped kernel (all ones, still all ones): [[1,1],[1,1]]
        // Internal conv with pad = kernel-1 = 1, stride=1 on 3x3:
        // Padded to 5x5:
        //   [[0, 0, 0, 0, 0],
        //    [0, 1, 0, 2, 0],
        //    [0, 0, 0, 0, 0],
        //    [0, 3, 0, 4, 0],
        //    [0, 0, 0, 0, 0]]
        // Convolve with 2x2 all-ones kernel, output 4x4:
        //   row 0: [1, 0+1, 2+0, 2] = [1, 1, 2, 2]
        //   row 1: [0+1, 1+0+0+0, 0+2+0+0, 0+2] = [1, 1, 2, 2]
        //   row 2: [3, 0+3, 4+0, 4] = [3, 3, 4, 4]
        //   row 3: [3, 3, 4, 4]
        //
        // Wait, let me recalculate more carefully.
        // After padding, we convolve (sum of 2x2 window at each position):
        // pos(0,0): 0+0+0+1 = 1
        // pos(0,1): 0+0+1+0 = 1
        // pos(0,2): 0+0+0+2 = 2
        // pos(0,3): 0+0+2+0 = 2
        // pos(1,0): 0+1+0+0 = 1
        // pos(1,1): 1+0+0+0 = 1
        // pos(1,2): 0+2+0+0 = 2
        // pos(1,3): 2+0+0+0 = 2
        // pos(2,0): 0+0+0+3 = 3
        // pos(2,1): 0+0+3+0 = 3
        // pos(2,2): 0+0+0+4 = 4
        // pos(2,3): 0+0+4+0 = 4
        // pos(3,0): 0+3+0+0 = 3
        // pos(3,1): 3+0+0+0 = 3
        // pos(3,2): 0+4+0+0 = 4
        // pos(3,3): 4+0+0+0 = 4

        let weight_data = vec![1.0f32; 4]; // [1, 1, 2, 2]
        let conv = ConvTranspose2d {
            weight: Parameter::from_slice(&weight_data, &[1, 1, 2, 2]).unwrap(),
            bias: None,
            in_channels: 1,
            out_channels: 1,
            kernel_size: (2, 2),
            stride: (2, 2),
            padding: (0, 0),
            output_padding: (0, 0),
            training: false,
        };

        let input = t(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 4, 4]);

        #[rustfmt::skip]
        let expected = [
            1.0, 1.0, 2.0, 2.0,
            1.0, 1.0, 2.0, 2.0,
            3.0, 3.0, 4.0, 4.0,
            3.0, 3.0, 4.0, 4.0,
        ];
        assert_close(output.data().unwrap(), &expected, 1e-5);
    }

    // -----------------------------------------------------------------------
    // ConvTranspose2d: bias
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose2d_bias() {
        let weight_data = vec![1.0f32]; // [1, 1, 1, 1] identity
        let bias_data = vec![5.0f32];
        let conv = ConvTranspose2d {
            weight: Parameter::from_slice(&weight_data, &[1, 1, 1, 1]).unwrap(),
            bias: Some(Parameter::from_slice(&bias_data, &[1]).unwrap()),
            in_channels: 1,
            out_channels: 1,
            kernel_size: (1, 1),
            stride: (1, 1),
            padding: (0, 0),
            output_padding: (0, 0),
            training: false,
        };

        let input = t(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let output = conv.forward(&input).unwrap();
        assert_close(output.data().unwrap(), &[6.0, 7.0, 8.0, 9.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // ConvTranspose2d: edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose2d_invalid_ndim() {
        let conv =
            ConvTranspose2d::<f32>::new(1, 1, (3, 3), (1, 1), (0, 0), (0, 0), false).unwrap();
        let input = t(&[1.0, 2.0, 3.0], &[1, 1, 3]);
        assert!(conv.forward(&input).is_err());
    }

    #[test]
    fn test_conv_transpose2d_channel_mismatch() {
        let conv =
            ConvTranspose2d::<f32>::new(3, 1, (3, 3), (1, 1), (0, 0), (0, 0), false).unwrap();
        let input = t(&[0.0; 5 * 5], &[1, 1, 5, 5]);
        assert!(conv.forward(&input).is_err());
    }

    #[test]
    fn test_conv_transpose2d_zero_channels_rejected() {
        assert!(ConvTranspose2d::<f32>::new(0, 1, (3, 3), (1, 1), (0, 0), (0, 0), false).is_err());
        assert!(ConvTranspose2d::<f32>::new(1, 0, (3, 3), (1, 1), (0, 0), (0, 0), false).is_err());
    }

    #[test]
    fn test_conv_transpose2d_output_padding_too_large() {
        // output_padding must be < stride
        assert!(ConvTranspose2d::<f32>::new(1, 1, (3, 3), (2, 2), (0, 0), (2, 2), false).is_err());
    }

    #[test]
    fn test_conv_transpose2d_parameter_count() {
        let conv =
            ConvTranspose2d::<f32>::new(8, 16, (3, 3), (2, 2), (1, 1), (0, 0), true).unwrap();
        // weight: 8 * 16 * 3 * 3 = 1152, bias: 16, total: 1168
        assert_eq!(conv.num_parameters(), 1168);
        assert_eq!(conv.parameters().len(), 2);
    }

    // ===================================================================
    // Conv3d tests
    // ===================================================================

    // -----------------------------------------------------------------------
    // Conv3d: output shape
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv3d_output_shape_no_padding() {
        // Input: [1, 1, 5, 5, 5], kernel 3x3x3, stride 1, padding 0
        // D_out = (5 - 3) / 1 + 1 = 3
        let conv = Conv3d::<f32>::new(1, 4, (3, 3, 3), (1, 1, 1), (0, 0, 0), false).unwrap();
        let input = t(&vec![0.0; 5 * 5 * 5], &[1, 1, 5, 5, 5]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 4, 3, 3, 3]);
    }

    #[test]
    fn test_conv3d_output_shape_with_padding() {
        // Input: [2, 3, 8, 8, 8], kernel 3x3x3, stride 1, padding 1
        // D_out = (8 + 2 - 3) / 1 + 1 = 8
        let conv = Conv3d::<f32>::new(3, 16, (3, 3, 3), (1, 1, 1), (1, 1, 1), true).unwrap();
        let input = t(&vec![0.0; 2 * 3 * 8 * 8 * 8], &[2, 3, 8, 8, 8]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 16, 8, 8, 8]);
    }

    #[test]
    fn test_conv3d_output_shape_with_stride() {
        // Input: [1, 1, 6, 6, 6], kernel 3x3x3, stride 2, padding 0
        // D_out = (6 - 3) / 2 + 1 = 2
        let conv = Conv3d::<f32>::new(1, 4, (3, 3, 3), (2, 2, 2), (0, 0, 0), false).unwrap();
        let input = t(&vec![0.0; 6 * 6 * 6], &[1, 1, 6, 6, 6]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 4, 2, 2, 2]);
    }

    // -----------------------------------------------------------------------
    // Conv3d: 1x1x1 kernel correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv3d_1x1x1_kernel_correctness() {
        // weight: [2, 1, 1, 1, 1] = [3.0, 5.0]
        // input:  [1, 1, 2, 1, 1] = [1.0, 2.0]
        // output: [1, 2, 2, 1, 1]
        //   out_ch 0: [3.0, 6.0]
        //   out_ch 1: [5.0, 10.0]
        let weight_data = vec![3.0f32, 5.0];
        let conv = Conv3d {
            weight: Parameter::from_slice(&weight_data, &[2, 1, 1, 1, 1]).unwrap(),
            bias: None,
            in_channels: 1,
            out_channels: 2,
            kernel_size: (1, 1, 1),
            stride: (1, 1, 1),
            padding: (0, 0, 0),
            dilation: (1, 1, 1),
            groups: 1,
            padding_mode: crate::padding::PaddingMode::Zeros,
            training: false,
        };

        let input = t(&[1.0, 2.0], &[1, 1, 2, 1, 1]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 2, 2, 1, 1]);
        assert_close(output.data().unwrap(), &[3.0, 6.0, 5.0, 10.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // Conv3d: forward correctness with a 3x3x3 kernel
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv3d_3x3x3_kernel_forward() {
        // Input: [1, 1, 3, 3, 3] (all ones), kernel: [1, 1, 3, 3, 3] (all ones)
        // Output: [1, 1, 1, 1, 1] = sum of 27 ones = 27.0
        let input_data = vec![1.0f32; 27];
        let weight_data = vec![1.0f32; 27];
        let conv = Conv3d {
            weight: Parameter::from_slice(&weight_data, &[1, 1, 3, 3, 3]).unwrap(),
            bias: None,
            in_channels: 1,
            out_channels: 1,
            kernel_size: (3, 3, 3),
            stride: (1, 1, 1),
            padding: (0, 0, 0),
            dilation: (1, 1, 1),
            groups: 1,
            padding_mode: crate::padding::PaddingMode::Zeros,
            training: false,
        };

        let input = t(&input_data, &[1, 1, 3, 3, 3]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 1, 1, 1]);
        assert_close(output.data().unwrap(), &[27.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // Conv3d: bias
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv3d_bias() {
        let conv = Conv3d {
            weight: Parameter::from_slice(&[1.0f32], &[1, 1, 1, 1, 1]).unwrap(),
            bias: Some(Parameter::from_slice(&[10.0f32], &[1]).unwrap()),
            in_channels: 1,
            out_channels: 1,
            kernel_size: (1, 1, 1),
            stride: (1, 1, 1),
            padding: (0, 0, 0),
            dilation: (1, 1, 1),
            groups: 1,
            padding_mode: crate::padding::PaddingMode::Zeros,
            training: false,
        };

        let input = t(&[2.0, 3.0], &[1, 1, 2, 1, 1]);
        let output = conv.forward(&input).unwrap();
        assert_close(output.data().unwrap(), &[12.0, 13.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // Conv3d: backward produces correct shapes
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv3d_backward_produces_correct_shapes() {
        let weight_data = vec![1.0f32; 2 * 3 * 3 * 3]; // [2, 1, 3, 3, 3]
        let input_data = vec![1.0f32; 5 * 5 * 5]; // [1, 1, 5, 5, 5]
        let bias_data = vec![0.0f32; 2];

        let conv = Conv3d {
            weight: Parameter::from_slice(&weight_data, &[2, 1, 3, 3, 3]).unwrap(),
            bias: Some(Parameter::from_slice(&bias_data, &[2]).unwrap()),
            in_channels: 1,
            out_channels: 2,
            kernel_size: (3, 3, 3),
            stride: (1, 1, 1),
            padding: (0, 0, 0),
            dilation: (1, 1, 1),
            groups: 1,
            padding_mode: crate::padding::PaddingMode::Zeros,
            training: false,
        };

        let input = leaf(&input_data, &[1, 1, 5, 5, 5]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 2, 3, 3, 3]);
        assert!(output.grad_fn().is_some());
        assert_eq!(output.grad_fn().unwrap().name(), "Conv3dBackward");

        let grad_output = t(&vec![1.0; 2 * 3 * 3 * 3], &[1, 2, 3, 3, 3]);
        let grads = output.grad_fn().unwrap().backward(&grad_output).unwrap();

        assert!(grads[0].is_some());
        assert_eq!(grads[0].as_ref().unwrap().shape(), &[1, 1, 5, 5, 5]);
        assert!(grads[1].is_some());
        assert_eq!(grads[1].as_ref().unwrap().shape(), &[2, 1, 3, 3, 3]);
        assert!(grads[2].is_some());
        assert_eq!(grads[2].as_ref().unwrap().shape(), &[2]);
    }

    // -----------------------------------------------------------------------
    // Conv3d: edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv3d_invalid_ndim() {
        let conv = Conv3d::<f32>::new(1, 1, (3, 3, 3), (1, 1, 1), (0, 0, 0), false).unwrap();
        let input = t(&[0.0; 25], &[1, 1, 5, 5]);
        assert!(conv.forward(&input).is_err());
    }

    #[test]
    fn test_conv3d_channel_mismatch() {
        let conv = Conv3d::<f32>::new(3, 1, (3, 3, 3), (1, 1, 1), (0, 0, 0), false).unwrap();
        let input = t(&vec![0.0; 5 * 5 * 5], &[1, 1, 5, 5, 5]);
        assert!(conv.forward(&input).is_err());
    }

    #[test]
    fn test_conv3d_zero_channels_rejected() {
        assert!(Conv3d::<f32>::new(0, 16, (3, 3, 3), (1, 1, 1), (0, 0, 0), false).is_err());
        assert!(Conv3d::<f32>::new(3, 0, (3, 3, 3), (1, 1, 1), (0, 0, 0), false).is_err());
    }

    #[test]
    fn test_conv3d_zero_kernel_rejected() {
        assert!(Conv3d::<f32>::new(1, 1, (0, 3, 3), (1, 1, 1), (0, 0, 0), false).is_err());
    }

    #[test]
    fn test_conv3d_zero_stride_rejected() {
        assert!(Conv3d::<f32>::new(1, 1, (3, 3, 3), (0, 1, 1), (0, 0, 0), false).is_err());
    }

    #[test]
    fn test_conv3d_parameter_count() {
        let conv = Conv3d::<f32>::new(3, 8, (3, 3, 3), (1, 1, 1), (0, 0, 0), true).unwrap();
        // weight: 8 * 3 * 3 * 3 * 3 = 648, bias: 8, total: 656
        assert_eq!(conv.num_parameters(), 656);
        assert_eq!(conv.parameters().len(), 2);
    }

    #[test]
    fn test_conv3d_named_parameters() {
        let conv = Conv3d::<f32>::new(1, 1, (3, 3, 3), (1, 1, 1), (0, 0, 0), true).unwrap();
        let named = conv.named_parameters();
        assert_eq!(named.len(), 2);
        assert_eq!(named[0].0, "weight");
        assert_eq!(named[1].0, "bias");
    }

    // ===================================================================
    // ConvTranspose1d tests
    // ===================================================================

    // -----------------------------------------------------------------------
    // ConvTranspose1d: output shape
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose1d_output_shape_basic() {
        // Input: [1, 1, 5], kernel 3, stride 1, padding 0, output_padding 0
        // L_out = (5 - 1) * 1 - 0 + 3 + 0 = 7
        let conv = ConvTranspose1d::<f32>::new(1, 1, 3, 1, 0, 0, false).unwrap();
        let input = t(&[0.0; 5], &[1, 1, 5]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 7]);
    }

    #[test]
    fn test_conv_transpose1d_output_shape_stride2() {
        // Input: [1, 1, 3], kernel 3, stride 2, padding 0, output_padding 0
        // L_out = (3 - 1) * 2 - 0 + 3 + 0 = 7
        let conv = ConvTranspose1d::<f32>::new(1, 1, 3, 2, 0, 0, false).unwrap();
        let input = t(&[0.0; 3], &[1, 1, 3]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 7]);
    }

    #[test]
    fn test_conv_transpose1d_output_shape_with_padding() {
        // Input: [1, 1, 5], kernel 3, stride 2, padding 1, output_padding 0
        // L_out = (5 - 1) * 2 - 2 + 3 + 0 = 9
        let conv = ConvTranspose1d::<f32>::new(1, 1, 3, 2, 1, 0, false).unwrap();
        let input = t(&[0.0; 5], &[1, 1, 5]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 9]);
    }

    #[test]
    fn test_conv_transpose1d_output_shape_with_output_padding() {
        // Input: [1, 1, 5], kernel 3, stride 2, padding 1, output_padding 1
        // L_out = (5 - 1) * 2 - 2 + 3 + 1 = 10
        let conv = ConvTranspose1d::<f32>::new(1, 1, 3, 2, 1, 1, false).unwrap();
        let input = t(&[0.0; 5], &[1, 1, 5]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 10]);
    }

    // -----------------------------------------------------------------------
    // ConvTranspose1d: 1x1 kernel correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose1d_1x1_kernel() {
        // With a kernel_size=1, stride 1, no padding, the transposed conv is
        // a per-position linear transform with channels transposed.
        // weight shape: [1, 2, 1] (in_channels=1, out_channels=2, k=1)
        let weight_data = vec![3.0f32, 7.0]; // [1, 2, 1]
        let conv = ConvTranspose1d {
            weight: Parameter::from_slice(&weight_data, &[1, 2, 1]).unwrap(),
            bias: None,
            in_channels: 1,
            out_channels: 2,
            kernel_size: 1,
            stride: 1,
            padding: 0,
            output_padding: 0,
            training: false,
        };

        let input = t(&[1.0, 2.0, 3.0], &[1, 1, 3]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 2, 3]);

        // out_ch 0: input * 3 = [3, 6, 9]
        // out_ch 1: input * 7 = [7, 14, 21]
        assert_close(
            output.data().unwrap(),
            &[3.0, 6.0, 9.0, 7.0, 14.0, 21.0],
            1e-5,
        );
    }

    // -----------------------------------------------------------------------
    // ConvTranspose1d: stride=2 correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose1d_stride2_correctness() {
        // Input: [1, 1, 2] = [1, 2]
        // Kernel: [1, 1, 2] = [1, 1] (all ones)
        // Stride=2, padding=0, output_padding=0
        // L_out = (2-1)*2 + 2 = 4
        //
        // Stride insertion produces [1, 0, 2]
        // Flipped kernel (all ones): [1, 1]
        // Internal conv with pad = 2-1 = 1, stride=1 on [1, 0, 2]:
        // Padded to [0, 1, 0, 2, 0]
        // Convolve with [1, 1] kernel, output 4:
        //   pos 0: 0+1 = 1
        //   pos 1: 1+0 = 1
        //   pos 2: 0+2 = 2
        //   pos 3: 2+0 = 2
        let weight_data = vec![1.0f32; 2]; // [1, 1, 2]
        let conv = ConvTranspose1d {
            weight: Parameter::from_slice(&weight_data, &[1, 1, 2]).unwrap(),
            bias: None,
            in_channels: 1,
            out_channels: 1,
            kernel_size: 2,
            stride: 2,
            padding: 0,
            output_padding: 0,
            training: false,
        };

        let input = t(&[1.0, 2.0], &[1, 1, 2]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 4]);
        assert_close(output.data().unwrap(), &[1.0, 1.0, 2.0, 2.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // ConvTranspose1d: bias
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose1d_bias() {
        let conv = ConvTranspose1d {
            weight: Parameter::from_slice(&[1.0f32], &[1, 1, 1]).unwrap(),
            bias: Some(Parameter::from_slice(&[5.0f32], &[1]).unwrap()),
            in_channels: 1,
            out_channels: 1,
            kernel_size: 1,
            stride: 1,
            padding: 0,
            output_padding: 0,
            training: false,
        };

        let input = t(&[1.0, 2.0, 3.0], &[1, 1, 3]);
        let output = conv.forward(&input).unwrap();
        assert_close(output.data().unwrap(), &[6.0, 7.0, 8.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // ConvTranspose1d: backward produces gradients
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose1d_backward_produces_gradients() {
        let weight_data = vec![1.0f32; 3]; // [1, 1, 3]
        let bias_data = vec![0.0f32; 1];

        let conv = ConvTranspose1d {
            weight: Parameter::from_slice(&weight_data, &[1, 1, 3]).unwrap(),
            bias: Some(Parameter::from_slice(&bias_data, &[1]).unwrap()),
            in_channels: 1,
            out_channels: 1,
            kernel_size: 3,
            stride: 1,
            padding: 0,
            output_padding: 0,
            training: false,
        };

        let input = leaf(&[1.0f32, 2.0, 3.0], &[1, 1, 3]);
        let output = conv.forward(&input).unwrap();
        // L_out = (3 - 1) * 1 - 0 + 3 + 0 = 5
        assert_eq!(output.shape(), &[1, 1, 5]);
        assert!(output.grad_fn().is_some());
        assert_eq!(output.grad_fn().unwrap().name(), "ConvTranspose1dBackward");

        let grad_output = t(&[1.0; 5], &[1, 1, 5]);
        let grads = output.grad_fn().unwrap().backward(&grad_output).unwrap();

        // grad_input shape: [1, 1, 3]
        assert!(grads[0].is_some());
        assert_eq!(grads[0].as_ref().unwrap().shape(), &[1, 1, 3]);
        // grad_weight shape: [1, 1, 3]
        assert!(grads[1].is_some());
        assert_eq!(grads[1].as_ref().unwrap().shape(), &[1, 1, 3]);
        // grad_bias shape: [1]
        assert!(grads[2].is_some());
        assert_eq!(grads[2].as_ref().unwrap().shape(), &[1]);
    }

    // -----------------------------------------------------------------------
    // ConvTranspose1d: edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose1d_invalid_ndim() {
        let conv = ConvTranspose1d::<f32>::new(1, 1, 3, 1, 0, 0, false).unwrap();
        let input = t(&[0.0; 4], &[1, 1, 2, 2]);
        assert!(conv.forward(&input).is_err());
    }

    #[test]
    fn test_conv_transpose1d_channel_mismatch() {
        let conv = ConvTranspose1d::<f32>::new(3, 1, 3, 1, 0, 0, false).unwrap();
        let input = t(&[0.0; 10], &[1, 1, 10]);
        assert!(conv.forward(&input).is_err());
    }

    #[test]
    fn test_conv_transpose1d_zero_channels_rejected() {
        assert!(ConvTranspose1d::<f32>::new(0, 1, 3, 1, 0, 0, false).is_err());
        assert!(ConvTranspose1d::<f32>::new(1, 0, 3, 1, 0, 0, false).is_err());
    }

    #[test]
    fn test_conv_transpose1d_output_padding_too_large() {
        assert!(ConvTranspose1d::<f32>::new(1, 1, 3, 2, 0, 2, false).is_err());
    }

    #[test]
    fn test_conv_transpose1d_parameter_count() {
        let conv = ConvTranspose1d::<f32>::new(8, 16, 5, 2, 1, 0, true).unwrap();
        // weight: 8 * 16 * 5 = 640, bias: 16, total: 656
        assert_eq!(conv.num_parameters(), 656);
        assert_eq!(conv.parameters().len(), 2);
    }

    // ===================================================================
    // ConvTranspose3d tests
    // ===================================================================

    // -----------------------------------------------------------------------
    // ConvTranspose3d: output shape
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose3d_output_shape_basic() {
        // Input: [1, 1, 3, 3, 3], kernel 3x3x3, stride 1, padding 0, output_padding 0
        // D_out = (3 - 1) * 1 - 0 + 3 + 0 = 5
        let conv =
            ConvTranspose3d::<f32>::new(1, 1, (3, 3, 3), (1, 1, 1), (0, 0, 0), (0, 0, 0), false)
                .unwrap();
        let input = t(&[0.0; 27], &[1, 1, 3, 3, 3]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 5, 5, 5]);
    }

    #[test]
    fn test_conv_transpose3d_output_shape_stride2() {
        // Input: [1, 1, 2, 2, 2], kernel 3x3x3, stride 2, padding 0, output_padding 0
        // D_out = (2 - 1) * 2 - 0 + 3 + 0 = 5
        let conv =
            ConvTranspose3d::<f32>::new(1, 1, (3, 3, 3), (2, 2, 2), (0, 0, 0), (0, 0, 0), false)
                .unwrap();
        let input = t(&[0.0; 8], &[1, 1, 2, 2, 2]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 5, 5, 5]);
    }

    #[test]
    fn test_conv_transpose3d_output_shape_with_padding() {
        // Input: [1, 1, 3, 3, 3], kernel 3x3x3, stride 2, padding 1, output_padding 0
        // D_out = (3 - 1) * 2 - 2 + 3 + 0 = 5
        let conv =
            ConvTranspose3d::<f32>::new(1, 1, (3, 3, 3), (2, 2, 2), (1, 1, 1), (0, 0, 0), false)
                .unwrap();
        let input = t(&[0.0; 27], &[1, 1, 3, 3, 3]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 5, 5, 5]);
    }

    #[test]
    fn test_conv_transpose3d_output_shape_with_output_padding() {
        // Input: [1, 1, 3, 3, 3], kernel 3x3x3, stride 2, padding 1, output_padding 1
        // D_out = (3 - 1) * 2 - 2 + 3 + 1 = 6
        let conv =
            ConvTranspose3d::<f32>::new(1, 1, (3, 3, 3), (2, 2, 2), (1, 1, 1), (1, 1, 1), false)
                .unwrap();
        let input = t(&[0.0; 27], &[1, 1, 3, 3, 3]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 6, 6, 6]);
    }

    // -----------------------------------------------------------------------
    // ConvTranspose3d: stride=2 upsamples (doubles spatial dims)
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose3d_stride2_upsamples() {
        // With stride=2, kernel=2x2x2, padding=0, output_padding=0:
        // D_out = (D - 1) * 2 + 2 = 2 * D
        let conv =
            ConvTranspose3d::<f32>::new(1, 1, (2, 2, 2), (2, 2, 2), (0, 0, 0), (0, 0, 0), false)
                .unwrap();
        let input = t(&vec![0.0; 4 * 4 * 4], &[1, 1, 4, 4, 4]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 8, 8, 8]);
    }

    // -----------------------------------------------------------------------
    // ConvTranspose3d: 1x1x1 kernel correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose3d_1x1x1_kernel() {
        // weight shape: [in=1, out=2, 1, 1, 1]
        let weight_data = vec![3.0f32, 7.0]; // [1, 2, 1, 1, 1]
        let conv = ConvTranspose3d {
            weight: Parameter::from_slice(&weight_data, &[1, 2, 1, 1, 1]).unwrap(),
            bias: None,
            in_channels: 1,
            out_channels: 2,
            kernel_size: (1, 1, 1),
            stride: (1, 1, 1),
            padding: (0, 0, 0),
            output_padding: (0, 0, 0),
            training: false,
        };

        let input = t(&[1.0, 2.0], &[1, 1, 2, 1, 1]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 2, 2, 1, 1]);
        assert_close(output.data().unwrap(), &[3.0, 6.0, 7.0, 14.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // ConvTranspose3d: bias
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose3d_bias() {
        let conv = ConvTranspose3d {
            weight: Parameter::from_slice(&[1.0f32], &[1, 1, 1, 1, 1]).unwrap(),
            bias: Some(Parameter::from_slice(&[5.0f32], &[1]).unwrap()),
            in_channels: 1,
            out_channels: 1,
            kernel_size: (1, 1, 1),
            stride: (1, 1, 1),
            padding: (0, 0, 0),
            output_padding: (0, 0, 0),
            training: false,
        };

        let input = t(&[1.0, 2.0], &[1, 1, 2, 1, 1]);
        let output = conv.forward(&input).unwrap();
        assert_close(output.data().unwrap(), &[6.0, 7.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // ConvTranspose3d: backward produces gradients
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose3d_backward_produces_gradients() {
        let weight_data = vec![1.0f32; 2 * 2 * 2]; // [1, 1, 2, 2, 2]
        let bias_data = vec![0.0f32; 1];

        let conv = ConvTranspose3d {
            weight: Parameter::from_slice(&weight_data, &[1, 1, 2, 2, 2]).unwrap(),
            bias: Some(Parameter::from_slice(&bias_data, &[1]).unwrap()),
            in_channels: 1,
            out_channels: 1,
            kernel_size: (2, 2, 2),
            stride: (1, 1, 1),
            padding: (0, 0, 0),
            output_padding: (0, 0, 0),
            training: false,
        };

        // D_out = (2-1)*1 - 0 + 2 + 0 = 3
        let input = leaf(&[1.0f32; 8], &[1, 1, 2, 2, 2]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 3, 3, 3]);
        assert!(output.grad_fn().is_some());
        assert_eq!(output.grad_fn().unwrap().name(), "ConvTranspose3dBackward");

        let grad_output = t(&[1.0; 27], &[1, 1, 3, 3, 3]);
        let grads = output.grad_fn().unwrap().backward(&grad_output).unwrap();

        assert!(grads[0].is_some());
        assert_eq!(grads[0].as_ref().unwrap().shape(), &[1, 1, 2, 2, 2]);
        assert!(grads[1].is_some());
        assert_eq!(grads[1].as_ref().unwrap().shape(), &[1, 1, 2, 2, 2]);
        assert!(grads[2].is_some());
        assert_eq!(grads[2].as_ref().unwrap().shape(), &[1]);
    }

    // -----------------------------------------------------------------------
    // ConvTranspose3d: edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose3d_invalid_ndim() {
        let conv =
            ConvTranspose3d::<f32>::new(1, 1, (3, 3, 3), (1, 1, 1), (0, 0, 0), (0, 0, 0), false)
                .unwrap();
        let input = t(&[0.0; 25], &[1, 1, 5, 5]);
        assert!(conv.forward(&input).is_err());
    }

    #[test]
    fn test_conv_transpose3d_channel_mismatch() {
        let conv =
            ConvTranspose3d::<f32>::new(3, 1, (3, 3, 3), (1, 1, 1), (0, 0, 0), (0, 0, 0), false)
                .unwrap();
        let input = t(&vec![0.0; 5 * 5 * 5], &[1, 1, 5, 5, 5]);
        assert!(conv.forward(&input).is_err());
    }

    #[test]
    fn test_conv_transpose3d_zero_channels_rejected() {
        assert!(
            ConvTranspose3d::<f32>::new(0, 1, (3, 3, 3), (1, 1, 1), (0, 0, 0), (0, 0, 0), false)
                .is_err()
        );
        assert!(
            ConvTranspose3d::<f32>::new(1, 0, (3, 3, 3), (1, 1, 1), (0, 0, 0), (0, 0, 0), false)
                .is_err()
        );
    }

    #[test]
    fn test_conv_transpose3d_output_padding_too_large() {
        assert!(
            ConvTranspose3d::<f32>::new(1, 1, (3, 3, 3), (2, 2, 2), (0, 0, 0), (2, 2, 2), false)
                .is_err()
        );
    }

    #[test]
    fn test_conv_transpose3d_parameter_count() {
        let conv =
            ConvTranspose3d::<f32>::new(8, 16, (3, 3, 3), (2, 2, 2), (1, 1, 1), (0, 0, 0), true)
                .unwrap();
        // weight: 8 * 16 * 3 * 3 * 3 = 3456, bias: 16, total: 3472
        assert_eq!(conv.num_parameters(), 3472);
        assert_eq!(conv.parameters().len(), 2);
    }

    // -----------------------------------------------------------------------
    // padding_mode threading — closes #1443
    //
    // Conv1d / Conv3d honor reflect/replicate/circular padding_mode for both
    // forward AND backward; ConvTranspose{1,2,3}d reject non-zeros modes
    // (matching `_ConvTransposeNd.__init__` ValueError, conv.py:755-758).
    //
    // All expected values are derived from a live PyTorch 2.11 oracle
    // (R-CHAR-3): the exact `torch.nn.Conv{1,3}d(..., padding_mode=...)` forward
    // outputs and `x.grad` after `out.sum().backward()`, with the same
    // deterministic weights/inputs reproduced below. The oracle script is in
    // the #1443 commit body. No tautological self-reference.
    // -----------------------------------------------------------------------

    /// Build a Conv1d with explicit weight/bias for deterministic oracle parity.
    fn conv1d_fixed(
        weight: &[f32],
        wshape: &[usize],
        bias: &[f32],
        kernel: usize,
        padding: usize,
        mode: crate::padding::PaddingMode,
    ) -> Conv1d<f32> {
        let w = Parameter::from_slice(weight, wshape).unwrap();
        let b = Parameter::from_slice(bias, &[wshape[0]]).unwrap();
        Conv1d {
            weight: w,
            bias: Some(b),
            in_channels: wshape[1],
            out_channels: wshape[0],
            kernel_size: kernel,
            stride: 1,
            padding,
            dilation: 1,
            groups: 1,
            padding_mode: mode,
            training: false,
        }
    }

    /// Conv1d reflect: forward output matches torch oracle.
    /// torch: Conv1d(1,1,3,padding=1,padding_mode='reflect'), w=[1,2,3], b=0.5,
    /// x=[1,2,3,4,5] -> out=[10.5, 14.5, 20.5, 26.5, 26.5].
    #[test]
    fn test_conv1d_reflect_forward_matches_torch() {
        let conv = conv1d_fixed(
            &[1.0, 2.0, 3.0],
            &[1, 1, 3],
            &[0.5],
            3,
            1,
            crate::padding::PaddingMode::Reflect,
        );
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5]);
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 1, 5]);
        assert_close(y.data().unwrap(), &[10.5, 14.5, 20.5, 26.5, 26.5], 1e-4);
    }

    /// Conv1d replicate: forward output matches torch oracle.
    /// torch out=[9.5, 14.5, 20.5, 26.5, 29.5].
    #[test]
    fn test_conv1d_replicate_forward_matches_torch() {
        let conv = conv1d_fixed(
            &[1.0, 2.0, 3.0],
            &[1, 1, 3],
            &[0.5],
            3,
            1,
            crate::padding::PaddingMode::Replicate,
        );
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5]);
        let y = conv.forward(&x).unwrap();
        assert_close(y.data().unwrap(), &[9.5, 14.5, 20.5, 26.5, 29.5], 1e-4);
    }

    /// Conv1d circular: forward output matches torch oracle.
    /// torch out=[13.5, 14.5, 20.5, 26.5, 17.5].
    #[test]
    fn test_conv1d_circular_forward_matches_torch() {
        let conv = conv1d_fixed(
            &[1.0, 2.0, 3.0],
            &[1, 1, 3],
            &[0.5],
            3,
            1,
            crate::padding::PaddingMode::Circular,
        );
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5]);
        let y = conv.forward(&x).unwrap();
        assert_close(y.data().unwrap(), &[13.5, 14.5, 20.5, 26.5, 17.5], 1e-4);
    }

    /// Conv1d reflect backward: input gradient flows through the non-zero pad
    /// (the #1550 regression class — a pad returning requires_grad=false would
    /// sever autograd and produce a None / zero grad here). torch grad_input
    /// for out.sum().backward() = [3, 7, 6, 9, 5].
    #[test]
    fn test_conv1d_reflect_backward_input_grad_matches_torch() {
        let conv = conv1d_fixed(
            &[1.0, 2.0, 3.0],
            &[1, 1, 3],
            &[0.5],
            3,
            1,
            crate::padding::PaddingMode::Reflect,
        );
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5]);
        let y = conv.forward(&x).unwrap();
        // grad_fn must be present — the autograd graph survives the pre-pad.
        assert!(
            y.grad_fn().is_some(),
            "Conv1d reflect output lost its grad_fn — pre-pad severed autograd (#1550 class)"
        );
        // `out.sum().backward()` — matches the torch oracle (grad_output = ones).
        let sum = ferrotorch_core::grad_fns::reduction::sum(&y).unwrap();
        ferrotorch_core::backward(&sum).unwrap();
        let xg = x
            .grad()
            .unwrap()
            .expect("input grad must be populated — pre-pad must be autograd-aware");
        assert_close(xg.data().unwrap(), &[3.0, 7.0, 6.0, 9.0, 5.0], 1e-4);
    }

    /// Conv1d circular backward input grad matches torch: [6, 6, 6, 6, 6].
    #[test]
    fn test_conv1d_circular_backward_input_grad_matches_torch() {
        let conv = conv1d_fixed(
            &[1.0, 2.0, 3.0],
            &[1, 1, 3],
            &[0.5],
            3,
            1,
            crate::padding::PaddingMode::Circular,
        );
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5]);
        let y = conv.forward(&x).unwrap();
        assert!(y.grad_fn().is_some());
        let sum = ferrotorch_core::grad_fns::reduction::sum(&y).unwrap();
        ferrotorch_core::backward(&sum).unwrap();
        let xg = x.grad().unwrap().expect("input grad must be populated");
        assert_close(xg.data().unwrap(), &[6.0, 6.0, 6.0, 6.0, 6.0], 1e-4);
    }

    // -----------------------------------------------------------------------
    // Conv1d groups + dilation (closes #1600) — oracle: live torch 2.11.0
    // -----------------------------------------------------------------------

    /// Build a grouped/dilated Conv1d through the production `new_full`
    /// constructor, then overwrite the weight/bias with deterministic
    /// caller-supplied tensors via `set_weight` / `set_data`. The weight must
    /// be `[out, in/groups, k]` (the grouped-conv layout, `conv.py:171`).
    fn conv1d_full_fixed(
        in_c: usize,
        out_c: usize,
        k: usize,
        dilation: usize,
        groups: usize,
        weight: &[f32],
        bias: Option<&[f32]>,
    ) -> Conv1d<f32> {
        let mut conv =
            Conv1d::<f32>::new_full(in_c, out_c, k, 1, 0, dilation, groups, bias.is_some())
                .unwrap();
        // Overwrite the Kaiming-initialised weight with the deterministic
        // tensor (direct field write — tests live in the same module).
        conv.weight = Parameter::from_slice(weight, &[out_c, in_c / groups, k]).unwrap();
        if let Some(bvals) = bias {
            conv.bias = Some(Parameter::from_slice(bvals, &[out_c]).unwrap());
        }
        conv
    }

    /// Grouped Conv1d, groups=2. Forward + grad_x + grad_w + grad_b all match
    /// the live torch 2.11 oracle (`F.conv1d(x, w, b, groups=2)`,
    /// `out.sum().backward()`). in=4 out=4 k=2 groups=2.
    #[test]
    fn test_conv1d_groups2_forward_and_backward_matches_torch() {
        // weight [4, 2, 2] = arange(1..=16) * 0.1; bias [0.5,-0.5,0.25,-0.25].
        let weight: Vec<f32> = (1..=16).map(|i| i as f32 * 0.1).collect();
        let bias = [0.5f32, -0.5, 0.25, -0.25];
        let conv = conv1d_full_fixed(4, 4, 2, 1, 2, &weight, Some(&bias));

        // x [1, 4, 5] = arange(1..=20).
        let x_data: Vec<f32> = (1..=20).map(|i| i as f32).collect();
        let x = leaf(&x_data, &[1, 4, 5]);
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 4, 4]);
        // torch A_fwd:
        assert_close(
            y.data().unwrap(),
            &[
                5.6, 6.6, 7.6, 8.6, 11.0, 13.6, 16.2, 18.8, 60.15, 64.35, 68.55, 72.75, 82.05,
                87.85, 93.65, 99.45,
            ],
            1e-3,
        );

        // out.sum().backward() => grad_output = ones.
        let grad_output = t(&[1.0f32; 16], &[1, 4, 4]);
        let grads = conv
            .forward(&x)
            .unwrap()
            .grad_fn()
            .unwrap()
            .backward(&grad_output)
            .unwrap();
        // grad_input (torch A_gx):
        assert_close(
            grads[0].as_ref().unwrap().data().unwrap(),
            &[
                0.6, 1.4, 1.4, 1.4, 0.8, 1.0, 2.2, 2.2, 2.2, 1.2, 2.2, 4.6, 4.6, 4.6, 2.4, 2.6,
                5.4, 5.4, 5.4, 2.8,
            ],
            1e-4,
        );
        // grad_weight (torch A_gw) — shape [4, 2, 2]:
        assert_eq!(grads[1].as_ref().unwrap().shape(), &[4, 2, 2]);
        assert_close(
            grads[1].as_ref().unwrap().data().unwrap(),
            &[
                10.0, 14.0, 30.0, 34.0, 10.0, 14.0, 30.0, 34.0, 50.0, 54.0, 70.0, 74.0, 50.0, 54.0,
                70.0, 74.0,
            ],
            1e-4,
        );
        // grad_bias (torch A_gb):
        assert_close(
            grads[2].as_ref().unwrap().data().unwrap(),
            &[4.0, 4.0, 4.0, 4.0],
            1e-4,
        );
    }

    /// Depthwise Conv1d, groups=3 (groups == in_channels), no bias. Forward +
    /// grad_x + grad_w match the live torch 2.11 oracle. in=3 out=3 k=2.
    #[test]
    fn test_conv1d_groups3_depthwise_forward_and_backward_matches_torch() {
        // weight [3, 1, 2] = arange(1..=6) * 0.5.
        let weight: Vec<f32> = (1..=6).map(|i| i as f32 * 0.5).collect();
        let conv = conv1d_full_fixed(3, 3, 2, 1, 3, &weight, None);

        // x [1, 3, 6] = arange(1..=18).
        let x_data: Vec<f32> = (1..=18).map(|i| i as f32).collect();
        let x = leaf(&x_data, &[1, 3, 6]);
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 3, 5]);
        // torch B_fwd:
        assert_close(
            y.data().unwrap(),
            &[
                2.5, 4.0, 5.5, 7.0, 8.5, 26.5, 30.0, 33.5, 37.0, 40.5, 74.5, 80.0, 85.5, 91.0, 96.5,
            ],
            1e-3,
        );

        let grad_output = t(&[1.0f32; 15], &[1, 3, 5]);
        let grads = conv
            .forward(&x)
            .unwrap()
            .grad_fn()
            .unwrap()
            .backward(&grad_output)
            .unwrap();
        // grad_input (torch B_gx):
        assert_close(
            grads[0].as_ref().unwrap().data().unwrap(),
            &[
                0.5, 1.5, 1.5, 1.5, 1.5, 1.0, 1.5, 3.5, 3.5, 3.5, 3.5, 2.0, 2.5, 5.5, 5.5, 5.5,
                5.5, 3.0,
            ],
            1e-4,
        );
        // grad_weight (torch B_gw) — shape [3, 1, 2]:
        assert_eq!(grads[1].as_ref().unwrap().shape(), &[3, 1, 2]);
        assert_close(
            grads[1].as_ref().unwrap().data().unwrap(),
            &[15.0, 20.0, 45.0, 50.0, 75.0, 80.0],
            1e-4,
        );
    }

    /// Dilated Conv1d, dilation=2, groups=1. Forward + grad_x + grad_w +
    /// grad_b match the live torch 2.11 oracle. in=2 out=2 k=3 dilation=2 =>
    /// eff_k=5, L=7 -> L_out=3.
    #[test]
    fn test_conv1d_dilation2_forward_and_backward_matches_torch() {
        // weight [2, 2, 3] = arange(1..=12) * 0.1; bias [1.0, -1.0].
        let weight: Vec<f32> = (1..=12).map(|i| i as f32 * 0.1).collect();
        let bias = [1.0f32, -1.0];
        let conv = conv1d_full_fixed(2, 2, 3, 2, 1, &weight, Some(&bias));

        // x [1, 2, 7] = arange(1..=14).
        let x_data: Vec<f32> = (1..=14).map(|i| i as f32).collect();
        let x = leaf(&x_data, &[1, 2, 7]);
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 2, 3]);
        // torch C_fwd:
        assert_close(
            y.data().unwrap(),
            &[18.6, 20.7, 22.8, 40.0, 45.7, 51.4],
            1e-3,
        );

        let grad_output = t(&[1.0f32; 6], &[1, 2, 3]);
        let grads = conv
            .forward(&x)
            .unwrap()
            .grad_fn()
            .unwrap()
            .backward(&grad_output)
            .unwrap();
        // grad_input (torch C_gx):
        assert_close(
            grads[0].as_ref().unwrap().data().unwrap(),
            &[
                0.8, 0.8, 1.8, 1.0, 2.2, 1.2, 1.2, 1.4, 1.4, 3.0, 1.6, 3.4, 1.8, 1.8,
            ],
            1e-4,
        );
        // grad_weight (torch C_gw) — shape [2, 2, 3]:
        assert_eq!(grads[1].as_ref().unwrap().shape(), &[2, 2, 3]);
        assert_close(
            grads[1].as_ref().unwrap().data().unwrap(),
            &[
                6.0, 12.0, 18.0, 27.0, 33.0, 39.0, 6.0, 12.0, 18.0, 27.0, 33.0, 39.0,
            ],
            1e-4,
        );
        // grad_bias (torch C_gb):
        assert_close(
            grads[2].as_ref().unwrap().data().unwrap(),
            &[3.0, 3.0],
            1e-4,
        );
    }

    /// `Conv1d::new_full` rejects `groups` that does not divide channels,
    /// matching `torch.nn.Conv1d`'s `ValueError` (`conv.py:107-110`).
    #[test]
    fn test_conv1d_groups_must_divide_channels() {
        // in_channels=3 not divisible by groups=2.
        assert!(Conv1d::<f32>::new_full(3, 4, 2, 1, 0, 1, 2, true).is_err());
        // out_channels=5 not divisible by groups=2 (in divisible).
        assert!(Conv1d::<f32>::new_full(4, 5, 2, 1, 0, 1, 2, true).is_err());
        // zero groups rejected.
        assert!(Conv1d::<f32>::new_full(4, 4, 2, 1, 0, 1, 0, true).is_err());
        // zero dilation rejected.
        assert!(Conv1d::<f32>::new_full(4, 4, 2, 1, 0, 0, 2, true).is_err());
        // valid grouped config accepted.
        assert!(Conv1d::<f32>::new_full(4, 4, 2, 1, 0, 1, 2, true).is_ok());
    }

    /// Build a Conv3d with explicit weight/bias for deterministic oracle parity.
    fn conv3d_fixed(
        weight: &[f32],
        wshape: &[usize],
        bias: &[f32],
        kernel: (usize, usize, usize),
        padding: (usize, usize, usize),
        mode: crate::padding::PaddingMode,
    ) -> Conv3d<f32> {
        let w = Parameter::from_slice(weight, wshape).unwrap();
        let b = Parameter::from_slice(bias, &[wshape[0]]).unwrap();
        Conv3d {
            weight: w,
            bias: Some(b),
            in_channels: wshape[1],
            out_channels: wshape[0],
            kernel_size: kernel,
            stride: (1, 1, 1),
            padding,
            dilation: (1, 1, 1),
            groups: 1,
            padding_mode: mode,
            training: false,
        }
    }

    /// Conv3d replicate forward matches torch oracle.
    /// torch: Conv3d(1,1,(2,2,2),padding=(1,1,1),padding_mode='replicate'),
    /// w=arange(1..=8), b=0, x=arange(1..=8) -> 27-element [1,1,3,3,3] output.
    #[test]
    fn test_conv3d_replicate_forward_matches_torch() {
        let w: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        let x_data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        let conv = conv3d_fixed(
            &w,
            &[1, 1, 2, 2, 2],
            &[0.0],
            (2, 2, 2),
            (1, 1, 1),
            crate::padding::PaddingMode::Replicate,
        );
        let x = t(&x_data, &[1, 1, 2, 2, 2]);
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 1, 3, 3, 3]);
        let expected = [
            36.0, 56.0, 72.0, 80.0, 100.0, 116.0, 108.0, 128.0, 144.0, 140.0, 160.0, 176.0, 184.0,
            204.0, 220.0, 212.0, 232.0, 248.0, 180.0, 200.0, 216.0, 224.0, 244.0, 260.0, 252.0,
            272.0, 288.0,
        ];
        assert_close(y.data().unwrap(), &expected, 1e-3);
    }

    /// Conv3d reflect forward matches torch oracle (same fixture, reflect mode).
    #[test]
    fn test_conv3d_reflect_forward_matches_torch() {
        let w: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        let x_data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        let conv = conv3d_fixed(
            &w,
            &[1, 1, 2, 2, 2],
            &[0.0],
            (2, 2, 2),
            (1, 1, 1),
            crate::padding::PaddingMode::Reflect,
        );
        let x = t(&x_data, &[1, 1, 2, 2, 2]);
        let y = conv.forward(&x).unwrap();
        let expected = [
            120.0, 124.0, 120.0, 136.0, 140.0, 136.0, 120.0, 124.0, 120.0, 184.0, 188.0, 184.0,
            200.0, 204.0, 200.0, 184.0, 188.0, 184.0, 120.0, 124.0, 120.0, 136.0, 140.0, 136.0,
            120.0, 124.0, 120.0,
        ];
        assert_close(y.data().unwrap(), &expected, 1e-3);
    }

    /// Conv3d circular forward matches torch oracle (discriminating asymmetric
    /// fixture: single-tap kernel + non-symmetric input so circular != reflect).
    /// torch: w[0,0,0,0,0]=1 else 0, x=[[1,2],[3,4]],[[5,6],[7,8]].
    #[test]
    fn test_conv3d_circular_forward_matches_torch() {
        let mut w = vec![0.0f32; 8];
        w[0] = 1.0;
        let x_data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        let conv = conv3d_fixed(
            &w,
            &[1, 1, 2, 2, 2],
            &[0.0],
            (2, 2, 2),
            (1, 1, 1),
            crate::padding::PaddingMode::Circular,
        );
        let x = t(&x_data, &[1, 1, 2, 2, 2]);
        let y = conv.forward(&x).unwrap();
        let expected = [
            8.0, 7.0, 8.0, 6.0, 5.0, 6.0, 8.0, 7.0, 8.0, 4.0, 3.0, 4.0, 2.0, 1.0, 2.0, 4.0, 3.0,
            4.0, 8.0, 7.0, 8.0, 6.0, 5.0, 6.0, 8.0, 7.0, 8.0,
        ];
        assert_close(y.data().unwrap(), &expected, 1e-3);
    }

    /// Conv3d replicate backward: input gradient flows through the non-zero pad
    /// (the #1550 regression class). torch grad_input for out.sum().backward()
    /// = [90, 99, 108, 117, 126, 135, 144, 153].
    #[test]
    fn test_conv3d_replicate_backward_input_grad_matches_torch() {
        let w: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        let x_data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        let conv = conv3d_fixed(
            &w,
            &[1, 1, 2, 2, 2],
            &[0.0],
            (2, 2, 2),
            (1, 1, 1),
            crate::padding::PaddingMode::Replicate,
        );
        let x = leaf(&x_data, &[1, 1, 2, 2, 2]);
        let y = conv.forward(&x).unwrap();
        assert!(
            y.grad_fn().is_some(),
            "Conv3d replicate output lost its grad_fn — pre-pad severed autograd (#1550 class)"
        );
        let sum = ferrotorch_core::grad_fns::reduction::sum(&y).unwrap();
        ferrotorch_core::backward(&sum).unwrap();
        let xg = x.grad().unwrap().expect("input grad must be populated");
        assert_close(
            xg.data().unwrap(),
            &[90.0, 99.0, 108.0, 117.0, 126.0, 135.0, 144.0, 153.0],
            1e-3,
        );
    }

    // -----------------------------------------------------------------------
    // Conv3d groups + dilation (closes #1601) — oracle: live torch 2.11.0
    // -----------------------------------------------------------------------

    /// Grouped + dilated Conv3d, groups=2, dilation=2. Forward + grad_x +
    /// grad_w + grad_b match the live torch 2.11 oracle. in=2 out=2
    /// k=(2,2,2) groups=2 dilation=(2,2,2) over a 4x4x4 volume => eff_k=3,
    /// out spatial = 2x2x2.
    #[test]
    fn test_conv3d_groups2_dilation2_forward_and_backward_matches_torch() {
        // weight [2, 1, 2, 2, 2] = arange(1..=16) * 0.01; bias [0.1, -0.1].
        let weight: Vec<f32> = (1..=16).map(|i| i as f32 * 0.01).collect();
        let bias = [0.1f32, -0.1];
        let mut conv =
            Conv3d::<f32>::new_full(2, 2, (2, 2, 2), (1, 1, 1), (0, 0, 0), (2, 2, 2), 2, true)
                .unwrap();
        conv.weight = Parameter::from_slice(&weight, &[2, 1, 2, 2, 2]).unwrap();
        conv.bias = Some(Parameter::from_slice(&bias, &[2]).unwrap());

        // x [1, 2, 4, 4, 4] = arange(1..=128).
        let x_data: Vec<f32> = (1..=128).map(|i| i as f32).collect();
        let x = leaf(&x_data, &[1, 2, 4, 4, 4]);
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 2, 2, 2, 2]);
        // torch D_fwd:
        assert_close(
            y.data().unwrap(),
            &[
                10.94, 11.3, 12.38, 12.74, 16.7, 17.06, 18.14, 18.5, 88.82, 89.82, 92.82, 93.82,
                104.82, 105.82, 108.82, 109.82,
            ],
            1e-3,
        );

        let grad_output = t(&[1.0f32; 16], &[1, 2, 2, 2, 2]);
        let grads = conv
            .forward(&x)
            .unwrap()
            .grad_fn()
            .unwrap()
            .backward(&grad_output)
            .unwrap();
        // grad_input (torch D_gx) — full 128 elements:
        #[rustfmt::skip]
        let d_gx: [f32; 128] = [
            0.01, 0.01, 0.02, 0.02, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.03, 0.03,
            0.04, 0.04, 0.01, 0.01, 0.02, 0.02, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04,
            0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06, 0.06, 0.05, 0.05, 0.06, 0.06, 0.07, 0.07,
            0.08, 0.08, 0.07, 0.07, 0.08, 0.08, 0.05, 0.05, 0.06, 0.06, 0.05, 0.05, 0.06, 0.06,
            0.07, 0.07, 0.08, 0.08, 0.07, 0.07, 0.08, 0.08, 0.09, 0.09, 0.1, 0.1, 0.09, 0.09, 0.1,
            0.1, 0.11, 0.11, 0.12, 0.12, 0.11, 0.11, 0.12, 0.12, 0.09, 0.09, 0.1, 0.1, 0.09, 0.09,
            0.1, 0.1, 0.11, 0.11, 0.12, 0.12, 0.11, 0.11, 0.12, 0.12, 0.13, 0.13, 0.14, 0.14, 0.13,
            0.13, 0.14, 0.14, 0.15, 0.15, 0.16, 0.16, 0.15, 0.15, 0.16, 0.16, 0.13, 0.13, 0.14,
            0.14, 0.13, 0.13, 0.14, 0.14, 0.15, 0.15, 0.16, 0.16, 0.15, 0.15, 0.16, 0.16,
        ];
        assert_close(grads[0].as_ref().unwrap().data().unwrap(), &d_gx, 1e-4);
        // grad_weight (torch D_gw) — shape [2, 1, 2, 2, 2]:
        assert_eq!(grads[1].as_ref().unwrap().shape(), &[2, 1, 2, 2, 2]);
        assert_close(
            grads[1].as_ref().unwrap().data().unwrap(),
            &[
                92.0, 108.0, 156.0, 172.0, 348.0, 364.0, 412.0, 428.0, 604.0, 620.0, 668.0, 684.0,
                860.0, 876.0, 924.0, 940.0,
            ],
            1e-3,
        );
        // grad_bias (torch D_gb):
        assert_close(
            grads[2].as_ref().unwrap().data().unwrap(),
            &[8.0, 8.0],
            1e-4,
        );
    }

    /// Grouped Conv3d (groups=2, dilation=1) sanity: a 1x1x1 grouped conv is
    /// a per-group channel mix. Forward + grad_x + grad_w match the live
    /// torch 2.11 oracle. in=2 out=4 k=(1,1,1) groups=2.
    #[test]
    fn test_conv3d_groups2_forward_and_backward_matches_torch() {
        // weight [4, 1, 1, 1, 1] = [1, 2, 3, 4], no bias.
        let weight = [1.0f32, 2.0, 3.0, 4.0];
        let mut conv =
            Conv3d::<f32>::new_full(2, 4, (1, 1, 1), (1, 1, 1), (0, 0, 0), (1, 1, 1), 2, false)
                .unwrap();
        conv.weight = Parameter::from_slice(&weight, &[4, 1, 1, 1, 1]).unwrap();

        // x [1, 2, 2, 2, 2] = arange(1..=16).
        let x_data: Vec<f32> = (1..=16).map(|i| i as f32).collect();
        let x = leaf(&x_data, &[1, 2, 2, 2, 2]);
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 4, 2, 2, 2]);
        // torch E_fwd:
        assert_close(
            y.data().unwrap(),
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0,
                27.0, 30.0, 33.0, 36.0, 39.0, 42.0, 45.0, 48.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0,
                60.0, 64.0,
            ],
            1e-3,
        );

        let grad_output = t(&[1.0f32; 32], &[1, 4, 2, 2, 2]);
        let grads = conv
            .forward(&x)
            .unwrap()
            .grad_fn()
            .unwrap()
            .backward(&grad_output)
            .unwrap();
        // grad_input (torch E_gx):
        assert_close(
            grads[0].as_ref().unwrap().data().unwrap(),
            &[
                3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
            ],
            1e-4,
        );
        // grad_weight (torch E_gw) — shape [4, 1, 1, 1, 1]:
        assert_eq!(grads[1].as_ref().unwrap().shape(), &[4, 1, 1, 1, 1]);
        assert_close(
            grads[1].as_ref().unwrap().data().unwrap(),
            &[36.0, 36.0, 100.0, 100.0],
            1e-4,
        );
    }

    /// `Conv3d::new_full` rejects `groups` that does not divide channels,
    /// matching `torch.nn.Conv3d`'s `ValueError` (`conv.py:107-110`).
    #[test]
    fn test_conv3d_groups_must_divide_channels() {
        // in_channels=3 not divisible by groups=2.
        assert!(
            Conv3d::<f32>::new_full(3, 4, (2, 2, 2), (1, 1, 1), (0, 0, 0), (1, 1, 1), 2, true)
                .is_err()
        );
        // out_channels=5 not divisible by groups=2.
        assert!(
            Conv3d::<f32>::new_full(4, 5, (2, 2, 2), (1, 1, 1), (0, 0, 0), (1, 1, 1), 2, true)
                .is_err()
        );
        // zero groups rejected.
        assert!(
            Conv3d::<f32>::new_full(4, 4, (2, 2, 2), (1, 1, 1), (0, 0, 0), (1, 1, 1), 0, true)
                .is_err()
        );
        // zero dilation rejected.
        assert!(
            Conv3d::<f32>::new_full(4, 4, (2, 2, 2), (1, 1, 1), (0, 0, 0), (0, 1, 1), 2, true)
                .is_err()
        );
        // valid grouped+dilated config accepted.
        assert!(
            Conv3d::<f32>::new_full(4, 4, (2, 2, 2), (1, 1, 1), (0, 0, 0), (2, 2, 2), 2, true)
                .is_ok()
        );
    }

    /// Conv1d with padding=0 ignores padding_mode (no pre-pad), matching torch
    /// (the `self.padding != 0` short-circuit in the forward).
    #[test]
    fn test_conv1d_reflect_zero_padding_is_noop() {
        let conv = conv1d_fixed(
            &[1.0, 2.0, 3.0],
            &[1, 1, 3],
            &[0.0],
            3,
            0,
            crate::padding::PaddingMode::Reflect,
        );
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5]);
        let y = conv.forward(&x).unwrap();
        // padding=0 -> output length 3, plain conv: [1*1+2*2+3*3, ...]
        assert_eq!(y.shape(), &[1, 1, 3]);
        assert_close(y.data().unwrap(), &[14.0, 20.0, 26.0], 1e-4);
    }

    // --- ConvTranspose: non-zeros padding_mode rejected (conv.py:755-758) ---

    #[test]
    fn test_conv_transpose1d_reflect_padding_mode_rejected() {
        let conv = ConvTranspose1d::<f32>::new(2, 2, 3, 1, 0, 0, false).unwrap();
        let err = conv
            .with_padding_mode(crate::padding::PaddingMode::Reflect)
            .unwrap_err();
        // Message matches torch exactly:
        // 'Only "zeros" padding mode is supported for ConvTranspose1d'.
        let msg = format!("{err}");
        assert!(
            msg.contains("Only \"zeros\" padding mode is supported for ConvTranspose1d"),
            "got: {msg}"
        );
    }

    #[test]
    fn test_conv_transpose2d_replicate_padding_mode_rejected() {
        let conv =
            ConvTranspose2d::<f32>::new(2, 2, (3, 3), (1, 1), (0, 0), (0, 0), false).unwrap();
        let err = conv
            .with_padding_mode(crate::padding::PaddingMode::Replicate)
            .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("Only \"zeros\" padding mode is supported for ConvTranspose2d"),
            "got: {msg}"
        );
    }

    #[test]
    fn test_conv_transpose3d_circular_padding_mode_rejected() {
        let conv =
            ConvTranspose3d::<f32>::new(2, 2, (3, 3, 3), (1, 1, 1), (0, 0, 0), (0, 0, 0), false)
                .unwrap();
        let err = conv
            .with_padding_mode(crate::padding::PaddingMode::Circular)
            .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("Only \"zeros\" padding mode is supported for ConvTranspose3d"),
            "got: {msg}"
        );
    }

    /// ConvTranspose accepts the `Zeros` mode (the only valid one) unchanged.
    #[test]
    fn test_conv_transpose2d_zeros_padding_mode_accepted() {
        let conv =
            ConvTranspose2d::<f32>::new(2, 2, (3, 3), (1, 1), (0, 0), (0, 0), false).unwrap();
        assert!(
            conv.with_padding_mode(crate::padding::PaddingMode::Zeros)
                .is_ok()
        );
    }
}
