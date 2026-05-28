//! Padding layers: constant, reflection, replication, and zero padding in 1-D, 2-D, 3-D.
//!
//! [CL-314] Add Conv3d, ConvTranspose1d/3d, and padding modules
//!
//! Each module pads the **last N** dimensions of the input tensor, matching
//! PyTorch semantics exactly.  Padding tuples specify *(left, right)* for 1-D,
//! *(left, right, top, bottom)* for 2-D, and
//! *(left, right, top, bottom, front, back)* for 3-D.
//!
//! ## REQ status (per `.design/ferrotorch-nn/padding.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | impl: `pub enum PaddingMode` here with 4 variants `Zeros` / `Reflect` / `Replicate` / `Circular`; non-test consumer: `ferrotorch-nn/src/conv.rs` uses `PaddingMode` as the `Conv{1,2,3}d` `padding_mode` field — the non-`Zeros` forward branch routes through `functional_pad_{1,2,3}d` (wiring landed in #1443), and `ConvTranspose{1,2,3}d::with_padding_mode` matches on it to reject non-`Zeros`. |
//! | REQ-2 | SHIPPED | impl: the grow-only `functional_pad_1d` / `functional_pad_2d` / `functional_pad_3d` entry points here dispatch on `PaddingMode`; the `Zeros`/constant arm routes through the crop-capable `functional_pad_1d_signed` / `functional_pad_2d_signed` / `functional_pad_3d_signed` (`isize` pads) which support NEGATIVE (crop) pads + mixed signs for `mode="constant"` via `pad_nd_signed_constant` + `PadNdSignedBackward`, mirroring `constant_pad_nd` at upstream `aten/src/ATen/native/PadNd.cpp:29-108` (#1611). Non-test consumer: the `usize` `functional_pad_{1,2,3}d` consume the signed entrypoints in production (the `Zeros` arm); `ferrotorch-nn/src/conv.rs` calls `functional_pad_{1,2,3}d` for the conv pre-pad; `ferrotorch-nn/src/functional.rs` re-exposes these as `nn::functional::pad`. |
//! | REQ-3 | SHIPPED | impl: `pub struct ConstantPad{1,2,3}d<T: Float>` here, mirroring `torch/nn/modules/padding.py` constant-pad family; non-test consumer: `pub use` in `lib.rs` exposes them to external crates; the vision-model code uses `ConstantPad2d` via the `lib.rs` re-export for padding non-square inputs. |
//! | REQ-4 | SHIPPED | impl: `pub struct ZeroPad{1,2,3}d<T: Float>` here; non-test consumer: `pub use` in `lib.rs` exposes them. |
//! | REQ-5 | SHIPPED | impl: `pub struct ReflectionPad{1,2,3}d<T: Float>` here with reflect-overflow check inside `pad_*d_reflect`; non-test consumer: `pub use` in `lib.rs`; reflection padding is the standard for U-nets and image-translation models. |
//! | REQ-6 | SHIPPED | impl: `pub struct ReplicationPad{1,2,3}d<T: Float>` here; non-test consumer: `pub use` in `lib.rs`. |
//! | REQ-7 | SHIPPED | impl: `pub struct CircularPad{1,2,3}d<T: Float>` here; non-test consumer: `pub use` in `lib.rs`. |
//! | REQ-8 | SHIPPED | impl: `macro_rules! impl_padding_module` here generates the `Module<T>` impls for all 12 structs; non-test consumer: `ferrotorch_optim` walks `Module::parameters()` of containers that include padding layers (every padding layer returns the empty parameter list, which is the correct behavior). |
//! | REQ-9 | NOT-STARTED | blocker #1441 (umbrella) — parity-sweep runner arms absent for all 6 padding ops. The impl is end-to-end verified by 40+ lib tests; only the runner-arm wiring is missing. |

use std::sync::Arc;

use ferrotorch_core::autograd::no_grad::is_grad_enabled;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::{GradFn, Tensor};
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float};

use crate::module::Module;
use crate::parameter::Parameter;

// ---------------------------------------------------------------------------
// Padding mode enum (used by conv layers with padding_mode)
// ---------------------------------------------------------------------------

/// Padding mode for convolution layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingMode {
    /// Zero padding (default).
    Zeros,
    /// Reflect padding.
    Reflect,
    /// Replicate padding (edge padding).
    Replicate,
    /// Circular padding (wrap-around).
    Circular,
}

// ---------------------------------------------------------------------------
// Low-level pad helpers (operate on raw data)
// ---------------------------------------------------------------------------

/// Pad the last dimension of a contiguous tensor.
///
/// `shape` has at least 1 dimension. The padding values `(left, right)` are
/// added to dimension `ndim-1`.
fn pad_1d_constant<T: Float>(
    data: &[T],
    shape: &[usize],
    pad_left: usize,
    pad_right: usize,
    value: T,
) -> (Vec<T>, Vec<usize>) {
    let ndim = shape.len();
    let inner = shape[ndim - 1];
    let new_inner = inner + pad_left + pad_right;

    // Number of "rows" = product of all dimensions except the last.
    let rows: usize = shape[..ndim - 1].iter().product();
    let rows = if rows == 0 { 1 } else { rows };

    let mut out = vec![value; rows * new_inner];
    // Degenerate input (numel 0 — e.g. an empty data buffer paired with a
    // non-empty declared shape, or `inner == 0`): there is no source data to
    // copy in. Mirror upstream `aten/src/ATen/native/PadNd.cpp:94-106`, which
    // `fill_(value)`s the output then `copy_`s the source — a no-op for a
    // zero-element input — leaving the correctly-shaped, value-filled output.
    // The guard prevents the out-of-bounds slice on `data` (#1551).
    if !data.is_empty() {
        for r in 0..rows {
            let src_start = r * inner;
            let dst_start = r * new_inner + pad_left;
            out[dst_start..dst_start + inner].copy_from_slice(&data[src_start..src_start + inner]);
        }
    }

    let mut new_shape = shape.to_vec();
    new_shape[ndim - 1] = new_inner;
    (out, new_shape)
}

/// Pad the last 2 dimensions of a contiguous tensor with a constant value.
fn pad_2d_constant<T: Float>(
    data: &[T],
    shape: &[usize],
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
    value: T,
) -> (Vec<T>, Vec<usize>) {
    let ndim = shape.len();
    let h = shape[ndim - 2];
    let w = shape[ndim - 1];
    let new_h = h + pad_top + pad_bottom;
    let new_w = w + pad_left + pad_right;

    let outer: usize = shape[..ndim - 2].iter().product();
    let outer = if outer == 0 { 1 } else { outer };

    let mut out = vec![value; outer * new_h * new_w];
    // Degenerate input (numel 0): no source data to copy in. Same rationale as
    // `pad_1d_constant` — mirror upstream `PadNd.cpp:94-106` (#1551).
    if !data.is_empty() {
        for o in 0..outer {
            for row in 0..h {
                let src_off = o * h * w + row * w;
                let dst_off = o * new_h * new_w + (row + pad_top) * new_w + pad_left;
                out[dst_off..dst_off + w].copy_from_slice(&data[src_off..src_off + w]);
            }
        }
    }

    let mut new_shape = shape.to_vec();
    new_shape[ndim - 2] = new_h;
    new_shape[ndim - 1] = new_w;
    (out, new_shape)
}

/// Pad the last 3 dimensions of a contiguous tensor with a constant value.
// Internal kernel: signature mirrors PyTorch's `F.pad` 3-axis layout
// (left, right, top, bottom, front, back); a config struct adds nothing.
#[allow(clippy::too_many_arguments)]
fn pad_3d_constant<T: Float>(
    data: &[T],
    shape: &[usize],
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
    pad_front: usize,
    pad_back: usize,
    value: T,
) -> (Vec<T>, Vec<usize>) {
    let ndim = shape.len();
    let d = shape[ndim - 3];
    let h = shape[ndim - 2];
    let w = shape[ndim - 1];
    let new_d = d + pad_front + pad_back;
    let new_h = h + pad_top + pad_bottom;
    let new_w = w + pad_left + pad_right;

    let outer: usize = shape[..ndim - 3].iter().product();
    let outer = if outer == 0 { 1 } else { outer };

    let mut out = vec![value; outer * new_d * new_h * new_w];
    // Degenerate input (numel 0): no source data to copy in. Same rationale as
    // `pad_1d_constant` — mirror upstream `PadNd.cpp:94-106` (#1551).
    if !data.is_empty() {
        for o in 0..outer {
            for dep in 0..d {
                for row in 0..h {
                    let src_off = o * d * h * w + dep * h * w + row * w;
                    let dst_off = o * new_d * new_h * new_w
                        + (dep + pad_front) * new_h * new_w
                        + (row + pad_top) * new_w
                        + pad_left;
                    out[dst_off..dst_off + w].copy_from_slice(&data[src_off..src_off + w]);
                }
            }
        }
    }

    let mut new_shape = shape.to_vec();
    new_shape[ndim - 3] = new_d;
    new_shape[ndim - 2] = new_h;
    new_shape[ndim - 1] = new_w;
    (out, new_shape)
}

// ---------------------------------------------------------------------------
// Reflection padding helpers
// ---------------------------------------------------------------------------

/// Reflect-pad the last dimension.
fn pad_1d_reflect<T: Float>(
    data: &[T],
    shape: &[usize],
    pad_left: usize,
    pad_right: usize,
) -> FerrotorchResult<(Vec<T>, Vec<usize>)> {
    let ndim = shape.len();
    let inner = shape[ndim - 1];
    if pad_left >= inner || pad_right >= inner {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "Reflection padding ({pad_left}, {pad_right}) must be less than input size ({inner})"
            ),
        });
    }
    let new_inner = inner + pad_left + pad_right;
    let rows: usize = shape[..ndim - 1].iter().copied().product::<usize>().max(1);

    let zero = <T as num_traits::Zero>::zero();
    let mut out = vec![zero; rows * new_inner];
    for r in 0..rows {
        let src = &data[r * inner..(r + 1) * inner];
        let dst = &mut out[r * new_inner..(r + 1) * new_inner];
        // Left reflection
        for i in 0..pad_left {
            dst[pad_left - 1 - i] = src[i + 1];
        }
        // Copy original
        dst[pad_left..pad_left + inner].copy_from_slice(src);
        // Right reflection
        for i in 0..pad_right {
            dst[pad_left + inner + i] = src[inner - 2 - i];
        }
    }

    let mut new_shape = shape.to_vec();
    new_shape[ndim - 1] = new_inner;
    Ok((out, new_shape))
}

/// Reflect-pad the last 2 dimensions.
fn pad_2d_reflect<T: Float>(
    data: &[T],
    shape: &[usize],
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
) -> FerrotorchResult<(Vec<T>, Vec<usize>)> {
    let ndim = shape.len();
    let h = shape[ndim - 2];
    let w = shape[ndim - 1];
    if pad_left >= w || pad_right >= w || pad_top >= h || pad_bottom >= h {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "Reflection padding ({pad_left}, {pad_right}, {pad_top}, {pad_bottom}) must be less than input size ({h}, {w})"
            ),
        });
    }
    let new_h = h + pad_top + pad_bottom;
    let new_w = w + pad_left + pad_right;
    let outer: usize = shape[..ndim - 2].iter().copied().product::<usize>().max(1);

    let zero = <T as num_traits::Zero>::zero();
    let mut out = vec![zero; outer * new_h * new_w];

    for o in 0..outer {
        let src_base = o * h * w;
        let dst_base = o * new_h * new_w;

        for new_row in 0..new_h {
            // Map new_row to source row via reflection
            let src_row = if new_row < pad_top {
                pad_top - new_row
            } else if new_row >= pad_top + h {
                h - 2 - (new_row - pad_top - h)
            } else {
                new_row - pad_top
            };

            for new_col in 0..new_w {
                let src_col = if new_col < pad_left {
                    pad_left - new_col
                } else if new_col >= pad_left + w {
                    w - 2 - (new_col - pad_left - w)
                } else {
                    new_col - pad_left
                };

                out[dst_base + new_row * new_w + new_col] = data[src_base + src_row * w + src_col];
            }
        }
    }

    let mut new_shape = shape.to_vec();
    new_shape[ndim - 2] = new_h;
    new_shape[ndim - 1] = new_w;
    Ok((out, new_shape))
}

/// Reflect-pad the last 3 dimensions.
// Internal kernel: same 3-axis pad descriptor as `pad_3d_constant`.
#[allow(clippy::too_many_arguments)]
fn pad_3d_reflect<T: Float>(
    data: &[T],
    shape: &[usize],
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
    pad_front: usize,
    pad_back: usize,
) -> FerrotorchResult<(Vec<T>, Vec<usize>)> {
    let ndim = shape.len();
    let d = shape[ndim - 3];
    let h = shape[ndim - 2];
    let w = shape[ndim - 1];
    if pad_left >= w
        || pad_right >= w
        || pad_top >= h
        || pad_bottom >= h
        || pad_front >= d
        || pad_back >= d
    {
        return Err(FerrotorchError::InvalidArgument {
            message: "Reflection padding must be less than corresponding input dimension".into(),
        });
    }
    let new_d = d + pad_front + pad_back;
    let new_h = h + pad_top + pad_bottom;
    let new_w = w + pad_left + pad_right;
    let outer: usize = shape[..ndim - 3].iter().copied().product::<usize>().max(1);

    let zero = <T as num_traits::Zero>::zero();
    let mut out = vec![zero; outer * new_d * new_h * new_w];

    for o in 0..outer {
        let src_base = o * d * h * w;
        let dst_base = o * new_d * new_h * new_w;

        for nd in 0..new_d {
            let sd = if nd < pad_front {
                pad_front - nd
            } else if nd >= pad_front + d {
                d - 2 - (nd - pad_front - d)
            } else {
                nd - pad_front
            };
            for nh in 0..new_h {
                let sh = if nh < pad_top {
                    pad_top - nh
                } else if nh >= pad_top + h {
                    h - 2 - (nh - pad_top - h)
                } else {
                    nh - pad_top
                };
                for nw in 0..new_w {
                    let sw = if nw < pad_left {
                        pad_left - nw
                    } else if nw >= pad_left + w {
                        w - 2 - (nw - pad_left - w)
                    } else {
                        nw - pad_left
                    };
                    out[dst_base + nd * new_h * new_w + nh * new_w + nw] =
                        data[src_base + sd * h * w + sh * w + sw];
                }
            }
        }
    }

    let mut new_shape = shape.to_vec();
    new_shape[ndim - 3] = new_d;
    new_shape[ndim - 2] = new_h;
    new_shape[ndim - 1] = new_w;
    Ok((out, new_shape))
}

// ---------------------------------------------------------------------------
// Replication padding helpers
// ---------------------------------------------------------------------------

/// Replicate-pad the last dimension (clamp to edges).
fn pad_1d_replicate<T: Float>(
    data: &[T],
    shape: &[usize],
    pad_left: usize,
    pad_right: usize,
) -> (Vec<T>, Vec<usize>) {
    let ndim = shape.len();
    let inner = shape[ndim - 1];
    let new_inner = inner + pad_left + pad_right;
    let rows: usize = shape[..ndim - 1].iter().copied().product::<usize>().max(1);

    let zero = <T as num_traits::Zero>::zero();
    let mut out = vec![zero; rows * new_inner];
    for r in 0..rows {
        let src = &data[r * inner..(r + 1) * inner];
        let dst = &mut out[r * new_inner..(r + 1) * new_inner];
        for (i, d) in dst.iter_mut().enumerate() {
            let src_idx = if i < pad_left {
                0
            } else if i >= pad_left + inner {
                inner - 1
            } else {
                i - pad_left
            };
            *d = src[src_idx];
        }
    }

    let mut new_shape = shape.to_vec();
    new_shape[ndim - 1] = new_inner;
    (out, new_shape)
}

/// Replicate-pad the last 2 dimensions.
fn pad_2d_replicate<T: Float>(
    data: &[T],
    shape: &[usize],
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
) -> (Vec<T>, Vec<usize>) {
    let ndim = shape.len();
    let h = shape[ndim - 2];
    let w = shape[ndim - 1];
    let new_h = h + pad_top + pad_bottom;
    let new_w = w + pad_left + pad_right;
    let outer: usize = shape[..ndim - 2].iter().copied().product::<usize>().max(1);

    let zero = <T as num_traits::Zero>::zero();
    let mut out = vec![zero; outer * new_h * new_w];

    for o in 0..outer {
        let src_base = o * h * w;
        let dst_base = o * new_h * new_w;
        for nr in 0..new_h {
            let sr = nr.saturating_sub(pad_top).min(h - 1);
            for nc in 0..new_w {
                let sc = nc.saturating_sub(pad_left).min(w - 1);
                out[dst_base + nr * new_w + nc] = data[src_base + sr * w + sc];
            }
        }
    }

    let mut new_shape = shape.to_vec();
    new_shape[ndim - 2] = new_h;
    new_shape[ndim - 1] = new_w;
    (out, new_shape)
}

/// Replicate-pad the last 3 dimensions.
// Internal kernel: same 3-axis pad descriptor as `pad_3d_constant`.
#[allow(clippy::too_many_arguments)]
fn pad_3d_replicate<T: Float>(
    data: &[T],
    shape: &[usize],
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
    pad_front: usize,
    pad_back: usize,
) -> (Vec<T>, Vec<usize>) {
    let ndim = shape.len();
    let d = shape[ndim - 3];
    let h = shape[ndim - 2];
    let w = shape[ndim - 1];
    let new_d = d + pad_front + pad_back;
    let new_h = h + pad_top + pad_bottom;
    let new_w = w + pad_left + pad_right;
    let outer: usize = shape[..ndim - 3].iter().copied().product::<usize>().max(1);

    let zero = <T as num_traits::Zero>::zero();
    let mut out = vec![zero; outer * new_d * new_h * new_w];

    for o in 0..outer {
        let src_base = o * d * h * w;
        let dst_base = o * new_d * new_h * new_w;
        for nd in 0..new_d {
            let sd = nd.saturating_sub(pad_front).min(d - 1);
            for nh in 0..new_h {
                let sh = nh.saturating_sub(pad_top).min(h - 1);
                for nw in 0..new_w {
                    let sw = nw.saturating_sub(pad_left).min(w - 1);
                    out[dst_base + nd * new_h * new_w + nh * new_w + nw] =
                        data[src_base + sd * h * w + sh * w + sw];
                }
            }
        }
    }

    let mut new_shape = shape.to_vec();
    new_shape[ndim - 3] = new_d;
    new_shape[ndim - 2] = new_h;
    new_shape[ndim - 1] = new_w;
    (out, new_shape)
}

// ---------------------------------------------------------------------------
// Circular padding helpers
// ---------------------------------------------------------------------------

/// Reject an all-non-negative circular pad that wraps around more than once.
///
/// The positive-only `pad_*_circular` helpers gather via `rem_euclid`, which
/// silently wraps a pad strictly larger than the axis size MULTIPLE times
/// (e.g. `circular [0,3]` on size 2 -> `[1,2,1,2,1]`). Upstream
/// `_pad_circular_symint` rejects this at `aten/src/ATen/native/PadNd.cpp:142`:
/// `TORCH_CHECK(pad_l <= size && pad_r <= size, "Padding value causes wrapping
/// around more than once.")`. For a non-negative pad the net extent is always
/// `>= size > 0`, so `:142` is the only check that can fire — mirror it here so
/// the positive circular path matches torch's accept/reject (`pad <= size`).
fn check_circular_positive(axes: &[(usize, usize)]) -> FerrotorchResult<()> {
    for (idx, &(size, pad)) in axes.iter().enumerate() {
        if pad > size {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Circular padding {pad} on axis (size {size}, position {idx}) causes wrapping around more than once (pad must be <= size)"
                ),
            });
        }
    }
    Ok(())
}

/// Circular-pad the last dimension (wrap-around).
fn pad_1d_circular<T: Float>(
    data: &[T],
    shape: &[usize],
    pad_left: usize,
    pad_right: usize,
) -> (Vec<T>, Vec<usize>) {
    let ndim = shape.len();
    let inner = shape[ndim - 1];
    let new_inner = inner + pad_left + pad_right;
    let rows: usize = shape[..ndim - 1].iter().copied().product::<usize>().max(1);

    let zero = <T as num_traits::Zero>::zero();
    let mut out = vec![zero; rows * new_inner];
    for r in 0..rows {
        let src = &data[r * inner..(r + 1) * inner];
        let dst = &mut out[r * new_inner..(r + 1) * new_inner];
        for (i, d) in dst.iter_mut().enumerate() {
            // Map to source via modulo
            let src_idx = ((i as isize - pad_left as isize).rem_euclid(inner as isize)) as usize;
            *d = src[src_idx];
        }
    }

    let mut new_shape = shape.to_vec();
    new_shape[ndim - 1] = new_inner;
    (out, new_shape)
}

/// Circular-pad the last 2 dimensions.
fn pad_2d_circular<T: Float>(
    data: &[T],
    shape: &[usize],
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
) -> (Vec<T>, Vec<usize>) {
    let ndim = shape.len();
    let h = shape[ndim - 2];
    let w = shape[ndim - 1];
    let new_h = h + pad_top + pad_bottom;
    let new_w = w + pad_left + pad_right;
    let outer: usize = shape[..ndim - 2].iter().copied().product::<usize>().max(1);

    let zero = <T as num_traits::Zero>::zero();
    let mut out = vec![zero; outer * new_h * new_w];

    for o in 0..outer {
        let src_base = o * h * w;
        let dst_base = o * new_h * new_w;
        for nr in 0..new_h {
            let sr = ((nr as isize - pad_top as isize).rem_euclid(h as isize)) as usize;
            for nc in 0..new_w {
                let sc = ((nc as isize - pad_left as isize).rem_euclid(w as isize)) as usize;
                out[dst_base + nr * new_w + nc] = data[src_base + sr * w + sc];
            }
        }
    }

    let mut new_shape = shape.to_vec();
    new_shape[ndim - 2] = new_h;
    new_shape[ndim - 1] = new_w;
    (out, new_shape)
}

/// Circular-pad the last 3 dimensions.
// Internal kernel: same 3-axis pad descriptor as `pad_3d_constant`.
#[allow(clippy::too_many_arguments)]
fn pad_3d_circular<T: Float>(
    data: &[T],
    shape: &[usize],
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
    pad_front: usize,
    pad_back: usize,
) -> (Vec<T>, Vec<usize>) {
    let ndim = shape.len();
    let d = shape[ndim - 3];
    let h = shape[ndim - 2];
    let w = shape[ndim - 1];
    let new_d = d + pad_front + pad_back;
    let new_h = h + pad_top + pad_bottom;
    let new_w = w + pad_left + pad_right;
    let outer: usize = shape[..ndim - 3].iter().copied().product::<usize>().max(1);

    let zero = <T as num_traits::Zero>::zero();
    let mut out = vec![zero; outer * new_d * new_h * new_w];

    for o in 0..outer {
        let src_base = o * d * h * w;
        let dst_base = o * new_d * new_h * new_w;
        for nd in 0..new_d {
            let sd = ((nd as isize - pad_front as isize).rem_euclid(d as isize)) as usize;
            for nh in 0..new_h {
                let sh = ((nh as isize - pad_top as isize).rem_euclid(h as isize)) as usize;
                for nw in 0..new_w {
                    let sw = ((nw as isize - pad_left as isize).rem_euclid(w as isize)) as usize;
                    out[dst_base + nd * new_h * new_w + nh * new_w + nw] =
                        data[src_base + sd * h * w + sh * w + sw];
                }
            }
        }
    }

    let mut new_shape = shape.to_vec();
    new_shape[ndim - 3] = new_d;
    new_shape[ndim - 2] = new_h;
    new_shape[ndim - 1] = new_w;
    (out, new_shape)
}

// ===========================================================================
// Public functional API — apply arbitrary padding to a Tensor
// ===========================================================================

// ---------------------------------------------------------------------------
// Autograd for the 1-D functional pad path (used by Conv1d's non-zero
// padding_mode pre-pad). Same gather/scatter-add adjoint as the 2-D case;
// see the `Pad2dBackward` block below for the full derivation. A pad that
// returns `requires_grad = false` severs autograd — the #1550 bug class that
// the 2-D path already fixed; the 1-D path needs the same `Pad1dBackward`
// node so Conv1d's input gradient flows through the reflect/replicate/circular
// pre-pad. Mirrors upstream `torch/nn/modules/conv.py:367-371` routing
// non-zero modes through the differentiable `F.pad`.
// ---------------------------------------------------------------------------

/// For an output element at `new_idx` in a 1-D pad, return the linear index
/// into the (single) source row, or `None` if the element comes from the
/// constant fill (Zeros mode) and has no source.
fn src_index_1d(mode: PaddingMode, new_idx: usize, inner: usize, pad_left: usize) -> Option<usize> {
    let s: usize = match mode {
        PaddingMode::Zeros => {
            if new_idx < pad_left || new_idx >= pad_left + inner {
                return None;
            }
            new_idx - pad_left
        }
        PaddingMode::Reflect => {
            if new_idx < pad_left {
                pad_left - new_idx
            } else if new_idx >= pad_left + inner {
                inner - 2 - (new_idx - pad_left - inner)
            } else {
                new_idx - pad_left
            }
        }
        PaddingMode::Replicate => new_idx.saturating_sub(pad_left).min(inner - 1),
        PaddingMode::Circular => {
            ((new_idx as isize - pad_left as isize).rem_euclid(inner as isize)) as usize
        }
    };
    Some(s)
}

/// Backward node for the 1-D functional pad. Scatter-adds the output gradient
/// back onto the unpadded input row using the per-output source-index map.
#[derive(Debug)]
struct Pad1dBackward<T: Float> {
    input: Tensor<T>,
    input_shape: Vec<usize>,
    mode: PaddingMode,
    pad_left: usize,
}

impl<T: Float> GradFn<T> for Pad1dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        let ndim = self.input_shape.len();
        let inner = self.input_shape[ndim - 1];
        let rows: usize = self.input_shape[..ndim - 1]
            .iter()
            .copied()
            .product::<usize>()
            .max(1);

        let go_shape = grad_output.shape();
        let new_inner = go_shape[ndim - 1];

        // The backward runs on host: scatter-add is data-dependent over the
        // index map. `data_vec` materialises the (possibly GPU) grad to CPU.
        let go = grad_output.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let mut grad_in = vec![zero; rows * inner];

        for r in 0..rows {
            let go_base = r * new_inner;
            let gi_base = r * inner;
            for ni in 0..new_inner {
                if let Some(src) = src_index_1d(self.mode, ni, inner, self.pad_left) {
                    grad_in[gi_base + src] += go[go_base + ni];
                }
            }
        }

        let grad_input =
            Tensor::from_storage(TensorStorage::cpu(grad_in), self.input_shape.clone(), false)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "Pad1dBackward"
    }
}

/// Apply padding to the last dimension of a tensor using the given mode.
///
/// This is the functional version used internally by conv layers with
/// `padding_mode`.
///
/// When `input` requires grad (and grad tracking is enabled) the returned
/// tensor carries a [`Pad1dBackward`] node so gradients flow back to `input`,
/// matching the differentiable `F.pad` that `torch/nn/modules/conv.py`
/// `_conv_forward` routes non-zero `padding_mode`s through (Conv1d at
/// `conv.py:367-371`).
pub fn functional_pad_1d<T: Float>(
    input: &Tensor<T>,
    pad_left: usize,
    pad_right: usize,
    mode: PaddingMode,
    value: T,
) -> FerrotorchResult<Tensor<T>> {
    // `Zeros` is the runner's mapping for torch `mode="constant"`; route it
    // through the crop-capable signed constant path — the single source of
    // truth for constant padding, mirroring torch dispatching `mode="constant"`
    // through `constant_pad_nd` (`aten/src/ATen/native/PadNd.cpp:214-215`). For
    // a non-negative `usize` pad the signed forward is byte-identical to the old
    // `pad_1d_constant` and its `PadNdSignedBackward` scatter-add equals the old
    // `Pad1dBackward` adjoint; the `value` fill (#1553) is preserved.
    if mode == PaddingMode::Zeros {
        return functional_pad_1d_signed(input, pad_left as isize, pad_right as isize, mode, value);
    }

    let data = input.data_vec()?;
    let shape = input.shape();
    let input_shape = shape.to_vec();
    // The `Zeros` (constant) arm is dispatched above through the crop-capable
    // signed path; the remaining gather modes never crop and keep their
    // existing positive-only helpers + `Pad1dBackward` adjoint.
    let (out_data, new_shape) = match mode {
        PaddingMode::Reflect => pad_1d_reflect(&data, shape, pad_left, pad_right)?,
        PaddingMode::Replicate => pad_1d_replicate(&data, shape, pad_left, pad_right),
        PaddingMode::Circular => {
            let inner = shape[shape.len() - 1];
            check_circular_positive(&[(inner, pad_left), (inner, pad_right)])?;
            pad_1d_circular(&data, shape, pad_left, pad_right)
        }
        PaddingMode::Zeros => {
            return functional_pad_1d_signed(
                input,
                pad_left as isize,
                pad_right as isize,
                mode,
                value,
            );
        }
    };

    // Grad path: attach Pad1dBackward so the autograd graph stays connected.
    // Without this the prior `from_storage(.., false)` severed it (#1550 bug
    // class), and Conv1d's input gradient would not flow through the non-zero
    // padding_mode pre-pad.
    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(Pad1dBackward {
            input: input.clone(),
            input_shape,
            mode,
            pad_left,
        });
        return Tensor::from_operation(TensorStorage::cpu(out_data), new_shape, grad_fn);
    }

    Tensor::from_storage(TensorStorage::cpu(out_data), new_shape, false)
}

// ---------------------------------------------------------------------------
// Autograd for the 2-D functional pad path (used by Conv2d's non-zero
// padding_mode pre-pad). Every pad mode is a pure *gather*:
//   out[k] = input[src_index_2d(k)]   (or 0 for the out-of-bounds Zeros case).
// The adjoint (VJP) of a gather is a scatter-add into the unpadded input:
//   grad_input[src_index_2d(k)] += grad_output[k].
// This single rule is correct for ALL modes — Zeros (interior crop, padded
// outputs have no source so contribute nothing), Reflect (the reflected
// boundary source indices repeat, so their grads fold/accumulate back onto
// the mirrored interior positions), Replicate (the edge source index repeats,
// summing into the edge), and Circular (wrapped source indices accumulate
// around). Mirrors upstream `torch/nn/modules/conv.py:367-371` routing
// non-zero modes through the differentiable `F.pad`.
// ---------------------------------------------------------------------------

/// For an output element at `(new_row, new_col)` in a 2-D pad, return the
/// linear index `sr * w + sc` into the (single) source plane, or `None` if the
/// element comes from the constant fill (Zeros mode) and has no source.
fn src_index_2d(
    mode: PaddingMode,
    new_row: usize,
    new_col: usize,
    h: usize,
    w: usize,
    pad_left: usize,
    pad_top: usize,
) -> Option<usize> {
    let sr: usize = match mode {
        PaddingMode::Zeros => {
            if new_row < pad_top || new_row >= pad_top + h {
                return None;
            }
            new_row - pad_top
        }
        PaddingMode::Reflect => {
            if new_row < pad_top {
                pad_top - new_row
            } else if new_row >= pad_top + h {
                h - 2 - (new_row - pad_top - h)
            } else {
                new_row - pad_top
            }
        }
        PaddingMode::Replicate => new_row.saturating_sub(pad_top).min(h - 1),
        PaddingMode::Circular => {
            ((new_row as isize - pad_top as isize).rem_euclid(h as isize)) as usize
        }
    };
    let sc: usize = match mode {
        PaddingMode::Zeros => {
            if new_col < pad_left || new_col >= pad_left + w {
                return None;
            }
            new_col - pad_left
        }
        PaddingMode::Reflect => {
            if new_col < pad_left {
                pad_left - new_col
            } else if new_col >= pad_left + w {
                w - 2 - (new_col - pad_left - w)
            } else {
                new_col - pad_left
            }
        }
        PaddingMode::Replicate => new_col.saturating_sub(pad_left).min(w - 1),
        PaddingMode::Circular => {
            ((new_col as isize - pad_left as isize).rem_euclid(w as isize)) as usize
        }
    };
    Some(sr * w + sc)
}

/// Backward node for the 2-D functional pad. Scatter-adds the output gradient
/// back onto the unpadded input plane using the per-output source-index map.
#[derive(Debug)]
struct Pad2dBackward<T: Float> {
    input: Tensor<T>,
    input_shape: Vec<usize>,
    mode: PaddingMode,
    pad_left: usize,
    pad_top: usize,
}

impl<T: Float> GradFn<T> for Pad2dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        let ndim = self.input_shape.len();
        let h = self.input_shape[ndim - 2];
        let w = self.input_shape[ndim - 1];
        let outer: usize = self.input_shape[..ndim - 2]
            .iter()
            .copied()
            .product::<usize>()
            .max(1);

        let go_shape = grad_output.shape();
        let new_h = go_shape[ndim - 2];
        let new_w = go_shape[ndim - 1];

        // The backward runs on host: scatter-add is data-dependent over the
        // index map. `data_vec` materialises the (possibly GPU) grad to CPU.
        let go = grad_output.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let mut grad_in = vec![zero; outer * h * w];

        for o in 0..outer {
            let go_base = o * new_h * new_w;
            let gi_base = o * h * w;
            for nr in 0..new_h {
                for nc in 0..new_w {
                    if let Some(src) =
                        src_index_2d(self.mode, nr, nc, h, w, self.pad_left, self.pad_top)
                    {
                        grad_in[gi_base + src] += go[go_base + nr * new_w + nc];
                    }
                }
            }
        }

        let grad_input =
            Tensor::from_storage(TensorStorage::cpu(grad_in), self.input_shape.clone(), false)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "Pad2dBackward"
    }
}

/// Apply padding to the last 2 dimensions of a tensor using the given mode.
///
/// When `input` requires grad (and grad tracking is enabled) the returned
/// tensor carries a [`Pad2dBackward`] node so gradients flow back to `input`,
/// matching the differentiable `F.pad` that `torch/nn/modules/conv.py`
/// `_conv_forward` routes non-zero `padding_mode`s through.
pub fn functional_pad_2d<T: Float>(
    input: &Tensor<T>,
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
    mode: PaddingMode,
    value: T,
) -> FerrotorchResult<Tensor<T>> {
    // `Zeros` (torch `mode="constant"`) routes through the crop-capable signed
    // path — see the `functional_pad_1d` note. The `value` fill (#1553) is
    // preserved; for non-negative `usize` pads the result is byte-identical.
    if mode == PaddingMode::Zeros {
        return functional_pad_2d_signed(
            input,
            pad_left as isize,
            pad_right as isize,
            pad_top as isize,
            pad_bottom as isize,
            mode,
            value,
        );
    }

    let data = input.data_vec()?;
    let shape = input.shape();
    let input_shape = shape.to_vec();
    let (out_data, new_shape) = match mode {
        PaddingMode::Reflect => {
            pad_2d_reflect(&data, shape, pad_left, pad_right, pad_top, pad_bottom)?
        }
        PaddingMode::Replicate => {
            pad_2d_replicate(&data, shape, pad_left, pad_right, pad_top, pad_bottom)
        }
        PaddingMode::Circular => {
            let nd = shape.len();
            let (h, w) = (shape[nd - 2], shape[nd - 1]);
            check_circular_positive(&[
                (w, pad_left),
                (w, pad_right),
                (h, pad_top),
                (h, pad_bottom),
            ])?;
            pad_2d_circular(&data, shape, pad_left, pad_right, pad_top, pad_bottom)
        }
        PaddingMode::Zeros => {
            return functional_pad_2d_signed(
                input,
                pad_left as isize,
                pad_right as isize,
                pad_top as isize,
                pad_bottom as isize,
                mode,
                value,
            );
        }
    };

    // Grad path: attach Pad2dBackward so the autograd graph stays connected
    // (the prior `from_storage(..., false)` severed it — #1550).
    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(Pad2dBackward {
            input: input.clone(),
            input_shape,
            mode,
            pad_left,
            pad_top,
        });
        return Tensor::from_operation(TensorStorage::cpu(out_data), new_shape, grad_fn);
    }

    Tensor::from_storage(TensorStorage::cpu(out_data), new_shape, false)
}

// ---------------------------------------------------------------------------
// Autograd for the 3-D functional pad path (used by Conv3d's non-zero
// padding_mode pre-pad). Same gather/scatter-add adjoint as the 1-D / 2-D
// cases; see the `Pad2dBackward` block above for the full derivation. Without
// the backward node, the pad returns `requires_grad = false` and severs
// autograd — the #1550 bug class. Mirrors upstream `torch/nn/modules/conv.py`
// `Conv3d._conv_forward` (`conv.py:717-721`) routing non-zero modes through
// the differentiable `F.pad`.
// ---------------------------------------------------------------------------

/// For an output element at `(nd, nh, nw)` in a 3-D pad, return the linear
/// index `sd*H*W + sh*W + sw` into the (single) source volume, or `None` if
/// the element comes from the constant fill (Zeros mode) and has no source.
// Internal helper: the 3-axis pad descriptor (d/h/w + per-axis pad) carries
// proportionally more arguments than the 1-D/2-D variants.
#[allow(clippy::too_many_arguments)]
fn src_index_3d(
    mode: PaddingMode,
    nd: usize,
    nh: usize,
    nw: usize,
    d: usize,
    h: usize,
    w: usize,
    pad_left: usize,
    pad_top: usize,
    pad_front: usize,
) -> Option<usize> {
    // Axis-wise source resolver shared across all three spatial axes. Returns
    // `None` only for the Zeros mode out-of-bounds case (constant fill).
    fn axis(mode: PaddingMode, new_idx: usize, size: usize, pad_lo: usize) -> Option<usize> {
        let s = match mode {
            PaddingMode::Zeros => {
                if new_idx < pad_lo || new_idx >= pad_lo + size {
                    return None;
                }
                new_idx - pad_lo
            }
            PaddingMode::Reflect => {
                if new_idx < pad_lo {
                    pad_lo - new_idx
                } else if new_idx >= pad_lo + size {
                    size - 2 - (new_idx - pad_lo - size)
                } else {
                    new_idx - pad_lo
                }
            }
            PaddingMode::Replicate => new_idx.saturating_sub(pad_lo).min(size - 1),
            PaddingMode::Circular => {
                ((new_idx as isize - pad_lo as isize).rem_euclid(size as isize)) as usize
            }
        };
        Some(s)
    }
    let sd = axis(mode, nd, d, pad_front)?;
    let sh = axis(mode, nh, h, pad_top)?;
    let sw = axis(mode, nw, w, pad_left)?;
    Some(sd * h * w + sh * w + sw)
}

/// Backward node for the 3-D functional pad. Scatter-adds the output gradient
/// back onto the unpadded input volume using the per-output source-index map.
#[derive(Debug)]
struct Pad3dBackward<T: Float> {
    input: Tensor<T>,
    input_shape: Vec<usize>,
    mode: PaddingMode,
    pad_left: usize,
    pad_top: usize,
    pad_front: usize,
}

impl<T: Float> GradFn<T> for Pad3dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        let ndim = self.input_shape.len();
        let d = self.input_shape[ndim - 3];
        let h = self.input_shape[ndim - 2];
        let w = self.input_shape[ndim - 1];
        let outer: usize = self.input_shape[..ndim - 3]
            .iter()
            .copied()
            .product::<usize>()
            .max(1);

        let go_shape = grad_output.shape();
        let new_d = go_shape[ndim - 3];
        let new_h = go_shape[ndim - 2];
        let new_w = go_shape[ndim - 1];

        // The backward runs on host: scatter-add is data-dependent over the
        // index map. `data_vec` materialises the (possibly GPU) grad to CPU.
        let go = grad_output.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let mut grad_in = vec![zero; outer * d * h * w];

        for o in 0..outer {
            let go_base = o * new_d * new_h * new_w;
            let gi_base = o * d * h * w;
            for ndp in 0..new_d {
                for nhp in 0..new_h {
                    for nwp in 0..new_w {
                        if let Some(src) = src_index_3d(
                            self.mode,
                            ndp,
                            nhp,
                            nwp,
                            d,
                            h,
                            w,
                            self.pad_left,
                            self.pad_top,
                            self.pad_front,
                        ) {
                            grad_in[gi_base + src] +=
                                go[go_base + ndp * new_h * new_w + nhp * new_w + nwp];
                        }
                    }
                }
            }
        }

        let grad_input =
            Tensor::from_storage(TensorStorage::cpu(grad_in), self.input_shape.clone(), false)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "Pad3dBackward"
    }
}

/// Apply padding to the last 3 dimensions of a tensor using the given mode.
///
/// When `input` requires grad (and grad tracking is enabled) the returned
/// tensor carries a [`Pad3dBackward`] node so gradients flow back to `input`,
/// matching the differentiable `F.pad` that `torch/nn/modules/conv.py`
/// `Conv3d._conv_forward` routes non-zero `padding_mode`s through
/// (`conv.py:717-721`).
// Public API: matches PyTorch's `torch.nn.functional.pad` signature for the
// 3-axis case (input + 6 pad amounts + mode + value); divergence would
// break parity with the upstream reference.
#[allow(clippy::too_many_arguments)]
pub fn functional_pad_3d<T: Float>(
    input: &Tensor<T>,
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
    pad_front: usize,
    pad_back: usize,
    mode: PaddingMode,
    value: T,
) -> FerrotorchResult<Tensor<T>> {
    // `Zeros` (torch `mode="constant"`) routes through the crop-capable signed
    // path — see the `functional_pad_1d` note. The `value` fill (#1553) is
    // preserved; for non-negative `usize` pads the result is byte-identical.
    if mode == PaddingMode::Zeros {
        return functional_pad_3d_signed(
            input,
            pad_left as isize,
            pad_right as isize,
            pad_top as isize,
            pad_bottom as isize,
            pad_front as isize,
            pad_back as isize,
            mode,
            value,
        );
    }

    let data = input.data_vec()?;
    let shape = input.shape();
    let input_shape = shape.to_vec();
    let (out_data, new_shape) = match mode {
        PaddingMode::Reflect => pad_3d_reflect(
            &data, shape, pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back,
        )?,
        PaddingMode::Replicate => pad_3d_replicate(
            &data, shape, pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back,
        ),
        PaddingMode::Circular => {
            let nd = shape.len();
            let (d, h, w) = (shape[nd - 3], shape[nd - 2], shape[nd - 1]);
            check_circular_positive(&[
                (w, pad_left),
                (w, pad_right),
                (h, pad_top),
                (h, pad_bottom),
                (d, pad_front),
                (d, pad_back),
            ])?;
            pad_3d_circular(
                &data, shape, pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back,
            )
        }
        PaddingMode::Zeros => {
            return functional_pad_3d_signed(
                input,
                pad_left as isize,
                pad_right as isize,
                pad_top as isize,
                pad_bottom as isize,
                pad_front as isize,
                pad_back as isize,
                mode,
                value,
            );
        }
    };

    // Grad path: attach Pad3dBackward so the autograd graph stays connected.
    // Without this the prior `from_storage(.., false)` severed it (#1550 bug
    // class), and Conv3d's input gradient would not flow through the non-zero
    // padding_mode pre-pad.
    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(Pad3dBackward {
            input: input.clone(),
            input_shape,
            mode,
            pad_left,
            pad_top,
            pad_front,
        });
        return Tensor::from_operation(TensorStorage::cpu(out_data), new_shape, grad_fn);
    }

    Tensor::from_storage(TensorStorage::cpu(out_data), new_shape, false)
}

// ===========================================================================
// Signed (crop-capable) functional pad — torch `constant_pad_nd` negative pad
// ===========================================================================
//
// `torch.nn.functional.pad` accepts NEGATIVE pad amounts: a negative pad on a
// side CROPS (removes) `|pad|` elements from that side instead of adding. ALL
// modes support this — upstream `aten/src/ATen/native/PadNd.cpp:207-242`
// (`_pad_enum_symint`) routes `constant` through `constant_pad_nd` (which
// narrows for negatives) and `reflect`/`replicate`/`circular` straight to the
// native `reflection_pad*` / `replication_pad*` / `_pad_circular` kernels, which
// compute `output = input + pad_l + pad_r` (a negative pad narrows the side)
// and gather with offset `max(0,-pad) - max(0,pad)` (ReflectionPad.cpp:46,
// PaddingKernel.cpp:63-65, PadNd.cpp:158-159). The non-constant modes still
// reject a non-zero `value` (PadNd.cpp:217-219). This signed-constant gather is
// the `PaddingMode::Zeros` forward; the other modes compose crop-then-pad
// (`functional_pad_nd_signed`), which is byte-identical to their native kernels.
//
// Forward (PadNd.cpp:29-108): for each padded dim with signed pads `(lo, hi)`
// the cropped input is `narrow(i, -lo, size+lo)` (when `lo<0`) then
// `narrow(i, 0, size'+hi)` (when `hi<0`); the output of size
// `new = size + lo + hi` is `fill_(value)`d and the cropped input copied into
// the `max(lo,0)` offset window (PadNd.cpp:94-106). Equivalently, an output
// index `o` reads source index `s = o - lo`: when `0 <= s < size` it is real
// data, otherwise (only possible for the POSITIVE-pad region) it is the `value`
// fill. This one rule handles MIXED signs per-dim correctly.
//
// Over-crop: torch's `narrow` rejects a negative length
// ("narrow(): length must be non-negative", from PadNd.cpp:49 / :54), and
// PadNd.cpp:76 `TORCH_CHECK(new_dim >= 0)`. We mirror BOTH: a left crop may not
// exceed `size`, and a right crop may not exceed the post-left-crop size — i.e.
// `size + min(lo,0) >= 0` AND `size + min(lo,0) + min(hi,0) >= 0`. A net size of
// exactly 0 is allowed (torch returns an empty dim, e.g. `F.pad(x3, [-1,-2])`).
//
// Backward: the adjoint of a crop-or-pad gather is a scatter-add into the
// (full, original-size) input — `grad_input[o - lo] += grad_output[o]` for the
// in-bounds outputs. Cropped-away positions receive no contribution (grad 0),
// matching torch's `constant_pad_nd` backward being itself a `constant_pad_nd`
// with negated pads.

/// Resolve, for a single axis, the source index a padded/cropped output index
/// reads from. Returns `None` for the constant-fill region (an output position
/// in the POSITIVE-pad area that has no source element). `lo` is the signed pad
/// on the low side of this axis.
#[inline]
fn signed_axis_src(new_idx: usize, size: usize, lo: isize) -> Option<usize> {
    let s = new_idx as isize - lo;
    if s >= 0 && (s as usize) < size {
        Some(s as usize)
    } else {
        None
    }
}

/// Validate the signed pads for one axis against torch's sequential-`narrow`
/// crop rule and return the new axis size. Errors when a crop removes more than
/// the (running) axis size — mirroring torch's
/// "narrow(): length must be non-negative" / `TORCH_CHECK(new_dim >= 0)`.
fn signed_axis_new_size(
    size: usize,
    lo: isize,
    hi: isize,
    axis_label: &str,
) -> FerrotorchResult<usize> {
    // Left crop applies first (PadNd.cpp:49): narrow length `size + lo` must be
    // non-negative when `lo < 0`.
    let after_left: isize = if lo < 0 {
        size as isize + lo
    } else {
        size as isize
    };
    if after_left < 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "constant pad: negative padding {lo} on {axis_label} crops more than the dimension size {size} (narrow length would be negative)"
            ),
        });
    }
    // Right crop applies to the post-left size (PadNd.cpp:54).
    let after_right: isize = if hi < 0 { after_left + hi } else { after_left };
    if after_right < 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "constant pad: negative padding ({lo}, {hi}) on {axis_label} crops more than the dimension size {size}, resulting in a negative output size"
            ),
        });
    }
    // The actual new size also adds the POSITIVE side of each pad back in.
    Ok((after_right + lo.max(0) + hi.max(0)) as usize)
}

/// Generic crop-capable constant pad over the last `npad` dimensions.
///
/// `pads` is `[lo_0, hi_0, lo_1, hi_1, ...]` ordered from the LAST padded axis
/// inward (i.e. matching the `(left, right, top, bottom, front, back)`
/// flattened layout the public entrypoints use). Returns `(data, new_shape)`.
fn pad_nd_signed_constant<T: Float>(
    data: &[T],
    shape: &[usize],
    pads: &[(isize, isize)],
    value: T,
) -> FerrotorchResult<(Vec<T>, Vec<usize>)> {
    let ndim = shape.len();
    let npad = pads.len();
    // `pads[0]` targets the LAST axis; map axis k (0-based from the last padded
    // axis) to absolute dim `ndim - 1 - k`.
    let mut new_shape = shape.to_vec();
    let mut new_sizes = vec![0usize; npad]; // new_sizes[k] for axis ndim-1-k
    for (k, &(lo, hi)) in pads.iter().enumerate() {
        let dim = ndim - 1 - k;
        let new_size = signed_axis_new_size(shape[dim], lo, hi, &format!("dimension {dim}"))?;
        new_sizes[k] = new_size;
        new_shape[dim] = new_size;
    }

    // Outer dims (everything before the first padded axis) are untouched.
    let first_padded = ndim - npad;
    let outer: usize = shape[..first_padded]
        .iter()
        .copied()
        .product::<usize>()
        .max(1);

    let new_total: usize = new_shape.iter().copied().product();
    let mut out = vec![value; new_total];

    // Degenerate input (numel 0 — e.g. shape `[0, 3]`: empty data buffer with a
    // non-empty declared dim): no source data to gather. Mirror upstream
    // `aten/src/ATen/native/PadNd.cpp:94-106`, which `fill_(value)`s the output
    // then `copy_`s the (empty) source — a no-op — leaving the value-filled
    // output. The guard prevents an out-of-bounds index into the empty `data`
    // (same #1551 bug class the positive-only `pad_*d_constant` helpers guard).
    if data.is_empty() {
        return Ok((out, new_shape));
    }

    // Per-element gather over the padded sub-volume. `npad` is at most 3 here,
    // so a small fixed-stride walk over the last axes is sufficient and clear.
    // Strides within the (single outer slice of the) input / output.
    let in_inner: usize = shape[first_padded..].iter().product();
    let out_inner: usize = new_shape[first_padded..].iter().product();

    // Source coordinate buffer reused per output element.
    for o in 0..outer {
        let in_base = o * in_inner;
        let out_base = o * out_inner;
        for flat in 0..out_inner {
            // Decode `flat` into per-axis output coords (last axis fastest).
            let mut rem = flat;
            let mut src_lin = 0usize;
            let mut src_stride = 1usize;
            let mut missing = false;
            // Walk axes from last (k=0) to first padded (k=npad-1).
            for k in 0..npad {
                let dim = ndim - 1 - k;
                let axis_new = new_shape[dim];
                let coord = rem % axis_new;
                rem /= axis_new;
                let lo = pads[k].0;
                match signed_axis_src(coord, shape[dim], lo) {
                    Some(s) => {
                        src_lin += s * src_stride;
                        src_stride *= shape[dim];
                    }
                    None => {
                        missing = true;
                        break;
                    }
                }
            }
            if !missing {
                out[out_base + flat] = data[in_base + src_lin];
            }
            // else: leave the `value` fill already in place.
        }
    }

    Ok((out, new_shape))
}

/// Backward node for the signed (crop-capable) constant functional pad. The
/// adjoint of the crop/pad gather is a scatter-add into the original-size
/// input: `grad_input[o - lo] += grad_output[o]` for in-bounds outputs. Cropped
/// positions get no contribution (grad 0). Mirrors torch's `constant_pad_nd`
/// backward (itself a `constant_pad_nd` with negated pads).
#[derive(Debug)]
struct PadNdSignedBackward<T: Float> {
    input: Tensor<T>,
    input_shape: Vec<usize>,
    /// `(lo, hi)` per padded axis, ordered LAST axis first (same as the forward).
    pads: Vec<(isize, isize)>,
}

impl<T: Float> GradFn<T> for PadNdSignedBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        let ndim = self.input_shape.len();
        let npad = self.pads.len();
        let first_padded = ndim - npad;
        let outer: usize = self.input_shape[..first_padded]
            .iter()
            .copied()
            .product::<usize>()
            .max(1);
        let in_inner: usize = self.input_shape[first_padded..].iter().product();

        let go_shape = grad_output.shape();
        let out_inner: usize = go_shape[first_padded..].iter().product();

        // The backward runs on host: scatter-add is data-dependent over the
        // index map. `data_vec` materialises the (possibly GPU) grad to CPU.
        let go = grad_output.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let mut grad_in = vec![zero; outer * in_inner];

        for o in 0..outer {
            let in_base = o * in_inner;
            let out_base = o * out_inner;
            for flat in 0..out_inner {
                let mut rem = flat;
                let mut src_lin = 0usize;
                let mut src_stride = 1usize;
                let mut missing = false;
                for k in 0..npad {
                    let dim = ndim - 1 - k;
                    let axis_new = go_shape[dim];
                    let coord = rem % axis_new;
                    rem /= axis_new;
                    let lo = self.pads[k].0;
                    match signed_axis_src(coord, self.input_shape[dim], lo) {
                        Some(s) => {
                            src_lin += s * src_stride;
                            src_stride *= self.input_shape[dim];
                        }
                        None => {
                            missing = true;
                            break;
                        }
                    }
                }
                if !missing {
                    grad_in[in_base + src_lin] += go[out_base + flat];
                }
            }
        }

        let grad_input =
            Tensor::from_storage(TensorStorage::cpu(grad_in), self.input_shape.clone(), false)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "PadNdSignedBackward"
    }
}

/// Apply the all-non-negative pad part of `pads` under a non-`Zeros` mode by
/// delegating to the positive-only helpers, so reflect/replicate/circular keep
/// their exact gather + autograd behaviour. `pads` is LAST axis first.
fn functional_pad_nd_positive<T: Float>(
    input: &Tensor<T>,
    pads: &[(isize, isize)],
    mode: PaddingMode,
    value: T,
) -> FerrotorchResult<Tensor<T>> {
    match pads.len() {
        1 => functional_pad_1d(input, pads[0].0 as usize, pads[0].1 as usize, mode, value),
        2 => functional_pad_2d(
            input,
            pads[0].0 as usize,
            pads[0].1 as usize,
            pads[1].0 as usize,
            pads[1].1 as usize,
            mode,
            value,
        ),
        3 => functional_pad_3d(
            input,
            pads[0].0 as usize,
            pads[0].1 as usize,
            pads[1].0 as usize,
            pads[1].1 as usize,
            pads[2].0 as usize,
            pads[2].1 as usize,
            mode,
            value,
        ),
        other => Err(FerrotorchError::InvalidArgument {
            message: format!("functional_pad_nd_signed supports 1-3 padded dims, got {other}"),
        }),
    }
}

/// Unified reflect index map matching upstream
/// `aten/src/ATen/native/cpu/PaddingKernel.cpp:63-80`. `j` is the output
/// position, `size` is the ORIGINAL input size on this axis, and `pad` is the
/// signed LOW-side pad. The window offset is
/// `offset = max(0, -pad) - max(0, pad)` (`PaddingKernel.cpp:63-65`); the
/// reflected index is then read as `i + offset` from the ORIGINAL input
/// (`PaddingKernel.cpp:71-80`). This reads the original window directly, so a
/// positive pad on a cropped side correctly reaches elements a crop-first pass
/// would have discarded. Caller guarantees the resolved index is in
/// `0..size` via the reflect legality check (`|pad| < size` per side).
#[inline]
fn reflect_axis_src(j: usize, size: usize, pad: isize) -> usize {
    let j = j as isize;
    let size_i = size as isize;
    let offset = 0i64.max(-(pad as i64)) - 0i64.max(pad as i64);
    let offset = offset as isize;
    let i = if j < pad {
        pad * 2 - j
    } else if j >= pad && j < size_i + pad {
        j
    } else {
        (size_i + pad - 1) * 2 - j
    };
    (i + offset) as usize
}

/// Unified replicate index map matching upstream `ReplicationPad::index`
/// (`aten/src/ATen/native/cpu/PaddingKernel.cpp:84-95`). `j` is the output
/// position, `size` is the ORIGINAL input size on this axis, and `pad` is the
/// signed LOW-side pad. The window offset is
/// `offset = max(0, -pad) - max(0, pad)` (`PaddingKernel.cpp:63-65`); the
/// CLAMPED index is then read as `i + offset` from the ORIGINAL input window
/// (`PaddingKernel.cpp:87-94`): a position before the (possibly cropped) window
/// clamps to the left boundary `pad`, a position past it clamps to the right
/// boundary `size + pad - 1`, and an interior position reads `j`. Because the
/// gather always resolves against the ORIGINAL window, an over-crop that leaves
/// a zero-size axis still reads the preserved edge element — no `inner - 1`
/// underflow, no panic (#1625, R-CODE-2). For a non-negative pad this is
/// byte-identical to the old crop-then-pad clamp. Caller guarantees `size >= 1`
/// (an empty original axis cannot be replicated; the legality check rejects it).
#[inline]
fn replicate_axis_src(j: usize, size: usize, pad: isize) -> usize {
    let j = j as isize;
    let size_i = size as isize;
    let offset = 0i64.max(-(pad as i64)) - 0i64.max(pad as i64);
    let offset = offset as isize;
    let i = if j < pad {
        pad
    } else if j >= pad && j < size_i + pad {
        j
    } else {
        size_i + pad - 1
    };
    (i + offset) as usize
}

/// Circular index map mirroring `_pad_circular`'s slice-copy gather
/// (`aten/src/ATen/native/PadNd.cpp:148-187`). The kernel first copies a
/// (possibly cropped) center slice `out[max(lo,0) .. out_w-max(hi,0)]` from
/// `in[max(-lo,0) .. size-max(-hi,0)]`, then wraps the left pad from the END of
/// the output and the right pad from the START. So a wrap reads from the
/// CROPPED center — NOT a plain modulo against the original window (which only
/// coincides when there is no crop). `j` is the output position, `size` the
/// ORIGINAL input size, `(lo, hi)` the signed pads on this axis. Returns the
/// RAW (signed) source index into the ORIGINAL input; it may fall outside
/// `0..size` for an illegal pad — `circular_axis_new_size` pre-validates every
/// index lies in `0..size` before the gather casts it to `usize`. Only called
/// for `out_w >= 1` (an empty `out_w == 0` axis runs the gather zero times).
#[inline]
fn circular_axis_src(j: usize, size: usize, lo: isize, hi: isize) -> isize {
    let j = j as isize;
    let size_i = size as isize;
    let out_w = size_i + lo + hi;
    let lo_pos = lo.max(0);
    let hi_pos = hi.max(0);
    // Resolve `j` to a center-region output index (left/right wraps copy from
    // the already-written center), then map that center index to the input.
    let center = if j < lo_pos {
        // Left wrap (`pad_l > 0`): out[0..lo] <- out[out_w-lo-hi_pos .. out_w-hi_pos].
        out_w - lo - hi_pos + j
    } else if j >= out_w - hi_pos {
        // Right wrap (`pad_r > 0`): out[out_w-hi .. out_w] <- out[lo_pos .. lo_pos+hi].
        lo_pos + (j - (out_w - hi))
    } else {
        j
    };
    // Center → input: in[max(-lo,0) + (center - max(lo,0))].
    lo.min(0).abs() + (center - lo_pos)
}

/// PER-AXIS circular-pad legality, returning the new axis extent
/// (`size + lo + hi`, which may be `0` for a net-zero crop → an empty dim).
///
/// This mirrors EXACTLY the two `TORCH_CHECK`s inside `_pad_circular_symint`'s
/// shape loop (`aten/src/ATen/native/PadNd.cpp:140-145`) — and ONLY those. The
/// center slice-copy (`:158-161`) and the wrap gather (`:169-187`) are NOT
/// per-axis legality: torch first allocates the FULL N-D output
/// (`:148 auto out = self.new_empty_symint(out_shape, ...)`) and only THEN does
/// the per-axis `copy_`. Those `copy_`s are validated in
/// [`circular_axis_validate_nonempty`], gated on the WHOLE output being
/// non-empty (when any axis is `0`, `out` has `numel 0` and every `copy_` is a
/// no-op — see the holistic restructure in [`pad_nd_signed_reflect_circular`]).
///
/// - `:140-142` `TORCH_CHECK(pad_l <= size && pad_r <= size, "Padding value
///   causes wrapping around more than once.")` — a pad strictly greater than
///   `size` wraps more than once → `Err`. This is the ONLY per-axis legality.
/// - `:143-145` `TORCH_CHECK(out_shape >= 0, "Negative padding value is
///   resulting in an empty dimension")` — a negative net extent → `Err`; a net
///   extent of EXACTLY `0` is allowed (an empty `[..,0]` dim, like
///   `constant_pad_nd`), distinct from reflect which demands `>= 1`.
fn circular_axis_legality(
    size: usize,
    lo: isize,
    hi: isize,
    dim: usize,
) -> FerrotorchResult<usize> {
    let size_i = size as isize;
    // `:140-142` — a pad larger than the dim wraps around more than once.
    if lo > size_i || hi > size_i {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "Circular padding ({lo}, {hi}) causes wrapping around more than once on dimension {dim} (size {size})"
            ),
        });
    }
    // `:143-145` — a negative net extent is an error; net zero is an empty dim.
    let out_w = size_i + lo + hi;
    if out_w < 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "Circular padding ({lo}, {hi}) on dimension {dim} of size {size} results in a negative output size {out_w} (empty dimension)"
            ),
        });
    }
    Ok(out_w as usize)
}

/// Normalize a `tensor.slice(dim, start, end)` to a clamped `[start, end)`
/// half-open range over a `length`-element axis, mirroring torch's
/// `slice_symint` index normalization (negative indices `+= length`, then clamp
/// to `[0, length]`). Used by [`circular_slicecopy_block`] to model every
/// `slice_symint` in `_pad_circular_symint` (`PadNd.cpp:148-187`).
#[inline]
fn circular_slice_range(length: isize, mut start: isize, mut end: isize) -> (usize, usize) {
    if start < 0 {
        start += length;
    }
    if end < 0 {
        end += length;
    }
    start = start.clamp(0, length);
    end = end.clamp(0, length);
    if end < start {
        end = start;
    }
    (start as usize, end as usize)
}

/// HOLISTIC faithful simulation of torch's `_pad_circular_symint` allocate-then-
/// copy algorithm (`aten/src/ATen/native/PadNd.cpp:148-187`) over the last
/// `npad` dims of ONE outer batch block. This replaces the prior per-axis
/// wrap-OOB / center-copy pre-validation (which rejected an axis whose ISOLATED
/// wrap was OOB even when a SIBLING axis had already emptied the whole output —
/// the #1628 cross-axis net-zero divergence). Instead of validating each axis in
/// isolation, we reproduce torch's exact sequence on the full N-D output buffer:
///
/// - `:148` `auto out = self.new_empty_symint(out_shape)` — a buffer with an
///   `init` mask (all `false`); uninitialized cells are tracked so an over-crop
///   that leaves a final cell unwritten is detected as the R-DEV-6 carve-out.
/// - `:154-161` ONE center `copy_`: narrow `out` and `self` on every padded dim
///   by `slice(dim, max(pad,0), …)` / `slice(dim, max(-pad,0), …)`, then copy.
///   `copy_` errors unless the source broadcasts to the destination shape (per
///   dim: sizes equal OR source size 1); a mismatch is a torch `RuntimeError`.
/// - `:169-187` the left/right wrap `copy_`s, each reading from `out` LIVE
///   (`in_slice = out.slice_symint(...)` aliases the buffer being written, so a
///   wrap reads cells the center or an earlier wrap just wrote — `:163-165`
///   "Corners will be written more than once"). Same broadcast-legality gate.
///
/// Because the wraps read from `out`, an axis whose isolated wrap would be OOB
/// is harmless when a different axis emptied the output (every `copy_` is then a
/// no-op over the empty extent), and torch's well-defined cross-axis wraps that
/// the prior per-axis check rejected are now reproduced byte-for-byte. After all
/// copies, any cell still uninitialized means torch read uninitialized memory
/// there (no reproducible byte-for-byte contract — R-DEV-6); ferrotorch rejects
/// such cases cleanly rather than returning nondeterministic garbage (R-CODE-2:
/// no panic). The legality `:140-145` is already enforced by
/// [`circular_axis_legality`] before this runs.
fn circular_slicecopy_block<T: Float>(
    in_block: &[T],
    in_inner_shape: &[usize],
    out_inner_shape: &[usize],
    pads: &[(isize, isize)],
) -> FerrotorchResult<Vec<T>> {
    let npad = pads.len();
    let ninner = in_inner_shape.len();
    let out_total: usize = out_inner_shape.iter().product();
    let zero = <T as num_traits::Zero>::zero();
    let mut out = vec![zero; out_total];
    let mut init = vec![false; out_total];

    // Row-major strides for the inner (padded-region) coordinate space.
    let mut in_strides = vec![1usize; ninner];
    let mut out_strides = vec![1usize; ninner];
    for d in (0..ninner.saturating_sub(1)).rev() {
        in_strides[d] = in_strides[d + 1] * in_inner_shape[d + 1];
        out_strides[d] = out_strides[d + 1] * out_inner_shape[d + 1];
    }

    // `pads` is ordered LAST padded axis first; the padded inner dims are the
    // trailing `npad` dims of the inner block. Inner-dim index for pad entry `k`
    // (which targets axis `dim = ninner - 1 - k`).
    let pad_for_inner_dim = |d: usize| -> (isize, isize) {
        // d in [ninner-npad, ninner-1] -> k = ninner - 1 - d
        pads[ninner - 1 - d]
    };

    // Per-dim half-open copy windows for the dst (`out`) and src.
    // `copy_block` copies `src[src_win]` (broadcast) into `out[dst_win]`,
    // propagating the init mask, and returns Err on a broadcast-illegal `copy_`.
    // `dst_win`/`src_win` are `(start,end)` per inner dim.
    //
    // When `read_data` is `Some`, the source is a SEPARATE buffer (the original
    // input, for the center copy — `read_strides` indexes it). When `read_data`
    // is `None`, the source is `out` ITSELF, read LIVE in the same pass: this
    // mirrors torch's `:169-187` wrap `copy_`s where `in_slice = out.slice(...)`
    // aliases the very `out` buffer being written (`read_strides` indexes `out`).
    // Iterating in row-major dst order, a wrap cell can therefore read a cell the
    // center (or an earlier dst cell in this same wrap) just wrote, deterministi-
    // cally propagating a narrow center band exactly as torch does (#1629).
    #[allow(clippy::too_many_arguments)]
    fn copy_block<T: Float>(
        out: &mut [T],
        init: &mut [bool],
        read_data: Option<&[T]>,
        read_init: Option<&[bool]>,
        ninner: usize,
        out_strides: &[usize],
        read_strides: &[usize],
        dst_win: &[(usize, usize)],
        src_win: &[(usize, usize)],
    ) -> FerrotorchResult<()> {
        // Broadcast-legality (torch `copy_`): per dim, dst extent must equal src
        // extent OR src extent must be 1. Otherwise torch raises.
        let mut dst_ext = vec![0usize; ninner];
        let mut src_ext = vec![0usize; ninner];
        for d in 0..ninner {
            dst_ext[d] = dst_win[d].1 - dst_win[d].0;
            src_ext[d] = src_win[d].1 - src_win[d].0;
            if dst_ext[d] != src_ext[d] && src_ext[d] != 1 {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "Circular padding: a slice copy of source extent {} into destination extent {} is not broadcastable on inner dim {d} (torch raises a size-mismatch here)",
                        src_ext[d], dst_ext[d]
                    ),
                });
            }
        }
        let total: usize = dst_ext.iter().product();
        if total == 0 {
            return Ok(()); // no-op over an empty extent (`:148` empty out_shape)
        }
        // torch `copy_` memory-overlap gate (live-read wraps only). When the wrap
        // reads from `out` itself (`read_data is None`), torch's `copy_` raises
        // `RuntimeError: ... refer to a single memory location` when the source
        // and destination slices each form a CONTIGUOUS memory run AND those runs
        // overlap by a non-identity offset (MEM_OVERLAP_YES). A wrap slices a
        // SINGLE dim `wd` (all other dims full-extent); its dst/src each form a
        // contiguous run iff every dim MORE MAJOR than `wd` has extent 1 (else the
        // slice repeats once per major index → strided, and torch's overlap
        // detector returns "too hard" and proceeds with the well-defined band
        // propagation, #1629). An EXACT-identity window pair is a self-copy no-op
        // torch always allows; disjoint windows never overlap. We mirror torch's
        // raise as a clean `Err` (R-CODE-2: never a panic).
        if read_data.is_none() {
            let mut wrap_dim: Option<usize> = None;
            for d in 0..ninner {
                if dst_win[d] != src_win[d] {
                    // a wrap differs on exactly one (the wrap) dim
                    wrap_dim = Some(d);
                    break;
                }
            }
            if let Some(wd) = wrap_dim {
                // contiguous run iff every more-major dim is collapsed to extent 1
                let runs_contiguous = (0..wd).all(|d| dst_ext[d] == 1);
                let ds = dst_win[wd];
                let ss = src_win[wd];
                let overlap = ds.0 < ss.1 && ss.0 < ds.1; // half-open range overlap
                let identical = ds == ss;
                if runs_contiguous && overlap && !identical {
                    return Err(FerrotorchError::InvalidArgument {
                        message:
                            "Circular padding: torch's wrap copy_ would read and write a single memory location over a contiguous slice (RuntimeError: some elements of the input and written-to tensor refer to a single memory location); ferrotorch rejects rather than fabricate (R-DEV-6)"
                                .to_string(),
                    });
                }
            }
        }
        // Iterate every dst coordinate; map to the (broadcast) src coordinate.
        let mut coord = vec![0usize; ninner];
        for _ in 0..total {
            let mut dst_off = 0usize;
            let mut src_off = 0usize;
            for d in 0..ninner {
                let dc = dst_win[d].0 + coord[d];
                dst_off += dc * out_strides[d];
                let sc = if src_ext[d] == 1 {
                    src_win[d].0
                } else {
                    src_win[d].0 + coord[d]
                };
                src_off += sc * read_strides[d];
            }
            // LIVE read: when `read_data` is `None` the source IS `out`/`init`
            // (torch's wrap `in_slice = out.slice(...)`), so we read the current
            // value at `src_off` — including a cell written earlier in this very
            // pass — before overwriting `dst_off`.
            let (v, src_inited) = match (read_data, read_init) {
                (Some(rd), ri) => (rd[src_off], ri.map(|m| m[src_off]).unwrap_or(true)),
                (None, _) => (out[src_off], init[src_off]),
            };
            out[dst_off] = v;
            init[dst_off] = src_inited;
            // advance coord (row-major over dst extents)
            let mut d = ninner;
            while d > 0 {
                d -= 1;
                coord[d] += 1;
                if coord[d] < dst_ext[d] {
                    break;
                }
                coord[d] = 0;
            }
        }
        Ok(())
    }

    // `:154-161` — the single CENTER copy. Build dst/src windows per inner dim.
    let mut dst_win = vec![(0usize, 0usize); ninner];
    let mut src_win = vec![(0usize, 0usize); ninner];
    for d in 0..ninner {
        let out_len = out_inner_shape[d] as isize;
        let in_len = in_inner_shape[d] as isize;
        if d < ninner - npad {
            // Non-padded inner dim: full extent on both sides.
            dst_win[d] = (0, out_inner_shape[d]);
            src_win[d] = (0, in_inner_shape[d]);
        } else {
            let (pl, pr) = pad_for_inner_dim(d);
            dst_win[d] = circular_slice_range(out_len, pl.max(0), out_len - pr.max(0));
            src_win[d] = circular_slice_range(in_len, (-pl).max(0), in_len - (-pr).max(0));
        }
    }
    copy_block(
        &mut out,
        &mut init,
        Some(in_block),
        None,
        ninner,
        &out_strides,
        &in_strides,
        &dst_win,
        &src_win,
    )?;

    // `:169-187` — the left/right wrap copies, each reading from `out` LIVE.
    // torch's `in_slice = out.slice_symint(...)` (`:176`/`:184`) aliases the SAME
    // `out` buffer the loop is writing, and `:163-165` is explicit that corners
    // are written more than once across the sequence. So each wrap reads the
    // CURRENT `out` (including cells the center or an earlier wrap just wrote),
    // deterministically propagating a narrow over-cropped center band exactly as
    // torch does (#1629). We pass `read_data = None` so `copy_block` reads `out`/
    // `init` in place — NOT a pre-copy snapshot. Cells torch never writes stay
    // uninit and are caught by the leftover-uninit R-DEV-6 check below.
    for (k, &(pl, pr)) in pads.iter().enumerate() {
        // i in torch is k counted from the FIRST padded axis; torch's `dim` is
        // the inner dim `ninner - npad + k`. Our `pads` is last-axis-first, so
        // entry k targets inner dim `ninner - 1 - k`. torch iterates i=0..npad
        // over `pad[2*i]` (first-axis-first); the set of (dim,pl,pr) visited is
        // identical, only the order differs — and torch's wraps on distinct dims
        // are order-independent for the WELL-DEFINED cases (the order-dependent
        // overlapping ones land in the R-DEV-6 leftover-uninit reject either way).
        let dim = ninner - 1 - k;
        let out_len = out_inner_shape[dim] as isize;
        if pl > 0 {
            let mut dwin = vec![(0usize, 0usize); ninner];
            let mut swin = vec![(0usize, 0usize); ninner];
            for d in 0..ninner {
                dwin[d] = (0, out_inner_shape[d]);
                swin[d] = (0, out_inner_shape[d]);
            }
            dwin[dim] = circular_slice_range(out_len, 0, pl);
            swin[dim] =
                circular_slice_range(out_len, out_len - pl - pr.max(0), out_len - pr.max(0));
            copy_block(
                &mut out,
                &mut init,
                None,
                None,
                ninner,
                &out_strides,
                &out_strides,
                &dwin,
                &swin,
            )?;
        }
        if pr > 0 {
            let mut dwin = vec![(0usize, 0usize); ninner];
            let mut swin = vec![(0usize, 0usize); ninner];
            for d in 0..ninner {
                dwin[d] = (0, out_inner_shape[d]);
                swin[d] = (0, out_inner_shape[d]);
            }
            dwin[dim] = circular_slice_range(out_len, out_len - pr, out_len);
            swin[dim] = circular_slice_range(out_len, pl.max(0), pl.max(0) + pr);
            copy_block(
                &mut out,
                &mut init,
                None,
                None,
                ninner,
                &out_strides,
                &out_strides,
                &dwin,
                &swin,
            )?;
        }
    }

    // R-DEV-6: if any output cell is still uninitialized, torch read freed /
    // uninitialized memory there (a mixed-sign over-crop where the cropped
    // center is narrower than the wrap, or an overlapping `copy_`). There is no
    // reproducible byte-for-byte contract, so ferrotorch rejects cleanly rather
    // than emit nondeterministic garbage (R-CODE-2: no panic).
    if init.iter().any(|&b| !b) {
        return Err(FerrotorchError::InvalidArgument {
            message:
                "Circular padding crops the center below the wrap width, so torch reads uninitialized memory (no byte-for-byte contract; R-DEV-6)"
                    .to_string(),
        });
    }
    Ok(out)
}

/// Resolve, for one axis, the source index a reflect/circular output index
/// reads from the ORIGINAL input window. Both modes always read a real element
/// (never a fill), so this returns a bare `usize`. `(lo, hi)` are the signed
/// pads on this axis. The circular index is pre-validated in
/// `circular_axis_new_size` to lie in `0..size`, so the `as usize` cast here is
/// always in-bounds (no OOB — R-CODE-2).
#[inline]
fn signed_mode_axis_src(mode: PaddingMode, j: usize, size: usize, lo: isize, hi: isize) -> usize {
    match mode {
        PaddingMode::Reflect => reflect_axis_src(j, size, lo),
        PaddingMode::Replicate => replicate_axis_src(j, size, lo),
        PaddingMode::Circular => circular_axis_src(j, size, lo, hi) as usize,
        // Zeros routes through the constant gather; this resolver is only invoked
        // for Reflect/Replicate/Circular (see `pad_nd_signed_reflect_circular` /
        // `PadNdSignedModeBackward`); the clamp here is a defensive in-bounds
        // fallback that never executes.
        PaddingMode::Zeros => (j as isize - lo).clamp(0, size as isize - 1) as usize,
    }
}

/// Crop-capable reflect/replicate/circular pad over the last `npad` dimensions
/// using the unified index map against the ORIGINAL input window. `pads` is
/// `[(lo,hi), ...]` ordered LAST padded axis first. Output extent per axis is
/// `size + lo + hi` (negative pads narrow). Reflect legality (SIGNED `lo < size`
/// and `hi < size` per axis, checked against the ORIGINAL size, mirroring
/// `aten/src/ATen/native/ReflectionPad.cpp:48-49`) is validated here. Reflect &
/// replicate use a RANK-DEPENDENT net-zero rule: 1-D requires output `>= 1`
/// while 2-D/3-D allow a per-axis net-zero (empty `[..,0,..]`) so long as one
/// padded axis survives (`ReflectionPad.cpp:251`/`:152`,
/// `ReplicationPadding.cpp:114`). Replicate gathers with the boundary clamp of
/// `ReplicationPad::index` (`cpu/PaddingKernel.cpp:84-95`), so an over-crop to a
/// zero-size axis never underflows (#1625).
fn pad_nd_signed_reflect_circular<T: Float>(
    data: &[T],
    shape: &[usize],
    pads: &[(isize, isize)],
    mode: PaddingMode,
) -> FerrotorchResult<(Vec<T>, Vec<usize>)> {
    let ndim = shape.len();
    let npad = pads.len();
    let mut new_shape = shape.to_vec();
    // Reflect's net-zero output rule is RANK-DEPENDENT (matches torch's per-rank
    // meta functions): 1-D `reflection_pad1d` requires `output_w >= 1`
    // (`aten/src/ATen/native/ReflectionPad.cpp:60-65`) so a net-zero axis Errs,
    // but 2-D `reflection_pad2d` (`:251`) and 3-D `reflection_pad3d` (`:152`)
    // require only `output_w >= 1 || output_h >= 1 (|| output_d >= 1)`, allowing
    // an INDIVIDUAL spatial axis to be net-zero (an empty `[..,0,..]` tensor) as
    // long as at least one spatial axis survives. Replicate has the identical
    // rank-dependent shape: `replication_pad1d` requires `owidth >= 1`
    // (`ReplicationPadding.cpp:49`) while `replication_pad2d`/`3d` use the same
    // OR (`:114`). So per-axis we reject a net-zero ONLY when `npad == 1` (the
    // 1-D kernel); for `npad >= 2` a single axis may be 0, and a final guard
    // below enforces that not ALL spatial axes are 0. (#1626)
    let per_axis_min: isize = isize::from(npad == 1);
    for (k, &(lo, hi)) in pads.iter().enumerate() {
        let dim = ndim - 1 - k;
        let size = shape[dim] as isize;
        // Reflect: torch's check is SIGNED, not absolute
        // (`aten/src/ATen/native/ReflectionPad.cpp:48-49`):
        // `TORCH_CHECK(pad_l < input_w && pad_r < input_w, ...)`. A NEGATIVE
        // (crop) pad is always `< input_w`, so torch only rejects POSITIVE pads
        // whose magnitude reaches `>= input_w`. Replicate has NO such
        // `pad < input` check upstream (`ReplicationPadding.cpp` only guards the
        // output extent), so this rejection is reflect-only.
        if mode == PaddingMode::Reflect && (lo >= size || hi >= size) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Reflection padding ({lo}, {hi}) must be less than input size ({size}) on dimension {dim}"
                ),
            });
        }
        // Replicate requires a non-empty ORIGINAL axis (the clamp gathers a real
        // boundary element). torch's `check_valid_input` rejects a zero-size
        // input plane, so size 0 here is impossible for a valid call; guard
        // defensively to keep the clamp index in `0..size`.
        if mode == PaddingMode::Replicate && size == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Replication padding cannot replicate an empty input dimension {dim} (size 0)"
                ),
            });
        }
        // Circular: torch's `_pad_circular_symint` is allocate-then-copy
        // (`aten/src/ATen/native/PadNd.cpp:140-187`). The PER-AXIS legality is
        // ONLY `:142` (reject `pad > size`, wraps more than once) and `:144`
        // (reject a negative net extent; allow exactly `0` → an empty dim) —
        // `circular_axis_legality`. The center copy (`:158-161`) and the wrap
        // gather (`:169-187`) operate on slices of the FULL `:148 new_empty`
        // output, so they are validated SEPARATELY below, gated on the WHOLE
        // output being non-empty (any `out_i == 0` ⇒ every `copy_` no-ops ⇒
        // torch returns the empty tensor without materializing ANY wrap index,
        // #1628). Reflect/Replicate use the rank-dependent `per_axis_min` reject.
        let new_size: usize = if mode == PaddingMode::Circular {
            circular_axis_legality(shape[dim], lo, hi, dim)?
        } else {
            let n = size + lo + hi;
            if n < per_axis_min {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "padding ({lo}, {hi}) on dimension {dim} of size {size} yields output size {n} below the minimum {per_axis_min} for this rank"
                    ),
                });
            }
            n as usize
        };
        new_shape[dim] = new_size;
    }

    // 2-D/3-D reflect & replicate: at least one padded spatial axis must survive
    // (`output_w >= 1 || output_h >= 1 (|| output_d >= 1)`,
    // `ReflectionPad.cpp:251`/`:152`, `ReplicationPadding.cpp:114`). When every
    // padded axis collapsed to 0, torch Errs "input is too small".
    if npad >= 2
        && matches!(mode, PaddingMode::Reflect | PaddingMode::Replicate)
        && pads
            .iter()
            .enumerate()
            .all(|(k, _)| new_shape[ndim - 1 - k] == 0)
    {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "{mode:?} padding collapses every padded spatial axis to size 0 (torch requires at least one >= 1)"
            ),
        });
    }

    let first_padded = ndim - npad;
    let outer: usize = shape[..first_padded]
        .iter()
        .copied()
        .product::<usize>()
        .max(1);
    let in_inner: usize = shape[first_padded..].iter().product();
    let out_inner: usize = new_shape[first_padded..].iter().product();
    let zero = <T as num_traits::Zero>::zero();
    let new_total: usize = new_shape.iter().copied().product();
    let mut out = vec![zero; new_total];

    // CIRCULAR: HOLISTIC allocate-then-copy (`PadNd.cpp:148-187`) per outer
    // batch, mirroring torch's `:148 new_empty(out_shape)` + center/wrap `copy_`
    // sequence on the FULL N-D output. This replaces the prior per-axis wrap-OOB
    // pre-validation + per-axis gather, which rejected an axis whose ISOLATED
    // wrap was OOB even when a SIBLING axis had already emptied the whole output
    // (the #1628 cross-axis net-zero divergence). The simulator reproduces the
    // empty short-circuit (any `out_i == 0` ⇒ every `copy_` no-ops ⇒ empty
    // tensor), the cross-axis well-defined wraps, AND the R-DEV-6 over-crop
    // rejection (leftover-uninit ⇒ Err, never a panic) in one faithful pass.
    if mode == PaddingMode::Circular {
        let in_inner_shape = &shape[first_padded..];
        let out_inner_shape = &new_shape[first_padded..];
        for o in 0..outer {
            let in_block = &data[o * in_inner..(o + 1) * in_inner];
            let out_block =
                circular_slicecopy_block(in_block, in_inner_shape, out_inner_shape, pads)?;
            out[o * out_inner..(o + 1) * out_inner].copy_from_slice(&out_block);
        }
        return Ok((out, new_shape));
    }

    // REFLECT / REPLICATE: the unified original-window per-axis gather
    // (`cpu/PaddingKernel.cpp:63-105`). Each output index reads a real input
    // element via the mode's per-axis index resolver.
    for o in 0..outer {
        let in_base = o * in_inner;
        let out_base = o * out_inner;
        for flat in 0..out_inner {
            let mut rem = flat;
            let mut src_lin = 0usize;
            let mut src_stride = 1usize;
            for k in 0..npad {
                let dim = ndim - 1 - k;
                let axis_new = new_shape[dim];
                let coord = rem % axis_new;
                rem /= axis_new;
                let (lo, hi) = pads[k];
                let s = signed_mode_axis_src(mode, coord, shape[dim], lo, hi);
                src_lin += s * src_stride;
                src_stride *= shape[dim];
            }
            out[out_base + flat] = data[in_base + src_lin];
        }
    }

    Ok((out, new_shape))
}

/// Backward for the signed reflect/circular pad: the adjoint of the unified
/// gather is a scatter-add into the original-size input
/// (`grad_input[src(o)] += grad_output[o]`), matching torch's
/// `reflection_pad*_backward` / `_pad_circular` backward.
#[derive(Debug)]
struct PadNdSignedModeBackward<T: Float> {
    input: Tensor<T>,
    input_shape: Vec<usize>,
    mode: PaddingMode,
    /// `(lo, hi)` per padded axis, ordered LAST axis first (same as the forward).
    pads: Vec<(isize, isize)>,
}

impl<T: Float> GradFn<T> for PadNdSignedModeBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        let ndim = self.input_shape.len();
        let npad = self.pads.len();
        let first_padded = ndim - npad;
        let outer: usize = self.input_shape[..first_padded]
            .iter()
            .copied()
            .product::<usize>()
            .max(1);
        let in_inner: usize = self.input_shape[first_padded..].iter().product();

        let go_shape = grad_output.shape();
        let out_inner: usize = go_shape[first_padded..].iter().product();

        let go = grad_output.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let mut grad_in = vec![zero; outer * in_inner];

        for o in 0..outer {
            let in_base = o * in_inner;
            let out_base = o * out_inner;
            for flat in 0..out_inner {
                let mut rem = flat;
                let mut src_lin = 0usize;
                let mut src_stride = 1usize;
                for k in 0..npad {
                    let dim = ndim - 1 - k;
                    let axis_new = go_shape[dim];
                    let coord = rem % axis_new;
                    rem /= axis_new;
                    let (lo, hi) = self.pads[k];
                    let s = signed_mode_axis_src(self.mode, coord, self.input_shape[dim], lo, hi);
                    src_lin += s * src_stride;
                    src_stride *= self.input_shape[dim];
                }
                grad_in[in_base + src_lin] += go[out_base + flat];
            }
        }

        let grad_input =
            Tensor::from_storage(TensorStorage::cpu(grad_in), self.input_shape.clone(), false)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "PadNdSignedModeBackward"
    }
}

/// Shared signed-pad driver for the 1-D/2-D/3-D public entrypoints. `pads` is
/// ordered LAST padded axis first.
///
/// For `PaddingMode::Zeros` (torch `mode="constant"`) negative pads narrow via
/// the signed-constant gather below. For reflect/replicate/circular, live torch
/// 2.11 does NOT reject a negative pad — `_pad_enum` dispatches straight to the
/// native `reflection_pad*` / `replication_pad*` / `_pad_circular` kernels,
/// which compute `output = input + pad_l + pad_r` directly (a negative pad
/// narrows the side) and offset the gather window by `max(0,-pad) - max(0,pad)`
/// (`aten/src/ATen/native/ReflectionPad.cpp:46`,
/// `aten/src/ATen/native/cpu/PaddingKernel.cpp:63-65`,
/// `aten/src/ATen/native/PadNd.cpp:158-159`). That is byte-identical to first
/// CROPPING the negative side(s) (constant-mode narrow) and then applying the
/// positive pad part with the mode's gather — verified against the live oracle
/// (`reflect [-1,0]` on `[1,2,3,4,5]` -> `[2,3,4,5]`; `replicate [1,-1]` ->
/// `[1,1,2,3,4]` grad `[2,1,1,1,0]`; `circular [-1,0]` -> `[2,3,4,5]` grad
/// `[0,1,1,1,1]`; `reflect2d [-1,1,0,0]` on the 3x3 -> `[[2,3,2],[5,6,5],
/// [8,9,8]]`). We compose crop-then-pad so the backward chains the crop adjoint
/// (zero-pad of the cropped side) with the mode-pad adjoint (the gather
/// scatter-add) through the normal autograd graph. Over-cropping a side
/// (`crop >= dim`) still errors via the signed-constant `narrow` check, matching
/// torch (`PadNd.cpp:221-242`).
fn functional_pad_nd_signed<T: Float>(
    input: &Tensor<T>,
    pads: &[(isize, isize)],
    mode: PaddingMode,
    value: T,
) -> FerrotorchResult<Tensor<T>> {
    let has_negative = pads.iter().any(|&(lo, hi)| lo < 0 || hi < 0);

    if mode != PaddingMode::Zeros {
        if !has_negative {
            // All-non-negative under a non-constant mode: pure mode-pad.
            return functional_pad_nd_positive(input, pads, mode, value);
        }
        // Reflect/Replicate/Circular with a negative (crop) pad: torch does NOT
        // crop first. It reflects/clamps/wraps against the ORIGINAL input window
        // via a single index map with offset `max(0,-pad) - max(0,pad)`
        // (`aten/src/ATen/native/cpu/PaddingKernel.cpp:63-95`,
        // `ReflectionPad.cpp:46-48`, `PadNd.cpp:158-159`). A positive pad on a
        // cropped side reads elements a crop-first pass would have discarded
        // (e.g. `reflect [-3,2]` on `[1,2,3,4]` -> `[4,3,2]`, not an error).
        //
        // Replicate in particular MUST use the original-window clamp rather than
        // crop-then-pad: when a crop reduces an axis to size 0, the crop-first
        // path fed a zero-size axis to `pad_*_replicate`, which computed
        // `inner - 1` / `h - 1` and PANICKED (subtract-overflow). torch's
        // `ReplicationPad::index` (`PaddingKernel.cpp:84-95`) clamps the gather
        // to `[pad, size+pad-1]` against the ORIGINAL window, so an over-crop
        // still reads the preserved boundary element — no underflow, no panic
        // (#1625, R-CODE-2). We gather directly from the original window and
        // scatter-add the adjoint through `PadNdSignedModeBackward` (#1620 #1621
        // #1625).
        let data = input.data_vec()?;
        let shape = input.shape();
        if pads.len() > shape.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "pad targets {} dims but input has only {} dims",
                    pads.len(),
                    shape.len()
                ),
            });
        }
        let input_shape = shape.to_vec();
        let (out_data, new_shape) = pad_nd_signed_reflect_circular(&data, shape, pads, mode)?;
        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(PadNdSignedModeBackward {
                input: input.clone(),
                input_shape,
                mode,
                pads: pads.to_vec(),
            });
            return Tensor::from_operation(TensorStorage::cpu(out_data), new_shape, grad_fn);
        }
        return Tensor::from_storage(TensorStorage::cpu(out_data), new_shape, false);
    }

    let data = input.data_vec()?;
    let shape = input.shape();
    if pads.len() > shape.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "pad targets {} dims but input has only {} dims",
                pads.len(),
                shape.len()
            ),
        });
    }
    let input_shape = shape.to_vec();
    let (out_data, new_shape) = pad_nd_signed_constant(&data, shape, pads, value)?;

    // Grad path: attach PadNdSignedBackward so autograd stays connected (same
    // #1550 bug class the positive-only paths fixed).
    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(PadNdSignedBackward {
            input: input.clone(),
            input_shape,
            pads: pads.to_vec(),
        });
        return Tensor::from_operation(TensorStorage::cpu(out_data), new_shape, grad_fn);
    }

    Tensor::from_storage(TensorStorage::cpu(out_data), new_shape, false)
}

/// Apply crop-capable padding to the last dimension of a tensor. Unlike
/// [`functional_pad_1d`] (which takes `usize`), the pad amounts are SIGNED: a
/// negative value crops `|pad|` elements off that side, mirroring
/// `torch.nn.functional.pad(input, [left, right], mode="constant", value=...)`
/// with negative `left`/`right` (`aten/src/ATen/native/PadNd.cpp:29-108`).
///
/// Negative (crop) pads are supported under EVERY mode: `Zeros` narrows via the
/// signed-constant gather, while reflect/replicate/circular crop the negative
/// side(s) then apply their gather on the positive part — byte-identical to
/// torch's native kernels, which compute `output = input + pad_l + pad_r`
/// directly (`PadNd.cpp:221-242`). Over-cropping (removing more than the
/// dimension holds) returns `InvalidArgument`, mirroring torch's
/// "narrow(): length must be non-negative".
pub fn functional_pad_1d_signed<T: Float>(
    input: &Tensor<T>,
    pad_left: isize,
    pad_right: isize,
    mode: PaddingMode,
    value: T,
) -> FerrotorchResult<Tensor<T>> {
    functional_pad_nd_signed(input, &[(pad_left, pad_right)], mode, value)
}

/// Crop-capable padding for the last 2 dimensions. Signed analogue of
/// [`functional_pad_2d`]; see [`functional_pad_1d_signed`] for the crop
/// semantics and constant-mode restriction.
pub fn functional_pad_2d_signed<T: Float>(
    input: &Tensor<T>,
    pad_left: isize,
    pad_right: isize,
    pad_top: isize,
    pad_bottom: isize,
    mode: PaddingMode,
    value: T,
) -> FerrotorchResult<Tensor<T>> {
    // `pads` is LAST axis (W: left/right) first, then 2nd-last (H: top/bottom).
    functional_pad_nd_signed(
        input,
        &[(pad_left, pad_right), (pad_top, pad_bottom)],
        mode,
        value,
    )
}

/// Crop-capable padding for the last 3 dimensions. Signed analogue of
/// [`functional_pad_3d`]; see [`functional_pad_1d_signed`] for the crop
/// semantics and constant-mode restriction.
// Public API: matches `torch.nn.functional.pad`'s 3-axis layout
// (left, right, top, bottom, front, back) — 6 signed pad amounts.
#[allow(clippy::too_many_arguments)]
pub fn functional_pad_3d_signed<T: Float>(
    input: &Tensor<T>,
    pad_left: isize,
    pad_right: isize,
    pad_top: isize,
    pad_bottom: isize,
    pad_front: isize,
    pad_back: isize,
    mode: PaddingMode,
    value: T,
) -> FerrotorchResult<Tensor<T>> {
    // LAST axis (W) first, then H, then D (front/back).
    functional_pad_nd_signed(
        input,
        &[
            (pad_left, pad_right),
            (pad_top, pad_bottom),
            (pad_front, pad_back),
        ],
        mode,
        value,
    )
}

// ===========================================================================
// Macro to reduce boilerplate for Module implementations on padding layers
// ===========================================================================

macro_rules! impl_padding_module {
    ($name:ident) => {
        impl<T: Float> Module<T> for $name<T> {
            fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
                self.pad(input)
            }

            fn parameters(&self) -> Vec<&Parameter<T>> {
                vec![]
            }

            fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
                vec![]
            }

            fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
                vec![]
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
    };
}

// ===========================================================================
// ConstantPad1d / ConstantPad2d / ConstantPad3d
// ===========================================================================

/// Pads the last dimension of the input tensor with a constant value.
///
/// # Shape
/// - Input: `[*, L]`
/// - Output: `[*, L + pad_left + pad_right]`
#[derive(Debug)]
pub struct ConstantPad1d<T: Float> {
    /// Padding `(left, right)`.
    pub padding: (usize, usize),
    /// Constant fill value.
    pub value: T,
    training: bool,
}

impl<T: Float> ConstantPad1d<T> {
    pub fn new(padding: (usize, usize), value: T) -> Self {
        Self {
            padding,
            value,
            training: true,
        }
    }

    fn pad(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let data = input.data_vec()?;
        let (out, new_shape) = pad_1d_constant(
            &data,
            input.shape(),
            self.padding.0,
            self.padding.1,
            self.value,
        );
        Tensor::from_storage(TensorStorage::cpu(out), new_shape, false)
    }
}

impl_padding_module!(ConstantPad1d);

/// Pads the last 2 dimensions with a constant value.
///
/// # Shape
/// - Input: `[*, H, W]`
/// - Output: `[*, H + top + bottom, W + left + right]`
#[derive(Debug)]
pub struct ConstantPad2d<T: Float> {
    /// Padding `(left, right, top, bottom)`.
    pub padding: (usize, usize, usize, usize),
    /// Constant fill value.
    pub value: T,
    training: bool,
}

impl<T: Float> ConstantPad2d<T> {
    pub fn new(padding: (usize, usize, usize, usize), value: T) -> Self {
        Self {
            padding,
            value,
            training: true,
        }
    }

    fn pad(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() < 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ConstantPad2d expects at least 2-D input, got {:?}",
                    input.shape()
                ),
            });
        }
        let data = input.data_vec()?;
        let (out, new_shape) = pad_2d_constant(
            &data,
            input.shape(),
            self.padding.0,
            self.padding.1,
            self.padding.2,
            self.padding.3,
            self.value,
        );
        Tensor::from_storage(TensorStorage::cpu(out), new_shape, false)
    }
}

impl_padding_module!(ConstantPad2d);

/// Pads the last 3 dimensions with a constant value.
///
/// # Shape
/// - Input: `[*, D, H, W]`
/// - Output: `[*, D + front + back, H + top + bottom, W + left + right]`
#[derive(Debug)]
pub struct ConstantPad3d<T: Float> {
    /// Padding `(left, right, top, bottom, front, back)`.
    pub padding: (usize, usize, usize, usize, usize, usize),
    /// Constant fill value.
    pub value: T,
    training: bool,
}

impl<T: Float> ConstantPad3d<T> {
    pub fn new(padding: (usize, usize, usize, usize, usize, usize), value: T) -> Self {
        Self {
            padding,
            value,
            training: true,
        }
    }

    fn pad(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() < 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ConstantPad3d expects at least 3-D input, got {:?}",
                    input.shape()
                ),
            });
        }
        let data = input.data_vec()?;
        let (out, new_shape) = pad_3d_constant(
            &data,
            input.shape(),
            self.padding.0,
            self.padding.1,
            self.padding.2,
            self.padding.3,
            self.padding.4,
            self.padding.5,
            self.value,
        );
        Tensor::from_storage(TensorStorage::cpu(out), new_shape, false)
    }
}

impl_padding_module!(ConstantPad3d);

// ===========================================================================
// ZeroPad1d / ZeroPad2d / ZeroPad3d
// ===========================================================================

/// Pads the last dimension with zeros.
#[derive(Debug)]
pub struct ZeroPad1d<T: Float> {
    pub padding: (usize, usize),
    training: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> ZeroPad1d<T> {
    pub fn new(padding: (usize, usize)) -> Self {
        Self {
            padding,
            training: true,
            _phantom: std::marker::PhantomData,
        }
    }

    fn pad(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let data = input.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let (out, new_shape) =
            pad_1d_constant(&data, input.shape(), self.padding.0, self.padding.1, zero);
        Tensor::from_storage(TensorStorage::cpu(out), new_shape, false)
    }
}

impl_padding_module!(ZeroPad1d);

/// Pads the last 2 dimensions with zeros.
#[derive(Debug)]
pub struct ZeroPad2d<T: Float> {
    pub padding: (usize, usize, usize, usize),
    training: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> ZeroPad2d<T> {
    pub fn new(padding: (usize, usize, usize, usize)) -> Self {
        Self {
            padding,
            training: true,
            _phantom: std::marker::PhantomData,
        }
    }

    fn pad(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() < 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ZeroPad2d expects at least 2-D input, got {:?}",
                    input.shape()
                ),
            });
        }
        let data = input.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let (out, new_shape) = pad_2d_constant(
            &data,
            input.shape(),
            self.padding.0,
            self.padding.1,
            self.padding.2,
            self.padding.3,
            zero,
        );
        Tensor::from_storage(TensorStorage::cpu(out), new_shape, false)
    }
}

impl_padding_module!(ZeroPad2d);

/// Pads the last 3 dimensions with zeros.
#[derive(Debug)]
pub struct ZeroPad3d<T: Float> {
    pub padding: (usize, usize, usize, usize, usize, usize),
    training: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> ZeroPad3d<T> {
    pub fn new(padding: (usize, usize, usize, usize, usize, usize)) -> Self {
        Self {
            padding,
            training: true,
            _phantom: std::marker::PhantomData,
        }
    }

    fn pad(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() < 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ZeroPad3d expects at least 3-D input, got {:?}",
                    input.shape()
                ),
            });
        }
        let data = input.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let (out, new_shape) = pad_3d_constant(
            &data,
            input.shape(),
            self.padding.0,
            self.padding.1,
            self.padding.2,
            self.padding.3,
            self.padding.4,
            self.padding.5,
            zero,
        );
        Tensor::from_storage(TensorStorage::cpu(out), new_shape, false)
    }
}

impl_padding_module!(ZeroPad3d);

// ===========================================================================
// ReflectionPad1d / ReflectionPad2d / ReflectionPad3d
// ===========================================================================

/// Pads the last dimension using reflection of the input boundary.
#[derive(Debug)]
pub struct ReflectionPad1d<T: Float> {
    pub padding: (usize, usize),
    training: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> ReflectionPad1d<T> {
    pub fn new(padding: (usize, usize)) -> Self {
        Self {
            padding,
            training: true,
            _phantom: std::marker::PhantomData,
        }
    }

    fn pad(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let data = input.data_vec()?;
        let (out, new_shape) =
            pad_1d_reflect(&data, input.shape(), self.padding.0, self.padding.1)?;
        Tensor::from_storage(TensorStorage::cpu(out), new_shape, false)
    }
}

impl_padding_module!(ReflectionPad1d);

/// Pads the last 2 dimensions using reflection.
#[derive(Debug)]
pub struct ReflectionPad2d<T: Float> {
    pub padding: (usize, usize, usize, usize),
    training: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> ReflectionPad2d<T> {
    pub fn new(padding: (usize, usize, usize, usize)) -> Self {
        Self {
            padding,
            training: true,
            _phantom: std::marker::PhantomData,
        }
    }

    fn pad(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() < 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ReflectionPad2d expects at least 2-D input, got {:?}",
                    input.shape()
                ),
            });
        }
        let data = input.data_vec()?;
        let (out, new_shape) = pad_2d_reflect(
            &data,
            input.shape(),
            self.padding.0,
            self.padding.1,
            self.padding.2,
            self.padding.3,
        )?;
        Tensor::from_storage(TensorStorage::cpu(out), new_shape, false)
    }
}

impl_padding_module!(ReflectionPad2d);

/// Pads the last 3 dimensions using reflection.
#[derive(Debug)]
pub struct ReflectionPad3d<T: Float> {
    pub padding: (usize, usize, usize, usize, usize, usize),
    training: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> ReflectionPad3d<T> {
    pub fn new(padding: (usize, usize, usize, usize, usize, usize)) -> Self {
        Self {
            padding,
            training: true,
            _phantom: std::marker::PhantomData,
        }
    }

    fn pad(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() < 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ReflectionPad3d expects at least 3-D input, got {:?}",
                    input.shape()
                ),
            });
        }
        let data = input.data_vec()?;
        let (out, new_shape) = pad_3d_reflect(
            &data,
            input.shape(),
            self.padding.0,
            self.padding.1,
            self.padding.2,
            self.padding.3,
            self.padding.4,
            self.padding.5,
        )?;
        Tensor::from_storage(TensorStorage::cpu(out), new_shape, false)
    }
}

impl_padding_module!(ReflectionPad3d);

// ===========================================================================
// ReplicationPad1d / ReplicationPad2d / ReplicationPad3d
// ===========================================================================

/// Pads the last dimension by replicating the edge values.
#[derive(Debug)]
pub struct ReplicationPad1d<T: Float> {
    pub padding: (usize, usize),
    training: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> ReplicationPad1d<T> {
    pub fn new(padding: (usize, usize)) -> Self {
        Self {
            padding,
            training: true,
            _phantom: std::marker::PhantomData,
        }
    }

    fn pad(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let data = input.data_vec()?;
        let (out, new_shape) =
            pad_1d_replicate(&data, input.shape(), self.padding.0, self.padding.1);
        Tensor::from_storage(TensorStorage::cpu(out), new_shape, false)
    }
}

impl_padding_module!(ReplicationPad1d);

/// Pads the last 2 dimensions by replicating edge values.
#[derive(Debug)]
pub struct ReplicationPad2d<T: Float> {
    pub padding: (usize, usize, usize, usize),
    training: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> ReplicationPad2d<T> {
    pub fn new(padding: (usize, usize, usize, usize)) -> Self {
        Self {
            padding,
            training: true,
            _phantom: std::marker::PhantomData,
        }
    }

    fn pad(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() < 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ReplicationPad2d expects at least 2-D input, got {:?}",
                    input.shape()
                ),
            });
        }
        let data = input.data_vec()?;
        let (out, new_shape) = pad_2d_replicate(
            &data,
            input.shape(),
            self.padding.0,
            self.padding.1,
            self.padding.2,
            self.padding.3,
        );
        Tensor::from_storage(TensorStorage::cpu(out), new_shape, false)
    }
}

impl_padding_module!(ReplicationPad2d);

/// Pads the last 3 dimensions by replicating edge values.
#[derive(Debug)]
pub struct ReplicationPad3d<T: Float> {
    pub padding: (usize, usize, usize, usize, usize, usize),
    training: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> ReplicationPad3d<T> {
    pub fn new(padding: (usize, usize, usize, usize, usize, usize)) -> Self {
        Self {
            padding,
            training: true,
            _phantom: std::marker::PhantomData,
        }
    }

    fn pad(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() < 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ReplicationPad3d expects at least 3-D input, got {:?}",
                    input.shape()
                ),
            });
        }
        let data = input.data_vec()?;
        let (out, new_shape) = pad_3d_replicate(
            &data,
            input.shape(),
            self.padding.0,
            self.padding.1,
            self.padding.2,
            self.padding.3,
            self.padding.4,
            self.padding.5,
        );
        Tensor::from_storage(TensorStorage::cpu(out), new_shape, false)
    }
}

impl_padding_module!(ReplicationPad3d);

// ===========================================================================
// CircularPad — wraps data circularly (periodic boundary conditions)
// ===========================================================================

/// 1-D circular padding: wraps the input circularly.
///
/// Input: [N, C, W]. Pads the W dimension with circular (periodic) values.
/// Matches PyTorch's `nn.CircularPad1d`.
#[derive(Debug, Clone)]
pub struct CircularPad1d<T: Float> {
    pub padding: (usize, usize),
    training: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> CircularPad1d<T> {
    pub fn new(padding: (usize, usize)) -> Self {
        Self {
            padding,
            training: true,
            _phantom: std::marker::PhantomData,
        }
    }

    fn pad(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "CircularPad1d: expected 3-D input [N,C,W], got {:?}",
                    input.shape()
                ),
            });
        }
        if input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "CircularPad1d",
            });
        }
        let shape = input.shape();
        let (n, c, w) = (shape[0], shape[1], shape[2]);
        let (pl, pr) = self.padding;
        let new_w = w + pl + pr;
        let data = input.data()?;
        let zero = <T as num_traits::Zero>::zero();
        let mut out = vec![zero; n * c * new_w];

        for batch in 0..n {
            for ch in 0..c {
                for ow in 0..new_w {
                    let iw = ((ow as isize - pl as isize).rem_euclid(w as isize)) as usize;
                    out[batch * c * new_w + ch * new_w + ow] = data[batch * c * w + ch * w + iw];
                }
            }
        }

        Tensor::from_storage(TensorStorage::cpu(out), vec![n, c, new_w], false)
    }
}

impl<T: Float> Default for CircularPad1d<T> {
    fn default() -> Self {
        Self::new((0, 0))
    }
}

impl_padding_module!(CircularPad1d);

/// 2-D circular padding. Input: [N, C, H, W].
/// Matches PyTorch's `nn.CircularPad2d`.
#[derive(Debug, Clone)]
pub struct CircularPad2d<T: Float> {
    pub padding: (usize, usize, usize, usize),
    training: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> CircularPad2d<T> {
    pub fn new(padding: (usize, usize, usize, usize)) -> Self {
        Self {
            padding,
            training: true,
            _phantom: std::marker::PhantomData,
        }
    }

    fn pad(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() != 4 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "CircularPad2d: expected 4-D input [N,C,H,W], got {:?}",
                    input.shape()
                ),
            });
        }
        if input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "CircularPad2d",
            });
        }
        let shape = input.shape();
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (pl, pr, pt, pb) = self.padding;
        let new_h = h + pt + pb;
        let new_w = w + pl + pr;
        let data = input.data()?;
        let zero = <T as num_traits::Zero>::zero();
        let mut out = vec![zero; n * c * new_h * new_w];

        for batch in 0..n {
            for ch in 0..c {
                for oh in 0..new_h {
                    let ih = ((oh as isize - pt as isize).rem_euclid(h as isize)) as usize;
                    for ow in 0..new_w {
                        let iw = ((ow as isize - pl as isize).rem_euclid(w as isize)) as usize;
                        out[batch * c * new_h * new_w + ch * new_h * new_w + oh * new_w + ow] =
                            data[batch * c * h * w + ch * h * w + ih * w + iw];
                    }
                }
            }
        }

        Tensor::from_storage(TensorStorage::cpu(out), vec![n, c, new_h, new_w], false)
    }
}

impl<T: Float> Default for CircularPad2d<T> {
    fn default() -> Self {
        Self::new((0, 0, 0, 0))
    }
}

impl_padding_module!(CircularPad2d);

/// 3-D circular padding. Input: [N, C, D, H, W].
/// Matches PyTorch's `nn.CircularPad3d`.
#[derive(Debug, Clone)]
pub struct CircularPad3d<T: Float> {
    pub padding: (usize, usize, usize, usize, usize, usize),
    training: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> CircularPad3d<T> {
    pub fn new(padding: (usize, usize, usize, usize, usize, usize)) -> Self {
        Self {
            padding,
            training: true,
            _phantom: std::marker::PhantomData,
        }
    }

    fn pad(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() != 5 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "CircularPad3d: expected 5-D input [N,C,D,H,W], got {:?}",
                    input.shape()
                ),
            });
        }
        if input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "CircularPad3d",
            });
        }
        let shape = input.shape();
        let (n, c, d, h, w) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
        let (pl, pr, pt, pb, pf, pk) = self.padding;
        let (new_d, new_h, new_w) = (d + pf + pk, h + pt + pb, w + pl + pr);
        let data = input.data()?;
        let zero = <T as num_traits::Zero>::zero();
        let mut out = vec![zero; n * c * new_d * new_h * new_w];

        for batch in 0..n {
            for ch in 0..c {
                for od in 0..new_d {
                    let id = ((od as isize - pf as isize).rem_euclid(d as isize)) as usize;
                    for oh in 0..new_h {
                        let ih = ((oh as isize - pt as isize).rem_euclid(h as isize)) as usize;
                        for ow in 0..new_w {
                            let iw = ((ow as isize - pl as isize).rem_euclid(w as isize)) as usize;
                            out[batch * c * new_d * new_h * new_w
                                + ch * new_d * new_h * new_w
                                + od * new_h * new_w
                                + oh * new_w
                                + ow] = data
                                [batch * c * d * h * w + ch * d * h * w + id * h * w + ih * w + iw];
                        }
                    }
                }
            }
        }

        Tensor::from_storage(
            TensorStorage::cpu(out),
            vec![n, c, new_d, new_h, new_w],
            false,
        )
    }
}

impl<T: Float> Default for CircularPad3d<T> {
    fn default() -> Self {
        Self::new((0, 0, 0, 0, 0, 0))
    }
}

impl_padding_module!(CircularPad3d);

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::Module;

    fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "length mismatch: {} vs {}",
            actual.len(),
            expected.len()
        );
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < tol, "index {i}: actual={a} expected={e}");
        }
    }

    // -----------------------------------------------------------------------
    // ConstantPad1d
    // -----------------------------------------------------------------------

    #[test]
    fn test_constant_pad1d_basic() {
        let pad = ConstantPad1d::<f32>::new((2, 3), 9.0);
        let input = t(&[1.0, 2.0, 3.0], &[1, 1, 3]);
        let output = pad.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 8]);
        assert_close(
            output.data().unwrap(),
            &[9.0, 9.0, 1.0, 2.0, 3.0, 9.0, 9.0, 9.0],
            1e-7,
        );
    }

    // -----------------------------------------------------------------------
    // ZeroPad1d
    // -----------------------------------------------------------------------

    #[test]
    fn test_zero_pad1d() {
        let pad = ZeroPad1d::<f32>::new((1, 2));
        let input = t(&[1.0, 2.0, 3.0], &[3]);
        let output = pad.forward(&input).unwrap();
        assert_eq!(output.shape(), &[6]);
        assert_close(
            output.data().unwrap(),
            &[0.0, 1.0, 2.0, 3.0, 0.0, 0.0],
            1e-7,
        );
    }

    // -----------------------------------------------------------------------
    // ZeroPad2d
    // -----------------------------------------------------------------------

    #[test]
    fn test_zero_pad2d() {
        let pad = ZeroPad2d::<f32>::new((1, 1, 1, 1));
        let input = t(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let output = pad.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 4, 4]);
        #[rustfmt::skip]
        let expected = [
            0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 2.0, 0.0,
            0.0, 3.0, 4.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        assert_close(output.data().unwrap(), &expected, 1e-7);
    }

    // -----------------------------------------------------------------------
    // ZeroPad3d
    // -----------------------------------------------------------------------

    #[test]
    fn test_zero_pad3d_shape() {
        let pad = ZeroPad3d::<f32>::new((1, 1, 1, 1, 1, 1));
        let input = t(&[1.0; 2 * 2 * 2], &[1, 1, 2, 2, 2]);
        let output = pad.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 4, 4, 4]);
    }

    // -----------------------------------------------------------------------
    // ReflectionPad1d
    // -----------------------------------------------------------------------

    #[test]
    fn test_reflection_pad1d() {
        let pad = ReflectionPad1d::<f32>::new((2, 2));
        // input = [1, 2, 3, 4]
        let input = t(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let output = pad.forward(&input).unwrap();
        assert_eq!(output.shape(), &[8]);
        // Reflect left: [3, 2, | 1, 2, 3, 4 | 3, 2]
        assert_close(
            output.data().unwrap(),
            &[3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0],
            1e-7,
        );
    }

    #[test]
    fn test_reflection_pad1d_too_large() {
        let pad = ReflectionPad1d::<f32>::new((4, 0));
        let input = t(&[1.0, 2.0, 3.0], &[3]); // size 3, pad 4 >= 3
        assert!(pad.forward(&input).is_err());
    }

    // -----------------------------------------------------------------------
    // ReflectionPad2d
    // -----------------------------------------------------------------------

    #[test]
    fn test_reflection_pad2d() {
        let pad = ReflectionPad2d::<f32>::new((1, 1, 1, 1));
        #[rustfmt::skip]
        let input = t(&[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ], &[1, 1, 3, 3]);
        let output = pad.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 5, 5]);
        // Corner (0,0) should reflect to (1,1) in src = 5.0
        let out = output.data().unwrap();
        assert_close(&out[0..1], &[5.0], 1e-7); // top-left corner
    }

    // -----------------------------------------------------------------------
    // ReplicationPad1d
    // -----------------------------------------------------------------------

    #[test]
    fn test_replication_pad1d() {
        let pad = ReplicationPad1d::<f32>::new((2, 3));
        let input = t(&[1.0, 2.0, 3.0], &[3]);
        let output = pad.forward(&input).unwrap();
        assert_eq!(output.shape(), &[8]);
        assert_close(
            output.data().unwrap(),
            &[1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0],
            1e-7,
        );
    }

    // -----------------------------------------------------------------------
    // ReplicationPad2d
    // -----------------------------------------------------------------------

    #[test]
    fn test_replication_pad2d() {
        let pad = ReplicationPad2d::<f32>::new((1, 1, 1, 1));
        #[rustfmt::skip]
        let input = t(&[
            1.0, 2.0,
            3.0, 4.0,
        ], &[1, 1, 2, 2]);
        let output = pad.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 4, 4]);
        #[rustfmt::skip]
        let expected = [
            1.0, 1.0, 2.0, 2.0,
            1.0, 1.0, 2.0, 2.0,
            3.0, 3.0, 4.0, 4.0,
            3.0, 3.0, 4.0, 4.0,
        ];
        assert_close(output.data().unwrap(), &expected, 1e-7);
    }

    // -----------------------------------------------------------------------
    // ConstantPad2d
    // -----------------------------------------------------------------------

    #[test]
    fn test_constant_pad2d() {
        let pad = ConstantPad2d::<f32>::new((1, 1, 1, 1), -1.0);
        let input = t(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let output = pad.forward(&input).unwrap();
        assert_eq!(output.shape(), &[4, 4]);
        #[rustfmt::skip]
        let expected = [
            -1.0, -1.0, -1.0, -1.0,
            -1.0, 5.0, 6.0, -1.0,
            -1.0, 7.0, 8.0, -1.0,
            -1.0, -1.0, -1.0, -1.0,
        ];
        assert_close(output.data().unwrap(), &expected, 1e-7);
    }

    // -----------------------------------------------------------------------
    // ConstantPad3d
    // -----------------------------------------------------------------------

    #[test]
    fn test_constant_pad3d_shape() {
        let pad = ConstantPad3d::<f32>::new((1, 2, 1, 2, 1, 2), 0.0);
        let input = t(&vec![1.0; 3 * 4 * 5], &[1, 1, 3, 4, 5]);
        let output = pad.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 6, 7, 8]);
    }

    // -----------------------------------------------------------------------
    // Circular padding (1D)
    // -----------------------------------------------------------------------

    #[test]
    fn test_circular_pad_1d() {
        // input = [1, 2, 3, 4], pad_left=1, pad_right=2
        // circular: [4, 1, 2, 3, 4, 1, 2]
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let (out, new_shape) = pad_1d_circular(&data, &[4], 1, 2);
        assert_eq!(new_shape, &[7]);
        assert_close(&out, &[4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0], 1e-7);
    }

    // -----------------------------------------------------------------------
    // Padding mode enum
    // -----------------------------------------------------------------------

    #[test]
    fn test_padding_mode_eq() {
        assert_eq!(PaddingMode::Zeros, PaddingMode::Zeros);
        assert_ne!(PaddingMode::Zeros, PaddingMode::Reflect);
    }

    // -----------------------------------------------------------------------
    // Module trait: no parameters
    // -----------------------------------------------------------------------

    #[test]
    fn test_padding_module_no_params() {
        let pad = ZeroPad2d::<f32>::new((1, 1, 1, 1));
        assert!(pad.parameters().is_empty());
        assert!(pad.named_parameters().is_empty());
    }

    #[test]
    fn test_padding_module_train_eval() {
        let mut pad = ReflectionPad1d::<f32>::new((1, 1));
        assert!(pad.is_training());
        pad.eval();
        assert!(!pad.is_training());
        pad.train();
        assert!(pad.is_training());
    }

    // -----------------------------------------------------------------------
    // Degenerate (numel-0) constant pad — regression for #1551.
    //
    // op_db emits pad samples whose input has an empty data buffer paired
    // with a non-empty *declared* last dim (e.g. shape `[0, 3]`: numel 0,
    // inner 3). Previously `pad_{1,2,3}d_constant` forced rows/outer to 1 and
    // then read `inner`/`w` elements from the empty `data` slice, panicking
    // with "range end index N out of range for slice of length 0" at the
    // `copy_from_slice`. Upstream `torch.nn.functional.pad`
    // (`aten/src/ATen/native/PadNd.cpp:94-106`) allocates the padded output,
    // `fill_(value)`s it, then `copy_`s the (empty) source — a no-op — so the
    // result is the correctly-shaped, value-filled tensor. These assert the
    // fixed behaviour: no panic + correct output shape on numel-0 input.
    // -----------------------------------------------------------------------

    #[test]
    fn test_constant_pad1d_empty_numel_no_panic() {
        // shape [0, 3]: numel 0 but inner = 3. data buffer is empty.
        let (out, new_shape) = pad_1d_constant::<f32>(&[], &[0, 3], 2, 3, 7.0);
        // last dim padded 3 -> 3+2+3 = 8; outer 0-dim with forced row count 1.
        assert_eq!(new_shape, vec![0, 8]);
        // value-filled output, no source copied in.
        assert!(out.iter().all(|&v| v == 7.0));
    }

    #[test]
    fn test_constant_pad2d_empty_numel_no_panic() {
        // shape [0, 2, 3]: numel 0, h = 2, w = 3, empty data.
        let (out, new_shape) = pad_2d_constant::<f32>(&[], &[0, 2, 3], 1, 1, 1, 1, 5.0);
        assert_eq!(new_shape, vec![0, 4, 5]);
        assert!(out.iter().all(|&v| v == 5.0));
    }

    #[test]
    fn test_constant_pad3d_empty_numel_no_panic() {
        // shape [0, 2, 2, 3]: numel 0, d = 2, h = 2, w = 3, empty data.
        let (out, new_shape) = pad_3d_constant::<f32>(&[], &[0, 2, 2, 3], 1, 1, 1, 1, 1, 1, 3.0);
        assert_eq!(new_shape, vec![0, 4, 4, 5]);
        assert!(out.iter().all(|&v| v == 3.0));
    }

    // -----------------------------------------------------------------------
    // Regression: `functional_pad_{1,2,3}d` constant-mode must use `value`.
    //
    // The runner maps torch `mode="constant"` -> `PaddingMode::Zeros` and passes
    // the `value` kwarg through. Pre-fix the `Zeros` arm hardcoded `T::zero()`
    // and dropped `value` (`let _ = value;`), so `F.pad(x, p, "constant", 2.0)`
    // filled 0 instead of 2 — 256 parity-sweep failures (ferrotorch=0 vs
    // torch=2). Upstream `aten/src/ATen/native/PadNd.cpp:94` does
    // `output.fill_(value)` before copying the source. #1553.
    // -----------------------------------------------------------------------

    #[test]
    fn test_functional_pad_1d_constant_uses_value() {
        let input = t(&[1.0, 2.0, 3.0], &[1, 1, 3]);
        let out = functional_pad_1d(&input, 1, 1, PaddingMode::Zeros, 2.0).unwrap();
        assert_eq!(out.shape(), &[1, 1, 5]);
        // Padded region (first + last) must be the fill `value` 2.0, not 0.0.
        assert_close(out.data().unwrap(), &[2.0, 1.0, 2.0, 3.0, 2.0], 1e-7);
    }

    #[test]
    fn test_functional_pad_2d_constant_uses_value() {
        // 1x1x2x2 input, pad (left, right, top, bottom) = (1, 1, 1, 1).
        let input = t(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let out = functional_pad_2d(&input, 1, 1, 1, 1, PaddingMode::Zeros, 2.0).unwrap();
        assert_eq!(out.shape(), &[1, 1, 4, 4]);
        #[rustfmt::skip]
        let expected = [
            2.0, 2.0, 2.0, 2.0,
            2.0, 1.0, 2.0, 2.0,
            2.0, 3.0, 4.0, 2.0,
            2.0, 2.0, 2.0, 2.0,
        ];
        assert_close(out.data().unwrap(), &expected, 1e-7);
        // The border is the fill value; no padded cell is 0.
        assert!(out.data().unwrap().iter().all(|&v| v != 0.0));
    }

    #[test]
    fn test_functional_pad_3d_constant_uses_value() {
        // 1x1x1x1x1 input, pad all six axes by 0 except left/right by 1.
        let input = t(&[5.0], &[1, 1, 1, 1, 1]);
        let out = functional_pad_3d(&input, 1, 1, 0, 0, 0, 0, PaddingMode::Zeros, 2.0).unwrap();
        assert_eq!(out.shape(), &[1, 1, 1, 1, 3]);
        assert_close(out.data().unwrap(), &[2.0, 5.0, 2.0], 1e-7);
    }

    // -----------------------------------------------------------------------
    // Autograd-aware functional pad (Pad1dBackward / Pad3dBackward) — #1443.
    //
    // These are the pre-pad helpers Conv1d/Conv3d route non-zero padding_modes
    // through; a pad returning requires_grad=false severs autograd (the #1550
    // bug class the 2-D path already fixed). Expected gradients are from a live
    // PyTorch 2.11 `F.pad(...).sum().backward()` oracle (R-CHAR-3); the oracle
    // script is in the #1443 commit body.
    // -----------------------------------------------------------------------

    /// Helper: leaf tensor that requires grad.
    fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
    }

    /// `functional_pad_1d` Reflect attaches `Pad1dBackward` and scatter-adds the
    /// grad back onto the source row. torch: F.pad([1,2,3,4], (2,2), 'reflect')
    /// -> out [3,2,1,2,3,4,3,2]; sum().backward() grad_input = [1,3,3,1].
    #[test]
    fn test_functional_pad_1d_reflect_backward_matches_torch() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let y = functional_pad_1d(&x, 2, 2, PaddingMode::Reflect, 0.0).unwrap();
        assert_eq!(y.shape(), &[1, 1, 8]);
        assert!(
            y.grad_fn().is_some(),
            "functional_pad_1d Reflect lost grad_fn — would sever Conv1d autograd (#1550 class)"
        );
        assert_eq!(y.grad_fn().unwrap().name(), "Pad1dBackward");
        let sum = ferrotorch_core::grad_fns::reduction::sum(&y).unwrap();
        ferrotorch_core::backward(&sum).unwrap();
        let g = x.grad().unwrap().expect("grad must be populated");
        assert_close(g.data().unwrap(), &[1.0, 3.0, 3.0, 1.0], 1e-5);
    }

    /// `functional_pad_3d` Circular attaches `Pad3dBackward`. torch: a circular
    /// pad of (1,1,1,1,1,1) on a 2x2x2 volume wraps every cell exactly 8 times,
    /// so the all-ones grad_output backprops to a uniform grad of 8.
    #[test]
    fn test_functional_pad_3d_circular_backward_matches_torch() {
        let x_data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        let x = leaf(&x_data, &[1, 1, 2, 2, 2]);
        let y = functional_pad_3d(&x, 1, 1, 1, 1, 1, 1, PaddingMode::Circular, 0.0).unwrap();
        assert_eq!(y.shape(), &[1, 1, 4, 4, 4]);
        assert!(y.grad_fn().is_some());
        assert_eq!(y.grad_fn().unwrap().name(), "Pad3dBackward");
        let sum = ferrotorch_core::grad_fns::reduction::sum(&y).unwrap();
        ferrotorch_core::backward(&sum).unwrap();
        let g = x.grad().unwrap().expect("grad must be populated");
        assert_close(g.data().unwrap(), &[8.0; 8], 1e-5);
    }

    // -----------------------------------------------------------------------
    // Negative (crop) padding — `torch.nn.functional.pad` with negative pad
    // amounts CROPS that side instead of adding. Only the constant
    // (`PaddingMode::Zeros`) path supports it; upstream
    // `aten/src/ATen/native/PadNd.cpp:29-108` (`constant_pad_nd`) narrows the
    // input for negative pads, fills the output with `value`, and copies the
    // cropped input into the positive-pad window. Reflect/replicate/circular
    // reject negative pads (PadNd.cpp:221-242). #1611.
    //
    // All expected forward + backward (sum().backward()) values below are from
    // a live PyTorch 2.11 oracle (R-CHAR-3); the deriving script is in the
    // #1611 commit body. Each block names the exact `F.pad(...)` call it pins.
    // -----------------------------------------------------------------------

    /// torch: `F.pad(torch.tensor([[[1,2,3,4,5]]]), [-1,-1], "constant")`
    /// -> out [2,3,4]; sum().backward() grad_input = [0,1,1,1,0].
    #[test]
    fn test_functional_pad_1d_signed_crop_both_matches_torch() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5]);
        let y = functional_pad_1d_signed(&x, -1, -1, PaddingMode::Zeros, 0.0).unwrap();
        assert_eq!(y.shape(), &[1, 1, 3]);
        assert_close(y.data().unwrap(), &[2.0, 3.0, 4.0], 1e-7);
        assert_eq!(y.grad_fn().unwrap().name(), "PadNdSignedBackward");
        let sum = ferrotorch_core::grad_fns::reduction::sum(&y).unwrap();
        ferrotorch_core::backward(&sum).unwrap();
        let g = x.grad().unwrap().expect("grad must be populated");
        assert_close(g.data().unwrap(), &[0.0, 1.0, 1.0, 1.0, 0.0], 1e-7);
    }

    /// Mixed signs: torch
    /// `F.pad(torch.tensor([[[1,2,3,4]]]), [-1,2], "constant", value=9)`
    /// -> out [2,3,4,9,9] (crop 1 from start, add 2 fill at end);
    /// sum().backward() grad_input = [0,1,1,1].
    #[test]
    fn test_functional_pad_1d_signed_mixed_matches_torch() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let y = functional_pad_1d_signed(&x, -1, 2, PaddingMode::Zeros, 9.0).unwrap();
        assert_eq!(y.shape(), &[1, 1, 5]);
        assert_close(y.data().unwrap(), &[2.0, 3.0, 4.0, 9.0, 9.0], 1e-7);
        let sum = ferrotorch_core::grad_fns::reduction::sum(&y).unwrap();
        ferrotorch_core::backward(&sum).unwrap();
        let g = x.grad().unwrap().expect("grad must be populated");
        assert_close(g.data().unwrap(), &[0.0, 1.0, 1.0, 1.0], 1e-7);
    }

    /// 2-D crop: torch `F.pad(3x3, [-1,0, 0,-1], "constant")` crops the right
    /// column (last dim) and the bottom row (2nd-last) -> 2x2 [[2,3],[5,6]];
    /// sum().backward() grad = [[0,1,1],[0,1,1],[0,0,0]] (flattened).
    #[test]
    fn test_functional_pad_2d_signed_crop_matches_torch() {
        #[rustfmt::skip]
        let x = leaf(&[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ], &[1, 1, 3, 3]);
        let y = functional_pad_2d_signed(&x, -1, 0, 0, -1, PaddingMode::Zeros, 0.0).unwrap();
        assert_eq!(y.shape(), &[1, 1, 2, 2]);
        assert_close(y.data().unwrap(), &[2.0, 3.0, 5.0, 6.0], 1e-7);
        let sum = ferrotorch_core::grad_fns::reduction::sum(&y).unwrap();
        ferrotorch_core::backward(&sum).unwrap();
        let g = x.grad().unwrap().expect("grad must be populated");
        assert_close(
            g.data().unwrap(),
            &[0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            1e-7,
        );
    }

    /// 2-D mixed signs: torch
    /// `F.pad(2x3, [-1,2, 1,-1], "constant", value=7)` (last dim crop1/add2,
    /// 2nd-last add1/crop1) -> 2x4 [[7,7,7,7],[2,3,7,7]];
    /// sum().backward() grad = [[0,1,1],[0,0,0]] (flattened).
    #[test]
    fn test_functional_pad_2d_signed_mixed_matches_torch() {
        #[rustfmt::skip]
        let x = leaf(&[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ], &[1, 1, 2, 3]);
        let y = functional_pad_2d_signed(&x, -1, 2, 1, -1, PaddingMode::Zeros, 7.0).unwrap();
        assert_eq!(y.shape(), &[1, 1, 2, 4]);
        #[rustfmt::skip]
        let expected = [
            7.0, 7.0, 7.0, 7.0,
            2.0, 3.0, 7.0, 7.0,
        ];
        assert_close(y.data().unwrap(), &expected, 1e-7);
        let sum = ferrotorch_core::grad_fns::reduction::sum(&y).unwrap();
        ferrotorch_core::backward(&sum).unwrap();
        let g = x.grad().unwrap().expect("grad must be populated");
        assert_close(g.data().unwrap(), &[0.0, 1.0, 1.0, 0.0, 0.0, 0.0], 1e-7);
    }

    /// 3-D crop: torch `F.pad(2x2x2 [1..8], [-1,0, 0,-1, -1,0], "constant")`
    /// (W crop right, H crop bottom, D crop front) -> 1x1x1 [6];
    /// sum().backward() grad = [0,0,0,0,0,1,0,0].
    #[test]
    fn test_functional_pad_3d_signed_crop_matches_torch() {
        let x_data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        let x = leaf(&x_data, &[1, 1, 2, 2, 2]);
        let y = functional_pad_3d_signed(&x, -1, 0, 0, -1, -1, 0, PaddingMode::Zeros, 0.0).unwrap();
        assert_eq!(y.shape(), &[1, 1, 1, 1, 1]);
        assert_close(y.data().unwrap(), &[6.0], 1e-7);
        let sum = ferrotorch_core::grad_fns::reduction::sum(&y).unwrap();
        ferrotorch_core::backward(&sum).unwrap();
        let g = x.grad().unwrap().expect("grad must be populated");
        assert_close(
            g.data().unwrap(),
            &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            1e-7,
        );
    }

    /// 3-D mixed signs incl. positive adds: torch
    /// `F.pad(2x2x2 [1..8], [1,-1, 0,1, -1,2], "constant", value=3)`
    /// -> 3x3x2; sum().backward() grad = [0,0,0,0,1,0,1,0].
    #[test]
    fn test_functional_pad_3d_signed_mixed_matches_torch() {
        let x_data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        let x = leaf(&x_data, &[1, 1, 2, 2, 2]);
        let y = functional_pad_3d_signed(&x, 1, -1, 0, 1, -1, 2, PaddingMode::Zeros, 3.0).unwrap();
        assert_eq!(y.shape(), &[1, 1, 3, 3, 2]);
        #[rustfmt::skip]
        let expected = [
            3.0, 5.0, 3.0, 7.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
        ];
        assert_close(y.data().unwrap(), &expected, 1e-7);
        let sum = ferrotorch_core::grad_fns::reduction::sum(&y).unwrap();
        ferrotorch_core::backward(&sum).unwrap();
        let g = x.grad().unwrap().expect("grad must be populated");
        assert_close(
            g.data().unwrap(),
            &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            1e-7,
        );
    }

    /// Over-crop: torch raises `RuntimeError: narrow(): length must be
    /// non-negative` when a single side crops more than the dim holds
    /// (`F.pad([[[1,2,3]]], [-4,0])`) or the combined net size is negative
    /// (`F.pad([[[1,2,3]]], [-2,-2])`). ferrotorch returns `InvalidArgument`.
    #[test]
    fn test_functional_pad_1d_signed_over_crop_errors() {
        // Single side over-crops (left 4 from size 3).
        let x = t(&[1.0, 2.0, 3.0], &[1, 1, 3]);
        assert!(
            functional_pad_1d_signed(&x, -4, 0, PaddingMode::Zeros, 0.0).is_err(),
            "single-side over-crop must error like torch narrow()"
        );
        // Combined net negative size (left 2 + right 2 from size 3 -> -1).
        assert!(
            functional_pad_1d_signed(&x, -2, -2, PaddingMode::Zeros, 0.0).is_err(),
            "combined net-negative crop must error like torch"
        );
        // Right side over-crops after left (left 1 -> size 2, right 3 -> -1).
        assert!(
            functional_pad_1d_signed(&x, -1, -3, PaddingMode::Zeros, 0.0).is_err(),
            "right-after-left over-crop must error like torch"
        );
    }

    /// Net-zero crop is NOT an error in torch: `F.pad([[[1,2,3]]], [-1,-2])`
    /// returns an empty dim `[1,1,0]`. ferrotorch must match (no error).
    #[test]
    fn test_functional_pad_1d_signed_net_zero_empty_dim_matches_torch() {
        let x = t(&[1.0, 2.0, 3.0], &[1, 1, 3]);
        let y = functional_pad_1d_signed(&x, -1, -2, PaddingMode::Zeros, 0.0).unwrap();
        assert_eq!(y.shape(), &[1, 1, 0]);
        assert!(y.data().unwrap().is_empty());
    }

    /// Negative (crop) pad under a non-constant mode CROPS — live torch 2.11's
    /// `_pad_enum` dispatches reflect/replicate/circular straight to the native
    /// kernels, which narrow for negative pads (`PadNd.cpp:221-242`). For
    /// `[-1, 0]` on `[1,2,3,4]` all three modes crop the left element, yielding
    /// `[2,3,4]` (the positive part of the pad is zero, so it is a pure crop).
    /// torch: `F.pad([[[1.,2.,3.,4.]]], [-1,0], mode=<m>)` -> shape [1,1,3],
    /// `[2,3,4]` for reflect/replicate/circular alike (#1620).
    #[test]
    fn test_functional_pad_signed_negative_non_constant_crops() {
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
        for mode in [
            PaddingMode::Reflect,
            PaddingMode::Replicate,
            PaddingMode::Circular,
        ] {
            let y = functional_pad_1d_signed(&x, -1, 0, mode, 0.0)
                .unwrap_or_else(|_| panic!("negative pad under {mode:?} must crop, not error"));
            assert_eq!(
                y.shape(),
                &[1, 1, 3],
                "{mode:?} crops left -> shape [1,1,3]"
            );
            assert_close(y.data().unwrap(), &[2.0, 3.0, 4.0], 1e-7);
        }
    }

    /// A non-negative signed pad must be byte-identical to the existing
    /// positive-only `functional_pad_1d` (the delegation invariant that makes
    /// the signed path the single source of truth for constant padding without
    /// changing conv.rs's production behaviour). torch:
    /// `F.pad([[[1,2,3]]], [1,1], "constant", value=2)` -> [2,1,2,3,2].
    #[test]
    fn test_functional_pad_1d_signed_nonneg_equals_positive_path() {
        let input = t(&[1.0, 2.0, 3.0], &[1, 1, 3]);
        let signed = functional_pad_1d_signed(&input, 1, 1, PaddingMode::Zeros, 2.0).unwrap();
        let positive = functional_pad_1d(&input, 1, 1, PaddingMode::Zeros, 2.0).unwrap();
        assert_eq!(signed.shape(), positive.shape());
        assert_close(signed.data().unwrap(), positive.data().unwrap(), 1e-7);
        assert_close(signed.data().unwrap(), &[2.0, 1.0, 2.0, 3.0, 2.0], 1e-7);
    }
}
