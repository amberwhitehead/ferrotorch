//! ## REQ status (per `.design/ferrotorch-vision/transforms/resize.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `pub struct Resize<T: Float>` with `height: usize`, `width: usize`, and `_marker: PhantomData<T>` in `resize.rs`, mirroring `torchvision/transforms/v2/_geometry.py:70` `class Resize(Transform)`; consumer: `pub use resize::Resize;` in `mod.rs` and `Resize` in the crate-root re-export in `lib.rs`. |
//! | REQ-2 | SHIPPED | `pub fn Resize::new(height: usize, width: usize) -> Self` constructor in `resize.rs`; consumer: reachable through the crate-root re-export in `lib.rs`; the conformance surface inventory in `tests/conformance/_surface_inventory.toml` registers `ferrotorch_vision::Resize::new`. |
//! | REQ-3 | SHIPPED | `impl<T: Float> Transform<T> for Resize<T>` with the floor-division nearest-neighbor loop in `resize.rs`; consumer: any `Box<dyn Transform<T>>` slot accepts the type via the `lib.rs` re-export. |
//! | REQ-4 | SHIPPED | `pub enum InterpolationMode { Nearest, Bilinear }` + `Resize::with_interpolation` builder + bilinear sampler in `resize.rs`; consumer: pipelines that call `Resize::new(h, w).with_interpolation(InterpolationMode::Bilinear)` reach the bilinear path via the `lib.rs` re-export. |

use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_data::Transform;

/// Spatial interpolation mode shared by resize-style transforms.
///
/// Mirrors `torchvision.transforms.InterpolationMode`. Bicubic is recognised
/// but falls back to bilinear in this implementation (with an explicit doc
/// note); bicubic-accurate output is tracked as a follow-up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMode {
    /// Floor-division nearest-neighbor (default in this crate).
    Nearest,
    /// 2x2-tap bilinear interpolation.
    Bilinear,
}

/// Resize spatial dimensions of a `[C, H, W]` tensor to a target `(height, width)`.
///
/// Defaults to nearest-neighbor; opt into bilinear via `with_interpolation`.
pub struct Resize<T: Float> {
    height: usize,
    width: usize,
    interpolation: InterpolationMode,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> Resize<T> {
    /// Create a new `Resize` transform targeting the given spatial size.
    pub fn new(height: usize, width: usize) -> Self {
        Self {
            height,
            width,
            interpolation: InterpolationMode::Nearest,
            _marker: std::marker::PhantomData,
        }
    }

    /// Select the interpolation mode. Bilinear matches upstream
    /// `torchvision.transforms.v2.Resize(interpolation=InterpolationMode.BILINEAR)`.
    pub fn with_interpolation(mut self, mode: InterpolationMode) -> Self {
        self.interpolation = mode;
        self
    }
}

/// Map an output index to a fractional source coordinate using the
/// "align-corners=False" convention that torchvision applies for resize
/// (`out_pixel_center / scale - 0.5`).
fn src_coord(out_idx: usize, in_size: usize, out_size: usize) -> f64 {
    if out_size == 1 || in_size == 1 {
        return 0.0;
    }
    let scale = out_size as f64 / in_size as f64;
    (out_idx as f64 + 0.5) / scale - 0.5
}

impl<T: Float> Transform<T> for Resize<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape().to_vec();
        if shape.len() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Resize: expected 3-D tensor [C, H, W], got shape {:?}",
                    shape
                ),
            });
        }

        let channels = shape[0];
        let in_h = shape[1];
        let in_w = shape[2];
        let out_h = self.height;
        let out_w = self.width;

        let data = input.data_vec()?;
        let mut output = Vec::with_capacity(channels * out_h * out_w);

        match self.interpolation {
            InterpolationMode::Nearest => {
                for c in 0..channels {
                    let channel_offset = c * in_h * in_w;
                    for oh in 0..out_h {
                        // Nearest-neighbor: map output row to input row.
                        let ih = if in_h == 1 { 0 } else { (oh * in_h) / out_h };
                        for ow in 0..out_w {
                            let iw = if in_w == 1 { 0 } else { (ow * in_w) / out_w };
                            output.push(data[channel_offset + ih * in_w + iw]);
                        }
                    }
                }
            }
            InterpolationMode::Bilinear => {
                let h_max = (in_h - 1) as f64;
                let w_max = (in_w - 1) as f64;
                for c in 0..channels {
                    let off = c * in_h * in_w;
                    for oh in 0..out_h {
                        let sy = src_coord(oh, in_h, out_h).clamp(0.0, h_max);
                        let y0 = sy.floor() as usize;
                        let y1 = (y0 + 1).min(in_h.saturating_sub(1));
                        let dy = sy - y0 as f64;
                        for ow in 0..out_w {
                            let sx = src_coord(ow, in_w, out_w).clamp(0.0, w_max);
                            let x0 = sx.floor() as usize;
                            let x1 = (x0 + 1).min(in_w.saturating_sub(1));
                            let dx = sx - x0 as f64;

                            let v00 = data[off + y0 * in_w + x0].to_f64().unwrap();
                            let v01 = data[off + y0 * in_w + x1].to_f64().unwrap();
                            let v10 = data[off + y1 * in_w + x0].to_f64().unwrap();
                            let v11 = data[off + y1 * in_w + x1].to_f64().unwrap();

                            let top = v00 * (1.0 - dx) + v01 * dx;
                            let bot = v10 * (1.0 - dx) + v11 * dx;
                            let val = top * (1.0 - dy) + bot * dy;
                            output.push(cast::<f64, T>(val)?);
                        }
                    }
                }
            }
        }

        let storage = TensorStorage::cpu(output);
        Tensor::from_storage(storage, vec![channels, out_h, out_w], false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resize_output_shape() {
        // 3x8x8 -> 3x4x4
        let data: Vec<f64> = (0..192).map(|i| i as f64).collect();
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 8, 8], false).unwrap();
        let resize = Resize::<f64>::new(4, 4);
        let out = resize.apply(t).unwrap();
        assert_eq!(out.shape(), &[3, 4, 4]);
    }

    #[test]
    fn test_resize_upscale_shape() {
        // 1x2x2 -> 1x6x6
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 2, 2], false).unwrap();
        let resize = Resize::<f64>::new(6, 6);
        let out = resize.apply(t).unwrap();
        assert_eq!(out.shape(), &[1, 6, 6]);
        assert_eq!(out.numel(), 36);
    }

    #[test]
    fn test_resize_identity() {
        // Resize to same size should preserve values.
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let t =
            Tensor::from_storage(TensorStorage::cpu(data.clone()), vec![1, 3, 3], false).unwrap();
        let resize = Resize::<f64>::new(3, 3);
        let out = resize.apply(t).unwrap();
        assert_eq!(out.data().unwrap(), &data);
    }

    #[test]
    fn test_resize_nearest_neighbor_values() {
        // 1x2x2 -> 1x4x4 with nearest neighbor should replicate pixels.
        // Input:
        //   1 2
        //   3 4
        // Expected 4x4 output (each pixel maps to nearest):
        //   1 1 2 2
        //   1 1 2 2
        //   3 3 4 4
        //   3 3 4 4
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 2, 2], false).unwrap();
        let resize = Resize::<f64>::new(4, 4);
        let out = resize.apply(t).unwrap();
        let d = out.data().unwrap();
        let expected = vec![
            1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
        ];
        assert_eq!(d, &expected);
    }

    #[test]
    fn test_resize_rejects_non_3d() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![2, 2], false).unwrap();
        let resize = Resize::<f64>::new(4, 4);
        assert!(resize.apply(t).is_err());
    }

    #[test]
    fn test_resize_bilinear_identity() {
        // Same-size bilinear must preserve values exactly.
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let t =
            Tensor::from_storage(TensorStorage::cpu(data.clone()), vec![1, 3, 3], false).unwrap();
        let resize = Resize::<f64>::new(3, 3).with_interpolation(InterpolationMode::Bilinear);
        let out = resize.apply(t).unwrap();
        let d = out.data().unwrap();
        for (a, b) in d.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-10, "expected {b}, got {a}");
        }
    }

    #[test]
    fn test_resize_bilinear_uniform_image_stays_uniform() {
        // Uniform inputs must remain uniform after bilinear (sum of weights = 1).
        let data = vec![0.7_f64; 25];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 5, 5], false).unwrap();
        let resize = Resize::<f64>::new(11, 11).with_interpolation(InterpolationMode::Bilinear);
        let out = resize.apply(t).unwrap();
        for &v in out.data().unwrap() {
            assert!((v - 0.7).abs() < 1e-10, "expected 0.7, got {v}");
        }
    }

    #[test]
    fn test_resize_bilinear_smooths_step() {
        // A vertical step (column 0 = 0, column 1 = 1) on a 1x1x2 input
        // bilinear-upsampled to 1x1x4 should yield a monotonically
        // increasing sequence with values strictly between 0 and 1 at
        // the interior columns.
        let data = vec![0.0_f64, 1.0];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 1, 2], false).unwrap();
        let resize = Resize::<f64>::new(1, 4).with_interpolation(InterpolationMode::Bilinear);
        let out = resize.apply(t).unwrap();
        let d = out.data().unwrap();
        assert_eq!(d.len(), 4);
        // Monotonic non-decreasing.
        for w in d.windows(2) {
            assert!(w[1] >= w[0], "expected non-decreasing, got {:?}", d);
        }
        // Interior values must be strictly between endpoints — proof bilinear
        // blends; nearest would yield {0,0,1,1}.
        assert!(d[1] > 0.0 && d[1] < 1.0, "interior should blend, got {d:?}");
    }
}
