// CL-332: Vision Transforms & Augmentation — RandomResizedCrop
//! ## REQ status (per `.design/ferrotorch-vision/transforms/random_resized_crop.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `pub struct RandomResizedCrop<T: Float>` with `height`, `width`, `scale_lo`, `scale_hi`, `ratio_lo`, `ratio_hi`, and `_marker: PhantomData<T>` in `random_resized_crop.rs`, mirroring `torchvision/transforms/v2/_geometry.py:197` `class RandomResizedCrop`; consumer: `pub use random_resized_crop::RandomResizedCrop;` in `mod.rs` and `RandomResizedCrop` in the crate-root re-export in `lib.rs`. |
//! | REQ-2 | SHIPPED | `pub fn RandomResizedCrop::new(height, width, scale, ratio) -> FerrotorchResult<Self>` constructor with scale/ratio range checks in `random_resized_crop.rs`; consumer: reachable through the crate-root re-export in `lib.rs`. |
//! | REQ-3 | SHIPPED | `pub(crate) fn nn_resize_channel<T: Float>(src, in_h, in_w, out_h, out_w, dst)` helper in `random_resized_crop.rs`; consumer: the impl in the same file calls `nn_resize_channel` within the per-channel resize loop. |
//! | REQ-4 | SHIPPED | `impl<T: Float> Transform<T> for RandomResizedCrop<T>` with the 10-attempt sampling, center-crop fallback, and per-channel crop plus nn-resize in `random_resized_crop.rs`; consumer: any `Box<dyn Transform<T>>` slot — typically the first stage of an Inception/ResNet ImageNet `Compose` training pipeline. |
//! | REQ-5 | SHIPPED | `RandomResizedCrop::with_interpolation` builder + bilinear sampler dispatch in `random_resized_crop.rs`; consumer: pipelines call `RandomResizedCrop::new(...).?.with_interpolation(InterpolationMode::Bilinear)` for the canonical ImageNet preset via the `lib.rs` re-export. |

use super::resize::InterpolationMode;
use super::rng::random_f64;
use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_data::Transform;

/// Crop a random region of the input, then resize to a target size.
///
/// This mirrors `torchvision.transforms.RandomResizedCrop`. A rectangular
/// region whose area is a random fraction (within `scale`) of the original
/// area and whose aspect ratio falls within `ratio` is sampled. The region
/// is then resized to `(height, width)` using nearest-neighbor interpolation.
///
/// If no valid crop can be found after a fixed number of attempts, a center
/// crop at the target aspect ratio is used as a fallback.
pub struct RandomResizedCrop<T: Float> {
    height: usize,
    width: usize,
    scale_lo: f64,
    scale_hi: f64,
    ratio_lo: f64,
    ratio_hi: f64,
    interpolation: InterpolationMode,
    antialias: bool,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> RandomResizedCrop<T> {
    /// Create a new `RandomResizedCrop`.
    ///
    /// * `height`, `width` — output spatial size.
    /// * `scale` — range of area fraction `(lo, hi)` relative to the input,
    ///   e.g. `(0.08, 1.0)`.
    /// * `ratio` — range of aspect ratio `(lo, hi)`, e.g. `(3.0/4.0, 4.0/3.0)`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if `scale` is not
    /// `(0, 1] × (0, 1]` with `lo <= hi`, or `ratio` is not positive with
    /// `lo <= hi`.
    pub fn new(
        height: usize,
        width: usize,
        scale: (f64, f64),
        ratio: (f64, f64),
    ) -> FerrotorchResult<Self> {
        if !(scale.0 > 0.0 && scale.0 <= scale.1 && scale.1 <= 1.0) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "RandomResizedCrop: scale must satisfy 0 < lo <= hi <= 1, got ({}, {})",
                    scale.0, scale.1,
                ),
            });
        }
        if !(ratio.0 > 0.0 && ratio.0 <= ratio.1) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "RandomResizedCrop: ratio must satisfy 0 < lo <= hi, got ({}, {})",
                    ratio.0, ratio.1,
                ),
            });
        }
        Ok(Self {
            height,
            width,
            scale_lo: scale.0,
            scale_hi: scale.1,
            ratio_lo: ratio.0,
            ratio_hi: ratio.1,
            interpolation: InterpolationMode::Nearest,
            antialias: false,
            _marker: std::marker::PhantomData,
        })
    }

    /// Select the interpolation mode used when resizing the sampled crop
    /// back to the target `(height, width)`. Mirrors upstream
    /// `RandomResizedCrop(interpolation=InterpolationMode.BILINEAR)`.
    pub fn with_interpolation(mut self, mode: InterpolationMode) -> Self {
        self.interpolation = mode;
        self
    }

    /// Enable antialiasing for downscale operations under bilinear. When
    /// `true`, a separable box-blur with kernel radius `≈ scale / 2` is
    /// applied before sampling, suppressing aliasing without the cost of
    /// a full bicubic filter. Mirrors the `antialias=True` default from
    /// upstream `_geometry.py:197-315`.
    pub fn with_antialias(mut self, antialias: bool) -> Self {
        self.antialias = antialias;
        self
    }
}

/// Nearest-neighbor resize of a single channel from `(in_h, in_w)` to
/// `(out_h, out_w)`, reading from `src` (length `in_h * in_w`).
pub(crate) fn nn_resize_channel<T: Float>(
    src: &[T],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    dst: &mut Vec<T>,
) {
    for oh in 0..out_h {
        let ih = if in_h == 1 { 0 } else { (oh * in_h) / out_h };
        for ow in 0..out_w {
            let iw = if in_w == 1 { 0 } else { (ow * in_w) / out_w };
            dst.push(src[ih * in_w + iw]);
        }
    }
}

/// Map an output index to a fractional source coordinate using the
/// "align-corners=False" convention used by torchvision.
fn src_coord(out_idx: usize, in_size: usize, out_size: usize) -> f64 {
    if out_size == 1 || in_size == 1 {
        return 0.0;
    }
    let scale = out_size as f64 / in_size as f64;
    (out_idx as f64 + 0.5) / scale - 0.5
}

/// Apply a 1-D separable box pre-filter for antialiasing on downscale.
/// `kernel` is the per-axis radius (in source pixels); rows then cols.
fn antialias_prefilter(src: &mut Vec<f64>, in_h: usize, in_w: usize, kr: usize, kc: usize) {
    // Horizontal pass.
    if kc > 0 {
        let mut out = vec![0.0_f64; in_h * in_w];
        let win = 2 * kc + 1;
        for r in 0..in_h {
            for c in 0..in_w {
                let mut acc = 0.0;
                for d in 0..win {
                    let s = c as i64 + d as i64 - kc as i64;
                    let s = s.clamp(0, in_w as i64 - 1) as usize;
                    acc += src[r * in_w + s];
                }
                out[r * in_w + c] = acc / win as f64;
            }
        }
        *src = out;
    }
    // Vertical pass.
    if kr > 0 {
        let mut out = vec![0.0_f64; in_h * in_w];
        let win = 2 * kr + 1;
        for r in 0..in_h {
            for c in 0..in_w {
                let mut acc = 0.0;
                for d in 0..win {
                    let s = r as i64 + d as i64 - kr as i64;
                    let s = s.clamp(0, in_h as i64 - 1) as usize;
                    acc += src[s * in_w + c];
                }
                out[r * in_w + c] = acc / win as f64;
            }
        }
        *src = out;
    }
}

/// Bilinear resize of a single channel from `(in_h, in_w)` to `(out_h, out_w)`.
/// Reads `src` (length `in_h * in_w`), appends `out_h * out_w` values to `dst`.
pub(crate) fn bilinear_resize_channel<T: Float>(
    src: &[T],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    antialias: bool,
    dst: &mut Vec<T>,
) -> FerrotorchResult<()> {
    // Promote to f64 for accumulation. Optional antialias prefilter.
    let mut buf: Vec<f64> = src.iter().map(|v| v.to_f64().unwrap()).collect();
    if antialias {
        // Box-prefilter only on the downscale axes.
        let scale_h = in_h as f64 / out_h as f64;
        let scale_w = in_w as f64 / out_w as f64;
        let kr = if scale_h > 1.0 {
            (scale_h * 0.5).floor() as usize
        } else {
            0
        };
        let kc = if scale_w > 1.0 {
            (scale_w * 0.5).floor() as usize
        } else {
            0
        };
        antialias_prefilter(&mut buf, in_h, in_w, kr, kc);
    }
    let h_max = (in_h - 1) as f64;
    let w_max = (in_w - 1) as f64;
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
            let v00 = buf[y0 * in_w + x0];
            let v01 = buf[y0 * in_w + x1];
            let v10 = buf[y1 * in_w + x0];
            let v11 = buf[y1 * in_w + x1];
            let top = v00 * (1.0 - dx) + v01 * dx;
            let bot = v10 * (1.0 - dx) + v11 * dx;
            let val = top * (1.0 - dy) + bot * dy;
            dst.push(cast::<f64, T>(val)?);
        }
    }
    Ok(())
}

impl<T: Float> Transform<T> for RandomResizedCrop<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape().to_vec();
        if shape.len() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "RandomResizedCrop: expected 3-D tensor [C, H, W], got shape {:?}",
                    shape
                ),
            });
        }

        let channels = shape[0];
        let in_h = shape[1];
        let in_w = shape[2];
        let area = (in_h * in_w) as f64;

        let data = input.data()?;

        // Try up to 10 times to find a valid crop.
        let mut crop_top = 0usize;
        let mut crop_left = 0usize;
        let mut crop_h = in_h;
        let mut crop_w = in_w;
        let mut found = false;

        for _ in 0..10 {
            let target_area =
                area * (self.scale_lo + random_f64() * (self.scale_hi - self.scale_lo));
            let log_lo = self.ratio_lo.ln();
            let log_hi = self.ratio_hi.ln();
            let aspect = (log_lo + random_f64() * (log_hi - log_lo)).exp();

            let w_f = (target_area * aspect).sqrt();
            let h_f = (target_area / aspect).sqrt();
            let w_candidate = w_f.round() as usize;
            let h_candidate = h_f.round() as usize;

            if w_candidate >= 1 && h_candidate >= 1 && w_candidate <= in_w && h_candidate <= in_h {
                crop_h = h_candidate;
                crop_w = w_candidate;
                crop_top = if in_h == crop_h {
                    0
                } else {
                    (random_f64() * (in_h - crop_h) as f64) as usize
                };
                crop_left = if in_w == crop_w {
                    0
                } else {
                    (random_f64() * (in_w - crop_w) as f64) as usize
                };
                found = true;
                break;
            }
        }

        if !found {
            // Fallback: center crop at the target aspect ratio.
            let target_ratio = self.width as f64 / self.height as f64;
            let in_ratio = in_w as f64 / in_h as f64;
            if in_ratio < target_ratio {
                crop_w = in_w;
                crop_h = ((in_w as f64 / target_ratio).round() as usize)
                    .max(1)
                    .min(in_h);
            } else {
                crop_h = in_h;
                crop_w = ((in_h as f64 * target_ratio).round() as usize)
                    .max(1)
                    .min(in_w);
            }
            crop_top = (in_h - crop_h) / 2;
            crop_left = (in_w - crop_w) / 2;
        }

        // Extract the crop, then resize to (self.height, self.width).
        let mut output = Vec::with_capacity(channels * self.height * self.width);

        for c in 0..channels {
            let ch_off = c * in_h * in_w;
            // Extract cropped channel into a temporary buffer.
            let mut cropped = Vec::with_capacity(crop_h * crop_w);
            for row in crop_top..crop_top + crop_h {
                let start = ch_off + row * in_w + crop_left;
                cropped.extend_from_slice(&data[start..start + crop_w]);
            }
            match self.interpolation {
                InterpolationMode::Nearest => {
                    nn_resize_channel(
                        &cropped,
                        crop_h,
                        crop_w,
                        self.height,
                        self.width,
                        &mut output,
                    );
                }
                InterpolationMode::Bilinear => {
                    bilinear_resize_channel(
                        &cropped,
                        crop_h,
                        crop_w,
                        self.height,
                        self.width,
                        self.antialias,
                        &mut output,
                    )?;
                }
            }
        }

        let storage = TensorStorage::cpu(output);
        Tensor::from_storage(storage, vec![channels, self.height, self.width], false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_resized_crop_output_shape() {
        let data: Vec<f64> = (0..300).map(|i| i as f64).collect();
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 10, 10], false).unwrap();
        let rrc = RandomResizedCrop::<f64>::new(5, 5, (0.08, 1.0), (0.75, 1.333)).unwrap();
        let out = rrc.apply(t).unwrap();
        assert_eq!(out.shape(), &[3, 5, 5]);
    }

    #[test]
    fn test_random_resized_crop_full_scale() {
        // scale=(1.0, 1.0), ratio=(1.0, 1.0): should crop the entire image
        // and resize to target.
        let data: Vec<f64> = (0..48).map(|i| i as f64).collect();
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 4, 4], false).unwrap();
        let rrc = RandomResizedCrop::<f64>::new(2, 2, (1.0, 1.0), (1.0, 1.0)).unwrap();
        let out = rrc.apply(t).unwrap();
        assert_eq!(out.shape(), &[3, 2, 2]);
    }

    #[test]
    fn test_random_resized_crop_values_from_input() {
        let data: Vec<f64> = (0..75).map(|i| i as f64).collect();
        let t =
            Tensor::from_storage(TensorStorage::cpu(data.clone()), vec![3, 5, 5], false).unwrap();
        let rrc = RandomResizedCrop::<f64>::new(3, 3, (0.5, 1.0), (0.75, 1.333)).unwrap();
        let out = rrc.apply(t).unwrap();
        let out_data = out.data().unwrap();
        let original: std::collections::HashSet<u64> = data.iter().map(|&v| v.to_bits()).collect();
        for &val in out_data {
            assert!(
                original.contains(&val.to_bits()),
                "Output value {val} not found in original"
            );
        }
    }

    #[test]
    fn test_random_resized_crop_rejects_non_3d() {
        let data = vec![1.0_f64; 8];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![2, 4], false).unwrap();
        let rrc = RandomResizedCrop::<f64>::new(2, 2, (0.08, 1.0), (0.75, 1.333)).unwrap();
        assert!(rrc.apply(t).is_err());
    }

    #[test]
    fn test_random_resized_crop_multichannel() {
        let data: Vec<f64> = (0..192).map(|i| i as f64).collect();
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 8, 8], false).unwrap();
        let rrc = RandomResizedCrop::<f64>::new(4, 4, (0.2, 0.8), (0.75, 1.333)).unwrap();
        let out = rrc.apply(t).unwrap();
        assert_eq!(out.shape(), &[3, 4, 4]);
        assert_eq!(out.numel(), 48);
    }

    #[test]
    fn test_nn_resize_channel_identity() {
        let src = vec![1.0_f64, 2.0, 3.0, 4.0];
        let mut dst = Vec::new();
        nn_resize_channel(&src, 2, 2, 2, 2, &mut dst);
        assert_eq!(dst, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_random_resized_crop_bilinear_output_shape() {
        let data: Vec<f64> = (0..300).map(|i| i as f64).collect();
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 10, 10], false).unwrap();
        let rrc = RandomResizedCrop::<f64>::new(5, 5, (0.08, 1.0), (0.75, 1.333))
            .unwrap()
            .with_interpolation(InterpolationMode::Bilinear);
        let out = rrc.apply(t).unwrap();
        assert_eq!(out.shape(), &[3, 5, 5]);
    }

    #[test]
    fn test_random_resized_crop_bilinear_uniform_input_stays_uniform() {
        // Uniform input under bilinear resize stays uniform.
        let data: Vec<f64> = vec![0.4; 75];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 5, 5], false).unwrap();
        let rrc = RandomResizedCrop::<f64>::new(3, 3, (0.5, 1.0), (0.75, 1.333))
            .unwrap()
            .with_interpolation(InterpolationMode::Bilinear);
        let out = rrc.apply(t).unwrap();
        for &v in out.data().unwrap() {
            assert!((v - 0.4).abs() < 1e-10, "expected 0.4, got {v}");
        }
    }

    #[test]
    fn test_random_resized_crop_bilinear_with_antialias_smoke() {
        // Antialiasing on a noisy 8x8 downscale must not panic and must
        // produce values within the original range.
        let data: Vec<f64> = (0..192).map(|i| (i % 7) as f64 * 0.1).collect();
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 8, 8], false).unwrap();
        let rrc = RandomResizedCrop::<f64>::new(4, 4, (0.2, 0.5), (0.75, 1.333))
            .unwrap()
            .with_interpolation(InterpolationMode::Bilinear)
            .with_antialias(true);
        let out = rrc.apply(t).unwrap();
        for &v in out.data().unwrap() {
            assert!((0.0..=0.7).contains(&v), "expected blurred range, got {v}");
        }
    }

    #[test]
    fn test_bilinear_resize_channel_identity() {
        let src = vec![1.0_f64, 2.0, 3.0, 4.0];
        let mut dst = Vec::new();
        bilinear_resize_channel(&src, 2, 2, 2, 2, false, &mut dst).unwrap();
        for (a, b) in dst.iter().zip(src.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_nn_resize_channel_upscale() {
        // 2x2 -> 4x4
        let src = vec![1.0_f64, 2.0, 3.0, 4.0];
        let mut dst = Vec::new();
        nn_resize_channel(&src, 2, 2, 4, 4, &mut dst);
        let expected = vec![
            1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
        ];
        assert_eq!(dst, expected);
    }
}
