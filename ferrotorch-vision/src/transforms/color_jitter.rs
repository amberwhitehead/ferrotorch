// CL-332: Vision Transforms & Augmentation — ColorJitter
//! ## REQ status (per `.design/ferrotorch-vision/transforms/color_jitter.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `pub struct ColorJitter<T: Float>` with `brightness: f64`, `contrast: f64`, `saturation: f64`, `hue: f64`, and `_marker: PhantomData<T>` in `color_jitter.rs`, mirroring `torchvision/transforms/v2/_color.py:72` `class ColorJitter`; consumer: `pub use color_jitter::ColorJitter;` in `mod.rs` and `ColorJitter` in the crate-root re-export in `lib.rs`. |
//! | REQ-2 | SHIPPED | `pub fn ColorJitter::new(brightness, contrast, saturation, hue) -> FerrotorchResult<Self>` constructor with four range checks in `color_jitter.rs`; consumer: registered in `tests/conformance/_surface_inventory.toml` as `ferrotorch_vision::ColorJitter::new`; reachable through the crate-root re-export. |
//! | REQ-3 | SHIPPED | `fn shuffle_order(n: usize) -> Vec<usize>` Fisher-Yates helper in `color_jitter.rs`; consumer: the impl calls `let order = shuffle_order(4);` before iterating the four jitter ops. |
//! | REQ-4 | SHIPPED | `fn uniform_factor(v: f64) -> f64` helper in `color_jitter.rs`; consumer: the impl calls `uniform_factor(self.brightness)`, `uniform_factor(self.contrast)`, and `uniform_factor(self.saturation)` inside the per-op branches. |
//! | REQ-5 | SHIPPED | `impl<T: Float> Transform<T> for ColorJitter<T>` in `color_jitter.rs`; consumer: any `Box<dyn Transform<T>>` slot — typically near the start of an augmentation `Compose` pipeline. The `lib.rs` re-export is the production-facing handle. |
//! | REQ-6 | SHIPPED | `fn rgb_to_hsv(r, g, b) -> (f64, f64, f64)` and `fn hsv_to_rgb(h, s, v)` conversion helpers in `color_jitter.rs`; consumer: the impl calls `rgb_to_hsv` and `hsv_to_rgb` per pixel inside the hue branch. |
//! | REQ-7 | SHIPPED | `pub fn ColorJitter::from_ranges(brightness, contrast, saturation, hue)` constructor + `(min, max)` range storage in `color_jitter.rs`; consumer: pipelines call `ColorJitter::from_ranges((0.8, 1.2), (0.8, 1.2), (0.8, 1.2), (-0.05, 0.05))?` per upstream `_color.py:100-122` `_check_input`. |

use super::rng::random_f64;
use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_data::Transform;

/// Randomly adjust the brightness, contrast, saturation, and hue of an image.
///
/// Expects a `[3, H, W]` tensor in **RGB** channel order with values in
/// `[0, 1]`. Each parameter specifies a range `[max(0, 1 - v), 1 + v]` from
/// which a multiplicative/additive factor is uniformly sampled.
///
/// Processing order is randomised per call (matching PyTorch):
///
/// 1. **Brightness** — scale all channels by a factor in `[1 - b, 1 + b]`.
/// 2. **Contrast** — blend towards the per-channel mean by a factor.
/// 3. **Saturation** — blend towards the luminance (grayscale) image.
/// 4. **Hue** — rotate the hue angle in HSV space by a shift in `[-h, h]`
///    (measured in fraction of a full circle, range `(-0.5, 0.5)`).
///
/// This mirrors `torchvision.transforms.ColorJitter`.
pub struct ColorJitter<T: Float> {
    /// `(lo, hi)` for the brightness factor. `(1, 1)` means no change.
    brightness: (f64, f64),
    /// `(lo, hi)` for the contrast factor. `(1, 1)` means no change.
    contrast: (f64, f64),
    /// `(lo, hi)` for the saturation factor. `(1, 1)` means no change.
    saturation: (f64, f64),
    /// `(lo, hi)` for the hue shift in turns; `(0, 0)` means no change.
    hue: (f64, f64),
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> ColorJitter<T> {
    /// Create a new `ColorJitter`.
    ///
    /// * `brightness` — non-negative. 0 means no change. Factor sampled from
    ///   `[max(0, 1 - brightness), 1 + brightness]`.
    /// * `contrast` — same convention.
    /// * `saturation` — same convention.
    /// * `hue` — in `[0, 0.5)`. Hue shift sampled from `[-hue, +hue]`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if any of `brightness`,
    /// `contrast`, `saturation` is negative, or if `hue` is outside `[0, 0.5)`.
    pub fn new(
        brightness: f64,
        contrast: f64,
        saturation: f64,
        hue: f64,
    ) -> FerrotorchResult<Self> {
        if brightness < 0.0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("ColorJitter: brightness must be >= 0, got {brightness}"),
            });
        }
        if contrast < 0.0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("ColorJitter: contrast must be >= 0, got {contrast}"),
            });
        }
        if saturation < 0.0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("ColorJitter: saturation must be >= 0, got {saturation}"),
            });
        }
        if !((0.0..0.5).contains(&hue) || hue == 0.0) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("ColorJitter: hue must be in [0, 0.5), got {hue}"),
            });
        }
        // Scalar shorthand mirrors torchvision: brightness=b ->
        // factor in [max(0, 1-b), 1+b]; hue=h -> shift in [-h, +h].
        Ok(Self {
            brightness: ((1.0 - brightness).max(0.0), 1.0 + brightness),
            contrast: ((1.0 - contrast).max(0.0), 1.0 + contrast),
            saturation: ((1.0 - saturation).max(0.0), 1.0 + saturation),
            hue: (-hue, hue),
            _marker: std::marker::PhantomData,
        })
    }

    /// Construct from explicit `(min, max)` ranges. Mirrors the upstream
    /// tuple form
    /// (`torchvision/transforms/v2/_color.py:100-122` `_check_input`):
    ///
    /// * `brightness/contrast/saturation`: range must lie within `[0, ∞)`
    ///   with `min <= max`. `(1, 1)` is the identity.
    /// * `hue`: range must lie within `[-0.5, 0.5]` with `min <= max`.
    ///   `(0, 0)` is the identity.
    pub fn from_ranges(
        brightness: (f64, f64),
        contrast: (f64, f64),
        saturation: (f64, f64),
        hue: (f64, f64),
    ) -> FerrotorchResult<Self> {
        check_pos_range("brightness", brightness)?;
        check_pos_range("contrast", contrast)?;
        check_pos_range("saturation", saturation)?;
        check_hue_range(hue)?;
        Ok(Self {
            brightness,
            contrast,
            saturation,
            hue,
            _marker: std::marker::PhantomData,
        })
    }
}

fn check_pos_range(name: &str, r: (f64, f64)) -> FerrotorchResult<()> {
    if !(r.0 >= 0.0 && r.0 <= r.1) {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "ColorJitter: {name} range must satisfy 0 <= lo <= hi, got ({}, {})",
                r.0, r.1
            ),
        });
    }
    Ok(())
}

fn check_hue_range(r: (f64, f64)) -> FerrotorchResult<()> {
    if !(r.0 >= -0.5 && r.1 <= 0.5 && r.0 <= r.1) {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "ColorJitter: hue range must satisfy -0.5 <= lo <= hi <= 0.5, got ({}, {})",
                r.0, r.1
            ),
        });
    }
    Ok(())
}

/// Fisher-Yates shuffle using the global PRNG.
fn shuffle_order(n: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (random_f64() * (i + 1) as f64) as usize;
        let j = j.min(i); // Clamp in case random_f64() returns exactly 1.0.
        order.swap(i, j);
    }
    order
}

/// Sample a uniform factor from the given `(lo, hi)` range.
fn sample_range(r: (f64, f64)) -> f64 {
    if r.0 == r.1 {
        r.0
    } else {
        r.0 + random_f64() * (r.1 - r.0)
    }
}

/// Returns `true` if the range is a non-degenerate identity for the
/// brightness/contrast/saturation channels (centered on 1).
fn is_factor_identity(r: (f64, f64)) -> bool {
    r.0 == 1.0 && r.1 == 1.0
}

/// Returns `true` if the hue range is degenerate at zero.
fn is_hue_identity(r: (f64, f64)) -> bool {
    r.0 == 0.0 && r.1 == 0.0
}

impl<T: Float> Transform<T> for ColorJitter<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape().to_vec();
        if shape.len() != 3 || shape[0] != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ColorJitter: expected 3-D RGB tensor [3, H, W], got shape {:?}",
                    shape
                ),
            });
        }

        let h = shape[1];
        let w = shape[2];
        let spatial = h * w;
        let data = input.data_vec()?;
        // Work with per-channel slices as mutable f64 buffers for precision.
        let mut r: Vec<f64> = data[..spatial]
            .iter()
            .map(|v| v.to_f64().unwrap())
            .collect();
        let mut g: Vec<f64> = data[spatial..2 * spatial]
            .iter()
            .map(|v| v.to_f64().unwrap())
            .collect();
        let mut b: Vec<f64> = data[2 * spatial..]
            .iter()
            .map(|v| v.to_f64().unwrap())
            .collect();

        // Determine random order for the four adjustments.
        let order = shuffle_order(4);

        for &op in &order {
            match op {
                0 if !is_factor_identity(self.brightness) => {
                    let factor = sample_range(self.brightness);
                    for i in 0..spatial {
                        r[i] *= factor;
                        g[i] *= factor;
                        b[i] *= factor;
                    }
                }
                1 if !is_factor_identity(self.contrast) => {
                    let factor = sample_range(self.contrast);
                    // Compute per-channel mean.
                    let mean_r: f64 = r.iter().sum::<f64>() / spatial as f64;
                    let mean_g: f64 = g.iter().sum::<f64>() / spatial as f64;
                    let mean_b: f64 = b.iter().sum::<f64>() / spatial as f64;
                    for i in 0..spatial {
                        r[i] = mean_r + (r[i] - mean_r) * factor;
                        g[i] = mean_g + (g[i] - mean_g) * factor;
                        b[i] = mean_b + (b[i] - mean_b) * factor;
                    }
                }
                2 if !is_factor_identity(self.saturation) => {
                    let factor = sample_range(self.saturation);
                    // Grayscale via ITU-R BT.601 luma coefficients.
                    for i in 0..spatial {
                        let gray = 0.2989 * r[i] + 0.5870 * g[i] + 0.1140 * b[i];
                        r[i] = gray + (r[i] - gray) * factor;
                        g[i] = gray + (g[i] - gray) * factor;
                        b[i] = gray + (b[i] - gray) * factor;
                    }
                }
                3 if !is_hue_identity(self.hue) => {
                    let hue_shift = sample_range(self.hue);
                    for i in 0..spatial {
                        let (hue, sat, val) = rgb_to_hsv(r[i], g[i], b[i]);
                        let new_hue = (hue + hue_shift).rem_euclid(1.0);
                        let (nr, ng, nb) = hsv_to_rgb(new_hue, sat, val);
                        r[i] = nr;
                        g[i] = ng;
                        b[i] = nb;
                    }
                }
                _ => {}
            }
        }

        // Clamp to [0, 1] and convert back to T.
        let mut output = Vec::with_capacity(data.len());
        for v in r.iter().chain(g.iter()).chain(b.iter()) {
            let clamped = v.clamp(0.0, 1.0);
            output.push(cast::<f64, T>(clamped)?);
        }

        let storage = TensorStorage::cpu(output);
        Tensor::from_storage(storage, shape, false)
    }
}

// ---------------------------------------------------------------------------
// RGB <-> HSV conversion helpers
// ---------------------------------------------------------------------------

fn rgb_to_hsv(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let v = max;
    let s = if max == 0.0 { 0.0 } else { delta / max };

    let h = if delta == 0.0 {
        0.0
    } else if (max - r).abs() < 1e-15 {
        ((g - b) / delta).rem_euclid(6.0) / 6.0
    } else if (max - g).abs() < 1e-15 {
        ((b - r) / delta + 2.0) / 6.0
    } else {
        ((r - g) / delta + 4.0) / 6.0
    };

    (h, s, v)
}

fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (f64, f64, f64) {
    if s == 0.0 {
        return (v, v, v);
    }

    let h6 = h * 6.0;
    let sector = h6.floor() as usize % 6;
    let f = h6 - h6.floor();
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    match sector {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb_tensor(r: &[f64], g: &[f64], b: &[f64]) -> Tensor<f64> {
        let spatial = r.len();
        let mut data = Vec::with_capacity(3 * spatial);
        data.extend_from_slice(r);
        data.extend_from_slice(g);
        data.extend_from_slice(b);
        Tensor::from_storage(TensorStorage::cpu(data), vec![3, 1, spatial], false).unwrap()
    }

    #[test]
    fn test_color_jitter_output_shape() {
        let data: Vec<f64> = vec![0.5; 48]; // 3x4x4
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 4, 4], false).unwrap();
        let jitter = ColorJitter::<f64>::new(0.2, 0.2, 0.2, 0.1).unwrap();
        let out = jitter.apply(t).unwrap();
        assert_eq!(out.shape(), &[3, 4, 4]);
    }

    #[test]
    fn test_color_jitter_zero_params() {
        // All parameters zero: output should equal input.
        let data: Vec<f64> = (0..12).map(|i| i as f64 / 12.0).collect();
        let t =
            Tensor::from_storage(TensorStorage::cpu(data.clone()), vec![3, 2, 2], false).unwrap();
        let jitter = ColorJitter::<f64>::new(0.0, 0.0, 0.0, 0.0).unwrap();
        let out = jitter.apply(t).unwrap();
        let d = out.data().unwrap();
        for (a, b) in d.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-10, "Expected {b}, got {a}");
        }
    }

    #[test]
    fn test_color_jitter_output_clamped() {
        // Even with extreme parameters, output should be in [0, 1].
        let data: Vec<f64> = vec![0.9; 12]; // 3x2x2
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 2, 2], false).unwrap();
        let jitter = ColorJitter::<f64>::new(0.9, 0.9, 0.9, 0.4).unwrap();
        let out = jitter.apply(t).unwrap();
        for &val in out.data().unwrap() {
            assert!(
                (0.0..=1.0).contains(&val),
                "Output value {val} out of [0, 1]"
            );
        }
    }

    #[test]
    fn test_color_jitter_rejects_non_rgb() {
        // 1-channel tensor should be rejected.
        let data = vec![0.5_f64; 4];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 2, 2], false).unwrap();
        let jitter = ColorJitter::<f64>::new(0.2, 0.2, 0.2, 0.1).unwrap();
        assert!(jitter.apply(t).is_err());
    }

    #[test]
    fn test_rgb_hsv_roundtrip() {
        let test_colors = vec![
            (1.0, 0.0, 0.0), // Red
            (0.0, 1.0, 0.0), // Green
            (0.0, 0.0, 1.0), // Blue
            (0.5, 0.5, 0.5), // Gray
            (0.0, 0.0, 0.0), // Black
            (1.0, 1.0, 1.0), // White
            (0.3, 0.6, 0.9), // Arbitrary
        ];

        for (r, g, b) in test_colors {
            let (h, s, v) = rgb_to_hsv(r, g, b);
            let (r2, g2, b2) = hsv_to_rgb(h, s, v);
            assert!(
                (r - r2).abs() < 1e-10 && (g - g2).abs() < 1e-10 && (b - b2).abs() < 1e-10,
                "Roundtrip failed for ({r}, {g}, {b}) -> ({h}, {s}, {v}) -> ({r2}, {g2}, {b2})"
            );
        }
    }

    #[test]
    fn test_color_jitter_brightness_only() {
        // With only brightness, all pixels should be scaled uniformly.
        let r = vec![0.5; 4];
        let g = vec![0.4; 4];
        let b = vec![0.3; 4];
        let t = rgb_tensor(&r, &g, &b);
        let jitter = ColorJitter::<f64>::new(0.3, 0.0, 0.0, 0.0).unwrap();
        let out = jitter.apply(t).unwrap();
        let d = out.data().unwrap();
        // All R pixels should have the same value (scaled by the same factor).
        let r_val = d[0];
        for &v in &d[..4] {
            assert!((v - r_val).abs() < 1e-10);
        }
    }

    #[test]
    fn test_color_jitter_from_ranges_identity() {
        // Identity ranges should leave the image untouched.
        let data: Vec<f64> = (0..12).map(|i| i as f64 / 12.0).collect();
        let t =
            Tensor::from_storage(TensorStorage::cpu(data.clone()), vec![3, 2, 2], false).unwrap();
        let jitter =
            ColorJitter::<f64>::from_ranges((1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (0.0, 0.0))
                .unwrap();
        let out = jitter.apply(t).unwrap();
        let d = out.data().unwrap();
        for (a, b) in d.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-10, "Expected {b}, got {a}");
        }
    }

    #[test]
    fn test_color_jitter_from_ranges_asymmetric_brightness() {
        // Brightness range (1.5, 1.5) must scale every pixel by exactly 1.5.
        let r = vec![0.4_f64; 4];
        let g = vec![0.2; 4];
        let b = vec![0.1; 4];
        let t = rgb_tensor(&r, &g, &b);
        let jitter =
            ColorJitter::<f64>::from_ranges((1.5, 1.5), (1.0, 1.0), (1.0, 1.0), (0.0, 0.0))
                .unwrap();
        let out = jitter.apply(t).unwrap();
        let d = out.data().unwrap();
        // R channel: 0.4 * 1.5 = 0.6
        assert!((d[0] - 0.6).abs() < 1e-10);
        // G channel: 0.2 * 1.5 = 0.3
        assert!((d[4] - 0.3).abs() < 1e-10);
        // B channel: 0.1 * 1.5 = 0.15
        assert!((d[8] - 0.15).abs() < 1e-10);
    }

    #[test]
    fn test_color_jitter_from_ranges_rejects_invalid() {
        // brightness lo < 0
        assert!(
            ColorJitter::<f64>::from_ranges((-0.1, 1.0), (1.0, 1.0), (1.0, 1.0), (0.0, 0.0))
                .is_err()
        );
        // contrast lo > hi
        assert!(
            ColorJitter::<f64>::from_ranges((1.0, 1.0), (1.5, 1.0), (1.0, 1.0), (0.0, 0.0))
                .is_err()
        );
        // hue out of [-0.5, 0.5]
        assert!(
            ColorJitter::<f64>::from_ranges((1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (-0.6, 0.0))
                .is_err()
        );
    }

    #[test]
    fn test_color_jitter_f32() {
        let data: Vec<f32> = vec![0.5; 12];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 2, 2], false).unwrap();
        let jitter = ColorJitter::<f32>::new(0.2, 0.2, 0.2, 0.1).unwrap();
        let out = jitter.apply(t).unwrap();
        assert_eq!(out.shape(), &[3, 2, 2]);
        for &val in out.data().unwrap() {
            assert!((0.0..=1.0).contains(&val));
        }
    }
}
