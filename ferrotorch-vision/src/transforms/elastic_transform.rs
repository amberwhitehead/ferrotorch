//! ElasticTransform: elastic deformation via a smoothed random
//! displacement field.
//!
//! Implements the elastic deformation described in Simard et al. 2003
//! ("Best Practices for Convolutional Neural Networks Applied to Visual
//! Document Analysis"). A per-pixel displacement field is sampled from
//! a uniform distribution, smoothed with a Gaussian, scaled by `alpha`,
//! and used to sample the source image via bilinear interpolation.
//!
//! Mirrors `torchvision.transforms.v2.ElasticTransform`. CL-458.
//!
//! ## REQ status (per `.design/ferrotorch-vision/transforms/elastic_transform.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `pub struct ElasticTransform<T: Float>` with `alpha: f64`, `sigma: f64`, and `_marker: PhantomData<T>` in `elastic_transform.rs`, mirroring `torchvision/transforms/v2/_geometry.py:999` `class ElasticTransform`; consumer: `pub use elastic_transform::ElasticTransform;` in `mod.rs`. |
//! | REQ-2 | SHIPPED | `pub fn ElasticTransform::new(alpha: f64, sigma: f64) -> FerrotorchResult<Self>` with `alpha >= 0` and `sigma > 0` validation in `elastic_transform.rs`; consumer: reachable through the `mod.rs` re-export. |
//! | REQ-3 | SHIPPED | `fn gaussian_kernel_1d` and `fn gaussian_filter_2d` helpers in `elastic_transform.rs`; consumer: the impl in the same file calls `gaussian_filter_2d(&dy_field, h, w, self.sigma)` and the dx-field counterpart inside the displacement-smoothing path. |
//! | REQ-4 | SHIPPED | `fn bilinear_sample(data, h, w, y, x) -> f64` with clamp-to-edge in `elastic_transform.rs`; consumer: the impl calls `bilinear_sample(&ch_data, h, w, src_y, src_x)` inside the per-output-pixel resampling loop. |
//! | REQ-5 | SHIPPED | `impl<T: Float> Transform<T> for ElasticTransform<T>` with shape/dim checks, random-field generation, Gaussian smoothing, and per-channel bilinear sampling in `elastic_transform.rs`; consumer: any `Box<dyn Transform<T>>` slot — composes into augmentation `Compose` pipelines via the `mod.rs` re-export. |
//! | REQ-6 | SHIPPED | `with_interpolation / with_fill / new_range` builders + nearest/bilinear+fill sampler dispatch in `elastic_transform.rs`; consumer: pipelines call `ElasticTransform::new_range((0.0, 60.0), (3.0, 7.0))?.with_interpolation(InterpolationMode::Nearest).with_fill(0.0)` per upstream `_geometry.py:999-1090`. |

use super::resize::InterpolationMode;
use super::rng::random_f64;
use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_data::Transform;

/// Elastic deformation via a Gaussian-smoothed random displacement field.
///
/// The strength of the deformation is controlled by `alpha` (scale of
/// the displacement field in pixels) and `sigma` (standard deviation of
/// the Gaussian smoother applied to the random field before scaling).
/// Larger `alpha` → bigger distortions; larger `sigma` → smoother,
/// more global distortions. A typical starting point is
/// `alpha=50, sigma=5`.
pub struct ElasticTransform<T: Float> {
    alpha_lo: f64,
    alpha_hi: f64,
    sigma_lo: f64,
    sigma_hi: f64,
    interpolation: InterpolationMode,
    fill: Option<f64>,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> ElasticTransform<T> {
    /// Create a new `ElasticTransform` with scalar `alpha` and `sigma`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if `alpha < 0` or
    /// `sigma <= 0`.
    pub fn new(alpha: f64, sigma: f64) -> FerrotorchResult<Self> {
        Self::new_range((alpha, alpha), (sigma, sigma))
    }

    /// Create a new `ElasticTransform` where each application samples
    /// `alpha` uniformly from `[alpha.0, alpha.1]` and `sigma` from
    /// `[sigma.0, sigma.1]`. Mirrors upstream
    /// `ElasticTransform(alpha=(lo, hi), sigma=(lo, hi))`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if either range is
    /// invalid.
    pub fn new_range(alpha: (f64, f64), sigma: (f64, f64)) -> FerrotorchResult<Self> {
        if !(alpha.0 >= 0.0 && alpha.0 <= alpha.1) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ElasticTransform: alpha must satisfy 0 <= lo <= hi, got ({}, {})",
                    alpha.0, alpha.1
                ),
            });
        }
        if !(sigma.0 > 0.0 && sigma.0 <= sigma.1) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ElasticTransform: sigma must satisfy 0 < lo <= hi, got ({}, {})",
                    sigma.0, sigma.1
                ),
            });
        }
        Ok(Self {
            alpha_lo: alpha.0,
            alpha_hi: alpha.1,
            sigma_lo: sigma.0,
            sigma_hi: sigma.1,
            interpolation: InterpolationMode::Bilinear,
            fill: None,
            _marker: std::marker::PhantomData,
        })
    }

    /// Select interpolation mode. Default is bilinear.
    pub fn with_interpolation(mut self, mode: InterpolationMode) -> Self {
        self.interpolation = mode;
        self
    }

    /// Use the given constant fill value for out-of-bounds samples.
    /// Default behaviour (when not set) is clamp-to-edge.
    pub fn with_fill(mut self, fill: f64) -> Self {
        self.fill = Some(fill);
        self
    }
}

/// Compute a 1-D Gaussian kernel, normalized.
fn gaussian_kernel_1d(size: usize, sigma: f64) -> Vec<f64> {
    let half = (size / 2) as i64;
    let mut kernel = Vec::with_capacity(size);
    let mut sum = 0.0_f64;
    for i in 0..size {
        let x = (i as i64 - half) as f64;
        let val = (-0.5 * (x / sigma).powi(2)).exp();
        kernel.push(val);
        sum += val;
    }
    for v in kernel.iter_mut() {
        *v /= sum;
    }
    kernel
}

/// Separable 2-D Gaussian filter (rows, then cols) on a flat HxW buffer.
/// Uses zero-padding at the borders.
fn gaussian_filter_2d(data: &[f64], h: usize, w: usize, sigma: f64) -> Vec<f64> {
    // Kernel radius of ~3*sigma covers ~99.7% of the mass.
    let radius = (3.0 * sigma).ceil() as usize;
    let size = 2 * radius + 1;
    let kernel = gaussian_kernel_1d(size, sigma);
    let half = size / 2;

    // Horizontal pass.
    let mut tmp = vec![0.0_f64; h * w];
    for row in 0..h {
        for col in 0..w {
            let mut acc = 0.0;
            for (ki, &kv) in kernel.iter().enumerate() {
                let src_col = col as i64 + ki as i64 - half as i64;
                if src_col >= 0 && (src_col as usize) < w {
                    acc += data[row * w + src_col as usize] * kv;
                }
            }
            tmp[row * w + col] = acc;
        }
    }
    // Vertical pass.
    let mut out = vec![0.0_f64; h * w];
    for row in 0..h {
        for col in 0..w {
            let mut acc = 0.0;
            for (ki, &kv) in kernel.iter().enumerate() {
                let src_row = row as i64 + ki as i64 - half as i64;
                if src_row >= 0 && (src_row as usize) < h {
                    acc += tmp[src_row as usize * w + col] * kv;
                }
            }
            out[row * w + col] = acc;
        }
    }
    out
}

/// Bilinear interpolation of a single-channel `[H, W]` image at
/// fractional coordinates `(y, x)`. Out-of-bounds samples are
/// clamped to the nearest edge (torchvision's default).
fn bilinear_sample(data: &[f64], h: usize, w: usize, y: f64, x: f64) -> f64 {
    // Clamp to [0, h-1] x [0, w-1].
    let y = y.clamp(0.0, (h - 1) as f64);
    let x = x.clamp(0.0, (w - 1) as f64);
    let y0 = y.floor() as usize;
    let x0 = x.floor() as usize;
    let y1 = (y0 + 1).min(h - 1);
    let x1 = (x0 + 1).min(w - 1);
    let dy = y - y0 as f64;
    let dx = x - x0 as f64;

    let v00 = data[y0 * w + x0];
    let v01 = data[y0 * w + x1];
    let v10 = data[y1 * w + x0];
    let v11 = data[y1 * w + x1];

    // Interpolate along x first, then y.
    let top = v00 * (1.0 - dx) + v01 * dx;
    let bot = v10 * (1.0 - dx) + v11 * dx;
    top * (1.0 - dy) + bot * dy
}

/// Bilinear sample with an out-of-bounds `fill` value.
fn bilinear_sample_with_fill(data: &[f64], h: usize, w: usize, y: f64, x: f64, fill: f64) -> f64 {
    if x < 0.0 || y < 0.0 || x > (w - 1) as f64 || y > (h - 1) as f64 {
        return fill;
    }
    bilinear_sample(data, h, w, y, x)
}

/// Nearest-neighbor sample. Uses `fill` for out-of-bounds when `fill` is
/// `Some`, else clamp-to-edge.
fn nearest_sample(data: &[f64], h: usize, w: usize, y: f64, x: f64, fill: Option<f64>) -> f64 {
    let xr = x.round();
    let yr = y.round();
    if let Some(f) = fill {
        if xr < 0.0 || yr < 0.0 || xr > (w - 1) as f64 || yr > (h - 1) as f64 {
            return f;
        }
    }
    let xi = xr.clamp(0.0, (w - 1) as f64) as usize;
    let yi = yr.clamp(0.0, (h - 1) as f64) as usize;
    data[yi * w + xi]
}

impl<T: Float> Transform<T> for ElasticTransform<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape().to_vec();
        if shape.len() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("ElasticTransform: expected 3-D tensor [C, H, W], got {shape:?}"),
            });
        }
        let c = shape[0];
        let h = shape[1];
        let w = shape[2];
        if h == 0 || w == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "ElasticTransform: image dimensions must be > 0".into(),
            });
        }
        // Sample per-call alpha and sigma from their (possibly degenerate)
        // ranges.
        let alpha = if self.alpha_lo == self.alpha_hi {
            self.alpha_lo
        } else {
            self.alpha_lo + random_f64() * (self.alpha_hi - self.alpha_lo)
        };
        let sigma = if self.sigma_lo == self.sigma_hi {
            self.sigma_lo
        } else {
            self.sigma_lo + random_f64() * (self.sigma_hi - self.sigma_lo)
        };

        if alpha == 0.0 {
            // Zero alpha means no displacement — identity transform.
            return Ok(input);
        }

        // Generate random displacement fields dx, dy in [-1, 1],
        // smooth with Gaussian, then scale by alpha.
        let numel = h * w;
        let mut dy_field = Vec::with_capacity(numel);
        let mut dx_field = Vec::with_capacity(numel);
        for _ in 0..numel {
            dy_field.push(2.0 * random_f64() - 1.0);
            dx_field.push(2.0 * random_f64() - 1.0);
        }
        let dy_field = gaussian_filter_2d(&dy_field, h, w, sigma);
        let dx_field = gaussian_filter_2d(&dx_field, h, w, sigma);
        // Scale by alpha.
        let dy_field: Vec<f64> = dy_field.iter().map(|v| v * alpha).collect();
        let dx_field: Vec<f64> = dx_field.iter().map(|v| v * alpha).collect();

        // Apply per channel.
        let data = input.data()?;
        let mut output = Vec::with_capacity(data.len());
        for ch in 0..c {
            let ch_data: Vec<f64> = data[ch * h * w..(ch + 1) * h * w]
                .iter()
                .map(|v| v.to_f64().unwrap())
                .collect();
            for row in 0..h {
                for col in 0..w {
                    let src_y = row as f64 + dy_field[row * w + col];
                    let src_x = col as f64 + dx_field[row * w + col];
                    let val = match self.interpolation {
                        InterpolationMode::Nearest => {
                            nearest_sample(&ch_data, h, w, src_y, src_x, self.fill)
                        }
                        InterpolationMode::Bilinear => match self.fill {
                            None => bilinear_sample(&ch_data, h, w, src_y, src_x),
                            Some(f) => bilinear_sample_with_fill(&ch_data, h, w, src_y, src_x, f),
                        },
                    };
                    output.push(cast::<f64, T>(val)?);
                }
            }
        }

        Tensor::from_storage(TensorStorage::cpu(output), shape, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transforms::rng::vision_manual_seed;

    #[test]
    fn test_elastic_output_shape_preserved() {
        let t: Tensor<f32> =
            Tensor::from_storage(TensorStorage::cpu(vec![0.5; 48]), vec![3, 4, 4], false).unwrap();
        let et = ElasticTransform::<f32>::new(5.0, 1.5).unwrap();
        let out = et.apply(t).unwrap();
        assert_eq!(out.shape(), &[3, 4, 4]);
    }

    #[test]
    fn test_elastic_zero_alpha_is_identity() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let t: Tensor<f32> =
            Tensor::from_storage(TensorStorage::cpu(data.clone()), vec![3, 2, 2], false).unwrap();
        let et = ElasticTransform::<f32>::new(0.0, 1.0).unwrap();
        let out = et.apply(t).unwrap();
        assert_eq!(out.data().unwrap(), data.as_slice());
    }

    #[test]
    fn test_elastic_constant_image_unchanged_interior() {
        // A uniform image should remain uniform after elastic
        // deformation because bilinear-sampling a constant gives
        // that constant everywhere.
        vision_manual_seed(99);
        let t: Tensor<f64> =
            Tensor::from_storage(TensorStorage::cpu(vec![0.7; 100]), vec![1, 10, 10], false)
                .unwrap();
        let et = ElasticTransform::<f64>::new(10.0, 2.0).unwrap();
        let out = et.apply(t).unwrap();
        for &v in out.data().unwrap() {
            assert!((v - 0.7).abs() < 1e-10, "expected 0.7, got {v}");
        }
    }

    #[test]
    fn test_elastic_rejects_non_3d() {
        let t: Tensor<f32> =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0; 4]), vec![2, 2], false).unwrap();
        let et = ElasticTransform::<f32>::new(1.0, 0.5).unwrap();
        assert!(et.apply(t).is_err());
    }

    #[test]
    fn test_elastic_rejects_zero_dim() {
        let t: Tensor<f32> =
            Tensor::from_storage(TensorStorage::cpu(vec![]), vec![3, 0, 4], false).unwrap();
        let et = ElasticTransform::<f32>::new(1.0, 0.5).unwrap();
        assert!(et.apply(t).is_err());
    }

    #[test]
    fn test_elastic_negative_alpha_errors() {
        let err = match ElasticTransform::<f32>::new(-1.0, 1.0) {
            Err(e) => e,
            Ok(_) => panic!("expected error for negative alpha"),
        };
        let msg = format!("{err}");
        assert!(msg.contains("alpha must"), "got: {msg}");
    }

    #[test]
    fn test_elastic_zero_sigma_errors() {
        let err = match ElasticTransform::<f32>::new(1.0, 0.0) {
            Err(e) => e,
            Ok(_) => panic!("expected error for zero sigma"),
        };
        let msg = format!("{err}");
        assert!(msg.contains("sigma must"), "got: {msg}");
    }

    #[test]
    fn test_bilinear_sample_corner() {
        let data = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        // At (0,0), exact corner = 1.0
        assert!((bilinear_sample(&data, 2, 2, 0.0, 0.0) - 1.0).abs() < 1e-10);
        // At (1,1), exact corner = 4.0
        assert!((bilinear_sample(&data, 2, 2, 1.0, 1.0) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_bilinear_sample_midpoint() {
        let data = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        // At (0.5, 0.5) = average of all four = 2.5
        let v = bilinear_sample(&data, 2, 2, 0.5, 0.5);
        assert!((v - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_elastic_new_range_samples_within_band() {
        // A (0, 0) alpha range is a no-op.
        let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let t: Tensor<f64> =
            Tensor::from_storage(TensorStorage::cpu(data.clone()), vec![3, 2, 2], false).unwrap();
        let et = ElasticTransform::<f64>::new_range((0.0, 0.0), (1.0, 2.0)).unwrap();
        let out = et.apply(t).unwrap();
        assert_eq!(out.data().unwrap(), data.as_slice());
    }

    #[test]
    fn test_elastic_with_nearest_yields_only_input_values() {
        // Nearest interpolation must return one of the original input
        // values (no in-between blending).
        let data = vec![0.0_f64, 1.0, 2.0, 3.0]; // 1x2x2
        let t: Tensor<f64> =
            Tensor::from_storage(TensorStorage::cpu(data.clone()), vec![1, 2, 2], false).unwrap();
        let et = ElasticTransform::<f64>::new(5.0, 1.5)
            .unwrap()
            .with_interpolation(InterpolationMode::Nearest);
        let out = et.apply(t).unwrap();
        for &v in out.data().unwrap() {
            assert!(
                data.contains(&v),
                "Nearest must yield input values, got {v}"
            );
        }
    }

    #[test]
    fn test_elastic_with_fill_replaces_oob_samples() {
        // With a large alpha relative to image size, many displacement
        // samples push outside the image. Fill must appear in the output.
        let data = vec![1.0_f64; 25]; // 1x5x5 all ones
        let t: Tensor<f64> =
            Tensor::from_storage(TensorStorage::cpu(data), vec![1, 5, 5], false).unwrap();
        let et = ElasticTransform::<f64>::new(100.0, 1.0)
            .unwrap()
            .with_fill(-7.0);
        let out = et.apply(t).unwrap();
        let saw_fill = out.data().unwrap().iter().any(|&v| v < 0.0);
        assert!(
            saw_fill,
            "expected fill -7.0 to appear after large displacement"
        );
    }

    #[test]
    fn test_elastic_new_range_validates_alpha() {
        assert!(ElasticTransform::<f64>::new_range((-1.0, 1.0), (1.0, 1.0)).is_err());
        assert!(ElasticTransform::<f64>::new_range((2.0, 1.0), (1.0, 1.0)).is_err());
    }

    #[test]
    fn test_bilinear_sample_out_of_bounds_clamps() {
        let data = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        // Outside → clamp to nearest corner.
        assert!((bilinear_sample(&data, 2, 2, -1.0, -1.0) - 1.0).abs() < 1e-10);
        assert!((bilinear_sample(&data, 2, 2, 5.0, 5.0) - 4.0).abs() < 1e-10);
    }
}
