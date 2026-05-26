// CL-332: Vision Transforms & Augmentation — RandomGaussianBlur
//! ## REQ status (per `.design/ferrotorch-vision/transforms/random_gaussian_blur.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `pub struct RandomGaussianBlur<T: Float>` with `kernel_size: usize`, `sigma_lo: f64`, `sigma_hi: f64`, and `_marker: PhantomData<T>` in `random_gaussian_blur.rs`, mirroring `torchvision/transforms/v2/_misc.py:177` `class GaussianBlur`; consumer: `pub use random_gaussian_blur::RandomGaussianBlur;` in `mod.rs` and `RandomGaussianBlur` in the crate-root re-export in `lib.rs`. |
//! | REQ-2 | SHIPPED | `pub fn RandomGaussianBlur::new(kernel_size: usize, sigma: (f64, f64)) -> FerrotorchResult<Self>` constructor with odd-positive kernel and sigma-ordering checks in `random_gaussian_blur.rs`; consumer: registered in the conformance surface inventory at `tests/conformance/_surface_inventory.toml`; reachable through the crate-root re-export. |
//! | REQ-3 | SHIPPED | `fn gaussian_kernel_1d(size: usize, sigma: f64) -> Vec<f64>` helper in `random_gaussian_blur.rs`; consumer: the impl in the same file calls `gaussian_kernel_1d(self.kernel_size, sigma)` inside the per-channel blur path. |
//! | REQ-4 | SHIPPED | `fn blur_rows(data, h, w, kernel) -> Vec<f64>` and `fn blur_cols(...)` separable-convolution helpers in `random_gaussian_blur.rs`; consumer: the impl chains `blur_cols(blur_rows(...))` per channel. |
//! | REQ-5 | SHIPPED | `impl<T: Float> Transform<T> for RandomGaussianBlur<T>` in `random_gaussian_blur.rs`; consumer: any `Box<dyn Transform<T>>` slot — composes into augmentation `Compose` pipelines via the `lib.rs` re-export. |
//! | REQ-6 | SHIPPED | `fn reflect_index` + `blur_rows` / `blur_cols` reflection-padded convolution in `random_gaussian_blur.rs`; consumer: the `impl Transform` in the same file calls `blur_cols(blur_rows(...))` per channel inside the apply path. Matches upstream default `padding_mode='reflect'` from `torchvision/transforms/v2/functional/_misc.py:gaussian_blur`. |

use super::rng::random_f64;
use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_data::Transform;

/// Apply Gaussian blur with a random sigma to a `[C, H, W]` tensor.
///
/// The kernel size is fixed at construction. The actual sigma is sampled
/// uniformly from `[sigma_lo, sigma_hi]` each time the transform is applied.
/// The Gaussian kernel is computed on-the-fly and applied as a separable
/// (row-then-column) convolution per channel.
///
/// This mirrors `torchvision.transforms.GaussianBlur`.
pub struct RandomGaussianBlur<T: Float> {
    kernel_size: usize,
    sigma_lo: f64,
    sigma_hi: f64,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> RandomGaussianBlur<T> {
    /// Create a new `RandomGaussianBlur`.
    ///
    /// * `kernel_size` — must be odd and >= 1.
    /// * `sigma` — `(lo, hi)`, the range from which sigma is sampled.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if `kernel_size` is even
    /// or zero, or if `sigma` is not positive with `lo <= hi`.
    pub fn new(kernel_size: usize, sigma: (f64, f64)) -> FerrotorchResult<Self> {
        if !(kernel_size >= 1 && kernel_size % 2 == 1) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "RandomGaussianBlur: kernel_size must be odd and >= 1, got {kernel_size}"
                ),
            });
        }
        if !(sigma.0 > 0.0 && sigma.0 <= sigma.1) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "RandomGaussianBlur: sigma must satisfy 0 < lo <= hi, got ({}, {})",
                    sigma.0, sigma.1,
                ),
            });
        }
        Ok(Self {
            kernel_size,
            sigma_lo: sigma.0,
            sigma_hi: sigma.1,
            _marker: std::marker::PhantomData,
        })
    }
}

/// Compute a 1-D Gaussian kernel of the given size and sigma, normalized to
/// sum to 1.
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

/// Reflect-pad an integer index against a `[0, size)` range. Mirrors
/// PyTorch's `padding_mode='reflect'` semantics: indices outside the range
/// bounce off the edges without repeating them, e.g. for size=5:
/// `-1 -> 1`, `-2 -> 2`, `5 -> 3`, `6 -> 2`.
fn reflect_index(i: i64, size: usize) -> usize {
    if size == 1 {
        return 0;
    }
    let n = size as i64;
    let period = 2 * (n - 1);
    let mut m = i.rem_euclid(period);
    if m >= n {
        m = period - m;
    }
    m as usize
}

/// Apply a 1-D convolution along rows (horizontal blur) with reflection
/// padding. Mirrors upstream torchvision `gaussian_blur` default
/// `padding_mode='reflect'`.
fn blur_rows(data: &[f64], h: usize, w: usize, kernel: &[f64]) -> Vec<f64> {
    let half = kernel.len() / 2;
    let mut out = vec![0.0; h * w];

    for row in 0..h {
        let row_off = row * w;
        for col in 0..w {
            let mut acc = 0.0;
            for (ki, &kv) in kernel.iter().enumerate() {
                let src_col = col as i64 + ki as i64 - half as i64;
                let idx = reflect_index(src_col, w);
                acc += data[row_off + idx] * kv;
            }
            out[row_off + col] = acc;
        }
    }
    out
}

/// Apply a 1-D convolution along columns (vertical blur) with reflection
/// padding.
fn blur_cols(data: &[f64], h: usize, w: usize, kernel: &[f64]) -> Vec<f64> {
    let half = kernel.len() / 2;
    let mut out = vec![0.0; h * w];

    for row in 0..h {
        for col in 0..w {
            let mut acc = 0.0;
            for (ki, &kv) in kernel.iter().enumerate() {
                let src_row = row as i64 + ki as i64 - half as i64;
                let idx = reflect_index(src_row, h);
                acc += data[idx * w + col] * kv;
            }
            out[row * w + col] = acc;
        }
    }
    out
}

impl<T: Float> Transform<T> for RandomGaussianBlur<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape().to_vec();
        if shape.len() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "RandomGaussianBlur: expected 3-D tensor [C, H, W], got shape {:?}",
                    shape
                ),
            });
        }

        let channels = shape[0];
        let h = shape[1];
        let w = shape[2];

        // Sample sigma.
        let sigma = self.sigma_lo + random_f64() * (self.sigma_hi - self.sigma_lo);
        let kernel = gaussian_kernel_1d(self.kernel_size, sigma);

        let data = input.data()?;
        let mut output = Vec::with_capacity(data.len());

        for c in 0..channels {
            let ch_data: Vec<f64> = data[c * h * w..(c + 1) * h * w]
                .iter()
                .map(|v| v.to_f64().unwrap())
                .collect();

            // Separable: horizontal then vertical.
            let blurred_h = blur_rows(&ch_data, h, w, &kernel);
            let blurred = blur_cols(&blurred_h, h, w, &kernel);

            for &v in &blurred {
                output.push(cast::<f64, T>(v)?);
            }
        }

        let storage = TensorStorage::cpu(output);
        Tensor::from_storage(storage, shape, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_blur_output_shape() {
        let data: Vec<f64> = vec![0.5; 48]; // 3x4x4
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 4, 4], false).unwrap();
        let blur = RandomGaussianBlur::<f64>::new(3, (0.1, 2.0)).unwrap();
        let out = blur.apply(t).unwrap();
        assert_eq!(out.shape(), &[3, 4, 4]);
    }

    #[test]
    fn test_gaussian_blur_uniform_image() {
        // With reflection padding, a uniform image stays uniform EVERYWHERE
        // (kernel weights sum to 1; reflected samples are still 0.7).
        let data: Vec<f64> = vec![0.7; 75]; // 3x5x5
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 5, 5], false).unwrap();
        let blur = RandomGaussianBlur::<f64>::new(3, (1.0, 1.0)).unwrap();
        let out = blur.apply(t).unwrap();
        let d = out.data().unwrap();
        for (i, &val) in d.iter().enumerate() {
            assert!(
                (val - 0.7).abs() < 1e-10,
                "Expected ~0.7 at pixel {i}, got {val}"
            );
        }
    }

    #[test]
    fn test_reflect_index_canonical_cases() {
        // size=5: indices 0..4 inclusive. Reflection: -1->1, -2->2, 5->3, 6->2.
        assert_eq!(reflect_index(0, 5), 0);
        assert_eq!(reflect_index(4, 5), 4);
        assert_eq!(reflect_index(-1, 5), 1);
        assert_eq!(reflect_index(-2, 5), 2);
        assert_eq!(reflect_index(5, 5), 3);
        assert_eq!(reflect_index(6, 5), 2);
        // size=1: always 0.
        assert_eq!(reflect_index(-3, 1), 0);
        assert_eq!(reflect_index(7, 1), 0);
    }

    #[test]
    fn test_gaussian_blur_border_pixels_not_dimmed() {
        // With reflection padding, border pixels in a constant image stay
        // at the input value — they don't get dimmed toward zero like with
        // zero-padding. Use a larger kernel to make the effect more obvious
        // if zero-padding were re-introduced.
        let data: Vec<f64> = vec![1.0; 25]; // 1x5x5 all ones
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 5, 5], false).unwrap();
        let blur = RandomGaussianBlur::<f64>::new(5, (1.5, 1.5)).unwrap();
        let out = blur.apply(t).unwrap();
        // Top-left corner pixel — with zero padding this would be ~0.6;
        // with reflection padding it stays at 1.0.
        let val = out.data().unwrap()[0];
        assert!(
            (val - 1.0).abs() < 1e-10,
            "Expected corner value 1.0 with reflection padding, got {val}"
        );
    }

    #[test]
    fn test_gaussian_blur_smooths_impulse() {
        // A single bright pixel in the center should spread out.
        let mut data = vec![0.0_f64; 25]; // 1x5x5
        data[12] = 1.0; // center
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 5, 5], false).unwrap();
        let blur = RandomGaussianBlur::<f64>::new(5, (1.0, 1.0)).unwrap();
        let out = blur.apply(t).unwrap();
        let d = out.data().unwrap();
        // Center should still be the brightest but less than 1.0.
        assert!(d[12] < 1.0, "Center should be reduced, got {}", d[12]);
        assert!(d[12] > 0.0, "Center should still be positive");
        // Neighbors should be non-zero.
        assert!(d[11] > 0.0, "Left neighbor should be non-zero");
        assert!(d[13] > 0.0, "Right neighbor should be non-zero");
        assert!(d[7] > 0.0, "Top neighbor should be non-zero");
        assert!(d[17] > 0.0, "Bottom neighbor should be non-zero");
    }

    #[test]
    fn test_gaussian_kernel_1d_sums_to_one() {
        let k = gaussian_kernel_1d(5, 1.0);
        let sum: f64 = k.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Kernel should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_gaussian_kernel_1d_symmetry() {
        let k = gaussian_kernel_1d(7, 2.0);
        let n = k.len();
        for i in 0..n / 2 {
            assert!(
                (k[i] - k[n - 1 - i]).abs() < 1e-15,
                "Kernel should be symmetric: k[{i}]={} != k[{}]={}",
                k[i],
                n - 1 - i,
                k[n - 1 - i]
            );
        }
    }

    #[test]
    fn test_gaussian_blur_rejects_non_3d() {
        let data = vec![0.5_f64; 4];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![2, 2], false).unwrap();
        let blur = RandomGaussianBlur::<f64>::new(3, (0.1, 2.0)).unwrap();
        assert!(blur.apply(t).is_err());
    }

    #[test]
    fn test_gaussian_blur_f32() {
        let data: Vec<f32> = vec![0.5; 12]; // 3x2x2
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 2, 2], false).unwrap();
        let blur = RandomGaussianBlur::<f32>::new(3, (0.5, 1.5)).unwrap();
        let out = blur.apply(t).unwrap();
        assert_eq!(out.shape(), &[3, 2, 2]);
    }
}
