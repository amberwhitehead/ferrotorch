//! RandomCrop — randomly crop a [C, H, W] tensor to a target size.
//!
//! ## REQ status (per `.design/ferrotorch-vision/transforms/random_crop.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `pub struct RandomCrop<T: Float>` with `crop_h: usize`, `crop_w: usize`, and `_marker: PhantomData<T>` in `random_crop.rs`, mirroring `torchvision/transforms/v2/_geometry.py:759` `class RandomCrop(Transform)`; consumer: `pub use random_crop::RandomCrop;` in `mod.rs`. |
//! | REQ-2 | SHIPPED | `pub fn RandomCrop::new(crop_h: usize, crop_w: usize) -> Self` constructor in `random_crop.rs`; consumer: reachable through the `mod.rs` re-export. |
//! | REQ-3 | SHIPPED | `pub fn RandomCrop::square(size: usize) -> Self` convenience constructor in `random_crop.rs`; consumer: reachable through the `mod.rs` re-export — user code calls `RandomCrop::square(224)` for the canonical square-crop ergonomics. |
//! | REQ-4 | SHIPPED | `impl<T: Float> Transform<T> for RandomCrop<T>` with shape, bounds, `random_usize`-sampled top-left corner, and region-copy in `random_crop.rs`; consumer: any `Box<dyn Transform<T>>` slot composes the type into `Compose<T>` pipelines via the `mod.rs` re-export. |
//! | REQ-5 | SHIPPED | `with_padding(usize) / with_padding_hw / with_fill / with_pad_if_needed` builders in `random_crop.rs` plus `apply` pad-then-crop dispatch; consumer: reachable through the `mod.rs` re-export — user pipelines composed into `Compose<T>` invoke `RandomCrop::new(h, w).with_padding(4).with_fill(0.0)` for the canonical CIFAR-style "pad-then-crop" augmentation. |

use super::rng::random_usize;
use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_data::Transform;

/// Randomly crop a `[C, H, W]` tensor to `(crop_h, crop_w)`.
///
/// A random top-left corner is chosen uniformly. If the input is smaller
/// than the crop size and `pad_if_needed` is false, an error is returned.
/// When padding is configured, the input is padded with `fill` (constant
/// fill, mirroring upstream's `padding_mode='constant'`) before the random
/// corner is drawn.
///
/// Matches `torchvision.transforms.RandomCrop` for the constant-fill
/// padding case (`padding_mode='constant'`).
pub struct RandomCrop<T: Float> {
    crop_h: usize,
    crop_w: usize,
    pad_top: usize,
    pad_bottom: usize,
    pad_left: usize,
    pad_right: usize,
    fill: f64,
    pad_if_needed: bool,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> RandomCrop<T> {
    pub fn new(crop_h: usize, crop_w: usize) -> Self {
        Self {
            crop_h,
            crop_w,
            pad_top: 0,
            pad_bottom: 0,
            pad_left: 0,
            pad_right: 0,
            fill: 0.0,
            pad_if_needed: false,
            _marker: std::marker::PhantomData,
        }
    }

    /// Square crop.
    pub fn square(size: usize) -> Self {
        Self::new(size, size)
    }

    /// Apply uniform `padding` on all four sides before cropping. Mirrors
    /// upstream `RandomCrop(padding=p)` integer-int form
    /// (`torchvision/transforms/v2/_geometry.py:759-913`).
    pub fn with_padding(mut self, padding: usize) -> Self {
        self.pad_top = padding;
        self.pad_bottom = padding;
        self.pad_left = padding;
        self.pad_right = padding;
        self
    }

    /// Apply distinct vertical (`pad_h`) and horizontal (`pad_w`) padding.
    /// Mirrors upstream's 2-tuple form `padding=(pad_w, pad_h)`.
    pub fn with_padding_hw(mut self, pad_h: usize, pad_w: usize) -> Self {
        self.pad_top = pad_h;
        self.pad_bottom = pad_h;
        self.pad_left = pad_w;
        self.pad_right = pad_w;
        self
    }

    /// Set the constant fill value applied during padding. Mirrors upstream
    /// `fill: number` for `padding_mode='constant'`.
    pub fn with_fill(mut self, fill: f64) -> Self {
        self.fill = fill;
        self
    }

    /// When `true`, automatically pads the input if it is smaller than the
    /// crop size along either dimension. Mirrors upstream
    /// `pad_if_needed=True` semantics.
    pub fn with_pad_if_needed(mut self, pad_if_needed: bool) -> Self {
        self.pad_if_needed = pad_if_needed;
        self
    }
}

impl<T: Float> Transform<T> for RandomCrop<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape().to_vec();
        if shape.len() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "RandomCrop: expected 3-D tensor [C, H, W], got shape {:?}",
                    shape
                ),
            });
        }

        let c = shape[0];
        let in_h = shape[1];
        let in_w = shape[2];

        // Initial explicit padding from `with_padding*`.
        let mut pad_top = self.pad_top;
        let mut pad_bottom = self.pad_bottom;
        let mut pad_left = self.pad_left;
        let mut pad_right = self.pad_right;

        // Effective input size after explicit padding.
        let mut h_eff = in_h + pad_top + pad_bottom;
        let mut w_eff = in_w + pad_left + pad_right;

        // pad_if_needed: top up to meet the crop size.
        if self.pad_if_needed {
            if h_eff < self.crop_h {
                let extra = self.crop_h - h_eff;
                pad_top += extra.div_ceil(2);
                pad_bottom += extra / 2;
                h_eff = self.crop_h;
            }
            if w_eff < self.crop_w {
                let extra = self.crop_w - w_eff;
                pad_left += extra.div_ceil(2);
                pad_right += extra / 2;
                w_eff = self.crop_w;
            }
        }

        if h_eff < self.crop_h || w_eff < self.crop_w {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "RandomCrop: padded input ({h_eff}x{w_eff}) is smaller than crop size ({}x{})",
                    self.crop_h, self.crop_w
                ),
            });
        }

        let any_pad = pad_top != 0 || pad_bottom != 0 || pad_left != 0 || pad_right != 0;
        let fill_t: T = cast::<f64, T>(self.fill)?;

        let top = if h_eff == self.crop_h {
            0
        } else {
            random_usize(h_eff - self.crop_h)
        };
        let left = if w_eff == self.crop_w {
            0
        } else {
            random_usize(w_eff - self.crop_w)
        };

        let data = input.data()?;
        let mut out = Vec::with_capacity(c * self.crop_h * self.crop_w);

        if any_pad {
            // Coordinates in the padded image map back into the source
            // tensor by subtracting (pad_top, pad_left); coordinates
            // outside that range are fill values.
            for ch in 0..c {
                for row in top..top + self.crop_h {
                    for col in left..left + self.crop_w {
                        let src_row = row as isize - pad_top as isize;
                        let src_col = col as isize - pad_left as isize;
                        if src_row >= 0
                            && src_col >= 0
                            && (src_row as usize) < in_h
                            && (src_col as usize) < in_w
                        {
                            out.push(
                                data[ch * in_h * in_w + src_row as usize * in_w + src_col as usize],
                            );
                        } else {
                            out.push(fill_t);
                        }
                    }
                }
            }
        } else {
            for ch in 0..c {
                for row in top..top + self.crop_h {
                    for col in left..left + self.crop_w {
                        out.push(data[ch * in_h * in_w + row * in_w + col]);
                    }
                }
            }
        }

        Tensor::from_storage(
            TensorStorage::cpu(out),
            vec![c, self.crop_h, self.crop_w],
            false,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transforms::rng::vision_manual_seed;

    #[test]
    fn test_random_crop_shape() {
        let crop: RandomCrop<f32> = RandomCrop::new(2, 3);
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![0.0f32; 3 * 5 * 7]),
            vec![3, 5, 7],
            false,
        )
        .unwrap();
        let out = crop.apply(input).unwrap();
        assert_eq!(out.shape(), &[3, 2, 3]);
    }

    #[test]
    fn test_random_crop_exact_size() {
        let crop: RandomCrop<f32> = RandomCrop::square(3);
        let data: Vec<f32> = (0..27).map(|i| i as f32).collect();
        let input =
            Tensor::from_storage(TensorStorage::cpu(data.clone()), vec![3, 3, 3], false).unwrap();
        let out = crop.apply(input).unwrap();
        // Exact size — no cropping needed, should return same data.
        assert_eq!(out.shape(), &[3, 3, 3]);
        assert_eq!(out.data().unwrap(), &data[..]);
    }

    #[test]
    fn test_random_crop_too_small() {
        let crop: RandomCrop<f32> = RandomCrop::new(10, 10);
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![0.0f32; 3 * 5 * 5]),
            vec![3, 5, 5],
            false,
        )
        .unwrap();
        assert!(crop.apply(input).is_err());
    }

    #[test]
    fn test_random_crop_with_padding_shape() {
        // CIFAR-style: pad by 4 then crop 32x32 from a 32x32 input — must work
        // because effective size after padding is 40x40.
        vision_manual_seed(123);
        let crop: RandomCrop<f32> = RandomCrop::square(32).with_padding(4);
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32; 3 * 32 * 32]),
            vec![3, 32, 32],
            false,
        )
        .unwrap();
        let out = crop.apply(input).unwrap();
        assert_eq!(out.shape(), &[3, 32, 32]);
    }

    #[test]
    fn test_random_crop_pad_if_needed_handles_small_input() {
        // 5x5 input, crop 10x10 — without `pad_if_needed` this errors; with
        // it the input is padded to 10x10 and the crop succeeds.
        vision_manual_seed(7);
        let crop: RandomCrop<f64> = RandomCrop::new(10, 10)
            .with_pad_if_needed(true)
            .with_fill(0.5);
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f64; 3 * 5 * 5]),
            vec![3, 5, 5],
            false,
        )
        .unwrap();
        let out = crop.apply(input).unwrap();
        assert_eq!(out.shape(), &[3, 10, 10]);
        // At least one fill pixel must be present (padding > 0).
        let d = out.data().unwrap();
        let mut saw_fill = false;
        let mut saw_one = false;
        for &v in d {
            if (v - 0.5).abs() < 1e-12 {
                saw_fill = true;
            } else if (v - 1.0).abs() < 1e-12 {
                saw_one = true;
            }
        }
        assert!(saw_fill, "expected fill=0.5 pixels");
        assert!(saw_one, "expected source-data pixels");
    }

    #[test]
    fn test_random_crop_with_fill_value_appears_in_border() {
        // Pad 1 on all sides; crop the entire padded image. The 4 corners
        // are guaranteed to be fill values.
        let crop: RandomCrop<f64> = RandomCrop::new(3, 3).with_padding(1).with_fill(-7.0);
        // 1x1 input with value 42 — padded becomes 3x3 with center=42,
        // corners=-7.
        let input =
            Tensor::from_storage(TensorStorage::cpu(vec![42.0f64]), vec![1, 1, 1], false).unwrap();
        let out = crop.apply(input).unwrap();
        let d = out.data().unwrap();
        // 3x3 layout (single channel): corners=-7, center=42, edges=-7
        assert_eq!(d[0], -7.0); // top-left
        assert_eq!(d[2], -7.0); // top-right
        assert_eq!(d[4], 42.0); // center
        assert_eq!(d[6], -7.0); // bottom-left
        assert_eq!(d[8], -7.0); // bottom-right
    }
}
