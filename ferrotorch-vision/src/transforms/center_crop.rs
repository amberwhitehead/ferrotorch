//! ## REQ status (per `.design/ferrotorch-vision/transforms/center_crop.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `pub struct CenterCrop<T: Float>` with `height: usize`, `width: usize`, and `_marker: PhantomData<T>` in `center_crop.rs`, mirroring `torchvision/transforms/v2/_geometry.py:171` `class CenterCrop(Transform)`; consumer: `pub use center_crop::CenterCrop;` in `mod.rs` and `CenterCrop` in the crate-root re-export in `lib.rs`. |
//! | REQ-2 | SHIPPED | `pub fn CenterCrop::new(height: usize, width: usize) -> Self` constructor in `center_crop.rs`; consumer: registered in the conformance surface inventory at `tests/conformance/_surface_inventory.toml` as `ferrotorch_vision::CenterCrop::new`; reachable via the crate-root re-export. |
//! | REQ-3 | SHIPPED | `impl<T: Float> Transform<T> for CenterCrop<T>` with shape, bounds, center-offset, and row-slice copy in `center_crop.rs`; consumer: any `Box<dyn Transform<T>>` slot accepts the type — the `lib.rs` re-export is the production-facing handle. |
//! | REQ-4 | SHIPPED | `with_fill(f64)` builder + auto-pad-with-fill dispatch in `apply` in `center_crop.rs`; consumer: pipelines that call `CenterCrop::new(h, w).with_fill(0.0)` reach the auto-pad path via the `lib.rs` re-export — matches upstream `_geometry.py:180-181`. |

use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_data::Transform;

/// Extract the center region of size `(height, width)` from a `[C, H, W]`
/// tensor.
///
/// When `with_fill` is set, inputs smaller than the crop along either spatial
/// dimension are padded with the fill value first (matching upstream
/// `torchvision.transforms.v2.CenterCrop`'s pad-with-zeros behaviour at
/// `_geometry.py:180-181`). Without `with_fill`, a too-small input still
/// produces `InvalidArgument`.
pub struct CenterCrop<T: Float> {
    height: usize,
    width: usize,
    fill: Option<f64>,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> CenterCrop<T> {
    /// Create a new `CenterCrop` with the desired output spatial size.
    pub fn new(height: usize, width: usize) -> Self {
        Self {
            height,
            width,
            fill: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Enable auto-pad-with-fill when input is smaller than the crop along
    /// either dimension. Mirrors upstream `_geometry.py:180-181` which pads
    /// with zeros; here the fill value is user-selected.
    pub fn with_fill(mut self, fill: f64) -> Self {
        self.fill = Some(fill);
        self
    }
}

impl<T: Float> Transform<T> for CenterCrop<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape().to_vec();
        if shape.len() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "CenterCrop: expected 3-D tensor [C, H, W], got shape {:?}",
                    shape
                ),
            });
        }

        let channels = shape[0];
        let in_h = shape[1];
        let in_w = shape[2];

        let needs_pad_h = self.height > in_h;
        let needs_pad_w = self.width > in_w;

        if (needs_pad_h || needs_pad_w) && self.fill.is_none() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "CenterCrop: crop size ({}x{}) is larger than input spatial size ({}x{})",
                    self.height, self.width, in_h, in_w
                ),
            });
        }

        let data = input.data_vec()?;

        if !needs_pad_h && !needs_pad_w {
            // Fast path: no fill required; bulk row-copy.
            let top = (in_h - self.height) / 2;
            let left = (in_w - self.width) / 2;
            let mut output = Vec::with_capacity(channels * self.height * self.width);
            for c in 0..channels {
                let channel_offset = c * in_h * in_w;
                for row in top..top + self.height {
                    let row_start = channel_offset + row * in_w + left;
                    output.extend_from_slice(&data[row_start..row_start + self.width]);
                }
            }
            let storage = TensorStorage::cpu(output);
            return Tensor::from_storage(storage, vec![channels, self.height, self.width], false);
        }

        // Pad-with-fill path: synthesize the padded image lazily, then
        // copy the center-cropped region out.
        let fill_t: T = cast::<f64, T>(self.fill.expect("fill checked above"))?;
        let h_eff = in_h.max(self.height);
        let w_eff = in_w.max(self.width);
        let pad_top = (h_eff - in_h) / 2;
        let pad_left = (w_eff - in_w) / 2;
        let top = (h_eff - self.height) / 2;
        let left = (w_eff - self.width) / 2;

        let mut output = Vec::with_capacity(channels * self.height * self.width);
        for c in 0..channels {
            let off = c * in_h * in_w;
            for row in top..top + self.height {
                for col in left..left + self.width {
                    let src_row = row as isize - pad_top as isize;
                    let src_col = col as isize - pad_left as isize;
                    if src_row >= 0
                        && src_col >= 0
                        && (src_row as usize) < in_h
                        && (src_col as usize) < in_w
                    {
                        output.push(data[off + src_row as usize * in_w + src_col as usize]);
                    } else {
                        output.push(fill_t);
                    }
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
    fn test_center_crop_output_shape() {
        let data: Vec<f64> = (0..75).map(|i| i as f64).collect();
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 5, 5], false).unwrap();
        let crop = CenterCrop::<f64>::new(3, 3);
        let out = crop.apply(t).unwrap();
        assert_eq!(out.shape(), &[3, 3, 3]);
    }

    #[test]
    fn test_center_crop_values() {
        // 1-channel 4x4 grid:
        //  0  1  2  3
        //  4  5  6  7
        //  8  9 10 11
        // 12 13 14 15
        //
        // Center 2x2 crop (top=1, left=1):
        //  5  6
        //  9 10
        let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 4, 4], false).unwrap();
        let crop = CenterCrop::<f64>::new(2, 2);
        let out = crop.apply(t).unwrap();
        let d = out.data().unwrap();
        assert_eq!(d, &[5.0, 6.0, 9.0, 10.0]);
    }

    #[test]
    fn test_center_crop_exact_size() {
        let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let t =
            Tensor::from_storage(TensorStorage::cpu(data.clone()), vec![1, 3, 4], false).unwrap();
        let crop = CenterCrop::<f64>::new(3, 4);
        let out = crop.apply(t).unwrap();
        assert_eq!(out.shape(), &[1, 3, 4]);
        assert_eq!(out.data().unwrap(), &data);
    }

    #[test]
    fn test_center_crop_multichannel() {
        // 2-channel 4x4:
        // Channel 0: 0..16
        // Channel 1: 16..32
        // Center 2x2 from each channel.
        let data: Vec<f64> = (0..32).map(|i| i as f64).collect();
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![2, 4, 4], false).unwrap();
        let crop = CenterCrop::<f64>::new(2, 2);
        let out = crop.apply(t).unwrap();
        assert_eq!(out.shape(), &[2, 2, 2]);
        let d = out.data().unwrap();
        // Channel 0: rows 1-2, cols 1-2 -> [5, 6, 9, 10]
        // Channel 1: rows 1-2, cols 1-2 -> [21, 22, 25, 26]
        assert_eq!(d, &[5.0, 6.0, 9.0, 10.0, 21.0, 22.0, 25.0, 26.0]);
    }

    #[test]
    fn test_center_crop_too_large() {
        let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 3, 4], false).unwrap();
        let crop = CenterCrop::<f64>::new(5, 4);
        assert!(crop.apply(t).is_err());
    }

    #[test]
    fn test_center_crop_rejects_non_3d() {
        let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![4, 4], false).unwrap();
        let crop = CenterCrop::<f64>::new(2, 2);
        assert!(crop.apply(t).is_err());
    }

    #[test]
    fn test_center_crop_with_fill_pads_small_input() {
        // 1x1 input centered in a 3x3 crop -> 4 corners + 4 edges are fill.
        let data = vec![42.0_f64];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 1, 1], false).unwrap();
        let crop = CenterCrop::<f64>::new(3, 3).with_fill(-1.0);
        let out = crop.apply(t).unwrap();
        let d = out.data().unwrap();
        // 3x3 single channel.
        assert_eq!(d.len(), 9);
        // Center pixel is the only source pixel.
        assert_eq!(d[4], 42.0);
        // All eight surrounding pixels are fill.
        for i in [0, 1, 2, 3, 5, 6, 7, 8] {
            assert_eq!(d[i], -1.0, "expected fill at d[{i}]");
        }
    }

    #[test]
    fn test_center_crop_with_fill_no_op_when_input_large_enough() {
        // When input meets or exceeds crop size, fill is unused.
        let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 4, 4], false).unwrap();
        let crop = CenterCrop::<f64>::new(2, 2).with_fill(99.0);
        let out = crop.apply(t).unwrap();
        assert_eq!(out.data().unwrap(), &[5.0, 6.0, 9.0, 10.0]);
    }
}
