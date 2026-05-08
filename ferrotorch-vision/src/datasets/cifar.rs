//! CIFAR-10 and CIFAR-100 image classification datasets.
//!
//! Provides the CIFAR-10 (10 classes) and CIFAR-100 (100 classes) datasets of
//! 32x32 colour images. Supports loading from the official binary batch files
//! on disk and generating synthetic data for pipeline testing.
//!
//! # CIFAR-10 binary batch format
//!
//! Each batch file contains a sequence of samples. Each sample is:
//! - 1 byte: class label (0..9)
//! - 3072 bytes: pixel data in channel-major order — the full R plane
//!   (32×32 = 1024 bytes), then G, then B.
//!
//! # CIFAR-100 binary batch format
//!
//! Same as CIFAR-10 but each sample header is 2 bytes:
//! - 1 byte: coarse label (0..19)  — stored but unused (we expose fine label)
//! - 1 byte: fine label (0..99)
//! - 3072 bytes: pixel data
//!
//! # Example
//!
//! ```ignore
//! use ferrotorch_vision::datasets::{Cifar10, Cifar100, Split};
//!
//! let train = Cifar10::<f32>::synthetic(Split::Train, 500);
//! assert_eq!(train.len(), 500);
//!
//! let sample = train.get(0).unwrap();
//! assert_eq!(sample.image.shape(), &[3, 32, 32]);
//! assert!(sample.label < 10);
//!
//! let c100 = Cifar100::<f32>::synthetic(Split::Train, 100);
//! let s = c100.get(0).unwrap();
//! assert!(s.label < 100);
//! ```

use std::path::Path;

use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_data::Dataset;

use super::mnist::Split;

/// A single CIFAR sample: a 3x32x32 RGB image and its class label.
///
/// Marked `#[non_exhaustive]` so future per-sample metadata (e.g. fine
/// labels for CIFAR-100, sample index) can be added without breaking
/// struct-literal construction outside this crate.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CifarSample<T: Float> {
    /// RGB image tensor with shape `[3, 32, 32]`, values in `[0, 1]`.
    pub image: Tensor<T>,
    /// Class label in `0..num_classes`.
    pub label: u8,
}

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------

/// Image height for CIFAR datasets.
const HEIGHT: usize = 32;
/// Image width for CIFAR datasets.
const WIDTH: usize = 32;
/// Number of colour channels (RGB).
const CHANNELS: usize = 3;

// ---------------------------------------------------------------------------
// CIFAR-10
// ---------------------------------------------------------------------------

/// The CIFAR-10 dataset: 60 000 32x32 colour images in 10 classes.
///
/// Construction modes:
///
/// - [`Cifar10::synthetic`]: generates random images and labels for testing.
#[derive(Debug)]
pub struct Cifar10<T: Float> {
    images: Vec<Tensor<T>>,
    labels: Vec<u8>,
    split: Split,
}

impl<T: Float> Cifar10<T> {
    /// Image height.
    pub const HEIGHT: usize = HEIGHT;
    /// Image width.
    pub const WIDTH: usize = WIDTH;
    /// Number of colour channels (RGB).
    pub const CHANNELS: usize = CHANNELS;
    /// Number of classes.
    pub const NUM_CLASSES: usize = 10;

    /// Create a synthetic CIFAR-10 dataset with `num_samples` randomly
    /// generated samples.
    ///
    /// Each image is filled with random values in `[0, 1]` and assigned a
    /// random label in `0..10`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if the generated `f64`
    /// pixel values cannot be cast to the target float type `T` (in practice
    /// this never happens for IEEE 754 floats given values in `[0, 1]`).
    pub fn synthetic(split: Split, num_samples: usize) -> FerrotorchResult<Self> {
        let (images, labels) = generate_synthetic::<T>(
            split,
            num_samples,
            Self::NUM_CLASSES,
            0xc1fa_0010_0001,
            0xc1fa_0010_0002,
        )?;
        Ok(Self {
            images,
            labels,
            split,
        })
    }

    /// Load CIFAR-10 from binary batch files in `root`.
    ///
    /// Reads the standard CIFAR-10 binary layout:
    /// - Train: `data_batch_1.bin` … `data_batch_5.bin`
    /// - Test:  `test_batch.bin`
    ///
    /// Each sample in a batch file is 1 label byte + 3072 pixel bytes
    /// (channel-major R→G→B, each plane 32×32). Pixel values are normalized
    /// to `[0, 1]` by dividing by 255.
    ///
    /// # Errors
    ///
    /// Returns an error if any batch file is missing or contains invalid data.
    pub fn from_dir<P: AsRef<Path>>(root: P, split: Split) -> FerrotorchResult<Self> {
        let root = root.as_ref();
        let batch_files: Vec<&str> = match split {
            Split::Train => vec![
                "data_batch_1.bin",
                "data_batch_2.bin",
                "data_batch_3.bin",
                "data_batch_4.bin",
                "data_batch_5.bin",
            ],
            Split::Test => vec!["test_batch.bin"],
        };

        let (images, labels) =
            load_cifar_batches::<T>(root, &batch_files, CifarFormat::Cifar10, Self::NUM_CLASSES)?;

        Ok(Self {
            images,
            labels,
            split,
        })
    }

    /// Which split this dataset represents.
    pub fn split(&self) -> Split {
        self.split
    }
}

impl<T: Float + 'static> Dataset for Cifar10<T> {
    type Sample = CifarSample<T>;

    fn len(&self) -> usize {
        self.images.len()
    }

    fn get(&self, index: usize) -> FerrotorchResult<Self::Sample> {
        if index >= self.images.len() {
            return Err(FerrotorchError::IndexOutOfBounds {
                index,
                axis: 0,
                size: self.images.len(),
            });
        }
        Ok(CifarSample {
            image: self.images[index].clone(),
            label: self.labels[index],
        })
    }
}

// ---------------------------------------------------------------------------
// CIFAR-100
// ---------------------------------------------------------------------------

/// The CIFAR-100 dataset: 60 000 32x32 colour images in 100 classes.
///
/// Construction modes:
///
/// - [`Cifar100::synthetic`]: generates random images and labels for testing.
#[derive(Debug)]
pub struct Cifar100<T: Float> {
    images: Vec<Tensor<T>>,
    labels: Vec<u8>,
    split: Split,
}

impl<T: Float> Cifar100<T> {
    /// Image height.
    pub const HEIGHT: usize = HEIGHT;
    /// Image width.
    pub const WIDTH: usize = WIDTH;
    /// Number of colour channels (RGB).
    pub const CHANNELS: usize = CHANNELS;
    /// Number of classes.
    pub const NUM_CLASSES: usize = 100;

    /// Create a synthetic CIFAR-100 dataset with `num_samples` randomly
    /// generated samples.
    ///
    /// Each image is filled with random values in `[0, 1]` and assigned a
    /// random label in `0..100`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if the generated `f64`
    /// pixel values cannot be cast to the target float type `T` (in practice
    /// this never happens for IEEE 754 floats given values in `[0, 1]`).
    pub fn synthetic(split: Split, num_samples: usize) -> FerrotorchResult<Self> {
        let (images, labels) = generate_synthetic::<T>(
            split,
            num_samples,
            Self::NUM_CLASSES,
            0xc1fa_0100_0001,
            0xc1fa_0100_0002,
        )?;
        Ok(Self {
            images,
            labels,
            split,
        })
    }

    /// Load CIFAR-100 from binary batch files in `root`.
    ///
    /// Reads the standard CIFAR-100 binary layout:
    /// - Train: `train.bin`
    /// - Test:  `test.bin`
    ///
    /// Each sample in a batch file is 2 label bytes (coarse, fine) + 3072
    /// pixel bytes (channel-major R→G→B, each plane 32×32). Only the fine
    /// label (0..99) is exposed; the coarse label is discarded. Pixel values
    /// are normalized to `[0, 1]` by dividing by 255.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch file is missing or contains invalid data.
    pub fn from_dir<P: AsRef<Path>>(root: P, split: Split) -> FerrotorchResult<Self> {
        let root = root.as_ref();
        let batch_files: Vec<&str> = match split {
            Split::Train => vec!["train.bin"],
            Split::Test => vec!["test.bin"],
        };

        let (images, labels) =
            load_cifar_batches::<T>(root, &batch_files, CifarFormat::Cifar100, Self::NUM_CLASSES)?;

        Ok(Self {
            images,
            labels,
            split,
        })
    }

    /// Which split this dataset represents.
    pub fn split(&self) -> Split {
        self.split
    }
}

impl<T: Float + 'static> Dataset for Cifar100<T> {
    type Sample = CifarSample<T>;

    fn len(&self) -> usize {
        self.images.len()
    }

    fn get(&self, index: usize) -> FerrotorchResult<Self::Sample> {
        if index >= self.images.len() {
            return Err(FerrotorchError::IndexOutOfBounds {
                index,
                axis: 0,
                size: self.images.len(),
            });
        }
        Ok(CifarSample {
            image: self.images[index].clone(),
            label: self.labels[index],
        })
    }
}

// ---------------------------------------------------------------------------
// Binary batch reader
// ---------------------------------------------------------------------------

/// Selects the on-disk sample layout for CIFAR-10 vs CIFAR-100.
#[derive(Clone, Copy)]
enum CifarFormat {
    /// CIFAR-10: 1 label byte + 3072 pixel bytes per sample.
    Cifar10,
    /// CIFAR-100: 2 label bytes (coarse, fine) + 3072 pixel bytes per sample.
    Cifar100,
}

/// Number of bytes in the image portion of every CIFAR sample (3 × 32 × 32).
const BYTES_PER_IMAGE: usize = CHANNELS * HEIGHT * WIDTH;

/// Load images and labels from one or more CIFAR binary batch files.
///
/// `batch_files` lists relative file names inside `root`. Files are
/// concatenated in the order given.
fn load_cifar_batches<T: Float>(
    root: &Path,
    batch_files: &[&str],
    format: CifarFormat,
    num_classes: usize,
) -> FerrotorchResult<(Vec<Tensor<T>>, Vec<u8>)> {
    let header_bytes = match format {
        CifarFormat::Cifar10 => 1usize,
        CifarFormat::Cifar100 => 2usize,
    };
    let bytes_per_sample = header_bytes + BYTES_PER_IMAGE;

    let mut images: Vec<Tensor<T>> = Vec::new();
    let mut labels: Vec<u8> = Vec::new();

    for &file_name in batch_files {
        let path = root.join(file_name);
        if !path.exists() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "CIFAR batch file not found: '{}'. \
                     Automatic download is not yet supported.",
                    path.display(),
                ),
            });
        }

        let bytes = std::fs::read(&path).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("failed to read '{}': {e}", path.display()),
        })?;

        if bytes.len() % bytes_per_sample != 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "batch file '{}': length {} is not a multiple of sample size {}",
                    path.display(),
                    bytes.len(),
                    bytes_per_sample,
                ),
            });
        }

        let n_samples = bytes.len() / bytes_per_sample;
        let inv_255 = 1.0_f64 / 255.0;

        for i in 0..n_samples {
            let sample_start = i * bytes_per_sample;

            // Extract label — for CIFAR-100 the fine label is the second byte.
            let label = match format {
                CifarFormat::Cifar10 => bytes[sample_start],
                CifarFormat::Cifar100 => bytes[sample_start + 1],
            };
            if label as usize >= num_classes {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "batch file '{}': sample {i} has label {label} >= num_classes {num_classes}",
                        path.display(),
                    ),
                });
            }

            // Pixel data follows the header bytes.
            let pixel_start = sample_start + header_bytes;
            let raw_pixels = &bytes[pixel_start..pixel_start + BYTES_PER_IMAGE];

            // Pixels are already in channel-major order (R plane, then G, then B).
            let data: Vec<T> = raw_pixels
                .iter()
                .map(|&b| cast::<f64, T>(b as f64 * inv_255))
                .collect::<FerrotorchResult<Vec<T>>>()?;

            let storage = TensorStorage::cpu(data);
            let tensor = Tensor::from_storage(storage, vec![CHANNELS, HEIGHT, WIDTH], false)?;
            images.push(tensor);
            labels.push(label);
        }
    }

    Ok((images, labels))
}

// ---------------------------------------------------------------------------
// Shared generation logic
// ---------------------------------------------------------------------------

/// Generate synthetic image/label pairs for a CIFAR-style dataset.
fn generate_synthetic<T: Float>(
    split: Split,
    num_samples: usize,
    num_classes: usize,
    seed_train: u64,
    seed_test: u64,
) -> FerrotorchResult<(Vec<Tensor<T>>, Vec<u8>)> {
    let mut images = Vec::with_capacity(num_samples);
    let mut labels = Vec::with_capacity(num_samples);

    let mut state: u64 = match split {
        Split::Train => seed_train,
        Split::Test => seed_test,
    };

    let numel = CHANNELS * HEIGHT * WIDTH;

    for _ in 0..num_samples {
        let mut data = Vec::with_capacity(numel);
        for _ in 0..numel {
            state = xorshift64(state);
            let f = (state as f64) / (u64::MAX as f64);
            data.push(cast::<f64, T>(f)?);
        }
        let storage = TensorStorage::cpu(data);
        let tensor = Tensor::from_storage(storage, vec![CHANNELS, HEIGHT, WIDTH], false)?;
        images.push(tensor);

        state = xorshift64(state);
        let label = (state % num_classes as u64) as u8;
        labels.push(label);
    }

    Ok((images, labels))
}

/// xorshift64 PRNG step.
#[inline]
fn xorshift64(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // CIFAR-10 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cifar10_synthetic_train_len() {
        let ds = Cifar10::<f32>::synthetic(Split::Train, 100).unwrap();
        assert_eq!(ds.len(), 100);
        assert!(!ds.is_empty());
    }

    #[test]
    fn test_cifar10_synthetic_test_len() {
        let ds = Cifar10::<f32>::synthetic(Split::Test, 50).unwrap();
        assert_eq!(ds.len(), 50);
    }

    #[test]
    fn test_cifar10_synthetic_empty() {
        let ds = Cifar10::<f32>::synthetic(Split::Train, 0).unwrap();
        assert!(ds.is_empty());
        assert_eq!(ds.len(), 0);
    }

    #[test]
    fn test_cifar10_sample_image_shape() {
        let ds = Cifar10::<f32>::synthetic(Split::Train, 10).unwrap();
        let sample = ds.get(0).unwrap();
        assert_eq!(sample.image.shape(), &[3, 32, 32]);
        assert_eq!(sample.image.numel(), 3 * 32 * 32);
    }

    #[test]
    fn test_cifar10_sample_values_in_range() {
        let ds = Cifar10::<f32>::synthetic(Split::Train, 5).unwrap();
        for i in 0..5 {
            let sample = ds.get(i).unwrap();
            let data = sample.image.data().unwrap();
            for &v in data {
                assert!((0.0..=1.0).contains(&v), "pixel value out of [0,1]: {v}");
            }
        }
    }

    #[test]
    fn test_cifar10_label_range() {
        let ds = Cifar10::<f32>::synthetic(Split::Train, 200).unwrap();
        for i in 0..200 {
            let sample = ds.get(i).unwrap();
            assert!(sample.label < 10, "label out of range: {}", sample.label);
        }
    }

    #[test]
    fn test_cifar10_out_of_bounds() {
        let ds = Cifar10::<f32>::synthetic(Split::Train, 10).unwrap();
        assert!(ds.get(10).is_err());
        assert!(ds.get(100).is_err());
    }

    #[test]
    fn test_cifar10_split_accessor() {
        let train = Cifar10::<f32>::synthetic(Split::Train, 1).unwrap();
        let test = Cifar10::<f32>::synthetic(Split::Test, 1).unwrap();
        assert_eq!(train.split(), Split::Train);
        assert_eq!(test.split(), Split::Test);
    }

    #[test]
    fn test_cifar10_train_test_different() {
        let train = Cifar10::<f32>::synthetic(Split::Train, 5).unwrap();
        let test = Cifar10::<f32>::synthetic(Split::Test, 5).unwrap();
        let t0 = train.get(0).unwrap();
        let e0 = test.get(0).unwrap();
        let t_data = t0.image.data().unwrap();
        let e_data = e0.image.data().unwrap();
        assert_ne!(t_data, e_data, "train and test splits should differ");
    }

    #[test]
    fn test_cifar10_f64() {
        let ds = Cifar10::<f64>::synthetic(Split::Train, 3).unwrap();
        let sample = ds.get(0).unwrap();
        assert_eq!(sample.image.shape(), &[3, 32, 32]);
    }

    #[test]
    fn test_cifar10_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Cifar10<f32>>();
        assert_send_sync::<CifarSample<f32>>();
    }

    // -----------------------------------------------------------------------
    // CIFAR-100 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cifar100_synthetic_len() {
        let ds = Cifar100::<f32>::synthetic(Split::Train, 80).unwrap();
        assert_eq!(ds.len(), 80);
    }

    #[test]
    fn test_cifar100_sample_shape() {
        let ds = Cifar100::<f32>::synthetic(Split::Test, 5).unwrap();
        let sample = ds.get(0).unwrap();
        assert_eq!(sample.image.shape(), &[3, 32, 32]);
    }

    #[test]
    fn test_cifar100_label_range() {
        let ds = Cifar100::<f32>::synthetic(Split::Train, 500).unwrap();
        for i in 0..500 {
            let sample = ds.get(i).unwrap();
            assert!(sample.label < 100, "label out of range: {}", sample.label);
        }
    }

    #[test]
    fn test_cifar100_out_of_bounds() {
        let ds = Cifar100::<f32>::synthetic(Split::Train, 10).unwrap();
        assert!(ds.get(10).is_err());
    }

    #[test]
    fn test_cifar100_different_from_cifar10() {
        // CIFAR-100 and CIFAR-10 use different seeds, so data should differ.
        let c10 = Cifar10::<f32>::synthetic(Split::Train, 5).unwrap();
        let c100 = Cifar100::<f32>::synthetic(Split::Train, 5).unwrap();
        let d10 = c10.get(0).unwrap().image.data().unwrap().to_vec();
        let d100 = c100.get(0).unwrap().image.data().unwrap().to_vec();
        assert_ne!(
            d10, d100,
            "CIFAR-10 and CIFAR-100 should use different seeds"
        );
    }

    #[test]
    fn test_cifar100_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Cifar100<f32>>();
    }
}
