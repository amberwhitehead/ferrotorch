//! Conformance suite for `ferrotorch-vision` — datasets module.
//!
//! Tracking issue: #870 (MNIST/CIFAR from_dir + synthetic fixtures).
//!
//! Reference: torchvision 0.21.x
//!
//! ## Scope
//!
//! Tests that `Mnist::from_dir`, `Cifar10::from_dir`, and `Cifar100::from_dir`
//! correctly parse the small synthetic binary fixtures in
//! `tests/conformance/fixtures/datasets_synthetic/`.
//!
//! The fixtures are deterministic (pixel values derived from a simple formula)
//! so we can verify exact shapes, dtypes, label values, and spot-checked pixel
//! values without bundling the real >100 MB datasets.
//!
//! ## cascade_skip convention
//!
//! `cascade_skip!(label)` prints a diagnostic and returns early. NOT `#[ignore]`.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::uninlined_format_args,
)]

use std::path::PathBuf;

use ferrotorch_data::Dataset;
use ferrotorch_vision::datasets::{Cifar10, Cifar100, Mnist, Split};

// ---------------------------------------------------------------------------
// Fixture path helpers
// ---------------------------------------------------------------------------

fn datasets_fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("conformance")
        .join("fixtures")
        .join("datasets_synthetic")
}

// ---------------------------------------------------------------------------
// PROBE: BEFORE state documented in fixture header
//
// BEFORE (#870): Cifar10::from_dir and Cifar100::from_dir did not exist.
//   Only `synthetic()` was available. from_dir returned a compile error.
//
// AFTER (#870): from_dir reads binary batch files and returns parsed datasets
//   with correct shapes, dtypes (f32/f64), and labels.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// MNIST::from_dir — synthetic fixture tests
// ---------------------------------------------------------------------------

#[test]
fn mnist_from_dir_train_len_and_shape() {
    let dir = datasets_fixture_dir();
    let ds = Mnist::<f32>::from_dir(&dir, Split::Train).expect("Mnist::from_dir(train)");
    assert_eq!(ds.len(), 4, "expected 4 synthetic train samples");
    assert!(!ds.is_empty());
}

#[test]
fn mnist_from_dir_train_labels() {
    let dir = datasets_fixture_dir();
    let ds = Mnist::<f32>::from_dir(&dir, Split::Train).expect("Mnist::from_dir(train)");

    // Fixture labels written by scripts: [3, 7, 1, 5]
    let expected = [3u8, 7, 1, 5];
    for (i, &exp) in expected.iter().enumerate() {
        let sample = ds.get(i).expect("get sample");
        assert_eq!(
            sample.label, exp,
            "MNIST train sample {i}: expected label {exp}, got {}",
            sample.label
        );
    }
}

#[test]
fn mnist_from_dir_train_image_shape() {
    let dir = datasets_fixture_dir();
    let ds = Mnist::<f32>::from_dir(&dir, Split::Train).expect("Mnist::from_dir(train)");

    // Fixture images are 4×4 (rows × cols) with 1 channel → shape [1, 4, 4].
    for i in 0..4 {
        let sample = ds.get(i).unwrap();
        assert_eq!(
            sample.image.shape(),
            &[1, 4, 4],
            "MNIST train sample {i}: wrong shape"
        );
    }
}

#[test]
fn mnist_from_dir_train_pixel_normalization() {
    let dir = datasets_fixture_dir();
    let ds = Mnist::<f32>::from_dir(&dir, Split::Train).expect("Mnist::from_dir(train)");

    let sample = ds.get(0).unwrap();
    let data = sample.image.data().unwrap();

    // All values must be in [0, 1].
    for &v in data {
        assert!(
            (0.0..=1.0).contains(&v),
            "MNIST pixel {v} outside [0, 1]"
        );
    }

    // Spot-check: fixture seed=0, formula = (0 + i*17) % 256 / 255.
    // pixel[0] = 0 / 255 = 0.0
    // pixel[1] = 17 / 255 ≈ 0.0667
    // pixel[2] = 34 / 255 ≈ 0.1333
    assert!(
        data[0].abs() < 1e-6,
        "pixel[0] expected ~0.0, got {}",
        data[0]
    );
    let expected_1 = 17.0_f32 / 255.0;
    assert!(
        (data[1] - expected_1).abs() < 1e-5,
        "pixel[1] expected ~{expected_1}, got {}",
        data[1]
    );
    let expected_2 = 34.0_f32 / 255.0;
    assert!(
        (data[2] - expected_2).abs() < 1e-5,
        "pixel[2] expected ~{expected_2}, got {}",
        data[2]
    );
}

#[test]
fn mnist_from_dir_test_len_and_labels() {
    let dir = datasets_fixture_dir();
    let ds = Mnist::<f32>::from_dir(&dir, Split::Test).expect("Mnist::from_dir(test)");

    assert_eq!(ds.len(), 4);
    // Fixture test labels: [0, 9, 4, 2]
    let expected = [0u8, 9, 4, 2];
    for (i, &exp) in expected.iter().enumerate() {
        assert_eq!(ds.get(i).unwrap().label, exp, "test label {i}");
    }
}

#[test]
fn mnist_from_dir_f64() {
    let dir = datasets_fixture_dir();
    let ds = Mnist::<f64>::from_dir(&dir, Split::Train).expect("Mnist::from_dir f64");
    let sample = ds.get(0).unwrap();
    assert_eq!(sample.image.shape(), &[1, 4, 4]);
    // f64 boundary: pixel[0] must be exactly 0.0 (raw byte = 0).
    let data = sample.image.data().unwrap();
    assert_eq!(data[0], 0.0_f64, "f64 pixel[0] should be exact 0");
}

#[test]
fn mnist_from_dir_split_accessor() {
    let dir = datasets_fixture_dir();
    let train = Mnist::<f32>::from_dir(&dir, Split::Train).unwrap();
    let test = Mnist::<f32>::from_dir(&dir, Split::Test).unwrap();
    assert_eq!(train.split(), Split::Train);
    assert_eq!(test.split(), Split::Test);
}

#[test]
fn mnist_from_dir_missing_returns_err() {
    let result = Mnist::<f32>::from_dir("/nonexistent/path", Split::Train);
    assert!(result.is_err(), "from_dir must return Err for missing files");
}

// ---------------------------------------------------------------------------
// Cifar10::from_dir — synthetic fixture tests
// ---------------------------------------------------------------------------

#[test]
fn cifar10_from_dir_test_len() {
    let dir = datasets_fixture_dir();
    let ds = Cifar10::<f32>::from_dir(&dir, Split::Test).expect("Cifar10::from_dir(test)");
    assert_eq!(ds.len(), 4, "expected 4 synthetic test samples");
    assert!(!ds.is_empty());
}

#[test]
fn cifar10_from_dir_test_labels() {
    let dir = datasets_fixture_dir();
    let ds = Cifar10::<f32>::from_dir(&dir, Split::Test).expect("Cifar10::from_dir(test)");

    // Fixture test_batch.bin labels: [3, 7, 1, 9]
    let expected = [3u8, 7, 1, 9];
    for (i, &exp) in expected.iter().enumerate() {
        let sample = ds.get(i).expect("get cifar10 test sample");
        assert_eq!(
            sample.label, exp,
            "CIFAR-10 test sample {i}: expected {exp}, got {}",
            sample.label
        );
    }
}

#[test]
fn cifar10_from_dir_test_image_shape() {
    let dir = datasets_fixture_dir();
    let ds = Cifar10::<f32>::from_dir(&dir, Split::Test).expect("Cifar10::from_dir(test)");

    // CIFAR-10 images are always [3, 32, 32].
    for i in 0..4 {
        let sample = ds.get(i).unwrap();
        assert_eq!(
            sample.image.shape(),
            &[3, 32, 32],
            "CIFAR-10 sample {i}: wrong shape"
        );
        assert_eq!(sample.image.numel(), 3 * 32 * 32);
    }
}

#[test]
fn cifar10_from_dir_test_pixel_normalization() {
    let dir = datasets_fixture_dir();
    let ds = Cifar10::<f32>::from_dir(&dir, Split::Test).expect("Cifar10::from_dir(test)");

    let sample = ds.get(0).unwrap();
    let data = sample.image.data().unwrap();

    // All values in [0, 1].
    for &v in data {
        assert!(
            (0.0..=1.0).contains(&v),
            "CIFAR-10 pixel {v} outside [0, 1]"
        );
    }

    // Spot-check: seed=0, label=3 → pixel[0] = (0 + 0*7 + 3*13) % 256 = 39
    // pixel[1] = (0 + 1*7 + 3*13) % 256 = 46
    let expected_0 = 39.0_f32 / 255.0;
    let expected_1 = 46.0_f32 / 255.0;
    assert!(
        (data[0] - expected_0).abs() < 1e-5,
        "CIFAR-10 pixel[0]: expected {expected_0}, got {}",
        data[0]
    );
    assert!(
        (data[1] - expected_1).abs() < 1e-5,
        "CIFAR-10 pixel[1]: expected {expected_1}, got {}",
        data[1]
    );
}

#[test]
fn cifar10_from_dir_train_len() {
    let dir = datasets_fixture_dir();
    // Train uses data_batch_1..5; fixture has 4 samples in batch_1, rest empty.
    let ds = Cifar10::<f32>::from_dir(&dir, Split::Train).expect("Cifar10::from_dir(train)");
    assert_eq!(ds.len(), 4, "expected 4 samples from data_batch_1");
}

#[test]
fn cifar10_from_dir_train_labels() {
    let dir = datasets_fixture_dir();
    let ds = Cifar10::<f32>::from_dir(&dir, Split::Train).expect("Cifar10::from_dir(train)");

    // data_batch_1.bin labels: [0, 5, 2, 8]
    let expected = [0u8, 5, 2, 8];
    for (i, &exp) in expected.iter().enumerate() {
        assert_eq!(
            ds.get(i).unwrap().label,
            exp,
            "CIFAR-10 train sample {i}"
        );
    }
}

#[test]
fn cifar10_from_dir_f64() {
    let dir = datasets_fixture_dir();
    let ds = Cifar10::<f64>::from_dir(&dir, Split::Test).expect("Cifar10::from_dir f64");
    let sample = ds.get(0).unwrap();
    assert_eq!(sample.image.shape(), &[3, 32, 32]);
    let data = sample.image.data().unwrap();
    // First pixel: raw byte 39 → 39.0/255.0
    let expected = 39.0_f64 / 255.0;
    assert!(
        (data[0] - expected).abs() < 1e-12,
        "f64 pixel[0]: expected {expected}, got {}",
        data[0]
    );
}

#[test]
fn cifar10_from_dir_split_accessor() {
    let dir = datasets_fixture_dir();
    let train = Cifar10::<f32>::from_dir(&dir, Split::Train).unwrap();
    let test = Cifar10::<f32>::from_dir(&dir, Split::Test).unwrap();
    assert_eq!(train.split(), Split::Train);
    assert_eq!(test.split(), Split::Test);
}

#[test]
fn cifar10_from_dir_missing_returns_err() {
    let result = Cifar10::<f32>::from_dir("/nonexistent/path", Split::Test);
    assert!(result.is_err(), "from_dir must Err for missing batch file");
}

// ---------------------------------------------------------------------------
// Cifar100::from_dir — synthetic fixture tests
// ---------------------------------------------------------------------------

#[test]
fn cifar100_from_dir_test_len() {
    let dir = datasets_fixture_dir();
    let ds = Cifar100::<f32>::from_dir(&dir, Split::Test).expect("Cifar100::from_dir(test)");
    assert_eq!(ds.len(), 4, "expected 4 synthetic CIFAR-100 test samples");
}

#[test]
fn cifar100_from_dir_test_fine_labels() {
    let dir = datasets_fixture_dir();
    let ds = Cifar100::<f32>::from_dir(&dir, Split::Test).expect("Cifar100::from_dir(test)");

    // test.bin fine labels: [17, 55, 88, 3]
    let expected = [17u8, 55, 88, 3];
    for (i, &exp) in expected.iter().enumerate() {
        let sample = ds.get(i).unwrap();
        assert_eq!(
            sample.label, exp,
            "CIFAR-100 test fine label {i}: expected {exp}, got {}",
            sample.label
        );
        assert!(sample.label < 100, "fine label must be < 100");
    }
}

#[test]
fn cifar100_from_dir_test_image_shape() {
    let dir = datasets_fixture_dir();
    let ds = Cifar100::<f32>::from_dir(&dir, Split::Test).expect("Cifar100::from_dir(test)");

    for i in 0..4 {
        let sample = ds.get(i).unwrap();
        assert_eq!(
            sample.image.shape(),
            &[3, 32, 32],
            "CIFAR-100 test image {i}: wrong shape"
        );
    }
}

#[test]
fn cifar100_from_dir_test_pixel_normalization() {
    let dir = datasets_fixture_dir();
    let ds = Cifar100::<f32>::from_dir(&dir, Split::Test).expect("Cifar100::from_dir(test)");

    for i in 0..4 {
        let sample = ds.get(i).unwrap();
        for &v in sample.image.data().unwrap() {
            assert!(
                (0.0..=1.0).contains(&v),
                "CIFAR-100 test sample {i} pixel {v} outside [0, 1]"
            );
        }
    }
}

#[test]
fn cifar100_from_dir_train_len_and_labels() {
    let dir = datasets_fixture_dir();
    let ds = Cifar100::<f32>::from_dir(&dir, Split::Train).expect("Cifar100::from_dir(train)");
    assert_eq!(ds.len(), 4);

    // train.bin fine labels: [10, 42, 73, 91]
    let expected = [10u8, 42, 73, 91];
    for (i, &exp) in expected.iter().enumerate() {
        assert_eq!(ds.get(i).unwrap().label, exp, "CIFAR-100 train label {i}");
    }
}

#[test]
fn cifar100_from_dir_missing_returns_err() {
    let result = Cifar100::<f32>::from_dir("/nonexistent/path", Split::Test);
    assert!(result.is_err());
}

#[test]
fn cifar100_from_dir_split_accessor() {
    let dir = datasets_fixture_dir();
    let train = Cifar100::<f32>::from_dir(&dir, Split::Train).unwrap();
    assert_eq!(train.split(), Split::Train);
}

// ---------------------------------------------------------------------------
// Cross-check: synthetic() and from_dir() produce same shapes/dtypes
// ---------------------------------------------------------------------------

#[test]
fn cifar10_from_dir_vs_synthetic_shape_parity() {
    let dir = datasets_fixture_dir();
    let from_dir = Cifar10::<f32>::from_dir(&dir, Split::Test).unwrap();
    let synthetic = Cifar10::<f32>::synthetic(Split::Test, 4).unwrap();

    let fd_sample = from_dir.get(0).unwrap();
    let syn_sample = synthetic.get(0).unwrap();

    // Both must produce [3, 32, 32] tensors.
    assert_eq!(fd_sample.image.shape(), syn_sample.image.shape());
    assert_eq!(fd_sample.image.ndim(), 3);
}
