//! Vision datasets: MNIST, CIFAR-10, CIFAR-100, ImageFolder, DatasetFolder.
//!
//! Each dataset implements [`ferrotorch_data::Dataset`] and provides a
//! `synthetic()` or `from_dir()` constructor.

//! ## REQ status (per `.design/ferrotorch-vision/datasets/mod.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `pub mod cifar; pub mod folder; pub mod mnist;` per upstream torchvision per-file layout (`torchvision/datasets/cifar.py`, `folder.py`, `mnist.py`); consumer: re-exports below + `ferrotorch-vision/src/lib.rs` glob. |
//! | REQ-2 | SHIPPED | `pub use cifar::{Cifar10, Cifar100, CifarSample};`, `pub use folder::{DatasetFolder, FolderSample, IMG_EXTENSIONS, ImageFolder, ImageSample};`, `pub use mnist::{Mnist, MnistSample, Split};` mirror torchvision's flat `datasets` namespace; consumer: `pub use datasets::{...}` at `ferrotorch-vision/src/lib.rs:99`. |
//! | REQ-3 | SHIPPED | `Split` is defined in `mnist.rs` and re-exported via `pub use mnist::{...Split};` for cross-dataset use; consumer: `use super::mnist::Split;` at `ferrotorch-vision/src/datasets/cifar.rs:44` and `use ferrotorch_vision::{Mnist, Split};` at `ferrotorch/examples/train_mnist.rs:22`.

pub mod cifar;
pub mod folder;
pub mod mnist;

pub use cifar::{Cifar10, Cifar100, CifarSample};
pub use folder::{DatasetFolder, FolderSample, IMG_EXTENSIONS, ImageFolder, ImageSample};
pub use mnist::{Mnist, MnistSample, Split};
