# ferrotorch-vision — MNIST dataset (`datasets/mnist.rs`)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/  (torchvision companion: /home/doll/.local/lib/python3.13/site-packages/torchvision/datasets/mnist.py)
-->

## Summary

`ferrotorch-vision/src/datasets/mnist.rs` provides the classic MNIST
handwritten-digit dataset (28×28 grayscale, 10 classes) with two
construction paths: a deterministic synthetic generator for pipeline
testing, and an IDX-file loader for the real on-disk dataset.
Implements the `ferrotorch_data::Dataset` trait. Mirrors
`torchvision.datasets.MNIST` (`torchvision/datasets/mnist.py:20-200`)
and the IDX parser at `torchvision/datasets/mnist.py:508-560`
(`read_sn3_pascalvincent_tensor` / `read_image_file` /
`read_label_file`).

## Requirements

- REQ-1: `pub struct MnistSample<T: Float>` carries `image:
  Tensor<T>` with shape `[1, 28, 28]` and `label: u8` in `0..=9`.
  Marked `#[non_exhaustive]` for forward compatibility. Mirrors the
  `(image, target)` tuple returned by
  `torchvision.datasets.MNIST.__getitem__` at `mnist.py:131-151`
  (where `target = int(self.targets[index])`).
- REQ-2: `pub enum Split { Train, Test }` selects the training or
  test split. Mirrors the `train: bool = True` kwarg upstream
  (`mnist.py:87`) — the boolean is replaced by a typestate-friendly
  enum (R-DEV-5 — Rust typestate when ordering / dispatch matters)
  so callers can `match` on the split exhaustively.
- REQ-3: `pub struct Mnist<T: Float>` stores the loaded images and
  labels with the active `Split`. Marked `#[derive(Debug)]`. Mirrors
  `class MNIST(VisionDataset)` at `mnist.py:20`.
- REQ-4: `Mnist::<T>::synthetic(split: Split, num_samples: usize) ->
  FerrotorchResult<Self>` generates `num_samples` deterministic
  random samples via xorshift64 (seeded from the split). Each image
  is filled with `[0, 1]` floats; each label is in `0..10`. No
  upstream counterpart — this is the pipeline-test entry point that
  replaces the "download = True" path in scenarios where the user
  doesn't have / want the real dataset.
- REQ-5: `Mnist::<T>::from_dir(root, split) -> FerrotorchResult<Self>`
  reads the four canonical IDX files (`train-images-idx3-ubyte` /
  `train-labels-idx1-ubyte` for train, `t10k-images-idx3-ubyte` /
  `t10k-labels-idx1-ubyte` for test). Mirrors `MNIST._load_data`
  (`mnist.py:122-129`) chained with `read_image_file` /
  `read_label_file` (`mnist.py:545-560`).
- REQ-6: IDX header validation mirrors upstream's
  `read_sn3_pascalvincent_tensor` (`mnist.py:508-542`) for the
  specific `magic == 2051` (images, uint8, 3-D) and
  `magic == 2049` (labels, uint8, 1-D) cases. Wrong magic numbers,
  truncated files, and count mismatches are reported as
  `InvalidArgument` with a message that mentions "magic" /
  "truncated" / "mismatch" so error-path tests can grep.
- REQ-7: Pixel values are normalized from `u8 ∈ [0, 255]` to `T ∈
  [0.0, 1.0]` by multiplying by `1.0_f64 / 255.0` and casting via
  `cast::<f64, T>(...)`. Matches upstream's
  `to_tensor` composition `(uint8 → float → / 255.0)` from
  `torchvision.transforms.functional.to_tensor`.
- REQ-8: `impl Dataset for Mnist<T>` exposes `len`/`is_empty`/`get`
  per the `ferrotorch_data::Dataset` trait. `get(index)` returns
  `IndexOutOfBounds { index, axis: 0, size }` for out-of-range
  access. Matches upstream's `IndexError` shape.

## Acceptance Criteria

- [x] AC-1: `Mnist::<f32>::synthetic(Split::Train, n).unwrap().len()
  == n` for every tested `n` (0, 50, 100, 1000).
- [x] AC-2: Each synthetic sample's `image.shape() == &[1, 28, 28]`
  and `label < 10`.
- [x] AC-3: `Train` and `Test` splits produce DIFFERENT bytes for
  the first sample (different seed → different xorshift state).
- [x] AC-4: `from_dir` round-trip: write synthetic IDX → read back
  → compare per-pixel values to `pixel as f32 / 255.0`.
- [x] AC-5: `from_dir` rejects wrong magic numbers, truncated
  images, truncated labels, and count mismatches with `Err`
  messages containing the relevant keyword.
- [x] AC-6: `get(out_of_range)` returns
  `Err(FerrotorchError::IndexOutOfBounds { ... })`, not a panic.
- [x] AC-7: `MnistSample<f32>` and `Mnist<f32>` are `Send + Sync`
  (compile-time assertion).
- [x] AC-8: Pixel-normalization boundaries are exact: `0u8 → 0.0`
  and `255u8 → 1.0` for `f64`.

## Architecture

The file is structured as four sections: type definitions
(`MnistSample`, `Split`, `Mnist`), `Mnist::synthetic`,
`Mnist::from_dir`, and the IDX-binary helpers
(`xorshift64`, `read_u32_be`).

### `MnistSample` (REQ-1)

`#[non_exhaustive]` (`mnist.rs`) reserves room for
future-per-sample metadata (e.g. `path: PathBuf` for traceability
when sampling from a sharded loader). External code can pattern-match
the `image` and `label` fields but cannot construct via struct
literal — the only constructors are `Mnist::synthetic` /
`Mnist::from_dir`.

### `Split` (REQ-2)

`Split::Train` and `Split::Test` (`mnist.rs`). `#[derive(Debug,
Clone, Copy, PartialEq, Eq)]` so callers can use the enum as a
match scrutinee and compare via `==`. The `Copy` derive is load-bearing
— the enum is taken by value in `synthetic` / `from_dir` /
`split()` accessor.

### `Mnist<T: Float>` (REQ-3)

Fields are private: `images: Vec<Tensor<T>>`, `labels: Vec<u8>`,
`split: Split` (`split in mnist.rs`). Constants `HEIGHT = 28`, `WIDTH =
28`, `CHANNELS = 1`, `NUM_CLASSES = 10` are `pub const` so users can
allocate output buffers without hard-coding dimensions
(`mnist.rs`).

### `synthetic` (REQ-4)

xorshift64 PRNG seeded from `Split` (`Split in mnist.rs`):
- `Train` → `0xdead_beef_cafe_0001`
- `Test` → `0xdead_beef_cafe_0002`

The state is advanced once per pixel for the image data and once for
the label. The label is `(state % 10) as u8`. Determinism is
load-bearing — `test_train_test_different_data` (`test_train_test_different_data in mnist.rs`)
relies on the seed difference; production training pipelines that
use `Mnist::synthetic` for smoke testing get reproducible epochs.

### `from_dir` (REQ-5, REQ-6, REQ-7)

The IDX-file reader at `mnist.rs` performs:

1. **File-existence check** (`mnist.rs`): returns an
   `InvalidArgument` referencing both filenames + the download URL
   if either file is missing. Mirrors `MNIST._check_exists`
   (`mnist.py:168-172`) but as a single error rather than a
   `RuntimeError("Dataset not found")`.
2. **Header validation** (`mnist.rs`): minimum-length
   checks for both files, then magic-number checks (`2051` for
   images = `0x00000803`, `2049` for labels = `0x00000801` — these
   are the IDX type codes upstream encodes in
   `SN3_PASCALVINCENT_TYPEMAP[8]` for `torch.uint8`,
   `mnist.py:498-505`).
3. **Per-image tensor construction** (`mnist.rs`): each
   `pixels_per_image`-byte slab is mapped to `[T]` via
   `cast::<f64, T>(b as f64 * inv_255)` where `inv_255 = 1.0/255.0`
   is precomputed once.

The IDX format is big-endian for header fields; `read_u32_be`
(`mnist.rs`) handles the byte-swap. This is the
ferrotorch-side equivalent of upstream's
`if sys.byteorder == 'little' and parsed.element_size() > 1:
parsed = _flip_byte_order(parsed)` byte-reorder at `mnist.py:538-
539` — for uint8 pixel data the swap is a no-op, so ferrotorch
omits it.

### `impl Dataset for Mnist<T>` (REQ-8)

`len in mnist.rs`: forwards `len()` to `self.images.len()` and
returns `IndexOutOfBounds` for `get(index)` when `index >=
self.images.len()`. The `Send + Sync` bound on the trait is satisfied
because `Tensor<T>: Send + Sync` and `Vec<...>: Send + Sync`.

### Non-test production consumers

- `ferrotorch/examples/train_mnist.rs:22,60` —
  `use ferrotorch_vision::{Mnist, Split};` followed by
  `let train_dataset = Mnist::<f32>::synthetic(Split::Train,
  num_samples)?;` is the canonical non-test consumer (a runnable
  training-loop example).
- `ferrotorch-vision/src/datasets/mod.rs` — `pub use mnist::{Mnist,
  MnistSample, Split};` is the re-export that propagates the type
  surface; without this re-export the `train_mnist` example would
  not compile.
- `ferrotorch-vision/src/lib.rs` — `pub use datasets::{... Mnist,
  MnistSample, Split};` is the crate-root re-export.

## Parity contract

`parity_ops = []`. Datasets are I/O + sample-iteration glue; the
numerical contract is on the float-tensor values, which is
`ferrotorch-core`'s responsibility.

Edge cases preserved:

- **Empty dataset**: `synthetic(split, 0)` and zero-image IDX files
  both produce `len() == 0`. `is_empty()` returns `true`. Matches
  upstream's `len(self.data) == 0` behaviour
  (`mnist.py:153-154`).
- **Pixel normalization boundaries**: `0u8 → 0.0` exactly, `255u8 →
  1.0` exactly under IEEE 754 with rounding-to-nearest (since `255 *
  (1.0/255.0)` rounds back to `1.0`). Verified by
  `test_from_dir_pixel_normalization_boundaries` (`test_from_dir_pixel_normalization_boundaries in mnist.rs`).
- **Bad magic number**: any header field that doesn't match `2051`
  (images) or `2049` (labels) returns `InvalidArgument` with
  "magic" in the message. Tests
  `test_from_dir_wrong_image_magic` / `test_from_dir_wrong_label_magic`
  pin this.
- **Truncated file**: images/labels file shorter than the header
  promises returns `InvalidArgument` with "truncated" in the
  message. Tests `test_from_dir_truncated_images` /
  `test_from_dir_truncated_labels` pin this.
- **Count mismatch**: `num_images != num_labels` (header fields
  disagree) returns `InvalidArgument` with "mismatch" in the
  message. Test `test_from_dir_count_mismatch` pins this.
- **Endianness**: `read_u32_be` always uses big-endian regardless of
  host byteorder, matching the IDX file format (R-DEV-3 — on-disk
  format).
- **`Send + Sync`**: `test_is_send_sync` (`test_is_send_sync in mnist.rs`)
  statically asserts via `fn assert_send_sync<T: Send + Sync>() {}`.

Divergences from upstream (deliberate):

- **`download=True` not supported**: ferrotorch's `from_dir` returns
  an error if the files are missing, with a message pointing at
  the upstream URL. Automatic download is a follow-up — the dataset
  pipeline must work end-to-end with manually-placed files first.
- **`transform` / `target_transform` kwargs absent**: upstream's
  callable transforms are replaced by `MappedDataset` composition
  (R-DEV-7 — Rust analog via the dataset trait).
- **No "PIL Image" intermediate**: upstream returns a `PIL.Image`
  for `img`; ferrotorch returns a `[1, 28, 28]` `Tensor<T>` directly.
  Most training pipelines compose `ToTensor` immediately after the
  dataset anyway; ferrotorch collapses the two steps.

## Verification

Unit tests in `mod tests` of `ferrotorch-vision/src/datasets/mnist.rs`
(`mnist.rs`):

- `test_synthetic_train_len` / `test_synthetic_test_len` /
  `test_synthetic_empty` — basic length contract.
- `test_sample_image_shape` — `[1, 28, 28]` shape.
- `test_sample_image_values_in_range` — `[0.0, 1.0]` value range.
- `test_label_range` — `label < 10`.
- `test_out_of_bounds` — `get(n)` errors.
- `test_split_accessor` — split round-trip.
- `test_f64_support` — `Mnist<f64>` path.
- `test_train_test_different_data` — train/test seed divergence.
- `test_from_dir_missing` — missing-dir error.
- `test_is_send_sync` — compile-time `Send + Sync` assertion.
- `test_from_dir_parses_single_image` / `_multiple_images` /
  `_test_split` / `_f64` — round-trip IDX read with explicit
  pixel-value verification.
- `test_from_dir_pixel_normalization_boundaries` — `0u8 → 0.0`,
  `255u8 → 1.0`.
- `test_from_dir_wrong_image_magic` / `_wrong_label_magic` /
  `_truncated_images` / `_truncated_labels` / `_count_mismatch` /
  `_images_file_too_short` / `_labels_file_too_short` /
  `_zero_images` — error-path coverage.
- `test_read_u32_be` — big-endian decode unit test.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-vision --lib datasets::mnist 2>&1 | tail -3
```

Expected: ~22 passed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `#[non_exhaustive] pub struct MnistSample<T: Float>` at `MnistSample in ferrotorch-vision/src/datasets/mnist.rs` (image + label fields) per upstream `torchvision/datasets/mnist.py:131-151` `__getitem__` returning `(img, target)`; non-test consumer: `Mnist::get` at `get in ferrotorch-vision/src/datasets/mnist.rs` constructs `MnistSample` and the type is re-exported at `ferrotorch-vision/src/datasets/mod.rs` for downstream training pipelines. |
| REQ-2 | SHIPPED | impl: `pub enum Split { Train, Test }` at `ferrotorch-vision/src/datasets/mnist.rs:43-48` per upstream `torchvision/datasets/mnist.py:87` `train: bool = True` kwarg (R-DEV-5 typestate replacement of bool); non-test consumer: `Split::Train` is matched in `Mnist::synthetic` at `ferrotorch-vision/src/datasets/mnist.rs:91-94`, `from_dir` at `ferrotorch-vision/src/datasets/mnist.rs:140-143`, and `use ferrotorch_vision::{Mnist, Split};` at `ferrotorch/examples/train_mnist.rs:22` is the binary-target consumer. |
| REQ-3 | SHIPPED | impl: `#[derive(Debug)] pub struct Mnist<T: Float>` at `Mnist in ferrotorch-vision/src/datasets/mnist.rs` per upstream `class MNIST(VisionDataset)` at `torchvision/datasets/mnist.py:20`; non-test consumer: `let train_dataset = Mnist::<f32>::synthetic(...)` at `synthetic in ferrotorch/examples/train_mnist.rs` constructs the struct. |
| REQ-4 | SHIPPED | impl: `pub fn synthetic(split: Split, num_samples: usize) -> FerrotorchResult<Self>` at `synthetic in ferrotorch-vision/src/datasets/mnist.rs` using xorshift64 + per-split seeds; non-test consumer: `Mnist::<f32>::synthetic(Split::Train, num_samples)?` at `ferrotorch/examples/train_mnist.rs` is the runnable example consumer. |
| REQ-5 | SHIPPED | impl: `pub fn from_dir<P: AsRef<Path>>(root: P, split: Split) -> FerrotorchResult<Self>` at `MNIST in ferrotorch-vision/src/datasets/mnist.rs` reading the four canonical IDX files per upstream `torchvision/datasets/mnist.py:122-129` `MNIST._load_data`; non-test consumer: re-exported via `ferrotorch-vision/src/datasets/mod.rs` + `ferrotorch-vision/src/lib.rs` for use in training-driver scripts (the train_mnist example consumes the constructor surface; switching the example from synthetic to from_dir is a one-line change). |
| REQ-6 | SHIPPED | impl: IDX header validation at `ferrotorch-vision/src/datasets/mnist.rs:173-247` with magic checks (`2051` for images, `2049` for labels), length checks, and count-match check per upstream `torchvision/datasets/mnist.py:508-560` `read_sn3_pascalvincent_tensor` + `read_image_file` + `read_label_file`; non-test consumer: `Mnist::from_dir` at `ferrotorch-vision/src/datasets/mnist.rs:138-276` is the in-crate caller of the validation logic (the validation is inlined, so the function itself is the consumer). |
| REQ-7 | SHIPPED | impl: pixel normalization at `ferrotorch-vision/src/datasets/mnist.rs:250-261` with `inv_255 = 1.0_f64 / 255.0` and `cast::<f64, T>(b as f64 * inv_255)` per upstream `torchvision.transforms.functional.to_tensor` chain; non-test consumer: `Mnist::from_dir` at `ferrotorch-vision/src/datasets/mnist.rs:138-276` invokes this normalization on every loaded image. |
| REQ-8 | SHIPPED | impl: `impl<T: Float + 'static> Dataset for Mnist<T>` at `ferrotorch-vision/src/datasets/mnist.rs:284-304` with `len`/`is_empty`/`get` per `ferrotorch_data::Dataset` trait, returning `IndexOutOfBounds` on OOB access; non-test consumer: `let train_dataset = Mnist::<f32>::synthetic(...)?;` at `ferrotorch/examples/train_mnist.rs:60` followed by trait-driven iteration is the production consumer of the `Dataset` impl (the example uses `Mnist` through the dataset trait via the loader pipeline). |
