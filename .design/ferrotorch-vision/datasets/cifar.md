# ferrotorch-vision — CIFAR-10/100 datasets (`datasets/cifar.rs`)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/  (torchvision companion: /home/doll/.local/lib/python3.13/site-packages/torchvision/datasets/cifar.py)
-->

## Summary

`ferrotorch-vision/src/datasets/cifar.rs` provides the CIFAR-10
(32×32 RGB, 10 classes) and CIFAR-100 (32×32 RGB, 100 classes) image-
classification datasets. Each dataset offers a deterministic
`synthetic()` constructor for pipeline testing and a `from_dir()`
loader for the official binary batch files. Both implement
`ferrotorch_data::Dataset` with `CifarSample<T>` as the sample type.
Mirrors `torchvision.datasets.CIFAR10` and `CIFAR100` at
`torchvision/datasets/cifar.py:13-167` — ferrotorch reads the
**binary batch format** (as documented at
https://www.cs.toronto.edu/~kriz/cifar.html) rather than upstream's
**python pickle format** (R-DEV-7 — Rust analog skipping the pickle
dependency).

## Requirements

- REQ-1: `pub struct CifarSample<T: Float>` carries `image:
  Tensor<T>` with shape `[3, 32, 32]` and `label: u8`. Marked
  `#[non_exhaustive]`. Mirrors the `(image, target)` tuple from
  `CIFAR10.__getitem__` at `cifar.py:104-124`.
- REQ-2: `pub struct Cifar10<T: Float>` stores 60 000 32×32 RGB
  samples with `label ∈ 0..10`. `pub const HEIGHT/WIDTH/CHANNELS/
  NUM_CLASSES` constants for ergonomic buffer allocation. Mirrors
  `class CIFAR10(VisionDataset)` at `cifar.py:13`.
- REQ-3: `pub struct Cifar100<T: Float>` is the 100-class variant
  storing the **fine label** (`label ∈ 0..100`). The coarse label
  is read but discarded — same as upstream's
  `entry["fine_labels"]` at `cifar.py:88` (CIFAR-100 pickle exposes
  both `coarse_labels` and `fine_labels`; torchvision picks the
  fine one).
- REQ-4: `Cifar10::<T>::synthetic(split, num_samples)` and
  `Cifar100::<T>::synthetic(split, num_samples)` generate
  deterministic random samples seeded by split and dataset (CIFAR-10
  vs CIFAR-100 use distinct seeds). No upstream counterpart — the
  pipeline-test entry point.
- REQ-5: `Cifar10::<T>::from_dir(root, split)` reads
  `data_batch_1.bin` .. `data_batch_5.bin` for the train split and
  `test_batch.bin` for the test split. CIFAR-100's `from_dir` reads
  `train.bin` and `test.bin`. Mirrors `CIFAR10.__init__`'s file
  enumeration (`cifar.py:35-45`) but for the **binary** format
  (which uses a 1-byte header for CIFAR-10 and a 2-byte header for
  CIFAR-100).
- REQ-6: Binary batch format parsing: each sample is `header_bytes +
  3072` (where `header_bytes == 1` for CIFAR-10 and `2` for
  CIFAR-100, with the fine label being the **second** byte). Pixel
  bytes are channel-major (R plane, then G, then B; each plane is
  `32 × 32 = 1024` bytes). Mirrors the CIFAR binary layout
  documented at https://www.cs.toronto.edu/~kriz/cifar.html and is
  the wire-format-mirror of upstream's `pickle.load` → `np.vstack`
  → `reshape(-1, 3, 32, 32)` chain (`cifar.py:79-91`).
- REQ-7: Pixel values are normalized from `u8 ∈ [0, 255]` to `T ∈
  [0.0, 1.0]` via `cast::<f64, T>(b as f64 * inv_255)` (where
  `inv_255 = 1.0 / 255.0`). Matches the
  `torchvision.transforms.functional.to_tensor` chain users apply
  after the dataset.
- REQ-8: `impl Dataset for Cifar10<T>` and `impl Dataset for
  Cifar100<T>` expose `len`/`is_empty`/`get` per
  `ferrotorch_data::Dataset`. `get(index)` returns
  `IndexOutOfBounds { index, axis: 0, size }` on OOB access.
- REQ-9: Label range is validated at load time: bytes ≥
  `NUM_CLASSES` (10 for CIFAR-10, 100 for CIFAR-100) return
  `InvalidArgument`. Upstream relies on the pickle's data being
  trusted; ferrotorch validates the on-disk bytes.

## Acceptance Criteria

- [x] AC-1: `CifarSample<T>` is `#[non_exhaustive]` with `image` /
  `label` fields.
- [x] AC-2: `Cifar10::<f32>::synthetic(Split::Train, n).unwrap()
  .len() == n` with sample images shaped `[3, 32, 32]` and
  `label < 10`.
- [x] AC-3: `Cifar100::<f32>::synthetic(Split::Train, n)` has
  `label < 100`.
- [x] AC-4: CIFAR-10 and CIFAR-100 synthetic outputs differ for
  identical `n` (distinct seeds).
- [x] AC-5: Train and test synthetic outputs differ for identical
  `n` (distinct seeds within each family).
- [x] AC-6: `from_dir` rejects missing batch files with
  `InvalidArgument`.
- [x] AC-7: `from_dir` rejects bad batch length (not a multiple of
  `bytes_per_sample`) with `InvalidArgument`.
- [x] AC-8: `from_dir` rejects out-of-range labels with
  `InvalidArgument`.
- [x] AC-9: `Cifar10` / `Cifar100` / `CifarSample` are `Send +
  Sync` (compile-time assertion).

## Architecture

The file is structured as: shared sample type and constants, the
`Cifar10` block (synthetic + from_dir + Dataset impl), the
`Cifar100` block (same shape), then the shared binary-batch reader
`load_cifar_batches` and the shared `generate_synthetic` PRNG core.

### Shared constants (`cifar.rs:65-69`, `cifar.rs:312`)

`HEIGHT = 32`, `WIDTH = 32`, `CHANNELS = 3`, and `BYTES_PER_IMAGE =
CHANNELS * HEIGHT * WIDTH = 3072` are file-private constants.
Re-exposed as `pub const Cifar10::HEIGHT` / etc. on each struct so
callers can write `let buf = vec![0.0; Cifar10::HEIGHT *
Cifar10::WIDTH * Cifar10::CHANNELS];` without hard-coding the values
(R-DEV-2 — public API ergonomics).

### `CifarSample<T>` (REQ-1)

Marked `#[non_exhaustive]` (`cifar.rs:52`) so future per-sample
metadata (e.g. coarse-label exposure for CIFAR-100, super-class id)
can land without breaking struct literals.

### `Cifar10` / `Cifar100` structs (REQ-2, REQ-3)

Each stores `images: Vec<Tensor<T>>`, `labels: Vec<u8>`, `split:
Split`. The difference is purely the `NUM_CLASSES` constant (10 vs
100) and the on-disk filenames (`data_batch_*.bin` vs `train.bin`).

For CIFAR-100, the **fine label** is the second byte of the
2-byte sample header (`cifar.rs:367-370`): `bytes[sample_start + 1]`.
The first byte (coarse label, 0..20) is read but discarded — this
matches upstream's `entry["fine_labels"]` selection at
`cifar.py:88`.

### `synthetic` (REQ-4)

The shared `generate_synthetic` helper (`cifar.rs:405-439`) is
parameterized by `num_classes` (10 or 100) and a pair of `(seed_train,
seed_test)` u64s. Each dataset picks its own seeds:
- CIFAR-10: `0xc1fa_0010_0001` (train), `0xc1fa_0010_0002` (test)
- CIFAR-100: `0xc1fa_0100_0001` (train), `0xc1fa_0100_0002` (test)

Identical `(num_samples, split)` produces byte-identical output
across runs and across machines (the xorshift64 is deterministic).
Distinct seeds ensure CIFAR-10 ≠ CIFAR-100 (test
`test_cifar100_different_from_cifar10` at `cifar.rs:579-590`).

### `from_dir` + `load_cifar_batches` (REQ-5, REQ-6, REQ-7, REQ-9)

`load_cifar_batches` (`cifar.rs:318-398`) is the shared binary
reader. The format selector enum `CifarFormat::{Cifar10, Cifar100}`
(`cifar.rs:303-309`) carries `header_bytes` (1 or 2) and the offset
into the header for the label byte (`0` or `1`).

Each batch file is read into memory in one `std::fs::read` call
(`cifar.rs:345-347`). The total length is validated to be a multiple
of `bytes_per_sample = header_bytes + 3072` (`cifar.rs:349-358`),
preventing partial-record reads.

Per-sample loop (`cifar.rs:363-394`):
1. Extract `label` from the header.
2. Validate `label < num_classes`; if not, error out with the file
   path + sample index.
3. Slice the 3072 pixel bytes (already in channel-major order — the
   CIFAR binary format stores them this way, so no transpose is
   needed; this is faster than upstream's
   `reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))` HWC conversion
   which adds an unnecessary memory-touch).
4. Map each byte through `cast::<f64, T>(b as f64 * inv_255)` to get
   the normalized float.
5. Build a `Tensor<T>` of shape `[3, 32, 32]` and push.

### `impl Dataset` (REQ-8)

`cifar.rs:165-185` (Cifar10) and `cifar.rs:276-296` (Cifar100) both
forward `len` to `self.images.len()` and return `IndexOutOfBounds`
on OOB `get`. Constructor returns owned `CifarSample` by cloning the
stored tensor (matches the Dataset trait's by-value semantics).

### Non-test production consumers

- `ferrotorch-vision/src/datasets/mod.rs:10` — `pub use cifar::{Cifar10,
  Cifar100, CifarSample};` re-exports the type surface.
- `ferrotorch-vision/src/lib.rs:99` — `pub use datasets::{Cifar10,
  Cifar100, CifarSample, ...};` is the crate-root re-export.
- `ferrotorch-vision/tests/conformance_vision_datasets.rs:30` — uses
  `Cifar10` and `Cifar100` (test consumer; counts towards REQ-1's
  documentation but the non-test consumer for SHIPPED claim is the
  re-export surface that downstream training drivers reach).

Honest underclaim: this crate ships `Cifar10` / `Cifar100` as
boundary public types reached via re-exports. The fully-runnable
downstream consumer that exists today is `train_mnist.rs` (for
`Mnist`); a parallel `train_cifar.rs` is the obvious next consumer
but does not exist yet. The re-export surface is the production
consumer per R-DEFER-1's "existing pub API is grandfathered" clause
— the types ARE the public boundary.

## Parity contract

`parity_ops = []`. Edge cases preserved:

- **Empty dataset**: `synthetic(split, 0)` returns `len() == 0`;
  `is_empty()` is `true`. No upstream constraint.
- **Train/test seed divergence**: distinct seeds guarantee different
  bytes between splits. Verified by `test_cifar10_train_test_different`
  (`cifar.rs:522-531`).
- **CIFAR-10 vs CIFAR-100 seed divergence**: different seed bases
  (`c1fa_0010` vs `c1fa_0100`) guarantee disjoint synthetic streams.
  Verified by `test_cifar100_different_from_cifar10`.
- **Pixel normalization**: `0u8 → 0.0`, `255u8 → 1.0` exactly in
  IEEE 754. Matches upstream's `to_tensor` convention.
- **Label validation**: bytes ≥ `num_classes` return
  `InvalidArgument` at load time. Upstream trusts the pickle.
- **Channel-major storage**: ferrotorch keeps the on-disk channel-
  major layout (R plane → G plane → B plane); upstream rearranges
  to HWC then back to CHW via `transpose((0, 2, 3, 1))` — both end
  up as `[3, 32, 32]` tensors with the same pixel values, so the
  consumer-facing contract is identical.
- **Send + Sync**: `test_cifar10_is_send_sync` and
  `test_cifar100_is_send_sync` (`cifar.rs:540-545` and
  `cifar.rs:592-596`) statically assert.

Divergences from upstream (deliberate):

- **Binary format, not pickle**: ferrotorch reads the upstream
  binary format (which is documented at the CIFAR website and is
  format-stable). Pickle would require either a pickle-parser
  dependency or a custom pickle subset reader — neither is worth
  it for a dataset whose binary form is canonical and trivially
  parseable (R-DEV-7 — Rust ecosystem analog).
- **No `transform` / `target_transform` kwargs**: composed externally
  via `MappedDataset`.
- **No `download = True`**: same as MNIST — files must be present.

## Verification

Unit tests in `mod tests` of `ferrotorch-vision/src/datasets/cifar.rs`
(`cifar.rs:450-596`):

**CIFAR-10**:
- `test_cifar10_synthetic_train_len` / `_test_len` / `_empty` —
  length contract.
- `test_cifar10_sample_image_shape` — `[3, 32, 32]` shape.
- `test_cifar10_sample_values_in_range` — `[0, 1]` range.
- `test_cifar10_label_range` — `label < 10`.
- `test_cifar10_out_of_bounds` — `get(n)` errors.
- `test_cifar10_split_accessor` — round-trip.
- `test_cifar10_train_test_different` — seed divergence.
- `test_cifar10_f64` — `f64` path.
- `test_cifar10_is_send_sync` — compile-time assertion.

**CIFAR-100**:
- `test_cifar100_synthetic_len` / `_sample_shape` /
  `_label_range` / `_out_of_bounds` / `_is_send_sync` — parallel
  contract.
- `test_cifar100_different_from_cifar10` — distinct seed family.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-vision --lib datasets::cifar 2>&1 | tail -3
```

Expected: 17 passed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `#[non_exhaustive] pub struct CifarSample<T: Float>` at `ferrotorch-vision/src/datasets/cifar.rs:51-58` (image + label fields) per upstream `torchvision/datasets/cifar.py:104-124` `__getitem__`; non-test consumer: `Cifar10::get` at `ferrotorch-vision/src/datasets/cifar.rs:172-184` and `Cifar100::get` at `ferrotorch-vision/src/datasets/cifar.rs:283-295` both construct `CifarSample`; re-exported via `ferrotorch-vision/src/datasets/mod.rs:10`. |
| REQ-2 | SHIPPED | impl: `#[derive(Debug)] pub struct Cifar10<T: Float>` at `ferrotorch-vision/src/datasets/cifar.rs:80-85` with `pub const HEIGHT/WIDTH/CHANNELS/NUM_CLASSES` at `ferrotorch-vision/src/datasets/cifar.rs:88-95` per upstream `torchvision/datasets/cifar.py:13`; non-test consumer: re-exported at `ferrotorch-vision/src/datasets/mod.rs:10` → `ferrotorch-vision/src/lib.rs:99` which `ferrotorch/src/lib.rs:71` glob-imports, surfacing `ferrotorch::Cifar10` to the meta-crate users. |
| REQ-3 | SHIPPED | impl: `#[derive(Debug)] pub struct Cifar100<T: Float>` at `ferrotorch-vision/src/datasets/cifar.rs:196-201` with `NUM_CLASSES = 100` at `ferrotorch-vision/src/datasets/cifar.rs:211`, and the fine-label extraction `bytes[sample_start + 1]` at `ferrotorch-vision/src/datasets/cifar.rs:368-370` per upstream `entry["fine_labels"]` at `torchvision/datasets/cifar.py:88`; non-test consumer: re-exported at `ferrotorch-vision/src/datasets/mod.rs:10` and used through that chain. |
| REQ-4 | SHIPPED | impl: `Cifar10::synthetic` at `ferrotorch-vision/src/datasets/cifar.rs:108-121` (seeds `0xc1fa_0010_0001/0002`) and `Cifar100::synthetic` at `ferrotorch-vision/src/datasets/cifar.rs:224-237` (seeds `0xc1fa_0100_0001/0002`), both calling the shared `generate_synthetic` at `ferrotorch-vision/src/datasets/cifar.rs:405-439`; non-test consumer: re-exported via `ferrotorch-vision/src/datasets/mod.rs:10` (the synthetic constructor is the documented test-pipeline entry point; downstream callers reach it through the re-export chain). |
| REQ-5 | SHIPPED | impl: `Cifar10::from_dir` at `ferrotorch-vision/src/datasets/cifar.rs:136-157` (5 train batches + test_batch.bin) and `Cifar100::from_dir` at `ferrotorch-vision/src/datasets/cifar.rs:253-268` (train.bin + test.bin) per upstream `torchvision/datasets/cifar.py:35-45` `train_list`/`test_list`; non-test consumer: re-exported through `ferrotorch-vision/src/datasets/mod.rs:10` → `ferrotorch-vision/src/lib.rs:99` for downstream training-driver scripts. |
| REQ-6 | SHIPPED | impl: `load_cifar_batches` at `ferrotorch-vision/src/datasets/cifar.rs:318-398` parses the CIFAR binary batch format with the `CifarFormat::{Cifar10, Cifar100}` enum at `ferrotorch-vision/src/datasets/cifar.rs:303-309` selecting 1-byte vs 2-byte header per the official CIFAR binary format documentation (cited at the cifar.rs module docstring); non-test consumer: invoked from `Cifar10::from_dir` at `ferrotorch-vision/src/datasets/cifar.rs:149-150` and `Cifar100::from_dir` at `ferrotorch-vision/src/datasets/cifar.rs:260-261`. |
| REQ-7 | SHIPPED | impl: pixel normalization at `ferrotorch-vision/src/datasets/cifar.rs:361,385-388` with `inv_255 = 1.0_f64 / 255.0` per upstream `torchvision.transforms.functional.to_tensor` chain; non-test consumer: `load_cifar_batches` applies this to every loaded pixel, and is itself called from both `Cifar10::from_dir` and `Cifar100::from_dir` (production load paths). |
| REQ-8 | SHIPPED | impl: `impl<T: Float + 'static> Dataset for Cifar10<T>` at `ferrotorch-vision/src/datasets/cifar.rs:165-185` and `impl<T: Float + 'static> Dataset for Cifar100<T>` at `ferrotorch-vision/src/datasets/cifar.rs:276-296` per `ferrotorch_data::Dataset` trait; non-test consumer: the re-export chain at `ferrotorch-vision/src/datasets/mod.rs:10` + `ferrotorch-vision/src/lib.rs:99` exposes the types implementing the trait to downstream training pipelines (the trait method `get` is the per-sample iteration contract). |
| REQ-9 | SHIPPED | impl: label-range validation at `ferrotorch-vision/src/datasets/cifar.rs:371-378` (`if label as usize >= num_classes { return Err(...) }`); non-test consumer: `load_cifar_batches` is the in-crate caller (the validation is inlined). |
