# ferrotorch-vision — ImageFolder / DatasetFolder (`datasets/folder.rs`)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/  (torchvision companion: /home/doll/.local/lib/python3.13/site-packages/torchvision/datasets/folder.py)
-->

## Summary

`ferrotorch-vision/src/datasets/folder.rs` provides `ImageFolder`
(class-per-subdirectory image dataset) and `DatasetFolder` (its
generic loader-parameterized variant). Mirrors
`torchvision.datasets.ImageFolder` and `DatasetFolder` at
`torchvision/datasets/folder.py:109-317`. Class names are the
alphabetically-sorted subdirectory basenames; per-class files are
discovered with a case-insensitive extension filter (default:
`IMG_EXTENSIONS = ["jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif",
"tiff", "webp"]` — matches upstream's `IMG_EXTENSIONS` at
`folder.py:257`). Image decoding is delegated to
`crate::io::read_image_as_tensor` (R-DEV-7 — Rust analog of the
PIL `pil_loader`).

## Requirements

- REQ-1: `pub struct ImageSample<T: Float>` carries `image:
  Tensor<T>` of shape `[C, H, W]` (typically `[3, H, W]` for RGB)
  and `label: u32`. Marked `#[non_exhaustive]`. Mirrors the
  `(sample, target)` tuple from `DatasetFolder.__getitem__` at
  `folder.py:236-251`.
- REQ-2: `pub const IMG_EXTENSIONS: &[&str]` lists the nine
  default extensions (without leading dot). Mirrors upstream's
  `IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp",
  ".pgm", ".tif", ".tiff", ".webp")` at `folder.py:257`. ferrotorch
  drops the leading dot since `Path::extension()` returns the
  extension without it.
- REQ-3: `pub struct ImageFolder<T: Float>` walks a root directory,
  discovers one class per subdirectory, and stores `(PathBuf,
  class_idx)` pairs. Class names are alphabetically sorted; per-
  class file lists are alphabetically sorted too. Mirrors
  `class ImageFolder(DatasetFolder)` at `folder.py:287-340`.
- REQ-4: `ImageFolder::from_dir(root)` is the default constructor;
  `from_dir_with_extensions(root, &extensions)` accepts a custom
  extension list (empty slice = accept all files);
  `from_dir_with_filter(root, predicate)` accepts a custom
  `Fn(&Path) -> bool` predicate (applied after the extension
  check). Mirrors upstream's `extensions` / `is_valid_file` kwargs
  to `DatasetFolder.__init__` at `folder.py:138-156`.
- REQ-5: `impl Dataset for ImageFolder<T>` defers image decode to
  `crate::io::read_image_as_tensor::<T>(path)` — files are NOT
  pre-loaded into memory. The directory scan only collects file
  paths. Mirrors upstream's lazy loader pattern: `loader(path)` at
  `folder.py:245` is called per-`__getitem__`.
- REQ-6: `pub struct DatasetFolder<S, F: Fn(&Path) ->
  FerrotorchResult<S>>` is the generic version — wraps any
  `Fn(&Path) -> FerrotorchResult<S>` loader. The `S` is whatever
  the loader returns. Mirrors `class DatasetFolder(VisionDataset)`
  at `folder.py:109`.
- REQ-7: `pub struct FolderSample<S>` is the `DatasetFolder` sample
  type — `(data: S, label: u32)`. Mirrors the `(sample, target)`
  tuple from `DatasetFolder.__getitem__`.
- REQ-8: The directory walk skips hidden dotfiles (`.DS_Store`,
  etc.) AND hidden subdirectories (`.hidden_class/`). Mirrors
  upstream's `os.scandir(directory) if entry.is_dir()` at
  `folder.py:41` — upstream does not explicitly skip dotfiles but
  the standard convention is that hidden dirs are not classes;
  ferrotorch makes this explicit (R-DEV-2 — match the user's
  expectation, not the literal upstream byte-for-byte behaviour).
- REQ-9: `class_to_idx()` returns a `HashMap<&str, u32>` matching
  upstream's `class_to_idx` dict attribute at `folder.py:162`.
  `classes()` returns the sorted class-name list.
- REQ-10: Non-directory root rejected with `InvalidArgument`.
  Empty root (no subdirectories) returns an empty dataset (matches
  torchvision's behavior when `allow_empty=False` and no classes
  exist — though ferrotorch is more permissive: returning empty
  rather than raising, matching the "empty classes are
  allowed if no subdirs at all" interpretation).

## Acceptance Criteria

- [x] AC-1: `ImageSample<T>` is `#[non_exhaustive]` with `image` /
  `label` fields.
- [x] AC-2: `IMG_EXTENSIONS` slice matches the nine
  torchvision-canonical extensions (without leading dot).
- [x] AC-3: `ImageFolder::from_dir(tempdir)` discovers classes in
  alphabetical order.
- [x] AC-4: Each class's files appear with their alphabetically-
  sorted order in `samples()`.
- [x] AC-5: Extension matching is case-insensitive (`upper.JPG`,
  `mixed.PnG` both accepted).
- [x] AC-6: Hidden files (`.DS_Store`) and hidden dirs
  (`.hidden_class/`) are skipped.
- [x] AC-7: `from_dir_with_filter` predicate is applied after the
  extension filter.
- [x] AC-8: `ImageFolder::get` actually decodes a 2×2 PNG into a
  `[1, 2, 2]` tensor (grayscale → single channel after `to_rgb8`
  is RGB-replicated to 3 channels, so shape is `[3, 2, 2]` in
  practice — the test asserts `shape.len() == 3 && shape[1] == 2
  && shape[2] == 2`).
- [x] AC-9: `DatasetFolder` is generic over the sample type — the
  test uses `DatasetFolder<usize, _>` returning byte-count.
- [x] AC-10: `from_dir` with empty extension slice accepts ALL
  files in each class dir.
- [x] AC-11: Non-directory root returns `InvalidArgument`.

## Architecture

### `ImageSample<T>` (REQ-1)

`#[non_exhaustive]` at `folder.rs:33` — same pattern as the other
sample types. `label` is `u32` (matches the class-count upper bound
of ~4 billion; upstream's `class_index` is a Python int).

### `IMG_EXTENSIONS` (REQ-2)

`pub const IMG_EXTENSIONS: &[&str]` at `folder.rs:43-45`. The slice
of 9 strings (no leading dot) is matched case-insensitively against
each file's `Path::extension()` via the `has_extension_ci` helper
at `folder.rs:346-354`.

### `ImageFolder<T>` (REQ-3, REQ-4)

The struct holds `samples: Vec<(PathBuf, u32)>`, `classes:
Vec<String>`, and a `PhantomData<T>` since `T` is only used in the
`Dataset` impl (`folder.rs:50-54`).

Three constructors:
- `from_dir(root)` — default, uses `IMG_EXTENSIONS`
  (`folder.rs:69-71`).
- `from_dir_with_extensions(root, extensions)` — custom list,
  empty slice means accept-all (`folder.rs:75-85`).
- `from_dir_with_filter(root, predicate)` — applies a closure
  predicate after the extension filter (`folder.rs:91-101`).

All three delegate to `scan_class_dirs` (`folder.rs:270-343`).

### `scan_class_dirs` (REQ-3, REQ-8)

1. Reject non-directory root with `InvalidArgument` (REQ-10).
2. `read_dir(root)` to discover class subdirectories. Skip
   non-directories and dotfile-named entries (REQ-8).
3. Sort class directories by basename
   (`class_dirs.sort_by(|a, b| a.0.cmp(&b.0))` at `folder.rs:305`).
4. For each class (in sorted order), `read_dir(class_dir)` to
   discover files. Skip non-files, dotfiles, and files whose
   extension is not in the allowlist (case-insensitive).
5. Sort files alphabetically within each class
   (`files.sort()` at `folder.rs:337`).

The double-sort is what gives ferrotorch the determinism property
that's load-bearing for reproducible training: a directory's
contents are identical across machines / OS file-system iteration
orders.

### `Dataset` impl (REQ-5)

`folder.rs:128-151`. `get(index)` looks up the `(path, label)` pair
and calls `crate::io::read_image_as_tensor::<T>(path)` — image
decode happens lazily on demand. This is critical for ImageFolder's
memory footprint: a 100k-image dataset keeps only paths in RAM, not
decoded tensors.

### `DatasetFolder<S, F>` (REQ-6, REQ-7)

`folder.rs:156-160` defines the generic loader-parameterized
variant. The `F: Fn(&Path) -> FerrotorchResult<S>` bound makes the
loader a closure or `fn` pointer; the resulting `S` is whatever the
loader returns (a tensor, a `Vec<u8>`, a custom sample struct,
etc.). The `Debug` impl skips the loader field with
`#[allow(clippy::missing_fields_in_debug)]` since closures don't
implement `Debug`.

Three constructors mirror the `ImageFolder` triplet:
`from_dir(root, &extensions, loader)`, `from_dir_with_filter(...)`,
and a `Send + Sync + 'static` bound on the loader for the
`Dataset` impl.

`FolderSample<S>` (`folder.rs:163-169`) is the sample type with
`data: S, label: u32`.

### `class_to_idx` (REQ-9)

`folder.rs:114-120`. Returns `HashMap<&str, u32>` mapping class
names to their sorted index. Matches upstream's `class_to_idx`
dict semantics (`folder.py:46`).

### Non-test production consumers

- `ferrotorch-vision/src/datasets/mod.rs:11` — `pub use folder::{DatasetFolder,
  FolderSample, IMG_EXTENSIONS, ImageFolder, ImageSample};` is the
  re-export.
- `ImageFolder::get` at `ferrotorch-vision/src/datasets/folder.rs:135-150`
  is itself the production consumer of `crate::io::read_image_as_tensor`
  — the cross-module call surfaces the io.rs production binding.
- `ferrotorch-vision/src/lib.rs:99` re-exports a subset (no `ImageFolder`
  / `DatasetFolder` listed at the crate root, only the dataset wrappers
  for the most-used types).

Honest underclaim: external production drivers (binaries, training
loops) that exercise `ImageFolder::from_dir` against a real
directory tree do not yet ship in the workspace. The non-test
consumer for the SHIPPED claim is the in-crate cross-module call
`ImageFolder::get → crate::io::read_image_as_tensor` (which IS
production code, not test code) + the re-export at
`datasets/mod.rs:11` that makes the type reachable from outside.
Per R-DEFER-1's grandfather clause, the boundary type IS the public
API and doesn't need a further downstream caller within the
ferrotorch monorepo.

## Parity contract

`parity_ops = []`. Edge cases preserved:

- **Sort order**: alphabetical class names, alphabetical file
  names per class. Verified by
  `image_folder_discovers_classes_in_alphabetical_order` and
  `image_folder_collects_all_files_per_class` tests.
- **Case-insensitive extension**: `upper.JPG`, `mixed.PnG` both
  accepted. Verified by `image_folder_extension_matching_is_case_insensitive`.
- **Dotfile skip**: `.DS_Store`, `.hidden_class/` both skipped.
  Verified by `image_folder_skips_dotfiles_and_dot_dirs`.
- **Empty-extensions = accept all**: passes the `extensions.is_empty()`
  short-circuit at `folder.rs:329` — every file passes the extension
  check. Verified by `dataset_folder_with_no_extensions_accepts_all_files`.
- **OOB get**: returns `InvalidArgument` (NOT
  `IndexOutOfBounds` — folder.rs uses
  `FerrotorchError::InvalidArgument` for the
  `ImageFolder::get(out-of-range)` case at `folder.rs:138-144`,
  unlike CIFAR/MNIST which use `IndexOutOfBounds`). This is a
  documented divergence; either error variant is acceptable per the
  trait contract. Verified by `image_folder_get_out_of_range_errors`.
- **Non-dir root**: `InvalidArgument`. Verified by
  `image_folder_rejects_non_directory_root`.

Divergences from upstream (deliberate):

- **`allow_empty` kwarg absent**: upstream raises if a class dir is
  empty (unless `allow_empty=True`); ferrotorch silently emits an
  empty class. The class list stays the union of present
  subdirectories. Acceptable since downstream training code will
  typically validate via `len() > 0` regardless.
- **No `transform` / `target_transform` kwargs**: composed externally
  via `MappedDataset`.
- **No `loader` kwarg on `ImageFolder`**: `ImageFolder` always uses
  `read_image_as_tensor`; for custom decoding, use `DatasetFolder`
  directly.

## Verification

Unit tests in `mod tests` of
`ferrotorch-vision/src/datasets/folder.rs` (`folder.rs:356-559`):

- `image_folder_discovers_classes_in_alphabetical_order`
- `image_folder_collects_all_files_per_class`
- `image_folder_filters_unknown_extensions`
- `image_folder_extension_matching_is_case_insensitive`
- `image_folder_skips_dotfiles_and_dot_dirs`
- `image_folder_with_filter_drops_predicate_rejects`
- `image_folder_get_reads_real_image` — end-to-end PNG decode
- `image_folder_get_out_of_range_errors`
- `image_folder_rejects_non_directory_root`
- `dataset_folder_uses_custom_loader` — generic loader path
- `dataset_folder_with_no_extensions_accepts_all_files`

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-vision --lib datasets::folder 2>&1 | tail -3
```

Expected: 11 passed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `#[non_exhaustive] pub struct ImageSample<T: Float>` at `ferrotorch-vision/src/datasets/folder.rs:32-39` per upstream `torchvision/datasets/folder.py:236-251` `__getitem__` returning `(sample, target)`; non-test consumer: `ImageFolder::get` at `ferrotorch-vision/src/datasets/folder.rs:135-150` constructs `ImageSample`, and the type is re-exported at `ferrotorch-vision/src/datasets/mod.rs:11`. |
| REQ-2 | SHIPPED | impl: `pub const IMG_EXTENSIONS: &[&str]` at `ferrotorch-vision/src/datasets/folder.rs:43-45` per upstream `IMG_EXTENSIONS` tuple at `torchvision/datasets/folder.py:257`; non-test consumer: `ImageFolder::from_dir` at `ferrotorch-vision/src/datasets/folder.rs:69-71` uses `IMG_EXTENSIONS` as the default extension list, and the constant is re-exported at `ferrotorch-vision/src/datasets/mod.rs:11`. |
| REQ-3 | SHIPPED | impl: `pub struct ImageFolder<T: Float>` at `ferrotorch-vision/src/datasets/folder.rs:50-54` per upstream `class ImageFolder(DatasetFolder)` at `torchvision/datasets/folder.py:287`; non-test consumer: re-exported at `ferrotorch-vision/src/datasets/mod.rs:11`, and `ImageFolder::get` at `ferrotorch-vision/src/datasets/folder.rs:135-150` is the cross-module production consumer that calls `crate::io::read_image_as_tensor` (linking the two routes' production surfaces). |
| REQ-4 | SHIPPED | impl: three constructors at `ferrotorch-vision/src/datasets/folder.rs:69-101` (`from_dir`, `from_dir_with_extensions`, `from_dir_with_filter`) per upstream's `extensions` / `is_valid_file` kwargs at `torchvision/datasets/folder.py:138-156`; non-test consumer: re-exported through `ferrotorch-vision/src/datasets/mod.rs:11`. (Honest underclaim: external training-driver consumers do not yet ship in the monorepo; the re-export IS the public binding per R-DEFER-1 grandfather.) |
| REQ-5 | SHIPPED | impl: `impl<T: Float + 'static> Dataset for ImageFolder<T>` at `ferrotorch-vision/src/datasets/folder.rs:128-151` with lazy decode via `crate::io::read_image_as_tensor::<T>(path)?` at `ferrotorch-vision/src/datasets/folder.rs:145` per upstream's `loader(path)` lazy pattern at `torchvision/datasets/folder.py:245`; non-test consumer: the `read_image_as_tensor` call at `folder.rs:145` is itself the cross-module production consumer of `ferrotorch-vision/src/io.rs:98-101`. |
| REQ-6 | SHIPPED | impl: `pub struct DatasetFolder<S, F: Fn(&Path) -> FerrotorchResult<S>>` at `ferrotorch-vision/src/datasets/folder.rs:156-160` with the `Send + Sync + 'static` Dataset impl at `ferrotorch-vision/src/datasets/folder.rs:233-260` per upstream `class DatasetFolder(VisionDataset)` at `torchvision/datasets/folder.py:109`; non-test consumer: re-exported at `ferrotorch-vision/src/datasets/mod.rs:11` so downstream code can plug in custom loaders (audio, segmentation-mask, etc.). |
| REQ-7 | SHIPPED | impl: `#[derive(Debug, Clone)] pub struct FolderSample<S>` at `ferrotorch-vision/src/datasets/folder.rs:163-169` per upstream's `(sample, target)` tuple convention; non-test consumer: `DatasetFolder::get` at `ferrotorch-vision/src/datasets/folder.rs:244-259` constructs `FolderSample`, and the type is re-exported at `ferrotorch-vision/src/datasets/mod.rs:11`. |
| REQ-8 | SHIPPED | impl: dotfile / dot-dir skip logic at `ferrotorch-vision/src/datasets/folder.rs:300-302` (class-dir check) and `ferrotorch-vision/src/datasets/folder.rs:324-328` (per-file check); non-test consumer: `scan_class_dirs` at `ferrotorch-vision/src/datasets/folder.rs:270-343` invokes this on every walk, and is called from all three `ImageFolder` constructors at `ferrotorch-vision/src/datasets/folder.rs:79,95,194,209`. |
| REQ-9 | SHIPPED | impl: `class_to_idx` at `ferrotorch-vision/src/datasets/folder.rs:114-120` returning `HashMap<&str, u32>`, and `classes` at `ferrotorch-vision/src/datasets/folder.rs:104-106` per upstream `class_to_idx` dict at `torchvision/datasets/folder.py:46,162`; non-test consumer: re-exported via `ImageFolder` whose `class_to_idx`/`classes` methods are reachable through the meta-crate. |
| REQ-10 | SHIPPED | impl: non-directory rejection at `ferrotorch-vision/src/datasets/folder.rs:275-282` (`if !root.is_dir() { return Err(InvalidArgument {...}) }`); non-test consumer: `scan_class_dirs` is the in-crate caller invoked by every `from_dir*` constructor at `ferrotorch-vision/src/datasets/folder.rs:79,95,194,209`. |
