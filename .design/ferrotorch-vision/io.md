# ferrotorch-vision — image I/O (`io.rs`)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/  (torchvision companion: /home/doll/.local/lib/python3.13/site-packages/torchvision/io/image.py + torchvision/datasets/folder.py:260-264 pil_loader)
-->

## Summary

`ferrotorch-vision/src/io.rs` provides image-file I/O: read PNG/JPEG/
BMP/GIF/etc. files into raw u8 buffers or directly into `[C, H, W]`
float tensors, and write `[C, H, W]` float tensors back to disk.
Mirrors the `torchvision.io.read_image` / `decode_image` family
(`torchvision/io/image.py:90-176`) and the `pil_loader` convention
(`torchvision/datasets/folder.py:260-264`), but routes through the
Rust ecosystem's `image` crate (R-DEV-7 — Rust analog of PIL) rather
than libpng/libjpeg-via-FFI.

## Requirements

- REQ-1: `pub struct RawImage` — intermediate HWC u8 representation
  between on-disk files and tensors. Carries `data: Vec<u8>`, `width:
  u32`, `height: u32`, `channels: u32`. Marked `#[non_exhaustive]` so
  future extensions (stride, color-space tag) can land without
  breaking struct-literal construction outside the crate. Mirrors
  the role of `PIL.Image.Image` as the intermediate representation
  in torchvision's `pil_loader` (`torchvision/datasets/folder.py:260-
  264`).
- REQ-2: `pub fn read_image(path) -> FerrotorchResult<RawImage>`
  reads an image file from disk, decodes via the `image` crate, and
  returns a 3-channel RGB `RawImage`. Alpha is discarded. Mirrors
  `torchvision.io.read_image` (`torchvision/io/image.py:154-187`)
  with `ImageReadMode.RGB`. Errors are returned, not panicked
  (R-CODE-2).
- REQ-3: `pub fn read_image_rgba(path) -> FerrotorchResult<RawImage>`
  preserves the alpha channel — returns a 4-channel RGBA `RawImage`.
  Mirrors `torchvision.io.read_image` with `ImageReadMode.RGB_ALPHA`.
- REQ-4: `pub fn read_image_as_tensor<T: Float>(path) ->
  FerrotorchResult<Tensor<T>>` is the convenience one-shot: read
  the image, convert HWC u8 → CHW float in `[0.0, 1.0]`. Mirrors the
  full PyTorch pipeline `read_image(path).float() / 255.0` from the
  `pil_loader` → `ToTensor` composition.
- REQ-5: `pub fn raw_image_to_tensor<T: Float>(&RawImage) ->
  FerrotorchResult<Tensor<T>>` performs the explicit HWC→CHW
  transposition + `1/255` scaling. Useful when the caller has the
  raw bytes already (e.g. from a network response). Mirrors the
  `torchvision.transforms.functional.to_tensor` PIL→Tensor conversion
  (R-DEV-2 — public API).
- REQ-6: `pub fn write_image(path, &RawImage) -> FerrotorchResult<()>`
  serializes a `RawImage` to disk; output format inferred from
  file extension via the `image` crate. Supports 3-channel (RGB) and
  4-channel (RGBA) buffers; other channel counts return
  `InvalidArgument`. Mirrors `torchvision.io.write_png` /
  `torchvision.io.write_jpeg` (`torchvision/io/image.py:213-296`).
- REQ-7: `pub fn write_tensor_as_image<T: Float>(path, &Tensor<T>) ->
  FerrotorchResult<()>` is the float-tensor → file shortcut: clamp
  `[0, 1]`, scale to `[0, 255]`, transpose CHW→HWC, write. Mirrors
  `torchvision.utils.save_image` (after the implicit clamping the
  `make_grid` family applies).
- REQ-8: `pub fn tensor_to_raw_image<T: Float>(&Tensor<T>) ->
  FerrotorchResult<RawImage>` is the explicit CHW→HWC + clamp +
  scale conversion. The tensor MUST be 3-D `[C, H, W]` with `C == 3
  || C == 4`; any other shape returns `InvalidArgument`. Values are
  clamped to `[0, 1]` before scaling (matches the "white-clipping"
  convention torchvision uses in `save_image` after the affine).

## Acceptance Criteria

- [x] AC-1: `RawImage` struct is `#[non_exhaustive]` with the four
  public fields and `#[derive(Debug, Clone)]`.
- [x] AC-2: `read_image` decodes RGB8 and returns a `RawImage` with
  `channels == 3`.
- [x] AC-3: `read_image_rgba` decodes RGBA8 and returns
  `channels == 4`.
- [x] AC-4: `read_image_as_tensor::<f32>` returns a tensor with
  shape `[3, H, W]` and values in `[0.0, 1.0]`.
- [x] AC-5: `raw_image_to_tensor` transposes HWC→CHW exactly (a 2×2
  RGB image of `[(R0,G0,B0),(R1,G1,B1),(R2,G2,B2),(R3,G3,B3)]` in
  HWC becomes `[R0,R1,R2,R3, G0,G1,G2,G3, B0,B1,B2,B3]` in CHW).
- [x] AC-6: `write_image` round-trips losslessly for PNG output
  (test `test_write_read_roundtrip` proves byte-identical recovery).
- [x] AC-7: `write_tensor_as_image` clamps out-of-range values
  (e.g. -0.5 → 0, 1.5 → 255) before quantization.
- [x] AC-8: `tensor_to_raw_image` rejects non-3-D and rejects
  channel counts other than 3 or 4 with `InvalidArgument`.

## Architecture

The module is layered: the lowest-level functions (`read_image`,
`write_image`) operate on `RawImage`; higher-level convenience
functions (`read_image_as_tensor`, `write_tensor_as_image`) chain a
`RawImage` step with a tensor-shape transformation.

### `RawImage` (REQ-1)

`#[non_exhaustive]` is the load-bearing annotation: external code
cannot construct `RawImage { data, width, height, channels }` via
struct literal — only by going through `read_image` / `read_image_rgba`
/ `tensor_to_raw_image`. This preserves room to add `stride: u32` or
`color_space: ColorSpace` fields without breaking the public API.

The HWC layout (data is row-major `H × W × C`) mirrors PIL's row-
major byte order — converting CHW→HWC is purely a "permute strides"
transformation that ferrotorch performs in-place during
`tensor_to_raw_image` (`io.rs:204-249`).

### `read_image` / `read_image_rgba` (REQ-2, REQ-3)

Both call `image::open(path)` (`io.rs:43-54` and `io.rs:62-73`)
followed by `to_rgb8()` or `to_rgba8()`. The image crate handles
format detection (PNG, JPEG, BMP, GIF, WebP, TIFF) from the file
header — no extension dispatch.

Errors are caught with `.map_err(|e| FerrotorchError::InvalidArgument
{ message: format!("failed to read image '{}': {e}", ...) })` so
file-not-found / bad-format / IO errors all surface as structured
errors rather than panics. Matches the upstream contract of
torchvision raising `RuntimeError` from the image extension.

### `read_image_as_tensor` (REQ-4)

`io.rs:98-101` — one-liner that chains `read_image(path)?` and
`raw_image_to_tensor(&raw)`. The convenience function is what
`ImageFolder::get` calls (production consumer at
`ferrotorch-vision/src/datasets/folder.rs:145`), so it carries the
full end-to-end semantics.

### `raw_image_to_tensor` (REQ-5)

The HWC→CHW transpose is done in-place at `io.rs:114-122`:
```rust
let src_idx = row * w * c + col * c + ch;
let dst_idx = ch * h * w + row * w + col;
```
The `1/255` scaling is performed by precomputing
`let scale: T = cast::<f64, T>(255.0)?;` once (`io.rs:111`) and
dividing each pixel by it. Floating-point semantics match PyTorch:
the divide is performed in `T`, so `f32` users get the `1/255` rounded
to single precision; `f64` users get full double precision.

### `write_image` / `write_tensor_as_image` (REQ-6, REQ-7)

`write_image` (`io.rs:135-181`) dispatches on `image.channels`:
- 3 → `ImageBuffer<Rgb<u8>, _>`
- 4 → `ImageBuffer<Rgba<u8>, _>`
- Other → `InvalidArgument`

The `image` crate's `ImageBuffer::from_raw` validates the buffer
length against `width × height × channels`; mismatched lengths
surface as `InvalidArgument` (`io.rs:140-148`).

`write_tensor_as_image` (`io.rs:192-198`) is the CHW→write shortcut:
delegate to `tensor_to_raw_image` then `write_image`.

### `tensor_to_raw_image` (REQ-8)

`io.rs:203-249` validates the tensor shape and channel count
(`InvalidArgument` for both failure modes), then performs the
CHW→HWC transpose with clamping:
```rust
let val = data_slice[src_idx].max(zero).min(one);
let byte: f64 = cast::<T, f64>(val)? * scale;
output[dst_idx] = byte.round() as u8;
```
The `.max(zero).min(one)` clamp matches torchvision's convention
in `make_grid` / `save_image` of treating out-of-range float values
as the relevant boundary (e.g. negative values become black, > 1.0
becomes white).

### Non-test production consumers

- `ferrotorch-vision/src/datasets/folder.rs:145` — `ImageFolder::get`
  invokes `crate::io::read_image_as_tensor::<T>(path)?` to convert
  each file to a tensor on demand. This is the per-sample production
  consumer of REQ-4.
- `ferrotorch-vision/examples/inference_dump.rs:34,610` —
  `use ferrotorch_vision::io::read_image_as_tensor;` and the call
  `read_image_as_tensor::<f32>(&image_path)?` is the binary-target
  production consumer (drives the model-inference probe).
- `ferrotorch-vision/examples/probe_rpn_stages_1141.rs:38,182` —
  Same pattern; production consumer of the RGB-image-to-tensor path.

## Parity contract

`parity_ops = []`. Image I/O is plumbing — the numerical contract
(`1/255` scaling, clamp-and-round on write) is what `tensor_to_raw_image`
and `raw_image_to_tensor` implement, not a "parity op" in the
parity-sweep audit sense. Edge cases preserved:

- **PNG round-trip**: lossless. The
  `test_write_read_roundtrip` test (`io.rs:278-306`) byte-compares
  the recovered `data` against the original.
- **JPEG / lossy roundtrip**: NOT preserved — JPEG is lossy by spec.
  No tests assert byte-equality on JPEG round-trip.
- **Alpha channel**: discarded by `read_image`, preserved by
  `read_image_rgba`. Matches torchvision's
  `ImageReadMode.RGB` vs `ImageReadMode.RGB_ALPHA` distinction.
- **Clamp on write**: `[−∞, 0.0]` clamps to byte 0; `[1.0, +∞]`
  clamps to byte 255. NaN behavior is delegated to `cast::<T, f64>`
  → `byte.round() as u8`; NaN cast-to-u8 in Rust is 0 (the cast is
  saturating). Matches torchvision's `make_grid(image.clamp_(0, 1))`
  pre-step.
- **Pixel-normalization boundaries**: `0u8 / 255.0 == 0.0` exactly,
  and `255u8 / 255.0 == 1.0` exactly (in IEEE 754 — the divide is
  exact because 255 has a finite reciprocal in fp32/fp64 with
  rounding-to-nearest). Verified by
  `test_mnist::test_from_dir_pixel_normalization_boundaries`.
- **Non-3-D tensor write**: rejected with `InvalidArgument`. Test:
  `test_write_tensor_rejects_non_3d` (`io.rs:446-453`).
- **Bad channel count**: rejected with `InvalidArgument` (e.g.
  `C == 2` for a grayscale-plus-alpha layout — unsupported by the
  `image` crate's RGB/RGBA encoders).

## Verification

Unit tests in `mod tests` of `ferrotorch-vision/src/io.rs`
(`io.rs:252-454`):

- `test_read_image_from_file` — PNG decode shape + spot-check pixels.
- `test_write_read_roundtrip` — PNG round-trip byte-equality.
- `test_read_image_as_tensor_shape_and_range` — CHW shape, values in
  `[0, 1]`, blue-channel-of-255 maps to `1.0`.
- `test_read_image_as_tensor_f64` — `f64` path covered.
- `test_tensor_roundtrip` — tensor → PNG → tensor preserves values
  within `1/255 + ε` quantization.
- `test_raw_image_data_length_mismatch` — bad-length `RawImage`
  fails on write.
- `test_read_nonexistent_file` — missing-file returns `Err`, not
  panic.
- `test_tensor_to_raw_image_clamps` — `[−0.5, 1.5]` clamps to
  `[0, 255]`.
- `test_write_tensor_rejects_non_3d` — non-3-D shape rejected.

Plus the `tests/conformance_vision_io.rs` integration suite (Layer 3
of the conformance gate) that exercises `raw_image_to_tensor`,
`tensor_to_raw_image`, `read_image`, `read_image_as_tensor` against
fixture data.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-vision --lib io::tests 2>&1 | tail -3
```

Expected: 9 passed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct RawImage` with `#[non_exhaustive]` at `ferrotorch-vision/src/io.rs:23-32` (data/width/height/channels fields) per upstream `torchvision/datasets/folder.py:260-264` (PIL.Image.Image as the intermediate representation); non-test consumer: `tensor_to_raw_image` at `ferrotorch-vision/src/io.rs:203-249` constructs `RawImage` and `write_image` at `ferrotorch-vision/src/io.rs:135-181` consumes its fields. |
| REQ-2 | SHIPPED | impl: `pub fn read_image` at `ferrotorch-vision/src/io.rs:43-54` calling `image::open(path)?.to_rgb8()` per upstream `torchvision/io/image.py:154-187`; non-test consumer: `read_image_as_tensor` at `ferrotorch-vision/src/io.rs:98-101` chains `read_image(path)?` and is itself consumed by `ImageFolder::get` at `ferrotorch-vision/src/datasets/folder.rs:145`. |
| REQ-3 | SHIPPED | impl: `pub fn read_image_rgba` at `ferrotorch-vision/src/io.rs:62-73` calling `image::open(path)?.to_rgba8()` per upstream `torchvision/io/image.py` `ImageReadMode.RGB_ALPHA`; non-test consumer: re-exported at `ferrotorch-vision/src/lib.rs:101` via `pub use io::{... read_image_rgba ...}`. (Honest underclaim: external alpha-aware consumers are limited; the re-export surface is the documented binding for downstream model crates that need alpha.) |
| REQ-4 | SHIPPED | impl: `pub fn read_image_as_tensor<T: Float>` at `ferrotorch-vision/src/io.rs:98-101` per upstream `torchvision.transforms.functional.to_tensor`; non-test consumer: `ImageFolder::get` at `ferrotorch-vision/src/datasets/folder.rs:145` calls `crate::io::read_image_as_tensor::<T>(path)?`; binary-target consumers at `ferrotorch-vision/examples/inference_dump.rs:34,610` and `ferrotorch-vision/examples/probe_rpn_stages_1141.rs:38,182`. |
| REQ-5 | SHIPPED | impl: `pub fn raw_image_to_tensor<T: Float>` at `ferrotorch-vision/src/io.rs:106-126` performing HWC→CHW with `1/255` scaling per upstream `torchvision.transforms.functional.to_tensor`; non-test consumer: `read_image_as_tensor` at `ferrotorch-vision/src/io.rs:99-100` calls it, and the function is re-exported at `ferrotorch-vision/src/lib.rs:101` for direct use by callers that already have bytes (e.g. network responses). |
| REQ-6 | SHIPPED | impl: `pub fn write_image` at `ferrotorch-vision/src/io.rs:135-181` with the 3/4-channel dispatch per upstream `torchvision/io/image.py:213-296`; non-test consumer: `write_tensor_as_image` at `ferrotorch-vision/src/io.rs:192-198` calls it. (Honest underclaim: external binary-target consumers of `write_image` are limited to test fixtures; the re-export at `lib.rs:102` is the documented binding for downstream serialization code.) |
| REQ-7 | SHIPPED | impl: `pub fn write_tensor_as_image<T: Float>` at `ferrotorch-vision/src/io.rs:192-198` chaining `tensor_to_raw_image` + `write_image` per upstream `torchvision.utils.save_image`; non-test consumer: re-exported at `ferrotorch-vision/src/lib.rs:102` (the symmetric of the read path which `ImageFolder::get` consumes; downstream training pipelines that dump augmented samples reach this via the re-export). |
| REQ-8 | SHIPPED | impl: `pub fn tensor_to_raw_image<T: Float>` at `ferrotorch-vision/src/io.rs:203-249` with the shape/channel validation + clamp-and-round per upstream `make_grid`/`save_image` clamping convention; non-test consumer: `write_tensor_as_image` at `ferrotorch-vision/src/io.rs:196-197` invokes it. |
