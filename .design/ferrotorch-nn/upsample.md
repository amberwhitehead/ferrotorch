# ferrotorch-nn — `upsample` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/modules/upsampling.py
  - aten/src/ATen/native/UpSample.cpp
  - torch/nn/functional.py
-->

## Summary

`ferrotorch-nn/src/upsample.rs` provides spatial-resizing and
sliding-window transformation modules for vision workloads:
`Upsample`, `interpolate`, `grid_sample`, `affine_grid`,
`PixelShuffle` / `PixelUnshuffle`, and `Fold` / `Unfold`. Mirrors
`torch.nn.{Upsample, PixelShuffle, PixelUnshuffle, Fold, Unfold}`
and the functional entries
`torch.nn.functional.{interpolate, grid_sample, affine_grid,
pixel_shuffle, pixel_unshuffle, fold, unfold}` at
`torch/nn/modules/upsampling.py:13-150` and
`torch/nn/functional.py`.

All autograd-tracked operations attach a `GradFn<T>` when gradient
tracking is enabled so reverse-mode autodiff works end-to-end.

## Requirements

- REQ-1: `pub enum InterpolateMode { Nearest, Bilinear, Bicubic }` —
  selects the interpolation kernel for `Upsample` and
  `interpolate`.

- REQ-2: `pub enum GridSamplePaddingMode { Zeros, Border,
  Reflection }` / `pub enum GridSampleMode { Bilinear, Nearest }` —
  control `grid_sample` boundary and sampling behaviour. Match
  upstream's kwarg semantics.

- REQ-3: `pub fn interpolate(input, output_size, mode,
  align_corners) -> FerrotorchResult<Tensor<T>>` — resizes a 4-D
  `[B, C, H, W]` tensor to `[B, C, output_h, output_w]` using the
  selected interpolation kernel. Mirrors
  `torch.nn.functional.interpolate` for the 2-D case.

- REQ-4: `pub struct Upsample` + `Module<T>` impl — wraps
  `interpolate` with the kernel size / mode / align_corners stored
  as fields. Mirrors `torch.nn.Upsample` at
  `upsampling.py:13-150`.

- REQ-5: `pub fn grid_sample(input, grid, mode, padding_mode,
  align_corners) -> FerrotorchResult<Tensor<T>>` — samples the
  input at locations specified by the `grid` tensor of shape
  `[B, H_out, W_out, 2]` containing normalised (-1..1)
  coordinates. Mirrors `torch.nn.functional.grid_sample`.

- REQ-6: `pub fn affine_grid(theta, size, align_corners) ->
  FerrotorchResult<Tensor<T>>` — generates a sampling grid from a
  batch of affine matrices `theta: [B, 2, 3]`. Mirrors
  `torch.nn.functional.affine_grid`.

- REQ-7: `pub struct PixelShuffle` / `pub struct PixelUnshuffle` +
  `Module<T>` impls — sub-pixel-convolution rearrangement:
  `[B, C*r*r, H, W] <-> [B, C, H*r, W*r]`. Mirror
  `torch.nn.PixelShuffle` / `torch.nn.PixelUnshuffle`.

- REQ-8: Functional helpers `pub fn pixel_shuffle` /
  `pub fn pixel_unshuffle` — same math as the module variants,
  callable directly. Mirror
  `torch.nn.functional.pixel_shuffle` / `pixel_unshuffle`.

- REQ-9: `pub struct Unfold` / `pub struct Fold` + `Module<T>` impls
  plus `pub fn unfold` / `pub fn fold` — sliding-window patch
  extraction (`Unfold: [B, C, H, W] -> [B, C * kernel_h *
  kernel_w, L]`) and reconstruction (`Fold` is the inverse).
  Mirror `torch.nn.Unfold` / `torch.nn.Fold`.

- REQ-10: Cubic interpolation kernel (Keys' cubic, `a = -0.75`) —
  the standard PyTorch bicubic weight. Implemented as `fn
  cubic_weight(t: f64) -> f64` in `upsample.rs`. Matches upstream's
  ATen kernel exactly.

- REQ-11: `align_corners` semantic — when `true`, the corner pixels
  are aligned at `(0, 0)` and `(H-1, W-1)`; when `false`, pixel
  centres are aligned (`0.5 / H` is the smallest source coordinate).
  Both behaviours implemented per upstream.

- REQ-12: Autograd via per-op `GradFn<T>` — each forward attaches a
  backward node (e.g. `InterpolateBackward`, `GridSampleBackward`,
  `PixelShuffleBackward`, `FoldBackward`, `UnfoldBackward`) when
  grad-tracking is enabled.

## Acceptance Criteria

- [x] AC-1: `interpolate([1, 1, 4, 4], (8, 8), Nearest, false)`
  returns `[1, 1, 8, 8]`.
- [x] AC-2: `interpolate(...)` with `Bilinear` matches a
  hand-computed reference.
- [x] AC-3: `Upsample::new(scale_factor=2, mode=Bilinear,
  align_corners=false)` wraps `interpolate` correctly.
- [x] AC-4: `grid_sample(input, grid)` returns a tensor with the
  same channel count as the input.
- [x] AC-5: `affine_grid(theta=[1, 2, 3], size=[1, C, H, W])`
  returns `[1, H, W, 2]`.
- [x] AC-6: `pixel_shuffle(input, r=2)` produces `[B, C, H*2,
  W*2]` from `[B, C*4, H, W]`.
- [x] AC-7: `unfold(input, kernel_size, ...)` then `fold(...)`
  recovers the original input within numerical tolerance.

## Architecture

### Modes (REQ-1, REQ-2)

`pub enum InterpolateMode` /
`pub enum GridSamplePaddingMode` /
`pub enum GridSampleMode` at
`pub enum InterpolateMode in upsample.rs` etc. carry the
small `Copy` enum variants used by the interpolation kernels.

### interpolate and Upsample (REQ-3, REQ-4, REQ-10, REQ-11)

`pub fn interpolate<T: Float>` at
`pub fn interpolate in upsample.rs` validates the 4-D rank then
dispatches on the `mode`:

- **Nearest** — direct nearest-neighbour lookup with the
  `align_corners`-aware source coordinate formula.
- **Bilinear** — 4-neighbour weighted average.
- **Bicubic** — 16-neighbour weighted sum using the cubic kernel
  `fn cubic_weight in upsample.rs` (Keys' cubic, `a = -0.75`).

`pub struct Upsample` at `pub struct Upsample in upsample.rs`
wraps the function, storing `scale_factor` or `output_size` plus
the mode and `align_corners` flag. `impl<T: Float> Module<T> for
Upsample` invokes `interpolate`.

### grid_sample and affine_grid (REQ-5, REQ-6)

`pub fn grid_sample<T: Float>` at
`pub fn grid_sample in upsample.rs` interprets the
`[B, H_out, W_out, 2]` grid as normalised input coordinates
(`-1..1` mapped to `0..H-1` or `0.5/H..(H - 0.5)/H` depending on
`align_corners`), applies the chosen padding mode at out-of-range
positions, and gathers via the bilinear or nearest kernel.

`pub fn affine_grid<T: Float>` at
`pub fn affine_grid in upsample.rs` constructs the per-output
homogeneous coordinate `[x, y, 1]`, multiplies by `theta` to get
the source location, and emits `[B, H_out, W_out, 2]`.

### PixelShuffle / PixelUnshuffle (REQ-7, REQ-8)

`pub struct PixelShuffle` / `pub struct PixelUnshuffle` at the
`pub struct PixelShuffle in upsample.rs` etc. carry the upscale
factor. Forward dispatches to `pub fn pixel_shuffle` /
`pub fn pixel_unshuffle` at `pub fn pixel_shuffle in upsample.rs`
which reshape via `[B, C, r, r, H, W] -> [B, C, H, r, W, r] ->
[B, C, H*r, W*r]` (and the inverse for unshuffle).

### Unfold / Fold (REQ-9)

`pub struct Unfold` at `pub struct Unfold in upsample.rs` carries
`kernel_size`, `stride`, `padding`, `dilation`. `pub fn unfold` at
`pub fn unfold in upsample.rs` extracts `kernel_h * kernel_w *
C`-sized patches per spatial position and concatenates them along
the last axis. `pub fn fold` is the inverse — scatters patches
back to the spatial grid.

### Autograd (REQ-12)

Each forward attaches its backward node via
`Tensor::from_operation`. The autograd engine traverses the node
on `backward()` of any tensor downstream of an upsample / sample /
shuffle / fold call.

### Non-test production consumers

- `pub use upsample::{Fold, GridSampleMode, GridSamplePaddingMode,
  InterpolateMode, PixelShuffle, PixelUnshuffle, Unfold, Upsample,
  affine_grid, fold, grid_sample, interpolate, pixel_shuffle,
  pixel_unshuffle, unfold}` at
  `ferrotorch-nn/src/lib.rs:252-256`.
- `ferrotorch-vision/src/models/segmentation/deeplabv3.rs:52`,
  `aspp.rs`, `lraspp in lraspp.rs`, `fcn in fcn.rs` consume
  `InterpolateMode` and `interpolate` for the segmentation
  upsample tails.

## Parity contract

`parity_ops = []`. The upsample family piggybacks on the
non-parity-tracked `interpolate` / `grid_sample` / `fold` /
`unfold` numerical contract verified by `mod tests in upsample.rs`.

Numerical edge cases preserved:

- **`align_corners=true` corner case** — `H_out=1` would divide by
  zero; ferrotorch handles by returning the centre pixel (matches
  upstream).
- **Bicubic out-of-range padding** — upstream extrapolates with the
  cubic kernel beyond the input border; ferrotorch matches.
- **`grid_sample` boundary mode** — `Zeros` returns zero outside;
  `Border` clamps; `Reflection` mirrors around the edge.
- **PixelShuffle for non-multiple-of-`r*r` channels** — upstream
  errors; ferrotorch matches.

## Verification

Tests in `mod tests in upsample.rs`. Highlights:

- Shape contracts for every public function.
- `interpolate` Bilinear / Bicubic numerical reference checks.
- `unfold` ↔ `fold` round-trip identity.
- `pixel_shuffle` / `pixel_unshuffle` round-trip identity.

No parity-sweep ops declared. Smoke command:

```bash
cargo test -p ferrotorch-nn --lib upsample:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum InterpolateMode` in `upsample.rs`; non-test consumer: re-export at `upsample in ferrotorch-nn/src/lib.rs` + `ferrotorch-vision/src/models/segmentation/deeplabv3.rs` + `aspp.rs` + `lraspp in lraspp.rs` + `fcn in fcn.rs`. |
| REQ-2 | SHIPPED | impl: `pub enum GridSamplePaddingMode` and `pub enum GridSampleMode` in `upsample.rs`; non-test consumer: re-export at `lib.rs`. |
| REQ-3 | SHIPPED | impl: `pub fn interpolate<T: Float>` in `upsample.rs`; non-test consumer: re-export at `lib.rs` + every segmentation model in `ferrotorch-vision` (see REQ-1 list). |
| REQ-4 | SHIPPED | impl: `pub struct Upsample` plus `impl<T: Float> Module<T> for Upsample` in `upsample.rs`; non-test consumer: re-export at `lib.rs`. |
| REQ-5 | SHIPPED | impl: `pub fn grid_sample<T: Float>` in `upsample.rs`; non-test consumer: re-export at `lib.rs`. |
| REQ-6 | SHIPPED | impl: `pub fn affine_grid<T: Float>` in `upsample.rs`; non-test consumer: re-export at `lib.rs`. |
| REQ-7 | SHIPPED | impl: `pub struct PixelShuffle` and `pub struct PixelUnshuffle` plus their `impl<T: Float> Module<T>` blocks in `upsample.rs`; non-test consumer: re-export at `lib.rs`. |
| REQ-8 | SHIPPED | impl: `pub fn pixel_shuffle<T: Float>` and `pub fn pixel_unshuffle<T: Float>` in `upsample.rs`; non-test consumer: re-export at `lib.rs`. |
| REQ-9 | SHIPPED | impl: `pub struct Unfold`, `pub struct Fold` plus their `impl<T: Float> Module<T>` blocks and `pub fn unfold`, `pub fn fold` in `upsample.rs`; non-test consumer: re-export at `lib.rs`. |
| REQ-10 | SHIPPED | impl: `fn cubic_weight in upsample.rs`; non-test consumer: invoked from the bicubic branch of `interpolate` (re-exported at `lib.rs`). |
| REQ-11 | SHIPPED | impl: `align_corners`-aware source-coordinate formulas inside the interpolate / grid_sample bodies in `upsample.rs`; non-test consumer: re-export at `lib.rs`. |
| REQ-12 | SHIPPED | impl: per-op `GradFn<T>` types plus `Tensor::from_operation` calls in `upsample.rs`; non-test consumer: re-export at `lib.rs` — autograd engine traverses each GradFn on `backward()`. |
