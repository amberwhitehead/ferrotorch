# ferrotorch-vision — `CenterCrop` transform

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (torchvision v0.26.0 site-packages)
upstream-paths:
  - torchvision/transforms/v2/_geometry.py
  - torchvision/transforms/v2/functional/_geometry.py
-->

## Summary

`ferrotorch-vision/src/transforms/center_crop.rs` provides
`CenterCrop<T: Float>`, which extracts the center region of size
`(height, width)` from a `[C, H, W]` input. Mirrors
`torchvision.transforms.v2.CenterCrop` at
`_geometry.py:171`.

## Requirements

- REQ-1: `pub struct CenterCrop<T: Float>` storing `height: usize`,
  `width: usize`, and `PhantomData<T>`. Mirrors
  `_geometry.py:171` `class CenterCrop(Transform)`.

- REQ-2: `pub fn CenterCrop::new(height: usize, width: usize) -> Self`
  infallible constructor. Mirrors `CenterCrop(size=(h, w))` upstream.

- REQ-3: `impl<T: Float> Transform<T> for CenterCrop<T>` — `apply`
  rejects non-3-D input, returns `InvalidArgument` if the crop size
  exceeds either input dimension (no auto-padding), then computes
  the deterministic top-left corner `(top = (in_h - height)/2,
  left = (in_w - width)/2)` and copies the cropped region per channel
  using contiguous-row slicing. Mirrors
  `torchvision.transforms.v2.functional.center_crop`.

- REQ-4: SHIPPED — `CenterCrop::with_fill(f64)` enables auto-pad-with-fill
  semantics matching upstream `_geometry.py:180-181`. The fill value is
  user-selected; upstream's default is zero.

## Acceptance Criteria

- [x] AC-1: `CenterCrop::new(3, 3)` applied to `[3, 5, 5]` produces
  `[3, 3, 3]` (verified by `test_center_crop_output_shape` at
  `center_crop.rs:75`).
- [x] AC-2: Center 2x2 crop of a 4x4 grid `[0..16]` yields
  `[5, 6, 9, 10]` (verified by `test_center_crop_values` at
  `center_crop.rs:84`).
- [x] AC-3: Same-size crop preserves data (verified by
  `test_center_crop_exact_size` at `center_crop.rs:103`).
- [x] AC-4: Multichannel input crops each channel (verified by
  `test_center_crop_multichannel` at `center_crop.rs:114`).
- [x] AC-5: Crop larger than input returns `Err` (verified by
  `test_center_crop_too_large` at `center_crop.rs:131`).
- [x] AC-6: Non-3-D input returns `Err` (verified by
  `test_center_crop_rejects_non_3d` at `center_crop.rs:139`).
- [x] AC-7: pad-if-smaller behavior with user-selected fill (verified
  by `test_center_crop_with_fill_pads_small_input` and
  `test_center_crop_with_fill_no_op_when_input_large_enough` in
  `center_crop.rs`).

## Architecture

### Struct + constructor (REQ-1, REQ-2)

```rust
pub struct CenterCrop<T: Float> {
    height: usize,
    width: usize,
    _marker: std::marker::PhantomData<T>,
}
impl<T: Float> CenterCrop<T> {
    pub fn new(height: usize, width: usize) -> Self { ... }
}
```

at `center_crop.rs:9-24`.

### Transform impl (REQ-3)

`fn apply` at `center_crop.rs:26-69`:

1. 3-D check.
2. `self.height > in_h || self.width > in_w` → `InvalidArgument`.
3. `top = (in_h - height) / 2`, `left = (in_w - width) / 2` —
   integer division gives the canonical center offset (matches
   upstream `torch.div(in_h - h, 2, rounding_mode='floor')`).
4. Per channel, per row in `top..top + height`, copy the slice
   `data[row_start..row_start + width]` into the output via
   `extend_from_slice` — one bulk memmove per row.

### NOT-STARTED gap (REQ-4)

Upstream auto-pads when input is smaller than the crop. ferrotorch
currently rejects. The pad-with-zeros behavior is a small extension —
the same `RandomCrop` padding work in blocker #1513 would solve this
too. Blocker #1515 tracks the CenterCrop-specific copy-then-pad
helper.

### Non-test production consumers

- `pub use center_crop::CenterCrop;` at
  `ferrotorch-vision/src/transforms/mod.rs:19` AND `CenterCrop` in the
  crate-root re-export at `ferrotorch-vision/src/lib.rs:113`. The
  conformance surface inventory registers
  `ferrotorch_vision::CenterCrop` and `::new` as the public API
  (`tests/conformance/_surface_inventory.toml:95`).

## Parity contract

`parity_ops = []`.

- **Same-size**: identity (`top = 0`, `left = 0`).
- **Asymmetric extra space** (e.g. crop 2 from 5): `top = (5-2)/2 = 1`
  — biased toward the top-left by one pixel for odd remainders.
  Matches upstream's integer-div semantics.
- **Smaller-than-crop input**: `InvalidArgument` (R-DEFER gap vs
  upstream auto-pad).
- **Non-3-D input**: `InvalidArgument`.

## Verification

Tests in `mod tests in center_crop.rs` (6 tests):

- `test_center_crop_output_shape` at `center_crop.rs:75`
- `test_center_crop_values` at `center_crop.rs:84`
- `test_center_crop_exact_size` at `center_crop.rs:103`
- `test_center_crop_multichannel` at `center_crop.rs:114`
- `test_center_crop_too_large` at `center_crop.rs:131`
- `test_center_crop_rejects_non_3d` at `center_crop.rs:139`

Smoke:

```bash
cargo test -p ferrotorch-vision --lib transforms::center_crop:: 2>&1 | tail -3
```

Expected: `6 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct CenterCrop<T: Float>` with `height, width, _marker` at `ferrotorch-vision/src/transforms/center_crop.rs:9-13`, mirroring `torchvision/transforms/v2/_geometry.py:171` `class CenterCrop(Transform)`; non-test consumer: `pub use center_crop::CenterCrop;` at `ferrotorch-vision/src/transforms/mod.rs:19` AND `CenterCrop` in the crate-root re-export at `ferrotorch-vision/src/lib.rs:113`. |
| REQ-2 | SHIPPED | impl: `pub fn CenterCrop::new(height: usize, width: usize) -> Self` at `center_crop.rs:17-23`; non-test consumer: registered in the conformance surface inventory at `ferrotorch-vision/tests/conformance/_surface_inventory.toml:95` as `ferrotorch_vision::CenterCrop::new`; reachable via the crate-root re-export. |
| REQ-3 | SHIPPED | impl: `impl<T: Float> Transform<T> for CenterCrop<T>` with shape + bounds + center-offset + row-slice copy at `center_crop.rs:26-69`; non-test consumer: any `Box<dyn Transform<T>>` slot accepts this — `lib.rs:113` re-export is the production-facing handle. |
| REQ-4 | SHIPPED | impl: `CenterCrop::with_fill(f64)` builder + auto-pad-with-fill dispatch at `ferrotorch-vision/src/transforms/center_crop.rs:24-44,82-119`; non-test consumer: reachable via the `lib.rs:113` re-export — pipelines call `CenterCrop::new(h, w).with_fill(0.0)` for the upstream `_geometry.py:180-181` pad-with-zeros equivalent. |
