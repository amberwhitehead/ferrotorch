# ferrotorch-vision — `TrivialAugmentWide` transform

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (torchvision v0.26.0 site-packages)
upstream-paths:
  - torchvision/transforms/v2/_auto_augment.py
  - torchvision/transforms/v2/functional/_color.py
-->

## Summary

`ferrotorch-vision/src/transforms/trivial_augment_wide.rs` provides
`TrivialAugmentWide<T: Float>`, the Müller & Hutter 2021
"TrivialAugment Wide" tuning-free data-augmentation strategy: pick
exactly ONE op at random from a wide augmentation space and apply it
with a strength sampled uniformly from a fixed range. Mirrors
`torchvision.transforms.v2.TrivialAugmentWide` at
`_auto_augment.py:438-507`.

## Requirements

- REQ-1: `pub struct TrivialAugmentWide<T: Float>` storing
  `num_magnitude_bins: usize` (default 31) and `PhantomData<T>`.
  Mirrors `_auto_augment.py:438` `class TrivialAugmentWide` and its
  `num_magnitude_bins=31` default (`_auto_augment.py:480`).

- REQ-2: `pub fn TrivialAugmentWide::new(num_magnitude_bins: usize) ->
  FerrotorchResult<Self>` constructor validating `num_magnitude_bins > 0`.

- REQ-3: `impl<T: Float> Default for TrivialAugmentWide<T>` returning
  `Self::new(31)` — matches upstream default.

- REQ-4: `enum Op { Identity, Brightness, Contrast, Sharpness,
  Posterize, Solarize, AutoContrast, Equalize, HorizontalFlip,
  TranslateX, TranslateY }` private enum with associated constant
  `Op::ALL: &'static [Op]` — the op space. This is a subset of the
  upstream `_AUGMENTATION_SPACE` (which includes ShearX/Y and Rotate
  via PIL/affine ops we don't yet have). Documented R-DEFER gap.

- REQ-5: `fn apply_op<T: Float>(data, h, w, c, op, num_bins) ->
  FerrotorchResult<Vec<T>>` private dispatcher — handles each op
  variant with its canonical strength range. Strengths sampled by
  drawing `level = random_usize(num_bins)` and computing
  `level_f = level / (num_bins - 1)` (the floor saturation at
  `level == num_bins - 1` gives the strongest setting).

- REQ-6: `fn box_blur_3x3(data, h, w) -> Vec<f64>` — single-channel
  3x3 box blur with zero-padding boundary, used by the `Sharpness` op.

- REQ-7: `impl<T: Float> Transform<T> for TrivialAugmentWide<T>` —
  `apply` rejects non-3-D or zero-dim input, picks `op_idx =
  random_usize(Op::ALL.len())`, calls `apply_op(data, h, w, c, op,
  self.num_magnitude_bins)`, returns the result.

- REQ-8: SHIPPED — `Op::ShearX`, `Op::ShearY`, `Op::Rotate`, and
  `Op::Color` are added to `Op::ALL` (now 15 entries, one more than
  upstream's 14 because ferrotorch keeps `Identity` as an explicit op).
  Shear/Rotate use inverse-mapped bilinear sampling with zero fill;
  Color uses the BT.601-luma blend for 3-channel RGB and falls back to
  identity for non-RGB inputs to avoid corrupting non-image tensors.

## Acceptance Criteria

- [x] AC-1: `TrivialAugmentWide::new(31)` constructs.
- [x] AC-2: `new(0)` returns `Err`.
- [x] AC-3: `Default::default()` produces `num_magnitude_bins == 31`
  (verified by `test_trivial_augment_default_num_bins` at
  `trivial_augment_wide.rs:372`).
- [x] AC-4: Output shape equals input shape (verified by
  `test_trivial_augment_output_shape_preserved` at
  `trivial_augment_wide.rs:363`).
- [x] AC-5: Non-3-D input returns `Err` (verified at
  `trivial_augment_wide.rs:388`).
- [x] AC-6: `Op::Identity` returns input unchanged (verified at
  `trivial_augment_wide.rs:396`).
- [x] AC-7: `Op::HorizontalFlip` reverses columns (verified at
  `trivial_augment_wide.rs:403`).
- [x] AC-8: `Op::Posterize` keeps output in `[0, 1]` (verified at
  `trivial_augment_wide.rs:426`).
- [x] AC-9: `Op::Solarize` at threshold=0 inverts all pixels
  (verified at `trivial_augment_wide.rs:440`).
- [x] AC-10: `Op::AutoContrast` stretches `[0.3, 0.5, 0.7]` to
  `[0, 0.5, 1]` (verified at `trivial_augment_wide.rs:457`).
- [x] AC-11: `Op::AutoContrast` no-op on a constant channel (verified
  at `trivial_augment_wide.rs:467`).
- [x] AC-12: `Op::Equalize` keeps output in `[0, 1]` (verified at
  `trivial_augment_wide.rs:478`).
- [x] AC-13: `box_blur_3x3` is a no-op on uniform interior pixels
  (verified at `trivial_augment_wide.rs:489`).
- [x] AC-14: ShearX/ShearY/Rotate/Color ops (verified by
  `test_op_all_includes_new_geometric_ops`,
  `test_op_shear_x_uniform_image_stays_uniform`,
  `test_op_rotate_uniform_image_stays_uniform_in_interior`,
  `test_op_color_uniform_image_stays_uniform`, and
  `test_op_color_non_rgb_is_identity` in `trivial_augment_wide.rs`).

## Architecture

### Struct + constructors (REQ-1, REQ-2, REQ-3)

```rust
pub struct TrivialAugmentWide<T: Float> {
    num_magnitude_bins: usize,
    _marker: std::marker::PhantomData<T>,
}
impl<T: Float> Default for TrivialAugmentWide<T> {
    fn default() -> Self {
        Self::new(31).expect("invariant: default num_magnitude_bins=31 is > 0")
    }
}
impl<T: Float> TrivialAugmentWide<T> {
    pub fn new(num_magnitude_bins: usize) -> FerrotorchResult<Self> {
        if num_magnitude_bins == 0 { return Err(...); }
        Ok(Self { num_magnitude_bins, _marker: PhantomData })
    }
}
```

at `trivial_augment_wide.rs:31-65`. The `Default` impl is at lines
38-44; `new` is at lines 54-64.

### Op enum (REQ-4)

```rust
enum Op {
    Identity, Brightness, Contrast, Sharpness,
    Posterize, Solarize, AutoContrast, Equalize,
    HorizontalFlip, TranslateX, TranslateY,
}
impl Op {
    const ALL: &'static [Op] = &[ Op::Identity, ..., Op::TranslateY ];
}
```

at `trivial_augment_wide.rs:74-102`. 11 ops total; the upstream space
is 14 (adds ShearX, ShearY, Rotate, Color). The omitted ops require
affine-transform infrastructure that we don't yet have.

### Op dispatcher (REQ-5)

`fn apply_op<T: Float>(data, h, w, c, op, num_bins)` at
`trivial_augment_wide.rs:107-304`. Each match arm maps a sampled
`level_f` into the op's canonical strength range, then applies the op.

Canonical ranges (from upstream `_AUGMENTATION_SPACE`):
- **Brightness/Contrast/Sharpness**: `factor = 0.01 + 1.98 * level_f`
  (range `[0.01, 1.99]`, value 1 = identity).
- **Posterize**: `bits = 2 + round(6 * level_f)` (range `[2, 8]`).
- **Solarize**: `threshold = level_f` (range `[0, 1]`).
- **AutoContrast / Equalize / HorizontalFlip / Identity**: no strength.
- **TranslateX/Y**: shift in pixels, sampled from `[-0.32W, +0.32W]`
  or `[-0.32H, +0.32H]`.

The math per op is straightforward; the most non-trivial is
`Op::Equalize` which builds per-channel histograms over 256 bins,
constructs the CDF, and replaces each pixel with its CDF value.

### Box blur (REQ-6)

`fn box_blur_3x3(data, h, w) -> Vec<f64>` at
`trivial_augment_wide.rs:307-327` — 3x3 mean filter with explicit
in-bounds guards. Used by `Op::Sharpness` to compute the blurred
reference: `output = blur + factor * (orig - blur)`.

### Transform impl (REQ-7)

`fn apply` at `trivial_augment_wide.rs:329-356`:

1. 3-D shape check.
2. Zero-dim guard.
3. `op_idx = random_usize(Op::ALL.len())`.
4. `apply_op(data, h, w, c, op, self.num_magnitude_bins)`.

### NOT-STARTED gap (REQ-8)

Upstream's `_AUGMENTATION_SPACE` includes:
- `ShearX`, `ShearY`: affine shear, needs grid_sample-equivalent.
- `Rotate`: needs the same affine machinery as `RandomRotation`.
- `Color`: saturation-style adjustment from `_color.py:adjust_saturation`.

Blocker #1523 tracks the missing ops.

### Non-test production consumers

- `pub use trivial_augment_wide::TrivialAugmentWide;` at
  `ferrotorch-vision/src/transforms/mod.rs:34`.
- (Note: `TrivialAugmentWide` is NOT re-exported at the crate root in
  `lib.rs:113-115`. Callers reach it via
  `ferrotorch_vision::transforms::TrivialAugmentWide`. Logged
  cleanup.)

## Parity contract

`parity_ops = []`. Per-op contracts:

- **`Op::Identity`**: returns input unchanged.
- **`Op::Brightness` factor=1**: identity (`v * 1 = v`).
- **`Op::Contrast` factor=1**: identity.
- **`Op::AutoContrast` on constant channel**: no-op (avoids
  divide-by-zero).
- **`Op::Equalize`**: produces CDF values in `[0, 1]` regardless of
  input range (clamped during histogram binning).
- **`Op::HorizontalFlip`**: row-by-row column reverse.
- **`Op::TranslateX/Y` shift=0**: identity.

## Verification

Tests in `mod tests in trivial_augment_wide.rs` (13 tests):

- `test_trivial_augment_output_shape_preserved` at `trivial_augment_wide.rs:363`
- `test_trivial_augment_default_num_bins` at `trivial_augment_wide.rs:372`
- `test_trivial_augment_zero_bins_errors` at `trivial_augment_wide.rs:378`
- `test_trivial_augment_rejects_non_3d` at `trivial_augment_wide.rs:388`
- `test_op_identity_returns_input_unchanged` at `trivial_augment_wide.rs:396`
- `test_op_horizontal_flip_reverses_columns` at `trivial_augment_wide.rs:403`
- `test_op_brightness_scales_pixels` at `trivial_augment_wide.rs:411`
- `test_op_posterize_preserves_length` at `trivial_augment_wide.rs:426`
- `test_op_solarize_at_threshold_zero_inverts_all` at `trivial_augment_wide.rs:440`
- `test_op_auto_contrast_stretches_range` at `trivial_augment_wide.rs:457`
- `test_op_auto_contrast_constant_channel_is_unchanged` at `trivial_augment_wide.rs:467`
- `test_op_equalize_cdf_is_monotonic` at `trivial_augment_wide.rs:478`
- `test_box_blur_uniform_is_unchanged_interior` at `trivial_augment_wide.rs:489`

Smoke:

```bash
cargo test -p ferrotorch-vision --lib transforms::trivial_augment_wide:: 2>&1 | tail -3
```

Expected: `13 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct TrivialAugmentWide<T: Float>` with `num_magnitude_bins` + `_marker` at `ferrotorch-vision/src/transforms/trivial_augment_wide.rs:31-36`, mirroring `torchvision/transforms/v2/_auto_augment.py:438` `class TrivialAugmentWide`; non-test consumer: `pub use trivial_augment_wide::TrivialAugmentWide;` at `mod.rs:34` exposes it through the public transforms namespace. |
| REQ-2 | SHIPPED | impl: `pub fn TrivialAugmentWide::new(num_magnitude_bins: usize) -> FerrotorchResult<Self>` at `trivial_augment_wide.rs:54-64`; non-test consumer: reachable via `mod.rs:34` re-export. |
| REQ-3 | SHIPPED | impl: `impl Default for TrivialAugmentWide<T>` returning `Self::new(31)` at `trivial_augment_wide.rs:38-44`; non-test consumer: reachable via the `pub use` re-export, called from `TrivialAugmentWide::default()` in config-driven augmentation pipelines. |
| REQ-4 | SHIPPED | impl: `enum Op { Identity, ..., TranslateY }` at `trivial_augment_wide.rs:74-86` and `Op::ALL: &'static [Op]` at `trivial_augment_wide.rs:88-102`; non-test consumer: `fn apply_op` matches every `Op` variant and `fn apply` calls `Op::ALL[random_usize(Op::ALL.len())]` at `trivial_augment_wide.rs:349-350`. |
| REQ-5 | SHIPPED | impl: `fn apply_op<T: Float>(data, h, w, c, op, num_bins) -> FerrotorchResult<Vec<T>>` at `trivial_augment_wide.rs:107-304`; non-test consumer: `fn apply` in this same file calls `apply_op(data, h, w, c, op, self.num_magnitude_bins)?` at `trivial_augment_wide.rs:353`. |
| REQ-6 | SHIPPED | impl: `fn box_blur_3x3(data, h, w) -> Vec<f64>` at `trivial_augment_wide.rs:307-327`; non-test consumer: `fn apply_op` calls `box_blur_3x3(&ch_slice, h, w)` inside the `Op::Sharpness` arm at `trivial_augment_wide.rs:159`. |
| REQ-7 | SHIPPED | impl: `impl<T: Float> Transform<T> for TrivialAugmentWide<T>` at `trivial_augment_wide.rs:329-356`; non-test consumer: any `Box<dyn Transform<T>>` slot — composes into augmentation `Compose` pipelines via the `mod.rs:34` re-export. |
| REQ-8 | SHIPPED | impl: `Op::ShearX`, `Op::ShearY`, `Op::Rotate`, `Op::Color` variants + `apply_op` dispatch arms + `shear_apply` / `rotate_apply` / `bilinear_sample_or_fill` helpers at `ferrotorch-vision/src/transforms/trivial_augment_wide.rs:88-128,318-386,419-509`; non-test consumer: `pub use trivial_augment_wide::TrivialAugmentWide;` at `mod.rs:42` — the impl picks an op via `Op::ALL[random_usize(Op::ALL.len())]`, so every new variant is reachable through the public API. |
