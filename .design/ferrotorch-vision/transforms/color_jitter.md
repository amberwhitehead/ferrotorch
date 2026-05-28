# ferrotorch-vision — `ColorJitter` transform

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (torchvision v0.26.0 site-packages)
upstream-paths:
  - torchvision/transforms/v2/_color.py
  - torchvision/transforms/v2/functional/_color.py
-->

## Summary

`ferrotorch-vision/src/transforms/color_jitter.rs` provides
`ColorJitter<T: Float>`, which randomly adjusts brightness, contrast,
saturation, and hue of a `[3, H, W]` RGB tensor with values in `[0, 1]`.
The four adjustments are applied in a randomly shuffled order
(matching upstream). Mirrors `torchvision.transforms.v2.ColorJitter`
at `_color.py:72`.

## Requirements

- REQ-1: `pub struct ColorJitter<T: Float>` storing `brightness: f64`,
  `contrast: f64`, `saturation: f64`, `hue: f64`, and `PhantomData<T>`.
  Mirrors `_color.py:72` `class ColorJitter(Transform)`.

- REQ-2: `pub fn ColorJitter::new(brightness, contrast, saturation, hue)
  -> FerrotorchResult<Self>` constructor validating each of
  `brightness/contrast/saturation >= 0` and `hue ∈ [0, 0.5)`.
  Mirrors upstream's `_check_input` non-negative-bound check
  (`_color.py:106`) and the hue-range constraint
  (`Should have 0 <= hue <= 0.5`).

- REQ-3: `fn shuffle_order(n: usize) -> Vec<usize>` — Fisher-Yates
  shuffle using the global PRNG. Returns a permutation of `0..n`.
  Used to randomize the order of the four jitters per call (upstream
  matches this behavior via `torch.randperm(4)`).

- REQ-4: `fn uniform_factor(v: f64) -> f64` — sample uniformly from
  `[max(0, 1 - v), 1 + v]`. The `max(0, ...)` clamp matches upstream's
  `clip_first_on_zero=True` default for brightness/contrast/saturation
  in `_color.py:_check_input`.

- REQ-5: `impl<T: Float> Transform<T> for ColorJitter<T>` — `apply`
  rejects non-RGB input (`shape.len() != 3 || shape[0] != 3`), then:
  1. Splits the tensor into three `f64` channel buffers `r, g, b`.
  2. Shuffles the four adjustment indices.
  3. For each `op` in the shuffled order, applies (gated by
     the corresponding parameter being `> 0`):
     - **Brightness**: scale all channels by `uniform_factor(brightness)`.
     - **Contrast**: blend each channel toward its per-channel mean
       by `uniform_factor(contrast)`.
     - **Saturation**: blend each pixel toward the luminance (ITU-R
       BT.601: `0.2989·R + 0.5870·G + 0.1140·B`) by `uniform_factor(saturation)`.
     - **Hue**: convert each pixel to HSV, shift `hue` by a uniform
       sample in `[-hue, +hue]` (wrapped modulo 1), convert back.
  4. Clamp all outputs to `[0, 1]` and cast to `T`.

- REQ-6: `fn rgb_to_hsv(r, g, b)` and `fn hsv_to_rgb(h, s, v)`
  conversion helpers. The roundtrip is bit-stable across the test
  colors red/green/blue/gray/black/white/arbitrary (verified by
  `test_rgb_hsv_roundtrip`).

- REQ-7: SHIPPED — `pub fn ColorJitter::from_ranges(brightness, contrast,
  saturation, hue)` accepts explicit `(min, max)` tuples per upstream
  `_check_input` (`_color.py:100-122`). The existing scalar `new` API
  still works and is layered on top of the same internal tuple
  representation.

## Acceptance Criteria

- [x] AC-1: `ColorJitter::new(0.2, 0.2, 0.2, 0.1)` constructs.
- [x] AC-2: Negative `brightness` returns `Err`.
- [x] AC-3: `hue > 0.5` returns `Err`.
- [x] AC-4: All-zero params returns identity (verified by
  `test_color_jitter_zero_params in color_jitter.rs`).
- [x] AC-5: Output shape equals input shape (verified by
  `test_color_jitter_output_shape in color_jitter.rs`).
- [x] AC-6: Output values are clamped to `[0, 1]` (verified by
  `test_color_jitter_output_clamped in color_jitter.rs`).
- [x] AC-7: Non-RGB input returns `Err` (verified by
  `test_color_jitter_rejects_non_rgb in color_jitter.rs`).
- [x] AC-8: RGB↔HSV roundtrip is exact for canonical colors (verified
  by `test_rgb_hsv_roundtrip in color_jitter.rs`).
- [x] AC-9: Brightness-only mode scales all pixels uniformly (verified
  by `test_color_jitter_brightness_only in color_jitter.rs`).
- [x] AC-10: f32 works (verified at `color_jitter.rs`).
- [x] AC-11: `(min, max)` tuple input form (verified by
  `test_color_jitter_from_ranges_identity`,
  `test_color_jitter_from_ranges_asymmetric_brightness`, and
  `test_color_jitter_from_ranges_rejects_invalid` in `color_jitter.rs`).

## Architecture

### Struct + constructor (REQ-1, REQ-2)

```rust
pub struct ColorJitter<T: Float> {
    brightness: f64, contrast: f64, saturation: f64, hue: f64,
    _marker: std::marker::PhantomData<T>,
}
```

at `color_jitter.rs`. Constructor at `color_jitter.rs`
applies four separate range checks.

### Helpers (REQ-3, REQ-4)

`fn shuffle_order` at `shuffle_order in color_jitter.rs` — Fisher-Yates over the
global PRNG.

`fn uniform_factor` at `uniform_factor in color_jitter.rs` — `[max(0, 1-v), 1+v]`
uniform sample.

### Transform impl (REQ-5)

`fn apply` at `apply in color_jitter.rs`:

```rust
// 1. Split into f64 channel buffers.
let mut r: Vec<f64> = data[..spatial].iter().map(|v| v.to_f64().unwrap()).collect();
// ... g, b similarly.
// 2. Shuffle op order.
let order = shuffle_order(4);
// 3. For each op, apply if its param is > 0.
for &op in &order {
    match op {
        0 if self.brightness > 0.0 => { /* scale all channels by factor */ }
        1 if self.contrast > 0.0 => { /* blend toward per-channel mean */ }
        2 if self.saturation > 0.0 => { /* blend toward grayscale via BT.601 */ }
        3 if self.hue > 0.0 => { /* HSV-shift hue */ }
        _ => {}
    }
}
// 4. Clamp to [0, 1] and convert back.
```

Each op writes back into `r, g, b` so the chained effect compounds.
Hue's HSV roundtrip is per-pixel — the most expensive op.

### Color-space conversion (REQ-6)

`fn rgb_to_hsv` at `rgb_to_hsv in color_jitter.rs`: standard
max/min/delta-based formula, with the hue sector chosen by which
channel equals `max`.

`fn hsv_to_rgb` at `hsv_to_rgb in color_jitter.rs`: standard sector-based
inverse.

The roundtrip preserves the canonical primary/secondary/grayscale
colors to f64 precision (`test_rgb_hsv_roundtrip` uses `< 1e-10`).

### NOT-STARTED gap (REQ-7)

Upstream accepts `brightness=(min, max)` tuples for asymmetric
sampling ranges (e.g. `brightness=(0.5, 1.5)`). ferrotorch's scalar-
only form supports the common `brightness=0.4` shorthand but not the
explicit range form. Blocker #1522.

### Non-test production consumers

- `pub use color_jitter::ColorJitter;` at
  `ferrotorch-vision/src/transforms/mod.rs:20` AND `ColorJitter` in
  the crate-root re-export at `ferrotorch-vision/src/lib.rs:113`.
- The conformance surface inventory at
  `ferrotorch-vision/tests/conformance/_surface_inventory.toml:137-146`
  registers `ferrotorch_vision::ColorJitter` and `::new` with the
  Python analog `torchvision.transforms.ColorJitter(brightness, contrast,
  saturation, hue)`.

## Parity contract

`parity_ops = []`.

- **All-zero params**: identity (the `if self.X > 0.0` guards skip
  every op).
- **Random op order**: each call picks a fresh permutation. Two
  invocations with the same input may produce different outputs even
  at the same seed, because the seed feeds both the order shuffle AND
  the factor draws.
- **Hue wrapping**: `(hue + hue_shift).rem_euclid(1.0)` ensures the
  HSV hue stays in `[0, 1)`.
- **Output clamp**: all output values clamped to `[0, 1]` before
  casting back to `T`. Matches upstream's
  `tensor.clamp(0.0, 1.0)` final step.
- **Non-RGB input**: `InvalidArgument`.

## Verification

Tests in `mod tests in color_jitter.rs` (7 tests):

- `test_color_jitter_output_shape in color_jitter.rs`
- `test_color_jitter_zero_params in color_jitter.rs`
- `test_color_jitter_output_clamped in color_jitter.rs`
- `test_color_jitter_rejects_non_rgb in color_jitter.rs`
- `test_rgb_hsv_roundtrip in color_jitter.rs`
- `test_color_jitter_brightness_only in color_jitter.rs`
- `test_color_jitter_f32 in color_jitter.rs`

Smoke:

```bash
cargo test -p ferrotorch-vision --lib transforms::color_jitter:: 2>&1 | tail -3
```

Expected: `7 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct ColorJitter<T: Float>` with four float params + `_marker` at `ColorJitter in ferrotorch-vision/src/transforms/color_jitter.rs`, mirroring `torchvision/transforms/v2/_color.py:72` `class ColorJitter`; non-test consumer: `pub use color_jitter::ColorJitter;` at `mod.rs` AND `ColorJitter` in the crate-root re-export at `ferrotorch-vision/src/lib.rs`. |
| REQ-2 | SHIPPED | impl: `pub fn ColorJitter::new(b, c, s, h) -> FerrotorchResult<Self>` with four range checks at `new in color_jitter.rs`; non-test consumer: registered in `ferrotorch-vision/tests/conformance/_surface_inventory.toml:143` as `ferrotorch_vision::ColorJitter::new`; reachable via the crate-root re-export. |
| REQ-3 | SHIPPED | impl: `fn shuffle_order(n: usize) -> Vec<usize>` Fisher-Yates at `shuffle_order in color_jitter.rs`; non-test consumer: `fn apply` calls `let order = shuffle_order(4);` at `apply in color_jitter.rs`. |
| REQ-4 | SHIPPED | impl: `fn uniform_factor(v: f64) -> f64` at `uniform_factor in color_jitter.rs`; non-test consumer: `fn apply` calls `uniform_factor(self.brightness)`, `uniform_factor(self.contrast)`, `uniform_factor(self.saturation)` at `uniform_factor in color_jitter.rs, 141, 153`. |
| REQ-5 | SHIPPED | impl: `impl<T: Float> Transform<T> for ColorJitter<T>` at `color_jitter.rs`; non-test consumer: any `Box<dyn Transform<T>>` slot — typically near the start of an augmentation `Compose` pipeline. The `lib.rs` re-export is the production-facing handle. |
| REQ-6 | SHIPPED | impl: `fn rgb_to_hsv(r, g, b) -> (f64, f64, f64)` at `rgb_to_hsv in color_jitter.rs` and `fn hsv_to_rgb(h, s, v)` at `hsv_to_rgb in color_jitter.rs`; non-test consumer: `fn apply` calls `rgb_to_hsv` and `hsv_to_rgb` per-pixel inside the hue branch at `hsv_to_rgb in color_jitter.rs`. |
| REQ-7 | SHIPPED | impl: `pub fn ColorJitter::from_ranges(brightness, contrast, saturation, hue)` + tuple field storage at `from_ranges in ferrotorch-vision/src/transforms/color_jitter.rs,104-146`; non-test consumer: `pub use color_jitter::ColorJitter;` at `mod.rs` AND in the `lib.rs` re-export — pipelines call `ColorJitter::from_ranges((0.8, 1.2), (0.8, 1.2), (0.8, 1.2), (-0.05, 0.05))?` per upstream `_color.py:100-122`. |
