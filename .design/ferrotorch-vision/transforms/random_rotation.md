# ferrotorch-vision — `RandomRotation` transform

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (torchvision v0.26.0 site-packages)
upstream-paths:
  - torchvision/transforms/v2/_geometry.py
  - torchvision/transforms/v2/functional/_geometry.py
-->

## Summary

`ferrotorch-vision/src/transforms/random_rotation.rs` provides
`RandomRotation<T: Float>`, which rotates a `[C, H, W]` tensor about its
center by a random angle in `[-degrees, +degrees]` using bilinear
interpolation, with out-of-image samples filled with zero. Mirrors
`torchvision.transforms.v2.RandomRotation` at `_geometry.py:560`.

## Requirements

- REQ-1: `pub struct RandomRotation<T: Float>` storing `degrees: f64`
  and `PhantomData<T>`. Mirrors `_geometry.py:560` `class RandomRotation`.

- REQ-2: `pub fn RandomRotation::new(degrees: f64) -> FerrotorchResult<Self>`
  validating `degrees >= 0` (the symmetric range `[-d, +d]`
  interpretation makes negative `d` meaningless). Mirrors upstream's
  scalar-degrees → `[-d, +d]` interpretation (`_geometry.py:608-616`).

- REQ-3: `impl<T: Float> Transform<T> for RandomRotation<T>` — `apply`
  rejects non-3-D input. If `degrees == 0` the input is returned
  unchanged (no draw — saves randomness). Otherwise samples
  `angle_deg ∈ [-d, +d]` uniformly, computes
  `cos(angle_rad), sin(angle_rad)`, and for each output pixel
  `(ox, oy)` computes the inverse-rotated source coordinate
  `(sx, sy)` around the image center `(cx = (w-1)/2, cy = (h-1)/2)`,
  then bilinear-samples the source.

- REQ-4: `fn bilinear_sample<T: Float>(data, h, w, y, x) ->
  FerrotorchResult<T>` private helper — interpolates a single channel
  at fractional `(y, x)`, returning `T::zero()` for out-of-bounds
  coordinates. Each of the 4 corner samples uses an explicit
  in-bounds check; weights are `cast::<f64, T>` for numerical
  precision.

- REQ-5: NOT-STARTED — upstream `RandomRotation` has `interpolation`
  (NEAREST / BILINEAR), `expand` (output size grows to contain the
  rotated image), `center: Option<[float; 2]>` (override the rotation
  center), and `fill` (out-of-bounds fill value) parameters not
  implemented. ferrotorch's version is BILINEAR-only, no expand, no
  center override, fill=0. Blocker #1518.

## Acceptance Criteria

- [x] AC-1: `RandomRotation::new(30.0)` constructs successfully;
  `new(-1.0)` returns `Err`.
- [x] AC-2: Output shape equals input shape (verified by
  `test_random_rotation_output_shape` at `random_rotation.rs:141`).
- [x] AC-3: `degrees == 0` returns input unchanged (verified at
  `random_rotation.rs:151`).
- [x] AC-4: Center pixel approximately preserved across any rotation
  angle (verified by `test_random_rotation_preserves_center_pixel`
  at `random_rotation.rs:161`).
- [x] AC-5: Non-3-D input returns `Err` (verified at
  `random_rotation.rs:180`).
- [x] AC-6: `bilinear_sample` returns the exact pixel value at integer
  coordinates (verified at `random_rotation.rs:188`).
- [x] AC-7: `bilinear_sample` returns the average of all 4 corners at
  `(0.5, 0.5)` (verified at `random_rotation.rs:200`).
- [x] AC-8: `bilinear_sample` returns zero for out-of-bounds coords
  (verified at `random_rotation.rs:208`).
- [ ] AC-9: NOT-STARTED — interpolation/expand/center/fill. Blocker #1518.

## Architecture

### Struct + constructor (REQ-1, REQ-2)

```rust
pub struct RandomRotation<T: Float> {
    degrees: f64,
    _marker: std::marker::PhantomData<T>,
}
impl<T: Float> RandomRotation<T> {
    pub fn new(degrees: f64) -> FerrotorchResult<Self> {
        if degrees < 0.0 {
            return Err(FerrotorchError::InvalidArgument { ... });
        }
        Ok(Self { degrees, _marker: PhantomData })
    }
}
```

at `random_rotation.rs:14-39`.

### Bilinear sampler (REQ-4)

`fn bilinear_sample` at `random_rotation.rs:43-83`:

1. Negative-x/y short-circuit → zero (anti-borders).
2. Floor to `(x0, y0)`; bounds-check `x0 >= w || y0 >= h` → zero.
3. Read four corners `v00, v10, v01, v11` with per-corner in-bounds
   guards (`x1 < w`, `y1 < h`) — the corner `v11` requires both.
4. Compute bilinear weights `w00 = (1-dx)(1-dy)`, etc., as
   `cast::<f64, T>` to keep the weight math in f64 precision before
   converting to the element type.
5. Output `v00*w00 + v10*w10 + v01*w01 + v11*w11`.

### Transform impl (REQ-3)

`fn apply` at `random_rotation.rs:85-135`:

```rust
let angle_deg = self.degrees * (2.0 * random_f64() - 1.0);
let angle_rad = angle_deg.to_radians();
let cos_a = angle_rad.cos();
let sin_a = angle_rad.sin();
let cx = (w as f64 - 1.0) / 2.0;
let cy = (h as f64 - 1.0) / 2.0;
// For each output pixel, invert the rotation to find the source.
let dx = ox as f64 - cx;
let dy = oy as f64 - cy;
let sx = cos_a * dx + sin_a * dy + cx;
let sy = -sin_a * dx + cos_a * dy + cy;
output.push(bilinear_sample(ch_data, h, w, sy, sx)?);
```

The inverse-rotation formula `(sx, sy) = R(-θ)(ox-cx, oy-cy) + (cx, cy)`
is the standard image-space rotation. Per-channel processing means a
3-channel RGB rotation runs 3 separate inverse-rotation grids; this is
inefficient compared to a precomputed grid + 3 bilinear lookups, but
correct.

### NOT-STARTED gap (REQ-5)

Upstream `RandomRotation(degrees, interpolation, expand, center, fill)`
has four params we don't ship. The most-requested in pretraining
pipelines is `fill` (mode-specific border fill instead of zero); the
others are niche. Blocker #1518.

### Non-test production consumers

- `pub use random_rotation::RandomRotation;` at
  `ferrotorch-vision/src/transforms/mod.rs:29` AND `RandomRotation`
  in the crate-root re-export at `ferrotorch-vision/src/lib.rs:114`.

## Parity contract

`parity_ops = []`.

- **`degrees == 0`**: identity (input returned unchanged, no draw).
- **`degrees == 180`**: every call rotates by a value in `[-180, +180]` —
  the full circle. Center pixel mapping holds; corner pixels frequently
  read out-of-bounds → zero.
- **Out-of-bounds source samples**: zero-fill. Upstream's default
  `fill=0` matches this.
- **Non-3-D input**: `InvalidArgument`.
- **Bilinear interpolation contract**: matches upstream's
  `align_corners=False` semantics — `(sx, sy)` in image-pixel
  coordinates, four-tap interpolation with linear weights.

## Verification

Tests in `mod tests in random_rotation.rs` (6 tests):

- `test_random_rotation_output_shape` at `random_rotation.rs:141`
- `test_random_rotation_zero_degrees` at `random_rotation.rs:151`
- `test_random_rotation_preserves_center_pixel` at `random_rotation.rs:161`
- `test_random_rotation_rejects_non_3d` at `random_rotation.rs:180`
- `test_bilinear_sample_exact_pixel` at `random_rotation.rs:188`
- `test_bilinear_sample_midpoint` at `random_rotation.rs:200`
- `test_bilinear_sample_out_of_bounds` at `random_rotation.rs:208`

Smoke:

```bash
cargo test -p ferrotorch-vision --lib transforms::random_rotation:: 2>&1 | tail -3
```

Expected: `7 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct RandomRotation<T: Float>` with `degrees, _marker` at `ferrotorch-vision/src/transforms/random_rotation.rs:14-17`, mirroring `torchvision/transforms/v2/_geometry.py:560` `class RandomRotation`; non-test consumer: `pub use random_rotation::RandomRotation;` at `mod.rs:29` AND `RandomRotation` in the crate-root re-export at `ferrotorch-vision/src/lib.rs:114`. |
| REQ-2 | SHIPPED | impl: `pub fn RandomRotation::new(degrees: f64) -> FerrotorchResult<Self>` with `degrees >= 0` check at `random_rotation.rs:28-38`; non-test consumer: reachable via the crate-root re-export at `lib.rs:114`. |
| REQ-3 | SHIPPED | impl: `impl<T: Float> Transform<T> for RandomRotation<T>` with shape check, zero-shortcut, inverse-rotation per-pixel + bilinear sample at `random_rotation.rs:85-135`; non-test consumer: any `Box<dyn Transform<T>>` slot accepts this — composes into augmentation `Compose` pipelines. |
| REQ-4 | SHIPPED | impl: `fn bilinear_sample<T: Float>(data, h, w, y, x) -> FerrotorchResult<T>` at `random_rotation.rs:43-83`; non-test consumer: `fn apply` in this same file calls `bilinear_sample(ch_data, h, w, sy, sx)?` at `random_rotation.rs:127`. |
| REQ-5 | NOT-STARTED | open prereq blocker #1518 — `interpolation` (NEAREST/BILINEAR), `expand`, `center`, `fill` parameters from `torchvision/transforms/v2/_geometry.py:560-638` are not implemented. ferrotorch is BILINEAR-only, no expand, no center override, fill=0. |
