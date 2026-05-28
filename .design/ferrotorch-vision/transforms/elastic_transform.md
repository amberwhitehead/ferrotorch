# ferrotorch-vision — `ElasticTransform` transform

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (torchvision v0.26.0 site-packages)
upstream-paths:
  - torchvision/transforms/v2/_geometry.py
  - torchvision/transforms/v2/functional/_geometry.py
-->

## Summary

`ferrotorch-vision/src/transforms/elastic_transform.rs` provides
`ElasticTransform<T: Float>`, which applies elastic deformation
(Simard et al. 2003) via a Gaussian-smoothed random displacement field
and bilinear resampling. Mirrors
`torchvision.transforms.v2.ElasticTransform` at `_geometry.py:999-1090`.

## Requirements

- REQ-1: `pub struct ElasticTransform<T: Float>` storing `alpha: f64`
  (displacement scale), `sigma: f64` (smoother std), and
  `PhantomData<T>`. Mirrors `_geometry.py:999` `class ElasticTransform`.

- REQ-2: `pub fn ElasticTransform::new(alpha: f64, sigma: f64) ->
  FerrotorchResult<Self>` constructor validating `alpha >= 0` and
  `sigma > 0`. Mirrors upstream's implicit non-negativity.

- REQ-3: `fn gaussian_kernel_1d(size, sigma)` and
  `fn gaussian_filter_2d(data, h, w, sigma)` — local helpers for the
  separable Gaussian-smoothing pass on the displacement fields.
  `gaussian_filter_2d` computes a `radius = ceil(3 * sigma)` kernel,
  then applies horizontal-then-vertical 1-D convolutions with
  zero-padding.

- REQ-4: `fn bilinear_sample(data, h, w, y, x) -> f64` — clamped-to-
  edge bilinear sampler for a single-channel `[H, W]` buffer.
  Out-of-bounds `(y, x)` are clamped to `[0, h-1] × [0, w-1]` — the
  upstream default ("border" mode in `torch.nn.functional.grid_sample`).
  Note this clamping behavior DIFFERS from
  `random_rotation.rs::bilinear_sample` which returns zero for OOB.

- REQ-5: `impl<T: Float> Transform<T> for ElasticTransform<T>` —
  `apply` rejects non-3-D input and zero-dim inputs. If `alpha == 0`
  returns identity (no displacement). Otherwise:
  1. Sample `dy_field`, `dx_field` per pixel from `Uniform[-1, 1]`.
  2. Smooth each field with `gaussian_filter_2d(_, _, _, sigma)`.
  3. Scale by `alpha`.
  4. Per channel, per output pixel `(row, col)`, sample from
     `(row + dy_field[row, col], col + dx_field[row, col])` via
     `bilinear_sample`.

- REQ-6: SHIPPED — `with_interpolation(InterpolationMode)` adds a
  nearest-neighbor sampler alongside the existing bilinear path;
  `with_fill(f64)` swaps clamp-to-edge for constant-fill on out-of-bounds
  samples; `new_range((alpha_lo, alpha_hi), (sigma_lo, sigma_hi))` adds
  tuple-form sampling per call (the scalar `new(alpha, sigma)` still
  works, mapped to the degenerate `(α, α)` / `(σ, σ)` range).

## Acceptance Criteria

- [x] AC-1: `ElasticTransform::new(5.0, 1.5)` constructs.
- [x] AC-2: `new(-1.0, 1.0)` returns `Err` (verified at
  `elastic_transform.rs`).
- [x] AC-3: `new(1.0, 0.0)` returns `Err` (verified at
  `elastic_transform.rs`).
- [x] AC-4: Output shape equals input shape (verified by
  `test_elastic_output_shape_preserved in elastic_transform.rs`).
- [x] AC-5: `alpha == 0` is identity (verified at `alpha in elastic_transform.rs`).
- [x] AC-6: Uniform-value image is uniform-preserving (verified by
  `test_elastic_constant_image_unchanged_interior` at
  `test_elastic_constant_image_unchanged_interior in elastic_transform.rs`).
- [x] AC-7: Non-3-D input returns `Err` (verified at
  `elastic_transform.rs`).
- [x] AC-8: Zero-dim input returns `Err` (verified at
  `elastic_transform.rs`).
- [x] AC-9: `bilinear_sample` corners are exact (verified at
  `elastic_transform.rs`).
- [x] AC-10: `bilinear_sample` midpoint averages 4 corners
  (verified at `elastic_transform.rs`).
- [x] AC-11: `bilinear_sample` out-of-bounds clamps to nearest corner
  (verified at `elastic_transform.rs`).
- [x] AC-12: interpolation/fill/tuple `alpha`/`sigma` params (verified
  by `test_elastic_new_range_samples_within_band`,
  `test_elastic_with_nearest_yields_only_input_values`,
  `test_elastic_with_fill_replaces_oob_samples`, and
  `test_elastic_new_range_validates_alpha` in `elastic_transform.rs`).
  Blocker #1521.

## Architecture

### Struct + constructor (REQ-1, REQ-2)

```rust
pub struct ElasticTransform<T: Float> {
    alpha: f64,
    sigma: f64,
    _marker: std::marker::PhantomData<T>,
}
```

at `elastic_transform.rs`. Constructor at
`elastic_transform.rs` validates both bounds.

### Gaussian helpers (REQ-3)

`fn gaussian_kernel_1d` at `gaussian_kernel_1d in elastic_transform.rs` is identical
to `random_gaussian_blur::gaussian_kernel_1d`; deliberate code
duplication to keep the file self-contained.

`fn gaussian_filter_2d(data, h, w, sigma)` at
`gaussian_filter_2d in elastic_transform.rs` computes `radius = ceil(3 * sigma)` to
cover ~99.7% of the Gaussian mass, then applies separable horizontal-
then-vertical 1-D convolutions with zero-padding.

### Bilinear sampler with clamp (REQ-4)

```rust
fn bilinear_sample(data: &[f64], h, w, y, x) -> f64 {
    let y = y.clamp(0.0, (h - 1) as f64);
    let x = x.clamp(0.0, (w - 1) as f64);
    // ... standard 4-corner interpolation
}
```

at `elastic_transform.rs`. The `.clamp` is the
"border-mode" sampler — out-of-bounds coordinates fall back to the
nearest edge pixel.

### Transform impl (REQ-5)

`fn apply` at `apply in elastic_transform.rs`:

```rust
// Generate random displacement fields, smooth, scale by alpha.
for _ in 0..numel {
    dy_field.push(2.0 * random_f64() - 1.0);
    dx_field.push(2.0 * random_f64() - 1.0);
}
let dy_field = gaussian_filter_2d(&dy_field, h, w, self.sigma);
let dx_field = gaussian_filter_2d(&dx_field, h, w, self.sigma);
let dy_field: Vec<f64> = dy_field.iter().map(|v| v * self.alpha).collect();
let dx_field: Vec<f64> = dx_field.iter().map(|v| v * self.alpha).collect();
// Per channel, sample each output pixel from displaced source.
for row in 0..h {
    for col in 0..w {
        let src_y = row as f64 + dy_field[row * w + col];
        let src_x = col as f64 + dx_field[row * w + col];
        let val = bilinear_sample(&ch_data, h, w, src_y, src_x);
        output.push(cast::<f64, T>(val)?);
    }
}
```

This is the Simard "best practices" elastic transform: a single
random-then-smooth displacement field shared across all channels (so
the per-channel deformation is coherent), with alpha scaling the
displacement magnitude in pixels.

### NOT-STARTED gap (REQ-6)

Upstream has `interpolation` (NEAREST/BILINEAR), `fill` (OOB fill
value), and tuple `alpha`/`sigma` (per-axis displacement) parameters.
Blocker #1521 covers all three.

### Non-test production consumers

- `pub use elastic_transform::ElasticTransform;` at
  `ferrotorch-vision/src/transforms/mod.rs:22`.
- (Note: `ElasticTransform` is NOT re-exported at the crate root in
  `lib.rs:113-115`. Callers reach it via
  `ferrotorch_vision::transforms::ElasticTransform`.)

## Parity contract

`parity_ops = []`.

- **`alpha == 0`**: identity (no draw, no smoothing).
- **Uniform-value input**: stays uniform — bilinear interpolation of
  a constant field is that constant.
- **Border samples**: clamp to edge (NOT zero like
  `random_rotation`). This matches upstream's
  `grid_sample(padding_mode='border')` default.
- **Cross-channel coherence**: all channels share the same
  displacement field, so RGB images deform coherently (no chromatic
  drift).

## Verification

Tests in `mod tests in elastic_transform.rs` (10 tests):

- `test_elastic_output_shape_preserved in elastic_transform.rs`
- `test_elastic_zero_alpha_is_identity in elastic_transform.rs`
- `test_elastic_constant_image_unchanged_interior in elastic_transform.rs`
- `test_elastic_rejects_non_3d in elastic_transform.rs`
- `test_elastic_rejects_zero_dim in elastic_transform.rs`
- `test_elastic_negative_alpha_errors in elastic_transform.rs`
- `test_elastic_zero_sigma_errors in elastic_transform.rs`
- `test_bilinear_sample_corner in elastic_transform.rs`
- `test_bilinear_sample_midpoint in elastic_transform.rs`
- `test_bilinear_sample_out_of_bounds_clamps in elastic_transform.rs`

Smoke:

```bash
cargo test -p ferrotorch-vision --lib transforms::elastic_transform:: 2>&1 | tail -3
```

Expected: `10 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct ElasticTransform<T: Float>` with `alpha, sigma, _marker` at `ElasticTransform in ferrotorch-vision/src/transforms/elastic_transform.rs`, mirroring `torchvision/transforms/v2/_geometry.py:999` `class ElasticTransform`; non-test consumer: `pub use elastic_transform::ElasticTransform;` at `mod.rs` exposes it through the public transforms namespace. |
| REQ-2 | SHIPPED | impl: `pub fn ElasticTransform::new(alpha: f64, sigma: f64) -> FerrotorchResult<Self>` with `alpha >= 0` and `sigma > 0` validation at `alpha in elastic_transform.rs`; non-test consumer: reachable via the `mod.rs` re-export. |
| REQ-3 | SHIPPED | impl: `fn gaussian_kernel_1d` at `gaussian_kernel_1d in elastic_transform.rs` and `fn gaussian_filter_2d` at `gaussian_filter_2d in elastic_transform.rs`; non-test consumer: `fn apply` in this same file calls `gaussian_filter_2d(&dy_field, h, w, self.sigma)` and `(&dx_field, ...)` at `gaussian_filter_2d in elastic_transform.rs`. |
| REQ-4 | SHIPPED | impl: `fn bilinear_sample(data, h, w, y, x) -> f64` with clamp-to-edge at `bilinear_sample in elastic_transform.rs`; non-test consumer: `fn apply` in this same file calls `bilinear_sample(&ch_data, h, w, src_y, src_x)` at `bilinear_sample in elastic_transform.rs`. |
| REQ-5 | SHIPPED | impl: `impl<T: Float> Transform<T> for ElasticTransform<T>` with shape/dim checks, random-field gen, Gaussian smooth, per-channel bilinear sample at `elastic_transform.rs`; non-test consumer: any `Box<dyn Transform<T>>` slot — composes into augmentation `Compose` pipelines via the `mod.rs` re-export. |
| REQ-6 | SHIPPED | impl: `with_interpolation` / `with_fill` builders + `new_range` tuple constructor + nearest/bilinear+fill dispatch at `new_range in ferrotorch-vision/src/transforms/elastic_transform.rs,200-260`; non-test consumer: `pub use elastic_transform::ElasticTransform;` at `mod.rs` — augmentation pipelines call `ElasticTransform::new_range((0.0, 60.0), (3.0, 7.0))?.with_interpolation(InterpolationMode::Nearest).with_fill(0.0)` per upstream `_geometry.py:999-1090`. |
