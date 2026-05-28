# ferrotorch-vision — `RandomResizedCrop` transform

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (torchvision v0.26.0 site-packages)
upstream-paths:
  - torchvision/transforms/v2/_geometry.py
  - torchvision/transforms/v2/functional/_geometry.py
-->

## Summary

`ferrotorch-vision/src/transforms/random_resized_crop.rs` provides
`RandomResizedCrop<T: Float>`, the canonical Inception-style
augmentation: sample a rectangular region whose area is a random
fraction within `scale` and whose aspect ratio is within `ratio`,
then resize the region to `(height, width)`. Mirrors
`torchvision.transforms.v2.RandomResizedCrop` at `_geometry.py:197`.

## Requirements

- REQ-1: `pub struct RandomResizedCrop<T: Float>` storing `height`,
  `width`, `scale_lo`, `scale_hi`, `ratio_lo`, `ratio_hi`, and
  `PhantomData<T>`. Mirrors `_geometry.py:197` `class RandomResizedCrop`.

- REQ-2: `pub fn RandomResizedCrop::new(height, width, scale: (f64, f64),
  ratio: (f64, f64)) -> FerrotorchResult<Self>` constructor validating:
  - `0 < scale.0 <= scale.1 <= 1`
  - `0 < ratio.0 <= ratio.1`

  Mirrors upstream's `Scale should be a sequence of two floats` and
  `Ratio should be a sequence of two floats` type checks plus the
  `(min, max)` ordering check.

- REQ-3: `pub(crate) fn nn_resize_channel<T: Float>(src, in_h, in_w,
  out_h, out_w, dst: &mut Vec<T>)` private helper — nearest-neighbor
  resize of a single channel from one buffer into a destination
  vector. Re-used by `RandomResizedCrop` and (by exported visibility)
  potentially by other geometric ops needing the same primitive.

- REQ-4: `impl<T: Float> Transform<T> for RandomResizedCrop<T>` —
  `apply` rejects non-3-D input, then repeatedly samples (up to 10
  attempts) a `(target_area, aspect)` pair from `scale × ratio` (with
  aspect drawn in log-space), computes `(h_candidate, w_candidate) =
  (sqrt(area/aspect), sqrt(area*aspect))`, and accepts if both
  candidates fit in the input. Falls back to a deterministic center
  crop at the target aspect ratio if no attempt fits in 10 tries.

  Per channel, extracts the cropped region into a temporary buffer,
  then nearest-neighbor resizes it to `(self.height, self.width)`.

- REQ-5: SHIPPED — `with_interpolation(InterpolationMode)` selects
  nearest (default) or bilinear; `with_antialias(true)` applies a
  separable box pre-filter on downscale axes before bilinear sampling
  (mirroring upstream's `antialias=True` default for `BILINEAR`).
  Bicubic remains a follow-up.

## Acceptance Criteria

- [x] AC-1: `RandomResizedCrop::new(5, 5, (0.08, 1.0), (0.75, 1.333))`
  constructs.
- [x] AC-2: Invalid scale or ratio bounds return `Err`.
- [x] AC-3: Output shape is `[C, height, width]` (verified by
  `test_random_resized_crop_output_shape` at
  `test_random_resized_crop_output_shape in random_resized_crop.rs`).
- [x] AC-4: `scale=(1.0,1.0), ratio=(1.0,1.0)` resizes the full
  image (verified by `test_random_resized_crop_full_scale` at
  `test_random_resized_crop_full_scale in random_resized_crop.rs`).
- [x] AC-5: Output values are a subset of input values (no
  interpolation in the NEAREST path) (verified by
  `test_random_resized_crop_values_from_input` at
  `test_random_resized_crop_values_from_input in random_resized_crop.rs`).
- [x] AC-6: Non-3-D input returns `Err` (verified at
  `random_resized_crop.rs`).
- [x] AC-7: Multichannel input is handled per-channel (verified at
  `random_resized_crop.rs`).
- [x] AC-8: `nn_resize_channel` identity (verified at
  `random_resized_crop.rs`).
- [x] AC-9: `nn_resize_channel` upscale replicates pixels (verified
  at `random_resized_crop.rs`).
- [x] AC-10: bilinear interpolation + optional antialias (verified
  by `test_random_resized_crop_bilinear_output_shape`,
  `test_random_resized_crop_bilinear_uniform_input_stays_uniform`, and
  `test_random_resized_crop_bilinear_with_antialias_smoke` in
  `random_resized_crop.rs`). Bicubic still NOT-STARTED.

## Architecture

### Struct + constructor (REQ-1, REQ-2)

```rust
pub struct RandomResizedCrop<T: Float> {
    height: usize,
    width: usize,
    scale_lo: f64,
    scale_hi: f64,
    ratio_lo: f64,
    ratio_hi: f64,
    _marker: std::marker::PhantomData<T>,
}
```

at `random_resized_crop.rs`. Constructor at
`random_resized_crop.rs` performs both range checks.

### Nearest-neighbor resize helper (REQ-3)

```rust
pub(crate) fn nn_resize_channel<T: Float>(
    src: &[T], in_h, in_w, out_h, out_w, dst: &mut Vec<T>,
) {
    for oh in 0..out_h {
        let ih = if in_h == 1 { 0 } else { (oh * in_h) / out_h };
        for ow in 0..out_w {
            let iw = if in_w == 1 { 0 } else { (ow * in_w) / out_w };
            dst.push(src[ih * in_w + iw]);
        }
    }
}
```

at `pub in random_resized_crop.rs`. `pub(crate)` visibility so other
crate-internal transforms can reuse this primitive without going
through the full `Resize` transform machinery.

### Transform impl (REQ-4)

`fn apply` at `apply in random_resized_crop.rs`:

1. 3-D check.
2. 10-iteration sample-and-validate loop:
   ```rust
   let target_area = area * (lo + random_f64() * (hi - lo));
   let aspect = (log_lo + random_f64() * (log_hi - log_lo)).exp();
   let w_candidate = (sqrt(area * aspect)).round() as usize;
   let h_candidate = (sqrt(area / aspect)).round() as usize;
   if 1 <= candidates && candidates <= input_dims { break; }
   ```
3. Fallback: deterministic center crop at the target aspect ratio.
4. Per channel: extract crop into temp buffer, call
   `nn_resize_channel` into the output vec.

The log-space aspect ratio sampling
(`(log_lo + random_f64() * (log_hi - log_lo)).exp()`) is upstream's
exact form (`_geometry.py:make_params` uses
`torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1]))`).

### NOT-STARTED gap (REQ-5)

Upstream defaults to BILINEAR + antialias=True for the final resize
step. Without these, ferrotorch's `RandomResizedCrop` produces
blocky-looking outputs at large size jumps. Blocker #1520.

### Non-test production consumers

- `pub use random_resized_crop::RandomResizedCrop;` at
  `ferrotorch-vision/src/transforms/mod.rs:28` AND
  `RandomResizedCrop` in the crate-root re-export at
  `ferrotorch-vision/src/lib.rs:114`.

## Parity contract

`parity_ops = []`.

- **Scale = ratio = (1, 1)**: deterministic — uses the entire image,
  resized to target.
- **No attempt fits in 10 tries**: deterministic center crop fallback
  matches upstream's same code path (`_geometry.py:make_params` has
  identical 10-attempt + center-crop fallback).
- **Aspect ratio sampled in log-space**: this matches PyTorch's
  geometric-mean-style sampling, which is the property the original
  Inception paper documented.

## Verification

Tests in `mod tests in random_resized_crop.rs` (7 tests):

- `test_random_resized_crop_output_shape in random_resized_crop.rs`
- `test_random_resized_crop_full_scale in random_resized_crop.rs`
- `test_random_resized_crop_values_from_input in random_resized_crop.rs`
- `test_random_resized_crop_rejects_non_3d in random_resized_crop.rs`
- `test_random_resized_crop_multichannel in random_resized_crop.rs`
- `test_nn_resize_channel_identity in random_resized_crop.rs`
- `test_nn_resize_channel_upscale in random_resized_crop.rs`

Smoke:

```bash
cargo test -p ferrotorch-vision --lib transforms::random_resized_crop:: 2>&1 | tail -3
```

Expected: `7 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct RandomResizedCrop<T: Float>` with 6 fields + `_marker` at `RandomResizedCrop in ferrotorch-vision/src/transforms/random_resized_crop.rs`, mirroring `torchvision/transforms/v2/_geometry.py:197` `class RandomResizedCrop`; non-test consumer: `pub use random_resized_crop::RandomResizedCrop;` at `mod.rs` AND `RandomResizedCrop` in the crate-root re-export at `ferrotorch-vision/src/lib.rs`. |
| REQ-2 | SHIPPED | impl: `pub fn RandomResizedCrop::new(height, width, scale, ratio) -> FerrotorchResult<Self>` with scale/ratio range checks at `new in random_resized_crop.rs`; non-test consumer: reachable via the crate-root re-export at `lib.rs`. |
| REQ-3 | SHIPPED | impl: `pub(crate) fn nn_resize_channel<T: Float>(...)` at `nn_resize_channel in random_resized_crop.rs`; non-test consumer: called from `fn apply` in this same file at `apply in random_resized_crop.rs` (within the per-channel loop). |
| REQ-4 | SHIPPED | impl: `impl<T: Float> Transform<T> for RandomResizedCrop<T>` with 10-attempt sampling, center-crop fallback, per-channel crop + nn-resize at `random_resized_crop.rs`; non-test consumer: any `Box<dyn Transform<T>>` slot — typically the first stage of an Inception/ResNet ImageNet `Compose` training pipeline. |
| REQ-5 | SHIPPED | impl: `with_interpolation` + `with_antialias` builders + `bilinear_resize_channel` separable antialias-prefilter sampler at `bilinear_resize_channel in ferrotorch-vision/src/transforms/random_resized_crop.rs,165-260`; non-test consumer: `pub use random_resized_crop::RandomResizedCrop;` at `mod.rs` AND in the `lib.rs` re-export — ImageNet pipelines call `RandomResizedCrop::new(224, 224, ...)?.with_interpolation(InterpolationMode::Bilinear).with_antialias(true)`. |
