# ferrotorch-vision — `VisionNormalize` transform

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (torchvision v0.26.0 site-packages)
upstream-paths:
  - torchvision/transforms/v2/_misc.py
-->

## Summary

`ferrotorch-vision/src/transforms/vision_normalize.rs` provides
`VisionNormalize<T: Float>`, a thin vision-flavored wrapper around
`ferrotorch_data::Normalize` that accepts `[f64; 3]` mean/std arrays
(matching torchvision's RGB convention) and exposes a canonical
`::imagenet()` shortcut. Mirrors `torchvision.transforms.v2.Normalize`
at `_misc.py:142-175`.

## Requirements

- REQ-1: `pub struct VisionNormalize<T: Float>` wrapping a single
  `inner: Normalize<T>` from `ferrotorch_data`. Mirrors `_misc.py:142`
  `class Normalize(Transform)`.

- REQ-2: `pub fn VisionNormalize::new(mean: [f64; 3], std: [f64; 3]) ->
  FerrotorchResult<Self>` constructor — converts the fixed-size arrays
  to `Vec<f64>` and forwards to `Normalize::new`. The `[f64; 3]` shape
  encodes "RGB" at the type level so the caller can't pass the wrong
  number of channels.

- REQ-3: `pub fn VisionNormalize::imagenet() -> Self` convenience
  constructor — calls `Self::new(IMAGENET_MEAN, IMAGENET_STD)`. The
  `.expect(...)` documents the invariant that ImageNet's published
  constants are representable in any `Float`. Mirrors the canonical
  ImageNet preprocessing chain documented at
  `torchvision.models.resnet50.IMAGENET1K_V2.transforms`.

- REQ-4: `impl<T: Float> Transform<T> for VisionNormalize<T>` —
  delegates to `self.inner.apply(input)`. The inner `Normalize::apply`
  does the per-channel `(input[c] - mean[c]) / std[c]` math (lives in
  `ferrotorch-data`).

## Acceptance Criteria

- [x] AC-1: `VisionNormalize::new([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])`
  constructs successfully.
- [x] AC-2: `VisionNormalize::<f64>::imagenet()` constructs the
  canonical ImageNet normalizer (verified by every
  `vision_normalize.rs` test).
- [x] AC-3: Normalizing the per-channel mean gives 0 (verified by
  `test_vision_normalize_known_values` at `vision_normalize.rs:64`).
- [x] AC-4: Channel-wise math is `(v - μ) / σ` (verified by
  `test_vision_normalize_non_zero_result` at `vision_normalize.rs:79`).
- [x] AC-5: Custom `[mean, std]` arrays work (verified by
  `test_vision_normalize_custom_stats` at `vision_normalize.rs:111`).
- [x] AC-6: Spatial broadcasting — normalizing a `[3, 2, 2]` tensor
  with all zeros yields `[3, 2, 2]` output with channel-specific
  values `-μ/σ` (verified by `test_vision_normalize_spatial` at
  `vision_normalize.rs:126`).
- [x] AC-7: Works for `f32` (verified by `test_vision_normalize_f32`
  at `vision_normalize.rs:167`).

## Architecture

### Struct (REQ-1)

```rust
pub struct VisionNormalize<T: Float> {
    inner: Normalize<T>,
}
```

at `vision_normalize.rs:18-20`. The wrapper exists for two reasons:
1. Type-level encoding that mean/std are length-3 (RGB), via `[f64; 3]`.
2. The `imagenet()` shortcut keeps the canonical constants
   discoverable in IDE autocomplete on `VisionNormalize::`.

### Constructors (REQ-2, REQ-3)

```rust
pub fn new(mean: [f64; 3], std: [f64; 3]) -> FerrotorchResult<Self> {
    Ok(Self { inner: Normalize::new(mean.to_vec(), std.to_vec())? })
}
pub fn imagenet() -> Self {
    Self::new(IMAGENET_MEAN, IMAGENET_STD)
        .expect("invariant: ImageNet constants are within Float range")
}
```

at `vision_normalize.rs:23-50`. The `.expect` is not a runtime panic
path; `ImageNet_MEAN ⊂ [0.4, 0.5]` and `IMAGENET_STD ⊂ [0.22, 0.23]`
are representable in every `Float` (`f32`, `f64`).

### Transform impl (REQ-4)

```rust
impl<T: Float> Transform<T> for VisionNormalize<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        self.inner.apply(input)
    }
}
```

at `vision_normalize.rs:52-56`. Single-line delegation; the math lives
in `ferrotorch_data::Normalize`. Treating the wrapper as a vision-side
configuration façade keeps the math centralised — `Normalize` is also
used by non-vision code paths.

### Non-test production consumers

- `pub use vision_normalize::VisionNormalize;` at
  `ferrotorch-vision/src/transforms/mod.rs:35` AND `VisionNormalize`
  in the crate-root re-export at `ferrotorch-vision/src/lib.rs:115`.
- Production users include any training driver that loads
  ImageNet-pretrained weights — they construct
  `VisionNormalize::imagenet()` at the end of their preprocessing
  `Compose`.
- The conformance surface inventory at
  `ferrotorch-vision/tests/conformance/_surface_inventory.toml:103-117`
  registers `ferrotorch_vision::VisionNormalize`, `::new`, and
  `::imagenet` as the public API.
- This is THE non-test production consumer of `IMAGENET_MEAN` /
  `IMAGENET_STD` (REQ-3 of `mod.md`): `Self::new(IMAGENET_MEAN,
  IMAGENET_STD)` at `vision_normalize.rs:47`.

## Parity contract

`parity_ops = []`. The numerical contract is delegated to
`ferrotorch_data::Normalize`. Edge cases:

- **`std[c] == 0`**: division by zero — `Normalize::new` errors out
  at construction (the underlying check lives in `ferrotorch-data`).
- **Non-3-channel input**: the inner `Normalize` enforces channel-
  count match against the mean/std length; mismatched channels return
  `Err`.
- **NaN payload**: passes through unchanged (NaN - μ / σ = NaN).
- **`f32` precision**: `vision_normalize_f32` test uses `< 1e-5` for
  the expected-0.0 comparison, accommodating f32 roundoff in the
  divide.

## Verification

Tests in `mod tests in vision_normalize.rs` (5 tests):

- `test_vision_normalize_known_values` at `vision_normalize.rs:64`
- `test_vision_normalize_non_zero_result` at `vision_normalize.rs:79`
- `test_vision_normalize_custom_stats` at `vision_normalize.rs:111`
- `test_vision_normalize_spatial` at `vision_normalize.rs:126`
- `test_vision_normalize_f32` at `vision_normalize.rs:167`

Smoke:

```bash
cargo test -p ferrotorch-vision --lib transforms::vision_normalize:: 2>&1 | tail -3
```

Expected: `5 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct VisionNormalize<T: Float>` wrapping `inner: Normalize<T>` at `ferrotorch-vision/src/transforms/vision_normalize.rs:18-20`, mirroring `torchvision/transforms/v2/_misc.py:142` `class Normalize(Transform)`; non-test consumer: `pub use vision_normalize::VisionNormalize;` at `mod.rs:35` AND `VisionNormalize` in the crate-root re-export at `ferrotorch-vision/src/lib.rs:115`. |
| REQ-2 | SHIPPED | impl: `pub fn VisionNormalize::new(mean: [f64; 3], std: [f64; 3]) -> FerrotorchResult<Self>` at `vision_normalize.rs:30-34`; non-test consumer: registered in `ferrotorch-vision/tests/conformance/_surface_inventory.toml:109` as `ferrotorch_vision::VisionNormalize::new`; reachable via the crate-root re-export. |
| REQ-3 | SHIPPED | impl: `pub fn VisionNormalize::imagenet() -> Self` at `vision_normalize.rs:46-49` reading `IMAGENET_MEAN`/`IMAGENET_STD` from the parent module; non-test consumer: registered in `ferrotorch-vision/tests/conformance/_surface_inventory.toml:115` as `ferrotorch_vision::VisionNormalize::imagenet`; this is the canonical-ImageNet entry point that downstream pretrained-classifier preprocessing pipelines invoke. |
| REQ-4 | SHIPPED | impl: `impl<T: Float> Transform<T> for VisionNormalize<T>` at `vision_normalize.rs:52-56` delegating to `self.inner.apply(input)`; non-test consumer: any `Box<dyn Transform<T>>` slot accepts this — typically the final stage of an ImageNet `Compose` pipeline. |
