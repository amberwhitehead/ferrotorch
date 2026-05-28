# ferrotorch-vision — `VisionToTensor` transform

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (torchvision v0.26.0 site-packages)
upstream-paths:
  - torchvision/transforms/v2/_type_conversion.py
  - torchvision/transforms/v1/functional.py
-->

## Summary

`ferrotorch-vision/src/transforms/to_tensor.rs` provides
`VisionToTensor<T: Float>`, which converts an image-shaped `[H, W, C]`
tensor with values in `[0, 255]` into a `[C, H, W]` tensor with values
scaled to `[0, 1]` (divide by 255). Mirrors the deprecated-but-still-
canonical `torchvision.transforms.ToTensor` (v1) which is reachable in
v2 via the `ToImage` + `ToDtype(torch.float32, scale=True)` composition.

## Requirements

- REQ-1: `pub struct VisionToTensor<T: Float>` carrying only
  `PhantomData<T>` (no runtime state). Mirrors the parameterless
  `class ToTensor` in torchvision v1.

- REQ-2: `pub fn VisionToTensor::new() -> Self` constructor and an
  `impl Default` returning the same. These are interchangeable; the
  `Default` impl exists for ergonomic `VisionToTensor::default()`
  use inside config-driven pipelines.

- REQ-3: `impl<T: Float> Transform<T> for VisionToTensor<T>` —
  `apply` rejects non-3-D input, then reads dims as `[H, W, C]` (NOT
  `[C, H, W]`) and emits `[C, H, W]` output with every value divided
  by 255. The dimension reinterpretation is the entire point: image
  decoders (libpng, libjpeg, image-rs) hand us HWC, but
  PyTorch conv layers expect CHW.

- REQ-4: Numeric cast — values are read as `T`, divided by `cast::<f64, T>(255.0)?`,
  then written as `T`. The divide is exact for representable values;
  the `cast::<f64, T>` propagates `Err` on non-representable f64s (the
  255.0 case is always representable in both `f32` and `f64`).

## Acceptance Criteria

- [x] AC-1: Construct via `VisionToTensor::<f64>::new()` or
  `VisionToTensor::<f64>::default()`.
- [x] AC-2: `[2, 3, 3]` HWC input transposes to `[3, 2, 3]` CHW
  with `/255` scaling (verified by
  `test_to_tensor_transposes_hwc_to_chw in to_tensor.rs`).
- [x] AC-3: All-255 input → all-1.0 output; all-0 input → all-0
  output (verified by `test_to_tensor_scales_to_unit_range` at
  `test_to_tensor_scales_to_unit_range in to_tensor.rs`).
- [x] AC-4: Single pixel `[51, 102, 153]` → `[51/255, 102/255, 153/255]`
  (verified by `test_to_tensor_single_pixel in to_tensor.rs`).
- [x] AC-5: Non-3-D input returns `Err` (verified by
  `test_to_tensor_rejects_non_3d in to_tensor.rs`).
- [x] AC-6: Works for both `f32` and `f64` element types (verified by
  `test_to_tensor_f32 in to_tensor.rs`).

## Architecture

### Struct (REQ-1) + constructor (REQ-2)

```rust
pub struct VisionToTensor<T: Float> {
    _marker: std::marker::PhantomData<T>,
}
impl<T: Float> VisionToTensor<T> {
    pub fn new() -> Self { Self { _marker: PhantomData } }
}
impl<T: Float> Default for VisionToTensor<T> {
    fn default() -> Self { Self::new() }
}
```

at `to_tensor.rs`.

### Transform impl (REQ-3, REQ-4)

`fn apply` at `apply in to_tensor.rs`:

```rust
let h = shape[0];   // input is HWC
let w = shape[1];
let c = shape[2];
let scale: T = cast::<f64, T>(255.0)?;
// output is CHW
for row in 0..h {
    for col in 0..w {
        for ch in 0..c {
            let src_idx = row * w * c + col * c + ch;
            let dst_idx = ch * h * w + row * w + col;
            output[dst_idx] = data[src_idx] / scale;
        }
    }
}
```

The index arithmetic is the explicit HWC→CHW transpose. Reading source
in HWC order (row-major) and writing destination in CHW order trades
sequential reads for non-sequential writes; for typical image sizes
this is bound by memory bandwidth either way.

The `/scale` divide is identical to upstream's `tensor.div(255)`
(`functional.py:to_tensor`) when the target dtype is float.

### Non-test production consumers

- `pub use to_tensor::VisionToTensor;` at
  `ferrotorch-vision/src/transforms/mod.rs:33` AND `VisionToTensor`
  in the crate-root re-export at `ferrotorch-vision/src/lib.rs:115`.
- The conformance inventory at
  `ferrotorch-vision/tests/conformance/_surface_inventory.toml:123`
  registers `ferrotorch_vision::VisionToTensor` and `::new` as the
  public surface.

## Parity contract

`parity_ops = []`. The behaviour is a deterministic linear scale +
permutation; no random gate.

- **All zeros**: output all zeros.
- **All 255s**: output all 1.0 (with f64-exact result; f32 has minor
  rounding because `1/255` isn't representable exactly).
- **Mid-range pixel `127`**: `127/255 ≈ 0.498…` — matches upstream's
  `torch.uint8 / 255` semantics.
- **Single-channel input** (`c == 1`): transpose still works; output
  shape is `[1, H, W]`.
- **Non-3-D input**: `InvalidArgument`. Upstream's v2 `ToImage` accepts
  PIL images and ndarrays directly; ferrotorch's version assumes the
  caller has already decoded to an HWC tensor.

## Verification

Tests in `mod tests in to_tensor.rs` (5 tests):

- `test_to_tensor_transposes_hwc_to_chw in to_tensor.rs`
- `test_to_tensor_scales_to_unit_range in to_tensor.rs`
- `test_to_tensor_single_pixel in to_tensor.rs`
- `test_to_tensor_rejects_non_3d in to_tensor.rs`
- `test_to_tensor_f32 in to_tensor.rs`

Smoke:

```bash
cargo test -p ferrotorch-vision --lib transforms::to_tensor:: 2>&1 | tail -3
```

Expected: `5 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct VisionToTensor<T: Float>` with `_marker: PhantomData<T>` at `VisionToTensor in ferrotorch-vision/src/transforms/to_tensor.rs`, mirroring the v1 `torchvision.transforms.ToTensor` parameterless class; non-test consumer: `pub use to_tensor::VisionToTensor;` at `mod.rs` AND `VisionToTensor` in the crate-root re-export at `ferrotorch-vision/src/lib.rs`. |
| REQ-2 | SHIPPED | impl: `pub fn VisionToTensor::new() -> Self` at `VisionToTensor in to_tensor.rs` and `impl Default for VisionToTensor<T>` at `new in to_tensor.rs`; non-test consumer: registered in the conformance surface inventory at `ferrotorch-vision/tests/conformance/_surface_inventory.toml:129` as `ferrotorch_vision::VisionToTensor::new`; reachable via the crate-root re-export. |
| REQ-3 | SHIPPED | impl: `impl<T: Float> Transform<T> for VisionToTensor<T>` with shape check + HWC→CHW index transpose + divide-by-255 at `to_tensor.rs`; non-test consumer: any `Box<dyn Transform<T>>` slot accepts this — typically inserted at the start of a `Compose` pipeline right after image decoding. |
| REQ-4 | SHIPPED | impl: `let scale: T = cast::<f64, T>(255.0)?;` followed by `data[src_idx] / scale` at `data in to_tensor.rs`; non-test consumer: same as REQ-3 — the divide happens unconditionally on every `apply` invocation through the crate-root `VisionToTensor` re-export. |
