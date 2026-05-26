# ferrotorch-vision — `Resize` transform

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (torchvision v0.26.0 site-packages)
upstream-paths:
  - torchvision/transforms/v2/_geometry.py
  - torchvision/transforms/v2/functional/_geometry.py
-->

## Summary

`ferrotorch-vision/src/transforms/resize.rs` provides `Resize<T: Float>`,
a spatial resize transform that maps a `[C, H, W]` input to
`[C, target_h, target_w]` using nearest-neighbor interpolation. Mirrors
`torchvision.transforms.v2.Resize` at `_geometry.py:70`
restricted to the nearest-neighbor `InterpolationMode.NEAREST` case.

## Requirements

- REQ-1: `pub struct Resize<T: Float>` storing `height: usize`,
  `width: usize`, and `PhantomData<T>`. Mirrors
  `torchvision/transforms/v2/_geometry.py:70` `class Resize(Transform)`.

- REQ-2: `pub fn Resize::new(height: usize, width: usize) -> Self`
  constructor — infallible. Mirrors upstream `Resize(size=(h, w))`
  with the explicit-tuple form.

- REQ-3: `impl<T: Float> Transform<T> for Resize<T>` — `apply` rejects
  non-3-D input, then per-channel maps each output pixel `(oh, ow)`
  to input pixel `(ih = (oh * in_h) / out_h, iw = (ow * in_w) / out_w)`
  with a guard `in_h == 1` → `ih = 0` (avoid div-by-zero in the
  degenerate single-row case). Mirrors
  `torchvision.transforms.v2.functional.resize` with `interpolation =
  InterpolationMode.NEAREST` and `antialias = False`.

- REQ-4: NOT-STARTED — `interpolation` parameter (BILINEAR / BICUBIC),
  `antialias` flag, and `max_size` constraint from upstream are not
  implemented. Resize is locked to nearest-neighbor + no max-size
  limit. Blocker #1514.

## Acceptance Criteria

- [x] AC-1: `Resize::new(4, 4)` applied to a `[3, 8, 8]` input
  produces a `[3, 4, 4]` output (verified by
  `test_resize_output_shape` at `resize.rs:69`).
- [x] AC-2: Upscale `[1, 2, 2]` → `[1, 6, 6]` succeeds (verified
  by `test_resize_upscale_shape` at `resize.rs:79`).
- [x] AC-3: Same-size resize preserves values exactly (verified by
  `test_resize_identity` at `resize.rs:90`).
- [x] AC-4: Nearest-neighbor mapping `[1,2;3,4]` → 4x4 replicates
  pixels into 2x2 blocks (verified by
  `test_resize_nearest_neighbor_values` at `resize.rs:101`).
- [x] AC-5: Non-3-D input returns `Err` (verified by
  `test_resize_rejects_non_3d` at `resize.rs:123`).
- [ ] AC-6: NOT-STARTED — `interpolation` / `antialias` / `max_size`.
  Blocker #1514.

## Architecture

### Struct (REQ-1) + constructor (REQ-2)

```rust
pub struct Resize<T: Float> {
    height: usize,
    width: usize,
    _marker: std::marker::PhantomData<T>,
}
impl<T: Float> Resize<T> {
    pub fn new(height: usize, width: usize) -> Self { ... }
}
```

at `resize.rs:9-24`.

### Transform impl (REQ-3)

`fn apply` at `resize.rs:26-62`:

```rust
for c in 0..channels {
    let channel_offset = c * in_h * in_w;
    for oh in 0..out_h {
        let ih = if in_h == 1 { 0 } else { (oh * in_h) / out_h };
        for ow in 0..out_w {
            let iw = if in_w == 1 { 0 } else { (ow * in_w) / out_w };
            output.push(data[channel_offset + ih * in_w + iw]);
        }
    }
}
```

This is the canonical floor-division nearest-neighbor mapping:
`ih = floor(oh * in_h / out_h)`, `iw = floor(ow * in_w / out_w)`.
Upstream's nearest-neighbor implementation
(`torchvision/transforms/v2/functional/_geometry.py:resize_image_tensor`
when `interpolation = InterpolationMode.NEAREST`) uses the same
formula via `torch.nn.functional.interpolate(..., mode='nearest')`.

### NOT-STARTED gap (REQ-4)

Upstream `Resize` takes:
- `interpolation: InterpolationMode = BILINEAR` (we ship NEAREST only)
- `antialias: Optional[bool] = True` (we don't antialias)
- `max_size: Optional[int] = None` (we don't enforce a max edge)
- `size: int | sequence[int]` — single-int means "scale shortest edge
  to size" (we require explicit `(h, w)`)

Blocker #1514: "Add bilinear/bicubic interpolation, antialias, max_size,
and shortest-edge sizing to Resize". The default upstream behavior is
BILINEAR + antialias, so the bilinear path is the highest-priority gap;
this is documented and tracked.

### Non-test production consumers

- `pub use resize::Resize;` at
  `ferrotorch-vision/src/transforms/mod.rs:31` AND `Resize` is in the
  crate-root re-export at `ferrotorch-vision/src/lib.rs:114` — callers
  reach it via `ferrotorch_vision::Resize`.

## Parity contract

`parity_ops = []`.

- **Same-size resize**: identity (each output pixel reads the same-
  index input pixel due to floor division).
- **2x upscale**: each input pixel is replicated into a `2 × 2` block
  in the output (`test_resize_nearest_neighbor_values` pins this).
- **Single-row/col input**: the `in_h == 1` / `in_w == 1` guards
  avoid the degenerate `(oh * 1) / out_h` evaluation that would
  always return 0 anyway, but the explicit branch makes the intent
  visible.
- **Non-3-D input**: `InvalidArgument`. Same scope limit as flips —
  no leading-batch-dims support yet.

## Verification

Tests in `mod tests in resize.rs` (5 tests):

- `test_resize_output_shape` at `resize.rs:69`
- `test_resize_upscale_shape` at `resize.rs:79`
- `test_resize_identity` at `resize.rs:90`
- `test_resize_nearest_neighbor_values` at `resize.rs:101`
- `test_resize_rejects_non_3d` at `resize.rs:123`

Smoke:

```bash
cargo test -p ferrotorch-vision --lib transforms::resize:: 2>&1 | tail -3
```

Expected: `5 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct Resize<T: Float>` with `height, width, _marker` at `ferrotorch-vision/src/transforms/resize.rs:9-13`, mirroring `torchvision/transforms/v2/_geometry.py:70` `class Resize(Transform)`; non-test consumer: `pub use resize::Resize;` at `ferrotorch-vision/src/transforms/mod.rs:31` AND `Resize` in the crate-root re-export at `ferrotorch-vision/src/lib.rs:114`. |
| REQ-2 | SHIPPED | impl: `pub fn Resize::new(height: usize, width: usize) -> Self` at `resize.rs:17-23`; non-test consumer: reachable via `lib.rs:114` re-export; the conformance inventory at `ferrotorch-vision/tests/conformance/_surface_inventory.toml:81` registers `ferrotorch_vision::Resize::new` as part of the public surface contract. |
| REQ-3 | SHIPPED | impl: `impl<T: Float> Transform<T> for Resize<T>` with floor-division nearest-neighbor loop at `resize.rs:26-62`; non-test consumer: any `Box<dyn Transform<T>>` slot accepts this — reachable via `lib.rs:114` re-export. |
| REQ-4 | NOT-STARTED | open prereq blocker #1514 — `interpolation` (BILINEAR/BICUBIC), `antialias`, `max_size`, and shortest-edge int-size handling from `torchvision/transforms/v2/_geometry.py:70-170` are not implemented. Resize is NEAREST-only, no antialias, no max_size. |
