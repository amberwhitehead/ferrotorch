# ferrotorch-vision — `RandomCrop` transform

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (torchvision v0.26.0 site-packages)
upstream-paths:
  - torchvision/transforms/v2/_geometry.py
-->

## Summary

`ferrotorch-vision/src/transforms/random_crop.rs` provides
`RandomCrop<T: Float>`, a transform that extracts a random
`(crop_h, crop_w)` rectangular region of a `[C, H, W]` input by
sampling a uniform top-left corner. Mirrors
`torchvision.transforms.v2.RandomCrop` at
`_geometry.py:759` (the geometric core; the upstream class also has
`padding`, `pad_if_needed`, `fill`, `padding_mode` features that are
NOT-STARTED here).

## Requirements

- REQ-1: `pub struct RandomCrop<T: Float>` storing `crop_h: usize`,
  `crop_w: usize`, and `PhantomData<T>`. Mirrors `_geometry.py:759`
  `class RandomCrop(Transform)`.

- REQ-2: `pub fn RandomCrop::new(crop_h, crop_w) -> Self` constructor
  (infallible — the only invalidity is non-positive sizes, but `usize`
  is non-negative by type, and `0` produces an empty crop which is
  legal). Mirrors upstream `RandomCrop.__init__` core (without the
  padding parameters).

- REQ-3: `pub fn RandomCrop::square(size: usize) -> Self` convenience
  for the common square-crop case (`size == crop_h == crop_w`).
  Mirrors upstream's `int size` interpretation in `_geometry.py:765-767`.

- REQ-4: `impl<T: Float> Transform<T> for RandomCrop<T>` — `apply`
  rejects non-3-D input, returns `InvalidArgument` if either input
  dimension is smaller than the corresponding crop dimension, then
  picks a uniform random top-left corner via `random_usize` and copies
  the crop region into a fresh storage. Mirrors upstream's
  `make_params` + `transform` decomposition (`_geometry.py`
  `RandomCrop.get_params` returns `(top, left, height, width)` exactly
  this way).

- REQ-5: NOT-STARTED — the upstream `padding`, `pad_if_needed`,
  `fill`, `padding_mode` parameters are not implemented. Cropping a
  smaller-than-target input returns `Err` here; upstream would pad and
  then crop. Blocker #1513.

## Acceptance Criteria

- [x] AC-1: `RandomCrop::new(2, 3)` constructs a transform.
- [x] AC-2: `RandomCrop::square(5)` constructs a square crop.
- [x] AC-3: Applying to a `[3, 5, 7]` input with `(2, 3)` crop produces
  a `[3, 2, 3]` output (verified by `test_random_crop_shape` at
  `random_crop.rs:94`).
- [x] AC-4: When input dims exactly match crop dims, output equals
  input data (verified by `test_random_crop_exact_size` at
  `random_crop.rs:106`).
- [x] AC-5: When input is smaller than crop, `apply` returns `Err`
  (verified by `test_random_crop_too_small` at `random_crop.rs:118`).
- [ ] AC-6: NOT-STARTED — `padding`, `pad_if_needed`, `fill`,
  `padding_mode` support. Blocker #1513.

## Architecture

### Struct + constructors (REQ-1, REQ-2, REQ-3)

```rust
pub struct RandomCrop<T: Float> {
    crop_h: usize,
    crop_w: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> RandomCrop<T> {
    pub fn new(crop_h: usize, crop_w: usize) -> Self { ... }
    pub fn square(size: usize) -> Self { Self::new(size, size) }
}
```

at `random_crop.rs:13-32`. The `square` helper exists because the
common ImageNet pipeline uses a single integer (e.g. `RandomCrop(224)`)
to mean a square crop; upstream Python handles this with a
`_setup_size` shim and a sequence-of-1 fallback.

### Transform impl (REQ-4)

`fn apply` at `random_crop.rs:34-87`:

1. Shape check: `shape.len() == 3` else `InvalidArgument`.
2. Bounds check: `h >= crop_h && w >= crop_w` else `InvalidArgument`.
3. Sample top-left:
   ```rust
   let top = if h == self.crop_h { 0 } else { random_usize(h - self.crop_h) };
   let left = if w == self.crop_w { 0 } else { random_usize(w - self.crop_w) };
   ```
   The `if dim == crop_dim` guard avoids `random_usize(0)` which would
   modulo by zero.
4. Triple loop copies the crop region into `out`.
5. Build the output tensor.

### NOT-STARTED gap (REQ-5)

Upstream `RandomCrop(_geometry.py:759-913)` has four
beyond-the-core parameters:
- `padding: int | sequence | None` — pad input before cropping
- `pad_if_needed: bool` — auto-pad when input is smaller than crop
- `fill: number | tuple | dict` — pad fill value
- `padding_mode: 'constant' | 'edge' | 'reflect' | 'symmetric'`

Without these, ferrotorch's RandomCrop returns `Err` when
`pad_if_needed` would have kicked in (input smaller than crop). This
is a documented R-DEFER gap. Blocker #1513 — "Add padding /
pad_if_needed / fill / padding_mode to RandomCrop".

### Non-test production consumers

- `pub use random_crop::RandomCrop;` at
  `ferrotorch-vision/src/transforms/mod.rs:25`.
- (Note: `RandomCrop` is NOT in the crate-root `pub use`
  list at `lib.rs:113-115`. Callers reach it via
  `ferrotorch_vision::transforms::RandomCrop`. Same minor
  inconsistency as `RandomHorizontalFlip` and `Compose`.)

## Parity contract

`parity_ops = []`.

- **Exact-size input** (`h == crop_h && w == crop_w`): crop returns
  identical data (top=0, left=0).
- **`h < crop_h` or `w < crop_w`**: `InvalidArgument`. Upstream
  with `pad_if_needed=True` would pad; without it, raises
  `ValueError`. Currently we ALWAYS error here.
- **Non-3-D input**: `InvalidArgument`.
- **Deterministic with seed**: `vision_manual_seed(s)` before `apply`
  pins the top-left corner deterministically.

## Verification

Tests in `mod tests in random_crop.rs` (3 tests):

- `test_random_crop_shape` at `random_crop.rs:94-104`
- `test_random_crop_exact_size` at `random_crop.rs:106-116`
- `test_random_crop_too_small` at `random_crop.rs:118-128`

Smoke:

```bash
cargo test -p ferrotorch-vision --lib transforms::random_crop:: 2>&1 | tail -3
```

Expected: `3 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct RandomCrop<T: Float>` with `crop_h, crop_w, _marker` at `ferrotorch-vision/src/transforms/random_crop.rs:13-17`, mirroring `torchvision/transforms/v2/_geometry.py:759` `class RandomCrop(Transform)`; non-test consumer: `pub use random_crop::RandomCrop;` at `ferrotorch-vision/src/transforms/mod.rs:25`. |
| REQ-2 | SHIPPED | impl: `pub fn RandomCrop::new(crop_h: usize, crop_w: usize) -> Self` at `random_crop.rs:20-26`; non-test consumer: reachable via `mod.rs:25` re-export. |
| REQ-3 | SHIPPED | impl: `pub fn RandomCrop::square(size: usize) -> Self` at `random_crop.rs:29-31`; non-test consumer: reachable via `mod.rs:25` re-export; called by user code wanting the canonical `RandomCrop(224)` square-crop ergonomics. |
| REQ-4 | SHIPPED | impl: `impl<T: Float> Transform<T> for RandomCrop<T>` with shape + bounds + random-corner + region-copy at `random_crop.rs:34-87`; non-test consumer: any `Box<dyn Transform<T>>` slot accepts this type — the `mod.rs:25` re-export makes it composable into `Compose<T>` pipelines. |
| REQ-5 | NOT-STARTED | open prereq blocker #1513 — `padding` / `pad_if_needed` / `fill` / `padding_mode` parameters from `torchvision/transforms/v2/_geometry.py:759-913` are not implemented. ferrotorch returns `Err` when input is smaller than crop; upstream would pad. |
