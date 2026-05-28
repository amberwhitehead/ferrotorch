# ferrotorch-vision — `RandomVerticalFlip` transform

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (torchvision v0.26.0 site-packages)
upstream-paths:
  - torchvision/transforms/v2/_geometry.py
  - torchvision/transforms/v2/functional/_geometry.py
-->

## Summary

`ferrotorch-vision/src/transforms/random_vertical_flip.rs` provides
`RandomVerticalFlip<T: Float>`, the spatial complement to
`RandomHorizontalFlip`: it reverses rows (top↔bottom) of a `[C, H, W]`
tensor with probability `p`. Mirrors
`torchvision.transforms.v2.RandomVerticalFlip` at
`_geometry.py:52-67`.

## Requirements

- REQ-1: `pub struct RandomVerticalFlip<T: Float>` storing `p: f64` and
  a `PhantomData<T>` marker. Mirrors
  `torchvision/transforms/v2/_geometry.py:52` `class RandomVerticalFlip`.

- REQ-2: `pub fn RandomVerticalFlip::new(p: f64) -> FerrotorchResult<Self>`
  with `p ∈ [0, 1]` validation, returning
  `FerrotorchError::InvalidArgument` otherwise.

- REQ-3: `impl<T: Float> Default for RandomVerticalFlip<T>` returning
  `Self::new(0.5)` — matches upstream default.

- REQ-4: `impl<T: Float> Transform<T> for RandomVerticalFlip<T>` —
  `apply` rejects non-3-D input, draws `random_f64() < p`, and on
  trigger reverses the H-axis (per channel: rows iterated
  `0..h` in reverse, copied into a contiguous output). Single-row
  inputs (`h <= 1`) take a fast no-op path that still returns a fresh
  tensor (matches upstream "flip of a 1-row image is the image").

## Acceptance Criteria

- [x] AC-1: `new(0.5)` succeeds; `new(2.0)` returns `Err`.
- [x] AC-2: `p = 1.0` over a `[1, 3, 2]` input `[1,2,3,4,5,6]`
  yields `[5,6,3,4,1,2]` (verified by
  `test_random_vertical_flip_always in random_vertical_flip.rs`).
- [x] AC-3: `p = 0.0` is identity (verified by
  `test_random_vertical_flip_never in random_vertical_flip.rs`).
- [x] AC-4: Multi-channel inputs flip each channel independently
  (verified by `test_random_vertical_flip_multichannel` at
  `test_random_vertical_flip_multichannel in random_vertical_flip.rs`).
- [x] AC-5: Non-3-D input returns `Err` (verified at `Err in random_vertical_flip.rs`).
- [x] AC-6: `h = 1` returns input unchanged (verified at
  `random_vertical_flip.rs`).

## Architecture

### Struct (REQ-1)

```rust
pub struct RandomVerticalFlip<T: Float> {
    p: f64,
    _marker: std::marker::PhantomData<T>,
}
```

at `random_vertical_flip.rs`. Same shape as
`RandomHorizontalFlip` because the two ops differ only in which axis
they reverse.

### Constructor (REQ-2)

`fn new` at `new in random_vertical_flip.rs` — identical structure to
the horizontal case, validating `p ∈ [0, 1]`.

### Default impl (REQ-3)

`impl Default` at `random_vertical_flip.rs` returning
`Self::new(0.5).expect(...)`.

### Transform impl (REQ-4)

`fn apply` at `apply in random_vertical_flip.rs`:

1. 3-D shape check.
2. Random gate; if not triggered, return input.
3. Fast path for `h <= 1`: copy storage as-is.
4. General case: per channel, iterate rows in reverse order, copying
   each row's W contiguous bytes into the output buffer.

The general case uses `data[start..start + w]` slicing + `extend_from_slice`
which is one bulk-copy per row — efficient because the row layout in
`[C, H, W]` is contiguous along W.

### Non-test production consumers

- `pub use random_vertical_flip::RandomVerticalFlip;` at
  `ferrotorch-vision/src/transforms/mod.rs:30`.
- `RandomVerticalFlip` IS re-exported at the crate root in
  `ferrotorch-vision/src/lib.rs:114` — callers reach it via
  `ferrotorch_vision::RandomVerticalFlip`.

## Parity contract

`parity_ops = []`.

- **`p == 0`**: input passes through (`Ok(input)`).
- **`p == 1`**: every call flips.
- **`h == 1`**: identity (flip of one row is itself).
- **Non-3-D input**: `InvalidArgument` — same scope limitation as
  `RandomHorizontalFlip`; torchvision supports `[..., C, H, W]`
  arbitrary-leading-batch.
- **Channel independence**: each channel reversed independently via
  the per-channel offset loop — matches upstream.

## Verification

Tests in `mod tests in random_vertical_flip.rs` (5 tests):

- `test_random_vertical_flip_always in random_vertical_flip.rs`
- `test_random_vertical_flip_never in random_vertical_flip.rs`
- `test_random_vertical_flip_multichannel in random_vertical_flip.rs`
- `test_random_vertical_flip_rejects_non_3d in random_vertical_flip.rs`
- `test_random_vertical_flip_single_row in random_vertical_flip.rs`

Smoke:

```bash
cargo test -p ferrotorch-vision --lib transforms::random_vertical_flip:: 2>&1 | tail -3
```

Expected: `5 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct RandomVerticalFlip<T: Float>` with `p: f64` + `_marker: PhantomData<T>` at `ferrotorch-vision/src/transforms/random_vertical_flip.rs:11-14`, mirroring `torchvision/transforms/v2/_geometry.py:52` `class RandomVerticalFlip`; non-test consumer: `pub use random_vertical_flip::RandomVerticalFlip;` at `ferrotorch-vision/src/transforms/mod.rs:30` AND `RandomVerticalFlip` in the crate-root re-export at `ferrotorch-vision/src/lib.rs:114`. |
| REQ-2 | SHIPPED | impl: `pub fn RandomVerticalFlip::new(p: f64) -> FerrotorchResult<Self>` with range check at `new in random_vertical_flip.rs`; non-test consumer: reachable via the crate-root re-export at `lib.rs`. |
| REQ-3 | SHIPPED | impl: `impl Default for RandomVerticalFlip<T>` at `default in random_vertical_flip.rs`; non-test consumer: reachable via `lib.rs` re-export; called by user code as `RandomVerticalFlip::default()`. |
| REQ-4 | SHIPPED | impl: `impl<T: Float> Transform<T> for RandomVerticalFlip<T>` with shape check, fast-path for h≤1, and per-channel row-reverse copy at `random_vertical_flip.rs`; non-test consumer: `Box<dyn Transform<T>>` slots (e.g. inside `Compose<T>`) accept this type — reachable via `lib.rs`. |
