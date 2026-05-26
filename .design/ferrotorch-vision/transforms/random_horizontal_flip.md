# ferrotorch-vision — `RandomHorizontalFlip` transform

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (torchvision v0.26.0 site-packages)
upstream-paths:
  - torchvision/transforms/v2/_geometry.py
  - torchvision/transforms/v2/functional/_geometry.py
-->

## Summary

`ferrotorch-vision/src/transforms/random_horizontal_flip.rs` provides
`RandomHorizontalFlip<T: Float>`, a transform that flips a `[C, H, W]`
tensor along the W (width) axis with probability `p`. Mirrors
`torchvision.transforms.v2.RandomHorizontalFlip` at
`_geometry.py:34-49`.

## Requirements

- REQ-1: `pub struct RandomHorizontalFlip<T: Float>` storing the
  probability `p: f64` and a `PhantomData<T>` marker (since `T` is only
  used for the generic Transform impl, not stored). Mirrors
  `torchvision/transforms/v2/_geometry.py:34` `class
  RandomHorizontalFlip(_RandomApplyTransform)`.

- REQ-2: `pub fn RandomHorizontalFlip::new(p: f64) -> FerrotorchResult<Self>`
  constructor that validates `p ∈ [0, 1]`, returning
  `FerrotorchError::InvalidArgument` otherwise. Mirrors the
  upstream `_RandomApplyTransform.__init__` probability check
  ("p should be a floating point value in the interval [0.0, 1.0]").

- REQ-3: `impl<T: Float> Default for RandomHorizontalFlip<T>` returning
  `Self::new(0.5)` — matches upstream's `p=0.5` default
  (`_geometry.py:43` `Default value is 0.5`).

- REQ-4: `impl<T: Float> Transform<T> for RandomHorizontalFlip<T>` —
  `apply` rejects non-3-D input, draws `random_f64() < p`, and if
  triggered reverses each row's columns. Returns input unchanged when
  not triggered. Mirrors `torchvision.transforms.v2.functional.horizontal_flip`
  on `[C, H, W]` tensors.

## Acceptance Criteria

- [x] AC-1: `RandomHorizontalFlip::new(0.5)` succeeds; `new(-0.1)` and
  `new(1.5)` return `Err(InvalidArgument)`.
- [x] AC-2: With `p = 1.0`, applying to row `[1,2,3]` yields `[3,2,1]`
  (verified by `test_horizontal_flip_shape` at `random_horizontal_flip.rs:80`).
- [x] AC-3: With `p = 0.0`, output equals input (verified by
  `test_horizontal_flip_zero_prob` at `random_horizontal_flip.rs:96`).
- [x] AC-4: Non-3-D input returns `Err(InvalidArgument)`.

## Architecture

### Struct (REQ-1)

```rust
pub struct RandomHorizontalFlip<T: Float> {
    p: f64,
    _marker: std::marker::PhantomData<T>,
}
```

at `random_horizontal_flip.rs:11-14`. `p` is `f64` because the
upstream API takes a Python float; `PhantomData<T>` lets the generic
parameter flow into the `Transform<T>` impl without storing it.

### Constructor (REQ-2)

```rust
pub fn new(p: f64) -> FerrotorchResult<Self> {
    if !(0.0..=1.0).contains(&p) {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("RandomHorizontalFlip: p must be in [0.0, 1.0], got {p}"),
        });
    }
    Ok(Self { p, _marker: std::marker::PhantomData })
}
```

at `random_horizontal_flip.rs:22-32`. Range-rejection matches upstream
behaviour.

### Default impl (REQ-3)

```rust
impl<T: Float> Default for RandomHorizontalFlip<T> {
    fn default() -> Self {
        Self::new(0.5).expect("invariant: default p=0.5 is in [0, 1]")
    }
}
```

at `random_horizontal_flip.rs:35-40`. The `.expect` documents the
construction invariant; `0.5` is in `[0, 1]` so `new` cannot return
`Err`. The `.expect` is not a panic path in practice.

### Transform impl (REQ-4)

`fn apply` at `random_horizontal_flip.rs:42-73`:

1. Check `shape.len() == 3` else return `InvalidArgument`.
2. Draw `random_f64()`; if `>= p`, return input unchanged.
3. Allocate `out = vec![zero(); c*h*w]`.
4. Triple-nested loop reverses each row's columns.
5. Build a new `Tensor` from the flipped storage.

The triple loop is `O(C·H·W)` — same complexity as upstream's
torch-level slice-reverse, no SIMD lowering yet.

### Non-test production consumers

- `pub use random_horizontal_flip::RandomHorizontalFlip;` at
  `ferrotorch-vision/src/transforms/mod.rs:27` — submodule re-export.
- (Note: `RandomHorizontalFlip` is NOT re-exported at the crate root
  in `lib.rs:113-115` — the crate-root re-export list omits it,
  inconsistent with `RandomVerticalFlip` which IS in `lib.rs:114`.
  Callers reach it via `ferrotorch_vision::transforms::RandomHorizontalFlip`.
  Logged as cleanup; not a blocker for this REQ since the `transforms::`
  path is a valid public surface.)

## Parity contract

`parity_ops = []`. The flip kernel is deterministic given the random
gate; numerical contract:

- **`p == 0`**: input returned by value (no clone, just `Ok(input)`).
- **`p == 1`**: every call flips.
- **Non-3-D input**: `InvalidArgument`. Upstream torchvision supports
  arbitrary leading batch dims `[..., C, H, W]`; this is a documented
  R-DEFER R-XLATE gap — batched flip is a follow-up; the underlying
  flip kernel is correct for the `[C, H, W]` case.
- **NaN/Inf payload**: passed through unchanged (the flip never reads
  values, only indexes).

## Verification

Tests in `mod tests in random_horizontal_flip.rs` (2 tests):

- `test_horizontal_flip_shape` at `random_horizontal_flip.rs:80-93`
  verifies the `[3,2,1] / [6,5,4]` column-reverse for a `[1,2,3]`
  tensor with `p = 1.0`.
- `test_horizontal_flip_zero_prob` at `random_horizontal_flip.rs:96-106`
  verifies identity at `p = 0.0`.

Smoke:

```bash
cargo test -p ferrotorch-vision --lib transforms::random_horizontal_flip:: 2>&1 | tail -3
```

Expected: `2 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct RandomHorizontalFlip<T: Float>` with `p: f64` and `_marker: PhantomData<T>` at `ferrotorch-vision/src/transforms/random_horizontal_flip.rs:11-14`, mirroring `torchvision/transforms/v2/_geometry.py:34` `class RandomHorizontalFlip(_RandomApplyTransform)`; non-test consumer: `pub use random_horizontal_flip::RandomHorizontalFlip;` at `ferrotorch-vision/src/transforms/mod.rs:27` exposes it through the public `transforms` namespace. |
| REQ-2 | SHIPPED | impl: `pub fn RandomHorizontalFlip::new(p: f64) -> FerrotorchResult<Self>` with `(0.0..=1.0).contains(&p)` validation at `random_horizontal_flip.rs:22-32`; non-test consumer: reachable via the `pub use` at `mod.rs:27`; user code calls `RandomHorizontalFlip::new(0.5)?` to construct the transform. |
| REQ-3 | SHIPPED | impl: `impl Default for RandomHorizontalFlip<T>` returning `Self::new(0.5).expect(...)` at `random_horizontal_flip.rs:35-40`; non-test consumer: the `Default` trait is reachable via the `pub use` re-export; downstream code uses `RandomHorizontalFlip::default()` when no custom probability is needed. |
| REQ-4 | SHIPPED | impl: `impl<T: Float> Transform<T> for RandomHorizontalFlip<T>` with shape check, random gate, and column-reverse loop at `random_horizontal_flip.rs:42-73`; non-test consumer: any `Box<dyn Transform<T>>` slot accepts this type, so it composes into `Compose<T>` or `RandomApply<T>` pipelines — that's the production surface. The `pub use` at `mod.rs:27` makes the impl reachable. |
