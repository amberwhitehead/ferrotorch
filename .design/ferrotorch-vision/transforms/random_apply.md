# ferrotorch-vision — `RandomApply` + `RandomChoice` transforms

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (torchvision v0.26.0 site-packages)
upstream-paths:
  - torchvision/transforms/v2/_container.py
-->

## Summary

`ferrotorch-vision/src/transforms/random_apply.rs` provides two related
container transforms:

- `RandomApply<T: Float>` — apply a chained list of transforms with
  probability `p`, otherwise return input unchanged. Mirrors
  `torchvision.transforms.v2.RandomApply` at `_container.py:63`.
- `RandomChoice<T: Float>` — pick exactly one transform from a list
  with uniform probability and apply it. Mirrors
  `torchvision.transforms.v2.RandomChoice` at `_container.py:119`.

## Requirements

- REQ-1: `pub struct RandomApply<T: Float>` storing
  `transforms: Vec<Box<dyn Transform<T>>>` and `p: f64`. Mirrors
  upstream `class RandomApply(Transform)`.

- REQ-2: `pub fn RandomApply::new(transforms, p) -> FerrotorchResult<Self>`
  constructor validating `p ∈ [0, 1]`. Mirrors upstream's
  `__init__` probability check.

- REQ-3: `impl<T: Float> Transform<T> for RandomApply<T>` — `apply`
  draws `random_f64() < p`; if not triggered returns input. If
  triggered, threads the input through each child transform in order
  (`Compose`-like). Matches upstream's `forward` semantics
  (`_container.py:104-109`).

- REQ-4: `pub struct RandomChoice<T: Float>` storing
  `transforms: Vec<Box<dyn Transform<T>>>`. Mirrors upstream
  `class RandomChoice(Transform)`.

- REQ-5: `pub fn RandomChoice::new(transforms) -> FerrotorchResult<Self>`
  validating that `transforms` is non-empty (a `RandomChoice` over
  zero options is ill-defined). Mirrors upstream's
  `if not transforms: raise ValueError`.

- REQ-6: `impl<T: Float> Transform<T> for RandomChoice<T>` — `apply`
  samples a uniform index in `[0, n)` via
  `(random_f64() * n) as usize` clamped with `.min(n - 1)` to defend
  against the `random_f64() == 1.0` boundary, then applies
  `self.transforms[idx]`. Mirrors `_container.py:147-149`.

- REQ-7: NOT-STARTED — upstream's `RandomChoice` accepts an optional
  `p: list[float]` per-transform probability vector (weighted choice);
  ferrotorch's version is uniform-only. Blocker #1517.

## Acceptance Criteria

- [x] AC-1: `RandomApply::new(vec![...], 0.5)` constructs.
- [x] AC-2: `RandomApply::new(vec![], 1.5)` returns `Err`.
- [x] AC-3: `p = 1.0` always applies (verified by
  `test_random_apply_always` at `random_apply.rs:91`).
- [x] AC-4: `p = 0.0` never applies (verified by
  `test_random_apply_never` at `random_apply.rs:109`).
- [x] AC-5: Empty `transforms` with `p = 1.0` is identity (verified by
  `test_random_apply_empty_transforms` at `random_apply.rs:126`).
- [x] AC-6: `RandomChoice::new(vec![])` returns `Err`.
- [x] AC-7: `RandomChoice` selects at least both transforms over many
  trials (verified by `test_random_choice_selects_one` at
  `random_apply.rs:136`).
- [x] AC-8: Single-transform `RandomChoice` always applies that one
  (verified at `random_apply.rs:165`).
- [x] AC-9: Both types are `Send + Sync` (verified at
  `random_apply.rs:178`).
- [ ] AC-10: NOT-STARTED — per-transform probability vector for
  `RandomChoice`. Blocker #1517.

## Architecture

### `RandomApply` struct + constructor (REQ-1, REQ-2)

```rust
pub struct RandomApply<T: Float> {
    transforms: Vec<Box<dyn Transform<T>>>,
    p: f64,
}
impl<T: Float> RandomApply<T> {
    pub fn new(transforms: Vec<Box<dyn Transform<T>>>, p: f64) -> FerrotorchResult<Self> {
        if !(0.0..=1.0).contains(&p) {
            return Err(FerrotorchError::InvalidArgument { ... });
        }
        Ok(Self { transforms, p })
    }
}
```

at `random_apply.rs:13-35`. Unlike upstream, an empty `transforms`
vector is accepted — it makes the apply path identity, which is
consistent with `Compose::new(vec![])`.

### `RandomApply` Transform impl (REQ-3)

```rust
impl<T: Float> Transform<T> for RandomApply<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if random_f64() >= self.p {
            return Ok(input);
        }
        let mut current = input;
        for t in &self.transforms {
            current = t.apply(current)?;
        }
        Ok(current)
    }
}
```

at `random_apply.rs:37-48`. One `random_f64()` draw per `apply` call
to gate the whole list (not per-transform — that would be `Compose +
RandomApply` per child).

### `RandomChoice` struct + constructor (REQ-4, REQ-5)

```rust
pub struct RandomChoice<T: Float> {
    transforms: Vec<Box<dyn Transform<T>>>,
}
impl<T: Float> RandomChoice<T> {
    pub fn new(transforms: Vec<Box<dyn Transform<T>>>) -> FerrotorchResult<Self> {
        if transforms.is_empty() {
            return Err(FerrotorchError::InvalidArgument { ... });
        }
        Ok(Self { transforms })
    }
}
```

at `random_apply.rs:55-73`. Empty vector is rejected because
`(random_f64() * 0) as usize % 0` is undefined.

### `RandomChoice` Transform impl (REQ-6)

```rust
impl<T: Float> Transform<T> for RandomChoice<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let n = self.transforms.len();
        let idx = (random_f64() * n as f64) as usize;
        let idx = idx.min(n - 1);
        self.transforms[idx].apply(input)
    }
}
```

at `random_apply.rs:75-82`. The `.min(n - 1)` clamp defends against
`random_f64()` returning exactly 1.0 — extremely rare but
deterministically possible.

### NOT-STARTED gap (REQ-7)

Upstream's `RandomChoice(transforms, p=None)` accepts an optional
weight vector. The non-uniform-weight code path is a minor extension
of `RandomChoice::apply` — sample cumulatively, look up the index.
Blocker #1517 tracks it.

### Non-test production consumers

- `pub use random_apply::{RandomApply, RandomChoice};` at
  `ferrotorch-vision/src/transforms/mod.rs:24` AND both types in the
  crate-root re-export at `ferrotorch-vision/src/lib.rs:113`.
- The conformance surface inventory at
  `ferrotorch-vision/tests/conformance/_surface_inventory.toml:151-174`
  registers `ferrotorch_vision::RandomApply` and `RandomChoice` as
  public surface.

## Parity contract

`parity_ops = []`. Both transforms are random-gated structural
combinators; numerics are delegated to the wrapped children.

- **`p == 0` on `RandomApply`**: input passes through.
- **`p == 1` on `RandomApply`**: always applies.
- **Empty `transforms` on `RandomApply` with `p == 1`**: identity (the
  for-loop trips zero times).
- **`RandomChoice` with one transform**: always selects that
  transform.
- **`RandomChoice` over an empty vec**: rejected at construction.
- **Send/Sync**: `assert_send_sync::<RandomApply<f32>>()` and
  `RandomChoice<f32>` pass — important because data loader workers
  hold these by reference across threads.

## Verification

Tests in `mod tests in random_apply.rs` (6 tests):

- `test_random_apply_always` at `random_apply.rs:91`
- `test_random_apply_never` at `random_apply.rs:109`
- `test_random_apply_empty_transforms` at `random_apply.rs:126`
- `test_random_choice_selects_one` at `random_apply.rs:136`
- `test_random_choice_single_transform` at `random_apply.rs:165`
- `test_random_apply_is_send_sync` at `random_apply.rs:178`

Smoke:

```bash
cargo test -p ferrotorch-vision --lib transforms::random_apply:: 2>&1 | tail -3
```

Expected: `6 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct RandomApply<T: Float>` with `transforms: Vec<Box<dyn Transform<T>>>` + `p: f64` at `ferrotorch-vision/src/transforms/random_apply.rs:13-16`, mirroring `torchvision/transforms/v2/_container.py:63` `class RandomApply`; non-test consumer: `pub use random_apply::{RandomApply, RandomChoice};` at `mod.rs:24` AND `RandomApply` in the crate-root re-export at `ferrotorch-vision/src/lib.rs:113`. |
| REQ-2 | SHIPPED | impl: `pub fn RandomApply::new(transforms, p) -> FerrotorchResult<Self>` with range check at `random_apply.rs:27-34`; non-test consumer: registered in `ferrotorch-vision/tests/conformance/_surface_inventory.toml:157` as `ferrotorch_vision::RandomApply::new`; reachable via the crate-root re-export. |
| REQ-3 | SHIPPED | impl: `impl<T: Float> Transform<T> for RandomApply<T>` with random gate + chained-apply at `random_apply.rs:37-48`; non-test consumer: any `Box<dyn Transform<T>>` slot accepts this — composes into nested `Compose` / `RandomApply` pipelines. |
| REQ-4 | SHIPPED | impl: `pub struct RandomChoice<T: Float>` with `transforms: Vec<Box<dyn Transform<T>>>` at `random_apply.rs:55-57`, mirroring `_container.py:119` `class RandomChoice`; non-test consumer: same `pub use` at `mod.rs:24` AND `RandomChoice` in the crate-root re-export at `lib.rs:113`. |
| REQ-5 | SHIPPED | impl: `pub fn RandomChoice::new(transforms) -> FerrotorchResult<Self>` with non-empty check at `random_apply.rs:65-72`; non-test consumer: registered in `ferrotorch-vision/tests/conformance/_surface_inventory.toml:171` as `ferrotorch_vision::RandomChoice::new`; reachable via the crate-root re-export. |
| REQ-6 | SHIPPED | impl: `impl<T: Float> Transform<T> for RandomChoice<T>` with uniform index sampling + `.min(n - 1)` clamp at `random_apply.rs:75-82`; non-test consumer: same `Box<dyn Transform<T>>` slot access via the crate-root re-export. |
| REQ-7 | NOT-STARTED | open prereq blocker #1517 — upstream's optional `p: list[float]` weight vector for `RandomChoice` from `torchvision/transforms/v2/_container.py:138-141` is not implemented; ferrotorch's `RandomChoice` is uniform-only. |
