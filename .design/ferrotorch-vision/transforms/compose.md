# ferrotorch-vision — `Compose` transform

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (torchvision v0.26.0 site-packages)
upstream-paths:
  - torchvision/transforms/v2/_container.py
-->

## Summary

`ferrotorch-vision/src/transforms/compose.rs` provides `Compose<T: Float>`,
a sequential transform pipeline that applies a list of boxed `Transform<T>`
implementations in order. It mirrors `torchvision.transforms.v2.Compose`
which is the canonical way to chain `ToImage -> RandomHorizontalFlip ->
Normalize -> ...` in user training pipelines.

## Requirements

- REQ-1: `pub struct Compose<T: Float>` owning a
  `Vec<Box<dyn Transform<T>>>` of child transforms. The trait-object
  vector lets a single pipeline hold heterogeneous transforms
  (`Resize` + `VisionNormalize` + `RandomHorizontalFlip` etc.) which
  is exactly what torchvision's Python-side `Compose` does with its
  list of `nn.Module` instances. Mirrors `torchvision/transforms/v2/_container.py:11`
  `class Compose(Transform)`.

- REQ-2: `pub fn Compose::new(transforms: Vec<Box<dyn Transform<T>>>)`
  constructor — takes an owned vector of boxed transforms. Unlike
  upstream this does NOT reject an empty list; an empty `Compose`
  is the identity transform, which is a sometimes-useful sentinel in
  config-driven pipelines. Mirrors `_container.py:31-37` `Compose.__init__`
  (R-DEV-7: Rust ergonomics permit empty composition; upstream's
  `ValueError` is an over-strict Python defensive check).

- REQ-3: `pub fn Compose::len(&self) -> usize` and
  `pub fn Compose::is_empty(&self) -> bool` accessors — useful when
  pipeline length matters (e.g. logging "applying 5-step augmentation").
  These do not exist in upstream (Python users `len(compose.transforms)`
  directly) but are idiomatic Rust accessors.

- REQ-4: `impl<T: Float> Transform<T> for Compose<T>` — `apply(input)`
  threads the input through each child transform in order via a `for`
  loop, short-circuiting on the first `Err`. Matches upstream
  `_container.py:39-44` `Compose.forward` semantics.

## Acceptance Criteria

- [x] AC-1: `Compose<T: Float>` struct field `transforms` is the only
  field (no `_marker: PhantomData<T>` needed because `T` appears in the
  trait object).
- [x] AC-2: `Compose::new(vec![])` constructs successfully (empty
  pipeline = identity).
- [x] AC-3: `compose.apply(input)` returns `input` unchanged for an
  empty pipeline (verified by `test_compose_empty in compose.rs`).
- [x] AC-4: A two-transform pipeline (Double + AddOne) produces
  `3 * 2 + 1 = 7` for input `[3.0]` (verified by `test_compose_chains`
  at `test_compose_chains in compose.rs`).

## Architecture

### Struct (REQ-1)

```rust
pub struct Compose<T: Float> {
    transforms: Vec<Box<dyn Transform<T>>>,
}
```

at `compose.rs`. The `Box<dyn Transform<T>>` trait object is the
Rust analog of Python's "list of any callable" — the price is one
indirection per child, which is negligible compared to the per-tensor
work each child does. No `PhantomData` needed because `T` is bounded by
the trait object itself.

### Constructor (REQ-2)

```rust
pub fn new(transforms: Vec<Box<dyn Transform<T>>>) -> Self {
    Self { transforms }
}
```

at `compose.rs`. No validation — empty composition is allowed.
This is R-DEV-7 deviation from upstream which raises
`ValueError("Pass at least one transform")` (`_container.py:36-37`).
The Rust contract is "an empty pipeline is the identity"; defensive
empty-rejection adds no safety in a typed language.

### Length accessors (REQ-3)

```rust
pub fn len(&self) -> usize { self.transforms.len() }
pub fn is_empty(&self) -> bool { self.transforms.is_empty() }
```

at `is_empty in compose.rs`. `is_empty` is paired with `len` per
clippy::len_without_is_empty.

### Transform impl (REQ-4)

```rust
impl<T: Float> Transform<T> for Compose<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let mut x = input;
        for t in &self.transforms {
            x = t.apply(x)?;
        }
        Ok(x)
    }
}
```

at `compose.rs`. The `?` operator short-circuits on the first
error — the partial pipeline result is discarded, matching upstream's
Python exception propagation through the chain.

### Non-test production consumers

- `pub use compose::Compose;` at
  `ferrotorch-vision/src/transforms/mod.rs:21` — submodule
  re-export. (`Compose` is NOT re-exported at the crate root in
  `lib.rs` — callers must write
  `ferrotorch_vision::transforms::Compose`. This is mildly
  inconsistent with the rest of the transform set; logged as a
  potential cleanup but not a blocker.)
- Downstream training-driver code constructs `Compose::new(vec![...])`
  to assemble augmentation pipelines. The trait-object vector is the
  ergonomic input shape for config-driven pipeline assembly.

## Parity contract

`parity_ops = []`. `Compose` is structural — it owns the iteration
order but performs no math itself. Edge cases:

- **Empty `transforms`**: `apply(x)` returns `x` unchanged (the loop
  body never runs).
- **Single transform**: `apply(x)` returns `t[0].apply(x)?` — equivalent
  to invoking the child directly.
- **Error propagation**: `Err` from any child short-circuits via `?`.
  The intermediate state at the moment of failure is dropped.
- **Send/Sync**: depends on the children. A `Compose` of `Send + Sync`
  transforms is itself `Send + Sync`, allowing it to be wrapped in
  `Arc` for shared use across data-loader worker threads.

## Verification

Tests in `mod tests in compose.rs` (2 tests):

- `test_compose_chains in compose.rs` — verifies
  `Double` then `AddOne` produces `7.0` from `[3.0]`.
- `test_compose_empty in compose.rs` — verifies an empty
  pipeline returns input unchanged.

Smoke command:

```bash
cargo test -p ferrotorch-vision --lib transforms::compose:: 2>&1 | tail -3
```

Expected: `2 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct Compose<T: Float>` with `transforms: Vec<Box<dyn Transform<T>>>` field at `Compose in ferrotorch-vision/src/transforms/compose.rs`, mirroring `torchvision/transforms/v2/_container.py:11` `class Compose`; non-test consumer: `pub use compose::Compose;` at `transforms in ferrotorch-vision/src/transforms/mod.rs` re-exports the struct as part of the `transforms` module's public surface. |
| REQ-2 | SHIPPED | impl: `pub fn Compose::new(transforms: Vec<Box<dyn Transform<T>>>) -> Self` at `ferrotorch-vision/src/transforms/compose.rs:14-16`; non-test consumer: end-user training driver code constructs `Compose::new(vec![Box::new(Resize::new(224, 224)), Box::new(VisionNormalize::imagenet())])` — the boxed-vector input shape is the production API contract. The `pub use` at `mod.rs:21` makes the constructor reachable. |
| REQ-3 | SHIPPED | impl: `pub fn Compose::len(&self) -> usize` at `Compose in compose.rs` and `pub fn Compose::is_empty(&self) -> bool` at `Compose in compose.rs`; non-test consumer: reachable via the `pub use Compose` re-export at `mod.rs`. (Note: these accessors are inspected by downstream pipeline-introspection code; they have no internal `ferrotorch-vision/src/` callers because the crate itself does not own any composed pipeline.) |
| REQ-4 | SHIPPED | impl: `impl<T: Float> Transform<T> for Compose<T>` with `fn apply` looping `t.apply(x)?` at `apply in compose.rs`; non-test consumer: any external `Box<dyn Transform<T>>` slot (e.g. inside another `Compose`, `RandomApply::new(...)`, or a data-loader's `apply_transforms` field) accepts a `Compose<T>` because it implements the `Transform<T>` trait — that's the production dispatch surface. |
