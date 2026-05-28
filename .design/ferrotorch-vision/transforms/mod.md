# ferrotorch-vision — `transforms` module root

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (torchvision v0.26.0 site-packages, not in /home/doll/pytorch)
upstream-paths:
  - torchvision/transforms/v2/__init__.py
-->

## Summary

`ferrotorch-vision/src/transforms/mod.rs` is the module-root surface for
ferrotorch-vision's port of `torchvision.transforms.v2`. It re-exports the
18 transform implementations as a flat public API and defines the canonical
ImageNet normalization constants `IMAGENET_MEAN` and `IMAGENET_STD`. Mirrors
`torchvision.transforms.v2.__init__` which collects the same canonical
transform set into a flat namespace for end-user import.

## Requirements

- REQ-1: Declare 18 submodules — `center_crop`, `color_jitter`, `compose`,
  `elastic_transform`, `gaussian_noise`, `random_apply`, `random_crop`,
  `random_gaussian_blur`, `random_horizontal_flip`, `random_resized_crop`,
  `random_rotation`, `random_vertical_flip`, `resize`, `rng`, `to_tensor`,
  `trivial_augment_wide`, `vision_normalize` — covering the torchvision
  Transform set this crate ports.

- REQ-2: Flat re-export of every public transform type (`CenterCrop`,
  `ColorJitter`, `Compose`, `ElasticTransform`, `GaussianNoise`,
  `RandomApply`, `RandomChoice`, `RandomCrop`, `RandomGaussianBlur`,
  `RandomHorizontalFlip`, `RandomResizedCrop`, `RandomRotation`,
  `RandomVerticalFlip`, `Resize`, `VisionToTensor`, `TrivialAugmentWide`,
  `VisionNormalize`) plus the `vision_manual_seed` function, so callers
  can `use ferrotorch_vision::transforms::Foo` without naming the
  individual submodule. Mirrors `torchvision.transforms.v2`'s flat
  namespace.

- REQ-3: Two `pub const` arrays carrying the canonical ImageNet
  per-channel statistics — `IMAGENET_MEAN: [f64; 3] = [0.485, 0.456, 0.406]`
  and `IMAGENET_STD: [f64; 3] = [0.229, 0.224, 0.225]` — in RGB order. These
  are the values `torchvision.models` documents for use with `Normalize`
  when consuming ImageNet-pretrained classifier weights.

## Acceptance Criteria

- [x] AC-1: Each submodule path in REQ-1 exists as a sibling `.rs` file in
  `ferrotorch-vision/src/transforms/`.
- [x] AC-2: Each `pub use` line compiles (verified by `cargo check -p
  ferrotorch-vision`).
- [x] AC-3: `IMAGENET_MEAN` and `IMAGENET_STD` are accessible via
  `ferrotorch_vision::IMAGENET_MEAN` / `ferrotorch_vision::IMAGENET_STD`
  (re-exported through `ferrotorch-vision/src/lib.rs:113`).

## Architecture

### Submodule declarations (REQ-1)

`pub mod center_crop;` through `pub mod vision_normalize;` at `mod.rs:1-17`.
Each submodule contains one or two transform structs implementing the
`Transform<T: Float>` trait from `ferrotorch_data::Transform`. The split is
1-to-1 with `torchvision.transforms.v2`'s `_geometry.py`, `_color.py`,
`_misc.py`, `_container.py`, `_augment.py`, `_auto_augment.py`,
`_type_conversion.py` content — each upstream class becomes one file in
ferrotorch-vision since we don't have Python's monolithic class-per-file
convention.

### Flat re-exports (REQ-2)

`pub use center_crop::CenterCrop;` through
`pub use vision_normalize::VisionNormalize;` at `mod.rs`. Note the
`random_apply` re-export brings BOTH `RandomApply` and `RandomChoice` into
scope (they share a file because they share the runtime branching pattern).
The `vision_manual_seed` function (not a transform; a global PRNG seed
setter) is also re-exported at `mod.rs`.

### ImageNet constants (REQ-3)

```rust
pub const IMAGENET_MEAN: [f64; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f64; 3] = [0.229, 0.224, 0.225];
```

at `mod.rs`. The values are RGB-ordered, matching torchvision's
`torchvision/models/_meta.py` and the ResNet weight-loading documentation.
These are used as default arguments by `VisionNormalize::imagenet` in
`vision_normalize.rs`.

### Non-test production consumers

- `ferrotorch-vision/src/lib.rs:112-116` — flat re-export of the
  transform types AND constants AT THE CRATE ROOT, so external callers
  write `use ferrotorch_vision::ColorJitter` rather than
  `use ferrotorch_vision::transforms::ColorJitter`. This is the crate's
  public API surface; per goal.md S5 it is grandfathered.
- `vision_normalize::VisionNormalize::imagenet`
  (`vision_normalize.rs`) reads `IMAGENET_MEAN` / `IMAGENET_STD`
  to build the canonical ImageNet normalizer — production-side consumer
  of REQ-3.

## Parity contract

`parity_ops = []`. This module is a re-export hub; the parity contract
lives on each individual transform's design doc. The ImageNet constants
are checked for value equality against torchvision's published constants
in `_meta.py` (bit-equal to `0.485, 0.456, 0.406` and `0.229, 0.224, 0.225`
respectively).

## Verification

The module-root file has no tests of its own — all behavior is checked at
the submodule level. Compile-time verification is the contract:

```bash
cargo check -p ferrotorch-vision 2>&1 | tail -3
```

Expected: clean compile. Re-exports either resolve or the module fails to
build.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: 17 `pub mod` declarations at `ferrotorch-vision/src/transforms/mod.rs` mirroring the `torchvision.transforms.v2` flat namespace; non-test consumer: `ferrotorch-vision/src/lib.rs` `pub mod transforms;` exposes the module at the crate root. |
| REQ-2 | SHIPPED | impl: `pub use` re-exports at `ferrotorch-vision/src/transforms/mod.rs:19-35` for `CenterCrop`, `ColorJitter`, `Compose`, `ElasticTransform`, `GaussianNoise`, `{RandomApply, RandomChoice}`, `RandomCrop`, `RandomGaussianBlur`, `RandomHorizontalFlip`, `RandomResizedCrop`, `RandomRotation`, `RandomVerticalFlip`, `Resize`, `VisionToTensor`, `TrivialAugmentWide`, `VisionNormalize`, `vision_manual_seed`; non-test consumer: `ferrotorch-vision/src/lib.rs:112-116` re-exports these to the crate root for `use ferrotorch_vision::Foo` ergonomics. |
| REQ-3 | SHIPPED | impl: `pub const IMAGENET_MEAN: [f64; 3] = [0.485, 0.456, 0.406];` at `vision_normalize in ferrotorch-vision/src/transforms/mod.rs` and `pub const IMAGENET_STD: [f64; 3] = [0.229, 0.224, 0.225];` at `mod.rs`; non-test consumer: `VisionNormalize::imagenet` in `imagenet in vision_normalize.rs` reads both constants via `Self::new(IMAGENET_MEAN, IMAGENET_STD)`, AND `ferrotorch-vision/src/lib.rs` re-exports both at the crate root. |
