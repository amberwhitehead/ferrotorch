# ferrotorch-vision — `models::segmentation` module root

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669 (working tree at /home/doll/pytorch)
baseline-torchvision: /home/doll/.local/lib/python3.13/site-packages/torchvision/models/segmentation/
upstream-paths:
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/segmentation/__init__.py
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/segmentation/deeplabv3.py
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/segmentation/fcn.py
  - /home/doll/.local/lib/python3.13/site-packages/torchvision/models/segmentation/lraspp.py
-->

## Summary

`ferrotorch-vision/src/models/segmentation/mod.rs` is the module-root re-export
hub for the three semantic-segmentation model families ferrotorch currently
mirrors from `torchvision.models.segmentation`: DeepLabV3 with ResNet-50 dilated
backbone, FCN with ResNet-50 dilated backbone, and LRASPP with
MobileNetV3-Large dilated backbone, plus the shared ASPP utility used by
DeepLabV3. The file itself contains only `pub mod` declarations and `pub use`
re-exports; the real implementations live in the per-model files
(`aspp.rs`, `deeplabv3.rs`, `fcn.rs`, `lraspp.rs`).

## Requirements

- REQ-1: Declare and publish four submodules `aspp`, `deeplabv3`, `fcn`,
  `lraspp` under `pub mod`. The submodule names mirror torchvision's
  `models/segmentation/{deeplabv3,fcn,lraspp}.py` plus the `ASPP` helper that
  torchvision keeps inside `deeplabv3.py` (split into its own file here for
  clarity).
- REQ-2: Re-export the public type and constructor surface of every model so
  callers can write `use ferrotorch_vision::models::segmentation::{Fcn,
  DeepLabV3, Lraspp, deeplabv3_resnet50, fcn_resnet50,
  lraspp_mobilenet_v3_large}` directly, matching the user-facing
  `torchvision.models.segmentation.{deeplabv3_resnet50, fcn_resnet50,
  lraspp_mobilenet_v3_large}` Python entry points.
- REQ-3: Re-export the per-model head types (`Aspp`, `DeepLabV3Head`, `FcnHead`,
  `LrasppHead`) and the dilated-ResNet wrapper `ResNet50Dilated` so downstream
  registries, tests, and probe scripts can construct heads independently of
  full-model entry points.

## Acceptance Criteria

- [x] AC-1: `cargo check -p ferrotorch-vision` resolves
  `ferrotorch_vision::models::segmentation::deeplabv3_resnet50` without an
  intermediate path import.
- [x] AC-2: `pub use segmentation::{Aspp, DeepLabV3, ..., fcn_resnet50, ...}`
  at `ferrotorch-vision/src/models/mod.rs` succeeds — the names declared by
  this file's `pub use` block exist.
- [x] AC-3: Registry entries `"deeplabv3_resnet50"`, `"fcn_resnet50"`, and
  `"lraspp_mobilenet_v3_large"` in
  `ferrotorch-vision/src/models/registry.rs` resolve their constructor
  closures through `super::segmentation::{deeplabv3_resnet50, fcn_resnet50,
  lraspp_mobilenet_v3_large}`.

## Architecture

`ferrotorch-vision/src/models/segmentation/mod.rs` is 25 lines total: a
module-level doc comment that names the three model families, four `pub mod`
declarations, and a single `pub use` block that re-exports each model's
visible type-and-constructor set.

The four submodules and their visible types:

- `aspp` — re-exports `Aspp` (the Atrous Spatial Pyramid Pooling utility used
  by DeepLabV3's head).
- `deeplabv3` — re-exports `DeepLabV3`, `DeepLabV3Head`, `ResNet50Dilated`,
  and the `deeplabv3_resnet50` constructor.
- `fcn` — re-exports `Fcn`, `FcnHead`, and the `fcn_resnet50` constructor.
- `lraspp` — re-exports `Lraspp`, `LrasppHead`, and the
  `lraspp_mobilenet_v3_large` constructor.

### Non-test production consumers

- `pub use segmentation::{Aspp, DeepLabV3, DeepLabV3Head, Fcn, FcnHead, Lraspp,
  LrasppHead, ResNet50Dilated, deeplabv3_resnet50, fcn_resnet50,
  lraspp_mobilenet_v3_large}` at `ferrotorch-vision/src/models/mod.rs:40-43`
  re-exports the entire surface one layer up.
- `default_registry()` in `ferrotorch-vision/src/models/registry.rs:313-340`
  invokes `super::segmentation::deeplabv3_resnet50::<f32>`,
  `super::segmentation::fcn_resnet50::<f32>`, and
  `super::segmentation::lraspp_mobilenet_v3_large::<f32>` to bind the three
  segmentation models into the global `REGISTRY`.

## Parity contract

`parity_ops = []`. This file declares no ops directly — it is a pure
re-export shim. The downstream files each carry their own parity contract
(none of the segmentation files own a parity op; they compose `Conv2d`,
`BatchNorm2d`, `Linear`, `cat`, `interpolate`, and the differentiable
activations from `ferrotorch-core` / `ferrotorch-nn`).

## Verification

This file is exercised transitively by every test in `aspp.rs`,
`deeplabv3.rs`, `fcn.rs`, and `lraspp.rs`; if the re-exports were missing
those tests would fail to compile.

Smoke command:

```bash
cargo test -p ferrotorch-vision --lib segmentation:: 2>&1 | tail -3
```

Expected: every segmentation test compiles via the re-exports and passes;
no `parity-sweep` ops to run.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: four `pub mod aspp; pub mod deeplabv3; pub mod fcn; pub mod lraspp;` lines at `ferrotorch-vision/src/models/segmentation/mod.rs:16-19`; non-test consumer: `pub use segmentation::{Aspp, ...}` at `ferrotorch-vision/src/models/mod.rs:40-43` resolves the submodule names. |
| REQ-2 | SHIPPED | impl: `pub use {aspp::Aspp; deeplabv3::{DeepLabV3, DeepLabV3Head, ResNet50Dilated, deeplabv3_resnet50}; fcn::{Fcn, FcnHead, fcn_resnet50}; lraspp::{Lraspp, LrasppHead, lraspp_mobilenet_v3_large}}` at `ferrotorch-vision/src/models/segmentation/mod.rs`; non-test consumer: `default_registry()` invokes `super::segmentation::deeplabv3_resnet50::<f32>` at `default_registry in ferrotorch-vision/src/models/registry.rs`, `super::segmentation::fcn_resnet50::<f32>` at `registry.rs`, `super::segmentation::lraspp_mobilenet_v3_large::<f32>` at `registry.rs`. |
| REQ-3 | SHIPPED | impl: same `pub use` block at `segmentation/mod.rs` re-exports `Aspp`, `DeepLabV3Head`, `FcnHead`, `LrasppHead`, `ResNet50Dilated`; non-test consumer: the `pub use segmentation::{... Aspp, DeepLabV3Head, FcnHead, LrasppHead, ResNet50Dilated ...}` re-export at `segmentation in ferrotorch-vision/src/models/mod.rs` makes them accessible at the crate-public path used by probe / diagnostic scripts. |
