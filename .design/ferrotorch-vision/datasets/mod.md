# ferrotorch-vision — datasets module root (`datasets/mod.rs`)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/  (torchvision companion: /home/doll/.local/lib/python3.13/site-packages/torchvision/datasets/__init__.py)
-->

## Summary

`ferrotorch-vision/src/datasets/mod.rs` is the module root for vision
dataset implementations. It declares the three concrete dataset
submodules (`cifar`, `folder`, `mnist`) and re-exports the public
type surface so users can write `use ferrotorch_vision::datasets::Mnist`
instead of `use ferrotorch_vision::datasets::mnist::Mnist`. Mirrors
the role of `torchvision/datasets/__init__.py` which gathers
`from .cifar import CIFAR10, CIFAR100`, `from .folder import
DatasetFolder, ImageFolder`, `from .mnist import MNIST, ...` into the
`torchvision.datasets` namespace.

## Requirements

- REQ-1: Declares `pub mod cifar; pub mod folder; pub mod mnist;` so
  each dataset family lives in its own translation unit. Mirrors the
  per-dataset file layout of `torchvision/datasets/*.py`.
- REQ-2: Re-exports the user-facing dataset types and sample structs
  from the three submodules: `Cifar10`, `Cifar100`, `CifarSample`
  from `cifar`; `DatasetFolder`, `FolderSample`, `IMG_EXTENSIONS`,
  `ImageFolder`, `ImageSample` from `folder`; `Mnist`, `MnistSample`,
  `Split` from `mnist`. Mirrors `torchvision/datasets/__init__.py`'s
  flat namespace.
- REQ-3: The `Split` enum is owned by `mnist.rs` and re-exported
  here. CIFAR and MNIST both consume the same `Split` (Train | Test);
  the type lives in `mnist.rs` because that was the first dataset
  added, and CIFAR re-imports it via `use super::mnist::Split` (per
  R-DEV-2 — the user-visible namespace is unified).

## Acceptance Criteria

- [x] AC-1: `pub mod cifar; pub mod folder; pub mod mnist;`
  declarations at the top of `datasets/mod.rs`.
- [x] AC-2: `pub use cifar::{Cifar10, Cifar100, CifarSample};` is
  present.
- [x] AC-3: `pub use folder::{DatasetFolder, FolderSample,
  IMG_EXTENSIONS, ImageFolder, ImageSample};` is present.
- [x] AC-4: `pub use mnist::{Mnist, MnistSample, Split};` is present.

## Architecture

`datasets/mod.rs` is a 12-line file — three `pub mod` declarations
and three `pub use` re-export blocks (`datasets/mod.rs`).

The structural decision here is: torchvision's `datasets` namespace
is flat (`torchvision.datasets.MNIST`, `torchvision.datasets.CIFAR10`,
`torchvision.datasets.ImageFolder` — no sub-namespacing by file).
ferrotorch matches that surface via re-exports: end users write
`use ferrotorch_vision::datasets::Mnist`, never
`use ferrotorch_vision::datasets::mnist::Mnist` (R-DEV-2 — Python
user-API ABI).

`Split` ownership is mildly unusual: it lives in `mnist.rs` because
MNIST was the first dataset added; CIFAR's `cifar.rs` reads
`use super::mnist::Split;` rather than duplicating the enum. The
re-export at `datasets/mod.rs` (`pub use mnist::{Mnist, MnistSample,
Split};`) makes `Split` a citizen of the `datasets` namespace —
internally it's a `mnist::Split` but externally it's the canonical
`ferrotorch_vision::datasets::Split` used by both CIFAR and MNIST.

### Non-test production consumers

- `ferrotorch-vision/src/lib.rs` — `pub use datasets::{Cifar10,
  Cifar100, CifarSample, Mnist, MnistSample, Split};` re-exports the
  surface at the crate root, reaching through THIS module's
  re-exports.
- `ferrotorch/examples/train_mnist.rs:22` — `use
  ferrotorch_vision::{Mnist, Split};` reaches transitively through
  `datasets in ferrotorch-vision/src/lib.rs` ← `datasets/mod.rs`.

## Parity contract

`parity_ops = []`. This module is purely a namespace organizer —
no numerical or behavioral contract beyond making the re-export
surface match `torchvision.datasets`'s flat layout.

Edge cases:
- **Symbol collision**: `Split` is exported from `mnist` only; CIFAR
  imports it via `super::mnist::Split`. A future `cifar::Split` would
  collide; the orchestrator must add a tracking issue if the CIFAR
  family ever needs a distinct split (e.g. CIFAR-10's "extra"
  subset). The current re-export is a single-symbol unambiguous
  binding.
- **`IMG_EXTENSIONS` constant**: re-exported alongside
  `ImageFolder` / `DatasetFolder` so users who want to extend the
  default extension list can write `let mut exts =
  IMG_EXTENSIONS.to_vec();` from `ferrotorch_vision::datasets::*`.

## Verification

This file has no inline tests — it's a 12-line declaration. The
re-exports are verified indirectly by:

- `ferrotorch-vision/tests/conformance_vision_datasets.rs:30` —
  `use ferrotorch_vision::datasets::{Cifar10, Cifar100, Mnist,
  Split};` compiles only if `datasets/mod.rs` correctly re-exports
  these symbols.
- `ferrotorch/examples/train_mnist.rs:22` — `use
  ferrotorch_vision::{Mnist, Split};` compiles only if the chain
  `datasets/mod.rs` → `datasets in ferrotorch-vision/src/lib.rs` is
  intact.

Smoke command (no parity ops):

```bash
cargo check -p ferrotorch-vision --lib 2>&1 | tail -3
```

If the re-exports are broken, `cargo check` fails with `unresolved
import`. The check is the verification.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub mod cifar; pub mod folder; pub mod mnist;` at `folder in ferrotorch-vision/src/datasets/mod.rs` per upstream torchvision per-file layout (`torchvision/datasets/cifar.py`, `folder.py`, `mnist.py`); non-test consumer: the conformance integration test `mnist in ferrotorch-vision/tests/conformance_vision_datasets.rs` reaches these modules through the re-exports, and the crate-root re-exports at `ferrotorch-vision/src/lib.rs` consume them as well. |
| REQ-2 | SHIPPED | impl: `pub use cifar::{Cifar10, Cifar100, CifarSample};` at `ferrotorch-vision/src/datasets/mod.rs`, `pub use folder::{DatasetFolder, FolderSample, IMG_EXTENSIONS, ImageFolder, ImageSample};` at `ferrotorch-vision/src/datasets/mod.rs`, and `pub use mnist::{Mnist, MnistSample, Split};` at `ferrotorch-vision/src/datasets/mod.rs` per upstream `torchvision/datasets/__init__.py`'s flat namespace; non-test consumer: `pub use datasets::{Cifar10, Cifar100, CifarSample, Mnist, MnistSample, Split};` at `ferrotorch-vision/src/lib.rs` reaches through these re-exports. |
| REQ-3 | SHIPPED | impl: `Split` is defined in `Split in ferrotorch-vision/src/datasets/mnist.rs` and re-exported at `ferrotorch-vision/src/datasets/mod.rs` via `pub use mnist::{Mnist, MnistSample, Split};`; cross-dataset consumer: `use super::mnist::Split;` at `ferrotorch-vision/src/datasets/cifar.rs` is the production consumer of the shared enum within the crate, and `use ferrotorch_vision::{Mnist, Split};` at `ferrotorch/examples/train_mnist.rs` is the cross-crate consumer. |
