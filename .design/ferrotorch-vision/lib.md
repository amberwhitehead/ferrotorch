# ferrotorch-vision — crate root (`lib.rs`)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/  (torchvision is an out-of-tree companion; the upstream surface
             mirrored is /home/doll/.local/lib/python3.13/site-packages/torchvision/__init__.py)
-->

## Summary

`ferrotorch-vision/src/lib.rs` is the crate root for ferrotorch's
torchvision-shaped surface: it declares the five top-level modules
(`datasets`, `io`, `models`, `ops`, `transforms`) and re-exports the
type/function symbols that downstream code reaches through the meta-crate
glob `pub use ferrotorch_vision::*;` (in `ferrotorch/src/lib.rs`).
Mirrors the layout of `torchvision/__init__.py` which imports
`extension, datasets, io, models, ops, transforms, utils` and exposes
them as attributes of the `torchvision` package.

## Requirements

- REQ-1: Crate root declares the five submodules (`datasets`, `io`,
  `models`, `ops`, `transforms`) as `pub mod`. Mirrors the imports in
  `torchvision/__init__.py:8` (`from torchvision import _meta_registrations,
  datasets, io, models, ops, transforms, utils`).
- REQ-2: Top-level re-exports surface the dataset wrapper types,
  io functions/structs, and transform functions at crate root so
  users can write `use ferrotorch_vision::Mnist` without traversing
  `datasets::mnist::Mnist`. Mirrors torchvision's
  `torchvision.datasets.MNIST` short-form access.
- REQ-3: Crate-level lint configuration applies `#![warn(clippy::all,
  clippy::pedantic)]` and `#![deny(unsafe_code, rust_2018_idioms)]`. The
  `unsafe_code` deny enforces R-CODE-1 across the entire image-processing
  surface — vision kernels must not introduce raw FFI.
- REQ-4: Documented per-lint `#![allow(...)]` exceptions name each
  accepted lint with a concrete one-line rationale. The
  `module_name_repetitions` allow is required because vision types
  deliberately echo their parent module name (e.g. `RandomGaussianBlur`
  in `random_gaussian_blur`) to mirror torchvision's naming convention
  (R-DEV-2 — Python user-API ABI).
- REQ-5: `missing_debug_implementations` is allowed at the crate level
  while the crate-wide `Debug` derive sweep is tracked as a follow-up
  (matches the pattern set by ferrotorch-jit #677). Public sample/storage
  types (`RawImage`, `MnistSample`, `CifarSample`, `ImageSample`,
  `Cifar10`, `Cifar100`, `Mnist`) already derive `Debug` so end-user
  inspection works; only ~44 internal model blocks remain.

## Acceptance Criteria

- [x] AC-1: `pub mod datasets; pub mod io; pub mod models; pub mod ops;
  pub mod transforms;` lives at the top of `lib.rs`.
- [x] AC-2: `pub use datasets::{Cifar10, Cifar100, CifarSample, Mnist,
  MnistSample, Split};` and `pub use io::{RawImage,
  raw_image_to_tensor, read_image, read_image_as_tensor,
  read_image_rgba, tensor_to_raw_image, write_image,
  write_tensor_as_image};` re-exports are present.
- [x] AC-3: `#![deny(unsafe_code, rust_2018_idioms)]` is in force; the
  crate compiles with no `unsafe` blocks.
- [x] AC-4: Every `#![allow(...)]` lint name has an inline comment
  documenting the rationale.
- [x] AC-5: `pub use ferrotorch_vision::*;` in `ferrotorch/src/lib.rs`
  is the production consumer of the crate-root surface.

## Architecture

`ferrotorch-vision/src/lib.rs` mirrors `torchvision/__init__.py` directly:
the Python `__init__` imports the submodules and assigns
`_image_backend = "PIL"`. ferrotorch's image backend is fixed (the
`image` crate from the Rust ecosystem, R-DEV-7 — Rust analog of PIL),
so the `set_image_backend` / `get_image_backend` indirection is
collapsed.

The crate-level lint configuration (`lib.rs:8-91`) is load-bearing:
- `#![warn(clippy::all, clippy::pedantic)]` — pedantic-on-by-default
  matches the rest of the workspace.
- `#![deny(unsafe_code, rust_2018_idioms)]` — vision-side code must
  not introduce raw FFI; image decoding goes through the `image` crate
  which is itself unsafe-audited.
- The `#![allow(missing_debug_implementations)]` block at
  `lib.rs:11-17` is annotated with a justification pointing to the
  follow-up sweep (mirrors jit #677).
- The pedantic-lint exceptions at `lib.rs:21-87` each name a concrete
  reason — `module_name_repetitions` mirrors torchvision; the cast
  lints support pixel-index arithmetic; `too_many_lines` covers the
  long `match` blocks in `TrivialAugmentWide`-style transforms.

Re-export surface (`lib.rs:99-116`):
- `pub use datasets::{Cifar10, Cifar100, CifarSample, Mnist,
  MnistSample, Split};` — five-symbol short form for dataset users.
- `pub use io::{RawImage, raw_image_to_tensor, read_image,
  read_image_as_tensor, read_image_rgba, tensor_to_raw_image,
  write_image, write_tensor_as_image};` — eight-symbol image-IO
  short form.
- `pub use models::{...};` — 20+ model constructors / configs.
- `pub use transforms::{...};` — augmentation transforms.

### Non-test production consumers

- `ferrotorch/src/lib.rs:71` — `pub use ferrotorch_vision::*;` is the
  meta-crate re-export consumer that surfaces every vision symbol at
  the `ferrotorch::` namespace. Without `lib.rs`'s re-exports, the
  meta-crate users would have to write `ferrotorch::vision::Mnist`.
- `ferrotorch/examples/train_mnist.rs` — `use
  ferrotorch_vision::{Mnist, Split};` reaches through the
  re-exports authored in this file.
- `ferrotorch-vision/examples/inference_dump.rs:34` and
  `ferrotorch-vision/examples/probe_rpn_stages_1141.rs:38` —
  `use ferrotorch_vision::io::read_image_as_tensor;` is the
  module-path consumer (validates that `pub mod io` is reachable).
- `ferrotorch-hub/src/registry.rs:1194` — `ferrotorch_vision::models`
  is the consumer for the `pub mod models` declaration.

## Parity contract

`parity_ops = []`. `lib.rs` is the crate root — it declares modules
and re-exports symbols. No numerical contract applies directly here.

Behavioral edge cases the lint configuration enforces:
- `unsafe_code` is denied — any new `unsafe` block must be
  per-item-allowed with a `// SAFETY:` comment (R-CODE-1).
- `rust_2018_idioms` is denied — no `extern crate` declarations, no
  bare-`fn`-pointer types in trait objects.
- Each `clippy::*` allow has an inline rationale; new code that
  triggers a non-allowed pedantic lint must fix the lint, not append
  to the allow list silently.

## Verification

`lib.rs` itself has no inline tests — verification is the cargo
gauntlet:

```bash
cargo check -p ferrotorch-vision 2>&1 | tail -3
cargo clippy -p ferrotorch-vision --lib -- -D warnings 2>&1 | tail -3
cargo test -p ferrotorch-vision --lib 2>&1 | tail -3
```

The `cargo clippy --lib -- -D warnings` invocation transitively
exercises every `#![allow]` and `#![warn]` declared here. If a new
crate-wide lint regresses, it fails the gauntlet.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub mod datasets; pub mod io; pub mod models; pub mod ops; pub mod transforms;` at `ferrotorch-vision/src/lib.rs` per upstream `torchvision/__init__.py:8`; non-test consumer: `pub use ferrotorch_vision::*;` at `ferrotorch/src/lib.rs` re-exports the surface for meta-crate users, and `use ferrotorch_vision::io::read_image_as_tensor;` at `ferrotorch-vision/examples/inference_dump.rs` exercises the `pub mod io` path. |
| REQ-2 | SHIPPED | impl: `pub use datasets::{Cifar10, Cifar100, CifarSample, Mnist, MnistSample, Split};` at `ferrotorch-vision/src/lib.rs` and `pub use io::{RawImage, raw_image_to_tensor, read_image, read_image_as_tensor, read_image_rgba, tensor_to_raw_image, write_image, write_tensor_as_image};` at `ferrotorch-vision/src/lib.rs`; non-test consumer: `use ferrotorch_vision::{Mnist, Split};` at `ferrotorch/examples/train_mnist.rs` resolves through these re-exports. |
| REQ-3 | SHIPPED | impl: `#![warn(clippy::all, clippy::pedantic)]` at `ferrotorch-vision/src/lib.rs:8` and `#![deny(unsafe_code, rust_2018_idioms)]` at `ferrotorch-vision/src/lib.rs:9`; non-test consumer: the cargo clippy gauntlet (`cargo clippy -p ferrotorch-vision --lib -- -D warnings`) runs in CI and on every commit, enforcing R-CODE-1 across the crate. |
| REQ-4 | SHIPPED | impl: `#![allow(...)]` block at `ferrotorch-vision/src/lib.rs:21-87` with inline rationale per lint (e.g. `clippy::module_name_repetitions` annotated "Vision types deliberately echo their parent module name (e.g. `RandomGaussianBlur` in `random_gaussian_blur`) to mirror torchvision"); non-test consumer: cargo clippy gauntlet validates the lint set on every workspace clippy run. |
| REQ-5 | SHIPPED | impl: `#![allow(missing_debug_implementations)]` at `ferrotorch-vision/src/lib.rs` with the documented follow-up rationale at `lib.rs` ("the pub sample types `RawImage`, `MnistSample`, `CifarSample`, `ImageSample` and the dataset wrappers `Cifar10`, `Cifar100`, `Mnist` already derive Debug"); non-test consumer: every public sample/dataset type derives `Debug` (e.g. `#[derive(Debug)] struct Cifar10<T: Float>` at `Cifar10 in ferrotorch-vision/src/datasets/cifar.rs`, `#[derive(Debug, Clone)] struct RawImage` at `RawImage in ferrotorch-vision/src/io.rs`) so end-user inspection through the re-export surface works. |
