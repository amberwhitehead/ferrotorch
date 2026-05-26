# ferrotorch-data — Crate root

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/utils/data/__init__.py
-->

## Summary

`ferrotorch-data/src/lib.rs` is the crate root that mirrors the public
surface of PyTorch's `torch.utils.data` package. It establishes the
workspace-standard lint baseline, declares the six submodules
(`collate`, `dataloader`, `dataset`, `interop`, `sampler`,
`transforms`), and re-exports the public types so callers can write
`use ferrotorch_data::{DataLoader, Dataset, ...}` without knowing the
internal module layout. Mirrors the role of
`torch/utils/data/__init__.py:1-79` which collects sub-module imports
into a single `__all__` list.

## Requirements

- REQ-1: Workspace-standard lint baseline. `#![warn(clippy::all,
  clippy::pedantic)] + #![deny(rust_2018_idioms)]` at the crate root,
  with per-item `#![allow(missing_docs, missing_debug_implementations,
  ...)]` justifications matching the ferrotorch-core / -nn / -vision
  precedent. No module-root `#![allow]` to silence a non-pedantic lint
  (per R-CODE-3).

- REQ-2: Six-submodule layout. `pub mod collate; pub mod dataloader;
  pub mod dataset; #[cfg(feature = "arrow")] pub mod interop; pub mod
  sampler; pub mod transforms;` — mirrors PyTorch's split of
  `torch.utils.data` into `dataset.py`, `dataloader.py`, `sampler.py`,
  `distributed.py`, `_utils/collate.py`. The `interop` module is
  feature-gated because tabular interop is optional.

- REQ-3: Public re-export surface for collate. `pub use
  collate::{default_collate, default_collate_pair};` mirroring the
  `default_collate` entry in `torch/utils/data/__init__.py:65`.

- REQ-4: Public re-export surface for dataloader. `pub use
  dataloader::{BatchIter, CollatedIter, DataLoader, MultiWorkerIter,
  PrefetchIter, ToDevice, WorkerMode};` — `DataLoader` matches
  `torch/utils/data/__init__.py:3` and is the headline export, the
  other types are the iterator family and helper traits the surface
  needs.

- REQ-5: Public re-export surface for dataset. `pub use
  dataset::{ChainDataset, ConcatDataset, Dataset, IterableDataset,
  MappedDataset, TensorDataset, VecDataset, WorkerInfo};` — every
  one mirrors a `torch.utils.data` class except `MappedDataset`
  (Rust-side composition helper) and `VecDataset` (test helper that
  doubles as a working in-memory dataset).

- REQ-6: Public re-export surface for sampler. `pub use
  sampler::{BatchSampler, DistributedSampler, RandomSampler, Sampler,
  SequentialSampler, WeightedRandomSampler, shuffle_with_seed};` —
  mirrors `torch/utils/data/__init__.py:33-40` plus the shared
  shuffle primitive used internally.

- REQ-7: Public re-export surface for transforms. `pub use
  transforms::{Compose, Normalize, RandomCrop, RandomHorizontalFlip,
  ToTensor, Transform, manual_seed};` — mirrors the
  `torchvision.transforms` v1 API (which ferrotorch consolidates into
  ferrotorch-data rather than splitting into a separate vision crate
  for the small data-augmentation surface).

## Acceptance Criteria

- [x] AC-1: Crate root `#![warn(clippy::all, clippy::pedantic)]
  + #![deny(rust_2018_idioms)]` and per-item `#![allow(...)]` block
  is present in `lib.rs` with one-line justification comments.
- [x] AC-2: All six `pub mod ...` declarations are present, with
  `interop` correctly `#[cfg(feature = "arrow")]`-gated.
- [x] AC-3: All seven `pub use ...` groups exactly match the
  symbol list in REQ-3..7.
- [x] AC-4: `cargo check -p ferrotorch-data` and
  `cargo check -p ferrotorch-data --features arrow,polars` both
  compile clean (the interop module exists only when feature-gated
  on; downstream code expecting `ferrotorch_data::interop::*` must
  enable the feature).
- [x] AC-5: `cargo clippy -p ferrotorch-data -- -D warnings` is
  clean (workspace-standard).

## Architecture

### Lint baseline (REQ-1)

The crate root mirrors the canonical baseline from `ferrotorch-core`
/ `-nn` / `-vision` / `-jit` / `-cubecl` / `-xpu`: `clippy::all` and
`clippy::pedantic` at warn, `rust_2018_idioms` at deny, plus an
explicit per-item `#![allow(...)]` block for the pedantic lints we
collectively waive. Every allow carries a one-line justification
comment naming the concrete reason (e.g. `module_name_repetitions` is
allowed because `DataLoader` lives in `dataloader.rs`). Adding to the
allow list without a justification is not allowed — the doc-comment
above the block makes this contract explicit.

The `missing_docs` and `missing_debug_implementations` lints are at
`allow` for now because the crate is mid-pass on the workspace
rustdoc / `Debug` discipline; core loader / dataset / iterator types
DO carry `Debug` impls (see e.g. `impl Debug for DataLoader<D> in
dataloader.rs`, manually written because the trait-object fields are
not `Debug`-bound).

### Submodule layout (REQ-2)

```
ferrotorch-data/src/
├── lib.rs           ← crate root, re-exports
├── collate.rs       ← default_collate, default_collate_pair
├── dataset.rs       ← Dataset trait + impls (Vec, Tensor, Concat, Chain, Mapped, IterableDataset)
├── sampler.rs       ← Sampler trait + impls (Sequential, Random, Distributed, Weighted, Batch)
├── dataloader.rs    ← DataLoader + BatchIter family (Sync/Prefetch/MultiWorker) + CollatedIter + ToDevice
├── transforms.rs    ← Transform trait + Compose/Normalize/ToTensor/RandomHorizontalFlip/RandomCrop
└── interop.rs       ← Arrow / Polars conversions (feature-gated)
```

The split is deliberately one Rust file per PyTorch upstream file
group: collate ↔ `_utils/collate.py`, dataset ↔ `dataset.py`,
sampler ↔ `sampler.py + distributed.py`, dataloader ↔
`dataloader.py + _utils/worker.py`. Transforms come from
`torchvision.transforms.v1` (a separate package upstream) but live
in this crate because the surface is small.

### Re-export discipline (REQ-3..7)

All re-exports are flat at the crate root — `pub use collate::*` style
glob re-exports are explicitly avoided in favour of named re-exports so
the surface is auditable by `grep -n "^pub use" lib.rs`. The
`pub use collate::{default_collate, default_collate_pair};` pattern
matches PyTorch's `torch.utils.data.default_collate` user-facing
spelling (PyTorch uses `__all__` with named entries; we use named
`pub use`).

### Non-test production consumers

- `ferrotorch/src/lib.rs` (the meta-crate) — `pub use
  ferrotorch_data::*;` glob propagates the entire surface to the
  application-facing `ferrotorch::` namespace. This is the primary
  consumer.
- `ferrotorch-vision`, `ferrotorch-text`, `ferrotorch-audio`, and the
  28 model-specific crates (ferrotorch-llama, etc.) all depend on
  `ferrotorch-data` and call `DataLoader::new` / `default_collate`
  / etc. from this surface.

## Parity contract

`parity_ops = []` for this route. Data-loading is plumbing — the
parity-sweep runner exercises numerical ops in `ferrotorch-core`, not
the iterator state machines here. Edge cases preserved at the
crate-root level:

- **Feature-gate consistency**: `interop` requires `arrow` (which
  drags in Apache Arrow). Compiling without `--features arrow` must
  succeed; the `pub mod interop;` line is correctly cfg-gated so it
  doesn't surface a missing-module error.
- **`pub use` glob avoidance**: each public re-export is named, so the
  surface a `--cfg arrow` build adds is exactly the `interop::*`
  free functions, not implicit prelude inclusion.
- **Workspace lint baseline**: the per-item `#![allow]` block is
  identical across `ferrotorch-core` / `-nn` / `-vision` / `-data`,
  so adding ferrotorch-data to a workspace build does not introduce
  new lint warnings.

## Verification

```bash
cargo check -p ferrotorch-data 2>&1 | tail -3
cargo check -p ferrotorch-data --features arrow,polars 2>&1 | tail -3
cargo clippy -p ferrotorch-data --all-targets --all-features -- -D warnings 2>&1 | tail -3
cargo test -p ferrotorch-data 2>&1 | tail -3
```

Expected: all four invocations return `Finished` (or `test result: ok`)
with 0 warnings, 0 failures.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `#![warn(clippy::all, clippy::pedantic)] + #![deny(rust_2018_idioms)] + #![allow(missing_docs, missing_debug_implementations, ...)]` block at top of `lib.rs` with per-item justification comments; non-test consumer: every other `ferrotorch-data/src/*.rs` file compiles under this baseline (verified by `cargo clippy -p ferrotorch-data -- -D warnings` PASS). |
| REQ-2 | SHIPPED | impl: `pub mod collate; pub mod dataloader; pub mod dataset; #[cfg(feature = "arrow")] pub mod interop; pub mod sampler; pub mod transforms;` in `lib.rs`; non-test consumer: every `pub use <module>::...` line below references these modules and the workspace `cargo check -p ferrotorch-data` PASSes. |
| REQ-3 | SHIPPED | impl: `pub use collate::{default_collate, default_collate_pair};` in `lib.rs`; non-test consumer: `ferrotorch/src/lib.rs` `pub use ferrotorch_data::*;` propagates these symbols to `ferrotorch::default_collate`. |
| REQ-4 | SHIPPED | impl: `pub use dataloader::{BatchIter, CollatedIter, DataLoader, MultiWorkerIter, PrefetchIter, ToDevice, WorkerMode};` in `lib.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-exports `ferrotorch::DataLoader`; downstream training-loop code in the ferrotorch-llama / -bert / -whisper crates constructs `DataLoader::new(...)` via this surface. |
| REQ-5 | SHIPPED | impl: `pub use dataset::{ChainDataset, ConcatDataset, Dataset, IterableDataset, MappedDataset, TensorDataset, VecDataset, WorkerInfo};` in `lib.rs`; non-test consumer: `DataLoader::new<D: Dataset>(...)` in `dataloader.rs` is generic over the `Dataset` trait re-exported here; meta-crate re-export through `ferrotorch::Dataset`. |
| REQ-6 | SHIPPED | impl: `pub use sampler::{BatchSampler, DistributedSampler, RandomSampler, Sampler, SequentialSampler, WeightedRandomSampler, shuffle_with_seed};` in `lib.rs`; non-test consumer: `fn DataLoader::build_indices in dataloader.rs` constructs `RandomSampler::new` / `SequentialSampler::new` through this re-export, and `fn DataLoader::with_sampler in dataloader.rs` accepts `Box<dyn Sampler>` from this trait. |
| REQ-7 | SHIPPED | impl: `pub use transforms::{Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor, Transform, manual_seed};` in `lib.rs`; non-test consumer: `ferrotorch/src/lib.rs` re-exports `ferrotorch::Compose` / `ferrotorch::Normalize`; data-augmentation pipelines in downstream training loops construct `Compose::new(vec![Box::new(Normalize::new(...))])` through this surface. |
