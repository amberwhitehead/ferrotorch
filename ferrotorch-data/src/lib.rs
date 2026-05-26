//! Data loading, batching, and transforms for ferrotorch.
//!
//! Mirrors PyTorch's `torch.utils.data` surface: [`Dataset`] / [`IterableDataset`]
//! traits, [`DataLoader`] with prefetch and multi-worker pipelines,
//! [`Sampler`] and `BatchSampler`, [`Transform`] and the standard
//! [`Compose`] / [`Normalize`] / [`RandomHorizontalFlip`] / [`RandomCrop`]
//! suite, and helper [`default_collate`] / [`default_collate_pair`]
//! collation functions.
//!
//! ## REQ status (per `.design/ferrotorch-data/lib.md`)
//!
//! Full evidence rows (impl + non-test production consumer + upstream
//! cites) live in the design doc; this synopsis is a one-line summary per
//! REQ.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 (lint baseline) | SHIPPED | `#![warn(clippy::all, clippy::pedantic)] + #![deny(rust_2018_idioms)] + #![allow(missing_docs, missing_debug_implementations, ...)]` block at top of `lib.rs` with per-item justification comments; consumer: every other `ferrotorch-data/src/*.rs` file compiles under this baseline (verified by `cargo clippy -p ferrotorch-data -- -D warnings`) |
//! | REQ-2 (six-submodule layout) | SHIPPED | `pub mod collate / dataloader / dataset / interop (cfg arrow) / sampler / transforms` in `lib.rs`; consumer: every `pub use <module>::...` below references these modules and `cargo check -p ferrotorch-data` PASSes |
//! | REQ-3 (collate re-export) | SHIPPED | `pub use collate::{default_collate, default_collate_pair};` in `lib.rs`; consumer: `ferrotorch/src/lib.rs` `pub use ferrotorch_data::*;` propagates to `ferrotorch::default_collate` |
//! | REQ-4 (dataloader re-export) | SHIPPED | `pub use dataloader::{BatchIter, CollatedIter, DataLoader, MultiWorkerIter, PrefetchIter, ToDevice, WorkerMode};` in `lib.rs`; consumer: meta-crate re-export `ferrotorch::DataLoader`; downstream model crates use the surface for training loops |
//! | REQ-5 (dataset re-export) | SHIPPED | `pub use dataset::{ChainDataset, ConcatDataset, Dataset, IterableDataset, MappedDataset, TensorDataset, VecDataset, WorkerInfo};` in `lib.rs`; consumer: `DataLoader<D: Dataset>` is generic over the re-exported trait; meta-crate `ferrotorch::Dataset` |
//! | REQ-6 (sampler re-export) | SHIPPED | `pub use sampler::{BatchSampler, DistributedSampler, RandomSampler, Sampler, SequentialSampler, WeightedRandomSampler, shuffle_with_seed};` in `lib.rs`; consumer: `fn DataLoader::build_indices in dataloader.rs` constructs `RandomSampler::new` / `SequentialSampler::new`; meta-crate re-export |
//! | REQ-7 (transforms re-export) | SHIPPED | `pub use transforms::{Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor, Transform, manual_seed};` in `lib.rs`; consumer: meta-crate `ferrotorch::Compose` / `ferrotorch::Normalize`; downstream augmentation pipelines compose `Compose::new(vec![Box::new(Normalize::new(...))])` |

// Lint baseline mirrors the workspace-standard pattern from
// `ferrotorch-core` / `-nn` / `-vision` / `-jit` / `-cubecl` / `-xpu`.
#![warn(clippy::all, clippy::pedantic)]
#![deny(rust_2018_idioms)]
// `missing_docs` and `missing_debug_implementations` are held at `allow`
// while the workspace-wide rustdoc / `Debug` pass is incremental
// (mirrors ferrotorch-core / -nn / -vision precedent — diverging
// unilaterally from a leaf crate would be architectural unilateralism).
// Core loader / dataset / iterator types do carry `Debug` impls;
// remaining holes are in auxiliary samplers and collate helpers.
#![allow(missing_docs, missing_debug_implementations)]
// Pedantic lints we explicitly accept across this crate. Each allow names
// a concrete reason — the alternative would be churn-for-zero-benefit or
// a worse API. Mirrors the ferrotorch-core / -nn / -vision baseline; add
// to this list only with a one-line justification.
#![allow(
    // The crate is laid out so submodule names (`dataloader::DataLoader`,
    // `dataset::Dataset`, `transforms::Transform`) match the public type
    // they export; renaming would force ergonomic breakage.
    clippy::module_name_repetitions,
    // `# Errors` / `# Panics` sections will be added during the
    // workspace-wide rustdoc pass tracked as a follow-up issue, not
    // gated on this lint baseline.
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    // `#[must_use]` on every getter / builder is churn for marginal
    // value; callers in this codebase already use the returned values.
    clippy::must_use_candidate,
    // Builder-style methods (`shuffle`, `drop_last`, `num_workers`, …)
    // already document their consume-and-return pattern; `#[must_use]`
    // is noise.
    clippy::return_self_not_must_use,
    // Multi-worker dispatcher and worker_loop have many tightly-coupled
    // parameters mirroring the iterator state; refactoring into a struct
    // is tracked as a separate (low-priority) follow-up.
    clippy::too_many_arguments,
    // Collation/loader paths use single-character names for indices and
    // dataset positions (i, j, p, c); requiring longer names hurts
    // readability of the iterator state machines.
    clippy::similar_names,
    // Doc comments here use technical prose with bracketed dimension
    // tags ([C, H, W]); pedantic doc-markdown warnings are too aggressive.
    clippy::doc_markdown,
    // Numeric casts between usize / i32 / f64 are pervasive in batch
    // index arithmetic, RNG state, and synthetic-data construction; the
    // explicit cast is more readable than alternatives. Mirrors the
    // ferrotorch-core baseline.
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    // Hex-encoded splitmix64 constants don't gain readability from
    // underscore separators; literal patterns mirror the canonical
    // upstream PRNG.
    clippy::unreadable_literal,
    // Sampler shuffle / dataset selection store small `usize` / `i32`
    // index slices; the stable `sort()` is fine and matches PyTorch's
    // sampler determinism. `sort_unstable()` is a separate optimization
    // pass tracked as a follow-up.
    clippy::stable_sort_primitive,
    // Float equality assertions in transforms tests check exact values
    // produced by deterministic transforms (flip permutations, identity
    // transforms); `abs_diff` would lose the regression-test specificity.
    clippy::float_cmp,
    // Builder methods on configs that take `&self` for ergonomic
    // chaining accept owned values for forward-compatibility with
    // wrapping types; rewriting to take `&` is a separate refactor.
    clippy::needless_pass_by_value,
    // The `worker_loop` lock-acquire / cv-wait pattern is clearer as a
    // `loop { match … }` than as `let-else`; the rewrite obscures the
    // wait-on-poison-then-extract sequence.
    clippy::manual_let_else,
    // Boolean-to-int conversion in batch-extra calculations is a
    // deliberate index-adjustment idiom that mirrors PyTorch's
    // `int(worker_id < remainder)`; the rewrite is less legible.
    clippy::bool_to_int_with_if,
    // `format!` with positional args is consistent with the
    // FerrotorchError construction sites elsewhere in the workspace; a
    // mass rewrite is a separate cosmetic pass.
    clippy::uninlined_format_args,
    // Manual `Debug` impls for iterator/loader types deliberately elide
    // closure / trait-object / channel fields with presence indicators
    // and bookkeeping summaries (see per-impl SAFETY comments). The
    // missing-fields lint is correct in 90% of cases but wrong here.
    clippy::missing_fields_in_debug,
)]

pub mod collate;
pub mod dataloader;
pub mod dataset;
#[cfg(feature = "arrow")]
pub mod interop;
pub mod sampler;
pub mod transforms;

pub use collate::{default_collate, default_collate_pair};
pub use dataloader::{
    BatchIter, CollatedIter, DataLoader, MultiWorkerIter, PrefetchIter, ToDevice, WorkerMode,
};
pub use dataset::{
    ChainDataset, ConcatDataset, Dataset, IterableDataset, MappedDataset, TensorDataset,
    VecDataset, WorkerInfo,
};
pub use sampler::{
    BatchSampler, DistributedSampler, RandomSampler, Sampler, SequentialSampler,
    WeightedRandomSampler, shuffle_with_seed,
};
pub use transforms::{
    Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor, Transform, manual_seed,
};
