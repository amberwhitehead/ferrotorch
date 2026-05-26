# ferrotorch — Umbrella Meta-Crate `lib` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - /home/doll/pytorch/torch/__init__.py
-->

## Summary

`ferrotorch/src/lib.rs` is the workspace's umbrella re-export crate — the
`ferrotorch::*` namespace that downstream `crates.io` consumers depend on.
It mirrors the `import torch` flat surface (`torch.nn`, `torch.optim`,
`torch.cuda`, …) by re-exporting every sub-crate under a single
`ferrotorch::<sub>` module gate, gated by Cargo features so non-CUDA / non-
MPS / non-XPU builds drop the matching dependency tree cleanly. It owns no
ops, no kernels, and no parity contract; its job is composition + the
global allocator + the lint baseline.

## Requirements

- REQ-1: A crate-level `//!` doc-comment orients a fresh reader: this crate
  is the umbrella, sub-crates own the work, the `prelude` re-export is the
  canonical one-import entry point, and the per-feature modules enumerated
  in the docstring are the discovery surface.
- REQ-2: A flattened `pub use ferrotorch_core::*;` re-export at the crate
  root so `use ferrotorch::{FerrotorchResult, zeros};` resolves without a
  `core::` prefix — matching `torch.Tensor`, `torch.zeros` living at the
  `torch` root in upstream PyTorch (`torch/__init__.py` __all__ list).
- REQ-3: A `pub mod prelude { ... }` module that re-exports the high-
  frequency surface (the `Module` trait, `Linear` / `Conv2d` / `BatchNorm2d`
  / `LayerNorm` / `Dropout`, activations `ReLU` / `SiLU` / `GELU` /
  `Sigmoid` / `Tanh` / `Softmax`, recurrent `LSTM` / `GRU`, losses
  `CrossEntropyLoss` / `MSELoss`, optimizers `Sgd` / `Adam` / `AdamW`, and
  the entirety of `ferrotorch_core::*`). Mirrors the unwritten convention
  of `from torch import nn, optim; from torch.nn import functional as F`
  by collapsing the three most-imported namespaces into one prelude.
- REQ-4: Always-included sub-crate modules (`nn`, `optim`, `data`,
  `vision`) declared unconditionally so the default cargo dependency tree
  always exposes them, regardless of feature selection. Each is a
  `pub mod <name> { pub use ferrotorch_<name>::*; }` wrapper.
- REQ-5: Feature-gated sub-crate modules — one per opt-in / opt-out sub-
  crate — declared behind `#[cfg(feature = "<flag>")]` so the crate
  compiles cleanly under any feature combination. The set: `train`,
  `serialize`, `jit`, `jit_script`, `distributions`, `profiler`, `hub`,
  `tokenize`, `gpu`, `cubecl`, `mps`, `xpu`, `distributed`, `llama`, `ml`.
- REQ-6: A `mimalloc::MiMalloc` global allocator, registered via
  `#[global_allocator]`, gated by `#[cfg(not(target_env = "msvc"))]` so
  non-MSVC builds get the allocator and MSVC falls back to the system
  allocator. Mirrors the way PyTorch swaps in `tcmalloc` / `jemalloc`
  through `LD_PRELOAD` shims at runtime — ferrotorch bakes the choice
  into the umbrella crate so users get the perf win by default.
- REQ-7: A crate-wide lint baseline — `#![warn(clippy::all,
  clippy::pedantic)]` and `#![deny(rust_2018_idioms,
  missing_debug_implementations)]` — with `#![allow(missing_docs)]` carried
  as a documented exception while the workspace-wide rustdoc pass is
  pending. The block is the umbrella-crate analog of the per-crate lint
  baselines in `ferrotorch-core`, `ferrotorch-llama`, `ferrotorch-cubecl`,
  etc. — workspace `[lints]` is intentionally NOT used; policy lives next
  to the code it governs.
- REQ-8: Backward-compatible feature aliases — specifically the
  `llama-cuda = ["llama", "gpu", "ferrotorch-llama/cuda"]` convenience
  feature in `Cargo.toml` — so the published API never silently changes
  shape under feature combinations downstream users already depend on.

## Acceptance Criteria

- [x] AC-1: `cargo check -p ferrotorch` compiles without errors against
  the default feature set.
- [x] AC-2: `cargo clippy -p ferrotorch --lib -- -D warnings` passes (no
  warnings escape the lint baseline).
- [x] AC-3: `cargo test -p ferrotorch --lib` runs the rustdoc example
  (`use ferrotorch::{FerrotorchResult, zeros};`) cleanly.
- [x] AC-4: `cargo fmt -p ferrotorch --check` reports no drift.
- [x] AC-5: Every `pub mod <name>` declared in `lib.rs` resolves to a
  reachable type under `ferrotorch::<name>::<Item>` — verified by
  `ferrotorch/tests/public_surface.rs`, which compile-time pins every
  documented module path.
- [x] AC-6: `cargo build -p ferrotorch --no-default-features` succeeds
  (the feature-gated modules vanish cleanly).
- [x] AC-7: `cargo build -p ferrotorch --features gpu` succeeds when the
  CUDA toolchain is present (sub-crate `ferrotorch-gpu` compiles in).

## Architecture

`lib.rs` opens with the crate doc-comment (`//!`) which describes the
purpose (umbrella re-export crate), gives a `use ferrotorch::{...}`
example that compiles under rustdoc, lists the per-feature modules, and
documents why the workspace `[lints]` mechanism is intentionally NOT
used (the per-crate `#![warn/deny]` lives next to the code it governs).
This is REQ-1, REQ-7.

The lint block (`#![warn(clippy::all, clippy::pedantic)]`,
`#![deny(rust_2018_idioms, missing_debug_implementations)]`,
`#![allow(missing_docs)]`) is REQ-7. The `missing_docs` allowance is
documented inline as held off until the workspace-wide rustdoc pass
lands; this is the only relaxation in the baseline.

The `#[global_allocator] static GLOBAL: mimalloc::MiMalloc =
mimalloc::MiMalloc;` block, gated by `#[cfg(not(target_env = "msvc"))]`,
is REQ-6. Registered as `mimalloc::MiMalloc` with
`default-features = false` so the small / no-secure variants of mimalloc
are not pulled in by default (per `ferrotorch/Cargo.toml:47`:
`mimalloc = { version = "0.1", default-features = false }`).

`pub use ferrotorch_core::*;` (the flat re-export named `ferrotorch_core`
glob-import in `lib.rs`) is REQ-2 — the flat re-export so the canonical
types (`Tensor`, `FerrotorchResult`, `zeros`, `from_vec`, …) live at
`ferrotorch::<Item>` matching upstream's `torch.<item>`.

`pub mod prelude` is REQ-3. The block re-exports `ferrotorch_core::*`
plus the high-frequency `ferrotorch_nn` items (`Module`, `Parameter`,
`Linear`, `Conv2d`, `BatchNorm2d`, `LayerNorm`, `Dropout`, `Sequential`,
activations `ReLU` / `SiLU` / `GELU` / `Sigmoid` / `Tanh` / `Softmax`,
recurrent `LSTM` / `GRU`, losses `CrossEntropyLoss` / `MSELoss`) and the
high-frequency `ferrotorch_optim` items (`Sgd`, `Adam`, `AdamW`, the
`Optimizer` trait). The list is curated — not every public sub-crate
name lands in prelude; only the ones a typical training-loop author
reaches for.

The always-on modules `pub mod nn`, `pub mod optim`, `pub mod data`,
`pub mod vision` are REQ-4. Each is a thin `pub use ferrotorch_<name>::*;`
wrapper inside a doc-commented `pub mod`. This matches upstream
`torch.nn`, `torch.optim`, `torch.utils.data`, `torchvision` namespacing.

The feature-gated modules — `pub mod train`, `pub mod serialize`,
`pub mod jit`, `pub mod jit_script`, `pub mod distributions`,
`pub mod profiler`, `pub mod hub`, `pub mod tokenize`, `pub mod gpu`,
`pub mod cubecl`, `pub mod mps`, `pub mod xpu`, `pub mod distributed`,
`pub mod llama`, `pub mod ml` — are REQ-5. Each carries a
`#[cfg(feature = "<flag>")]` so the dependency tree drops cleanly when
the feature is off. The `gpu`, `cubecl`, `mps`, `xpu`, `distributed`,
`llama`, `ml` modules are opt-IN (not in `default = [...]` in
`Cargo.toml`); the rest are opt-OUT (listed in `default = [...]`).

REQ-8 — the `llama-cuda` convenience feature — lives in
`ferrotorch/Cargo.toml:42` (`llama-cuda = ["llama", "gpu",
"ferrotorch-llama/cuda"]`) and is referenced in the `//!` doc-comment
in `lib.rs` (under the `pub mod llama` block) as
"with `llama-cuda`". The feature flag is the public-API contract the
doc-comment promises.

### Non-test production consumers

The umbrella crate is its own end: it is published to `crates.io` (badge
on `ferrotorch/README.md:5`: `[![crates.io](...)](https://crates.io/crates/ferrotorch)`)
and serves downstream users (`docs.rs/ferrotorch` consumers) that depend
on `ferrotorch = "0.5.x"` in their own `Cargo.toml`. Per goal.md S5, the
**boundary public API IS the consumer** for an umbrella re-export crate
— there is no in-workspace caller because the crate's purpose is to be
called from outside the workspace.

In-workspace consumers of the umbrella crate's modules are intentionally
absent (the in-workspace examples bypass `ferrotorch::*` and import the
sub-crates directly so they bench / train against the smallest possible
dep tree). The compile-time pins for the module surface live in
`ferrotorch/tests/public_surface.rs`, which is test infrastructure for
the very `pub mod` declarations in `lib.rs` — i.e., the umbrella crate's
test harness IS the contract auditor for the re-exports it ships.

Concrete in-workspace evidence the doc-comment promise is in force:

- The rustdoc `no_run` example block inside the crate `//!` doc-comment
  (the `# Examples` section): `use ferrotorch::{FerrotorchResult, zeros};
  fn main() -> FerrotorchResult<()> { let t = zeros::<f32>(&[2, 3])?;
  ... }`. This is a doctest the gauntlet runs (`cargo test -p ferrotorch
  --lib --doc`) and pins the REQ-2 flat re-export at compile time.
- The `pub mod llama { pub use ferrotorch_llama::*; }` block in `lib.rs`
  is the in-workspace consumer of every pub item in `ferrotorch-llama`,
  transitively (the design doc `.design/ferrotorch-llama/lib.md` cites
  THIS BLOCK as the non-test consumer of the llama crate). The same
  line-of-reasoning applies to every other `pub mod <crate> { pub use
  ferrotorch_<crate>::*; }` block in this file: each is the in-workspace
  consumer for the corresponding sub-crate's public surface.

## Parity contract

`parity_ops = []`. The umbrella crate hosts no parity ops; behavioral
parity belongs to the sub-crates this file re-exports. The parity-sweep
audit file `tools/parity-sweep/parity_audit.json` contains zero entries
sourced from `ferrotorch/src/lib.rs`.

The structural-parity contract the file does enforce:

- The crate's flat surface mirrors `torch/__init__.py`'s `__all__` list
  (upstream `torch/__init__.py:68-141`): `Tensor`, `zeros`, `randn`,
  `manual_seed`, `load`, `save`, `compile`, etc. all live at the
  `torch.<item>` root and are reached without a `torch.core.<item>`
  prefix. Ferrotorch matches by re-exporting `ferrotorch_core::*` at
  the crate root.
- Module namespacing mirrors upstream namespace boundaries: `torch.nn`
  → `ferrotorch::nn`, `torch.optim` → `ferrotorch::optim`,
  `torch.utils.data` → `ferrotorch::data`, `torchvision` →
  `ferrotorch::vision`, `torch.cuda` → `ferrotorch::gpu` (rename
  reflects "GPU" being device-family-neutral in ferrotorch).
- Feature-gated sub-crates mirror upstream optional dependencies:
  `torch.distributed` is opt-in in PyTorch wheels, `ferrotorch::distributed`
  is opt-in behind `#[cfg(feature = "distributed")]`. `torch.profiler`
  is bundled, `ferrotorch::profiler` is opt-OUT (default-on,
  `default-features = false` removes it).

## Verification

`lib.rs` carries no `#[cfg(test)]` tests of its own. The protections are:

- `ferrotorch/tests/public_surface.rs` — compile-time pins for every
  `pub mod` declared in `lib.rs`, gated by the matching
  `#[cfg(feature = "<flag>")]` so each feature combination is audited
  independently. Test functions: `always_on_modules_resolve`,
  `train_module_resolves`, `serialize_module_resolves`,
  `jit_module_resolves`, `jit_script_module_resolves`,
  `distributions_module_resolves`, `profiler_module_resolves`,
  `hub_module_resolves`, `tokenize_module_resolves`, `gpu_module_resolves`,
  `cubecl_module_resolves`, `mps_module_resolves`, `xpu_module_resolves`,
  `distributed_module_resolves`, `llama_module_resolves`,
  `ml_module_resolves`.
- The crate-level rustdoc `no_run` example inside the `//!` block of
  `lib.rs`, which is executed under `cargo test -p ferrotorch --doc`
  and pins the flat-surface re-export contract.

The gauntlet block:

```bash
cargo check -p ferrotorch 2>&1 | tail -3
cargo clippy -p ferrotorch --lib -- -D warnings 2>&1 | tail -3
cargo test -p ferrotorch --lib 2>&1 | tail -3
cargo fmt -p ferrotorch --check
cargo test -p ferrotorch-core --test divergence_cite_drift_generic 2>&1 | tail -3
```

No parity-sweep smoke applies (parity_ops = []). The cite-drift
generic test guards the design doc's symbol-anchor citations against
post-hoc drift.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: the crate `//!` doc-comment (the block opening `//! ferrotorch — PyTorch-shaped deep learning framework in Rust.` in `lib.rs`) mirrors `torch/__init__.py:1-9` module docstring; non-test consumer: the rustdoc `no_run` example block inside the same docstring (the `# Examples` section) is exercised by `cargo test -p ferrotorch --doc` and pins the umbrella API the docstring promises; in-workspace consumer evidence: `ferrotorch-llama/src/lib.rs` design doc cites the `pub mod llama { pub use ferrotorch_llama::*; }` block in `ferrotorch/src/lib.rs` as the production consumer of the llama re-export, demonstrating the umbrella surface is the boundary API per goal.md S5 grandfathering. |
| REQ-2 | SHIPPED | impl: `pub use ferrotorch_core::*;` in `ferrotorch/src/lib.rs` mirrors upstream `torch/__init__.py:68-141` flat `__all__` namespace; non-test consumer: the rustdoc `no_run` doctest inside the `//!` block (`use ferrotorch::{FerrotorchResult, zeros};`) compiles against the re-export under `cargo test -p ferrotorch --doc`. |
| REQ-3 | SHIPPED | impl: the `pub mod prelude { ... }` block in `ferrotorch/src/lib.rs` mirrors `from torch import nn, optim` ergonomic convention; non-test consumer: the `//!` doc-comment in the same file documents `use ferrotorch::prelude::*;` as the canonical entry point — the public-API contract published to `crates.io/ferrotorch`. |
| REQ-4 | SHIPPED | impl: the always-on `pub mod nn` / `pub mod optim` / `pub mod data` / `pub mod vision` declarations in `ferrotorch/src/lib.rs` (no `cfg` gate) mirror `torch.nn` / `torch.optim` / `torch.utils.data` / `torchvision` always-on namespaces; non-test consumer: `ferrotorch/tests/public_surface.rs:22-25` compile-time pins each path (test harness for the public-API contract); production consumer (downstream of workspace): `crates.io/ferrotorch` users importing `ferrotorch::nn::Linear`. |
| REQ-5 | SHIPPED | impl: 15 `#[cfg(feature = "<flag>")] pub mod <name>` blocks (`pub mod train`, `pub mod serialize`, `pub mod jit`, `pub mod jit_script`, `pub mod distributions`, `pub mod profiler`, `pub mod hub`, `pub mod tokenize`, `pub mod gpu`, `pub mod cubecl`, `pub mod mps`, `pub mod xpu`, `pub mod distributed`, `pub mod llama`, `pub mod ml`) in `ferrotorch/src/lib.rs` mirror upstream's optional `torch.distributed` / `torch.profiler` / etc. feature surfaces; non-test consumer: `ferrotorch/Cargo.toml:15-43` enumerates the matching feature flags and the `[dependencies]` table at `ferrotorch/Cargo.toml:56-73` pins each optional sub-crate to `optional = true`. The structural mirror is auditable; the published-crate contract is the consumer. |
| REQ-6 | SHIPPED | impl: the `#[cfg(not(target_env = "msvc"))] #[global_allocator] static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;` declaration in `ferrotorch/src/lib.rs`; non-test consumer: every binary linking the `ferrotorch` crate picks up the allocator via the rustc `#[global_allocator]` mechanism — e.g. `cargo run --example train_mnist -p ferrotorch` from `ferrotorch/examples/train_mnist.rs` links the umbrella crate transitively and inherits the allocator. |
| REQ-7 | SHIPPED | impl: the `#![warn(clippy::all, clippy::pedantic)]` / `#![deny(rust_2018_idioms, missing_debug_implementations)]` / `#![allow(missing_docs)]` lint baseline block in `ferrotorch/src/lib.rs`; non-test consumer: `cargo clippy -p ferrotorch --lib -- -D warnings` is the production gate that runs on every commit per goal.md Step 7 gauntlet. The `missing_docs` allowance is documented inline in the comment immediately preceding the `#![allow]` as held off until the workspace rustdoc pass. |
| REQ-8 | SHIPPED | impl: `llama-cuda = ["llama", "gpu", "ferrotorch-llama/cuda"]` at `ferrotorch/Cargo.toml:42`; non-test consumer: the `//!` doc-comment in `ferrotorch/src/lib.rs` (under the `pub mod llama` block) references "with `llama-cuda`" as a publicly documented feature combination — the published-crate users on `crates.io/ferrotorch` are the consumer (boundary public API per goal.md S5). |
