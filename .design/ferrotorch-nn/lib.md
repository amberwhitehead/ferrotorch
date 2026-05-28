# ferrotorch-nn — crate root (`lib.rs`)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/__init__.py
  - torch/nn/modules/__init__.py
-->

## Summary

`ferrotorch-nn/src/lib.rs` is the crate root for the neural-network module
layer. It declares every submodule, re-exports the canonical public surface
(layer types, `Module` trait, `Parameter`, `Buffer`, container types,
gradient-clipping helpers, the `Module` derive macro), and provides a
`prelude` module that mirrors the ergonomics of `from torch import nn` in
upstream PyTorch. The file owns the lint baseline for the crate (the
documented `#![allow(...)]` block).

## Requirements

- REQ-1: Crate-wide lint baseline matches the workspace-standard pattern
  (`warn(clippy::all, clippy::pedantic)`, `deny(rust_2018_idioms)`,
  per-item-justified pedantic `allow`s only) so the `nn` crate keeps the
  same bar as `ferrotorch-core`/`-distributed`/`-jit`/`-cubecl`/`-xpu`.
  Mirrors upstream PyTorch's convention of a single style/lint baseline
  applied at the `torch/nn/__init__.py` import edge.
- REQ-2: Every layer module under `ferrotorch-nn/src/` is declared as a
  `pub mod`, making the per-module file the canonical home for its
  contents. Mirrors `torch/nn/modules/__init__.py:1-200` which imports
  every concrete `nn.<Module>` from `torch/nn/modules/*.py`.
- REQ-3: The crate publishes a flat re-export surface
  (`pub use module::{Module, Reduction, StateDict}` and ~50 more
  `pub use` lines) so callers can write `use ferrotorch_nn::Linear`
  instead of `use ferrotorch_nn::linear::Linear`. Mirrors
  `torch/nn/__init__.py:11-50` which star-imports the concrete modules
  from `torch.nn.modules`.
- REQ-4: The crate re-exports the `Module` derive macro from
  `ferrotorch-nn-derive` under the same name as the `Module` trait
  (both resolve in the user's namespace: the trait via
  `use ferrotorch_nn::Module`, the macro via `#[derive(Module)]`).
  This is the Rust-ecosystem analog (R-DEV-7) of PyTorch's `nn.Module`
  base class — Rust uses a trait + derive macro rather than runtime
  metaclass introspection.
- REQ-5: A `prelude` submodule re-exports the high-leverage subset
  (`Module`, `Parameter`, `Buffer`, `StateDict`, `Reduction`, the
  derive macro under the name `DeriveModule`, the canonical layers
  `Linear`/`Conv1d`/`Conv2d`/`Conv3d`/`Dropout`/`Embedding`/
  `LayerNorm`/`BatchNorm1d`/`BatchNorm2d`/`GroupNorm`/`RMSNorm`/
  `MaxPool2d`/`AdaptiveAvgPool2d`/`GELU`/`ReLU`/`Sigmoid`/`Softmax`/
  `Tanh`/`Sequential`/`ModuleList`/`ModuleDict`, the canonical losses
  `MSELoss`/`L1Loss`/`CrossEntropyLoss`/`NLLLoss`/`BCELoss`/
  `BCEWithLogitsLoss`, and the gradient-clipping helpers
  `clip_grad_norm_`/`clip_grad_value_`) so callers can write
  `use ferrotorch_nn::prelude::*` and have the standard model-building
  surface in scope. Mirrors `from torch import nn` ergonomics.
- REQ-6: The `extern crate self as ferrotorch_nn` declaration is
  load-bearing for the derive macro's hygienic path resolution
  (`::ferrotorch_nn::Module` etc) when the macro is invoked from
  within the `ferrotorch-nn` crate's own integration tests. Mirrors
  the standard pattern for proc-macro–consuming crates that also
  host their own usages.

## Acceptance Criteria

- [x] AC-1: `#![warn(clippy::all, clippy::pedantic)]` and
  `#![deny(rust_2018_idioms)]` at the top of `lib.rs`.
- [x] AC-2: Every per-layer file (`activation.rs`, `attention.rs`, …,
  `utils.rs`) is declared as a `pub mod` in `lib.rs`.
- [x] AC-3: Flat `pub use` re-exports cover every type the prelude
  re-exports and the standard layer surface.
- [x] AC-4: `pub use ferrotorch_nn_derive::Module` re-exports the
  derive macro under the name `Module` (resolves alongside the trait).
- [x] AC-5: `pub mod prelude` exists and contains the high-leverage
  subset described in REQ-5.
- [x] AC-6: `extern crate self as ferrotorch_nn` is present (guarded
  by `#[allow(unused_extern_crates)]`).

## Architecture

### Lint baseline (REQ-1)

The file opens with `#![warn(clippy::all, clippy::pedantic)]` followed
by `#![deny(rust_2018_idioms)]`. The pedantic-allow block lists every
lint suppressed across the crate, each with a one-line rationale that
maps to either (a) an upstream-PyTorch parity choice (e.g.
`clippy::module_name_repetitions` because submodule files mirror their
type's name, matching the `torch.nn.modules.linear.Linear` pattern),
(b) a math-heavy code idiom that the lint mis-fires on (e.g.
`clippy::cast_*` for tensor-shape ↔ float casts in numeric kernels),
or (c) a churn-vs-benefit judgment (e.g. `clippy::uninlined_format_args`
deferred to a workspace-wide pass). `missing_docs` and
`missing_debug_implementations` are intentionally held at `allow` while
the workspace-wide rustdoc / `Debug` pass is tracked separately —
matches the `ferrotorch-core`/`-gpu`/`-distributed` precedent.

### Module declarations (REQ-2)

`lib.rs` declares 31 `pub mod` entries: `activation`, `attention`,
`buffer`, `container`, `conv`, `dropout`, `embedding`,
`flash_attention`, `flex_attention`, `functional`, `hooks`, `identity`,
`init`, `lazy_conv`, `lazy_conv_transpose`, `lazy_linear`, `lazy_norm`,
`linear`, `lora`, `loss`, `module`, `norm`, `padding`,
`paged_attention`, `parameter`, `parameter_container`, `pooling`,
`qat`, `rnn`, `rnn_utils`, `se`, `transformer`, `upsample`, `utils`.
Each is a separate file under `ferrotorch-nn/src/`. The 1:1 file ↔
module mapping mirrors PyTorch's `torch/nn/modules/*.py` layout
(`torch/nn/modules/__init__.py:1-100`).

### Re-export surface (REQ-3)

After the module declarations, `lib.rs` publishes a flat re-export
list (the `pub use activation::{...}`, `pub use attention::{...}`,
etc. block). Each layer file's pub types are surfaced here so callers
can write `use ferrotorch_nn::Linear` directly. The pattern mirrors
`torch/nn/__init__.py:11-50` which star-imports the concrete modules.

### Derive macro re-export (REQ-4)

`pub use ferrotorch_nn_derive::Module` republishes the derive macro
under the name `Module`. The macro and the trait live in different
namespaces (macro vs type), so both `use ferrotorch_nn::Module` (the
trait) and `#[derive(Module)]` (the macro) resolve simultaneously
without conflict. This is the Rust analog (R-DEV-7) of PyTorch's
`nn.Module` metaclass — Rust achieves the same registration ergonomic
through a procedural macro rather than runtime attribute walking.

### Prelude (REQ-5)

`pub mod prelude` collects the core abstractions, standard layers,
canonical losses, and gradient-clipping helpers. The selection mirrors
`from torch import nn`'s actual surface: enough to declare and train
a model without hunting through submodules, but not so wide that the
glob-import becomes lossy. Note the derive macro is re-exported under
the name `DeriveModule` inside the prelude to avoid colliding with
the `Module` trait in the same glob — callers do
`use ferrotorch_nn::prelude::*; #[derive(DeriveModule)]`.

### `extern crate self` (REQ-6)

`#[allow(unused_extern_crates)] extern crate self as ferrotorch_nn;`
binds the crate's own name to its root so the derive macro's
hygienic paths (`::ferrotorch_nn::Module`, etc.) resolve when the
macro is invoked from inside the crate's integration tests. Without
this binding, the macro's expansion would fail to find the trait at
the absolute path it generates. The `#[allow]` is necessary because
the binding looks unused to the compiler — the macro-expansion site
references it indirectly.

### Non-test production consumers

`lib.rs`'s consumers are external crates: `ferrotorch-optim`,
`ferrotorch-train`, `ferrotorch-vision`, `ferrotorch-llama`,
`ferrotorch-bert`, every downstream model crate. Concrete sites:

- `ferrotorch-optim/src/optimizer.rs` line 5 — `use ferrotorch_nn::Parameter`
- `ferrotorch-optim/src/adam.rs` line 17 — `use ferrotorch_nn::Parameter`
- `ferrotorch-train/src/grad_utils.rs` — `pub use ferrotorch_nn::utils::{clip_grad_norm_, clip_grad_value_}`
- 28 model crates compose the layer types re-exported from this file.

The crate root is the canonical entry point; every external
consumer touches it directly.

## Parity contract

`parity_ops = []`. `lib.rs` is a structural/declarative file with no
runtime semantics of its own — its correctness is "every per-module
file is reachable and every pub type is re-exported under the
documented name". Verified by `cargo check -p ferrotorch-nn` (which
fails if any `pub mod` references a missing file) plus downstream
crates importing the re-exported names (which fail to compile if a
name is missing).

## Verification

- `cargo check -p ferrotorch-nn` — module graph + re-exports compile.
- Downstream crates (`ferrotorch-optim`, `ferrotorch-train`, every
  model crate) link against the re-exports — any missing name is a
  compile-time failure visible in the workspace build.
- The lint baseline is exercised by `cargo clippy -p ferrotorch-nn
  --lib -- -D warnings` (the gauntlet's clippy step).
- No per-module test file specifically targets `lib.rs`; its surface
  is verified indirectly through every other module's tests.

Smoke command (no parity ops):

```bash
cargo check -p ferrotorch-nn 2>&1 | tail -3
cargo clippy -p ferrotorch-nn --lib -- -D warnings 2>&1 | tail -3
```

Expected: `Finished` on both.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: crate-root `#![warn(clippy::all, clippy::pedantic)]` + `#![deny(rust_2018_idioms)]` + per-lint-justified `#![allow(...)]` block in `lib.rs`, mirroring the workspace-standard pattern; non-test consumer: every Edit to `ferrotorch-nn/src/**/*.rs` is gated by this baseline, and `cargo clippy -p ferrotorch-nn --lib -- -D warnings` enforces it on every downstream build. |
| REQ-2 | SHIPPED | impl: 31 `pub mod` declarations in `lib.rs` covering every per-layer file; non-test consumer: each `pub mod` resolves to a real file (e.g. `pub mod linear;` ↔ `linear.rs`) — `cargo check -p ferrotorch-nn` fails if any module file is missing. |
| REQ-3 | SHIPPED | impl: flat `pub use activation::{...}`, `pub use container::{...}`, etc. block in `lib.rs`, mirroring `torch/nn/__init__.py:11-50`; non-test consumer: `ferrotorch-optim/src/optimizer.rs` `use ferrotorch_nn::Parameter`, `ferrotorch-optim/src/adam.rs` likewise, plus the 28 model crates. |
| REQ-4 | SHIPPED | impl: `pub use ferrotorch_nn_derive::Module` in `lib.rs` republishes the derive macro under the name `Module`, mirroring upstream `nn.Module`'s registration semantics via R-DEV-7 (proc macro replacing Python metaclass); non-test consumer: every concrete layer file that uses `#[derive(Module)]` resolves to this re-export — e.g. derive-macro–generated paths in `ferrotorch-nn-derive`'s expansion. |
| REQ-5 | SHIPPED | impl: `pub mod prelude` block at the bottom of `lib.rs` collecting `Module`, `Parameter`, `Buffer`, `StateDict`, `Reduction`, `DeriveModule`, standard layers, canonical losses, gradient-clipping helpers; non-test consumer: downstream model authors who write `use ferrotorch_nn::prelude::*` (e.g. example training scripts and tutorial bundles consume the prelude as the documented onboarding surface). |
| REQ-6 | SHIPPED | impl: `#[allow(unused_extern_crates)] extern crate self as ferrotorch_nn;` in `lib.rs`, load-bearing for the derive macro's `::ferrotorch_nn::Module` hygienic path; non-test consumer: the derive-macro–expanded code in any `#[derive(Module)]` invocation inside this crate references this binding (compile-fail without it). |
