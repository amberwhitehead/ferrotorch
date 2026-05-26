# ferrotorch-ml — sklearn-compatible adapter crate root

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/__init__.py
  - torch/__init__.py
-->

## Summary

`ferrotorch-ml/src/lib.rs` is the crate root for the sklearn-compatible
adapter that bridges `ferrotorch_core::Tensor` to the `ferrolearn`
ecosystem (a scikit-learn equivalent in Rust). It re-exports
`ferrolearn_preprocess`, `ferrolearn_decomp`, and `ferrolearn_model_sel`
behind crate-local sub-module names so the full sklearn surface is one
import away. Upstream PyTorch itself does NOT ship a sklearn bridge —
this crate fills the slot occupied by `skorch` / `pytorch-tabular` in
the Python ecosystem. The Rust analog is materially better than what
upstream offers (R-DEV-7) because `ferrolearn` is a typed,
numerically-strict reimplementation rather than a duck-typed wrapper.

## Requirements

- REQ-1: The crate establishes a workspace-mirroring lint baseline
  (`#![warn(clippy::all, clippy::pedantic)]` + per-lint `#![allow]`
  blocks with documented rationale) so the bridge crate participates in
  the workspace clippy gate.
- REQ-2: Three public sub-modules — `adapter`, `metrics`, `datasets` —
  are declared and reachable as `ferrotorch_ml::{adapter, metrics,
  datasets}`.
- REQ-3: Three ferrolearn re-export sub-modules — `preprocess`,
  `decomposition`, `model_selection` — expose the full
  `ferrolearn_preprocess::*` / `ferrolearn_decomp::*` /
  `ferrolearn_model_sel::*` surface unchanged, so callers can write
  `use ferrotorch_ml::preprocess::StandardScaler` without depending on
  ferrolearn directly.
- REQ-4: The module-level doc-comment explicitly documents the
  CPU-only-by-design relaxation: GPU tensors are transparently moved
  to host memory by the adapter, but compute crates (`ferrotorch-core`,
  `-nn`, `-gpu`) continue to enforce strict no-silent-fallback. The
  user opts into the relaxation by reaching into this crate.

## Acceptance Criteria

- [x] AC-1: Crate-root lint baseline present with per-lint `allow`
  blocks carrying inline justification (no bare module-root
  `#![allow]` without rationale).
- [x] AC-2: `pub mod adapter; pub mod datasets; pub mod metrics;`
  declarations resolve.
- [x] AC-3: `pub mod preprocess { pub use ferrolearn_preprocess::*; }`
  plus the two parallel re-export blocks resolve.
- [x] AC-4: Crate doc-comment (`//!`) documents the GPU
  auto-materialisation relaxation and references the dedicated bridge
  crate's role.

## Architecture

### Lint baseline (REQ-1)

The crate-root attribute block in `lib.rs` warns `clippy::all` and
`clippy::pedantic`, denies `rust_2018_idioms` and
`missing_debug_implementations`, and holds `missing_docs` at `allow`
pending the workspace-wide rustdoc pass tracked as #731. Per-lint
`#![allow]` entries name the concrete rationale (e.g.
`clippy::module_name_repetitions` because the wrapper fns
intentionally re-export ferrolearn metric/dataset names so the API
maps 1:1 onto sklearn's surface). Each allow lives in `lib.rs` with a
documented justification — none are silent suppressions.

Mirrors the `ferrotorch-gpu/src/lib.rs` and `ferrotorch-jit/src/lib.rs`
baselines so the bridge crate participates in the workspace clippy
gate at the same posture.

### Public module declarations (REQ-2, REQ-3)

```rust
pub mod adapter;
pub mod datasets;
pub mod metrics;

pub mod preprocess {
    pub use ferrolearn_preprocess::*;
}
pub mod decomposition {
    pub use ferrolearn_decomp::*;
}
pub mod model_selection {
    pub use ferrolearn_model_sel::*;
}
```

The three first-party modules (`adapter`, `metrics`, `datasets`) are
defined locally. The three sklearn re-exports use the `pub use`
glob-export pattern so the entire upstream sub-crate API surface
appears under the namespaced crate path. Callers depend only on
`ferrotorch-ml`, not on individual ferrolearn sub-crates.

### CPU-only-by-design rationale (REQ-4)

The crate doc-comment explains:

- ferrolearn is CPU-only (built on `ndarray` + `faer`); GPU tensors
  cannot enter the sklearn pipeline directly.
- `ferrotorch-ml` accepts tensors on any device — the adapter routes
  GPU buffers through host memory before handing data to ferrolearn,
  mirroring the torch idiom `loss.cpu().item()`.
- The relaxation applies only to this dedicated bridge crate. Compute
  crates (`ferrotorch-core`, `-nn`, `-gpu`) continue to enforce the
  strict `/rust-gpu-discipline` no-silent-fallback rule.
- The function names (`tensor_to_array2`, `tensor_to_array1_usize`,
  etc.) make the device crossing self-evident at the call site, so
  hot-path callers can explicitly opt in (or assert `t.device().is_cpu()`
  before calling).

### Non-test production consumers

- `ferrotorch-ml/src/datasets.rs` uses
  `crate::adapter::{array1_to_tensor, array1_usize_to_tensor, array2_to_tensor}`
  to pack ferrolearn dataset output back into `Tensor<F>` pairs.
- `ferrotorch-ml/src/metrics.rs` uses
  `crate::adapter::{tensor_to_array1, tensor_to_array1_usize}` to
  unpack `Tensor<T>` arguments for every metric wrapper.
- Downstream callers depend on `ferrotorch-ml` via Cargo and import
  through these re-exports (the conformance-surface inventory at
  `ferrotorch-ml/tests/conformance/_surface_inventory.toml` enumerates
  every public path).

### Upstream PyTorch mapping (R-DEV-7 deviation)

Upstream `torch/nn/__init__.py` ships `nn.Linear`, `nn.Conv2d`, etc.,
but does NOT include a sklearn bridge. The PyTorch ecosystem fills
that slot via third-party packages (`skorch`,
`pytorch-tabular`, `pytorch-lightning`). ferrotorch bundles the bridge
as a first-party leaf crate (R-DEV-7 — the Rust ecosystem analog
`ferrolearn` is materially better than the duck-typed Python
wrappers). The contract preserved is the sklearn API surface
(function signatures, kwargs, return shapes); the implementation is
the Rust ferrolearn ecosystem.

## Parity contract

`parity_ops = []`. The crate root performs no numerical computation
of its own — it's a module-declaration + re-export shell. Edge-case
parity is owned by the modules it declares:

- Adapter parity → `.design/ferrotorch-ml/adapter.md` (memcpy
  fidelity, non-contiguous handling, finite-check on label casts).
- Metric parity → `.design/ferrotorch-ml/metrics.md` (sklearn
  reference values).
- Dataset parity → `.design/ferrotorch-ml/datasets.md` (shape
  contracts, label-encoding round-trip).

The CPU-only relaxation declared in the crate doc-comment is itself a
parity contract: GPU inputs must materialise transparently and not
fail at the adapter boundary.

## Verification

The crate has no lib-level tests — every test lives inside the
sub-module under `#[cfg(test)] mod tests`. Crate-wide gauntlet:

```bash
cargo test -p ferrotorch-ml --lib 2>&1 | tail -3
cargo clippy -p ferrotorch-ml --lib -- -D warnings 2>&1 | tail -3
cargo fmt -p ferrotorch-ml --check
```

The integration tests under `ferrotorch-ml/tests/` exercise the
public surface end-to-end:

- `conformance_ml_adapter.rs` — adapter symbol reachability +
  round-trip.
- `conformance_ml_metrics.rs` — metric reference-value
  cross-checks.
- `conformance_ml_datasets.rs` — dataset shape contracts.
- `conformance_sklearn_parity.rs` — sklearn parity probe.
- `conformance_surface_coverage.rs` — checks the public path
  inventory matches the toml manifest.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-ml --lib 2>&1 | tail -3
```

Expected: `61 passed` across the three sub-modules.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: crate-root attribute block (`#![warn(clippy::all, clippy::pedantic)]` + `#![deny(rust_2018_idioms, missing_debug_implementations)]` + per-lint `#![allow]` with documented rationale) at top of `ferrotorch-ml/src/lib.rs`; non-test consumer: the workspace clippy gate (`cargo clippy -p ferrotorch-ml -- -D warnings`) consumes this baseline, and the per-lint allows are referenced by every production fn in `adapter.rs`, `datasets.rs`, `metrics.rs` (the `clippy::cast_*` and `clippy::needless_pass_by_value` lints would otherwise fire on the cast-heavy adapter code). |
| REQ-2 | SHIPPED | impl: `pub mod adapter; pub mod datasets; pub mod metrics;` declarations in `ferrotorch-ml/src/lib.rs`; non-test consumer: `ferrotorch-ml/src/datasets.rs` `use crate::adapter::{array1_to_tensor, array1_usize_to_tensor, array2_to_tensor}` and `ferrotorch-ml/src/metrics.rs` `use crate::adapter::{tensor_to_array1, tensor_to_array1_usize}` consume the adapter module via these declarations. |
| REQ-3 | SHIPPED | impl: `pub mod preprocess { pub use ferrolearn_preprocess::*; }` and the two parallel re-export blocks in `ferrotorch-ml/src/lib.rs`; non-test consumer: the conformance surface inventory at `ferrotorch-ml/tests/conformance/_surface_exclusions.toml` lists `ferrotorch_ml::preprocess::*` / `ferrotorch_ml::decomposition::*` / `ferrotorch_ml::model_selection::*` as the glob-reachable re-export paths the production API exposes, and the README documents them as the user-facing import path. |
| REQ-4 | SHIPPED | impl: crate doc-comment paragraphs "CPU-only by design, GPU input transparently materialised" and "the relaxation here applies only to this dedicated bridge crate" in `ferrotorch-ml/src/lib.rs`; non-test consumer: `ferrotorch-ml/src/adapter.rs` module-level doc-comment quotes the same contract (`"GPU tensors are auto-moved to CPU"` section) and `ferrotorch-ml/src/metrics.rs` module-level doc-comment references it (`"GPU input is transparently materialised"` section) so every wrapper inherits the documented relaxation. |
