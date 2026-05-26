# ops/ — module-namespace declarations for the non-autograd op layer

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158 (Revert "[dynamo] Implement nb_or/nb_inplace_or slot dispatch for | and |= operators (#181326)")
upstream-paths:
  - aten/src/ATen/native/
-->

## Summary

`ferrotorch-core/src/ops/mod.rs` is a single-purpose module declaration
file. It enumerates the nine sub-modules that constitute the kernel
(non-autograd) op layer:

```rust
pub mod cumulative;
pub mod elementwise;
pub mod higher_order;
pub mod indexing;
pub mod linalg;
pub mod phase2c;
pub mod scatter;
pub mod search;
pub mod tensor_ops;
```

That is the entire file (9 lines). Each sub-module owns its own design
doc (`.design/ferrotorch-core/ops/cumulative.md`, etc.); this design
doc documents only the **namespace** at `ferrotorch-core::ops::*`,
which mirrors PyTorch's `aten/src/ATen/native/` directory-as-namespace
convention (`ReduceOps.cpp` for cumulative, `BinaryOps.cpp` for
elementwise, `Indexing.cpp` for indexing, etc.).

## Requirements

- REQ-1: The `pub mod` declarations expose nine sub-modules under
  `ferrotorch_core::ops::*`. Each is the **kernel layer** for the
  corresponding op family — forward-only math, no autograd tracking.
  The autograd layer lives in `ferrotorch-core/src/grad_fns/*.rs` and
  delegates to the kernel layer via
  `crate::ops::<family>::<fn>(...)` calls.
- REQ-2: The split between kernel layer (`ops::*`) and autograd layer
  (`grad_fns::*`) mirrors PyTorch's `_<op>` (private dispatcher) vs `<op>`
  (user-facing) convention at `aten/src/ATen/native/ReduceOps.cpp:465`
  (`_cummax_helper`, `_logcumsumexp_cpu`, …) vs the namespace functions
  (`cummax`, `logcumsumexp`).
- REQ-3: No re-exports happen in this file — each sub-module exposes its
  own public symbols, and the top-level `lib.rs:173-177` re-export block
  picks specific symbols (`CumExtremeResult`, `gather`, `masked_select`,
  `topk`, `unique`, `meshgrid`, `cdist`, `diag`, `roll`, `tril`, `triu`,
  `searchsorted`, `bucketize`, `histc`, `scatter_add_segments`,
  `where_cond`, `scatter`, `scatter_add`) from those sub-modules for
  unqualified access at `ferrotorch_core::*`.

## Acceptance Criteria

- [x] AC-1: All 9 sub-modules exist on disk (mechanical: `ls
  ferrotorch-core/src/ops/` returns nine `.rs` files plus `mod.rs`).
- [x] AC-2: `cargo check -p ferrotorch-core` compiles — every `pub mod`
  declaration resolves to a `<name>.rs` file (mechanical: the gauntlet
  pass is the proof).
- [x] AC-3: `lib.rs` `pub use` re-exports of `ops::*` symbols (lib.rs:173
  / :174 / :175 / :176 / :177) resolve — the public surface that
  downstream crates import.

## Architecture

### The 9 sub-modules

Each sub-module is a separate translation unit with its own design doc.
A one-line summary:

- `cumulative` — scan ops (`cumsum`, `cumprod`, `cummax`, `cummin`,
  `logcumsumexp`). Mirrors `aten/src/ATen/native/ReduceOps.cpp`.
  Design: `.design/ferrotorch-core/ops/cumulative.md` (not yet written;
  the autograd-layer doc at `.design/ferrotorch-core/grad_fns/cumulative.md`
  is currently the authoritative reference for the cumulative family).
- `elementwise` — elementwise unary / binary ops (broadcast helpers,
  `fast_exp`, `fast_log`, `fast_sin`, `fast_cos`, `unary_map`). Mirrors
  `aten/src/ATen/native/UnaryOps.cpp` + `BinaryOps.cpp`.
- `higher_order` — higher-order helpers (map / fold patterns over
  tensors). Mirrors `aten/src/ATen/native/native_functions.yaml`'s
  vmap-friendly entries.
- `indexing` — `gather` / `scatter` / `masked_select` / `where_cond`.
  Mirrors `aten/src/ATen/native/Indexing.cpp`. Has its own design doc.
- `linalg` — `mm`, `bmm`, `matmul`, …. Mirrors
  `aten/src/ATen/native/LinearAlgebra.cpp`. Has its own design doc
  (`.design/ferrotorch-core/ops/linalg.md`).
- `phase2c` — integer cast / argmax / argmin kernels staged for
  crosslink #1185 Phase 2c.
- `scatter` — `scatter_add_segments` and segment-aggregation kernels.
  Mirrors `aten/src/ATen/native/TensorAdvancedIndexing.cpp`.
- `search` — `topk`, `unique`, `unique_consecutive`, `meshgrid`,
  `bucketize`, `histc`, `searchsorted`. Mirrors
  `aten/src/ATen/native/Sorting.cpp` + `Bucketization.cpp`.
- `tensor_ops` — `cdist`, `diag`, `diagflat`, `roll`, `tril`, `triu`.
  Mirrors `aten/src/ATen/native/TensorTransformations.cpp`.

### Why a separate kernel namespace (REQ-2)

The kernel layer is forward-only — no autograd graph manipulation, no
`requires_grad` checks, no grad-fn attachment. This split allows:
- Direct kernel reuse by the parity-sweep runner (which calls
  `ops::cumulative::cumsum_forward(...)` rather than the
  autograd-wrapped `grad_fns::cumulative::cumsum(...)` to isolate kernel
  correctness from autograd contributions).
- Internal call paths in `grad_fns/*` that just need a forward
  computation (e.g. `LogcumsumexpBackward::backward` calls
  `crate::ops::cumulative::reverse_cumsum`).
- A cleaner per-op file (the autograd-layer file is smaller because the
  forward kernel lives elsewhere).

Mirrors upstream's `_<op>` private-helper convention at
`aten/src/ATen/native/`.

### Production consumers

`crate::ops::*` is consumed throughout `crate::grad_fns::*`. Concrete
non-test consumers:
- `ferrotorch-core/src/grad_fns/cumulative.rs:32` `use
  crate::ops::cumulative::{...};` and downstream forward-kernel calls.
- `ferrotorch-core/src/grad_fns/transcendental.rs:15` `use
  crate::ops::elementwise::{fast_cos, fast_sin, unary_map};`
- `ferrotorch-core/src/tensor.rs:1146`
  `crate::ops::indexing::masked_select(self, mask)` — the boundary
  method consumes the kernel directly.
- `ferrotorch-core/src/lib.rs:173-177` `pub use` re-exports lift
  specific `ops::*` symbols to the crate-root namespace for downstream
  crates.

## Parity contract

`parity_ops = []`. This file is purely structural. The parity surface is
the union of the 9 sub-modules' parity ops — when each sub-module's design
doc lands, its parity contract is documented there.

## Verification

```
cargo check -p ferrotorch-core
cargo test -p ferrotorch-core --lib ops
```

Expected: `cargo check` passes (compilation proof that all 9 `pub mod`
declarations resolve); `cargo test -p ferrotorch-core --lib ops` runs
every `#[cfg(test)] mod tests` inside the 9 sub-modules.

There is no `#[cfg(test)] mod tests` block in `ops/mod.rs` itself —
the file has only `pub mod` declarations. The verification is
mechanical (compilation) + delegated (sub-module tests).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: 9 `pub mod <name>;` declarations at `ferrotorch-core/src/ops/mod.rs:1-9` exposing the kernel-layer sub-modules. Each declared module resolves to a `<name>.rs` file under `ferrotorch-core/src/ops/`. Non-test production consumer: `ferrotorch-core/src/grad_fns/cumulative.rs:32` (`use crate::ops::cumulative::{...}`), `ferrotorch-core/src/grad_fns/transcendental.rs:15` (`use crate::ops::elementwise::{fast_cos, fast_sin, unary_map}`), `ferrotorch-core/src/tensor.rs:1146` (`crate::ops::indexing::masked_select`). |
| REQ-2 | SHIPPED | impl: the split is **the** organizational primitive — each sub-module file is the kernel layer, and the corresponding `ferrotorch-core/src/grad_fns/<family>.rs` is the autograd wrapper. Non-test production consumer: `ferrotorch-core/src/grad_fns/cumulative.rs:32-35` imports from `crate::ops::cumulative` and the body of `pub fn cumsum` (`grad_fns/cumulative.rs:104`) delegates the forward to `ops::cumulative::cumsum_forward(...)`. This is the upstream `aten::cummax` (user) vs `_cummax_helper` (private) split mirrored 1:1. |
| REQ-3 | SHIPPED | impl: this file has no `pub use` (mechanical: only nine `pub mod` lines). Non-test production consumer: `ferrotorch-core/src/lib.rs:173-177` `pub use ops::indexing::{gather, masked_select, scatter, ...}` etc. lifts specific symbols — the picking-by-symbol pattern requires the sub-modules to NOT pre-re-export, which mod.rs preserves by being a pure-declaration file. |
