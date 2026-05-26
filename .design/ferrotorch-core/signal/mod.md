# Signal module root

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - torch/signal/__init__.py
  - torch/signal/windows/__init__.py
  - torch/_torch_docs.py
  - torch/overrides.py
-->

## Summary

`ferrotorch-core/src/signal/mod.rs` is a 12-line re-export hub that
exposes `torch.signal.windows.*` window-function APIs as
`ferrotorch_core::signal::windows::*` and convenience re-exports for the
top-level `bartlett` / `blackman` / `hann` / ... functions. The whole
module mirrors `torch/signal/__init__.py` (which in turn re-exports
from `torch/signal/windows/__init__.py`).

## Requirements

- REQ-1: `pub mod windows` — expose the `windows` submodule
  (`ferrotorch-core/src/signal/windows.rs`). Mirrors
  `torch.signal.windows` being a public module of `torch.signal`.
- REQ-2: Convenience re-exports — `pub use windows::{bartlett, blackman,
  cosine, exponential, gaussian, general_cosine, general_hamming,
  hamming, hann, hanning, kaiser, nuttall, parzen, taylor, tukey};`
  so callers can write `ferrotorch_core::signal::hann(N)` without
  reaching through the submodule path. Mirrors
  `torch.signal.__init__.from .windows import ...`.

## Acceptance Criteria

- [x] AC-1: `use ferrotorch_core::signal::windows;` resolves
  (mod declaration at `signal/mod.rs:7`).
- [x] AC-2: `use ferrotorch_core::signal::hann;` resolves
  (re-export at `signal/mod.rs:9-12`).
- [x] AC-3: All 15 window names are exposed
  (`signal/mod.rs:10-11` lists the 15 names).

## Architecture

The file is 12 lines:

```rust
//! Signal-processing utilities.
//!
//! Mirrors `torch.signal.*`. Currently exposes the [`windows`] submodule;
//! future work may add filter design, convolution helpers, and other
//! `scipy.signal`-shaped primitives.

pub mod windows;

pub use windows::{
    bartlett, blackman, cosine, exponential, gaussian, general_cosine, general_hamming, hamming,
    hann, hanning, kaiser, nuttall, parzen, taylor, tukey,
};
```

There is no executable code, just module-system glue.

## Parity contract

`parity_ops = []`. Re-export modules have no parity contract beyond
exposing the underlying functions, whose contracts live in
`signal/windows.md`.

## Verification

- The whole-crate build asserts the module-system glue is sound:

  ```bash
  cargo check -p ferrotorch-core
  ```

  Expected: 0 errors. If a function name is removed from
  `windows.rs` without removing it from `mod.rs`'s `use` block, the
  compile fails.
- The `output_lives_on_cpu` test at `signal/windows.rs:343-366`
  enumerates all 15 window names and verifies they return CPU tensors
  via the convenience re-exports — implicit confirmation that the
  re-export surface is complete.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub mod windows;` at `ferrotorch-core/src/signal/mod.rs:7`; non-test consumer: `ferrotorch-core/src/signal/windows.rs` is the module body; downstream callers reach `ferrotorch_core::signal::windows::hann(N)` via this declaration. The signal module is also re-exported at `ferrotorch-core/src/lib.rs` (`pub mod signal;` registration). |
| REQ-2 | SHIPPED | impl: `pub use windows::{...}` at `ferrotorch-core/src/signal/mod.rs:9-12` lists all 15 window-function names; non-test consumer: the windows are reachable as `ferrotorch_core::signal::hann(N)` etc.; the test at `ferrotorch-core/src/signal/windows.rs:343-366` exercises all 15 names from the top-level path (production call surface). |
