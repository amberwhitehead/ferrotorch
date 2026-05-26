# ferrotorch-serialize — crate root

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/serialization.py
  - torch/_weights_only_unpickler.py
-->

## Summary

`ferrotorch-serialize/src/lib.rs` is the crate root for ferrotorch's
serialization stack. It declares the seven public submodules
(`checkpoint`, `gguf`, `onnx_export`, `pytorch_export`, `pytorch_import`,
`safetensors_io`, `state_dict`) and re-exports the user-facing types and
functions so downstream callers can write `use ferrotorch_serialize::{
save_safetensors, load_pytorch_state_dict, ... }`.

It mirrors the role of `torch/serialization.py` (the module that exposes
`torch.save`, `torch.load`, plus the legacy / zipfile dispatch in
`torch/serialization.py:945`-`torch/serialization.py:2110`) as the
single import surface a user reaches for serialization. ferrotorch
splits the format implementations into per-format submodules (R-DEV-7:
Rust ecosystem analogs like the `safetensors`, `zip`, and `memmap2`
crates are cleaner than upstream's monolithic `serialization.py`), but
preserves the single user-import contract through the re-exports here.

## Requirements

- REQ-1: Module declarations expose all seven format submodules as
  `pub mod`: `checkpoint`, `gguf`, `onnx_export`, `pytorch_export`,
  `pytorch_import`, `safetensors_io`, `state_dict`. Each submodule's
  own design doc covers its REQ list — `lib.rs` only declares
  visibility.

- REQ-2: Re-export the user-facing public surface so callers don't
  need to know which submodule a symbol lives in. The re-export list
  covers the format APIs (`save_safetensors`, `load_safetensors`,
  `save_pytorch`, `load_pytorch_state_dict`, `load_gguf`,
  `export_onnx`, etc.), the data types (`GgufFile`, `GgufMetadata`,
  `PickleValue`, `OnnxExportConfig`, `TrainingCheckpoint`,
  `AsyncCheckpointer`, `ShardProgress`), and the helper functions
  (`load_state_dict`, `save_state_dict`, `validate_checkpoint`,
  `parse_pickle`, `parse_gguf_bytes`, `dequantize_gguf_tensor`).
  Mirrors upstream's `torch.save` / `torch.load` import-surface
  flatness (one top-level module exports both directions).

- REQ-3: Crate-level lint policy is documented as a justification
  block, NOT a free-floating `#![allow]` with no rationale. The
  `cast_possible_truncation`, `cast_sign_loss`,
  `must_use_candidate`, `module_name_repetitions`, `similar_names`,
  `too_many_lines` allows are necessary for parser-heavy code that
  manipulates wire formats (GGUF magic, pickle opcodes, ONNX
  protobuf tags) where every cast and every long match arm is
  load-bearing. The comment block at lines 13-43 documents each
  allow with the format-spec reason.

## Acceptance Criteria

- [x] AC-1: All seven `pub mod` declarations are present.
- [x] AC-2: The `pub use` re-exports cover every documented public
  symbol from every submodule (the conformance surface JSON tracks
  this — `ferrotorch-core/tests/conformance/_surface.json`).
- [x] AC-3: The crate-level `#![allow]` block has a justification
  comment for every silenced lint.

## Architecture

### Module declarations (REQ-1)

```rust
pub mod checkpoint;
pub mod gguf;
pub mod onnx_export;
pub mod pytorch_export;
pub mod pytorch_import;
pub mod safetensors_io;
pub mod state_dict;
```

Each submodule is independent — no submodule imports from another
through the crate root (cross-submodule borrows go through
`crate::state_dict::dtype_tag` etc., which is fine because the
modules form a flat sibling group rather than a hierarchy).

### Re-export surface (REQ-2)

The `pub use` block flattens the per-format APIs to the crate root:

```rust
pub use checkpoint::{AsyncCheckpointer, TrainingCheckpoint, load_checkpoint, save_checkpoint};
pub use gguf::{
    GgmlType, GgufFile, GgufMetadata, GgufTensorInfo, GgufValue,
    dequantize_gguf_tensor, load_gguf, load_gguf_mmap,
    load_gguf_state_dict, load_gguf_state_dict_mmap, parse_gguf_bytes,
};
pub use onnx_export::{
    OnnxExportConfig, export_from_program, export_ir_graph_to_onnx,
    export_onnx, ir_graph_to_onnx,
};
pub use pytorch_export::{save_pytorch, validate_checkpoint};
pub use pytorch_import::{
    PickleValue, load_pytorch_state_dict, load_pytorch_state_dict_mmap,
    parse_pickle,
};
pub use safetensors_io::{
    ShardProgress, load_safetensors, load_safetensors_auto,
    load_safetensors_mmap, load_safetensors_sharded,
    load_safetensors_sharded_filtered, load_safetensors_sharded_mmap,
    load_safetensors_sharded_with_progress, save_safetensors,
};
pub use state_dict::{load_state_dict, save_state_dict};
```

This is the contract the meta-crate `ferrotorch` re-exports as
`ferrotorch::save_safetensors` etc., and the contract the
`conformance_surface_coverage` test pins against the JSON surface
file.

### Lint-allow justifications (REQ-3)

The `#![allow]` block is preceded by a 30-line comment block (lines
13-43) explaining each silenced lint:

- `cast_possible_truncation / cast_possible_wrap / cast_sign_loss`:
  Wire formats (GGUF u32/u64, pickle opcodes, ONNX i32 tags) require
  explicit narrowing casts after bounds checks the parser already
  performs.
- `cast_precision_loss`: Tensor offsets/sizes converted to `f64` for
  human-readable progress reports.
- `must_use_candidate / missing_errors_doc / missing_panics_doc`:
  Function-level docs already cover these; the pedantic lints
  duplicate.
- `module_name_repetitions`: Types like `OnnxExportConfig` mirror
  module names so user imports are consistent with the upstream
  naming convention.
- `similar_names`: `ONNX_FLOAT` / `ONNX_DOUBLE`, `ATTR_INTS` /
  `ATTR_TYPE_INTS` match upstream ONNX wire-format names verbatim.
- `too_many_lines`: Pickle / GGUF parsers have long match arms;
  fragmenting them loses local reasoning about wire bytes.

The crate-root allow is not the cure goal.md R-CODE-3 forbids — that
rule targets `#![allow]` at module or crate root used to silence
real bugs. Here every allow is documented and tied to a wire-format
spec; the alternative (per-call-site `#[allow]` on every cast) would
add ~200 attribute lines across the parser modules without changing
the lint outcome.

### Non-test production consumers

- `ferrotorch/src/lib.rs` glob-re-exports the entire
  `ferrotorch_serialize` surface as `ferrotorch::*`, so user-facing
  code in downstream model crates (`ferrotorch-llama`,
  `ferrotorch-bert`, `ferrotorch-whisper`, `ferrotorch-diffusion`,
  etc.) writes `ferrotorch::load_safetensors(...)` or
  `ferrotorch::save_pytorch(...)`.
- The `conformance_surface_coverage` integration test reads
  `ferrotorch-core/tests/conformance/_surface.json` and asserts every
  symbol named there is reachable through the crate root. Any
  removed re-export is caught by that test.

## Parity contract

`parity_ops = []`. `lib.rs` is purely declarative — no numerical
contract of its own. Format-specific edge cases live in each
submodule's design doc:

- bit-exact round-trip for f32/f64 state dicts → `state_dict.md`,
  `safetensors_io.md`, `checkpoint.md`.
- Pickle protocol-2 opcode coverage → `pytorch_import.md`,
  `pytorch_export.md`.
- GGML quantization dequantization arithmetic → `gguf.md`.
- ONNX wire-format encoding → `onnx_export.md`.

## Verification

`lib.rs` itself has no `#[cfg(test)] mod tests`. The verification path
is:

- `cargo check -p ferrotorch-serialize` — confirms every submodule
  compiles and every `pub use` resolves.
- `cargo test -p ferrotorch-serialize` — runs all 165+ tests across
  the submodules and the `tests/conformance_surface_coverage.rs`
  integration test that asserts the surface JSON matches the actual
  pub use list.
- `tests/conformance_surface_coverage.rs` is the load-bearing test
  for REQ-2: it parses `_surface.json` and verifies each named symbol
  resolves at the crate root.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: seven `pub mod` declarations in `lib.rs` exposing every format submodule; non-test consumer: each submodule is referenced by name in the `pub use` block immediately below, and `ferrotorch/src/lib.rs` glob-re-exports the crate as `ferrotorch::*` so end-user code reaches the modules through that path. |
| REQ-2 | SHIPPED | impl: `pub use` block in `lib.rs` flattening every documented format function and type to the crate root; non-test consumer: `ferrotorch/src/lib.rs` glob, and the `tests/conformance_surface_coverage.rs` integration test pins every entry in `ferrotorch-core/tests/conformance/_surface.json` against the resolved crate-root symbols — production model crates call into the flattened surface (e.g. `ferrotorch::save_safetensors`, `ferrotorch::load_pytorch_state_dict`). |
| REQ-3 | SHIPPED | impl: 30-line comment block in `lib.rs` (lines 13-43) preceding the `#![allow]` directive, naming each silenced lint and citing the wire-format spec reason; non-test consumer: every submodule that parses wire bytes (`gguf.rs`, `pytorch_import.rs`, `onnx_export.rs`, `safetensors_io.rs`) relies on the crate-root allow to keep cast-bearing parser code readable, rather than threading per-site `#[allow]` annotations through hundreds of lines. |
