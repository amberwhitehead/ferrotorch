# ferrotorch-jit — `export` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/export/__init__.py
  - torch/export/exported_program.py
  - torch/export/dynamic_shapes.py
-->

## Summary

`ferrotorch-jit/src/export.rs` provides the `torch.export`-style API:
`ExportedProgram` (an `IrGraph` paired with its state dict, input
specs, and output shape), `export()` (single-input static-shape
flavour), and `export_with_dynamic_shapes()` (dynamic-shape flavour
with per-dim `DimSpec`). Mirrors `torch.export.export` at
`torch/export/__init__.py:59-205` and the
`ExportedProgram` class at `torch/export/exported_program.py`.

## Requirements

- REQ-1: `pub enum DimSpec { Static(usize), Dynamic { name: String,
  min: Option<usize>, max: Option<usize> } }` describes per-dimension
  symbolic / static specification. Mirrors
  `torch.export.Dim(name, min=None, max=None)` at
  `torch/export/dynamic_shapes.py`.
- REQ-2: `pub struct InputSpec { shape: Vec<DimSpec> }` holds the
  per-input shape spec produced by tracing. `InputSpec::new`,
  `InputSpec::all_static`, `InputSpec::has_dynamic_dims`,
  `InputSpec::rank` form the public surface.
- REQ-3: `pub struct ExportedProgram` holds the traced `IrGraph`,
  `state_dict: HashMap<String, Vec<f32>>`, `input_shapes`,
  `input_specs`, and `output_shape`. Mirrors the
  `ExportedProgram` dataclass at
  `torch/export/exported_program.py:200`.
- REQ-4: `ExportedProgram::run(inputs)` interprets the graph
  against new inputs (mirrors `ExportedProgram.module().forward()`);
  `ExportedProgram::run_with_guards(inputs)` validates against
  `input_specs` before running (mirrors `torch.export`'s
  `_check_input_constraints`).
- REQ-5: `ExportedProgram::check_inputs` exposes the guard logic
  standalone — input-count check, per-input rank check, static-dim
  equality check, dynamic-dim `[min, max]` range check (CL-461).
- REQ-6: `ExportedProgram::serialize` and `deserialize` round-trip
  the program through a custom binary format (graph payload via
  `IrGraph::serialize` plus an outer header carrying state_dict +
  input_specs + output_shape). Mirrors `torch.export.save` /
  `torch.export.load`.
- REQ-7: `ExportedProgram::save(path)` / `load(path)` are the
  on-disk variants — write `serialize()` to a file and read back.
  Mirrors `torch.export.save(ep, path)`.
- REQ-8: `ExportedProgram::to_json` and `parse_json_metadata`
  produce / parse a simple JSON view of the program's metadata for
  diagnostics. (Non-canonical; the binary format is the source of
  truth.)
- REQ-9: `pub fn export<T, M>(module, example_inputs) ->
  FerrotorchResult<ExportedProgram>` is the single-input static
  flavour. Single-input only (documented limitation in the
  module-level doc-comment).
- REQ-10: `pub fn export_with_dynamic_shapes<T, M>(module,
  example_inputs, input_specs) -> FerrotorchResult<ExportedProgram>`
  accepts user-declared `InputSpec` per input, building a polymorphic
  trace that matches `torch.export(model, args, dynamic_shapes=...)`.

## Acceptance Criteria

- [x] AC-1: `DimSpec::Static(8)` is constructible;
  `DimSpec::dynamic("batch")` returns `DimSpec::Dynamic { name:
  "batch".into(), min: None, max: None }`;
  `DimSpec::dynamic_range("batch", 1, 32)` returns the bounded
  variant.
- [x] AC-2: `InputSpec::all_static(&[1, 3, 224, 224])` rank-4 spec
  has `has_dynamic_dims() == false` and `rank() == 4`.
- [x] AC-3: `export(module, &[input])` returns an `ExportedProgram`
  with a non-empty `graph`, matching `input_shapes[0] ==
  input.shape()`.
- [x] AC-4: `ep.run(&[other_input])` reproduces the eager output
  within 1e-5.
- [x] AC-5: `ep.run_with_guards(&[wrong_rank_input])` returns
  `Err(InvalidArgument)` citing the rank mismatch.
- [x] AC-6: `ep.serialize()` then `ExportedProgram::deserialize(...)`
  round-trips byte-exactly.
- [x] AC-7: `ep.save(path)` then `ExportedProgram::load(path)`
  reproduces numerical output.
- [x] AC-8: `ep.to_json()` produces a string parseable by
  `parse_json_metadata`.

## Architecture

`DimSpec` is a non-exhaustive enum with two variants. The
`dynamic` / `dynamic_range` constructors are sugar for the
`Dynamic` branch; `is_dynamic` is a one-line predicate.

`InputSpec` wraps `Vec<DimSpec>` so a multi-dim tensor's shape can
mix static and dynamic dimensions. `all_static(&[1, 3, 224, 224])`
is the common case: every dim is `Static(_)`.

`ExportedProgram` holds the graph + weights + shapes. The
`state_dict` is `HashMap<String, Vec<f32>>` (limitation: f32 only;
models trained in f64 are converted on export, documented at the
struct's rustdoc and REQ-3). The `input_specs` enable a guarded
`run_with_guards` path (REQ-4/REQ-5).

`serialize` writes a 4-byte magic + version + state_dict-key count +
each (key, value-len, value-bytes) + input_shapes count + each
shape + input_specs + output_shape + the IR-graph bytes
(`IrGraph::serialize`). `deserialize` mirrors the format.

`to_json` produces a one-pass diagnostic string; `parse_json_metadata`
parses it. Both are explicitly non-canonical — the binary format is
the source of truth (documented above the functions).

`export` (`pub fn export<T, M: Module<T>>(module, example_inputs)`)
runs the module's forward in a closure suitable for `trace`, then
gathers the module's `named_parameters` into a state_dict, then
builds `ExportedProgram` with all-static `InputSpec`s derived from
the example shapes.

`export_with_dynamic_shapes` is the polymorphic variant: callers
supply per-input `InputSpec`s, and the returned program has
`input_specs` exactly matching what was passed in. The traced graph
is patched (via `symbolic::patch_reshape_for_symbolic_dims`) so any
`Reshape` whose target shape literally encodes the trace-time
symbolic dim becomes `-1` (infer-this-dim sentinel).

### Non-test production consumers

- `pub use export::{DimSpec, ExportedProgram,
  ExportedProgramMetadata, InputSpec, export,
  export_with_dynamic_shapes}` at
  `ferrotorch-jit/src/lib.rs:100`.
- `ferrotorch-serialize/src/onnx_export.rs` is the
  cross-crate consumer that consumes an `ExportedProgram` and
  emits ONNX — the `OnnxDimSpec` enum at
  `OnnxDimSpec in onnx_export.rs` and the `encode_value_info_dynamic`
  function at `onnx_export.rs` translate the JIT's
  `DimSpec::Dynamic` into ONNX's `dim_param` entries (CL-396).

## Parity contract

`parity_ops = []`. Export is structural; numerics are the union of
the graph's nodes (covered elsewhere).

Edge cases:

- **Single-input only**: `export` rejects `example_inputs.len() !=
  1` with a clear error (documented at the module level).
- **f32 state_dict only**: Models trained in f64 will have weights
  truncated to f32 on export. Documented at REQ-3.
- **Single-output only**: `ExportedProgram::run` requires
  `outputs.len() == 1`. Documented inline.

## Verification

Tests in `ferrotorch-jit/src/export.rs` `mod tests` cover: DimSpec
constructors, InputSpec all_static / has_dynamic_dims,
ExportedProgram run / run_with_guards / check_inputs (passing and
failing paths), serialize / deserialize round-trip, save / load
disk round-trip, to_json / parse_json_metadata round-trip,
export and export_with_dynamic_shapes end-to-end. The
`ferrotorch-serialize` crate has cross-crate tests exercising
`ExportedProgram` → ONNX.

Smoke command:

```bash
cargo test -p ferrotorch-jit --lib export:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum DimSpec` with `Static` / `Dynamic` variants + `dynamic` / `dynamic_range` / `is_dynamic` constructors in `export.rs`; non-test consumer: `ferrotorch-serialize/src/onnx_export.rs:805` matches on `DimSpec::Dynamic` to emit `OnnxDimSpec::Dynamic` in the ONNX export path. |
| REQ-2 | SHIPPED | impl: `pub struct InputSpec` + `new` / `all_static` / `has_dynamic_dims` / `rank` in `export.rs`; non-test consumer: `ExportedProgram::input_specs: Vec<InputSpec>` field consumed by `run_with_guards` and by `ferrotorch-serialize`. |
| REQ-3 | SHIPPED | impl: `pub struct ExportedProgram { graph, state_dict, input_shapes, input_specs, output_shape }` in `export.rs`; non-test consumer: `export in ferrotorch-serialize/src/onnx_export.rs` `// ExportedProgram -> ONNX` section. |
| REQ-4 | SHIPPED | impl: `pub fn run`, `pub fn run_with_guards` on `ExportedProgram` in `export.rs`; non-test consumer: re-export at `lib.rs:100`; ONNX export path round-trips through `.run` for end-to-end validation. |
| REQ-5 | SHIPPED | impl: `pub fn check_inputs` in `export.rs`; non-test consumer: `pub fn run_with_guards` invokes it (`self.check_inputs(inputs)?`). |
| REQ-6 | SHIPPED | impl: `pub fn serialize`, `pub fn deserialize` on `ExportedProgram` in `export.rs`; non-test consumer: `pub fn save` / `pub fn load` invoke them. |
| REQ-7 | SHIPPED | impl: `pub fn save`, `pub fn load` in `export.rs`; non-test consumer: re-export at `lib.rs`. |
| REQ-8 | SHIPPED | impl: `pub fn to_json`, `pub fn parse_json_metadata` in `export.rs`; non-test consumer: re-export at `lib.rs` plus `pub struct ExportedProgramMetadata` returned by the parser. |
| REQ-9 | SHIPPED | impl: `pub fn export<T, M: Module<T>>` in `export.rs`; non-test consumer: re-export at `lib.rs`; `ferrotorch-serialize::export_from_program` documented as consuming this path (`export in export.rs`, `export in export.rs`). |
| REQ-10 | SHIPPED | impl: `pub fn export_with_dynamic_shapes<T, M: Module<T>>` in `export.rs`; non-test consumer: re-export at `lib.rs:100`; `ferrotorch-serialize/src/onnx_export.rs:800-807` constructs `OnnxDimSpec::Dynamic` from the `ExportedProgram::input_specs` produced by this path. |
