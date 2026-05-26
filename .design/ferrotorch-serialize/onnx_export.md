# ferrotorch-serialize — `onnx_export` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/serialization.py
  - torch/_weights_only_unpickler.py
-->

## Summary

`ferrotorch-serialize/src/onnx_export.rs` (2105 LOC) exports a
ferrotorch-traced `IrGraph` (from `ferrotorch-jit`) as an ONNX
[ModelProto](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto)
binary file. The implementation includes a hand-written protobuf
wire-format encoder (varint / fixed32 / fixed64 / length-delimited),
ONNX message builders (`OperatorSetIdProto`, `TensorShapeProto`,
`ValueInfoProto`, `TensorProto`, `AttributeProto`, `NodeProto`,
`GraphProto`, `ModelProto`), an `IrOpKind` → ONNX op-type mapper,
and the topological-order walk that emits the ONNX graph from the
IR. The module is fully self-contained: no `protoc`, no
code-generated stubs, no `prost` / `protobuf` dependency.

Mirrors `torch.onnx.export(model, args, path, opset_version=17,
dynamic_axes=...)` in `torch/onnx/__init__.py:65`. The ferrotorch
path traces a Rust closure via `ferrotorch_jit::trace`, walks the
resulting `IrGraph`, and emits ONNX bytes — semantically the same
as upstream's "trace the model + emit ONNX" but built bottom-up in
Rust (R-DEV-7: hand-written protobuf is cleaner than the
`prost-build` setup for the narrow set of ONNX messages we emit).

## Requirements

- REQ-1: `#[non_exhaustive] pub struct OnnxExportConfig` with `pub
  opset_version: usize`, `pub model_name: String`, `pub
  dynamic_axes: HashMap<usize, Vec<(usize, String)>>`. Defaults
  (`impl Default`): opset_version = 17, model_name =
  "ferrotorch_model", dynamic_axes = empty. The struct is
  `#[non_exhaustive]` so new export options (FP16 weights,
  optimization passes, custom op-domain mappings) land without
  breaking caller struct-literal initializers.

- REQ-2: Protobuf wire-format encoder. `struct ProtobufWriter` ships
  the wire types ONNX uses:
  - varint (wire type 0) via `write_varint(u64)`,
  - 32-bit fixed (wire type 5) for `float`,
  - 64-bit fixed (wire type 1) for `double`,
  - length-delimited (wire type 2) for `bytes`, `string`, embedded
    messages.

  Plus high-level helpers: `write_tag(field, wire_type)`,
  `write_uint64(field, v)`, `write_int64(field, v)`,
  `write_int32(field, v)`, `write_string(field, s)`,
  `write_bytes(field, b)`, `write_message(field, msg_bytes)`,
  `write_float`, `write_double`. The `#[allow(dead_code)]` on
  several reserved methods (`write_bytes`, lower-level numeric
  writers) is justified inline: they're kept implemented so the
  protobuf encoder is a complete reusable component for future
  ONNX features (training-info graphs, quantized tensor protos,
  sparse weights).

- REQ-3: ONNX dtype enum dispatch. `const ONNX_FLOAT: i32 = 1`,
  `const ONNX_DOUBLE: i32 = 11`. `fn onnx_dtype<T: Float>() ->
  FerrotorchResult<i32>` dispatches on `size_of::<T>()`; anything
  other than f32/f64 errors. These constants match the ONNX
  `TensorProto.DataType` enum exactly.

- REQ-4: Protobuf message builders. One function per ONNX message
  type:
  - `fn encode_opset(domain, version) -> Vec<u8>` →
    `OperatorSetIdProto`.
  - `fn encode_dim(value) -> Vec<u8>` →
    `TensorShapeProto.Dimension` with `dim_value`.
  - `pub fn encode_dim_param(param) -> Vec<u8>` → same with
    `dim_param` (symbolic name).
  - `fn encode_shape(dims)` and `pub fn encode_shape_with_dynamic(
    dims: &[OnnxDimSpec])` → `TensorShapeProto`.
  - `fn encode_tensor_type` → `TypeProto.Tensor`.
  - `fn encode_type_proto` → `TypeProto`.
  - `fn encode_value_info` / `pub fn encode_value_info_dynamic`
    → `ValueInfoProto`.
  - `fn encode_tensor_proto` → `TensorProto` (initializer with raw
    data bytes).
  - `fn encode_attr_int` / `fn encode_attr_ints` →
    `AttributeProto`.
  - `fn encode_node` → `NodeProto`.
  - `fn encode_graph` → `GraphProto`.
  - `fn encode_model` → `ModelProto`.

- REQ-5: Dynamic-axes encoding. `pub enum OnnxDimSpec { Static(
  usize), Dynamic(String) }` and `pub fn encode_shape_with_dynamic(
  dims: &[OnnxDimSpec]) -> Vec<u8>` + `pub fn
  encode_value_info_dynamic(name, elem_type, dims)`. Used when the
  caller specifies `dynamic_axes` so a model accepts variable
  batch / sequence sizes at inference time — critical for
  serving transformer models that need to run on arbitrary
  prompt lengths.

- REQ-6: `IrOpKind` → ONNX op mapper. `fn map_ir_op(op,
  node_name, elem_type) -> FerrotorchResult<OpMapping>` covers
  every supported `IrOpKind`: arithmetic (`Add`, `Sub`, `Mul`,
  `Div`), unary (`Neg`, `Sqrt`, `Exp`, `Log`, `Sigmoid`,
  `Tanh`, `Relu`, `Erf`, `Sin`, `Cos`), reductions (`Sum`,
  `Mean`, `Max`, `Min`), linear (`MatMul`, `MatMulT` →
  `Gemm{transB=1}`), shape ops (`Reshape`, `Transpose`,
  `Permute`, `Squeeze`, `Unsqueeze`, `Concat`), comparisons
  (`Eq`, `Lt`, `Gt`), control (`Where`), more. Unsupported
  ops (`FusedElementwise`, `FusedLinearActivation`,
  `FusedAttention`, `Cond`, `Scan`) error with the "must be
  lowered before ONNX export" message naming the op.

- REQ-7: Op decomposition for portability. Some ferrotorch ops
  don't have a single ONNX op-type so they're decomposed:
  - `IrOpKind::Silu` → `Sigmoid` + `Mul` (portable to opset
    >= 13).
  - `IrOpKind::Gelu` → `Div / Erf / Add / Mul / Mul` using
    `y = x * 0.5 * (1 + Erf(x / sqrt(2)))` with the three
    constants (sqrt(2), 0.5, 1.0) emitted as initializers.
    CL-375.

- REQ-8: Graph topological walk. `pub fn ir_graph_to_onnx(graph,
  config, elem_type) -> FerrotorchResult<Vec<u8>>` walks the IR
  in topological order:
  1. Iterate `graph.topological_order()`.
  2. For each node: dispatch on `IrOpKind::Input` (emit
     ValueInfoProto + record as graph input), `IrOpKind::Constant`
     (emit TensorProto initializer + matching ValueInfoProto),
     `IrOpKind::Output` (skip — handled separately), or any
     other op (call `map_ir_op` + emit NodeProto).
  3. Auxiliary initializers (e.g. `Reshape`'s shape tensor as
     INT64) are emitted as int64 TensorProtos.
  4. Graph outputs are emitted from `graph.output_values`.
  5. Final ModelProto wraps the GraphProto + the opset import +
     IR version 8.

- REQ-9: `pub fn export_onnx<T: Float>(trace_fn, example_inputs,
  path, config) -> FerrotorchResult<()>` is the highest-level
  entry: traces the closure via `ferrotorch_jit::trace`, runs
  `ir_graph_to_onnx`, writes the bytes to disk. Validates
  `config.opset_version >= 17` (anything older lacks the ops
  ferrotorch emits).

- REQ-10: `pub fn export_ir_graph_to_onnx(graph, path, config,
  elem_type) -> FerrotorchResult<()>` exports an already-traced
  graph without re-tracing. Same opset validation.

- REQ-11: `pub fn export_from_program(program, path, config) ->
  FerrotorchResult<()>` exports an `ExportedProgram` (a graph +
  state-dict bundle from `ferrotorch_jit::export`). Merges the
  program's `input_specs` dynamic-axes spec into the caller's
  `config.dynamic_axes` (caller entries win on conflict per CL-396).
  Uses `ONNX_FLOAT` as the element type since the current
  `ExportedProgram` state dict is f32-only.

- REQ-12: Output file write. Each public entry writes the encoded
  ONNX bytes to the caller's `path` via `std::fs::write` and
  surfaces I/O errors as `InvalidArgument` with the file path in
  the message. `#[allow(clippy::needless_pass_by_value)]` on
  `OnnxExportConfig` parameters is justified inline: the config
  has owned `String` + `HashMap` fields and callers construct a
  fresh one per call, so by-value is the natural ownership.

## Acceptance Criteria

- [x] AC-1: `export_onnx` produces a file that
  [onnx.checker.check_model](https://onnx.ai/onnx/api/checker.html)
  accepts (covered by the Python inter-op fixture in
  `tests/conformance_serialize.rs` when ONNX is available).
- [x] AC-2: Every supported `IrOpKind` round-trips through the
  mapper.
- [x] AC-3: Silu and Gelu decomposition emits the expected
  sub-node sequence (covered by inline tests in `mod tests`).
- [x] AC-4: `dynamic_axes` produces `dim_param` ValueInfoProtos
  with the caller-supplied symbolic names.
- [x] AC-5: `export_from_program` merges program-level dynamic
  dims with caller-supplied ones, caller wins on conflict.
- [x] AC-6: Unsupported opset_version (<17) is rejected
  pre-trace.
- [x] AC-7: Unsupported `IrOpKind` (`FusedElementwise`, etc.)
  errors with the "must be lowered" message.
- [x] AC-8: A complete MLP traced + exported can be loaded by
  ONNX Runtime (covered by the inter-op fixture when ORT is
  available).

## Architecture

### Config (REQ-1)

```rust
#[derive(Debug)]
#[non_exhaustive]
pub struct OnnxExportConfig {
    pub opset_version: usize,
    pub model_name: String,
    pub dynamic_axes: HashMap<usize, Vec<(usize, String)>>,
}
```

`dynamic_axes` maps input index → list of `(axis, symbolic_name)`.
The default is opset 17 because that's the first opset with full
support for the ferrotorch op set (`Erf`, `Sigmoid`, `Gemm`,
modern broadcasting semantics).

### Protobuf encoder (REQ-2)

```rust
struct ProtobufWriter { buf: Vec<u8> }
impl ProtobufWriter {
    fn write_varint(&mut self, mut v: u64) {
        loop {
            let b = (v & 0x7F) as u8;
            v >>= 7;
            if v == 0 { self.buf.push(b); return; }
            self.buf.push(b | 0x80);
        }
    }
    fn write_tag(&mut self, field: u32, wire_type: u32) {
        self.write_varint((field as u64) << 3 | wire_type as u64);
    }
    fn write_string(&mut self, field: u32, s: &str) {
        self.write_tag(field, 2);  // wire type 2 = length-delimited
        self.write_varint(s.len() as u64);
        self.buf.extend_from_slice(s.as_bytes());
    }
    fn write_message(&mut self, field: u32, msg: &[u8]) {
        self.write_tag(field, 2);
        self.write_varint(msg.len() as u64);
        self.buf.extend_from_slice(msg);
    }
    // ... write_float, write_double, write_int32, write_int64, write_uint64, write_bytes
}
```

The encoder is ~100 LOC and covers exactly the wire types ONNX
uses. The `#[allow(dead_code)]` at the impl level keeps the
unused-but-reserved methods (`write_bytes`, etc.) compiled.

### ONNX dtype constants (REQ-3)

The values match the ONNX spec:
[`onnx/onnx.proto`
TensorProto.DataType](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto).
ferrotorch's `Float` trait is two impls (f32, f64); the dispatch is
a `size_of` match.

### Message builders (REQ-4)

Each `encode_*` function constructs a fresh `ProtobufWriter`,
writes the fields in protobuf-tag order, and returns the
serialized bytes. The functions compose: `encode_graph` calls
`encode_node`, `encode_value_info`, `encode_tensor_proto`, etc. on
the things it's wrapping.

The protobuf field constants (`OPSET_VERSION`, `SHAPE_DIM`,
`TENSOR_DIMS`, `TENSOR_DATA_TYPE`, etc.) are private to this
module — copied from the `onnx.proto` spec.

### Dynamic axes (REQ-5)

```rust
pub enum OnnxDimSpec { Static(usize), Dynamic(String) }

pub fn encode_shape_with_dynamic(dims: &[OnnxDimSpec]) -> Vec<u8> {
    let mut w = ProtobufWriter::new();
    for dim in dims {
        let dim_bytes = match dim {
            OnnxDimSpec::Static(v) => encode_dim(*v as u64),
            OnnxDimSpec::Dynamic(name) => encode_dim_param(name),
        };
        w.write_message(SHAPE_DIM, &dim_bytes);
    }
    w.into_bytes()
}
```

When the caller's config sets `dynamic_axes[input_idx]`, the
walker (REQ-8) emits the input's shape via
`encode_value_info_dynamic` instead of `encode_value_info`.

### Op mapping (REQ-6, REQ-7)

`map_ir_op` returns `OpMapping { op_type: &'static str,
attributes: Vec<Vec<u8>>, aux_initializer: Option<(name, raw,
dims)> }`. Most ops map 1-to-1: `Add → "Add"`, `MatMul →
"MatMul"`, etc. A few need attributes (`Transpose` carries
`perm` as an `ints` attribute, `Gemm` carries `transA`/`transB` as
`int` attrs, `Reshape` carries the target shape as an int64
auxiliary initializer).

`Silu` and `Gelu` are decomposed inline in `ir_graph_to_onnx`
(the walker handles them as special cases) because they need to
emit MULTIPLE NodeProtos, which doesn't fit the single-mapping
return type.

### Topological walk (REQ-8)

```rust
for &nid in &topo {
    let node = node_map[&nid];
    match &node.op {
        IrOpKind::Input { .. } => onnx_inputs.push(encode_value_info(...)),
        IrOpKind::Constant { data, shape } => {
            onnx_initializers.push(encode_tensor_proto(...));
            onnx_inputs.push(encode_value_info(...));
        }
        IrOpKind::Output => /* skip */,
        IrOpKind::Silu => /* emit Sigmoid + Mul */,
        IrOpKind::Gelu => /* emit Div + Erf + Add + Mul + Mul */,
        op => {
            let mapping = map_ir_op(op, &node_name, elem_type)?;
            // emit auxiliary initializer if present, then NodeProto
        }
    }
}
```

The walker is the load-bearing piece. The two-pass-style approach
(emit initializers + inputs + nodes into separate `Vec<Vec<u8>>`,
then assemble) lets us produce ONNX's required structural order
(initializers + inputs precede nodes in the GraphProto).

### Public entries (REQ-9, REQ-10, REQ-11)

```rust
pub fn export_onnx<T: Float>(trace_fn, example_inputs, path, config) {
    if config.opset_version < 17 { return err; }
    let elem_type = onnx_dtype::<T>()?;
    let graph = trace(trace_fn, example_inputs)?;
    let onnx_bytes = ir_graph_to_onnx(&graph, &config, elem_type)?;
    std::fs::write(path, &onnx_bytes)?;
    Ok(())
}

pub fn export_ir_graph_to_onnx(graph, path, config, elem_type) { ... }

pub fn export_from_program(program, path, mut config) {
    // Merge program.input_specs.shape dynamic dims into config.dynamic_axes,
    // caller wins on conflict (CL-396).
    let onnx_bytes = ir_graph_to_onnx(&program.graph, &config, ONNX_FLOAT)?;
    std::fs::write(path, &onnx_bytes)?;
    Ok(())
}
```

### Non-test production consumers

- `pub use onnx_export::{OnnxExportConfig, OnnxDimSpec,
  export_onnx, export_ir_graph_to_onnx, export_from_program,
  ir_graph_to_onnx, encode_dim_param,
  encode_shape_with_dynamic, encode_value_info_dynamic,
  DIM_PARAM}` in `lib.rs`, reachable through the meta-crate
  glob.
- Production export paths in user code call
  `ferrotorch::export_onnx(model_fn, inputs, "out.onnx",
  config)?` to ship a model to ONNX Runtime / TensorRT / CoreML.
- `export_from_program` is the path AOT-compiled
  `ExportedProgram` bundles use to ship — it preserves the
  dynamic-axes contract through the JIT export hand-off.

## Parity contract

`parity_ops = []`. The byte format is the ONNX spec
(R-DEV-3 external). Parity gates:

- **Output passes `onnx.checker.check_model`**: covered by the
  inter-op fixture.
- **ONNX Runtime executes a traced ferrotorch MLP**: the
  cross-validation fixture in `tests/conformance_serialize.rs`
  runs ferrotorch forward + ONNX Runtime forward on the same
  inputs and asserts the outputs are within fp32 tolerance.
- **Dynamic axes appear as `dim_param`**: a model exported with
  `dynamic_axes = {0: [(0, "batch")]}` has ValueInfoProtos with
  `dim_param: "batch"` on input 0's axis 0.
- **Opset version validation**: any value <17 errors before
  tracing.
- **Silu / Gelu decomposition**: emit the expected sub-node
  sequence for opset 17+ portability.

## Verification

Tests in `mod tests in onnx_export.rs` (29 unit tests) covering:

- Protobuf wire-format primitives (varint, tag, fixed32, fixed64).
- Each ONNX message builder against a known-good byte sequence.
- Every `IrOpKind` arm of `map_ir_op`.
- `OnnxDimSpec` + dynamic-axes encoding.
- Silu / Gelu decomposition.
- Full traced-MLP export end-to-end.
- `ExportedProgram` → ONNX flow.

Integration tests in `tests/conformance_serialize.rs` validate
against `onnx.checker.check_model` and ONNX Runtime when those
are available.

Smoke command:

```bash
cargo test -p ferrotorch-serialize --lib onnx_export:: 2>&1 | tail -3
```

Expected: 29 passed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `#[non_exhaustive] pub struct OnnxExportConfig` + `impl Default` in `onnx_export.rs`; non-test consumer: `pub use onnx_export::OnnxExportConfig` in `lib.rs`; every public export entry takes `OnnxExportConfig` by value, and production model crates construct configs in their save paths. |
| REQ-2 | SHIPPED | impl: `struct ProtobufWriter` + `impl ProtobufWriter` with varint / fixed32 / fixed64 / length-delimited writers in `onnx_export.rs`; non-test consumer: every ONNX message builder (`encode_node`, `encode_graph`, etc.) constructs a `ProtobufWriter` to assemble its byte sequence — the encoder is the production substrate for every byte the module writes. |
| REQ-3 | SHIPPED | impl: `const ONNX_FLOAT: i32 = 1` + `const ONNX_DOUBLE: i32 = 11` + `fn onnx_dtype<T: Float>` in `onnx_export.rs`; non-test consumer: `export_onnx`, `export_ir_graph_to_onnx`, and `export_from_program` all resolve the element type via `onnx_dtype` (or the `ONNX_FLOAT` constant directly for `export_from_program`). |
| REQ-4 | SHIPPED | impl: 13 `encode_*` functions in `onnx_export.rs` covering every ONNX message ferrotorch emits; non-test consumer: `ir_graph_to_onnx` composes them — every encoded ONNX file is built from these primitives. |
| REQ-5 | SHIPPED | impl: `pub enum OnnxDimSpec` + `pub fn encode_dim_param` + `pub fn encode_shape_with_dynamic` + `pub fn encode_value_info_dynamic` + `pub const DIM_PARAM: u32 = 2` in `onnx_export.rs`; non-test consumer: `ir_graph_to_onnx` calls `encode_value_info_dynamic` whenever `config.dynamic_axes.get(&input_counter)` is `Some`. |
| REQ-6 | SHIPPED | impl: `fn map_ir_op(op, node_name, elem_type) -> FerrotorchResult<OpMapping>` in `onnx_export.rs` covering every supported `IrOpKind`; non-test consumer: `ir_graph_to_onnx` calls `map_ir_op` for every non-special-cased `IrOpKind` in the topological walk. |
| REQ-7 | SHIPPED | impl: `IrOpKind::Silu` and `IrOpKind::Gelu` arms in the `ir_graph_to_onnx` match in `onnx_export.rs` emitting multi-node decompositions; non-test consumer: every production trace that includes a Silu / Gelu (LLM models, transformers) hits these arms during export. |
| REQ-8 | SHIPPED | impl: `pub fn ir_graph_to_onnx(graph, config, elem_type)` in `onnx_export.rs` with the topological walk + initializer/node assembly + final ModelProto wrapper; non-test consumer: `pub use onnx_export::ir_graph_to_onnx` in `lib.rs`; `export_onnx`, `export_ir_graph_to_onnx`, and `export_from_program` all delegate to `ir_graph_to_onnx`. |
| REQ-9 | SHIPPED | impl: `pub fn export_onnx<T: Float>(trace_fn, example_inputs, path, config)` in `onnx_export.rs` doing opset check + trace + `ir_graph_to_onnx` + `fs::write`; non-test consumer: `pub use onnx_export::export_onnx` in `lib.rs`; production export paths in user code call `ferrotorch::export_onnx(...)`. |
| REQ-10 | SHIPPED | impl: `pub fn export_ir_graph_to_onnx(graph, path, config, elem_type)` in `onnx_export.rs`; non-test consumer: `pub use onnx_export::export_ir_graph_to_onnx` in `lib.rs`; consumers that already have a hand-built `IrGraph` (e.g. test fixtures, model-converter tools) skip the trace step via this entry. |
| REQ-11 | SHIPPED | impl: `pub fn export_from_program(program, path, mut config)` in `onnx_export.rs` merging `program.input_specs` dynamic dims into `config.dynamic_axes` per CL-396; non-test consumer: `pub use onnx_export::export_from_program` in `lib.rs`; the AOT-compile path in `ferrotorch-jit::export::ExportedProgram` consumers calls this entry. |
| REQ-12 | SHIPPED | impl: `std::fs::write(path, &onnx_bytes)` in `export_onnx`, `export_ir_graph_to_onnx`, and `export_from_program` in `onnx_export.rs`, each surfacing I/O errors as `InvalidArgument` with the path in the message + `#[allow(clippy::needless_pass_by_value)]` justifications; non-test consumer: every production caller writes ONNX bytes to disk via these three entries. |
