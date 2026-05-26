# ferrotorch-jit — `serialize` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/jit/_serialization.py
  - torch/csrc/jit/serialization/import.cpp
  - torch/csrc/jit/serialization/export.cpp
-->

## Summary

`ferrotorch-jit/src/serialize.rs` provides the binary serialisation
format for `IrGraph`. It implements `IrGraph::serialize` and
`IrGraph::deserialize` using a hand-rolled little-endian binary
format with a magic header (`b"FTIR"`) and a version field
(currently `2`). Mirrors the role of
`torch/jit/_serialization.py:save` /
`torch/csrc/jit/serialization/import.cpp` — except where PyTorch
uses a ZIP archive of pickled tensors plus a TorchScript IR proto,
ferrotorch ships a single flat binary blob suited to its much
smaller IR surface (R-DEV-7: the Rust analog is materially
simpler).

## Requirements

- REQ-1: `impl IrGraph` block providing `pub fn serialize(&self) ->
  Vec<u8>` and `pub fn deserialize(data: &[u8]) ->
  FerrotorchResult<IrGraph>` as inherent methods on `IrGraph`.
- REQ-2: The on-disk format begins with magic bytes `b"FTIR"`
  followed by a `u32` version field. Magic / version mismatches on
  deserialize surface a clear `JitError::SerializationError`.
- REQ-3: The format encodes (in order): magic, version, value count
  + each value (`u32 id, shape, producer-node-id (-1 if None), dtype
  tag`), node count + each node (`u32 id, op tag + payload, inputs,
  outputs`), input value count + ids, output value count + ids.
- REQ-4: Format version `2` adds a single dtype tag byte after each
  value's `producer` field (`0` = `F32`, `1` = `F64`); v1 readers
  default unspecified values to `Dtype::F32`. The deserialiser
  accepts both versions.
- REQ-5: All multi-byte integers are little-endian; the
  serialiser is deterministic so two serialisations of the same
  graph yield byte-identical output (subject to topo-stable
  ordering of the graph's internal `nodes`/`values` vectors).
- REQ-6: `Writer` (internal) is the append-only byte-builder; the
  `Reader<'a>` (internal) is the bounds-checked cursor used by
  `deserialize`. Both are private to this module.

## Acceptance Criteria

- [x] AC-1: `serialize()` of a simple Input → Add → Relu →
  Output graph produces a non-empty byte vector starting with
  `b"FTIR\x02\x00\x00\x00"` (magic + version 2 LE).
- [x] AC-2: `deserialize(&bytes)` of the above byte vector
  reproduces an `IrGraph` whose `node_count` and `value_count`
  match the original.
- [x] AC-3: `serialize(deserialize(serialize(g))?)?` is byte-equal
  to `serialize(g)` (idempotent round-trip).
- [x] AC-4: Garbage bytes (e.g. `[0xFF, 0xFE, 0xFD]`) surface an
  error from `deserialize`, not a panic.
- [x] AC-5: A v1-formatted blob (no dtype byte) deserialises with
  every value tagged `Dtype::F32`.
- [x] AC-6: A v2-formatted blob containing `Dtype::F64` values
  round-trips dtype correctly.

## Architecture

The format is laid out as:

| Offset | Field | Type |
|--------|-------|------|
| 0 | Magic | `[u8; 4]` (= `b"FTIR"`) |
| 4 | Version | `u32` LE |
| 8 | Value count | `u32` |
| 12 | Value records (variable) | per-value: `u32 id, u32 rank, u32 dims[rank], i32 producer_node_id (-1 for input), u8 dtype_tag` |
| ... | Node count | `u32` |
| ... | Node records (variable) | per-node: `u32 id, u32 op_tag, payload (variable per op), u32 input_count, u32 inputs[], u32 output_count, u32 outputs[]` |
| ... | Input value count | `u32` |
| ... | Input value ids | `u32` each |
| ... | Output value count | `u32` |
| ... | Output value ids | `u32` each |

`Writer` is a `struct Writer { buf: Vec<u8> }` with `write_u8`,
`write_u32`, `write_i32`, `write_bytes` helpers. `Reader<'a>` is a
`struct Reader<'a> { buf: &'a [u8], pos: usize }` with bounds-checked
`read_u8`, `read_u32`, `read_i32`, `read_bytes` helpers that surface
`JitError::SerializationError` on under-run.

`IrGraph::serialize` walks `self.values` and `self.nodes` in order,
emitting each record. `IrGraph::deserialize` reverses the walk,
constructing a fresh `IrGraph` and (per v1) tagging values with
`Dtype::F32` when the dtype byte is absent.

The `IrOpKind` discriminant is encoded as a `u32` tag; payload-bearing
variants (`Constant`, `Pow`, `Reshape`, `Squeeze`, `Unsqueeze`,
`Cat`, `Input`, `FusedElementwise`, `FusedLinearActivation`,
`FusedAttention`) write the payload after the tag. The decoder
matches on the tag and reads the matching payload.

### Non-test production consumers

- `ferrotorch-jit/src/module.rs:146-153` —
  `TracedModule::to_bytes` calls `self.graph.serialize()`;
  `TracedModule::from_bytes` calls `IrGraph::deserialize(data)?`.
- `ferrotorch-jit/src/export.rs:260-407` —
  `ExportedProgram::serialize` uses `IrGraph::serialize` for the
  IR-graph payload inside its outer header;
  `ExportedProgram::deserialize` at `export.rs:407` calls
  `IrGraph::deserialize(graph_bytes)?`.

## Parity contract

`parity_ops = []`. Serialisation is structural; byte equivalence is
the contract.

Edge cases:

- **Magic mismatch** — error citing the magic bytes seen vs
  expected.
- **Version mismatch beyond v2** — error citing the version.
- **Truncated buffer** — `Reader` bounds-check surfaces an error
  with the byte offset.
- **Unknown op-tag** — error citing the tag value, suggesting the
  format may be from a newer ferrotorch version.

## Verification

Tests in `ferrotorch-jit/src/serialize.rs` `mod tests`:
`test_round_trip_simple_graph` and others covering each op-payload
shape, the v1-fallback dtype path, and garbage-input error paths.

The cross-module round-trip is also tested via
`module.rs`'s `test_traced_module_to_bytes_from_bytes_roundtrip`
and `test_traced_module_save_load_disk_roundtrip`.

Smoke command:

```bash
cargo test -p ferrotorch-jit --lib serialize:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `impl IrGraph { pub fn serialize, pub fn deserialize }` in `serialize.rs`; non-test consumer: `ferrotorch-jit/src/module.rs:146` `pub fn to_bytes` calls `self.graph.serialize()` and `module.rs:152` `from_bytes` calls `IrGraph::deserialize(data)?`. |
| REQ-2 | SHIPPED | impl: `b"FTIR"` magic write + version `u32` LE in `serialize.rs::serialize`; magic/version check in `serialize.rs::deserialize` with `JitError::SerializationError`; non-test consumer: every `TracedModule::to_bytes` call writes through this path. |
| REQ-3 | SHIPPED | impl: the `Writer` helper sequence in `serialize.rs::serialize` writes value records then node records then input/output ids; non-test consumer: `module.rs:146` and `export.rs:260-407` are production callers. |
| REQ-4 | SHIPPED | impl: v2 dtype tag byte after each value's `producer` field; v1-compat path inside `serialize.rs::deserialize` (defaults to `Dtype::F32`); non-test consumer: every `IrGraph::deserialize` call in `module.rs` and `export.rs`. |
| REQ-5 | SHIPPED | impl: little-endian writes throughout `serialize.rs`; deterministic walk over `self.values` and `self.nodes`; non-test consumer: the format contract is the same one `module.rs:146` depends on. |
| REQ-6 | SHIPPED | impl: private `Writer` and `Reader<'a>` structs in `serialize.rs:120-180`; non-test consumer: `serialize` and `deserialize` use these internally. |
