# ferrotorch-serialize — `state_dict` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/serialization.py
  - torch/_weights_only_unpickler.py
-->

## Summary

`ferrotorch-serialize/src/state_dict.rs` implements a self-contained
JSON-header + little-endian binary-body file format for
`StateDict<T>` (`HashMap<String, Tensor<T>>`). It is the simplest of
the seven serialization paths and the only one with NO upstream
on-disk parity contract — the format is ferrotorch-native. It exists
as the substrate for `checkpoint.rs` (which embeds the same byte
layout inside its length-prefixed sections) and as a developer-mode
"just dump tensors to disk" path for tests and debugging.

The conceptual upstream is `torch.save(state_dict, path)` in
`torch/serialization.py:945` — same use case (snapshot a parameter
dict to disk) but ferrotorch picks a hand-written line-oriented
header instead of pickle for security + auditability (R-DEV-4: Rust
eliminates the Python pickle attack surface). Cross-format
interchange with PyTorch happens through `safetensors_io.rs` and
`pytorch_export.rs` instead.

## Requirements

- REQ-1: `pub fn save_state_dict<T: Float>(state: &StateDict<T>,
  path: impl AsRef<Path>) -> FerrotorchResult<()>` writes the state
  dict to disk. Keys are sorted alphabetically for deterministic
  byte output. The file layout is: one JSON-encoded `TensorMeta` per
  line (`{"name":...,"shape":[...],"dtype":...,"byte_offset":N,
  "byte_length":N}`), then the `\n---\n` separator, then the raw
  little-endian bytes for each tensor concatenated in key order.

- REQ-2: `pub fn load_state_dict<T: Float>(path: impl AsRef<Path>)
  -> FerrotorchResult<StateDict<T>>` reads the format back. It
  parses the header line-by-line until it hits `---`, then reads
  the binary body. Each tensor's bytes are decoded via
  `f32::from_le_bytes` or `f64::from_le_bytes` and the resulting
  value is `numeric_cast`'d into `T` (so loading an `f32`-stored
  file into `StateDict<f64>` works; loading `f64` into `f32`
  errors via `cast` if a value isn't representable).

- REQ-3: Dtype tag dispatch. `pub(crate) fn dtype_tag<T: Float>() ->
  &'static str` returns `"f32"` / `"f64"` / `"unknown"` based on
  `size_of::<T>()`. Used in BOTH the save and load paths AND by the
  `checkpoint.rs` embedded state-dict section so all three paths
  agree on the wire string.

- REQ-4: Hand-written JSON metadata parser. `pub(crate) fn
  parse_meta_line(line: &str) -> FerrotorchResult<TensorMeta>` does
  a minimal serde-free extraction of the five JSON keys: `name`
  (string), `shape` (array of usize), `dtype` (string),
  `byte_offset` (usize), `byte_length` (usize). The parser is
  intentionally minimal — it rejects malformed input with a
  structured `FerrotorchError::InvalidArgument` rather than
  panicking, but does NOT handle JSON niceties like escape
  sequences or unicode escapes (the header writer never emits
  those).

- REQ-5: Endianness contract. The file format assumes little-endian
  byte order — every byte slice is reinterpreted directly as
  `&[u8]` via `slice::from_raw_parts` rather than being converted
  per-element. A `compile_error!` at the top of the file rejects
  big-endian targets at compile time so the assumption is
  mechanically enforced.

- REQ-6: Shape + dtype validation on load. Each tensor's
  `expected_numel = shape.iter().product()` is cross-checked
  against `byte_length / size_of::<T>()`; mismatch is
  `FerrotorchError::ShapeMismatch`. Dtype tag mismatch is
  `FerrotorchError::DtypeMismatch`. Truncated body (a tensor's
  range extends beyond `body.len()`) is
  `FerrotorchError::InvalidArgument`.

- REQ-7: Edge cases preserved: 0-rank scalars (`shape: []`,
  `numel: 1`), empty state dicts (zero header lines, then
  `\n---\n`, then zero body bytes), arbitrarily deep
  hierarchical key names (`conv.weight`, `transformer.layer.0.q`).

## Acceptance Criteria

- [x] AC-1: `save_state_dict` writes deterministic byte output
  for sorted keys (`test_deterministic_ordering`).
- [x] AC-2: `load_state_dict` recovers the saved data for f32
  and f64 (`test_save_load_roundtrip_f32` / `_f64`).
- [x] AC-3: `parse_meta_line` parses both standard and empty-shape
  metadata lines (`test_parse_meta_line` /
  `test_parse_meta_line_empty_shape`).
- [x] AC-4: A dtype mismatch produces `FerrotorchError::DtypeMismatch`
  (`test_dtype_mismatch`).
- [x] AC-5: Missing files return an error with the
  `"failed to open file"` prefix (`test_load_missing_file`).
- [x] AC-6: Empty state dicts round-trip (`test_empty_state_dict`).
- [x] AC-7: Scalar (0-rank) tensors round-trip
  (`test_shape_preservation_scalar`).
- [x] AC-8: High-rank (4D) tensors round-trip with byte order
  preserved (`test_shape_preservation_high_rank`).

## Architecture

### Save path (REQ-1)

`save_state_dict` does three passes over the sorted keys:

1. Compute `byte_offset` for every tensor (`elem_size * numel`
   cumulative) AND build the header JSON line.
2. Write each header line + newline to the file.
3. Write the `\n---\n` separator, then iterate keys again and
   reinterpret each tensor's `&[T]` data slice as `&[u8]` via
   `slice::from_raw_parts(data.as_ptr().cast::<u8>(),
   size_of_val(data))` and write the bytes verbatim.

The `unsafe` block carries a documented `SAFETY:` comment that names
the four invariants — `T` is one of f32/f64/bf16 (POD, Copy, no
padding, no Drop), the byte length equals `data.len() *
size_of::<T>()`, the borrow is reseated by `from_raw_parts` so the
returned `&[u8]` shares the lifetime of `&[T]`, and the crate-level
`compile_error!` forbids big-endian targets.

### Load path (REQ-2)

`load_state_dict` reads line-by-line until it hits the trimmed
`---` separator. Each line is parsed via `parse_meta_line` (REQ-4).
The dtype is validated against `dtype_tag::<T>()` (REQ-3); a
mismatch errors immediately. After the header loop, the rest of the
file is `read_to_end`'d into `body: Vec<u8>`, and each tensor's
bytes are sliced out + decoded:

```rust
for chunk in byte_slice.chunks_exact(elem_size) {
    let value: T = match elem_size {
        4 => cast::<f32, T>(f32::from_le_bytes(arr))?,
        8 => cast::<f64, T>(f64::from_le_bytes(arr))?,
        other => Err(FerrotorchError::InvalidArgument { ... })?,
    };
}
```

The `numeric_cast::cast` call handles the cross-dtype case (loading
a wider dtype into a narrower `Float` impl) — it returns
`Err(DtypeMismatch)` if the source value isn't representable.

### dtype_tag dispatch (REQ-3)

`pub(crate) fn dtype_tag<T: Float>() -> &'static str` is a
size-of dispatch: `4 → "f32"`, `8 → "f64"`, anything else →
`"unknown"`. The function is called from BOTH this module's save +
load paths AND from `checkpoint.rs::serialize_state_dict_to_bytes`
and `checkpoint.rs::deserialize_state_dict_from_bytes`, which is
why it's `pub(crate)` rather than `pub`.

### Header parser (REQ-4)

`parse_meta_line` uses three closures: `extract_string("name")`,
`extract_usize("byte_offset")`, `extract_shape()`. Each builds a
search pattern (`"name":"` etc.), finds the substring start /
end, and parses. The `shape` closure splits the comma-separated
array contents and parses each element as `usize`. All failures
produce `FerrotorchError::InvalidArgument` with the malformed line
quoted in the message.

The parser does not handle JSON escape sequences (`\"`, `\\`,
`\n`); ferrotorch state-dict files never have escaped characters
in tensor names (the writer doesn't emit any), so the simpler
parser is correct for this format. A user who hand-edits the
header could in principle introduce escapes; we treat that as
malformed input and return an error rather than try to handle it.

### Endianness (REQ-5)

```rust
#[cfg(not(target_endian = "little"))]
compile_error!(
    "ferrotorch state dict serialization assumes little-endian byte order. \
     Big-endian platforms are not supported."
);
```

The save path uses raw byte reinterpretation rather than
per-element `to_le_bytes`; the compile-time check is what makes
that sound on the only platforms we support (x86, ARM).

### Validation on load (REQ-6)

Three checks fire in `load_state_dict`:

1. `if end > body.len()` → InvalidArgument (truncated body).
2. `if numel * elem_size != meta.byte_length` → InvalidArgument
   (corrupt metadata: byte_length doesn't divide evenly).
3. `if expected_numel != numel` → ShapeMismatch (declared shape's
   product doesn't equal element count).

### Non-test production consumers

- `crate::checkpoint::deserialize_state_dict_from_bytes` calls
  `crate::state_dict::parse_meta_line` and
  `crate::state_dict::dtype_tag::<T>` to share the same on-disk
  format inside the checkpoint's embedded section. This is a
  same-crate cross-module consumer — visible in
  `checkpoint.rs` at the imports and call sites.
- `ferrotorch_serialize::{save_state_dict, load_state_dict}` is
  re-exported through `lib.rs` and reaches end-users via the
  meta-crate glob (`ferrotorch::save_state_dict`).
- Documented in `.design/phase-3-optim-serialize.md` as the dev-mode
  snapshot path for unit tests; not the production user-facing
  format (which is safetensors).

## Parity contract

`parity_ops = []`. This module has no PyTorch-side counterpart at
the bit level — it's a ferrotorch-native format. The closest
upstream analog is `torch.save(state_dict, path)` which uses pickle
inside a zip; ferrotorch users who need that exact format use
`pytorch_export.rs` instead.

Edge-case contract this module ships:

- Empty state dict: header is empty, file is exactly the
  `\n---\n` bytes (5 bytes). `test_empty_state_dict`.
- Scalar (0-rank) tensor: shape array is `[]`, `byte_length` is
  one element. `test_shape_preservation_scalar`.
- 4D conv weight: byte order preserved through high-rank shapes.
  `test_shape_preservation_high_rank`.
- Dtype mismatch on load: f32-saved file loaded as f64 errors with
  `FerrotorchError::DtypeMismatch` rather than producing garbage.
  `test_dtype_mismatch`.
- Missing file: `FerrotorchError::InvalidArgument` with the
  `"failed to open file"` prefix. `test_load_missing_file`.

## Verification

Tests in `mod tests in state_dict.rs` (10 tests):

- Roundtrip: `test_save_load_roundtrip_f64`,
  `test_save_load_roundtrip_f32`.
- Failure modes: `test_load_missing_file`, `test_dtype_mismatch`.
- Shape edge cases: `test_shape_preservation_scalar`,
  `test_shape_preservation_high_rank`, `test_empty_state_dict`.
- Header parser: `test_parse_meta_line`,
  `test_parse_meta_line_empty_shape`.
- Determinism: `test_deterministic_ordering`.

Smoke command:

```bash
cargo test -p ferrotorch-serialize --lib state_dict:: 2>&1 | tail -3
```

Expected: 10 passed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn save_state_dict<T: Float>` in `state_dict.rs` writes sorted header lines + `\n---\n` + raw LE body bytes; non-test consumer: `pub use state_dict::save_state_dict` in `lib.rs` and the meta-crate glob makes it available as `ferrotorch::save_state_dict`; `crate::checkpoint::serialize_state_dict_to_bytes` reuses the same byte layout via `crate::state_dict::dtype_tag` for the embedded section. |
| REQ-2 | SHIPPED | impl: `pub fn load_state_dict<T: Float>` in `state_dict.rs` parses the header line-by-line, validates dtype/shape, and decodes LE bytes via `f{32,64}::from_le_bytes` + `numeric_cast::cast`; non-test consumer: `pub use state_dict::load_state_dict` in `lib.rs`; downstream training-resume code reads checkpoints via this entry. |
| REQ-3 | SHIPPED | impl: `pub(crate) fn dtype_tag<T: Float>() -> &'static str` in `state_dict.rs` dispatching by `size_of::<T>()`; non-test consumer: `checkpoint.rs::serialize_state_dict_to_bytes` and `checkpoint.rs::deserialize_state_dict_from_bytes` both call `crate::state_dict::dtype_tag::<T>()` for the embedded state-dict section. |
| REQ-4 | SHIPPED | impl: `pub(crate) fn parse_meta_line` in `state_dict.rs` extracting `name`/`shape`/`dtype`/`byte_offset`/`byte_length` without serde; non-test consumer: `checkpoint.rs::deserialize_state_dict_from_bytes` calls `crate::state_dict::parse_meta_line` on every header line inside the embedded section. |
| REQ-5 | SHIPPED | impl: `compile_error!` at `state_dict.rs` top-of-file plus `slice::from_raw_parts(data.as_ptr().cast::<u8>(), ...)` in `save_state_dict` and the SAFETY comment block citing the LE invariant; non-test consumer: the saved bytes are read on the same platform by `load_state_dict` and by `checkpoint.rs::deserialize_state_dict_from_bytes`, both of which depend on the same LE assumption. |
| REQ-6 | SHIPPED | impl: three guard arms in `load_state_dict` returning `InvalidArgument`/`ShapeMismatch`/`DtypeMismatch`; non-test consumer: `checkpoint.rs::deserialize_state_dict_from_bytes` performs the same checks on the embedded section. |
| REQ-7 | SHIPPED | impl: scalar handling via `shape.iter().product()` returning 1 for empty shape, empty state dict produces a zero-line header, hierarchical keys are stored verbatim; non-test consumer: every production caller that writes scalars (regularization losses) or empty optimizer-only checkpoints hits this path; covered by `test_shape_preservation_scalar`, `test_empty_state_dict`, `test_shape_preservation_high_rank` in `mod tests`. |
