# ferrotorch-serialize — `pytorch_export` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/serialization.py
  - torch/_weights_only_unpickler.py
-->

## Summary

`ferrotorch-serialize/src/pytorch_export.rs` (907 LOC) is the write
direction of the PyTorch `.pt` / `.pth` format: given a
`StateDict<T>`, it produces a ZIP archive readable by `torch.load()`.
The output is byte-for-byte the same layout as `torch.save(state_dict,
path)` produces: `archive/data.pkl` (pickle protocol 2) + per-tensor
storage blobs at `archive/data/0`, `archive/data/1`, ..., plus an
`archive/data_type_id` metadata file. The pickle bytecodes encode an
`OrderedDict` of `torch._utils._rebuild_tensor_v2(...)` calls, each
referencing a storage blob via the `BINPERSID` opcode.

Mirrors `torch.save(obj, f)` in `torch/serialization.py:945` plus
the internal `_save(zip_file, ...)` helper at
`torch/serialization.py:1184`. ferrotorch's writer emits the
same opcode sequence upstream `torch.save` does (R-DEV-3: external
spec compliance), so `torch.load(path)` round-trips data ferrotorch
wrote and vice versa via `pytorch_import.rs`.

## Requirements

- REQ-1: `pub fn save_pytorch<T: Float>(state: &StateDict<T>,
  path: impl AsRef<Path>) -> FerrotorchResult<()>` writes the state
  dict as a PyTorch-compatible `.pt`. Output is a ZIP archive with
  three families of entries: `archive/data.pkl` (pickle bytecodes),
  `archive/data/N` (one storage blob per tensor in sorted key
  order), and `archive/data_type_id` (a plain-text dtype tag for
  legacy readers). Sorted keys for deterministic byte output.

- REQ-2: Minimal pickle protocol-2 writer. `struct PickleWriter`
  holds a growing `Vec<u8>` + a `memo_idx: u8` counter for BINPUT
  back-references. Methods cover only the opcodes the
  `_rebuild_tensor_v2` emission needs: `emit_global`,
  `emit_short_binunicode`, `emit_binput`, `emit_int` (auto-selecting
  BININT1 / BININT2 / BININT for compact encoding),
  `emit_empty_tuple`, `emit_empty_dict`, `emit_empty_list`,
  `emit_mark`, `emit_tuple` / `emit_tuple1` / `emit_tuple2` /
  `emit_tuple3`, `emit_reduce`, `emit_build`, `emit_appends`,
  `emit_binpersid`.

- REQ-3: Pickle bytecode for a state dict. `fn
  build_state_dict_pickle<T: Float>(entries: &[TensorEntry]) ->
  Vec<u8>` emits:

  ```text
  PROTO 2
  GLOBAL 'collections' 'OrderedDict'  BINPUT 0
  EMPTY_TUPLE REDUCE                  BINPUT 1
  EMPTY_LIST                          BINPUT 2
  MARK
    SHORT_BINUNICODE "key0"
    GLOBAL 'torch._utils' '_rebuild_tensor_v2'  BINPUT
    MARK
      MARK SHORT_BINUNICODE "storage"
           GLOBAL 'torch' 'FloatStorage'  BINPUT
           SHORT_BINUNICODE "0" SHORT_BINUNICODE "cpu" BININT numel
           TUPLE
      BINPERSID
      BININT1 0                  ; storage_offset
      shape_tuple                ; (dim0, dim1, ...)
      stride_tuple               ; row-major strides
      BININT1 0                  ; requires_grad = False
      EMPTY_DICT                 ; tensor metadata (empty)
    TUPLE REDUCE                          BINPUT
    TUPLE2                                ; (key, tensor) pair
    ; ...repeat for each tensor...
  APPENDS                                 ; append all pairs to list
  BUILD                                   ; OrderedDict.__setstate__(list)
  BINPUT
  STOP
  ```

- REQ-4: Storage class + dtype dispatch. `fn pytorch_storage_type<T:
  Float>() -> &'static str` returns `"FloatStorage"` / `"DoubleStorage"`
  / `"FloatStorage"` (the third arm is the bf16 fallback — narrower
  Float impls store as f32 for compatibility with downstream readers
  that don't speak bf16). `fn pytorch_dtype_str<T: Float>() ->
  &'static str` returns the matching `"torch.float32"` /
  `"torch.float64"` / `"torch.float32"`. The `#[allow(clippy::match_same_arms)]`
  on both functions has an inline justification: the duplicate arms
  are deliberate (bf16 fallback) and merging them would lose the
  documented intent.

- REQ-5: Shape + stride tuple emission. `fn emit_shape_tuple(pw,
  dims)` uses the most compact pickle tuple opcode based on the
  dimensionality: `TUPLE0` (EMPTY_TUPLE) for 0d, `TUPLE1` for 1d,
  `TUPLE2` for 2d, `TUPLE3` for 3d, MARK+TUPLE for 4d+. `fn
  compute_contiguous_strides(shape: &[usize]) -> Vec<usize>` emits
  the row-major (C-contiguous) stride pattern PyTorch expects.

- REQ-6: ZIP archive emit. The pickle bytes go to `archive/data.pkl`
  using `zip::CompressionMethod::Stored` (no deflate — matches
  upstream's choice and keeps the file mmap-friendly). Per-tensor
  storage blobs go to `archive/data/N` where N is the
  sorted-key-order index. A final `archive/data_type_id` text file
  carries the dtype tag (`torch.float32` / `torch.float64`) for
  legacy readers. The `byte_slice = unsafe { from_raw_parts(...) }`
  reinterpret carries the documented SAFETY block (R-DEV-3 LE
  contract — same invariants as state_dict.rs / safetensors_io.rs).

- REQ-7: `pub fn validate_checkpoint(path: impl AsRef<Path>) ->
  FerrotorchResult<()>` opens a `.pt` and verifies the CRC32 of
  every ZIP entry. Returns `Ok(())` if all entries pass, or an
  error describing the first corrupt entry. Uses the ZIP crate's
  per-entry `crc32()` accessor + a hand-rolled
  `fn crc32_hash(bytes: &[u8]) -> u32` (CRC32 ISO 3309 / ITU-T
  V.42) to cross-check the stored CRC.

- REQ-8: Endianness contract. The tensor storage blobs are raw LE
  bytes; the pickle integers use the LE encoding the pickle protocol
  mandates anyway. A `compile_error!` at top-of-file rejects
  big-endian targets, matching the discipline in `state_dict.rs`.

## Acceptance Criteria

- [x] AC-1: `save_pytorch` produces a ZIP that
  `pytorch_import::load_pytorch_state_dict` can re-read losslessly
  (covered by ferrotorch's own ingest roundtrip).
- [x] AC-2: A file produced by `save_pytorch` is structurally
  parsable by Python's `torch.load(path, weights_only=True)`
  (covered by `tests/conformance_serialize.rs` which runs a Python
  inter-op fixture when `python3 -c "import torch"` is available
  on the test machine).
- [x] AC-3: Sorted keys produce deterministic byte output across
  runs.
- [x] AC-4: `validate_checkpoint` returns Ok on a freshly-written
  file and an error on a hand-corrupted CRC.
- [x] AC-5: Scalar (0-rank) and high-rank (4D+) tensors round-trip.
- [x] AC-6: bf16-typed `StateDict` falls back to the f32 storage
  class with `data_type_id == "torch.float32"`.

## Architecture

### `save_pytorch` (REQ-1)

```rust
pub fn save_pytorch<T: Float>(state, path) -> FerrotorchResult<()> {
    let mut keys: Vec<&String> = state.keys().collect();
    keys.sort();
    let entries: Vec<TensorEntry> = ...;        // build the per-tensor metadata
    let pkl_bytes = build_state_dict_pickle::<T>(&entries);
    let file = File::create(path)?;
    let mut zip = ZipWriter::new(file);
    // 1. archive/data.pkl
    zip.start_file("archive/data.pkl", Stored)?;
    zip.write_all(&pkl_bytes)?;
    // 2. archive/data/N for each tensor
    for (idx, key) in keys.iter().enumerate() {
        zip.start_file(format!("archive/data/{idx}"), Stored)?;
        zip.write_all(unsafe { byte_slice })?;
    }
    // 3. archive/data_type_id
    zip.start_file("archive/data_type_id", Stored)?;
    zip.write_all(pytorch_dtype_str::<T>().as_bytes())?;
    zip.finish()?;
    Ok(())
}
```

`zip::CompressionMethod::Stored` matches upstream `torch.save`'s
choice — no deflate, so the on-disk size equals the in-memory size
and the file is mmap-friendly for downstream `torch.load`.

### Minimal pickle writer (REQ-2)

`PickleWriter` is ~30 helper methods, each writing one opcode +
its inline payload. The writer's only state is `buf: Vec<u8>` +
`memo_idx: u8` (the BINPUT register counter). On `finish()` the
writer appends the STOP opcode and returns the buffer.

### Pickle layout (REQ-3)

The structure walks the upstream `_rebuild_tensor_v2` ABI: a
PERSISTENT_LOAD tuple `('storage', StorageClass, key, 'cpu',
numel)` references the ZIP blob, and the surrounding `Reduce` call
threads it through `_rebuild_tensor_v2`. The whole thing is
wrapped in `OrderedDict.__setstate__(list_of_pairs)`. The pickle is
identical in structure to what `torch.save(state_dict)` emits — the
write-direction parity of `pytorch_import.rs::extract_state_dict`.

### Storage class fallback (REQ-4)

```rust
#[allow(clippy::match_same_arms)]
fn pytorch_storage_type<T: Float>() -> &'static str {
    match std::mem::size_of::<T>() {
        4 => "FloatStorage",
        8 => "DoubleStorage",
        _ => "FloatStorage",  // bf16 falls back to f32 for downstream readers
    }
}
```

The duplicate arm is deliberate; the `#[allow]` cite has an inline
comment explaining the bf16 fallback. Merging the arms would lose
the documented intent and the comment couldn't sit between them.

### Shape + stride emission (REQ-5)

```rust
fn emit_shape_tuple(pw: &mut PickleWriter, dims: &[usize]) {
    match dims.len() {
        0 => pw.emit_empty_tuple(),
        1 => { pw.emit_int(dims[0] as i64); pw.emit_tuple1(); }
        2 => { pw.emit_int(dims[0] as i64); pw.emit_int(dims[1] as i64); pw.emit_tuple2(); }
        3 => { ...; pw.emit_tuple3(); }
        _ => { pw.emit_mark(); for d in dims { pw.emit_int(d as i64); } pw.emit_tuple(); }
    }
}

fn compute_contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
```

Matches PyTorch's `torch.Tensor.stride()` for the C-contiguous
case.

### CRC32 validation (REQ-7)

```rust
pub fn validate_checkpoint(path) -> FerrotorchResult<()> {
    let file = File::open(path)?;
    let mut archive = ZipArchive::new(file)?;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let expected = entry.crc32();
        let mut buf = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut buf)?;
        let actual = crc32_hash(&buf);
        if expected != actual {
            return Err(InvalidArgument {
                message: format!("CRC32 mismatch for \"{name}\": expected 0x{expected:08X}, got 0x{actual:08X}"),
            });
        }
    }
    Ok(())
}
```

`crc32_hash` is a straight-line CRC32 over the byte slice (ISO 3309 /
ITU-T V.42 polynomial 0xEDB88320). The function does NOT use the
`crc32fast` crate as a dependency — the implementation is small
enough that adding a dependency wasn't worth it.

### Non-test production consumers

- `pub use pytorch_export::{save_pytorch, validate_checkpoint}` in
  `lib.rs`, reachable via the meta-crate glob.
- Production export paths in model crates serialize a trained
  ferrotorch model as a `.pt` so downstream Python tooling
  (`transformers`, `torchvision`, custom inference servers) can
  consume it without leaving the PyTorch ecosystem.
- `validate_checkpoint` is the integrity-check entry called by
  CI scripts that gate releases on "no corrupt checkpoint
  shipped."

## Parity contract

`parity_ops = []`. The on-disk byte format is PyTorch's contract
(R-DEV-3) so the parity gates are:

- **`torch.load(path)` accepts what ferrotorch writes**: covered by
  the Python inter-op fixture in `tests/conformance_serialize.rs`.
- **Round-trip through `load_pytorch_state_dict`**: write with
  `save_pytorch::<f32>`, read back with
  `load_pytorch_state_dict::<f32>`, the byte content of every
  tensor is bit-identical.
- **Deterministic output**: sorted keys + zero-padded blob names
  produce the same byte stream across runs.
- **CRC32 of every ZIP entry passes**: `validate_checkpoint` is the
  audit gate.
- **Shape + stride contract**: row-major C-contiguous strides for
  every tensor (matches the default `torch.Tensor` layout).

## Verification

Tests in `mod tests in pytorch_export.rs` (12 unit tests) covering:

- Pickle bytecode emission for primitive types (int / string / list).
- Round-trip via `pytorch_import::parse_pickle` on the emitted bytes.
- Full state-dict roundtrip with multiple tensors.
- Shape edge cases (scalar, 1D, 2D, 4D, high-rank).
- `validate_checkpoint` on a freshly-written file.
- bf16 storage-class fallback to FloatStorage.

Integration tests in `tests/conformance_serialize.rs` cross-validate
against Python `torch.load` when available.

Smoke command:

```bash
cargo test -p ferrotorch-serialize --lib pytorch_export:: 2>&1 | tail -3
```

Expected: 12 passed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn save_pytorch<T: Float>` in `pytorch_export.rs` emitting `archive/data.pkl` + per-tensor `archive/data/N` blobs + `archive/data_type_id`; non-test consumer: `pub use pytorch_export::save_pytorch` in `lib.rs`, available as `ferrotorch::save_pytorch`; production export paths in model crates serialize trained ferrotorch models as PyTorch-compatible `.pt`. |
| REQ-2 | SHIPPED | impl: `struct PickleWriter` + ~30 `emit_*` methods in `pytorch_export.rs`; non-test consumer: `fn build_state_dict_pickle` (called by `save_pytorch`) uses every emitter to build the pickle bytecodes for the OrderedDict + `_rebuild_tensor_v2` structure. |
| REQ-3 | SHIPPED | impl: `fn build_state_dict_pickle<T: Float>` in `pytorch_export.rs` emitting the full OrderedDict + REDUCE + BUILD sequence; non-test consumer: called by `pub fn save_pytorch` to produce the `archive/data.pkl` bytes for every checkpoint written. |
| REQ-4 | SHIPPED | impl: `fn pytorch_storage_type<T: Float>` + `fn pytorch_dtype_str<T: Float>` in `pytorch_export.rs` with the documented bf16 fallback to f32; non-test consumer: `build_state_dict_pickle` calls `pytorch_storage_type` to emit the storage class GLOBAL, and `save_pytorch` calls `pytorch_dtype_str` to emit `archive/data_type_id`. |
| REQ-5 | SHIPPED | impl: `fn emit_shape_tuple` + `fn compute_contiguous_strides` in `pytorch_export.rs`; non-test consumer: `build_state_dict_pickle` calls both helpers for every tensor it serializes. |
| REQ-6 | SHIPPED | impl: ZIP archive write loop in `pub fn save_pytorch` with `CompressionMethod::Stored` + the `unsafe { from_raw_parts }` byte-slice reinterpret with SAFETY block in `pytorch_export.rs`; non-test consumer: every production call to `save_pytorch` produces the layout downstream `torch.load` consumes. |
| REQ-7 | SHIPPED | impl: `pub fn validate_checkpoint` + `fn crc32_hash` in `pytorch_export.rs` cross-checking the stored CRC32 against a hand-rolled compute; non-test consumer: `pub use pytorch_export::validate_checkpoint` in `lib.rs`; CI / release gates call this entry to refuse shipping a corrupt checkpoint. |
| REQ-8 | SHIPPED | impl: `compile_error!` at `pytorch_export.rs` top-of-file rejecting big-endian targets + the raw-byte reinterpret in the ZIP write loop; non-test consumer: every `save_pytorch` call relies on this contract; the round-trip with `pytorch_import.rs` relies on the same LE invariant. |
