# ferrotorch-serialize — `pytorch_import` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/serialization.py
  - torch/_weights_only_unpickler.py
-->

## Summary

`ferrotorch-serialize/src/pytorch_import.rs` (1982 LOC) parses
PyTorch's `.pt` / `.pth` checkpoint files in pure Rust. A `.pt` file
is a ZIP archive containing a pickle-encoded structure description
(`archive/data.pkl`) plus raw little-endian tensor byte blobs
(`archive/data/0`, `archive/data/1`, ...) referenced by the pickle
through the `PERSISTENT_LOAD` (`BINPERSID`) opcode. This module
implements a minimal pickle protocol-2 / protocol-3 virtual machine,
a `PickleValue` AST, and the interpretation logic that walks the
parsed pickle tree to extract tensor metadata and load the
corresponding raw bytes from the ZIP.

Mirrors `torch.load(path, weights_only=True)` in
`torch/serialization.py:1316` + the unpickler subset in
`torch/_weights_only_unpickler.py:589`. ferrotorch's pickle parser
intentionally implements ONLY the opcode subset PyTorch's state dict
emits — the parser explicitly rejects unknown opcodes rather than
silently producing garbage (the entire `weights_only=True`
restriction goal). R-DEV-4: pickle is a Python attack surface;
re-implementing only the safe subset is the Rust analog.

## Requirements

- REQ-1: `pub enum PickleValue` — the pickle VM value tree. Variants
  cover the pickle types PyTorch state dicts actually emit:
  `None`, `Bool`, `Int(i64)`, `Float(f64)`, `Bytes(Vec<u8>)`,
  `String(String)`, `Tuple(Vec<PickleValue>)`,
  `List(Vec<PickleValue>)`, `Dict(Vec<(PickleValue, PickleValue)>)`,
  `Global { module, name }`, `Reduce { callable, args }`,
  `Build { obj, state }`, `PersistentLoad(Box<PickleValue>)`.

- REQ-2: `pub fn parse_pickle(data: &[u8]) ->
  FerrotorchResult<PickleValue>` runs the pickle VM over a byte
  slice. Handles ~30 opcodes from protocol 2 (the original PyTorch
  default) AND protocol 3+ subset (FRAME, BINUNICODE,
  SHORT_BINUNICODE, BINBYTES, SHORT_BINBYTES, STACK_GLOBAL,
  MEMOIZE, NEWOBJ). The opcode dispatch is an explicit match;
  unknown opcodes produce `FerrotorchError::InvalidArgument` with
  the byte and stream position in the message rather than
  panicking.

- REQ-3: Bug-fixed opcode constants (ferrotorch #1169). The
  constants for `BINUNICODE` (0x58), `BINBYTES` (0x42), and
  `SHORT_BINBYTES` (0x43) were previously misnumbered (BINUNICODE
  was 0x8d; BINBYTES and SHORT_BINBYTES were swapped). The doc
  comments above each constant call out the previous bug and
  cite the pickle protocol spec by version. This was the load-bearing
  fix that made `resnet18-f37072fd.pth` (and every other modern
  ZIP-pickle .pth) decode correctly.

- REQ-4: ZIP archive enumeration. `fn find_pkl_name<R>(archive)`
  searches for the pickle in standard locations (`archive/data.pkl`,
  `data.pkl`) and falls back to any `*.pkl` suffix (case-insensitive
  — some downstream tooling capitalizes). `fn find_data_prefix<R>(
  archive, infos)` probes for the tensor blob directory prefix
  (`archive/data/`, `data/`, or empty); both prefixes show up in
  the wild depending on which torch.save version produced the file.

- REQ-5: State dict extraction. `fn extract_state_dict(root:
  &PickleValue) -> FerrotorchResult<Vec<TensorInfo>>` walks the
  pickle tree (a Reduce/Build wrapping an OrderedDict of
  `(name, _rebuild_tensor_v2(...))` pairs) and pulls out per-tensor
  metadata: storage key (the ZIP blob path suffix), storage offset,
  dtype string, shape. The dtype string comes from the
  `PERSISTENT_LOAD` tuple's storage class name
  (`FloatStorage` → `f32`, `DoubleStorage` → `f64`,
  `HalfStorage` → `f16`, `BFloat16Storage` → `bf16`, etc.).

- REQ-6: Tensor byte → `Tensor<T>` conversion. `fn
  convert_bytes_to_float<T: Float>(bytes: &[u8], dtype_str: &str,
  numel: usize) -> FerrotorchResult<Vec<T>>` dispatches on the
  upstream stored dtype: `f32` raw, `f64` raw, `f16` upcast via
  hand-rolled half-to-float, `bf16` upcast via byte-shift, and
  finally `numeric_cast` into `T` if the stored dtype is wider
  or narrower than `T`.

- REQ-7: `pub fn load_pytorch_state_dict<T: Float>(path) ->
  FerrotorchResult<StateDict<T>>` is the high-level entry: opens
  the ZIP, finds the pickle, parses it, walks the tree, and reads
  each tensor's raw bytes. The output is a `StateDict<T>` where
  `T` is the target dtype (any `Float`); cross-dtype conversion
  is handled by `convert_bytes_to_float`.

- REQ-8: `pub fn load_pytorch_state_dict_mmap<T: Float>(path) ->
  FerrotorchResult<StateDict<T>>` (#629) is the mmap counterpart.
  Uses `memmap2::Mmap` + a `Cursor<&[u8]>` so the ZIP reader
  doesn't slurp the file into a heap `Vec<u8>` first. Same return
  contract — the mmap is dropped before the function returns and
  tensor data is copied into owned buffers via
  `convert_bytes_to_float`.

- REQ-9: Shared inner dispatch. `fn load_pytorch_state_dict_inner<
  T, R: Read + Seek>(archive: ZipArchive<R>) ->
  FerrotorchResult<StateDict<T>>` is generic over the reader so
  both the `File`-backed and `Cursor<&[u8]>`-backed paths funnel
  through the same logic. This is the structural fix from #629
  that keeps the file-backed and mmap'd paths from diverging.

## Acceptance Criteria

- [x] AC-1: `parse_pickle` decodes a synthetic PyTorch-style state
  dict pickle blob into a `PickleValue` tree (covered by tests in
  `mod tests`).
- [x] AC-2: `load_pytorch_state_dict::<f32>` round-trips a
  ferrotorch-produced `.pt` (via `pytorch_export.rs`).
- [x] AC-3: Real-world checkpoint files (resnet18, distilbert)
  load without error (covered by
  `tests/conformance_serialize.rs`).
- [x] AC-4: The bug-fixed opcode constants (BINUNICODE,
  BINBYTES, SHORT_BINBYTES) are pinned at the spec values by an
  inline test (`test_opcode_constants`).
- [x] AC-5: `load_pytorch_state_dict_mmap` produces a
  `StateDict<T>` byte-identical to
  `load_pytorch_state_dict` on the same input file.
- [x] AC-6: Unknown pickle opcodes produce a structured
  `FerrotorchError::InvalidArgument` rather than panicking
  (`test_unknown_opcode`).
- [x] AC-7: Truncated pickle data produces a structured error
  (`test_truncated_pickle`).

## Architecture

### Pickle VM value tree (REQ-1)

```rust
pub enum PickleValue {
    None, Bool(bool), Int(i64), Float(f64), Bytes(Vec<u8>),
    String(String),
    Tuple(Vec<PickleValue>), List(Vec<PickleValue>),
    Dict(Vec<(PickleValue, PickleValue)>),
    Global { module: String, name: String },
    Reduce { callable: Box<PickleValue>, args: Box<PickleValue> },
    Build { obj: Box<PickleValue>, state: Box<PickleValue> },
    PersistentLoad(Box<PickleValue>),
}
```

Only types that appear in PyTorch state-dict pickles are
modelled. There is no `Class` variant, no `Object` variant, no
arbitrary callable — `weights_only=True` semantics deliberately.

### Pickle VM (REQ-2)

`parse_pickle` is a loop over the byte stream. The state is:
- `stack: Vec<PickleValue>` — the pickle VM's evaluation stack.
- `memo: HashMap<u32, PickleValue>` — back-references from
  BINPUT / BINGET / MEMOIZE.
- `mark_stack: Vec<usize>` — the MARK opcode pushes the current
  stack length; matching opcodes (DICT, LIST, TUPLE) pop everything
  pushed since.

The opcode dispatch is a flat `match opcode`. Each arm reads its
payload (length-prefixed string, varint, etc.) and either pushes
a `PickleValue` onto the stack or consumes some items and pushes a
combined value.

Unknown opcodes return `pickle_err("unsupported pickle opcode
0xNN at position M")` with the byte and stream offset in the
message. There is no `panic!`, no `unwrap`, no `unreachable!`.

### Bug-fixed opcode constants (REQ-3)

```rust
// Pickle protocol-2 BINUNICODE opcode is 0x58 ('X')
// Prior to #1169 this was incorrectly 0x8d.
const BINUNICODE: u8 = 0x58;

// Pickle protocol-3 BINBYTES = 0x42 ('B'), SHORT_BINBYTES = 0x43 ('C').
// Prior to #1169 these constants were swapped.
const BINBYTES: u8 = 0x42;
const SHORT_BINBYTES: u8 = 0x43;
```

Each fix has its own inline doc-comment block explaining the
previous bug and the spec reference, so future readers don't
"clean up" the comments without understanding the regression they
prevent.

### ZIP enumeration (REQ-4)

`find_pkl_name`:
```rust
for candidate in &["archive/data.pkl", "data.pkl"] { ... }
// fallback: any *.pkl (case-insensitive)
```

`find_data_prefix`:
```rust
for prefix in &["archive/data/", "data/", ""] {
    if names.contains(&format!("{prefix}{first_key}")) { return prefix; }
}
```

Both probe-and-fallback because PyTorch's ZIP layout has shifted
across versions (1.x used `archive/`, some custom wrappers strip
the prefix).

### State dict tree walk (REQ-5)

`extract_state_dict` is the load-bearing interpreter pass. It
expects the pickle root to be a `Reduce { callable: Global {
module: "collections", name: "OrderedDict" }, args: ... }` (or
the `Build`-wrapped variant). The OrderedDict's contents are
walked pair-by-pair; each value should be a
`Reduce { callable: Global { module: "torch._utils", name:
"_rebuild_tensor_v2" }, args: Tuple(...) }`. The args tuple yields
`(storage_persistent_load, storage_offset, shape_tuple,
stride_tuple, requires_grad, metadata_dict)`. We then deconstruct
the `PersistentLoad` payload `(b"storage", Global{name:
"FloatStorage"}, b"0", b"cpu", numel)` to get the storage key
and dtype.

### Bytes → Tensor (REQ-6)

`convert_bytes_to_float<T: Float>(bytes, dtype_str, numel)`:

```rust
match dtype_str {
    "f32" => f32::from_le_bytes per element → numeric_cast::<f32, T>,
    "f64" => f64::from_le_bytes per element → numeric_cast::<f64, T>,
    "f16" => f16_to_f32_bits per element → numeric_cast::<f32, T>,
    "bf16" => (u16 bits << 16) bit-cast to f32 → numeric_cast::<f32, T>,
    other => InvalidArgument,
}
```

The `numeric_cast` calls handle the cross-dtype path; loading a
f64 PyTorch checkpoint into `StateDict<f32>` is a lossy cast
that succeeds for finite values and errors via the cast for
out-of-range.

### Public entry (REQ-7)

```rust
pub fn load_pytorch_state_dict<T: Float>(path) -> FerrotorchResult<StateDict<T>> {
    let file = File::open(path)?;
    let archive = ZipArchive::new(file)?;
    load_pytorch_state_dict_inner(archive)
}
```

The mmap variant (REQ-8):

```rust
pub fn load_pytorch_state_dict_mmap<T: Float>(path) -> FerrotorchResult<StateDict<T>> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file) }?;
    let cursor = Cursor::new(&mmap[..]);
    let archive = ZipArchive::new(cursor)?;
    load_pytorch_state_dict_inner(archive)
}
```

The mmap is dropped at function exit; tensor bytes are copied by
`convert_bytes_to_float` so no borrow escapes.

### Shared inner (REQ-9)

```rust
fn load_pytorch_state_dict_inner<T: Float, R: Read + Seek>(
    mut archive: zip::ZipArchive<R>,
) -> FerrotorchResult<StateDict<T>>
```

Genericized over the reader so File-backed and Cursor-backed paths
share every line of the pickle parse + tree walk + tensor decode.
This is what #629 fixed — previously the two paths had drifted
and reported different errors on the same input.

### Non-test production consumers

- `pub use pytorch_import::{PickleValue, load_pytorch_state_dict,
  load_pytorch_state_dict_mmap, parse_pickle}` in `lib.rs`, reachable
  through the meta-crate glob.
- Production cross-framework ingest paths in model crates
  (`ferrotorch-llama`, `ferrotorch-bert`, etc.) call
  `ferrotorch::load_pytorch_state_dict::<f32>(path)?` to load
  HuggingFace `pytorch_model.bin` or pre-safetensors `.pth`
  checkpoints.

## Parity contract

`parity_ops = []`. The on-disk byte format is PyTorch's contract
(R-DEV-3). Edge cases:

- **Protocol 2 + 3 opcode coverage**: every opcode PyTorch state
  dict emits is handled; unknown opcodes produce a structured
  error.
- **Dtype upcast on load**: f16/bf16 stored tensors upcast to f32
  when loading as `StateDict<f32>`. Numeric semantics delegated
  to `numeric_cast::cast`.
- **Both ZIP layouts**: `archive/data/N` and `data/N` blob
  prefixes both work.
- **Bug-fix regressions pinned**: the BINUNICODE / BINBYTES /
  SHORT_BINBYTES opcode constants are at the spec values; an
  inline test guards against revert.
- **mmap byte equality**: `load_pytorch_state_dict` and
  `load_pytorch_state_dict_mmap` produce the same `StateDict`
  for the same file (#629).

## Verification

Tests in `mod tests in pytorch_import.rs` (48 unit tests) covering:

- Every pickle opcode in isolation (synthetic minimal pickles).
- The full PyTorch state-dict tree walk against fixture pickles.
- Real-world checkpoint files (resnet18, distilbert) under
  `tests/conformance_serialize.rs`.
- Truncated / malformed pickle handling.
- Cross-dtype conversion (f64 stored → f32 loaded).
- mmap-vs-non-mmap byte equality
  (`tests/conformance_serialize_mmap.rs`).

Smoke command:

```bash
cargo test -p ferrotorch-serialize --lib pytorch_import:: 2>&1 | tail -3
```

Expected: 48 passed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum PickleValue` with 14 variants in `pytorch_import.rs`; non-test consumer: `pub use pytorch_import::PickleValue` in `lib.rs`; `parse_pickle` returns `PickleValue` and `extract_state_dict` consumes the variants — production code that wants to inspect a raw pickle tree (e.g. checkpoint metadata extraction) uses this surface. |
| REQ-2 | SHIPPED | impl: `pub fn parse_pickle(data: &[u8])` in `pytorch_import.rs` running the opcode-dispatch loop with stack/memo/mark state; non-test consumer: `pub use pytorch_import::parse_pickle` in `lib.rs`; `fn load_pytorch_state_dict_inner` is the production caller of `parse_pickle` (which then drives the tensor extraction). |
| REQ-3 | SHIPPED | impl: `const BINUNICODE: u8 = 0x58`, `const BINBYTES: u8 = 0x42`, `const SHORT_BINBYTES: u8 = 0x43` with inline regression-cite doc comments in `pytorch_import.rs`; non-test consumer: the opcode constants are matched in `parse_pickle`'s match arms; the regression-fix lives in production code (without it `resnet18.pth` doesn't load). |
| REQ-4 | SHIPPED | impl: `fn find_pkl_name<R>` + `fn find_data_prefix<R>` in `pytorch_import.rs` probing standard layouts; non-test consumer: `load_pytorch_state_dict_inner` calls both helpers as part of every `.pt` load. |
| REQ-5 | SHIPPED | impl: `fn extract_state_dict(&PickleValue)` in `pytorch_import.rs` walking the OrderedDict + `_rebuild_tensor_v2` tree; non-test consumer: called by `load_pytorch_state_dict_inner` to convert the parsed pickle into `Vec<TensorInfo>`. |
| REQ-6 | SHIPPED | impl: `fn convert_bytes_to_float<T: Float>` in `pytorch_import.rs` dispatching on `dtype_str` (f32/f64/f16/bf16) + `numeric_cast` into `T`; non-test consumer: called by `load_pytorch_state_dict_inner` once per tensor to turn raw blob bytes into a `Vec<T>`. |
| REQ-7 | SHIPPED | impl: `pub fn load_pytorch_state_dict<T: Float>` in `pytorch_import.rs` opening the ZIP and routing through `load_pytorch_state_dict_inner`; non-test consumer: `pub use pytorch_import::load_pytorch_state_dict` in `lib.rs`; production cross-framework ingest paths call this entry. |
| REQ-8 | SHIPPED | impl: `pub fn load_pytorch_state_dict_mmap<T: Float>` in `pytorch_import.rs` using `Mmap::map` + `Cursor`; non-test consumer: `pub use pytorch_import::load_pytorch_state_dict_mmap` in `lib.rs`; production inference servers loading 7B-param transformer checkpoints prefer this entry. |
| REQ-9 | SHIPPED | impl: `fn load_pytorch_state_dict_inner<T, R: Read + Seek>` generic-over-reader function in `pytorch_import.rs`; non-test consumer: both `load_pytorch_state_dict` and `load_pytorch_state_dict_mmap` funnel through this function so the parse + tree walk + tensor decode logic is shared. |
