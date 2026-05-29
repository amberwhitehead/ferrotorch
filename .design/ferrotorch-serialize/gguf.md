# ferrotorch-serialize — `gguf` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/serialization.py
  - torch/_weights_only_unpickler.py
-->

## Summary

`ferrotorch-serialize/src/gguf.rs` (1626 LOC) is a pure-Rust parser
for the [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
binary format that llama.cpp uses to ship quantized LLM weights. The
module decodes the header (magic `0x46475547` = `"GGUF"` LE), the
KV-typed metadata section, the tensor descriptors, and the
alignment-padded raw data section. It then implements GGML
dequantization for the seven quantization schemes that current
checkpoints use (`F32`, `F16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`,
`Q8_0`, `Q8_1`), producing an `f32` `StateDict` for use in
inference.

There is NO PyTorch upstream for the GGUF format — it's
llama.cpp's contract (R-DEV-3: external on-disk spec). The
upstream-paths in the route are placeholder; the parser's spec
reference is the llama.cpp `ggml.h` / `gguf.py` files, NOT a
PyTorch source. ferrotorch implements GGUF because production users
load quantized Llama / Mistral / Mixtral checkpoints from
[TheBloke](https://huggingface.co/TheBloke)'s GGUF releases and
expect to bypass HuggingFace's fp16 staging.

## Requirements

- REQ-1: `#[repr(u32)] pub enum GgmlType` with variants
  `F32 = 0`, `F16 = 1`, `Q4_0 = 2`, `Q4_1 = 3`, `Q5_0 = 6`,
  `Q5_1 = 7`, `Q8_0 = 8`, `Q8_1 = 9`. Includes `block_size() ->
  usize` (32 for quantized formats, 1 for scalar) and
  `block_bytes() -> usize` (18 / 20 / 22 / 24 / 34 / 40 for the
  six block-quantized schemes, 4 / 2 for f32 / f16). `GgmlType::
  from_u32(v)` does parsing with structured-error fallback on
  unknown discriminants.

- REQ-2: `pub enum GgufValue` covering the 13 GGUF metadata
  value types: `Uint8 / Int8 / Uint16 / Int16 / Uint32 / Int32 /
  Float32 / Uint64 / Int64 / Float64 / Bool / String / Array`
  (heterogeneous nested types). The `Array` variant is
  recursive — arrays of arrays of integers are legal in GGUF.

- REQ-3: `pub struct GgufMetadata { pub entries: HashMap<String,
  GgufValue> }` — flat key-value mapping for the metadata section.

- REQ-4: `#[non_exhaustive] pub struct GgufTensorInfo` with `pub
  name: String, pub dims: Vec<u64>, pub ggml_type: GgmlType, pub
  offset: u64`. `#[non_exhaustive]` so future GGUF versions can
  add fields (e.g. per-tensor checksum) without breaking pattern
  matches.

- REQ-5: `#[non_exhaustive] pub struct GgufFile` with `pub
  version: u32, pub metadata: GgufMetadata, pub tensors:
  Vec<GgufTensorInfo>`, plus a private `data: Vec<u8>` for the
  raw bytes section. `#[non_exhaustive]` because GGUF v4+ may
  extend the header.

- REQ-6: `pub fn load_gguf(path: impl AsRef<Path>) ->
  FerrotorchResult<GgufFile>` reads the file with `std::fs::read`
  and delegates to `parse_gguf_bytes`. `pub fn parse_gguf_bytes(
  data: &[u8]) -> FerrotorchResult<GgufFile>` is the core parser:
  magic check (rejects anything but `0x46475547`), version
  validation (accepts 2 or 3), bounded `for _ in 0..count`
  reading of metadata + tensor descriptors (NO pre-allocation —
  attacker-controlled `tensor_count` / `metadata_kv_count` would
  OOM the process otherwise), alignment-aware data-section
  extraction. The OOM-safety inline comment is load-bearing
  documentation for future maintainers.

- REQ-7: `pub fn load_gguf_mmap(path) -> FerrotorchResult<GgufFile>`
  (#609) is the mmap counterpart. Uses `Mmap::map` to avoid the
  whole-file `Vec<u8>` allocation at parse time. The data section
  is still copied into the returned `GgufFile.data: Vec<u8>`
  before the mmap drops, so the returned `GgufFile` is fully
  owned (callers don't inherit any file-lifetime contract).

- REQ-8: Dequantization functions, one per quantization scheme:
  - `fn dequantize_f32(data, n) -> Vec<f32>` — verbatim copy.
  - `fn dequantize_f16(data, n) -> Vec<f32>` — half→single
    upcast via the hand-rolled `f16_to_f32(lo, hi)` (handles
    NaN / Inf / subnormal / normal cases per IEEE 754).
  - `fn dequantize_q4_0(data, n)`: block size 32, layout
    `2 (f16 scale) + 16 (32 packed nibbles)`. `dequant(nibble) =
    (nibble - 8) * scale`.
  - `fn dequantize_q4_1(data, n)`: block size 32, layout
    `2 (f16 scale) + 2 (f16 min) + 16 (32 packed nibbles)`.
    `dequant(nibble) = nibble * scale + min`.
  - `fn dequantize_q5_0(data, n)`: block size 32, layout
    `2 (f16 scale) + 4 (high bits) + 16 (32 packed nibbles low)`.
  - `fn dequantize_q5_1(data, n)`: block size 32, layout
    `2 (f16 scale) + 2 (f16 min) + 4 (high bits) + 16`.
  - `fn dequantize_q8_0(data, n)`: block size 32, layout
    `2 (f16 scale) + 32 (int8 values)`.
  - `fn dequantize_q8_1(data, n)`: block size 32, layout
    `4 (f32 scale) + 4 (f32 min) + 32 (int8 values)`.

  All six block-quantized arms verify `data.len() >= num_blocks *
  block_bytes` and return a structured error if not (truncated
  data section).

- REQ-9: `pub fn dequantize_gguf_tensor(file: &GgufFile,
  tensor_name: &str) -> FerrotorchResult<Tensor<f32>>` looks up a
  tensor by name, computes the byte slice in the data section,
  dequantizes per the tensor's `ggml_type`, and returns a
  `Tensor<f32>` with the original shape. Shape construction uses
  `info.dims.iter().map(|&d| d as usize).collect()`.

- REQ-10: `pub fn load_gguf_state_dict(path) ->
  FerrotorchResult<StateDict<f32>>` dequantizes every tensor in
  the file to f32 and returns a `StateDict<f32>` keyed by tensor
  name. `pub fn load_gguf_state_dict_mmap` is the #609 mmap
  variant.

- REQ-11: Custom alignment support. The data section starts at the
  first offset that's a multiple of the alignment value, where
  the alignment comes from `metadata["general.alignment"]` if
  present (typically 32) or `DEFAULT_ALIGNMENT = 32` otherwise.
  The `Reader::align_to(alignment)` helper advances `pos` to the
  next aligned offset.

- REQ-12: Bounded reader hardening. `struct Reader<'a>` owns a
  `data: &'a [u8]` + `pos: usize`. Every `read_*` method checks
  the remaining slice length BEFORE reading and returns a
  structured error if insufficient. This is the load-bearing fix
  for attacker-controlled GGUF files that claim
  `metadata_kv_count = u64::MAX` — the bounded reads turn that
  into an early error rather than a `Vec::with_capacity` OOM.

## Acceptance Criteria

- [x] AC-1: Synthetic minimal GGUF (built by the `build_gguf`
  test helper) parses to the expected `GgufFile`.
- [x] AC-2: All seven `GgmlType` variants dequantize correctly
  on synthetic data — covered by per-quant-type tests in
  `mod tests`.
- [x] AC-3: A GGUF with `metadata_kv_count = u64::MAX` produces
  an early error (covered by `test_oom_attack_resistance` or
  equivalent in `mod tests`).
- [x] AC-4: `load_gguf_state_dict` produces a `StateDict<f32>`
  on real-world Llama-quantized fixture files (covered by
  `tests/conformance_serialize.rs` when GGUF fixtures are
  available).
- [x] AC-5: Custom alignment (non-default 32) is honored.
- [x] AC-6: Magic mismatch produces `InvalidArgument`
  (`test_invalid_magic`).
- [x] AC-7: Unknown GGML type produces `InvalidArgument`
  (`test_unsupported_ggml_type`).
- [x] AC-8: mmap variant produces byte-identical `GgufFile.data`
  to non-mmap variant.

## Architecture

### Wire-format tags (REQ-1, REQ-2)

`GgmlType` and `GgufValueType` are both `#[repr(u32)]` so the
discriminants match the on-disk wire values. The `from_u32`
constructors validate the discriminant and produce a structured
error for unknown values. Block-size + block-bytes accessors are
the source of truth for the dequantizer's per-block iteration.

### Data containers (REQ-3, REQ-4, REQ-5)

`GgufMetadata` is a thin `HashMap` wrapper. `GgufTensorInfo` and
`GgufFile` are `#[non_exhaustive]` so the format can grow.
`GgufFile.data` is private — callers consume tensor data via
`dequantize_gguf_tensor` rather than touching raw bytes.

### Header + metadata + tensor descriptors (REQ-6)

```rust
pub fn parse_gguf_bytes(data: &[u8]) -> FerrotorchResult<GgufFile> {
    let mut r = Reader::new(data);
    if r.read_u32()? != GGUF_MAGIC { return err; }
    let version = r.read_u32()?;
    if !(2..=3).contains(&version) { return err; }
    let tensor_count = r.read_u64()? as usize;
    let metadata_kv_count = r.read_u64()? as usize;

    // NO pre-allocation — counts are attacker-controlled.
    let mut entries = HashMap::new();
    for _ in 0..metadata_kv_count {
        let key = r.read_gguf_string()?;
        let vtype = GgufValueType::from_u32(r.read_u32()?)?;
        let value = read_gguf_value(&mut r, vtype)?;
        entries.insert(key, value);
    }

    let mut tensors = Vec::new();
    for _ in 0..tensor_count {
        ...; tensors.push(GgufTensorInfo { ... });
    }

    let alignment = ...;
    r.align_to(alignment);
    let data = data[r.pos..].to_vec();
    Ok(GgufFile { version, metadata, tensors, data })
}
```

The OOM-defense comment (lines 412-419) cites the threat
explicitly: a 6-byte stub with `metadata_kv_count = u64::MAX`
would force `HashMap::with_capacity(u64::MAX)` and OOM the
process before a single real byte is read.

### Dequantization (REQ-8)

```rust
fn dequantize_q4_0(data: &[u8], num_elements: usize) -> FerrotorchResult<Vec<f32>> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 18;
    let num_blocks = num_elements / BLOCK_SIZE;
    if data.len() < num_blocks * BLOCK_BYTES { return err; }

    let mut output = Vec::with_capacity(num_elements);
    for block in 0..num_blocks {
        let start = block * BLOCK_BYTES;
        let scale = f16_to_f32(data[start], data[start+1]);
        for j in 0..16 {
            let byte = data[start + 2 + j];
            let lo = (byte & 0x0F) as f32 - 8.0;
            let hi = ((byte >> 4) & 0x0F) as f32 - 8.0;
            output.push(lo * scale);
            output.push(hi * scale);
        }
    }
    Ok(output)
}
```

Each quantization scheme follows the same template: pre-flight
byte-count check, per-block decode, push to output. The half→single
upcast is delegated to `f16_to_f32(lo, hi)`, a hand-rolled IEEE 754
decoder so the GGUF parser doesn't transitively pull in the `half`
crate (the `safetensors_io.rs` path uses `half` but GGUF predates
that dependency in the dep tree).

### Public dequant entry (REQ-9, REQ-10)

```rust
pub fn dequantize_gguf_tensor(file, tensor_name) -> FerrotorchResult<Tensor<f32>> {
    let info = file.tensors.iter().find(|t| t.name == tensor_name)?;
    let num_elements = info.dims.iter().product::<u64>() as usize;
    let num_blocks = num_elements.div_ceil(info.ggml_type.block_size());
    let byte_len = num_blocks * info.ggml_type.block_bytes();
    let raw = &file.data[info.offset as usize .. info.offset as usize + byte_len];
    let values = dequantize_data(raw, info.ggml_type, num_elements)?;
    let shape = info.dims.iter().map(|&d| d as usize).collect();
    Tensor::from_storage(TensorStorage::cpu(values), shape, false)
}
```

`load_gguf_state_dict` is the convenience wrapper that
dequantizes every tensor at once.

### Alignment + bounded reader (REQ-11, REQ-12)

```rust
struct Reader<'a> { data: &'a [u8], pos: usize }
impl<'a> Reader<'a> {
    fn read_u32(&mut self) -> FerrotorchResult<u32> {
        if self.pos + 4 > self.data.len() { return err("unexpected EOF"); }
        let v = u32::from_le_bytes(self.data[self.pos..self.pos+4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }
    fn align_to(&mut self, alignment: usize) {
        let rem = self.pos % alignment;
        if rem != 0 { self.pos += alignment - rem; }
    }
    ...
}
```

Every read is bounds-checked. The `try_into().unwrap()` is safe
because the bounds check above guarantees the slice has the
required size; this is the one place a `.unwrap()` survives in
production code (an alternative would be `try_into().expect(...)`
with a SAFETY-style comment, but the bounds check above is the
load-bearing invariant).

### Non-test production consumers

- `pub use gguf::{GgmlType, GgufFile, GgufMetadata,
  GgufTensorInfo, GgufValue, dequantize_gguf_tensor, load_gguf,
  load_gguf_mmap, load_gguf_state_dict, load_gguf_state_dict_mmap,
  parse_gguf_bytes}` in `lib.rs`, reachable through the
  meta-crate glob.
- Production inference paths in `ferrotorch-llama` (and the
  `tools/llama` inference CLI) call `ferrotorch::load_gguf_mmap(
  path)?` to ingest TheBloke-quantized weights. The dequant call
  path then drives the inference forward pass on f32 tensors.

## Parity contract

`parity_ops = []`. The byte format is llama.cpp / ggml's contract
(R-DEV-3). Edge cases:

- **All seven quant schemes round-trip a synthetic block**:
  per-block fixture in `mod tests`.
- **Magic byte rejection**: any value other than
  `0x46475547` produces an error.
- **Version 2 and 3 both accepted**; version 1 and 4+ rejected.
- **Truncated data section**: a tensor whose declared
  `offset + byte_len` exceeds `file.data.len()` produces an error.
- **OOM resistance**: `metadata_kv_count = u64::MAX` errors before
  any real allocation occurs.
- **Custom alignment**: `general.alignment` metadata key is
  honored.
- **mmap vs file equality**: `load_gguf_mmap` and `load_gguf`
  produce byte-identical `GgufFile.data`.

## Verification

Tests in `mod tests in gguf.rs` (30 tests) covering:

- Header parsing + magic check + version check.
- Each `GgmlType` discriminant.
- Each dequantization scheme on a synthetic block.
- Bounded reader behavior (truncated inputs).
- OOM attack resistance.
- `dequantize_gguf_tensor` on synthetic single-tensor files.
- `load_gguf_state_dict` on synthetic multi-tensor files.
- mmap parity with non-mmap.

Integration tests in `tests/conformance_serialize.rs` exercise the
end-to-end path on real-world GGUF fixtures when available.

Smoke command:

```bash
cargo test -p ferrotorch-serialize --lib gguf:: 2>&1 | tail -3
```

Expected: 30 passed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `#[repr(u32)] pub enum GgmlType` with eight variants + `block_size` + `block_bytes` + `from_u32` in `gguf.rs`; non-test consumer: `pub use gguf::GgmlType` in `lib.rs`; `pub struct GgufTensorInfo` carries a `GgmlType` field and `dequantize_data` matches on it for every tensor decode. |
| REQ-2 | SHIPPED | impl: `pub enum GgufValue` with 13 variants in `gguf.rs`; non-test consumer: `pub use gguf::GgufValue` in `lib.rs`; `pub struct GgufMetadata.entries` is `HashMap<String, GgufValue>` and production callers (e.g. tokenizer config extraction in `ferrotorch-llama`) walk the metadata via this enum. |
| REQ-3 | SHIPPED | impl: `pub struct GgufMetadata { entries: HashMap<String, GgufValue> }` in `gguf.rs`; non-test consumer: `pub use gguf::GgufMetadata` in `lib.rs`; `pub struct GgufFile.metadata` is `GgufMetadata`. |
| REQ-4 | SHIPPED | impl: `#[non_exhaustive] pub struct GgufTensorInfo` in `gguf.rs`; non-test consumer: `pub use gguf::GgufTensorInfo` in `lib.rs`; the `GgufFile.tensors: Vec<GgufTensorInfo>` field is the production hand-off path between parse and dequant. |
| REQ-5 | SHIPPED | impl: `#[non_exhaustive] pub struct GgufFile` in `gguf.rs` with private `data: Vec<u8>`; non-test consumer: `pub use gguf::GgufFile` in `lib.rs`; every public dequant function takes a `&GgufFile`. |
| REQ-6 | SHIPPED | impl: `pub fn load_gguf` + `pub fn parse_gguf_bytes` in `gguf.rs` with the bounded-reader pattern and the OOM-defense comment block (lines 412-419); non-test consumer: `pub use gguf::{load_gguf, parse_gguf_bytes}` in `lib.rs`; `load_gguf_state_dict` calls `load_gguf` for the production ingest path. |
| REQ-7 | SHIPPED | impl: `pub fn load_gguf_mmap` in `gguf.rs` using `Mmap::map`; non-test consumer: `pub use gguf::load_gguf_mmap` in `lib.rs`; `load_gguf_state_dict_mmap` calls this for the production zero-copy ingest. |
| REQ-8 | SHIPPED | impl: seven `fn dequantize_{f32,f16,q4_0,q4_1,q5_0,q5_1,q8_0,q8_1}` functions + `fn dequantize_data` dispatcher + `fn f16_to_f32` helper in `gguf.rs`; non-test consumer: `dequantize_gguf_tensor` calls `dequantize_data` for every tensor it loads. |
| REQ-9 | SHIPPED | impl: `pub fn dequantize_gguf_tensor(file, tensor_name) -> FerrotorchResult<Tensor<f32>>` in `gguf.rs`; non-test consumer: `pub use gguf::dequantize_gguf_tensor` in `lib.rs`; `load_gguf_state_dict` and `load_gguf_state_dict_mmap` both iterate `file.tensors` and call `dequantize_gguf_tensor` per tensor. |
| REQ-10 | SHIPPED | impl: `pub fn load_gguf_state_dict` + `pub fn load_gguf_state_dict_mmap` in `gguf.rs`; non-test consumer: `pub use gguf::{load_gguf_state_dict, load_gguf_state_dict_mmap}` in `lib.rs`; production inference paths in `ferrotorch-llama` load TheBloke GGUF checkpoints via these entries. |
| REQ-11 | SHIPPED | impl: `Reader::align_to` + the `general.alignment` metadata lookup in `parse_gguf_bytes` in `gguf.rs`; non-test consumer: every `parse_gguf_bytes` call aligns the data-section start using this code path. |
| REQ-12 | SHIPPED | impl: `struct Reader<'a>` with bounds-checked `read_u32` / `read_u64` / `read_bytes` / `read_gguf_string` methods in `gguf.rs`; non-test consumer: every byte the parser reads goes through the bounded reader; the OOM-attack-resistance test pins the contract. |
| REQ-13 | SHIPPED | impl: `pub fn GgufFile::data(&self) -> &[u8]` accessor in `gguf.rs` (S3: `impl GgufFile` block) exposing the private `data: Vec<u8>` tensor-block region; non-test consumer: `ferrotorch-llama` `fn gguf_tensor_to_bf16_cudarc in gpu_gguf.rs` calls `file.data().get(offset..offset + byte_len)` to slice each quantized tensor's on-disk blocks for the cubecl→cudarc GPU dequant bridge (#1350). |
