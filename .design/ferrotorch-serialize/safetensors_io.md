# ferrotorch-serialize — `safetensors_io` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/serialization.py
  - torch/_weights_only_unpickler.py
-->

## Summary

`ferrotorch-serialize/src/safetensors_io.rs` (2025 LOC) is the
production-grade save/load path for the HuggingFace
[safetensors](https://huggingface.co/docs/safetensors/) format,
covering single-file, mmap, and sharded (`model.safetensors.index.json`)
checkpoint layouts that current LLM/Vision releases ship in. The
underlying wire layout is delegated to the [`safetensors`
crate](https://crates.io/crates/safetensors) — this module wires
that crate into ferrotorch's `StateDict<T>` type, adds the sharded
index handling HuggingFace formalizes for >2 GB checkpoints, and
ships the rayon-parallel + mmap performance paths called out in
ferrotorch issues #587 / #1127 / #1178.

The conceptual upstream is `torch.load("model.safetensors")` /
`torch.save(..., "model.safetensors")` (when HuggingFace's
`safetensors` Python package is installed) — both upstream PyTorch
and ferrotorch defer the actual format spec to the `safetensors`
ecosystem (R-DEV-3: external on-disk spec; R-DEV-7: Rust analog is
materially better). The serial-vs-parallel decode strategy and the
sharded loader are ferrotorch-side ergonomics for huge HF
checkpoints (Llama 3 8B / 70B, Mixtral, SDXL UNet).

## Requirements

- REQ-1: `pub fn save_safetensors<T: Float>(state: &StateDict<T>,
  path: impl AsRef<Path>) -> FerrotorchResult<()>` writes a
  `StateDict<T>` (T = f32 or f64) as a HuggingFace-compatible
  `.safetensors` file. Sorted keys for deterministic output. The
  byte layout is delegated to `safetensors::serialize_to_file`;
  ferrotorch builds the `Vec<(String, TensorView)>` from the
  state dict's `&[T]` reinterpreted as `&[u8]` via the documented
  `unsafe` reborrow.

- REQ-2: `pub fn load_safetensors<T: Float>(path) ->
  FerrotorchResult<StateDict<T>>` reads a single-file
  safetensors checkpoint into memory and decodes every tensor.
  Supports `f32` and `f64` as the target type; dtype mismatches
  surface via `FerrotorchError::DtypeMismatch`. Auto-promotes
  `bf16` and `f16` stored tensors to `f32` (the dominant inference
  dtype) via `half::{bf16, f16}::to_f32` — required for HuggingFace
  models since modern transformer checkpoints are bf16 by default.

- REQ-3: `pub fn load_safetensors_mmap<T: Float>(path) ->
  FerrotorchResult<StateDict<T>>` (#587) memory-maps the on-disk
  bytes with `memmap2::Mmap` instead of slurping into a heap
  `Vec<u8>`. Halves peak RSS during the decode phase: file pages
  are owned by the OS page cache, only the decoded `Tensor<T>`
  buffers are Rust-owned. The mmap is dropped before the function
  returns; tensor data is copied into owned buffers by
  `decode_view`, so callers don't inherit any file-lifetime
  contract.

- REQ-4: Sharded HuggingFace index support. `pub struct
  SafeTensorsIndex` (`#[non_exhaustive]`, derives `Deserialize`)
  with `pub metadata: SafeTensorsIndexMetadata` (carrying
  `total_size: u64`) and `pub weight_map: HashMap<String, String>`
  (tensor name → shard filename). Helper methods `shard_files() ->
  Vec<String>` and `group_by_shard() -> HashMap<String,
  Vec<String>>`. Parsed via `SafeTensorsIndex::from_file(path)
  -> FerrotorchResult<Self>`. Mirrors HuggingFace's
  `model.safetensors.index.json` layout used by Llama / Mistral /
  Mixtral / Falcon checkpoints.

- REQ-5: `pub fn load_safetensors_sharded<T: Float>(index_path)
  -> FerrotorchResult<StateDict<T>>` loads every shard the index
  declares and merges into one `StateDict<T>`. Shards are loaded
  in parallel via rayon (CL-1127 part A) — `par_iter` over the
  sorted shard list, each shard decoded into its own
  `StateDict<T>` on a worker thread, then a serial merge at the
  end. Every key the index declares must be found in some shard;
  a missing key produces `InvalidArgument`.

- REQ-6: `pub fn load_safetensors_sharded_mmap<T: Float>(index_path)
  -> FerrotorchResult<StateDict<T>>` is the mmap counterpart of
  REQ-5. Mirrors the same parallel-shards strategy but each shard
  is mmap'd instead of read into a `Vec<u8>` (#587). Peak RSS is
  bounded by the decoded shard size only.

- REQ-7: `pub fn load_safetensors_auto<T: Float>(path) ->
  FerrotorchResult<StateDict<T>>` dispatches on the filename
  suffix: `*.index.json` → sharded loader, anything else →
  single-file loader. This is the user-friendly entry that
  matches HuggingFace's pattern of "give me a path, I'll figure
  out if it's a single file or a sharded directory."

- REQ-8: `pub struct ShardProgress<'a>` (`#[non_exhaustive]`,
  `Copy`) + `pub fn load_safetensors_sharded_with_progress<T, F>(
  index_path, progress: F) -> FerrotorchResult<StateDict<T>>` where
  `F: FnMut(ShardProgress<'_>)`. The callback fires once before
  each shard is opened with shard_index / shard_count /
  shard_file / tensors_in_shard / tensors_loaded_so_far /
  total_tensors. (#586) Useful for progress bars on the ~140 GB
  Llama 3 70B checkpoint load.

- REQ-9: `pub fn load_safetensors_sharded_filtered<T, F>(
  index_path, predicate: F) -> FerrotorchResult<StateDict<T>>`
  where `F: Fn(&str) -> bool`. Loads only tensors whose name
  matches `predicate`. The typical use is "load only the encoder
  weights" or "load layer 12 only" for inference servers and
  LoRA / adapter training that wants to skip the base model.
  (#586) Shards containing zero matching tensors are skipped
  entirely.

- REQ-10: f16/bf16 → f32 upcast vectorization (CL-1127 part B).
  `fn decode_view` reinterprets a half-precision byte buffer as
  `&[u16]` via `bytemuck::cast_slice` and delegates the conversion
  to `half::f16::to_f32` / `half::bf16::to_f32`, which LLVM
  auto-vectorizes (`cvtph2ps` for f16 with F16C, `unpcklwd` +
  `pslld` shift for bf16). Specialized `T == f32` path avoids the
  `numeric_cast` call entirely so the loop collapses to a pure
  half→f32 map.

- REQ-11: Decode strategy dispatch (CL-1127). `fn
  decode_tensor_list` peeks the first tensor's dtype: if it's
  `BF16` / `F16` it dispatches to `rayon par_iter` (the SIMD
  upcast scales linearly with worker count); for native `F32` /
  `F64` it falls back to a serial loop (the pure mmap → Vec memcpy
  is memory-bandwidth bound and extra workers add page-fault
  contention — measured 315s → 379s regression on SD-1.5 UNet
  with 16 workers). An operator-controlled env var
  `FERROTORCH_FORCE_SERIAL_LOAD=1` overrides for
  diagnostics.

- REQ-12: Endianness contract. Safetensors mandates little-endian
  on-disk. `fn as_le_bytes` reinterprets `&[T]` as `&[u8]` via
  `slice::from_raw_parts(data.as_ptr().cast::<u8>(),
  size_of_val(data))`. The crate-root LE assumption (no
  `compile_error!` in this file specifically, but the platform
  invariant is documented in the SAFETY block).

## Acceptance Criteria

- [x] AC-1: Single-file f32 + f64 round-trip
  (`test_save_load_roundtrip_f32` / `_f64`).
- [x] AC-2: Tensor names and shapes survive round-trip
  (`test_correct_tensor_names_and_shapes`).
- [x] AC-3: bf16 / f16 stored tensors upcast to f32 (relevant
  tests in the integration suite).
- [x] AC-4: `SafeTensorsIndex::from_file` parses
  HuggingFace-format index JSON.
- [x] AC-5: `load_safetensors_sharded` aggregates a multi-shard
  checkpoint (covered by `tests/conformance_serialize_sharded.rs`).
- [x] AC-6: `load_safetensors_mmap` produces byte-identical output
  to `load_safetensors` (covered by
  `tests/conformance_serialize_mmap.rs`).
- [x] AC-7: `load_safetensors_auto` dispatches on `.index.json`
  vs anything else.
- [x] AC-8: `load_safetensors_sharded_filtered` skips shards with
  no matching tensors and the resulting StateDict contains only
  predicate-accepted keys.
- [x] AC-9: `load_safetensors_sharded_with_progress` fires the
  callback exactly once per shard with monotonically increasing
  `tensors_loaded_so_far`.
- [x] AC-10: `FERROTORCH_FORCE_SERIAL_LOAD=1` env override forces
  the serial decode path on a bf16 file.

## Architecture

### Save (REQ-1)

`save_safetensors` builds a `Vec<TensorEntry>` (intermediate retainer
for the owned `Vec<usize>` shape + borrowed `&[u8]` byte data),
constructs the `Vec<(String, TensorView)>` the safetensors crate
expects, and delegates writing to `safetensors::serialize_to_file`.
Sorting the keys before building the entry list guarantees
deterministic byte output across runs.

### Single-file load (REQ-2)

`load_safetensors` `std::fs::read`'s the entire file, calls
`SafeTensors::deserialize`, and runs the result through
`decode_tensor_list` (REQ-11). `decode_view` is the per-tensor
decoder: it handles dtype matching, the f16/bf16 upcast (REQ-10),
and the shape validation.

### MMap load (REQ-3)

`load_safetensors_mmap` opens the file, calls `Mmap::map(&file)`
(documented `unsafe` block — the SAFETY comment cites the
mmap-vs-mutation contract), parses the safetensors header in
place, then runs `decode_view` per tensor copying into owned
buffers. The mmap is dropped at function exit so no borrow
escapes.

### Sharded index format (REQ-4)

```rust
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct SafeTensorsIndex {
    pub metadata: SafeTensorsIndexMetadata,
    pub weight_map: HashMap<String, String>,
}
```

`weight_map` maps tensor name (e.g.
`"model.layers.0.self_attn.q_proj.weight"`) to shard filename
(`"model-00001-of-00004.safetensors"`). `group_by_shard()`
inverts the map so the loader knows which tensors to expect per
shard.

### Parallel sharded load (REQ-5, REQ-6)

```rust
let per_shard: Vec<StateDict<T>> = shard_files
    .par_iter()
    .map(|shard_file| load_one_shard_owned::<T>(...))
    .collect::<FerrotorchResult<Vec<_>>>()?;
```

Each shard is decoded on its own rayon worker into an owned
`StateDict<T>`. The merge at the end is `HashMap::extend` — O(N)
hash inserts dwarfed by the per-shard byte traversal we just
parallelized. The previous loop was strictly sequential even
though each shard is on disk and embarrassingly parallel.
`load_one_shard_owned` (read-into-Vec) and
`load_one_shard_owned_mmap` (mmap'd) share the inner
`decode_shard_tensors` so the dtype handling and missing-key
validation stay in one place.

### Auto-detect (REQ-7)

```rust
if filename.ends_with(".index.json") {
    load_safetensors_sharded(p)
} else {
    load_safetensors(p)
}
```

Simple suffix dispatch; the convention is HuggingFace's.

### Progress callback (REQ-8)

```rust
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ShardProgress<'a> {
    pub shard_index: usize,
    pub shard_count: usize,
    pub shard_file: &'a str,
    pub tensors_in_shard: usize,
    pub tensors_loaded_so_far: usize,
    pub total_tensors: usize,
}
```

The shard list is iterated SERIALLY (no rayon parallelism) so the
callback fires in the order the user expects — using
`par_iter` would interleave callbacks unpredictably. This is the
explicit ergonomics trade-off for the progress-bar use case.

### Filtered load (REQ-9)

`load_safetensors_sharded_filtered` filters per-shard expected
keys before calling `load_one_shard_into`. Shards whose filtered
key list is empty are skipped entirely (no file open, no parse).
The serial-shard-iteration here is deliberate: the predicate may
have side effects (logging, accounting); we don't want to call it
concurrently across shards.

### f16/bf16 vectorized upcast (REQ-10)

```rust
let raw_u16: &[u16] = bytemuck::cast_slice(byte_data);
// T == f32 fast path:
let data: Vec<f32> = if is_bf16 {
    raw_u16.iter().map(|&b| bf16::from_bits(b).to_f32()).collect()
} else {
    raw_u16.iter().map(|&b| f16::from_bits(b).to_f32()).collect()
};
```

CL-1127 part B audit goal: the original loop did per-byte
`u16::from_le_bytes` + a fallible `numeric_cast` per element,
which blocked SIMD. The current loop is a pure `map` LLVM
auto-vectorizes — `unpcklwd` + `pslld` for bf16, `cvtph2ps` (F16C)
or table-lookup for f16.

### Serial-vs-parallel decode dispatch (REQ-11)

```rust
let parallel_dtype = tensor_list
    .first()
    .is_some_and(|(_, v)| matches!(v.dtype(), Dtype::BF16 | Dtype::F16));
let force_serial = std::env::var_os("FERROTORCH_FORCE_SERIAL_LOAD").is_some();
if parallel_dtype && !force_serial { /* rayon par_iter */ }
else { /* serial loop */ }
```

The rationale block in the file documents the SD-1.5 UNet
benchmark (315 s serial → 379 s parallel on F32) and the bf16 case
(linear speedup with worker count). Files are nearly always
homogeneous in dtype; a single peek at `first()` is sufficient.

### Endianness (REQ-12)

```rust
unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), size_of_val(data)) }
```

The SAFETY block names the four invariants: `T` is POD/Copy/no
padding, byte length matches `size_of_val`, the returned slice
reborrows, big-endian targets would produce wrong bytes (the
crate-root LE assumption applies).

### Non-test production consumers

- `pub use safetensors_io::{save_safetensors, load_safetensors,
  load_safetensors_auto, load_safetensors_mmap,
  load_safetensors_sharded, load_safetensors_sharded_filtered,
  load_safetensors_sharded_mmap,
  load_safetensors_sharded_with_progress, ShardProgress}` in
  `lib.rs` and the meta-crate glob make these reachable as
  `ferrotorch::save_safetensors` etc.
- Downstream model crates (`ferrotorch-llama`,
  `ferrotorch-bert`, `ferrotorch-whisper`, `ferrotorch-diffusion`)
  call `ferrotorch::load_safetensors_auto(path)?` to ingest
  HuggingFace checkpoints — this is the primary production
  consumer path; sharded loaders are how the 8B/70B Llama
  weights actually load.

## Parity contract

`parity_ops = []`. The on-disk byte format is the safetensors
crate's contract (R-DEV-3) so ferrotorch's correctness is checked
against the spec, not against PyTorch. Edge-case contract:

- **Bit-exact round-trip for f32/f64**: write → read produces
  byte-identical `Vec<T>` data. Verified by
  `test_save_load_roundtrip_f32` / `_f64`.
- **bf16/f16 upcast to f32**: the half-precision dtypes are
  upcast on load (lossy in the bf16→f32 direction is exact since
  bf16 is a strict subset of f32; f16→f32 is exact since f16's
  exponent/mantissa fit inside f32). NaN / Inf / denormal handling
  delegated to `half::{bf16, f16}::to_f32`.
- **Shard ordering deterministic**: shard files sorted
  lexicographically before the rayon dispatch (so error messages
  name the same shard run-to-run for the same input).
- **Missing tensor in index → error**: a key declared in
  `weight_map` but not present in any shard produces
  `InvalidArgument` with the tensor name in the message.
- **`load_safetensors_mmap` vs `load_safetensors` byte equality**:
  both paths produce the same `StateDict<T>` for the same input
  file. Tested by `tests/conformance_serialize_mmap.rs`.

## Verification

Tests in `mod tests in safetensors_io.rs` (31 unit tests) cover:

- Single-file round-trip for f32 + f64.
- Tensor name/shape preservation.
- Sharded index parsing.
- Sharded round-trip with synthetic multi-shard fixtures.
- mmap path.
- Filtered + progress-callback paths.

Integration tests in `tests/conformance_serialize.rs`,
`tests/conformance_serialize_mmap.rs`, and
`tests/conformance_serialize_sharded.rs` exercise the public API
end-to-end against fixture safetensors files.

Smoke command:

```bash
cargo test -p ferrotorch-serialize --lib safetensors_io:: 2>&1 | tail -3
```

Expected: 31 passed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn save_safetensors<T: Float>` in `safetensors_io.rs` building the `Vec<(String, TensorView)>` and delegating to `safetensors::serialize_to_file`; non-test consumer: `pub use safetensors_io::save_safetensors` in `lib.rs`, available as `ferrotorch::save_safetensors`; training checkpoint export paths in production code call this entry. |
| REQ-2 | SHIPPED | impl: `pub fn load_safetensors<T: Float>` in `safetensors_io.rs` + `fn decode_tensor_list` + `fn decode_view` doing dtype-aware decode; non-test consumer: `pub use safetensors_io::load_safetensors` in `lib.rs`; downstream model loaders call this entry for single-file HuggingFace checkpoints. |
| REQ-3 | SHIPPED | impl: `pub fn load_safetensors_mmap<T: Float>` in `safetensors_io.rs` using `memmap2::Mmap::map`; non-test consumer: `pub use safetensors_io::load_safetensors_mmap` in `lib.rs`; production inference servers prefer this entry on large checkpoints to halve peak RSS. |
| REQ-4 | SHIPPED | impl: `#[non_exhaustive] pub struct SafeTensorsIndex` + `SafeTensorsIndexMetadata` + `impl SafeTensorsIndex::{from_file, shard_files, group_by_shard}` in `safetensors_io.rs`; non-test consumer: `load_safetensors_sharded` / `_mmap` / `_with_progress` / `_filtered` all construct an index via `SafeTensorsIndex::from_file`, and the index type is `pub`-re-exported via the sharded loaders. |
| REQ-5 | SHIPPED | impl: `pub fn load_safetensors_sharded<T: Float>` in `safetensors_io.rs` with `par_iter` over sorted shard files + serial merge + cross-check for missing keys; non-test consumer: `pub use safetensors_io::load_safetensors_sharded` in `lib.rs`; downstream `ferrotorch::load_safetensors_auto` dispatches to this path when handed a `*.index.json` file. |
| REQ-6 | SHIPPED | impl: `pub fn load_safetensors_sharded_mmap<T: Float>` + `fn load_one_shard_owned_mmap` in `safetensors_io.rs`; non-test consumer: `pub use safetensors_io::load_safetensors_sharded_mmap` in `lib.rs`; production loaders prefer this on disk-bound multi-shard transformer checkpoints. |
| REQ-7 | SHIPPED | impl: `pub fn load_safetensors_auto<T: Float>` in `safetensors_io.rs` dispatching on `.index.json` suffix; non-test consumer: `pub use safetensors_io::load_safetensors_auto` in `lib.rs`; the meta-crate glob makes `ferrotorch::load_safetensors_auto` the primary user-facing entry. |
| REQ-8 | SHIPPED | impl: `#[non_exhaustive] pub struct ShardProgress<'a>` + `pub fn load_safetensors_sharded_with_progress<T, F>` in `safetensors_io.rs` firing the callback once per shard pre-open; non-test consumer: `pub use safetensors_io::{ShardProgress, load_safetensors_sharded_with_progress}` in `lib.rs`; CLI / TUI tools for loading 70B+ checkpoints wire progress bars through this entry. |
| REQ-9 | SHIPPED | impl: `pub fn load_safetensors_sharded_filtered<T, F>` in `safetensors_io.rs` filtering per-shard expected keys before opening; non-test consumer: `pub use safetensors_io::load_safetensors_sharded_filtered` in `lib.rs`; LoRA / adapter training loops use this entry to skip base-model weights. |
| REQ-10 | SHIPPED | impl: `fn half_to_f32` / `fn bf16_to_f32` + `bytemuck::cast_slice` half-buffer reinterpret + `T == f32` fast-path specialization in `decode_view` in `safetensors_io.rs`; non-test consumer: every bf16 / f16 safetensors load hits this code path through `decode_view`, including the production HuggingFace transformer paths. |
| REQ-11 | SHIPPED | impl: `fn decode_tensor_list` in `safetensors_io.rs` with the bf16/f16 → rayon, f32/f64 → serial dispatch + `FERROTORCH_FORCE_SERIAL_LOAD` override; non-test consumer: both `load_safetensors` and `load_safetensors_mmap` route through `decode_tensor_list`, so every single-file load benefits from the dispatch. |
| REQ-12 | SHIPPED | impl: `fn as_le_bytes` in `safetensors_io.rs` with the SAFETY block; non-test consumer: `save_safetensors` calls `as_le_bytes` for every tensor; the crate-root LE invariant (mirrored from `state_dict.rs`'s `compile_error!`) governs platform support. |
