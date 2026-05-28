# Distributed checkpointing

<!--
tier: 3-component
status: draft
baseline-pytorch: main (user's local clone /home/doll/pytorch)
upstream-paths:
  - torch/distributed/checkpoint/__init__.py
  - torch/distributed/checkpoint/api.py
  - torch/distributed/checkpoint/state_dict_saver.py
  - torch/distributed/checkpoint/state_dict_loader.py
-->

## Summary

`ferrotorch-distributed/src/checkpoint.rs` implements per-rank shard
saving / loading / resharding for distributed training state dicts.
Each rank writes its own SafeTensors file to a shared directory; rank
0 additionally writes a JSON metadata file describing the sharding
layout. On load, if the world size has changed since save, the
resharding logic reconstructs the full tensor by concatenating
shards along the shard dimension, then re-splits for the new world
size. An async checkpointer (`AsyncCheckpointer`) stages tensor data
to CPU memory then writes in a background thread without blocking
the training loop. Mirrors `torch.distributed.checkpoint`'s
save/load/`async_save` surface.

## Requirements

- REQ-1: `pub enum DistCheckpointError` with `Debug` +
  `thiserror::Error` + `#[non_exhaustive]`. Variants: `Io`,
  `Serialization`, `Metadata`, `MissingShard`, `Tensor`,
  `InvalidArgument`, `AsyncFailed`. `From<DistCheckpointError> for
  FerrotorchError` bridge via `InvalidArgument`. `From<std::io::Error>
  for DistCheckpointError` shortcut for I/O paths. Mirrors
  `class CheckpointException(BaseException)` at
  `torch/distributed/checkpoint/api.py:30` (PyTorch wraps per-rank
  failures; ferrotorch errors are surfaced inline since checkpoint
  ops are per-process).
- REQ-2: `pub struct TensorShardSpec { full_shape: Vec<usize>,
  shard_dim: usize, shard_sizes: Vec<usize> }` with `Serialize +
  Deserialize + Clone + Debug + #[non_exhaustive]`. Describes how
  one tensor is partitioned across ranks: which dimension splits,
  per-rank sizes along that dimension.
- REQ-3: `pub struct ShardMetadata { num_ranks: usize, tensor_specs:
  HashMap<String, TensorShardSpec> }` with the same trait set. This
  is the JSON-serializable metadata file written by rank 0.
- REQ-4: `pub struct DistributedCheckpoint { checkpoint_dir: PathBuf,
  shard_metadata: ShardMetadata }` with `#[non_exhaustive]`. Owning
  handle for a checkpoint directory.
- REQ-5: `pub fn save_distributed<T: Float>(state_dict, dir, rank,
  world_size, shard_spec) -> Result<(), DistCheckpointError>` writes
  this rank's shard to `dir/rank_{rank}.safetensors` and (if rank 0)
  writes `dir/metadata.json` via `serde_json::to_string_pretty`.
  Validates `world_size >= 1` and `rank < world_size`. Creates the
  directory if it doesn't exist.
- REQ-6: `pub fn load_distributed<T: Float>(dir, rank, world_size) ->
  Result<HashMap<String, Tensor<T>>, DistCheckpointError>` reads
  `dir/metadata.json`, compares the saved `num_ranks` against the
  current `world_size`. If equal, loads this rank's shard file
  directly. If different, delegates to `reshard`. Handles the
  legitimate empty-checkpoint case (no tensor_specs) without
  erroring.
- REQ-7: `pub fn reshard<T: Float>(dir, old_world_size, new_world_size,
  new_rank) -> Result<HashMap<String, Tensor<T>>,
  DistCheckpointError>` reconstructs each tensor by loading all
  `old_world_size` shards and concatenating along `shard_dim`, then
  re-splits the full tensor for `new_world_size` using
  `compute_shard_sizes` (even split with `total % num_parts` parts
  getting one extra element). Returns this `new_rank`'s slice.
- REQ-8: SafeTensors interop: `fn st_dtype<T>` maps `size_of::<T>()`
  to `Dtype::F32` / `Dtype::F64` (bf16/f16 reach `InvalidArgument`).
  `fn as_le_bytes<T>(&[T]) -> &[u8]` byte-reinterprets via
  `from_raw_parts` inside an `unsafe { ... }` block with a 6-bullet
  SAFETY substantiation. `fn save_tensors_to_file` /
  `fn load_tensors_from_file` are the file-level helpers. Load uses
  `ptr::read_unaligned` to handle source-buffer alignment, also
  SAFETY-documented.
- REQ-9: `fn concat_along_dim<T>(shard_datas, shard_shapes, dim,
  full_shape) -> Result<Vec<T>, DistCheckpointError>` walks the
  outer / middle / inner dimensions and copies each shard's data to
  the right offset in the output. Validates non-`dim` dimensions
  match `full_shape`.
- REQ-10: `fn slice_along_dim<T>(data, shape, dim, offset, size) ->
  Vec<T>` is the inverse of `concat_along_dim`: extracts a
  contiguous slice along `dim`.
- REQ-11: `pub struct CheckpointFuture` and `pub struct
  AsyncCheckpointer`. `AsyncCheckpointer::save_async` stages tensor
  data to CPU memory then spawns a background thread writing the
  shard file and (for rank 0) the metadata. A `Arc<Mutex<bool>>`
  `in_flight` guard prevents concurrent saves. `CheckpointFuture::wait`
  blocks until the background thread completes; idempotent on
  repeat calls (cached result). `CheckpointFuture::is_done` peeks
  without joining.
- REQ-12: `pub fn flat_shard_metadata(state_dict, world_size) ->
  ShardMetadata` is the convenience constructor for the FSDP-style
  flat-1D-shard case: every tensor is treated as a 1-D buffer with
  `shard_dim = 0` and `shard_sizes = vec![shard_numel; world_size]`.

## Acceptance Criteria

- [x] AC-1: Single-rank save → load round-trips
  `state_dict = {"weight": [1,2,3,4], "bias": [0.1, 0.2]}`.
- [x] AC-2: Two-rank save with `weight` half-split → two-rank load
  reconstructs each rank's expected half.
- [x] AC-3: 2-rank save → 4-rank load resplits along dim 0.
- [x] AC-4: 4-rank save → 2-rank load (scale-down) resplits along
  dim 0.
- [x] AC-5: 2-D tensor sharded along dim 0 or dim 1 round-trips
  through resharding.
- [x] AC-6: Uneven 3→2 resharding splits the full tensor as evenly
  as possible.
- [x] AC-7: `load_distributed` automatically reshards when the
  saved `num_ranks` differs from the current `world_size`.
- [x] AC-8: `compute_shard_sizes` produces even splits when
  `total % num_parts == 0` and uneven splits (first `remainder`
  parts get +1) otherwise.
- [x] AC-9: `concat_along_dim` / `slice_along_dim` round-trip 1-D
  and 2-D inputs along both axes.
- [x] AC-10: `AsyncCheckpointer::save_async` writes a valid
  checkpoint; `wait()` is idempotent on repeat calls; `is_done()`
  reports completion.
- [x] AC-11: Save with `rank >= world_size` is rejected.
- [x] AC-12: Load with no metadata file errors; load with
  metadata but missing shard file errors.

## Architecture

### Error type (REQ-1)

`pub enum DistCheckpointError` (in `checkpoint.rs`) carries 7
variants; `#[non_exhaustive]` enables future categories. The two
`From` impls (`From<DistCheckpointError> for FerrotorchError`,
`From<std::io::Error> for DistCheckpointError`) keep error-handling
ergonomic at the call site (single `?` operator works).

### Metadata types (REQ-2, REQ-3, REQ-4)

`pub struct TensorShardSpec` (in `checkpoint.rs`) is the per-tensor
descriptor; `pub struct ShardMetadata` is the overall manifest. Both
derive `Serialize` / `Deserialize` and `Clone`, enabling round-trip
through serde_json. `pub struct DistributedCheckpoint` is the
owning handle pairing a directory with its metadata.

### Save / load (REQ-5, REQ-6)

`pub fn save_distributed` (in `checkpoint.rs`):

1. Validate `world_size >= 1` and `rank < world_size`.
2. `std::fs::create_dir_all(dir)`.
3. Save this rank's shard to `dir/rank_{rank}.safetensors` via
   `save_tensors_to_file`.
4. If rank 0: serialize `shard_spec` to pretty JSON and write to
   `dir/metadata.json`.

`pub fn load_distributed` (in `checkpoint.rs`):

1. Validate same inputs.
2. Read and parse `dir/metadata.json` into `ShardMetadata`.
3. Compare `metadata.num_ranks` against `world_size`. If equal,
   load the shard file directly. Otherwise call `reshard`.
4. Empty-checkpoint case: if `tensor_specs` is empty and the shard
   file is missing, return an empty `HashMap` instead of erroring.

### Resharding (REQ-7, REQ-9, REQ-10)

`pub fn reshard` (in `checkpoint.rs`) is the workhorse for world-size
changes:

1. Load every old shard via `load_tensors_from_file`.
2. For each tensor in the metadata:
   - Collect each old shard's data and shape.
   - Concatenate along `shard_dim` via `concat_along_dim` to get the
     full tensor.
   - Compute new shard sizes via `compute_shard_sizes(full_dim_size,
     new_world_size)`.
   - Slice along `shard_dim` at `new_offset = sum(new_shard_sizes[..new_rank])`
     for `new_size = new_shard_sizes[new_rank]`.

`fn compute_shard_sizes` (in `checkpoint.rs`) returns `Vec<usize>` of
length `num_parts`; the first `total % num_parts` parts get an
extra element. This matches PyTorch's `_chunk_by_size`-style
distribution.

`fn concat_along_dim` and `fn slice_along_dim` are 3-level loop
helpers (outer/middle/inner) that handle multi-dimensional tensors
laid out in row-major order.

### SafeTensors interop (REQ-8)

`fn st_dtype<T>` (in `checkpoint.rs`) maps element size to
`safetensors::tensor::Dtype`. `fn as_le_bytes<T>(&[T]) -> &[u8]`
byte-reinterprets via `std::slice::from_raw_parts` inside a
documented `unsafe { ... }` block; the 6-bullet SAFETY comment
covers VALIDITY / LIFETIME / LENGTH / ALIGNMENT / ALIASING /
ENDIANNESS. `fn save_tensors_to_file` builds
`Vec<(String, TensorView<'_>)>` entries (sorted by key to keep
on-disk layout deterministic) and writes via
`safetensors::serialize_to_file`.

`fn load_tensors_from_file` (in `checkpoint.rs`) reads the bytes,
calls `SafeTensors::deserialize`, then for each tensor:

- Validates dtype matches `T`.
- Validates byte count matches `numel * elem_size`.
- Decodes via `chunks_exact(elem_size).map(...)` where the inner
  map uses `std::ptr::read_unaligned(bytes.as_ptr() as *const T)`
  inside an `unsafe { ... }` block. SAFETY is documented at length
  (VALIDITY / LENGTH / ALIGNMENT / LIFETIME / PROVENANCE /
  ENDIANNESS) — `read_unaligned` is required because the byte
  buffer is `[u8; 8]` (1-aligned) which doesn't satisfy `T`'s
  alignment requirement.

### Async checkpointing (REQ-11)

`pub struct CheckpointFuture` (in `checkpoint.rs`) holds an
`Option<JoinHandle>` and a cached `Option<Result>`. `wait()` joins
the handle on first call, caches the result, and returns the cached
result on subsequent calls. `is_done()` peeks via `JoinHandle::is_finished`
without joining.

`pub struct AsyncCheckpointer` (in `checkpoint.rs`) holds `dir`,
`rank`, `world_size`, `shard_spec`, and an `Arc<Mutex<bool>>`
`in_flight` guard. `pub fn save_async`:

1. Acquire the `in_flight` lock; reject if another save is running.
2. Stage every tensor's data to a CPU-owned `Vec<f32>` on the
   calling thread (this is the part that touches GPU memory and
   must not move).
3. Capture the dir + rank + spec + the staged data + the lock arc
   into a `move` closure; spawn a thread.
4. The thread rebuilds tensors from the staged data, calls
   `save_tensors_to_file`, and (for rank 0) writes metadata. On
   completion (success or failure), releases the `in_flight`
   guard.
5. Return a `CheckpointFuture` wrapping the join handle.

The implementation is hard-coded to `Tensor<f32>` (not generic over
`T`) because the staging step needs a concrete element type for the
captured closure's `Send` bound. Generic-over-`T` async support is
a follow-up.

### Flat-shard convenience (REQ-12)

`pub fn flat_shard_metadata(state_dict, world_size)` (in
`checkpoint.rs`) iterates the state dict and emits a
`ShardMetadata` with every tensor treated as 1-D
`[shard_numel * world_size]` with equal `shard_sizes`. This is the
FSDP-default sharding shape; it covers the most common case so
callers don't have to spell out `TensorShardSpec` per tensor.

### Consumer sites (production, non-test)

- `ferrotorch-distributed/src/lib.rs` `pub use checkpoint::{
  AsyncCheckpointer, CheckpointFuture, DistCheckpointError,
  DistributedCheckpoint, ShardMetadata, TensorShardSpec,
  flat_shard_metadata, load_distributed, reshard, save_distributed
};` re-exports the full surface. `ferrotorch/src/lib.rs` re-exports
  the whole crate via `pub use ferrotorch_distributed::*;`.
- Within `checkpoint.rs`, `load_distributed` is a production
  consumer of `reshard` (called when world sizes differ); `reshard`
  is a production consumer of `load_tensors_from_file`,
  `concat_along_dim`, `compute_shard_sizes`, and `slice_along_dim`.
- `save_async` is a production consumer of `save_tensors_to_file`
  via the background thread closure.

## Parity contract

No parity-sweep ops in the route (`parity_ops = []`). The contract is
the PyTorch checkpoint shape:

- `torch.distributed.checkpoint.save(state_dict, ...)` and
  `torch.distributed.checkpoint.load(state_dict, ...)` re-exported
  from `torch/distributed/checkpoint/__init__.py:20` → ferrotorch's
  `save_distributed` / `load_distributed`. R-DEV-7 deviation:
  PyTorch's planner / storage-reader / storage-writer abstractions
  are flattened into direct save/load functions; the underlying
  on-disk format is SafeTensors (a Rust-ecosystem analog of
  PyTorch's serialization).
- `torch.distributed.checkpoint.async_save` re-exported from
  `torch/distributed/checkpoint/__init__.py:20` → ferrotorch's
  `AsyncCheckpointer::save_async`. R-DEV-7 deviation: returns a
  thread `JoinHandle`-wrapped `CheckpointFuture` instead of a
  `torch.futures.Future`.
- `CheckpointException` at `torch/distributed/checkpoint/api.py:30`
  → ferrotorch's `DistCheckpointError`. R-DEV-1 deviation: error
  variants are typed rather than wrapping a `dict[int,
  WRAPPED_EXCEPTION]` since ferrotorch errors fire per-process.

## Verification

`cargo test -p ferrotorch-distributed --lib checkpoint::` runs the
`#[cfg(test)] mod tests` block at lines 920-1738 covering 25 tests:

- Save/load: `test_save_load_single_rank`,
  `test_save_load_two_ranks`, `test_save_invalid_rank`,
  `test_load_missing_metadata`, `test_load_missing_shard`,
  `test_load_distributed_reshards_when_world_size_differs`.
- Resharding: `test_reshard_2_to_4`, `test_reshard_4_to_2`,
  `test_reshard_2d_tensor`, `test_reshard_dim1`,
  `test_reshard_3_to_2_uneven`, `test_reshard_same_world_size`,
  `test_reshard_multiple_tensors`.
- Shard-size math: `test_compute_shard_sizes_even`,
  `test_compute_shard_sizes_uneven`.
- Concat/slice: `test_concat_1d`, `test_concat_2d_dim0`,
  `test_concat_2d_dim1`, `test_slice_1d`, `test_slice_2d_dim0`,
  `test_slice_2d_dim1`.
- Metadata + struct: `test_metadata_roundtrip`,
  `test_flat_shard_metadata`, `test_distributed_checkpoint_struct`.
- Async: `test_async_checkpoint_basic`,
  `test_async_checkpoint_wait_idempotent`,
  `test_async_checkpoint_is_done`.

Lint: `cargo clippy -p ferrotorch-distributed -- -D warnings` clean.
Parity-sweep: no ops; integer grep count is 0 by construction.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum DistCheckpointError`, `impl From<DistCheckpointError> for FerrotorchError`, `impl From<std::io::Error> for DistCheckpointError` in `ferrotorch-distributed/src/checkpoint.rs`; non-test consumer: `ferrotorch-distributed/src/lib.rs` `pub use checkpoint::{...DistCheckpointError...};` → `ferrotorch/src/lib.rs`. |
| REQ-2 | SHIPPED | impl: `pub struct TensorShardSpec` in `ferrotorch-distributed/src/checkpoint.rs`; non-test consumer: `lib.rs` re-exports `TensorShardSpec`; used as a field of `ShardMetadata::tensor_specs`. |
| REQ-3 | SHIPPED | impl: `pub struct ShardMetadata` in `ferrotorch-distributed/src/checkpoint.rs`; non-test consumer: `lib.rs` re-exports `ShardMetadata`; consumed by `save_distributed`, `load_distributed`, `reshard`, `flat_shard_metadata`, and the `AsyncCheckpointer`. |
| REQ-4 | SHIPPED | impl: `pub struct DistributedCheckpoint` in `ferrotorch-distributed/src/checkpoint.rs`; non-test consumer: `lib.rs` re-exports `DistributedCheckpoint`. |
| REQ-5 | SHIPPED | impl: `pub fn save_distributed<T>` in `ferrotorch-distributed/src/checkpoint.rs`; non-test consumer: `lib.rs` re-export of `save_distributed`; production consumer of `save_tensors_to_file` (same file). |
| REQ-6 | SHIPPED | impl: `pub fn load_distributed<T>` in `ferrotorch-distributed/src/checkpoint.rs`; non-test consumer: `lib.rs` re-export of `load_distributed`; production consumer of `reshard` (same file). |
| REQ-7 | SHIPPED | impl: `pub fn reshard<T>` in `ferrotorch-distributed/src/checkpoint.rs`; non-test consumer: `lib.rs` re-export of `reshard`; invoked by `pub fn load_distributed` (same file) on world-size mismatch. |
| REQ-8 | SHIPPED | impl: `fn st_dtype`, `fn as_le_bytes`, `fn save_tensors_to_file`, `fn load_tensors_from_file` in `ferrotorch-distributed/src/checkpoint.rs`; non-test consumer: invoked by `pub fn save_distributed`, `pub fn reshard`, `AsyncCheckpointer::save_async` (all same file, all reachable via `lib.rs`). |
| REQ-9 | SHIPPED | impl: `fn concat_along_dim` in `ferrotorch-distributed/src/checkpoint.rs`; non-test consumer: invoked by `pub fn reshard` (same file). |
| REQ-10 | SHIPPED | impl: `fn slice_along_dim` in `ferrotorch-distributed/src/checkpoint.rs`; non-test consumer: invoked by `pub fn reshard` (same file). |
| REQ-11 | SHIPPED | impl: `pub struct CheckpointFuture`, `pub struct AsyncCheckpointer`, `pub fn save_async`, `pub fn wait`, `pub fn is_done` in `ferrotorch-distributed/src/checkpoint.rs`; non-test consumer: `lib.rs` re-exports `AsyncCheckpointer` and `CheckpointFuture`. |
| REQ-12 | SHIPPED | impl: `pub fn flat_shard_metadata` in `ferrotorch-distributed/src/checkpoint.rs`; non-test consumer: `lib.rs` re-export of `flat_shard_metadata`. |
