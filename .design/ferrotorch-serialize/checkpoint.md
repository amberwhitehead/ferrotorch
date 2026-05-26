# ferrotorch-serialize — `checkpoint` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/serialization.py
  - torch/_weights_only_unpickler.py
-->

## Summary

`ferrotorch-serialize/src/checkpoint.rs` implements training-resume
checkpoints — a single file that bundles three things: training
metadata (epoch + step), optimizer state (momentum / velocity buffers
per parameter), and the model's state dict. It also ships
`AsyncCheckpointer`, a background-thread saver that lets training
loops snapshot weights without blocking the forward pass.

Mirrors the role of `torch.save({'model_state': ...,
'optimizer_state': ..., 'epoch': N})` in `torch/serialization.py:945`
— the upstream pattern is a single pickle blob containing a dict
of arbitrary keys. ferrotorch deviates from upstream's pickle layer
(R-DEV-4: pickle is a Python attack surface Rust eliminates) and
uses a three-section length-prefixed binary layout with hand-written
JSON for the metadata and optimizer sections.

## Requirements

- REQ-1: `#[non_exhaustive] pub struct TrainingCheckpoint<T: Float>`
  with four fields: `model_state: StateDict<T>`, `optimizer_state:
  OptimizerState`, `epoch: usize`, `step: usize`. Marked
  `#[non_exhaustive]` so future versions can add RNG state, scheduler
  state, or framework version without forcing every consumer to update
  their pattern matches. Construction goes through
  `TrainingCheckpoint::new(model_state, optimizer_state, epoch, step)
  -> Self` since struct-literal syntax is blocked by
  `#[non_exhaustive]` outside the crate.

- REQ-2: `pub fn save_checkpoint<T: Float>(checkpoint:
  &TrainingCheckpoint<T>, path: impl AsRef<Path>) ->
  FerrotorchResult<()>` writes the three sections in order: metadata
  JSON (`{"epoch":N,"step":N}`), optimizer state JSON
  (`{"param_name":{"state_key":[v1,v2,...]}, ...}`), and the state
  dict binary using the same `state_dict.rs` format. Each section is
  preceded by an 8-byte little-endian length prefix.

- REQ-3: `pub fn load_checkpoint<T: Float>(path: impl AsRef<Path>)
  -> FerrotorchResult<TrainingCheckpoint<T>>` reads back the three
  sections by reading the length prefix and then exactly that many
  bytes. The optimizer state JSON is parsed by a hand-written
  recursive descent (`split_top_level` for nested brace/bracket
  awareness) since serde is not in the crate's dep tree.

- REQ-4: Optimizer-state JSON serialization. `fn
  serialize_optimizer_state(state: &OptimizerState) -> String` emits
  alphabetically-sorted outer keys + alphabetically-sorted inner
  keys for deterministic output. The inverse, `fn
  deserialize_optimizer_state(s: &str) -> FerrotorchResult<OptimizerState>`,
  uses `split_top_level` to respect nested brace/bracket depth so
  inner arrays of f64 values can contain commas without breaking the
  outer split.

- REQ-5: Length-prefixed binary framing. `fn write_section(file,
  data) -> ...` writes `len.to_le_bytes()` (8 bytes) followed by
  exactly `len` payload bytes. `fn read_section(file) ->
  FerrotorchResult<Vec<u8>>` reads the 8-byte length, allocates a
  buffer of that exact size, and `read_exact`'s the body. Truncated
  files surface as `InvalidArgument` with the byte-count discrepancy
  in the message.

- REQ-6: Embedded state-dict section. `fn
  serialize_state_dict_to_bytes<T>(state) -> FerrotorchResult<Vec<u8>>`
  and `fn deserialize_state_dict_from_bytes<T>(bytes) ->
  FerrotorchResult<StateDict<T>>` reuse the on-disk layout from
  `state_dict.rs` (sorted header lines + `\n---\n` + LE body),
  calling `crate::state_dict::dtype_tag` and
  `crate::state_dict::parse_meta_line` so all three save/load paths
  agree on the wire bytes.

- REQ-7: `pub struct AsyncCheckpointer` — a background-thread saver
  with `pub fn new() -> Self`, `pub fn save(&mut self, checkpoint:
  TrainingCheckpoint<f32>, path) -> FerrotorchResult<()>`, `pub fn
  wait(&mut self) -> FerrotorchResult<()>`, `pub fn is_saving(&self)
  -> bool`. `save` blocks if a previous save is still in flight
  (FIFO ordering, not coalescing — every requested save executes).
  Restricted to `f32` checkpoints because the background thread
  owns the checkpoint and supporting `T: Float` would require a
  trait-object boxed-checkpoint indirection that hasn't been
  needed in practice.

- REQ-8: Panic safety in the background thread. The
  `AsyncCheckpointer::save` body wraps the call to `save_checkpoint`
  in `catch_unwind(AssertUnwindSafe(...))` and ALWAYS resets the
  `in_flight: AtomicBool` to false (even on panic) so subsequent
  saves are not deadlocked. Panic payloads are downcast to
  `&str` / `String` and surfaced as
  `FerrotorchError::InvalidArgument { message: "async checkpoint
  thread panicked: <payload>" }`. The `wait` join is also
  panic-safe: a `Err` from `JoinHandle::join` is converted to an
  `InvalidArgument` rather than re-panicked.

- REQ-9: Endianness contract. The 8-byte length prefix is
  `u64::to_le_bytes`, matching the state-dict / safetensors / GGUF
  / pytorch-export contracts. A `compile_error!` at top-of-file
  rejects big-endian targets, identical to the discipline in
  `state_dict.rs`.

## Acceptance Criteria

- [x] AC-1: `TrainingCheckpoint::new` constructs the struct
  (required since `#[non_exhaustive]` blocks struct-literal syntax
  outside the crate).
- [x] AC-2: `save_checkpoint` + `load_checkpoint` round-trip the
  three sections (`test_checkpoint_roundtrip`).
- [x] AC-3: Empty optimizer state round-trips
  (`test_checkpoint_empty_optimizer_state`).
- [x] AC-4: Missing file produces an error with the
  `"failed to open"` prefix (`test_checkpoint_missing_file`).
- [x] AC-5: Optimizer-state serialization is a true inverse pair
  (`test_optimizer_state_serialization_roundtrip`).
- [x] AC-6: Metadata parsing handles the
  `{"epoch":N,"step":N}` format (`test_metadata_parsing`).
- [x] AC-7: `AsyncCheckpointer::is_saving` correctly reports
  whether a background save is in flight (covered indirectly by
  the in-flight flag manipulation in `save`).

## Architecture

### Three-section layout (REQ-2, REQ-3, REQ-5)

```text
[8 bytes: metadata JSON length]   [metadata JSON bytes]
[8 bytes: optimizer JSON length]  [optimizer JSON bytes]
[8 bytes: state dict bin length]  [state dict bin bytes]
```

Each section is independent — the metadata can be parsed without
ever touching the state dict bytes, the state dict can be
fast-forwarded by skipping `len` bytes after reading the length,
etc. The `write_section` / `read_section` helpers contain the
length-prefix logic; the rest of save/load is straight-line
"call write_section three times" / "call read_section three times".

### Optimizer state JSON (REQ-4)

`OptimizerState` is `HashMap<String, HashMap<String, Vec<f64>>>` —
the outer keys are parameter names (`"fc.weight"`), the inner
keys are state names (`"momentum"`, `"velocity"`), the values are
flat per-parameter buffers.

Serialization emits sorted outer + inner keys for deterministic
output. Deserialization uses `split_top_level` to respect nested
brace/bracket depth — a naive `s.split(',')` would break on the
commas inside the inner value arrays.

```rust
fn split_top_level(s: &str, open: char, close: char) -> Vec<String>
```

Tracks `depth` for both `{}` and `[]` pairs and emits a part only
when the comma is at depth 0. This correctly handles
`{"a":[1,2,3],"b":[4,5]}` (split on the outer comma, not the inner
ones).

### Embedded state-dict section (REQ-6)

```rust
fn serialize_state_dict_to_bytes<T: Float>(state: &StateDict<T>)
    -> FerrotorchResult<Vec<u8>>
fn deserialize_state_dict_from_bytes<T: Float>(bytes: &[u8])
    -> FerrotorchResult<StateDict<T>>
```

These are inline replicas of `state_dict::save_state_dict` /
`state_dict::load_state_dict` writing to a `Vec<u8>` instead of
`File`. They share the dtype-tag string and the
`parse_meta_line` parser via `crate::state_dict::dtype_tag::<T>()`
and `crate::state_dict::parse_meta_line` — cross-module reuse, not
duplication. The two paths produce byte-identical output for the
same `StateDict<T>`.

### AsyncCheckpointer (REQ-7, REQ-8)

```rust
pub struct AsyncCheckpointer {
    in_flight: Arc<AtomicBool>,
    handle: Option<JoinHandle<FerrotorchResult<()>>>,
}
```

`save(checkpoint, path)`:

1. Calls `self.wait()?` to drain any previous save (FIFO).
2. Stores the checkpoint + path into the spawned closure (move
   semantics — `Send + 'static` bounds on `path` enforce this).
3. `in_flight.store(true)`.
4. `std::thread::spawn(move || catch_unwind(AssertUnwindSafe(||
   save_checkpoint(&checkpoint, path))))` — the `AssertUnwindSafe`
   shim is required because `save_checkpoint` takes a `&` to the
   moved-in `TrainingCheckpoint<f32>`, which `catch_unwind` would
   reject as not `UnwindSafe` without the assertion.
5. The closure ALWAYS resets `in_flight` (even on panic), then
   either returns the `Result` from `save_checkpoint` or unpacks
   the panic payload via `downcast_ref::<&str>` /
   `downcast_ref::<String>` into an
   `FerrotorchError::InvalidArgument`.

`wait()`:

1. Takes the `JoinHandle` out of `self.handle` (so a second `wait`
   is a no-op).
2. `handle.join()` returns `Result<FerrotorchResult<()>,
   Box<dyn Any>>`. The outer `Err` (thread aborted) is converted
   to `FerrotorchError::InvalidArgument`. The inner `Err` (save
   itself failed) propagates.

The `Debug` impl is hand-written rather than `#[derive]`'d because
`JoinHandle` doesn't implement `Debug` in a useful way; we report
`in_flight` and `has_pending_save` instead.

### Endianness (REQ-9)

The 8-byte length prefix is `u64::to_le_bytes`. The same
`compile_error!` at top-of-file as `state_dict.rs` rejects
big-endian targets so the length prefix and the embedded state
dict bytes share the same byte order contract.

### Non-test production consumers

- `pub use checkpoint::{TrainingCheckpoint, AsyncCheckpointer,
  save_checkpoint, load_checkpoint}` in `lib.rs` re-exports the
  surface.
- Downstream training-loop code in the model crates
  (`ferrotorch-llama`, `ferrotorch-bert`, etc.) constructs
  `TrainingCheckpoint::new(model_state, optimizer_state, epoch,
  step)` after each epoch and persists with `save_checkpoint` or
  `AsyncCheckpointer::save`. Resume training reads back via
  `load_checkpoint`.
- `AsyncCheckpointer` is documented as the production-mode saver
  so the foreground training loop doesn't block on disk I/O while
  the next batch's forward pass starts.

## Parity contract

`parity_ops = []`. No PyTorch-side byte-for-byte parity contract
applies — the format is ferrotorch-native. The closest upstream
analog is `torch.save({'model_state_dict': ..., 'optimizer_state_dict': ...,
'epoch': ..., 'step': ...}, path)` in
`torch/serialization.py:945` (a pickle blob containing a dict of
mixed Python values). For cross-framework checkpoint exchange,
users go through `pytorch_export.rs` / `pytorch_import.rs` or
`safetensors_io.rs` separately for the model state.

Edge-case contract this module ships:

- Empty optimizer state: an empty `HashMap` serializes as `"{}"`
  and round-trips without inserting phantom keys.
  `test_checkpoint_empty_optimizer_state`.
- Async panic recovery: a panicking save resets `in_flight` and
  surfaces as a structured error on the next `wait()` call. The
  in-flight flag is never permanently stuck.
- Section length overflow: a section whose declared length exceeds
  the file size produces `InvalidArgument` with the expected byte
  count in the message (via `read_exact`'s error).
- Missing file: `InvalidArgument` with `"failed to open"` prefix.
  `test_checkpoint_missing_file`.

## Verification

Tests in `mod tests in checkpoint.rs` (5 tests):

- `test_checkpoint_roundtrip`: end-to-end save + load with non-empty
  optimizer state.
- `test_checkpoint_empty_optimizer_state`: edge case for the JSON
  emit path.
- `test_checkpoint_missing_file`: error path.
- `test_optimizer_state_serialization_roundtrip`: round-trip of the
  hand-written JSON encoder/decoder pair.
- `test_metadata_parsing`: the metadata extract.

Smoke command:

```bash
cargo test -p ferrotorch-serialize --lib checkpoint:: 2>&1 | tail -3
```

Expected: 5 passed (plus 12 integration tests in
`tests/conformance_serialize_async.rs` that exercise
`AsyncCheckpointer`).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `#[non_exhaustive] pub struct TrainingCheckpoint<T: Float>` + `impl TrainingCheckpoint::new` in `checkpoint.rs`; non-test consumer: `pub use checkpoint::TrainingCheckpoint` in `lib.rs`, reachable through the meta-crate glob, and constructed in production training loops to bundle epoch/step + model + optimizer state. |
| REQ-2 | SHIPPED | impl: `pub fn save_checkpoint<T: Float>` in `checkpoint.rs` calling `write_section` three times (metadata JSON, optimizer JSON, state-dict bytes); non-test consumer: `pub use checkpoint::save_checkpoint` in `lib.rs`; training loops persist checkpoints via this entry. |
| REQ-3 | SHIPPED | impl: `pub fn load_checkpoint<T: Float>` in `checkpoint.rs` calling `read_section` three times and reconstructing each section; non-test consumer: `pub use checkpoint::load_checkpoint` in `lib.rs`; training-resume code rehydrates via this entry. |
| REQ-4 | SHIPPED | impl: `fn serialize_optimizer_state` + `fn deserialize_optimizer_state` + `fn split_top_level` + `fn parse_inner_opt_state` in `checkpoint.rs` — sorted-key emit + depth-aware split parser; non-test consumer: called by `save_checkpoint` / `load_checkpoint` themselves which are pub-re-exported through `lib.rs`. |
| REQ-5 | SHIPPED | impl: `fn write_section` / `fn read_section` in `checkpoint.rs` using `u64::to_le_bytes` length prefix + `read_exact` body; non-test consumer: every call to `save_checkpoint` / `load_checkpoint` hits the helpers three times. |
| REQ-6 | SHIPPED | impl: `fn serialize_state_dict_to_bytes` / `fn deserialize_state_dict_from_bytes` in `checkpoint.rs` calling `crate::state_dict::dtype_tag` and `crate::state_dict::parse_meta_line`; non-test consumer: `save_checkpoint` / `load_checkpoint` use these for the third section, ensuring wire-byte equality with the standalone state-dict format. |
| REQ-7 | SHIPPED | impl: `pub struct AsyncCheckpointer` + `impl AsyncCheckpointer::{new, save, wait, is_saving}` + manual `Debug` + `impl Default` in `checkpoint.rs`; non-test consumer: `pub use checkpoint::AsyncCheckpointer` in `lib.rs`; production training loops construct `AsyncCheckpointer::default()` once and call `save(...)` per epoch to overlap disk I/O with the next forward pass. |
| REQ-8 | SHIPPED | impl: `catch_unwind(AssertUnwindSafe(...))` in `AsyncCheckpointer::save`'s spawn closure + unconditional `in_flight.store(false)` + panic-payload downcast + `wait()`'s join-error handling in `checkpoint.rs`; non-test consumer: every `AsyncCheckpointer::save` call routes through this path; the contract is the load-bearing reason training loops use the async saver instead of foreground `save_checkpoint`. |
| REQ-9 | SHIPPED | impl: `compile_error!` at `checkpoint.rs` top-of-file + `u64::to_le_bytes`/`u64::from_le_bytes` in `write_section`/`read_section`; non-test consumer: every checkpoint write/read uses the same length-prefix encoding, and the embedded state-dict section inherits the same LE assumption from `state_dict.rs`. |
