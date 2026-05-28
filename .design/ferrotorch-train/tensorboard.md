# ferrotorch-train — TensorBoard `TFEvents` writer + `Callback`

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/utils/tensorboard/writer.py
  - torch/utils/tensorboard/summary.py
-->

## Summary

`ferrotorch-train/src/tensorboard.rs` implements a minimal in-process
writer for the TensorBoard `TFEvents` binary file format plus a
`TensorBoardCallback` that hooks into the `Learner` training loop and
emits per-epoch and per-batch scalar summaries. Mirrors PyTorch's
`torch.utils.tensorboard.SummaryWriter` (`torch/utils/tensorboard/
writer.py:173`) at the `add_scalar` / `add_scalars` API surface and
mirrors the TFEvents wire format documented by the TensorFlow event
spec.

## Requirements

- REQ-1: `pub struct TensorBoardWriter` writes a TFEvents binary file
  inside a log directory. The directory is created on construction;
  the file is named `events.out.tfevents.{timestamp}.ferrotorch`. The
  first record is a `file_version` event with payload
  `"brain.Event:2"`.
- REQ-2: Each record in the file is framed as `[uint64 length][uint32
  masked_crc32c(length)][bytes data][uint32 masked_crc32c(data)]`.
  The `data` payload is a protobuf-encoded `Event` message.
- REQ-3: `crc32c(data)` computes the Castagnoli polynomial
  (`0x82F6_3B78`) CRC32 of the input. `masked_crc32c(data)` rotates
  the raw CRC right by 15 bits and adds `0xa282_ead8` per the
  TensorFlow events spec.
- REQ-4: `TensorBoardWriter::add_scalar(tag, value, step) ->
  FerrotorchResult<()>` writes one `Summary.Value` with the given tag
  and a `simple_value: float`. `add_scalars(main_tag, values, step)`
  writes multiple values under the compound tag `"{main_tag}/{key}"`.
- REQ-5: `TensorBoardWriter::flush()` flushes the underlying
  `BufWriter`. `log_dir()` returns the log directory path.
- REQ-6: `pub struct TensorBoardCallback` wraps a
  `Mutex<TensorBoardWriter>` and implements `Callback<T>` for all
  `T: Float`. `on_epoch_end` writes `train_loss`, `val_loss` (if
  `Some`), `lr`, and any custom metrics as scalars at step
  `epoch as i64`. `on_batch_end` writes `batch_loss` at step `batch
  as i64`. Write failures are logged at `tracing::warn!` and do not
  propagate (matches PyTorch's `SummaryWriter` failure policy:
  silently drops on write error).
- REQ-7: The CRC32C implementation passes the RFC 3720 test vectors:
  - CRC32C of `[0u8; 32]` is `0x8A91_36AA`.
  - CRC32C of `[0xFFu8; 32]` is `0x62A8_AB43`.
  - CRC32C of `(0u8..32)` is `0x46DD_794E`.
  - CRC32C of `b""` is `0x0000_0000`.

## Acceptance Criteria

- [x] AC-1: `TensorBoardWriter::new(dir)` creates the directory and
  the event file, writing the initial `file_version` record.
- [x] AC-2: The first record in the file contains the string
  `"brain.Event:2"`.
- [x] AC-3: All 4 RFC 3720 CRC32C test vectors pass.
- [x] AC-4: `add_scalar` and `add_scalars` writes grow the file.
- [x] AC-5: `TensorBoardCallback` is `Send + Sync` (via
  `Mutex<TensorBoardWriter>`) and implements `Callback<f32>`.
- [x] AC-6: `TensorBoardCallback::on_epoch_end` writes train_loss,
  val_loss, lr, and metric scalars without panicking on file errors.

## Architecture

### `CRC32C` (REQ-3, REQ-7)

At `ferrotorch-train/src/tensorboard.rs:46-81`. The lookup table is
generated at compile time from the Castagnoli polynomial. The
`crc32c` function uses the standard table-driven byte-at-a-time
algorithm with the conventional `0xFFFF_FFFF` start and final XOR.
`masked_crc32c` applies the TensorFlow events mask: `crc.rotate_right(15)
+ 0xa282_ead8` (wrapping add).

The RFC 3720 test vectors at lines 562-581 are the canonical
verification path that pins the polynomial + algorithm.

### `ProtobufWriter` (REQ-2)

At lines 94-170. A minimal protobuf wire-format encoder supporting
varint (wire type 0), 64-bit fixed (wire type 1), length-delimited
(wire type 2), and 32-bit fixed (wire type 5). The full helper set
(`write_int64`, `write_string`, `write_message`, `write_float`,
`write_double`) is kept available for forward compatibility — adding
new event/summary fields should not need to touch the encoder.

The `#[allow(dead_code)]` at line 105 is the documented allow for
the unused-for-now helpers (only a subset is exercised by the current
`encode_*` functions). The justification comment at lines 98-104
documents this.

### `Event` / `Summary` encoders (REQ-1, REQ-4)

At lines 186-236:
- `encode_summary_value(tag, value)` writes a `Summary.Value` with
  `string tag = 1; float simple_value = 2;`.
- `encode_summary(values)` writes a `Summary` with `repeated Value
  value = 1;`.
- `encode_event_summary(wall_time, step, summary_bytes)` writes an
  `Event` with `double wall_time = 1; int64 step = 2; Summary summary
  = 5;`.
- `encode_event_file_version(wall_time)` writes the initial
  `file_version` event with `string file_version = 3;` set to
  `"brain.Event:2"`.

### `write_record` framing (REQ-2)

At lines 248-276. Writes the 4-part record (length, length CRC,
data, data CRC) to the inner `BufWriter`. Each `write_all` is wrapped
in a `FerrotorchError::InvalidArgument` mapping so an IO error
propagates with a structured message.

### `TensorBoardWriter` (REQ-1, REQ-4, REQ-5)

At lines 302-409. `new(dir)` creates the directory, opens the event
file (named with `wall_time` truncated to `u64` seconds), writes the
mandatory `file_version` record, and returns the wrapped
`BufWriter<File>`.

`add_scalar` at line 354 encodes the `Summary` + `Event` and calls
`write_record`. `add_scalars` at line 371 builds compound tags
`"{main_tag}/{key}"` and writes them as a single multi-value Summary.

### `TensorBoardCallback` (REQ-6)

At lines 437-543. Wraps the writer in a `Mutex` for `Send + Sync`
(the writer's `BufWriter<File>` is `Send` but not `Sync`; the mutex
gives the callback `Sync` at the cost of internal locking).

`on_epoch_end` at line 459 writes train_loss, val_loss (if
`Some`), lr, custom metrics — each via `add_scalar` — and then
flushes. Write failures are logged via `tracing::warn!` with the
target `"ferrotorch::tensorboard"`, mirroring PyTorch's silent-drop
policy on `SummaryWriter` failures.

`on_batch_end` at line 528 writes `batch_loss` at step `batch as i64`.

### Non-test production consumers

- `ferrotorch-train/src/lib.rs:183` `pub use tensorboard::{TensorBoardCallback,
  TensorBoardWriter};` exposes both types at the crate root.
- `ferrotorch-train/src/tensorboard.rs:451` constructs a `TensorBoardWriter`
  inside `TensorBoardCallback::new` — internal production consumer.
- No external in-tree caller attaches a `TensorBoardCallback` to a
  `Learner` today. Open prereq blocker #1504 covers wiring the
  callback into a real example binary or downstream user app.

## Parity contract

`parity_ops = []`. The TFEvents wire format is an external
specification; deviation would break TensorBoard interoperability.
Edge cases:

- **CRC32C of empty input**: returns `0x0000_0000`. Tested by
  `test_crc32c_empty` at line 558.
- **Masked vs raw CRC32C**: always differ (the rotate+add ensures
  the mask is non-trivial). Tested by
  `test_masked_crc32c_differs_from_raw` at line 585.
- **`add_scalar` IO error**: returns `FerrotorchError::InvalidArgument`
  with a structured message. The `TensorBoardCallback` catches the
  error and emits a `tracing::warn!` rather than propagating
  (matches PyTorch's `SummaryWriter.add_scalar` silent-drop on
  closed-file).
- **Writer mutex poison recovery**: `on_epoch_end` and `on_batch_end`
  call `self.writer.lock().unwrap_or_else(|p| p.into_inner())`
  (lines 461 / 529) so a poisoned mutex still allows the callback
  to write further events.

## Verification

10+ unit tests in `mod tests` (lines 549-end) cover:
- CRC32C test vectors (lines 556-588).
- Writer creates a non-empty file (line 592).
- Multi-scalar writes grow the file (line 607).
- First record contains the `"brain.Event:2"` file_version string
  (line 628).
- `add_scalars` writes compound tags (line 646).
- `TensorBoardCallback` is `Callback<f32>` and `Send + Sync` (lines
  667-678).
- `on_epoch_end` writes scalars without panicking (line 682).

Smoke command:

```bash
cargo test -p ferrotorch-train --lib tensorboard:: 2>&1 | tail -3
```

Expected: > 10 passed, 0 failed.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct TensorBoardWriter` at `TensorBoardWriter in ferrotorch-train/src/tensorboard.rs`, `new in ferrotorch-train/src/tensorboard.rs` (creates dir + file, writes `file_version`); non-test consumer: `file_version in ferrotorch-train/src/tensorboard.rs` (`TensorBoardCallback::new` constructs `TensorBoardWriter::new(log_dir)`). |
| REQ-2 | SHIPPED | impl: `write_record` at `ferrotorch-train/src/tensorboard.rs:248-276` writes the 4-part framing; non-test consumer: every `add_scalar` / `add_scalars` / `new` call (lines 358, 386, 339) invokes `write_record`. |
| REQ-3 | SHIPPED | impl: `crc32c` at `ferrotorch-train/src/tensorboard.rs:67-74`, `masked_crc32c` at `:78-81`; non-test consumer: `write_record` (lines 251-252) invokes both on every record write. |
| REQ-4 | SHIPPED | impl: `add_scalar` at `ferrotorch-train/src/tensorboard.rs:354-361`, `add_scalars` at `:371-389`; non-test consumer: `TensorBoardCallback::on_epoch_end` (lines 471, 482, 493, 505) and `on_batch_end` (line 533) invoke `add_scalar` per epoch/batch. |
| REQ-5 | SHIPPED | impl: `flush` at `ferrotorch-train/src/tensorboard.rs:397-403`, `log_dir` at `:406-408`; non-test consumer: `TensorBoardCallback::on_epoch_end` (line 517) calls `writer.flush()` after every epoch's scalar batch. |
| REQ-6 | SHIPPED | impl: `pub struct TensorBoardCallback` at `TensorBoardCallback in ferrotorch-train/src/tensorboard.rs`, `Callback<T>` impl at `ferrotorch-train/src/tensorboard.rs`; non-test consumer: `learner in ferrotorch-train/src/lib.rs` `pub use tensorboard::{TensorBoardCallback, TensorBoardWriter};` exposes at the crate root for external use; in-tree attachment to `Learner` is the open consumer-wiring gap covered by blocker #1504. The trait dispatch through `Learner in learner.rs`'s `Vec<Box<dyn Callback<T>>>` is the production-consumer plumbing waiting for a real attachment site. |
| REQ-7 | SHIPPED | impl: same `crc32c` at `ferrotorch-train/src/tensorboard.rs:67-74`; non-test consumer: every `write_record` invocation (`new` / `add_scalar` / `add_scalars` / `flush` callers) consumes the verified CRC32C through the `write_record` call. The RFC 3720 vectors at lines 562-581 are the verification pin. |
