# Async collective handles (compute / collective overlap)

<!--
tier: 3-component
status: draft
baseline-pytorch: main (user's local clone /home/doll/pytorch)
upstream-paths:
  - torch/distributed/distributed_c10d.py
-->

## Summary

`ferrotorch-distributed/src/async_collective.rs` wraps the
synchronous `all_gather` and `reduce_scatter` primitives in
background threads and exposes a `PendingCollective<T>` handle that
the caller `.wait()`s on after local compute. This enables FSDP-style
backward prefetch (all-gather the next layer's parameters in
parallel with the current layer's backward) and gradient
reduce-scatter during backward. Mirrors PyTorch's `async_op=True`
shape on `dist.all_gather` / `dist.reduce_scatter` (the returned
`Work` handle's `.wait()` method).

## Requirements

- REQ-1: `pub struct PendingCollective<T: Float>` carrying
  `recv: Option<Receiver<FerrotorchResult<Tensor<T>>>>`,
  `handle: Option<JoinHandle<()>>`, and an `op_name: &'static str`
  used in error messages.
- REQ-2: `pub fn PendingCollective::wait(self) ->
  FerrotorchResult<Tensor<T>>` — consumes the handle, blocks on the
  `mpsc::Receiver`, joins the background thread, and propagates any
  panic as `FerrotorchError::InvalidArgument`. Calling `.wait()` on
  an already-consumed handle (e.g., manual take of the receiver)
  returns `Err`.
- REQ-3: `pub fn op_name(&self) -> &'static str` accessor for
  diagnostics.
- REQ-4: `pub fn async_all_gather<T: Float + 'static>(tensor:
  Tensor<T>, backend: Arc<dyn Backend>) -> PendingCollective<T>`
  spawns a background thread that calls
  `crate::collective::all_gather` on the provided tensor and signals
  completion through the `mpsc::channel`.
- REQ-5: `pub fn async_reduce_scatter<T: Float + 'static>(tensor:
  Tensor<T>, backend: Arc<dyn Backend>, op: ReduceOp) ->
  PendingCollective<T>` symmetric to REQ-4 calling
  `crate::collective::reduce_scatter`.
- REQ-6: Drop-without-wait semantics: dropping a `PendingCollective`
  without `.wait()` detaches the background thread; the collective
  still runs to completion on the backend, the result is discarded.
  Documented in the struct docstring.
- REQ-7: Single-outstanding-op invariant: callers must not initiate a
  second async collective on the same backend until the prior
  handle has been `.wait()`ed on. The Backend's send/recv channels
  are untagged, so interleaving would corrupt the message stream.
  Documented in the module docstring; not enforced at the type
  level (would require a typestate on `Backend`).

## Acceptance Criteria

- [x] AC-1: `async_all_gather(t, backend).wait()` returns the same
  result as `all_gather(&t, &backend)` for a 2-rank in-process
  simulated test.
- [x] AC-2: `async_reduce_scatter(t, backend, ReduceOp::Mean).wait()`
  matches `reduce_scatter(&t, &backend, ReduceOp::Mean)`.
- [x] AC-3: `async_all_gather` with `world_size == 1` returns the
  input unchanged after `.wait()`.

## Architecture

### `PendingCollective<T>` shape (REQ-1 / REQ-2 / REQ-3)

`pub struct PendingCollective<T: Float>` carries:

- `recv: Option<Receiver<FerrotorchResult<Tensor<T>>>>` — single-
  message `mpsc` channel from the background thread.
- `handle: Option<JoinHandle<()>>` — the spawned thread's handle.
- `op_name: &'static str` — stored at spawn time for diagnostic
  messages.

`wait(mut self)` consumes `self.recv` and `self.handle` via
`Option::take`, blocks on the receiver, joins the background thread,
and converts any thread panic into `FerrotorchError::InvalidArgument`.
Calling `.wait()` after the receiver has already been taken (e.g.,
in the unlikely concurrent-access edge case) returns
`InvalidArgument` ("wait() called on already-consumed handle").

`op_name()` returns the stored static string for callers that want to
log the in-flight collective without consuming the handle.

### Background-thread spawn (REQ-4 / REQ-5)

`async_all_gather`:

```text
let (tx, rx) = mpsc::channel();
let handle = thread::spawn(move || {
    let result = all_gather(&tensor, backend.as_ref());
    let _ = tx.send(result);
});
PendingCollective { recv: Some(rx), handle: Some(handle),
                    op_name: "async_all_gather" }
```

The owned `tensor` is moved into the background thread. The
`backend: Arc<dyn Backend>` is cloned across the boundary (via the
move) so the parent thread doesn't have to outlive the child. The
`let _ = tx.send(...)` swallows `SendError` when the parent has
dropped the receiver — that's the documented fire-and-forget path
(REQ-6).

`async_reduce_scatter` is structurally identical with `all_gather`
replaced by `reduce_scatter` and an additional `op: ReduceOp`
parameter folded into the closure.

### Drop semantics (REQ-6)

If the caller drops a `PendingCollective` without `.wait()`:

- The `Receiver` is dropped → the background thread's `tx.send(...)`
  becomes `SendError`, which is swallowed by `let _ = ...`.
- The `JoinHandle` is dropped → the thread becomes detached.
- The collective still runs to completion on the backend (the send/
  recv pairs the synchronous collective issued still complete from
  the other ranks' point of view).
- The result is discarded.

This is the right shape for a true "fire and forget" (e.g., a
broadcast where the local rank doesn't need the broadcast result).
It's the WRONG shape for the common FSDP prefetch case — that case
must `.wait()`.

### Single-outstanding-op invariant (REQ-7)

The Backend's send/recv channels are untagged. If a caller spawns
two `async_all_gather`s on the same backend, the second's
recv-from-rank-0 will arbitrarily interleave with the first's
recv-from-rank-0 and the byte stream gets shredded. The module
docstring at lines 17-24 documents this; the FSDP module respects it
by tracking `pending_prefetch: Option<Vec<PendingCollective<T>>>`
explicitly (drains the previous prefetch before queueing a new one).

Enforcing the invariant at the type level would require a
typestate-bearing `Backend` where the `&dyn Backend` becomes
exclusive while an async collective is outstanding — a non-local
refactor. R-DEV-5 would apply if we wanted that guarantee. For now
the convention is a doc contract; the FSDP prefetch hook is the only
in-tree caller and it respects it.

### Consumer sites (production, non-test)

- `ferrotorch-distributed/src/fsdp.rs` — `use
  crate::async_collective::{PendingCollective, async_all_gather};`
- `ferrotorch-distributed/src/fsdp.rs` — stores `pending_prefetch:
  Option<Vec<PendingCollective<T>>>` field on the FSDP wrapper.
- `ferrotorch-distributed/src/fsdp.rs` — calls
  `async_all_gather(shard, Arc::clone(&self.backend))` in the
  prefetch path.
- `ferrotorch-distributed/src/lib.rs` — re-exports
  `PendingCollective`, `async_all_gather`, `async_reduce_scatter`.
- `ferrotorch/src/lib.rs` — meta-crate `pub use
  ferrotorch_distributed::*;` exposes the async surface.

## Parity contract

No parity-sweep ops in the route (`parity_ops = []`). The contract is
the PyTorch `async_op=True` shape:

- `dist.all_gather(tensor_list, tensor, group, async_op=True)`
  returns a `Work` object whose `.wait()` blocks for the collective.
  ferrotorch's `async_all_gather(t, b).wait()` is the equivalent
  shape. The deviation: PyTorch fills a caller-allocated tensor
  list; ferrotorch returns a freshly-allocated `Tensor<T>` whose
  dim-0 is `world_size * input_dim0` (matching the synchronous
  `all_gather` shape).
- `dist.reduce_scatter(output, input_list, op, group, async_op=True)`
  similarly returns a `Work`; ferrotorch returns a fresh tensor of
  size `numel / world_size`.

R-DEV-4 / R-DEV-7 deviations: explicit `Arc<dyn Backend>` ownership
across the thread boundary, freshly-allocated outputs rather than
caller-mutated buffers (the borrow checker would complain about the
buffer outliving the spawn).

## Verification

- `cargo test -p ferrotorch-distributed --lib` runs the
  `#[cfg(test)] mod tests` at lines 135-234 covering:
  - `test_async_all_gather_matches_sync` (2-rank in-process simulated
    backend, async result matches sync).
  - `test_async_reduce_scatter_matches_sync` (2-rank, Mean
    reduction).
  - `test_async_all_gather_world_size_1` (degenerate case).
- Lint: `cargo clippy -p ferrotorch-distributed -- -D warnings` PASS.
- Parity-sweep: no ops; integer grep count is 0 by construction.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct PendingCollective<T: Float>` in `ferrotorch-distributed/src/async_collective.rs`; non-test consumer: `ferrotorch-distributed/src/fsdp.rs` stores `pending_prefetch: Option<Vec<PendingCollective<T>>>` field on FSDP. |
| REQ-2 | SHIPPED | impl: `pub fn wait` in `ferrotorch-distributed/src/async_collective.rs` consuming `self.recv` / `self.handle` with thread-panic propagation at lines 68-75; non-test consumer: re-exported via `PendingCollective` at `ferrotorch-distributed/src/lib.rs` — FSDP's prefetch drain path is the production call site (boundary API reached through `crate::async_collective`). |
| REQ-3 | SHIPPED | impl: `pub fn op_name` in `ferrotorch-distributed/src/async_collective.rs`; non-test consumer: re-exported via `PendingCollective` at `ferrotorch-distributed/src/lib.rs`. |
| REQ-4 | SHIPPED | impl: `pub fn async_all_gather` in `ferrotorch-distributed/src/async_collective.rs`; non-test consumer: `ferrotorch-distributed/src/fsdp.rs` invokes `async_all_gather(shard, Arc::clone(&self.backend))` in the forward-prefetch path. |
| REQ-5 | SHIPPED | impl: `pub fn async_reduce_scatter` in `ferrotorch-distributed/src/async_collective.rs`; non-test consumer: re-exported at `ferrotorch-distributed/src/lib.rs`, reached through `ferrotorch/src/lib.rs` meta-crate path (boundary API for FSDP-style gradient reduce-scatter during backward). |
| REQ-6 | SHIPPED | impl: `let _ = tx.send(result);` in `ferrotorch-distributed/src/async_collective.rs` (and `:126`) is the silent-on-drop semantics; struct doc at `ferrotorch-distributed/src/async_collective.rs` documents the contract; non-test consumer: re-export via `PendingCollective` at `ferrotorch-distributed/src/lib.rs`. |
| REQ-7 | SHIPPED | impl: module docstring at `ferrotorch-distributed/src/async_collective.rs` documents the single-outstanding-op invariant; non-test consumer: `ferrotorch-distributed/src/fsdp.rs` (the in-tree caller) tracks `pending_prefetch: Option<Vec<PendingCollective<T>>>` so it drains the previous prefetch before queueing a new one — respects the contract. |
