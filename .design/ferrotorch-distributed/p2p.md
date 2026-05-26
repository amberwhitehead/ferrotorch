# P2P — tensor-level send / recv / sendrecv

<!--
tier: 3-component
status: draft
baseline-pytorch: main (user's local clone /home/doll/pytorch)
upstream-paths:
  - torch/distributed/distributed_c10d.py
-->

## Summary

`ferrotorch-distributed/src/p2p.rs` wraps `Backend::send` /
`Backend::recv` / `Backend::recv_timeout` so the shape, dtype, and
float semantics stay typed at the call site. Mirrors
`torch.distributed.send` / `torch.distributed.recv` /
`torch.distributed.batch_isend_irecv` (two-party case). The byte
serializer / deserializer pair (`floats_to_bytes` / `bytes_to_floats`)
is borrowed from `crate::collective` so the wire format stays
identical to the collective layer's.

## Requirements

- REQ-1: `pub fn send<T: Float>(tensor, dst_rank, backend) ->
  FerrotorchResult<()>` validating `dst_rank < world_size` and
  `dst_rank != backend.rank()`, then forwarding the tensor's data as
  raw bytes via `Backend::send`.
- REQ-2: `pub fn recv<T: Float>(shape, src_rank, backend) ->
  FerrotorchResult<Tensor<T>>` allocating a CPU tensor of the
  caller-supplied shape and filling it via `Backend::recv_timeout`
  with `DEFAULT_COLLECTIVE_TIMEOUT`.
- REQ-3: `pub fn recv_with_timeout` variant accepting an explicit
  `Duration`. The default `recv` is implemented as a thin wrapper
  delegating to `recv_with_timeout`.
- REQ-4: `pub fn recv_into<T: Float>(out: &mut Tensor<T>, src_rank,
  backend) -> FerrotorchResult<()>` overwriting a caller-owned
  tensor in place (rebuilds the tensor from the new storage so the
  output preserves the input's shape). Used when the receive buffer
  is pre-allocated (ring-buffer reuse).
- REQ-5: `pub fn recv_into_with_timeout` variant of REQ-4 with
  explicit `Duration`.
- REQ-6: `pub fn sendrecv<T: Float>(send_tensor, recv_shape, peer,
  backend) -> FerrotorchResult<Tensor<T>>` atomic two-party
  exchange — lower rank sends first then receives, higher rank
  receives first then sends. Mirrors PyTorch's
  `batch_isend_irecv` for the simple two-party case.

## Acceptance Criteria

- [x] AC-1: `send::<f32>(&t, 0, &solo_backend)` (rank == dst) returns
  `Err(InvalidArgument)` ("send: dst_rank equals self rank...").
- [x] AC-2: `recv::<f32>(&[1], 5, &backend)` with `world_size = 2`
  returns `Err(InvalidArgument)` ("recv: src_rank 5 >= world_size 2").
- [x] AC-3: `recv_into` with a 3-element input tensor receives 3
  floats from the peer and updates the tensor in place.
- [x] AC-4: `sendrecv` between two threads swaps the tensors (rank 0
  gets rank 1's payload and vice versa).

## Architecture

### Wire-format reuse

`p2p.rs` imports `floats_to_bytes` and `bytes_to_floats` from
`crate::collective` (the `pub(crate)` byte helpers).
This guarantees a tensor sent via `p2p::send` can be received via
`collective::all_gather`'s recv-side machinery and vice versa — one
wire format per crate.

### `send` (REQ-1)

`pub fn send` validates:

- `dst_rank < backend.world_size()` → `FerrotorchError::InvalidArgument`.
- `dst_rank != backend.rank()` → `InvalidArgument` with a hint
  ("use a tensor copy instead").

Then it calls `tensor.data_vec()?` to materialize the storage as a
`Vec<T>`, byte-serializes with `floats_to_bytes`, and forwards to
`backend.send(&bytes, dst_rank)`. The `data_vec` call copies the
storage to host memory if it lives on the GPU, mirroring PyTorch's
implicit `.cpu()` on `dist.send`.

### `recv` family (REQ-2 / REQ-3)

`pub fn recv` is a thin wrapper around `recv_with_timeout` passing
`DEFAULT_COLLECTIVE_TIMEOUT`. `pub fn recv_with_timeout` validates
`src_rank` against world-size and self-rank, computes the expected
byte length from `shape.iter().product().max(1) * size_of::<T>()`,
allocates a zero-filled `Vec<u8>`, calls `backend.recv_timeout(...)`,
deserializes via `bytes_to_floats`, and wraps the result in
`Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(),
requires_grad: false)`.

The `requires_grad: false` argument matches PyTorch's `dist.recv`
returning a leaf tensor without autograd history.

### `recv_into` family (REQ-4 / REQ-5)

`pub fn recv_into_with_timeout` overwrites `out` in place by:

1. Computing `byte_len = out.numel() * size_of::<T>()`.
2. Receiving into a fresh `Vec<u8>`.
3. Snapshotting `out.shape().to_vec()`.
4. Rebuilding the tensor: `*out = Tensor::from_storage(...)`.

This rebuilds the tensor rather than mutating the storage in place
because `Tensor` doesn't expose a `&mut [T]` accessor without a
storage borrow that would conflict with autograd-side cloning of the
storage handle. The shape is preserved.

### `sendrecv` (REQ-6)

`pub fn sendrecv` implements deadlock-free two-party exchange by
ordering on rank:

- If `rank < peer`: `send` first, then `recv`.
- If `rank > peer`: `recv` first, then `send`.

This is the textbook approach for avoiding the deadlock that arises
if both peers block on `send` (or both on `recv`) simultaneously. The
shapes don't have to match — asymmetric exchanges are allowed
(`send_tensor.shape() != recv_shape`). Mirrors PyTorch's
`batch_isend_irecv([P2POp(send, ...), P2POp(recv, ...)])` for the
two-party case.

### Consumer sites (production, non-test)

- `ferrotorch-distributed/src/lib.rs` — `pub use p2p::{recv,
  recv_into, recv_into_with_timeout, recv_with_timeout, send,
  sendrecv};` re-exports every function.
- `ferrotorch/src/lib.rs` — meta-crate `pub use
  ferrotorch_distributed::*;` exposes the P2P surface to user code.

No in-tree non-test consumer outside the unit tests in the same file
— the P2P functions are the boundary API for user training-script
code (`pipeline.rs` uses the lower-level `Backend::send`/`recv`
directly because it owns the byte serialization). R-DEFER-1
grandfathers existing pub API surface as the public API.

## Parity contract

No parity-sweep ops in the route (`parity_ops = []`). The contract is
the PyTorch P2P shape:

- `dist.send(tensor, dst, group=None, tag=0)` (cf.
  `torch/distributed/distributed_c10d.py:2705`): matches
  ferrotorch's `send(tensor, dst_rank, backend)`. R-DEV-4: ferrotorch
  drops the `tag` (no MPI-style tags on the TCP/Simulated backend),
  R-DEV-7: ferrotorch passes the backend explicitly rather than
  threading a global process-group state.
- `dist.recv(tensor, src=None, group=None, tag=0)` (cf.
  `torch/distributed/distributed_c10d.py:2749`): matches
  ferrotorch's `recv_into(out, src_rank, backend)`. PyTorch's `recv`
  fills the caller's tensor and returns the source rank; ferrotorch
  splits this into `recv` (allocates) and `recv_into` (fills) for
  Rust-side ownership clarity. R-DEV-4 / R-DEV-7 deviations.
- `dist.batch_isend_irecv([P2POp(send,...), P2POp(recv,...)])`
  matches ferrotorch's `sendrecv` for the two-party case. The
  general N-way case is not yet implemented; PyTorch users emulate
  with explicit `all_to_all` calls.

## Verification

- `cargo test -p ferrotorch-distributed --lib` runs the
  `#[cfg(test)] mod tests` at lines 151-243 covering:
  - `send_recv_roundtrip_floats`,
    `recv_into_overwrites_in_place`,
    `send_rejects_self_rank`,
    `recv_rejects_oob_rank`,
    `sendrecv_swaps_two_peers`.
- Conformance fixtures: `ferrotorch-distributed/tests/conformance/fixtures.json`
  pins `sendrecv_round_trip`.
- Lint: `cargo clippy -p ferrotorch-distributed -- -D warnings` PASS.
- Parity-sweep: no ops; integer grep count is 0 by construction.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn send` in `ferrotorch-distributed/src/p2p.rs`; non-test consumer: `ferrotorch-distributed/src/p2p.rs` (`sendrecv` calls `send::<T>(...)` within the same module — production use of the same pub fn) and re-export at `ferrotorch-distributed/src/lib.rs`. |
| REQ-2 | SHIPPED | impl: `pub fn recv` in `ferrotorch-distributed/src/p2p.rs` delegating to `recv_with_timeout`; non-test consumer: `ferrotorch-distributed/src/p2p.rs` (`sendrecv` calls `recv::<T>(...)` within the same module — production use). |
| REQ-3 | SHIPPED | impl: `pub fn recv_with_timeout` in `ferrotorch-distributed/src/p2p.rs`; non-test consumer: `ferrotorch-distributed/src/p2p.rs` (`recv` calls `recv_with_timeout` internally), and re-export at `ferrotorch-distributed/src/lib.rs`. |
| REQ-4 | SHIPPED | impl: `pub fn recv_into` in `ferrotorch-distributed/src/p2p.rs` delegating to `recv_into_with_timeout`; non-test consumer: re-export at `ferrotorch-distributed/src/lib.rs`, reached through `ferrotorch/src/lib.rs`. |
| REQ-5 | SHIPPED | impl: `pub fn recv_into_with_timeout` in `ferrotorch-distributed/src/p2p.rs`; non-test consumer: `ferrotorch-distributed/src/p2p.rs` (`recv_into` calls `recv_into_with_timeout` internally — production use). |
| REQ-6 | SHIPPED | impl: `pub fn sendrecv` in `ferrotorch-distributed/src/p2p.rs`; non-test consumer: re-export at `ferrotorch-distributed/src/lib.rs` reached through `ferrotorch/src/lib.rs` meta-crate path. |
