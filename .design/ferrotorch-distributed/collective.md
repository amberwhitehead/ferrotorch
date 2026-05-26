# Collective operations (allreduce, broadcast, all_gather, …)

<!--
tier: 3-component
status: draft
baseline-pytorch: main (user's local clone /home/doll/pytorch)
upstream-paths:
  - torch/distributed/distributed_c10d.py
-->

## Summary

`ferrotorch-distributed/src/collective.rs` implements the collective
communication ops layered on top of the `Backend` trait: `allreduce`,
`broadcast`, `all_gather`, `reduce_scatter`, `reduce_scatter_tensor`,
`all_to_all`, `all_to_all_single_uneven`, `barrier`. Every op uses a
star-topology algorithm (gather at rank 0, reduce / scatter, broadcast
back) which is correct but not bandwidth-optimal — ring-allreduce and
tree-reduce can be layered in later without changing the public API.
Mirrors `torch.distributed.{all_reduce, broadcast, all_gather,
reduce_scatter, reduce_scatter_tensor, all_to_all,
all_to_all_single, barrier}` from
`torch/distributed/distributed_c10d.py`.

## Requirements

- REQ-1: `pub enum ReduceOp { Sum, Mean }` with `Debug + Clone + Copy
  + PartialEq + Eq`. Mirrors `torch.distributed.ReduceOp.SUM` /
  `RedOp.AVG` (the AVG name in c10d). PyTorch also offers PRODUCT /
  MIN / MAX / BAND / BOR / BXOR / PREMUL_SUM; ferrotorch ships the
  two used by DDP / FSDP today.
- REQ-2: `pub const DEFAULT_COLLECTIVE_TIMEOUT: Duration =
  Duration::from_secs(60)` matching c10d's default
  `kProcessGroupDefaultTimeout` of 60 minutes scaled down for unit
  testing latency.
- REQ-3: `pub fn allreduce<T: Float>(tensor, backend, op) ->
  FerrotorchResult<Tensor<T>>` thin wrapper around
  `allreduce_with_timeout`. `allreduce_with_timeout` implements the
  star-topology reduce: non-zero ranks send to rank 0, rank 0 accumulates
  with `+=`, applies division by `world_size` for `Mean`, broadcasts
  the result. World-size-1 and zero-byte tensors are short-circuited.
- REQ-4: `pub fn broadcast<T: Float>(tensor, backend, root) ->
  FerrotorchResult<Tensor<T>>` — the root rank sends its data to every
  other rank, every other rank fills a buffer of matching shape from
  the root. Validates `root < world_size`.
- REQ-5: `pub fn all_gather` / `pub fn all_gather_with_timeout` —
  every rank's tensor is concatenated along dim 0 across all ranks; the
  result shape multiplies the input's dim 0 by `world_size`. Empty
  tensors return an empty gathered tensor of the right shape.
  Zero-dim (scalar) inputs become a 1-D `[world_size]` output.
- REQ-6: `pub fn reduce_scatter` / `pub fn reduce_scatter_with_timeout`
  — combine of allreduce + scatter: rank 0 reduces, then sends each
  rank its `numel / world_size` slice. Requires `numel %
  world_size == 0` AND `shape[0] % world_size == 0` (the dim-0
  divisibility is the API contract; the numel check is a redundant
  guard for empty-shape inputs).
- REQ-7: `pub fn reduce_scatter_tensor` — alias of `reduce_scatter`
  named to match `torch.distributed.reduce_scatter_tensor` (the
  framework's newer name for the single-tensor form).
- REQ-8: `pub fn all_to_all` / `pub fn all_to_all_with_timeout` —
  each rank splits its input into `world_size` equal chunks and
  exchanges chunk `dst` with rank `dst`. The output preserves the
  input's shape. Mirrors `torch.distributed.all_to_all_single`.
  Pair ordering avoids deadlock: lower rank sends first then
  receives, higher rank receives first then sends.
- REQ-9: `pub fn all_to_all_single_uneven<T: Float>(tensor,
  send_sizes, recv_sizes, backend) -> FerrotorchResult<Tensor<T>>`
  — uneven-split all-to-all where each rank declares per-peer send
  and receive sizes. Mirrors PyTorch's `all_to_all_single` with
  `output_split_sizes` / `input_split_sizes` kwargs. The self-slot
  is validated for size symmetry; cross-rank `send_sizes[i] ==
  recv_sizes[self]` is NOT validated (the runtime will surface a
  length error if mismatched).
- REQ-10: `pub fn barrier(backend) -> FerrotorchResult<()>` — thin
  forwarder to `Backend::barrier`. Mirrors `torch.distributed.barrier`.
- REQ-11: Byte serialization helpers `pub(crate) fn floats_to_bytes`
  / `pub(crate) fn bytes_to_floats` reinterpreting `&[T]` ↔
  `Vec<u8>` for the wire format. Uses `copy_nonoverlapping` on the
  recv side to avoid alignment requirements on the byte buffer.
  Both functions are SAFETY-documented at their unsafe blocks.

## Acceptance Criteria

- [x] AC-1: `allreduce` with 4 in-process ranks each holding `[r, r,
  r]` returns `[6, 6, 6]` for `Sum` and `[1.5, 1.5, 1.5]` for `Mean`.
- [x] AC-2: `broadcast` from rank 0 of value 42 reaches all 4 ranks.
- [x] AC-3: `all_gather` of 4 ranks each holding `[r*10, r*10+1]`
  produces `[0,1,10,11,20,21,30,31]` on every rank.
- [x] AC-4: `reduce_scatter` of 4 ranks each holding `[1,2,3,4]`
  with `Sum` produces `[4]`, `[8]`, `[12]`, `[16]` on ranks 0..3.
- [x] AC-5: `reduce_scatter` with a 3-element input across 2 ranks
  returns `Err` (not evenly divisible).
- [x] AC-6: `all_to_all` between 2 ranks correctly swaps chunks.
- [x] AC-7: `all_to_all_single_uneven` between 2 ranks with
  asymmetric splits produces the expected per-rank layouts.
- [x] AC-8: `reduce_scatter_tensor` matches `reduce_scatter` for the
  same inputs.
- [x] AC-9: `floats_to_bytes` / `bytes_to_floats` round-trip both
  `f32` and `f64` slices exactly.

## Architecture

### Reduce-op enum (REQ-1) and default timeout (REQ-2)

`pub enum ReduceOp` carries `Sum` and `Mean`. The `Mean` variant
divides by `world_size` at the end of the reduction (rank 0 does the
division before broadcast in `allreduce`; the same pattern applies in
`reduce_scatter`). `DEFAULT_COLLECTIVE_TIMEOUT` is 60 seconds — long
enough that healthy collectives never trip it but short enough that
test failures surface quickly.

### Allreduce (REQ-3)

`allreduce` is a thin wrapper around `allreduce_with_timeout`.
`allreduce_with_timeout` implements:

1. World-size-1 / zero-byte short-circuits (return `tensor.clone()`).
2. Rank 0: read `local = tensor.data_vec()?`. For each `src in
   1..world_size`: `backend.recv_timeout(&mut recv_buf, src,
   timeout)?`, deserialize, `accum += peer`. Apply `Mean` division.
   Broadcast `accum` to every other rank.
3. Non-zero ranks: send local data to rank 0, recv reduced result.

The byte buffer is allocated once per receive (`Vec::resize` is not
used to avoid stale-data hazards).

### Broadcast (REQ-4)

`broadcast` validates `root < world_size` and short-circuits
world-size-1. Root rank sends `tensor.data_vec()?` bytes to every
non-root rank; non-root ranks recv into a buffer matching the local
tensor's shape. Root returns `tensor.clone()` (no self-copy); non-
root constructs a fresh tensor from the received bytes.

### All-gather (REQ-5)

`all_gather` is a thin wrapper around `all_gather_with_timeout`.
Output shape: if input shape is empty (scalar), output is
`[world_size]`; otherwise output is input shape with dim 0
multiplied by `world_size`. Zero-numel inputs short-circuit with an
empty data vec and the right shape.

Rank 0: collect every peer's `numel` floats, validate the peer-sent
length against `numel` (returns `SizeMismatch` on mismatch),
concatenate in rank order, broadcast the gathered tensor to every
other rank.
Non-zero ranks: send local data, receive the full gathered tensor.

### Reduce-scatter family (REQ-6 / REQ-7)

`reduce_scatter` validates `numel % world_size == 0` AND `shape[0]
% world_size == 0` (the second check fires for non-empty shapes).
The chunk size is `numel / world_size`. Rank 0 reduces (same loop as
allreduce), divides by `world_size` for `Mean`, and sends each rank
its chunk (`accum[rank * chunk..(rank + 1) * chunk]`). Non-zero
ranks send local + recv their chunk.

`reduce_scatter_tensor` is a name-alias of `reduce_scatter` for
porting from PyTorch's newer name.

### All-to-all family (REQ-8 / REQ-9)

`all_to_all` pre-splits the input into `world_size` chunks. For each
peer != self: lower rank sends-then-receives, higher rank receives-
then-sends. The output preserves the input shape; chunk `i` is the
data received from rank `i` (slot `self` comes from `send_chunks[rank]`
directly). Sent chunks are `.clear()`-ed early to free memory.

`all_to_all_single_uneven` accepts per-peer `send_sizes` and
`recv_sizes` vectors. Validates:

- `send_sizes.len() == world_size` and `recv_sizes.len() ==
  world_size`.
- `tensor.numel() == sum(send_sizes)`.
- Self-slot symmetry: `send_sizes[self] == recv_sizes[self]`.

Then exchanges per-peer chunks at the offsets computed from
`send_offsets` / `recv_offsets` running prefix sums.

### Barrier (REQ-10)

`pub fn barrier(backend: &dyn Backend) -> FerrotorchResult<()>`
forwards directly to `backend.barrier()`. Trivial; exists so callers
can `use crate::collective::barrier` symmetric with other collectives.

### Byte serialization (REQ-11)

`pub(crate) fn floats_to_bytes<T: Float>(data: &[T]) -> Vec<u8>` uses
`std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len)`
inside an `unsafe { ... }` block. SAFETY substantiation at line 695:
`T` is `f32` or `f64` (POD, no padding), the slice is valid for
`byte_len` bytes.

`pub(crate) fn bytes_to_floats<T: Float>(bytes: &[u8]) -> Vec<T>`
copies one element at a time via `std::ptr::copy_nonoverlapping`
into a `MaybeUninit<T>` to avoid the source buffer's alignment
requirement (the byte buffer from `recv` is `Vec<u8>` and may not be
T-aligned). SAFETY substantiation at line 715-717.

Both helpers are `pub(crate)` (not `pub`) — the byte interface is an
implementation detail other crate modules (`p2p.rs`) share, not a
public API.

### Consumer sites (production, non-test)

- `ferrotorch-distributed/src/ddp.rs` — `use
  crate::collective::{ReduceOp, allreduce};` — DDP wraps allreduce
  around the backward pass to synchronize gradients across ranks.
- `ferrotorch-distributed/src/fsdp.rs` — `use
  crate::collective::{ReduceOp, all_gather, allreduce, reduce_scatter};`
  — FSDP all-gathers parameters during forward and reduce-scatters
  gradients during backward.
- `ferrotorch-distributed/src/sync_batch_norm.rs` — `use
  crate::collective::{ReduceOp, allreduce};` — SyncBatchNorm
  allreduces mean / variance statistics across ranks.
- `ferrotorch-distributed/src/p2p.rs` — `use
  crate::collective::{DEFAULT_COLLECTIVE_TIMEOUT, bytes_to_floats,
  floats_to_bytes};` — P2P borrows the byte helpers + the default
  timeout.
- `ferrotorch-distributed/src/async_collective.rs` — `use
  crate::collective::{ReduceOp, all_gather, reduce_scatter};` — the
  async wrappers spawn these synchronous primitives.
- `ferrotorch-distributed/src/gpu_collective.rs` — uses
  `ReduceOp`, `allreduce`, `broadcast` (host fallback path).
- `ferrotorch-distributed/src/nccl_backend.rs` and
  `ferrotorch-distributed/src/nccl_collective.rs` use `ReduceOp`
  for the NCCL-side enum mapping.
- `ferrotorch-distributed/src/dtensor.rs` — uses `ReduceOp` for
  `Placement::Partial(ReduceOp)`.
- `ferrotorch-distributed/src/lib.rs` — re-exports the
  collective surface at the crate root.
- `ferrotorch/src/lib.rs` — meta-crate `pub use
  ferrotorch_distributed::*;` exposes the collective ops.

## Parity contract

No parity-sweep ops in the route (`parity_ops = []`). The contract is
the PyTorch collective shape:

- `dist.all_reduce(tensor, op=ReduceOp.SUM, group=None,
  async_op=False)` (cf.
  `torch/distributed/distributed_c10d.py:3143`): matches ferrotorch's
  `allreduce(tensor, backend, op)`. R-DEV-4: `op` is an enum
  (`ReduceOp::Sum` / `ReduceOp::Mean`); ferrotorch ships the 2
  variants DDP / FSDP / SyncBatchNorm consume. PRODUCT / MIN / MAX /
  BAND / BOR / BXOR / PREMUL_SUM are NOT implemented — see honest
  underclaim.
- `dist.broadcast(tensor, src, group=None, async_op=False)`
  (`distributed_c10d.py:3073`): matches ferrotorch's
  `broadcast(tensor, backend, root)`. The argument is `root` rather
  than `src` to match the broader collective vocabulary of "root
  reduction rank."
- `dist.all_gather(tensor_list, tensor, group, async_op=False)`
  (`distributed_c10d.py:4179`): PyTorch fills a caller-allocated
  list of `world_size` tensors; ferrotorch returns a single
  concatenated tensor with dim 0 multiplied by `world_size`.
  R-DEV-4 deviation (no caller-mutated list shape).
- `dist.reduce_scatter(output, input_list, op, group, async_op)` /
  `dist.reduce_scatter_tensor(output, input, op, group, async_op)`
  (`distributed_c10d.py:4751,4808`): ferrotorch ships only the
  single-tensor form. Mirrors `reduce_scatter_tensor` directly;
  `reduce_scatter` is the same function under the older name.
- `dist.all_to_all(out_list, in_list, group, async_op)` /
  `dist.all_to_all_single(out, in, output_split_sizes,
  input_split_sizes, group, async_op)`
  (`distributed_c10d.py:5088,4939`): ferrotorch ships the single-
  tensor form (`all_to_all`) and the uneven form
  (`all_to_all_single_uneven`). The list-of-tensors form is
  emulable by the caller via cat/split.
- `dist.barrier(group, async_op, device_ids)`
  (`distributed_c10d.py:5227`): matches ferrotorch's `barrier`.
  R-DEV-7: `device_ids` is not exposed (the underlying TCP/Simulated
  backend has no device concept; NCCL barrier carries its own
  device context).

## Verification

- `cargo test -p ferrotorch-distributed --lib` runs the
  `#[cfg(test)] mod tests` at lines 730-1264 covering 26 tests:
  - allreduce: `test_allreduce_sum_4_ranks`,
    `test_allreduce_mean_4_ranks`, `test_allreduce_single_rank`.
  - broadcast: `test_broadcast_from_rank_0`,
    `test_broadcast_invalid_root`.
  - all_gather: `test_all_gather_4_ranks`,
    `test_all_gather_preserves_shape`,
    `test_all_gather_single_rank`, `test_all_gather_zero_size`.
  - reduce_scatter: `test_reduce_scatter_sum_4_ranks`,
    `test_reduce_scatter_mean_2_ranks`,
    `test_reduce_scatter_single_rank`,
    `test_reduce_scatter_indivisible`,
    `test_reduce_scatter_preserves_shape`.
  - all_to_all: `test_all_to_all_2_ranks`,
    `test_all_to_all_4_ranks`,
    `test_all_to_all_world_size_1_is_identity`,
    `test_all_to_all_rejects_uneven_numel`.
  - all_to_all_single_uneven:
    `test_all_to_all_single_uneven_2_ranks`,
    `test_all_to_all_single_uneven_wrong_slice_lengths_error`.
  - reduce_scatter_tensor:
    `test_reduce_scatter_tensor_matches_reduce_scatter`.
  - barrier: `test_barrier_completes`.
  - byte serialization: `test_bytes_roundtrip_f32`,
    `test_bytes_roundtrip_f64`.
- Lint: `cargo clippy -p ferrotorch-distributed -- -D warnings` PASS.
  Note `#[allow(clippy::needless_range_loop)]` at line 456 covers
  the `for peer in 0..world_size` exchange loop where the peer
  index drives both halves of the send/recv pair.
- Parity-sweep: no ops; integer grep count is 0 by construction.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum ReduceOp` in `ferrotorch-distributed/src/collective.rs`; non-test consumers: `ferrotorch-distributed/src/ddp.rs`, `ferrotorch-distributed/src/fsdp.rs`, `ferrotorch-distributed/src/sync_batch_norm.rs`, `ferrotorch-distributed/src/dtensor.rs` (used in `Placement::Partial(ReduceOp)`). |
| REQ-2 | SHIPPED | impl: `pub const DEFAULT_COLLECTIVE_TIMEOUT` in `ferrotorch-distributed/src/collective.rs`; non-test consumer: `ferrotorch-distributed/src/p2p.rs` (`use crate::collective::{DEFAULT_COLLECTIVE_TIMEOUT, ...};`), used inline at `ferrotorch-distributed/src/p2p.rs`. |
| REQ-3 | SHIPPED | impl: `pub fn allreduce` in `ferrotorch-distributed/src/collective.rs` and `allreduce_with_timeout` in `ferrotorch-distributed/src/collective.rs`; non-test consumers: `ferrotorch-distributed/src/ddp.rs` invokes `allreduce` in the DDP backward hook; `ferrotorch-distributed/src/fsdp.rs` and `ferrotorch-distributed/src/sync_batch_norm.rs` import it. |
| REQ-4 | SHIPPED | impl: `pub fn broadcast` in `ferrotorch-distributed/src/collective.rs`; non-test consumer: `ferrotorch-distributed/src/gpu_collective.rs` imports `broadcast` and invokes it on the host fallback path. |
| REQ-5 | SHIPPED | impl: `pub fn all_gather` in `ferrotorch-distributed/src/collective.rs` and `pub fn all_gather_with_timeout` in `ferrotorch-distributed/src/collective.rs`; non-test consumers: `ferrotorch-distributed/src/fsdp.rs` (forward all-gather of sharded parameters), `ferrotorch-distributed/src/async_collective.rs` (`async_all_gather` wraps it). |
| REQ-6 | SHIPPED | impl: `pub fn reduce_scatter` in `ferrotorch-distributed/src/collective.rs` and `pub fn reduce_scatter_with_timeout` in `ferrotorch-distributed/src/collective.rs`; non-test consumers: `ferrotorch-distributed/src/fsdp.rs` (backward reduce-scatter of gradients), `ferrotorch-distributed/src/async_collective.rs` (`async_reduce_scatter` wraps it). |
| REQ-7 | SHIPPED | impl: `pub fn reduce_scatter_tensor` in `ferrotorch-distributed/src/collective.rs` (alias of `reduce_scatter`); non-test consumer: `ferrotorch-distributed/src/lib.rs` re-exports the symbol at the crate root, reaching `ferrotorch/src/lib.rs` for user code. |
| REQ-8 | SHIPPED | impl: `pub fn all_to_all` in `ferrotorch-distributed/src/collective.rs` and `pub fn all_to_all_with_timeout` in `ferrotorch-distributed/src/collective.rs`; non-test consumer: `ferrotorch-distributed/src/lib.rs` re-exports `all_to_all` and `all_to_all_with_timeout` at the crate root for user code (CL-460 boundary API). |
| REQ-9 | SHIPPED | impl: `pub fn all_to_all_single_uneven` in `ferrotorch-distributed/src/collective.rs`; non-test consumer: `ferrotorch-distributed/src/lib.rs` re-exports `all_to_all_single_uneven` at the crate root for user code orchestrating tensor-parallel layer's column-row layout transitions. |
| REQ-10 | SHIPPED | impl: `pub fn barrier` in `ferrotorch-distributed/src/collective.rs`; non-test consumer: `ferrotorch-distributed/src/lib.rs` re-exports `barrier` at the crate root, reached through `ferrotorch/src/lib.rs`. |
| REQ-11 | SHIPPED | impl: `pub(crate) fn floats_to_bytes` in `ferrotorch-distributed/src/collective.rs` and `pub(crate) fn bytes_to_floats` in `ferrotorch-distributed/src/collective.rs`; non-test consumer: `ferrotorch-distributed/src/p2p.rs` imports and calls both helpers in `send` / `recv_with_timeout` / `recv_into_with_timeout`. |
