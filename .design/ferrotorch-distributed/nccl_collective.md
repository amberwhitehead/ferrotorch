# NCCL collective wrappers (GpuBufferHandle layer)

<!--
tier: 3-component
status: draft
baseline-pytorch: main (user's local clone /home/doll/pytorch)
upstream-paths:
  - torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp
  - torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp
-->

## Summary

`ferrotorch-distributed/src/nccl_collective.rs` is the thin
type-aware layer that sits between the workspace `GpuBufferHandle`
abstraction and the `NcclBackend::*_raw` FFI surface. It infers
the NCCL data type from the buffer's element size, extracts the
raw CUDA device pointer via the GPU dispatch interface, and
delegates to `NcclBackend`. Mirrors the role of PyTorch's
`ProcessGroupNCCL` collective-method handlers — they take a typed
PyTorch tensor, infer NCCL dtype from `tensor.scalar_type()`, and
call the matching `ncclXxx` FFI symbol.

## Requirements

- REQ-1: `pub fn nccl_allreduce(buffer: &mut GpuBufferHandle,
  backend: &NcclBackend, op: &ReduceOp) -> FerrotorchResult<()>`
  auto-detects f32 vs f64 from the buffer's elem size and forwards
  to `nccl_allreduce_dtype`.
- REQ-2: `pub fn nccl_allreduce_dtype(buffer, backend, op,
  dtype: NcclDataType) -> FerrotorchResult<()>` extracts a raw
  device pointer, converts `op` via `reduce_op_to_nccl`, and
  invokes `NcclBackend::allreduce_raw` in-place
  (`sendbuf == recvbuf`).
- REQ-3: `pub fn nccl_broadcast(buffer: &mut GpuBufferHandle,
  backend: &NcclBackend, root: usize) -> FerrotorchResult<()>`
  auto-detects dtype and forwards to `nccl_broadcast_dtype`. The
  explicit-dtype variant invokes `NcclBackend::broadcast_raw`
  in-place; `root` is cast to `i32` (the NCCL ABI shape).
- REQ-4: `pub fn nccl_all_gather(send_buf: &GpuBufferHandle,
  recv_buf: &mut GpuBufferHandle, backend: &NcclBackend) ->
  FerrotorchResult<()>` auto-detects dtype FROM `send_buf` and
  forwards to `nccl_all_gather_dtype`. The explicit-dtype variant
  invokes `NcclBackend::all_gather_raw` with `sendcount =
  send_buf.len()`; the caller is responsible for sizing `recv_buf`
  to `sendcount * world_size` elements (otherwise NCCL writes
  past the buffer).
- REQ-5: `pub fn nccl_reduce_scatter(send_buf: &GpuBufferHandle,
  recv_buf: &mut GpuBufferHandle, backend: &NcclBackend, op:
  &ReduceOp) -> FerrotorchResult<()>` auto-detects dtype FROM
  `recv_buf` and forwards to `nccl_reduce_scatter_dtype`. The
  explicit-dtype variant invokes `NcclBackend::reduce_scatter_raw`
  with `recvcount = recv_buf.len()`.
- REQ-6: `fn infer_dtype(handle: &GpuBufferHandle) ->
  FerrotorchResult<NcclDataType>` reads the active GPU backend's
  `buffer_elem_size(handle)` and maps:
  - 4 → `Float32`
  - 8 → `Float64`
  - 0 → `InvalidArgument` "unrecognized buffer type"
  - other → `InvalidArgument` "unsupported element size {n}"
- REQ-7: `fn get_ptr` / `fn get_ptr_mut` extract a raw device
  pointer via `gpu_backend().raw_device_ptr` / `_mut` and
  reject null pointers with `InvalidArgument`.

## Acceptance Criteria

- [x] AC-1: Module compiles only under `#[cfg(feature =
  "nccl")]` (gated in `lib.rs` line 207).
- [x] AC-2: Auto-detect dtype path returns `Float32` for a
  4-byte-elem buffer; `Float64` for an 8-byte-elem buffer
  (`infer_dtype` match arms).
- [x] AC-3: Null device pointer raises `InvalidArgument`
  with message "buffer has no valid device pointer"
  (`get_ptr` / `get_ptr_mut`).
- [x] AC-4: `nccl_allreduce` invokes the raw method in-place
  (the same ptr is used for `sendbuf` and `recvbuf` via
  `ptr.cast_const(), ptr`).
- [x] AC-5: `nccl_reduce_scatter` uses `recv_buf.len()` for
  `recvcount` (the receive-side chunk count, not the
  send-side total).

## Architecture

`fn infer_dtype` queries `gpu_dispatch::gpu_backend()` for the
active GPU backend handle, errors with `DeviceUnavailable` if
none is registered, then reads `backend.buffer_elem_size(handle)`.
The match covers `4 → Float32`, `8 → Float64`, `0 → unrecognised`,
`other → unsupported`. Note that f16 / bf16 / int dtypes are NOT
covered by this layer (their elem sizes are 2 / 2 / 4 / 8 — only
the 4 and 8 paths trigger). Wider dtype coverage would need to
plumb a typed dtype tag through `GpuBufferHandle` (tracked in the
broader GpuBufferHandle dtype-tagging work).

`fn get_ptr` returns `*const c_void`; `fn get_ptr_mut` returns
`*mut c_void`. Both error with `InvalidArgument` on null
pointers (the GPU dispatch interface returns null for unbacked
handles).

Each public collective fn does the same three steps:
1. Infer the NCCL dtype from a buffer.
2. Extract the device pointer(s).
3. Invoke the matching `NcclBackend::*_raw` inside an `unsafe {
   ... }` block with a `SAFETY:` comment discharging each
   documented obligation.

The SAFETY: comments are quite detailed: each one walks through
the device-pointer validity, in-place vs disjoint-buffer mode,
buffer-size contract (e.g., "recv_buf must have capacity for
sendcount * world_size elements"), the dtype/buffer-element-layout
match, cross-rank consistency, and stream lifetime obligations.
The Rust borrow checker enforces that `&GpuBufferHandle` and `&mut
GpuBufferHandle` don't alias for the all_gather / reduce_scatter
shapes.

### Consumer sites (production, non-test)

- `ferrotorch-distributed/src/lib.rs` — `pub use nccl_collective::
  {nccl_all_gather, nccl_allreduce, nccl_broadcast,
  nccl_reduce_scatter};` at line 245.
- `ferrotorch/src/lib.rs` — meta-crate re-export reaches every
  workspace user.
- Downstream (NN layers / training loops in ferrotorch-nn,
  ferrotorch-llama, etc.) that hold a `GpuBufferHandle` and want
  to allreduce gradients call `nccl_allreduce` directly.

Note: this module's public surface (the four `nccl_*` helpers)
is grandfathered per goal.md S5 — the `pub use` re-exports in
`lib.rs` establish it as part of the crate's API. The downstream
training-loop integration (DDP / FSDP gradient sync on `GpuBufferHandle`)
is the next translation iteration's work.

## Parity contract

No parity-sweep ops. The contract is the per-collective shape:

- `nccl_allreduce` ↔ `ncclAllReduce` (in-place mode allowed).
- `nccl_broadcast` ↔ `ncclBroadcast` (in-place mode allowed
  on root; non-root overwrites `recvbuf`).
- `nccl_all_gather` ↔ `ncclAllGather` (must NOT alias; caller
  sizes `recv_buf` to `sendcount * world_size`).
- `nccl_reduce_scatter` ↔ `ncclReduceScatter` (must NOT alias;
  caller sizes `send_buf` to `recvcount * world_size`).
- F32/F64 auto-detect ↔ PyTorch `ProcessGroupNCCL`'s
  `getNcclDataType(scalar_type)` dispatch (which covers more
  dtypes; ferrotorch's gloo_native / nccl_collective is f32/f64
  scoped per the current GpuBufferHandle elem-size signal).

## Verification

`cargo test -p ferrotorch-distributed --features nccl --lib`
runs no in-file tests (the file has no `#[cfg(test)] mod tests`).
The dispatch wiring is verified at compile time by the
`#[cfg(feature = "nccl")]` gating in `lib.rs` and via the
hardware-gated tests in `gpu_collective::tests` that exercise
the full path from `gpu_allreduce` through `NcclBackend::allreduce_raw`.

No parity-sweep ops; integer grep count is 0 by construction.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn nccl_allreduce` in `ferrotorch-distributed/src/nccl_collective.rs`; non-test consumer: `pub use nccl_collective::{nccl_allreduce, ...}` in `ferrotorch-distributed/src/lib.rs` line 245 reaches `ferrotorch/src/lib.rs`. |
| REQ-2 | SHIPPED | impl: `pub fn nccl_allreduce_dtype` in `ferrotorch-distributed/src/nccl_collective.rs`; non-test consumer: same `lib.rs` line 245 re-export. `nccl_allreduce` itself calls `nccl_allreduce_dtype`. |
| REQ-3 | SHIPPED | impl: `pub fn nccl_broadcast` and `pub fn nccl_broadcast_dtype` in `ferrotorch-distributed/src/nccl_collective.rs`; non-test consumer: `lib.rs` line 245 re-export. |
| REQ-4 | SHIPPED | impl: `pub fn nccl_all_gather` and `pub fn nccl_all_gather_dtype` in `ferrotorch-distributed/src/nccl_collective.rs`; non-test consumer: `lib.rs` line 245 re-export. |
| REQ-5 | SHIPPED | impl: `pub fn nccl_reduce_scatter` and `pub fn nccl_reduce_scatter_dtype` in `ferrotorch-distributed/src/nccl_collective.rs`; non-test consumer: `lib.rs` line 245 re-export. |
| REQ-6 | SHIPPED | impl: `fn infer_dtype` in `ferrotorch-distributed/src/nccl_collective.rs`; non-test consumer: every auto-detect entry point (`nccl_allreduce`, `nccl_broadcast`, `nccl_all_gather`, `nccl_reduce_scatter`) in the same file invokes it. |
| REQ-7 | SHIPPED | impl: `fn get_ptr` / `fn get_ptr_mut` in `ferrotorch-distributed/src/nccl_collective.rs`; non-test consumer: every `*_dtype` helper in the same file uses them to extract device pointers before the FFI call. |
