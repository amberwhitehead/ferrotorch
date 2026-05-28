# NCCL backend (GPU-native collective communication)

<!--
tier: 3-component
status: draft
baseline-pytorch: main (user's local clone /home/doll/pytorch)
upstream-paths:
  - torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp
  - torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp
  - torch/distributed/distributed_c10d.py
-->

## Summary

`ferrotorch-distributed/src/nccl_backend.rs` is the feature-gated
public handle for NCCL-backed distributed training. It owns an NCCL
communicator (`Mutex<NcclComm>` raw FFI pointer) and a dedicated CUDA
stream for NCCL operations, exposing both the generic `Backend` trait
(via `as_nccl_backend` downcast for the GPU fast path) and four
GPU-native raw collective methods (`allreduce_raw`, `broadcast_raw`,
`all_gather_raw`, `reduce_scatter_raw`) that operate directly on
device pointers without CPU round-trips. Mirrors PyTorch's
`ProcessGroupNCCL` shape: NCCL handles collectives on GPU buffers;
byte-level P2P send/recv falls back to Gloo/TCP (we return
`UnsupportedOp` for that case, expecting callers to pair with a
`TcpBackend` via `HybridBackend`). The CUDA stream is loaded lazily
via `dlopen`/`dlsym` so the crate compiles on machines without
`libcudart.so.12`.

## Requirements

- REQ-1: `pub struct NcclBackend` carries `comm: Mutex<NcclComm>`,
  `rank: usize`, `world_size: usize`, `stream: *mut c_void` (dedicated
  CUDA stream or null for default), `owns_stream: bool`. `unsafe
  impl Send + Sync for NcclBackend` because the underlying NCCL
  comm is safe to share when protected by the mutex.
- REQ-2: `pub fn NcclBackend::new(rank, world_size, unique_id) ->
  FerrotorchResult<Self>` invokes `nccl_sys::comm_init_rank` then
  attempts to create a dedicated CUDA stream via `create_nccl_stream`
  (falls back to null/default stream on dlopen failure).
- REQ-3: `pub fn NcclBackend::with_stream(rank, world_size,
  unique_id, stream) -> FerrotorchResult<Self>` is the caller-owns-
  stream variant; `owns_stream = false` so `Drop` does not destroy
  it.
- REQ-4: `pub fn NcclBackend::synchronize(&self) -> FerrotorchResult<()>`
  blocks until all enqueued NCCL operations have completed on
  `self.stream` (no-op if stream is null — the default stream is
  implicitly synchronized).
- REQ-5: `pub unsafe fn allreduce_raw(&self, sendbuf, recvbuf, count,
  datatype, op) -> FerrotorchResult<()>` enqueues `ncclAllReduce` on
  `self.stream`. The `# Safety` rustdoc enumerates the cross-rank
  contract (same `count`/`datatype`/`op` on every rank), buffer-
  validity, device binding, and lifetime obligations.
- REQ-6: `pub unsafe fn broadcast_raw(&self, sendbuf, recvbuf, count,
  datatype, root) -> FerrotorchResult<()>` enqueues `ncclBroadcast`,
  with `root` consistency obligation documented.
- REQ-7: `pub unsafe fn all_gather_raw(&self, sendbuf, recvbuf,
  sendcount, datatype) -> FerrotorchResult<()>` enqueues
  `ncclAllGather`. The `# Safety` rustdoc explicitly warns that
  `recvbuf` must have capacity for `sendcount * world_size *
  size_of(datatype)` bytes — NCCL writes past undersized buffers
  with no bounds check.
- REQ-8: `pub unsafe fn reduce_scatter_raw(&self, sendbuf, recvbuf,
  recvcount, datatype, op) -> FerrotorchResult<()>` enqueues
  `ncclReduceScatter`. Same buffer-size warning as REQ-7 on the
  send side (`recvcount * world_size`).
- REQ-9: `impl Backend for NcclBackend` returns `UnsupportedOp`
  for byte-level `send` / `recv`, returns `Some(self)` from
  `as_nccl_backend()`, and implements `barrier()` as a zero-count
  `ncclAllReduce` (a documented NCCL synchronisation point).
- REQ-10: `impl Drop for NcclBackend` destroys the comm via
  `nccl_sys::comm_destroy` under the mutex guard, and destroys the
  CUDA stream via `destroy_stream` iff `owns_stream`.
- REQ-11: `pub fn reduce_op_to_nccl(op: &ReduceOp) -> NcclRedOp`
  converts the workspace `ReduceOp` enum (Sum / Mean) to the
  matching NCCL enum (`Sum` / `Avg`).
- REQ-12: `pub fn is_nccl_available() -> bool` returns
  `nccl_sys::is_available()` (dlopen-probes `libnccl.so.2`).

## Acceptance Criteria

- [x] AC-1: `pub struct NcclBackend` is `Send + Sync` (unsafe
  impls under the documented mutex-protected contract).
- [x] AC-2: Module compiles only under `#[cfg(feature = "nccl")]`
  (gated in `lib.rs` line 205).
- [x] AC-3: `unsafe fn allreduce_raw` carries a `SAFETY:`
  block at the FFI call site that maps every caller obligation
  to an `nccl_sys::all_reduce` precondition.
- [x] AC-4: `impl Drop` runs `comm_destroy` exactly once
  (Rust's `Drop` language guarantee) under the mutex guard.
- [x] AC-5: `Backend::send` / `Backend::recv` return
  `UnsupportedOp` with a message naming the GPU-collective /
  `TcpBackend` upgrade paths.
- [x] AC-6: `Backend::as_nccl_backend(&self)` returns
  `Some(self)` (the trait's default returns `None`).
- [x] AC-7: `Backend::barrier()` invokes `ncclAllReduce` with
  `count = 0` on the default Sum / Float32 — a valid NCCL
  sync point per the NCCL programming guide.

## Architecture

`pub struct NcclBackend` has five fields. The `comm: Mutex<NcclComm>`
wraps the opaque FFI pointer so it can be safely shared across
threads; every NCCL call acquires the lock via `fn lock_comm` and
dereferences (`*` copies the `*mut c_void` handle out of the
guard). The `stream: *mut c_void` is either null (= default
stream, always valid per CUDA semantics) or a dedicated stream
created by `fn create_nccl_stream` in `Self::new`. `owns_stream`
distinguishes the `new` / `with_stream` paths so `Drop` knows
whether to destroy.

The four `pub unsafe fn *_raw` methods are the GPU-native
collective surface. Each one:
1. Acquires the comm mutex (`*self.lock_comm()?`).
2. Calls `nccl_sys::*` inside an `unsafe { ... }` block.
3. Maps any `NcclError` to `DistributedError::Io { message: ... }`.

The `# Safety` rustdoc on each method enumerates the caller's
obligations: buffer validity (`count * size_of(datatype)` bytes
addressable), aliasing rules (`sendbuf` and `recvbuf` may equal
for in-place ops on allreduce / broadcast; must differ for
all_gather / reduce_scatter), cross-rank consistency (same
`count` / `datatype` / `op` on every rank), and stream
lifetime ("buffers remain alive and unmodified until the NCCL
stream is synchronised"). The SAFETY block inside each fn body
maps every documented obligation to a `nccl_sys::*`
precondition, completing the chain.

`fn create_nccl_stream` / `fn synchronize_stream` /
`fn destroy_stream` lazily `dlopen` `libcudart.so.12` (with a
fallback to the unversioned soname for older toolkits) and
`dlsym` the three CUDA Runtime symbols
(`cudaStreamCreateWithFlags`, `cudaStreamSynchronize`,
`cudaStreamDestroy`). The function pointers are transmuted to
typed `extern "C"` shapes; each `unsafe` block carries a
detailed SAFETY: comment documenting the ABI mapping
(`cudaStream_t = *mut c_void`, `cudaError_t = i32`, `unsigned
int = u32`). The dlopen handle is intentionally never
`dlclose`-d — CUDA driver convention is that `libcudart`
remains loaded for the process lifetime.

`impl Drop for NcclBackend` runs under the mutex guard: only
if `*comm != null`, call `nccl_sys::comm_destroy(*comm)`
inside an `unsafe { ... }` block with a discharge-all SAFETY
comment. Errors are discarded with `let _ = ...` since `Drop`
cannot return them; a failed destroy leaks the NCCL communicator
on the GPU-side state machine — preferable to panicking in Drop.
If `owns_stream && !stream.is_null()`, the stream is destroyed
via `destroy_stream`.

`impl Backend for NcclBackend` returns `UnsupportedOp` for
byte-level send / recv (NCCL's P2P is GPU-buffer-only; CPU-
side bytes must go through `TcpBackend`). `as_nccl_backend()`
returns `Some(self)` — this is the downcast hook that
`gpu_collective::gpu_allreduce` / `gpu_broadcast` query to
pick the NCCL fast path. `barrier()` is a documented
zero-count `ncclAllReduce` synchronisation point.

`pub fn reduce_op_to_nccl` is a 4-line match: `ReduceOp::Sum →
NcclRedOp::Sum`, `ReduceOp::Mean → NcclRedOp::Avg`. NCCL has
no Mean; `Avg` is the rename PyTorch's `ProcessGroupNCCL` does.

### Consumer sites (production, non-test)

- `ferrotorch-distributed/src/nccl_collective.rs` —
  `nccl_allreduce` / `nccl_broadcast` / `nccl_all_gather` /
  `nccl_reduce_scatter` take `backend: &NcclBackend` and call
  the `*_raw` methods.
- `ferrotorch-distributed/src/hybrid_backend.rs` —
  `HybridBackend.nccl: NcclBackend` holds an NCCL backend
  alongside a `TcpBackend`; `pub fn nccl(&self) -> &NcclBackend`
  hands it out.
- `ferrotorch-distributed/src/gpu_collective.rs` — `fn
  nccl_path_allreduce` / `_broadcast` take `nccl: &NcclBackend`
  and call `allreduce_raw` / `broadcast_raw`.
- `ferrotorch-distributed/src/ucc_backend.rs` —
  `UccBackend.gpu_inner: Mutex<Option<Arc<NcclBackend>>>`
  holds an attached NCCL communicator for `ucc-native-gpu`
  builds.
- `ferrotorch-distributed/src/lib.rs` — `pub use
  nccl_backend::{NcclBackend, is_nccl_available};` at line 243.

## Parity contract

No parity-sweep ops. The contract is the C10d
`ProcessGroupNCCL` shape:

- `ProcessGroupNCCL::allreduce` ↔ `NcclBackend::allreduce_raw`
  via `gpu_collective::gpu_allreduce`.
- `ProcessGroupNCCL::broadcast` ↔ `NcclBackend::broadcast_raw`
  via `gpu_collective::gpu_broadcast`.
- `ProcessGroupNCCL::allgather` ↔ `NcclBackend::all_gather_raw`
  via `nccl_collective::nccl_all_gather`.
- `ProcessGroupNCCL::reduce_scatter` ↔
  `NcclBackend::reduce_scatter_raw` via
  `nccl_collective::nccl_reduce_scatter`.
- `ProcessGroupNCCL::barrier` (zero-count allreduce on default
  stream) ↔ `NcclBackend::barrier`.
- `ProcessGroupNCCL` rejection of CPU-side P2P send/recv ↔
  `impl Backend for NcclBackend::send` / `recv` return
  `UnsupportedOp`.
- `ProcessGroupNCCL`'s reliance on a dedicated CUDA stream
  for collective-compute overlap ↔ `self.stream: *mut c_void`.

## Verification

`cargo test -p ferrotorch-distributed --lib` does NOT exercise
this module on default builds (it's `#[cfg(feature = "nccl")]`-
gated). With `--features=nccl`, hardware-gated tests (each
`#[ignore]`'d so they don't run unattended) live in
`gpu_collective::tests` and `ucc_backend::tests`:

- `gpu_allreduce_dispatches_to_nccl_in_single_rank_mode`
  exercises `NcclBackend::new` → `allreduce_raw` → expected-
  identity result for single rank.
- `gpu_broadcast_dispatches_to_nccl_in_single_rank_mode`
  same for broadcast.
- `ucc_native_gpu_allreduce_via_nccl_single_rank` exercises
  the `UccBackend.with_nccl` dispatch shape.

These tests are `#[ignore]`'d because they require both
`libnccl2` and a CUDA device — out of scope for the standard
test gauntlet, in scope for the targeted GPU CI lane.

`cargo clippy -p ferrotorch-distributed --features nccl --
-D warnings`: PASS.

No parity-sweep ops; integer grep count is 0 by construction.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct NcclBackend` + `unsafe impl Send/Sync` in `ferrotorch-distributed/src/nccl_backend.rs`; non-test consumer: `ferrotorch-distributed/src/hybrid_backend.rs` `HybridBackend.nccl: NcclBackend` field. |
| REQ-2 | SHIPPED | impl: `pub fn NcclBackend::new` in `ferrotorch-distributed/src/nccl_backend.rs`; non-test consumer: `ferrotorch-distributed/src/hybrid_backend.rs` `HybridBackend::new` invokes `NcclBackend::new(rank, world_size, unique_id)?`. |
| REQ-3 | SHIPPED | impl: `pub fn NcclBackend::with_stream` in `ferrotorch-distributed/src/nccl_backend.rs`; non-test consumer: re-export at `ferrotorch-distributed/src/lib.rs` line 243 reaches `ferrotorch/src/lib.rs`; the `with_stream` constructor is part of the public API surface for callers wanting to share a CUDA stream with the compute path. |
| REQ-4 | SHIPPED | impl: `pub fn NcclBackend::synchronize` in `ferrotorch-distributed/src/nccl_backend.rs`; non-test consumer: `ferrotorch-distributed/src/gpu_collective.rs` `fn nccl_path_allreduce` and `fn nccl_path_broadcast` both call `nccl.synchronize()?` before returning the result tensor; `ferrotorch-distributed/src/hybrid_backend.rs` `pub fn synchronize_nccl` forwards to it. |
| REQ-5 | SHIPPED | impl: `pub unsafe fn allreduce_raw` in `ferrotorch-distributed/src/nccl_backend.rs`; non-test consumer: `ferrotorch-distributed/src/nccl_collective.rs` `pub fn nccl_allreduce_dtype` invokes it; `ferrotorch-distributed/src/gpu_collective.rs` `fn nccl_path_allreduce` invokes it. |
| REQ-6 | SHIPPED | impl: `pub unsafe fn broadcast_raw` in `ferrotorch-distributed/src/nccl_backend.rs`; non-test consumer: `ferrotorch-distributed/src/nccl_collective.rs` `pub fn nccl_broadcast_dtype` invokes it; `ferrotorch-distributed/src/gpu_collective.rs` `fn nccl_path_broadcast` invokes it. |
| REQ-7 | SHIPPED | impl: `pub unsafe fn all_gather_raw` in `ferrotorch-distributed/src/nccl_backend.rs`; non-test consumer: `ferrotorch-distributed/src/nccl_collective.rs` `pub fn nccl_all_gather_dtype` invokes it. |
| REQ-8 | SHIPPED | impl: `pub unsafe fn reduce_scatter_raw` in `ferrotorch-distributed/src/nccl_backend.rs`; non-test consumer: `ferrotorch-distributed/src/nccl_collective.rs` `pub fn nccl_reduce_scatter_dtype` invokes it. |
| REQ-9 | SHIPPED | impl: `impl Backend for NcclBackend` in `ferrotorch-distributed/src/nccl_backend.rs`; non-test consumer: `ferrotorch-distributed/src/gpu_collective.rs` `pub fn gpu_allreduce` and `pub fn gpu_broadcast` invoke `backend.as_nccl_backend()` to detect and route through the NCCL fast path. |
| REQ-10 | SHIPPED | impl: `impl Drop for NcclBackend` in `ferrotorch-distributed/src/nccl_backend.rs`; non-test consumer: implicit — every drop site for an `NcclBackend` (e.g., when `HybridBackend` is dropped in `hybrid_backend.rs`) runs the destroy path. |
| REQ-11 | SHIPPED | impl: `pub fn reduce_op_to_nccl` in `ferrotorch-distributed/src/nccl_backend.rs`; non-test consumer: `ferrotorch-distributed/src/nccl_collective.rs` `pub fn nccl_allreduce_dtype` and `nccl_reduce_scatter_dtype` call `reduce_op_to_nccl(op)`; `ferrotorch-distributed/src/gpu_collective.rs` `fn nccl_path_allreduce` calls `reduce_op_to_nccl(&op)`. |
| REQ-12 | SHIPPED | impl: `pub fn is_nccl_available` in `ferrotorch-distributed/src/nccl_backend.rs`; non-test consumer: re-exported at `ferrotorch-distributed/src/lib.rs` line 243 (`pub use nccl_backend::{NcclBackend, is_nccl_available};`); reached through `ferrotorch/src/lib.rs`. |
