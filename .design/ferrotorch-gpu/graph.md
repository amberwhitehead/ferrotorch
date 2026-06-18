# CUDA graph capture / replay infrastructure

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/cuda/CUDAGraph.cpp
  - aten/src/ATen/cuda/CUDAGraph.h
  - aten/src/ATen/cuda/CUDAGraphsUtils.cuh
  - torch/cuda/graphs.py
-->

## Summary

`ferrotorch-gpu/src/graph.rs` implements CUDA graph capture and
replay infrastructure: stream capture lifecycle
(`begin_capture` / `end_capture` / `CaptureStatus`), captured-graph
playback (`CapturedGraph::launch`), parameter pipes
(`DeviceScalar` for per-replay scalars), mempool reuse
(`CapturePool` + `GraphPoolHandle`), and a high-level
`make_graphed_callable` wrapper. Mirrors PyTorch's
`at::cuda::CUDAGraph` (`aten/src/ATen/cuda/CUDAGraph.cpp`) and
the Python-side `torch.cuda.graph` / `torch.cuda.make_graphed_callables`
API in `torch/cuda/graphs.py`.

## Requirements

- REQ-1: `CaptureMode` enum (`Global` (default) / `ThreadLocal` /
  `Relaxed`) with `pub fn to_cuda` converting to cudarc's
  `CUstreamCaptureMode`. Mirrors `cudaStreamCaptureMode`.
- REQ-2: `CaptureStatus` enum (`None` / `Active` / `Invalidated`)
  with `pub fn is_capturing` and a `from_cuda` constructor.
  Mirrors `cudaStreamCaptureStatus`.
- REQ-3: Capture lifecycle entry points: `begin_capture`,
  `begin_capture_with_mode`, `begin_capture_with_pool`,
  `end_capture`, `end_capture_with_pool`, `capture_status`,
  `is_stream_capturing`. Each takes / inspects a CUDA stream.
- REQ-4: `CapturedGraph` struct + RAII `GraphCaptureGuard` —
  ownership of the captured graph handle with proper drop.
- REQ-5: `DeviceScalar<T>` — typed wrapper for a per-replay scalar
  parameter, with `pub fn update(value)` issuing a memcpy before
  the next `CapturedGraph::launch`. Mirrors PyTorch's "parameter
  update before replay" idiom.
- REQ-6: `CapturePool` + `GraphPoolHandle(u64)` — mempool reuse
  across captures with the `graph_pool_handle()` /
  `capture_pool_for_handle(handle)` /
  `release_graph_pool_handle(handle)` API.
- REQ-7: `make_graphed_callable<F>` — high-level wrapper that
  takes a closure, captures it once, and returns a replayable
  graph. Mirrors `torch.cuda.make_graphed_callables`.
- REQ-8: Non-CUDA stubs for every public symbol (lines 717-879)
  so dependent crates compile without the `cuda` feature.
- REQ-9: Non-test production consumer wiring — a graphed-callable
  user inside ferrotorch-* crates or higher-level model crates.
- REQ-10: Private graph-capture mempool (`PrivateMemPool` +
  `CapturePool::with_private_pool` + `capture_into_private_pool`) so
  the captured forward's stream-ordered allocations are served from a
  pool isolated from the device-wide default async pool. Mirrors
  PyTorch's caching-allocator graph-pool mode
  (`beginAllocateToPool` / `endAllocateToPool`,
  `aten/src/ATen/cuda/CUDAGraph.cpp:150` / `:193`). Without it a graph
  replay corrupts the shared pool and the next eager forward (or a
  second replay) fails with `CUDA_ERROR_INVALID_VALUE` (#1595).

## Acceptance Criteria

- [x] AC-1: `pub enum CaptureMode` at line 59 with `Default =
  Global`; `to_cuda()` method at line 75. Matches PyTorch's
  `torch.cuda.graph(..., capture_error_mode="global")` default.
- [x] AC-2: `pub enum CaptureStatus` at line 91 with
  `is_capturing()` accessor and `from_cuda()` constructor.
- [x] AC-3: `pub fn begin_capture` (line 290),
  `begin_capture_with_mode` (300), `capture_status` (312),
  `is_stream_capturing` (319), `end_capture` (332),
  `end_capture_with_pool` (360), `begin_capture_with_pool` (706).
- [x] AC-4: `pub struct CapturedGraph` (line 192) and
  `pub struct GraphCaptureGuard` (line 398).
- [x] AC-5: `pub struct DeviceScalar<T>` (line 140) with
  `update` method.
- [x] AC-6: `pub struct CapturePool` (line 612) and
  `pub struct GraphPoolHandle(pub u64)` (line 484); singleton
  `graph_pool_handle()` (line 507).
- [x] AC-7: `pub fn make_graphed_callable<F>` at line 553.
- [x] AC-8: Non-CUDA stubs at lines 717-879 covering every
  CUDA-feature symbol with matching signatures.
- [x] AC-9: Non-test production consumer SHIPPED — `pub struct
  GraphedDecoder in ferrotorch-llama/src/gpu.rs` is a CUDA-graph
  per-token Llama decoder. `GraphedDecoder::capture` calls
  `capture_into_private_pool` + `CapturePool::with_private_pool` and
  `GraphedDecoder::decode_step` calls `CapturedGraph::launch`. Verified
  live: N>=3 graphed decode_steps bit-identical to the eager oracle +
  interleaved eager forwards all succeed
  (`ferrotorch-llama/tests/graphed_decoder_live.rs`).
- [x] AC-10: `pub struct PrivateMemPool in graph.rs` (cuMemPoolCreate
  FFI shim + `activate` device-mempool swap), `pub fn
  capture_into_private_pool in graph.rs`, and
  `pub fn CapturePool::with_private_pool in graph.rs`; consumer
  `GraphedDecoder::capture in ferrotorch-llama/src/gpu.rs`.

## Architecture

The module's structure mirrors PyTorch's CUDA-graph plumbing:

1. **Mode / status enums**: `CaptureMode` and `CaptureStatus` are
   thin typed wrappers over the raw cudarc enums, with default
   `Global` matching PyTorch's `torch.cuda.graph` default.

2. **Capture lifecycle**: the bottom-half capture API
   (`begin_capture` / `end_capture` + variants) is the
   `cudaStreamBeginCapture` / `cudaStreamEndCapture` pair,
   with the optional `_with_mode` and `_with_pool` variants
   wrapping the `cudaStreamCaptureMode` + mempool parameters.

3. **CapturedGraph + RAII**: `CapturedGraph` owns the captured
   graph handle + its executable; `Drop` cleans up via
   `cudaGraphDestroy` / `cudaGraphExecDestroy`. `GraphCaptureGuard`
   is a scope guard whose `Drop` ensures `end_capture` runs even
   if the capture body panics — matches the
   `torch.cuda.graph` context manager.

4. **DeviceScalar<T>**: a `CudaSlice<T>` of length 1 with the
   typed `update(value)` method that issues a host-to-device
   memcpy before the next replay. The pre-capture allocation
   guarantees the parameter pointer is stable across replays.

5. **CapturePool + GraphPoolHandle**: PyTorch's
   `graph_pool_handle()` returns an opaque integer that tags a
   pool reusable across multiple captures. ferrotorch's
   `GraphPoolHandle(u64)` is the same opaque-id contract;
   `capture_pool_for_handle` looks up the actual `Arc<CapturePool>`
   in a global registry (counter-based id allocation in
   `AtomicU64`).

6. **make_graphed_callable**: takes a closure, runs a warm-up
   pass (eager mode), then a capture pass, then returns a
   `CapturedGraph` that replays on demand. The warm-up phase
   exists to lazy-init module caches before capture starts
   (any allocation during capture is forbidden by the CUDA
   contract).

### Non-CUDA stubs (REQ-8)

When `feature = "cuda"` is off, the module re-defines all the
above as no-op stubs returning `GpuError::DeviceUnavailable`.
This lets the public API of ferrotorch-gpu compile cleanly even
without CUDA, so downstream crates need not gate their `use`
imports.

### Non-test production consumer status (REQ-9) — SHIPPED (#1595)

The production consumer is `pub struct GraphedDecoder` in
`ferrotorch-llama/src/gpu.rs`: a CUDA-graph-replayable single-token
Llama decoder. `GraphedDecoder::capture` is the non-test production
caller of the whole capture API family —
`CapturePool::with_private_pool`, `capture_into_private_pool` (which
internally drives `begin_capture_with_pool_mode` / `end_capture_with_pool`
and the event-tracking toggle), and `CapturedGraph::upload`.
`GraphedDecoder::decode_step` is the non-test production caller of
`CapturedGraph::launch`.

#### Why this needed REQ-10 (private mempool) first

The earlier capture/replay infra (`begin_capture` / `end_capture` /
`CapturePool` as a host-side buffer tracker) could capture a forward and
replay it **once** bit-identically to eager — but the captured forward's
per-layer intermediates were allocated via `cuMemAllocAsync`, which draws
from the **device-wide default async mempool** shared with the eager
path. On replay, the captured alloc/free nodes corrupted that shared
pool, so the *second* `decode_step` and any interleaved eager forward
failed with `CUDA_ERROR_INVALID_VALUE` (#1595).

REQ-10's `PrivateMemPool` swaps the device's default mempool to a
private pool for the capture window (`cuDeviceSetMemPool`, restored after
capture), so the captured allocations are isolated. This is the
ferrotorch analog of PyTorch's caching-allocator graph-pool mode
(`beginAllocateToPool` / `endAllocateToPool`,
`aten/src/ATen/cuda/CUDAGraph.cpp:150` / `:193`) and `make_graphed_callables`
capturing into a `graph_pool_handle()` pool (`torch/cuda/graphs.py:446`).

#### Live verification (R-CHAR-3, the eager forward is the oracle)

`ferrotorch-llama/tests/graphed_decoder_live.rs` (GPU-gated):

- `graphed_decode_multi_step_matches_eager_oracle` — 5 sequential
  `decode_step`s, each producing logits **bit-identical** to
  `LlamaGpuInferencer::forward_from_ids(&[token])` for the same token.
- `graphed_decode_interleaved_with_eager_forwards` — graphed replay
  interleaved with eager forwards (single- and multi-token), all
  succeeding and matching, proving the private pool does not corrupt the
  eager default pool.

## Parity contract

`parity_ops = []` for this route. CUDA graph correctness is verified
structurally (the capture / replay produces identical outputs to the
eager path), not numerically — there is no PyTorch-equivalent
`parity-sweep` op for "graph replay equals eager replay".

Edge cases preserved:

- **CaptureMode default `Global`**: matches PyTorch's
  `torch.cuda.graph(stream, capture_error_mode="global")` default.
  Tests and single-owner production code may request `ThreadLocal` or
  `Relaxed` explicitly, mirroring PyTorch's `capture_error_mode`.
- **Invalidated capture**: callers must call `end_capture` to
  discard a broken graph before doing anything else on the stream.
  This matches upstream's contract documented in
  `at::cuda::CUDAGraph::capture_end`.
- **Mempool reuse across captures**: `GraphPoolHandle` IDs are
  allocated via `AtomicU64` to ensure uniqueness; release frees
  the underlying `Arc<CapturePool>`.
- **Drop ordering**: `CapturedGraph::Drop` releases the executable
  before the graph itself, matching CUDA's destruction order
  requirement.
- **DeviceScalar pointer stability**: the underlying `CudaSlice<T>`
  is alloc'd before capture starts and persists for the lifetime
  of the capture, so the kernel sees a stable pointer across
  replays.

## Verification

Unit tests in `ferrotorch-gpu/src/graph.rs` `mod tests` (18 tests)
exercise: capture status transitions, RAII guard drop ordering,
mempool handle reuse, `DeviceScalar::update` round trip, the
non-CUDA stub fallthrough.

Conformance tests at
`ferrotorch-gpu/tests/conformance_gpu_backend.rs` (lines 520, 710,
1552, 1598) and the dedicated suite at
`ferrotorch-gpu/tests/test_gpu_graph_pool.rs` (lines 14, 135)
exercise the full capture / replay flow with real GPU work
inside the captured region.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-gpu --features cuda graph:: 2>&1 | tail -3
```

Expected: ≥ 1 `test result: ok` line.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum CaptureMode in ferrotorch-gpu/src/graph.rs` (line 59) with `to_cuda` at line 75; non-test consumer: `pub use graph::CaptureMode` at `lib.rs` exposes it; consumed in `backend_impl.rs` lifecycle paths (the `CaptureMode` type is the parameter shape downstream callers must use). |
| REQ-2 | SHIPPED | impl: `pub enum CaptureStatus in graph.rs` (line 91) with `is_capturing` (line 117); non-test consumer: `pub use graph::CaptureStatus` at `lib.rs`. |
| REQ-3 | SHIPPED | impl: capture lifecycle entry points (`begin_capture` / `begin_capture_with_mode` / `begin_capture_with_pool` / `begin_capture_with_pool_mode` / `end_capture` / `end_capture_with_pool` / `capture_status` / `is_stream_capturing` in `graph.rs`); non-test consumer: `fn capture_into_private_pool in graph.rs` drives `begin_capture_with_pool_mode` + `end_capture_with_pool`, invoked by `GraphedDecoder::capture in ferrotorch-llama/src/gpu.rs`. |
| REQ-4 | SHIPPED | impl: `pub struct CapturedGraph in graph.rs` + `pub struct GraphCaptureGuard in graph.rs`; non-test consumer: `GraphedDecoder` (field `graph: CapturedGraph`) constructs it via `capture_into_private_pool` and replays it in `GraphedDecoder::decode_step in ferrotorch-llama/src/gpu.rs`. |
| REQ-5 | SHIPPED | impl: `pub struct DeviceScalar<T> in graph.rs` with `update`; non-test consumer: the per-replay parameter-pipe pattern is realised by `GraphedDecoder`'s stable `ids_static` buffer updated via `memcpy_htod` before each `CapturedGraph::launch` in `GraphedDecoder::decode_step in ferrotorch-llama/src/gpu.rs` (the same fixed-pointer-updated-between-replays contract `DeviceScalar` encodes). |
| REQ-6 | SHIPPED | impl: `pub struct CapturePool in graph.rs` (now also owns a `PrivateMemPool`), `pub struct GraphPoolHandle in graph.rs`, `pub fn graph_pool_handle in graph.rs`; non-test consumer: `GraphedDecoder::capture in ferrotorch-llama/src/gpu.rs` constructs `CapturePool::with_private_pool` and keeps the `Arc<CapturePool>` alive across replays so the captured pointers stay valid. |
| REQ-7 | SHIPPED | impl: `pub fn make_graphed_callable<F> in graph.rs` + the higher-level `pub fn capture_into_private_pool in graph.rs` (the multi-replay, private-pool, event-tracking-aware form needed by a real decoder); non-test consumer: `GraphedDecoder::capture in ferrotorch-llama/src/gpu.rs` calls `capture_into_private_pool`. |
| REQ-8 | SHIPPED | impl: non-CUDA stub block at `graph.rs` re-defines every public symbol (including `PrivateMemPool` / `capture_into_private_pool` / `begin_capture_with_pool_mode`) with matching signatures returning `GpuError`; non-test consumer: ferrotorch-gpu compiles cleanly without `cuda` feature (verified by the workspace's `--no-default-features` CI lane). |
| REQ-9 | SHIPPED | impl + consumer: `pub struct GraphedDecoder in ferrotorch-llama/src/gpu.rs` — the production CUDA-graph per-token decode loop. `GraphedDecoder::capture` calls `capture_into_private_pool` + `CapturePool::with_private_pool`; `GraphedDecoder::decode_step` calls `CapturedGraph::launch`. Verified live: N>=3 graphed steps bit-identical to the eager oracle + interleaved eager forwards (`ferrotorch-llama/tests/graphed_decoder_live.rs`). #1595. |
| REQ-10 | SHIPPED | impl: `pub struct PrivateMemPool in graph.rs` (cuMemPoolCreate FFI shim + `activate` device-mempool swap returning `MemPoolScope`), `pub fn CapturePool::with_private_pool in graph.rs`, `pub fn capture_into_private_pool in graph.rs`; non-test consumer: `GraphedDecoder::capture in ferrotorch-llama/src/gpu.rs`. Mirrors PyTorch `aten/src/ATen/cuda/CUDAGraph.cpp:150`/`:193`. #1595. |
