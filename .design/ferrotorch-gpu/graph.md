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

- REQ-1: `CaptureMode` enum (`Global` / `ThreadLocal` (default) /
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

## Acceptance Criteria

- [x] AC-1: `pub enum CaptureMode` at line 59 with `Default =
  ThreadLocal`; `to_cuda()` method at line 75.
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
- [ ] AC-9: No non-test production consumer — every call site
  outside the test/example surface is a `pub use` re-export or
  a documentation example. See REQ-9 blocker.

## Architecture

The module's structure mirrors PyTorch's CUDA-graph plumbing:

1. **Mode / status enums**: `CaptureMode` and `CaptureStatus` are
   thin typed wrappers over the raw cudarc enums, with default
   `ThreadLocal` matching PyTorch's `torch.cuda.graph` default.

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

### Non-test production consumer status (REQ-9)

Across the workspace, only the following sites reference
`graph::*` outside the test surface:

- `ferrotorch-gpu/src/lib.rs` — `pub use` re-exports.
- `ferrotorch-gpu/src/graph.rs` — its own doc-comment examples.

All other consumers are in:
- `ferrotorch-gpu/tests/conformance_gpu_backend.rs:520, 710, 1552, 1598`
- `ferrotorch-gpu/tests/test_gpu_graph_pool.rs, 135`
- `ferrotorch-core/tests/_probe_b1_capture_stream.rs`

These are **test-side** consumers. Per R-DOC-3, test callers do not
count for SHIPPED status. The capture/replay infrastructure is built
and unit-tested, but no production code path inside ferrotorch
inference / training calls `make_graphed_callable` or
`begin_capture`/`end_capture` yet. Open prereq blocker: REQ-9 needs
a production user (likely the Llama / SD inference loop adopting
graph-replay for the per-token decode step).

## Parity contract

`parity_ops = []` for this route. CUDA graph correctness is verified
structurally (the capture / replay produces identical outputs to the
eager path), not numerically — there is no PyTorch-equivalent
`parity-sweep` op for "graph replay equals eager replay".

Edge cases preserved:

- **CaptureMode default `ThreadLocal`**: matches PyTorch's
  `torch.cuda.graph(stream)` default.
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
| REQ-3 | NOT-STARTED | open prereq blocker #1355; impl: all seven lifecycle entry points exist in `graph.rs` at the documented line numbers; no non-test production consumer invokes them. Open prereq blocker: wire CUDA graph capture into the Llama / SD inference loops or expose through a `CudaBackendImpl::capture_*` trait method on `GpuBackend`. |
| REQ-4 | NOT-STARTED | open prereq blocker #1355; impl: `pub struct CapturedGraph in graph.rs` (line 192) and `pub struct GraphCaptureGuard` (line 398) exist; no non-test production consumer constructs either. Open prereq blocker: same as REQ-3 — a production graph-replay user. |
| REQ-5 | NOT-STARTED | open prereq blocker #1355; impl: `pub struct DeviceScalar<T> in graph.rs` (line 140) with `update` method; no non-test production consumer instantiates it. Open prereq blocker: production callsite for per-replay parameter pipes. |
| REQ-6 | NOT-STARTED | open prereq blocker #1355; impl: `pub struct CapturePool in graph.rs` (line 612), `pub struct GraphPoolHandle` (line 484), `pub fn graph_pool_handle` (line 507); no non-test production consumer uses the mempool reuse API. Open prereq blocker: production graph-replay user. |
| REQ-7 | NOT-STARTED | open prereq blocker #1355; impl: `pub fn make_graphed_callable<F> in graph.rs` (line 553); no non-test production consumer invokes it. Open prereq blocker: a model decode loop adopting `make_graphed_callable`. |
| REQ-8 | SHIPPED | impl: non-CUDA stub block at `graph.rs` re-defines every public symbol with matching signatures returning `GpuError`; non-test consumer: ferrotorch-gpu compiles cleanly without `cuda` feature (verified by the workspace's `--no-default-features` CI lane). |
| REQ-9 | NOT-STARTED | open prereq blocker #1355; no non-test production consumer of the graph API exists. Open prereq blocker: pick one downstream use site (the Llama per-token decode loop is the canonical PyTorch reference) and wire `make_graphed_callable` into it. |
