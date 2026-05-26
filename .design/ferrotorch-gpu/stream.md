# CUDA stream pool + priority + events + thread-local current stream

<!--
tier: 3-component
status: draft
baseline-pytorch: 2fa9c68b1 (working tree at /home/doll/pytorch)
upstream-paths:
  - aten/src/ATen/cuda/
  - aten/src/ATen/native/cuda/
  - c10/cuda/
  - torch/cuda/
-->

## Summary

`ferrotorch-gpu/src/stream.rs` is the CUDA stream layer: a per-device
pool of non-blocking streams (round-robin allocation), a priority
pool for `(High/Normal/Low)` streams, the safe `CudaEvent` wrapper,
the thread-local "current stream per device" map, and the RAII
`StreamGuard` that scopes a stream change to a block. This is the
ferrotorch analog of `c10::cuda::CUDAStream` plus
`c10::cuda::CUDAEvent` plus `at::cuda::getCurrentCUDAStream(idx)`,
all in one file (`c10/cuda/CUDAStream.h` + `CUDAStream.cpp` + the
thread-local stream registry in `c10/cuda/impl/`).

## Requirements

- REQ-1: `StreamPriority` enum (`High`, `Normal`, `Low`, default
  `Normal`). Resolves to a CUDA priority integer via
  `to_cuda_priority((least, greatest))` where the CUDA convention is
  lower-int-is-higher-priority. `High` → greatest (numerically
  smallest); `Low` → least (numerically largest); `Normal` → midpoint
  (clamped sensibly on collapsed ranges). Mirrors
  `at::cuda::CUDAStream::Priority` in
  `aten/src/ATen/cuda/CUDAStream.cpp`.

- REQ-2: `get_stream_priority_range(ctx) -> GpuResult<(i32, i32)>` that
  wraps `cuCtxGetStreamPriorityRange`. The `unsafe` block carries a
  `SAFETY:` comment per goal.md R-CODE-1 documenting that the
  out-pointers are valid stack locals and the current context is
  bound.

- REQ-3: Layout-mirror trick for priority-stream construction.
  `cudarc 0.19`'s `CudaStream` has private fields; constructing one
  from a raw `CUstream` produced by `cuStreamCreateWithPriority` is
  done via a `CudaStreamMirror` struct with identical layout and a
  `mem::transmute`. The const `_CUDA_STREAM_LAYOUT_GUARD` block
  asserts at compile time that the size and alignment match (so any
  future cudarc layout drift fails the build rather than producing
  UB). Per R-DEV-7 the transmute is the documented Rust-side
  workaround for cudarc's missing public constructor; the SAFETY
  comment cites the const guard.

- REQ-4: `pub fn new_stream_with_priority(ctx, priority)
  -> GpuResult<Arc<CudaStream>>` calls `cuStreamCreateWithPriority`
  with `CU_STREAM_NON_BLOCKING`, the clamped priority, and wraps the
  raw stream via the layout-mirror transmute.

- REQ-5: `CudaEventWrapper` safe wrapper around `cudarc::driver::CudaEvent`.
  Provides `new(ctx)` (no timing flag → cheaper), `new_with_timing(ctx)`
  (needed for `elapsed_ms`), `record(stream)`, `synchronize()`,
  `query() -> bool`, `wait_on(stream)`, `elapsed_ms(&end) -> f32`,
  `elapsed_us(&end) -> u64`, `inner() -> &CudaEvent`. Returns
  `GpuResult<T>` for every fallible op. Mirrors PyTorch's
  `torch.cuda.Event` API surface.

- REQ-6: Per-device stream pool. `STREAMS_PER_DEVICE = 8` non-blocking
  streams per device, lazily created on first `StreamPool::get_stream`
  call, distributed round-robin via an atomic counter. `MAX_DEVICES = 64`
  guards against unbounded allocation if a caller passes a bogus
  ordinal. Mirrors upstream's stream-pool note in
  `c10/cuda/CUDAStream.h` ("32 streams per device, round-robin").

- REQ-7: Priority pool. `STREAMS_PER_PRIORITY = 4` streams per
  `(device, priority)` pair. Same lazy/round-robin pattern as the
  default pool but keyed by `(device_ordinal, StreamPriority)` in a
  `Mutex<HashMap>` because the key space is dynamic.

- REQ-8: Thread-local "current stream per device". `thread_local!
  static CURRENT_STREAMS: RefCell<HashMap<usize, Arc<CudaStream>>>`
  maps device ordinal to the per-thread "current" stream. Provides
  `get_current_stream(device)`, `set_current_stream(device, stream)`,
  `clear_current_stream(device)`, and `current_stream_or_default(device)`
  (which resolves to the thread-local OR the device's default).
  Mirrors PyTorch's `at::cuda::getCurrentCUDAStream(idx)`.

- REQ-9: `StreamGuard` RAII guard. Construct with `StreamGuard::new(device,
  stream)` to set the thread-local stream; the `Drop` impl restores
  the previous stream (or clears if none). Mirrors PyTorch's
  `c10::cuda::CUDAStreamGuard` semantics in `c10/cuda/CUDAGuard.h`.

- REQ-10: Stubs when `cuda` feature is disabled. `CudaEventWrapper`,
  `StreamPool`, `StreamGuard`, `get_current_stream`,
  `set_current_stream`, `clear_current_stream` all have host-only
  stub implementations that return `GpuError::NoCudaFeature` or are
  no-ops. Keeps downstream `use ferrotorch_gpu::StreamGuard`
  resolvable in host-only builds.

## Acceptance Criteria

- [x] AC-1: `get_stream_priority_range` returns `(least, greatest)`
  with `greatest <= least` always — verified by
  `priority_range_returns_sane_values` (line 987).
- [x] AC-2: `StreamPriority::to_cuda_priority((5, -5))` returns
  `(5, midpoint, -5)` for `Low/Normal/High` — verified by
  `stream_priority_resolves_within_range` (line 1001).
- [x] AC-3: `StreamPriority` resolves to `0` for all three variants on
  a collapsed `(0, 0)` range — verified by
  `stream_priority_collapsed_range_resolves_to_zero` (line 1013).
- [x] AC-4: `new_stream_with_priority` succeeds for all three priority
  levels and the returned streams synchronize without segfault —
  verified by `new_stream_with_priority_actually_runs_kernels` (line
  1037).
- [x] AC-5: `StreamPool::get_stream(ctx, 0)` returns a round-robin
  stream and wraps after `STREAMS_PER_DEVICE` calls — verified by
  `stream_pool_round_robin` (line 810).
- [x] AC-6: `StreamPool::get_stream(ctx, MAX_DEVICES + 1)` returns
  `GpuError::InvalidDevice` — verified by `stream_pool_invalid_device`
  (line 845).
- [x] AC-7: `StreamGuard::new(0, s)` restores the previous stream on
  drop — verified by `stream_guard_restores_previous` (line 852).
- [x] AC-8: `StreamGuard::new(0, s)` with no previous stream clears on
  drop — verified by `stream_guard_clears_when_no_previous` (line 902).
- [x] AC-9: `current_stream_or_default(&device)` returns the
  thread-local stream when set, otherwise the device's default —
  verified by `current_stream_or_default_fallback` (line 928).
- [x] AC-10: `CudaEventWrapper::record + synchronize + query` is the
  documented sequence — verified by `event_record_sync` (line 775).
- [x] AC-11: `CudaEventWrapper::wait_on(stream2)` makes `stream2`
  GPU-side-wait — verified by `event_wait_on_stream` (line 966).

## Architecture

### StreamPriority + range query (REQ-1, REQ-2)

`pub enum StreamPriority in stream.rs` at `stream.rs` with three
variants and a default of `Normal`. `to_cuda_priority(range)` at
`stream.rs` resolves the variant to the device's priority
integer (with the CUDA convention that lower-int = higher-priority).
Normal sits at the midpoint with a special case for collapsed ranges
(both ends equal) to avoid div-by-zero / signed midpoint surprises.

`pub fn get_stream_priority_range in stream.rs` at `stream.rs`
wraps `cuCtxGetStreamPriorityRange`. The `unsafe` block carries a
SAFETY comment naming the invariants (out-pointers, bound context).

### Layout-mirror transmute (REQ-3)

`struct CudaStreamMirror in stream.rs` at `stream.rs` has the
exact same field types in the same order as cudarc 0.19's
`CudaStream` (which has private fields). The compile-time guard
`const _CUDA_STREAM_LAYOUT_GUARD: () = { ... }` at `stream.rs`
asserts `size_of` and `align_of` match. Any future cudarc version
that changes the layout fails the build (clearly named).

`pub fn new_stream_with_priority in stream.rs` at `stream.rs`
uses `cuStreamCreateWithPriority` + transmute via the mirror. Two
`unsafe` blocks each carry a SAFETY comment: one for the raw FFI
call (`stream.rs`), one for the transmute
(`stream.rs`). Per R-CODE-1 these are leaf-primitive `unsafe`
blocks with explicit invariants documented.

### CudaEventWrapper (REQ-5)

`pub struct CudaEventWrapper in stream.rs` at `stream.rs` is a
thin newtype around `cudarc::driver::CudaEvent`. Methods at
`stream.rs`:
- `new(ctx)` — non-timing event (default).
- `new_with_timing(ctx)` — timing event for `elapsed_ms`.
- `record(stream)` / `synchronize()` / `query()` / `wait_on(stream)`
  — the standard event API mapped to cudarc.
- `elapsed_ms(&end) -> f32` / `elapsed_us(&end) -> u64` — timing
  queries; `_us` rounds to u64 microseconds for callers that store
  durations as integers.
- `inner() -> &CudaEvent` — escape hatch for cudarc-direct calls.

Each fallible method returns `GpuResult<T>` lifting cudarc errors
through the `From` impls on `GpuError`.

### Stream pool (REQ-6)

`pub struct StreamPool in stream.rs` at `stream.rs` is a zero-sized
namespace; the actual state is `static STREAM_POOL: OnceLock<Vec<OnceLock<DeviceStreams>>>`
at `stream.rs`. The flat array is indexed by device ordinal, so
no locking on the hot path. `DeviceStreams` (`stream.rs`) holds
the round-robin counter plus the `Vec<Arc<CudaStream>>`.

`pub fn StreamPool::get_stream(ctx, device_ordinal) in stream.rs` at
`stream.rs`. On first access to a device, lazily creates
`STREAMS_PER_DEVICE` streams; on subsequent calls, round-robins via
`fetch_add % len`. Returns `GpuError::InvalidDevice` if the ordinal
exceeds `MAX_DEVICES`.

`pub fn StreamPool::pool_size(device_ordinal) in stream.rs` at
`stream.rs` reports the current pool size (0 if not yet
initialised).

Non-test production consumer: `ferrotorch-gpu/src/backend_impl.rs`
— `crate::stream::StreamPool::pool_size(device)` is read for
observability inside the `CudaBackendImpl` trait method
`stream_count`.

### Priority pool (REQ-7)

`pub fn StreamPool::get_priority_stream(ctx, ordinal, priority) in stream.rs`
at `stream.rs`. Uses a `Mutex<HashMap<(usize, StreamPriority),
Vec<Arc<CudaStream>>>>` (`stream.rs`) keyed by `(device,
priority)`. Lazily populates the bucket on first access, then
round-robins via per-key `Arc<AtomicUsize>` counters
(`stream.rs`). Race-safe insert: re-check the bucket under
lock before overwriting (`stream.rs`).

### Thread-local current stream (REQ-8)

`thread_local! static CURRENT_STREAMS: RefCell<HashMap<usize,
Arc<CudaStream>>>` at `stream.rs`. `get_current_stream(device)`,
`set_current_stream(device, stream)`, `clear_current_stream(device)`
at `stream.rs` are the explicit setters.
`current_stream_or_default(device)` at `stream.rs` resolves
to the thread-local OR the device's default — this is the fn
`crate::device::GpuDevice::stream` calls.

Non-test production consumer:
`ferrotorch-gpu/src/device.rs` —
`crate::stream::current_stream_or_default(self)` inside
`GpuDevice::stream`. Every kernel launch goes through this
resolution.

### StreamGuard RAII (REQ-9)

`pub struct StreamGuard in stream.rs` at `stream.rs` holds the
device ordinal and the previous stream (or None). Construction at
`stream.rs` saves the previous stream and sets the new one;
the `Drop` impl at `stream.rs` restores via
`set_current_stream` or `clear_current_stream`.

Non-test production consumer: any kernel that wants to scope its
launches to a non-default stream. The boundary contract is the
crate-root `pub use stream::StreamGuard` (not currently in the
re-export set — `StreamGuard` lives under
`ferrotorch_gpu::stream::StreamGuard`); external consumers absent on
`main`. Grandfathered API surface per goal.md S5; pinned by
`stream_guard_restores_previous` / `stream_guard_clears_when_no_previous`.

### Host-only stubs (REQ-10)

`#[cfg(not(feature = "cuda"))]` impls at `stream.rs`:
- `CudaEventWrapper` becomes a unit struct.
- `StreamPool::get_stream` returns `GpuError::NoCudaFeature`;
  `pool_size` returns 0.
- `StreamGuard` becomes a unit struct.
- `get_current_stream` returns `None`; `set_current_stream` and
  `clear_current_stream` are no-ops.

Non-test production consumer: host-only build path keeps every
`use ferrotorch_gpu::StreamGuard` reachable.

## Parity contract

`parity_ops = []`. The stream pool is INFRASTRUCTURE — no parity-sweep
op verifies it directly. Correctness is enforced structurally:
- Round-robin distribution prevents one-stream serialisation in
  multi-stream workloads.
- Thread-local current-stream resolution gives every kernel launch
  the "current stream" PyTorch semantics expect.
- Priority pool gives the high/normal/low scheduling buckets a
  PyTorch user sees through `torch.cuda.Stream(priority=...)`.

Edge cases handled:
- Priority on devices without priority support: range collapses to
  `(0, 0)`; all three variants return 0; streams are functionally
  equivalent.
- WSL2 or other configurations where stream creation fails: the
  pool returns whatever it managed to create (at least 1 fallback
  via `ctx.default_stream().fork()`), or `CUDA_ERROR_OUT_OF_MEMORY`
  if even that fails (`stream.rs`).
- Mutex poison in the priority pool: silently silenced via
  `lock().unwrap_or_else(|p| p.into_inner())` — the priority pool
  proceeds with whatever state was in place.
- cudarc layout drift: `_CUDA_STREAM_LAYOUT_GUARD` const-asserts
  size+alignment at compile time; a layout change in a future
  cudarc bump fails the build with a clear diagnostic.

## Verification

Tests in `mod tests in stream.rs` (lines 765–1075):
- `event_record_sync`, `event_query_before_record`, `event_wait_on_stream`
  exercise the CudaEventWrapper API.
- `stream_pool_round_robin`, `stream_pool_invalid_device` exercise the
  default stream pool.
- `stream_guard_restores_previous`, `stream_guard_clears_when_no_previous`,
  `current_stream_or_default_fallback` exercise the thread-local
  current-stream + RAII guard.
- `priority_range_returns_sane_values`,
  `stream_priority_resolves_within_range`,
  `stream_priority_collapsed_range_resolves_to_zero`,
  `new_stream_with_priority_succeeds_for_all_three_levels`,
  `new_stream_with_priority_actually_runs_kernels`,
  `priority_pool_caches_streams_per_device_and_priority`,
  `priority_pool_invalid_device` exercise the priority pool.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-gpu --features cuda --lib stream::tests 2>&1 | tail -3
```

Expected: 14 passed on a CUDA system; tests that need a real GPU
return early via `test_ctx()` returning `None`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum StreamPriority in stream.rs` at `stream.rs` + `impl StreamPriority::to_cuda_priority` at `stream.rs`. Non-test production consumer: `StreamPool::get_priority_stream` consumes the enum at `stream.rs`; `stream::tests::new_stream_with_priority_actually_runs_kernels` pins the round-trip. |
| REQ-2 | SHIPPED | impl: `pub fn get_stream_priority_range in stream.rs` at `stream.rs`. Non-test production consumer: `new_stream_with_priority` at `stream.rs` calls `get_stream_priority_range(ctx)?` to bracket the priority integer. |
| REQ-3 | SHIPPED | impl: `struct CudaStreamMirror in stream.rs` at `stream.rs`, `const _CUDA_STREAM_LAYOUT_GUARD` at `stream.rs`. Non-test production consumer: `new_stream_with_priority` at `stream.rs` performs the transmute under the layout guard. |
| REQ-4 | SHIPPED | impl: `pub fn new_stream_with_priority in stream.rs` at `stream.rs`. Non-test production consumer: `StreamPool::get_priority_stream` at `stream.rs` calls `new_stream_with_priority(ctx, priority)` inside the lazy-init path. |
| REQ-5 | SHIPPED | impl: `pub struct CudaEventWrapper in stream.rs` at `stream.rs` + methods at `stream.rs`. Non-test production consumer: the wider observability layer in `ferrotorch-gpu/src/graph.rs` (CL-454 `replay_count` / `uploaded` machinery) uses event-recording semantics; the API surface is exercised by `event_record_sync`. |
| REQ-6 | SHIPPED | impl: `pub struct StreamPool in stream.rs` at `stream.rs` + `pub fn get_stream` at `stream.rs` + `pub fn pool_size` at `stream.rs`. Non-test production consumer: `ferrotorch-gpu/src/backend_impl.rs` calls `crate::stream::StreamPool::pool_size(device)` in the `CudaBackendImpl` trait impl. |
| REQ-7 | SHIPPED | impl: `pub fn StreamPool::get_priority_stream in stream.rs` at `stream.rs` + `priority_pool_size` at `stream.rs`. Non-test production consumer: the API surface is exposed via `StreamPool::get_priority_stream`; pinned by `priority_pool_caches_streams_per_device_and_priority`. |
| REQ-8 | SHIPPED | impl: `thread_local! CURRENT_STREAMS` at `stream.rs` + `get_current_stream / set_current_stream / clear_current_stream / current_stream_or_default` at `stream.rs`. Non-test production consumer: `ferrotorch-gpu/src/device.rs` calls `crate::stream::current_stream_or_default(self)` in `GpuDevice::stream`. |
| REQ-9 | SHIPPED | impl: `pub struct StreamGuard in stream.rs` at `stream.rs` + `impl StreamGuard::new + impl Drop` at `stream.rs`. Non-test production consumer: the API surface; pinned by `stream_guard_restores_previous` (line 852) and `stream_guard_clears_when_no_previous` (line 902). Grandfathered per goal.md S5. |
| REQ-10 | SHIPPED | impl: `#[cfg(not(feature = "cuda"))]` stubs at `stream.rs`. Non-test production consumer: `cargo build -p ferrotorch-gpu --no-default-features` succeeds; the stubs keep the types resolvable in host-only builds. |
