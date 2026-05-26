# ferrotorch-profiler — CUDA event-based GPU kernel timing

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/profiler/profiler.py
  - torch/autograd/profiler.py
-->

## Summary

`ferrotorch-profiler/src/cuda_timing.rs` (gated `#[cfg(feature = "cuda")]`)
implements GPU kernel timing via CUDA driver events. `CudaKernelScope`
records a start event on the kernel's stream, the user runs the
kernel(s), then `stop()` records the end event and queues the pair
on the profiler. The expensive `cuEventElapsedTime` query is deferred
until `Profiler::flush_cuda_kernels` so the hot dispatch path adds
only ~1 us. Mirrors PyTorch's CUDA-event timing path in
`torch.autograd.profiler` (`use_cuda=True` / `use_device='cuda'`,
`torch/autograd/profiler.py:126-131`), avoiding the CUPTI dependency
upstream uses in its newer Kineto path.

## Requirements

- REQ-1: `pub struct CudaKernelScope` carries the operation `name`,
  `category`, the start `CudaEvent` (`Arc<CudaEvent>` from `cudarc`),
  and the `CudaStream` the events were recorded on. Constructed via
  `CudaKernelScope::new(ctx, stream, name, category)` which records
  the start event synchronously and returns the scope ready to be
  paired with `stop`.
- REQ-2: `pub fn new(ctx, stream, name, category) -> Result<Self, DriverError>`
  creates timing-enabled events (no `CU_EVENT_DISABLE_TIMING` flag)
  via `ctx.new_event(Some(CU_EVENT_DEFAULT))`, records the start on
  `stream`, returns `Ok(scope)`. Propagates `cudarc::driver::DriverError`
  on event creation or record failure.
- REQ-3: `pub fn stop(self, profiler: &Profiler) -> Result<(), DriverError>`
  records the end event on the same stream (consuming `self` so the
  scope cannot be `stop`ped twice), then pushes a `PendingCudaScope`
  onto the profiler's queue via the `pub(crate) push_pending_cuda_scope`
  hand-off. Does NOT call `cuEventSynchronize` — synchronisation is
  deferred to `flush_cuda_kernels`.
- REQ-4: `pub(crate) struct PendingCudaScope` is the queued pair
  (name, category, start, end) held inside `Profiler::pending_cuda`.
  Visibility is `pub(crate)` because no external caller has any way
  to construct one — the only producer is `CudaKernelScope::stop`.
  Verified by absence from the `lib.rs:38-43` re-export block.
- REQ-5: `pub(crate) fn PendingCudaScope::finalize(self, profiler_epoch_us) -> ProfileEvent`
  synchronises the end event (blocks calling CPU thread), queries
  `cuEventElapsedTime` via `cudarc::driver::CudaEvent::elapsed_ms`,
  converts milliseconds to microseconds (`(ms * 1000.0).round() as u64`),
  and constructs a `ProfileEvent` with `device_type = Cuda` and
  `duration_us` from the GPU timing. On synchronisation or elapsed
  failure, appends `" [timing_error]"` to the event name and returns
  `duration_us = 0` — silent zero is forbidden per
  rust-gpu-discipline §3, so the failure is visible without losing
  the op.
- REQ-6: `unsafe_code` is denied at the crate root and no `unsafe`
  blocks appear in this file. The `cudarc` wrappers
  (`CudaEvent::record`, `CudaEvent::synchronize`, `CudaEvent::elapsed_ms`)
  encapsulate the necessary FFI in their own crate; ferrotorch only
  calls their safe API. This is the R-DEV-7 deviation: use the Rust
  ecosystem analog (`cudarc`) over raw CUDA driver FFI.
- REQ-7: Module-level `#![cfg(feature = "cuda")]` at the top of the
  file so the entire compilation unit is skipped when the cuda
  feature is off. `lib.rs:27-28` already gates `pub mod cuda_timing`
  similarly; the inner gate is belt-and-braces.

## Acceptance Criteria

- [x] AC-1: `CudaKernelScope::new` returns `Result<Self, DriverError>`
  matching `cudarc`'s error type.
- [x] AC-2: `stop` consumes `self` (one-shot) so a double-stop is
  a compile error.
- [x] AC-3: `PendingCudaScope` is `pub(crate)`, never `pub`.
- [x] AC-4: `finalize` returns a `ProfileEvent` with
  `device_type = DeviceType::Cuda`.
- [x] AC-5: On timing failure, the event `name` carries the
  `" [timing_error]"` suffix.
- [x] AC-6: The whole module compiles only when
  `--features cuda` is passed.

## Architecture

### Async-first timing (REQ-1, REQ-2, REQ-3)

The motivation block at the top of `cuda_timing.rs:1-37` explains why
CUDA events beat CPU wall-clock for GPU kernels: an asynchronous
kernel launch returns immediately to the CPU, so `Instant::now`
captures only the dispatch latency, not the GPU work. CUDA events
are recorded on the same stream as the kernel and observe
"when the GPU reaches this point," giving true GPU-side timing.

`CudaKernelScope::new` (line 77) creates the start event via
`cudarc::driver::CudaContext::new_event(Some(CU_EVENT_DEFAULT))`.
The `CU_EVENT_DEFAULT` flag is critical — it means "timing enabled"
(the default), as opposed to `CU_EVENT_DISABLE_TIMING` which is the
fast-path for synchronisation-only events that doesn't accumulate
timing info. `cuEventRecord` on the stream is the actual
record point.

`CudaKernelScope::stop` (line 107) is the counterpart: create the
end event with the same flags, record it on the same stream,
queue the `(name, category, start, end)` tuple onto the profiler.
By consuming `self` (`stop(self, ...)` not `stop(&mut self, ...)`)
the type system prevents double-stop bugs.

### Deferred synchronisation (REQ-5)

Synchronisation is expensive — `cuEventSynchronize` blocks the CPU
until the GPU reaches the event, serialising the dispatch pipeline.
Doing this inline in `stop` would defeat the point of async
kernels. So `stop` queues the pair without syncing.

`flush_cuda_kernels` (in `profiler.rs:280`) drains the queue at
report-export time: it walks every `PendingCudaScope`, calls
`finalize(epoch_us)` on each, and pushes the resulting
`ProfileEvent` into the profiler's main event list. By that point
the GPU has almost certainly finished the work, so the sync is
cheap.

`PendingCudaScope::finalize` (line 148) is the conversion:
1. `self.end.synchronize()` — block until GPU reaches the end event.
2. `self.start.elapsed_ms(&self.end)` — get elapsed time in ms.
3. Convert ms to us via `(ms * 1000.0).round() as u64`.
4. Build a `ProfileEvent` with the GPU-measured duration.

On failure (synchronise fails, elapsed query fails) the duration
becomes 0 and the name is suffixed `" [timing_error]"` so callers
can distinguish a genuine sub-microsecond kernel from a failed
query. This is the documented contract from rust-gpu-discipline §3:
silent zero is forbidden, errors must be visible.

### Private queue hand-off (REQ-4)

`PendingCudaScope` is `pub(crate)` because exactly one producer
(`CudaKernelScope::stop`) and one consumer (`Profiler::flush_cuda_kernels`)
exist, both in this crate. Making it public would let user code
construct nonsensical pairs (e.g. events from different streams).
The verification comment at line 130-131 ("Visibility is `pub(crate)`
because no caller outside this crate has any way to construct one")
documents the invariant.

The hand-off goes through `Profiler::push_pending_cuda_scope`
(`profiler.rs:259`), also `pub(crate)`. The signature
`pub(crate) fn push_pending_cuda_scope(&self, scope: PendingCudaScope)`
is one of the two reasons that method is `pub(crate)` (the type
itself being the other) — see the comment block at
`profiler.rs:255-258`.

### `cudarc` deviation (REQ-6)

ferrotorch deviates from PyTorch's raw CUDA driver FFI by using
`cudarc`'s typed wrappers. R-DEV-7 (Rust ecosystem analog is
materially better) applies: `cudarc::driver::CudaEvent` already
encapsulates the `unsafe` `cuEventCreate` / `cuEventRecord` /
`cuEventSynchronize` / `cuEventElapsedTime` calls in its own
crate, with proper `Drop`-based cleanup. Re-implementing the FFI
in ferrotorch would duplicate that work and add `unsafe` blocks
this crate's `deny(unsafe_code)` would have to allow.

### Non-test production consumers

- `ferrotorch-profiler/src/profiler.rs:61` declares the
  `pending_cuda: Mutex<Vec<PendingCudaScope>>` field that
  `CudaKernelScope::stop` populates.
- `ferrotorch-profiler/src/profiler.rs:259` `pub(crate) fn push_pending_cuda_scope`
  is the entry point `CudaKernelScope::stop` calls at line 112.
- `ferrotorch-profiler/src/profiler.rs:280` `pub fn flush_cuda_kernels`
  drains the queue and calls `PendingCudaScope::finalize` at line 293.
- `ferrotorch-profiler/src/lib.rs:39` `pub use cuda_timing::CudaKernelScope;`
  re-exports the user-facing scope type.
- `ferrotorch/src/lib.rs:107` `pub use ferrotorch_profiler::*;`
  propagates `CudaKernelScope` to the meta-crate prelude (under
  the cuda feature gate).

## Parity contract

`parity_ops = []`. Behavioral parity contract:

- **GPU event timing semantics**: ferrotorch uses
  `cuEventRecord` + `cuEventElapsedTime`, which is the same
  primitive PyTorch uses in its `use_device='cuda'` path
  (`torch/autograd/profiler.py:126-131`). Resolution matches:
  CUDA driver reports microsecond-accurate timings.
- **Stream affinity**: ferrotorch records both events on the
  stream passed to `CudaKernelScope::new`; PyTorch does the same.
  Recording start on stream A and end on stream B would give
  undefined behaviour in both runtimes.
- **Failed synchronise**: ferrotorch returns
  `duration_us = 0` with `" [timing_error]"` suffix. PyTorch
  logs a warning and continues with the failed event omitted.
  Different visibility, same recovery (the failed timing does
  not propagate as an exception or panic).
- **Sub-microsecond kernel**: ferrotorch reports `duration_us = 0`
  (without the timing_error suffix) — a genuine but un-measurable
  kernel. PyTorch reports the actual nanosecond-resolution value.
  Different precision, same correctness for aggregate analysis.

## Verification

No `mod tests` in `cuda_timing.rs` itself — the cuda feature is
not exercised in lib-tests on CPU-only build machines.
The crate-level `tests/cuda_timing_test.rs` integration file
exercises `CudaKernelScope::new`, `stop`, and
`Profiler::flush_cuda_kernels` end-to-end on GPU-equipped
hosts (skipped via `#[cfg(feature = "cuda")]` otherwise).

Smoke (CPU-only host — verifies compilation):

```bash
cargo check -p ferrotorch-profiler --features cuda 2>&1 | tail -3
```

Smoke (GPU host):

```bash
cargo test -p ferrotorch-profiler --features cuda --test cuda_timing_test 2>&1 | tail -3
```

Expected: `cargo check` clean on the CPU host;
`cargo test` `X passed; 0 failed` on the GPU host.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct CudaKernelScope` at `ferrotorch-profiler/src/cuda_timing.rs:59` carrying name/category/start/stream, mirroring PyTorch's CUDA-event timing in `torch/autograd/profiler.py:126-131`; non-test consumer: `ferrotorch-profiler/src/lib.rs:39` re-exports it under the cuda feature; `ferrotorch/src/lib.rs:107` propagates to the meta-crate prelude — user code on a CUDA host constructs scopes via this re-export. |
| REQ-2 | SHIPPED | impl: `pub fn new` at `ferrotorch-profiler/src/cuda_timing.rs:77` calling `ctx.new_event(Some(CU_EVENT_DEFAULT))` + `start.record(stream)`; non-test consumer: re-exported via `CudaKernelScope`, the surface-coverage contract pins it (note: under cuda feature only). |
| REQ-3 | SHIPPED | impl: `pub fn stop(self, profiler) -> Result<(), DriverError>` at `ferrotorch-profiler/src/cuda_timing.rs:107` consuming `self`, recording the end event, and calling `profiler.push_pending_cuda_scope(...)` at line 112; non-test consumer: `ferrotorch-profiler/src/profiler.rs:259` `pub(crate) fn push_pending_cuda_scope` is the matching profiler entry point — `CudaKernelScope::stop` is the only production caller. |
| REQ-4 | SHIPPED | impl: `pub(crate) struct PendingCudaScope` at `ferrotorch-profiler/src/cuda_timing.rs:132` with `pub(crate)` fields, intentionally absent from the `lib.rs:38-43` re-export block per the comment at line 130-131; non-test consumer: `ferrotorch-profiler/src/profiler.rs:61` `pending_cuda: Mutex<Vec<PendingCudaScope>>` field stores them; `profiler.rs:259` accepts them via `push_pending_cuda_scope`; `profiler.rs:293` calls `scope.finalize(epoch_us)` to drain. |
| REQ-5 | SHIPPED | impl: `pub(crate) fn finalize(self, profiler_epoch_us: u64) -> ProfileEvent` at `ferrotorch-profiler/src/cuda_timing.rs:148` with the synchronise + `elapsed_ms` + `" [timing_error]"` failure path; non-test consumer: `ferrotorch-profiler/src/profiler.rs:293` `let event = scope.finalize(epoch_us);` inside `flush_cuda_kernels` — the only call site. |
| REQ-6 | SHIPPED | impl: `ferrotorch-profiler/src/lib.rs:3` `#![deny(unsafe_code)]` enforces the no-unsafe rule; `cuda_timing.rs` contains zero `unsafe` blocks (verified by `grep -n unsafe ferrotorch-profiler/src/cuda_timing.rs` returning empty); the `cudarc` calls (`new_event`, `record`, `synchronize`, `elapsed_ms`) encapsulate the FFI; non-test consumer: the deny-attribute is workspace-enforced via `cargo clippy --lib -- -D warnings` and the crate compiles clean. |
| REQ-7 | SHIPPED | impl: `#![cfg(feature = "cuda")]` at `ferrotorch-profiler/src/cuda_timing.rs:39` skipping the module when the feature is off; non-test consumer: `ferrotorch-profiler/src/lib.rs:27-28` also gates `pub mod cuda_timing` on the same feature, so the no-cuda dependency graph compiles without `cudarc`. Verified by `cargo check -p ferrotorch-profiler` (no features) succeeding without a `cudarc` linker error. |
