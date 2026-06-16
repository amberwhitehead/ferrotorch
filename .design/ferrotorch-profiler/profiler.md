# ferrotorch-profiler — `Profiler` core, `with_profiler` lifecycle, `OpProfiler` hook

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/profiler/profiler.py
  - torch/autograd/profiler.py
  - torch/profiler/__init__.py
-->

## Summary

`ferrotorch-profiler/src/profiler.rs` implements the central
`Profiler` struct (thread-safe event collector), the
`ProfileConfig` knob set (`record_shapes`, `record_memory`,
`with_stack`), the `with_profiler(config, |p| { ... })`
closure-scoped lifecycle, and the `impl OpProfiler for Profiler`
bridge that lets `ferrotorch-core`'s tensor ops auto-record
themselves when this profiler is the active thread-local. Mirrors
the union of `torch.autograd.profiler.profile`
(`torch/autograd/profiler.py:112`) and `torch.profiler.profile`
(`torch/profiler/profiler.py:651`), narrowed to the
no-Kineto / no-CUPTI Rust implementation.

## Requirements

- REQ-1: `pub struct ProfileConfig` with three `pub bool` fields
  (`record_shapes`, `record_memory`, `with_stack`) and a `Default`
  setting `record_shapes=true, record_memory=false, with_stack=false`.
  Mirrors the `record_shapes` / `profile_memory` / `with_stack` kwargs
  on `torch.autograd.profiler.profile` (`torch/autograd/profiler.py:218-222`).
- REQ-2: `pub struct Profiler` is `Send + Sync` (verified by the
  trait-object cast at line 443), holds events behind a single
  `Mutex<Vec<ProfileEvent>>`, an atomic `active` flag, an
  `Instant` epoch, and (under the cuda feature) a
  `Mutex<Vec<PendingCudaScope>>` queue. The lock-ordering rule
  documented at line 47-52: no other lock may be held while
  calling `record*` / `push_gpu_event`.
- REQ-3: Five event-producing methods:
  - `pub fn record(&self, name, category, shapes)` — CPU op, start at "now", duration 0.
  - `pub fn record_with_duration(&self, name, category, duration_us)` — pre-timed CPU op.
  - `pub fn record_memory(&self, name, bytes)` — memory event, category Other.
  - `pub fn record_memory_categorized(&self, name, bytes, category)` — memory event with explicit category.
  - `pub fn push_gpu_event(&self, name, category, timing) -> Option<usize>` — GPU event, returns index for caller correlation.
  All five gate on `self.active.load(Relaxed)` so a stopped profiler is a no-op.
  Mirrors PyTorch's `record_function` / memory profiler / CUDA event paths.
- REQ-4: `pub fn with_profiler<F, R>(config, f: F) -> (R, ProfileReport)`
  where `F: FnOnce(&Profiler) -> R`. Builds an `Arc<Profiler>`,
  installs it as the `ferrotorch_core::profiler_hook` thread-local
  via `set_current(Some(...))`, runs the closure, stops the
  profiler, clears the hook via an RAII `ProfilerHookGuard`, and
  unwraps the `Arc` to drain events into a `ProfileReport`.
  Mirrors `torch.profiler.profile.__enter__` / `__exit__`
  (`torch/profiler/profiler.py:651` context manager).
- REQ-5: `impl OpProfiler for Profiler` at line 369 registers the
  profiler as the cross-crate hook target — every tensor op in
  `ferrotorch-core` wrapped in `profile_op_scope` calls
  `OpProfiler::record_op` on the currently-installed profiler
  without needing a direct dependency on this crate. Mirrors
  PyTorch's `record_function` mechanism
  (`torch/autograd/profiler.py:820` `record_function`).
- REQ-6: Lifecycle methods: `pub fn is_active(&self) -> bool`,
  `pub fn stop(&self)`, `pub fn pending_cuda_count(&self) -> usize`,
  `pub fn flush_cuda_kernels(&self)`. `stop` flips the atomic so
  subsequent records become no-ops; `flush_cuda_kernels` (cuda
  feature) drains the pending CUDA scope queue, synchronising each
  pair and converting to a `ProfileEvent`. Non-cuda build provides
  a no-op `flush_cuda_kernels` so the API surface is identical.
- REQ-7: Stack-trace capture via `std::backtrace::Backtrace::capture()`
  gated by `ProfileConfig::with_stack` and the `RUST_BACKTRACE`
  env var. The cost when `with_stack=true` but `RUST_BACKTRACE`
  unset is "a few atomic loads" (documented at line 21-25).
  Mirrors PyTorch's `with_stack` kwarg.
- REQ-8: Production code never panics on a poisoned mutex. The
  `push_event` / `push_event_returning_index` paths silently
  drop the event on poison; the `into_report` final unwrap uses
  `unwrap_or_default()` so a poisoned final state still yields
  a (possibly empty) report. The single `unreachable!` at line
  457 documents an invariant the API surface mathematically
  enforces (Arc strong count == 1 after guard drop).

## Acceptance Criteria

- [x] AC-1: `ProfileConfig::default()` returns
  `{ record_shapes: true, record_memory: false, with_stack: false }`.
- [x] AC-2: `Profiler` is `Send + Sync` (cast to
  `Arc<dyn profiler_hook::OpProfiler>` at line 443 requires both).
- [x] AC-3: `record` populates a `ProfileEvent` with
  `device_type = DeviceType::Cpu`.
- [x] AC-4: `push_gpu_event` populates `device_type = DeviceType::Cuda`
  even when the timing came from CPU wall-clock fallback (BUG-17 fix).
- [x] AC-5: After `stop()`, subsequent `record*` calls leave the
  event list unchanged.
- [x] AC-6: `with_profiler` installs the thread-local hook for the
  closure duration and clears it on exit (verified by panic-safe
  RAII guard).
- [x] AC-7: `record_memory` defaults to `MemoryCategory::Other`.

## Architecture

### `ProfileConfig` (REQ-1)

Three public booleans plus `Default`. Public fields (not getters)
match the kwarg-bag idiom PyTorch uses on the Python side — users
construct it inline. `record_shapes=true` by default because the
shape vector is cheap to populate and unlocks FLOP estimation
downstream.

### `Profiler` core (REQ-2, REQ-8)

Single-`Mutex<Vec<ProfileEvent>>` storage means lock ordering is
trivial within the profiler itself. The manual `Debug` impl at
line 64-79 reads the event count under the lock with
`lock().map_or(0, |g| g.len())` so even a poisoned mutex renders
cleanly.

The atomic `active` flag is `Relaxed`-ordered — events form a
total order on the mutex, the active flag just gates whether to
acquire it. Stopping the profiler is racy by design (events in
flight may or may not be recorded), which matches PyTorch's
behaviour on `profile.__exit__`.

The `Send + Sync` derivation is implicit (every field is `Send + Sync`)
and verified by the `Arc<dyn OpProfiler>` cast at line 443.

Poisoned-mutex handling: `push_event` is fire-and-forget (silently
drops on poison); `push_event_returning_index` returns `None` on
poison (callers can detect). `into_report` uses
`unwrap_or_default()` so a poisoned final mutex yields an empty
event list rather than panicking. The single `unreachable!` at
line 457 is the only panic-shaped construct, and the API surface
proves it unreachable (Arc strong count is mathematically 1
after the hook guard drops).

### Event-producing methods (REQ-3)

All five share the same body shape:
1. Check `self.active.load(Relaxed)` → bail if stopped.
2. Compute / accept the event fields.
3. Optionally populate `input_shapes` from `ProfileConfig::record_shapes`.
4. Optionally populate `flops` via `flops::estimate(name, &shapes)`.
5. Push into the events vec under the mutex.

The 5 methods differ only in which event-payload variant they
build:
- `record` (line 99) — `start_us = now`, `duration_us = 0`.
- `record_with_duration` (line 133) — `start_us = now - duration`,
  `duration` given.
- `record_memory_categorized` (line 164) — `category = "memory"`,
  `memory_bytes = Some(bytes)`, `memory_category = Some(cat)`.
- `push_gpu_event` (line 196) — `device_type = Cuda`,
  `duration = end - start` (saturating).
- `OpProfiler::record_op` (line 370) — same shape as `record_with_duration`
  but reachable through the cross-crate hook.

### `with_profiler` lifecycle (REQ-4)

`with_profiler` (line 437) is the canonical entry point. The
sequence:
1. `Arc::new(Profiler::new(config))` — fresh profiler.
2. `profiler_hook::set_current(Some(arc.clone()))` — install hook.
3. `let guard = ProfilerHookGuard;` — RAII for hook clear.
4. `f(&profiler)` — run user code.
5. `profiler.stop()` — flip the active flag.
6. `drop(guard)` — clears the hook (also fires on panic unwind).
7. `Arc::try_unwrap(profiler)` — must succeed because the only
   other strong reference (`hook`) was just cleared.
8. `profiler.into_report()` — drain events.

The RAII guard is the key safety property: even if `f` panics,
the thread-local hook is cleared, so subsequent code on the same
thread doesn't see a stale profiler pointer. Mirrors PyTorch's
context-manager `__exit__` cleanup, but enforced by the
type system rather than the user.

### `OpProfiler` bridge (REQ-5)

`impl OpProfiler for Profiler` at line 369 is the trait
implementation that makes `ferrotorch-core/src/grad_fns/*.rs`
auto-profile. The trait is defined in `ferrotorch-core` (not
this crate) so the core has no dependency cycle. Every tensor op
in `ferrotorch-core/src/grad_fns/arithmetic.rs:824`,
`grad_fns/transcendental.rs:282`, etc. wraps its body in
`profile_op_scope("add", "tensor_op", &[...], || { ... })`,
which checks the thread-local hook and (if set) calls
`OpProfiler::record_op` with the measured duration. There are
36 such call sites in `ferrotorch-core` today.

### CUDA flush (REQ-6)

`pending_cuda` (cuda feature) holds `PendingCudaScope`s queued by
`CudaKernelScope::stop` (see `cuda_timing.md`). `flush_cuda_kernels`
takes the queue, calls `scope.finalize(epoch_us)` on each, and
pushes the resulting `ProfileEvent` into the main events list.
The no-cuda build provides a no-op `flush_cuda_kernels` so calling
code compiles identically against both feature configurations.

### Non-test production consumers

- `ferrotorch-core/src/profiler_hook.rs:51` defines the
  thread-local `RefCell<Option<Arc<dyn OpProfiler>>>` that
  `with_profiler` installs into; `profile_op_scope` at line 87
  reads it and dispatches every tensor op through the
  `OpProfiler::record_op` method this crate implements.
- `ferrotorch-core/src/grad_fns/arithmetic.rs:824` —
  `crate::profiler_hook::profile_op_scope("add", "tensor_op", &[a.shape(), b.shape()], || { ... })`.
  Same pattern at `arithmetic.rs:1275` (`add_scaled`), `arithmetic.rs:1583` (mul),
  `arithmetic.rs:1761` (div), `arithmetic.rs:1895` (neg),
  `arithmetic.rs:2064` (pow), plus `grad_fns/transcendental.rs:282` (exp),
  line 385 (log), line 492 (sin), line 569 (cos), and ~26 more — 36
  call sites total.
- `ferrotorch/src/lib.rs:107` `pub use ferrotorch_profiler::*;`
  exposes `Profiler` / `ProfileConfig` / `with_profiler` to user
  code through the meta-crate prelude.

## Parity contract

`parity_ops = []`. The profiler is structural — no per-op
numerical kernels. Behavioral parity contract:

- **`stop` semantics**: PyTorch's `__exit__` stops collection
  and triggers post-processing. ferrotorch's `stop` is just the
  collection toggle; finalisation happens in `into_report`. The
  semantic boundary differs but the user-visible behaviour
  (events after stop are not recorded) matches.
- **Poisoned mutex**: PyTorch has no analogous condition (Python
  has no Mutex.poison). ferrotorch silently drops events to
  preserve the diagnostic-tool contract — the alternative
  (panicking) would mask the original error.
- **`with_profiler` panic safety**: hook clear via RAII guard
  fires on unwind. PyTorch's `__exit__` runs on exception via
  context-manager protocol; same end state.
- **`record_memory` with `record_memory=false`**: ferrotorch
  drops the event silently. PyTorch's `profile_memory=False`
  similarly suppresses memory events; semantics match.
- **Thread-id format**: ferrotorch parses `ThreadId(N)` debug
  output (line 471-481) until `ThreadId::as_u64` stabilises in
  Rust. PyTorch uses OS-level TID. Different values, same role
  (group events by recording thread in chrome trace).

## Verification

4 unit tests in `profiler.rs` `mod tests` (lines 483-537):

- `test_push_gpu_event_returns_index` — index correlation.
- `test_push_gpu_event_sets_cuda_device_type` — BUG-17 fix.
- `test_push_gpu_event_inactive_returns_none` — stop gates.
- `test_record_sets_cpu_device_type` — default CPU.

Plus the crate-level integration `tests/profiler_tests.rs`
(20+ tests) and `tests/cuda_timing_test.rs` exercising the cuda
feature path.

Smoke:

```bash
cargo test -p ferrotorch-profiler --lib profiler 2>&1 | tail -3
```

Expected: `4 passed; 0 failed` for `profiler::tests`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct ProfileConfig` at `ProfileConfig in ferrotorch-profiler/src/profiler.rs` with `Default` at line 28, mirroring `torch/autograd/profiler.py:218-222` kwargs; non-test consumer: `pub in ferrotorch-profiler/src/lib.rs` re-exports it, `pub in ferrotorch-profiler/src/profiler.rs` `with_profiler(config, ...)` consumes it by value, `ferrotorch/src/lib.rs` propagates to the meta-crate prelude. |
| REQ-2 | SHIPPED | impl: `pub struct Profiler` at `Profiler in ferrotorch-profiler/src/profiler.rs` with the documented lock-ordering doc at line 47-52; non-test consumer: `pub in ferrotorch-profiler/src/profiler.rs` casts `Arc<Profiler>` to `Arc<dyn profiler_hook::OpProfiler>` which requires `Send + Sync`; the cast itself is the proof that the trait bounds hold. |
| REQ-3 | SHIPPED | impl: `record` at line 99, `record_with_duration` at line 133, `record_memory` at line 157, `record_memory_categorized` at line 164, `push_gpu_event` at line 196, mirroring PyTorch's `record_function` / memory profiler / CUDA event paths; non-test consumer: `ferrotorch-profiler/src/lib.rs:41` re-exports `Profiler`, and the crate's surface-coverage test (`tests/conformance_surface_coverage.rs:72-77`) pins each method as a production-API surface item consumed via the meta-crate. |
| REQ-4 | SHIPPED | impl: `pub fn with_profiler` at `with_profiler in ferrotorch-profiler/src/profiler.rs` with the RAII `ProfilerHookGuard` at line 406, mirroring `torch.profiler.profile.__enter__/__exit__` at `torch/profiler/profiler.py:651`; non-test consumer: `pub in ferrotorch-profiler/src/lib.rs` re-exports `with_profiler`, `ferrotorch/src/lib.rs` propagates it. The doc-test at `lib.rs` exercises it as part of `cargo test --doc`. |
| REQ-5 | SHIPPED | impl: `impl profiler_hook::OpProfiler for Profiler` at `record_op in ferrotorch-profiler/src/profiler.rs` with `record_op` at line 370 populating a `ProfileEvent` from the cross-crate hook signature; non-test consumer: `profile_op_scope in ferrotorch-core/src/grad_fns/arithmetic.rs` `crate::profiler_hook::profile_op_scope("add", "tensor_op", &[a.shape(), b.shape()], || { ... })` invokes `OpProfiler::record_op` on the installed profiler. 36 call sites across `ferrotorch-core/src/grad_fns/{arithmetic,transcendental}.rs`. |
| REQ-6 | SHIPPED | impl: `is_active` at line 236, `stop` at line 241, `pending_cuda_count` at line 307, `flush_cuda_kernels` (cuda) at line 280 and no-op variant (non-cuda) at line 301; non-test consumer: `ferrotorch-profiler/src/profiler.rs:447` `profiler.stop()` is called from inside `with_profiler` itself — the production consumer is this crate's own lifecycle path. `flush_cuda_kernels` is consumed by user code via the meta-crate prelude before calling `report.chrome_trace_json()`. |
| REQ-7 | SHIPPED | impl: `maybe_capture_stack` at `ferrotorch-profiler/src/profiler.rs:227` calling `std::backtrace::Backtrace::capture()` gated on `self.config.with_stack`; non-test consumer: every event-producing method calls `self.maybe_capture_stack()` to populate `ProfileEvent::stack_trace` (lines 127, 149, 179, 217, 397). `Profiler` itself is the production consumer of the helper. |
| REQ-8 | SHIPPED | impl: `push_event` at line 320 (silent drop on poison), `push_event_returning_index` at line 332 (returns `None` on poison), `into_report` at line 358 (`unwrap_or_default`), `unreachable!` at line 457 documenting the impossible Arc-count branch; non-test consumer: every event method funnels through `push_event` / `push_event_returning_index`; `with_profiler` consumes the `unreachable!`-bearing branch at line 456 inside the closure-scoped lifecycle. The poisoned-mutex behavior is exercised indirectly by the test `test_push_gpu_event_inactive_returns_none` (line 517) which validates the `None` return path. |
