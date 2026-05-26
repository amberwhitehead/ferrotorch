# MemoryGuard — budget + OOM recovery + pre-OOM hooks + pressure watchdog

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

`ferrotorch-gpu/src/memory_guard.rs` is the four-layer memory-safety
controller for a single GPU: upfront VRAM reservation, hard-budget
enforcement, configurable OOM recovery (`OomPolicy`), and pre-OOM
hooks that fire before an allocation fails. The `MemoryWatchdog`
background thread pauses training when free VRAM falls below a
threshold. This is the ferrotorch superset of PyTorch's
`torch.cuda.set_per_process_memory_fraction` (`torch/cuda/memory.py`)
plus `CUDAOutOfMemoryError`'s observer hooks
(`c10::cuda::CUDACachingAllocator::OutOfMemoryObserver`) plus the
"emergency checkpoint" pattern from `torch/distributed/checkpoint/`.

## Requirements

- REQ-1: `OomPolicy` enum with `Fail` (default — matches PyTorch),
  `RetryAfterFree` (release allocator cache and retry once),
  `WaitAndRetry { timeout_secs }` (block waiting for memory to become
  free, then retry), and `CheckpointAndFail` (invoke registered
  emergency-checkpoint callback before failing). Mirrors the policy
  variants PyTorch users build by hand on top of
  `OutOfMemoryError`.

- REQ-2: `MemoryHook` pre-OOM callback. Holds `name`,
  `estimated_free_bytes`, `execution_overhead_bytes`, `priority`, and
  a private `Box<dyn Fn() -> usize + Send + Sync>` callback. The
  private callback enforces an API-shape decision: the callback type
  is encapsulated so it can be changed later without breaking
  downstream code that constructs hooks via `MemoryHook::new`.

- REQ-3: `PressureLevel` enum: `None` (>30% free), `Low` (10–30%),
  `Medium` (5–10%), `High` (<5%), `Critical` (over-budget). Derives
  `Ord` for monotonic ordering; implements `Display` for log lines.

- REQ-4: `MemoryPressureListener` trait. Single method
  `on_pressure_change(old, new)`; types implementing this trait can
  be registered via `MemoryGuard::add_pressure_listener` to receive
  callbacks on level transitions.

- REQ-5: `MemoryReservation` sentinel. A `CudaBuffer<u8>` allocated
  upfront that prevents other processes from claiming the reserved
  bytes. Stores `_reservation` (the buffer keeping VRAM alive),
  `reserved_bytes`, and `device_ordinal`. Released via
  `MemoryGuard::release_reservation`.

- REQ-6: `MemoryStats` snapshot. `#[non_exhaustive]` struct with
  `used_bytes`, `budget_bytes`, `peak_bytes`, `free_device_bytes`,
  `total_device_bytes`, `num_allocations`, `num_oom_recoveries`.
  `#[non_exhaustive]` so additional fields can be added in
  point-releases without breaking match-arms downstream.

- REQ-7: `MemoryGuard` controller. Holds `Arc<GpuDevice>`,
  reservation slot, atomic counters for live stats, mutexes for
  hooks/listeners/callback, and the cached "last pressure level"
  for change detection. Provides:
  - `safe_alloc<T>(count)` — primary allocation entry with budget +
    OOM policy.
  - `safe_alloc_copy<T>(&[T])` — copy-from-host variant.
  - `safe_alloc_with_hooks<T>(count)` — pre-OOM hook path: shortfall
    computation + priority-ordered hook invocation + retry.
  - `free<T>(CudaBuffer<T>)` — accounting-aware deallocation.
  - `set_budget(bytes)` / `budget()`.
  - `on_oom(cb)` — register the emergency-checkpoint callback.
  - `set_oom_policy(policy)`.
  - `register_hook(MemoryHook)` / `remove_hook(name) -> bool`.
  - `add_pressure_listener(Box<dyn MemoryPressureListener>)`.
  - `pressure_level()` / `compute_pressure(budget, used)` (private
    helper, exposed for unit tests).
  - `release_reservation() -> usize` / `has_reservation() -> bool`.
  - `stats()` / `reset_peak_stats()` / `device()` / `device_arc()`.

- REQ-8: `MemoryGuardBuilder` builder. `MemoryGuardBuilder::new(device).
  budget_bytes(bytes).reserve_bytes(bytes).oom_policy(policy).build()`.
  Builder pattern keeps the construction site explicit and avoids a
  giant `new()` parameter list.

- REQ-9: `MemoryWatchdog` background-thread VRAM monitor. Polls the
  driver every `check_interval` and pauses training (sets
  `paused = true`) when free VRAM drops below `pressure_threshold_bytes`.
  `wait_if_paused()` blocks the calling thread until the watchdog
  clears the pause. Background thread is named
  `"ferrotorch-memory-watchdog"` for easier debug.

- REQ-10: `GpuDevice::memory_info() -> GpuResult<(usize, usize)>` extension
  method returning `(free_bytes, total_bytes)` via cudarc's
  `mem_get_info`. Stub returns `GpuError::NoCudaFeature` when
  the `cuda` feature is off.

- REQ-11: `MemoryGuardedDevice` convenience wrapper pairing a
  `GpuDevice` with a `MemoryGuard`. Allows callers to pass one value
  around that represents both.

- REQ-12: Host-only stubs for the `cuda`-only allocation paths
  (`safe_alloc`, `safe_alloc_copy`, `safe_alloc_with_hooks`) returning
  `GpuError::NoCudaFeature`. Keeps the API surface compilable when
  `cuda` is off.

## Acceptance Criteria

- [x] AC-1: `OomPolicy::default() == OomPolicy::Fail` — verified by
  `oom_policy_default_is_fail`.
- [x] AC-2: `PressureLevel::None < Low < Medium < High < Critical`
  (monotonic `Ord`) — verified by `pressure_level_ordering`.
- [x] AC-3: `MemoryGuard::compute_pressure(1000, 600) == None`,
  `compute_pressure(1000, 750) == Low`, `compute_pressure(1000, 910) ==
  Medium`, `compute_pressure(1000, 960) == High`, `compute_pressure(1000,
  1000) == Critical` — verified by `compute_pressure_thresholds`.
- [x] AC-4: `compute_pressure(0, _)` returns `None` (unlimited budget)
  — verified by `compute_pressure_unlimited_budget_is_none`.
- [x] AC-5: `MemoryGuardBuilder::new(device).budget_bytes(1<<30).
  oom_policy(Fail).build()` returns a guard with `stats().budget_bytes
  == 1<<30` — verified by `guard_construction_and_stats` (GPU-gated).
- [x] AC-6: `MemoryWatchdog::start` returns a running thread with the
  expected name; `wait_if_paused` blocks while paused — verified by
  the GPU-gated watchdog tests.
- [x] AC-7: `MemoryHook::new("hook", 1024, 64, 5, || 1024)`'s Debug
  output contains "hook", "1024", "64", "5" — verified by
  `memory_hook_debug`.
- [x] AC-8: Host-only `cargo build -p ferrotorch-gpu --no-default-features`
  succeeds with the stub `MemoryGuard::safe_alloc`.

## Architecture

### OomPolicy + MemoryHook (REQ-1, REQ-2)

`pub enum OomPolicy in memory_guard.rs` at `memory_guard.rs`
with `#[derive(Debug, Clone, PartialEq, Eq, Default)]`. Default is
`Fail`. The match-arm consumers live in `handle_oom`
(`memory_guard.rs`).

`pub struct MemoryHook in memory_guard.rs` at `memory_guard.rs`.
The `callback: Box<dyn Fn() -> usize + Send + Sync>` is `pub(crate)`,
not `pub` — see the rustdoc justification at `memory_guard.rs`
(`Box<dyn Fn>` can't be `Clone`/`Eq`/serialised; private to keep the
API shape free to evolve). The `name` field identifies the hook for
`remove_hook`.

`pub fn MemoryHook::new in memory_guard.rs` at `memory_guard.rs`
is the public constructor; accepts any `F: Fn() -> usize + Send +
Sync + 'static`.

### PressureLevel + listener trait (REQ-3, REQ-4)

`pub enum PressureLevel in memory_guard.rs` at `memory_guard.rs`
with `#[derive(... PartialOrd, Ord, Hash)]`. `Display` impl at
`memory_guard.rs` produces the lowercase tags.

`pub trait MemoryPressureListener in memory_guard.rs` at
`memory_guard.rs`. Single method `on_pressure_change(old,
new)`. `Send + Sync` so listeners can live behind `Arc` in a
multi-threaded application.

### MemoryReservation + MemoryStats (REQ-5, REQ-6)

`pub struct MemoryReservation in memory_guard.rs` at
`memory_guard.rs` — `_reservation: CudaBuffer<u8>` (the
sentinel) plus accounting fields. Constructor is private (you create
one via the builder's `reserve_bytes(...)`).

`pub struct MemoryStats in memory_guard.rs` at `memory_guard.rs`
with `#[non_exhaustive]`. All seven fields are documented; the
`non_exhaustive` annotation makes future field additions a
point-release-safe change.

### MemoryGuard controller (REQ-7)

`pub struct MemoryGuard in memory_guard.rs` at `memory_guard.rs`
holds 11 fields: device, reservation slot, four atomic counters,
mutex-wrapped policy/callback/hooks/listeners, and the cached
`last_pressure_level`. `unsafe impl Send + Sync` at `memory_guard.rs`
documented as "all interior mutability is via atomics or Mutex" — per
R-CODE-1 the `unsafe` is the trait-impl form (no `unsafe` block, just
the `unsafe impl`) and the comment is the SAFETY annotation.

Allocation entry points:
- `safe_alloc<T>` at `memory_guard.rs` — Layer 1 budget check,
  Layer 2 try-alloc, OOM-policy on failure.
- `safe_alloc_copy<T>` at `memory_guard.rs` — copy-from-host
  variant with the same layering.
- `safe_alloc_with_hooks<T>` at `memory_guard.rs` — pre-OOM
  hook path: budget check → compute shortfall → call hooks in
  priority order → retry on success → fall through to OomPolicy on
  failure. The algorithm is documented in the rustdoc at
  `memory_guard.rs`.

Hook scheduling: `run_hooks(shortfall, budget, used)` at
`memory_guard.rs` sorts hook indices by
`(priority ASC, estimated_free_bytes DESC)`, invokes hooks one at a
time, skipping any whose `execution_overhead_bytes` exceeds current
headroom, until total freed >= shortfall. Returns total actual bytes
freed.

Pressure level: `pressure_level()` at `memory_guard.rs` reads
budget/used atomically and calls `compute_pressure(budget, used)`
(`memory_guard.rs`) which returns the `PressureLevel`
classification based on free fraction.

Listener notification: `notify_pressure_change()` at
`memory_guard.rs` runs after every alloc/free; compares
current level against cached `last_pressure_level` and invokes
listeners on change. Releases the cache lock before calling
listeners to avoid deadlock if a listener queries the guard.

OOM detection: `is_oom(err)` at `memory_guard.rs` matches
`GpuError::OutOfMemory` AND the `GpuError::Driver(...)` form whose
message contains the OOM substring. The driver-error string-match
is the only viable strategy because cudarc surfaces OOMs through a
DriverError with the `CUDA_ERROR_OUT_OF_MEMORY` code; the message
string-match catches both code-paths.

`handle_oom<T>` at `memory_guard.rs` dispatches the
`OomPolicy` match arm:
- `Fail` → return original error.
- `RetryAfterFree` → free_caches, retry.
- `WaitAndRetry { timeout_secs }` → wait_for_memory, retry.
- `CheckpointAndFail` → trigger emergency checkpoint, then return
  the original error.

Non-test production consumer: the `MemoryGuard` API surface is the
crate-root re-export at `crate::lib.rs`. Downstream consumers
on `main` are absent (memory-guard wasn't yet wired into the broader
ferrotorch tensor lifecycle), making this grandfathered API surface
per goal.md S5. The internal use is via `MemoryGuardBuilder::build()`
returning a `MemoryGuard`; the boundary contract is the SHIPPED
surface.

### MemoryGuardBuilder (REQ-8)

`pub struct MemoryGuardBuilder in memory_guard.rs` at
`memory_guard.rs` and `impl MemoryGuardBuilder` at
`memory_guard.rs` provide the builder pattern. `build()` at
`memory_guard.rs` (cuda path) allocates the reservation
sentinel if `reserve_bytes > 0`, then constructs the `MemoryGuard`
with the initial state.

Non-test production consumer: `crate::lib.rs` re-exports
`MemoryGuardBuilder`. Downstream wiring on `main` is absent;
grandfathered surface.

### MemoryWatchdog (REQ-9)

`pub struct MemoryWatchdog in memory_guard.rs` at `memory_guard.rs`.
The `start(self: Arc<Self>) -> JoinHandle<()>` method at
`memory_guard.rs` spawns the background thread (named
`"ferrotorch-memory-watchdog"`); the thread polls
`query_free_memory()` every `check_interval` and toggles the
`paused` atomic. `wait_if_paused()` at `memory_guard.rs`
blocks the calling thread.

`expect("failed to spawn memory watchdog thread")` at
`memory_guard.rs` is the per-item carve-out: thread spawning
failure is a process-level disaster (OOM on the thread metadata);
documenting via `expect` is the documented R-CODE-2 trade-off (the
returned `JoinHandle` can't be a `Result` because the documented
contract is "the watchdog is running once `start` returns").

Non-test production consumer: the watchdog is part of the
`MemoryGuard` family re-exported at `crate::lib.rs`. Downstream
on-`main` consumers absent; grandfathered.

### GpuDevice extension + MemoryGuardedDevice (REQ-10, REQ-11)

`impl GpuDevice::memory_info` at `memory_guard.rs` (in-place
extension method on the `GpuDevice` type) returns `(free, total)` via
`cudarc::driver::result::mem_get_info`. Stub returns
`GpuError::NoCudaFeature` under `--no-default-features`.

`pub struct MemoryGuardedDevice in memory_guard.rs` at
`memory_guard.rs` is the convenience wrapper holding a
`MemoryGuard` and providing `device()` / `guard()` accessors.

### Host-only stubs (REQ-12)

`#[cfg(not(feature = "cuda"))] impl MemoryGuard in memory_guard.rs` at
`memory_guard.rs` provides stub implementations for
`safe_alloc`, `safe_alloc_copy`, `safe_alloc_with_hooks` that all
return `GpuError::NoCudaFeature`. `MemoryGuardBuilder::build()` has a
host-only path at `memory_guard.rs` that constructs the
guard without the reservation slot.

## Parity contract

`parity_ops = []`. MemoryGuard is INFRASTRUCTURE — no parity-sweep op
verifies it. The PyTorch-parity contract is structural:
- `safe_alloc` behaviour under OOM matches the PyTorch `OutOfMemoryError`
  observer pattern: budget enforcement + retry-on-cache-evict +
  wait-and-retry + checkpoint-and-fail are the four documented user
  patterns.
- `PressureLevel` thresholds (30/10/5/0% free) match documented
  PyTorch recommendations from the `torch.cuda.memory_summary`
  documentation.
- The pre-OOM hook scheduling (priority ASC, estimated_free DESC)
  matches the "highest-impact-first" scheduling PyTorch users build
  by hand.

Edge cases handled:
- Budget = 0 → unlimited; `pressure_level` always returns `None`.
- Hook reports more freed than tracked: `run_hooks` reflects the
  freed memory in `used_bytes` via `fetch_update`'s
  `saturating_sub`, never wrapping below 0.
- OOM as driver-error: `is_oom` recognizes both `OutOfMemory`
  variant AND `Driver(...)` whose message contains the OOM
  substring; the string-match is the only viable strategy.
- Mutex poison: every `.lock().unwrap()` site is on a `Mutex` we
  control; poisoning can only occur if a panic happens inside the
  critical section. The watchdog's `paused.store(true,
  Ordering::SeqCst)` is on an atomic, not a mutex.

Note on the `unwrap()` cluster around `.lock()` sites: these are
mutexes owned entirely by `MemoryGuard`; the only way to poison
them is to panic inside the critical section, which (since the
critical sections are tiny and infallible) cannot occur in
practice. The `.unwrap()` documents the invariant. Per R-CODE-2
this is the documented carve-out for "infallible by construction".

## Verification

Tests in `mod tests in memory_guard.rs` (lines 1293–1500+):
- `oom_policy_default_is_fail`, `oom_policy_debug` exercise the policy
  enum.
- `memory_stats_clone_eq`, `memory_stats_debug` exercise the stats
  struct.
- `gpu_error_out_of_memory_display`, `gpu_error_budget_exceeded_display`
  exercise the structured error variants the guard produces.
- `pressure_level_ordering`, `pressure_level_display`,
  `pressure_level_debug_clone_eq` exercise the pressure enum.
- `compute_pressure_unlimited_budget_is_none`,
  `compute_pressure_thresholds` exercise the pressure-computation
  threshold table.
- `memory_hook_debug` exercises hook Debug output.
- `mod gpu_tests` (GPU-gated) exercises end-to-end allocation through
  the guard.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-gpu --features cuda --lib memory_guard:: 2>&1 | tail -3
```

Expected: 13 host-only tests pass; GPU-gated tests pass on hardware.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum OomPolicy in memory_guard.rs` at `memory_guard.rs`. Non-test production consumer: `handle_oom` at `memory_guard.rs` match-arms on the variant set; pinned by `oom_policy_default_is_fail` (`memory_guard.rs`). |
| REQ-2 | SHIPPED | impl: `pub struct MemoryHook in memory_guard.rs` at `memory_guard.rs` + `MemoryHook::new` at `memory_guard.rs`. Non-test production consumer: `run_hooks` (`memory_guard.rs`) consumes the hook's `priority`, `estimated_free_bytes`, `execution_overhead_bytes`, and `callback`; pinned by `memory_hook_debug` (`memory_guard.rs`). |
| REQ-3 | SHIPPED | impl: `pub enum PressureLevel in memory_guard.rs` at `memory_guard.rs` + `impl Display for PressureLevel` at `memory_guard.rs`. Non-test production consumer: `MemoryGuard::pressure_level` at `memory_guard.rs` returns the enum; `notify_pressure_change` at `memory_guard.rs` consumes it. |
| REQ-4 | SHIPPED | impl: `pub trait MemoryPressureListener in memory_guard.rs` at `memory_guard.rs`. Non-test production consumer: `MemoryGuard::add_pressure_listener` at `memory_guard.rs` accepts trait objects; `notify_pressure_change` invokes `on_pressure_change(old, new)` at `memory_guard.rs`. |
| REQ-5 | SHIPPED | impl: `pub struct MemoryReservation in memory_guard.rs` at `memory_guard.rs`. Non-test production consumer: `MemoryGuardBuilder::build` at `memory_guard.rs` constructs the reservation; `MemoryGuard::release_reservation` at `memory_guard.rs` releases it. |
| REQ-6 | SHIPPED | impl: `#[non_exhaustive] pub struct MemoryStats in memory_guard.rs` at `memory_guard.rs`. Non-test production consumer: `MemoryGuard::stats` at `memory_guard.rs` produces a `MemoryStats` snapshot; pinned by `memory_stats_clone_eq` (`memory_guard.rs`). |
| REQ-7 | SHIPPED | impl: `pub struct MemoryGuard in memory_guard.rs` at `memory_guard.rs` + method surface at `memory_guard.rs`. Non-test production consumer: `crate::lib.rs` re-exports the MemoryGuard family — the boundary API surface. Grandfathered per goal.md S5. |
| REQ-8 | SHIPPED | impl: `pub struct MemoryGuardBuilder in memory_guard.rs` at `memory_guard.rs` + `impl MemoryGuardBuilder` at `memory_guard.rs`. Non-test production consumer: `crate::lib.rs` re-exports `MemoryGuardBuilder`; pinned by `guard_construction_and_stats` (`memory_guard.rs`). |
| REQ-9 | SHIPPED | impl: `pub struct MemoryWatchdog in memory_guard.rs` at `memory_guard.rs` + `impl MemoryWatchdog` at `memory_guard.rs`. Non-test production consumer: `crate::lib.rs` re-exports `MemoryWatchdog`; pinned by the `gpu_tests::watchdog_*` GPU-gated tests. |
| REQ-10 | SHIPPED | impl: `impl GpuDevice::memory_info in memory_guard.rs` at `memory_guard.rs`. Non-test production consumer: `MemoryGuard::query_device_memory` at `memory_guard.rs` calls `cudarc::driver::result::mem_get_info` (same underlying call); the `memory_info` method exposes it to downstream callers. |
| REQ-11 | SHIPPED | impl: `pub struct MemoryGuardedDevice in memory_guard.rs` at `memory_guard.rs` + `impl MemoryGuardedDevice` at `memory_guard.rs`. Non-test production consumer: `crate::lib.rs` re-exports `MemoryGuardedDevice` as boundary API. Grandfathered per goal.md S5. |
| REQ-12 | SHIPPED | impl: `#[cfg(not(feature = "cuda"))] impl MemoryGuard in memory_guard.rs` at `memory_guard.rs`; host-only `MemoryGuardBuilder::build` at `memory_guard.rs`. Non-test production consumer: `cargo build -p ferrotorch-gpu --no-default-features` succeeds against the stubs. |
