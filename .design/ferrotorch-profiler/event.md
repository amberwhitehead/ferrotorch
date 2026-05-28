# ferrotorch-profiler — `ProfileEvent` value type, device + memory tagging

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/profiler/__init__.py
  - torch/profiler/profiler.py
  - torch/autograd/profiler.py
  - torch/profiler/_memory_profiler.py
-->

## Summary

`ferrotorch-profiler/src/event.rs` defines the value types every other
profiler module produces or consumes: `DeviceType` (`Cpu` / `Cuda`),
`MemoryCategory` (Activations / Parameters / `OptimizerState` /
Gradients / Other), the `ProfileEvent` record, and the `GpuTimingPair`
returned by `push_gpu_event`. Mirrors the per-event payload PyTorch's
`torch.autograd.profiler.FunctionEvent` and `torch.profiler.profile`
emit, narrowed to the shape-driven, dependency-free subset
ferrotorch needs (no Kineto / CUPTI / pickled C++ FunctionEvent —
those are deviations recorded in `profiler.md`).

## Requirements

- REQ-1: `pub enum DeviceType { Cpu, Cuda }` with `#[derive(Default)]`
  → `Cpu`, plus `Display` mirroring PyTorch's `DeviceType.CPU` /
  `DeviceType.CUDA` names returned by
  `torch._C._autograd.DeviceType` (re-exported at
  `torch/profiler/__init__.py:15`).
- REQ-2: `pub enum MemoryCategory` with the five variants
  `Activations`, `Parameters`, `OptimizerState`, `Gradients`, `Other`
  (default `Other`), plus `Display` returning the lowercase
  underscore names. Mirrors `torch.profiler._memory_profiler.Category`
  at `torch/profiler/_memory_profiler.py:37-44` whose variants are
  `INPUT`, `TEMPORARY`, `ACTIVATION`, `GRADIENT`, `AUTOGRAD_DETAIL`,
  `PARAMETER`, `OPTIMIZER_STATE`. The ferrotorch variant set is the
  user-meaningful five; the `INPUT` / `TEMPORARY` / `AUTOGRAD_DETAIL`
  categories upstream are debugging detail folded into `Other`.
- REQ-3: `pub struct ProfileEvent` with the 11 public fields
  documented in the source. Required fields: `name: String`,
  `category: String`, `start_us: u64`, `duration_us: u64`,
  `input_shapes: Vec<Vec<usize>>`, `memory_bytes: Option<i64>`,
  `memory_category: Option<MemoryCategory>`, `thread_id: u64`,
  `device_type: DeviceType`, `flops: Option<u64>`,
  `stack_trace: Option<String>`. Mirrors the union of fields
  `FunctionEvent` exposes through `key_averages()` (see
  `torch/autograd/profiler.py:494` `table`).
- REQ-4: `pub struct GpuTimingPair { start_us: u64, end_us: u64 }`
  carrying the GPU-event-measured (or wall-clock fallback) start/end
  microseconds. Returned by callers wrapping kernel launches so the
  profiler can record them without owning the timing primitive.
- REQ-5: Every type derives the trait set the downstream modules
  rely on: `DeviceType` and `MemoryCategory` are
  `Clone, Copy, Default, PartialEq, Eq, Hash, Debug` (Hash so they
  can be `HashMap` keys in `ProfileReport::memory_by_category`).
  `ProfileEvent` is `Clone, Debug` (no `Copy` — it owns three
  heap allocations). `GpuTimingPair` is `Clone, Copy, Debug`.

## Acceptance Criteria

- [x] AC-1: `DeviceType::default() == DeviceType::Cpu`.
- [x] AC-2: `MemoryCategory::default() == MemoryCategory::Other`.
- [x] AC-3: `format!("{}", DeviceType::Cuda) == "CUDA"`, `Cpu == "CPU"`.
- [x] AC-4: `format!("{}", MemoryCategory::OptimizerState) == "optimizer_state"`.
- [x] AC-5: `ProfileEvent` clones cleanly without panicking on empty
  input-shape vectors (used by `record_with_duration` /
  `push_gpu_event` paths).
- [x] AC-6: `GpuTimingPair` is `Copy` so callers can stash it in
  arrays without lifetime gymnastics.

## Architecture

### `DeviceType` (REQ-1, REQ-5)

`pub enum DeviceType` in `event.rs` carries only the two variants
ferrotorch currently distinguishes — host CPU and CUDA GPU. The
`Display` impl yields `"CPU"` / `"CUDA"` to match the strings
PyTorch's Chrome trace `args.device` field uses (consumed by
`chrome://tracing`'s grouping). Default is `Cpu` so callers that
construct an event via `ProfileEvent { device_type: Default::default(), ... }`
get the safe-on-host value without an explicit annotation.

### `MemoryCategory` (REQ-2, REQ-5)

Five variants mirroring the categories PyTorch's memory profiler
emits in its summary table (Activations dominate during training,
Parameters are static, `OptimizerState` is 2x parameters for
Adam-style, Gradients track the backward pass, Other catches
anything that doesn't fit). `Display` returns the lowercase
underscore names so the table renderer in `report.rs` can emit
`"optimizer_state"` directly into the trace JSON without a second
mapping. The PyTorch upstream uses `enum.auto()`-numbered variants
with no stable wire format; ferrotorch fixes the names so JSON
trace consumers (TensorBoard, Perfetto) get stable labels.

### `ProfileEvent` (REQ-3)

The 11-field struct is the union of every kind of event the
profiler records:
- CPU op via `Profiler::record` — shapes populated, memory_bytes
  None, flops estimated from shapes.
- Pre-timed CPU op via `record_with_duration` — duration known
  ahead of insertion, shapes empty.
- Memory event via `record_memory_categorized` — memory_bytes +
  memory_category populated, duration 0.
- GPU event via `push_gpu_event` — device_type Cuda, duration from
  the `GpuTimingPair`.
- CUDA-event-timed kernel via the cuda-feature path — device_type
  Cuda, duration computed from `cuEventElapsedTime`.

All five paths produce the same record type so downstream
aggregation (`ProfileReport::top_ops`, `memory_by_category`,
`chrome_trace_json`) doesn't need to branch on event shape.

### `GpuTimingPair` (REQ-4)

`Copy + Clone` so callers can pass it by value into
`push_gpu_event` without juggling references. The duration of the
GPU op is `end_us - start_us` (saturating); start_us anchors the
event to the profiler's epoch so chrome trace lines up GPU events
with concurrent CPU events.

### Non-test production consumers

- `ferrotorch-profiler/src/profiler.rs` `use crate::event::{DeviceType, GpuTimingPair, MemoryCategory, ProfileEvent};` — the profiler constructs every kind of event through this re-export.
- `top_ops in ferrotorch-profiler/src/report.rs` `use crate::event::{DeviceType, ProfileEvent};` — `ProfileReport::top_ops` partitions by `DeviceType`, `memory_by_category` groups by `MemoryCategory`.
- `finalize in ferrotorch-profiler/src/cuda_timing.rs` `use crate::event::{DeviceType, ProfileEvent};` — `PendingCudaScope::finalize` constructs the post-sync GPU event.
- `ferrotorch-profiler/src/lib.rs:40` re-exports all four types (`DeviceType`, `GpuTimingPair`, `MemoryCategory`, `ProfileEvent`) at the crate root.
- `ferrotorch/src/lib.rs:107` `pub use ferrotorch_profiler::*;` propagates them into the meta-crate `prelude` so user code can write `ferrotorch::profiler::ProfileEvent`.

## Parity contract

`parity_ops = []`. This file holds value types; numerical parity is
owned by `profiler.md`, `report.md`, `flops.md`. Edge cases the value
types own:

- **`MemoryCategory::Hash` stability**: derived from variant
  discriminant, stable for a given build but not across compilers.
  Acceptable since the wire format goes through `Display` (string
  names), not the discriminant.
- **`ProfileEvent::input_shapes` empty vs absent**: the value type
  treats empty `Vec<Vec<usize>>` and absence identically — both
  render as `[]` in chrome trace JSON. PyTorch distinguishes them
  via `with_shapes=False` (field omitted) vs `[]` (empty); ferrotorch
  always carries the field and uses emptiness as the off-signal,
  losing no information for callers that respect the
  `ProfileConfig::record_shapes` flag.
- **`GpuTimingPair::end_us < start_us`**: `push_gpu_event` uses
  `saturating_sub` so duration becomes 0 rather than underflowing
  to `u64::MAX`. Matches PyTorch's `cudaEventElapsedTime` clamp at
  zero for out-of-order events.

## Verification

No tests live in `event.rs` itself (it's pure data). The types are
exercised by 4 tests in `profiler.rs` (`mod tests`), 7 tests in
`report.rs`, and the crate's `tests/conformance_event.rs` integration
file. Smoke:

```bash
cargo test -p ferrotorch-profiler --lib event 2>&1 | tail -3
```

Expected: `0 passed; 0 failed` for `event::tests` (no direct tests)
and `33 passed; 0 failed` for the full lib-test sweep that
exercises these types through `Profiler::record*`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum DeviceType` at `ferrotorch-profiler/src/event.rs` with `Display` at line 12, mirroring `torch/profiler/__init__.py:15` (`from torch._C._autograd import DeviceType`); non-test consumer: `pub in ferrotorch-profiler/src/profiler.rs` sets `device_type: DeviceType::Cpu` on every CPU event, `pub in ferrotorch-profiler/src/profiler.rs` sets `DeviceType::Cuda` on every GPU event, `pub in ferrotorch-profiler/src/report.rs` branches on it for per-device aggregation. |
| REQ-2 | SHIPPED | impl: `pub enum MemoryCategory` at `ferrotorch-profiler/src/event.rs:25` with `Display` at line 44, mirroring `torch/profiler/_memory_profiler.py:37-44`; non-test consumer: `ferrotorch-profiler/src/profiler.rs:164` `record_memory_categorized` accepts it, `ferrotorch-profiler/src/report.rs:122` `memory_by_category` groups events by it. |
| REQ-3 | SHIPPED | impl: `pub struct ProfileEvent` at `ProfileEvent in ferrotorch-profiler/src/event.rs` with the 11 fields documented in source comments, mirroring `torch.autograd.profiler.FunctionEvent`'s public field set (`torch/autograd/profiler.py:494` `table()` columns); non-test consumer: every event-producing method in `ferrotorch-profiler/src/profiler.rs` (`record` at line 116, `record_with_duration` at line 138, `record_memory_categorized` at line 168, `push_gpu_event` at line 205, `OpProfiler::record_op` at line 386), plus `record in ferrotorch-profiler/src/cuda_timing.rs` `PendingCudaScope::finalize` constructs one. |
| REQ-4 | SHIPPED | impl: `pub struct GpuTimingPair` at `GpuTimingPair in ferrotorch-profiler/src/event.rs`; non-test consumer: `pub in ferrotorch-profiler/src/profiler.rs` `push_gpu_event(timing: GpuTimingPair)` accepts it, computing `duration_us = timing.end_us.saturating_sub(timing.start_us)` at line 209. |
| REQ-5 | SHIPPED | impl: derive attrs at `DeviceType in ferrotorch-profiler/src/event.rs` (`DeviceType`), line 24 (`MemoryCategory`), line 57 (`ProfileEvent`), line 97 (`GpuTimingPair`); non-test consumer: `Hash + Eq` on `MemoryCategory` used by `new in ferrotorch-profiler/src/report.rs` `HashMap<MemoryCategory, i64>`; `Clone` on `ProfileEvent` used by `new in ferrotorch-profiler/src/report.rs` `ProfileReport::new(events: Vec<ProfileEvent>)` which clones the slice for `events()` access; `Copy` on `GpuTimingPair` used at `events in ferrotorch-profiler/src/profiler.rs` (test) and consumed by-value at the production call site. |
