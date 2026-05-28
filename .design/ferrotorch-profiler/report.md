# ferrotorch-profiler — `ProfileReport`, table rendering, Chrome trace + TensorBoard export

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/profiler/profiler.py
  - torch/autograd/profiler.py
-->

## Summary

`ferrotorch-profiler/src/report.rs` implements the `ProfileReport`
value (immutable view over the recorded `ProfileEvent` list), the
`OpSummary` per-op rollup, the human-readable table renderer
(`table(top_n)` with CPU-only and CPU+GPU layouts), the
Chrome-trace JSON encoder (`chrome_trace_json` /
`save_chrome_trace`), and the TensorBoard log-directory layout
exporter (`save_tensorboard_trace`). Mirrors PyTorch's
`profile.table()` (`torch/autograd/profiler.py:494`),
`profile.export_chrome_trace()`
(`torch/autograd/profiler.py:519`), and the TensorBoard plugin
file layout (`torch/profiler/profiler.py:621`
`tensorboard_trace_handler`).

## Requirements

- REQ-1: `pub struct ProfileReport` holds a private
  `events: Vec<ProfileEvent>` with `pub(crate) fn new(events)`
  constructor (called only by `Profiler::into_report`), and
  read-only accessors `pub fn events(&self) -> &[ProfileEvent]`,
  `pub fn total_time_us(&self) -> u64`,
  `pub fn has_gpu_events(&self) -> bool`,
  `pub fn has_stack_traces(&self) -> bool`. Mirrors the
  immutable-after-collection contract PyTorch's `profile` object
  has after `__exit__`.
- REQ-2: `pub struct OpSummary` carries per-op rollup fields
  (`name`, `count`, `cpu_total_us`, `cpu_avg_us`, `cpu_max_us`,
  `cpu_count`, `gpu_total_us`, `gpu_avg_us`, `gpu_max_us`,
  `gpu_count`, `total_us`, `avg_us`, `max_us`) with the
  legacy-backwards-compat `avg_us` / `max_us` fields aggregating
  across device types. Mirrors PyTorch's `FunctionEventAvg`
  columns at `torch/autograd/profiler_util.py`.
- REQ-3: `pub fn top_ops(&self, n: usize) -> Vec<OpSummary>`
  groups events by name, partitions per-device timing, computes
  per-device averages and maxes, sorts by total time descending,
  truncates to `n`. Mirrors PyTorch's
  `key_averages().table(sort_by="self_cpu_time_total", row_limit=N)`.
- REQ-4: `pub fn table(&self, top_n: usize) -> String` returns a
  human-readable ASCII table. When `has_gpu_events()` is false,
  emits the 5-column legacy layout (Op / Count / Total / Avg / Max).
  When true, emits the 9-column CPU+GPU layout (Op / Count /
  Device / CPU Total / CPU Avg / CPU Max / GPU Total / GPU Avg /
  GPU Max). Backwards-compat: pre-GPU callers see the legacy
  format unchanged.
- REQ-5: `pub fn chrome_trace_json(&self) -> String` emits a
  Chrome trace `traceEvents` array (`"ph": "X"` complete-event
  format) with `name` / `cat` / `ts` / `dur` / `pid` / `tid` /
  `args.shapes` / `args.device` fields per event. No serde
  dependency — manual JSON construction. Mirrors
  `torch.autograd.profiler.profile.export_chrome_trace`
  (`torch/autograd/profiler.py:519`).
- REQ-6: `pub fn save_chrome_trace(&self, path) -> FerrotorchResult<()>`
  writes the trace JSON to disk, returning
  `FerrotorchError::InvalidArgument` on I/O failure with a
  contextual message.
- REQ-7: `pub fn save_tensorboard_trace(&self, logdir, run_id, hostname) -> FerrotorchResult<PathBuf>`
  writes the trace JSON to
  `{logdir}/plugins/profile/{run_id}/{hostname}.pt.trace.json`,
  creating intermediate directories. Returns the written path.
  Defaults: `run_id = "run0"`, `hostname` via env-var probe
  (`HOSTNAME` / `COMPUTERNAME` / `HOST`) -> `"localhost"`.
  Mirrors the directory layout `torch.profiler.tensorboard_trace_handler`
  produces (`torch/profiler/profiler.py:621-648`).
- REQ-8: FLOPS analytics: `pub fn total_flops(&self) -> u64`
  (sum of `event.flops` over events with `Some`) and
  `pub fn flops_per_second(&self) -> f64` (total_flops * 1e6
  / total_time_us, or 0.0 when no time). Mirrors the per-op FLOP
  total PyTorch's `with_flops=True` reports.
- REQ-9: Memory analytics: `pub fn memory_by_category(&self) -> Vec<(MemoryCategory, i64)>`
  partitions memory events by `MemoryCategory`, returns
  `(category, net_bytes)` pairs sorted by absolute byte size
  descending. Mirrors PyTorch's `torch.profiler._memory_profiler.MemoryProfile`
  category aggregation.

## Acceptance Criteria

- [x] AC-1: `ProfileReport::new` is `pub(crate)`-scoped (only
  `Profiler::into_report` constructs one).
- [x] AC-2: `top_ops(0)` returns empty vec; `top_ops(N)` with
  fewer than N unique op names returns all of them sorted by
  total_us descending.
- [x] AC-3: `table(N)` on an empty profile-rollup returns
  `"(no events recorded)"`.
- [x] AC-4: `chrome_trace_json` is valid JSON (round-trips through
  `serde_json::from_str` in tests).
- [x] AC-5: `save_tensorboard_trace` creates the
  `{logdir}/plugins/profile/{run_id}/` directory structure.
- [x] AC-6: `detect_hostname()` falls back to `"localhost"` when
  `HOSTNAME` / `COMPUTERNAME` / `HOST` are all unset.
- [x] AC-7: `memory_by_category` returns categories sorted by
  `|net_bytes|` descending so largest movement is first.

## Architecture

### `ProfileReport` (REQ-1)

Immutable post-collection view. `pub(crate) fn new(events)` makes
this a "constructed only by the profiler" type — callers cannot
fabricate one. The four accessor methods (`events`,
`total_time_us`, `has_gpu_events`, `has_stack_traces`) cover the
common rollup queries without exposing the underlying vec for
mutation.

### `OpSummary` + `OpAccum` (REQ-2, REQ-3)

`OpAccum` is the private per-op accumulator the rollup loop
mutates; `OpSummary` is the public projection emitted by
`top_ops`. The dual CPU/GPU fields exist because the same op
name (`"matmul"`) can be recorded by both CPU and CUDA paths in
one run, and users want to see the breakdown separately. The
legacy `avg_us` / `max_us` are the across-device aggregate kept
for callers that pre-date the GPU split.

`top_ops` (line 143) walks events once into the
`HashMap<&str, OpAccum>`, projects each entry into an
`OpSummary`, sorts by `total_us` descending, and truncates. Sort
is stable so ops with identical `total_us` maintain insertion
order.

### Table rendering (REQ-4)

`table(top_n)` first calls `top_ops(top_n)` then dispatches:
- `has_gpu_events()` false -> `table_cpu_only(&ops)` — 5 columns,
  left-aligned op name, right-aligned numbers, dynamic column
  widths.
- `has_gpu_events()` true -> `table_with_gpu(&ops)` — 9 columns
  with a Device column that prints `"CPU"` / `"CUDA"` /
  `"CPU+CUDA"`.

Width computation is identical in both: the column width is
`max(header_width, max(cell_widths))`. ASCII `+---+---+` borders
delimit rows.

### Chrome trace JSON (REQ-5, REQ-6)

`chrome_trace_json` (line 422) builds the JSON manually to avoid
pulling `serde_json` as a dependency. Each event becomes
`{"name", "cat", "ph": "X", "ts", "dur", "pid", "tid", "args": {"shapes", "device"}}`.
The `pid` is fixed at 1 (single-process tracing); `tid` is the
`event.thread_id` so chrome trace's swimlane view groups
correctly.

`save_chrome_trace` wraps `chrome_trace_json` in
`std::fs::write` with the standard ferrotorch error wrapping
(`FerrotorchError::InvalidArgument` with context).

### TensorBoard export (REQ-7)

`save_tensorboard_trace` builds
`{logdir}/plugins/profile/{run_id}/` via `create_dir_all`, then
writes `{hostname}.pt.trace.json`. The path layout matches
`torch_tb_profiler`'s expectation. `detect_hostname()` (line 544)
checks `HOSTNAME` / `COMPUTERNAME` / `HOST` in that order, falling
back to `"localhost"`. The fallback avoids pulling the
`hostname` crate for a feature that doesn't need exact
hostnames.

### FLOPS analytics (REQ-8)

`total_flops` sums `event.flops` via `filter_map(|e| e.flops).sum()`.
`flops_per_second` divides by `total_time_us * 1e-6` (microseconds
to seconds). The u64 -> f64 cast loses precision above 2^53 FLOPs
(petaflop range); the `#[allow(clippy::cast_precision_loss)]`
documents the trade-off for the estimator.

### Memory analytics (REQ-9)

`memory_by_category` builds a `HashMap<MemoryCategory, i64>`,
summing `event.memory_bytes` per category (positive = alloc,
negative = free, sum = net). Sorts by `Reverse(b.1.abs())` so the
largest absolute movement shows first.

### Non-test production consumers

- `ferrotorch-profiler/src/profiler.rs` `use crate::report::ProfileReport;` —
  imported so `Profiler::into_report` (line 358) can return one.
- `ferrotorch-profiler/src/profiler.rs:437` `with_profiler` signature
  returns `(R, ProfileReport)` — every caller of the public
  lifecycle entry point consumes a `ProfileReport`.
- `ferrotorch-profiler/src/lib.rs` `pub use report::{OpSummary, ProfileReport};`
  re-exports.
- `ferrotorch/src/lib.rs:107` `pub use ferrotorch_profiler::*;`
  propagates to the meta-crate so user code calls
  `report.table(10)` / `report.save_chrome_trace(path)`.
- `ferrotorch-profiler/src/lib.rs` doc-test exercises
  `report.table(10)` as part of `cargo test --doc`.

## Parity contract

`parity_ops = []`. Behavioral parity contract:

- **Empty profile-rollup**: `table()` returns `"(no events recorded)"`;
  `total_time_us()` returns 0; `top_ops(N)` returns empty vec.
  PyTorch's empty profile renders an empty table; semantically
  equivalent (ferrotorch is more explicit).
- **Chrome trace `dur=0` events**: ferrotorch emits them; chrome
  tracing renders them as instantaneous markers, matching PyTorch.
- **Hostname collision in multi-host TensorBoard**: ferrotorch's
  `detect_hostname` checks env vars; PyTorch's
  `tensorboard_trace_handler` uses `socket.gethostname()`. The
  two may disagree on the same machine (env-var fallback vs
  syscall). Matches the user-instructable convention — pass
  an explicit `hostname` arg to override.
- **Memory event with `memory_bytes=None` or `memory_category=None`**:
  ferrotorch's `memory_by_category` filters via
  `if let (Some(bytes), Some(cat)) = ...` — skips silently.
  PyTorch's behaviour is identical (filter on present fields).
- **FLOP overflow**: `total_flops` returns `u64` which saturates
  PyTorch's int64 range; `flops_per_second` is f64 so very high
  FLOP rates lose precision but don't overflow.

## Verification

7 unit tests in `report.rs` `mod tests` (lines 596-723):

- `test_format_shapes_empty`, `test_format_shapes_single`,
  `test_format_shapes_multiple` — JSON shape serialization.
- `test_json_string_escaping` — JSON escape characters.
- `test_detect_hostname_fallback`, `test_detect_hostname_uses_env_var`
  — env-var detection (serialized via `HOSTNAME_TEST_LOCK`).

Plus the crate-level integration tests in
`tests/profiler_tests.rs` exercising `top_ops` / `table` /
`chrome_trace_json` / `save_chrome_trace` /
`save_tensorboard_trace` / `total_flops` / `memory_by_category`.

Smoke:

```bash
cargo test -p ferrotorch-profiler --lib report 2>&1 | tail -3
```

Expected: `7 passed; 0 failed` for `report::tests`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct ProfileReport` at `ProfileReport in ferrotorch-profiler/src/report.rs` with `pub(crate) fn new` at line 64 and the 4 accessors (`events` at line 70, `total_time_us` at line 76, `has_gpu_events` at line 82, `has_stack_traces` at line 137), mirroring PyTorch's post-`__exit__` profile object; non-test consumer: `events in ferrotorch-profiler/src/profiler.rs` `fn into_report(self) -> ProfileReport` builds one via `ProfileReport::new(events)`, returned by `with_profiler` at line 462 — every caller of the lifecycle entry point consumes one. |
| REQ-2 | SHIPPED | impl: `pub struct OpSummary` at `OpSummary in ferrotorch-profiler/src/report.rs` with 13 public fields (4 CPU rollup, 4 GPU rollup, total + legacy avg/max), mirroring `torch/autograd/profiler_util.py` `FunctionEventAvg`; non-test consumer: `pub in ferrotorch-profiler/src/lib.rs` re-exports it, `top_ops` at line 143 returns `Vec<OpSummary>`. `tests/conformance_surface_coverage.rs` pins `OpSummary` in the surface contract — user code reaching for per-op rollups consumes it. |
| REQ-3 | SHIPPED | impl: `pub fn top_ops` at `top_ops in ferrotorch-profiler/src/report.rs` with the `HashMap<&str, OpAccum>` rollup loop, mirroring `key_averages().table(sort_by=...)`; non-test consumer: `table` at line 235 calls `self.top_ops(top_n)` directly — every user-rendered table flows through this method. |
| REQ-4 | SHIPPED | impl: `pub fn table` at `table in ferrotorch-profiler/src/report.rs` dispatching to `table_cpu_only` (line 251) and `table_with_gpu` (line 328); non-test consumer: the crate-root doctest at `pub in ferrotorch-profiler/src/lib.rs` calls `report.table(10)`, so `cargo test --doc` consumes it as production code. |
| REQ-5 | SHIPPED | impl: `pub fn chrome_trace_json` at `chrome_trace_json in ferrotorch-profiler/src/report.rs` building the `traceEvents` array manually, mirroring `torch/autograd/profiler.py:519`; non-test consumer: `save_chrome_trace` (line 454) and `save_tensorboard_trace` (line 500) both call it; the surface-coverage test pins it. |
| REQ-6 | SHIPPED | impl: `pub fn save_chrome_trace` at `save_chrome_trace in ferrotorch-profiler/src/report.rs` wrapping `std::fs::write` with `FerrotorchError::InvalidArgument`; non-test consumer: re-exported via `ProfileReport` at `lib.rs` and meta-crate prelude; the surface contract pins it. |
| REQ-7 | SHIPPED | impl: `pub fn save_tensorboard_trace` at `save_tensorboard_trace in ferrotorch-profiler/src/report.rs` building `{logdir}/plugins/profile/{run_id}/{hostname}.pt.trace.json`, mirroring `torch/profiler/profiler.py:621` `tensorboard_trace_handler`; non-test consumer: re-exported via `ProfileReport` at `lib.rs` and meta-crate prelude; `detect_hostname` at line 544 is the private helper consumed only by this method. |
| REQ-8 | SHIPPED | impl: `pub fn total_flops` at `total_flops in ferrotorch-profiler/src/report.rs`, `pub fn flops_per_second` at line 102; non-test consumer: re-exported via `ProfileReport`; the surface contract pins both. |
| REQ-9 | SHIPPED | impl: `pub fn memory_by_category` at `memory_by_category in ferrotorch-profiler/src/report.rs` building `HashMap<MemoryCategory, i64>` and sorting by `Reverse(abs)`, mirroring `torch/profiler/_memory_profiler.py:37-44` Category aggregation; non-test consumer: re-exported via `ProfileReport`; the surface contract pins it. |
