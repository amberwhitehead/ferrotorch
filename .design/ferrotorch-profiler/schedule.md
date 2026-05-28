# ferrotorch-profiler — `ProfileSchedule` wait/warmup/active/repeat state machine

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/profiler/profiler.py
  - torch/profiler/__init__.py
-->

## Summary

`ferrotorch-profiler/src/schedule.rs` implements the
`ProfileSchedule` state machine that drives a profiler through
`wait → warmup → active → repeat` cycles, plus the
`SchedulePhase` enum (`Waiting` / `Warmup` / `Active` / `Done`).
Mirrors PyTorch's `torch.profiler.schedule(wait, warmup, active, repeat)`
factory function at `torch/profiler/profiler.py:551`, with the
runtime state explicitly held inside the schedule (Rust prefers
a struct over upstream's closure capturing `step` from the outer
`profile` context manager).

## Requirements

- REQ-1: `pub enum SchedulePhase { Waiting, Warmup, Active, Done }`
  with `Display` (capitalised names) and the standard derive set.
  Mirrors `torch.profiler.profiler.ProfilerAction` at
  `torch/profiler/profiler.py:540-548` whose four variants are
  `NONE` / `WARMUP` / `RECORD` / `RECORD_AND_SAVE`. ferrotorch
  collapses `RECORD` + `RECORD_AND_SAVE` into a single `Active`
  state since the on-trace-ready callback (REQ-5) is the
  ferrotorch equivalent of the upstream `RECORD_AND_SAVE`
  distinction.
- REQ-2: `pub struct ProfileSchedule` carrying configuration
  (`wait`, `warmup`, `active`, `repeat`) plus runtime state
  (`current_step`, `current_cycle`, `phase`) plus the optional
  `on_trace_ready: Box<dyn FnMut(u64) + Send>` callback. Storage
  via owned `Box<dyn ...>` (not generics) so the schedule can be
  consumed by a `Profiler` whose type parameter doesn't track the
  callback closure.
- REQ-3: `pub fn new(wait, warmup, active, repeat) -> Self`
  constructor with `assert!(active > 0)` and `assert!(repeat > 0)`
  panics — matches PyTorch's runtime `AssertionError` at
  `torch/profiler/profiler.py:600-604` (`active <= 0` → assert).
  Returns the schedule pre-positioned at the correct phase for
  step 0 (Waiting if wait > 0, else Warmup if warmup > 0, else
  Active).
- REQ-4: `pub fn step(&mut self) -> SchedulePhase` advances the
  state machine by one step, deriving the new phase from the
  current step position within the cycle. End-of-cycle fires the
  `on_trace_ready` callback (REQ-5), increments `current_cycle`,
  and either restarts a new cycle or transitions to `Done`.
  Mirrors the schedule-fn closure at
  `torch/profiler/profiler.py:575-598`.
- REQ-5: `pub fn set_on_trace_ready(&mut self, cb: impl FnMut(u64) + Send + 'static)`
  registers a callback that fires at the end of each active
  window. Mirrors PyTorch's `on_trace_ready` callback parameter
  on `profile(...)` (`torch/profiler/profiler.py:701`).
- REQ-6: `Clone` impl carries config + runtime state but
  intentionally drops the `on_trace_ready` callback (a
  `Box<dyn FnMut + Send>` is not `Clone`). The doc-comment at
  `schedule in schedule.rs` explains the rationale: re-attaching the
  callback would alias state the original owner did not consent
  to share. This is the type-system enforced version of the
  "callback per profile" invariant PyTorch documents informally.
- REQ-7: Read-only accessors `pub fn phase(&self) -> SchedulePhase`,
  `pub fn is_active(&self) -> bool`, `pub fn current_step(&self) -> u64`,
  `pub fn current_cycle(&self) -> u64`. `is_active` is the
  convenience method used by the integration layer that gates
  whether to record events.

## Acceptance Criteria

- [x] AC-1: `ProfileSchedule::new(1, 1, 2, 1).phase() == SchedulePhase::Waiting`.
- [x] AC-2: `ProfileSchedule::new(0, 0, 3, 1).phase() == SchedulePhase::Active`
  (no wait, no warmup → start active).
- [x] AC-3: Cycle of (wait=1, warmup=1, active=2, repeat=1) walks
  Waiting → Warmup → Active → Active → Done.
- [x] AC-4: `new(_, _, 0, _)` panics with
  `"`active` must be > 0"`.
- [x] AC-5: `new(_, _, _, 0)` panics with
  `"`repeat` must be > 0"`.
- [x] AC-6: `on_trace_ready` fires once per cycle end and is
  given the 0-based cycle index.
- [x] AC-7: `clone()` preserves config + position; cloned
  schedule does NOT fire the original's callback.

## Architecture

### `SchedulePhase` (REQ-1)

`pub enum SchedulePhase` with the four variants `Waiting`,
`Warmup`, `Active`, `Done`. `Display` renders them as their
capitalised variant names for log output. The collapse vs.
PyTorch's four `ProfilerAction` variants (we don't distinguish
`RECORD` from `RECORD_AND_SAVE`) is intentional: the
`on_trace_ready` callback fires at end-of-cycle and that's the
only operationally distinct moment.

### `ProfileSchedule` (REQ-2, REQ-3, REQ-4)

The struct holds 8 fields: 4 configuration (`wait`, `warmup`,
`active`, `repeat`), 3 runtime state (`current_step`,
`current_cycle`, `phase`), and 1 optional callback. All
non-callback fields are `u64`; the callback is a
`Box<dyn FnMut(u64) + Send>` (Send so the schedule can move
between threads, no `Sync` because `FnMut` requires `&mut`).

Construction (`schedule in schedule.rs`) is a one-shot panic point:
`active == 0` or `repeat == 0` is a programming error caught at
profile-config time, not a runtime condition.

`step()` (line 179) is the heart of the machine. It increments
`current_step`, checks whether the position is at end-of-cycle
(`pos >= cycle_length`), fires the callback if so, advances the
cycle, and either rewraps or transitions to `Done`. The phase
within a cycle is derived from the position:
`pos < wait` → Waiting, `pos < wait+warmup` → Warmup, else
Active.

### Callback (REQ-5)

`set_on_trace_ready` stashes a `Box<dyn FnMut(u64) + Send + 'static>`
in the optional field. The callback receives the cycle index so
multi-cycle traces can disambiguate which cycle they're handling
(important for the TensorBoard export path that names files by
cycle).

### Clone semantics (REQ-6)

Manual `impl Clone for ProfileSchedule` at line 74-90 carries
every field EXCEPT the callback (set to `None` on the clone).
The doc-comment explains why this is the only sound choice for
a `Box<dyn FnMut>`. Callers that want the cloned schedule to
fire callbacks must call `set_on_trace_ready` on the clone
explicitly.

### Read-only accessors (REQ-7)

`#[must_use]` on every accessor (`phase`, `is_active`,
`current_step`, `current_cycle`). `is_active` returns
`self.phase == SchedulePhase::Active` — the canonical "should I
record now?" check the host integration layer calls.

### Non-test production consumers

- `pub in ferrotorch-profiler/src/lib.rs` `pub mod schedule;`
  exposes the module at the crate root, with line 43
  `pub use schedule::{ProfileSchedule, SchedulePhase};` lifting
  the types into the prelude.
- `ferrotorch/src/lib.rs:107` `pub use ferrotorch_profiler::*;`
  propagates to the meta-crate so user code can write
  `ferrotorch::profiler::ProfileSchedule::new(1, 1, 5, 3)`.
- `ferrotorch-profiler/tests/conformance_surface_coverage.rs:83`
  pins `ProfileSchedule` + `SchedulePhase` in the surface
  contract.

## Parity contract

`parity_ops = []`. The state machine is structural — no
numerical kernels. Behavioral parity contract:

- **Phase at construction with `wait=0, warmup=0`**: starts at
  `Active` (matches PyTorch's `mod_step < wait + warmup` becoming
  vacuously false when both are zero).
- **`active=0`**: panics in `new`. PyTorch raises
  `AssertionError("Invalid profiler schedule arguments")`. The
  message differs (ferrotorch is more specific), the contract
  ("active must be positive") matches.
- **`repeat=0`**: ferrotorch panics; PyTorch interprets `repeat=0`
  as "infinite" (`torch/profiler/profiler.py:586`
  `if repeat > 0 and step / num_steps >= repeat`). This is a
  DEVIATION recorded as a known difference: ferrotorch's
  schedule requires bounded cycles. Users wanting unbounded
  profiling re-create the schedule between cycles. See REQ-3
  for the rationale (Rust prefers explicit "no" over silent
  forever-loops).
- **`on_trace_ready` callback parameter**: ferrotorch passes the
  0-based cycle index. PyTorch passes the `profile` instance.
  Different parameter, same intent — both let the handler save
  a per-cycle trace file.

## Verification

8 unit tests in `schedule.rs` `mod tests` (lines 227-374):

- `basic_phases` — full cycle walk for `(1, 1, 2, 1)`.
- `repeat_cycles` — `(0, 0, 2, 2)` walks two cycles to Done.
- `on_trace_ready_fires` — callback observes 3 cycle indices.
- `no_wait_no_warmup` — starts Active when `wait=warmup=0`.
- `only_warmup` — `(0, 2, 1, 1)` starts Warmup.
- `zero_active_panics`, `zero_repeat_panics` — argument validation.
- `clone_preserves_config_and_position_drops_callback` — Clone
  semantics.
- `current_step_and_cycle_tracking` — accessor read-back.

Smoke:

```bash
cargo test -p ferrotorch-profiler --lib schedule 2>&1 | tail -3
```

Expected: `8 passed; 0 failed` for `schedule::tests`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum SchedulePhase` at `phase in ferrotorch-profiler/src/schedule.rs` with `Display` at line 17, mirroring `torch.profiler.profiler.ProfilerAction` at `torch/profiler/profiler.py:540`; non-test consumer: `phase in ferrotorch-profiler/src/schedule.rs` `ProfileSchedule::phase` field carries it, line 148 `pub fn phase(&self) -> SchedulePhase` exposes it to callers (re-exported at `lib.rs`); the surface-coverage test consumes the variant set. |
| REQ-2 | SHIPPED | impl: `pub struct ProfileSchedule` at `ProfileSchedule in ferrotorch-profiler/src/schedule.rs` with the 8 fields (config + runtime + callback) and manual `Debug` at line 92; non-test consumer: the type is re-exported at `lib.rs` and propagated to the meta-crate at `ferrotorch/src/lib.rs`; `tests/conformance_surface_coverage.rs` pins it in the surface contract. |
| REQ-3 | SHIPPED | impl: `pub fn new` at `new in ferrotorch-profiler/src/schedule.rs` with `assert!(active > 0)` and `assert!(repeat > 0)`, mirroring `torch/profiler/profiler.py:600-604`; non-test consumer: the constructor is the only entry point — every `ProfileSchedule` instance flows through it. Re-exported at `lib.rs` → `ferrotorch/src/lib.rs`. |
| REQ-4 | SHIPPED | impl: `pub fn step` at `step in ferrotorch-profiler/src/schedule.rs` advancing position + phase, firing callback at end-of-cycle, mirroring the schedule_fn closure at `torch/profiler/profiler.py:575-598`; non-test consumer: same prelude path; the surface contract at `tests/conformance_surface_coverage.rs` pins `ProfileSchedule::step`. |
| REQ-5 | SHIPPED | impl: `pub fn set_on_trace_ready` at `set_on_trace_ready in ferrotorch-profiler/src/schedule.rs` taking `impl FnMut(u64) + Send + 'static`; non-test consumer: re-exported at `lib.rs` and meta-crate prelude; the surface contract pins it. |
| REQ-6 | SHIPPED | impl: manual `impl Clone for ProfileSchedule` at `ferrotorch-profiler/src/schedule.rs:74` with explicit `on_trace_ready: None` on the clone and rationale comment at lines 42-52; non-test consumer: the `Clone` derive is required by the `pub use` propagation chain — any caller cloning a schedule transitively consumes this impl. The behavior is verified by the lib test `clone_preserves_config_and_position_drops_callback` at line 327. |
| REQ-7 | SHIPPED | impl: `phase` at line 148, `is_active` at line 154, `current_step` at line 160, `current_cycle` at line 166, all `#[must_use]`; non-test consumer: the accessors are part of the surface contract pinned by `tests/conformance_surface_coverage.rs:85-` and reachable through the meta-crate prelude — user code that drives a schedule outside `Profiler` flows through these getters. |
