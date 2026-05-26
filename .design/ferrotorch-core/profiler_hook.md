# Profiler hook

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - torch/profiler/profiler.py
  - torch/csrc/profiler/orchestration/observer.cpp
  - torch/csrc/profiler/orchestration/observer.h
  - torch/_torch_docs.py
  - torch/overrides.py
-->

## Summary

`ferrotorch-core/src/profiler_hook.rs` defines the trait-and-thread-local
seam between `ferrotorch-core` (low-level tensor ops) and `ferrotorch-profiler`
(the user-facing observer). `ferrotorch-core` cannot depend on
`ferrotorch-profiler` because the dependency runs the other way, so this
module exposes a minimal `OpProfiler` trait that the profiler crate
plugs into; tensor ops in core wrap themselves in `profile_op_scope`.
Mirrors the `RecordFunction` callback pattern in
`torch/csrc/profiler/orchestration/observer.h`.

## Requirements

- REQ-1: `trait OpProfiler: Send + Sync` — minimal callback surface
  (`record_op(name, category, shapes, duration_us)`). Mirrors the
  `at::RecordFunction::Callback` signature in
  `torch/csrc/profiler/orchestration/observer.cpp`.
- REQ-2: Thread-local `CURRENT: RefCell<Option<Arc<dyn OpProfiler>>>` —
  one active profiler per thread, settable via `set_current` and
  readable via `current`. Mirrors
  `torch.autograd.profiler._set_current_thread_function_metadata`.
- REQ-3: `profile_op_scope(name, category, shapes, f) -> R` — wraps a
  closure; when a profiler is active, times the closure with
  `Instant::now()` and reports the duration via `record_op`. When no
  profiler is active, the overhead is one thread-local read + one
  `Option::is_none` check. Mirrors `at::RecordFunction::Guard`.
- REQ-4: Nested op scopes work — an outer op body calling
  `profile_op_scope` from within its own scope records both the inner
  and outer ops (LIFO order: inner completes first).
- REQ-5: Setting a profiler on one thread does NOT propagate to
  another thread (per-thread isolation).

## Acceptance Criteria

- [x] AC-1: `current()` is `None` by default in any fresh thread
  (`profiler_hook.rs:126-134`).
- [x] AC-2: `profile_op_scope` with no profiler runs the closure and
  returns its value (`profiler_hook.rs:137-145`).
- [x] AC-3: `profile_op_scope` with an active profiler records exactly
  one event with the supplied name/category/shapes
  (`profiler_hook.rs:148-171`).
- [x] AC-4: Nested scopes record both inner and outer events in LIFO
  order (`profiler_hook.rs:187-209`).
- [x] AC-5: Cross-thread isolation: parent thread's profiler is
  invisible to a child `std::thread::spawn`
  (`profiler_hook.rs:211-226`).
- [x] AC-6: `cargo test -p ferrotorch-core --lib profiler_hook` passes.

## Architecture

- `trait OpProfiler` (`profiler_hook.rs:39-43`) — single method
  `record_op`; the trait is `Send + Sync` so profilers can be shared
  across threads.
- `CURRENT` thread-local (`profiler_hook.rs:45-52`) holds an
  `Arc<dyn OpProfiler>` so the profiler can outlive the closure that
  installed it.
- `set_current` (`profiler_hook.rs:59-61`) and `current`
  (`profiler_hook.rs:70-72`) are thin wrappers around the thread-local
  cell. `current` clones the `Arc` rather than holding the borrow for
  the closure body — this avoids deadlocks on nested ops.
- `profile_op_scope` (`profiler_hook.rs:87-101`) is the universal
  entry point; the hot path is the no-profiler branch (`f()` directly)
  so unprofile op execution pays only the cost of an atomic
  thread-local read.

The non-test production consumer for this module is
`ferrotorch-profiler` (downstream crate). Inside `ferrotorch-core` itself
the `profile_op_scope` calls are sprinkled across tensor op
implementations (e.g. `crate::grad_fns::arithmetic::add` wraps a
`profile_op_scope("add", "tensor_op", ...)`), so any op file that uses
the helper is a consumer.

## Parity contract

`parity_ops = []`. The profiler is a host-side observability shim; it
records timing metadata but does not change any tensor value. PyTorch's
profiler likewise has no numerical contract — it is purely diagnostic.

## Verification

- Unit tests at `profiler_hook.rs:103-228` exercise: no-profiler
  default, no-profiler closure invocation, single-event recording,
  setter-can-clear, nested scopes, and thread isolation.
- Indirect: every op that calls `profile_op_scope` exercises the
  no-profiler branch in normal test runs. Run:

  ```bash
  cargo test -p ferrotorch-core --lib profiler_hook
  ```

  Expected: 6 tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `trait OpProfiler` at `ferrotorch-core/src/profiler_hook.rs:39-43` mirrors `at::RecordFunction::Callback` in `torch/csrc/profiler/orchestration/observer.h`; non-test consumer: `ferrotorch-profiler` crate (downstream) implements this trait — the trait is the contract, and core-internal callers go through `profile_op_scope` rather than reading the trait directly. |
| REQ-2 | SHIPPED | impl: `CURRENT` thread-local at `ferrotorch-core/src/profiler_hook.rs:45-52`, `set_current` at `:59`, `current` at `:70`; non-test consumer: `ferrotorch-profiler`'s `with_profiler` guard pattern calls `set_current(Some(_))` on entry and `set_current(None)` on drop. |
| REQ-3 | SHIPPED | impl: `profile_op_scope` at `ferrotorch-core/src/profiler_hook.rs:87`; non-test consumer: ferrotorch-core op modules wrap their public ops (e.g. tensor ops in `grad_fns::arithmetic`) with `profile_op_scope(name, category, shapes, || { ... })` — the helper is the canonical entry point and is called from every instrumented op. |
| REQ-4 | SHIPPED | impl: `profile_op_scope`'s `Option::is_some` branch (`profiler_hook.rs:91-97`) supports re-entry because `current()` clones the `Arc` and releases the cell borrow before invoking the closure; non-test consumer: any composite op (e.g. `matmul` -> GEMM + bias-add) demonstrates nesting via `profile_op_scope` chained calls in production. The unit test at `:187-209` pins this behaviour explicitly. |
| REQ-5 | SHIPPED | impl: `thread_local!` keyword at `ferrotorch-core/src/profiler_hook.rs:45` enforces per-thread storage by construction; non-test consumer: rayon worker threads picking up tensor work get their own (initially empty) thread-local — there is no path by which a `set_current` on thread A could leak to thread B. The unit test at `:211-226` pins this. |
