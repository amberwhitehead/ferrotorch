# ferrotorch-jit — `autotune` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/_inductor/select_algorithm.py
  - torch/_inductor/runtime/triton_heuristics.py
  - torch/_inductor/autotune_process.py
-->

## Summary

`ferrotorch-jit/src/autotune.rs` benchmarks candidate `Codegen`
backends against a traced graph and picks the fastest one. Results
are cached keyed by a shape-fingerprint so the same problem only
gets timed once. Mirrors PyTorch Inductor's
`torch._inductor.select_algorithm.AlgorithmSelectorCache` +
`runtime.triton_heuristics`: the user supplies candidates (different
backends, different configs, different block sizes) and the tuner
returns the fastest one with the winning compiled graph.

## Requirements

- REQ-1: `pub struct AutotuneKey` — stable cache key combining a
  graph's structural fingerprint
  (`u64` from `IrGraph::fingerprint`) with the input shapes
  (`Vec<Vec<usize>>`). `PartialEq + Eq + Hash + Clone + Debug`.

- REQ-2: `AutotuneKey::from_graph(graph, input_shapes) -> Self`
  reads the cached `u64` fingerprint off the graph (audit #1128:
  no per-call rebuild of the full structural walk) and appends the
  input shapes.

- REQ-3: `pub struct AutotuneCandidate` — a `(name: String,
  backend: Box<dyn Codegen>)` pair. Constructor: `new(name,
  backend)`.

- REQ-4: `pub struct AutotuneResult` — outcome of a single `tune`
  call. Carries `winner_name`, `winner_time: Duration`,
  `winner_compiled: CompiledGraph`, and
  `all_timings: Vec<(String, Duration)>` (one row per timed
  candidate; cache hits contain only the winner row).

- REQ-5: `pub struct Autotuner` — the tuner state. Holds
  `candidates: Vec<AutotuneCandidate>`, `iterations: usize`
  (default 5), `warmup: usize` (default 1), and `cache:
  Mutex<HashMap<AutotuneKey, (String, Duration)>>`. Builder
  methods: `new`, `with_candidate`, `with_iterations`,
  `with_warmup`. Accessors: `iterations`, `warmup`,
  `candidate_count`, `cached`, `cache_size`, `clear_cache`.

- REQ-6: `Autotuner::tune(&self, graph, inputs) ->
  FerrotorchResult<AutotuneResult>` — the main entry point.
  Derives input shapes from the graph (O(values) one-shot map
  build per audit #1128), builds the cache key, and either:
  - **Cache hit**: re-compile the winning candidate, return a
    one-row `AutotuneResult`.
  - **Cache miss**: compile + warm up + median-time every
    candidate, pick the fastest, cache the decision, re-compile
    the winner for the return value, return the full timing
    table.

- REQ-7: Median timing — each candidate runs `iterations` timed
  passes (after `warmup` discarded passes); the median elapsed
  time is the candidate's score.

- REQ-8: Empty-candidate handling — `tune` returns
  `Err(InvalidArgument)` ("no candidates registered") when
  `self.candidates.is_empty()`.

- REQ-9: Cache-miss-after-rename — if a cached candidate name no
  longer exists in the current candidate list, return
  `Err(InvalidArgument)` ("cached winner ... is not among current
  candidates") rather than silently picking another candidate.

## Acceptance Criteria

- [x] AC-1: An empty tuner returns `Err(InvalidArgument)` on
  `tune`.
- [x] AC-2: Two candidates (`InterpreterBackend`, `NativeBackend`)
  on a `Relu → Sqrt` chain return a winner whose
  `winner_compiled().execute(...)` yields the expected scalar
  output.
- [x] AC-3: First `tune` call records 2 timing rows; second call
  for the same graph + shapes returns 1 timing row (cache hit).
- [x] AC-4: After `clear_cache()`, a re-tune returns 2 timing
  rows again.
- [x] AC-5: Two graphs with the same op but different input
  shapes produce different `AutotuneKey` values.
- [x] AC-6: Two graphs with the same shape but different ops
  produce different `AutotuneKey` values.
- [x] AC-7: `Autotuner::new().with_iterations(7).with_warmup(3)`
  carries those configs.
- [x] AC-8: `with_iterations(0)` panics (assertion).
- [x] AC-9: A single-candidate tune returns 1 timing row with
  that candidate as the winner.
- [x] AC-10: Two `AutotuneKey::from_graph` calls for the same
  graph return equal keys; mutating the graph changes the
  fingerprint.

## Architecture

### `AutotuneKey` (REQ-1, REQ-2)

`pub struct AutotuneKey` at
`pub struct AutotuneKey in autotune.rs` carries
`graph_fingerprint: u64` (from `IrGraph::fingerprint`, which is
lazily cached on the graph object) and `input_shapes:
Vec<Vec<usize>>`. The audit-#1128 invariant: the hot path is a
single `u64` read plus the input-shape slice clone, not a
per-call stringification of every op.

### `AutotuneCandidate` + `AutotuneResult` (REQ-3, REQ-4)

`pub struct AutotuneCandidate` at
`pub struct AutotuneCandidate in autotune.rs` is the `(name,
boxed-backend)` pair. `pub struct AutotuneResult` at
`pub struct AutotuneResult in autotune.rs` carries the winner's
name + time + compiled graph plus the full timing table.

### `Autotuner` + `tune` (REQ-5, REQ-6, REQ-7, REQ-8, REQ-9)

`pub struct Autotuner` at `pub struct Autotuner in autotune.rs`
holds the candidate list, timing config, and the cache
`Mutex<HashMap<AutotuneKey, (String, Duration)>>`. The cache
stores only the *decision* (which candidate to use), not the
compiled graph itself (since `CompiledGraph` is not `Clone`).

`pub fn tune` at `impl Autotuner in autotune.rs` runs:

1. Validate `!candidates.is_empty()` (REQ-8); else
   `Err(InvalidArgument)`.
2. Derive `input_shapes` via a single `HashMap<value_id, shape>`
   build (audit #1128: O(values) one-shot, not O(inputs ×
   values) per call).
3. `let key = AutotuneKey::from_graph(graph, &input_shapes);`.
4. Cache lookup: if hit, re-compile the winning candidate via
   `candidate.backend.compile(graph)?`, return a one-row
   `AutotuneResult`. Mismatched-name returns
   `Err(InvalidArgument)` (REQ-9).
5. Cache miss: for each candidate, compile + run `warmup`
   discarded passes + run `iterations` timed passes, collect into
   `samples: Vec<Duration>`, sort, take median, push to
   `all_timings`. Track the best `(idx, time)`.
6. Insert the decision into the cache.
7. Re-compile the winner for the return value (the first
   compilation was consumed during timing).
8. Return the `AutotuneResult` with the full timing table.

### Configuration (REQ-7)

`with_iterations(n)` asserts `n > 0` (panicking on 0) — the
median calculation requires at least one sample. `with_warmup(n)`
accepts 0 (no warmup discards). Median = `samples[samples.len() /
2]` after sorting.

### Non-test production consumers

- `pub use autotune::{AutotuneCandidate, AutotuneKey,
  AutotuneResult, Autotuner}` at
  `ferrotorch-jit/src/lib.rs:88` — grandfathered public API.

This module is a leaf consumer of `crate::codegen::{Codegen,
CompiledGraph}` and `crate::graph::IrGraph`; it has no internal
ferrotorch-jit downstream consumers (the lib re-export is the
public boundary, per S5 of goal.md).

## Parity contract

`parity_ops = []`. This is a benchmark-and-pick harness; it
produces no values of its own. Numerical invariants:

- **Winning result equals interpreter result** — all candidates
  must produce the same `Vec<f64>` output for the same input.
  Tested in `test_autotune_picks_a_winner_from_two_candidates`
  (which validates the winning compiled graph's output against
  the known scalar result).
- **Cache stability** — `AutotuneKey::from_graph` reads the
  cached `u64` fingerprint; two builds with the same graph
  produce equal keys (REQ-1, REQ-10).
- **Audit-#1128 invariant** — the cache key build path is
  O(values) one-shot map build + O(inputs) lookup, not the prior
  O(inputs × values) walk per `tune` call.

## Verification

Tests in `mod tests in autotune.rs`:
`test_autotune_empty_candidates_errors` (REQ-8),
`test_autotune_picks_a_winner_from_two_candidates` (full
benchmark + verify),
`test_autotune_cache_hit_returns_single_timing_row` (REQ-6 cache
path),
`test_autotune_key_is_shape_sensitive` (REQ-1),
`test_autotune_key_is_op_sensitive` (REQ-1),
`test_autotune_honors_iterations_and_warmup_config` (REQ-5),
`test_autotune_rejects_zero_iterations` (panic),
`test_autotune_with_single_candidate_still_works`,
`test_autotune_key_uses_cached_fingerprint` (audit #1128).

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-jit --lib autotune:: 2>&1 | tail -3
```

Expected: all 9 tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct AutotuneKey` in `autotune.rs`; non-test consumer: re-export at `ferrotorch-jit/src/lib.rs:88` — this is the `Autotuner::tune` cache-key type, used internally by every cache lookup. |
| REQ-2 | SHIPPED | impl: `pub fn from_graph` on `impl AutotuneKey` in `autotune.rs`; non-test consumer: `autotune.rs::tune` calls `let key = AutotuneKey::from_graph(graph, &input_shapes);` on every `Autotuner::tune` invocation. |
| REQ-3 | SHIPPED | impl: `pub struct AutotuneCandidate` + `pub fn new` in `autotune.rs`; non-test consumer: re-export at `lib.rs:88` + `Autotuner::with_candidate` constructs candidates via `AutotuneCandidate::new(name, backend)`. |
| REQ-4 | SHIPPED | impl: `pub struct AutotuneResult` + the `winner_name` / `winner_time` / `winner_compiled` / `all_timings` accessors in `autotune.rs`; non-test consumer: re-export at `lib.rs:88` — the public return type of `Autotuner::tune`. |
| REQ-5 | SHIPPED | impl: `pub struct Autotuner` with `candidates` / `iterations` / `warmup` / `cache` fields + all builder methods (`new`, `with_candidate`, `with_iterations`, `with_warmup`) and accessors (`iterations`, `warmup`, `candidate_count`, `cached`, `cache_size`, `clear_cache`) in `autotune.rs`; non-test consumer: re-export at `lib.rs:88` — this is the canonical public type for kernel-tuning workflows. |
| REQ-6 | SHIPPED | impl: `pub fn tune` on `impl Autotuner` in `autotune.rs` covering both the cache-hit (single-row result + winner re-compile) and cache-miss (full benchmark + record decision + re-compile winner) paths; non-test consumer: re-export at `lib.rs:88` makes `Autotuner::tune` the public benchmark entry point. |
| REQ-7 | SHIPPED | impl: the `samples.sort(); samples[samples.len() / 2]` median computation inside `pub fn tune` (`autotune.rs`); non-test consumer: invoked on every cache-miss `tune` call (via `lib.rs:88`). |
| REQ-8 | SHIPPED | impl: `if self.candidates.is_empty() { return Err(FerrotorchError::InvalidArgument { ... }); }` at the top of `pub fn tune` in `autotune.rs`; non-test consumer: same path as REQ-6. |
| REQ-9 | SHIPPED | impl: `let candidate = self.candidates.iter().find(\|c\| c.name == cached_name).ok_or_else(\|\| FerrotorchError::InvalidArgument { ... })?;` inside the cache-hit branch of `pub fn tune` (`autotune.rs`); non-test consumer: triggered on any `tune` call whose cache holds a name not in the current candidate list. |
