# ferrotorch-jit — `graph_break` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/_dynamo/output_graph.py
  - torch/_dynamo/symbolic_convert.py
  - torch/_dynamo/exc.py
-->

## Summary

`ferrotorch-jit/src/graph_break.rs` handles graph breaks during
tracing — the situation where the tracer encounters an op whose
`GradFn::name()` isn't in the known op-mapping table. Instead of
failing outright, it inserts a *graph break*: split execution into
segments where each segment is either a compiled `TracedModule`
subgraph OR an eager-mode fallback closure. The `SegmentedModule`
threads them in order. When `CompileConfig::fullgraph == true`,
graph breaks are rejected instead (mirrors
`torch.compile(model, fullgraph=True)`).

Mirrors `torch._dynamo.output_graph.OutputGraph` and the
graph-break logic in `torch/_dynamo/symbolic_convert.py:600-1500`,
plus the user-facing `fullgraph=True` enforcement.

## Requirements

- REQ-1: `pub enum GraphSegment<T: Float>` is the per-segment
  marker: `Compiled(TracedModule<T>)` for a JIT-compiled subgraph,
  `Eager(Box<dyn Fn(&Tensor<T>) -> FerrotorchResult<Tensor<T>>>)`
  for a closure that runs the op in eager mode.
- REQ-2: `pub struct SegmentedModule<T: Float>` wraps a
  `Vec<GraphSegment<T>>`. `forward(input)` threads the tensor
  output of one segment as the input to the next.
- REQ-3: `SegmentedModule::segments`, `segment_count`,
  `is_fully_compiled` are accessors for diagnostics.
- REQ-4: `pub enum TraceResult<T: Float>` is the result of
  `trace_with_breaks`: `Compiled(TracedModule<T>)` if no graph
  breaks were needed, `Segmented(SegmentedModule<T>)` if breaks
  produced multiple segments.
- REQ-5: `pub fn trace_with_breaks<T, F>(f, example_inputs,
  config: CompileConfig) -> FerrotorchResult<TraceResult<T>>`
  attempts to trace; on unknown-op encounters, splits into segments;
  on `config.fullgraph == true`, rejects the break with
  `JitError::GraphBreak`.
- REQ-6: A canonical `KNOWN_OP_NAMES` set inside the module
  controls which `GradFn` names map to recognised ops. Must stay in
  sync with `trace::map_name_to_op` (documented at
  `graph_break in graph_break.rs` and `graph_break in graph_break.rs`).

## Acceptance Criteria

- [x] AC-1: `SegmentedModule::new(vec![GraphSegment::Compiled(m1),
  GraphSegment::Compiled(m2)])` constructs.
- [x] AC-2: `segmented.forward(&input)` threads each segment's
  output to the next segment's input.
- [x] AC-3: `is_fully_compiled` returns `true` when every segment
  is `Compiled`, `false` when any segment is `Eager`.
- [x] AC-4: `trace_with_breaks` for a forward containing only known
  ops returns `TraceResult::Compiled(TracedModule)`.
- [x] AC-5: `trace_with_breaks` for a forward containing an
  unknown op (e.g. a custom autograd Fn whose name isn't in
  `KNOWN_OP_NAMES`) returns `TraceResult::Segmented(SegmentedModule)`
  when `config.fullgraph == false`.
- [x] AC-6: `trace_with_breaks` for that same forward returns
  `Err(JitError::GraphBreak { ... })` when `config.fullgraph ==
  true`.

## Architecture

`GraphSegment<T>` is a non-exhaustive enum. The `Compiled` variant
holds a `TracedModule<T>` (single-input forward); the `Eager`
variant holds a `Box<dyn Fn>` closure that runs the op directly via
`ferrotorch_core::grad_fns`. A manual `Debug` impl prints the
segment kind + node count for the compiled case.

`SegmentedModule<T>` wraps `Vec<GraphSegment<T>>`. `forward(input)`
walks the segments, calling `m.forward(...)` on the compiled
segments and the closure on eager segments, threading the output
through. The `is_fully_compiled` predicate is the key diagnostic
for "did we successfully fold this whole forward into one compiled
graph?".

`TraceResult<T>` is a non-exhaustive enum with two variants:
`Compiled(TracedModule<T>)` and `Segmented(SegmentedModule<T>)`. The
caller dispatches on the result.

`trace_with_breaks` is implemented in stages:

1. Run the forward to build the autograd graph.
2. Walk the autograd graph, mapping each grad_fn name to a known op
   via `KNOWN_OP_NAMES` (`name in graph_break.rs`).
3. On unknown op:
   - If `config.fullgraph == true`, surface `JitError::GraphBreak`.
   - Otherwise: cut the graph at this op, build a `TracedModule` for
     the prefix, build an eager closure for the unknown op, and
     recurse on the suffix. The resulting segments are concatenated.
4. If every op is recognised, return
   `TraceResult::Compiled(TracedModule::new(optimized))`.

The full-graph mode is the strict mode users opt into with
`CompileConfig { fullgraph: true, ... }`; it's the equivalent of
`torch.compile(model, fullgraph=True)` (which raises rather than
falling back).

### Non-test production consumers

- `pub use graph_break::{GraphSegment, SegmentedModule, TraceResult,
  trace_with_breaks}` at `ferrotorch-jit/src/lib.rs:108`.
- The `CompileConfig::fullgraph` flag is wired here: while the
  `compile` path doesn't currently call `trace_with_breaks` (it
  goes through `trace` directly), the public surface
  (`trace_with_breaks`) is the path users take when they want
  graph-break tolerance.

## Parity contract

`parity_ops = []`. Graph-break behaviour is structural — the
numerical result of a segmented forward is exactly what the eager
forward would produce, by construction.

Edge cases:

- **`fullgraph = true` + any unknown op** — error, no segmented
  fallback.
- **Empty segment list** — a no-op forward (`forward(input)` just
  returns input).
- **`KNOWN_OP_NAMES` drift vs `trace::map_name_to_op`** — both
  tables must list the same op-name set. The comment at
  `graph_break in graph_break.rs` pins this contract; drift is a defect.

## Verification

Tests in `ferrotorch-jit/src/graph_break.rs` `mod tests` cover:
SegmentedModule construction + forward (multi-segment threading,
fully-compiled path, mixed compiled+eager path), trace_with_breaks
happy path, trace_with_breaks fullgraph rejection,
trace_with_breaks segmentation on unknown op,
is_fully_compiled predicate.

Smoke command:

```bash
cargo test -p ferrotorch-jit --lib graph_break:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum GraphSegment<T: Float>` in `graph_break.rs`; non-test consumer: `pub struct SegmentedModule<T>` (`SegmentedModule in graph_break.rs`) holds `Vec<GraphSegment<T>>`; `forward` dispatches on the variant. |
| REQ-2 | SHIPPED | impl: `pub struct SegmentedModule<T: Float>` + `pub fn new` + `pub fn forward` in `graph_break.rs`; non-test consumer: re-export at `lib.rs`; `pub fn trace_with_breaks` returns it inside `TraceResult::Segmented`. |
| REQ-3 | SHIPPED | impl: `pub fn segment_count`, `pub fn segments`, `pub fn is_fully_compiled` in `graph_break.rs`; non-test consumer: re-export at `lib.rs`. |
| REQ-4 | SHIPPED | impl: `pub enum TraceResult<T: Float>` in `graph_break.rs`; non-test consumer: `pub fn trace_with_breaks` returns this type. |
| REQ-5 | SHIPPED | impl: `pub fn trace_with_breaks<T, F>` in `graph_break.rs`; non-test consumer: re-export at `lib.rs:108`. |
| REQ-6 | SHIPPED | impl: `KNOWN_OP_NAMES` set / `map_name_to_op_kind` inside `graph_break.rs` (mirrors `trace::map_name_to_op` per the comment at `lib.rs` and `lib.rs`); non-test consumer: `pub fn trace_with_breaks` consults it during the autograd-graph walk. |
