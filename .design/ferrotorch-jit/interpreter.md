# ferrotorch-jit — `interpreter` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/fx/interpreter.py
  - torch/csrc/jit/runtime/interpreter.cpp
  - torch/_inductor/codegen/wrapper.py
-->

## Summary

`ferrotorch-jit/src/interpreter.rs` is the reference execution
backend for an `IrGraph`. It walks the graph in topological order
and dispatches each `IrOpKind` to the corresponding
`ferrotorch-core` op (`grad_fns::arithmetic::add`, `::mul`,
`grad_fns::activation::relu`, etc.). The interpreter is the
baseline that every other backend (CPU codegen, GPU codegen,
fused-kernel runtimes) is checked against — and the path the
graph-break / symbolic / AOT-autograd paths all converge on for
correctness. Mirrors the role of `torch.fx.Interpreter`
(`torch/fx/interpreter.py:60`) plus the C++ JIT interpreter
(`torch/csrc/jit/runtime/interpreter.cpp`).

## Requirements

- REQ-1: `pub fn interpret<T: Float>(graph: &IrGraph, inputs:
  &[Tensor<T>]) -> FerrotorchResult<Tensor<T>>` is the single-output
  entry point. Validates input count, walks topo order, returns the
  graph's first output value.
- REQ-2: `pub fn interpret_multi<T: Float>(graph, inputs) ->
  FerrotorchResult<Vec<Tensor<T>>>` is the multi-output variant.
  Returns the graph's `output_values` in declaration order. Used by
  AOT autograd / graph-break paths where the graph may legitimately
  produce a tuple of tensors.
- REQ-3: `pub fn interpret_multi_with_captures<T: Float>(graph,
  inputs, capture_indices: &[usize]) ->
  FerrotorchResult<(Vec<Tensor<T>>, Vec<Tensor<T>>)>` runs the graph
  AND captures the first output value of each topologically-indexed
  node into the second return slot. Used by
  `AotCompiledModule::forward_with_ctx` so the saved-by-index
  contract (`AotGraphPair::saved_tensor_indices`) is honoured in a
  single forward pass (audit #1110 finding-A).
- REQ-4: Every `IrOpKind` recognised by `graph.rs` is dispatched to
  a matching `ferrotorch-core` call OR — for higher-order ops that
  require lowering (`Cond`, `Scan`) — surfaces a clear
  "must be lowered before interpretation" error.
- REQ-5: The interpreter's results agree (within 1e-5 fp tolerance)
  with the eager `ferrotorch-core` execution that produced the
  graph via `trace`. This is the parity contract for the
  trace→interpret round-trip.

## Acceptance Criteria

- [x] AC-1: `interpret(&graph, &[input])` for a single-input,
  single-output graph returns one tensor of the expected shape.
- [x] AC-2: `interpret(&graph, &[])` for a graph with
  `input_count > 0` returns an error citing the count mismatch.
- [x] AC-3: `interpret_multi(&graph, &inputs)` for a 2-output graph
  returns a 2-tensor `Vec`.
- [x] AC-4: `interpret_multi_with_captures(&fwd, &[a, b], &[0, 1, 2])`
  returns the forward output AND the first output of each named
  topologically-indexed node — the canonical AOT-autograd contract.
- [x] AC-5: For the `sum(mul(a, b))` graph,
  `interpret_multi_with_captures` captures the Mul intermediate
  with value `[4, 10, 18]` at the `Mul`-node index.

## Architecture

`interpret` operates on a `HashMap<IrValueId, Tensor<T>>` (the
runtime value map) keyed by `IrValueId`. The Input nodes are seeded
with `inputs[index]`; each subsequent node in topo order computes
its output value(s) by:

1. Resolving its `IrNode::inputs` to the runtime tensors via the
   value map.
2. Dispatching on `IrNode::op` to the matching `ferrotorch-core`
   call (e.g. `IrOpKind::Add => grad_fns::arithmetic::add(&lhs,
   &rhs)?`, `IrOpKind::Mul`, `::Relu`, etc.).
3. Inserting the returned tensors into the value map under the
   `IrNode::outputs` IDs.

For `IrOpKind::Constant { data, shape }`, the interpreter
materialises a `Tensor<T>` via `Tensor::from_storage` (with the
appropriate dtype cast from `f64` to `T`).

For higher-order ops (`Cond`, `Scan`), the interpreter returns
`JitError::UnsupportedOp { op: "...".into() }` — these must be
lowered (currently a future-work item; documented at the
`IrOpKind` rustdoc in `graph.rs`).

`interpret_multi_with_captures` augments the topo walk with a
per-node check: if the node's topological index is in
`capture_indices`, push its first output tensor into the captures
vec. This is the primitive `AotCompiledModule::forward_with_ctx`
relies on (`module.rs:355-380`) — without it, the AOT contract
silently desynced (audit #1110 finding-A).

### Non-test production consumers

- `pub use interpreter::{interpret, interpret_multi,
  interpret_multi_with_captures}` at
  `ferrotorch-jit/src/lib.rs:109`.
- `ferrotorch-jit/src/module.rs:16, 125, 198, 363, 397` —
  `TracedModule::forward` / `forward_multi` and
  `AotCompiledModule::forward_with_ctx` / `backward` all dispatch
  through `interpret` and `interpret_multi_with_captures`.
- `ferrotorch-jit/src/symbolic.rs:63, 315` —
  `SymbolicTracedModule::forward_symbolic` calls
  `interpret(self.inner.graph(), inputs)` after the shape guard
  fires.
- `ferrotorch-jit/src/graph_break.rs` — `SegmentedModule::forward`
  threads tensor outputs through `interpret` (when the segment is
  `GraphSegment::Compiled`) or through the eager closure (when the
  segment is `GraphSegment::Eager`).
- `ferrotorch-jit/src/export.rs:162` — `ExportedProgram::run`
  delegates to `crate::interpret(&self.graph, inputs)`.

## Parity contract

`parity_ops = []`. The interpreter is the canonical
trace-correctness reference — every other backend is checked
against it. The numerical contract is whatever `ferrotorch-core`'s
ops produce.

Edge cases:

- **Empty graph** (no nodes): the input list is returned
  unchanged; `output_values.is_empty()` is an error from the
  caller's perspective.
- **Higher-order ops** (`Cond`, `Scan`): surface
  `JitError::UnsupportedOp` — these are placeholders requiring a
  lowering pass before the interpreter can execute them.

## Verification

Tests in `ferrotorch-jit/src/interpreter.rs` `mod tests` (smoke /
basic chain of ops) plus the comprehensive integration coverage
through `ferrotorch-jit/src/module.rs`'s 20+ tests (every test in
that file routes through `interpret` or
`interpret_multi_with_captures`).

Smoke command:

```bash
cargo test -p ferrotorch-jit --lib interpreter:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn interpret<T: Float>` in `interpreter.rs`; non-test consumer: `ferrotorch-jit/src/module.rs:125` (`TracedModule::forward_multi`), `module.rs:198` (`TracedModule::forward`), `module.rs:397` (`AotCompiledModule::backward`), `symbolic.rs:315`, `export.rs:162`. |
| REQ-2 | SHIPPED | impl: `pub fn interpret_multi<T: Float>` in `interpreter.rs`; non-test consumer: re-export at `lib.rs:109`. |
| REQ-3 | SHIPPED | impl: `pub fn interpret_multi_with_captures<T: Float>` in `interpreter.rs`; non-test consumer: `ferrotorch-jit/src/module.rs:363` `AotCompiledModule::forward_with_ctx` (the fix for audit #1110 finding-A). |
| REQ-4 | SHIPPED | impl: the `match node.op { ... }` dispatch arm-set in `interpreter.rs` covers every `IrOpKind` variant except the higher-order `Cond`/`Scan` which surface `JitError::UnsupportedOp`; non-test consumer: every interpreter call site listed above. |
| REQ-5 | SHIPPED | impl: the trace→interpret round-trip is the implementation strategy of `module::compile`; non-test consumer: `module.rs:276-280` `compile` traces then optimises then wraps in `TracedModule`, every subsequent `forward` call interprets the result and reproduces the eager numerics (tests in `module.rs` verify within 1e-5). |
