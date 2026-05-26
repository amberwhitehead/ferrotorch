# ferrotorch-jit — `trace` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/jit/_trace.py
  - torch/fx/symbolic_trace.py
  - torch/_dynamo/__init__.py
-->

## Summary

`ferrotorch-jit/src/trace.rs` provides the `trace` entry point that
captures an `IrGraph` from a real forward execution. The strategy
is post-hoc: run the user-supplied function with real
`requires_grad` tensors, then walk the resulting autograd graph
(via `GradFn::inputs()`) backwards from the output to the leaves
and emit one `IrNode` per visited op. Mirrors the executor of
`torch.jit.trace` (`torch/jit/_trace.py:838-1042`) and the FX
`torch.fx.symbolic_trace` flow (`torch/fx/symbolic_trace.py:1100`)
— both build a graph by observing real execution.

## Requirements

- REQ-1: `pub fn trace<T, F>(f: F, example_inputs: &[Tensor<T>]) ->
  FerrotorchResult<IrGraph>` is the sole public entry point. It
  executes `f(example_inputs)`, derives the trace dtype from `T`
  via `Dtype::from_type_name(std::any::type_name::<T>())`, walks
  the autograd graph of the output tensor, and emits an `IrGraph`
  whose `input_values` align (in order) with `example_inputs` and
  whose `output_values` contain the single traced output.
- REQ-2: The tracer rejects unsupported `T` dtypes (e.g. `bf16`,
  `f16`, integer types) up front with a clear `InvalidArgument`
  error rather than producing a wrongly-tagged kernel downstream
  (#721-A safety guard).
- REQ-3: The tracer rejects outputs with no `grad_fn` (no autograd
  graph was built) with a tracing error.
- REQ-4: Each visited `GradFn` maps to an `IrOpKind` via a
  `map_name_to_op` table that must stay in sync with
  `graph_break.rs`'s `KNOWN_OP_NAMES` set; an unknown op-name
  surfaces `JitError::UnsupportedOp` (which through the `From`
  impl appears as a `FerrotorchError::InvalidArgument`).

## Acceptance Criteria

- [x] AC-1: `trace(|inputs| sum(&mul(&inputs[0], &inputs[1])?),
  &[a, b])` (where both inputs have `requires_grad`) returns an
  `IrGraph` whose topo order is `Input, Input, Mul, Sum`.
- [x] AC-2: `trace::<bf16, _>(...)` (any non-f32/f64 dtype) returns
  `Err(FerrotorchError::InvalidArgument)` containing the phrase
  "unsupported tensor dtype".
- [x] AC-3: `trace(|inputs| inputs[0].clone(), &[no_grad_tensor])`
  (no autograd graph built) returns an error.
- [x] AC-4: `trace(|inputs| relu(&inputs[0]), &[grad_tensor])` —
  the produced `IrGraph` has dtype-tagged values: each `IrValue`'s
  `dtype` matches the input tensor's element type.

## Architecture

`trace` runs in five conceptual steps (all inside the same function
body in `trace.rs`):

1. **Dtype resolution** — `std::any::type_name::<T>()` is fed to
   `Dtype::from_type_name`. Failure surfaces a fast `InvalidArgument`
   error citing #721-A.
2. **Forward execution** — call `f(example_inputs)`; propagate any
   error from the user function up unchanged.
3. **Autograd-graph traversal** — walk from `output.grad_fn()` back
   to the leaves. Each visited node provides its `inputs`
   (themselves `Tensor<T>` references with their own `grad_fn`), so
   the traversal is a topological reverse-walk.
4. **`IrNode` emission** — for each visited grad_fn, look up its
   name in `map_name_to_op` (a hand-maintained `&str ->
   IrOpKind` map; the canonical source must stay in sync with
   `graph_break.rs` per the comment at `graph_break.rs:35`). On
   miss, return `JitError::UnsupportedOp` converted to
   `FerrotorchError`. Build the `IrNode` via
   `IrGraph::add_node_with_dtype`, threading the resolved dtype.
5. **Output marking** — call `set_outputs` with the IDs corresponding
   to the output tensor's `IrValue`.

The tracer is intentionally simple — no proxy tensors, no
interpreter — and produces an exact transcription of what the
user's code actually executed. Mirrors the executor at
`torch/jit/_trace.py:838-1042` which similarly runs the model and
captures node-by-node.

### Non-test production consumers

- `pub use trace::trace` at `ferrotorch-jit/src/lib.rs:117` —
  public surface.
- `ferrotorch-jit/src/module.rs:18` — `use crate::trace::trace;`
  then `compile` calls `trace(f, example_inputs)?`.
- `ferrotorch-jit/src/symbolic.rs:66` —
  `use crate::trace::trace;` then `compile_symbolic` calls
  `trace(f, example_inputs)?`.
- `ferrotorch-jit/src/aot_autograd.rs:466` —
  `let mut graph = crate::trace::trace(f, example_inputs)?;` is
  the first step of `compile_aot`.
- `ferrotorch-jit/src/export.rs:21` — `use crate::trace;` then
  `export` calls into the tracer.

## Parity contract

`parity_ops = []`. The tracer is structural — it captures whatever
the user's code did. The captured graph's numerical behaviour is
the union of `ferrotorch-core`'s ops; parity of those is enforced
elsewhere.

Edge cases pinned in the test suite:

- **Empty graph** — tracing a function that returns one of its
  inputs unchanged (no autograd activity) is an error (REQ-3).
- **Unsupported dtype** — non-f32/f64 surfaces a structured error
  before any execution (REQ-2). The error message names
  `f32` / `f64` as the supported set.
- **`map_name_to_op` drift** — the op-name table must match
  `graph_break.rs`'s `KNOWN_OP_NAMES`. Drift is caught by the
  full-graph fail-fast in `graph_break.rs:600`.

## Verification

Tests in `ferrotorch-jit/src/trace.rs` `mod tests` exercise the
common forward shapes (add / mul / sum / relu / linear); each test
verifies the resulting `IrGraph` has the expected node count and
operation kinds. The integration tests in
`ferrotorch-jit/src/module.rs` exercise `trace` indirectly through
`compile`.

Smoke command:

```bash
cargo test -p ferrotorch-jit --lib trace:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn trace<T, F>` in `trace.rs`; non-test consumer: `ferrotorch-jit/src/module.rs:18, 276` (`compile` calls `trace(f, example_inputs)?`), `symbolic.rs:66`, `aot_autograd.rs:466`, `export.rs:21`. |
| REQ-2 | SHIPPED | impl: `Dtype::from_type_name(std::any::type_name::<T>())` check in `trace.rs` (~line 237); non-test consumer: every call site that monomorphises `T` on a non-`f32`/`f64` type. Pinned by `test_dtype_from_actual_type_name` in `graph.rs`. |
| REQ-3 | SHIPPED | impl: `output.grad_fn().ok_or_else(...)` guard in `trace.rs`; non-test consumer: `module.rs:276` and `symbolic.rs:404` rely on the error surface. |
| REQ-4 | SHIPPED | impl: `map_name_to_op` in `trace.rs`; non-test consumer: `graph_break.rs:35-600` comments pin that this table is the canonical source kept in sync with `KNOWN_OP_NAMES`. |
