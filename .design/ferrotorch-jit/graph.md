# ferrotorch-jit — `graph` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/fx/graph.py
  - torch/fx/node.py
  - torch/fx/graph_module.py
  - torch/csrc/jit/ir/ir.h
-->

## Summary

`ferrotorch-jit/src/graph.rs` defines the JIT's high-level IR:
`IrGraph`, `IrNode`, `IrValue`, `IrValueId`, `IrNodeId`, `IrOpKind`,
and `Dtype`. The IR is the frontend representation consumed by every
optimisation pass, the interpreter, the codegen lowering, and the
serialiser. Mirrors the role of `torch.fx.Graph` /
`torch.fx.Node` (`torch/fx/graph.py:300`, `torch/fx/node.py`) and the
underlying `torch.jit.Graph` IR at `torch/csrc/jit/ir/ir.h:200-700`.

## Requirements

- REQ-1: `pub struct IrValueId(pub usize)` and `pub struct
  IrNodeId(pub usize)` are stable newtypes that identify edges and
  nodes in the graph. Hashable, comparable, copy.
- REQ-2: `pub enum IrOpKind` enumerates the op vocabulary
  recognised by the IR: `Input { index }`, `Constant { data, shape
  }`, `Output`, arithmetic (`Add` / `Sub` / `Mul` / `Div` / `Neg` /
  `Pow { exponent }` / `Sqrt` / `Abs` / `Exp` / `Log`), reduction
  (`Sum` / `Mean` / `Prod`), linalg (`Matmul` / `Mm` / `Mv` / `Dot`
  / `Transpose` / `Linear`), activations (`Relu` / `Sigmoid` /
  `Tanh` / `Gelu` / `Silu` / `Softmax` / `LogSoftmax`), shape
  (`Reshape { shape }` / `Flatten` / `Squeeze { axis }` /
  `Unsqueeze { axis }` / `Cat { axis }`), higher-order (`Cond` /
  `Scan`), and fused (`FusedElementwise { ops }` /
  `FusedLinearActivation { activation }` / `FusedAttention {
  head_dim }`). Mirrors `torch.fx.Node.op + target` taxonomy.
- REQ-3: `pub enum Dtype { F32, F64 }` is the IR-edge dtype marker
  (currently f32 / f64 only). `Dtype::name` returns the Rust
  primitive name; `Dtype::from_type_name` matches the
  `std::any::type_name::<T>()` output for the bare and
  stable-rustc-path-qualified variants so `trace.rs` can derive
  `Dtype` from the monomorphic `T`.
- REQ-4: `pub struct IrValue { id, shape, producer, dtype }` (with
  `#[non_exhaustive]`) is the per-edge metadata. Every value is
  produced by exactly one node (or `None` for graph inputs in the
  legacy path), carries a shape, and tags a `Dtype`.
- REQ-5: `pub struct IrNode { id, op, inputs, outputs }` (with
  `#[non_exhaustive]`) is the per-op record. Inputs and outputs
  are `Vec<IrValueId>` so multi-output ops (and `Cat` / `Cond`)
  fit the same shape.
- REQ-6: `pub struct IrGraph { nodes, values, input_values,
  output_values, cached_fingerprint }` is the IR container. Builder
  methods (`new`, `add_input` / `add_input_with_dtype`,
  `add_constant` / `add_constant_with_dtype`, `add_node` /
  `add_node_with_dtype`, `set_outputs`, `alloc_node_id`,
  `alloc_value_id`, `remove_node`) construct the graph; query
  methods (`node_count`, `value_count`, `topological_order`,
  `is_constant`, `fingerprint`) consume it.
- REQ-7: `IrGraph::topological_order` computes a Kahn's-algorithm
  topo order (producer before consumer) used by the interpreter,
  optimiser, fusion passes, and the fingerprint hasher.
- REQ-8: `IrGraph::fingerprint` returns a stable `u64` structural
  hash of the graph (op kinds in topo order + shapes + dtypes),
  cached in a `OnceLock` and invalidated by every mutation method,
  for the autotune cache (audit #1128).
- REQ-9: Default-dtype constructors (`add_input`, `add_constant`,
  `add_node`) tag values with `Dtype::F32` for backward
  compatibility; the `*_with_dtype` family propagates an explicit
  dtype to every output edge.

## Acceptance Criteria

- [x] AC-1: `IrGraph::new()` produces an empty graph with
  `node_count() == 0`, `value_count() == 0`.
- [x] AC-2: `add_input(vec![2, 3])` adds one Input node + one value
  and returns the value's id.
- [x] AC-3: `add_node(IrOpKind::Add, vec![x, y], vec![vec![3]])`
  appends an Add node and returns one output value id.
- [x] AC-4: `topological_order` orders producers before consumers
  (Input < Add < Relu in the canonical chain).
- [x] AC-5: `is_constant(c)` returns `true` for an id produced by
  `add_constant`; `false` for any other value.
- [x] AC-6: `Dtype::from_type_name("f32")` is `Some(Dtype::F32)`;
  `from_type_name("bf16")` is `None`.
- [x] AC-7: `Dtype::from_type_name(std::any::type_name::<f32>())`
  is `Some(Dtype::F32)` and the same for `f64` — the binding
  constraint trace.rs relies on.
- [x] AC-8: `add_input_with_dtype(vec![2], Dtype::F64)` tags the
  produced value with `Dtype::F64`.
- [x] AC-9: `IrGraph::fingerprint` is stable across reads (cached)
  and changes after a mutation (cache invalidated).
- [x] AC-10: `remove_node(id)` deletes the node and its produced
  values and cleans them out of `input_values`/`output_values`.

## Architecture

`IrGraph` carries four vectors: `nodes` (ordered insertion order),
`values` (same), `input_values` (declared inputs, in order), and
`output_values` (declared outputs). Two `usize` counters
(`next_value_id`, `next_node_id`) ensure ID uniqueness across
mutations. A `OnceLock<u64>` (`cached_fingerprint`) caches the
structural hash for the autotune lookup path.

`add_node_with_dtype` panics if `output_dtypes.len() !=
output_shapes.len()` — a programmer-error assertion that should
never be hit by external callers, since the builder is the only way
to construct nodes. The legacy `add_node` wraps this with
`Dtype::F32` defaults.

`Clone` for `IrGraph` is implemented manually so the cloned fingerprint
cache reuses the source's cached value (via a fresh `OnceLock::set`)
when present; otherwise the clone starts with an empty cell.
Invalidation funnels through `invalidate_fingerprint`, which every
mutation method calls.

`hash_op` (private) handles the `f64` payloads (`Constant`, `Pow`)
by feeding `.to_bits()` to the hasher — `f64` doesn't derive `Hash`.
For composite variants (`FusedElementwise`, `FusedLinearActivation`),
the function recurses.

`Dtype::from_type_name` accepts the bare primitive names plus the
stable rustc-emitted path-qualified variants
(`core::primitive::f32`, `std::primitive::f32`, `core::f32`,
`std::f32`, and the f64 equivalents). The `test_dtype_from_actual_type_name`
test pins the agreement with `std::any::type_name::<T>()` on the
current rustc.

### Non-test production consumers

- `pub use ferrotorch_jit::IrGraph` (re-exported transitively via
  multiple modules' re-exports of types that hold `IrGraph`).
- `ferrotorch-jit/src/module.rs:15` — `use crate::graph::IrGraph;`
- `ferrotorch-jit/src/optimize.rs:10` —
  `use crate::graph::{IrGraph, IrNode, IrNodeId, IrOpKind, IrValue, IrValueId};`
- `ferrotorch-jit/src/autotune.rs:57` —
  `use crate::graph::IrGraph;`
- `ferrotorch-jit/src/aot_autograd.rs:18` —
  `use crate::graph::{IrGraph, IrNodeId, IrOpKind, IrValueId};`
- `ferrotorch-jit/src/trace.rs:15` —
  `use crate::graph::{Dtype, IrGraph, IrOpKind, IrValueId};`
- `ferrotorch-jit/src/symbolic.rs` — uses
  `IrGraph::output_values` via the wrapped `TracedModule`.
- `ferrotorch-jit/src/export.rs` — holds an `IrGraph` field on
  `ExportedProgram`.
- `ferrotorch-jit/src/serialize.rs` — `impl IrGraph` `serialize`
  and `deserialize`.

## Parity contract

`parity_ops = []`. The IR is structural; it doesn't host numerical
ops directly. Its correctness is checked by every consumer
(interpreter, optimiser, codegen, serialiser) preserving the
contract.

Edge cases tracked in the design:

- **Dtype inference falls back to f32** when a constructor is
  called without an explicit dtype. The fallback is intentional for
  backward compatibility with the original f32-only API.
- **`is_constant` returns `false` for input values** (no producer
  node of `IrOpKind::Constant`).
- **`Cond` / `Scan` higher-order ops must be lowered** before the
  interpreter sees them — the interpreter cannot execute them
  directly. (Documented on the variants.)

## Verification

Tests in `ferrotorch-jit/src/graph.rs` `mod tests`:
`test_simple_graph_input_add_relu_output`,
`test_topological_order`, `test_add_constant_and_is_constant`,
`test_node_value_counts`, `test_dtype_from_type_name_stable_variants`,
`test_dtype_from_actual_type_name`,
`test_default_construction_is_f32`,
`test_explicit_dtype_construction`, `test_remove_node`.

Smoke command:

```bash
cargo test -p ferrotorch-jit --lib graph:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct IrValueId(pub usize)` and `pub struct IrNodeId(pub usize)` in `graph.rs`; non-test consumer: `optimize.rs:10` and 7 other modules import these ids. |
| REQ-2 | SHIPPED | impl: `pub enum IrOpKind` in `graph.rs`; non-test consumer: `trace.rs:15`, `optimize.rs:10`, `aot_autograd.rs:18` import it; `graph_break.rs`, `interpreter.rs` match on its variants. |
| REQ-3 | SHIPPED | impl: `pub enum Dtype` with `name` and `from_type_name` in `graph.rs`; non-test consumer: `trace.rs:237-245` resolves the trace dtype from `std::any::type_name::<T>()`. |
| REQ-4 | SHIPPED | impl: `#[non_exhaustive] pub struct IrValue { id, shape, producer, dtype }` in `graph.rs`; non-test consumer: `optimize.rs`, `interpreter.rs`, `codegen_*.rs` all read `.shape` and `.dtype` off `IrValue`s. |
| REQ-5 | SHIPPED | impl: `#[non_exhaustive] pub struct IrNode { id, op, inputs, outputs }` in `graph.rs`; non-test consumer: `optimize.rs`, `interpreter.rs`, `codegen.rs` traverse `IrGraph::nodes`. |
| REQ-6 | SHIPPED | impl: `pub struct IrGraph` + builder methods in `graph.rs`; non-test consumer: `module.rs:485-565` constructs graphs via these methods (also production code paths in `trace.rs`, `graph_break.rs`, `aot_autograd.rs` rely on them). |
| REQ-7 | SHIPPED | impl: `pub fn topological_order` in `graph.rs`; non-test consumer: `interpreter.rs` walks nodes in topo order; `optimize.rs` and `fusion.rs` use the order to schedule passes. |
| REQ-8 | SHIPPED | impl: `pub fn fingerprint` in `graph.rs` with `cached_fingerprint: OnceLock<u64>`; non-test consumer: `autotune.rs:57` imports `IrGraph` and keys the autotune cache by `fingerprint()` (audit #1128). |
| REQ-9 | SHIPPED | impl: `pub fn add_input`, `pub fn add_input_with_dtype`, `pub fn add_constant`, `pub fn add_constant_with_dtype`, `pub fn add_node`, `pub fn add_node_with_dtype` in `graph.rs`; non-test consumer: `trace.rs` builds graphs through the dtype-aware constructors. |
