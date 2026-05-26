# ferrotorch-jit — `optimize` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/_inductor/fx_passes/post_grad.py
  - torch/_inductor/fx_passes/group_batch_fusion.py
  - torch/_inductor/constant_folding.py
  - torch/fx/passes/dce.py
-->

## Summary

`ferrotorch-jit/src/optimize.rs` runs the configurable graph
optimization pipeline over an `IrGraph` before codegen: constant
folding, dead-code elimination, operator fusion (pattern + linear
elementwise), and memory planning. Mirrors
`torch._inductor.fx_passes.post_grad.post_grad_passes` + the
`constant_folding` + `dce` passes upstream, gated by
`OptimizationConfig` toggles that mirror
`inductor.config.{constant_folding, dead_code_elimination,
operator_fusion, memory_planning}`.

## Requirements

- REQ-1: `pub struct OptimizationConfig` — four-bool config
  carrying `constant_folding`, `dead_code_elimination`,
  `operator_fusion`, `memory_planning`. Each toggle gates one
  pipeline stage. `Default::default()` enables all four.

- REQ-2: `pub fn optimize(graph: &mut IrGraph, config:
  &OptimizationConfig) -> Option<MemoryPlan>` — runs enabled
  passes in canonical order (CF → DCE → pattern-fuse →
  elementwise-fuse → memory-plan). Returns `Some(MemoryPlan)`
  when memory planning is enabled, `None` otherwise.

- REQ-3: `pub fn constant_fold(graph: &mut IrGraph)` — folds
  subgraphs whose inputs are all `Constant` nodes into a single
  `Constant` node, iterating to a fixed point. Auto-chains DCE
  after the final fold to sweep orphan input-constants (matches
  `torch.fx`/Inductor `constant_folding` behaviour, fix for #885).

- REQ-4: `pub fn dead_code_eliminate(graph: &mut IrGraph)` —
  removes nodes whose outputs are not consumed by any other node
  and not graph outputs. Iterates to a fixed point. `Input` nodes
  are never dead.

- REQ-5: `pub fn pattern_fuse(graph: &mut IrGraph)` — recognizes
  high-level patterns and rewrites them as fused ops:
  `Linear → Activation` → `FusedLinearActivation`;
  `Matmul → Mul/Div → Softmax → Matmul` → `FusedAttention`.

- REQ-6: `pub fn fuse_elementwise(graph: &mut IrGraph)` — fuses
  chains of unary elementwise ops where every intermediate has
  exactly one consumer into a single
  `IrOpKind::FusedElementwise { ops }` node. Iterates to a fixed
  point.

- REQ-7: Dtype preservation on constant folding — the replacement
  `Constant` node carries the original value's dtype, so
  folding does not silently change an edge's dtype.

- REQ-8: Graph-output preservation on constant folding — when a
  folded node's output value was also a graph output, the
  replacement `Constant` node's output is restored into
  `graph.output_values` after `remove_node` strips it.

## Acceptance Criteria

- [x] AC-1: `OptimizationConfig::default()` has all four flags
  enabled.
- [x] AC-2: `optimize(&mut g, &Default::default())` on a graph
  with memory planning enabled returns `Some(MemoryPlan)`.
- [x] AC-3: `constant_fold` on `Constant + Constant → Add` rewrites
  to a single `Constant` whose data is the elementwise sum.
- [x] AC-4: `dead_code_eliminate` removes unreferenced
  intermediate nodes.
- [x] AC-5: `pattern_fuse` on `Linear → Relu` produces a single
  `FusedLinearActivation { activation: Relu }` node.
- [x] AC-6: `fuse_elementwise` on `Neg → Relu → Sigmoid` chain
  produces one `FusedElementwise { ops }` node.
- [x] AC-7: Constant-folding a value that is also a graph output
  preserves it in `output_values` and its dtype is unchanged.
- [x] AC-8: `pattern_fuse` on `Mm → Mul → Softmax → Mm`
  produces one `FusedAttention` node.

## Architecture

### `OptimizationConfig` + `optimize` (REQ-1, REQ-2)

`pub struct OptimizationConfig` at
`pub struct OptimizationConfig in optimize.rs` carries the four
bool toggles with `#[allow(clippy::struct_excessive_bools, reason
= "mirrors PyTorch's inductor.config.* style")]`. `Default::default()`
sets all four to `true`.

`pub fn optimize` at `pub fn optimize in optimize.rs` runs:

1. `if config.constant_folding { constant_fold(graph); }`
2. `if config.dead_code_elimination { dead_code_eliminate(graph); }`
3. `if config.operator_fusion { pattern_fuse(graph); fuse_elementwise(graph); }`
4. `if config.memory_planning { Some(memory_plan::plan_memory(graph)) } else { None }`

### Constant folding (REQ-3, REQ-7, REQ-8)

`pub fn constant_fold` at `pub fn constant_fold in optimize.rs`
loops until no fold happens. Each iteration:

1. Snapshot node IDs (so we can mutate during iteration).
2. For each node where `is_simple_elementwise(&node.op)` AND
   every input value is produced by a `Constant` node:
   a. Capture the original output value's dtype (REQ-7).
   b. Note whether the output is a graph output (REQ-8).
   c. Call `eval_elementwise(&op, &constant_inputs)` to compute
      the folded data.
   d. `graph.remove_node(node_id)` (also strips outputs from
      `output_values`).
   e. Re-insert an `IrValue` with the same `IrValueId`, original
      dtype, and a new `producer = const_node_id`.
   f. Insert a new `Constant` node with the folded data + shape.
   g. Restore graph-output membership if applicable (REQ-8).
3. After fixed point reached, run `dead_code_eliminate(graph)` to
   sweep orphan input-constants (fix for #885).

`fn eval_elementwise` at `fn eval_elementwise in optimize.rs`
handles the simple elementwise op space (`Add`, `Sub`, `Mul`,
`Div`, `Neg`, `Relu`, `Sigmoid`, `Tanh`). Length mismatch
returns `None` (cannot fold).

### Dead-code elimination (REQ-4)

`pub fn dead_code_eliminate` at
`pub fn dead_code_eliminate in optimize.rs` loops until no
dead nodes remain. Per iteration:
1. Collect all live value IDs: outputs of every node consumed by
   another node, plus all graph outputs.
2. Find nodes whose every output is not live AND whose op is not
   `Input { .. }`.
3. `graph.remove_node(id)` for each.

### Pattern fusion (REQ-5)

`pub fn pattern_fuse` at `pub fn pattern_fuse in optimize.rs`
delegates to `fn fuse_linear_activation` and
`fn fuse_attention_pattern`.

`fn fuse_linear_activation` walks topo order, finds each `Linear`
node, checks that the sole consumer is an activation (`Relu`,
`Gelu`, `Silu`, `Sigmoid`, `Tanh`), and rewrites the Linear's op
to `FusedLinearActivation { activation: Box::new(act_op) }` while
removing the activation node.

`fn fuse_attention_pattern` walks topo order looking for
`Matmul → Mul/Div → Softmax → Matmul` sequences and rewrites them
to a single `FusedAttention` node.

### Elementwise fusion (REQ-6)

`pub fn fuse_elementwise` at
`pub fn fuse_elementwise in optimize.rs` loops on
`fn try_fuse_one_chain` until no chain ≥ 2 remains.

`fn try_fuse_one_chain` builds a consumer-count map, finds the
start of a fusible chain (a unary elementwise node whose single
output has exactly one consumer which is also unary
elementwise), walks the chain forward, and rewrites it via
`fn fuse_chain` into a single `FusedElementwise { ops }` node.

`fn is_fusable_elementwise` at
`fn is_fusable_elementwise in optimize.rs` restricts fusion to
unary ops (Neg, Relu, Sigmoid, Tanh, Sqrt, Abs, Gelu, Silu, Exp,
Log, Pow). Binary ops are excluded because the
`FusedElementwise` interpreter path applies ops sequentially on
a single tensor.

### Non-test production consumers

- `pub use optimize::{OptimizationConfig, optimize}` at
  `ferrotorch-jit/src/lib.rs:112` — grandfathered public API.
- `ferrotorch-jit/src/module.rs:17` `use crate::optimize::
  {OptimizationConfig, optimize};` — `pub fn compile_with_config`
  calls `optimize(&mut graph, &config.optimization)?` before
  codegen.
- `ferrotorch-jit/src/aot_autograd.rs:19` `use
  crate::optimize::{OptimizationConfig, optimize};` —
  `pub fn compile_aot` runs optimization on the decomposed
  forward + backward graphs.
- `ferrotorch-jit/src/symbolic.rs:65` `use
  crate::optimize::{OptimizationConfig, optimize};` —
  `compile_symbolic` reruns optimization for each symbolic shape
  specialisation.
- `ferrotorch-jit/src/graph_break.rs:27` `use
  crate::optimize::optimize;` — `pub fn trace_with_breaks` runs
  optimization on every segment.

## Parity contract

`parity_ops = []`. This is a graph-rewrite pass; correctness is
determined by:

- **Constant-fold equivalence** — the folded constant's data
  must equal the elementwise op applied to the input constants
  (tested in `test_constant_fold_add`,
  `test_constant_fold_chained`, etc.). Dtype is preserved
  (REQ-7).
- **DCE preserves outputs** — graph outputs are always live, so
  DCE never removes a node producing a graph output.
- **Fusion preserves semantics** — `FusedElementwise { ops }`
  applied to an input must match applying each op separately,
  to within f64 precision.
- **Pattern-fusion mirrors upstream** — `FusedLinearActivation`
  fuses `Linear → Activation` matching upstream `torch.fx`'s
  `fuse_linear_activation` rewrite.

## Verification

Tests in `mod tests in optimize.rs`:
`test_constant_fold_add`,
`test_constant_fold_iterates_to_fixed_point`,
`test_constant_fold_preserves_graph_output`,
`test_constant_fold_preserves_dtype`,
`test_dead_code_eliminate_removes_unused`,
`test_dead_code_eliminate_preserves_outputs`,
`test_fuse_linear_relu`, `test_fuse_attention_pattern`,
`test_fuse_elementwise_chain`,
`test_optimize_full_pipeline_returns_memory_plan`.

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-jit --lib optimize:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct OptimizationConfig` + `impl Default for OptimizationConfig` in `optimize.rs`; non-test consumer: re-export at `ferrotorch-jit/src/lib.rs:112` + `ferrotorch-jit/src/module.rs:17` `use crate::optimize::{OptimizationConfig, optimize};` + `ferrotorch-jit/src/aot_autograd.rs:19` + `ferrotorch-jit/src/symbolic.rs:65`. |
| REQ-2 | SHIPPED | impl: `pub fn optimize` in `optimize.rs`; non-test consumer: `ferrotorch-jit/src/module.rs::compile_with_config` calls `optimize(&mut graph, &config.optimization)?` + `aot_autograd.rs` calls `optimize(&mut forward, &config)?` on the decomposed graphs + `symbolic.rs::compile_symbolic` + `graph_break.rs:27`. |
| REQ-3 | SHIPPED | impl: `pub fn constant_fold` in `optimize.rs`; non-test consumer: invoked by `pub fn optimize` when `config.constant_folding` is true; that public fn is called from all four downstream modules (module / aot_autograd / symbolic / graph_break). |
| REQ-4 | SHIPPED | impl: `pub fn dead_code_eliminate` in `optimize.rs`; non-test consumer: invoked by `pub fn optimize` when `config.dead_code_elimination` is true + also chained at the tail of `constant_fold` per #885 (so it runs at least once whenever CF runs). |
| REQ-5 | SHIPPED | impl: `pub fn pattern_fuse` delegating to `fn fuse_linear_activation` + `fn fuse_attention_pattern` in `optimize.rs`; non-test consumer: invoked by `pub fn optimize` when `config.operator_fusion` is true. |
| REQ-6 | SHIPPED | impl: `pub fn fuse_elementwise` + `fn try_fuse_one_chain` + `fn fuse_chain` in `optimize.rs`; non-test consumer: invoked by `pub fn optimize` when `config.operator_fusion` is true. |
| REQ-7 | SHIPPED | impl: `preserved_dtype` capture + restore inside `pub fn constant_fold` in `optimize.rs`; non-test consumer: transitive via `pub fn optimize` (REQ-3 chain). |
| REQ-8 | SHIPPED | impl: `is_graph_output` check + post-replacement restore inside `pub fn constant_fold` in `optimize.rs`; non-test consumer: transitive via `pub fn optimize` (REQ-3 chain). |
