# ferrotorch-jit — `dag_fusion` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/_inductor/scheduler.py
  - torch/_inductor/fx_passes/group_batch_fusion.py
  - torch/_inductor/codegen/common.py
-->

## Summary

`ferrotorch-jit/src/dag_fusion.rs` performs DAG-level fusion over an
`IrGraph`: it identifies connected subgraphs of compatible
elementwise / reduction / matmul ops and groups them into
`FusionGroup`s that can be lowered to a single fused `LoopIR`
program. Mirrors `torch._inductor.scheduler.Scheduler.fuse_nodes` /
the `group_batch_fusion` FX pass: ops with compatible iteration
domains merge; reductions, matmuls, and shape ops form group
boundaries.

## Requirements

- REQ-1: `pub struct FusionGroup` — a group of ops sharing a single
  fused-kernel lowering. Carries `node_ids: Vec<IrNodeId>`,
  `ops: Vec<IrOpKind>`, `external_inputs: Vec<IrValueId>`,
  `external_outputs: Vec<IrValueId>`, and
  `kind: FusionGroupKind`. Topologically sorted.

- REQ-2: `pub enum FusionGroupKind` — five variants:
  `Elementwise` (unary + binary fuse together), `Reduction`,
  `MatMul`, `Linear`, `Opaque` (shape ops, softmax, etc.).
  Determines the lowering strategy.

- REQ-3: `pub fn find_fusion_groups(graph: &IrGraph) ->
  Vec<FusionGroup>` — walks the topological order assigning each
  non-Input/Constant/Output node to a group. Returns groups in
  topological order. Fusion is greedy: an elementwise node joins
  the group of its single producer when safe (no fan-out across
  group boundary, no cycles, intermediate value not externally
  consumed unless explicitly external).

- REQ-4: `pub fn fuse_dag(groups: &[FusionGroup], graph: &IrGraph) ->
  Vec<Vec<LoopIR>>` — lowers each `FusionGroup` to a `LoopIR`
  program. Elementwise → single fused loop; Reduction → accumulator
  + loop; MatMul → triple loop nest; Linear/Opaque → comment node
  (handled by the per-target emitter directly).

- REQ-5: Op classification — `fn classify_op(op: &IrOpKind) ->
  FusionGroupKind` maps every `IrOpKind` variant into the group
  kind taxonomy. Reductions terminate fusion; linalg ops are
  standalone; shape ops are `Opaque`.

- REQ-6: Safe fusion — `find_mergeable_group` rejects merges that
  would (a) require an intermediate value with multiple
  consumers across group boundaries, (b) create a cycle, or (c)
  mix elementwise with non-elementwise kinds. The result is the
  same node-grouping invariants Inductor's `Scheduler.fuse_nodes`
  enforces.

- REQ-7: External I/O computation — for each group, identifies
  inputs that come from outside the group (graph inputs, constants,
  or other groups' outputs) and outputs consumed outside the group
  (or that are graph outputs).

- REQ-8: Shape estimation helpers — `estimate_numel_for_inputs` and
  `estimate_matmul_dims` infer the kernel iteration shape from the
  group's external input values. Used by `fuse_dag` to size the
  emitted loop bounds.

## Acceptance Criteria

- [x] AC-1: A single `IrOpKind::Relu` produces one `FusionGroup` of
  kind `Elementwise` with `ops.len() == 1`.
- [x] AC-2: A chain `Neg → Relu → Sigmoid` produces one
  `FusionGroup` with `ops.len() == 3`.
- [x] AC-3: A `Relu → Sum` chain produces two groups
  (`Elementwise`, `Reduction`).
- [x] AC-4: A `Mm → Relu` produces two groups (`MatMul`,
  `Elementwise`).
- [x] AC-5: A graph where a value has two consumers (fan-out)
  produces ≥ 2 groups (fusion cannot cross the fan-out).
- [x] AC-6: `external_inputs` and `external_outputs` are correctly
  populated for every group.
- [x] AC-7: `fuse_dag` returns one `Vec<LoopIR>` per group, with
  non-empty lowering for Elementwise/Reduction/MatMul groups.
- [x] AC-8: `classify_op(&IrOpKind::Add) ==
  FusionGroupKind::Elementwise`; `classify_op(&IrOpKind::Sum) ==
  FusionGroupKind::Reduction`; `classify_op(&IrOpKind::Mm) ==
  FusionGroupKind::MatMul`.

## Architecture

### `FusionGroup` + `FusionGroupKind` (REQ-1, REQ-2)

`pub struct FusionGroup` at
`pub struct FusionGroup in dag_fusion.rs` carries the group's
node IDs, op sequence, external inputs/outputs, and kind. The kind
controls how `fuse_dag` lowers the group.

### Group discovery (REQ-3, REQ-5, REQ-6, REQ-7)

`pub fn find_fusion_groups` at
`pub fn find_fusion_groups in dag_fusion.rs` walks the topological
order. For each non-Input/Constant/Output node:

1. Classify via `fn classify_op`.
2. If elementwise, call `fn find_mergeable_group` to try joining
   an existing producer group. If safe, merge; otherwise, create
   a new group.
3. Non-elementwise nodes always create their own group.

After all nodes are assigned, compute `external_inputs` and
`external_outputs` per group by scanning each node's inputs /
outputs and checking against the group's internal value set
(REQ-7).

`fn find_mergeable_group` at
`fn find_mergeable_group in dag_fusion.rs` enforces the safety
rules: (a) the producer must be in an Elementwise group, (b) the
intermediate value can only be safely fused if all consumers are
in the same group OR if the value is also a graph output (in
which case the group will carry an additional external output),
(c) no cycle is created (the new node's outputs must not
transitively reach any existing group member).

### Lowering (REQ-4)

`pub fn fuse_dag` at `pub fn fuse_dag in dag_fusion.rs` walks the
groups and dispatches to `fn lower_group`. For each group:

- Elementwise → `fn lower_elementwise_group` calls
  `codegen_ir::lower_to_loops(&group.ops, &in_refs, "out",
  numel)`.
- Reduction (single op) → `codegen_ir::lower_to_loops` for the
  reduction op (multi-op reductions are not yet supported and
  emit a Comment placeholder).
- MatMul → `codegen_ir::lower_matmul("in0", "in1", "out", m, k,
  n)` with `(m, k, n)` from `estimate_matmul_dims`.
- Linear / Opaque → emit a Comment placeholder. The per-target
  emitter handles these directly (e.g. by calling the underlying
  Linear/Softmax/Reshape kernel at runtime).

### Shape estimation (REQ-8)

`fn estimate_numel_for_inputs` at
`fn estimate_numel_for_inputs in dag_fusion.rs` returns the first
non-zero `shape.iter().product()` from the group's external
inputs (fallback `1` for scalars). `fn estimate_matmul_dims`
returns `(M, K, N)` from a 2-input group whose two inputs are
both 2-D shapes (fallback `(1, 1, 1)`).

### Non-test production consumers

- `pub use dag_fusion::{FusionGroup, FusionGroupKind}` at
  `ferrotorch-jit/src/lib.rs:97` — grandfathered public API.
- `ferrotorch-jit/src/codegen.rs:823` calls
  `crate::dag_fusion::find_fusion_groups(graph)` from
  `InductorBackend::generate`.
- `ferrotorch-jit/src/codegen.rs:824` calls
  `crate::dag_fusion::fuse_dag(&groups, graph)` for the
  lowering pass.
- `ferrotorch-jit/src/codegen.rs:1201` `use
  crate::dag_fusion::{FusionGroupKind, find_fusion_groups,
  fuse_dag};` inside `try_jit_compile_cpu_rust` — the JIT path's
  shape validation (single Elementwise group) calls these.

## Parity contract

`parity_ops = []`. This is a fusion-planning pass; the output is
the same value graph reshaped into fewer kernels. Numerical edge
cases preserved:

- **Fusion preserves results bit-for-bit on CPU** — the fused
  loop applies the same chain of f64 operations the
  unfused interpreter would, so the result must be bit-identical
  (assuming the interpreter doesn't reassociate independently).
  Tests pin this for unary chains and binary-with-constant
  shapes.
- **Reduction boundaries** — reductions never fuse with following
  ops (the partial sum cannot feed a per-element transform until
  it's complete). Mirrors Inductor's reduction-boundary rule.
- **Fan-out blocks fusion** — when an intermediate value has
  multiple consumers, it cannot be hidden inside a fused kernel
  (the second consumer would need to read it). Mirrors Inductor's
  multi-consumer rule.

## Verification

Tests in `mod tests in dag_fusion.rs`: classification tests
(`test_classify_elementwise`, `test_classify_reduction`,
`test_classify_matmul`, `test_classify_opaque`), fusion topology
tests (`test_chain_fuses_into_one_group`,
`test_branch_prevents_merge`, `test_reduction_breaks_group`,
`test_matmul_standalone_group`), and lowering tests
(`test_fuse_dag_elementwise_group`, `test_fuse_dag_matmul_group`,
`test_fuse_dag_reduction_group`).

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-jit --lib dag_fusion:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct FusionGroup` in `dag_fusion.rs`; non-test consumer: re-export at `dag_fusion in ferrotorch-jit/src/lib.rs` + `ferrotorch-jit/src/codegen.rs` iterates `for (i, (group, loops)) in groups.iter().zip(loops_per_group.iter()).enumerate()` reading `group.external_inputs` etc. |
| REQ-2 | SHIPPED | impl: `pub enum FusionGroupKind` in `dag_fusion.rs`; non-test consumer: re-export at `lib.rs` + `codegen in codegen.rs` `if group.kind != FusionGroupKind::Elementwise { return Ok(None); }` in `try_jit_compile_cpu_rust`. |
| REQ-3 | SHIPPED | impl: `pub fn find_fusion_groups` in `dag_fusion.rs`; non-test consumer: `find_fusion_groups in codegen.rs` `let groups = crate::dag_fusion::find_fusion_groups(graph);` (every InductorBackend::generate call) + `codegen in codegen.rs` (every InductorBackend::compile CpuRust JIT-path call). |
| REQ-4 | SHIPPED | impl: `pub fn fuse_dag` in `dag_fusion.rs`; non-test consumer: `fuse_dag in codegen.rs` `let loops_per_group = crate::dag_fusion::fuse_dag(&groups, graph);` + `codegen in codegen.rs` `let loops_per_group = fuse_dag(&groups, graph);` in the JIT path. |
| REQ-5 | SHIPPED | impl: `fn classify_op` in `dag_fusion.rs`; non-test consumer: invoked by `find_fusion_groups` once per non-Input/Constant/Output node before assignment to a group. |
| REQ-6 | SHIPPED | impl: `fn find_mergeable_group` in `dag_fusion.rs`; non-test consumer: invoked by `find_fusion_groups` for every elementwise node before group assignment. |
| REQ-7 | SHIPPED | impl: the external-inputs / external-outputs scan loop at the bottom of `pub fn find_fusion_groups` in `dag_fusion.rs`; non-test consumer: `find_fusion_groups in codegen.rs` reads `group.external_inputs.len()` to size the per-kernel input count + `codegen in codegen.rs` uses `group.external_inputs` to map runtime sources. |
| REQ-8 | SHIPPED | impl: `fn estimate_numel_for_inputs` + `fn estimate_matmul_dims` in `dag_fusion.rs`; non-test consumer: called by `fn lower_group` (transitively via `fuse_dag` from `codegen in codegen.rs`). |
