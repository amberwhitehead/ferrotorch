# ferrotorch-jit — `memory_plan` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/_inductor/memory.py
  - torch/_inductor/scheduler.py
  - torch/csrc/jit/runtime/memory_planner.cpp
-->

## Summary

`ferrotorch-jit/src/memory_plan.rs` computes static buffer-slot
assignments for an `IrGraph` by analysing per-value liveness ranges
over the topological execution order. Mirrors
`torch._inductor.memory.MemoryPlanner` and the legacy
`torch/csrc/jit/runtime/memory_planner.cpp`: non-overlapping
values share the same physical slot, so the backend allocator
preallocates one buffer per slot instead of one per IR value.

## Requirements

- REQ-1: `pub struct MemoryPlan` — result of memory planning.
  Carries `assignments: HashMap<IrValueId, usize>` (value-id →
  slot), `slot_sizes: Vec<usize>` (size of each slot in
  elements), `num_slots: usize`, `naive_total: usize`,
  `planned_total: usize`.

- REQ-2: `MemoryPlan::savings_percent(&self) -> f64` — returns
  `(naive_total - planned_total) / naive_total * 100.0`. Returns
  `0.0` when `naive_total == 0` (avoids NaN from a divide-by-zero
  on empty graphs).

- REQ-3: `pub fn plan_memory(graph: &IrGraph) -> MemoryPlan` — the
  three-stage planner:
  1. **Liveness analysis** — per-value `(born, last_use)`
     topological indices.
  2. **Greedy first-fit allocation** — values in birth order pick
     the smallest free slot that fits, or allocate a new slot.
  3. **Statistics** — `num_slots`, `planned_total`, and
     `naive_total` for the savings comparison.

- REQ-4: Graph-output liveness — values that appear in
  `graph.output_values` have `last_use = max_topo` (the last
  topological index). They cannot be overwritten by later values.

- REQ-5: Empty-graph handling — `plan_memory(&IrGraph::new())`
  returns an empty plan with `num_slots = 0`, `naive_total = 0`,
  `planned_total = 0`, and no `assignments`.

- REQ-6: Determinism — ties between values with the same `born`
  index are broken by value-id ordering, so two plans for the
  same graph have identical slot assignments. Mirrors Inductor's
  deterministic memory-planning contract.

## Acceptance Criteria

- [x] AC-1: A chain `x[100] → relu → sigmoid → output` produces a
  plan with `num_slots < 3` (sequential intermediates share a
  slot).
- [x] AC-2: A diamond graph `x → {relu, sigmoid} → add → output`
  places `relu_out` and `sigmoid_out` in different slots (they
  are concurrently live).
- [x] AC-3: An empty graph returns a plan with `num_slots = 0`,
  `savings_percent() == 0.0`.
- [x] AC-4: Two graph outputs that are simultaneously live each
  occupy a distinct slot.
- [x] AC-5: A mixed-size graph (large input, small reshape
  output) places the small value in a larger slot when one is
  free.
- [x] AC-6: Every IR value in the graph appears in the
  `assignments` map, and every slot index is `< num_slots`.
- [x] AC-7: A 5-step long chain (1000 elements each) achieves
  `> 20%` savings via slot reuse.

## Architecture

### `MemoryPlan` (REQ-1, REQ-2)

`pub struct MemoryPlan` at `pub struct MemoryPlan in memory_plan.rs`
carries the assignment map, slot-size vector, totals, and a
`savings_percent()` accessor. `savings_percent` uses
`naive_total.saturating_sub(planned_total)` to avoid underflow
when planning increases total memory (which shouldn't happen but
is defended against).

### `plan_memory` (REQ-3, REQ-4)

`pub fn plan_memory` at `pub fn plan_memory in memory_plan.rs`
runs three stages:

**Stage 1 — Liveness analysis:**
- Walk `graph.values`; for each value with a `producer` node in
  the topological order, record `(born, last_use = born)`.
- Walk `graph.nodes`; for each input value, extend its `last_use`
  to `max(last_use, consumer's topo index)`.
- Walk `graph.output_values`; pin `last_use = max_topo` for any
  graph output (REQ-4).

**Stage 2 — Greedy first-fit allocation:**
- Sort values by `(born, value_id)` for deterministic ordering
  (REQ-6).
- For each value, scan existing slots; find the smallest slot
  whose `slot_sizes[i] >= value_size` AND whose `occupancy[i]`
  (latest `last_use` of any value previously assigned) is
  strictly less than this value's `born`.
- On match: assign + update `occupancy[slot_idx] =
  value.last_use`.
- On miss: allocate a new slot with `slot_sizes.push(size)`.

**Stage 3 — Statistics:**
- `num_slots = slot_sizes.len()`.
- `planned_total = slot_sizes.iter().sum()`.
- `naive_total` was accumulated during stage 2.

### Empty-graph handling (REQ-5)

`pub fn plan_memory` early-returns an empty `MemoryPlan` when
`graph.topological_order().is_empty()`. The `savings_percent`
helper short-circuits to `0.0` when `naive_total == 0` to avoid
NaN.

### `LiveInterval` + `value_num_elements` (helpers)

`struct LiveInterval { born: usize, last_use: usize }` at
`struct LiveInterval in memory_plan.rs` is the per-value liveness
summary. `fn value_num_elements(graph, value)` returns
`value.shape.iter().product()` (or `1` for scalars / unknown
values).

### Non-test production consumers

- `pub use memory_plan::{MemoryPlan, plan_memory}` at
  `ferrotorch-jit/src/lib.rs:110` — grandfathered public API.
- `ferrotorch-jit/src/optimize.rs` `use crate::memory_plan::
  {self, MemoryPlan};` — `pub fn optimize` returns
  `Some(memory_plan::plan_memory(graph))` when
  `config.memory_planning` is enabled.

## Parity contract

`parity_ops = []`. This is an analysis pass; correctness is
determined by:

1. **No-overlap invariant** — for any two values `(a, b)`
   assigned to the same slot, their liveness intervals are
   strictly disjoint (`a.last_use < b.born` or vice versa). The
   greedy first-fit enforces this constraint.
2. **Graph-output preservation** — graph outputs are pinned to
   the last topological index, so they cannot be overwritten by
   later non-output values. Tested in
   `test_graph_outputs_pinned_to_end`.
3. **Determinism** — sort-by-(born, value_id) means the same
   graph produces the same plan across runs.

## Verification

Tests in `mod tests in memory_plan.rs`:
`test_simple_chain_reuses_buffers` (chain savings),
`test_diamond_needs_two_concurrent_slots` (concurrency),
`test_savings_percentage` (long-chain savings),
`test_empty_graph` (empty-graph edge case),
`test_mixed_sizes` (small-in-large-slot reuse),
`test_graph_outputs_pinned_to_end` (REQ-4),
`test_all_values_assigned` (REQ-6 + sanity).

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-jit --lib memory_plan:: 2>&1 | tail -3
```

Expected: all 7 tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct MemoryPlan` in `memory_plan.rs`; non-test consumer: re-export at `memory_plan in ferrotorch-jit/src/lib.rs` + `plan_memory in ferrotorch-jit/src/optimize.rs` `Some(memory_plan::plan_memory(graph))` returns `Option<MemoryPlan>` from `pub fn optimize` which is consumed by `module in module.rs`, `optimize in aot_autograd.rs`, `optimize in symbolic.rs`, `graph_break in graph_break.rs` (all `use crate::optimize::{OptimizationConfig, optimize};`). |
| REQ-2 | SHIPPED | impl: `pub fn savings_percent` on `impl MemoryPlan` in `memory_plan.rs`; non-test consumer: re-export at `lib.rs:110` makes this accessible to any downstream consumer of the `MemoryPlan` produced by `optimize` — including the JIT module's compile path. |
| REQ-3 | SHIPPED | impl: `pub fn plan_memory` in `memory_plan.rs`; non-test consumer: `plan_memory in optimize.rs` `Some(memory_plan::plan_memory(graph))` inside `pub fn optimize`, which is called from `module.rs`, `aot_autograd.rs`, `symbolic.rs`, `graph_break.rs`. |
| REQ-4 | SHIPPED | impl: the graph-output `last_use = max_topo` pin in `pub fn plan_memory` (`memory_plan.rs`); non-test consumer: invoked transitively from `optimize in optimize.rs` for every plan whose graph has output values. |
| REQ-5 | SHIPPED | impl: empty-graph early return inside `pub fn plan_memory` (`memory_plan.rs`); non-test consumer: same as REQ-3. |
| REQ-6 | SHIPPED | impl: `values_by_birth.sort_by(\|a, b\| ia.born.cmp(&ib.born).then_with(\|\| a.0.cmp(&b.0)))` in `pub fn plan_memory` (`memory_plan.rs`); non-test consumer: same as REQ-3. |
