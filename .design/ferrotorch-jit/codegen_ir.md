# ferrotorch-jit — `codegen_ir` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/_inductor/ir.py
  - torch/_inductor/loop_body.py
  - torch/_inductor/codegen/common.py
-->

## Summary

`ferrotorch-jit/src/codegen_ir.rs` defines the low-level loop-based
intermediate representation (`LoopIR`, `Expr`, `BinOpKind`,
`UnaryOpKind`) and the lowering pass from the high-level `IrOpKind`
graph to that IR. Mirrors `torch._inductor.ir.Loops` /
`torch._inductor.loop_body.LoopBody` as the iteration-explicit IR
that sits between the graph and the per-target source emitters.

## Requirements

- REQ-1: `pub enum Expr` — scalar-expression AST with variants
  `Var`, `Const`, `IntConst`, `BinOp`, `UnaryOp`, `FnCall`,
  `Index`, `Cast`. Mirrors Inductor's `sympy.Expr`-backed
  expression layer in `ir.py`.

- REQ-2: `pub enum BinOpKind` — binary operators: `Add`, `Sub`,
  `Mul`, `Div`, `Mod`. Mirrors `torch._inductor.ir`'s `BinOp`.

- REQ-3: `pub enum UnaryOpKind` — unary operators: `Neg`, `Exp`,
  `Log`, `Sqrt`, `Abs`, `Sigmoid`, `Tanh`, `Relu`, `Gelu`, `Silu`.
  Mirrors Inductor's elementwise unary op set.

- REQ-4: `pub enum LoopIR` — statement layer with variants `Loop`,
  `Store`, `Let`, `Assign`, `Accumulate`, `If`, `Comment`. Mirrors
  `torch._inductor.loop_body.LoopBody` statement nodes.

- REQ-5: `Expr` builder helpers — `var`, `constant`, `int`, `bin`,
  `unary`, `index`, `call`, `sum`, `prod` shorthand constructors.

- REQ-6: `ir_op_to_unary` / `ir_op_to_binary` / `is_unary_elementwise` /
  `is_binary_elementwise` / `is_elementwise` / `is_reduction` —
  classification helpers mapping high-level `IrOpKind` into the
  low-level taxonomy.

- REQ-7: `pub fn lower_to_loops(ops, input_names, output_name,
  numel) -> Vec<LoopIR>` — the lowering entry point. Dispatches on
  op count + shape: single-op uses `lower_single_op`; all-elementwise
  fuses into one loop; mixed sequences lower op-by-op (with the
  previous output feeding the next input).

- REQ-8: Per-op lowering — `lower_unary_elementwise`,
  `lower_binary_elementwise`, `lower_sum_reduction`,
  `lower_mean_reduction`, `lower_prod_reduction`, `lower_matmul`,
  `lower_fused_elementwise`. Each produces a `Vec<LoopIR>` matching
  the upstream Inductor lowering for that op family.

- REQ-9: Chunked-accumulator reduction lowering (audit #1128) — Sum
  / Mean / Prod for `numel >= REDUCTION_CHUNK_THRESHOLD (64)` emit
  `REDUCTION_CHUNK_WIDTH (8)` parallel scalar accumulators that
  LLVM's autovectorizer can pack into a single vector register;
  below the threshold, fall back to a single scalar accumulator
  matching the pre-#1128 lowering.

## Acceptance Criteria

- [x] AC-1: `Expr::var("i")`, `Expr::constant(2.0)`, `Expr::int(8)`,
  `Expr::bin(BinOpKind::Add, ...)` all compile and produce the
  expected variants.
- [x] AC-2: `BinOpKind::Add` formats as `+` via `Display`;
  `UnaryOpKind::Tanh` formats as `tanh`.
- [x] AC-3: `ir_op_to_unary(&IrOpKind::Neg)` returns
  `Some(UnaryOpKind::Neg)`; `ir_op_to_unary(&IrOpKind::Add)`
  returns `None`.
- [x] AC-4: `lower_to_loops(&[IrOpKind::Neg], &["in0"], "out", 4)`
  returns a single `LoopIR::Loop` containing one `Store`.
- [x] AC-5: `lower_to_loops(&[IrOpKind::Sum], &["in0"], "out", 8)`
  returns the scalar-accumulator pattern (Sum below the chunk
  threshold).
- [x] AC-6: `lower_to_loops(&[IrOpKind::Sum], &["in0"], "out", 128)`
  returns the 8-wide chunked-accumulator pattern (Sum above the
  chunk threshold).
- [x] AC-7: `lower_matmul("a", "b", "out", 2, 3, 4)` returns a triple
  loop nest with `let mut acc` + `acc +=`.
- [x] AC-8: `lower_to_loops(&[IrOpKind::Neg, IrOpKind::Relu], &["in0"],
  "out", 4)` returns a single loop containing the fused chain.

## Architecture

### `Expr` AST (REQ-1, REQ-2, REQ-3, REQ-5)

`pub enum Expr` at `pub enum Expr in codegen_ir.rs` is the scalar
expression layer. `BinOpKind` and `UnaryOpKind` at
`pub enum BinOpKind in codegen_ir.rs` /
`pub enum UnaryOpKind in codegen_ir.rs` enumerate the operator
spaces. Builder helpers (`var`, `constant`, `int`, `bin`, `unary`,
`index`, `call`, `sum`, `prod`) at
`impl Expr in codegen_ir.rs` keep call sites readable.

### `LoopIR` statement layer (REQ-4)

`pub enum LoopIR` at `pub enum LoopIR in codegen_ir.rs` carries the
statement variants. Every variant is a documented one-line
construct (Loop, Store, Let, Assign, Accumulate, If, Comment)
that the per-target emitters dispatch on without further
interpretation.

### Classification + conversion helpers (REQ-6)

`fn ir_op_to_unary` at `fn ir_op_to_unary in codegen_ir.rs` and
`fn ir_op_to_binary` at `fn ir_op_to_binary in codegen_ir.rs` map
the high-level taxonomy. `fn is_unary_elementwise`,
`fn is_binary_elementwise`, `fn is_elementwise`, `fn is_reduction`
build on top of those.

### Lowering pipeline (REQ-7, REQ-8)

`pub fn lower_to_loops` at `pub fn lower_to_loops in codegen_ir.rs`
is the entry point. The dispatch is:

1. Empty ops → empty `Vec<LoopIR>`.
2. Single op → `lower_single_op`.
3. All-elementwise → `lower_fused_elementwise` (single loop).
4. Mixed → lower op-by-op with the previous output feeding the next
   input as a single buffer name.

Per-op lowering paths:
- `lower_unary_elementwise` → one flat loop with one Store.
- `lower_binary_elementwise` → one flat loop with one Store
  reading from `in0` and `in1`.
- `lower_sum_reduction` / `lower_mean_reduction` /
  `lower_prod_reduction` → see chunked path (REQ-9).
- `lower_matmul` (`pub fn lower_matmul`) → triple loop nest
  (`for i in 0..M`, `for j in 0..N`, `let mut acc = 0; for p in
  0..K { acc += a[i,p] * b[p,j] }; out[i,j] = acc`).
- `lower_fused_elementwise` → one loop with the chained Expr
  built by repeatedly applying `apply_op_expr` to a running
  expression.

### Chunked reductions (REQ-9)

`fn emit_chunked_reduction_prelude` at
`fn emit_chunked_reduction_prelude in codegen_ir.rs` switches on
`numel >= REDUCTION_CHUNK_THRESHOLD (64)` to choose between:

- Below threshold: single `let mut acc = init; for i in 0..numel
  { acc op= in[i]; }`.
- Above threshold: 8 parallel accumulators (`acc0..acc7`), a
  stride-8 outer loop, a horizontal combine via repeated `Expr::bin`,
  and a scalar tail loop for `numel % 8` elements via
  `emit_reduction_tail`. The 8-wide layout matches AVX2 f32 lanes /
  AVX-512 f64 lanes; LLVM's loop vectorizer packs the lane
  accumulators into a single vector register.

The `_finalize` step on `lower_mean_reduction` post-multiplies by
`1/numel` (or divides by `numel`); on Prod, the init is `1.0`
instead of `0.0`.

### Non-test production consumers

- `pub use codegen_ir::{BinOpKind, Expr, LoopIR, UnaryOpKind}` at
  `ferrotorch-jit/src/lib.rs:95` — grandfathered public API.
- `ferrotorch-jit/src/codegen_cpu.rs:12` `use crate::codegen_ir::
  {Expr, LoopIR, UnaryOpKind};` consumed by every Rust emission
  helper.
- `ferrotorch-jit/src/codegen_gpu.rs:43` `use crate::codegen_ir::
  {BinOpKind, Expr, LoopIR, UnaryOpKind};` consumed by every CUDA
  / PTX emission helper.
- `ferrotorch-jit/src/codegen_jit.rs:73` `use crate::codegen_ir::
  {BinOpKind, Expr, LoopIR, UnaryOpKind};` consumed by the
  cranelift JIT lowering pipeline.
- `ferrotorch-jit/src/dag_fusion.rs:32` `use crate::codegen_ir::
  {self, LoopIR};` calls `lower_to_loops` and `lower_matmul` for
  every fusion group via the `fuse_dag` lowering pipeline.

## Parity contract

`parity_ops = []`. This module is a target-agnostic IR layer;
parity is determined by how the per-target emitters consume it.
Numerical edge cases preserved across all lowering paths:

- **Reduction associativity** — Sum/Mean/Prod use the chunked
  8-wide accumulator layout above 64 elements. Float associativity
  is not guaranteed by IEEE 754, so the result may differ from a
  naive `sum(..)` by a few ULPs (matching upstream Inductor's
  vectorized lowering for the same shape).
- **Matmul accumulator** — the inner `p` loop uses
  `acc += a[i,p] * b[p,j]` accumulating into a fresh `acc = 0`;
  one `acc` per (i,j) output element. Matches the canonical
  textbook matmul lowering used by upstream Inductor on CPU.
- **Mean rounding** — `mean(x) = sum(x) / n` is computed
  post-reduction (not as a running average). This is the upstream
  convention and avoids the accumulation-bias of running averages.

## Verification

Tests in `mod tests in codegen_ir.rs`: every variant of Expr /
LoopIR has a construction test; every `lower_*` helper has a
shape-validation test asserting the produced `LoopIR` skeleton
matches the expected loop structure. The chunked-reduction
threshold transitions (`numel = 63` vs `numel = 64`) are pinned
explicitly.

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-jit --lib codegen_ir:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum Expr` in `codegen_ir.rs`; non-test consumer: re-export at `ferrotorch-jit/src/lib.rs:95` + `ferrotorch-jit/src/codegen_gpu.rs:43` + `ferrotorch-jit/src/codegen_cpu.rs:12` + `ferrotorch-jit/src/codegen_jit.rs:73`. |
| REQ-2 | SHIPPED | impl: `pub enum BinOpKind` in `codegen_ir.rs`; non-test consumer: re-export at `lib.rs:95` + `codegen_gpu.rs:43` + `codegen_jit.rs:73`. |
| REQ-3 | SHIPPED | impl: `pub enum UnaryOpKind` in `codegen_ir.rs`; non-test consumer: re-export at `lib.rs:95` + `codegen_cpu.rs:12` + `codegen_gpu.rs:43` + `codegen_jit.rs:73`. |
| REQ-4 | SHIPPED | impl: `pub enum LoopIR` in `codegen_ir.rs`; non-test consumer: re-export at `lib.rs:95` + `codegen_cpu.rs:12` + `codegen_gpu.rs:43` + `codegen_jit.rs:73`. |
| REQ-5 | SHIPPED | impl: builder methods on `impl Expr` (`var`, `constant`, `int`, `bin`, `unary`, `index`, `call`, `sum`, `prod`) in `codegen_ir.rs`; non-test consumer: every `lower_*` helper in this file builds expression trees via these helpers, which then flow through `codegen.rs:823` `crate::dag_fusion::find_fusion_groups` → `crate::dag_fusion::fuse_dag` → the per-target emitters. |
| REQ-6 | SHIPPED | impl: `pub fn ir_op_to_unary`, `pub fn ir_op_to_binary`, `pub fn is_unary_elementwise`, `pub fn is_binary_elementwise`, `pub fn is_elementwise`, `pub fn is_reduction` in `codegen_ir.rs`; non-test consumer: `lower_to_loops` calls `is_elementwise` to choose the fused-elementwise path; `lower_single_op` calls `is_unary_elementwise` / `is_binary_elementwise`. |
| REQ-7 | SHIPPED | impl: `pub fn lower_to_loops` in `codegen_ir.rs`; non-test consumer: `ferrotorch-jit/src/dag_fusion.rs:405` `codegen_ir::lower_to_loops(&group.ops, &in_refs, "out", numel)` from `fn lower_group`, which `InductorBackend::generate` drives for every fusion group. |
| REQ-8 | SHIPPED | impl: `lower_single_op`, `lower_unary_elementwise`, `lower_binary_elementwise`, `lower_sum_reduction`, `lower_mean_reduction`, `lower_prod_reduction`, `pub fn lower_matmul`, `lower_fused_elementwise` in `codegen_ir.rs`; non-test consumer: `dag_fusion.rs:416` calls `codegen_ir::lower_matmul("in0", "in1", "out", m, k, n)` from the MatMul arm of `lower_group`. |
| REQ-9 | SHIPPED | impl: `fn emit_chunked_reduction_prelude` + `REDUCTION_CHUNK_WIDTH (8)` + `REDUCTION_CHUNK_THRESHOLD (64)` constants in `codegen_ir.rs`; non-test consumer: `lower_sum_reduction` / `lower_mean_reduction` / `lower_prod_reduction` invoke it for every reduction lowered through `dag_fusion::fuse_dag` (transitively via `codegen.rs:823`). |
