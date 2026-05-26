# ferrotorch-jit — `codegen` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/_inductor/codegen/common.py
  - torch/_inductor/codegen/cpp.py
  - torch/_inductor/select_algorithm.py
  - torch/_inductor/scheduler.py
-->

## Summary

`ferrotorch-jit/src/codegen.rs` defines the `Codegen` trait and three
production backends (`InterpreterBackend`, `NativeBackend`,
`InductorBackend`) that compile an `IrGraph` into an executable
`CompiledGraph`. Mirrors `torch._inductor.codegen.common.PythonPrinter`
+ `torch._inductor.scheduler.Scheduler.codegen` + `select_algorithm`
backend selection at the conceptual level: each backend is a strategy
that converts a captured graph into callable native code (or, for the
interpreter, a captured closure).

## Requirements

- REQ-1: `pub trait Codegen` — a backend-agnostic compile-time strategy
  trait with `compile(&self, graph: &IrGraph) -> FerrotorchResult<CompiledGraph>`
  and `name(&self) -> &str`. Implementations are `Send + Sync + Debug`.
  Mirrors PyTorch Inductor's `WrapperCodegen` family (multiple per-target
  subclasses sharing an entry contract).

- REQ-2: `pub struct CompiledGraph` — execution closure
  `Box<dyn Fn(&[Vec<f64>]) -> FerrotorchResult<Vec<f64>>>` plus
  `num_inputs` and `output_shape`. `execute` validates input arity then
  invokes the closure. Mirrors `torch._inductor.codecache.PyCodeCache`'s
  compiled-fn wrapper.

- REQ-3: `pub struct InterpreterBackend` — fallback backend that wraps
  the existing IR interpreter as a `Codegen` impl. Used as a baseline
  and as the safety net for graphs the JIT path cannot handle.

- REQ-4: `pub struct NativeBackend` — composes Rust closures directly
  for simple elementwise / binary-with-constant / self-binary graphs,
  bypassing interpreter dispatch overhead. Falls back to
  `InterpreterBackend` for graphs it cannot natively compile.

- REQ-5: `pub enum InductorTarget` — the three code-emission targets:
  `CpuRust`, `GpuCuda`, `GpuPtx`. Selects the per-fusion-group emitter.

- REQ-6: `pub struct InductorBackend` — DAG-fusion → loop-IR → native
  source emitter pipeline. `generate()` returns source strings (one per
  fusion group). `compile()` JIT-compiles single-elementwise-group
  graphs via `codegen_jit` for `CpuRust`; raises
  `JitError::GpuBackendUnavailable` for GPU targets without runtime
  wiring; raises `JitError::UnsupportedOp` for non-JIT-able CpuRust
  graphs (callers must use `compile_with_status` to opt-in to fallback).

- REQ-7: `pub enum InductorCompileStatus` — structured return for
  `compile_with_status`: `Compiled(CompiledGraph)` or
  `FellBackToInterpreter(CompiledGraph)`. Audit #1110 requires the
  backend choice be observable at the call site; silent fallback through
  the `Codegen::compile` trait is forbidden.

- REQ-8: Per-fusion-group uniform-dtype enforcement (#729): GPU codegen
  refuses mixed-dtype groups with `JitError::CodegenError`. Mirrors
  Inductor's per-kernel single-scalar-width contract.

## Acceptance Criteria

- [x] AC-1: `InterpreterBackend.compile(&g)` returns a `CompiledGraph`
  whose `execute` matches the interpreter result element-by-element.
- [x] AC-2: `NativeBackend.compile(&g)` falls back to the interpreter
  for `Mm`/`Matmul` graphs and produces the correct matmul result.
- [x] AC-3: `InductorBackend::new(CpuRust).compile(&g)` for a single
  unary chain returns a JIT-compiled `CompiledGraph`.
- [x] AC-4: `InductorBackend::new(GpuPtx).compile(&g)` returns
  `Err(JitError::GpuBackendUnavailable)` (no runtime executor wired
  through `compile()`).
- [x] AC-5: `InductorBackend::new(CpuRust).compile_with_status(&g)`
  returns `InductorCompileStatus::FellBackToInterpreter(_)` for
  multi-group graphs.
- [x] AC-6: `InductorBackend::generate(&g)` returns one source string
  per fusion group; an identity graph still gets a `kernel_identity`
  stub.
- [x] AC-7: `InductorBackend` rejects mixed-dtype GPU groups with
  `JitError::CodegenError`.

## Architecture

### `Codegen` trait + `CompiledGraph` (REQ-1, REQ-2)

`pub trait Codegen` at `pub trait Codegen in codegen.rs` is the
extension point. `CompiledExecFn` (type alias for
`Box<dyn Fn(&[Vec<f64>]) -> FerrotorchResult<Vec<f64>> + Send + Sync>`)
backs `CompiledGraph::execute`. The struct holds the closure plus
`num_inputs` and `output_shape` so callers can validate before invoking
without rebuilding the graph.

### `InterpreterBackend` (REQ-3)

`pub struct InterpreterBackend` at
`pub struct InterpreterBackend in codegen.rs` implements `Codegen` by
capturing the graph in a closure, converting flat `Vec<f64>` inputs to
`Tensor<f64>` per execution, calling
`crate::interpreter::interpret(&graph, &tensors)`, and extracting the
result data. Input shapes are computed once at compile time via
`collect_input_shapes`.

### `NativeBackend` (REQ-4)

`pub struct NativeBackend` at `pub struct NativeBackend in codegen.rs`
walks the topological order classifying each value as `Input`,
`Constant`, `Computed(Vec<ElementwiseOp>)`, or `BinaryComputed { lhs,
rhs, binary, unary_chain }`. Three pattern families are supported:
unary chains, binary-with-constant (broadcast over the constant), and
self-binary (`x + x`). `try_compile_native` returns `None` for any
unsupported shape and the trait impl falls through to
`InterpreterBackend::compile`. Multi-element-constant ops carry a
positional `AtomicUsize` counter that is reset to 0 at the start of
each `execute` call (so element indexing is stable across invocations).

### `InductorBackend` (REQ-5, REQ-6, REQ-7, REQ-8)

`pub struct InductorBackend` at
`pub struct InductorBackend in codegen.rs` ties together
`crate::dag_fusion::find_fusion_groups` / `fuse_dag`, the per-target
emitters (`crate::codegen_cpu::CpuCodegen::generate_rust_source`,
`crate::codegen_gpu::GpuCodegen::generate_cuda_source`,
`crate::codegen_gpu::GpuCodegen::generate_ptx_source`), and the
in-process cranelift JIT path
`crate::codegen_jit::compile_loop_ir_kernel`. Three constructors:
`new(target)` (default `block_size = 256`), `with_block_size`. The
generate-then-validate flow at
`pub fn generate in codegen.rs` invokes the appropriate emitter for
every fusion group and feeds `resolve_group_dtype` to enforce REQ-8.

The `Codegen::compile` impl on `InductorBackend` refuses to silently
delegate to the interpreter (audit #1110) and instead returns
`Err(JitError::UnsupportedOp)` for non-JIT-able `CpuRust` graphs +
`Err(JitError::GpuBackendUnavailable)` for GPU targets without runtime
wiring. The structured `compile_with_status` variant exposes the
fallback explicitly as `InductorCompileStatus::FellBackToInterpreter`.

JIT path: `try_jit_compile_cpu_rust` at
`fn try_jit_compile_cpu_rust in codegen.rs` validates the graph is a
single elementwise fusion group with no aliased inputs, lowers it via
`fuse_dag`, and calls `compile_loop_ir_kernel(loops, num_inputs,
output_len)` to produce a `JitCompiledKernel`. The resulting
`CompiledGraph::execute` builds the per-call input-pointer list and
calls `kernel.execute(...)`.

### Non-test production consumers

- `pub use codegen::{Codegen, CompiledGraph, InductorBackend,
  InductorCompileStatus, InductorTarget, InterpreterBackend,
  NativeBackend}` at `ferrotorch-jit/src/lib.rs:89-92` — grandfathered
  public API surface.
- `ferrotorch-jit/src/autotune.rs:56` consumes `Codegen` and
  `CompiledGraph` via `use crate::codegen::{Codegen, CompiledGraph};`
  to dispatch candidate backends.

## Parity contract

`parity_ops = []`. This module is a backend-selection harness; it
does not own a parity-sweep op. Parity is determined by the underlying
emitters and the interpreter the fallback wraps. Numerical edge cases
preserved:

- **Fallback identity** — `NativeBackend.compile` on a graph it can't
  handle delegates to `InterpreterBackend.compile`; results must be
  bit-identical to the interpreter.
- **JIT vs interpreter parity** — Tests assert
  `JIT(InductorBackend(CpuRust)).execute(x) == InterpreterBackend.execute(x)`
  for the supported single-elementwise-group shape.
- **Mixed-dtype refusal** — A fusion group with mixed
  `Dtype::{F32, F64}` values is rejected (per upstream Inductor:
  one kernel = one scalar width).
- **NaN / Inf** — Pass-through. Elementwise closures use the
  builtin `f64` operations, so NaN propagation matches IEEE 754.

## Verification

Tests in `mod tests in codegen.rs`: interpreter/native consistency
tests (`test_native_matches_interpreter_unary_chain`,
`test_native_matches_interpreter_constant_sub`), inductor target tests
(`test_inductor_backend_name`, plus full integration tests that go
through `compile_with_status`), and zero-element edge-case tests.

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-jit --lib codegen:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub trait Codegen` in `codegen.rs`; non-test consumer: `ferrotorch-jit/src/autotune.rs:56` `use crate::codegen::{Codegen, CompiledGraph};` + the `AutotuneCandidate` field `backend: Box<dyn Codegen>`. |
| REQ-2 | SHIPPED | impl: `pub struct CompiledGraph` in `codegen.rs`; non-test consumer: re-export at `ferrotorch-jit/src/lib.rs:89-92` + `autotune.rs:56` (Autotuner returns `CompiledGraph` for the winning candidate). |
| REQ-3 | SHIPPED | impl: `pub struct InterpreterBackend` + `impl Codegen for InterpreterBackend` in `codegen.rs`; non-test consumer: re-export at `lib.rs:89-92` + used internally as the fallback target inside `NativeBackend::compile` and `InductorBackend::compile_with_status`. |
| REQ-4 | SHIPPED | impl: `pub struct NativeBackend` + `impl Codegen for NativeBackend` + `fn try_compile_native` in `codegen.rs`; non-test consumer: re-export at `lib.rs:89-92`. |
| REQ-5 | SHIPPED | impl: `pub enum InductorTarget { CpuRust, GpuCuda, GpuPtx }` in `codegen.rs`; non-test consumer: re-export at `lib.rs:89-92` + used by every `InductorBackend` constructor. |
| REQ-6 | SHIPPED | impl: `pub struct InductorBackend` + `impl Codegen` + `InductorBackend::generate` in `codegen.rs`; non-test consumer: re-export at `lib.rs:89-92`. Internal: `InductorBackend::compile` calls `crate::codegen_jit::compile_loop_ir_kernel` (the cranelift JIT path) for `CpuRust` and `crate::codegen_gpu::GpuCodegen::generate_{cuda,ptx}_source` for GPU targets. |
| REQ-7 | SHIPPED | impl: `pub enum InductorCompileStatus` + `impl InductorBackend::compile_with_status` in `codegen.rs`; non-test consumer: re-export at `lib.rs:89-92`. |
| REQ-8 | SHIPPED | impl: `fn resolve_group_dtype` in `codegen.rs` returning `Err(JitError::CodegenError)` on dtype mismatch; non-test consumer: `InductorBackend::generate` calls it for every GPU group before lowering. |
