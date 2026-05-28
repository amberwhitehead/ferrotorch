# ferrotorch-jit — `codegen_cpu` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/_inductor/codegen/cpp.py
  - torch/_inductor/codegen/cpp_template_kernel.py
  - torch/_inductor/codegen/common.py
-->

## Summary

`ferrotorch-jit/src/codegen_cpu.rs` emits Rust source code from a
`LoopIR` program. Mirrors `torch._inductor.codegen.cpp.CppKernel`'s
role as the CPU-side source emitter, but produces idiomatic Rust
(`#[inline(always)]` + `f64::{exp,ln,sqrt,...}` method dispatch)
rather than C++. The generated source is consumed downstream by
`codegen_jit::compile_loop_ir_kernel`, which lowers it through
cranelift's JIT (not rustc).

## Requirements

- REQ-1: `pub struct CpuCodegen` — zero-sized type holding the
  emission entry points. Mirrors Inductor's `CppKernel` as an
  emitter object.

- REQ-2: `pub fn generate_rust_source(loops, fn_name) -> String` —
  emits a `#[inline(always)] pub unsafe fn <name>(inputs: &[&[f64]],
  output: &mut [f64])` Rust function from a slice of `LoopIR`
  statements.

- REQ-3: SIMD-friendly emission — inner loops with no nested loops
  get a `// SIMD: sequential access, no dependencies` comment; large
  outer loops (`>= PARALLEL_THRESHOLD == 1024`) with nested loops get
  a `// NOTE: candidate for rayon par_iter (n=<N>)` annotation.

- REQ-4: Buffer name mapping — `in0`, `in1`, ... map to
  `inputs[0]`, `inputs[1]`, ...; `out` / `output` map to `output`;
  arbitrary names pass through verbatim.

- REQ-5: Special-float literal handling — `f64::INFINITY`,
  `f64::NEG_INFINITY`, `f64::NAN`, `0.0`, `1.0` get canonical
  literal forms; all other values format as `<v>_f64`.

- REQ-6: Unary op emission — every `UnaryOpKind` variant
  (`Neg`, `Exp`, `Log`, `Sqrt`, `Abs`, `Tanh`, `Sigmoid`, `Relu`,
  `Gelu`, `Silu`) emits a Rust expression matching the upstream
  numerical contract (e.g. `Gelu` uses the tanh-approximation
  matching PyTorch's `nn.functional.gelu(approximate='tanh')`).

- REQ-7: Statement emission — every `LoopIR` statement variant
  (`Loop`, `Store`, `Let`, `Assign`, `Accumulate`, `If`, `Comment`)
  emits a valid Rust statement with consistent indentation
  (4-space steps).

## Acceptance Criteria

- [x] AC-1: `CpuCodegen::generate_rust_source(&[], "kernel_empty")`
  returns a well-formed empty Rust function.
- [x] AC-2: A single `IrOpKind::Neg` lowered then emitted contains
  `inputs[0]`, `output[`, and `for i in`.
- [x] AC-3: A sum-reduction emission contains `let mut acc` and
  `acc +=`.
- [x] AC-4: A matmul emission contains nested `for i in`, `for j in`,
  `for p in` loops plus `let mut acc` + `acc +=`.
- [x] AC-5: `format_f64_rust(0.0)` returns `"0.0_f64"`;
  `format_f64_rust(f64::INFINITY)` returns `"f64::INFINITY"`.
- [x] AC-6: `rust_buffer_access("in42")` returns `"inputs[42]"`.
- [x] AC-7: `Sigmoid` emission contains `1.0_f64` and `.exp()`;
  `Pow { exponent }` emission contains `.powf(`.

## Architecture

### `CpuCodegen` entry point (REQ-1, REQ-2)

`pub struct CpuCodegen` at `pub struct CpuCodegen in codegen_cpu.rs`
is a unit struct. Its sole public emission method
`pub fn generate_rust_source` at
`impl CpuCodegen in codegen_cpu.rs` walks the `LoopIR` program and
emits a Rust function with `#[inline(always)]` plus the canonical
`pub unsafe fn <name>(inputs: &[&[f64]], output: &mut [f64])`
signature. The trampoline that bridges this Rust signature to the
FFI-friendly `extern "C" fn(*const *const f64, *mut f64, i32)` lives
in `codegen_jit` (the `compile_loop_ir_kernel` function builds the
cranelift IR matching this trampoline shape).

### Statement / expression emitters (REQ-6, REQ-7)

`fn emit_rust_stmt` at `fn emit_rust_stmt in codegen_cpu.rs`
dispatches on `LoopIR` variants. `Loop` recursively emits inner
statements with `indent + 1`; `Store` emits
`<buf>[<idx> as usize] = <val>;`; `Let` / `Assign` / `Accumulate` emit
the obvious Rust forms; `If` emits the `if ... else ...` block;
`Comment` emits `// <text>`.

`fn emit_rust_expr` at `fn emit_rust_expr in codegen_cpu.rs`
dispatches on `Expr` variants. Unary ops use the `f64::<method>()`
form (`.exp()`, `.ln()`, `.sqrt()`, `.abs()`, `.tanh()`).
`Sigmoid` emits `(1.0_f64 / (1.0_f64 + (-x).exp()))`. `Relu`
emits the explicit `if-else` form (avoiding `f64::max` for clarity
of the zero-comparison). `Gelu` emits the tanh-approximation with
`0.7978845608` and `0.044715` constants matching upstream
`torch.nn.functional.gelu(approximate='tanh')` at
`/home/doll/pytorch/aten/src/ATen/native/cpu/Activation.cpp`.
`Silu` emits `(x / (1.0 + (-x).exp()))`.

### Buffer + float helpers (REQ-4, REQ-5)

`fn rust_buffer_access` at
`fn rust_buffer_access in codegen_cpu.rs` strips the `in` prefix and
parses the suffix as a `usize` to produce `inputs[<idx>]`; `out` and
`output` map to `output`; any other name passes through verbatim.

`fn format_f64_rust` at `fn format_f64_rust in codegen_cpu.rs` uses
exact float comparison (`#[allow(clippy::float_cmp,
reason = "canonical literal selection requires bit-identical match")]`)
to emit short literals for `0.0` and `1.0` and the canonical
non-literal names for `Inf`, `-Inf`, `NaN`.

### SIMD / parallel hints (REQ-3)

The inner-loop SIMD comment is emitted unconditionally for loops
whose body contains no nested `Loop`. The parallel hint requires the
loop's `end` expression to be an `Expr::IntConst(n)` with `n as
usize >= 1024` AND the body to contain at least one nested loop.
The hints are advisory only — they do not change codegen, but
downstream readers (and the cranelift backend's autovectorizer) can
use them.

### Non-test production consumers

- The crate does not currently re-export `CpuCodegen` from `lib.rs`
  (`grep -n "CpuCodegen" ferrotorch-jit/src/lib.rs` returns
  `pub use codegen_cpu::CpuCodegen;` at line 93).
- `ferrotorch-jit/src/codegen.rs` calls
  `crate::codegen_cpu::CpuCodegen::generate_rust_source(loops, &fn_name)`
  from the `InductorBackend::generate` `CpuRust` arm.
- `ferrotorch-jit/src/codegen.rs` calls the same emitter from the
  identity-graph fallback.

## Parity contract

`parity_ops = []`. This is a source emitter, not an op
implementation. Numerical edge cases preserved:

- **Gelu approximation** — emits the `tanh` form
  (`x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`)
  matching `torch.nn.functional.gelu(approximate='tanh')` upstream.
  The exact form is reused identically in the GPU codegen + the
  CPU fusion path for cross-backend bit stability.
- **Sigmoid** — emits the form `1 / (1 + exp(-x))` (not the
  `0.5 * (1 + tanh(x/2))` form). Matches upstream
  `at::native::sigmoid` byte-for-byte under default rounding.
- **Special floats** — `Inf`, `-Inf`, `NaN` get their canonical
  `f64` constant names; `0.0` / `1.0` get short literals; everything
  else gets the full numeric value with `_f64` suffix.

## Verification

Tests in `mod tests in codegen_cpu.rs`: every `UnaryOpKind` variant
gets an explicit emission test (`test_rust_sigmoid`, `test_rust_pow`,
`test_rust_silu`, `test_rust_log`, ...), shape-emission tests
(`test_rust_matmul`, `test_rust_sum_reduction`), and
helper-function tests (`test_rust_special_float_values`,
`test_rust_buffer_mapping`).

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-jit --lib codegen_cpu:: 2>&1 | tail -3
```

Expected: all 14 tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct CpuCodegen` in `codegen_cpu.rs`; non-test consumer: `generate_rust_source in ferrotorch-jit/src/codegen.rs` `crate::codegen_cpu::CpuCodegen::generate_rust_source(loops, &fn_name)` from `InductorBackend::generate` (CpuRust arm). |
| REQ-2 | SHIPPED | impl: `pub fn generate_rust_source` in `codegen_cpu.rs`; non-test consumer: `generate_rust_source in codegen.rs` (main InductorBackend path) and `codegen in codegen.rs` (identity-graph fallback). |
| REQ-3 | SHIPPED | impl: SIMD-comment + parallel-hint emission in `fn emit_rust_stmt` (`codegen_cpu.rs`); non-test consumer: transitively via `codegen in codegen.rs`. Hints are emitted into every inner loop / large outer loop emitted by the InductorBackend pipeline. |
| REQ-4 | SHIPPED | impl: `fn rust_buffer_access` in `codegen_cpu.rs`; non-test consumer: invoked by both `emit_rust_stmt` (Store / Let) and `emit_rust_expr` (Index) in the generated kernel for every `IrOpKind` lowered through `InductorBackend::generate`. |
| REQ-5 | SHIPPED | impl: `fn format_f64_rust` in `codegen_cpu.rs`; non-test consumer: called inside `emit_rust_expr` for every `Expr::Const` emitted by the InductorBackend pipeline. |
| REQ-6 | SHIPPED | impl: `fn emit_rust_expr` UnaryOp arm in `codegen_cpu.rs` with one match arm per `UnaryOpKind` variant; non-test consumer: transitively via `codegen in codegen.rs`. |
| REQ-7 | SHIPPED | impl: `fn emit_rust_stmt` in `codegen_cpu.rs` with one arm per `LoopIR` variant; non-test consumer: transitively via `codegen in codegen.rs`. |
