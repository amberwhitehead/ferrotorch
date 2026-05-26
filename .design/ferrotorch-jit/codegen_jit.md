# ferrotorch-jit — `codegen_jit` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/_inductor/codecache.py
  - torch/_inductor/runtime/compile_tasks.py
  - torch/_inductor/async_compile.py
-->

## Summary

`ferrotorch-jit/src/codegen_jit.rs` JIT-compiles a `LoopIR` kernel
into native code in-process via cranelift, producing a
`JitCompiledKernel` that exposes a callable
`extern "C" fn(*const *const f64, *mut f64, i32)` entry. Mirrors
`torch._inductor.codecache.PyCodeCache`'s role as the
JIT-compile + cache layer, but uses pure-Rust cranelift instead of
rustc shell-out + `libloading`. No subprocess fork, no
on-disk `.so`, no `dlopen`.

## Requirements

- REQ-1: `pub struct JitCompiledKernel` — owns the `JITModule`
  (executable memory pages), holds a typed function pointer to the
  trampoline entry, plus `num_inputs` and `output_len` for arity
  validation. `Send + Sync`-safe with SAFETY comments on every
  `unsafe impl`.

- REQ-2: `JitCompiledKernel::execute(&self, inputs: &[&[f64]],
  output: &mut [f64])` — validates arity + buffer sizes, then
  invokes the kernel via `extern "C" fn(*const *const f64,
  *mut f64, i32)`. Returns `Err(InvalidArgument)` on any mismatch.

- REQ-3: `pub fn compile_loop_ir_kernel(loops, num_inputs,
  output_len) -> FerrotorchResult<JitCompiledKernel>` — the
  cranelift-based compile entry point. Lowers the `LoopIR` into
  cranelift SSA, JIT-compiles, and resolves the entry function
  pointer.

- REQ-4: `pub fn jit_supports(loops: &[LoopIR]) -> bool` — the
  predicate the caller checks before invoking
  `compile_loop_ir_kernel`. Rejects `If` statements,
  `BinOpKind::Mod`, and any `FnCall` other than `powf`. Used by
  `codegen.rs::try_jit_compile_cpu_rust` to short-circuit to the
  interpreter for unsupported shapes.

- REQ-5: Math intrinsic bindings — `exp`, `log`, `sqrt`, `sin`,
  `cos`, `tanh`, `pow` are bound at JIT time to the pure-Rust
  `libm` implementations via cranelift's symbol resolution. The
  bindings are stable symbol names the JIT can call; `f64::exp()`
  / `f64::ln()` / `f64::sqrt()` / `f64::sin()` / `f64::cos()` /
  `f64::tanh()` / `f64::powf()` in `std` dispatch to the same code.

- REQ-6: Compile cache — `KERNEL_CACHE` (`OnceLock<Mutex<HashMap<u64,
  Arc<JitCompiledKernel>>>>`) keyed by the FNV-1a hash of the
  `LoopIR`'s `Debug` representation plus `(num_inputs, output_len)`.
  Identical kernels are reused across compile calls. Cranelift
  compiles in milliseconds, so the cache exists for cache-hit
  speed (sub-microsecond) rather than to avoid expensive compiles.

- REQ-7: `Drop`-safe executable memory — `JITModule` is wrapped in
  `Mutex<JITModule>` so we can deallocate the pages on `Drop`
  without requiring `&mut self`. The kernel function pointer
  remains valid for the entire `JitCompiledKernel` lifetime.

## Acceptance Criteria

- [x] AC-1: `compile_loop_ir_kernel` for a single `IrOpKind::Neg`
  lowered to `LoopIR` produces a `JitCompiledKernel` with
  `num_inputs == 1` and `output_len == 4` (matches the lowered
  numel).
- [x] AC-2: `kernel.execute(&[&[-1.0, 2.0, -3.0]], &mut [0.0; 3])`
  populates the output with `[1.0, -2.0, 3.0]`.
- [x] AC-3: `jit_supports(&[LoopIR::If { ... }])` returns `false`.
- [x] AC-4: `jit_supports(&[LoopIR with BinOpKind::Mod expression])`
  returns `false`.
- [x] AC-5: A two-input `IrOpKind::Add` lowered to `LoopIR`
  compiles and executes correctly.
- [x] AC-6: `kernel.execute(&[&[]], &mut [])` with
  `inputs.len() != num_inputs` returns
  `Err(InvalidArgument)`.
- [x] AC-7: Repeated `compile_loop_ir_kernel` calls with the same
  `LoopIR` hit the cache (same `Arc<JitCompiledKernel>` returned).

## Architecture

### `JitCompiledKernel` (REQ-1, REQ-2, REQ-7)

`pub struct JitCompiledKernel` at
`pub struct JitCompiledKernel in codegen_jit.rs` holds three fields:

1. `_module: Mutex<JITModule>` — owns the executable memory pages.
   Wrapping in `Mutex` lets `Drop` free the pages from a `&self`
   context. Excluded from `Debug` since its internal state has no
   diagnostic value.
2. `kernel_fn: KernelEntry` — the resolved typed function pointer.
3. `num_inputs: usize` + `output_len: usize` — for execute-time
   arity / buffer-length validation.

`pub fn execute` at `impl JitCompiledKernel in codegen_jit.rs`
validates `inputs.len() == num_inputs`, `output.len() >=
output_len`, and every `inputs[i].len() >= output_len`, then
builds a contiguous `Vec<*const f64>` and calls the kernel
function pointer inside an `unsafe { ... }` block whose SAFETY
comment documents:
- The function pointer was produced by cranelift in `build_kernel`
  with the matching `KernelEntry` ABI.
- The `JITModule` owns the pages and is held alive by `_module`
  for the entire `self` lifetime.

The `unsafe impl Send for JitCompiledKernel` and `unsafe impl Sync
for JitCompiledKernel` blocks have SAFETY comments documenting
the no-shared-mutable-state invariant.

### `compile_loop_ir_kernel` (REQ-3, REQ-5)

`pub fn compile_loop_ir_kernel` at
`pub fn compile_loop_ir_kernel in codegen_jit.rs` builds a
cranelift `Function` matching the `KernelEntry` ABI:

```ignore
extern "C" fn ferrotorch_kernel_entry(
    inputs: *const *const f64,
    output: *mut f64,
    n: i32,
)
```

The internal helper `build_kernel` walks the `LoopIR` translating
each statement / expression into cranelift IR. Math intrinsics
(`exp`, `log`, `sqrt`, `sin`, `cos`, `tanh`, `powf`) are emitted
as `call_extern_fn` to externally-resolved symbols; the
`JITBuilder` is configured with symbol resolutions binding those
names to the `libm` implementations under stable Rust ABIs.

### `jit_supports` predicate (REQ-4)

`pub fn jit_supports` at `pub fn jit_supports in codegen_jit.rs`
checks every statement / expression in the `LoopIR`:

- `LoopIR::If` → unsupported (cranelift can synthesize branches,
  but the JIT path doesn't lower them yet).
- `BinOpKind::Mod` → unsupported.
- `Expr::FnCall { name, args }` → supported only if `name ==
  "powf"`.
- All other shapes → supported.

The caller `codegen.rs::try_jit_compile_cpu_rust` short-circuits
to `Ok(None)` (interpreter fallback) when `jit_supports(loops)`
returns `false`.

### Compile cache (REQ-6)

`static KERNEL_CACHE: OnceLock<Mutex<HashMap<u64,
Arc<JitCompiledKernel>>>>` at the module-private location in
`codegen_jit.rs`. The key is the FNV-1a hash of:

1. The `Debug` representation of the `LoopIR` (so structurally
   identical kernels collide deterministically across processes).
2. The `(num_inputs, output_len)` pair.

`compile_loop_ir_kernel` checks the cache first; on miss, it
compiles, wraps in `Arc<JitCompiledKernel>`, inserts, and returns
a clone of the `Arc`. The cache exists for sub-microsecond cache
hits (compile itself is in the millisecond range).

### Non-test production consumers

- `pub use codegen_jit::{JitCompiledKernel, compile_loop_ir_kernel}`
  at `ferrotorch-jit/src/lib.rs:96` — grandfathered public API.
- `ferrotorch-jit/src/codegen.rs:1302` (inside
  `try_jit_compile_cpu_rust`) `crate::codegen_jit::jit_supports(
  kernel_loops)` — the gate that short-circuits to the interpreter
  for unsupported shapes.
- `ferrotorch-jit/src/codegen.rs:1306` (inside
  `try_jit_compile_cpu_rust`) `crate::codegen_jit::
  compile_loop_ir_kernel(kernel_loops, num_inputs, output_len)?`
  — the JIT-compile call that produces the kernel
  `InductorBackend::compile` invokes per `execute`.

## Parity contract

`parity_ops = []`. This is a JIT compile + execute engine; parity
is determined by:

1. The `LoopIR` it consumes (lowered by `codegen_ir`).
2. The intrinsic bindings (`f64::exp`, etc.) it links to.

Numerical edge cases preserved:

- **Intrinsic ULPs** — `libm` `exp`, `log`, `sqrt`, `tanh`,
  `powf` match the Rust `std::f64::<method>` exact bit pattern
  (they call the same code). Tests assert
  `kernel.execute(x) == InterpreterBackend.execute(x)` to within
  1e-10 across the supported `LoopIR` shapes.
- **`If` rejection** — graphs with `If` go through the interpreter
  instead. Mixing the JIT and interpreter paths in the same graph
  is not supported (the caller chooses one based on
  `jit_supports`).
- **Cache-key collisions** — FNV-1a is non-cryptographic; the
  cache value is keyed on `Debug`-string content, so two
  structurally identical kernels (same `LoopIR` debug-output,
  same shape) get the same key.

## Verification

Tests in `mod tests in codegen_jit.rs`:
`jit_supports_elementwise_loops`,
`jit_supports_rejects_if_statement`,
`jit_supports_rejects_modulus`,
`compile_loop_ir_kernel_neg_single_input`,
`compile_loop_ir_kernel_add_two_inputs`, plus
parity tests against the interpreter for unary chains, binary
ops, and reductions.

The `ferrotorch-jit/src/codegen_ir.rs:1245` end-to-end test
(`mod tests`) calls `compile_loop_ir_kernel(&loops, 1, 1)`
through the IR layer and asserts the JIT result matches the
expected scalar.

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-jit --lib codegen_jit:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct JitCompiledKernel` + `unsafe impl Send/Sync for JitCompiledKernel` in `codegen_jit.rs`; non-test consumer: re-export at `ferrotorch-jit/src/lib.rs:96` + `ferrotorch-jit/src/codegen.rs:1306` `let kernel = crate::codegen_jit::compile_loop_ir_kernel(...)?` (the kernel returned is the JitCompiledKernel from REQ-1). |
| REQ-2 | SHIPPED | impl: `pub fn execute` on `impl JitCompiledKernel` in `codegen_jit.rs`; non-test consumer: `codegen.rs:1355` `kernel.execute(&kernel_inputs, &mut output)?` inside the closure returned by `try_jit_compile_cpu_rust` — that closure is the body of every `CompiledGraph::execute` for a JIT-compiled CpuRust graph. |
| REQ-3 | SHIPPED | impl: `pub fn compile_loop_ir_kernel` in `codegen_jit.rs`; non-test consumer: `codegen.rs:1306` `let kernel = crate::codegen_jit::compile_loop_ir_kernel(kernel_loops, num_inputs, output_len)?`. |
| REQ-4 | SHIPPED | impl: `pub fn jit_supports` in `codegen_jit.rs`; non-test consumer: `codegen.rs:1302` `if !crate::codegen_jit::jit_supports(kernel_loops) { return Ok(None); }` — the gate before the JIT compile call. |
| REQ-5 | SHIPPED | impl: math intrinsic symbol bindings inside `build_kernel` of `codegen_jit.rs` (cranelift `JITBuilder::symbol(...)` bindings for `exp`, `log`, `sqrt`, `sin`, `cos`, `tanh`, `powf`); non-test consumer: transitively invoked by every `compile_loop_ir_kernel` call from `codegen.rs:1306`. |
| REQ-6 | SHIPPED | impl: `KERNEL_CACHE: OnceLock<Mutex<HashMap<u64, Arc<JitCompiledKernel>>>>` plus FNV-1a hash + insert in `pub fn compile_loop_ir_kernel` (`codegen_jit.rs`); non-test consumer: the cache services every call site `codegen.rs:1306` makes — the second compile of an identical `LoopIR` returns from cache. |
| REQ-7 | SHIPPED | impl: `_module: Mutex<JITModule>` field on `pub struct JitCompiledKernel` + the implicit `Drop` from `JITModule`'s own `Drop` impl in `codegen_jit.rs`; non-test consumer: every `JitCompiledKernel` returned from `compile_loop_ir_kernel` (via `codegen.rs:1306`) carries the `Mutex<JITModule>` and unmaps its pages on drop. |
