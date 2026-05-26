# ferrotorch-jit — `module` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/jit/_trace.py
  - torch/jit/__init__.py
  - torch/_functorch/aot_autograd.py
  - torch/_dynamo/__init__.py
-->

## Summary

`ferrotorch-jit/src/module.rs` is the `torch.compile` /
`torch.jit.trace` equivalent. It defines `TracedModule<T>` (a
compiled module wrapping an optimised `IrGraph`),
`AotCompiledModule<T>` (the AOT-autograd analog holding paired
forward+backward graphs), and the `compile` / `compile_with_config`
entry points that one-shot trace → optimise → wrap. Mirrors the
`TracedModule` class at `torch/jit/_trace.py:1272-1377` and the
`torch.compile(model)` / `aot_function(...)` flow at
`torch/_dynamo/__init__.py` + `torch/_functorch/aot_autograd.py:698-833`.

## Requirements

- REQ-1: `pub struct CompileConfig` holds the user-tunable settings
  (`optimization: OptimizationConfig`, `fullgraph: bool`,
  `cache_size: usize`) consumed by the `compile_with_config` entry
  point. Mirrors the `compile(model, mode=..., fullgraph=...,
  dynamic=...)` kwargs at `torch/_dynamo/__init__.py:compile`.
- REQ-2: `pub struct TracedModule<T: Float>` wraps an optimised
  `IrGraph` and exposes it through the standard `Module<T>` trait,
  so a traced graph is a drop-in replacement for the eager module it
  was traced from. Mirrors `torch.jit.TracedModule` at
  `torch/jit/_trace.py:1272`.
- REQ-3: `TracedModule::forward` (single-input entry) and
  `TracedModule::forward_multi` (slice-of-inputs entry) interpret
  the wrapped graph against the supplied inputs, validating that
  the input count matches the trace-time signature. Mirrors the
  `__call__` dispatch on `TracedModule` and `torch.jit.ScriptModule`.
- REQ-4: `pub struct AotCompiledModule<T: Float>` holds the
  forward+backward graphs returned by
  `aot_autograd::decompose_forward_backward`, plus the
  `saved_tensor_indices` contract that pins which intermediates the
  backward depends on. Mirrors the AOTAutograd dispatch at
  `torch/_functorch/aot_autograd.py:aot_function`.
- REQ-5: `AotCompiledModule::forward_with_ctx` executes the forward
  graph AND captures the intermediates declared by
  `saved_tensor_indices` via `interpret_multi_with_captures` (audit
  #1110 finding-A: the pre-fix implementation stashed
  `inputs.to_vec()` and silently lied about the contract).
- REQ-6: `AotCompiledModule::backward(grad_output)` runs the
  backward graph against the saved intermediates + `grad_output`
  and returns the input-gradient tensor. Mirrors the back-half of
  AOTAutograd's joint trace.
- REQ-7: `pub fn compile<T, F>(f, example_inputs, config) ->
  FerrotorchResult<TracedModule<T>>` is the one-call
  `torch.compile`-style entry point: trace, run all optimisations,
  wrap. Mirrors `torch.compile(model)` at
  `torch/_dynamo/__init__.py:compile`.
- REQ-8: `pub fn compile_with_config<T, F>(f, example_inputs,
  config)` accepts the extended `CompileConfig` for future settings;
  delegates to `compile` after extracting the `optimization` field.
- REQ-9: Save/load round-trip: `TracedModule::to_bytes` /
  `from_bytes` / `save` / `load` mirror `torch.jit.save` /
  `torch.jit.load` for the in-memory and on-disk cases (#620).

## Acceptance Criteria

- [x] AC-1: `CompileConfig::default()` has
  `fullgraph=false`, `cache_size=8`, and all optimisation passes
  enabled.
- [x] AC-2: `TracedModule::new(graph)` succeeds for an `IrGraph`
  with one input and one output and reports
  `input_count == 1` plus the captured `output_shape`.
- [x] AC-3: `TracedModule::forward_multi` rejects an input slice of
  the wrong length with `FerrotorchError::InvalidArgument`.
- [x] AC-4: `compile(fn, &[a, b], None)` traces, optimises, and
  returns a `TracedModule` whose `forward_multi` reproduces the
  eager numerical result within 1e-5.
- [x] AC-5: `compile` honours a user-supplied `OptimizationConfig`
  with all passes disabled (no constant-folding away of weights).
- [x] AC-6: `TracedModule::to_bytes` round-trips through
  `from_bytes` with byte-exact reproduction.
- [x] AC-7: `TracedModule::save(path)` followed by
  `TracedModule::load(path)` reproduces the numerical result.
- [x] AC-8: `AotCompiledModule::forward_with_ctx` captures one
  tensor per `saved_tensor_indices` entry (not just the inputs);
  the captured value at the `Mul` intermediate position equals the
  expected `[4, 10, 18]` for the canonical `sum(mul(a, b))` test
  case.

## Architecture

`TracedModule<T>` holds the optimised `IrGraph`, the input count
(captured at construction), and the trace-time output shape. Its
`Module<T>` impl exposes `forward` (single-input), `parameters`
(empty in this MVP — all weights are baked in as constants or
explicit inputs), `train`/`eval` (no-op — traced modules are always
in eval), and `is_training` (always `false`). The save/load family
calls `IrGraph::serialize` / `IrGraph::deserialize` (REQ-9).

`AotCompiledModule<T>` is the AOT-autograd analog. Construction
(`new`) takes the forward graph, backward graph, and
`saved_tensor_indices` produced by
`aot_autograd::decompose_forward_backward`. Its `forward_with_ctx`
calls `interpret_multi_with_captures` with the saved indices so
every saved intermediate is captured in a single forward pass — no
re-execution. `backward(grad_output)` builds the backward input
list as `[saved_tensors..., grad_output]` and calls
`interpret` on the backward graph.

The fix for audit #1110 finding-A: the pre-fix `forward_with_ctx`
stashed `inputs.to_vec()` into `saved_tensors`, length 2, then
appended `grad_output` — producing 3 backward inputs against the
required 4. The fixed implementation uses
`interpret_multi_with_captures` (`module.rs` forward_with_ctx
body) to honour the `saved_tensor_indices` contract literally.

`compile` and `compile_with_config` are top-level entry points:
`compile` traces (`trace::trace`), optimises
(`optimize::optimize`), and wraps with `TracedModule::new`.
`compile_with_config` accepts the full `CompileConfig` and delegates
to `compile`.

### Non-test production consumers

- `pub use module::{AotCompiledModule, CompileConfig, TracedModule,
  compile, compile_with_config}` at `ferrotorch-jit/src/lib.rs:111`
  — grandfathered public surface.
- `ferrotorch-jit/src/graph_break.rs:26` imports
  `crate::module::{CompileConfig, TracedModule}`, then
  `graph_break.rs:472` and `graph_break.rs:507` invoke
  `TracedModule::new(optimized)` to wrap segments of a graph break.
- `ferrotorch-jit/src/symbolic.rs:64` imports
  `crate::module::TracedModule` and uses it as the inner module of
  `SymbolicTracedModule`.

## Parity contract

`parity_ops = []`. The traced/compiled-module surface composes ops
covered elsewhere (the graph's nodes invoke
`ferrotorch_core::grad_fns` primitives through the interpreter).

Numerical edge cases:

- **Empty parameters in MVP**: `TracedModule::parameters` returns
  an empty `Vec` — the MVP bakes constants and treats explicit
  inputs as the only mutable state. Differs from upstream's
  `TracedModule.named_parameters()` which exposes the captured
  weights; documented at REQ-2.
- **Always-eval mode**: `TracedModule::train`/`eval` are no-ops, so
  no Dropout/BatchNorm switch is honoured at runtime. Models that
  need a training-mode trace must trace twice and dispatch
  themselves. Documented at REQ-2/REQ-3.
- **Single-output AOT**: `AotCompiledModule::forward_with_ctx`
  requires `outputs.len() == 1`. Multi-output AOT is future work.

## Verification

Tests in `ferrotorch-jit/src/module.rs` `mod tests`:
`test_traced_module_new_and_forward`,
`test_traced_module_forward_multi`,
`test_forward_on_multi_input_graph_errors`,
`test_forward_multi_input_count_mismatch`, `test_graph_accessor`,
`test_module_trait_empty_parameters`,
`test_module_trait_forward`,
`test_trace_optimize_execute`,
`test_compile_produces_working_module`,
`test_compile_with_custom_config`,
`test_compile_with_compile_config`,
`test_compiled_module_with_different_inputs`,
`test_traced_module_implements_module_trait`,
`test_trace_linear_layer`,
`test_compile_config_default`,
`test_compile_config_from_optimization`,
`test_traced_module_is_send_sync`,
`test_traced_module_to_bytes_from_bytes_roundtrip`,
`test_traced_module_save_load_disk_roundtrip`,
`test_traced_module_from_bytes_garbage_input_errors`,
`test_forward_with_ctx_captures_intermediate_not_just_inputs`
(audit #1110 finding-A discriminator),
`test_forward_with_ctx_captured_intermediate_value_matches_interpreter`.

Smoke command:

```bash
cargo test -p ferrotorch-jit --lib module:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct CompileConfig` in `module.rs`; non-test consumer: re-export at `ferrotorch-jit/src/lib.rs:111`. |
| REQ-2 | SHIPPED | impl: `pub struct TracedModule<T: Float>` + `impl<T: Float> Module<T> for TracedModule<T>` in `module.rs`; non-test consumer: `ferrotorch-jit/src/graph_break.rs:472` constructs `TracedModule::new(optimized)` per segment, `ferrotorch-jit/src/symbolic.rs:64` wraps it. |
| REQ-3 | SHIPPED | impl: `pub fn forward_multi` and `impl<T: Float> Module<T> for TracedModule<T>::forward` in `module.rs`; non-test consumer: `ferrotorch-jit/src/symbolic.rs:312` `forward_symbolic` invokes `interpret(self.inner.graph(), inputs)` on the wrapped traced module. |
| REQ-4 | SHIPPED | impl: `pub struct AotCompiledModule<T: Float>` + `new` in `module.rs`; non-test consumer: re-export at `ferrotorch-jit/src/lib.rs:111` — public surface returned by `aot_autograd::compile_aot` at `ferrotorch-jit/src/aot_autograd.rs:455`. |
| REQ-5 | SHIPPED | impl: `pub fn forward_with_ctx` in `module.rs` (calls `interpret_multi_with_captures`); non-test consumer: re-export at `ferrotorch-jit/src/lib.rs:111`. |
| REQ-6 | SHIPPED | impl: `pub fn backward` in `module.rs`; non-test consumer: re-export at `ferrotorch-jit/src/lib.rs:111`. |
| REQ-7 | SHIPPED | impl: `pub fn compile<T, F>` in `module.rs`; non-test consumer: re-export at `ferrotorch-jit/src/lib.rs:111`. |
| REQ-8 | SHIPPED | impl: `pub fn compile_with_config<T, F>` in `module.rs`; non-test consumer: re-export at `ferrotorch-jit/src/lib.rs:111`. |
| REQ-9 | SHIPPED | impl: `pub fn to_bytes`, `pub fn from_bytes`, `pub fn save`, `pub fn load` in `module.rs`; non-test consumer: re-export at `ferrotorch-jit/src/lib.rs:111`; uses `IrGraph::serialize` / `IrGraph::deserialize` from `ferrotorch-jit/src/serialize.rs:431-491`. |
