# ferrotorch-jit — `aot_autograd` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/_functorch/aot_autograd.py
  - torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py
  - torch/_functorch/_aot_autograd/runtime_wrappers.py
-->

## Summary

`ferrotorch-jit/src/aot_autograd.rs` implements the ahead-of-time
autograd decomposition: given a forward `IrGraph`, produce a
separate forward graph (that computes the output AND saves
intermediates) plus a backward graph (that consumes the saved
intermediates and a `grad_output` and produces input gradients).
Mirrors `torch._functorch.aot_autograd.aot_function` at
`torch/_functorch/aot_autograd.py:698-833` and the joint-trace
machinery in
`torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py`.

## Requirements

- REQ-1: `pub struct AotGraphPair { forward, backward,
  saved_tensor_indices }` is the decomposition's return shape — the
  forward graph augmented with the saved intermediates, the
  separate backward graph, and the topo indices of the nodes whose
  outputs the backward depends on.
- REQ-2: `pub fn decompose_forward_backward(forward_graph:
  &IrGraph) -> FerrotorchResult<AotGraphPair>` performs the
  decomposition: walks the forward graph in topo order, builds the
  per-op gradient rules into a separate backward graph, and tracks
  which forward intermediates must be saved.
- REQ-3: The forward graph in the returned `AotGraphPair` is
  augmented with the saved intermediates as additional outputs (in
  the order matching `saved_tensor_indices`) so the interpreter
  can capture them in a single forward pass via
  `interpret_multi_with_captures`.
- REQ-4: The backward graph expects inputs ordered
  `[saved_intermediates..., grad_output]` and produces gradients
  for each forward input in declaration order.
- REQ-5: `pub fn compile_aot<T, F>(f, example_inputs) ->
  FerrotorchResult<AotCompiledModule<T>>` is the one-call entry
  point: trace `f`, decompose into a pair, wrap in an
  `AotCompiledModule` whose `forward_with_ctx` + `backward` execute
  the pair correctly.

## Acceptance Criteria

- [x] AC-1: For the forward graph `y = sum(mul(a, b))`,
  `decompose_forward_backward` returns an `AotGraphPair` whose
  `saved_tensor_indices` includes BOTH the inputs (topo 0/1) AND
  the Mul intermediate (topo 2). At minimum: `saved_tensor_indices.len()
  > 2`.
- [x] AC-2: The returned `forward` graph produces the same output
  as the original forward when interpreted on the same inputs
  (e.g. `sum([1*4, 2*5, 3*6]) = 32`).
- [x] AC-3: The returned `backward` graph, when supplied
  `[saved_intermediates..., grad_output=1.0]`, produces gradients
  matching the analytic derivatives (e.g. `d sum(a*b) / da = b`,
  `d sum(a*b) / db = a`).
- [x] AC-4: `compile_aot(f, &[a, b])` returns an
  `AotCompiledModule<T>` whose `forward_with_ctx` + `backward` pair
  reproduces the eager `backward()` result within 1e-5.

## Architecture

`AotGraphPair` is a plain struct: `forward: IrGraph`, `backward:
IrGraph`, `saved_tensor_indices: Vec<usize>`. The indices are
positions into the forward graph's topological order; each names a
node whose first output is the saved intermediate the backward
relies on.

`decompose_forward_backward` walks the forward graph in topo order:

1. **Build a node-id map** so each node can be looked up by its
   `IrNodeId` (`aot_autograd.rs:186`).
2. **Per-op gradient rules** — for each forward op (`Add`,
   `Sub`, `Mul`, `Sum`, `Relu`, `Sigmoid`, `Tanh`, `Matmul`,
   `Linear`, etc.), emit the matching backward IR nodes into the
   backward graph. Each rule documents which intermediates it
   needs and adds them to `saved_tensor_indices`.
3. **Topological dependency resolution** — backward nodes consume
   the saved intermediates (which are augmented as outputs of the
   forward graph) and the `grad_output` input.
4. **Output binding** — the backward graph's outputs are the
   gradients for each forward input (in declaration order).

For ops the decomposer doesn't yet recognise, the function returns
a `JitError::UnsupportedOp` converted to `FerrotorchError`. The
known set covers the ops that appear in tested forwards
(`Add`, `Sub`, `Mul`, `Sum`, `Relu`, `Linear`, `Matmul`, more); the
"add a new op" path is "extend the match arm + add a test".

`compile_aot<T, F>` is the integrated entry point: call
`crate::trace::trace(f, example_inputs)?` (`aot_autograd.rs:466`),
call `decompose_forward_backward`, wrap the pair with
`AotCompiledModule::new(pair.forward, pair.backward,
pair.saved_tensor_indices)`.

### Non-test production consumers

- `pub use aot_autograd::{AotGraphPair, compile_aot,
  decompose_forward_backward}` at
  `ferrotorch-jit/src/lib.rs:87`.
- `ferrotorch-jit/src/module.rs:310, 348` — `AotCompiledModule`'s
  rustdoc explicitly references `aot_autograd::compile_aot` and
  `aot_autograd::decompose_forward_backward` as the producer; the
  `AotCompiledModule::new(forward_graph, backward_graph,
  saved_tensor_indices)` constructor consumes exactly the fields
  of `AotGraphPair`.

## Parity contract

`parity_ops = []`. AOT autograd is structural — it composes the
forward IR's ops with the matching backward IR. Numerical equivalence
to eager `.backward()` is the only correctness criterion.

Edge cases:

- **Unknown op** — surface `JitError::UnsupportedOp`; do not
  silently emit a no-op backward.
- **No `requires_grad` inputs** — `compile_aot` rejects via the
  underlying `trace` (which requires at least one grad input).
- **Multi-output forward** — currently single-output only
  (mirrors the existing tracer / interpreter limitation).

## Verification

Tests in `ferrotorch-jit/src/aot_autograd.rs` `mod tests`:
`test_decompose_simple_add` and additional decomposition tests
covering the supported op set. Plus the audit-#1110 discriminator
tests in `module.rs` that pin the interplay between
`decompose_forward_backward`'s `saved_tensor_indices` contract and
`AotCompiledModule::forward_with_ctx`.

Smoke command:

```bash
cargo test -p ferrotorch-jit --lib aot_autograd:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct AotGraphPair { forward, backward, saved_tensor_indices }` in `aot_autograd.rs`; non-test consumer: `ferrotorch-jit/src/module.rs:998-1014` `AotCompiledModule::new(pair.forward.clone(), pair.backward.clone(), pair.saved_tensor_indices.clone())` (test sites) AND production `pub fn compile_aot` returns this pair internally (`aot_autograd.rs:466-470`). |
| REQ-2 | SHIPPED | impl: `pub fn decompose_forward_backward` in `aot_autograd.rs`; non-test consumer: `pub fn compile_aot` invokes it as the second step. |
| REQ-3 | SHIPPED | impl: the per-op-rule `match` body inside `decompose_forward_backward` augments the forward graph with saved-intermediate outputs aligned to `saved_tensor_indices`; non-test consumer: `module.rs:355-380` `AotCompiledModule::forward_with_ctx` calls `interpret_multi_with_captures(&self.forward_graph, inputs, &self.saved_tensor_indices)`. |
| REQ-4 | SHIPPED | impl: the input-binding logic inside `decompose_forward_backward` (backward inputs = saved intermediates then grad_output); non-test consumer: `module.rs:392-398` `AotCompiledModule::backward` builds `backward_inputs = self.saved_tensors.clone(); backward_inputs.push(grad_output.clone());`. |
| REQ-5 | SHIPPED | impl: `pub fn compile_aot<T, F>` in `aot_autograd.rs`; non-test consumer: re-export at `lib.rs:87`. |
