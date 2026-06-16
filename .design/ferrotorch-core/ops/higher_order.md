# Higher-Order Tensor Operations

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/
  - c10/
  - torch/_torch_docs.py
  - torch/overrides.py
-->

## Summary

`ferrotorch-core/src/ops/higher_order.rs` implements `cond` and
`scan` — the two higher-order primitives that enable conditional
subgraph execution and sequential state accumulation under autograd.
They mirror `torch._higher_order_ops` / `torch.cond` and the
`torch.func.scan` / state-space-model patterns used by Mamba, S4,
RWKV, etc. The implementation composes user-supplied step / branch
functions and wraps each autograd-tracked output with a thin zero-copy
`CondBackward` / `ScanBackward` stride view on the original output
storage. The backward node routes the upstream gradient to the target
tensor's own grad-fn chain — the chain that the user's callback built
on the forward pass — and also exposes explicit zero-gradient edges to
the original control-flow inputs for detached targets, matching
PyTorch's higher-order autograd invariant.

## Requirements

- REQ-1: `cond(pred, true_fn, false_fn, operands)` — evaluate both
  branch callables once, validate their output metadata, and return the
  tensors from `true_fn(operands)` if `pred` is nonzero, else from
  `false_fn(operands)`. A CUDA-resident scalar predicate is read by an
  explicit one-element synchronization; branch outputs and operands are
  not moved by predicate evaluation. Mirrors `torch.cond`
  (`torch/_higher_order_ops/cond.py`), whose compile/tracing path
  validates both branch output specs before selecting the result.
- REQ-2: `cond` autograd — gradients flow through the taken branch
  ONLY. When any operand requires grad, each branch output is wrapped
  with a zero-copy `CondBackward` view that preserves the output's
  storage, strides, offset, and device. The branch's own grad_fn chain
  (built during the forward callback) carries real per-operand VJPs;
  `CondBackward` forwards `grad_output` to it and returns zeros for
  original operands that the branch output does not reach.
- REQ-3: `validate_cond_branches(true_outputs, false_outputs)` —
  user-callable utility to eagerly check both branches return the
  same number of tensors with matching static shapes, devices, strides,
  and storage offsets. Dtype is fixed by the generic `Tensor<T>` output
  type in this API and by storage tag invariants. `cond` calls this
  helper internally after evaluating both branch callables.
- REQ-4: `scan(fn_step, init, xs)` — sequential fold-with-outputs.
  Calls `fn_step(carry, x) -> (new_carry, output)` once per element
  of `xs`. Returns `(final_carry, [outputs...])`. Mirrors the
  `scan(...)` pattern in
  `torch/_higher_order_ops/` and the `flax.linen.scan` /
  `jax.lax.scan` analogs from the JAX side. This is the primitive
  state-space models (Mamba, S4, RWKV) build on.
- REQ-5: `scan` autograd — gradients flow through ALL steps via the
  per-step grad_fn chains built by `fn_step`. Each wrapped output /
  final-carry attaches its own zero-copy `ScanBackward` view routing
  the upstream gradient to the raw target and zeros to disconnected
  `init` / `xs` inputs.
- REQ-6: Predicate validation — `cond` errors with
  `InvalidArgument` if `pred.numel() != 1` (scalar requirement).

## Acceptance Criteria

- [x] AC-1: `cargo test -p ferrotorch-core --lib ops::higher_order`
  passes.
- [x] AC-2: `cond` with `pred=1.0` returns `true_fn` outputs;
  `pred=0.0` returns `false_fn` outputs. The untaken branch callable is
  still evaluated once for metadata validation, matching PyTorch's
  branch-output contract.
- [x] AC-3: `cond` predicate with `numel != 1` errors.
- [x] AC-4: `scan` with empty `xs` returns
  `(init.clone(), Vec::new())`.
- [x] AC-5: `scan` autograd — calling `.backward()` on the final
  carry propagates gradients back to `init` and every `xs[i]`.
- [x] AC-6: `cond` / `validate_cond_branches` reject mismatched output
  counts, static shapes, devices, strides, and storage offsets before
  returning the selected branch result.
- [x] AC-7: CUDA `cond` / `scan` outputs and backward gradients remain
  CUDA-resident when autograd is active; detached branch / step outputs
  produce explicit zero gradients for the original inputs.
- [x] AC-8: CPU and CUDA `cond` predicates match PyTorch
  single-element tensor truthiness: exact zero is false; nonzero,
  including subunit values and NaN, is true.

## Architecture

`CondBackward<T>` holds the raw `branch_output: Tensor<T>` plus the
original `operands`. Its `backward` forwards `grad_output.clone()` to
the branch output and returns `zeros_like(operand)` for each operand.
Its `inputs()` returns the branch output followed by those operands, so
the autograd engine traverses INTO the branch output's own grad-fn
chain (which the user's callback already built during the forward
pass), while detached outputs still materialise PyTorch-style zero
operand gradients.

`cond` at `:79-134` walks:

1. Validate `pred.numel() == 1`.
2. Read the logical single element with `data_vec()`, which is an
   explicit scalar D2H synchronization for CUDA predicates.
3. Treat the predicate as true iff it is not exactly zero. This mirrors
   PyTorch tensor truthiness for float predicates: `0.25`, `0.5`,
   negative values, and `NaN` all take the true branch.
4. Invoke both branch callables once and validate their output
   metadata. This is the eager analogue of PyTorch's fake-tensor branch
   spec validation. Branch callables are expected to be pure; side
   effects in branch callables are outside the structured control-flow
   contract.
5. Select the true outputs when the predicate is true, otherwise the
   false outputs. The unselected outputs are dropped after validation.
6. If no operand requires grad / grad is disabled, return raw selected
   outputs.
7. Otherwise, wrap each selected output with `CondBackward { branch_output:
   out.clone(), operands }` through `wrap_control_flow_output`, which
   attaches the grad node to a stride view sharing the output's
   existing storage/device instead of copying through host memory.

`validate_cond_branches` at `:142-194` is the eager validator —
walks zipped output vectors, checking length match and per-position
shape, device, stride, and storage-offset match. Returns
`InvalidArgument` / `ShapeMismatch` / `DeviceMismatch` errors.

`ScanBackward<T>` is analogous to `CondBackward`: it holds the raw
`target: Tensor<T>` and cloned zero-gradient inputs (`init` plus
`xs`). Each wrapped step output / final carry gets its own
`ScanBackward` instance, so real gradients flow through the target's
step graph while disconnected scan inputs receive explicit zeros.

`scan` at `:236-295` walks:

1. Empty `xs` early-return at `:245`.
2. Walk `xs` left-to-right, calling `fn_step(&current_carry, x)`
   at each step; accumulate `outputs` and update `current_carry`
   (`:253-257`).
3. Decide whether autograd is needed: `is_grad_enabled() &&
   (init.requires_grad() || any xs[i].requires_grad())`.
4. If not, return `(current_carry, outputs)` raw.
5. Otherwise, wrap the final carry with a zero-copy `ScanBackward`
   routing to the raw `current_carry`, then wrap each step output
   similarly. The wrapper shares output storage/stride metadata and
   never re-materialises CUDA tensors through CPU storage.

The implementation has been refactored — earlier versions held
`Vec<carries>`, `Vec<xs>`, `Vec<outputs>` plus an `OutputKind` enum
to disambiguate which `ScanBackward` instance was which. The
current design recognises that the held tensor already encodes the
role (it IS the step output or final carry), making the enum +
multi-vec storage vestigial. The in-line comment at `:185-189`
documents this.

**Non-test consumers**: `crate::autograd::mod` at `autograd/mod.rs:26`
re-exports as `pub use crate::ops::higher_order::{cond, scan,
validate_cond_branches}` — the autograd module is the canonical
re-export path. Tests for `cond` / `scan` exist in
`ferrotorch-core/tests/higher_order_*.rs` and downstream Mamba /
S4 implementations in `ferrotorch-nn` consume `scan` as their
primary state-update mechanism.

## Parity contract

`parity_ops = []` (no torch op_db entry for `torch.cond` /
`torch.func.scan` in the eager parity-sweep oracle). The numeric
contract is "behaviour matches torch.cond / jax.lax.scan
semantics" — verified through unit tests + the downstream
Mamba/S4 integration tests that compare against reference PyTorch
implementations.

## Verification

`cargo test -p ferrotorch-core --lib ops::higher_order` exercises
the local tests at `ops/higher_order.rs` (predicate validation, branch
execution, scan empty-input early-return, connected autograd flow, and
detached-output zero-gradient semantics).

`cargo test -p ferrotorch-core --features gpu --test
audit_core116_117_higher_order_control_flow` exercises CPU and CUDA
probes for PyTorch-style detached-output zero gradients, connected
CUDA gradients, and non-contiguous CUDA branch outputs that must stay
device-resident.

`cargo test -p ferrotorch-core --features gpu --test
audit_core119_cond_cuda_predicate` exercises CPU and CUDA predicate
truthiness, including CUDA scalar predicates, f32/f64 branch execution,
resident CUDA outputs, resident CUDA gradients, zero predicates, and
NaN predicates.

`cargo test -p ferrotorch-core --features gpu --test
audit_core118_cond_branch_metadata` exercises PyTorch-style branch
metadata validation for output count, static shape, device, stride, and
storage offset mismatches, including CUDA device-residency probes.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `cond` at `ops/higher_order.rs:79`; non-test consumer: re-exported through `crate::autograd::mod` at `autograd/mod.rs:26` as `ferrotorch_core::autograd::cond` (boundary public API per goal.md S5) |
| REQ-2 | SHIPPED | impl: `CondBackward in ops/higher_order.rs` + wrap logic at `CondBackward in ops/higher_order.rs`; non-test consumer: `cond` itself, called via the `crate::autograd::cond` re-export |
| REQ-3 | SHIPPED | impl: `validate_cond_branches` at `ops/higher_order.rs:142`; non-test consumer: re-exported as `ferrotorch_core::autograd::validate_cond_branches` at `autograd/mod.rs:26` |
| REQ-4 | SHIPPED | impl: `scan` at `ops/higher_order.rs:236`; non-test consumer: re-exported as `ferrotorch_core::autograd::scan` at `autograd/mod.rs:26` |
| REQ-5 | SHIPPED | impl: `ScanBackward in ops/higher_order.rs` + per-output wrap at `ScanBackward in ops/higher_order.rs`; non-test consumer: `scan` itself, called via the `crate::autograd::scan` re-export |
| REQ-6 | SHIPPED | impl: `pred.numel() != 1` check at `ops/higher_order.rs:91`; non-test consumer: `cond` entry point |
