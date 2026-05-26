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
functions and wraps each output with a thin `CondBackward` /
`ScanBackward` that simply routes the upstream gradient to the
target tensor's own grad-fn chain — the chain that the user's
callback built on the forward pass.

## Requirements

- REQ-1: `cond(pred, true_fn, false_fn, operands)` — evaluate
  `true_fn(operands)` if `pred > 0.5`, else `false_fn(operands)`.
  Only the taken branch is called. Mirrors `torch.cond`
  (`torch/_higher_order_ops/cond.py`).
- REQ-2: `cond` autograd — gradients flow through the taken branch
  ONLY. Wraps each branch output with `CondBackward { branch_output:
  out.clone() }`. The branch's own grad_fn chain (built during the
  forward callback) carries the per-operand VJP; `CondBackward`
  simply forwards `grad_output` to it.
- REQ-3: `validate_cond_branches(true_outputs, false_outputs)` —
  user-callable utility to eagerly check both branches return the
  same number of tensors with matching shapes. Required because
  `cond` only executes one branch at runtime; the shape mismatch
  can't be detected by `cond` itself.
- REQ-4: `scan(fn_step, init, xs)` — sequential fold-with-outputs.
  Calls `fn_step(carry, x) -> (new_carry, output)` once per element
  of `xs`. Returns `(final_carry, [outputs...])`. Mirrors the
  `scan(...)` pattern in
  `torch/_higher_order_ops/` and the `flax.linen.scan` /
  `jax.lax.scan` analogs from the JAX side. This is the primitive
  state-space models (Mamba, S4, RWKV) build on.
- REQ-5: `scan` autograd — gradients flow through ALL steps via the
  per-step grad_fn chains built by `fn_step`. Each wrapped output /
  final-carry attaches its own `ScanBackward` instance routing the
  upstream gradient to the raw target.
- REQ-6: Predicate validation — `cond` errors with
  `InvalidArgument` if `pred.numel() != 1` (scalar requirement).

## Acceptance Criteria

- [x] AC-1: `cargo test -p ferrotorch-core --lib ops::higher_order`
  passes.
- [x] AC-2: `cond` with `pred=1.0` calls `true_fn` only;
  `pred=0.0` calls `false_fn` only.
- [x] AC-3: `cond` predicate with `numel != 1` errors.
- [x] AC-4: `scan` with empty `xs` returns
  `(init.clone(), Vec::new())`.
- [x] AC-5: `scan` autograd — calling `.backward()` on the final
  carry propagates gradients back to `init` and every `xs[i]`.
- [x] AC-6: `validate_cond_branches` rejects mismatched output counts
  and shape mismatches.

## Architecture

`CondBackward<T>` at `ops/higher_order.rs:31-48` holds a single
`branch_output: Tensor<T>`. Its `backward` forwards `grad_output.clone()`
unchanged. Its `inputs()` returns `vec![&self.branch_output]` so the
autograd engine traverses INTO the branch output's own grad-fn chain
(which the user's callback already built during the forward pass) —
that's where per-operand VJPs are produced. The earlier implementation
held the raw operands and returned identity grads to each one,
bypassing the branch's grad-fn chain; that produced wrong gradients
for any branch that wasn't a pure pass-through. The current shape is
documented in the in-line comment at `:26-30`.

`cond` at `:79-134` walks:

1. Validate `pred.numel() == 1` (`:91`).
2. Read scalar pred value at `:101`; compare against `T::from(0.5)`.
3. Invoke `true_fn(operands)` or `false_fn(operands)` based on the
   comparison (`:105-109`).
4. If no operand requires grad / grad is disabled, return raw
   outputs.
5. Otherwise, wrap each output with `CondBackward { branch_output:
   out.clone() }` via `Tensor::from_operation` (`:122-131`).

`validate_cond_branches` at `:142-169` is the eager validator —
walks zipped output vectors, checking length match and per-position
shape match. Returns `InvalidArgument` / `ShapeMismatch` errors.

`ScanBackward<T>` at `:191-207` is analogous to `CondBackward`:
holds a single `target: Tensor<T>` and routes grad through. Each
wrapped step output / final carry gets its own `ScanBackward`
instance.

`scan` at `:236-295` walks:

1. Empty `xs` early-return at `:245`.
2. Walk `xs` left-to-right, calling `fn_step(&current_carry, x)`
   at each step; accumulate `outputs` and update `current_carry`
   (`:253-257`).
3. Decide whether autograd is needed: `is_grad_enabled() &&
   (init.requires_grad() || any xs[i].requires_grad())`.
4. If not, return `(current_carry, outputs)` raw.
5. Otherwise, wrap the final carry with a `ScanBackward` routing
   to the raw `current_carry` (`:272-278`), then wrap each step
   output similarly (`:282-292`).

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
the local tests at `ops/higher_order.rs:300-...` (predicate
validation, branch execution, scan empty-input early-return,
autograd flow).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `cond` at `ops/higher_order.rs:79`; non-test consumer: re-exported through `crate::autograd::mod` at `autograd/mod.rs:26` as `ferrotorch_core::autograd::cond` (boundary public API per goal.md S5) |
| REQ-2 | SHIPPED | impl: `CondBackward` at `ops/higher_order.rs:31` + wrap logic at `:122-131`; non-test consumer: `cond` itself, called via the `crate::autograd::cond` re-export |
| REQ-3 | SHIPPED | impl: `validate_cond_branches` at `ops/higher_order.rs:142`; non-test consumer: re-exported as `ferrotorch_core::autograd::validate_cond_branches` at `autograd/mod.rs:26` |
| REQ-4 | SHIPPED | impl: `scan` at `ops/higher_order.rs:236`; non-test consumer: re-exported as `ferrotorch_core::autograd::scan` at `autograd/mod.rs:26` |
| REQ-5 | SHIPPED | impl: `ScanBackward` at `ops/higher_order.rs:191` + per-output wrap at `:282-292`; non-test consumer: `scan` itself, called via the `crate::autograd::scan` re-export |
| REQ-6 | SHIPPED | impl: `pred.numel() != 1` check at `ops/higher_order.rs:91`; non-test consumer: `cond` entry point |
