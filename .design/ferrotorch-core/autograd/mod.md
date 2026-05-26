# Autograd module root

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (Revert "feat(gpu): route bf16 buffers through f32 elementwise dispatchers (#23) (#24)")
upstream-paths:
  - aten/src/ATen/
  - c10/
  - torch/_torch_docs.py
  - torch/overrides.py
  - torch/autograd/__init__.py
-->

## Summary

`ferrotorch-core/src/autograd/mod.rs` is the public-facing module root for
ferrotorch's automatic-differentiation stack. It declares the 13 sibling
submodules (`anomaly`, `autocast`, `autocast_ops`, `checkpoint`,
`fixed_point`, `forward_ad`, `grad_penalty`, `gradcheck`, `graph`,
`higher_order`, `hooks`, `no_grad`, `saved_tensors`) and `pub use`-re-exports
the user-visible surface, mirroring the namespacing strategy of
`torch/autograd/__init__.py`. The module also re-exports
`crate::ops::higher_order::{cond, scan, validate_cond_branches}` so the
legacy `autograd::cond` import path stays compatible with prior callers.

## Requirements

- REQ-1: Declare every autograd submodule as `pub mod` so downstream
  crates can reach into the namespace (e.g. `crate::autograd::no_grad::no_grad`).
- REQ-2: Re-export `autocast`, `autocast_ops`, `forward_ad`,
  `gradcheck::gradcheck`, `graph::{backward, backward_with_grad}`,
  `higher_order::{grad, hessian, jacobian}`, `no_grad::{no_grad,
  enable_grad, inference_mode, set_grad_enabled, is_grad_enabled,
  is_inference_mode}`, `fixed_point::fixed_point`,
  `grad_penalty::{grad_norm, gradient_penalty, jvp, vjp}` so consumers
  can write `crate::autograd::backward(...)` rather than the full
  `crate::autograd::graph::backward(...)` path. Mirrors the PyTorch
  `__all__` list at `torch/autograd/__init__.py:41-50`.
- REQ-3: Re-export `cond`, `scan`, `validate_cond_branches` from
  `crate::ops::higher_order` so downstream code keeps the legacy
  `autograd::{cond, scan}` import paths working — these belong here
  per upstream's `torch.cond` / `torch.scan` exposure on the
  `torch.autograd` namespace (via `torch/_higher_order_ops/`).

## Acceptance Criteria

- [x] AC-1: `cargo check -p ferrotorch-core` builds without errors —
  every `pub mod` declaration matches an existing `.rs` file in the
  `autograd/` subdirectory.
- [x] AC-2: `crate::autograd::no_grad` and `crate::autograd::no_grad::no_grad`
  both resolve (the latter via `pub mod no_grad`, the former as the
  re-exported function from `no_grad::no_grad`). Verified by 30+ callers
  across `ferrotorch-core/src/{tensor.rs,einsum.rs,stride_tricks.rs,methods.rs,
  ops/*.rs,grad_fns/*.rs}` and `ferrotorch-nn/src/{flash_attention.rs,
  transformer.rs,dropout.rs}` that import via the module path.
- [x] AC-3: `pub use crate::ops::higher_order::{cond, scan,
  validate_cond_branches}` at `mod.rs:26` keeps the legacy
  `autograd::{cond, scan, validate_cond_branches}` import path
  alive — verified by `crate::lib.rs:127` re-export
  `cond, ..., scan, ..., validate_cond_branches` originating here.

## Architecture

The module root is intentionally thin (38 LOC). Per upstream's
`torch/autograd/__init__.py:13-50` namespace strategy, ferrotorch
mirrors:

- `torch.autograd.{backward, grad, no_grad, enable_grad,
  set_grad_enabled, inference_mode, gradcheck}` → re-exported
  through `pub use` declarations at `mod.rs:15-38`.
- `torch.amp.autocast` / `torch.autograd.forward_ad` → re-exported
  via `pub use autocast::*` and `pub use forward_ad::*` (R-DEV-2 API
  shape match — Python users writing
  `torch.autocast(device_type='cuda', dtype=torch.float16)` translate
  to `autograd::autocast(AutocastDtype::F16, || ...)`).
- `torch.autograd.functional.{jvp, vjp, jacobian, hessian}` → routed
  through `grad_penalty` (jvp/vjp) and `higher_order` (jacobian/hessian)
  sub-modules.

The 13 submodules form 4 conceptual clusters:

1. **Graph engine** — `graph.rs` (Kahn's algorithm topological-sort
   backward, parallel backward), `hooks.rs` (HookStorage and
   GradHook/PostAccumulateGradHook types stored on every TensorInner),
   `saved_tensors.rs` (pack/unpack-hook offloading), and `no_grad.rs`
   (thread-local grad-tracking + inference mode).

2. **Forward AD** — `forward_ad.rs` (DualTensor + dual_* operator rules
   + jvp_exact + jacfwd).

3. **Higher-order** — `higher_order.rs` (the `grad` function with
   `create_graph=true`, Jacobian/Hessian helpers), `fixed_point.rs`
   (implicit-function-theorem differentiation via Neumann series),
   `grad_penalty.rs` (WGAN-GP, grad_norm, finite-difference jvp,
   autograd-vjp).

4. **Modes and validation** — `autocast.rs` + `autocast_ops.rs`
   (mixed-precision policy engine), `anomaly.rs` (NaN/Inf backtrace
   capture), `checkpoint.rs` (forward+recompute trade-off),
   `gradcheck.rs` (numerical-gradient verification).

The non-test production consumer for REQ-1/REQ-2/REQ-3 is the
workspace-level `ferrotorch-core/src/lib.rs:121-133` `pub use` chain
that exposes `AnomalyMode`, `ForwardBacktrace`,
`check_gradient_anomaly`, `detect_anomaly`, `HookHandle`,
`AutocastDtype`, `AutocastEvent`, `autocast`, `autocast_dtype`,
`autocast_guard`, `backward`, `backward_with_grad`, `cond`,
`enable_grad`, `fixed_point`, `grad`, `grad_norm`, `gradient_penalty`,
`hessian`, `is_autocast_debug`, `is_autocast_enabled`, `is_grad_enabled`,
`jacobian`, `jvp`, `no_grad`, `scan`, `set_autocast_debug`,
`set_grad_enabled`, `validate_cond_branches`, `vjp`, `DualTensor`,
`dual_add`, `dual_cos`, `dual_div`, `dual_exp`, `dual_log`,
`dual_matmul`, `dual_mul`, `dual_neg`, `dual_relu`, `dual_sigmoid`,
`dual_sin`, `dual_sub`, `dual_tanh`, `jacfwd`, and `jvp_exact` to all
downstream crates. This `pub use` cascade is the ferrotorch analog of
Python's `from torch.autograd import *` import chain.

## Parity contract

`parity_ops = []` — `mod.rs` declares modules and re-exports; it has no
direct ops. Parity coverage for the cluster lands on the sibling
submodules' design docs (each lists `parity_ops = []` too, because the
autograd subsystem is mode/state plumbing rather than tensor-valued op
implementation).

## Verification

The 38-line module file is a declaration file; verification is
transitive through the sibling submodules' tests. Compile-time
verification: every `pub use` and `pub mod` at `mod.rs:1-38` must
resolve, enforced by `cargo check -p ferrotorch-core`. The four
clusters' tests live in their respective `#[cfg(test)] mod tests`
blocks (35+ tests in `graph.rs`, 14 in `no_grad.rs`, 16 in
`autocast_ops.rs`, 9 in `anomaly.rs`, 7 in `hooks.rs`, 4 in
`gradcheck.rs`, etc.).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub mod` declarations at `ferrotorch-core/src/autograd/mod.rs:1-13` declare 13 submodules; non-test production consumer: `ferrotorch-core/src/lib.rs:121-133` `pub use autograd::*` chain re-exports the surface to the crate root, then downstream callers like `ferrotorch-core/src/tensor.rs:87 hooks: Mutex<crate::autograd::hooks::HookStorage<T>>` and `ferrotorch-nn/src/transformer.rs:41 use ferrotorch_core::autograd::no_grad::is_grad_enabled` reach into the namespace. |
| REQ-2 | SHIPPED | impl: `pub use` chain at `mod.rs:15-38` re-exports `backward`, `grad`, `no_grad`, `autocast`, `gradcheck`, `fixed_point`, `grad_norm`, `gradient_penalty`, `jvp`, `vjp`, `DualTensor`, `jacfwd`, `jvp_exact`, `hessian`, `jacobian`, `enable_grad`, `inference_mode`, `set_grad_enabled`, `is_grad_enabled`, `is_inference_mode`, `AutocastDtype`, `AutocastCategory`, `AutocastEvent`, `autocast_category`, `autocast_guard`, `autocast_log`, `drain_autocast_events`, `should_cast_to_reduced`, `should_keep_full_precision`, `is_autocast_debug`, `set_autocast_debug`, `autocast_dtype`; non-test production consumer: `ferrotorch-core/src/lib.rs:125-133` `pub use autograd::{...}` cascade exposes the same identifiers at the crate root for use by every downstream model crate (ferrotorch-nn, ferrotorch-vision, ferrotorch-train, the 28 model crates). |
| REQ-3 | SHIPPED | impl: `pub use crate::ops::higher_order::{cond, scan, validate_cond_branches}` at `ferrotorch-core/src/autograd/mod.rs:26` keeps the legacy `autograd::cond` / `autograd::scan` import path alive; non-test production consumer: `ferrotorch-core/src/lib.rs:127` `pub use autograd::{... cond, ..., scan, ..., validate_cond_branches, ...}` flows the same identifiers up to the crate root and out to every downstream consumer that imports via `ferrotorch_core::cond` etc. |
