# ferrotorch-nn — `Module` trait + `Reduction` + `StateDict`

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/modules/module.py
  - torch/nn/modules/__init__.py
-->

## Summary

`ferrotorch-nn/src/module.rs` defines the `Module<T>` trait that every
neural-network layer implements (the Rust analog of PyTorch's
`torch.nn.Module` base class), the `Reduction` enum used by loss
functions to select mean/sum/none reduction, and the `StateDict<T>`
type alias used for serialization. The trait composes parameter
iteration, buffer iteration, train/eval mode, device transfer,
sub-module walks, hook registration, gradient zeroing, and
state-dict load/save. Where upstream PyTorch implements this via a
runtime metaclass with hidden `_parameters`/`_buffers`/`_modules`
dicts, ferrotorch uses an explicit trait + derive macro
(R-DEV-7 + R-DEV-4 deviations).

## Requirements

- REQ-1: `pub type StateDict<T> = HashMap<String, Tensor<T>>` —
  the serialization map from dot-separated parameter/buffer paths to
  tensors. Mirrors `torch.nn.Module.state_dict()`'s `Dict[str, Tensor]`
  return type (`torch/nn/modules/module.py:1980-2060`).
- REQ-2: `pub enum Reduction { Mean, Sum, None }` — the reduction
  mode for loss functions. Mirrors PyTorch's
  `reduction: Literal["mean", "sum", "none"]` kwarg pattern used
  across `torch/nn/modules/loss.py`.
- REQ-3: `pub trait Module<T: Float>: Send + Sync` — the base trait.
  `Send + Sync` matches `Tensor<T>`'s thread-safety guarantees and is
  required because models are routinely moved across thread boundaries
  for parallel training. Mirrors `torch.nn.Module`
  (`torch/nn/modules/module.py:407-441`).
- REQ-4: Forward + parameter iteration: `fn forward(&self, input:
  &Tensor<T>) -> FerrotorchResult<Tensor<T>>`, `fn parameters(&self)
  -> Vec<&Parameter<T>>`, `fn parameters_mut(&mut self) -> Vec<&mut
  Parameter<T>>`, `fn named_parameters(&self) -> Vec<(String,
  &Parameter<T>)>`. Mirrors `torch.nn.Module.forward`,
  `.parameters()`, `.named_parameters()`
  (`torch/nn/modules/module.py:1660-1740`).
- REQ-5: Train/eval mode: `fn train(&mut self)`, `fn eval(&mut self)`,
  `fn is_training(&self) -> bool`. Mirrors
  `torch.nn.Module.train()` / `.eval()` / `.training` attribute
  (`torch/nn/modules/module.py:2350-2400`).
- REQ-6: Device transfer: `fn to_device(&mut self, device: Device)
  -> FerrotorchResult<()>` with a default implementation iterating
  `parameters_mut()` and `buffers_mut()` and transferring each via
  the `to(device)` method on each. Mirrors
  `torch.nn.Module.to(device)` (`torch/nn/modules/module.py:1180-1260`).
- REQ-7: State-dict serialization: `fn state_dict(&self) ->
  StateDict<T>` with a default implementation that unions
  `named_parameters()` with `named_buffers()`. Mirrors
  `torch.nn.Module.state_dict()` (`module.py:1980-2060`).
- REQ-8: State-dict deserialization: `fn load_state_dict(&mut self,
  state: &StateDict<T>, strict: bool) -> FerrotorchResult<()>`. When
  `strict == true`, unexpected or missing keys are errors; when
  `false`, extras are silently ignored and missing keys leave
  existing values unchanged. Shape mismatches always error.
  Mirrors `torch.nn.Module.load_state_dict(state, strict)`
  (`module.py:2150-2310`).
- REQ-9: Buffer iteration (#583): `fn buffers(&self) -> Vec<&Buffer<T>>`,
  `fn buffers_mut(&mut self) -> Vec<&mut Buffer<T>>`,
  `fn named_buffers(&self) -> Vec<(String, &Buffer<T>)>`. Default
  impls return empty (leaf module with no buffers). Mirrors
  `torch.nn.Module.buffers()` / `.named_buffers()`
  (`module.py:2430-2490`).
- REQ-10: `fn as_any(&self) -> Option<&dyn std::any::Any>` downcast
  hook (#984) for type-erased buffer-loader dispatch, used by
  BatchNorm's running-mean/variance/`num_batches_tracked` state that
  lives outside the `Buffer<T>` abstraction. Default returns `None`.
  Rust-specific (R-DEV-7) — no upstream analog because Python's
  duck-typing makes the downcast unnecessary.
- REQ-11: Submodule iteration: `fn children(&self) -> Vec<&dyn
  Module<T>>`, `fn named_children(&self) -> Vec<(String, &dyn
  Module<T>)>`, `fn modules(&self) -> Vec<&dyn Module<T>>`
  (depth-first including self, `Self: Sized`), `fn descendants_dyn(&self)
  -> Vec<&dyn Module<T>>` (object-safe; descendants only), `fn
  named_modules(...)` and `fn named_descendants_dyn(...)` paired
  with the above. Default impls return empty. Mirrors
  `torch.nn.Module.children()` / `.named_children()` /
  `.modules()` / `.named_modules()` (`module.py:2510-2640`).
- REQ-12: Empty-parent path normalisation in
  `named_descendants_dyn`. When a transparent-wrapper module
  exposes its inner child as `("", inner)`, the walker MUST emit
  the grandchild's path as `"backbone"` (not `".backbone"`) so
  state-dict keys round-trip. Pins the #1142 DeepLabV3 BN-buffer
  routing fix.
- REQ-13: Hook registration ergonomics (#606): `fn
  with_forward_hook(self, hook) -> (HookedModule<Self, T>,
  HookHandle)`, `fn with_forward_pre_hook(...)`, `fn
  with_backward_hook(...)` — all gated on `Self: Sized` so the
  trait stays dyn-compatible. Mirrors
  `torch.nn.Module.register_*_hook` calling conventions but returns
  the wrapped module rather than mutating in place (Rust-friendly).
- REQ-14: `fn zero_grad(&self) -> FerrotorchResult<()>` —
  iterates `parameters()` and calls `tensor.zero_grad()` on each.
  Mirrors `torch.nn.Module.zero_grad()` (`module.py:2700-2740`).
- REQ-15: `fn requires_grad_(&mut self, requires_grad: bool)` —
  toggles `requires_grad` on every parameter to freeze/unfreeze the
  module. Mirrors `torch.nn.Module.requires_grad_(bool)`
  (`module.py:2680-2700`).
- REQ-16: `fn apply_to_parameters(&mut self, f: &mut dyn
  FnMut(&mut Parameter<T>))` — applies a callback to every
  parameter. Mirrors `torch.nn.Module.apply(...)` for the
  parameter case. Object-safe (takes `&mut dyn FnMut` rather than a
  generic closure).

## Acceptance Criteria

- [x] AC-1: `pub type StateDict<T> = HashMap<String, Tensor<T>>` in `module.rs`.
- [x] AC-2: `pub enum Reduction { Mean, Sum, None }` with
  `#[derive(Debug, Clone, Copy, PartialEq, Eq)]`.
- [x] AC-3: `pub trait Module<T: Float>: Send + Sync` with all
  required methods.
- [x] AC-4: Default `to_device` iterates parameters + buffers.
- [x] AC-5: Default `state_dict` unions parameters + buffers.
- [x] AC-6: `load_state_dict(strict=true)` errors on unexpected and
  missing keys; shape mismatch always errors.
- [x] AC-7: `buffers`/`buffers_mut`/`named_buffers` default to empty.
- [x] AC-8: `as_any` default returns `None`.
- [x] AC-9: `modules` (self-first, `Self: Sized`) and
  `descendants_dyn` (object-safe, descendants-only) both exist.
- [x] AC-10: Empty-parent path branch in
  `named_descendants_dyn` avoids leading-dot.
- [x] AC-11: `with_forward_hook` / `with_forward_pre_hook` /
  `with_backward_hook` return `(HookedModule<Self, T>, HookHandle)`.
- [x] AC-12: `zero_grad`, `requires_grad_`, `apply_to_parameters`
  default impls present.
- [x] AC-13: Test `module_named_descendants_dyn_empty_parent_no_leading_dot`
  pins the #1142 regression.

## Architecture

### `StateDict<T>` (REQ-1)

```rust
pub type StateDict<T> = HashMap<String, Tensor<T>>;
```

Keys are dot-separated paths like `"layer1.weight"` or
`"backbone.bn1.running_mean"`. Values are `Tensor<T>` (the underlying
tensor, not `Parameter<T>` / `Buffer<T>` — the wrapper type is
re-inferred at load time). Mirrors `torch.nn.Module.state_dict()`'s
`Dict[str, Tensor]` return type.

### `Reduction` (REQ-2)

The variants exactly mirror PyTorch's `Literal["mean", "sum", "none"]`
loss reduction string kwarg. Stored as a Rust enum rather than a
string for compile-time validation and pattern-match exhaustiveness
(R-DEV-7 — Rust ecosystem analog is materially better).

### `Module<T>` trait (REQ-3..16)

The trait is generic over `T: Float` (the element dtype). `Send + Sync`
is a hard requirement — every concrete module type must be safely
transferable across threads, matching `Tensor<T>`'s thread-safety
guarantee. Object-safe methods can be used through
`&dyn Module<T>` / `Box<dyn Module<T>>`; `Self: Sized` methods are
the inherent-syntax sugar variants (`modules()`, `named_modules()`,
`with_*_hook()`).

The trait composes:
- **Forward + parameters** (REQ-4): `forward`, `parameters`,
  `parameters_mut`, `named_parameters`.
- **Mode** (REQ-5): `train`, `eval`, `is_training`.
- **Device** (REQ-6): `to_device` (default impl iterates
  parameters + buffers and calls `to(device)` on each).
- **State-dict** (REQ-7, REQ-8): `state_dict`, `load_state_dict`.
- **Buffers** (REQ-9, REQ-10): `buffers`, `buffers_mut`,
  `named_buffers`, `as_any` (downcast hook for BN-style state
  outside the `Buffer<T>` abstraction — see #984 for the rationale
  preventing a premature unifying setter).
- **Submodules** (REQ-11, REQ-12): `children`, `named_children`,
  `modules`, `descendants_dyn`, `named_modules`,
  `named_descendants_dyn`. The empty-parent branch in
  `named_descendants_dyn` is load-bearing for the #1142
  DeepLabV3 BN-buffer routing fix — without it, transparent wrappers
  produce paths with a leading dot that mismatches state-dict keys.
- **Hooks** (REQ-13): `with_forward_hook`, `with_forward_pre_hook`,
  `with_backward_hook` return a `(HookedModule<Self, T>, HookHandle)`
  pair — wrapping the module in `HookedModule` and registering the
  hook in one call.
- **Helpers** (REQ-14..16): `zero_grad`, `requires_grad_`,
  `apply_to_parameters`.

### Default `load_state_dict` (REQ-8)

Two-pass approach: first collect known keys (union of
`named_parameters()` and `named_buffers()`), then iterate the
state-dict and reject unexpected keys when `strict == true`.
Second pass collects parameter names + `parameters_mut()` and
zips them, reading the corresponding tensor from the state-dict
(shape-check, then `*param = Parameter::new(tensor.clone())`). Same
dance for buffers. Mirrors `torch.nn.Module.load_state_dict`'s
two-pass shape-check + assign behaviour
(`torch/nn/modules/module.py:2150-2310`).

### Default `to_device` (REQ-6)

```rust
fn to_device(&mut self, device: Device) -> FerrotorchResult<()> {
    for param in self.parameters_mut() {
        *param = param.to(device)?;
    }
    for buffer in self.buffers_mut() {
        *buffer = buffer.to(device)?;
    }
    Ok(())
}
```

Concrete layers override only if they have non-`Parameter<T>` /
`Buffer<T>` state that also needs device transfer (rare — BatchNorm's
`Mutex<Vec<f64>>` running stats stay on host by design).

### Hook-registration ergonomics (REQ-13)

The `with_*_hook` methods consume `self` and return a tuple
`(HookedModule<Self, T>, HookHandle)`. They're gated on `Self: Sized`
so the trait stays dyn-compatible. The naming uses `with_*` rather
than `register_*` to avoid clashing with `HookedModule`'s own
inherent `register_*` methods (which append a hook to an
already-wrapped instance), so the two compose:
`Linear::new(..)?.with_forward_hook(h1).0.register_forward_hook(h2)`.

### Non-test production consumers

- `ferrotorch-nn/src/linear.rs` — `impl<T: Float> Module<T> for Linear<T>`; declares parameters via `Parameter<T>` and forwards through linear's matmul + optional bias.
- `ferrotorch-nn/src/conv.rs` — every `Conv*` impl.
- `ferrotorch-nn/src/norm.rs` — `LayerNorm`/`BatchNorm*`/`GroupNorm`/`RMSNorm`; BN also overrides `as_any()` for the running-stats downcast hook.
- `ferrotorch-nn/src/container.rs` — `Sequential`/`ModuleList`/`ModuleDict` consume the trait via `Box<dyn Module<T>>` to compose layers.
- `ferrotorch-nn/src/hooks.rs` — `HookedModule<M, T>` consumes `M: Module<T>`.
- `ferrotorch-train/src/grad_utils.rs` and `ferrotorch-optim/src/optimizer.rs` consume `.parameters()`.

The trait is the central abstraction; every layer file in the crate
implements it, and `ferrotorch-train`, `ferrotorch-optim`, every
model crate consumes it.

## Parity contract

`parity_ops = []`. The trait is a structural abstraction with no
numerical contract of its own — the parity is on its consumers
(`Linear`, `Conv2d`, `LayerNorm`, etc.). Edge-case parity the trait
itself owns:

- **Empty-parent transparent wrappers** (#1142): `named_descendants_dyn`
  MUST NOT prepend a leading `.` when the parent path is `""`. Pinned
  by `module_named_descendants_dyn_empty_parent_no_leading_dot`.
- **Strict state-dict load**: unexpected keys → `InvalidArgument`,
  missing keys → `InvalidArgument`, shape mismatch → `ShapeMismatch`.
- **Buffers included in state_dict** by default — matches upstream
  (BN's `running_mean` is in `state_dict()`).
- **`Module: Send + Sync`** — every concrete impl in the crate
  satisfies this; the marker bound is checked at impl site.

## Verification

Tests in `mod tests in module.rs` (24 tests):

- `test_module_parameters` / `test_module_named_parameters` /
  `test_module_train_eval` — basic trait surface.
- `test_module_state_dict_roundtrip` / `_strict_extra_key` /
  `_shape_mismatch` — load_state_dict strictness.
- `test_module_is_send_sync` — auto-trait assertion.
- `test_reduction_enum` — Reduction equality.
- `test_to_device_cpu_preserves_weights` / `_cuda_without_backend`.
- `module_buffers_default_is_empty` / `_listed_for_overriding_module`.
- `module_children_listed_for_parent`.
- `module_named_modules_includes_self_and_descendants` /
  `module_modules_includes_self_and_descendants`.
- `module_zero_grad_succeeds`.
- `module_requires_grad_toggles_all_parameters`.
- `module_apply_to_parameters_visits_all`.
- `module_state_dict_includes_buffers` /
  `module_load_state_dict_with_buffer`.
- `module_descendants_dyn_excludes_self` /
  `module_named_descendants_dyn_paths`.
- `module_named_descendants_dyn_empty_parent_no_leading_dot` —
  #1142 regression lock.
- `with_forward_hook_wraps_and_fires` / `_pre_hook_wraps_and_fires` /
  `_backward_hook_returns_handle`.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-nn --lib module:: 2>&1 | tail -3
```

Expected: `24 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub type StateDict<T> = HashMap<String, Tensor<T>>` in `module.rs` mirroring `torch/nn/modules/module.py:1980-2060`; non-test consumer: `pub use module::{Module, Reduction, StateDict}` in `lib.rs:218` and `ferrotorch-nn/src/hooks.rs` (`use crate::module::{Module, StateDict}`) plus every layer impl that returns `state_dict()`. |
| REQ-2 | SHIPPED | impl: `pub enum Reduction { Mean, Sum, None }` in `module.rs` mirroring `torch/nn/modules/loss.py`'s `reduction` kwarg pattern; non-test consumer: `ferrotorch-nn/src/loss.rs:19` `use crate::module::Reduction` (consumed by every loss type) and `ferrotorch-nn/src/functional.rs:1798` `use crate::module::Reduction`. |
| REQ-3 | SHIPPED | impl: `pub trait Module<T: Float>: Send + Sync` in `module.rs` mirroring `torch/nn/modules/module.py:407-441`; non-test consumer: every layer file in `ferrotorch-nn/src/` (`linear.rs`, `conv.rs`, `norm.rs`, `embedding.rs`, `dropout.rs`, `identity.rs`, `container.rs`, etc.) `impl Module<T>`. |
| REQ-4 | SHIPPED | impl: `forward` / `parameters` / `parameters_mut` / `named_parameters` as required methods in the trait; non-test consumer: `ferrotorch-optim/src/optimizer.rs:5` and `ferrotorch-optim/src/adam.rs:17` consume `.parameters()` via `use ferrotorch_nn::Parameter`. |
| REQ-5 | SHIPPED | impl: `train` / `eval` / `is_training` required methods in the trait; non-test consumer: `ferrotorch-nn/src/container.rs` propagates `train()`/`eval()` to child layers in `Sequential::train`/`Sequential::eval`. |
| REQ-6 | SHIPPED | impl: `fn to_device` with default impl iterating `parameters_mut()` and `buffers_mut()` in `module.rs` mirroring `torch/nn/modules/module.py:1180-1260`; non-test consumer: model-composition code in downstream crates calls `model.to_device(Device::Cuda(0))` to move whole sub-trees. |
| REQ-7 | SHIPPED | impl: default `fn state_dict` unioning `named_parameters()` with `named_buffers()` in `module.rs` mirroring `module.py:1980-2060`; non-test consumer: `ferrotorch-nn/src/hooks.rs` `HookedModule::state_dict` delegates to inner; `ferrotorch-serialize` reads `state_dict()` for SafeTensors export. |
| REQ-8 | SHIPPED | impl: `load_state_dict(strict)` two-pass strict-check + shape-validate + assign in `module.rs` mirroring `module.py:2150-2310`; non-test consumer: `ferrotorch-nn/src/hooks.rs` `HookedModule::load_state_dict` delegates; downstream model loaders (SafeTensors, GGUF) call it. |
| REQ-9 | SHIPPED | impl: `buffers` / `buffers_mut` / `named_buffers` default-empty methods in `module.rs` mirroring `module.py:2430-2490`; non-test consumer: `ferrotorch-nn/src/module.rs` line 374 `*buf = Buffer::new(tensor.clone())` inside `load_state_dict`; `ferrotorch-nn/src/norm.rs` `BatchNorm*` overrides `buffers()` / `named_buffers()`. |
| REQ-10 | SHIPPED | impl: `fn as_any(&self) -> Option<&dyn Any>` default `None` in `module.rs`; non-test consumer: `ferrotorch-nn/src/norm.rs` `BatchNorm*` overrides `as_any` to expose the running-stats downcast hook (#984) consumed by `ferrotorch-vision`'s state-dict loader walking `named_modules()`. |
| REQ-11 | SHIPPED | impl: `children` / `named_children` / `modules` (`Self: Sized`) / `descendants_dyn` (object-safe) / `named_modules` / `named_descendants_dyn` in `module.rs` mirroring `module.py:2510-2640`; non-test consumer: `ferrotorch-nn/src/container.rs` `Sequential`/`ModuleList`/`ModuleDict` traverse children; state-dict loaders walk `named_modules()`. |
| REQ-12 | SHIPPED | impl: empty-parent branch (the `if name.is_empty()` arm) in `named_descendants_dyn` inside `module.rs`, fixing #1142 (DeepLabV3 BN-buffer routing); non-test consumer: `ferrotorch-vision`'s DeepLabV3 backbone walker — the BN-buffer load path consumes the leading-dot-free paths via `named_modules()`. |
| REQ-13 | SHIPPED | impl: `with_forward_hook` / `with_forward_pre_hook` / `with_backward_hook` `Self: Sized` methods in `module.rs` returning `(HookedModule<Self, T>, HookHandle)`; non-test consumer: `ferrotorch-nn/src/hooks.rs` `HookedModule` is the production wrapper instantiated by these methods; downstream observability code wraps layers via `linear.with_forward_hook(...)`. |
| REQ-14 | SHIPPED | impl: default `fn zero_grad` iterating `parameters()` in `module.rs` mirroring `module.py:2700-2740`; non-test consumer: `ferrotorch-optim/src/optimizer.rs` and `ferrotorch-train/src/grad_utils.rs` call `module.zero_grad()` before each training step. |
| REQ-15 | SHIPPED | impl: default `fn requires_grad_` toggling all params in `module.rs` mirroring `module.py:2680-2700`; non-test consumer: transfer-learning code in downstream crates freezes the backbone via `backbone.requires_grad_(false)` before fine-tuning. |
| REQ-16 | SHIPPED | impl: default `fn apply_to_parameters(&mut self, f: &mut dyn FnMut(&mut Parameter<T>))` in `module.rs` mirroring upstream `torch.nn.Module.apply` for the parameter case; non-test consumer: init code in downstream layers (e.g. lazy parameter materialization in `ferrotorch-nn/src/lazy_linear.rs`) walks parameters via this hook. |
