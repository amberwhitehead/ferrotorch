# ferrotorch-nn — `Parameter<T>`

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/parameter.py
-->

## Summary

`ferrotorch-nn/src/parameter.rs` defines the `Parameter<T>` wrapper: a
thin newtype around `Tensor<T>` that signals "this tensor participates
in autograd as a trainable parameter". It enforces `requires_grad =
true` on construction, derefs to `Tensor<T>` for all tensor
operations, and supports the optimizer-facing API (`set_data`,
`set_requires_grad`, `to(device)`). Mirrors PyTorch's
`torch.nn.Parameter` (`torch/nn/parameter.py:30-105`).

## Requirements

- REQ-1: `pub struct Parameter<T: Float> { data: Tensor<T> }` —
  Arc-backed via the inner `Tensor<T>` so cloning shares identity.
  Mirrors `torch.nn.Parameter` as a Tensor subclass with the
  `_is_param` flag (`torch/nn/parameter.py:30-70`); the Rust analog
  (R-DEV-7) replaces Python subclassing with a newtype.
- REQ-2: `Parameter::new(tensor)` constructor that forces
  `requires_grad = true` regardless of the input tensor's state.
  Mirrors `nn.Parameter(data, requires_grad=True)`
  (`torch/nn/parameter.py:51-70`).
- REQ-3: Convenience constructors `Parameter::zeros(shape)`,
  `Parameter::ones(shape)`, `Parameter::from_slice(data, shape)`
  for the common initialization patterns. Mirrors upstream's
  `nn.Parameter(torch.zeros(shape))` / `nn.Parameter(torch.ones(shape))`
  composition shorthand.
- REQ-4: `pub fn tensor(&self) -> &Tensor<T>` accessor and
  `pub fn into_tensor(self) -> Tensor<T>` consumer for unwrapping
  in callers that need the bare tensor. Mirrors implicit
  Tensor-of-a-Parameter access in PyTorch (you can call any
  `torch.Tensor` method on a `Parameter` directly because of
  subclassing).
- REQ-5: `pub fn set_data(&mut self, tensor: Tensor<T>)` — replaces
  the underlying data while re-enforcing `requires_grad = true`.
  Used by optimizers (`adam.step()`, etc.) to update parameter
  values without breaking the parameter identity semantics.
  Mirrors `Parameter.data = ...` assignment in upstream.
- REQ-6: `pub fn set_requires_grad(&mut self, requires_grad: bool)`
  for freezing/unfreezing parameters (#583). Mirrors
  `Parameter.requires_grad_(bool)` from
  `torch/nn/parameter.py:30-105` (inherited from Tensor).
- REQ-7: `pub fn to(&self, device: Device) -> FerrotorchResult<Self>`
  — returns a new `Parameter<T>` on the target device. Mirrors
  `Parameter.to(device)` (inherited from `torch.Tensor`).
- REQ-8: `impl Deref<Target = Tensor<T>>` so callers can transparently
  invoke any `Tensor<T>` method on a `Parameter<T>`. This is the
  Rust analog (R-DEV-7) of Python's class-subclass automatic method
  inheritance.
- REQ-9: `#[derive(Debug, Clone)]` — `Clone` is shallow (shares the
  inner `Arc<TensorStorage>` via `Tensor::clone`), so two clones
  of the same `Parameter<T>` point to the same storage. Matches
  PyTorch's `Parameter.__deepcopy__` semantics for "shallow
  reference, deep on demand".

## Acceptance Criteria

- [x] AC-1: `pub struct Parameter<T: Float>` with `data: Tensor<T>`.
- [x] AC-2: `Parameter::new(tensor)` enforces `requires_grad = true`.
- [x] AC-3: `zeros`, `ones`, `from_slice` factories.
- [x] AC-4: `tensor(&self)` and `into_tensor(self)` accessors.
- [x] AC-5: `set_data` re-enforces `requires_grad = true`.
- [x] AC-6: `set_requires_grad(bool)`.
- [x] AC-7: `to(Device)` returns a new on-device Parameter.
- [x] AC-8: `Deref<Target = Tensor<T>>`.
- [x] AC-9: `#[derive(Debug, Clone)]` with shallow clone semantics.
- [x] AC-10: `test_parameter_clone_shares_identity` pins that two
  clones share the underlying tensor identity (`is_same`).

## Architecture

### The wrapper struct (REQ-1, REQ-2)

```rust
#[derive(Debug, Clone)]
pub struct Parameter<T: Float> {
    data: Tensor<T>,
}

impl<T: Float> Parameter<T> {
    pub fn new(tensor: Tensor<T>) -> Self {
        Self { data: tensor.requires_grad_(true) }
    }
}
```

`Tensor::requires_grad_(true)` returns a new Tensor with the
flag set (immutable update through the Arc-backed storage). Any
input tensor — even one with `requires_grad = false` — is
upgraded to a trainable parameter on construction. This matches
upstream's invariant that "a Parameter is always trainable unless
explicitly frozen via `requires_grad_(false)`" — the default-true
behavior at construction.

### Convenience factories (REQ-3)

```rust
pub fn zeros(shape: &[usize]) -> FerrotorchResult<Self> {
    let t = ferrotorch_core::zeros::<T>(shape)?;
    Ok(Self::new(t))
}
```

The two `Parameter::ones` and `Parameter::from_slice` factories
delegate to `ferrotorch_core::ones` / `::from_slice` respectively.
These cover ~95% of parameter construction in concrete layer files
(`Linear::new` uses `Parameter::zeros` for weight + bias before
initialization runs over the tensor).

### Accessors (REQ-4)

```rust
pub fn tensor(&self) -> &Tensor<T> { &self.data }
pub fn into_tensor(self) -> Tensor<T> { self.data }
```

`tensor()` is the canonical "give me the underlying tensor"
accessor; `into_tensor` consumes the Parameter to avoid
unnecessary clones when the caller no longer needs the wrapper.

### `set_data` / `set_requires_grad` (REQ-5, REQ-6)

```rust
pub fn set_data(&mut self, tensor: Tensor<T>) {
    self.data = tensor.requires_grad_(true);
}

pub fn set_requires_grad(&mut self, requires_grad: bool) {
    let cloned = self.data.clone();
    self.data = cloned.requires_grad_(requires_grad);
}
```

`set_data` re-enforces `requires_grad = true` to maintain the
invariant ("a Parameter is always trainable"). `set_requires_grad`
goes through a `clone` because `Tensor::requires_grad_` takes
`self` by value — Tensor's design favors immutable updates, so
the freeze/unfreeze path allocates a new Arc handle (cheap).

### `to(device)` (REQ-7)

```rust
pub fn to(&self, device: Device) -> FerrotorchResult<Self> {
    Ok(Self::new(self.data.to(device)?))
}
```

Returns a new Parameter on the target device. Used by
`Module::to_device`'s default impl, which iterates `parameters_mut()`
and replaces each with `param.to(device)?`.

### `Deref<Target = Tensor<T>>` (REQ-8)

```rust
impl<T: Float> std::ops::Deref for Parameter<T> {
    type Target = Tensor<T>;
    fn deref(&self) -> &Self::Target { &self.data }
}
```

This is the load-bearing trait for ergonomics — callers write
`param.shape()` / `param.numel()` / `param.device()` directly,
the same way Python users call `parameter.shape` on a
`nn.Parameter`. Rust's Deref-as-method-resolution is the R-DEV-7
analog of Python's class-subclass method inheritance.

### Non-test production consumers

- `pub use parameter::Parameter` in `lib.rs:235`.
- `ferrotorch-nn/src/lib.rs` prelude re-exports `Parameter`.
- `ferrotorch-nn/src/linear.rs` lines 22, 46, 48, 83, 88 —
  `use crate::parameter::Parameter`, `pub weight: Parameter<T>`,
  `pub bias: Option<Parameter<T>>`, plus construction via
  `Parameter::zeros(&[out_features, in_features])` and bias.
- `ferrotorch-nn/src/conv.rs`, `embedding.rs`, `norm.rs`, every
  weight-bearing layer file — stores `Parameter<T>` fields.
- `ferrotorch-nn/src/init.rs` line 9 — Kaiming/Xavier init walks
  `Parameter<T>` references.
- `ferrotorch-optim/src/adam.rs` line 17 (and every optimizer in
  the crate: `adadelta`, `asgd`, `rprop`, `rmsprop`, `nadam`,
  `adagrad`, `adamax`) — `use ferrotorch_nn::Parameter` is the
  consumer of every parameter's gradient + value.
- `ferrotorch-nn/src/parameter_container.rs` — `ParameterList` and
  `ParameterDict` store `Parameter<T>` directly.
- `ferrotorch-nn/src/utils.rs` — `clip_grad_norm_` and
  `clip_grad_value_` take `&[&Parameter<T>]` slices.
- `ferrotorch-nn/src/hooks.rs` — `HookedModule::parameters` returns
  `Vec<&Parameter<T>>` from the inner module.

`Parameter<T>` is the canonical optimizer ↔ layer hand-off type.
Every layer stores parameters via this wrapper, and every optimizer
consumes them via the same wrapper.

## Parity contract

`parity_ops = []`. The wrapper is a structural type — its
correctness is "construction enforces `requires_grad = true` and
`Deref` exposes the underlying Tensor". Edge cases:

- **Construction from a `requires_grad = false` tensor**:
  upgraded to `true`. Matches upstream's
  `nn.Parameter(t, requires_grad=True)` default.
- **`set_data` from a `requires_grad = false` tensor**: same
  upgrade applies. The invariant "Parameter is always trainable
  unless explicitly frozen" is preserved across mutations.
- **Clone**: shallow — `param.clone()` produces a second handle to
  the same Arc-backed storage. `test_parameter_clone_shares_identity`
  pins `tensor.is_same(clone.tensor())`.
- **`to(Device::Cuda(n))` without a CUDA backend**: error
  surfaces from `Tensor::to`; the wrapper does not swallow it.

## Verification

Tests in `mod tests in parameter.rs` (5 tests):

- `test_parameter_requires_grad` — `zeros(...)` produces a
  parameter with `requires_grad == true`.
- `test_parameter_deref_to_tensor` — `param.shape()` /
  `param.numel()` work directly via Deref.
- `test_parameter_clone_shares_identity` — shallow clone preserves
  Arc identity.
- `test_parameter_to_cpu_preserves_data` — `to(Cpu)` roundtrip.
- `test_parameter_to_cuda_without_backend` — error case (no CUDA).

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-nn --lib parameter:: 2>&1 | tail -3
```

Expected: `5 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct Parameter<T: Float> { data: Tensor<T> }` with `#[derive(Debug, Clone)]` in `parameter.rs` mirroring `torch/nn/parameter.py:30-70` (R-DEV-7 newtype replacing the Tensor subclass); non-test consumer: `ferrotorch-nn/src/linear.rs:46` `pub weight: Parameter<T>` and dozens of other layer fields. |
| REQ-2 | SHIPPED | impl: `Parameter::new(tensor)` enforces `requires_grad = true` via `tensor.requires_grad_(true)` in `parameter.rs` mirroring `nn.Parameter(data, requires_grad=True)` (`torch/nn/parameter.py:51-70`); non-test consumer: `ferrotorch-nn/src/linear.rs:83-88` constructs `Parameter::zeros(...)` (which calls `Parameter::new`); same pattern in every weight-bearing layer. |
| REQ-3 | SHIPPED | impl: `Parameter::zeros` / `::ones` / `::from_slice` factories in `parameter.rs`; non-test consumer: `ferrotorch-nn/src/linear.rs:83` `Parameter::zeros(&[out_features, in_features])` and `linear.rs:88` `Parameter::zeros(&[out_features])`; `conv.rs`, `embedding.rs`, `norm.rs` follow the same pattern. |
| REQ-4 | SHIPPED | impl: `tensor(&self) -> &Tensor<T>` and `into_tensor(self) -> Tensor<T>` accessors in `parameter.rs`; non-test consumer: `ferrotorch-nn/src/module.rs` line 74 `param.tensor().clone()` inside `state_dict`; every optimizer reads `param.tensor()` to access the underlying storage. |
| REQ-5 | SHIPPED | impl: `pub fn set_data(&mut self, tensor)` re-enforces `requires_grad = true` in `parameter.rs` mirroring upstream `Parameter.data = ...`; non-test consumer: `ferrotorch-optim/src/adam.rs` and every optimizer step uses this path (via `parameters_mut()`) to update parameter values without breaking identity. |
| REQ-6 | SHIPPED | impl: `pub fn set_requires_grad(&mut self, bool)` in `parameter.rs` mirroring `torch/nn/parameter.py`'s `requires_grad_` (inherited from Tensor); non-test consumer: `ferrotorch-nn/src/module.rs` `Module::requires_grad_` calls `param.set_requires_grad(requires_grad)` in its default impl — invoked by transfer-learning code that freezes backbones. |
| REQ-7 | SHIPPED | impl: `pub fn to(&self, device) -> FerrotorchResult<Self>` in `parameter.rs`; non-test consumer: `ferrotorch-nn/src/module.rs` `Module::to_device` default impl calls `param.to(device)?` for each parameter — invoked by every model-to-device transfer in downstream code. |
| REQ-8 | SHIPPED | impl: `impl<T: Float> std::ops::Deref for Parameter<T>` with `Target = Tensor<T>` in `parameter.rs` (R-DEV-7 — Rust analog of Python class-subclass method inheritance); non-test consumer: every callsite that invokes a `Tensor<T>` method on a `Parameter<T>` (e.g. `param.shape()`, `param.device()`, `param.zero_grad()` in `Module::zero_grad`). |
| REQ-9 | SHIPPED | impl: `#[derive(Debug, Clone)]` on `Parameter<T>` in `parameter.rs` with shallow Arc-backed clone via the inner `Tensor<T>::clone`; non-test consumer: `Module::state_dict` default impl calls `param.tensor().clone()` for serialization; every layer that re-uses a shared parameter via clone (parameter-tying patterns) relies on the shallow Arc semantics. |
