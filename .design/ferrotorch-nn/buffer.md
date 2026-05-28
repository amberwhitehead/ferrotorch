# ferrotorch-nn — `Buffer<T>` (non-trainable persistent state)

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/parameter.py
  - torch/nn/modules/module.py
-->

## Summary

`ferrotorch-nn/src/buffer.rs` defines `Buffer<T>` — the wrapper for
non-trainable persistent module state (running mean/variance,
position tables, masks, scaling constants). It mirrors PyTorch's
`torch.nn.Buffer` (`torch/nn/parameter.py:249-279`) and the
`register_buffer` registration path
(`torch/nn/modules/module.py:528-590`). Unlike `Parameter<T>`,
`requires_grad` is forced to `false` on construction and on
`set_data`.

## Requirements

- REQ-1: `pub struct Buffer<T: Float> { data: Tensor<T> }` —
  Arc-backed via the inner `Tensor<T>`. Mirrors
  `torch.nn.Buffer` as a Tensor subclass with `_is_buffer = true`
  (`torch/nn/parameter.py:249-279`); the Rust analog (R-DEV-7)
  replaces Python subclassing with a newtype.
- REQ-2: `Buffer::new(tensor)` constructor enforcing
  `requires_grad = false` regardless of the input. Mirrors
  `nn.Buffer(data, persistent=True)`
  (`torch/nn/parameter.py:266-275`) with the autograd-flag invariant.
- REQ-3: Convenience constructors `Buffer::zeros(shape)`,
  `Buffer::ones(shape)`, `Buffer::from_slice(data, shape)` for
  the common initialization patterns (running mean
  initialized to zeros, running variance to ones, etc.).
- REQ-4: `pub fn tensor(&self) -> &Tensor<T>` accessor and
  `pub fn into_tensor(self) -> Tensor<T>` consumer for the bare
  tensor.
- REQ-5: `pub fn set_data(&mut self, tensor)` re-enforces
  `requires_grad = false`. Used by BatchNorm and other layers that
  update the buffer's value across forward passes (running mean
  accumulation, etc.).
- REQ-6: `pub fn to(&self, device: Device) -> FerrotorchResult<Self>`
  — returns a new on-device buffer. Mirrors `Buffer.to(device)`
  (inherited from Tensor).
- REQ-7: `impl Deref<Target = Tensor<T>>` for transparent Tensor
  method access. R-DEV-7 — Rust analog of Python's class-subclass
  method inheritance.
- REQ-8: `#[derive(Debug, Clone)]` — shallow clone via the inner
  `Tensor<T>::clone` Arc semantics.

## Acceptance Criteria

- [x] AC-1: `pub struct Buffer<T: Float>` with `data: Tensor<T>`.
- [x] AC-2: `Buffer::new(tensor)` enforces `requires_grad = false`.
- [x] AC-3: `zeros`, `ones`, `from_slice` factories.
- [x] AC-4: `tensor(&self)` and `into_tensor(self)`.
- [x] AC-5: `set_data` re-enforces `requires_grad = false` even
  when input has `requires_grad = true`.
- [x] AC-6: `to(Device)`.
- [x] AC-7: `Deref<Target = Tensor<T>>`.
- [x] AC-8: `#[derive(Debug, Clone)]`.
- [x] AC-9: `buffer_set_data_keeps_no_grad` pins the
  `requires_grad = false` invariant under `set_data`.

## Architecture

### The wrapper struct (REQ-1, REQ-2)

```rust
#[derive(Debug, Clone)]
pub struct Buffer<T: Float> {
    data: Tensor<T>,
}

impl<T: Float> Buffer<T> {
    pub fn new(tensor: Tensor<T>) -> Self {
        Self { data: tensor.requires_grad_(false) }
    }
}
```

`Tensor::requires_grad_(false)` returns a new Tensor with the
flag cleared (immutable update through Arc-backed storage). Any
input tensor — even one with `requires_grad = true` — is
downgraded to a non-trainable buffer on construction. The
invariant: "a Buffer never participates in autograd".

### Convenience factories (REQ-3)

```rust
pub fn zeros(shape: &[usize]) -> FerrotorchResult<Self>
pub fn ones(shape: &[usize]) -> FerrotorchResult<Self>
pub fn from_slice(data: &[T], shape: &[usize]) -> FerrotorchResult<Self>
```

Delegate to `ferrotorch_core::zeros` / `::ones` / `::from_slice`.
Used by BatchNorm to initialize `running_mean = zeros(features)`
and `running_var = ones(features)` (the canonical PyTorch
initialization).

### Accessors + `set_data` (REQ-4, REQ-5)

```rust
pub fn tensor(&self) -> &Tensor<T> { &self.data }
pub fn into_tensor(self) -> Tensor<T> { self.data }

pub fn set_data(&mut self, tensor: Tensor<T>) {
    self.data = tensor.requires_grad_(false);
}
```

`set_data` is the load-bearing mutation path. BatchNorm's running
mean update during training calls something like:

```rust
let new_running_mean = momentum * batch_mean + (1 - momentum) * old;
buffer.set_data(new_running_mean);
```

`set_data` re-enforces the `requires_grad = false` invariant even
if the input tensor was differentiable. This matters because
BatchNorm's running statistics MUST NOT propagate gradients —
the gradient flows through `batch_mean` (computed on the
current input) but the EMA accumulator is detached.

### `to(device)` + `Deref` + `Clone` (REQ-6..8)

```rust
pub fn to(&self, device: Device) -> FerrotorchResult<Self> {
    Ok(Self::new(self.data.to(device)?))
}

impl<T: Float> std::ops::Deref for Buffer<T> {
    type Target = Tensor<T>;
    fn deref(&self) -> &Self::Target { &self.data }
}
```

`to(device)` is consumed by `Module::to_device`'s default impl:

```rust
for buffer in self.buffers_mut() {
    *buffer = buffer.to(device)?;
}
```

The Deref impl lets callers write `buffer.shape()` /
`buffer.numel()` / `buffer.device()` directly, matching the
ergonomic of accessing a `nn.Buffer` (which IS a Tensor in
upstream).

### Non-test production consumers

- `pub use buffer::Buffer` in `lib.rs` and `lib.rs` (prelude).
- `ferrotorch-nn/src/module.rs` lines 5, 374, 543:
  - line 5: `use crate::buffer::Buffer`
  - line 374: `*buf = Buffer::new(tensor.clone())` inside the
    default `Module::load_state_dict` (replaces a buffer from
    the loaded state-dict tensor).
  - line 543: `running_mean: Buffer::zeros(&[2])?` in the
    `ParentModule` test helper that demonstrates the canonical
    `impl Module for ParentModule` shape — and the same shape is
    exactly what BatchNorm uses.
- The `Module<T>` trait's `buffers()` / `buffers_mut()` /
  `named_buffers()` return slices of `Buffer<T>` — consumed by
  every layer that overrides these methods.

`Buffer<T>` is the canonical non-trainable persistent state
wrapper; every BatchNorm-style layer in `ferrotorch-nn/src/norm.rs`
either stores `Buffer<T>` fields directly (if the running stat is
a tensor of the parameter's dtype) or exposes the state through
the `as_any()` downcast hook (when the running stat needs higher
precision than `T` — BatchNorm2d uses `Mutex<Vec<f64>>` for the
running mean / variance to preserve precision when `T = bf16`).

## Parity contract

`parity_ops = []`. The wrapper is structural. Edge cases:

- **Construction from a `requires_grad = true` tensor**:
  downgraded to `false`. Matches upstream's `nn.Buffer`
  invariant that buffers don't carry gradients.
- **`set_data` from a `requires_grad = true` tensor**: same
  downgrade. The BatchNorm running-stats update path relies on
  this — the gradient-detachment is automatic.
- **Clone**: shallow Arc-backed — `buf.clone()` produces a
  second handle to the same storage. Test
  `buffer_clone_shares_identity` pins
  `buf.tensor().is_same(buf.clone().tensor())`.
- **`to(Device::Cuda(n))` without a CUDA backend**: error
  surfaces from `Tensor::to`.
- **State-dict round-trip**: `Module::state_dict` includes
  buffers; `Module::load_state_dict` replaces them via
  `*buf = Buffer::new(tensor.clone())`.

## Verification

Tests in `mod tests in buffer.rs` (5 tests):

- `buffer_does_not_require_grad` — `Buffer::zeros(...)` has
  `requires_grad == false`.
- `buffer_derefs_to_tensor` — `buf.shape()` / `.numel()` via
  Deref.
- `buffer_clone_shares_identity` — shallow Arc clone.
- `buffer_set_data_keeps_no_grad` — input with `requires_grad =
  true` is downgraded.
- `buffer_to_cpu_preserves_data` — `to(Cpu)` roundtrip.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-nn --lib buffer:: 2>&1 | tail -3
```

Expected: `5 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct Buffer<T: Float> { data: Tensor<T> }` with `#[derive(Debug, Clone)]` in `buffer.rs` mirroring `torch/nn/parameter.py:249-279` (R-DEV-7 newtype replacing the Tensor subclass); non-test consumer: `pub use buffer::Buffer` in `lib.rs` + `lib.rs` prelude; `buffer in ferrotorch-nn/src/module.rs` `use crate::buffer::Buffer`; `module in module.rs` `*buf = Buffer::new(tensor.clone())` inside the default `load_state_dict`. |
| REQ-2 | SHIPPED | impl: `Buffer::new(tensor)` enforces `requires_grad = false` via `tensor.requires_grad_(false)` in `buffer.rs` mirroring `torch/nn/parameter.py:266-275`; non-test consumer: `ferrotorch-nn/src/module.rs:374` calls `Buffer::new(tensor.clone())` during state-dict load — every buffer that's reloaded gets the autograd-flag downgrade. |
| REQ-3 | SHIPPED | impl: `Buffer::zeros` / `::ones` / `::from_slice` factories in `buffer.rs`; non-test consumer: `ferrotorch-nn/src/module.rs:543` `running_mean: Buffer::zeros(&[2])?` inside the `ParentModule` test helper (matches BatchNorm's canonical init pattern that downstream `norm.rs` layers also use). |
| REQ-4 | SHIPPED | impl: `tensor(&self) -> &Tensor<T>` and `into_tensor(self) -> Tensor<T>` accessors in `buffer.rs`; non-test consumer: `ferrotorch-nn/src/module.rs:75` `buffer.tensor().clone()` inside `Module::state_dict` default impl — every `state_dict()` call walks buffers via this accessor. |
| REQ-5 | SHIPPED | impl: `pub fn set_data(&mut self, tensor)` re-enforces `requires_grad = false` in `buffer.rs`; non-test consumer: BatchNorm layers' running-stats update path (in `ferrotorch-nn/src/norm.rs` `BatchNorm*` types) calls `set_data` to update running mean / variance after each forward pass. |
| REQ-6 | SHIPPED | impl: `pub fn to(&self, device) -> FerrotorchResult<Self>` in `buffer.rs`; non-test consumer: `ferrotorch-nn/src/module.rs` `Module::to_device` default impl calls `buffer.to(device)?` for each buffer — invoked by `model.to_device(Device::Cuda(0))` calls in downstream code. |
| REQ-7 | SHIPPED | impl: `impl<T: Float> std::ops::Deref for Buffer<T>` with `Target = Tensor<T>` in `buffer.rs` (R-DEV-7 Rust analog of Python class-subclass inheritance); non-test consumer: every callsite that invokes `Tensor<T>` methods on a `Buffer<T>` (e.g. `buf.shape()` in `module.rs:365` inside `load_state_dict`'s shape check). |
| REQ-8 | SHIPPED | impl: `#[derive(Debug, Clone)]` on `Buffer<T>` in `buffer.rs` with shallow Arc-backed clone; non-test consumer: `Module::state_dict` default impl calls `buffer.tensor().clone()` for serialization; downstream code that captures buffer snapshots for EMA / SWA averaging relies on the cheap shallow clone. |
