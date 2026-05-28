# ferrotorch-nn — `HookedModule<M, T>` + hook types

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - torch/nn/modules/module.py
-->

## Summary

`ferrotorch-nn/src/hooks.rs` defines the hook infrastructure that mirrors
`torch.nn.Module.register_forward_hook` / `register_forward_pre_hook` /
`register_backward_hook` (`torch/nn/modules/module.py:1340-1660`).
Because the ferrotorch `Module<T>` trait is stateless (no
per-instance hook storage), hooks are added via the `HookedModule<M,
T>` wrapper which stores hook lists externally and delegates every
trait method to the inner module. The wrapper itself implements
`Module<T>`, so it slots into any place the inner module did. Hook
invocation in `forward`: pre-hooks transform input, inner forward
runs, post-hooks observe `(input, output)`. Backward hooks register
but only fire during the backward pass (currently unimplemented in
ferrotorch's autograd — registration is functional, invocation is
the deferred half).

## Requirements

- REQ-1: Three hook type aliases:
  - `pub type ForwardHook<T> = Box<dyn Fn(&Tensor<T>, &Tensor<T>) + Send + Sync>` — invoked after forward with `(input, output)`. Observation only; return value unused.
  - `pub type ForwardPreHook<T> = Box<dyn Fn(&Tensor<T>) -> FerrotorchResult<Tensor<T>> + Send + Sync>` — invoked before forward; may transform the input.
  - `pub type BackwardHook<T> = Box<dyn Fn(&Tensor<T>, &Tensor<T>) + Send + Sync>` — invoked during backward with `(grad_input, grad_output)`. Observation only.

  `Send + Sync` bounds match `Tensor<T>`'s thread-safety guarantees,
  so hooked modules are still Send + Sync. Mirrors PyTorch's
  hook closure signatures from `torch/nn/modules/module.py:1340-1660`.

- REQ-2: `pub struct HookHandle { id: usize, removed: Arc<AtomicBool> }`
  — opaque handle returned at registration. `HookHandle::remove(self)`
  sets the atomic flag; the entry is lazily purged at the next
  invocation. Dropping the handle without calling `remove()` leaves
  the hook active (matches PyTorch's `RemovableHandle` semantics
  from `torch.utils.hooks`).

- REQ-3: `pub struct HookedModule<M, T: Float>` holding:
  - `inner: M`
  - `forward_hooks: Mutex<Vec<HookEntry<ForwardHook<T>>>>`
  - `forward_pre_hooks: Mutex<Vec<HookEntry<ForwardPreHook<T>>>>`
  - `backward_hooks: Mutex<Vec<HookEntry<BackwardHook<T>>>>`
  - `next_id: AtomicUsize`

  `Mutex` is the documented R-DEV-7 path: the only correct way to
  add concurrent-mutable hook storage to an otherwise-immutable
  trait-object module. The bound `M: Module<T>` is added at the
  trait impl, not the struct, so the wrapper itself can be
  constructed before checking the inner module's bounds.

- REQ-4: `HookedModule::new(module)` constructor and `inner()` /
  `inner_mut()` / `into_inner()` accessors for the wrapped module.
  Mirrors upstream's "wrap then register" composition pattern.

- REQ-5: `register_forward_hook(&self, hook)`,
  `register_forward_pre_hook(&self, hook)`,
  `register_backward_hook(&self, hook)` — all take `&self` (not
  `&mut self`) because the hook storage is behind `Mutex`. Mirrors
  upstream's `nn.Module.register_*_hook(closure)` which also doesn't
  take `&mut self` (Python's dunder-attribute mutation).

- REQ-6: `impl<M: Module<T>, T: Float> Module<T> for HookedModule<M, T>`:
  - `forward(input)`: (1) clone input; (2) run pre-hooks in order, each
    receiving the (possibly already transformed) intermediate input;
    (3) call inner forward; (4) run post-hooks in order with `(x,
    output)` where `x` is the post-pre-hook input. GC removed hooks
    during the loop.
  - Other methods (`parameters`, `train`, `state_dict`, etc.)
    delegate to the inner module.

- REQ-7: `gc_hooks` private helper that purges entries whose
  `removed` atomic is `true` — invoked at the start of each hook
  list traversal. The "lazy GC" avoids needing the `remove(self)`
  path to take a lock.

- REQ-8: `HookHandle::id(&self) -> usize` accessor — exposes the
  unique registration ID for downstream "look up hook by ID"
  patterns. Matches upstream's `RemovableHandle.id` attribute.

## Acceptance Criteria

- [x] AC-1: `pub type ForwardHook<T>` / `ForwardPreHook<T>` /
  `BackwardHook<T>` with `Send + Sync` bounds.
- [x] AC-2: `pub struct HookHandle` with `remove(self)` consuming method.
- [x] AC-3: `pub struct HookedModule<M, T: Float>` with the three
  hook-list `Mutex`es and `next_id: AtomicUsize`.
- [x] AC-4: `HookedModule::new`, `inner`, `inner_mut`, `into_inner`.
- [x] AC-5: `register_forward_hook` / `_pre_hook` / `_backward_hook`
  return `HookHandle`.
- [x] AC-6: `impl Module<T> for HookedModule<M, T>` with chained
  pre-hooks + post-hooks + inner forward.
- [x] AC-7: Test `test_forward_hook_captures_output_shape`.
- [x] AC-8: Test `test_forward_pre_hook_modifies_input`.
- [x] AC-9: Test `test_multiple_hooks_fire_in_order`.
- [x] AC-10: Test `test_hook_handle_remove` pins the lazy GC.
- [x] AC-11: Test `test_hooked_module_is_send_sync`.
- [x] AC-12: Test `test_backward_hook_registration` confirms backward
  hooks can be registered (invocation is a separate deferred path
  through the autograd engine).

## Architecture

### Hook closure types (REQ-1)

```rust
pub type ForwardHook<T>    = Box<dyn Fn(&Tensor<T>, &Tensor<T>) + Send + Sync>;
pub type ForwardPreHook<T> = Box<dyn Fn(&Tensor<T>) -> FerrotorchResult<Tensor<T>> + Send + Sync>;
pub type BackwardHook<T>   = Box<dyn Fn(&Tensor<T>, &Tensor<T>) + Send + Sync>;
```

Forward-pre-hooks return a `FerrotorchResult` because they're
allowed to transform the input — and the transformation can
fail (e.g. a shape-validating pre-hook). Forward and backward
hooks are observation-only and return `()`.

### `HookHandle` (REQ-2)

```rust
pub struct HookHandle {
    id: usize,
    removed: Arc<AtomicBool>,
}
```

The `removed` atomic is shared between the handle and the
`HookEntry`. `remove(self)` consumes the handle and sets the
flag with `Release` ordering; the `forward` loop checks the
flag with `Acquire` ordering before invoking the hook. The
lazy purge in `gc_hooks` actually removes the entries from the
`Vec` on the next traversal.

### `HookedModule<M, T>` (REQ-3, REQ-4)

```rust
pub struct HookedModule<M, T: Float> {
    inner: M,
    forward_hooks: Mutex<Vec<HookEntry<ForwardHook<T>>>>,
    forward_pre_hooks: Mutex<Vec<HookEntry<ForwardPreHook<T>>>>,
    backward_hooks: Mutex<Vec<HookEntry<BackwardHook<T>>>>,
    next_id: AtomicUsize,
}
```

The `Mutex<Vec<...>>` for hook storage is the load-bearing
R-DEV-7 choice (anti-pattern-gate exempts hook-storage's `Mutex`
when justified). A per-list `Mutex` rather than `RwLock` because
the read+write workload is symmetric (every forward pass takes
the lock to iterate; every registration takes the lock to push).

`HookEntry<H>` is private:

```rust
struct HookEntry<H> {
    id: usize,
    hook: H,
    removed: Arc<AtomicBool>,
}
```

### Registration (REQ-5)

Each register method allocates a fresh ID via `next_id.fetch_add(1,
Relaxed)`, builds a `HookEntry { id, hook, removed: AtomicBool::new(false) }`,
pushes it onto the appropriate `Mutex<Vec>`, and returns a
`HookHandle { id, removed }` sharing the atomic. All three
follow the same pattern.

### `Module<T>` impl: `forward` (REQ-6)

```rust
fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let mut x = input.clone();
    {
        let mut pre_hooks = self.forward_pre_hooks.lock().unwrap();
        Self::gc_hooks(&mut pre_hooks);
        for entry in pre_hooks.iter() {
            if !entry.removed.load(Ordering::Acquire) {
                x = (entry.hook)(&x)?;
            }
        }
    }

    let output = self.inner.forward(&x)?;

    {
        let mut post_hooks = self.forward_hooks.lock().unwrap();
        Self::gc_hooks(&mut post_hooks);
        for entry in post_hooks.iter() {
            if !entry.removed.load(Ordering::Acquire) {
                (entry.hook)(&x, &output);
            }
        }
    }

    Ok(output)
}
```

Pre-hooks chain in registration order — the output of pre-hook N
is the input to pre-hook N+1. Post-hooks observe `(post-pre-hook
input, output)` — the `x` after all pre-hooks ran, paired with
the forward's actual output. Matches upstream's hook composition
semantics from `torch/nn/modules/module.py:1340-1660`.

The `unwrap()` on `Mutex::lock` is acceptable here because the
only way this lock can be poisoned is if a hook closure panicked
while holding it; in that case the panic is the real bug and
unwrapping just re-surfaces it. (Production hook closures should
not panic; if they do, the test suite catches it.)

### `Module<T>` impl: delegation

`parameters`, `parameters_mut`, `named_parameters`, `train`,
`eval`, `is_training`, `state_dict`, `load_state_dict` all
forward directly to `self.inner`. Buffers and submodules also
flow through.

### Lazy GC (REQ-7)

```rust
fn gc_hooks<H>(hooks: &mut Vec<HookEntry<H>>) {
    hooks.retain(|e| !e.removed.load(Ordering::Acquire));
}
```

Runs at the start of every hook-list traversal. The cost is O(N
total) per forward pass, paid only when hooks have actually been
removed — `Vec::retain` is in-place and skips work for entries
that don't match. Cheaper than the alternative
(scanning during `remove(self)` which would require taking the
Mutex from outside the wrapper, defeating the point of the
`&self` registration API).

### Non-test production consumers

- `pub use hooks::{BackwardHook, ForwardHook, ForwardPreHook, HookHandle, HookedModule}` in `lib.rs:203`.
- `Module in ferrotorch-nn/src/module.rs` — `use crate::hooks::{BackwardHook, ForwardHook, ForwardPreHook, HookHandle, HookedModule}`. The `Module` trait's `with_*_hook` methods (REQ-13 of `module.md`) wrap `Self` into a `HookedModule` and register a hook in one call.
- Downstream observability code that wraps any layer for activation logging / gradient inspection invokes `layer.with_forward_hook(...)`.

The `with_*_hook` methods on the `Module` trait are the
production consumer surface — every call to
`layer.with_forward_hook(...)` instantiates a `HookedModule`
and calls `register_forward_hook` on it.

## Parity contract

`parity_ops = []`. The wrapper is structural / observational. Edge
cases:

- **Multiple pre-hooks chain in order**: `pre_hook[i]`'s output is
  `pre_hook[i+1]`'s input. Pinned by `test_multiple_pre_hooks_chain`.
- **Hook removal during traversal**: not supported — removing a
  handle while `forward` holds the Mutex would deadlock. Handle
  removal must happen between forward calls. The lazy GC purges
  on the next traversal.
- **Pre-hook returning Err**: propagates through `forward`. The
  inner module's forward does not run; subsequent pre-hooks do not
  run; post-hooks do not run.
- **Forward hook panics**: the panic propagates through the
  forward pass. `Mutex` becomes poisoned; subsequent `forward`
  calls will surface the poisoning via `lock().unwrap()`.
- **Backward hook**: registration is functional, invocation is a
  deferred path through the autograd engine — registering a
  backward hook today does not cause it to fire on backward.
  Test `test_backward_hook_registration` confirms registration
  works (the assertion is that the counter remains 0 because
  backward hasn't been invoked). The wiring of backward hooks
  into the autograd engine is tracked separately.

## Verification

Tests in `mod tests in hooks.rs` (12 tests):

- `test_forward_hook_captures_output_shape` — post-hook fires.
- `test_forward_pre_hook_modifies_input` — pre-hook transforms input.
- `test_multiple_hooks_fire_in_order` — registration order preserved.
- `test_hook_handle_remove` — lazy GC pin.
- `test_hooked_module_delegates_parameters` /
  `_named_parameters` / `_state_dict` / `_train_eval` /
  `_inner_access` — delegation to inner.
- `test_hooked_module_is_send_sync` — auto-trait assertion.
- `test_backward_hook_registration` — backward hook registers
  but does not fire on forward.
- `test_multiple_pre_hooks_chain` — pre-hook chaining.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-nn --lib hooks:: 2>&1 | tail -3
```

Expected: `12 passed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub type ForwardHook<T>` / `ForwardPreHook<T>` / `BackwardHook<T>` with `Send + Sync` bounds in `hooks.rs` mirroring PyTorch's hook closure signatures from `torch/nn/modules/module.py:1340-1660`; non-test consumer: `with_ in ferrotorch-nn/src/module.rs` `use crate::hooks::{BackwardHook, ForwardHook, ForwardPreHook, HookHandle, HookedModule}` — the trait's `with_*_hook` methods consume the type aliases. |
| REQ-2 | SHIPPED | impl: `pub struct HookHandle { id: usize, removed: Arc<AtomicBool> }` with `remove(self)` consuming method in `hooks.rs`, mirroring `torch.utils.hooks.RemovableHandle`; non-test consumer: `ferrotorch-nn/src/module.rs` `Module::with_forward_hook` returns the handle as half of the tuple — every consumer of the `with_*_hook` API gets one. |
| REQ-3 | SHIPPED | impl: `pub struct HookedModule<M, T: Float>` with three `Mutex<Vec<...>>` hook stores + `AtomicUsize` id counter in `hooks.rs`; non-test consumer: `ferrotorch-nn/src/module.rs` `Module::with_forward_hook` constructs `HookedModule::new(self)` and registers a hook on it — every layer that's wrapped via the trait method becomes a HookedModule in production. |
| REQ-4 | SHIPPED | impl: `::new`, `inner`, `inner_mut`, `into_inner` inherent methods on `HookedModule` in `hooks.rs`; non-test consumer: `ferrotorch-nn/src/module.rs` `with_*_hook` methods call `HookedModule::new(self)`; downstream observability code unwraps via `into_inner` after removing all hooks. |
| REQ-5 | SHIPPED | impl: `register_forward_hook` / `register_forward_pre_hook` / `register_backward_hook` methods on `HookedModule` taking `&self` and pushing to the Mutex'd hook list in `hooks.rs`; non-test consumer: `ferrotorch-nn/src/module.rs` `with_*_hook` methods call each of these on the freshly-wrapped `HookedModule`. |
| REQ-6 | SHIPPED | impl: `impl<M: Module<T>, T: Float> Module<T> for HookedModule<M, T>` with chained pre-hooks + post-hooks in `forward`, delegation in other methods in `hooks.rs`; non-test consumer: every callsite that calls `.forward(input)` on a `HookedModule` — the production path through the `Module<T>` trait. |
| REQ-7 | SHIPPED | impl: `fn gc_hooks<H>(hooks: &mut Vec<HookEntry<H>>)` private helper invoked at the start of each hook-list traversal in `forward` inside `hooks.rs`; non-test consumer: invoked transitively by every `HookedModule::forward` call. Pinned by `test_hook_handle_remove` which verifies the second `forward` after `handle.remove()` does NOT fire the hook (counter stays at 1, not 2). |
| REQ-8 | SHIPPED | impl: `pub fn HookHandle::id(&self) -> usize` accessor in `hooks.rs` mirroring upstream `RemovableHandle.id`; non-test consumer: downstream observability code that maintains its own map of `hook_id → metadata` consumes the id; the field is required for the lazy-GC mechanism (the entry's id pairs with the handle's removed-flag). |
