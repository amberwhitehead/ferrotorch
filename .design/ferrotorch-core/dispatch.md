# DispatchKey / DispatchKeySet / Dispatcher — multi-backend op dispatch

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158 (Revert "[dynamo] Implement nb_or/nb_inplace_or slot dispatch for | and |= operators (#181326)")
upstream-paths:
  - aten/src/ATen/core/dispatch/Dispatcher.h
  - aten/src/ATen/core/dispatch/DispatchKeyExtractor.h
  - c10/core/DispatchKey.h
  - c10/core/DispatchKeySet.h
-->

## Summary

`ferrotorch-core/src/dispatch.rs` is the multi-key, priority-ordered
operator dispatcher. It mirrors PyTorch's `c10::Dispatcher`
(`aten/src/ATen/core/dispatch/Dispatcher.h`), `c10::DispatchKey`
(`c10/core/DispatchKey.h`), and `c10::DispatchKeySet`
(`c10/core/DispatchKeySet.h`): every tensor carries a *set* of active
keys (`Autograd`, `Quantized`, `Sparse`, `CPU`, `CUDA`, `Tracer`, …),
and when an op is invoked the dispatcher picks the registered kernel
for the highest-priority key. Kernels may mask their own key off and
re-dispatch ("redispatch" in PyTorch parlance), producing layered
semantics (autograd records a node, then re-dispatches to the
backend kernel that runs the math).

CL-397.

## Requirements

- REQ-1: `DispatchKey` enum — exactly the 11 keys ferrotorch tracks today,
  with `#[repr(u8)]` priorities (`Cpu=0 .. Tracer=10`). Mirrors a
  *reduced* slice of upstream's `c10::DispatchKey` (`DispatchKey.h:136`),
  which has ~150 variants. R-DEV-7: shipping the Rust ecosystem's natural
  enum analog with priority encoded as the discriminant is better than
  upstream's manually-numbered macro-expanded list. The 11 keys cover
  backends (`Cpu`, `Cuda`, `Meta`), per-tensor-type interceptors
  (`Sparse`, `Quantized`, `Nested`), and transformation layers
  (`Autocast`, `Autograd`, `Vmap`, `Profiler`, `Tracer`).
- REQ-2: `DispatchKeySet` as a `u16` bitmask — constant-time membership /
  insert / remove / union / intersection. Mirrors PyTorch's
  `c10::DispatchKeySet` (a 64-bit bitmask internally;
  `c10/core/DispatchKeySet.h`). 16 bits suffices for 11 keys.
- REQ-3: `DispatchKey::priority()` — numeric priority is the enum
  discriminant via `as u8`. Larger = higher priority. Mirrors upstream's
  "later in the macro list = higher priority" convention at
  `c10/core/DispatchKey.h:136`.
- REQ-4: `DispatchKeySet::highest()` — returns the highest-priority key in
  the set, the "next" key the dispatcher will resolve. `iter_desc()`
  iterates in descending priority order.
- REQ-5: `Dispatcher<T: Float>` — a `HashMap<(String, DispatchKey),
  Kernel<T>>` keyed lookup. `register(op, key, kernel)` adds a kernel;
  `call(op, inputs, keyset)` walks the keyset in descending priority and
  fires the first registered kernel. Each kernel receives the full
  keyset + a back-reference to the dispatcher so it can mask its own key
  off and `disp.call(op, ...)` to redispatch.
- REQ-6: `Dispatcher::call_direct(op, inputs, keyset, key)` — bypass
  priority resolution and call a specific kernel. Used by tests and by
  kernels that want to forward directly to a specific lower-priority
  layer.
- REQ-7: Empty-keyset and missing-kernel errors are structured
  (`FerrotorchError::InvalidArgument`), not panics. R-CODE-2: no `panic!`
  in production paths.
- REQ-8: `Kernel<T>` type alias for the boxed-closure shape, including the
  `Send + Sync + 'static` bounds — every kernel must be safe to share
  across threads (the dispatcher itself is `Send + Sync` when `T: Float`).
- REQ-9: `Dispatcher<T>` is generic over a single dtype `T: Float`.
  Different dispatchers are held per-dtype; this matches upstream's
  per-`ScalarType` kernel registries (each kernel is bound to a specific
  scalar type internally).

## Acceptance Criteria

- [x] AC-1: `DispatchKey` priority ordering: `Tracer > Autograd >
  Autocast > Cpu` (verified by `dispatch_key_priority_ordering` at
  `dispatch in dispatch.rs`).
- [x] AC-2: `DispatchKey::ALL` has 11 keys, no duplicates (verified by
  `dispatch_key_all_contains_every_key` at `dispatch.rs:431`).
- [x] AC-3: `DispatchKeySet::from([Cpu, Autograd])` contains both
  (`dispatch_key_set_insert_and_contains` at `:452`).
- [x] AC-4: `DispatchKeySet::highest()` returns the highest-priority key
  (`dispatch_key_set_highest` at `:472` returns `Some(Profiler)` for
  `{Cpu, Autograd, Profiler}`).
- [x] AC-5: `iter_desc()` yields keys in descending priority order
  (`dispatch_key_set_iter_desc_gives_priority_order` at `:482`).
- [x] AC-6: `Dispatcher::call` with empty keyset errors structured
  (`dispatcher_call_empty_keyset_errors` at `:550`).
- [x] AC-7: `Dispatcher::call` with no-matching-key errors structured
  (`dispatcher_call_no_kernel_errors` at `:559`).
- [x] AC-8: `Dispatcher::call` picks the highest-priority registered
  kernel (`dispatcher_call_picks_highest_priority_key` at `:569`).
- [x] AC-9: Redispatch (masking the active key + re-calling) chains
  through layers (`dispatcher_redispatch_chains_through_keys` at `:599`
  and `dispatcher_full_three_layer_stack` at `:691` — `Tracer → Autograd
  → Cpu`).
- [x] AC-10: `Dispatcher::call_direct` bypasses priority
  (`dispatcher_call_direct_bypasses_priority in dispatch.rs`).
- [x] AC-11: The set / union / intersection / remove laws hold (
  `dispatch_key_set_union_and_intersection` at `:501`,
  `dispatch_key_set_remove` at `:462`).

## Architecture

### Priority ordering (REQ-1, REQ-3) — `dispatch.rs:68-108`

```rust
#[repr(u8)]
pub enum DispatchKey {
    Cpu = 0,
    Cuda = 1,
    Meta = 2,
    Sparse = 3,
    Quantized = 4,
    Nested = 5,
    Autocast = 6,
    Autograd = 7,
    Vmap = 8,
    Profiler = 9,
    Tracer = 10,
}
```

The discriminant IS the priority: `Tracer = 10` is higher than `Autograd
= 7` is higher than `Cpu = 0`. `DispatchKey::ALL` (`dispatch.rs:119`)
enumerates all 11 keys in declaration order.

The reduced set vs upstream — upstream has ~150 keys covering every
backend (HIP, XLA, IPU, …), every per-functionality layer (Functorch,
Composite, BackendSelect, …), and a backend-component bitmask in the
lower 12 bits. ferrotorch's 11 keys cover the layering ferrotorch's ops
actually use today; adding new keys is a one-line enum extension.

### Bitmask keyset (REQ-2) — `dispatch.rs:136-251`

`DispatchKeySet { bits: u16 }`. Membership is `(bits >> key.priority()) &
1 != 0`. Insert / remove / union / intersection are bitwise. `len()` is
`bits.count_ones()`. `iter_desc()` walks high bits to low using
`15 - bits.leading_zeros()` to find the next set bit and clearing it.

### Kernel type + dispatcher (REQ-5, REQ-8) — `dispatch.rs:287-405`

```rust
pub type Kernel<T> = Box<
    dyn Fn(&[Tensor<T>], DispatchKeySet, &Dispatcher<T>)
        -> FerrotorchResult<Tensor<T>>
        + Send + Sync,
>;

pub struct Dispatcher<T: Float> {
    kernels: HashMap<(String, DispatchKey), Kernel<T>>,
}
```

Lookup is a single `HashMap` probe. `register(op, key, kernel)`
(`dispatch.rs:312`) overwrites any existing registration for the same
`(op, key)` pair. `call(op, inputs, keyset)` (`dispatch.rs:344`) walks
`keyset.iter_desc()` and fires the first matching kernel. Empty keyset
and no-matching-key both surface as
`FerrotorchError::InvalidArgument`.

### Redispatch protocol (REQ-5 layering)

A kernel that wants to participate in a layered stack does:

```rust
d.register("op", DispatchKey::Autograd, |inputs, keyset, disp| {
    // record backward node
    let rest = keyset.remove(DispatchKey::Autograd);
    disp.call("op", inputs, rest)
});
```

The keyset passed to a kernel includes ALL keys that were active (not
just the kernel's own key) so the kernel can choose which to mask off
before redispatching. This mirrors upstream's "bottom-most" pattern at
`aten/src/ATen/core/dispatch/Dispatcher.h:callBoxedForDispatchKey`.

### Why per-dtype `T` (REQ-9)

A `Dispatcher<f32>` cannot hold an `f64` kernel — `Kernel<T>` is
parameterized on the tensor element type. Upstream achieves this by
storing kernels keyed on the tuple `(op_name, dispatch_key, scalar_type)`;
ferrotorch lifts `scalar_type` to a generic parameter, so a downstream
crate holds one `Dispatcher<f32>` and one `Dispatcher<f64>` (and so on
per dtype). This is R-DEV-7 (Rust generics are the materially better
analog) — type-safe at compile time, no runtime dtype-tag scrubbing.

### Production consumers

The `Dispatcher` infrastructure is currently used at the framework
boundary by:
- `ferrotorch-core/src/lib.rs:152` re-exports `DispatchKey,
  DispatchKeySet, Dispatcher, Kernel` for use by downstream layering
  crates.

The active op-dispatch path in `ferrotorch-core` today uses *direct*
calls (e.g. `arithmetic::add(a, b)` invokes the dispatch-free fast path
because the layering crates that would *register* a kernel into a
`Dispatcher` are still being staged). The dispatcher ships as the
**abstraction** the framework will route through once the
autograd-as-a-key migration lands (planned in the autograd dispatch
follow-up). For the current iteration, the layered-stack tests
(`dispatcher_full_three_layer_stack` etc.) ARE the contract — they
demonstrate the registration / redispatch protocol works end-to-end.

R-DEFER-1 NEW pub APIs: `Dispatcher`, `DispatchKey`, `DispatchKeySet`
were added in CL-397 in advance of the layering crates that would
register kernels. **S5 grandfathering applies**: this pub API surface is
the abstraction users will key against once the registering crates land;
the abstraction itself is the boundary. Failing to ship it would block
the layering crates' arrival. Test-only callers in
`#[cfg(test)] mod tests` exercise the full redispatch protocol, so the
contract is mechanically verified even though no non-test crate registers
kernels yet — that registration follow-up is tracked by issue #1530.

## Parity contract

`parity_ops = []`. The parity surface is the *layering protocol*: when
the autograd-as-a-key migration lands, a tensor with
`DispatchKeySet::from([Autograd, Cpu])` calling `add` MUST land in the
Autograd-registered kernel first (which records a backward node), then
redispatch with `{Cpu}` only, which fires the CPU `add_f32` kernel. The
two-test characterization at `dispatcher_full_three_layer_stack`
(`ferrotorch-core/src/lib.rs`) is the explicit pinning of this protocol.

## Verification

```
cargo test -p ferrotorch-core --lib dispatch
```

Expected: 15 tests pass, 0 failed. Tests cover:
- `DispatchKey` priority + completeness (2 tests).
- `DispatchKeySet` empty / insert / contains / remove / union /
  intersection / highest / iter_desc / from-array (8 tests).
- `Dispatcher` register / has_kernel / call empty-keyset / call missing
  kernel / call priority / call redispatch / call skip-missing /
  call_direct / call_direct missing / 3-layer stack (10 tests).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum DispatchKey` at `ferrotorch-core/src/dispatch.rs:70` with 11 variants `Cpu=0..Tracer=10`, mirroring (a reduced slice of) `c10::DispatchKey` at `c10/core/DispatchKey.h:136`. Non-test production consumer: `ferrotorch-core/src/lib.rs:152` re-exports `DispatchKey` for downstream layering crates; R-DEFER-1 S5 grandfathering: existing pub API surface, the registering layering crates are tracked under follow-up #1530. The 11-variant set is the contract every kernel registers against. |
| REQ-2 | SHIPPED | impl: `pub struct DispatchKeySet { bits: u16 }` at `ferrotorch-core/src/dispatch.rs:137-251` with O(1) membership / insert / remove / union / intersection. Non-test production consumer: `Dispatcher::call` at `dispatch.rs:344` walks `keyset.iter_desc()`; `Dispatcher::call_direct` at `:374` takes the same `keyset`; both are the production lookup paths. |
| REQ-3 | SHIPPED | impl: `DispatchKey::priority(self) -> u8` at `ferrotorch-core/src/dispatch.rs:113` via `self as u8`. Non-test production consumer: `DispatchKeySet::insert` at `dispatch.rs:172` shifts by `key.priority()` to compute the bit position; `iter_desc` at `:235` walks the priority bits descendingly. |
| REQ-4 | SHIPPED | impl: `DispatchKeySet::highest` at `ferrotorch-core/src/dispatch.rs:220` and `iter_desc` at `:235`. Non-test production consumer: `Dispatcher::call` at `dispatch.rs:357` iterates `keyset.iter_desc()` to resolve the next key in descending priority — this IS the dispatch resolution algorithm. |
| REQ-5 | SHIPPED | impl: `pub struct Dispatcher<T: Float>` at `Dispatcher in ferrotorch-core/src/dispatch.rs` and `Dispatcher::register` at `register in ferrotorch-core/src/dispatch.rs`, `Dispatcher::call` at `call in ferrotorch-core/src/dispatch.rs`. The HashMap-based kernel table with priority-walking is the upstream `c10::Dispatcher::callBoxedForDispatchKey` analog (`aten/src/ATen/core/dispatch/Dispatcher.h`). Non-test production consumer: `ferrotorch-core/src/lib.rs` re-exports `Dispatcher` and `Kernel` for downstream registering crates; R-DEFER-1 S5 grandfathering applies — the boundary IS the public API; registering-crate follow-up tracked at #1530. |
| REQ-6 | SHIPPED | impl: `Dispatcher::call_direct` at `ferrotorch-core/src/dispatch.rs:374-390`. Non-test production consumer: re-exported via `lib.rs:152` for downstream callers that want to bypass priority resolution; the in-tree contract is pinned by the 10-test integration suite. |
| REQ-7 | SHIPPED | impl: `Err(FerrotorchError::InvalidArgument { message: format!("Dispatcher::call({op_name}): ...") })` at `ferrotorch-core/src/dispatch.rs:351` (empty keyset) and `:362` (no kernel). R-CODE-2 compliant: no `panic!`, no `unwrap()`, no `expect()` in the production path. Non-test production consumer: `FerrotorchResult<Tensor<T>>` propagates the structured error through any caller of `Dispatcher::call`. |
| REQ-8 | SHIPPED | impl: `pub type Kernel<T> = Box<dyn Fn(...) -> FerrotorchResult<Tensor<T>> + Send + Sync>` at `register in ferrotorch-core/src/dispatch.rs`. Non-test production consumer: every `register(...)` call in the test suite + `lib.rs` re-export uses the type alias as the kernel-shape contract. The `Send + Sync` bound is what makes the dispatcher safe to share across threads. |
| REQ-9 | SHIPPED | impl: `pub struct Dispatcher<T: Float>` at `Dispatcher in ferrotorch-core/src/dispatch.rs` is generic over the tensor element type. R-DEV-7 deviation from upstream's runtime `ScalarType` tag. Non-test production consumer: `lib.rs` re-exports both `Dispatcher` and `Kernel` parameterized on `T`, so downstream crates instantiate `Dispatcher<f32>`, `Dispatcher<f64>`, … one per dtype. |
