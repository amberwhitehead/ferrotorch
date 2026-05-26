# CudaBuffer — pool-aware GPU memory buffer

<!--
tier: 3-component
status: draft
baseline-pytorch: 2fa9c68b1 (working tree at /home/doll/pytorch)
upstream-paths:
  - aten/src/ATen/cuda/
  - aten/src/ATen/native/cuda/
  - c10/cuda/
  - torch/cuda/
-->

## Summary

`ferrotorch-gpu/src/buffer.rs` defines `CudaBuffer<T>`, the owned GPU
memory handle every kernel and bridge in the crate passes around. A
`CudaBuffer<T>` wraps a `cudarc::driver::CudaSlice<T>`, the logical
element count, the rounded pool-key allocation size, the device
ordinal, and an optional pool-return function pointer that the `Drop`
impl uses to return the slice to the global pool (`crate::pool`)
instead of freeing it. This is the ferrotorch analog of PyTorch's
`at::DataPtr` (`c10/core/Allocator.h`) plus the
`CUDACachingAllocator::recordStream`-aware lifecycle PyTorch wires into
`at::Tensor::storage_`.

## Requirements

- REQ-1: Generic `CudaBuffer<T>` parameterised by element type, holding
  `Option<CudaSlice<T>>` (option so `Drop` can `take()` without
  double-free), the logical element count `len`, the rounded `alloc_len`
  used as the pool key, the device ordinal, and an optional
  `PoolReturnFn<T>` function pointer.

- REQ-2: Type-erased pool-return mechanism. `PoolReturnFn<T> =
  Option<fn(usize, usize, CudaSlice<T>)>`. `None` means "do not pool —
  drop via `CudaSlice::Drop` (`cuMemFreeAsync`)". `Some(fn)` means call
  the fn in `Drop` to return the slice to the pool. The pool-return fns
  for `f32` (`return_f32`) and `f64` (`return_f64`) are private module
  members so the pool key (`device, alloc_len, elem_size_bytes`)
  matches the lookup in `crate::pool::pool_take`.

- REQ-3: Pooled-construction helpers for `f32` and `f64`:
  `CudaBuffer<f32>::new_pooled(slice, len, alloc_len, device)` and
  `CudaBuffer<f64>::new_pooled(slice, len, alloc_len, device)`. These
  are `pub(crate)` — they are crate-internal because pool keys must
  match the rounded length convention; downstream callers obtain pooled
  buffers via `crate::transfer::alloc_zeros` and similar high-level
  fns, not by directly calling the constructor.

- REQ-4: Drop impl returning to pool on `Some(pool_fn)`, else freeing
  naturally. `impl<T> Drop for CudaBuffer<T>` calls `self.data.take()`
  to extract the slice, then either invokes `pool_fn(device, alloc_len,
  slice)` for pooled buffers or lets `slice` drop normally (which fires
  cudarc's `CudaSlice::Drop` → `cuMemFreeAsync`). The `Option` extraction
  is the standard Rust idiom for moving owned data out of a `&mut self`
  Drop signature.

- REQ-5: Read-side accessors: `len()`, `alloc_len()`, `is_empty()`,
  `device_ordinal()`, `inner() -> &CudaSlice<T>`, `inner_mut() -> &mut
  CudaSlice<T>`. The `inner` / `inner_mut` accessors `expect` that the
  internal `Option` is `Some` — the only way it can be `None` is after
  the `Drop` impl has `take()`d it, which is unreachable from the live
  `&self` borrow. Per R-CODE-2 the `expect` documents the borrow-checker
  invariant.

- REQ-6: Manual `Debug` impl. cudarc's `CudaSlice<T>` doesn't impl
  `Debug` for all `T` (the slice contents are device memory and can't be
  formatted), so we derive `Debug` manually showing `len`,
  `device_ordinal`, and `pooled` (whether `pool_fn.is_some()`). The
  device pointer / inner slice is omitted intentionally.

- REQ-7: Host-only stub when `cuda` feature is disabled. The stub
  `CudaBuffer<T>` has `_phantom: PhantomData<T>`, `len: usize`,
  `device_ordinal: usize` and provides `len()` / `is_empty()` /
  `device_ordinal()`. Keeps the type resolvable in host-only builds so
  downstream `use ferrotorch_gpu::CudaBuffer` compiles.

## Acceptance Criteria

- [x] AC-1: `CudaBuffer<f32>::new_pooled(slice, 1024, 1024, 0)` constructs
  a pooled buffer that returns to the global pool on drop — verified
  by `pool.rs::tests::pool_return_then_take`.
- [x] AC-2: Dropping a non-pooled `CudaBuffer<T>` (constructed by
  `alloc_zeros` with `pool_fn: None`) frees the underlying GPU memory
  via cudarc — verified by every `CudaAllocator::free` call in
  `allocator.rs::tests::free_decreases_allocated_bytes`.
- [x] AC-3: `buf.len()` returns the logical element count;
  `buf.alloc_len()` returns the rounded pool-key size.
- [x] AC-4: `buf.inner()` returns the underlying `CudaSlice<T>` for use
  with cudarc launch APIs — verified by every kernel module that calls
  `buf.inner()` before launching.
- [x] AC-5: Host-only `cargo build -p ferrotorch-gpu --no-default-features`
  succeeds against `CudaBuffer<T>`'s stub.

## Architecture

### Struct + pool-fn pointer (REQ-1, REQ-2)

`pub struct CudaBuffer<T> in buffer.rs` at `buffer.rs` holds:
- `data: Option<CudaSlice<T>>` — wrapped in `Option` so the `Drop` impl
  can `.take()` ownership.
- `len: usize` — logical element count visible to callers.
- `alloc_len: usize` — rounded element count used as the pool key.
- `device_ordinal: usize` — owning device.
- `pool_fn: PoolReturnFn<T>` — `None` for direct-free, `Some(fn)` for
  pooled.

The `PoolReturnFn<T>` typedef at `buffer.rs` is the function-pointer
shape. `return_f32` (`buffer.rs`) and `return_f64` (`buffer.rs`)
are the concrete instances; each calls
`crate::pool::pool_return::<CudaSlice<T>>(device, len, elem_size, slice)`.
The element size (4 / 8 bytes) is baked into the fn so the pool's
byte-level accounting stays correct.

Why function pointers rather than a generic dispatch? Because
`CudaBuffer<T>` is generic but the pool-return path is monomorphic per
`T`; storing a `fn(usize, usize, CudaSlice<T>)` keeps the buffer struct
size at 1 word per pool-fn rather than dragging a closure or vtable.

### Pooled constructors (REQ-3)

`impl CudaBuffer<f32> in buffer.rs::new_pooled` at `buffer.rs`
and the `f64` sibling at `buffer.rs`. Both are `pub(crate)` and
return `Self` with `pool_fn: Some(return_f32 / return_f64)`. Callers
use them via `crate::transfer::alloc_zeros_f32` etc.

Non-test production consumer: `ferrotorch-gpu/src/transfer.rs` (the
high-level allocation entry point) calls `CudaBuffer::<f32>::new_pooled`
when the pool path is taken; `CudaAllocator` calls it indirectly via
the pool. Outside the crate, callers see a finished `CudaBuffer<f32>`.

### Drop returning to pool (REQ-4)

`impl<T> Drop for CudaBuffer<T> in buffer.rs` at `buffer.rs`.
Sequence:
1. `if let Some(slice) = self.data.take()` — extract the slice from the
   `Option`.
2. If `pool_fn.is_some()`, call it with `(device, alloc_len, slice)`.
   Note `alloc_len` not `len` — the pool key is the rounded length, so
   subsequent `pool_take` with the same rounded length finds the buffer.
3. Else: `slice` drops out of scope, firing cudarc's `CudaSlice::Drop`
   which calls `cuMemFreeAsync` on the device pointer.

Non-test production consumer: every kernel-output `CudaBuffer<f32>` in
`ferrotorch-jit/src/fusion_gpu.rs` (`apply_fused: CUDA tensor's
GPU handle is not a CudaBuffer<f32>` is the consumer downcast site)
goes through this Drop path when the apply finishes.

### Accessors (REQ-5)

`impl<T> CudaBuffer<T> in buffer.rs` at `buffer.rs` provides
the read-side accessors. `inner()` / `inner_mut()` use
`.expect("CudaBuffer: inner slice already taken")` — the only way to
hit this panic is to call `inner` AFTER `drop` has run, which the
borrow checker prevents on live `&self` / `&mut self`. Per R-CODE-2,
the `expect` documents the borrow-checker invariant rather than
silencing a real failure case.

Non-test production consumer: `ferrotorch-diffusion/src/gpu/vae_encoder.rs`
takes `x: &CudaBuffer<f32>` and forwards to kernels that call
`x.inner()` to get the `CudaSlice<f32>` for launches.

### Manual Debug (REQ-6)

`impl<T> std::fmt::Debug for CudaBuffer<T> in buffer.rs` at
`buffer.rs`. Shows `len`, `device_ordinal`, `pooled`. The
inner slice and `alloc_len` are omitted — pool key details are
implementation detail; the device pointer can't be displayed safely.

Non-test production consumer: any log line that includes
`format!("{:?}", buf)` — typical in `ferrotorch-llama` debug
diagnostics.

### Host-only stub (REQ-7)

`#[cfg(not(feature = "cuda"))] pub struct CudaBuffer<T> in buffer.rs`
at `buffer.rs` is the host-only stub. `_phantom: PhantomData<T>`
preserves the generic parameter; the stub has no constructor (you
can't construct a stub buffer because allocation requires a real
device), only `len()`, `is_empty()`, `device_ordinal()` accessors that
return whatever the stub was initialised to.

Non-test production consumer: downstream `use ferrotorch_gpu::CudaBuffer`
in host-only builds (e.g. `ferrotorch-distributions/src/fallback.rs`
under `--no-default-features`) still resolves; the stub keeps the type
name alive.

## Parity contract

`parity_ops = []`. `CudaBuffer<T>` is INFRASTRUCTURE — a smart pointer
plus a `Drop` hook. Parity-sweep correctness depends on it (every
parity op materialises results into `CudaBuffer`s before host
comparison), so a regression in Drop would surface as the entire sweep
leaking GPU memory or freeing twice — caught by CUDA's internal
double-free protection.

Edge cases handled:
- Zero-length buffer: `alloc_zeros::<f32>(0)` returns a `CudaBuffer`
  with `len = 0`, `is_empty() == true`. Drop is a no-op
  (cudarc's `CudaSlice::Drop` no-ops on empty slices). Pinned by
  `allocator.rs::tests::zero_element_alloc`.
- `pool_fn: None` with non-empty buffer: drops via cudarc's
  `cuMemFreeAsync`; the pool sees nothing. The
  `CudaAllocator::alloc_zeros` (`allocator.rs`) constructs with
  `pool_fn: None` so its `free` accounting is exact.
- `inner` after `take`: documented `expect` panic; the borrow checker
  prevents the live access pattern.

## Verification

Tests in consuming modules:
- `pool.rs::tests::pool_return_then_take` exercises the
  `new_pooled` → drop → pool_take roundtrip via the underlying
  `pool_return` / `pool_take` primitives.
- `allocator.rs::tests::cuda_tests::alloc_increases_allocated_bytes`,
  `free_decreases_allocated_bytes`, `zero_element_alloc` exercise the
  non-pooled Drop path through `CudaAllocator::alloc_zeros` /
  `CudaAllocator::free`.

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-gpu --features cuda buffer:: 2>&1 | tail -3
cargo test -p ferrotorch-gpu --features cuda --lib pool::tests 2>&1 | tail -3
```

Expected: `0 failed`.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct CudaBuffer<T> in buffer.rs` at `buffer.rs` with `Option<CudaSlice<T>> + len + alloc_len + device_ordinal + pool_fn`. Non-test production consumer: `ferrotorch-jit/src/fusion_gpu.rs` downcasts `GpuBufferHandle` to `&CudaBuffer<f32>`. |
| REQ-2 | SHIPPED | impl: `type PoolReturnFn<T> = Option<fn(usize, usize, CudaSlice<T>)>` at `buffer.rs`; `fn return_f32 / return_f64` at `buffer.rs`. Non-test production consumer: `crate::pool::pool_return::<CudaSlice<f32>>` invocation in `buffer.rs` is the wiring; the pool sees the call at `pool.rs`. |
| REQ-3 | SHIPPED | impl: `impl CudaBuffer<f32>::new_pooled in buffer.rs` at `buffer.rs`, `impl CudaBuffer<f64>::new_pooled in buffer.rs` at `buffer.rs`. Non-test production consumer: `crate::transfer::alloc_zeros_f32` (referenced from `ferrotorch-jit/src/fusion_gpu.rs`) builds pooled buffers via this path. |
| REQ-4 | SHIPPED | impl: `impl<T> Drop for CudaBuffer<T> in buffer.rs` at `buffer.rs`. Non-test production consumer: every dropped buffer in `ferrotorch-llama/src/gpu.rs` flows through this Drop — KV cache evictions, attention output buffers, etc. |
| REQ-5 | SHIPPED | impl: accessors `len() / alloc_len() / is_empty() / device_ordinal() / inner() / inner_mut()` at `buffer.rs`. Non-test production consumer: `ferrotorch-diffusion/src/gpu/vae_encoder.rs` consumes `&CudaBuffer<f32>` via `x.inner()` for kernel launches. |
| REQ-6 | SHIPPED | impl: `impl<T> std::fmt::Debug for CudaBuffer<T> in buffer.rs` at `buffer.rs`. Non-test production consumer: structured logging in the broader workspace; `format!("{buf:?}")` formatted output is the API contract. |
| REQ-7 | SHIPPED | impl: `#[cfg(not(feature = "cuda"))] pub struct CudaBuffer<T> in buffer.rs` at `buffer.rs`. Non-test production consumer: host-only build path (`cargo build -p ferrotorch-gpu --no-default-features`) compiles; the stub keeps the `use ferrotorch_gpu::CudaBuffer` line at `ferrotorch-jit/src/fusion_gpu.rs` valid in host builds. |
