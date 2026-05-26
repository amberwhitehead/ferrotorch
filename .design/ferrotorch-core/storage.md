# Tensor storage

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - c10/core/Storage.h
  - c10/core/StorageImpl.h
  - c10/core/Allocator.h
  - aten/src/ATen/core/Tensor.h
  - torch/_torch_docs.py
  - torch/overrides.py
-->

## Summary

`ferrotorch-core/src/storage.rs` defines `TensorStorage<T>` and the
underlying `StorageBuffer<T>` enum that tags the actual data with a
device. Mirrors `c10::Storage` / `c10::StorageImpl` (`c10/core/Storage.h`)
plus the `at::Tensor` storage-ownership pattern in
`aten/src/ATen/core/Tensor.h`. Storage variants: CPU `Vec<T>`, CUDA
`GpuBufferHandle`, CubeCL `Box<dyn CubeStorageHandle>` (XPU), and Meta
(no backing memory, only `numel` + optional fill value).

## Requirements

- REQ-1: `StorageBuffer<T>` 4-variant enum — `Cpu(Vec<T>)`,
  `Gpu(GpuBufferHandle)`, `Cubecl(Box<dyn CubeStorageHandle>)`,
  `Meta { numel, fill_value: Option<T> }`. Mirrors PyTorch's
  multi-device storage discriminator behind
  `c10::TensorImpl::storage()`.
- REQ-2: `CubeStorageHandle` trait — trait-erased CubeCL device handle
  with `as_any`, `len`, `is_empty`, `ordinal`, `read_to_host`,
  `clone_handle`. The concrete implementation lives in
  `ferrotorch-cubecl` to avoid a circular dependency. Mirrors the
  same pattern used by `GpuBackend` for CUDA.
- REQ-3: Constructors — `TensorStorage::cpu(data)`,
  `TensorStorage::gpu(handle)`, `TensorStorage::xpu_from_handle(handle,
  ordinal)`, `TensorStorage::meta(numel)`,
  `TensorStorage::meta_filled(numel, value)`. The `on_device` and
  `on_device_pinned` constructors take a CPU Vec + target device and
  either wrap directly (CPU) or upload (CUDA via the registered
  `GpuBackend`).
- REQ-4: Fallible accessors — `try_as_slice()`,
  `try_as_mut_slice()` return `Err(GpuTensorNotAccessible)` for
  non-CPU storage; the deprecated `as_slice`/`as_mut_slice` panic
  instead.
- REQ-5: Variant predicates — `is_cpu`, `is_gpu`, `is_cubecl`,
  `is_meta` and the handle accessors `gpu_handle`,
  `gpu_handle_mut`, `cubecl_handle`.
- REQ-6: `try_clone` — fallible deep-clone. CPU clones the Vec; GPU
  routes through `backend.clone_buffer`; Cubecl uses
  `handle.clone_handle`; Meta clones the descriptor. `Clone::clone`
  delegates to `try_clone` and panics on backend failure (documented).
- REQ-7: `try_clone_subregion(offset, numel)` — clone a contiguous
  sub-slice for narrow / select views. CPU slices the Vec
  zero-copy-ish; GPU goes through D2H + H2D under the source handle's
  dtype tag; Cubecl currently errors (sub-region upload not
  implemented).
- REQ-8: `Drop` impl returns CPU `Vec` to `cpu_pool::pool_return_cpu`
  (`storage.rs:517-528`); GPU / Cubecl handles cleanup via their own
  Drop chain.
- REQ-9: Meta fill value — `meta_fill_value()` returns
  `Option<&T>` so `full_meta(shape, value)` round-trips the requested
  fill without allocating data.

## Acceptance Criteria

- [x] AC-1: `TensorStorage::cpu(vec![1.0, 2.0]).device() == Device::Cpu`.
- [x] AC-2: `TensorStorage::meta(N)` carries `numel` only.
- [x] AC-3: `try_as_slice` on a GPU storage returns
  `GpuTensorNotAccessible`.
- [x] AC-4: `Drop` returns CPU buffers to the pool (verified via
  `cpu_pool` hit rates in the `cpu_pool::test_pool_miss_then_hit`
  test at `cpu_pool.rs:248-285`, which exercises the storage drop
  path transitively when tensors go out of scope).
- [x] AC-5: `try_clone` on GPU storage uses `backend.clone_buffer`
  rather than D2H+H2D.
- [x] AC-6: `meta_fill_value` returns `Some(&v)` after
  `meta_filled(N, v)` and `None` after `meta(N)`.
- [x] AC-7: `cargo test -p ferrotorch-core --lib storage` passes.

## Architecture

The file is ~540 LOC, mostly variant dispatch and SAFETY-block
documentation.

- `TensorStorage<T>` is a 2-field struct: `data: StorageBuffer<T>` +
  `device: Device`. The device is redundant when the variant is CPU
  / Gpu / Meta (recoverable from the buffer), but explicit for XPU
  where the handle alone doesn't tell you the ordinal in a way the
  outer machinery wants.
- `on_device(data, target_device)` (`storage.rs:148-180`) is the
  canonical "construct + upload" path. CPU is direct wrap; CUDA uses
  `backend.cpu_to_gpu` with `unsafe` byte-slice reinterpretation
  (16-line SAFETY comment at `:154-163`). XPU and MPS error out —
  XPU must go through `ferrotorch-xpu::XpuDevice::upload` (it owns the
  CubeRuntime), MPS is not yet wired.
- `on_device_pinned` (`storage.rs:186-215`) — same shape but uses
  `backend.cpu_to_gpu_pinned` (pinned host memory for the H2D, ~2x
  faster on large buffers).
- `try_clone` (`storage.rs:397-427`) dispatches per variant; for GPU,
  it requires the global backend to be registered (else
  `DeviceUnavailable`). The blanket `Clone` impl
  (`storage.rs:496-515`) panics with a descriptive message on
  failure; callers should prefer `try_clone` if they want to handle
  the error.
- `try_clone_subregion` (`storage.rs:434-493`) for GPU goes through
  host bytes preserving the source `dtype` tag — see the comment at
  `:457-460`. The Cubecl path is currently a stub returning
  `InvalidArgument` (see issue #673) until a kernel-side `upload_slice`
  lands.

Non-test production consumers of `TensorStorage`:

- Every `Tensor::from_storage` / `Tensor::from_operation` call site
  is a consumer; the type is the foundational data carrier.
- `tensor.rs:1232` (`update_data` GPU branch) constructs a fresh
  `StorageBuffer::Gpu` from a backend handle.
- `tensor.rs:825, 903, 989` (`to()` device transfers) construct
  `TensorStorage::gpu(handle)`, `TensorStorage::cpu(data)`,
  `TensorStorage::meta(numel)`.
- `creation::*` (`creation.rs:10, 17, 24, 29, 34`, ...) all build
  storage via `TensorStorage::cpu(data)`.

## Parity contract

`parity_ops = []`. Storage is plumbing; the parity contract lives at
the op level. The CPU/GPU/XPU/Meta dispatch is observably correct
when every op that depends on storage runs through its parity sweep
without failure.

## Verification

- Indirect: every CPU/GPU op test exercises the storage variant
  paths. There is no dedicated test mod in this file.
- The `Drop` round-trip with the CPU pool is pinned by
  `cpu_pool::test_pool_miss_then_hit` at `cpu_pool.rs:248-285`.
- The `try_clone` GPU path is exercised by every CUDA op that
  internally clones a buffer (e.g. `as_strided_copy`'s
  `materialize_strided_cuda` at `stride_tricks.rs:394-417`).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `enum StorageBuffer<T>` at `ferrotorch-core/src/storage.rs:63-86` with 4 variants; non-test consumer: `Tensor::storage()` at `ferrotorch-core/src/tensor.rs:423` exposes the storage to every op, which then dispatches on the variant (e.g. `gpu_dispatch.rs:7011` reads the backend through the storage). |
| REQ-2 | SHIPPED | impl: `trait CubeStorageHandle` at `ferrotorch-core/src/storage.rs:18-45`; non-test consumer: `ferrotorch-cubecl` implements this trait (downstream crate); `Tensor::cubecl_handle` at `ferrotorch-core/src/storage.rs:388-393` exposes it to XPU-aware ops. |
| REQ-3 | SHIPPED | impl: constructors at `ferrotorch-core/src/storage.rs:90` (`cpu`), `:100` (`meta`), `:116` (`meta_filled`), `:148` (`on_device`), `:186` (`on_device_pinned`), `:225` (`xpu_from_handle`), `:233` (`gpu`); non-test consumers: `creation.rs:10, 17, 24, 29, 34` and many `tensor.rs` sites construct via these. |
| REQ-4 | SHIPPED | impl: `try_as_slice` at `ferrotorch-core/src/storage.rs:317`, `try_as_mut_slice` at `:333`; non-test consumer: `Tensor::data` at `tensor.rs:655` and `Tensor::data_mut` at `tensor.rs:1172` route through these — every CPU element-access call hits this path. |
| REQ-5 | SHIPPED | impl: `is_cpu` at `ferrotorch-core/src/storage.rs:344`, `is_gpu` at `:350`, `is_cubecl` at `:356`, `is_meta` at `:362`, `gpu_handle` at `:367`, `gpu_handle_mut` at `:380`, `cubecl_handle` at `:388`; non-test consumer: every CUDA-dispatched op tests `is_gpu` or calls `gpu_handle` (e.g. `tensor.rs:835` in `to(Device::Cpu)`). |
| REQ-6 | SHIPPED | impl: `try_clone` at `ferrotorch-core/src/storage.rs:397`, `Clone::clone` at `:506`; non-test consumer: `Tensor::clone` and any deep-clone path in autograd accumulation (`tensor.rs:565`) flow through this. |
| REQ-7 | SHIPPED | impl: `try_clone_subregion` at `ferrotorch-core/src/storage.rs:434`; non-test consumer: `Tensor::into_storage_and_shape` at `tensor.rs:763, 768, 776` calls `try_clone_subregion(offset, numel)` to materialise narrow/select views before returning ownership. |
| REQ-8 | SHIPPED | impl: `Drop for TensorStorage<T>` at `ferrotorch-core/src/storage.rs:517-528`; non-test consumer: every temporary CPU tensor that goes out of scope in any op hits this drop. The CPU pool hit-rate test at `cpu_pool.rs:248-285` pins the integration. |
| REQ-9 | SHIPPED | impl: `meta_filled` at `ferrotorch-core/src/storage.rs:116`, `meta_fill_value` at `:129`; non-test consumer: `Tensor::meta_fill_value` at `tensor.rs:1089` exposes the value to `creation::full_meta` callers — production user surface for meta-tensor inspection. |
