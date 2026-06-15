# `Tensor<T>` — the central type

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/core/Tensor.h
  - aten/src/ATen/core/TensorBase.h
  - c10/core/TensorImpl.h
  - c10/core/Storage.h
  - torch/_torch_docs.py
  - torch/overrides.py
-->

## Summary

`ferrotorch-core/src/tensor.rs` defines `Tensor<T>` — the central type
of the whole framework — plus its inner `TensorInner<T>` storage,
`TensorId` identity, `MemoryFormat` enum, and the `GradFn<T>` trait
that every backward node implements. Mirrors `at::Tensor` /
`at::TensorImpl` (`aten/src/ATen/core/Tensor.h`, `c10/core/TensorImpl.h`)
plus the channels-last memory-format machinery in
`c10/core/MemoryFormat.h`.

`Tensor<T>` is an `Arc<TensorInner<T>>`; cloning is cheap and preserves
identity. Storage is shared `Arc<TensorStorage<T>>` so view ops
(`view_reshape`, `stride_view`, `narrow`, `permute`, …) are zero-copy.
Gradient state lives behind a `Mutex` so backward can write into any
clone.

## Requirements

- REQ-1: `Tensor<T: Float>` — `Arc`-shared inner with `id`,
  `storage: Arc<TensorStorage<T>>`, `shape: Vec<usize>`,
  `strides: Vec<isize>`, `offset: usize`, `grad: Mutex<Option<Box<Tensor>>>`,
  `grad_fn: Option<Arc<dyn GradFn<T>>>`, `requires_grad: bool`,
  `is_leaf: bool`, `hooks: Mutex<HookStorage<T>>`. Mirrors the
  `at::TensorImpl` invariants in `c10/core/TensorImpl.h`.
- REQ-2: `from_storage(storage, shape, requires_grad)` — leaf tensor
  constructor with C-contiguous strides; rejects numel > storage.len.
- REQ-3: `from_operation(storage, shape, grad_fn)` — op-result
  constructor with `requires_grad=true`, `is_leaf=false`, the supplied
  `grad_fn`. In inference mode short-circuits to `from_storage`
  without grad.
- REQ-4: Zero-copy views — `view_reshape`, `view_operation`,
  `stride_view`, `stride_view_operation`. Non-contiguous tensors are
  materialised first (CUDA via `strided_copy_*`, CPU via
  `data_vec()`).
- REQ-5: `to(device)` — device transfer between CPU, CUDA, XPU,
  Meta. CUDA→CPU readback materialises non-contiguous views on-device
  via `strided_copy_*` rather than copying the full buffer (#802).
  XPU↔CPU goes through the `CubeStorageHandle::read_to_host`. Meta is
  destination-only (you cannot materialise data out of meta).
- REQ-6: `to_pinned(device)` — variant of `to` that uses pinned
  host memory for CPU→CUDA (~2× faster on large buffers).
- REQ-7: `data()` / `data_ref()` / `data_vec()` — element accessors.
  `data()` returns a zero-copy slice (CPU + contiguous only).
  `data_vec()` always returns an owned `Vec<T>`, transparently
  handling non-contiguous + GPU storage.
- REQ-8: Autograd surface — `grad()`, `set_grad()`, `zero_grad()`,
  `accumulate_grad(incoming)`. Gradient accumulation has a GPU-native
  fast path that dispatches `backend.add_f32` / `f64` when both
  existing and incoming grads are on CUDA (#789/#788/#800).
- REQ-9: `detach()` — return a new tensor sharing storage but with
  no grad_fn / requires_grad. `requires_grad_(b)` — return a new
  tensor with the flag updated.
- REQ-10: `is_contiguous()` — C-order contiguity check; size-1 dims
  are wildcards. `is_contiguous_for(format)` — channels-last
  contiguity check. `to_memory_format(format)` — physical
  rearrangement to the target stride pattern (GPU fast path via
  `strided_copy_*` with a permuted shape/stride feed; CPU fallback
  via multi-index walk).
- REQ-11: `as_strided` family methods — exposed via the inherent
  impl in `stride_tricks.rs`. The `view_reshape`, `stride_view`,
  `stride_view_operation` constructors are the substrate.
- REQ-12: `gpu_handle()` / `gpu_handle_mut` (via storage) — typed
  access to the CUDA buffer for kernel dispatch.
- REQ-13: `update_data` / `update_storage` / `update_storage_and_shape`
  / `with_gpu_handle_mut` — `unsafe` in-place mutation entry points
  for optimizer `step` fast paths. Each carries a SAFETY block
  documenting the exclusive-access contract and the leak-preventing
  `ptr::replace` pattern (the previous `ptr::write` leaked the old
  storage).
- REQ-14: Hooks — `register_hook`, `register_post_accumulate_grad_hook`,
  `remove_hook`. Mirror `torch.Tensor.register_hook`.
- REQ-15: `masked_fill` / `masked_select` — bool-tensor-driven
  fill / gather. GPU-resident (`bool_fill_f32`/`f64` kernels;
  stream-compaction for `masked_select`).
- REQ-16: `item()` — scalar extractor for `numel == 1` tensors.
- REQ-17: `into_storage_and_shape()` — consume the tensor, return
  ownership of the storage (zero-copy when refcount allows).
- REQ-18: `is_same`, `inner_refcount`, `storage_refcount`,
  `inner_storage_arc` — identity / refcount inspection used by the
  backward engine to gate in-place gradient accumulation.
- REQ-19: `trait GradFn<T>` — every backward node implements
  `backward(grad_output) -> Vec<Option<Tensor>>`, `inputs()`,
  `name()`, and optional `scalar_args()` for JIT-trace reconstruction.
- REQ-20: `MemoryFormat` enum — `Contiguous`, `ChannelsLast` (4D
  NHWC), `ChannelsLast3d` (5D NDHWC). Mirrors `c10::MemoryFormat`.

## Acceptance Criteria

- [x] AC-1: `from_storage(cpu(vec![1..6]), [2,3], false)` → contiguous
  CPU tensor with id, strides `[3,1]`, device CPU.
- [x] AC-2: Cloning preserves id (tests at `tensor.rs:1900-1908`).
- [x] AC-3: Detach returns a non-tracking tensor sharing storage.
- [x] AC-4: Gradient accumulation across clones is observed from any
  clone (`tensor.rs:1924-1939`).
- [x] AC-5: `Tensor<f32>` and `Tensor<f64>` are `Send + Sync`.
- [x] AC-6: `view_operation` shares storage with the source
  (`tensor.rs:1911-1922`).
- [x] AC-7: `cargo test -p ferrotorch-core --lib tensor` passes.

## Architecture

The file is ~1.97k LOC. Sections:

- **Lines 1-103**: imports, `MemoryFormat`, `TensorId`,
  `trait GradFn<T>`, `TensorInner<T>`, `Tensor<T>` struct decl.
- **Lines 107-341**: construction — `from_storage`,
  `view_reshape`, `view_operation`, `stride_view`,
  `stride_view_operation`, `from_operation`.
- **Lines 343-372**: `ToDeviceBackward` (cross-device autograd node).
- **Lines 375-540**: accessors — `id`, `shape`, `strides`, `numel`,
  `storage_offset`, `storage_len`, `storage`, `device`,
  `requires_grad`, `is_leaf`, `grad_fn`, hook registration,
  `grad`, `set_grad`, `zero_grad`.
- **Lines 554-625**: `accumulate_grad` with the GPU-native fast path
  (`backend.add_f32`/`add_f64` when both grads are on CUDA).
- **Lines 627-781**: data access — `data`, `data_ref`, `data_vec`
  (incl. non-contiguous gather walk), `into_storage_and_shape`.
- **Lines 783-1053**: device transfer — `to(device)` with the full
  CPU↔CUDA↔XPU↔Meta matrix; `to_pinned`; `cuda`; `cpu`;
  variant predicates `is_cpu`, `is_meta`, `is_cuda`, `is_xpu`.
- **Lines 1093-1180**: `gpu_handle()`, `masked_fill`,
  `masked_select`, `data_mut`.
- **Lines 1182-1413**: `unsafe` in-place mutation —
  `update_data`, `update_storage`, `update_storage_and_shape`,
  `with_gpu_handle_mut`. Every `unsafe` block carries a SAFETY
  comment; the `update_storage` impl explicitly documents the
  `ptr::replace` (drop-the-old) pattern that fixed a multi-step
  optimizer leak.
- **Lines 1415-1755**: `detach`, `requires_grad_`,
  `is_contiguous`, `is_contiguous_for(MemoryFormat)`,
  `to_memory_format`, `contiguous_in(MemoryFormat)`,
  `materialize_format` (GPU fast path + CPU fallback),
  `materialize_format_cpu`, `is_scalar`, `item`, `is_same`,
  refcount helpers.
- **Lines 1757-1808**: `strides_match_with_size1`,
  `format_permutation` (the NHWC / NDHWC permutation used by the GPU
  `strided_copy_*` fast path for memory-format changes).
- **Lines 1810-1834**: `Clone` impl (Arc bump),
  `Debug` impl.
- **Lines 1835-1966**: in-file test mod with ~10 unit tests covering
  construction, identity, detach, autograd accumulation, view
  sharing.

Non-test production consumers are *every* op in `ferrotorch-core`,
`ferrotorch-nn`, `ferrotorch-gpu`, etc. — `Tensor<T>` is the data type
the whole workspace operates on. The accessors are foundational; any
attempt to enumerate consumer sites would just list the entire
workspace.

## Parity contract

`parity_ops = []`. The tensor type is plumbing; parity is enforced
at the op level. The properties this type owns (zero-copy views,
correct stride computation, gradient accumulation, device transfer,
identity preservation across clones) are pinned by the in-file unit
tests plus the indirect coverage of every op test.

## Verification

```bash
cargo test -p ferrotorch-core --lib tensor
```

Expected: 10 tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `Tensor<T>` struct at `TensorInner in ferrotorch-core/src/tensor.rs`, `TensorInner in ferrotorch-core/src/tensor.rs`; non-test consumer: every op in the workspace constructs / consumes `Tensor<T>` — see e.g. `add in grad_fns/arithmetic.rs` (`add`'s forward). |
| REQ-2 | SHIPPED | impl: `from_storage in ferrotorch-core/src/tensor.rs`; non-test consumer: `creation::zeros` / `ones` / `tensor in creation.rs, 17, 24` and ~all op forward paths. |
| REQ-3 | SHIPPED | impl: `from_operation` at `ferrotorch-core/src/tensor.rs:299`; non-test consumer: every grad-attaching forward op (e.g. `grad_fns/arithmetic.rs:404` in `add_inner`). |
| REQ-4 | SHIPPED | impl: `view_reshape in ferrotorch-core/src/tensor.rs`, `view_operation in ferrotorch-core/src/tensor.rs`, `stride_view in ferrotorch-core/src/tensor.rs`, `stride_view_operation in ferrotorch-core/src/tensor.rs`; non-test consumer: `grad_fns/shape.rs` reshape / flatten / squeeze / unsqueeze ops, `methods.rs::view_t`. |
| REQ-5 | SHIPPED | impl: `to(device)` at `cuda in ferrotorch-core/src/tensor.rs` covers CPU/CUDA/XPU/Meta matrix incl. the #802 non-contiguous CUDA→CPU materialise path at `cuda in ferrotorch-core/src/tensor.rs`; non-test consumer: `Tensor::cuda` at `cuda in ferrotorch-core/src/tensor.rs`, `Tensor::cpu` at `cpu in ferrotorch-core/src/tensor.rs`, plus every model state-dict load that targets a device. |
| REQ-6 | SHIPPED | impl: `to_pinned(device)` at `ferrotorch-core/src/tensor.rs:1573`; non-test consumer: `ferrotorch-data::DataLoader` calls this when `pin_memory(true)` is set. |
| REQ-7 | SHIPPED | impl: `data` at `ferrotorch-core/src/tensor.rs:1132`, `data_ref` at `:1182`, `data_vec` at `:1192`; non-test consumer: every CPU op that reads tensor data (e.g. `pruning::magnitude_prune` at `pruning.rs:71`, `signal::windows`'s round-trip helper). |
| REQ-8 | SHIPPED | impl: `grad` at `ferrotorch-core/src/tensor.rs:995`, `set_grad` at `:1007`, `zero_grad` at `:1023`, `accumulate_grad` at `:1033` with GPU-native path at `:1049-1075`; non-test consumer: `autograd::backward` engine writes via `accumulate_grad` for every leaf reachable in the graph. |
| REQ-9 | SHIPPED | impl: `detach` at `ferrotorch-core/src/tensor.rs:2446`, `requires_grad_` at `:2466`; non-test consumer: `autograd::no_grad` blocks call `detach`; downstream model init code calls `requires_grad_(true)` on parameters. |
| REQ-10 | SHIPPED | impl: `is_contiguous in ferrotorch-core/src/tensor.rs`, `is_contiguous_for in ferrotorch-core/src/tensor.rs`, `to_memory_format in ferrotorch-core/src/tensor.rs`, `contiguous_in in ferrotorch-core/src/tensor.rs`, `materialize_format in ferrotorch-core/src/tensor.rs` (GPU fast path at `materialize_format in ferrotorch-core/src/tensor.rs`); non-test consumer: `ferrotorch-nn::Conv2d` calls `to_memory_format(MemoryFormat::ChannelsLast)` before cuDNN dispatch. |
| REQ-11 | SHIPPED | impl: see `.design/ferrotorch-core/stride_tricks.md` — the inherent impl `impl Tensor` block at `stride_tricks.rs:183` defines `as_strided` / `as_strided_copy` / `as_strided_scatter`; non-test consumer: `crate::einsum`. |
| REQ-12 | SHIPPED | impl: `gpu_handle` at `ferrotorch-core/src/tensor.rs:1815`; non-test consumer: every CUDA op that dispatches a kernel (e.g. `grad_fns/arithmetic.rs::add_inner` for the CUDA branch, `stride_tricks.rs:1110-1156`). |
| REQ-13 | SHIPPED | impl: `update_data` at `ferrotorch-core/src/tensor.rs:1949`, `update_storage` at `:2202`, `update_storage_and_shape` at `:2057`, `with_gpu_handle_mut` at `:2423`; non-test consumer: `ferrotorch-nn::optim::adamw` / `sgd` `step()` calls these to write the updated parameter; the `out=`-style add path at `grad_fns/arithmetic.rs::add_scaled_out` calls `update_storage_and_shape`. |
| REQ-14 | SHIPPED | impl: `register_hook` at `ferrotorch-core/src/tensor.rs:933`, `register_post_accumulate_grad_hook` at `:953`, `remove_hook` at `:979`; non-test consumer: `autograd::hooks::HookStorage` integration; user-callable surface for gradient monitoring. |
| REQ-15 | SHIPPED | impl: `masked_fill` at `ferrotorch-core/src/tensor.rs:1839`, `masked_select` at `:1855`; non-test consumer: production ops in `grad_fns/indexing.rs::masked_fill_bt` (called via the inherent method delegate). |
| REQ-16 | SHIPPED | impl: `item` at `ferrotorch-core/src/tensor.rs:2777`; non-test consumer: every scalar-returning op result (loss values, mean, sum_all, etc.) — used pervasively by training-loop diagnostics. |
| REQ-17 | SHIPPED | impl: `into_storage_and_shape` at `ferrotorch-core/src/tensor.rs:1249`; non-test consumer: `accumulate_grad` at `:1044` uses it to take ownership of the incoming grad's storage without copying. |
| REQ-18 | SHIPPED | impl: `pub fn is_same in ferrotorch-core/src/tensor.rs`, `inner_refcount in ferrotorch-core/src/tensor.rs`, `storage_refcount in ferrotorch-core/src/tensor.rs`, `inner_storage_arc in ferrotorch-core/src/tensor.rs`; non-test consumer: the autograd backward engine (`autograd::graph`) uses refcount checks to gate in-place gradient accumulation. |
| REQ-19 | SHIPPED | impl: `trait GradFn<T>` at `ferrotorch-core/src/tensor.rs:46-68`; non-test consumer: every grad-fn struct in `grad_fns/*` implements this — see `grad_fns/arithmetic.rs::AddBackward`, `AddScaledBackward`, etc. |
| REQ-20 | SHIPPED | impl: `enum MemoryFormat` at `ferrotorch-core/src/tensor.rs:17-26`; non-test consumer: `Tensor::to_memory_format` / `contiguous_in` consume it; `ferrotorch-nn::Conv2d` chooses NHWC for cuDNN. |
