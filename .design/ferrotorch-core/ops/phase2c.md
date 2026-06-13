# Phase 2c Cross-World Integer Ops

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

`ferrotorch-core/src/ops/phase2c.rs` (crosslink #1185 Phase 2c)
implements the cross-world ops between `Tensor<T: Float>` and
`IntTensor<I: IntElement>`: `argmax`/`argmin` producing
`IntTensor<i64>` indices, `index_select`/`gather` driven by an
`IntTensor` index (GPU-resident on CUDA), and dtype casts
`Tensor::to_int` / `IntTensor::to_float` / `IntTensor::cast_gpu`.
These mirror `torch.argmax`, `torch.argmin`, `torch.index_select`,
`torch.gather`, and `tensor.to(dtype)`. Each op runs on CUDA when
the input is GPU-resident — real PTX kernels (`backend.argmax`,
`backend.gather_intidx`, `backend.cast_f_to_i`, etc.) — and on CPU
otherwise via a reference loop. `gather` uses the compact
`backend.gather_intidx` CUDA path when non-gather dimensions match the
input, and the rank-aware `backend.gather_intidx_nd` CUDA path when
PyTorch-legal index shapes are smaller on non-gather axes. In both
cases the value buffer stays GPU-resident; only the integer index
buffer is read back for the existing bounds-validation gate.

## Requirements

- REQ-1: `Tensor::argmax(dim)` — return `IntTensor<i64>` index of
  the maximum along `dim`, or flat 0-d scalar when `dim = None`.
  Ties resolve to the FIRST (lowest) index. Mirrors
  `torch.argmax(input, dim=None, keepdim=False)`.
- REQ-2: `Tensor::argmin(dim)` — symmetric with `argmax`. Mirrors
  `torch.argmin`.
- REQ-3: `Tensor::index_select(dim, indices)` —
  `indices: &IntTensor<I>` (1-D). Output keeps `self`'s dtype;
  shape is `self.shape` with `shape[dim]` replaced by
  `indices.numel()`. GPU-resident on CUDA; same-device requirement
  for `indices`. Mirrors `torch.index_select`.
- REQ-4: `Tensor::gather(dim, index)` — `index: &IntTensor<I>` with
  matching ndim. Output shape = `index.shape`; dtype = `self.dtype`.
  GPU-resident on CUDA. Mirrors `torch.gather`.
- REQ-5: `Tensor::to_int::<I>()` — cast float to int dtype,
  TRUNCATE toward zero (PyTorch `tensor.to(int)` semantics). GPU-
  resident on CUDA via `backend.cast_f_to_i(handle, I::dtype())`.
- REQ-6: `IntTensor::argmax(dim)` / `argmin(dim)` — integer-tensor
  arg-reduction returning `IntTensor<i64>`. Same first-index tie-
  breaking.
- REQ-7: `IntTensor::index_select(dim, indices)` /
  `IntTensor::gather(dim, index)` — same as REQ-3/REQ-4 but on
  integer dtype.
- REQ-8: `IntTensor::to_float::<T>()` — cast int to float
  (round-to-nearest-even). GPU-resident on CUDA via
  `backend.cast_i_to_f`.
- REQ-9: `IntTensor::cast_gpu::<J>()` — i32 ↔ i64 GPU dtype cast.
  Returns `Option` so the caller's CPU path handles non-CUDA tensors;
  `Some(Ok/Err)` on CUDA. `pub(crate)` — accessed via
  `IntTensor::cast`'s GPU branch.

## Acceptance Criteria

- [x] AC-1: `cargo test -p ferrotorch-core --lib ops::phase2c`
  (and the cross-world conformance tests) pass.
- [x] AC-2: `argmax` on CPU `[3.0, 1.0, 3.0, 0.0]` returns `0`
  (ties → first; matches torch).
- [x] AC-3: `argmax` on CUDA stays GPU-resident — result is
  `IntTensor<i64>` on the same device as input.
- [x] AC-4: `index_select` rejects 2-D `indices` with
  `InvalidArgument`.
- [x] AC-5: `gather` rejects `index.ndim() != input.ndim()` with
  `ShapeMismatch`.
- [x] AC-6: `to_int` of `3.7` → `3`, `-3.7` → `-3` (truncate toward
  zero).
- [x] AC-7: `to_int` of out-of-range float (e.g. `f32::INFINITY` for
  `i32`) errors with `InvalidArgument`.

## Architecture

The module's purpose is to break the round-trip pattern Llama
generation hit: argmax for sampling AND embedding gather both
required CPU tensors before this Phase 2c work. Now both are
GPU-resident end-to-end.

Shape helpers at `ops/phase2c.rs:24-36`: `factor(shape, dim)` returns
`(outer, dim_size, inner)`; `shape_without(shape, dim)` removes the
reduced axis.

`arg_reduce_ref<V>` at `:43-67` is the CPU reference for argmax /
argmin generic over the comparable type. Walks `[outer, dim_size,
inner]` with first-index tie-breaking (`if better(candidate,
current)` — strict `>` for max, strict `<` for min — keeps the
earliest index).

`tensor_arg<T: Float>` at `:71-111` is the dispatcher for float
inputs: CUDA branch calls `backend.argmax(h, outer, dim_size, inner)`
/ `argmin`; CPU branch reads `data_vec` and calls `arg_reduce_ref`.
`inttensor_arg<I: IntElement>` at `:114-153` is the integer-tensor
counterpart; same dispatch structure.

`index_select_ref<V>` at `index_select_ref in ops/phase2c.rs` is the
axis-factorized CPU reference. `gather_ref<V>` at `gather_ref in
ops/phase2c.rs` walks the actual `index.shape()` coordinates and
substitutes only the gather-axis coordinate with the selected source
index, so smaller non-axis index shapes mirror `torch.gather` rather
than reading a full-input layout.

On CUDA, `Tensor::gather` / `IntTensor::gather` choose
`backend.gather_intidx` only when the compact `[outer, out_dim,
inner]` layout is valid. If `index.shape()` is smaller than
`input.shape()` on any non-gather axis, they call
`backend.gather_intidx_nd`, which mirrors PyTorch's iterator/restride
contract by making index/output shape authoritative while keeping the
value buffer on device.

The public methods on `Tensor` at `:212-350` and on `IntTensor` at
`gpu_backend in ops/phase2c.rs` dispatch through the helpers + `gpu_dispatch::gpu_backend()`
on the CUDA branch.

`float_to_i64_trunc` at `:354-360` is the helper for `to_int`:
`v.trunc()` then `as i64` (Rust 1.45+ saturating semantics, matching
PyTorch's `.to(int64)` clamp-on-overflow). For non-i64 targets
(`i32`), `I::try_from_i64` reports `None` for out-of-range and the
public path returns `InvalidArgument`.

`check_same_device` at `:504` and `gather_check_shapes` at `:519`
are shared validators — same-device requirement for `index_select`
and `gather` on CUDA; gather requires matching ndim and per-axis
`index.shape[ax] <= input.shape[ax]` (PyTorch allows smaller index
off the gather axis).

**Non-test consumers**:

- `crate::tensor::Tensor::argmax` etc. — re-exported as method on
  the `Tensor` type via the `impl Tensor` block at
  `ops/phase2c.rs:212-350`. Note that this is a separate `impl
  Tensor` block in this file, supplementing the main `impl` in
  `tensor.rs`.
- Llama / token-sampling code in `ferrotorch-llama` calls
  `tensor.argmax(None)` to pick the next-token index; the result
  is then fed straight to `tensor.index_select(0, &indices)` to
  gather the embedding row. Both stay GPU-resident.
- The `IntTensor::cast_gpu` method is `pub(crate)` — its sole
  consumer is `IntTensor::cast<J>` in `int_tensor.rs` which branches
  on `is_cuda()` and routes through `cast_gpu` for the GPU half.

## Parity contract

`parity_ops = []` (the route does not declare any). Coverage
through `argmax` / `argmin` / `index_select` / `gather` parity-sweep
arms (if/when those land); current parity-sweep coverage runs through
`grad_fns::reduction::argmax_dim` (the differentiable wrapper) which
chains to `phase2c::Tensor::argmax`.

## Verification

`cargo test -p ferrotorch-core --lib ops::phase2c` covers the CPU
references for argmax/argmin/index_select/gather. CUDA-side
conformance lives in `ferrotorch-core/tests/conformance_phase2c.rs`
(GPU-gated).

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `Tensor::argmax` at `argmax in ops/phase2c.rs`; non-test consumer: `crate::methods::Tensor::argmax_t` at `argmax_t in methods.rs` (the autograd-wrapper) and `crate::grad_fns::reduction::argmax` at `argmax in grad_fns/reduction.rs` route through `Tensor::argmax` |
| REQ-2 | SHIPPED | impl: `Tensor::argmin` at `argmin in ops/phase2c.rs`; non-test consumer: `Tensor::argmin_t` at `argmin_t in methods.rs` |
| REQ-3 | SHIPPED | impl: `Tensor::index_select` at `index_select in ops/phase2c.rs`; non-test consumer: `crate::grad_fns::indexing::index_select_differentiable` at `index_select in grad_fns/indexing.rs` invokes `Tensor::index_select` for its forward |
| REQ-4 | SHIPPED | impl: `Tensor::gather` at `ops/phase2c.rs:283`; non-test consumer: `crate::grad_fns::indexing::GatherBackward::backward` recurses through `Tensor::gather` for the VJP construction |
| REQ-5 | SHIPPED | impl: `Tensor::to_int` at `ops/phase2c.rs:326`; non-test consumer: `crate::int_tensor::Tensor::to_int` re-export path used by quantization / discretization paths in `ferrotorch-llama` and `ferrotorch-quant` |
| REQ-6 | SHIPPED | impl: `IntTensor::argmax`/`argmin` at `ops/phase2c.rs:369,374`; non-test consumer: every downstream caller that argmax's a logit-index tensor goes through this |
| REQ-7 | SHIPPED | impl: `IntTensor::index_select`/`gather` at `ops/phase2c.rs:380,423`; non-test consumer: re-exported via the `IntTensor` method surface |
| REQ-8 | SHIPPED | impl: `IntTensor::to_float` at `ops/phase2c.rs:458`; non-test consumer: re-exported via the `IntTensor` method surface; embedding-table reverse-lookup paths |
| REQ-9 | SHIPPED | impl: `IntTensor::cast_gpu` at `ops/phase2c.rs:481`; non-test consumer: `IntTensor::cast<J>` in `int_tensor.rs` invokes `cast_gpu` for the CUDA branch |
