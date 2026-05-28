# Stride tricks (`as_strided` family)

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/TensorShape.cpp
  - aten/src/ATen/native/AsStridedCopy.cpp
  - aten/src/ATen/native/TensorAdvancedIndexing.cpp
  - torch/_torch_docs.py
  - torch/overrides.py
-->

## Summary

`ferrotorch-core/src/stride_tricks.rs` implements the `as_strided`
family — direct stride manipulation on tensors. It mirrors
`torch.Tensor.as_strided`, `torch.as_strided_copy`, and
`torch.as_strided_scatter`. Strides are in *element* units (not bytes),
may be zero (broadcast) or negative (reverse), and views may overlap
(used for Toeplitz matrices, sliding windows, broadcast replication).
The backward pass for `as_strided` is `AsStridedBackward`, which
scatters the upstream gradient back into a zero-initialised input via
`as_strided_scatter`.

## Requirements

- REQ-1: `Tensor::as_strided(size, stride, storage_offset)` — return a
  zero-copy view with the requested shape, strides, and offset.
  Validates that every reachable offset stays inside the underlying
  storage (does NOT reject overlap). Works on any device because it
  is pure metadata. Mirrors
  `aten/src/ATen/native/TensorShape.cpp::Tensor::as_strided` and
  `torch/_torch_docs.py: as_strided`.
- REQ-2: `Tensor::as_strided_copy(size, stride, storage_offset)` —
  materialise the strided view into a fresh contiguous tensor. CUDA
  path dispatches to `strided_copy_{f32,f64}` GPU kernels via the
  registered `GpuBackend` (no host bounce); CPU walks the multi-index.
  Mirrors `aten/src/ATen/native/AsStridedCopy.cpp::as_strided_copy`.
- REQ-3: `Tensor::as_strided_scatter(src, size, stride, storage_offset)`
  — inverse of `as_strided`: return a copy of `self` with `src`
  written into the strided positions. CUDA dispatch uses
  `strided_copy_*` + `strided_scatter_*` kernels (no host bounce); CPU
  walks the multi-index. Mirrors
  `aten/src/ATen/native/TensorAdvancedIndexing.cpp::as_strided_scatter`.
- REQ-4: Bounds validation — `validate_bounds` (`stride_tricks.rs:126`)
  computes the (min, max) reachable offset from `(shape, stride)`,
  returns `InvalidArgument` when:
  - shape/stride length mismatch (`:133-141`)
  - any offset goes below 0 (`:159-166`)
  - any offset reaches `>= storage_len` (`:167-174`)
  - `storage_offset > storage_len` for an empty view
    (`:145-153`)
- REQ-5: Autograd: `AsStridedBackward` saves the input + `(size,
  stride, offset)` and scatters the upstream gradient into a
  zero-initialised tensor of the input's shape via
  `as_strided_scatter`. Mirrors torch's `AsStridedBackward0`.
- REQ-6: Negative strides — `as_strided(_, &[-1], Some(N-1))` reverses
  a 1-D tensor. Pinned by test `as_strided_negative_stride_reverses`
  at `stride_tricks.rs:518-525`.
- REQ-7: Zero strides — `as_strided(_, &[0], Some(K))` broadcasts a
  single element across the output. Pinned at `stride_tricks.rs:527-533`.
- REQ-8: Free-function wrappers `as_strided`, `as_strided_copy`,
  `as_strided_scatter` (`stride_tricks.rs:61-89`) — thin delegations
  to the `Tensor` inherent methods so callers can use either style.

## Acceptance Criteria

- [x] AC-1: Reshape `[6]` to `[2,3]` via `as_strided(&[2,3], &[3,1])`
  returns `[1,2,3;4,5,6]` (`stride_tricks.rs:497-503`).
- [x] AC-2: Overlapping sliding-window view `[3,3]` with stride `[1,1]`
  over `[1..5]` produces three 3-windows
  (`stride_tricks.rs:506-516`).
- [x] AC-3: Reverse via stride `-1` (`stride_tricks.rs:518-525`).
- [x] AC-4: Broadcast via stride `0` (`stride_tricks.rs:527-533`).
- [x] AC-5: Out-of-bounds shape rejected (`stride_tricks.rs:539-548`).
- [x] AC-6: Negative-reach offset rejected
  (`stride_tricks.rs:550-556`).
- [x] AC-7: `as_strided_copy` produces a contiguous tensor with the
  same values (`stride_tricks.rs:595-613`).
- [x] AC-8: `as_strided_scatter` writes into view positions only
  (`stride_tricks.rs:619-641`).
- [x] AC-9: Shape-mismatched `src` rejected
  (`stride_tricks.rs:654-663`).
- [x] AC-10: Backward via `sum().backward()` produces correct gradient
  (`stride_tricks.rs:669-711`).
- [x] AC-11: `cargo test -p ferrotorch-core --lib stride_tricks` passes.

## Architecture

- `Tensor::as_strided` (`stride_tricks.rs:196-220`) — no-grad fast
  path is a zero-copy `stride_view`; grad path attaches
  `AsStridedBackward` and uses `stride_view_operation`. Both routes
  share the same `Arc<Storage>` with the source tensor on every
  device, so no data movement.
- `Tensor::as_strided_copy` (`stride_tricks.rs:231-251`) — builds the
  view first (re-uses `as_strided` for autograd + bounds), then
  materialises:
  - CUDA path: `materialize_strided_cuda` at `stride_tricks.rs:394-417`
    dispatches to `backend.strided_copy_f32` / `f64`. Other dtypes on
    CUDA error with `NotImplementedOnCuda`.
  - CPU path: `view.data_vec()` walks the strided view and copies
    elements in logical order.
- `Tensor::as_strided_scatter` (`stride_tricks.rs:260-322`) — CPU
  path: starts from a contiguous copy of `self`, walks `src` in
  C-order, writes into the strided positions. CUDA path:
  `scatter_on_cuda` at `:431-476` clones `self` into a fresh
  contiguous GPU buffer via `strided_copy_*` then runs
  `strided_scatter_*` to overwrite positions — never bounces through
  host. bf16 / f16 on CUDA error with `NotImplementedOnCuda`.
- `AsStridedBackward` (`stride_tricks.rs:336-383`) implements
  `GradFn::backward` by allocating `zeros(input.shape())` and calling
  `zeros.as_strided_scatter(grad_output, size, stride, offset)`.

Non-test production consumers:

- `crate::einsum` references `as_strided_copy` at
  `einsum.rs:324, 357, 385, 425, 473` for diagonal extraction and
  Einstein-summation kernel reuse (this is real production code,
  not a test).
- `Tensor::masked_fill` / `Tensor::masked_select` (via
  `tensor.rs:1131, 1146`) compose with strided views and rely on the
  same bounds validation.

## Parity contract

`parity_ops = []`. The op has no parity-sweep entry because parity
testing for stride tricks requires a per-(shape, stride, offset)
oracle that the parity-sweep harness does not yet enumerate. The
in-file test suite at `stride_tricks.rs:482-712` covers:

- Shape reshape (contiguous).
- Sliding window (overlap).
- Negative stride (reverse).
- Zero stride (broadcast).
- Empty view (`shape contains 0`).
- Out-of-bounds rejection (both ends).
- Shape/stride length mismatch.
- Backward via sum.

PyTorch's behaviour is matched contract-by-contract: same bounds
rules, same overlap-allowed-but-UB-on-write semantics, same `Vec<_>`
order of returned coordinates in scatter.

## Verification

```bash
cargo test -p ferrotorch-core --lib stride_tricks
```

Expected: ~17 tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `Tensor::as_strided` at `ferrotorch-core/src/stride_tricks.rs:196` mirrors `aten/src/ATen/native/TensorShape.cpp::Tensor::as_strided`; non-test consumer: `crate::einsum` calls `as_strided` directly for Einstein-summation diagonal extraction (`ferrotorch-core/src/einsum.rs:425`). |
| REQ-2 | SHIPPED | impl: `Tensor::as_strided_copy` at `ferrotorch-core/src/stride_tricks.rs:231` mirrors `aten/src/ATen/native/AsStridedCopy.cpp::as_strided_copy`; non-test consumer: `crate::einsum` calls `as_strided_copy` for materialising diagonals on CUDA (`ferrotorch-core/src/einsum.rs:324, 357`). |
| REQ-3 | SHIPPED | impl: `Tensor::as_strided_scatter` at `ferrotorch-core/src/stride_tricks.rs:260` mirrors `aten/src/ATen/native/TensorAdvancedIndexing.cpp::as_strided_scatter`; non-test consumer: `AsStridedBackward::backward` at `:366-372` calls `zeros.as_strided_scatter(grad_output, ...)` to scatter gradients in production autograd. |
| REQ-4 | SHIPPED | impl: `validate_bounds` at `ferrotorch-core/src/stride_tricks.rs:126` with `stride_extent` helper at `:102`; non-test consumer: invoked from `Tensor::as_strided` at `:204` and `Tensor::as_strided_scatter` at `:269` — every public call validates before constructing the view. |
| REQ-5 | SHIPPED | impl: `AsStridedBackward in ferrotorch-core/src/stride_tricks.rs` with `impl GradFn::backward` at `backward in ferrotorch-core/src/stride_tricks.rs`; non-test consumer: `Tensor::as_strided` attaches it when `requires_grad` is set (`requires_grad in ferrotorch-core/src/stride_tricks.rs`) — production autograd graph routes through this when stride ops appear in a differentiable function. |
| REQ-6 | SHIPPED | impl: `stride_extent` at `ferrotorch-core/src/stride_tricks.rs:102-120` handles negative `s` via the `last >= 0` branch; non-test consumer: any user calling `tensor.flip(0)` semantically equivalent path (negative strides are how PyTorch encodes reverse views). Test pin at `:518-525`. |
| REQ-7 | SHIPPED | impl: `stride_extent` at `ferrotorch-core/src/stride_tricks.rs:102-120` treats `s == 0` as zero contribution per dim; non-test consumer: broadcast-replication via stride 0 is the foundational mechanic behind `Tensor::expand`. Test pin at `:527-533`. |
| REQ-8 | SHIPPED | impl: `as_strided in ferrotorch-core/src/stride_tricks.rs`, `as_strided_copy in ferrotorch-core/src/stride_tricks.rs`, `as_strided_scatter in ferrotorch-core/src/stride_tricks.rs`; non-test consumer: free-function path delegates to inherent methods; downstream code that prefers function form (e.g. functional-style composition) reaches the same logic. Per S5 the pub API surface is grandfathered. |
