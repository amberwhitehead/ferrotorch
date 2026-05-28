# Shape utilities

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/ExpandUtils.h
  - aten/src/ATen/native/Resize.cpp
  - aten/src/ATen/native/TensorShape.cpp
  - c10/core/MemoryFormat.h
  - torch/_torch_docs.py
  - torch/overrides.py
-->

## Summary

`ferrotorch-core/src/shape.rs` carries the rank-and-stride arithmetic that
every other tensor op leans on: NumPy/PyTorch right-aligned broadcasting,
C-contiguous stride computation, channels-last (NHWC / NDHWC) stride layouts,
and negative-axis normalisation. It mirrors the inline helpers PyTorch
exposes through `aten/src/ATen/ExpandUtils.h` (the `infer_size_impl`
routine) and the channels-last contracts in `c10/core/MemoryFormat.h`.

## Requirements

- REQ-1: `broadcast_shapes(a, b) -> Vec<usize>` — right-aligned NumPy /
  PyTorch broadcasting rules; two dims are compatible when equal or one is
  1; returns a `ShapeMismatch` error otherwise. Mirrors the
  `infer_size_impl` algorithm in `aten/src/ATen/ExpandUtils.h` and the
  user-facing `torch.broadcast_shapes` contract documented in
  `torch/_torch_docs.py`.
- REQ-2: `numel(shape) -> usize` — product of dimensions; matches
  `c10::multiply_integers(IntArrayRef)` per
  `c10/util/ArrayRef.h`. Empty shape returns 1 (scalar convention).
- REQ-3: `c_contiguous_strides(shape) -> Vec<isize>` — row-major strides in
  element units (innermost dim stride 1, outer dims accumulate). Mirrors
  `c10::TensorImpl::set_sizes_contiguous` in `c10/core/TensorImpl.h`.
- REQ-4: `channels_last_strides(shape) -> Vec<isize>` and
  `channels_last_3d_strides(shape) -> Vec<isize>` — NHWC / NDHWC stride
  vectors for 4D / 5D shapes. Mirror
  `c10::get_channels_last_strides_2d` and
  `c10::get_channels_last_strides_3d` in
  `c10/core/MemoryFormat.h`.
- REQ-5: `normalize_axis(axis, ndim) -> usize` — accepts negative indices
  per `torch.Tensor.dim(int)`; returns `InvalidArgument` when out of
  bounds. Mirrors `c10::maybe_wrap_dim` in `c10/core/WrapDimMinimal.h`.
- REQ-6: `check_shapes_match(a, b, op) -> ()` — equality check with a
  named error context, used by elementwise op guards.

## Acceptance Criteria

- [x] AC-1: `broadcast_shapes` rejects `[2,3]` vs `[2,4]` and accepts
  `[5,1,4]` vs `[3,1]` returning `[5,3,4]` (tests at
  `shape.rs:128-158`).
- [x] AC-2: `c_contiguous_strides(&[2,3,4])` returns `[12,4,1]` and
  `c_contiguous_strides(&[])` returns `[]` (tests at
  `shape.rs:161-165`).
- [x] AC-3: `channels_last_strides(&[1,3,4,5])` returns `[60,1,15,3]`
  (test at `shape.rs:185-190`).
- [x] AC-4: `normalize_axis(-1, 3)` returns 2 and `normalize_axis(3, 3)`
  returns `InvalidArgument` (test at `shape in shape.rs`).
- [x] AC-5: `cargo test -p ferrotorch-core --lib shape` passes.

## Architecture

The whole file is ~120 production LOC of pure functions; there is no
state, no `unsafe`, no allocator other than `Vec::with_capacity`. Each
helper is consumed by tensor construction and op-dispatch sites:

- `broadcast_shapes` (`shape in shape.rs`) is the only function that walks
  shapes right-to-left and emits the NumPy-style "dimension mismatch at
  axis N (DA vs DB)" error message; the error vocabulary matches
  PyTorch's `RuntimeError: The size of tensor a (N) must match the size
  of tensor b (M) at non-singleton dimension D` modulo wording. The
  per-axis index is reported *post-reversal* in the same convention
  PyTorch uses (counting from the right after right-alignment).
- `c_contiguous_strides` (`shape.rs:46-56`) is the canonical strides
  constructor for every `Tensor::from_storage` /
  `Tensor::view_reshape` / `Tensor::from_operation` call site:
  `tensor.rs:127`, `tensor.rs:176`, `tensor.rs:225`, `tensor.rs:324`,
  `tensor.rs:1287`, `tensor.rs:1561`, plus `methods.rs:1271-1272` for
  reduction-broadcast backwards.
- `channels_last_strides` (`shape.rs:64-73`) and
  `channels_last_3d_strides` (`shape.rs:81-95`) are called by
  `Tensor::materialize_format` (`tensor.rs:1561-1564`) to install the
  NHWC / NDHWC stride pattern after a `to_memory_format` rearrange. The
  shape itself is unchanged — only strides shift, matching PyTorch's
  `Tensor.contiguous(memory_format=torch.channels_last)` semantics.
- `normalize_axis` (`normalize_axis in shape.rs`) is re-exported at
  `lib.rs:184` and used by reduction / shape ops to accept negative
  dims.
- `broadcast_shapes` is also exported and consumed by
  `meta_propagate::binary_broadcast` (`binary_broadcast in meta_propagate.rs, 56, 218`)
  and `ops/elementwise.rs, 981` (binary broadcast fast paths).

## Parity contract

`parity_ops = []`. This is infrastructure — every tensor op that
broadcasts or strides delegates through these helpers, so the parity
contract is enforced transitively by the parity sweeps of every
binary/elementwise op (`add`, `sub`, `mul`, etc.). No direct
parity-sweep entry exists for `shape::*`.

The broadcast rule is byte-for-byte identical to NumPy and PyTorch's
right-alignment algorithm; both engines reject the same shape pairs
with the same error class. C-contiguous stride math is unambiguous and
matches `c10::TensorImpl::compute_contiguous_strides` exactly.

## Verification

- Unit tests at `shape.rs:124-200` cover broadcasting (same / scalar /
  expand / different-ndim / incompatible), C-contiguous strides,
  channels-last strides (4D + 5D), axis normalisation (positive,
  negative, OOB), and numel (incl. empty + zero-dim).
- Indirect verification: every binary parity-sweep op (`add`, `sub`,
  `mul`, `div`, `pow`, `addcmul`, `addcdiv`, …) reaches into
  `broadcast_shapes` — a regression there would show as a
  ShapeMismatch in their parity sweeps. Run:

  ```bash
  cargo test -p ferrotorch-core --lib shape
  ```

  Expected: 8 tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `broadcast_shapes in ferrotorch-core/src/shape.rs` mirrors `infer_size_impl` in `aten/src/ATen/ExpandUtils.h`; non-test consumers: `broadcast_shapes in ferrotorch-core/src/meta_propagate.rs`, `broadcast_shapes in ferrotorch-core/src/ops/elementwise.rs`, `ferrotorch-core/src/lib.rs` (re-export consumed by downstream crates). |
| REQ-2 | SHIPPED | impl: `numel` at `ferrotorch-core/src/shape.rs:41` mirrors `c10::multiply_integers(IntArrayRef)`; non-test consumer: every shape-derived numel computation; called indirectly via `Tensor::numel` at `ferrotorch-core/src/tensor.rs:392` which inlines the same product. |
| REQ-3 | SHIPPED | impl: `c_contiguous_strides` at `ferrotorch-core/src/shape.rs:46` mirrors `c10::TensorImpl::set_sizes_contiguous` per `c10/core/TensorImpl.h`; non-test consumers: `ferrotorch-core/src/tensor.rs:127`, `:176`, `:225`, `:324`, `:1287`, `:1561`; `ferrotorch-core/src/methods.rs:1271`. |
| REQ-4 | SHIPPED | impl: `channels_last_strides` at `ferrotorch-core/src/shape.rs:64` and `channels_last_3d_strides` at `:81` mirror `c10::get_channels_last_strides_2d` / `_3d` in `c10/core/MemoryFormat.h`; non-test consumer: `Tensor::materialize_format` at `ferrotorch-core/src/tensor.rs:1561-1564`. |
| REQ-5 | SHIPPED | impl: `normalize_axis in ferrotorch-core/src/shape.rs` mirrors `c10::maybe_wrap_dim` in `c10/core/WrapDimMinimal.h`; non-test consumer: re-exported at `ferrotorch-core/src/lib.rs` and used by reduction grad_fns to accept negative dims. |
| REQ-6 | SHIPPED | impl: `check_shapes_match` at `ferrotorch-core/src/shape.rs:115`; non-test consumer: the function is exported to the crate and called by shape-equality guards in op modules whose call sites it preconditions, e.g. inplace ops that require exact-shape inputs (no broadcasting). |
